"""Stage 2 matrix: composition estimators over forecast-kind transformers.

Covers that FeatureUnion, FeaturePipeline, and ColumnTransformer each operate on
forecast-kind children, propagate ``kind``, reject mixed-kind compositions, and
that ColumnTransformer protects the ``vintage_time`` index column.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest
from sklearn.linear_model import Ridge

from yohou.base import BaseForecastTransformer
from yohou.compose import ColumnTransformer, FeaturePipeline, FeatureUnion, PerVintageActualTransformer
from yohou.point import PointReductionForecaster
from yohou.preprocessing import FunctionTransformer


def _forecast_frame() -> pl.DataFrame:
    return pl.DataFrame({
        "vintage_time": [datetime(2020, 1, 1)] * 2 + [datetime(2020, 1, 2)] * 2,
        "time": [datetime(2020, 1, 2), datetime(2020, 1, 3), datetime(2020, 1, 3), datetime(2020, 1, 4)],
        "load": [100.0, 110.0, 200.0, 210.0],
        "wind": [10.0, 20.0, 30.0, 25.0],
    })


def _net_load_pv() -> PerVintageActualTransformer:
    return PerVintageActualTransformer(
        FunctionTransformer(
            func=lambda df: df.select((pl.col("load") - pl.col("wind")).alias("net_load")),
            feature_names_out=lambda self, names: ["net_load"],
        )
    )


def _identity_pv() -> PerVintageActualTransformer:
    return PerVintageActualTransformer(FunctionTransformer())


# --- 9.1: composers operate on forecast-kind children --------------------------


def _forecast_union() -> FeatureUnion:
    return FeatureUnion([("nl", _net_load_pv())])


def _forecast_pipeline() -> FeaturePipeline:
    return FeaturePipeline([("nl", _net_load_pv())])


def _forecast_column_transformer() -> ColumnTransformer:
    return ColumnTransformer([("load", _identity_pv(), ["load"])], remainder="drop")


@pytest.mark.parametrize(
    "make_composer",
    [_forecast_union, _forecast_pipeline, _forecast_column_transformer],
    ids=["union", "pipeline", "column_transformer"],
)
def test_forecast_kind_composition_preserves_index_and_kind(make_composer):
    """Each composer transforms a forecast frame, keeping the vintage/time index."""
    composer = make_composer()
    assert composer.__sklearn_tags__().transformer_tags.kind == "forecast"

    out = composer.fit_transform(_forecast_frame())
    assert "vintage_time" in out.columns
    assert "time" in out.columns
    assert out.height == 4
    # exactly the two index columns plus feature columns, in canonical order
    assert out.columns[:2] == ["vintage_time", "time"]
    assert len(out.columns) > 2


def test_actual_kind_composition_still_reports_actual():
    """A composer of actual children reports kind='actual' (regression guard)."""
    union = FeatureUnion([("id", FunctionTransformer())])
    assert union.__sklearn_tags__().transformer_tags.kind == "actual"


# --- 9.2: mixing raises, per composer ------------------------------------------


def test_feature_union_rejects_mixed_kinds():
    union = FeatureUnion([("actual", FunctionTransformer()), ("forecast", _net_load_pv())])
    with pytest.raises(ValueError, match="cannot mix actual and forecast"):
        union.fit(_forecast_frame())


def test_feature_pipeline_rejects_mixed_kinds():
    pipeline = FeaturePipeline([("actual", FunctionTransformer()), ("forecast", _net_load_pv())])
    with pytest.raises(ValueError, match="cannot mix actual and forecast"):
        pipeline.fit(_forecast_frame())


def test_column_transformer_rejects_mixed_kinds():
    ct = ColumnTransformer([("actual", FunctionTransformer(), ["load"]), ("forecast", _identity_pv(), ["wind"])])
    with pytest.raises(ValueError, match="cannot mix actual and forecast"):
        ct.fit(_forecast_frame())


# --- 9.3: ColumnTransformer protects the vintage_time index --------------------


def test_column_transformer_protects_vintage_time_with_remainder_drop():
    """vintage_time survives routing even with remainder='drop'."""
    ct = ColumnTransformer([("load", _identity_pv(), ["load"])], remainder="drop")
    out = ct.fit_transform(_forecast_frame())
    assert "vintage_time" in out.columns
    assert "time" in out.columns
    # 'wind' was neither routed nor kept (remainder='drop'); vintage_time is not a feature
    assert "wind" not in out.columns


# --- 9.4: forecast transformers rejected from a forecaster's actual slot --------


def _target_series() -> pl.DataFrame:
    time = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 3, 1), interval="1d", eager=True)
    return pl.DataFrame({"time": time, "v": [float(i) for i in range(len(time))]})


def test_forecast_leaf_rejected_from_actual_transformer():
    """A forecast leaf is rejected from actual_transformer (by the parameter constraint)."""
    with pytest.raises(Exception, match="actual_transformer"):
        PointReductionForecaster(estimator=Ridge(), actual_transformer=_net_load_pv())._validate_params()


def test_forecast_kind_composer_rejected_from_actual_transformer():
    """A forecast-kind composer is rejected from actual_transformer at fit (by the kind guard)."""
    forecaster = PointReductionForecaster(estimator=Ridge(), actual_transformer=_forecast_union())
    with pytest.raises(ValueError, match="actual-kind transformer"):
        forecaster.fit(_target_series(), forecasting_horizon=3)


# --- the memory API is actual-kind only ----------------------------------------


@pytest.mark.parametrize("method", ["observe", "rewind", "observe_transform", "rewind_transform"])
@pytest.mark.parametrize(
    "make_composer",
    [_forecast_union, _forecast_pipeline, _forecast_column_transformer],
    ids=["union", "pipeline", "column_transformer"],
)
def test_memory_api_rejected_on_forecast_kind_composition(make_composer, method):
    """The memory API means nothing on the vintage axis and must not be accepted.

    A forecast-kind composition is structurally a ``BaseActualTransformer``, so it
    inherits the memory API. The buffer it would maintain needs contiguous recent
    rows, which the discontinuous vintage axis cannot supply. Covers every method
    across every composer because they share no single implementation: the
    composite ``observe_transform``/``rewind_transform`` are separate overrides
    from ``observe``/``rewind``, so guarding one pair does not cover the other.
    """
    frame = _forecast_frame()
    composer = make_composer()
    composer.fit(frame)

    with pytest.raises(ValueError, match="actual-kind transformer"):
        getattr(composer, method)(frame)


def test_memory_api_still_works_on_actual_kind_composition():
    """The guard is a no-op for actual-kind compositions."""
    union = FeatureUnion([("id", FunctionTransformer())])
    union.fit(_target_series())

    assert union.observe(_target_series()) is union
    assert union.rewind(_target_series()) is union


def test_weighted_actual_union_observe_transforms():
    """Weight scaling excludes the index columns and leaves their dtype intact.

    The weight sites live only in ``_observe_transform_one`` /
    ``_rewind_transform_one``, which the memory guard now closes to forecast-kind
    compositions, so an actual-kind composition is the reachable path. The sites
    exclude every index column rather than just ``"time"``, which is
    behaviour-identical here (a single-axis frame's only index column is
    ``"time"``) and correct if a forecast frame ever reaches them.
    """
    time = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 6), interval="1d", eager=True)
    frame = pl.DataFrame({"time": time, "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})

    union = FeatureUnion(
        [
            (
                "a",
                FunctionTransformer(
                    func=lambda df: df.select(pl.col("v").alias("x")),
                    feature_names_out=lambda self, names: ["x"],
                ),
            )
        ],
        transformer_weights={"a": 2.0},
    )
    union.fit(frame)
    out = union.observe_transform(frame)

    assert out["a_x"].to_list() == pytest.approx([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    assert out["time"].dtype == frame["time"].dtype


class _OrderPreservingForecastTransformer(BaseForecastTransformer):
    """A forecast transformer that emits rows in input order, without re-grouping.

    ``PerVintageActualTransformer`` re-groups rows by ``vintage_time``; this does
    not. A union of the two therefore has children that disagree on row order,
    which is the shape that exposes positional (rather than index-based)
    alignment.
    """

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        return X.select("vintage_time", "time", (pl.col("load") * 100).alias("scaled"))

    def get_feature_names_out(self, input_features=None):
        return ["scaled"]


def test_union_aligns_children_that_disagree_on_row_order():
    """A re-grouping child beside an order-preserving one still aligns by index.

    The frame is sorted by ``time`` with vintages interleaved, so the
    per-vintage child emits rows grouped by vintage while the sibling keeps the
    input order. Values must attach to their own (vintage_time, time) row.
    """
    t = datetime(2020, 1, 1)
    v1, v2 = t, t + timedelta(days=1)
    # interleaved by time, so the two children's row orders genuinely differ
    frame = pl.DataFrame({
        "vintage_time": [v1, v2, v1, v2],
        "time": [t + timedelta(days=2), t + timedelta(days=2), t + timedelta(days=3), t + timedelta(days=3)],
        "load": [10.0, 20.0, 30.0, 40.0],
    })

    union = FeatureUnion([
        (
            "pv",
            PerVintageActualTransformer(
                FunctionTransformer(
                    func=lambda df: df.select(pl.col("load").alias("passthrough")),
                    feature_names_out=lambda self, names: ["passthrough"],
                )
            ),
        ),
        ("plain", _OrderPreservingForecastTransformer()),
    ])
    out = union.fit_transform(frame).sort("vintage_time", "time")

    # scaled must be exactly 100x passthrough on every row, whatever order each child emitted
    assert out["plain_scaled"].to_list() == pytest.approx([v * 100 for v in out["pv_passthrough"].to_list()])
