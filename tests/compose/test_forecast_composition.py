"""Stage 2 matrix: composition estimators over forecast-kind transformers.

Covers that FeatureUnion, FeaturePipeline, and ColumnTransformer each operate on
forecast-kind children, propagate ``kind``, reject mixed-kind compositions, and
that ColumnTransformer protects the ``vintage_time`` index column.
"""

from datetime import datetime

import polars as pl
import pytest
from sklearn.linear_model import Ridge

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


def test_forecast_leaf_rejected_from_feature_transformer():
    """A forecast leaf is rejected from feature_transformer (by the parameter constraint)."""
    with pytest.raises(Exception, match="feature_transformer"):
        PointReductionForecaster(estimator=Ridge(), feature_transformer=_net_load_pv())._validate_params()


def test_forecast_kind_composer_rejected_from_feature_transformer():
    """A forecast-kind composer is rejected from feature_transformer at fit (by the kind guard)."""
    forecaster = PointReductionForecaster(estimator=Ridge(), feature_transformer=_forecast_union())
    with pytest.raises(ValueError, match="actual-kind transformer"):
        forecaster.fit(_target_series(), forecasting_horizon=3)
