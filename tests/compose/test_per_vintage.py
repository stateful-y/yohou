"""Tests for PerVintageActualTransformer and the actual/forecast kind split."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from yohou.base import BaseActualTransformer, BaseForecastTransformer
from yohou.compose import FeatureUnion, PerVintageActualTransformer
from yohou.preprocessing import (
    Downsampler,
    FunctionTransformer,
    LagTransformer,
    SimpleImputer,
    StandardScaler,
)
from yohou.testing.forecast_transformer import FORECAST_TRANSFORMER_CHECKS


def _sample_forecast_frame() -> pl.DataFrame:
    """Two vintages, three horizon steps each, with load/wind feature columns."""
    return pl.DataFrame({
        "vintage_time": [datetime(2020, 1, 1)] * 3 + [datetime(2020, 1, 2)] * 3,
        "time": [
            datetime(2020, 1, 2),
            datetime(2020, 1, 3),
            datetime(2020, 1, 4),
            datetime(2020, 1, 3),
            datetime(2020, 1, 4),
            datetime(2020, 1, 5),
        ],
        "load": [100.0, 110.0, 120.0, 200.0, 210.0, 220.0],
        "wind": [10.0, 20.0, 15.0, 30.0, 25.0, 35.0],
    })


def _net_load_transformer() -> FunctionTransformer:
    """A stateless FunctionTransformer emitting net_load = load - wind."""
    return FunctionTransformer(
        func=lambda df: df.select((pl.col("load") - pl.col("wind")).alias("net_load")),
        feature_names_out=lambda self, names: ["net_load"],
    )


def _cumsum_transformer() -> FunctionTransformer:
    """A stateless but position-dependent transform (cum_sum has no NaN warmup)."""
    return FunctionTransformer(
        func=lambda df: df.select(pl.col("load").cum_sum().alias("cum")),
        feature_names_out=lambda self, names: ["cum"],
    )


@pytest.mark.parametrize("check", FORECAST_TRANSFORMER_CHECKS, ids=lambda c: c.__name__)
def test_forecast_transformer_contract(check):
    """PerVintageActualTransformer satisfies the forecast-transformer contract."""
    check(PerVintageActualTransformer(_net_load_transformer()), _sample_forecast_frame())


def test_net_load_applied_per_vintage():
    """Each vintage's net_load uses only that vintage's rows."""
    tx = PerVintageActualTransformer(_net_load_transformer())
    out = tx.fit_transform(_sample_forecast_frame())
    assert out.columns == ["vintage_time", "time", "net_load"]
    assert out["net_load"].to_list() == [90.0, 90.0, 105.0, 170.0, 185.0, 185.0]


def test_position_dependent_transform_does_not_bleed_across_vintages():
    """A cumulative sum resets at each vintage boundary (no cross-vintage bleed)."""
    tx = PerVintageActualTransformer(_cumsum_transformer())
    out = tx.fit_transform(_sample_forecast_frame())
    # If it bled, vintage 2 would continue from 330: [430, 540, 650].
    assert out["cum"].to_list() == [100.0, 210.0, 330.0, 200.0, 410.0, 630.0]


def _two_scale_forecast_frame() -> pl.DataFrame:
    """Two vintages with deliberately different scales.

    Vintage 1's ``a`` spans [10, 20, 30] (mean 20); vintage 2's spans
    [1000, 2000, 3000] (mean 2000). The scales differ so that per-vintage
    fitting is distinguishable from a single shared fit: a global scaler would
    center both vintages on ~1010, leaving neither vintage's output centered.
    """
    return pl.DataFrame({
        "vintage_time": [datetime(2020, 1, 1)] * 3 + [datetime(2020, 1, 2)] * 3,
        "time": [
            datetime(2020, 1, 2),
            datetime(2020, 1, 3),
            datetime(2020, 1, 4),
            datetime(2020, 1, 3),
            datetime(2020, 1, 4),
            datetime(2020, 1, 5),
        ],
        "a": [10.0, 20.0, 30.0, 1000.0, 2000.0, 3000.0],
    })


def test_each_vintage_is_scaled_by_its_own_statistics():
    """A StandardScaler standardizes each vintage against that vintage alone.

    Fails under a shared/global fit: with vintages at scales ~20 and ~2000,
    global centering leaves each vintage's output far from mean 0.
    """
    out = PerVintageActualTransformer(StandardScaler()).fit_transform(_two_scale_forecast_frame())

    for part in out.partition_by("vintage_time", maintain_order=True):
        assert part["a"].mean() == pytest.approx(0.0, abs=1e-9)
        # ddof=0 matches the population standard deviation the scaler divides by.
        assert part["a"].std(ddof=0) == pytest.approx(1.0)


def test_per_vintage_mean_imputation():
    """Each vintage's missing value is filled from that vintage's own mean."""
    frame = pl.DataFrame({
        "vintage_time": [datetime(2020, 1, 1)] * 3 + [datetime(2020, 1, 2)] * 3,
        "time": [
            datetime(2020, 1, 2),
            datetime(2020, 1, 3),
            datetime(2020, 1, 4),
            datetime(2020, 1, 3),
            datetime(2020, 1, 4),
            datetime(2020, 1, 5),
        ],
        # vintage 1 non-null mean = 20, vintage 2 non-null mean = 2000
        "a": [10.0, None, 30.0, 1000.0, None, 3000.0],
    })
    out = PerVintageActualTransformer(SimpleImputer(strategy="mean")).fit_transform(frame)

    filled = out.sort("vintage_time", "time")["a"].to_list()
    assert filled == pytest.approx([10.0, 20.0, 30.0, 1000.0, 2000.0, 3000.0])


def test_vintage_transform_is_independent_of_other_vintages():
    """A vintage's output is identical alone or inside the full frame (leakage-free)."""
    frame = _two_scale_forecast_frame()
    v0_time = datetime(2020, 1, 1)
    v0_alone = frame.filter(pl.col("vintage_time") == v0_time)

    full = PerVintageActualTransformer(StandardScaler()).fit_transform(frame)
    alone = PerVintageActualTransformer(StandardScaler()).fit_transform(v0_alone)

    full_v0 = full.filter(pl.col("vintage_time") == v0_time).sort("time")["a"].to_list()
    assert full_v0 == pytest.approx(alone.sort("time")["a"].to_list())


def test_stateful_inner_is_rejected():
    """A stateful wrapped transformer (observation_horizon > 0) is rejected at fit."""
    tx = PerVintageActualTransformer(LagTransformer(lag=1))  # LagTransformer is stateful
    with pytest.raises(ValueError, match="stateless"):
        tx.fit(_sample_forecast_frame())


def test_stateful_inner_too_large_for_the_vintage_names_the_constraint():
    """An inner whose horizon exceeds the vintage length still names the real constraint.

    Vintages are short by nature, so a stateful inner often fails *inside* the
    inner before the composite's own observation_horizon check runs. The raw
    inner message ("Not enough input data to set the transformer memory") never
    mentions lifting, leaving the user no idea that stateful transformers cannot
    be lifted at all.
    """
    tx = PerVintageActualTransformer(LagTransformer(lag=5))  # vintages here are 3 rows
    with pytest.raises(ValueError, match="stateless") as excinfo:
        tx.fit(_sample_forecast_frame())

    # the inner's own diagnosis is preserved as the chained cause
    assert excinfo.value.__cause__ is not None
    assert "observation_horizon" in str(excinfo.value.__cause__)


def test_stateless_function_transformer_is_still_accepted():
    """A stateless FunctionTransformer is accepted, despite reporting stateful=True pre-fit.

    ``FunctionTransformer`` cannot know whether its function is stateful without
    probing it at fit, so its tag is conservatively True beforehand. Enforcing
    statelessness by reading that tag before fitting would reject this, the
    wrapper's primary use case, which is why the horizon is measured after fit.
    """
    inner = _net_load_transformer()
    assert inner.__sklearn_tags__().transformer_tags.stateful is True  # conservative, pre-fit

    tx = PerVintageActualTransformer(inner)
    out = tx.fit_transform(_sample_forecast_frame())
    assert out["net_load"].to_list() == [90.0, 90.0, 105.0, 170.0, 185.0, 185.0]


def test_fit_on_empty_frame_raises():
    """Fitting on an empty X_forecast frame raises a clear error."""
    tx = PerVintageActualTransformer(_net_load_transformer())
    with pytest.raises(ValueError, match="empty"):
        tx.fit(_sample_forecast_frame().clear())


def test_get_feature_names_out_delegates_to_inner():
    """Output feature names come from the wrapped transformer."""
    tx = PerVintageActualTransformer(_net_load_transformer())
    tx.fit(_sample_forecast_frame())
    assert tx.get_feature_names_out() == ["net_load"]


def test_sub_two_row_vintages_are_dropped_with_a_warning():
    """A vintage too small to fit per vintage is dropped, with a warning, not transformed.

    The truncated tail of a real forecast frame is full of single-row vintages
    (the horizon runs off the end of the series). Such a vintage has no
    per-vintage statistic to compute, so it is dropped and reported rather than
    scaled/imputed with borrowed parameters.
    """
    # two full vintages plus a trailing single-row vintage
    frame = pl.DataFrame({
        "vintage_time": [datetime(2020, 1, 1)] * 3 + [datetime(2020, 1, 2)] * 3 + [datetime(2020, 1, 3)],
        "time": [
            datetime(2020, 1, 2),
            datetime(2020, 1, 3),
            datetime(2020, 1, 4),
            datetime(2020, 1, 3),
            datetime(2020, 1, 4),
            datetime(2020, 1, 5),
            datetime(2020, 1, 4),
        ],
        "a": [10.0, 20.0, 30.0, 1000.0, 2000.0, 3000.0, 99.0],
    })
    tx = PerVintageActualTransformer(StandardScaler())

    with pytest.warns(UserWarning, match="dropped 1 vintage"):
        out = tx.fit_transform(frame)

    # the two full vintages survive; the single-row vintage is gone
    assert out["vintage_time"].unique().sort().to_list() == [datetime(2020, 1, 1), datetime(2020, 1, 2)]
    assert out.height == 6


def test_irregular_grid_inner_is_lifted_per_vintage():
    """An inner tagged accepts_irregular_grid tolerates a jittered vintage.

    Pins the interaction between the per-vintage fit and the
    ``accepts_irregular_grid`` tag: each vintage is fitted on its own, so an
    inner that opts into the tag (here a ``Downsampler``) is validated per
    vintage and accepts a non-uniform axis within it. An inner that does not opt
    in still requires a uniform grid per vintage.
    """
    t0 = datetime(2024, 1, 1)

    def jittered(vintage: datetime, minutes: list[int]) -> pl.DataFrame:
        return pl.DataFrame({
            "vintage_time": [vintage] * len(minutes),
            "time": [t0 + timedelta(minutes=m) for m in minutes],
            "a": [float(i) for i in range(len(minutes))],
        })

    # each vintage is ~15m-spaced but jittered, so no strict interval exists
    frame = pl.concat([jittered(t0, [0, 14, 31, 44]), jittered(t0 + timedelta(hours=1), [60, 76, 89, 104])])

    out = PerVintageActualTransformer(Downsampler(interval="1h")).fit_transform(frame)

    # one hourly bin per vintage, each aggregated from that vintage's own rows
    assert out["vintage_time"].to_list() == [t0, t0 + timedelta(hours=1)]
    assert out["a"].to_list() == pytest.approx([1.5, 1.5])


def test_all_vintages_too_small_raises_at_fit():
    """If no vintage is large enough to fit, fit raises rather than warning."""
    frame = pl.DataFrame({
        "vintage_time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        "time": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
        "a": [10.0, 20.0],
    })  # every vintage has a single row
    tx = PerVintageActualTransformer(StandardScaler())
    with pytest.raises(ValueError, match="at least one vintage"):
        tx.fit(frame)


def test_feature_schema_mismatch_at_transform_raises():
    """Transforming a frame with different feature columns than fit raises."""
    tx = PerVintageActualTransformer(_net_load_transformer())
    tx.fit(_sample_forecast_frame())
    wrong = _sample_forecast_frame().rename({"wind": "solar"})
    with pytest.raises(ValueError, match="do not match"):
        tx.transform(wrong)


# --- the kind split ------------------------------------------------------------


def test_actual_leaves_report_actual_kind():
    """Existing single-axis leaves report kind='actual' by default."""
    for tx in (FunctionTransformer(), LagTransformer(lag=1)):
        assert tx.__sklearn_tags__().transformer_tags.kind == "actual"


def test_forecast_transformer_reports_forecast_kind():
    """A forecast transformer reports kind='forecast'."""
    tx = PerVintageActualTransformer(_net_load_transformer())
    assert tx.__sklearn_tags__().transformer_tags.kind == "forecast"


def test_forecast_transformer_excluded_from_actual_hierarchy():
    """A forecast transformer is not a BaseActualTransformer, so actual-only slots reject it.

    This is the durable invariant the ``isinstance``-based slot guards rely on:
    ``BaseForecastTransformer`` is a sibling of ``BaseActualTransformer`` under the
    private ``_BaseTransformer`` root, not a subclass.
    """
    tx = PerVintageActualTransformer(_net_load_transformer())
    assert not isinstance(tx, BaseActualTransformer)
    assert not issubclass(BaseForecastTransformer, BaseActualTransformer)


def test_compose_then_lift_wraps_an_actual_union():
    """Stage 1 supports compose-then-lift: lift a FeatureUnion of actual transformers per vintage."""
    inner = FeatureUnion([
        ("net_load", _net_load_transformer()),
        ("cum", _cumsum_transformer()),
    ])
    tx = PerVintageActualTransformer(inner)
    out = tx.fit_transform(_sample_forecast_frame())
    assert "vintage_time" in out.columns and "time" in out.columns
    # FeatureUnion prefixes each output with its step name.
    assert "net_load_net_load" in out.columns and "cum_cum" in out.columns
    # cum resets per vintage (no bleed) even through the union
    assert out["cum_cum"].to_list() == [100.0, 210.0, 330.0, 200.0, 410.0, 630.0]
