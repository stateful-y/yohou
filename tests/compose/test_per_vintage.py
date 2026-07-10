"""Tests for PerVintageActualTransformer and the actual/forecast kind split."""

from datetime import datetime

import polars as pl
import pytest

from yohou.base import BaseActualTransformer, BaseForecastTransformer
from yohou.compose import FeatureUnion, PerVintageActualTransformer
from yohou.preprocessing import FunctionTransformer, LagTransformer
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


def test_stateful_inner_is_rejected():
    """A stateful wrapped transformer (observation_horizon > 0) is rejected at fit."""
    tx = PerVintageActualTransformer(LagTransformer(lag=1))  # LagTransformer is stateful
    with pytest.raises(ValueError, match="stateless"):
        tx.fit(_sample_forecast_frame())


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
