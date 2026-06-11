"""Tests for estimator-based time-axis weighters."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from yohou.weighting import (
    BaseWeighter,
    CompositeWeighter,
    ExponentialDecayWeighter,
    LinearDecayWeighter,
    LookupWeighter,
    SeasonalEmphasisWeighter,
    TableWeighter,
)
from yohou.weighting.weighters import (
    _combine_weight_vectors,
    _normalize_weights,
    _resolve_weighter_to_array,
)


@pytest.fixture
def times() -> pl.Series:
    """Three consecutive daily timestamps."""
    return pl.Series("time", [datetime(2024, 1, d) for d in range(1, 4)])


@pytest.fixture
def steps() -> pl.Series:
    """Integer forecasting steps."""
    return pl.Series("forecasting_step", [1, 2, 3])


# ---------------------------------------------------------------------------
# Legacy-equivalence golden values (formerly the factory functions)
# ---------------------------------------------------------------------------


def test_exponential_decay_matches_legacy_values(times: pl.Series) -> None:
    """ExponentialDecayWeighter reproduces the old exponential_decay_weight output."""
    weights = ExponentialDecayWeighter(half_life=1).compute_weights(times).to_list()
    assert weights == pytest.approx([0.25, 0.5, 1.0])


def test_linear_decay_matches_legacy_values() -> None:
    """LinearDecayWeighter reproduces the old linear_decay_weight output."""
    t = pl.Series("time", [datetime(2024, 1, d) for d in range(1, 5)])
    weights = LinearDecayWeighter().compute_weights(t).to_list()
    assert weights == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])


def test_linear_decay_max_steps_zeros_older() -> None:
    """max_steps zeros out observations beyond the window."""
    t = pl.Series("time", [datetime(2024, 1, d) for d in range(1, 6)])
    weights = LinearDecayWeighter(max_steps=3).compute_weights(t).to_numpy()
    assert weights[0] == 0.0
    assert weights[1] == 0.0
    assert weights[-1] == 1.0


def test_seasonal_emphasis_matches_legacy_values() -> None:
    """SeasonalEmphasisWeighter emphasizes in-phase positions."""
    t = pl.Series(
        "time",
        [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 8), datetime(2024, 1, 9)],
    )
    weights = SeasonalEmphasisWeighter(seasonality=7, emphasis=2.0).compute_weights(t).to_list()
    assert weights == pytest.approx([1.0, 1.0, 1.0, 2.0])


# ---------------------------------------------------------------------------
# ExponentialDecayWeighter: scale handling (thread C1')
# ---------------------------------------------------------------------------


def test_exponential_scale_inferred_elapsed_for_datetime(times: pl.Series) -> None:
    """Datetime keys default to elapsed scale."""
    w = ExponentialDecayWeighter(half_life=1)
    assert w._resolve_scale(times) == "elapsed"


def test_exponential_scale_inferred_position_for_integer(steps: pl.Series) -> None:
    """Integer keys default to position scale."""
    w = ExponentialDecayWeighter(half_life=1)
    assert w._resolve_scale(steps) == "position"


def test_exponential_position_serves_step_keys(steps: pl.Series) -> None:
    """Exponential decay works over integer steps without a datetime key."""
    weights = ExponentialDecayWeighter(half_life=1).compute_weights(steps).to_list()
    # Position decay from the largest step: step 3 -> 1.0, step 2 -> 0.5, step 1 -> 0.25
    assert weights == pytest.approx([0.25, 0.5, 1.0])


def test_exponential_timedelta_with_position_scale_raises(steps: pl.Series) -> None:
    """A timedelta half_life is invalid with a non-elapsed scale."""
    w = ExponentialDecayWeighter(half_life=timedelta(days=7), scale="position")
    with pytest.raises(ValueError, match="elapsed"):
        w.compute_weights(steps)


def test_exponential_timedelta_elapsed_ok(times: pl.Series) -> None:
    """A timedelta half_life is valid with elapsed scale."""
    weights = ExponentialDecayWeighter(half_life=timedelta(days=1), scale="elapsed").compute_weights(times)
    assert weights.to_list() == pytest.approx([0.25, 0.5, 1.0])


# ---------------------------------------------------------------------------
# LookupWeighter / TableWeighter (thread A3)
# ---------------------------------------------------------------------------


def test_lookup_default_applies_to_missing_keys(steps: pl.Series) -> None:
    """Keys absent from mapping receive the default."""
    weights = LookupWeighter(mapping={1: 1.0}, default=0.0).compute_weights(steps).to_list()
    assert weights == pytest.approx([1.0, 0.0, 0.0])


def test_lookup_default_is_tunable(steps: pl.Series) -> None:
    """The default is a real, settable parameter."""
    w = LookupWeighter(mapping={1: 1.0}, default=0.0)
    w.set_params(default=0.5)
    weights = w.compute_weights(steps).to_list()
    assert weights == pytest.approx([1.0, 0.5, 0.5])


def test_lookup_wildcard_compat(steps: pl.Series) -> None:
    """A '*' entry in the mapping still overrides the default (compat)."""
    weights = LookupWeighter(mapping={"*": 0.0, 1: 2.0}).compute_weights(steps).to_list()
    assert weights == pytest.approx([2.0, 0.0, 0.0])


def test_table_weighter_resolves_by_join(times: pl.Series) -> None:
    """TableWeighter looks up weights by joining on the key column."""
    frame = pl.DataFrame({"time": times, "weight": [0.1, 0.2, 0.7]})
    weights = TableWeighter(frame=frame, on="time").compute_weights(times).to_list()
    assert weights == pytest.approx([0.1, 0.2, 0.7])


def test_table_weighter_unmatched_key_raises(times: pl.Series) -> None:
    """A key with no row in the frame raises ValueError."""
    frame = pl.DataFrame({"time": times[:2], "weight": [0.5, 0.5]})
    with pytest.raises(ValueError, match="no values"):
        TableWeighter(frame=frame, on="time").compute_weights(times)


def test_table_weighter_panel_group_column(times: pl.Series) -> None:
    """Group-specific weight columns are used for panel data."""
    frame = pl.DataFrame({"time": times, "A_weight": [1.0, 2.0, 3.0], "weight": [9.0, 9.0, 9.0]})
    weights = TableWeighter(frame=frame, on="time").compute_weights(times, group_name="A").to_list()
    assert weights == pytest.approx([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# CompositeWeighter (thread D)
# ---------------------------------------------------------------------------


def test_composite_multiplies_components(times: pl.Series) -> None:
    """CompositeWeighter multiplies component outputs element-wise."""
    a = ExponentialDecayWeighter(half_life=2)
    b = SeasonalEmphasisWeighter(seasonality=2, emphasis=1.5)
    expected = (a.compute_weights(times) * b.compute_weights(times)).to_list()
    got = CompositeWeighter([("a", a), ("b", b)]).compute_weights(times).to_list()
    assert got == pytest.approx(expected)


def test_base_weighter_is_abstract() -> None:
    """BaseWeighter cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseWeighter()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# _resolve_weighter_to_array + validation helpers
# ---------------------------------------------------------------------------


def test_normalize_weights_sums_to_n() -> None:
    """_normalize_weights scales the array to sum to its length."""
    out = _normalize_weights(np.array([1.0, 3.0]))
    assert out.sum() == pytest.approx(2.0)


def test_combine_weight_vectors_multiplies_and_normalizes() -> None:
    """_combine_weight_vectors multiplies inputs and normalizes to n."""
    out = _combine_weight_vectors(np.array([1.0, 2.0]), np.array([2.0, 1.0]), n=2)
    assert out is not None
    assert out.sum() == pytest.approx(2.0)


def test_combine_weight_vectors_all_none_returns_none() -> None:
    """All-None inputs return None."""
    assert _combine_weight_vectors(None, None, n=3) is None


def test_normalize_weights_zero_sum_raises() -> None:
    """_normalize_weights rejects an all-zero array (cannot scale)."""
    with pytest.raises(ValueError, match="sum is zero"):
        _normalize_weights(np.array([0.0, 0.0]))


def test_combine_weight_vectors_zero_product_raises() -> None:
    """_combine_weight_vectors rejects inputs whose product sums to zero."""
    with pytest.raises(ValueError, match="sum to zero"):
        _combine_weight_vectors(np.array([1.0, 0.0]), np.array([0.0, 1.0]), n=2)


# ---------------------------------------------------------------------------
# Single-element key short-circuits
# ---------------------------------------------------------------------------


def test_linear_decay_single_key_returns_one() -> None:
    """A length-1 key yields weight 1.0 (no rank range to decay over)."""
    weights = LinearDecayWeighter().compute_weights(pl.Series("time", [datetime(2024, 1, 1)])).to_list()
    assert weights == pytest.approx([1.0])


def test_seasonal_emphasis_single_key_returns_one() -> None:
    """A length-1 key yields weight 1.0 (no phase to emphasize)."""
    weights = SeasonalEmphasisWeighter(seasonality=7).compute_weights(pl.Series("time", [datetime(2024, 1, 1)]))
    assert weights.to_list() == pytest.approx([1.0])


# ---------------------------------------------------------------------------
# TableWeighter error/branch paths
# ---------------------------------------------------------------------------


def test_table_weighter_none_frame_raises(times: pl.Series) -> None:
    """A None frame cannot resolve any weights."""
    with pytest.raises(ValueError, match="non-None"):
        TableWeighter(frame=None).compute_weights(times)


def test_table_weighter_falls_back_to_weight_column(times: pl.Series) -> None:
    """When the group column is absent, the generic 'weight' column is used."""
    frame = pl.DataFrame({"time": times, "weight": [1.0, 2.0, 3.0]})
    weights = TableWeighter(frame=frame, on="time").compute_weights(times, group_name="B").to_list()
    assert weights == pytest.approx([1.0, 2.0, 3.0])


def test_table_weighter_missing_weight_column_raises(times: pl.Series) -> None:
    """A frame without a 'weight' column raises for global data."""
    frame = pl.DataFrame({"time": times, "other": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="must have 'weight' column"):
        TableWeighter(frame=frame, on="time").compute_weights(times)


def test_table_weighter_missing_panel_columns_raises(times: pl.Series) -> None:
    """A frame missing both '{group}_weight' and 'weight' raises for panel data."""
    frame = pl.DataFrame({"time": times, "other": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="missing both 'C_weight' and 'weight'"):
        TableWeighter(frame=frame, on="time").compute_weights(times, group_name="C")


# ---------------------------------------------------------------------------
# CompositeWeighter get_params / set_params edge paths
# ---------------------------------------------------------------------------


def test_composite_set_params_replaces_weighters_list() -> None:
    """set_params replaces the whole weighters list."""
    w = CompositeWeighter([("decay", ExponentialDecayWeighter(1))])
    new = [("lin", LinearDecayWeighter(2))]
    w.set_params(weighters=new)
    assert w.weighters is new


# ---------------------------------------------------------------------------
# _resolve_weighter_to_array contract checks
# ---------------------------------------------------------------------------


def test_resolve_weighter_to_array_rejects_non_series(times: pl.Series) -> None:
    """A weighter returning a non-Series is rejected."""

    class _BadType(BaseWeighter):
        def compute_weights(self, key: pl.Series, group_name: str | None = None) -> pl.Series:
            return np.ones(len(key))  # type: ignore[return-value]

    with pytest.raises(ValueError, match="must return pl.Series"):
        _resolve_weighter_to_array(_BadType(), times)


def test_resolve_weighter_to_array_rejects_mismatched_length(times: pl.Series) -> None:
    """A weighter returning the wrong number of weights is rejected."""

    class _BadLength(BaseWeighter):
        def compute_weights(self, key: pl.Series, group_name: str | None = None) -> pl.Series:
            return pl.Series([1.0], dtype=pl.Float64)

    with pytest.raises(ValueError, match="expected 3 rows"):
        _resolve_weighter_to_array(_BadLength(), times)


# ---------------------------------------------------------------------------
# CompositeWeighter combination = "mean" and per-component weights
# ---------------------------------------------------------------------------


def test_composite_mean_combination(times: pl.Series) -> None:
    """combination='mean' averages the component weight series."""
    a = ExponentialDecayWeighter(half_life=2)
    b = SeasonalEmphasisWeighter(seasonality=2, emphasis=1.5)
    expected = ((a.compute_weights(times) + b.compute_weights(times)) / 2).to_list()
    got = CompositeWeighter([("a", a), ("b", b)], combination="mean").compute_weights(times).to_list()
    assert got == pytest.approx(expected)


def test_composite_mean_weighted_coefficients(times: pl.Series) -> None:
    """Per-component coefficients give a weighted average under mean."""
    a = ExponentialDecayWeighter(half_life=2)
    b = SeasonalEmphasisWeighter(seasonality=2, emphasis=1.5)
    expected = ((a.compute_weights(times) * 0.25 + b.compute_weights(times) * 0.75) / 1.0).to_list()
    got = (
        CompositeWeighter([("a", a), ("b", b)], combination="mean", weights=[0.25, 0.75])
        .compute_weights(times)
        .to_list()
    )
    assert got == pytest.approx(expected)


def test_composite_multiply_exponents(times: pl.Series) -> None:
    """Per-component exponents apply under multiply."""
    a = ExponentialDecayWeighter(half_life=2)
    b = SeasonalEmphasisWeighter(seasonality=2, emphasis=1.5)
    expected = (a.compute_weights(times).pow(2.0) * b.compute_weights(times).pow(0.5)).to_list()
    got = (
        CompositeWeighter([("a", a), ("b", b)], combination="multiply", weights=[2.0, 0.5])
        .compute_weights(times)
        .to_list()
    )
    assert got == pytest.approx(expected)


def test_weighters_not_importable_from_old_utils_path() -> None:
    """Weighters moved to yohou.weighting; the old utils path is gone."""
    with pytest.raises((ImportError, ModuleNotFoundError)):
        from yohou.utils.weighting import ExponentialDecayWeighter  # noqa: F401


def test_weighters_not_reexported_from_utils() -> None:
    """yohou.utils no longer exposes weighter estimators."""
    from yohou import utils

    for name in (
        "BaseWeighter",
        "ExponentialDecayWeighter",
        "LinearDecayWeighter",
        "SeasonalEmphasisWeighter",
        "LookupWeighter",
        "TableWeighter",
        "CompositeWeighter",
    ):
        assert name not in utils.__all__
        assert not hasattr(utils, name)


def test_public_weighters_are_discoverable() -> None:
    """all_estimators finds the weighters under the new yohou.weighting path."""
    from yohou.utils.discovery import all_estimators

    discovered = dict(all_estimators())
    # BaseWeighter is abstract and is excluded by all_estimators; only the
    # concrete weighters are discoverable.
    for name in (
        "ExponentialDecayWeighter",
        "LinearDecayWeighter",
        "SeasonalEmphasisWeighter",
        "LookupWeighter",
        "TableWeighter",
        "CompositeWeighter",
    ):
        assert name in discovered, f"{name} not discovered"
        assert discovered[name].__module__.startswith("yohou.weighting")
