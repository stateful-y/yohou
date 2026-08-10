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


def test_linear_decay_max_steps_one_keeps_only_most_recent() -> None:
    """max_steps=1 zeros every key except the most recent, which gets 1.0."""
    t = pl.Series("time", [datetime(2024, 1, d) for d in range(1, 4)])
    weights = LinearDecayWeighter(max_steps=1).compute_weights(t).to_list()
    assert weights == pytest.approx([0.0, 0.0, 1.0])


def test_linear_decay_max_steps_exceeds_length() -> None:
    """max_steps > len(key) computes without an OverflowError.

    Ranks are cast to a signed type so that ``ranks - cutoff`` stays correct
    (rather than underflowing) when max_steps exceeds the series length.
    """
    t = pl.Series("time", [datetime(2024, 1, d) for d in range(1, 4)])
    weights = LinearDecayWeighter(max_steps=10).compute_weights(t).to_numpy()
    assert len(weights) == 3
    assert weights[-1] == 1.0
    assert all(w >= 0.0 for w in weights)


def test_seasonal_emphasis_list_emphasizes_union_of_periods() -> None:
    """A list of seasonalities emphasizes keys in phase with ANY period."""
    t = pl.Series("time", [datetime(2024, 1, d) for d in range(1, 8)])
    w_list = SeasonalEmphasisWeighter(seasonality=[2, 3], emphasis=2.0).compute_weights(t).to_list()
    w2 = SeasonalEmphasisWeighter(seasonality=2, emphasis=2.0).compute_weights(t).to_list()
    w3 = SeasonalEmphasisWeighter(seasonality=3, emphasis=2.0).compute_weights(t).to_list()

    expected_union = [2.0 if (a == 2.0 or b == 2.0) else 1.0 for a, b in zip(w2, w3, strict=True)]
    assert w_list == pytest.approx(expected_union)
    # The union strictly differs from either single period, exercising the loop.
    assert w_list != w2
    assert w_list != w3


def test_seasonal_emphasis_matches_legacy_values() -> None:
    """SeasonalEmphasisWeighter emphasizes in-phase positions."""
    t = pl.Series(
        "time",
        [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 8), datetime(2024, 1, 9)],
    )
    weights = SeasonalEmphasisWeighter(seasonality=7, emphasis=2.0).compute_weights(t).to_list()
    assert weights == pytest.approx([1.0, 1.0, 1.0, 2.0])


def test_seasonal_emphasis_anchors_to_most_recent_unsorted() -> None:
    """The emphasized phase is the most-recent timestamp's, regardless of order."""
    # Most-recent timestamp (2024-01-09) is in the first position, not the last.
    t = pl.Series(
        "time",
        [datetime(2024, 1, 9), datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 8)],
    )
    weights = SeasonalEmphasisWeighter(seasonality=7, emphasis=2.0).compute_weights(t).to_list()
    # Phase is rank-based; with 4 distinct timestamps only the most-recent (rank 3,
    # here in the first position) matches its own phase and is emphasized.
    assert weights == pytest.approx([2.0, 1.0, 1.0, 1.0])


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


def test_table_weighter_explicit_null_weight_raises(times: pl.Series) -> None:
    """A key present in the frame but with an explicit null weight raises ValueError."""
    frame = pl.DataFrame({"time": times, "weight": [0.5, None, 0.5]})
    with pytest.raises(ValueError, match="explicit null"):
        TableWeighter(frame=frame, on="time").compute_weights(times)


def test_table_weighter_panel_group_column(times: pl.Series) -> None:
    """Group-specific weight columns are used for panel data.

    The per-group column uses the library-wide ``group__column`` double
    underscore separator (regression for the 2026-06-21 QA finding, which
    used a single underscore ``{group}_weight``).
    """
    frame = pl.DataFrame({"time": times, "A__weight": [1.0, 2.0, 3.0], "weight": [9.0, 9.0, 9.0]})
    weights = TableWeighter(frame=frame, on="time").compute_weights(times, group_name="A").to_list()
    assert weights == pytest.approx([1.0, 2.0, 3.0])

    single_underscore = pl.DataFrame({"time": times, "A_weight": [1.0, 2.0, 3.0], "weight": [9.0, 9.0, 9.0]})
    fallback = TableWeighter(frame=single_underscore, on="time").compute_weights(times, group_name="A").to_list()
    assert fallback == pytest.approx([9.0, 9.0, 9.0])


def test_composite_multiplies_components(times: pl.Series) -> None:
    """CompositeWeighter multiplies component outputs element-wise."""
    a = ExponentialDecayWeighter(half_life=2)
    b = SeasonalEmphasisWeighter(seasonality=2, emphasis=1.5)
    expected = (a.compute_weights(times) * b.compute_weights(times)).to_list()
    got = CompositeWeighter([("a", a), ("b", b)]).compute_weights(times).to_list()
    assert got == pytest.approx(expected)


def test_composite_empty_weighters_raises(times: pl.Series) -> None:
    """An empty (but correctly named-list) weighters argument is rejected."""
    with pytest.raises(ValueError, match="at least one weighter"):
        CompositeWeighter(weighters=[]).compute_weights(times)


def test_base_weighter_is_abstract() -> None:
    """BaseWeighter cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseWeighter()  # type: ignore[abstract]


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


def test_linear_decay_single_key_returns_one() -> None:
    """A length-1 key yields weight 1.0 (no rank range to decay over)."""
    weights = LinearDecayWeighter().compute_weights(pl.Series("time", [datetime(2024, 1, 1)])).to_list()
    assert weights == pytest.approx([1.0])


def test_seasonal_emphasis_single_key_returns_one() -> None:
    """A length-1 key yields weight 1.0 (no phase to emphasize)."""
    weights = SeasonalEmphasisWeighter(seasonality=7).compute_weights(pl.Series("time", [datetime(2024, 1, 1)]))
    assert weights.to_list() == pytest.approx([1.0])


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
    """A frame missing both '{group}__weight' and 'weight' raises for panel data."""
    frame = pl.DataFrame({"time": times, "other": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="missing both 'C__weight' and 'weight'"):
        TableWeighter(frame=frame, on="time").compute_weights(times, group_name="C")


def test_composite_set_params_replaces_weighters_list() -> None:
    """set_params replaces the whole weighters list."""
    w = CompositeWeighter([("decay", ExponentialDecayWeighter(1))])
    new = [("lin", LinearDecayWeighter(2))]
    w.set_params(weighters=new)
    assert w.weighters is new


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


def test_composite_mean_zero_coefficient_sum_raises(times: pl.Series) -> None:
    """combination='mean' with zero-sum coefficients raises rather than returning zeros."""
    a = ExponentialDecayWeighter(half_life=2)
    b = SeasonalEmphasisWeighter(seasonality=2, emphasis=1.5)
    weighter = CompositeWeighter([("a", a), ("b", b)], combination="mean", weights=[0.0, 0.0])
    with pytest.raises(ValueError, match="sum to a nonzero value"):
        weighter.compute_weights(times)


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


class TestCompositeNamedAccess:
    """Sub-weighters are reachable by name.

    No trailing underscore: ``CompositeWeighter`` uses its sub-weighters in place rather
    than cloning them, so this is a view over parameters, not fitted state.
    """

    @staticmethod
    def _parts():
        return ExponentialDecayWeighter(half_life=5), SeasonalEmphasisWeighter(seasonality=7)

    def test_returns_each_sub_weighter_under_its_name(self):
        decay, seasonal = self._parts()
        composite = CompositeWeighter(weighters=[("decay", decay), ("seasonal", seasonal)])
        assert set(composite.named_weighters) == {"decay", "seasonal"}

    def test_does_not_require_fit_and_returns_the_same_objects(self):
        """The missing underscore claims in-place semantics; this pins that claim."""
        decay, seasonal = self._parts()
        composite = CompositeWeighter(weighters=[("decay", decay), ("seasonal", seasonal)])
        assert composite.named_weighters["decay"] is decay
        assert composite.named_weighters["seasonal"] is seasonal

    def test_repeated_names_are_rejected(self):
        decay, seasonal = self._parts()
        composite = CompositeWeighter(weighters=[("dup", decay), ("dup", seasonal)])
        with pytest.raises(ValueError, match="not unique"):
            _ = composite.named_weighters

    def test_distinct_names_still_validate(self):
        decay, seasonal = self._parts()
        CompositeWeighter(weighters=[("decay", decay), ("seasonal", seasonal)])._check_weighters()
