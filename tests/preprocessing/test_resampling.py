"""Tests for resampling transformers.

Tests Downsampler and Upsampler transformer-specific behavior.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone

from yohou.preprocessing.resampling import Downsampler, Upsampler


def create_hourly_data(length: int = 48, seed: int = 42) -> pl.DataFrame:
    """Create hourly time series data for testing.

    Parameters
    ----------
    length : int
        Number of samples (hours).
    seed : int
        Random seed.

    Returns
    -------
    pl.DataFrame
        DataFrame with time column and value columns.

    """
    np.random.seed(seed)
    time = [datetime(2021, 1, 1) + timedelta(hours=i) for i in range(length)]
    return pl.DataFrame({
        "time": time,
        "value_a": np.cumsum(np.random.randn(length)) + 100,
        "value_b": np.sin(np.linspace(0, 4 * np.pi, length)) * 10 + 50,
    })


def create_daily_data(length: int = 30, seed: int = 42) -> pl.DataFrame:
    """Create daily time series data for testing.

    Parameters
    ----------
    length : int
        Number of samples (days).
    seed : int
        Random seed.

    Returns
    -------
    pl.DataFrame
        DataFrame with time column and value columns.

    """
    np.random.seed(seed)
    time = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(length)]
    return pl.DataFrame({
        "time": time,
        "value_a": np.cumsum(np.random.randn(length)) + 100,
        "value_b": np.sin(np.linspace(0, 4 * np.pi, length)) * 10 + 50,
    })


class TestDownsampler:
    """Tests for Downsampler transformer."""

    def test_downsample_hourly_to_daily(self) -> None:
        """Test downsampling from hourly to daily."""
        X = create_hourly_data(length=48)  # 2 days of data
        downsampler = Downsampler(interval="1d", aggregation="mean")
        downsampler.fit(X)
        X_daily = downsampler.transform(X)

        assert len(X_daily) == 2
        assert "time" in X_daily.columns
        assert "value_a" in X_daily.columns
        assert "value_b" in X_daily.columns

    def test_downsample_aggregation_sum(self) -> None:
        """Test downsampling with sum aggregation."""
        X = create_hourly_data(length=24)
        downsampler = Downsampler(interval="1d", aggregation="sum")
        downsampler.fit(X)
        X_daily = downsampler.transform(X)

        # Sum of 24 hourly values should equal the daily value
        expected_sum_a = X["value_a"].sum()
        actual_sum_a = X_daily["value_a"].sum()
        assert abs(expected_sum_a - actual_sum_a) < 1e-10

    def test_downsample_aggregation_min_max(self) -> None:
        """Test downsampling with min/max aggregations."""
        X = create_hourly_data(length=24)

        downsampler_min = Downsampler(interval="1d", aggregation="min")
        downsampler_min.fit(X)
        X_min = downsampler_min.transform(X)

        downsampler_max = Downsampler(interval="1d", aggregation="max")
        downsampler_max.fit(X)
        X_max = downsampler_max.transform(X)

        # Min should be <= max for all values
        assert X_min["value_a"][0] <= X_max["value_a"][0]
        assert X_min["value_b"][0] <= X_max["value_b"][0]

    def test_downsample_aggregation_first_last(self) -> None:
        """Test downsampling with first/last aggregations."""
        X = create_hourly_data(length=24)

        downsampler_first = Downsampler(interval="1d", aggregation="first")
        downsampler_first.fit(X)
        X_first = downsampler_first.transform(X)

        downsampler_last = Downsampler(interval="1d", aggregation="last")
        downsampler_last.fit(X)
        X_last = downsampler_last.transform(X)

        # First value should match X[0], last should match X[-1]
        assert X_first["value_a"][0] == X["value_a"][0]
        assert X_last["value_a"][0] == X["value_a"][-1]

    def test_downsample_aggregation_median(self) -> None:
        """Test downsampling with median aggregation."""
        X = create_hourly_data(length=24)
        downsampler = Downsampler(interval="1d", aggregation="median")
        downsampler.fit(X)
        X_daily = downsampler.transform(X)

        # Median should be within range of min/max
        assert X_daily["value_a"][0] >= X["value_a"].min()
        assert X_daily["value_a"][0] <= X["value_a"].max()

    def test_downsample_aggregation_std(self) -> None:
        """Test downsampling with std aggregation.

        Unlike the other aggregations, ``std`` describes how much the series moved
        *within* each bin rather than where it sat, which is the only way to keep a
        sub-interval's shape once it has been reduced to a coarser grid.
        """
        times = [datetime(2020, 1, 1) + timedelta(hours=i) for i in range(4)]
        X = pl.DataFrame({"time": times, "value_a": [1.0, 2.0, 3.0, 4.0]})

        downsampler = Downsampler(interval="1d", aggregation="std")
        downsampler.fit(X)
        X_daily = downsampler.transform(X)

        # Sample standard deviation of 1, 2, 3, 4 is sqrt(5/3).
        assert X_daily.height == 1
        assert X_daily["value_a"][0] == pytest.approx((5 / 3) ** 0.5)

    def test_downsample_std_is_null_for_a_single_point_bin(self) -> None:
        """A spread is undefined for one observation, so that bin yields null.

        The frame still needs two time points overall, because ``fit`` infers the input
        interval from them; it is the second *bin* that holds a lone observation.
        """
        times = [datetime(2020, 1, 1), datetime(2020, 1, 1, 1), datetime(2020, 1, 2)]
        X = pl.DataFrame({"time": times, "value_a": [1.0, 2.0, 9.0]})

        downsampler = Downsampler(interval="1d", aggregation="std")
        downsampler.fit(X)
        X_daily = downsampler.transform(X)

        assert X_daily["value_a"][0] == pytest.approx(0.5**0.5)
        assert X_daily["value_a"][1] is None

    def test_downsample_closed_label_options(self) -> None:
        """Test downsampling with different closed/label options."""
        X = create_hourly_data(length=24)

        downsampler_left = Downsampler(interval="1d", closed="left", label="left")
        downsampler_left.fit(X)
        X_left = downsampler_left.transform(X)

        downsampler_right = Downsampler(interval="1d", closed="right", label="right")
        downsampler_right.fit(X)
        X_right = downsampler_right.transform(X)

        # Both should produce output
        assert len(X_left) > 0
        assert len(X_right) > 0

    def test_downsample_preserves_column_order(self) -> None:
        """Test that downsampling preserves column order."""
        X = create_hourly_data(length=24)
        downsampler = Downsampler(interval="1d")
        downsampler.fit(X)
        X_daily = downsampler.transform(X)

        # First column should be time, then data columns in order
        assert X_daily.columns[0] == "time"
        # Data columns follow (may not preserve exact order due to group_by)

    def test_downsample_fit_transform(self) -> None:
        """Test fit_transform convenience method."""
        X = create_hourly_data(length=48)
        downsampler = Downsampler(interval="1d")

        X_daily = downsampler.fit_transform(X)
        assert len(X_daily) == 2

    def test_downsample_clone(self) -> None:
        """Test that Downsampler can be cloned."""
        downsampler = Downsampler(interval="1d", aggregation="sum")
        cloned = clone(downsampler)

        assert cloned.interval == downsampler.interval
        assert cloned.aggregation == downsampler.aggregation

    def test_downsample_rejects_smaller_target(self) -> None:
        """Test that fitting rejects target interval smaller than input."""
        # Create daily data and try to "downsample" to hourly (should fail)
        X = create_daily_data(length=30)
        downsampler = Downsampler(interval="1h", aggregation="mean")

        with pytest.raises(ValueError, match="smaller than input interval"):
            downsampler.fit(X)


class TestUpsampler:
    """Tests for Upsampler transformer."""

    def test_upsample_daily_to_hourly(self) -> None:
        """Test upsampling from daily to hourly."""
        X = create_daily_data(length=3)  # 3 days
        upsampler = Upsampler(interval="12h", interpolation="linear")
        upsampler.fit(X)
        X_12h = upsampler.transform(X)

        # 3 days should become 5 12-hour periods (start to end inclusive)
        # days: 0, 1, 2 -> 12h: 0, 0.5, 1, 1.5, 2 = 5 points
        assert len(X_12h) == 5
        assert "time" in X_12h.columns
        assert "value_a" in X_12h.columns

    def test_upsample_interpolation_linear(self) -> None:
        """Test linear interpolation."""
        time = [datetime(2021, 1, 1), datetime(2021, 1, 3)]  # 2-day gap
        X = pl.DataFrame({
            "time": time,
            "value": [0.0, 2.0],
        })
        upsampler = Upsampler(interval="1d", interpolation="linear")
        upsampler.fit(X)
        X_daily = upsampler.transform(X)

        # Should have day 1, 2, 3 with values 0, 1, 2
        assert len(X_daily) == 3
        assert X_daily["value"][0] == 0.0
        assert X_daily["value"][1] == 1.0  # Linear interpolation
        assert X_daily["value"][2] == 2.0

    def test_upsample_interpolation_forward(self) -> None:
        """Test forward fill interpolation."""
        time = [datetime(2021, 1, 1), datetime(2021, 1, 3)]
        X = pl.DataFrame({
            "time": time,
            "value": [10.0, 20.0],
        })
        upsampler = Upsampler(interval="1d", interpolation="forward")
        upsampler.fit(X)
        X_daily = upsampler.transform(X)

        # Middle value should be forward-filled from first
        assert len(X_daily) == 3
        assert X_daily["value"][0] == 10.0
        assert X_daily["value"][1] == 10.0  # Forward fill
        assert X_daily["value"][2] == 20.0

    def test_upsample_interpolation_backward(self) -> None:
        """Test backward fill interpolation."""
        time = [datetime(2021, 1, 1), datetime(2021, 1, 3)]
        X = pl.DataFrame({
            "time": time,
            "value": [10.0, 20.0],
        })
        upsampler = Upsampler(interval="1d", interpolation="backward")
        upsampler.fit(X)
        X_daily = upsampler.transform(X)

        # Middle value should be backward-filled from last
        assert len(X_daily) == 3
        assert X_daily["value"][0] == 10.0
        assert X_daily["value"][1] == 20.0  # Backward fill
        assert X_daily["value"][2] == 20.0

    def test_upsample_interpolation_nearest_by_distance(self) -> None:
        """'nearest' assigns each gap point the temporally closest anchor."""
        X = pl.DataFrame({
            "time": [datetime(2021, 1, 1), datetime(2021, 1, 5)],
            "value": [10.0, 50.0],
        })
        X_daily = Upsampler(interval="1d", interpolation="nearest").fit(X).transform(X)

        # Jan 1..5; Jan 2 nearer to Jan 1, Jan 4 nearer to Jan 5, Jan 3 ties -> trailing anchor's prev.
        assert X_daily["value"].to_list() != [10.0, 10.0, 10.0, 10.0, 50.0]
        assert X_daily["value"].to_list() == [10.0, 10.0, 10.0, 50.0, 50.0]

    def test_upsample_fit_transform(self) -> None:
        """Test fit_transform convenience method."""
        X = create_daily_data(length=3)
        upsampler = Upsampler(interval="12h")

        X_12h = upsampler.fit_transform(X)
        assert len(X_12h) == 5

    def test_upsample_clone(self) -> None:
        """Test that Upsampler can be cloned."""
        upsampler = Upsampler(interval="1h", interpolation="nearest")
        cloned = clone(upsampler)

        assert cloned.interval == upsampler.interval
        assert cloned.interpolation == upsampler.interpolation

    def test_upsample_rejects_larger_target(self) -> None:
        """Test that fitting rejects target interval larger than input."""
        # Create hourly data and try to "upsample" to daily (should fail)
        X = create_hourly_data(length=48)
        upsampler = Upsampler(interval="1d", interpolation="linear")

        with pytest.raises(ValueError, match="larger than input interval"):
            upsampler.fit(X)

    def test_upsample_empty_series_raises(self) -> None:
        """Transforming an empty series raises a clear error."""
        X = create_hourly_data(length=48)
        upsampler = Upsampler(interval="30m", interpolation="linear")
        upsampler.fit(X)

        with pytest.raises(ValueError, match="empty time series"):
            upsampler.transform(X.head(0))


class TestDownsamplerBoundaries:
    """Tests for Downsampler include_boundaries handling."""

    def test_include_boundaries_dropped_from_output(self) -> None:
        """include_boundaries=True must not leak boundary columns into output."""
        X = create_hourly_data(length=48)
        downsampler = Downsampler(interval="1d", aggregation="mean", include_boundaries=True)
        downsampler.fit(X)
        result = downsampler.transform(X)

        assert "_lower_boundary" not in result.columns
        assert "_upper_boundary" not in result.columns
        assert result.columns == ["time", "value_a", "value_b"]


def create_irregular_subhourly_data() -> pl.DataFrame:
    """Create a jittered, gapped sub-hourly frame that fails the strict interval check.

    Roughly a 5-minute cadence with second-level jitter and one multi-hour gap, so
    ``check_interval_consistency`` rejects it (the gap widens the delta spread past
    the sub-day tolerance) while ``group_by_dynamic`` can still bin it. Deterministic.

    Returns
    -------
    pl.DataFrame
        DataFrame with an irregular "time" column and one "value_a" column.

    """
    steps = [300, 297, 305, 300, 301, 299, 300, 303, 300, 298, 302, 300]
    seconds: list[int] = []
    t = 0
    for step in steps:
        seconds.append(t)
        t += step
    t += 5400  # a ~90 minute gap widens the delta spread past the 1 hour tolerance
    for step in [300, 302, 298, 300, 301, 300, 299, 300, 304, 300, 296, 300]:
        seconds.append(t)
        t += step
    seconds.append(t)
    base = datetime(2021, 1, 1)
    time = [base + timedelta(seconds=s) for s in seconds]
    return pl.DataFrame({"time": time, "value_a": [100.0 + i for i in range(len(time))]})


class TestDownsamplerIrregularGrid:
    """Downsampler accepts a non-uniform (jittered/gapped) input grid at fit and transform."""

    def test_tag_declared(self) -> None:
        """Downsampler opts into the irregular-grid contract; a resampler that does not is False."""
        assert Downsampler().__sklearn_tags__().transformer_tags.accepts_irregular_grid is True
        assert Upsampler().__sklearn_tags__().transformer_tags.accepts_irregular_grid is False

    def test_fixture_is_irregular(self) -> None:
        """Guard the fixture: it really is rejected by the strict interval check."""
        from yohou.utils.validation import check_interval_consistency

        with pytest.raises(ValueError):
            check_interval_consistency(create_irregular_subhourly_data())

    def test_fits_and_bins_irregular_to_hourly(self) -> None:
        """An irregular sub-hourly frame downsamples to the same hourly means as a manual group_by."""
        X = create_irregular_subhourly_data()
        out = Downsampler(interval="1h", aggregation="mean").fit_transform(X).sort("time")
        expected = (
            X
            .group_by(pl.col("time").dt.truncate("1h").alias("time"))
            .agg(value_a=pl.col("value_a").mean())
            .sort("time")
        )
        assert out.height == expected.height
        assert out["value_a"].to_list() == pytest.approx(expected["value_a"].to_list())

    def test_records_representative_interval(self) -> None:
        """On irregular input, the recorded interval is a representative (median) one, not a raise."""
        ds = Downsampler(interval="1h").fit(create_irregular_subhourly_data())
        assert ds.input_interval_str_ == "5m"

    def test_regular_grid_interval_unchanged(self) -> None:
        """On a uniform grid the strict interval is recorded, exactly as before the opt-in."""
        ds = Downsampler(interval="1d").fit(create_hourly_data(length=48))
        assert ds.input_interval_str_ == "1h"

    def test_non_optin_transformer_still_rejects_irregular(self) -> None:
        """A transformer that does not opt in still rejects the same irregular frame at fit."""
        from yohou.preprocessing import LagTransformer

        with pytest.raises(ValueError):
            LagTransformer(lag=[1]).fit(create_irregular_subhourly_data())

    def test_modest_gaps_record_the_frequency_weighted_interval(self) -> None:
        """Outlier gaps must not skew the recorded interval.

        The strict check tolerates a sub-day delta spread and returns the median
        of the *unique* deltas, which a few gaps drag upward: for
        ``{300, 600, 900}`` it returns 10m. The frequency-weighted median is 5m,
        which is what the feed actually is. The strict check succeeds here, so
        gating the robust function behind ``except ValueError`` never reaches it.
        """
        base = datetime(2021, 1, 1)
        seconds: list[int] = []
        t = 0
        for step in [300] * 12 + [600, 900]:
            seconds.append(t)
            t += step
        seconds.append(t)
        frame = pl.DataFrame({
            "time": [base + timedelta(seconds=s) for s in seconds],
            "value_a": [100.0 + i for i in range(len(seconds))],
        })

        ds = Downsampler(interval="10m").fit(frame)
        assert ds.input_interval_str_ == "5m"

    def test_jittered_feed_can_be_binned_onto_its_own_cadence(self) -> None:
        """Regularizing a jittered 5m feed onto a clean 5m grid is accepted.

        This is the advertised use case. It fails when the recorded input
        interval is skewed above the true cadence, because the target >= input
        guard then rejects the target.
        """
        base = datetime(2021, 1, 1)
        seconds: list[int] = []
        t = 0
        for step in [300] * 12 + [600, 900]:
            seconds.append(t)
            t += step
        seconds.append(t)
        frame = pl.DataFrame({
            "time": [base + timedelta(seconds=s) for s in seconds],
            "value_a": [100.0 + i for i in range(len(seconds))],
        })

        out = Downsampler(interval="5m").fit_transform(frame)
        assert "time" in out.columns

    def test_all_identical_timestamps_raise(self) -> None:
        """A frame with no positive interval is rejected, not recorded as 0d."""
        base = datetime(2021, 1, 1)
        frame = pl.DataFrame({"time": [base] * 4, "value_a": [1.0, 2.0, 3.0, 4.0]})

        with pytest.raises(ValueError):
            Downsampler(interval="5m").fit(frame)
