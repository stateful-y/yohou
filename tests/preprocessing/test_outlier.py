"""Tests for outlier handling transformers.

Tests OutlierThresholdHandler and OutlierPercentileHandler basic functionality.
"""

import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from conftest import run_checks
from yohou.preprocessing import (
    OutlierPercentileHandler,
    OutlierThresholdHandler,
)
from yohou.segmenting import CalendarSegmenter, IntersectionSegmenter, UnionSegmenter
from yohou.testing import _yield_yohou_transformer_checks


class TestOutlierThresholdHandler:
    """Test OutlierThresholdHandler functionality."""

    def test_systematic_checks(self, time_series_train_test_factory):
        """Run all systematic checks for OutlierThresholdHandler."""
        transformer = OutlierThresholdHandler(low=-3.0, high=3.0, strategy="clip")

        X_train, X_test = time_series_train_test_factory(train_length=100, test_length=50)

        transformer.fit(X_train)

        run_checks(
            transformer,
            _yield_yohou_transformer_checks(transformer, X_train, None, X_test),
        )

    def test_clip_strategy(self, time_series_with_outliers_factory):
        """Test OutlierThresholdHandler clip strategy."""
        X = time_series_with_outliers_factory(length=50, n_components=2, outlier_magnitude=10.0)
        handler = OutlierThresholdHandler(low=-5.0, high=5.0, strategy="clip")
        handler.fit(X)
        X_clipped = handler.transform(X)

        # Check all values are within bounds
        assert "time" in X_clipped.columns
        for col in X_clipped.columns:
            if col != "time":
                assert X_clipped[col].min() >= -5.0
                assert X_clipped[col].max() <= 5.0

    def test_nan_strategy(self, time_series_with_outliers_factory):
        """Test OutlierThresholdHandler nan strategy nulls out-of-bound values."""
        X = time_series_with_outliers_factory(length=50, n_components=2, outlier_magnitude=10.0)
        handler = OutlierThresholdHandler(low=-5.0, high=5.0, strategy="nan")
        handler.fit(X)
        X_nan = handler.transform(X)

        assert "time" in X_nan.columns
        # Outliers (magnitude 10 > threshold 5) must be replaced by nulls, so the
        # input (which had no nulls) gains at least one null after transform.
        feature_cols = [c for c in X_nan.columns if c != "time"]
        n_nulls = X_nan.select(feature_cols).null_count().sum_horizontal().sum()
        assert n_nulls > 0, "nan strategy should null at least one out-of-bound value"

    def test_inverted_thresholds_leave_unfitted(self, time_series_with_outliers_factory):
        """A failed fit (low > high) must not assign threshold attributes."""
        X = time_series_with_outliers_factory(length=50, n_components=2)
        handler = OutlierThresholdHandler(low=5.0, high=-5.0, strategy="clip")

        with pytest.raises(ValueError, match="must be <= high"):
            handler.fit(X)

        # The ordering is validated before any fitted threshold attribute is
        # assigned, so transform cannot silently apply inverted thresholds.
        assert not hasattr(handler, "low_")
        assert not hasattr(handler, "high_")
        with pytest.raises(AttributeError):
            handler.transform(X)


class TestOutlierPercentileHandler:
    """Test OutlierPercentileHandler functionality."""

    def test_systematic_checks(self, time_series_train_test_factory):
        """Run all systematic checks for OutlierPercentileHandler."""
        transformer = OutlierPercentileHandler(low=5, high=95, strategy="clip")

        X_train, X_test = time_series_train_test_factory(train_length=100, test_length=50)

        transformer.fit(X_train)

        run_checks(
            transformer,
            _yield_yohou_transformer_checks(transformer, X_train, None, X_test),
        )

    def test_clip_strategy(self, time_series_with_outliers_factory):
        """Test OutlierPercentileHandler clip strategy."""
        X = time_series_with_outliers_factory(length=50, n_components=2, outlier_magnitude=10.0)
        handler = OutlierPercentileHandler(low=5.0, high=95.0, strategy="clip")
        handler.fit(X)
        X_clipped = handler.transform(X)

        assert "time" in X_clipped.columns

    def test_nan_strategy(self, time_series_with_outliers_factory):
        """Test OutlierPercentileHandler nan strategy nulls out-of-bound values."""
        X = time_series_with_outliers_factory(length=50, n_components=2, outlier_magnitude=10.0)
        handler = OutlierPercentileHandler(low=5.0, high=95.0, strategy="nan")
        handler.fit(X)
        X_nan = handler.transform(X)

        assert "time" in X_nan.columns
        # Values beyond the 5th/95th percentile must be replaced by nulls, so the
        # null-free input gains at least one null after transform.
        feature_cols = [c for c in X_nan.columns if c != "time"]
        n_nulls = X_nan.select(feature_cols).null_count().sum_horizontal().sum()
        assert n_nulls > 0, "nan strategy should null at least one out-of-bound value"

    def test_inverted_percentiles_leave_unfitted(self, time_series_with_outliers_factory):
        """A failed fit (low > high) must not assign threshold attributes.

        Mirrors test_inverted_thresholds_leave_unfitted for the threshold variant:
        the percentile ordering guard (outlier.py) raises before thresholds_ is set.
        """
        X = time_series_with_outliers_factory(length=50, n_components=2)
        handler = OutlierPercentileHandler(low=95.0, high=5.0, strategy="clip")

        with pytest.raises(ValueError, match="must be <= high"):
            handler.fit(X)

        # Ordering is validated before thresholds_ is assigned, so transform
        # cannot silently apply inverted percentile bounds.
        assert not hasattr(handler, "thresholds_")
        with pytest.raises(AttributeError):
            handler.transform(X)


def _hourly(days: int, value_fn, start=None, columns=("value",)) -> pl.DataFrame:
    """Hourly frame over ``days`` days whose value columns are ``value_fn(timestamp)``."""
    start = start or datetime(2024, 1, 1)  # a Monday
    times = [start + timedelta(hours=h) for h in range(24 * days)]
    return pl.DataFrame({"time": times, **{c: [value_fn(t) for t in times] for c in columns}})


def _business_hours() -> CalendarSegmenter:
    return CalendarSegmenter("hour", values=list(range(8, 19)))


class TestSegmentMask:
    """Tests for the segmenter parameter on both handlers."""

    def test_rows_outside_pass_through(self):
        """Test only in-segment rows are handled by the threshold handler."""
        X = pl.DataFrame({"time": [datetime(2024, 1, 1, 3), datetime(2024, 1, 1, 12)], "value": [500.0, 500.0]})
        handler = OutlierThresholdHandler(high=100.0, segmenter=_business_hours()).fit(X)
        assert handler.transform(X)["value"].to_list() == [500.0, 100.0]

    def test_nan_strategy_outside_untouched(self):
        """Test the nan strategy also leaves rows outside the segment alone."""
        X = pl.DataFrame({"time": [datetime(2024, 1, 1, 3), datetime(2024, 1, 1, 12)], "value": [500.0, 500.0]})
        handler = OutlierThresholdHandler(high=100.0, strategy="nan", segmenter=_business_hours()).fit(X)
        assert handler.transform(X)["value"].to_list() == [500.0, None]

    @pytest.mark.parametrize("handler_cls", [OutlierThresholdHandler, OutlierPercentileHandler])
    def test_no_segmenter_keeps_behaviour(self, handler_cls):
        """Test segmenter=None gives the same result as the handler always gave."""
        rng = np.random.default_rng(0)
        X = _hourly(5, lambda t: 0.0).with_columns(pl.Series("value", rng.normal(size=120)))
        kwargs = {"low": -1.0, "high": 1.0} if handler_cls is OutlierThresholdHandler else {"low": 5, "high": 95}
        out = handler_cls(**kwargs).fit(X).transform(X)["value"].to_numpy()
        lo, hi = (-1.0, 1.0) if handler_cls is OutlierThresholdHandler else np.percentile(X["value"], [5, 95])
        np.testing.assert_allclose(out, np.clip(X["value"].to_numpy(), lo, hi))

    def test_segmenter_cloned_at_fit(self):
        """Test changing the segmenter after fit does not change transform."""
        X = pl.DataFrame({"time": [datetime(2024, 1, 1, 3), datetime(2024, 1, 1, 12)], "value": [500.0, 500.0]})
        handler = OutlierThresholdHandler(high=100.0, segmenter=_business_hours()).fit(X)
        handler.segmenter.set_params(values=[3])
        assert handler.segmenter_ is not handler.segmenter
        assert handler.transform(X)["value"].to_list() == [500.0, 100.0]

    def test_percentiles_from_segment_only(self):
        """Test pooled thresholds use hours 8 to 18 only."""
        X = _hourly(20, lambda t: float(t.hour * 10 + t.day))
        handler = OutlierPercentileHandler(high=95, segmenter=_business_hours()).fit(X)
        daytime = X.filter(pl.col("time").dt.hour().is_between(8, 18))["value"].to_numpy()
        assert handler.thresholds_["value"] == (None, pytest.approx(np.percentile(daytime, 95)))

    def test_nan_values_not_counted(self):
        """Test NaN rows do not change the fitted thresholds."""
        X = _hourly(20, lambda t: float(t.hour * 10 + t.day))
        X_nan = X.with_columns(
            pl.when(pl.col("time").dt.day() == 3).then(float("nan")).otherwise(pl.col("value")).alias("value")
        )
        with_nan = OutlierPercentileHandler(high=95, segmenter=_business_hours()).fit(X_nan).thresholds_
        manual = np.percentile(
            X_nan.filter(pl.col("time").dt.hour().is_between(8, 18))["value"].drop_nans().to_numpy(), 95
        )
        assert with_nan["value"] == (None, pytest.approx(manual))

    def test_empty_segment_warns_and_passes_through(self):
        """Test an hour segment on daily data warns at fit and leaves values unchanged."""
        X = pl.DataFrame({"time": [datetime(2024, 1, d) for d in range(1, 11)], "value": [float(d) for d in range(10)]})
        handler = OutlierPercentileHandler(high=95, segmenter=CalendarSegmenter("hour", values=[12]))
        with pytest.warns(UserWarning, match="selects no finite value of column 'value'"):
            handler.fit(X)
        assert handler.transform(X)["value"].to_list() == X["value"].to_list()


class TestBySegment:
    """Tests for per-segment percentile thresholds."""

    @staticmethod
    def _daily_cycle(days: int = 365) -> pl.DataFrame:
        rng = np.random.default_rng(1)
        X = _hourly(days, lambda t: 60.0 if t.hour < 6 or t.hour > 20 else 300.0)
        return X.with_columns(pl.col("value") + pl.Series(rng.normal(scale=5.0, size=len(X))))

    def test_same_value_different_hours(self):
        """Test 150 is clipped at 03:00 and kept at 12:00."""
        handler = OutlierPercentileHandler(high=95, segmenter=CalendarSegmenter("hour"), by_segment=True)
        handler.fit(self._daily_cycle())
        X_new = pl.DataFrame({"time": [datetime(2025, 6, 2, 3), datetime(2025, 6, 2, 12)], "value": [150.0, 150.0]})
        out = handler.transform(X_new)["value"].to_list()
        assert out[0] == pytest.approx(handler.segment_thresholds_["value"][3][1])
        assert out[0] < 100.0
        assert out[1] == 150.0

    def test_per_label_thresholds_match_numpy(self):
        """Test each label's thresholds are the percentiles of its own rows."""
        X = self._daily_cycle(60)
        handler = OutlierPercentileHandler(low=5, high=95, segmenter=CalendarSegmenter("hour"), by_segment=True).fit(X)
        assert set(handler.segment_thresholds_["value"]) == set(range(24))
        for hour in (3, 12):
            rows = X.filter(pl.col("time").dt.hour() == hour)["value"].to_numpy()
            expected = tuple(np.percentile(rows, [5, 95]))
            assert handler.segment_thresholds_["value"][hour] == pytest.approx(expected)

    def test_columns_keep_separate_thresholds(self):
        """Test two columns get their own per-label thresholds."""
        X = _hourly(60, lambda t: float(t.hour), columns=("a", "b")).with_columns(pl.col("b") * 10)
        handler = OutlierPercentileHandler(high=95, segmenter=CalendarSegmenter("hour"), by_segment=True).fit(X)
        assert handler.segment_thresholds_["a"][5] == (None, pytest.approx(5.0))
        assert handler.segment_thresholds_["b"][5] == (None, pytest.approx(50.0))

    def test_composite_labels_are_tuples(self):
        """Test struct labels become tuples of field values."""
        seg = IntersectionSegmenter([
            ("days", CalendarSegmenter("day_of_week", values=[1])),
            ("hours", CalendarSegmenter("hour", values=[9])),
        ])
        X = _hourly(7 * 40, lambda t: float(t.day))
        handler = OutlierPercentileHandler(high=95, segmenter=seg, by_segment=True).fit(X)
        assert list(handler.segment_thresholds_["value"]) == [(1, 9)]

    def test_requires_segmenter(self):
        """Test by_segment without a segmenter fails at fit."""
        with pytest.raises(ValueError, match="requires a segmenter"):
            OutlierPercentileHandler(high=95, by_segment=True).fit(_hourly(2, lambda t: 1.0))

    def test_no_limits_is_silent_noop(self):
        """Test no low/high gives (None, None) per label, no warning, unchanged output."""
        X = _hourly(2, lambda t: float(t.hour))
        handler = OutlierPercentileHandler(segmenter=CalendarSegmenter("hour"), by_segment=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            handler.fit(X)
            out = handler.transform(X)
        assert set(handler.segment_thresholds_["value"].values()) == {(None, None)}
        assert out["value"].to_list() == X["value"].to_list()

    def test_unseen_label_uses_pooled(self):
        """Test a June row after January-to-March training uses the pooled thresholds, with no warning."""
        X = _hourly(31 + 29 + 31, lambda t: float(t.hour))  # 2024-01-01 to 2024-03-31, contiguous
        handler = OutlierPercentileHandler(high=95, segmenter=CalendarSegmenter("month"), by_segment=True)
        X_june = pl.DataFrame({"time": [datetime(2024, 6, 3, 12)], "value": [1000.0]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            handler.fit(X)
            out = handler.transform(X_june)
        assert set(handler.segment_thresholds_["value"]) == {1, 2, 3}
        assert out["value"].to_list() == [pytest.approx(handler.thresholds_["value"][1])]

    def test_label_below_min_samples_warns_and_falls_back(self):
        """Test two weeks of daily data leave every weekday below min_samples."""
        X = pl.DataFrame({
            "time": [datetime(2024, 1, 1) + timedelta(days=d) for d in range(14)],
            "value": [float(d) for d in range(14)],
        })
        handler = OutlierPercentileHandler(
            high=95, segmenter=CalendarSegmenter("day_of_week"), by_segment=True, min_samples=30
        )
        with pytest.warns(UserWarning, match=r"7 \(column, label\) pairs .* min_samples=30") as record:
            handler.fit(X)
        assert len(record) == 1
        assert set(handler.segment_thresholds_["value"].values()) == {handler.thresholds_["value"]}

    def test_fallback_is_per_column(self):
        """Test only the column short on Monday data falls back for Monday."""
        X = _hourly(7 * 10, lambda t: float(t.hour), columns=("a", "b"))
        X = X.with_columns(pl.when(pl.col("time").dt.weekday() == 1).then(None).otherwise(pl.col("b")).alias("b"))
        handler = OutlierPercentileHandler(high=95, segmenter=CalendarSegmenter("day_of_week"), by_segment=True)
        with pytest.warns(UserWarning, match=r"1 \(column, label\) pairs .*: b/1$"):
            handler.fit(X)
        assert handler.segment_thresholds_["b"][1] == handler.thresholds_["b"]
        monday_a = X.filter(pl.col("time").dt.weekday() == 1)["a"].to_numpy()
        assert handler.segment_thresholds_["a"][1] == (None, pytest.approx(np.percentile(monday_a, 95)))

    def test_warning_names_at_most_ten_pairs(self):
        """Test the warning counts every pair but lists only ten."""
        X = _hourly(1, lambda t: float(t.hour))
        handler = OutlierPercentileHandler(high=95, segmenter=CalendarSegmenter("hour"), by_segment=True)
        with pytest.warns(UserWarning, match=r"24 \(column, label\) pairs") as record:
            handler.fit(X)
        assert str(record[0].message).endswith(", ...")
        assert str(record[0].message).count("value/") == 10

    def test_chunked_transform_matches_full(self):
        """Test transforming in two chunks equals transforming whole."""
        seg = CalendarSegmenter("hour", values=list(range(6, 22)), time_zone="Europe/Paris")
        X = self._daily_cycle(40).with_columns(pl.col("time").dt.replace_time_zone("UTC"))
        handler = OutlierPercentileHandler(low=5, high=95, segmenter=seg, by_segment=True).fit(X)
        whole = handler.transform(X)
        chunks = pl.concat([handler.transform(X[:500]), handler.transform(X[500:])])
        assert whole.equals(chunks)

    @pytest.mark.parametrize("by_segment", [False, True])
    def test_percentile_rows_outside_pass_through(self, by_segment):
        """Test the percentile handler leaves rows outside the segment unchanged and clips inside rows."""
        handler = OutlierPercentileHandler(high=95, segmenter=_business_hours(), by_segment=by_segment, min_samples=5)
        handler.fit(self._daily_cycle(30))
        X_new = pl.DataFrame({"time": [datetime(2025, 6, 2, 3), datetime(2025, 6, 2, 12)], "value": [5000.0, 5000.0]})
        out = handler.transform(X_new)["value"].to_list()
        assert out[0] == 5000.0
        assert out[1] < 5000.0

    def test_refit_without_by_segment_clears_segment_state(self):
        """Test a refit with by_segment=False drops per-segment state from the earlier fit."""
        X = self._daily_cycle(10)
        handler = OutlierPercentileHandler(high=95, segmenter=CalendarSegmenter("hour"), by_segment=True, min_samples=5)
        handler.fit(X)
        assert hasattr(handler, "segment_thresholds_")
        handler.set_params(by_segment=False).fit(X)
        assert not hasattr(handler, "segment_thresholds_")
        assert not hasattr(handler, "_segment_bounds")

    def test_nested_composite_labels(self):
        """Test by_segment over a union containing an intersection keys thresholds by nested tuples."""
        business = IntersectionSegmenter([
            ("days", CalendarSegmenter("day_of_week", values=[1, 2, 3, 4, 5])),
            ("hours", CalendarSegmenter("hour", values=list(range(8, 19)))),
        ])
        seg = UnionSegmenter([("business", business), ("weekend", CalendarSegmenter("day_of_week", values=[6, 7]))])
        X = _hourly(7 * 10, lambda t: float(t.hour))
        handler = OutlierPercentileHandler(high=95, segmenter=seg, by_segment=True, min_samples=5).fit(X)
        keys = set(handler.segment_thresholds_["value"])
        assert ((1, 9), None) in keys
        assert (None, 6) in keys
        X_new = pl.DataFrame({"time": [datetime(2024, 3, 11, 9)], "value": [1000.0]})  # Monday 09:00
        assert handler.transform(X_new)["value"].to_list() == [pytest.approx(9.0)]

    def test_docstring_states_fallback(self):
        """Test the API docstring states which thresholds unseen and small labels use."""
        doc = OutlierPercentileHandler.__doc__
        assert "never saw" in doc
        assert "fewer than ``min_samples`` finite values" in doc
        assert "pooled" in doc


class TestSegmentSystematicChecks:
    """Run the transformer check suite with segment-aware configurations."""

    @pytest.mark.parametrize(
        "transformer",
        [
            OutlierThresholdHandler(low=-1.0, high=1.0, segmenter=CalendarSegmenter("day_of_week", values=[1, 2, 3])),
            OutlierPercentileHandler(
                low=5,
                high=95,
                segmenter=CalendarSegmenter("day_of_week"),
                by_segment=True,
                min_samples=5,
            ),
        ],
        ids=["threshold-mask", "percentile-by-segment"],
    )
    def test_systematic_checks(self, transformer, time_series_train_test_factory):
        """Run all systematic transformer checks with a segmenter configured."""
        X_train, X_test = time_series_train_test_factory(train_length=100, test_length=50)
        transformer.fit(X_train)
        run_checks(transformer, _yield_yohou_transformer_checks(transformer, X_train, None, X_test))
