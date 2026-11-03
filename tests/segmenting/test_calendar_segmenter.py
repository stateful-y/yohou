"""Tests for CalendarSegmenter and the BaseSegmenter contract."""

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from yohou.segmenting import BaseSegmenter, CalendarSegmenter, ComplementSegmenter


@pytest.fixture
def hourly_day() -> pl.Series:
    """Twenty-four hourly timestamps covering 2024-01-01."""
    return pl.Series("time", [datetime(2024, 1, 1, h) for h in range(24)])


class TestBaseContract:
    """Tests for the segmenter estimator contract."""

    def test_is_base_segmenter(self):
        """Test CalendarSegmenter is a BaseSegmenter."""
        assert isinstance(CalendarSegmenter(), BaseSegmenter)

    def test_labels_aligned(self, hourly_day):
        """Test one label per input element."""
        labels = CalendarSegmenter().compute_labels(hourly_day)
        assert isinstance(labels, pl.Series)
        assert len(labels) == len(hourly_day)
        assert labels.to_list() == list(range(24))

    def test_fit_is_noop(self, hourly_day):
        """Test fit returns self and leaves parameters unchanged."""
        seg = CalendarSegmenter("hour", values=[1, 2])
        before = seg.get_params()
        assert seg.fit(hourly_day) is seg
        assert seg.get_params() == before

    def test_tunable_through_host(self):
        """Test a host routes segmenter__values to the segmenter."""
        host = ComplementSegmenter(CalendarSegmenter("hour", values=[8]))
        host.set_params(segmenter__values=[9, 10])
        assert host.segmenter.values == [9, 10]
        assert host.get_params(deep=True)["segmenter__values"] == [9, 10]

    def test_two_halves_equal_whole(self, hourly_day):
        """Test labelling consecutive parts gives the same labels as labelling the whole."""
        seg = CalendarSegmenter("hour", values=range(8, 19))
        whole = seg.compute_labels(hourly_day)
        parts = pl.concat([seg.compute_labels(hourly_day[:10]), seg.compute_labels(hourly_day[10:])])
        assert whole.equals(parts)


class TestCalendarLabels:
    """Tests for the labels CalendarSegmenter produces."""

    def test_hour_window(self, hourly_day):
        """Test rows from 08:00 to 18:00 carry their hour, others are null."""
        labels = CalendarSegmenter("hour", values=list(range(8, 19))).compute_labels(hourly_day).to_list()
        assert labels == [h if 8 <= h <= 18 else None for h in range(24)]

    def test_no_values_means_all_inside(self):
        """Test values=None labels every row with its weekday, Monday being 1."""
        week = pl.Series("time", [datetime(2024, 1, 1) + timedelta(days=d) for d in range(7)])  # Monday first
        labels = CalendarSegmenter("day_of_week").compute_labels(week)
        assert labels.null_count() == 0
        assert labels.to_list() == [1, 2, 3, 4, 5, 6, 7]

    def test_window_across_midnight(self, hourly_day):
        """Test a list spanning midnight selects 22, 23, 0 and 1."""
        labels = CalendarSegmenter("hour", values=[22, 23, 0, 1]).compute_labels(hourly_day)
        assert [h for h, lab in enumerate(labels.to_list()) if lab is not None] == [0, 1, 22, 23]

    def test_range_equals_list(self, hourly_day):
        """Test a range and the equivalent list give the same labels."""
        from_range = CalendarSegmenter("hour", values=range(8, 19)).compute_labels(hourly_day)
        from_list = CalendarSegmenter("hour", values=list(range(8, 19))).compute_labels(hourly_day)
        assert from_range.equals(from_list)

    def test_empty_values_means_all_outside(self, hourly_day):
        """Test values=[] puts every row outside."""
        labels = CalendarSegmenter("hour", values=[]).compute_labels(hourly_day)
        assert labels.null_count() == len(hourly_day)

    def test_boolean_feature(self):
        """Test a boolean feature labels inside rows 1."""
        days = pl.Series("time", [datetime(2024, 1, 5), datetime(2024, 1, 6)])  # Friday, Saturday
        assert CalendarSegmenter("is_weekend", values=[1]).compute_labels(days).to_list() == [None, 1]

    def test_null_timestamp_is_outside(self):
        """Test a null timestamp gets a null label."""
        times = pl.Series("time", [datetime(2024, 1, 1, 9), None])
        assert CalendarSegmenter("hour").compute_labels(times).to_list() == [9, None]


class TestCalendarValidation:
    """Tests for the errors CalendarSegmenter raises."""

    def test_hour_24_rejected(self, hourly_day):
        """Test an hour outside 0 to 23 is named with the range."""
        with pytest.raises(ValueError, match=r"\[24\].*0 to 23"):
            CalendarSegmenter("hour", values=[23, 24]).compute_labels(hourly_day)

    def test_non_integer_rejected(self, hourly_day):
        """Test a float value is rejected and named."""
        with pytest.raises(ValueError, match=r"8\.5"):
            CalendarSegmenter("hour", values=[8.5]).compute_labels(hourly_day)

    def test_unknown_feature_rejected(self, hourly_day):
        """Test an unknown feature name is rejected."""
        with pytest.raises(ValueError, match="feature"):
            CalendarSegmenter("fortnight").compute_labels(hourly_day)

    def test_integer_steps_rejected(self):
        """Test integer forecasting steps are rejected."""
        with pytest.raises(ValueError, match="Date or Datetime"):
            CalendarSegmenter("hour").compute_labels(pl.Series([1, 2, 3]))

    def test_hour_on_dates_rejected(self):
        """Test reading an hour from a Date series is rejected."""
        with pytest.raises(ValueError, match="time of day"):
            CalendarSegmenter("hour").compute_labels(pl.Series("time", [date(2024, 1, 1)]))

    def test_day_feature_on_dates(self):
        """Test day-level features work on Date series."""
        days = pl.Series("time", [date(2024, 1, 1), date(2024, 1, 6)])
        assert CalendarSegmenter("day_of_week", values=[6]).compute_labels(days).to_list() == [None, 6]


class TestCalendarTimeZone:
    """Tests for local-time evaluation."""

    def test_local_hour_from_utc(self):
        """Test 18:00 UTC in winter is 12:00 in Chicago."""
        times = pl.Series("time", [datetime(2024, 1, 15, 17), datetime(2024, 1, 15, 18)]).dt.replace_time_zone("UTC")
        labels = CalendarSegmenter("hour", values=[12], time_zone="America/Chicago").compute_labels(times)
        assert labels.to_list() == [None, 12]

    def test_input_not_modified(self):
        """Test the input series keeps its zone and values."""
        times = pl.Series("time", [datetime(2024, 1, 15, 18)]).dt.replace_time_zone("UTC")
        original = times.clone()
        CalendarSegmenter("hour", time_zone="America/Chicago").compute_labels(times)
        assert times.equals(original)
        assert times.dtype == pl.Datetime("us", "UTC")

    def test_naive_with_time_zone_rejected(self, hourly_day):
        """Test a time zone on naive timestamps is rejected."""
        with pytest.raises(ValueError, match="timezone-aware"):
            CalendarSegmenter("hour", time_zone="America/Chicago").compute_labels(hourly_day)
