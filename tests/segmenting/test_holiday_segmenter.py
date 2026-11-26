"""Tests for HolidaySegmenter."""

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from yohou.segmenting import CalendarSegmenter, ComplementSegmenter, HolidaySegmenter, IntersectionSegmenter


@pytest.fixture
def christmas() -> pl.DataFrame:
    """Holiday table holding 2024-12-25."""
    return pl.DataFrame({"date": [date(2024, 12, 25)]})


@pytest.fixture
def hourly_christmas_week() -> pl.Series:
    """Hourly timestamps from 2024-12-24 00:00 to 2024-12-26 23:00."""
    return pl.Series("time", [datetime(2024, 12, 24) + timedelta(hours=h) for h in range(72)])


class TestLabels:
    """Tests for the labels HolidaySegmenter assigns."""

    def test_christmas_on_hourly_data(self, christmas, hourly_christmas_week):
        """The 24 rows dated 2024-12-25 are True and every other row is null."""
        labels = HolidaySegmenter(holidays=christmas).compute_labels(hourly_christmas_week)
        on_christmas = hourly_christmas_week.dt.date() == date(2024, 12, 25)
        assert labels.filter(on_christmas).to_list() == [True] * 24
        assert labels.filter(~on_christmas).null_count() == 48

    def test_local_date(self, christmas):
        """2024-12-24 16:00 UTC is 2024-12-25 01:00 in Tokyo, so it is a holiday there."""
        times = pl.Series("time", [datetime(2024, 12, 24, 16)]).dt.replace_time_zone("UTC")
        assert HolidaySegmenter(holidays=christmas).compute_labels(times).to_list() == [None]
        local = HolidaySegmenter(holidays=christmas, time_zone="Asia/Tokyo").compute_labels(times)
        assert local.to_list() == [True]

    def test_date_keys(self, christmas):
        """A Date series is matched directly."""
        days = pl.date_range(date(2024, 12, 24), date(2024, 12, 26), eager=True).alias("time")
        assert HolidaySegmenter(holidays=christmas).compute_labels(days).to_list() == [None, True, None]

    def test_date_keys_with_time_zone_rejected(self, christmas):
        """A Date series has no source zone to convert from."""
        days = pl.date_range(date(2024, 12, 24), date(2024, 12, 26), eager=True).alias("time")
        with pytest.raises(ValueError, match="timezone-aware"):
            HolidaySegmenter(holidays=christmas, time_zone="Europe/Paris").compute_labels(days)

    def test_naive_timestamps_with_time_zone_rejected(self, christmas, hourly_christmas_week):
        """Naive timestamps cannot be converted to a local date."""
        with pytest.raises(ValueError, match="timezone-aware"):
            HolidaySegmenter(holidays=christmas, time_zone="Europe/Paris").compute_labels(hourly_christmas_week)

    def test_integer_keys_rejected(self, christmas):
        """Integer forecasting steps carry no date."""
        with pytest.raises(ValueError, match="Date or Datetime"):
            HolidaySegmenter(holidays=christmas).compute_labels(pl.Series("time", [1, 2, 3]))

    def test_holidays_excluded_from_business_hours(self, christmas):
        """A holiday at 10:00 is outside business hours that exclude holidays."""
        business = IntersectionSegmenter([
            ("hours", CalendarSegmenter("hour", values=list(range(8, 19)))),
            ("not_holiday", ComplementSegmenter(HolidaySegmenter(holidays=christmas))),
        ])
        times = pl.Series("time", [datetime(2024, 12, 25, 10), datetime(2024, 12, 24, 10)])
        assert business.compute_labels(times).to_list() == [None, {"hours": 10, "not_holiday": True}]

    def test_datetime_holiday_column(self, hourly_christmas_week):
        """A Datetime 'date' column is matched by its date."""
        holidays = pl.DataFrame({"date": [datetime(2024, 12, 25)]})
        labels = HolidaySegmenter(holidays=holidays).compute_labels(hourly_christmas_week)
        assert labels.drop_nulls().len() == 24

    def test_other_columns_ignored(self, hourly_christmas_week):
        """A holiday name column does not change the labels."""
        holidays = pl.DataFrame({"date": [date(2024, 12, 25)], "name": ["Christmas"]})
        labels = HolidaySegmenter(holidays=holidays).compute_labels(hourly_christmas_week)
        assert set(labels.drop_nulls().to_list()) == {True}


class TestHolidayTableValidation:
    """Tests for the holiday table checks, which match HolidayFeatureTransformer's."""

    def test_missing_table(self, hourly_christmas_week):
        """holidays=None raises when labels are computed."""
        with pytest.raises(ValueError, match="got None"):
            HolidaySegmenter().compute_labels(hourly_christmas_week)

    def test_wrong_column(self, hourly_christmas_week):
        """A table without a 'date' column raises naming it."""
        with pytest.raises(ValueError, match="'date' column"):
            HolidaySegmenter(holidays=pl.DataFrame({"day": [date(2024, 12, 25)]})).compute_labels(hourly_christmas_week)

    def test_not_a_dataframe(self, hourly_christmas_week):
        """A list of dates raises asking for a polars DataFrame."""
        with pytest.raises(ValueError, match="must be a polars DataFrame, got list"):
            HolidaySegmenter(holidays=[date(2024, 12, 25)]).compute_labels(hourly_christmas_week)

    def test_wrong_date_dtype(self, hourly_christmas_week):
        """A string 'date' column raises asking for Date or Datetime."""
        with pytest.raises(ValueError, match="must be Date or Datetime type"):
            HolidaySegmenter(holidays=pl.DataFrame({"date": ["2024-12-25"]})).compute_labels(hourly_christmas_week)
