"""Tests for calendar and holiday feature transformers.

Tests CalendarFeatureTransformer and HolidayFeatureTransformer using both
the systematic check generator pattern and transformer-specific tests.
"""

from datetime import date, datetime

import polars as pl
import pytest
from sklearn.base import clone

from conftest import run_checks
from yohou.preprocessing.calendar import CalendarFeatureTransformer, HolidayFeatureTransformer
from yohou.testing import _yield_yohou_transformer_checks


class TestCalendarFeatureTransformerSystematic:
    """Systematic check generator tests for CalendarFeatureTransformer."""

    @pytest.mark.parametrize(
        "transformer,expected_failures",
        [
            (CalendarFeatureTransformer(), []),
            (CalendarFeatureTransformer(features=["month", "day_of_week"]), []),
        ],
        ids=["default", "specific_features"],
    )
    def test_systematic_checks(
        self,
        transformer,
        expected_failures,
        time_series_train_test_factory,
    ):
        """Run all applicable checks for CalendarFeatureTransformer."""
        X_train, X_test = time_series_train_test_factory(
            train_length=60,
            test_length=30,
        )

        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
            expected_failures=set(expected_failures),
        )


class TestCalendarFeatureTransformerFeatures:
    """Tests for CalendarFeatureTransformer feature selection and values."""

    def test_auto_select_daily_data(self):
        """Test auto-selection excludes hour/minute for daily data."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 3, 1), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = CalendarFeatureTransformer()
        transformer.fit(X)

        assert "hour" not in transformer.applicable_features_
        assert "minute" not in transformer.applicable_features_
        assert "month" in transformer.applicable_features_
        assert "day_of_week" in transformer.applicable_features_

    def test_auto_select_hourly_data(self):
        """Test auto-selection includes hour for hourly data."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1h", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = CalendarFeatureTransformer()
        transformer.fit(X)

        assert "hour" in transformer.applicable_features_
        assert "month" in transformer.applicable_features_

    def test_specific_features(self):
        """Test extracting specific features produces correct columns."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 3, 1), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = CalendarFeatureTransformer(features=["month", "day_of_week"])
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert "cal_month" in X_t.columns
        assert "cal_day_of_week" in X_t.columns
        assert "cal_year" not in X_t.columns

    def test_invalid_feature_raises(self):
        """Test unknown feature raises ValueError."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 31), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = CalendarFeatureTransformer(features=["month", "nonexistent"])
        with pytest.raises(ValueError, match="Unknown features"):
            transformer.fit(X)

    def test_incompatible_feature_raises(self):
        """Test hourly feature on daily data raises ValueError."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 3, 1), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = CalendarFeatureTransformer(features=["hour"])
        with pytest.raises(ValueError, match="not applicable"):
            transformer.fit(X)

    def test_correct_month_values(self):
        """Test month values are 1-12."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2021, 1, 1), interval="1mo", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = CalendarFeatureTransformer(features=["month"])
        transformer.fit(X)
        X_t = transformer.transform(X)

        months = X_t["cal_month"].to_list()
        assert months[0] == 1
        assert months[5] == 6
        assert months[11] == 12

    def test_correct_day_of_week_values(self):
        """Test day_of_week uses polars convention (1=Monday to 7=Sunday)."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 6),
            end=datetime(2020, 1, 13),
            interval="1d",
            eager=True,
        )
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = CalendarFeatureTransformer(features=["day_of_week"])
        transformer.fit(X)
        X_t = transformer.transform(X)

        dow = X_t["cal_day_of_week"].to_list()
        assert dow[0] == 1  # Monday

    def test_is_weekend_values(self):
        """Test is_weekend produces 0/1 for weekday/weekend."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 6),
            end=datetime(2020, 1, 13),
            interval="1d",
            eager=True,
        )
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = CalendarFeatureTransformer(features=["is_weekend"])
        transformer.fit(X)
        X_t = transformer.transform(X)

        weekend = X_t["cal_is_weekend"].to_list()
        assert weekend[5] == 1  # Saturday
        assert weekend[6] == 1  # Sunday
        assert weekend[0] == 0  # Monday

    def test_column_name_conflict_raises(self):
        """Test that conflicting column names raise ValueError."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 31), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "cal_month": range(len(time))})

        transformer = CalendarFeatureTransformer(features=["month"])
        with pytest.raises(ValueError, match="conflict"):
            transformer.fit(X)


class TestCalendarFeatureTransformerPanel:
    """Tests for panel data support."""

    def test_panel_data_support(self, panel_time_series_factory):
        """Test transformer handles panel data (prefixed columns)."""
        X = panel_time_series_factory(length=50, n_series=2, n_groups=2)

        transformer = CalendarFeatureTransformer(features=["month", "day_of_week"])
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert "time" in X_t.columns
        assert "cal_month" in X_t.columns
        assert "cal_day_of_week" in X_t.columns


class TestCalendarFeatureTransformerEdgeCases:
    """Edge case tests for CalendarFeatureTransformer."""

    def test_single_row(self):
        """Test transformer works with two-row DataFrame (minimum for interval detection)."""
        X = pl.DataFrame({
            "time": [datetime(2020, 6, 15), datetime(2020, 6, 16)],
            "value": [1.0, 2.0],
        })

        transformer = CalendarFeatureTransformer(features=["month", "quarter"])
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert X_t["cal_month"][0] == 6
        assert X_t["cal_quarter"][0] == 2

    def test_get_feature_names_out(self):
        """Test get_feature_names_out matches transform output."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 31), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = CalendarFeatureTransformer(features=["month", "is_weekend"])
        transformer.fit(X)
        X_t = transformer.transform(X)

        feature_names = transformer.get_feature_names_out()
        assert feature_names == ["cal_month", "cal_is_weekend"]
        for name in feature_names:
            assert name in X_t.columns


class TestHolidayFeatureTransformerSystematic:
    """Systematic check generator tests for HolidayFeatureTransformer."""

    @pytest.fixture()
    def holidays_df(self):
        """Create a sample holidays DataFrame."""
        return pl.DataFrame({
            "date": [date(2021, 1, 1), date(2021, 12, 25), date(2021, 7, 4)],
        })

    @pytest.mark.parametrize(
        "days_to_next,days_since_last,expected_failures",
        [
            (False, False, []),
            (True, True, []),
        ],
        ids=["binary_only", "with_proximity"],
    )
    def test_systematic_checks(
        self,
        holidays_df,
        days_to_next,
        days_since_last,
        expected_failures,
        time_series_train_test_factory,
    ):
        """Run all applicable checks for HolidayFeatureTransformer."""
        X_train, X_test = time_series_train_test_factory(
            train_length=60,
            test_length=30,
        )

        transformer = HolidayFeatureTransformer(
            holidays=holidays_df, days_to_next=days_to_next, days_since_last=days_since_last
        )
        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
            expected_failures=set(expected_failures),
        )


class TestHolidayFeatureTransformerBasic:
    """Basic functionality tests for HolidayFeatureTransformer."""

    def test_binary_output(self):
        """Test correct binary output for known holidays."""
        time = pl.datetime_range(start=datetime(2020, 12, 23), end=datetime(2020, 12, 27), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})
        holidays = pl.DataFrame({"date": [date(2020, 12, 25)]})

        transformer = HolidayFeatureTransformer(holidays=holidays)
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert X_t["holiday_indicator"].to_list() == [0, 0, 1, 0, 0]

    def test_no_holidays_match(self):
        """Test all zeros when no holidays match."""
        time = pl.datetime_range(start=datetime(2020, 6, 1), end=datetime(2020, 6, 5), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})
        holidays = pl.DataFrame({"date": [date(2020, 12, 25)]})

        transformer = HolidayFeatureTransformer(holidays=holidays)
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert all(v == 0 for v in X_t["holiday_indicator"].to_list())

    def test_missing_date_column_raises(self):
        """Test ValueError for missing date column."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 10), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})
        holidays = pl.DataFrame({"holiday_date": [date(2020, 1, 5)]})

        transformer = HolidayFeatureTransformer(holidays=holidays)
        with pytest.raises(ValueError, match="'date' column"):
            transformer.fit(X)

    def test_invalid_holidays_type_raises(self):
        """Test ValueError for non-DataFrame holidays."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 10), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = HolidayFeatureTransformer(holidays="not_a_df")
        with pytest.raises(ValueError, match="polars DataFrame"):
            transformer.fit(X)

    def test_none_holidays_raises(self):
        """Test ValueError when holidays is None."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 10), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})

        transformer = HolidayFeatureTransformer()
        with pytest.raises(ValueError, match="holidays must be provided"):
            transformer.fit(X)

    def test_wrong_date_dtype_raises(self):
        """Test ValueError when date column has wrong dtype."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 10), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})
        holidays = pl.DataFrame({"date": ["2020-01-05"]})

        transformer = HolidayFeatureTransformer(holidays=holidays)
        with pytest.raises(ValueError, match="Date or Datetime type"):
            transformer.fit(X)


class TestHolidayFeatureTransformerProximity:
    """Tests for proximity feature output."""

    def test_proximity_columns(self):
        """Test days_to_next/days_since_last produce distance columns."""
        time = pl.datetime_range(start=datetime(2020, 12, 23), end=datetime(2020, 12, 27), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})
        holidays = pl.DataFrame({"date": [date(2020, 12, 25)]})

        transformer = HolidayFeatureTransformer(holidays=holidays, days_to_next=True, days_since_last=True)
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert "holiday_days_to_next" in X_t.columns
        assert "holiday_days_since_last" in X_t.columns

        to_next = X_t["holiday_days_to_next"].to_list()
        assert to_next[0] == 2  # Dec 23 -> Dec 25
        assert to_next[1] == 1  # Dec 24 -> Dec 25
        assert to_next[2] == 0  # Dec 25 -> Dec 25

    def test_proximity_null_at_boundary(self):
        """Test null when no next/previous holiday exists."""
        time = pl.datetime_range(start=datetime(2020, 12, 26), end=datetime(2020, 12, 28), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})
        holidays = pl.DataFrame({"date": [date(2020, 12, 25)]})

        transformer = HolidayFeatureTransformer(holidays=holidays, days_to_next=True, days_since_last=True)
        transformer.fit(X)
        X_t = transformer.transform(X)

        to_next = X_t["holiday_days_to_next"].to_list()
        assert all(v is None for v in to_next)

        since_last = X_t["holiday_days_since_last"].to_list()
        assert since_last[0] == 1  # Dec 26.

    def test_days_to_next_only(self):
        """Test days_to_next=True without days_since_last."""
        time = pl.datetime_range(start=datetime(2020, 12, 23), end=datetime(2020, 12, 27), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})
        holidays = pl.DataFrame({"date": [date(2020, 12, 25)]})

        transformer = HolidayFeatureTransformer(holidays=holidays, days_to_next=True)
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert "holiday_days_to_next" in X_t.columns
        assert "holiday_days_since_last" not in X_t.columns
        assert X_t["holiday_days_to_next"][0] == 2

    def test_days_since_last_only(self):
        """Test days_since_last=True without days_to_next."""
        time = pl.datetime_range(start=datetime(2020, 12, 26), end=datetime(2020, 12, 28), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})
        holidays = pl.DataFrame({"date": [date(2020, 12, 25)]})

        transformer = HolidayFeatureTransformer(holidays=holidays, days_since_last=True)
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert "holiday_days_since_last" in X_t.columns
        assert "holiday_days_to_next" not in X_t.columns
        assert X_t["holiday_days_since_last"][0] == 1


class TestHolidayFeatureTransformerEdgeCases:
    """Edge case tests for HolidayFeatureTransformer."""

    def test_empty_holiday_list(self):
        """Test with empty holidays DataFrame."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})
        holidays = pl.DataFrame({"date": pl.Series([], dtype=pl.Date)})

        transformer = HolidayFeatureTransformer(holidays=holidays)
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert all(v == 0 for v in X_t["holiday_indicator"].to_list())

    def test_empty_holidays_with_proximity(self):
        """Test proximity with empty holidays produces all nulls."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})
        holidays = pl.DataFrame({"date": pl.Series([], dtype=pl.Date)})

        transformer = HolidayFeatureTransformer(holidays=holidays, days_to_next=True, days_since_last=True)
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert all(v is None for v in X_t["holiday_days_to_next"].to_list())
        assert all(v is None for v in X_t["holiday_days_since_last"].to_list())

    def test_empty_holidays_days_to_next_only(self):
        """Test empty holidays with only days_to_next produces nulls."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})
        holidays = pl.DataFrame({"date": pl.Series([], dtype=pl.Date)})

        transformer = HolidayFeatureTransformer(holidays=holidays, days_to_next=True)
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert "holiday_days_to_next" in X_t.columns
        assert "holiday_days_since_last" not in X_t.columns
        assert all(v is None for v in X_t["holiday_days_to_next"].to_list())

    def test_empty_holidays_days_since_last_only(self):
        """Test empty holidays with only days_since_last produces nulls."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})
        holidays = pl.DataFrame({"date": pl.Series([], dtype=pl.Date)})

        transformer = HolidayFeatureTransformer(holidays=holidays, days_since_last=True)
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert "holiday_days_since_last" in X_t.columns
        assert "holiday_days_to_next" not in X_t.columns
        assert all(v is None for v in X_t["holiday_days_since_last"].to_list())

    def test_all_dates_are_holidays(self):
        """Test when every date in data is a holiday."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 3), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})
        holidays = pl.DataFrame({
            "date": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
        })

        transformer = HolidayFeatureTransformer(holidays=holidays)
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert all(v == 1 for v in X_t["holiday_indicator"].to_list())

    def test_datetime_holidays_column(self):
        """Test holidays with Datetime column type works."""
        time = pl.datetime_range(start=datetime(2020, 12, 24), end=datetime(2020, 12, 26), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})
        holidays = pl.DataFrame({"date": [datetime(2020, 12, 25)]})

        transformer = HolidayFeatureTransformer(holidays=holidays)
        transformer.fit(X)
        X_t = transformer.transform(X)

        assert X_t["holiday_indicator"].to_list() == [0, 1, 0]

    def test_column_name_conflict_raises(self):
        """Test that conflicting column names raise ValueError."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "holiday_indicator": range(len(time))})
        holidays = pl.DataFrame({"date": [date(2020, 1, 3)]})

        transformer = HolidayFeatureTransformer(holidays=holidays)
        with pytest.raises(ValueError, match="conflict"):
            transformer.fit(X)

    def test_get_feature_names_out(self):
        """Test get_feature_names_out returns correct names."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})
        holidays = pl.DataFrame({"date": [date(2020, 1, 3)]})

        transformer = HolidayFeatureTransformer(holidays=holidays, days_to_next=True, days_since_last=True)
        transformer.fit(X)

        names = transformer.get_feature_names_out()
        assert names == ["holiday_indicator", "holiday_days_to_next", "holiday_days_since_last"]

    def test_get_feature_names_out_no_proximity(self):
        """Test get_feature_names_out without proximity."""
        time = pl.datetime_range(start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1d", eager=True)
        X = pl.DataFrame({"time": time, "value": range(len(time))})
        holidays = pl.DataFrame({"date": [date(2020, 1, 3)]})

        transformer = HolidayFeatureTransformer(holidays=holidays)
        transformer.fit(X)

        names = transformer.get_feature_names_out()
        assert names == ["holiday_indicator"]
