"""Tests for calendar and holiday feature transformers.

Tests CalendarFeatureTransformer, HolidayFeatureTransformer, and
DaylightSavingFeatureTransformer using both the systematic check generator
pattern and transformer-specific tests.
"""

from datetime import UTC, date, datetime

import polars as pl
import pytest
from sklearn.base import clone

from conftest import run_checks
from yohou.preprocessing.calendar import (
    CalendarFeatureTransformer,
    DaylightSavingFeatureTransformer,
    HolidayFeatureTransformer,
)
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


@pytest.fixture
def spring_frame() -> pl.DataFrame:
    """Hourly UTC frame spanning the 2026-03-08 US spring-forward (02:00 CST -> 03:00 CDT)."""
    return pl.DataFrame({
        "time": pl.datetime_range(
            datetime(2026, 3, 6, tzinfo=UTC), datetime(2026, 3, 10, tzinfo=UTC), interval="1h", eager=True
        )
    })


@pytest.fixture
def fall_frame() -> pl.DataFrame:
    """Hourly UTC frame spanning the 2026-11-01 US fall-back (02:00 CDT -> 01:00 CST)."""
    return pl.DataFrame({
        "time": pl.datetime_range(
            datetime(2026, 10, 30, tzinfo=UTC), datetime(2026, 11, 3, tzinfo=UTC), interval="1h", eager=True
        )
    })


def _row_at(out: pl.DataFrame, *cols: str, y: int, m: int, d: int, h: int) -> tuple:
    return out.filter(pl.col("time") == datetime(y, m, d, h, tzinfo=UTC)).select(*cols).row(0)


class TestDaylightSavingFeatureTransformerSystematic:
    """Systematic check generator tests for DaylightSavingFeatureTransformer."""

    @pytest.mark.parametrize(
        "transformer,expected_failures",
        [
            (DaylightSavingFeatureTransformer(), []),
            (DaylightSavingFeatureTransformer(features=["in_effect", "transition_day", "transition_type"]), []),
        ],
        ids=["default", "all_features"],
    )
    def test_systematic_checks(self, transformer, expected_failures, time_series_train_test_factory):
        """Run all applicable checks for DaylightSavingFeatureTransformer."""
        X_train, X_test = time_series_train_test_factory(train_length=60, test_length=30)
        # The transformer requires a timezone-aware "time" column; the factory is tz-naive.
        X_train = X_train.with_columns(pl.col("time").dt.replace_time_zone("UTC"))
        X_test = X_test.with_columns(pl.col("time").dt.replace_time_zone("UTC"))

        transformer_fitted = clone(transformer)
        transformer_fitted.fit(X_train)

        run_checks(
            transformer_fitted,
            _yield_yohou_transformer_checks(transformer_fitted, X_train, None, X_test),
            expected_failures=set(expected_failures),
        )


class TestDaylightSavingFeatureTransformerFeatures:
    """Feature-value tests for DaylightSavingFeatureTransformer."""

    def test_in_effect_across_spring_forward(self):
        """DST turns on at 03:00 CDT (08:00 UTC); the 02:00 local hour is skipped."""
        times = pl.datetime_range(
            datetime(2026, 3, 8, 6, tzinfo=UTC), datetime(2026, 3, 8, 10, tzinfo=UTC), interval="1h", eager=True
        )
        out = DaylightSavingFeatureTransformer(features=["in_effect"]).fit_transform(pl.DataFrame({"time": times}))
        assert out["dst_in_effect"].to_list() == [0, 0, 1, 1, 1]

    def test_transition_day_and_type_spring(self, spring_frame):
        """The spring-forward local date is flagged (+1); neighbouring dates are not."""
        out = DaylightSavingFeatureTransformer(features=["transition_day", "transition_type"]).fit_transform(
            spring_frame
        )
        assert _row_at(out, "dst_transition_day", "dst_transition_type", y=2026, m=3, d=8, h=12) == (1, 1)
        assert _row_at(out, "dst_transition_day", "dst_transition_type", y=2026, m=3, d=7, h=12) == (0, 0)
        assert _row_at(out, "dst_transition_day", "dst_transition_type", y=2026, m=3, d=9, h=12) == (0, 0)

    def test_transition_type_fall(self, fall_frame):
        """The fall-back local date is flagged (-1); by noon UTC the clock has already fallen back to CST."""
        out = DaylightSavingFeatureTransformer(
            features=["in_effect", "transition_day", "transition_type"]
        ).fit_transform(fall_frame)
        assert _row_at(out, "dst_in_effect", "dst_transition_day", "dst_transition_type", y=2026, m=11, d=1, h=12) == (
            0,
            1,
            -1,
        )

    def test_transition_day_equals_type_nonzero(self, spring_frame, fall_frame):
        """dst_transition_day is exactly dst_transition_type != 0."""
        for frame in (spring_frame, fall_frame):
            out = DaylightSavingFeatureTransformer(features=["transition_day", "transition_type"]).fit_transform(frame)
            expected = (out["dst_transition_type"] != 0).cast(pl.Int32)
            assert out["dst_transition_day"].to_list() == expected.to_list()

    def test_evaluation_zone_independent_of_input_zone(self):
        """The same instants give the same features whether the input is UTC or Central."""
        utc = pl.datetime_range(
            datetime(2026, 3, 8, 6, tzinfo=UTC), datetime(2026, 3, 8, 10, tzinfo=UTC), interval="1h", eager=True
        )
        X_utc = pl.DataFrame({"time": utc})
        X_central = pl.DataFrame({"time": utc.dt.convert_time_zone("America/Chicago")})
        tx = DaylightSavingFeatureTransformer(features=["in_effect"])
        assert (
            tx.fit_transform(X_utc)["dst_in_effect"].to_list() == tx.fit_transform(X_central)["dst_in_effect"].to_list()
        )

    def test_default_features_and_contract(self, spring_frame):
        """Default emits time + in_effect + transition_day, dropping input columns."""
        tx = DaylightSavingFeatureTransformer()
        out = tx.fit_transform(spring_frame)
        assert out.columns == ["time", "dst_in_effect", "dst_transition_day"]
        assert tx.applicable_features_ == ["in_effect", "transition_day"]
        assert tx.get_feature_names_out() == ["dst_in_effect", "dst_transition_day"]


class TestDaylightSavingFeatureTransformerContract:
    """Input-contract and gating tests for DaylightSavingFeatureTransformer."""

    def test_tz_naive_rejected(self):
        """A timezone-naive time column raises at fit time."""
        naive = pl.DataFrame({
            "time": pl.datetime_range(datetime(2026, 3, 6), datetime(2026, 3, 10), interval="1h", eager=True)
        })
        with pytest.raises(ValueError, match="timezone-aware"):
            DaylightSavingFeatureTransformer().fit(naive)

    def test_date_dtype_rejected(self):
        """A pl.Date time column raises at fit time (cannot be timezone-aware)."""
        dates = pl.DataFrame({"time": pl.date_range(date(2026, 3, 1), date(2026, 3, 20), interval="1d", eager=True)})
        with pytest.raises(ValueError, match="timezone-aware"):
            DaylightSavingFeatureTransformer().fit(dates)

    def test_in_effect_gated_on_daily_data(self):
        """Explicit in_effect on daily data raises; the default drops it silently."""
        daily = pl.DataFrame({
            "time": pl.datetime_range(
                datetime(2026, 3, 1, tzinfo=UTC), datetime(2026, 3, 20, tzinfo=UTC), interval="1d", eager=True
            )
        })
        with pytest.raises(ValueError, match="sub-daily"):
            DaylightSavingFeatureTransformer(features=["in_effect"]).fit(daily)

        tx = DaylightSavingFeatureTransformer().fit(daily)
        assert tx.applicable_features_ == ["transition_day"]
        assert "dst_in_effect" not in tx.transform(daily).columns

    def test_unknown_feature_raises(self, spring_frame):
        """An unrecognized feature name raises at fit time."""
        with pytest.raises(ValueError, match="Unknown DST features"):
            DaylightSavingFeatureTransformer(features=["nope"]).fit(spring_frame)

    def test_output_conflict_raises(self, spring_frame):
        """A generated dst_* name colliding with an input column raises."""
        clash = spring_frame.with_columns(pl.lit(0).alias("dst_in_effect"))
        with pytest.raises(ValueError, match="conflict"):
            DaylightSavingFeatureTransformer().fit(clash)

    def test_feature_names_out_requires_fit(self):
        """get_feature_names_out raises before fit."""
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            DaylightSavingFeatureTransformer().get_feature_names_out()

    def test_composes_in_feature_union(self, spring_frame):
        """The transformer composes inside a FeatureUnion alongside calendar features."""
        from yohou.compose import FeatureUnion

        union = FeatureUnion(
            transformer_list=[
                ("cal", CalendarFeatureTransformer(features=["hour"])),
                ("dst", DaylightSavingFeatureTransformer(features=["in_effect"])),
            ]
        )
        out = union.fit_transform(spring_frame)
        assert any("dst_in_effect" in c for c in out.columns)
