"""Tests for synthetic dataset generators."""

from __future__ import annotations

import polars as pl

from yohou.datasets._generators import (
    _HOLIDAY_MONTH_DAYS,
    make_exogenous_classification,
    make_exogenous_regression,
)


class TestMakeExogenousRegression:
    """Tests for make_exogenous_regression."""

    def test_bunch_keys(self):
        """Returned Bunch has all expected keys."""
        data = make_exogenous_regression()
        assert set(data.keys()) == {
            "y",
            "X_actual",
            "X_future",
            "X_forecast",
            "frame",
            "feature_names",
            "target_names",
            "frequency",
            "DESCR",
        }

    def test_y_schema(self):
        """y has [time, price] columns with correct types."""
        data = make_exogenous_regression()
        assert data.y.columns == ["time", "price"]
        assert data.y["time"].dtype == pl.Datetime
        assert data.y["price"].dtype == pl.Float64

    def test_x_actual_schema(self):
        """X_actual has [time, temperature] columns."""
        data = make_exogenous_regression()
        assert data.X_actual.columns == ["time", "temperature"]
        assert data.X_actual["time"].dtype == pl.Datetime

    def test_x_future_schema(self):
        """X_future has [time, is_holiday] columns."""
        data = make_exogenous_regression()
        assert data.X_future.columns == ["time", "is_holiday"]
        assert data.X_future["time"].dtype == pl.Datetime

    def test_x_forecast_schema(self):
        """X_forecast has [vintage_time, time, wx_temp] columns."""
        data = make_exogenous_regression()
        assert data.X_forecast.columns == ["vintage_time", "time", "wx_temp"]
        assert data.X_forecast["vintage_time"].dtype == pl.Datetime
        assert data.X_forecast["time"].dtype == pl.Datetime

    def test_row_counts(self):
        """y, X_actual, X_future all have n_samples rows."""
        n = 150
        data = make_exogenous_regression(n_samples=n)
        assert len(data.y) == n
        assert len(data.X_actual) == n
        assert len(data.X_future) == n

    def test_x_forecast_one_vintage_per_observation(self):
        """X_forecast has one vintage per observation from H onward."""
        n, H = 100, 4
        data = make_exogenous_regression(n_samples=n, forecasting_horizon=H)
        vintage_times = set(data.X_forecast["vintage_time"].unique().to_list())
        obs_times = set(data.y["time"][H:].to_list())
        # All present vintages must be valid observation times
        assert vintage_times <= obs_times

    def test_x_forecast_step_coverage(self):
        """Each vintage covers up to forecasting_horizon steps."""
        data = make_exogenous_regression(n_samples=100, forecasting_horizon=4)
        for vt in data.X_forecast["vintage_time"].unique():
            subset = data.X_forecast.filter(pl.col("vintage_time") == vt)
            assert len(subset) <= 4

    def test_frame_join(self):
        """frame is y + X_actual + X_future joined on time."""
        data = make_exogenous_regression()
        assert "price" in data.frame.columns
        assert "temperature" in data.frame.columns
        assert "is_holiday" in data.frame.columns
        assert len(data.frame) == len(data.y)

    def test_feature_names(self):
        """feature_names lists all three feature columns."""
        data = make_exogenous_regression()
        assert data.feature_names == ["temperature", "is_holiday", "wx_temp"]

    def test_target_names(self):
        """target_names is ['price']."""
        data = make_exogenous_regression()
        assert data.target_names == ["price"]

    def test_frequency(self):
        """frequency is '1h'."""
        data = make_exogenous_regression()
        assert data.frequency == "1h"

    def test_determinism_same_seed(self):
        """Same random_state produces identical output."""
        a = make_exogenous_regression(random_state=0)
        b = make_exogenous_regression(random_state=0)
        assert a.y.equals(b.y)
        assert a.X_forecast.equals(b.X_forecast)

    def test_different_seed(self):
        """Different random_state produces different output."""
        a = make_exogenous_regression(random_state=0)
        b = make_exogenous_regression(random_state=1)
        assert not a.y.equals(b.y)

    def test_custom_parameters(self):
        """Custom parameters are respected."""
        data = make_exogenous_regression(n_samples=50, forecasting_horizon=3, noise=1.0, forecast_bias=2.0)
        assert len(data.y) == 50
        for vt in data.X_forecast["vintage_time"].unique():
            subset = data.X_forecast.filter(pl.col("vintage_time") == vt)
            assert len(subset) <= 3


class TestHolidayCalendarIsAnEventFeature:
    """The generators' is_holiday must be a genuine X_future feature.

    A weekday predicate would be a clock feature: derivable from the timestamp,
    and therefore something a feature_transformer should produce rather than
    something the X_future channel must carry. These tests pin the property the
    channel's semantics depend on.
    """

    def test_is_holiday_is_not_a_weekday_predicate(self):
        """is_holiday matches no single-weekday indicator."""
        X_future = make_exogenous_regression(n_samples=24 * 60).X_future
        weekday = X_future["time"].dt.weekday()
        for day in range(1, 8):
            indicator = (weekday == day).cast(pl.Float64)
            assert not (indicator == X_future["is_holiday"]).all(), f"is_holiday is just weekday == {day}"

    def test_holidays_span_at_least_three_weekdays(self):
        """Over a year the holiday dates fall on at least three weekdays.

        Regression guard: the month/day lookup key is built with arithmetic on
        Int8 columns, which silently overflowed past January and collapsed the
        calendar onto the wrong dates. A January-only window cannot catch that.
        """
        X_future = make_exogenous_regression(n_samples=24 * 366).X_future
        holidays = X_future.filter(pl.col("is_holiday") == 1.0)
        weekdays = {t.strftime("%a") for t in holidays["time"]}
        assert len(weekdays) >= 3, f"holidays only fall on {sorted(weekdays)}"

    def test_every_calendar_entry_lands_within_a_year(self):
        """A full year marks exactly one date per calendar entry."""
        X_future = make_exogenous_regression(n_samples=24 * 366).X_future
        holidays = X_future.filter(pl.col("is_holiday") == 1.0)
        dates = {t.date() for t in holidays["time"]}
        assert len(dates) == len(_HOLIDAY_MONTH_DAYS)

    def test_default_window_has_distributed_holidays(self):
        """The default window carries at least two dates, not all at the start."""
        X_future = make_exogenous_regression().X_future
        holidays = X_future.filter(pl.col("is_holiday") == 1.0)
        dates = sorted({t.date() for t in holidays["time"]})
        assert len(dates) >= 2
        assert any(d > X_future["time"][0].date() for d in dates)

    def test_calendar_is_seed_independent(self):
        """The holiday calendar is a property of the dataset, not the seed."""
        a = make_exogenous_regression(random_state=0).X_future
        b = make_exogenous_regression(random_state=99).X_future
        assert a["is_holiday"].equals(b["is_holiday"])

    def test_classification_holiday_is_also_an_event_feature(self):
        """The classification generator's X_future is not a weekend predicate."""
        X_future = make_exogenous_classification(n_samples=24 * 60).X_future
        weekend = (X_future["time"].dt.weekday() >= 6).cast(pl.Float64)
        assert not (weekend == X_future["is_holiday"]).all()


class TestMakeExogenousClassification:
    """Tests for make_exogenous_classification."""

    def test_bunch_keys(self):
        """Returned Bunch has all expected keys including classes."""
        data = make_exogenous_classification()
        assert set(data.keys()) == {
            "y",
            "X_actual",
            "X_future",
            "X_forecast",
            "frame",
            "feature_names",
            "target_names",
            "classes",
            "frequency",
            "DESCR",
        }

    def test_y_schema(self):
        """y has [time, air_quality] columns."""
        data = make_exogenous_classification()
        assert data.y.columns == ["time", "air_quality"]
        assert data.y["time"].dtype == pl.Datetime

    def test_x_actual_schema(self):
        """X_actual has [time, pollutant] columns."""
        data = make_exogenous_classification()
        assert data.X_actual.columns == ["time", "pollutant"]

    def test_x_future_schema(self):
        """X_future has [time, is_holiday] columns."""
        data = make_exogenous_classification()
        assert data.X_future.columns == ["time", "is_holiday"]

    def test_x_forecast_schema(self):
        """X_forecast has [vintage_time, time, pollutant_forecast] columns."""
        data = make_exogenous_classification()
        assert data.X_forecast.columns == ["vintage_time", "time", "pollutant_forecast"]

    def test_row_counts(self):
        """y, X_actual, X_future all have n_samples rows."""
        n = 200
        data = make_exogenous_classification(n_samples=n)
        assert len(data.y) == n
        assert len(data.X_actual) == n
        assert len(data.X_future) == n

    def test_classes_match_y_values(self):
        """classes attribute matches actual y values."""
        data = make_exogenous_classification()
        y_values = set(data.y["air_quality"].unique().to_list())
        assert y_values <= set(data.classes)

    def test_classes_content(self):
        """classes contains good, moderate, poor."""
        data = make_exogenous_classification()
        assert data.classes == ["good", "moderate", "poor"]

    def test_x_forecast_one_vintage_per_observation(self):
        """X_forecast has one vintage per observation from H onward."""
        n, H = 200, 4
        data = make_exogenous_classification(n_samples=n, forecasting_horizon=H)
        vintage_times = set(data.X_forecast["vintage_time"].unique().to_list())
        obs_times = set(data.y["time"][H:].to_list())
        assert vintage_times <= obs_times

    def test_determinism_same_seed(self):
        """Same random_state produces identical output."""
        a = make_exogenous_classification(random_state=0)
        b = make_exogenous_classification(random_state=0)
        assert a.y.equals(b.y)
        assert a.X_forecast.equals(b.X_forecast)

    def test_different_seed(self):
        """Different random_state produces different output."""
        a = make_exogenous_classification(random_state=0)
        b = make_exogenous_classification(random_state=1)
        assert not a.y.equals(b.y)

    def test_feature_names(self):
        """feature_names lists all three feature columns."""
        data = make_exogenous_classification()
        assert data.feature_names == ["pollutant", "is_holiday", "pollutant_forecast"]

    def test_target_names(self):
        """target_names is ['air_quality']."""
        data = make_exogenous_classification()
        assert data.target_names == ["air_quality"]
