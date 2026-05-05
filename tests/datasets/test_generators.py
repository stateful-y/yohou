"""Tests for synthetic dataset generators."""

from __future__ import annotations

import polars as pl

from yohou.datasets._generators import (
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
        """X_future has [time, is_weekend] columns."""
        data = make_exogenous_classification()
        assert data.X_future.columns == ["time", "is_weekend"]

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
        assert data.feature_names == ["pollutant", "is_weekend", "pollutant_forecast"]

    def test_target_names(self):
        """target_names is ['air_quality']."""
        data = make_exogenous_classification()
        assert data.target_names == ["air_quality"]
