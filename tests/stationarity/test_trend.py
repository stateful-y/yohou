"""Tests for PolynomialTrendForecaster."""

from datetime import datetime

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline

from conftest import run_checks
from yohou.stationarity import PolynomialTrendForecaster
from yohou.testing import _yield_yohou_forecaster_checks


class TestPolynomialTrendForecaster:
    @pytest.mark.parametrize(
        "forecaster,expected_failures",
        [
            (
                PolynomialTrendForecaster(degree=1),
                [],
            ),
            (
                PolynomialTrendForecaster(degree=2),
                [],
            ),
            (
                PolynomialTrendForecaster(degree=3),
                [],
            ),
        ],
    )
    def test_polynomial_trend_checks(self, forecaster, expected_failures, y_X_factory):
        """Run systematic checks on PolynomialTrendForecaster."""
        # Generate data with trend
        y, X_actual = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        # Add linear trend to data
        y = y.with_columns([(pl.col(col) + pl.Series(range(len(y)))).alias(col) for col in y.columns if col != "time"])

        y_train, y_test = y[:80], y[80:]
        X_actual_train, X_actual_test = (X_actual[:80], X_actual[80:]) if X_actual is not None else (None, None)

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_actual_train, forecasting_horizon=3)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(forecaster_fitted, y_train, X_actual_train, y_test, X_actual_test),
            expected_failures=set(expected_failures),
        )

    def test_polynomial_linear_analytical(self):
        """Test linear trend forecaster on known linear process."""
        # Create perfect linear trend: y = 2*t + 5
        from datetime import timedelta

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": [2 * i + 5 for i in range(50)]})

        # Fit on first 40, predict next 10
        forecaster = PolynomialTrendForecaster(degree=1, estimator=ElasticNet(alpha=0.0, l1_ratio=0.0))
        forecaster.fit(y[:40], forecasting_horizon=1)

        # Predict 10 steps
        y_pred = forecaster.predict(forecasting_horizon=10)

        # Check predictions match exact linear continuation
        expected_values = [2 * i + 5 for i in range(40, 50)]

        # Allow small numerical error (floating point precision)
        pred_values = y_pred["value"].to_numpy().flatten()
        assert np.allclose(pred_values, expected_values, atol=1e-10), (
            "Linear trend predictions should match exact linear process"
        )

    def test_polynomial_quadratic_analytical(self):
        """Test polynomial trend on known quadratic process."""
        # Create perfect quadratic: y = 0.5*t^2 + 2*t + 1
        from datetime import timedelta

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": [0.5 * i**2 + 2 * i + 1 for i in range(50)]})

        # Fit polynomial degree 2
        forecaster = PolynomialTrendForecaster(degree=2, estimator=ElasticNet(alpha=0.0, l1_ratio=0.0))
        forecaster.fit(y[:40], forecasting_horizon=1)

        # Predict 10 steps
        y_pred = forecaster.predict(forecasting_horizon=10)

        # Check predictions
        expected_values = [0.5 * i**2 + 2 * i + 1 for i in range(40, 50)]

        # Polynomial fitting may have small numerical errors
        pred_values = y_pred["value"].to_numpy().flatten()
        assert np.allclose(pred_values, expected_values, atol=1e-1), (
            "Polynomial trend should match quadratic process closely"
        )

    def test_polynomial_different_horizons(self):
        """Test that different forecasting horizons work correctly."""
        from datetime import timedelta

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": [2 * i + 5 for i in range(50)]})

        forecaster = PolynomialTrendForecaster(degree=1)
        forecaster.fit(y[:40], forecasting_horizon=5)

        # Predict different horizon than fit
        y_pred_10 = forecaster.predict(forecasting_horizon=10)
        assert len(y_pred_10) == 10

        y_pred_3 = forecaster.predict(forecasting_horizon=3)
        assert len(y_pred_3) == 3

    def test_polynomial_panel_data(self, panel_time_series_factory):
        """Test PolynomialTrendForecaster with panel data."""
        y_panel = panel_time_series_factory(length=100, n_series=3, seed=42)

        # Add linear trends with different slopes per series
        for i in range(3):
            col_name = f"panel__series_{i}"
            y_panel = y_panel.with_columns((pl.col(col_name) + (i + 1) * pl.Series(range(100))).alias(col_name))

        forecaster = PolynomialTrendForecaster(degree=1)
        forecaster.fit(y_panel[:80], forecasting_horizon=5)

        # Predict all groups
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Should have predictions for all series
        assert "panel__series_0" in y_pred.columns
        assert "panel__series_1" in y_pred.columns
        assert "panel__series_2" in y_pred.columns
        assert len(y_pred) == 5

    def test_polynomial_observe_predict(self):
        """Test observe_predict method."""
        from datetime import timedelta

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": [2 * i + 5 for i in range(50)]})

        forecaster = PolynomialTrendForecaster(degree=1)
        fit_forecasting_horizon = 5
        forecaster.fit(y[:30], forecasting_horizon=fit_forecasting_horizon)

        # Observe with new data and predict
        n_new = 10
        predict_forecasting_horizon = 5
        y_new = y[30 : 30 + n_new]
        y_pred = forecaster.observe_predict(y_new, forecasting_horizon=predict_forecasting_horizon)

        assert len(y_pred) == predict_forecasting_horizon * (1 + n_new // fit_forecasting_horizon)
        assert "value" in y_pred.columns

    def test_polynomial_panel_pooled_strategy(self, panel_time_series_factory):
        """Test PolynomialTrendForecaster with pooled panel strategy."""
        y_panel = panel_time_series_factory(length=100, n_series=3, seed=42)

        # Add distinct linear trends with different slopes per series
        for i in range(3):
            col_name = f"panel__series_{i}"
            # Series 0: slope=1, Series 1: slope=3, Series 2: slope=5
            slope = 1 + i * 2
            y_panel = y_panel.with_columns((pl.col(col_name) + slope * pl.Series(range(100))).alias(col_name))

        # Fit forecaster with default pooled strategy
        forecaster = PolynomialTrendForecaster(degree=1)
        forecaster.fit(y_panel[:80], forecasting_horizon=5)

        # Pooled strategy: estimator_ should be single Pipeline object
        assert isinstance(forecaster.estimator_, Pipeline), "Pooled should store single Pipeline"
        assert not isinstance(forecaster.estimator_, dict), "Pooled should not be a dict"

        # Predict all groups
        y_pred = forecaster.predict(forecasting_horizon=5)

        # Basic structure checks
        assert len(y_pred) == 5
        assert "panel__series_0" in y_pred.columns
        assert "panel__series_1" in y_pred.columns
        assert "panel__series_2" in y_pred.columns


class TestPolynomialTrendWithoutExogenous:
    """Tests for PolynomialTrendForecaster with X_actual=None."""

    def test_fit_predict_without_exogenous(self):
        """PolynomialTrendForecaster should work without exogenous features."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 2, 19),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": [float(i) for i in range(50)]})

        forecaster = PolynomialTrendForecaster(degree=1)
        forecaster.fit(y[:40], X_actual=None, forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns
        assert len(y_pred) == 5

    def test_requires_exogenous_tag(self):
        """PolynomialTrendForecaster should have requires_exogenous=False tag."""
        forecaster = PolynomialTrendForecaster(degree=1)
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.requires_exogenous is False

    def test_panel_rewind_uses_per_group_observation_horizon(self):
        """Each panel group's rewind uses its own transformer horizon.

        Panel rewind must read the observation horizon from each group's own
        transformer rather than from the first group's transformer
        (``next(iter(...))``) and applying it to every group. With groups whose
        transformers have different horizons, each group must use its own.
        """
        from yohou.stationarity import SeasonalDifferencing

        forecaster = PolynomialTrendForecaster(degree=1, target_transformer=SeasonalDifferencing(seasonality=2))
        # Simulate a fitted panel state with per-group transformers whose
        # observation horizons (== seasonality) differ across groups.
        forecaster.target_transformer_ = {
            "g0": SeasonalDifferencing(seasonality=2),
            "g1": SeasonalDifferencing(seasonality=5),
        }

        assert forecaster._group_target_observation_horizon("g0") == 2
        assert forecaster._group_target_observation_horizon("g1") == 5

    def test_panel_rewind_updates_first_observed_time_per_group(self):
        """A real panel fit -> observe -> rewind cycle updates each group's anchor.

        Exercises the panel branch of ``_BaseTrendForecaster.rewind`` (which reads
        each group's own ``get_group_df`` slice and per-group observation horizon)
        rather than unit-testing the horizon helper in isolation.
        """
        from datetime import timedelta

        from yohou.stationarity import SeasonalDifferencing

        length = 60
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
            interval="1s",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "g0__series_0": [float(i) for i in range(length)],
            "g1__series_0": [float(i + 10) for i in range(length)],
        })

        forecaster = PolynomialTrendForecaster(degree=1, target_transformer=SeasonalDifferencing(seasonality=2))
        forecaster.fit(y[:40], forecasting_horizon=5)
        assert isinstance(forecaster.target_transformer_, dict)

        forecaster.observe(y=y[40:50])
        forecaster.rewind(y=y[30:40])

        # Each group's first observed time is anchored at its own horizon (== 2)
        # within the rewound slice that begins at index 30, i.e. time[32].
        expected = time[32]
        assert forecaster._first_observed_time == {"g0": expected, "g1": expected}


class TestPolynomialTrendDefaultEstimator:
    def test_default_estimator_is_none_no_shared_mutable(self):
        """estimator defaults to None (not a shared mutable ElasticNet); built at fit."""
        import inspect

        default = inspect.signature(PolynomialTrendForecaster).parameters["estimator"].default
        assert default is None
        assert PolynomialTrendForecaster().estimator is None
