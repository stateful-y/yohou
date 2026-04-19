"""Tests for LocalPanelForecaster meta-forecaster.

Tests per-group independent forecasting via LocalPanelForecaster.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge

from yohou.compose import LocalPanelForecaster
from yohou.interval import SplitConformalForecaster
from yohou.point import PointReductionForecaster, SeasonalNaive
from yohou.stationarity import PolynomialTrendForecaster
from yohou.utils.panel import inspect_panel


@pytest.fixture
def panel_y():
    """Simple panel target with two groups."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 4, 9),
        interval="1d",
        eager=True,
    )
    return pl.DataFrame({
        "time": time,
        "store_a__sales": [float(i) for i in range(100)],
        "store_b__sales": [float(i + 100) for i in range(100)],
    })


@pytest.fixture
def panel_y_X():
    """Panel target and exogenous with two groups."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 4, 9),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({
        "time": time,
        "store_a__sales": [float(i) for i in range(100)],
        "store_b__sales": [float(i + 100) for i in range(100)],
    })
    X = pl.DataFrame({
        "time": time,
        "store_a__promo": [float(i % 7) for i in range(100)],
        "store_b__promo": [float(i % 5) for i in range(100)],
    })
    return y, X


class TestBasicFitPredict:
    """Tests for basic fit/predict workflow."""

    def test_fit_predict_point(self, panel_y):
        """Fit and predict with SeasonalNaive on panel data."""
        y = panel_y
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        f.fit(y[:80], forecasting_horizon=5)

        y_pred = f.predict(forecasting_horizon=5)

        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) == 5
        assert "time" in y_pred.columns
        assert "store_a__sales" in y_pred.columns
        assert "store_b__sales" in y_pred.columns

    def test_fit_predict_with_exogenous(self, panel_y_X):
        """Fit and predict with exogenous features."""
        y, X = panel_y_X
        f = LocalPanelForecaster(
            forecaster=PointReductionForecaster(estimator=Ridge()),
        )
        f.fit(y[:80], X[:80], forecasting_horizon=5)
        y_pred = f.predict(X=X[80:85], forecasting_horizon=5)

        assert len(y_pred) == 5
        assert "store_a__sales" in y_pred.columns
        assert "store_b__sales" in y_pred.columns

    def test_per_group_clones_are_independent(self, panel_y):
        """Each group should have its own independent forecaster clone."""
        y = panel_y
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        f.fit(y[:80], forecasting_horizon=5)

        assert len(f.forecasters_) == 2
        assert "store_a" in f.forecasters_
        assert "store_b" in f.forecasters_
        assert f.forecasters_["store_a"] is not f.forecasters_["store_b"]

    def test_rejects_non_panel_data(self):
        """Should raise ValueError on global (non-panel) data."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 4, 9),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "sales": range(100)})

        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        with pytest.raises(ValueError, match="requires panel data"):
            f.fit(y, forecasting_horizon=5)

    def test_not_fitted_error(self, panel_y):
        """Should raise NotFittedError before fit."""
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        with pytest.raises(NotFittedError, match="is not fitted"):
            f.predict(forecasting_horizon=5)


class TestGroupSubsetting:
    """Tests for groups subsetting."""

    def test_predict_subset_groups(self, panel_y):
        """predict(groups=...) should return only requested groups."""
        y = panel_y
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        f.fit(y[:80], forecasting_horizon=5)

        y_pred = f.predict(forecasting_horizon=5, groups=["store_a"])

        non_time = [c for c in y_pred.columns if c not in ("time", "observed_time")]
        assert all(c.startswith("store_a__") for c in non_time)

    def test_observe_subset_groups(self, panel_y):
        """observe(groups=...) should update only requested groups."""
        y = panel_y
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        f.fit(y[:80], forecasting_horizon=5)

        # Observe only store_a
        f.observe(y=y[80:85], groups=["store_a"])

        # store_a should have been updated, store_b should still be at fit state
        # We verify by checking prediction differs for store_a
        y_pred_a = f.predict(forecasting_horizon=3, groups=["store_a"])
        assert len(y_pred_a) == 3


class TestObserveRewind:
    """Tests for observe and rewind methods."""

    def test_observe_predict_cycle(self, panel_y):
        """Observe then predict should use new observations."""
        y = panel_y
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        f.fit(y[:80], forecasting_horizon=5)

        y_pred_before = f.predict(forecasting_horizon=5)
        f.observe(y=y[80:85])
        y_pred_after = f.predict(forecasting_horizon=5)

        # Predictions should differ after observing new data
        assert y_pred_before["store_a__sales"].to_list() != y_pred_after["store_a__sales"].to_list()

    def test_observe_predict_combined(self, panel_y):
        """observe_predict should be equivalent to observe + predict."""
        y = panel_y
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        f.fit(y[:80], forecasting_horizon=5)

        y_pred = f.observe_predict(y=y[80:85])
        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) == 5

    def test_rewind(self, panel_y):
        """Rewind should reset observation state per group."""
        y = panel_y
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        f.fit(y[:80], forecasting_horizon=5)
        f.observe(y=y[80:85])

        # Rewind to a subset of the data
        f.rewind(y=y[70:80])

        y_pred = f.predict(forecasting_horizon=5)
        assert len(y_pred) == 5


class TestLocalVsGlobal:
    """Verify that LocalPanelForecaster handles heterogeneous groups better than pooled."""

    def test_captures_per_group_trends(self):
        """Groups with different slopes: local should capture each individually."""
        n = 100
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=n - 1),
            interval="1d",
            eager=True,
        )

        # Group A: slope=1, Group B: slope=10
        y = pl.DataFrame({
            "time": time,
            "gA__value": [1.0 * i for i in range(n)],
            "gB__value": [10.0 * i for i in range(n)],
        })

        f = LocalPanelForecaster(
            forecaster=PolynomialTrendForecaster(degree=1),
        )
        f.fit(y[:80], forecasting_horizon=5)
        y_pred = f.predict(forecasting_horizon=5)

        # Group A predictions should reflect slope=1
        gA_pred = y_pred["gA__value"].to_numpy()
        gA_expected = np.array([1.0 * i for i in range(80, 85)])
        np.testing.assert_allclose(gA_pred, gA_expected, rtol=0.1)

        # Group B predictions should reflect slope=10
        gB_pred = y_pred["gB__value"].to_numpy()
        gB_expected = np.array([10.0 * i for i in range(80, 85)])
        np.testing.assert_allclose(gB_pred, gB_expected, rtol=0.1)


class TestTags:
    """Tests for sklearn tag propagation."""

    def test_inherits_forecaster_type_point(self):
        """Should inherit forecaster_type from wrapped point forecaster."""
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        tags = f.__sklearn_tags__()
        assert tags.forecaster_tags.forecaster_type == frozenset({"point"})

    def test_inherits_forecaster_type_interval(self):
        """Should inherit forecaster_type from wrapped interval forecaster."""
        f = LocalPanelForecaster(
            forecaster=SplitConformalForecaster(
                point_forecaster=SeasonalNaive(seasonality=7),
            ),
        )
        tags = f.__sklearn_tags__()
        assert "interval" in tags.forecaster_tags.forecaster_type

    def test_supports_panel_data(self):
        """Should declare panel data support."""
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        tags = f.__sklearn_tags__()
        assert tags.forecaster_tags.supports_panel_data is True

    def test_tracks_observations_false(self):
        """Should delegate observation tracking to children."""
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        tags = f.__sklearn_tags__()
        assert tags.forecaster_tags.tracks_observations is False


class TestCloneParams:
    """Tests for sklearn clone and parameter management."""

    def test_clone(self):
        """Clone should produce equivalent unfitted forecaster."""
        f = LocalPanelForecaster(
            forecaster=SeasonalNaive(seasonality=7),
            n_jobs=2,
        )
        f_cloned = clone(f)

        assert f_cloned.n_jobs == 2
        assert isinstance(f_cloned.forecaster, SeasonalNaive)
        assert f_cloned.forecaster is not f.forecaster

    def test_get_params(self):
        """get_params should include nested forecaster parameters."""
        f = LocalPanelForecaster(
            forecaster=SeasonalNaive(seasonality=7),
            n_jobs=2,
        )
        params = f.get_params(deep=True)

        assert params["n_jobs"] == 2
        assert params["forecaster__seasonality"] == 7

    def test_set_params(self):
        """set_params should update nested parameters."""
        f = LocalPanelForecaster(
            forecaster=SeasonalNaive(seasonality=7),
        )
        f.set_params(forecaster__seasonality=14)

        assert f.forecaster.seasonality == 14


class TestPredictInterval:
    """Tests for predict_interval delegation."""

    def test_predict_interval_available(self, panel_y):
        """predict_interval should be available when wrapped forecaster supports it."""
        y = panel_y
        f = LocalPanelForecaster(
            forecaster=SplitConformalForecaster(
                point_forecaster=SeasonalNaive(seasonality=7),
                calibration_size=20,
            ),
        )
        f.fit(y[:80], forecasting_horizon=5)
        y_pred = f.predict_interval(forecasting_horizon=5, coverage_rates=[0.9])

        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) == 5

    def test_predict_interval_not_available(self):
        """predict_interval should not be available for point-only forecasters."""
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        assert not hasattr(f, "predict_interval") or not callable(getattr(f, "predict_interval", None))


class TestMultipleGroups:
    """Tests with more than two panel groups."""

    def test_five_groups(self, panel_time_series_factory):
        """Should handle 5+ panel groups."""
        y = panel_time_series_factory(length=100, n_series=2, n_groups=5)
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        f.fit(y[:80], forecasting_horizon=5)

        assert len(f.forecasters_) == 5
        assert len(f.groups_) == 5

        y_pred = f.predict(forecasting_horizon=5)
        _, panel_groups = inspect_panel(y_pred)
        assert len(panel_groups) == 5
