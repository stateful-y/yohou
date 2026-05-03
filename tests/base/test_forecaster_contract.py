"""Contract tests for BaseForecaster lifecycle methods.

Verifies fit, observe, rewind, predict, observation_horizon,
tags, and fitted attributes using a minimal concrete forecaster.
"""

import sys
from pathlib import Path

import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import SimpleTransformer
from yohou.point import PointReductionForecaster


class TestBaseForecasterFit:
    """Tests for fit lifecycle and fitted attributes."""

    def test_fit_returns_self(self, y_X_factory):
        """Fit returns the forecaster instance."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster()
        result = f.fit(y, forecasting_horizon=3)
        assert result is f

    def test_fit_sets_forecasting_horizon(self, y_X_factory):
        """Fit sets fit_forecasting_horizon_ attribute."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster()
        f.fit(y, forecasting_horizon=5)
        assert f.fit_forecasting_horizon_ == 5

    def test_fit_sets_interval(self, y_X_factory):
        """Fit sets interval_ from time series."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster()
        f.fit(y, forecasting_horizon=1)
        assert hasattr(f, "interval_")

    def test_fit_sets_y_observed(self, y_X_factory):
        """Fit sets _y_observed buffer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster()
        f.fit(y, forecasting_horizon=1)
        assert hasattr(f, "_y_observed")


class TestBaseForecasterPredict:
    """Tests for predict lifecycle."""

    def test_predict_returns_dataframe(self, y_X_factory):
        """Predict returns a polars DataFrame."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster()
        f.fit(y, forecasting_horizon=3)
        result = f.predict()
        assert isinstance(result, pl.DataFrame)

    def test_predict_has_time_columns(self, y_X_factory):
        """Predict output has vintage_time and time columns."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster()
        f.fit(y, forecasting_horizon=3)
        result = f.predict()
        assert "time" in result.columns
        assert "vintage_time" in result.columns

    def test_predict_length_matches_horizon(self, y_X_factory):
        """Predict returns rows equal to forecasting_horizon."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster()
        f.fit(y, forecasting_horizon=5)
        result = f.predict()
        assert len(result) == 5

    def test_predict_different_horizon(self, y_X_factory):
        """Predict with different horizon than fit."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster()
        f.fit(y, forecasting_horizon=3)
        result = f.predict(forecasting_horizon=7)
        assert len(result) == 7


class TestBaseForecasterObserve:
    """Tests for observe() method."""

    def test_observe_returns_self(self, y_X_factory):
        """Observe returns the forecaster instance."""
        y, X = y_X_factory(length=60, n_targets=1, n_features=0)
        f = PointReductionForecaster()
        f.fit(y[:50], forecasting_horizon=1)
        result = f.observe(y[50:])
        assert result is f

    def test_observe_not_fitted_raises(self, y_X_factory):
        """Observe before fit raises NotFittedError."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster()
        with pytest.raises(NotFittedError):
            f.observe(y)


class TestBaseForecasterRewind:
    """Tests for rewind() method."""

    def test_rewind_returns_self(self, y_X_factory):
        """Rewind returns the forecaster instance."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster()
        f.fit(y, forecasting_horizon=1)
        result = f.rewind(y)
        assert result is f


class TestBaseForecasterObservationHorizon:
    """Tests for observation_horizon property."""

    def test_horizon_after_fit(self, y_X_factory):
        """Observation horizon is accessible after fit."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster()
        f.fit(y, forecasting_horizon=1)
        assert f.observation_horizon >= 0

    def test_horizon_includes_transformer(self, y_X_factory):
        """Observation horizon includes target transformer horizon."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster(
            target_transformer=SimpleTransformer(observation_horizon=5),
        )
        f.fit(y, forecasting_horizon=1)
        assert f.observation_horizon >= 5


class TestBaseForecasterTags:
    """Tests for __sklearn_tags__()."""

    def test_estimator_type_is_forecaster(self):
        """Tags report estimator_type as forecaster."""
        f = PointReductionForecaster()
        tags = f.__sklearn_tags__()
        assert tags.estimator_type == "forecaster"

    def test_forecaster_type_is_point(self):
        """PointReductionForecaster has forecaster_type=point tag."""
        f = PointReductionForecaster()
        tags = f.__sklearn_tags__()
        assert tags.forecaster_tags.forecaster_type == frozenset({"point"})


class TestBaseForecasterWithExogenous:
    """Tests for forecasters with exogenous features."""

    def test_fit_with_X(self, y_X_factory):
        """Fit with exogenous features completes successfully."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2)
        f = PointReductionForecaster()
        f.fit(y, X, forecasting_horizon=1)
        # Verify the forecaster can produce predictions
        result = f.predict()
        assert isinstance(result, pl.DataFrame)

    def test_observe_with_X(self, y_X_factory):
        """Observe with exogenous features succeeds."""
        y, X = y_X_factory(length=60, n_targets=1, n_features=2)
        f = PointReductionForecaster()
        f.fit(y[:50], X[:50], forecasting_horizon=1)
        result = f.observe(y[50:], X[50:])
        assert result is f


class TestBaseForecasterPreFitValidation:
    """Tests for _validate_pre_fit error paths."""

    def test_feature_transformer_without_X_raises(self, y_X_factory):
        """Feature transformer with target_as_feature=None and no X raises."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster(
            feature_transformer=SimpleTransformer(observation_horizon=0),
            target_as_feature=None,
        )
        with pytest.raises(ValueError, match="feature_transformer requires X"):
            f.fit(y, forecasting_horizon=1)

    def test_no_X_with_exogenous_forecaster_raises(self, y_X_factory):
        """Forecaster with ignores_exogenous=False and no X raises."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster(target_as_feature=None)
        with pytest.raises(ValueError, match="target_as_feature=None requires X"):
            f.fit(y, forecasting_horizon=1)


class TestBaseForecasterRewindObservationHorizonZero:
    """Tests for rewind with observation_horizon == 0."""

    def test_rewind_zero_observation_horizon(self, y_X_factory):
        """Rewind special-cases observation_horizon == 0."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster()
        f.fit(y[:40], forecasting_horizon=3)
        assert f.observation_horizon == 0
        result = f.rewind(y[:40])
        assert result is f

    def test_rewind_zero_horizon_predict_matches(self, y_X_factory):
        """Predictions after rewind with horizon=0 remain consistent."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=0)
        f = PointReductionForecaster()
        f.fit(y[:40], forecasting_horizon=3)
        pred_before = f.predict()
        f.rewind(y[:40])
        pred_after = f.predict()
        assert pred_before.equals(pred_after)


class TestBaseForecasterPanelObserve:
    """Tests for panel observe path."""

    def test_panel_observe(self, y_X_factory):
        """Observe dispatches to panel observe when fitted on panel data."""
        y, X = y_X_factory(length=60, n_targets=1, n_features=0, panel=True, n_groups=2)
        f = PointReductionForecaster()
        f.fit(y[:50], forecasting_horizon=1)
        result = f.observe(y[50:])
        assert result is f

    def test_panel_rewind(self, y_X_factory):
        """Rewind dispatches to panel rewind when fitted on panel data."""
        y, X = y_X_factory(length=60, n_targets=1, n_features=0, panel=True, n_groups=2)
        f = PointReductionForecaster()
        f.fit(y[:50], forecasting_horizon=1)
        result = f.rewind(y[:50])
        assert result is f
