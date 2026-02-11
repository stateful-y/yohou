"""Tests for yohou.testing.forecaster check functions."""

from yohou.point.naive import SeasonalNaive
from yohou.point.reduction import PointReductionForecaster
from yohou.preprocessing.window import LagTransformer
from yohou.stationarity.transformers import LogTransformer
from yohou.testing.forecaster import (
    check_clone_preserves_forecaster_params,
    check_fit_sets_forecaster_attributes,
    check_forecaster_not_fitted_error,
    check_forecaster_tags_accessible_before_fit,
    check_forecaster_tags_match_capabilities,
    check_forecaster_tags_static_after_fit,
    check_forecasting_horizon_validation,
    check_observe_extends_observations,
    check_predict_time_columns,
    check_prediction_types_property,
    check_rewind_propagates_to_transformers,
    check_rewind_replaces_observations,
)


class TestForecasterFitChecks:
    """Tests for forecaster fit-related check functions."""

    def test_fit_sets_attributes(self, y_X_factory):
        """Test check_fit_sets_forecaster_attributes passes for valid forecaster."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        forecaster = SeasonalNaive(seasonality=12)

        # Should not raise
        check_fit_sets_forecaster_attributes(forecaster, y[:40], X[:40], forecasting_horizon=3)

    def test_fit_sets_attributes_with_transformers(self, y_X_factory):
        """Test check validates transformer attributes are set."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        forecaster = PointReductionForecaster(
            target_transformer=LogTransformer(), feature_transformer=LagTransformer(lag=3)
        )

        # Should not raise - validates transformer_ attributes
        check_fit_sets_forecaster_attributes(forecaster, y[:40], X[:40], forecasting_horizon=3)

    def test_not_fitted_error(self, y_X_factory):
        """Test check_forecaster_not_fitted_error passes for unfitted forecaster."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        forecaster = SeasonalNaive(seasonality=12)

        # Should not raise - unfitted forecaster correctly raises NotFittedError
        check_forecaster_not_fitted_error(forecaster, y, X)

    def test_not_fitted_error_raises_on_predict(self, y_X_factory):
        """Test check validates predict raises NotFittedError."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        forecaster = SeasonalNaive(seasonality=12)

        # Check validates predict/observe/rewind all raise NotFittedError
        # Should not raise - unfitted forecaster correctly raises
        check_forecaster_not_fitted_error(forecaster, y, X)


class TestForecasterPredictChecks:
    """Tests for forecaster predict-related check functions."""

    def test_predict_time_columns(self, y_X_factory):
        """Test check_predict_time_columns passes for valid forecaster."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        forecaster = SeasonalNaive(seasonality=12)
        forecaster.fit(y[:40], X[:40], forecasting_horizon=3)

        # Should not raise
        check_predict_time_columns(forecaster, y[:40], X[:40])

    def test_forecasting_horizon_validation(self, y_X_factory):
        """Test check_forecasting_horizon_validation passes for valid forecaster."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        forecaster = SeasonalNaive(seasonality=12)
        forecaster.fit(y[:40], X[:40], forecasting_horizon=3)

        # Should not raise
        check_forecasting_horizon_validation(forecaster, y[:40], X[:40])

    def test_prediction_types_property(self):
        """Test check_prediction_types_property passes for valid forecaster."""
        forecaster = SeasonalNaive(seasonality=12)

        # Should not raise
        check_prediction_types_property(forecaster)


class TestForecasterObserveRewindChecks:
    """Tests for forecaster observe/rewind check functions."""

    def test_observe_extends_observations(self, y_X_factory):
        """Test check_observe_extends_observations passes for valid forecaster."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42, panel=False)
        forecaster = SeasonalNaive(seasonality=12)
        forecaster.fit(y[:30], X[:30], forecasting_horizon=3)

        # Should not raise
        check_observe_extends_observations(forecaster, y[:30], y[30:40], X[:30], X[30:40])

    def test_rewind_replaces_observations(self, y_X_factory):
        """Test check_rewind_replaces_observations passes for valid forecaster."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42, panel=False)
        forecaster = SeasonalNaive(seasonality=12)
        forecaster.fit(y[:30], X[:30], forecasting_horizon=3)

        # Should not raise
        check_rewind_replaces_observations(forecaster, y[:30], y[20:40], X[:30], X[20:40])

    def test_rewind_propagates_to_transformers(self, y_X_factory):
        """Test check_rewind_propagates_to_transformers passes for valid forecaster."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42, panel=False)
        forecaster = PointReductionForecaster(
            target_transformer=LogTransformer(), feature_transformer=LagTransformer(lag=3)
        )
        forecaster.fit(y[:30], X[:30], forecasting_horizon=3)

        # Should not raise
        check_rewind_propagates_to_transformers(forecaster, y[:30], y[20:40], X[:30], X[20:40])


class TestForecasterCloneAndTagChecks:
    """Tests for forecaster clone and tag check functions."""

    def test_clone_preserves_params(self):
        """Test check_clone_preserves_forecaster_params passes for valid forecaster."""
        forecaster = SeasonalNaive(seasonality=12)

        # Should not raise
        check_clone_preserves_forecaster_params(forecaster)

    def test_clone_preserves_params_with_transformers(self):
        """Test check validates transformer parameters are preserved."""
        forecaster = PointReductionForecaster(
            target_transformer=LogTransformer(offset=1.0), feature_transformer=LagTransformer(lag=3)
        )

        # Should not raise
        check_clone_preserves_forecaster_params(forecaster)

    def test_tags_accessible_before_fit(self, y_X_factory):
        """Test check_forecaster_tags_accessible_before_fit passes for valid forecaster."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        forecaster = SeasonalNaive(seasonality=12)

        # Should not raise
        check_forecaster_tags_accessible_before_fit(forecaster, y, X)

    def test_tags_static_after_fit(self, y_X_factory):
        """Test check_forecaster_tags_static_after_fit passes for valid forecaster."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        forecaster = SeasonalNaive(seasonality=12)
        forecaster.fit(y[:40], X[:40], forecasting_horizon=3)

        # Should not raise
        check_forecaster_tags_static_after_fit(forecaster, y[:40], X[:40])

    def test_tags_match_capabilities(self, y_X_factory):
        """Test check_forecaster_tags_match_capabilities passes for valid forecaster."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        forecaster = SeasonalNaive(seasonality=12)
        forecaster.fit(y[:40], X[:40], forecasting_horizon=3)

        # Should not raise
        check_forecaster_tags_match_capabilities(forecaster)
