"""Tests for yohou.testing.forecaster check functions."""

from yohou.point_forecaster.naive import SeasonalNaive
from yohou.point_forecaster.reduction import PointReductionForecaster
from yohou.preprocessing.stationarization import LogTransform
from yohou.preprocessing.window import LagTransformer
from yohou.testing.forecaster import (
    check_clone_preserves_forecaster_params,
    check_fit_sets_forecaster_attributes,
    check_forecaster_not_fitted_error,
    check_forecaster_tags_accessible_before_fit,
    check_forecaster_tags_match_capabilities,
    check_forecaster_tags_static_after_fit,
    check_forecasting_horizon_validation,
    check_predict_time_columns,
    check_prediction_types_property,
    check_reset_propagates_to_transformers,
    check_reset_replaces_observations,
    check_update_extends_observations,
)


def test_check_fit_sets_forecaster_attributes(y_X_factory):
    """Test check_fit_sets_forecaster_attributes passes for valid forecaster."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    forecaster = SeasonalNaive(seasonality=12)

    # Should not raise
    check_fit_sets_forecaster_attributes(forecaster, y[:40], X[:40], forecasting_horizon=3)


def test_check_fit_sets_forecaster_attributes_with_transformers(y_X_factory):
    """Test check validates transformer attributes are set."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    forecaster = PointReductionForecaster(target_transformer=LogTransform(), feature_transformer=LagTransformer(lag=3))

    # Should not raise - validates transformer_ attributes
    check_fit_sets_forecaster_attributes(forecaster, y[:40], X[:40], forecasting_horizon=3)


def test_check_forecaster_not_fitted_error(y_X_factory):
    """Test check_forecaster_not_fitted_error passes for unfitted forecaster."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    forecaster = SeasonalNaive(seasonality=12)

    # Should not raise - unfitted forecaster correctly raises NotFittedError
    check_forecaster_not_fitted_error(forecaster, y, X)


def test_check_forecaster_not_fitted_error_raises_on_predict(y_X_factory):
    """Test check validates predict raises NotFittedError."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    forecaster = SeasonalNaive(seasonality=12)

    # Check validates predict/update/reset all raise NotFittedError
    # Should not raise - unfitted forecaster correctly raises
    check_forecaster_not_fitted_error(forecaster, y, X)


def test_check_predict_time_columns(y_X_factory):
    """Test check_predict_time_columns passes for valid forecaster."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    forecaster = SeasonalNaive(seasonality=12)
    forecaster.fit(y[:40], X[:40], forecasting_horizon=3)

    # Should not raise
    check_predict_time_columns(forecaster, y[:40], X[:40])


def test_check_update_extends_observations(y_X_factory):
    """Test check_update_extends_observations passes for valid forecaster."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42, panel=False)
    forecaster = SeasonalNaive(seasonality=12)
    forecaster.fit(y[:30], X[:30], forecasting_horizon=3)

    # Should not raise
    check_update_extends_observations(forecaster, y[:30], y[30:40], X[:30], X[30:40])


def test_check_reset_replaces_observations(y_X_factory):
    """Test check_reset_replaces_observations passes for valid forecaster."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42, panel=False)
    forecaster = SeasonalNaive(seasonality=12)
    forecaster.fit(y[:30], X[:30], forecasting_horizon=3)

    # Should not raise
    check_reset_replaces_observations(forecaster, y[:30], y[20:40], X[:30], X[20:40])


def test_check_reset_propagates_to_transformers(y_X_factory):
    """Test check_reset_propagates_to_transformers passes for valid forecaster."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42, panel=False)
    forecaster = PointReductionForecaster(target_transformer=LogTransform(), feature_transformer=LagTransformer(lag=3))
    forecaster.fit(y[:30], X[:30], forecasting_horizon=3)

    # Should not raise
    check_reset_propagates_to_transformers(forecaster, y[:30], y[20:40], X[:30], X[20:40])


def test_check_forecasting_horizon_validation(y_X_factory):
    """Test check_forecasting_horizon_validation passes for valid forecaster."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    forecaster = SeasonalNaive(seasonality=12)
    forecaster.fit(y[:40], X[:40], forecasting_horizon=3)

    # Should not raise
    check_forecasting_horizon_validation(forecaster, y[:40], X[:40])


def test_check_prediction_types_property():
    """Test check_prediction_types_property passes for valid forecaster."""
    forecaster = SeasonalNaive(seasonality=12)

    # Should not raise
    check_prediction_types_property(forecaster)


def test_check_clone_preserves_forecaster_params():
    """Test check_clone_preserves_forecaster_params passes for valid forecaster."""
    forecaster = SeasonalNaive(seasonality=12)

    # Should not raise
    check_clone_preserves_forecaster_params(forecaster)


def test_check_clone_preserves_forecaster_params_with_transformers():
    """Test check validates transformer parameters are preserved."""
    forecaster = PointReductionForecaster(
        target_transformer=LogTransform(offset=1.0), feature_transformer=LagTransformer(lag=3)
    )

    # Should not raise
    check_clone_preserves_forecaster_params(forecaster)


def test_check_forecaster_tags_accessible_before_fit(y_X_factory):
    """Test check_forecaster_tags_accessible_before_fit passes for valid forecaster."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    forecaster = SeasonalNaive(seasonality=12)

    # Should not raise
    check_forecaster_tags_accessible_before_fit(forecaster, y, X)


def test_check_forecaster_tags_static_after_fit(y_X_factory):
    """Test check_forecaster_tags_static_after_fit passes for valid forecaster."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    forecaster = SeasonalNaive(seasonality=12)
    forecaster.fit(y[:40], X[:40], forecasting_horizon=3)

    # Should not raise
    check_forecaster_tags_static_after_fit(forecaster, y[:40], X[:40])


def test_check_forecaster_tags_match_capabilities(y_X_factory):
    """Test check_forecaster_tags_match_capabilities passes for valid forecaster."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
    forecaster = SeasonalNaive(seasonality=12)
    forecaster.fit(y[:40], X[:40], forecasting_horizon=3)

    # Should not raise
    check_forecaster_tags_match_capabilities(forecaster)
