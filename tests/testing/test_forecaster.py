"""Tests for yohou.testing.forecaster check functions."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from yohou.compose import ColumnForecaster
from yohou.interval.split_conformal import SplitConformalForecaster
from yohou.point.naive import SeasonalNaive
from yohou.point.reduction import PointReductionForecaster
from yohou.preprocessing.window import LagTransformer
from yohou.stationarity.transformers import LogTransformer
from yohou.testing.forecaster import (
    check_clone_preserves_forecaster_params,
    check_fit_predict_with_X_forecast,
    check_fit_predict_with_X_future,
    check_fit_predict_without_exogenous,
    check_fit_sets_forecaster_attributes,
    check_forecaster_not_fitted_error,
    check_forecaster_tags_accessible_before_fit,
    check_forecaster_tags_match_capabilities,
    check_forecaster_tags_static_after_fit,
    check_forecasting_horizon_validation,
    check_observe_auto_rederives_step_columns,
    check_observe_extends_observations,
    check_observe_predict_interval_with_step_columns,
    check_observe_predict_with_step_columns,
    check_predict_time_columns,
    check_predict_X_forecast_override,
    check_prediction_types_property,
    check_rewind_propagates_to_transformers,
    check_rewind_replaces_observations,
)


def test_interval_step_column_check_reexported_from_public_api():
    """check_observe_predict_interval_with_step_columns is reachable from yohou.testing."""
    import yohou.testing as testing_pkg

    assert hasattr(testing_pkg, "check_observe_predict_interval_with_step_columns"), (
        "check_observe_predict_interval_with_step_columns must be re-exported from yohou.testing"
    )
    assert "check_observe_predict_interval_with_step_columns" in testing_pkg.__all__


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
            target_transformer=LogTransformer(), actual_transformer=LagTransformer(lag=3)
        )

        # Should not raise - validates transformer_ attributes
        check_fit_sets_forecaster_attributes(forecaster, y[:40], X[:40], forecasting_horizon=3)

    def test_not_fitted_error(self, y_X_factory):
        """Test check_forecaster_not_fitted_error passes for unfitted forecaster."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        forecaster = SeasonalNaive(seasonality=12)

        # Should not raise - unfitted forecaster correctly raises NotFittedError
        # on predict, observe, and rewind (all three covered by the check).
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
        check_observe_extends_observations(forecaster, y[30:40], X[30:40])

    def test_rewind_replaces_observations(self, y_X_factory):
        """Test check_rewind_replaces_observations passes for valid forecaster."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42, panel=False)
        forecaster = SeasonalNaive(seasonality=12)
        forecaster.fit(y[:30], X[:30], forecasting_horizon=3)

        # Should not raise
        check_rewind_replaces_observations(forecaster, y[20:40], X[20:40])

    def test_rewind_propagates_to_transformers(self, y_X_factory):
        """Test check_rewind_propagates_to_transformers passes for valid forecaster."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42, panel=False)
        forecaster = PointReductionForecaster(
            target_transformer=LogTransformer(), actual_transformer=LagTransformer(lag=3)
        )
        forecaster.fit(y[:30], X[:30], forecasting_horizon=3)

        # Should not raise
        check_rewind_propagates_to_transformers(forecaster, y[20:40], X[20:40])


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
            target_transformer=LogTransformer(offset=1.0), actual_transformer=LagTransformer(lag=3)
        )

        # Should not raise
        check_clone_preserves_forecaster_params(forecaster)

    def test_tags_accessible_before_fit(self, y_X_factory):
        """Test check_forecaster_tags_accessible_before_fit passes for valid forecaster."""
        forecaster = SeasonalNaive(seasonality=12)

        # Should not raise
        check_forecaster_tags_accessible_before_fit(forecaster)

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


class TestForecasterPanelObserveRewind:
    """Tests for observe/rewind check functions with panel data."""

    @pytest.fixture(scope="class")
    def panel_data(self):
        """Generate panel (y, X) pair with 2 groups."""
        n = 60
        times = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=n - 1),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": times,
            "g0__val": [float(i % 7) for i in range(n)],
            "g1__val": [float(i % 7) + 1 for i in range(n)],
        })
        return y

    def test_predict_time_columns_panel_stacked(self, panel_data):
        """check_predict_time_columns tolerates stacked panel predictions.

        A panel forecaster that stacks predictions per group emits
        ``G * forecasting_horizon`` rows, so the check must not assert a strict
        single-group row count equal to the horizon when ``groups_`` is set.
        """
        y = panel_data
        n_groups = 2
        forecasting_horizon = 3

        class _StackedPanelForecaster:
            """Wraps a fitted panel forecaster but emits stacked per-group rows."""

            groups_ = ["g0", "g1"]

            def __init__(self, inner):
                self._inner = inner

            def predict(self, forecasting_horizon):
                wide = self._inner.predict(forecasting_horizon=forecasting_horizon)
                # Stack: one block of `horizon` rows per group (G * H rows total).
                return pl.concat([wide.select("vintage_time", "time") for _ in self.groups_])

        inner = SeasonalNaive(seasonality=7)
        inner.fit(y.head(40), forecasting_horizon=forecasting_horizon)
        stacked = _StackedPanelForecaster(inner)

        produced = stacked.predict(forecasting_horizon=forecasting_horizon)
        assert len(produced) == n_groups * forecasting_horizon  # sanity: more than horizon

        # Must not raise even though row count != forecasting_horizon.
        check_predict_time_columns(stacked, y.head(40))

    def test_observe_extends_panel_observations(self, panel_data):
        """check_observe_extends_observations passes for panel forecaster."""
        y = panel_data
        forecaster = SeasonalNaive(seasonality=7)
        forecaster.fit(y.head(40), forecasting_horizon=3)
        check_observe_extends_observations(
            forecaster,
            y.slice(40, 10),
            None,
        )

    def test_rewind_replaces_panel_observations(self, panel_data):
        """check_rewind_replaces_observations passes for panel forecaster."""
        y = panel_data
        forecaster = SeasonalNaive(seasonality=7)
        forecaster.fit(y.head(40), forecasting_horizon=3)
        check_rewind_replaces_observations(
            forecaster,
            y.slice(20, 20),
            None,
        )


class TestForecasterCloneComposition:
    """Tests for clone check with composition forecasters (list-of-tuples params)."""

    def test_clone_column_forecaster(self):
        """ColumnForecaster clone preserves 3-tuple params."""
        forecaster = ColumnForecaster(
            forecasters=[
                ("naive_a", SeasonalNaive(seasonality=7), ["val_a"]),
                ("naive_b", SeasonalNaive(seasonality=14), ["val_b"]),
            ],
        )
        check_clone_preserves_forecaster_params(forecaster)


class TestForecasterTagsWithTransformers:
    """Tests for tag capability checks with transformer-bearing forecasters."""

    def test_tags_match_with_target_transformer(self, y_X_factory):
        """Tag check passes for forecaster with target_transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        forecaster = PointReductionForecaster(
            target_transformer=LogTransformer(),
            actual_transformer=LagTransformer(lag=3),
        )
        forecaster.fit(y[:40], X[:40], forecasting_horizon=3)
        check_forecaster_tags_match_capabilities(forecaster)


class TestForecasterFitPredictWithoutX:
    """Tests for check_fit_predict_without_exogenous."""

    def test_requires_exogenous_forecaster_succeeds(self, y_X_factory):
        """Forecaster with requires_exogenous=False succeeds without X."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        forecaster = SeasonalNaive(seasonality=7)
        check_fit_predict_without_exogenous(
            forecaster,
            y[:40],
            requires_exogenous=False,
            forecasting_horizon=3,
        )

    def test_target_as_feature_none_raises(self, y_X_factory):
        """Forecaster with target_as_feature=None and no X raises ValueError."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        forecaster = PointReductionForecaster(target_as_feature=None)
        check_fit_predict_without_exogenous(
            forecaster,
            y[:40],
            requires_exogenous=True,
            target_as_feature=None,
            forecasting_horizon=3,
        )


class TestForecasterIntervalTags:
    """Tests for tag checks with interval/both-type forecasters."""

    def test_interval_forecaster_tags(self, y_X_factory):
        """Tag check passes for interval (both-type) forecaster."""
        y, _ = y_X_factory(length=200, n_targets=1, n_features=0, seed=42)
        forecaster = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
        )
        forecaster.fit(y[:150], forecasting_horizon=3)
        check_forecaster_tags_match_capabilities(forecaster)


class TestStepColumnChecks:
    """Tests for X_future/X_forecast step-column check functions."""

    def test_check_fit_predict_with_X_future(self, y_X_factory):
        """check_fit_predict_with_X_future passes for a reduction forecaster."""
        y, X, X_future, X_forecast = y_X_factory(
            length=60,
            n_targets=1,
            n_features=2,
            seed=42,
            n_future_features=2,
            n_forecast_features=0,
            return_exogenous=True,
        )
        forecaster = PointReductionForecaster()
        check_fit_predict_with_X_future(
            forecaster,
            y[:50],
            X[:50],
            X_future=X_future,
            forecasting_horizon=3,
        )

    def test_check_fit_predict_with_X_forecast(self, y_X_factory):
        """check_fit_predict_with_X_forecast passes for a reduction forecaster."""
        y, X, X_future, X_forecast = y_X_factory(
            length=60,
            n_targets=1,
            n_features=2,
            seed=42,
            n_future_features=0,
            n_forecast_features=2,
            return_exogenous=True,
        )
        forecaster = PointReductionForecaster()
        check_fit_predict_with_X_forecast(
            forecaster,
            y[:50],
            X[:50],
            X_forecast=X_forecast,
            forecasting_horizon=3,
        )

    def test_check_predict_X_forecast_override(self, y_X_factory):
        """check_predict_X_forecast_override passes for a fitted forecaster."""
        y, X, X_future, X_forecast = y_X_factory(
            length=60,
            n_targets=1,
            n_features=2,
            seed=42,
            n_future_features=0,
            n_forecast_features=2,
            return_exogenous=True,
        )
        forecaster = PointReductionForecaster()
        forecaster.fit(y[:50], X[:50], forecasting_horizon=3, X_forecast=X_forecast)
        check_predict_X_forecast_override(
            forecaster,
            X_forecast=X_forecast,
            forecasting_horizon=3,
        )

    def test_check_observe_auto_rederives_step_columns(self, y_X_factory):
        """check_observe_auto_rederives_step_columns passes for a fitted forecaster."""
        y, X, X_future, X_forecast = y_X_factory(
            length=60,
            n_targets=1,
            n_features=2,
            seed=42,
            n_future_features=2,
            n_forecast_features=0,
            return_exogenous=True,
        )
        forecaster = PointReductionForecaster()
        forecaster.fit(y[:50], X[:50], forecasting_horizon=3, X_future=X_future)
        check_observe_auto_rederives_step_columns(
            forecaster,
            y[50:53],
            X_actual_observe=X[50:53],
            X_future=X_future,
        )

    def test_check_observe_predict_with_step_columns(self, y_X_factory):
        """check_observe_predict_with_step_columns passes for a reduction forecaster."""
        y, X, X_future, X_forecast = y_X_factory(
            length=80,
            n_targets=1,
            n_features=2,
            seed=42,
            n_future_features=2,
            n_forecast_features=2,
            return_exogenous=True,
        )
        forecaster = PointReductionForecaster()
        check_observe_predict_with_step_columns(
            forecaster,
            y[:60],
            X[:60],
            y[60:80],
            X_actual_test=X[60:80],
            X_future=X_future,
            X_forecast=X_forecast,
            forecasting_horizon=3,
        )

    def test_check_observe_predict_interval_with_step_columns(self, y_X_factory):
        """check_observe_predict_interval_with_step_columns passes for a conformal forecaster."""
        y, X, X_future, X_forecast = y_X_factory(
            length=80,
            n_targets=1,
            n_features=2,
            seed=42,
            n_future_features=2,
            n_forecast_features=2,
            return_exogenous=True,
        )
        forecaster = SplitConformalForecaster(calibration_size=20)
        check_observe_predict_interval_with_step_columns(
            forecaster,
            y[:60],
            X[:60],
            y[60:80],
            X_actual_test=X[60:80],
            X_future=X_future,
            X_forecast=X_forecast,
            forecasting_horizon=3,
        )
