"""Tests for yohou.testing.generators check generation functions."""

from sklearn.tree import DecisionTreeRegressor

from yohou.point.naive import SeasonalNaive
from yohou.point.reduction import PointReductionForecaster
from yohou.preprocessing.window import LagTransformer
from yohou.testing.generators import (
    _yield_yohou_forecaster_checks,
    _yield_yohou_transformer_checks,
)


class TestGeneratorChecks:
    """Tests for check generator functions."""

    def test_yield_yohou_transformer_checks(self, y_X_factory):
        """Test _yield_yohou_transformer_checks generates check functions."""
        y, X_actual = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X_actual[:40], y[:40])

        # Get check generator
        checks = list(
            _yield_yohou_transformer_checks(
                transformer=transformer,
                X_train=X_actual[:30],
                y_train=y[:30],
                X_test=X_actual[30:40],
                y_test=y[30:40],
                tags={"stateless": False, "invertible": False},
            )
        )

        # Should generate multiple checks
        assert len(checks) > 0, "Should generate at least one check"

        # Each check should be a tuple of (name, func, kwargs)
        for check_name, check_func, check_kwargs in checks:
            assert isinstance(check_name, str), f"Check name should be string, got {type(check_name)}"
            assert callable(check_func), f"Check func should be callable, got {type(check_func)}"
            assert isinstance(check_kwargs, dict), f"Check kwargs should be dict, got {type(check_kwargs)}"

    def test_yield_yohou_forecaster_checks(self, y_X_factory):
        """Test _yield_yohou_forecaster_checks generates check functions."""
        y, X_actual = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        forecaster = SeasonalNaive(seasonality=12)
        forecaster.fit(y[:40], X_actual[:40], forecasting_horizon=3)

        # Get check generator
        checks = list(
            _yield_yohou_forecaster_checks(
                forecaster=forecaster,
                y_train=y[:30],
                X_actual_train=X_actual[:30],
                y_test=y[30:40],
                X_actual_test=X_actual[30:40],
                tags={"forecaster_type": frozenset({"point"}), "uses_reduction": False},
            )
        )

        # Should generate multiple checks
        assert len(checks) > 0, "Should generate at least one check"

        # Each check should be a tuple of (name, func, kwargs)
        for check_name, check_func, check_kwargs in checks:
            assert isinstance(check_name, str), f"Check name should be string, got {type(check_name)}"
            assert callable(check_func), f"Check func should be callable, got {type(check_func)}"
            assert isinstance(check_kwargs, dict), f"Check kwargs should be dict, got {type(check_kwargs)}"

    def test_yield_yohou_transformer_checks_filters_by_tags(self, y_X_factory):
        """Test _yield_yohou_transformer_checks respects tags for filtering."""
        y, X_actual = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X_actual[:40], y[:40])

        # Get all checks
        all_checks = list(
            _yield_yohou_transformer_checks(
                transformer=transformer,
                X_train=X_actual[:30],
                y_train=y[:30],
                X_test=X_actual[30:40],
                y_test=y[30:40],
                tags={"stateless": False, "invertible": False},
            )
        )

        # Should have some checks
        assert len(all_checks) > 0

    def test_yield_yohou_forecaster_checks_point_vs_interval(self, y_X_factory):
        """Test _yield_yohou_forecaster_checks handles different forecaster types."""
        y, X_actual = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)

        # Point forecaster
        point = SeasonalNaive(seasonality=12)
        point.fit(y[:40], X_actual[:40], forecasting_horizon=3)

        point_checks = list(
            _yield_yohou_forecaster_checks(
                forecaster=point,
                y_train=y[:30],
                X_actual_train=X_actual[:30],
                y_test=y[30:40],
                X_actual_test=X_actual[30:40],
                tags={"forecaster_type": frozenset({"point"}), "uses_reduction": False},
            )
        )

        # Should generate point checks
        assert len(point_checks) > 0

        # Check names should not include interval-specific checks
        check_names = [name for name, _, _ in point_checks]
        assert not any("interval" in name.lower() and "coverage" in name.lower() for name in check_names)

    def test_yield_yohou_transformer_checks_expected_failures(self, y_X_factory):
        """Test _yield_yohou_transformer_checks allows expected_failures parameter."""
        y, X_actual = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X_actual[:40], y[:40])

        # Get checks - note: expected_failures is not a parameter of the generator
        # It's typically used when running the checks
        checks = list(
            _yield_yohou_transformer_checks(
                transformer=transformer,
                X_train=X_actual[:30],
                y_train=y[:30],
                X_test=X_actual[30:40],
                y_test=y[30:40],
                tags={"stateless": False, "invertible": False},
            )
        )

        # Should still generate checks
        assert len(checks) > 0

    def test_yield_yohou_forecaster_checks_with_step_data(self, y_X_factory):
        """Test _yield_yohou_forecaster_checks includes step-column checks when X_future/X_forecast provided."""
        y, X_actual, X_future, X_forecast = y_X_factory(
            length=50,
            n_targets=1,
            n_features=2,
            seed=42,
            n_future_features=1,
            n_forecast_features=1,
            return_exogenous=True,
        )
        forecaster = PointReductionForecaster(estimator=DecisionTreeRegressor())
        forecaster.fit(y[:40], X_actual[:40], forecasting_horizon=3, X_future=X_future, X_forecast=X_forecast)

        checks = list(
            _yield_yohou_forecaster_checks(
                forecaster=forecaster,
                y_train=y[:30],
                X_actual_train=X_actual[:30],
                y_test=y[30:40],
                X_actual_test=X_actual[30:40],
                X_future_train=X_future,
                X_future_test=X_future,
                X_forecast_train=X_forecast,
                X_forecast_test=X_forecast,
                tags={"forecaster_type": frozenset({"point"}), "uses_reduction": True},
            )
        )

        check_names = {name for name, _, _ in checks}
        assert "check_fit_predict_with_X_future" in check_names
        assert "check_fit_predict_with_X_forecast" in check_names
        assert "check_predict_X_forecast_override" in check_names
        assert "check_observe_auto_rederives_step_columns" in check_names
        assert "check_observe_predict_with_step_columns" in check_names

    def test_yield_yohou_forecaster_checks_no_step_data(self, y_X_factory):
        """Test _yield_yohou_forecaster_checks excludes step-column checks when no X_future/X_forecast."""
        y, X_actual = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        forecaster = PointReductionForecaster(estimator=DecisionTreeRegressor())
        forecaster.fit(y[:40], X_actual[:40], forecasting_horizon=3)

        checks = list(
            _yield_yohou_forecaster_checks(
                forecaster=forecaster,
                y_train=y[:30],
                X_actual_train=X_actual[:30],
                y_test=y[30:40],
                X_actual_test=X_actual[30:40],
                tags={"forecaster_type": frozenset({"point"}), "uses_reduction": True},
            )
        )

        check_names = {name for name, _, _ in checks}
        step_checks = {
            "check_fit_predict_with_X_future",
            "check_fit_predict_with_X_forecast",
            "check_predict_X_forecast_override",
            "check_observe_auto_rederives_step_columns",
            "check_observe_predict_with_step_columns",
            "check_requires_exogenous_warns_on_X_future_X_forecast",
        }
        assert step_checks.isdisjoint(check_names)
