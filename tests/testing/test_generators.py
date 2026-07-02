"""Tests for yohou.testing.generators check generation functions."""

from sklearn.tree import DecisionTreeRegressor
from sklearn.utils._param_validation import Interval

from yohou.point.naive import SeasonalNaive
from yohou.point.reduction import PointReductionForecaster
from yohou.preprocessing.window import LagTransformer
from yohou.testing.generators import (
    _interval_lower_bound,
    _yield_yohou_forecaster_checks,
    _yield_yohou_transformer_checks,
)


class TestIntervalLowerBound:
    """Tests for the _interval_lower_bound constraint helper."""

    def test_returns_int_lower_bound(self):
        """The integer lower bound of the first Interval constraint is returned."""
        import numbers

        constraints = [Interval(numbers.Integral, 1, None, closed="left")]
        assert _interval_lower_bound(constraints) == 1

    def test_returns_none_without_interval(self):
        """A constraint list with no bound returns None."""
        assert _interval_lower_bound([list, None]) is None


class TestGeneratorChecks:
    """Tests for check generator functions."""

    def test_yield_yohou_transformer_checks(self, y_X_factory):
        """Generator yields well-formed tuples and gates inverse checks on the invertible tag."""
        y, X_actual = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X_actual[:40], y[:40])

        inverse_checks = {
            "check_inverse_transform_identity",
            "check_inverse_transform_round_trip",
            "check_inverse_observe_transform_identity",
        }

        def _checks(invertible):
            return list(
                _yield_yohou_transformer_checks(
                    transformer=transformer,
                    X_train=X_actual[:30],
                    y_train=y[:30],
                    X_test=X_actual[30:40],
                    y_test=y[30:40],
                    tags={"stateless": False, "invertible": invertible},
                )
            )

        non_invertible_checks = _checks(invertible=False)
        invertible_checks = _checks(invertible=True)

        # Every yielded item is a (name, callable, kwargs) tuple.
        assert len(non_invertible_checks) > 0, "Should generate at least one check"
        for check_name, check_func, check_kwargs in non_invertible_checks:
            assert isinstance(check_name, str), f"Check name should be string, got {type(check_name)}"
            assert callable(check_func), f"Check func should be callable, got {type(check_func)}"
            assert isinstance(check_kwargs, dict), f"Check kwargs should be dict, got {type(check_kwargs)}"

        # The invertible tag is the only difference, and it gates the inverse checks.
        non_invertible_names = {name for name, _, _ in non_invertible_checks}
        invertible_names = {name for name, _, _ in invertible_checks}
        assert inverse_checks.isdisjoint(non_invertible_names)
        assert inverse_checks.issubset(invertible_names)

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
