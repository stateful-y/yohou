"""Tests for conditional yield logic in testing generators.

Step 30 of codebase quality plan: verify that the _yield_yohou_*_checks
generator functions yield the correct set of checks based on tags,
data shape, and estimator attributes.
"""

from __future__ import annotations

import polars as pl
from sklearn.tree import DecisionTreeRegressor

from yohou.interval.split_conformal import SplitConformalForecaster
from yohou.metrics.point import MeanAbsoluteError
from yohou.model_selection import GridSearchCV, RandomizedSearchCV
from yohou.model_selection.split import ExpandingWindowSplitter
from yohou.point.naive import SeasonalNaive
from yohou.point.reduction import PointReductionForecaster
from yohou.preprocessing.window import LagTransformer
from yohou.testing.generators import (
    _yield_yohou_forecaster_checks,
    _yield_yohou_scorer_checks,
    _yield_yohou_search_checks,
    _yield_yohou_splitter_checks,
    _yield_yohou_transformer_checks,
)


def _check_names(generator):
    """Collect check names from a generator."""
    return [name for name, _, _ in generator]


class TestTransformerGeneratorConditions:
    """Verify conditional branches in _yield_yohou_transformer_checks."""

    def test_stateful_true_yields_stateful_checks(self, y_X_factory):
        """Stateful tag should emit observe/rewind checks."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        t = LagTransformer(lag=3)
        t.fit(X[:40], y[:40])

        names = _check_names(
            _yield_yohou_transformer_checks(t, X[:30], y[:30], X[30:40], y[30:40], tags={"stateful": True})
        )
        stateful_checks = {
            "check_observe_concatenates_memory",
            "check_observe_transform_equivalence",
            "check_rewind_updates_memory",
            "check_memory_bounded",
            "check_insufficient_data_raises",
            "check_observe_transform_sequential_consistency",
        }
        assert stateful_checks.issubset(set(names)), f"Missing stateful checks: {stateful_checks - set(names)}"

    def test_stateful_false_skips_stateful_checks(self, y_X_factory):
        """Non-stateful tag should NOT emit stateful checks."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        t = LagTransformer(lag=3)
        t.fit(X[:40], y[:40])

        names = _check_names(
            _yield_yohou_transformer_checks(t, X[:30], y[:30], X[30:40], y[30:40], tags={"stateful": False})
        )
        assert "check_observe_concatenates_memory" not in names
        assert "check_memory_bounded" not in names

    def test_invertible_true_yields_inverse_checks(self, y_X_factory):
        """Invertible tag should emit inverse_transform checks."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        t = LagTransformer(lag=3)
        t.fit(X[:40], y[:40])

        names = _check_names(
            _yield_yohou_transformer_checks(t, X[:30], y[:30], X[30:40], y[30:40], tags={"invertible": True})
        )
        inverse_checks = {
            "check_inverse_transform_identity",
            "check_inverse_transform_round_trip",
            "check_inverse_observe_transform_identity",
        }
        assert inverse_checks.issubset(set(names)), f"Missing inverse checks: {inverse_checks - set(names)}"

    def test_invertible_false_skips_inverse_checks(self, y_X_factory):
        """Non-invertible tag should skip inverse checks."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        t = LagTransformer(lag=3)
        t.fit(X[:40], y[:40])

        names = _check_names(
            _yield_yohou_transformer_checks(t, X[:30], y[:30], X[30:40], y[30:40], tags={"invertible": False})
        )
        assert "check_inverse_transform_identity" not in names
        assert "check_inverse_transform_round_trip" not in names

    def test_panel_data_yields_panel_check(self, y_X_factory):
        """Panel data + supports_panel_data=True should yield panel check."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42, panel=True, n_groups=2)
        t = LagTransformer(lag=3)
        t.fit(X[:40], y[:40])

        names = _check_names(
            _yield_yohou_transformer_checks(
                t,
                X[:30],
                y[:30],
                X[30:40],
                y[30:40],
                tags={"supports_panel_data": True},
            )
        )
        assert "check_panel_data_support" in names

    def test_non_panel_data_skips_panel_check(self, y_X_factory):
        """Non-panel data should skip panel check even if tag is True."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42, panel=False)
        t = LagTransformer(lag=3)
        t.fit(X[:40], y[:40])

        names = _check_names(
            _yield_yohou_transformer_checks(
                t,
                X[:30],
                y[:30],
                X[30:40],
                y[30:40],
                tags={"supports_panel_data": True},
            )
        )
        assert "check_panel_data_support" not in names


class TestForecasterGeneratorConditions:
    """Verify conditional branches in _yield_yohou_forecaster_checks."""

    def test_point_type_yields_point_checks(self, y_X_factory):
        """Point forecaster type should yield point-specific checks."""
        y, X = y_X_factory(length=80, n_targets=1, n_features=2, seed=42)
        f = SeasonalNaive(seasonality=5)
        f.fit(y[:60], X[:60], forecasting_horizon=5)

        names = _check_names(
            _yield_yohou_forecaster_checks(
                f,
                y[:50],
                X[:50],
                y[50:60],
                X[50:60],
                tags={"forecaster_type": "point"},
            )
        )
        assert "check_point_prediction_structure" in names
        assert "check_point_prediction_types" in names
        assert "check_interval_prediction_columns" not in names

    def test_interval_type_yields_interval_checks(self, y_X_factory):
        """Interval forecaster type should yield interval-specific checks."""
        y, X = y_X_factory(length=200, n_targets=1, n_features=2, seed=42)
        f = SplitConformalForecaster(
            point_forecaster=PointReductionForecaster(
                estimator=DecisionTreeRegressor(),
            ),
            calibration_size=30,
        )
        f.fit(y[:150], X[:150], forecasting_horizon=5)

        names = _check_names(
            _yield_yohou_forecaster_checks(
                f,
                y[:140],
                X[:140],
                y[140:150],
                X[140:150],
                tags={"forecaster_type": "interval"},
            )
        )
        interval_checks = {
            "check_interval_prediction_columns",
            "check_interval_bounds",
            "check_interval_prediction_types",
            "check_coverage_rates_parameter",
            "check_coverage_rates_validation",
        }
        assert interval_checks.issubset(set(names)), f"Missing interval checks: {interval_checks - set(names)}"
        assert "check_point_prediction_structure" not in names

    def test_reduction_yields_estimator_checks(self, y_X_factory):
        """uses_reduction=True should yield reduction-specific checks."""
        y, X = y_X_factory(length=80, n_targets=1, n_features=2, seed=42)
        f = PointReductionForecaster(
            estimator=DecisionTreeRegressor(),
        )
        f.fit(y[:60], X[:60], forecasting_horizon=5)

        names = _check_names(
            _yield_yohou_forecaster_checks(
                f,
                y[:50],
                X[:50],
                y[50:60],
                X[50:60],
                tags={"forecaster_type": "point", "uses_reduction": True},
            )
        )
        assert "check_estimator_parameter" in names
        assert "check_reduction_strategy" in names

    def test_non_reduction_skips_estimator_checks(self, y_X_factory):
        """uses_reduction=False should skip reduction checks."""
        y, X = y_X_factory(length=80, n_targets=1, n_features=2, seed=42)
        f = SeasonalNaive(seasonality=5)
        f.fit(y[:60], X[:60], forecasting_horizon=5)

        names = _check_names(
            _yield_yohou_forecaster_checks(
                f,
                y[:50],
                X[:50],
                y[50:60],
                X[50:60],
                tags={"forecaster_type": "point", "uses_reduction": False},
            )
        )
        assert "check_estimator_parameter" not in names
        assert "check_reduction_strategy" not in names

    def test_short_y_test_skips_observe_rewind(self, y_X_factory):
        """y_test < 10 rows should skip observe/rewind checks."""
        y, X = y_X_factory(length=80, n_targets=1, n_features=2, seed=42)
        f = SeasonalNaive(seasonality=5)
        f.fit(y[:60], X[:60], forecasting_horizon=5)

        # Only 5 rows of test data (< 10)
        names = _check_names(
            _yield_yohou_forecaster_checks(
                f,
                y[:50],
                X[:50],
                y[50:55],
                X[50:55],
                tags={"forecaster_type": "point", "tracks_observations": True},
            )
        )
        assert "check_observe_extends_observations" not in names
        assert "check_rewind_replaces_observations" not in names

    def test_sufficient_y_test_yields_observe_rewind(self, y_X_factory):
        """y_test >= 10 rows with tracks_observations=True yields observe/rewind checks."""
        y, X = y_X_factory(length=80, n_targets=1, n_features=2, seed=42)
        f = SeasonalNaive(seasonality=5)
        f.fit(y[:60], X[:60], forecasting_horizon=5)

        # 15 rows of test data (>= 10)
        names = _check_names(
            _yield_yohou_forecaster_checks(
                f,
                y[:50],
                X[:50],
                y[50:65],
                X[50:65],
                tags={"forecaster_type": "point", "tracks_observations": True},
            )
        )
        assert "check_observe_extends_observations" in names
        assert "check_rewind_replaces_observations" in names

    def test_tracks_observations_false_skips_observe(self, y_X_factory):
        """tracks_observations=False should skip observe/rewind even with enough data."""
        y, X = y_X_factory(length=80, n_targets=1, n_features=2, seed=42)
        f = SeasonalNaive(seasonality=5)
        f.fit(y[:60], X[:60], forecasting_horizon=5)

        names = _check_names(
            _yield_yohou_forecaster_checks(
                f,
                y[:50],
                X[:50],
                y[50:65],
                X[50:65],
                tags={"forecaster_type": "point", "tracks_observations": False},
            )
        )
        assert "check_observe_extends_observations" not in names
        assert "check_rewind_replaces_observations" not in names

    def test_panel_data_yields_panel_checks(self, y_X_factory):
        """Panel data with supports_panel_data=True should yield panel checks."""
        y, X = y_X_factory(length=80, n_targets=1, n_features=2, seed=42, panel=True, n_groups=2)
        f = SeasonalNaive(seasonality=5)
        f.fit(y[:60], X[:60], forecasting_horizon=5)

        names = _check_names(
            _yield_yohou_forecaster_checks(
                f,
                y[:50],
                X[:50],
                y[50:60],
                X[50:60],
                tags={"forecaster_type": "point", "supports_panel_data": True},
            )
        )
        assert "check_panel_data" in names
        assert "check_panel_single_group" in names

    def test_non_panel_data_skips_panel_checks(self, y_X_factory):
        """Non-panel data should skip panel checks."""
        y, X = y_X_factory(length=80, n_targets=1, n_features=2, seed=42, panel=False)
        f = SeasonalNaive(seasonality=5)
        f.fit(y[:60], X[:60], forecasting_horizon=5)

        names = _check_names(
            _yield_yohou_forecaster_checks(
                f,
                y[:50],
                X[:50],
                y[50:60],
                X[50:60],
                tags={"forecaster_type": "point", "supports_panel_data": True},
            )
        )
        assert "check_panel_data" not in names


class TestSplitterGeneratorConditions:
    """Verify conditional branches in _yield_yohou_splitter_checks."""

    def test_yields_core_checks(self, y_X_factory):
        """Splitter generator should always yield core checks."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)
        splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)

        names = _check_names(_yield_yohou_splitter_checks(splitter, y, X, tags={"supports_panel_data": False}))
        core_checks = {
            "check_splitter_tags_accessible_before_fit",
            "check_splitter_tags_static_after_fit",
            "check_splitter_tags_match_capabilities",
            "check_splitter_produces_valid_indices",
            "check_splitter_n_splits_consistency",
            "check_splitter_non_overlapping_tests",
        }
        assert core_checks.issubset(set(names)), f"Missing core checks: {core_checks - set(names)}"

    def test_panel_support_yields_panel_check(self, y_X_factory):
        """supports_panel_data=True should yield panel check."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)
        splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)

        names = _check_names(_yield_yohou_splitter_checks(splitter, y, X, tags={"supports_panel_data": True}))
        assert "check_splitter_panel_data_support" in names

    def test_no_panel_support_skips_panel_check(self, y_X_factory):
        """supports_panel_data=False should skip panel check."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)
        splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)

        names = _check_names(_yield_yohou_splitter_checks(splitter, y, X, tags={"supports_panel_data": False}))
        assert "check_splitter_panel_data_support" not in names

    def test_parameter_constraints_yield_validation(self, y_X_factory):
        """Classes with _parameter_constraints should yield parametrized checks."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)
        splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)

        names = _check_names(_yield_yohou_splitter_checks(splitter, y, X, tags={"supports_panel_data": False}))
        constraint_checks = [n for n in names if "check_splitter_parameter_constraints" in n]
        if hasattr(ExpandingWindowSplitter, "_parameter_constraints"):
            assert len(constraint_checks) > 0, "Should yield parameter constraint checks"


class TestScorerGeneratorConditions:
    """Verify conditional branches in _yield_yohou_scorer_checks."""

    def _make_scorer_data(self, y_X_factory):
        """Create y_truth and y_pred for scorer testing."""
        y, _ = y_X_factory(length=20, n_targets=1, n_features=0, seed=42)
        y_truth = y[:10]
        y_pred = y_truth.with_columns((pl.col(c) + 0.1) for c in y_truth.columns if c != "time")
        return y_truth, y_pred

    def test_point_scorer_yields_core_checks(self, y_X_factory):
        """Point scorer should yield all core checks."""
        y_truth, y_pred = self._make_scorer_data(y_X_factory)
        scorer = MeanAbsoluteError()

        names = _check_names(
            _yield_yohou_scorer_checks(
                scorer,
                y_truth,
                y_pred,
                tags={"prediction_type": "point", "lower_is_better": True},
            )
        )
        core_checks = {
            "check_scorer_tags_accessible_before_fit",
            "check_scorer_tags_static_after_fit",
            "check_scorer_tags_match_capabilities",
            "check_scorer_lower_is_better",
            "check_scorer_methods_call_check_is_fitted",
        }
        assert core_checks.issubset(set(names))

    def test_aggregation_method_yields_aggregation_check(self, y_X_factory):
        """Scorers with aggregation_method should yield aggregation check."""
        y_truth, y_pred = self._make_scorer_data(y_X_factory)
        scorer = MeanAbsoluteError()

        names = _check_names(
            _yield_yohou_scorer_checks(
                scorer,
                y_truth,
                y_pred,
                tags={"prediction_type": "point"},
            )
        )
        if hasattr(scorer, "aggregation_method"):
            assert "check_scorer_aggregation_methods" in names

    def test_point_scorer_skips_coverage_check(self, y_X_factory):
        """Point scorer should NOT yield coverage rate check."""
        y_truth, y_pred = self._make_scorer_data(y_X_factory)
        scorer = MeanAbsoluteError()

        names = _check_names(
            _yield_yohou_scorer_checks(
                scorer,
                y_truth,
                y_pred,
                tags={"prediction_type": "point"},
            )
        )
        assert "check_scorer_coverage_rate_subselection" not in names


class TestSearchGeneratorConditions:
    """Verify conditional branches in _yield_yohou_search_checks."""

    def test_grid_yields_grid_specific_checks(self, y_X_factory):
        """Grid search should yield grid-specific checks."""
        y, X = y_X_factory(length=80, n_targets=1, n_features=0, seed=42)
        f = SeasonalNaive(seasonality=5)
        search = GridSearchCV(
            forecaster=f,
            param_grid={"seasonality": [3, 5]},
            scoring=MeanAbsoluteError(),
            cv=ExpandingWindowSplitter(n_splits=2, test_size=5),
        )
        search.fit(y[:60], forecasting_horizon=5)

        names = _check_names(
            _yield_yohou_search_checks(
                search,
                y[:50],
                None,
                y[50:60],
                None,
                tags={"search_type": "grid", "refit": True},
            )
        )
        assert "check_grid_search_exhaustive" in names
        assert "check_grid_search_param_grid_validation" in names
        assert "check_randomized_search_n_iter" not in names

    def test_randomized_yields_randomized_specific_checks(self, y_X_factory):
        """Randomized search should yield randomized-specific checks."""
        y, X = y_X_factory(length=80, n_targets=1, n_features=0, seed=42)
        f = SeasonalNaive(seasonality=5)
        search = RandomizedSearchCV(
            forecaster=f,
            param_distributions={"seasonality": [3, 5, 7]},
            n_iter=2,
            scoring=MeanAbsoluteError(),
            cv=ExpandingWindowSplitter(n_splits=2, test_size=5),
            random_state=42,
        )
        search.fit(y[:60], forecasting_horizon=5)

        names = _check_names(
            _yield_yohou_search_checks(
                search,
                y[:50],
                None,
                y[50:60],
                None,
                tags={"search_type": "randomized", "refit": True},
            )
        )
        assert "check_randomized_search_n_iter" in names
        assert "check_randomized_search_distributions" in names
        assert "check_grid_search_exhaustive" not in names

    def test_refit_true_yields_delegation_checks(self, y_X_factory):
        """refit=True should yield delegation checks."""
        y, _ = y_X_factory(length=80, n_targets=1, n_features=0, seed=42)
        f = SeasonalNaive(seasonality=5)
        search = GridSearchCV(
            forecaster=f,
            param_grid={"seasonality": [3, 5]},
            scoring=MeanAbsoluteError(),
            cv=ExpandingWindowSplitter(n_splits=2, test_size=5),
        )
        search.fit(y[:60], forecasting_horizon=5)

        names = _check_names(
            _yield_yohou_search_checks(
                search,
                y[:50],
                None,
                y[50:65],
                None,
                tags={"search_type": "grid", "refit": True},
            )
        )
        assert "check_search_predict_delegates" in names
        assert "check_search_refit_false_no_forecaster" not in names

    def test_refit_false_yields_no_forecaster_check(self, y_X_factory):
        """refit=False should yield refit_false check, not delegation."""
        y, _ = y_X_factory(length=80, n_targets=1, n_features=0, seed=42)
        f = SeasonalNaive(seasonality=5)
        search = GridSearchCV(
            forecaster=f,
            param_grid={"seasonality": [3, 5]},
            scoring=MeanAbsoluteError(),
            cv=ExpandingWindowSplitter(n_splits=2, test_size=5),
            refit=False,
        )
        search.fit(y[:60], forecasting_horizon=5)

        names = _check_names(
            _yield_yohou_search_checks(
                search,
                y[:50],
                None,
                y[50:60],
                None,
                tags={"search_type": "grid", "refit": False},
            )
        )
        assert "check_search_refit_false_no_forecaster" in names
        assert "check_search_predict_delegates" not in names
