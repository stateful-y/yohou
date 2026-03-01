"""Systematic tests for GridSearchCV and RandomizedSearchCV using check generators."""

import numpy as np
import pytest
from scipy.stats import randint
from sklearn.base import clone

from conftest import run_checks
from yohou.metrics import MeanAbsoluteError, RootMeanSquaredError
from yohou.model_selection import GridSearchCV, RandomizedSearchCV
from yohou.point import SeasonalNaive
from yohou.testing import _yield_yohou_search_checks


class TestSystematicChecks:
    """Parametrized systematic validation of search CV classes."""

    @pytest.mark.parametrize(
        "search_cv_class,params,tags,expected_failures",
        [
            # GridSearchCV with single metric
            (
                GridSearchCV,
                {
                    "param_grid": {"seasonality": [1, 5, 10]},
                    "scoring": MeanAbsoluteError(),
                    "cv": 2,
                    "refit": True,
                },
                {"search_type": "grid", "refit": True, "multimetric": False},
                [],
            ),
            # GridSearchCV with multi-metric
            (
                GridSearchCV,
                {
                    "param_grid": {"seasonality": [1, 7]},
                    "scoring": {
                        "mae": MeanAbsoluteError(),
                        "rmse": RootMeanSquaredError(),
                    },
                    "cv": 2,
                    "refit": "mae",
                },
                {"search_type": "grid", "refit": True, "multimetric": True},
                [],
            ),
            # GridSearchCV with refit=False
            (
                GridSearchCV,
                {
                    "param_grid": {"seasonality": [1, 5]},
                    "scoring": MeanAbsoluteError(),
                    "cv": 2,
                    "refit": False,
                },
                {"search_type": "grid", "refit": False, "multimetric": False},
                [],
            ),
            # RandomizedSearchCV with single metric
            (
                RandomizedSearchCV,
                {
                    "param_distributions": {"seasonality": [1, 5, 10, 15]},
                    "n_iter": 3,
                    "scoring": MeanAbsoluteError(),
                    "cv": 2,
                    "random_state": 42,
                    "refit": True,
                },
                {"search_type": "randomized", "refit": True, "multimetric": False},
                [],
            ),
            # RandomizedSearchCV with scipy distributions
            (
                RandomizedSearchCV,
                {
                    "param_distributions": {"seasonality": randint(low=1, high=11)},
                    "n_iter": 5,
                    "scoring": MeanAbsoluteError(),
                    "cv": 2,
                    "random_state": 42,
                    "refit": True,
                },
                {"search_type": "randomized", "refit": True, "multimetric": False},
                [],
            ),
            # RandomizedSearchCV with multi-metric
            (
                RandomizedSearchCV,
                {
                    "param_distributions": {"seasonality": [1, 5, 10]},
                    "n_iter": 2,
                    "scoring": {
                        "mae": MeanAbsoluteError(),
                        "rmse": RootMeanSquaredError(),
                    },
                    "cv": 2,
                    "random_state": 42,
                    "refit": "rmse",
                },
                {"search_type": "randomized", "refit": True, "multimetric": True},
                [],
            ),
        ],
    )
    @pytest.mark.slow
    def test_search_cv_systematic_checks(self, search_cv_class, params, tags, expected_failures, y_X_factory):
        """Run systematic checks on search CV classes using generator pattern."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)
        y_train, y_test = y[:80], y[80:]
        X_train, X_test = (X[:80], X[80:]) if X is not None else (None, None)

        forecaster = SeasonalNaive()
        search_cv = search_cv_class(forecaster=forecaster, **params)
        search_cv_fitted = clone(search_cv)
        search_cv_fitted.fit(y_train, X_train, forecasting_horizon=3)

        run_checks(
            search_cv_fitted,
            _yield_yohou_search_checks(search_cv_fitted, y_train, X_train, y_test, X_test, tags=tags),
            expected_failures=set(expected_failures),
        )


class TestPanelData:
    """Tests for search CV with panel data."""

    @pytest.mark.parametrize(
        "search_cv_class,params",
        [
            (
                GridSearchCV,
                {
                    "param_grid": {"seasonality": [1, 5]},
                    "scoring": MeanAbsoluteError(),
                    "cv": 2,
                },
            ),
            (
                RandomizedSearchCV,
                {
                    "param_distributions": {"seasonality": [1, 5, 10]},
                    "n_iter": 2,
                    "scoring": MeanAbsoluteError(),
                    "cv": 2,
                    "random_state": 42,
                },
            ),
        ],
    )
    def test_search_cv_panel_data(self, search_cv_class, params, y_X_panel_factory):
        """Test search CV with panel data using prefixed columns."""
        y_panel, X_panel = y_X_panel_factory(n_groups=2, length=80, n_targets=1, n_features=2, seed=42)
        y_train, _y_test = y_panel[:60], y_panel[60:]
        X_train, X_test = (X_panel[:60], X_panel[60:]) if X_panel is not None else (None, None)

        forecaster = SeasonalNaive()
        search_cv = search_cv_class(forecaster=forecaster, **params)
        search_cv.fit(y_train, X_train, forecasting_horizon=3)

        y_pred = search_cv.predict(forecasting_horizon=3, X=X_test)

        assert "time" in y_pred.columns
        assert len(y_pred) == 3

        non_time_cols = [c for c in y_pred.columns if c not in {"time", "observed_time"}]
        assert len(non_time_cols) > 0, "Should have prediction columns"
        assert any("__" in col for col in non_time_cols), "Should have panel prefixes"


class TestEdgeCases:
    """Tests for edge cases in search CV."""

    def test_grid_search_empty_param_grid(self, y_X_factory):
        """Test GridSearchCV with single-value param_grid (degenerate case)."""
        y, X = y_X_factory(length=100, seed=42)
        y_train = y[:80]
        X_train = X[:80] if X is not None else None

        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [5]},
            scoring=MeanAbsoluteError(),
            cv=2,
        )
        search.fit(y_train, X_train, forecasting_horizon=3)

        assert len(search.cv_results_["params"]) == 1
        assert search.best_params_ == {"seasonality": 5}

    def test_randomized_search_n_iter_exceeds_space(self, y_X_factory):
        """Test RandomizedSearchCV when n_iter exceeds parameter space."""
        y, X = y_X_factory(length=100, seed=42)
        y_train = y[:80]
        X_train = X[:80] if X is not None else None

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The total space of parameters.*is smaller than n_iter",
                category=UserWarning,
            )
            search = RandomizedSearchCV(
                forecaster=SeasonalNaive(),
                param_distributions={"seasonality": [1, 5, 10]},
                n_iter=10,
                scoring=MeanAbsoluteError(),
                cv=2,
                random_state=42,
            )
            search.fit(y_train, X_train, forecasting_horizon=3)

        actual_n_iter = len(search.cv_results_["params"])
        assert actual_n_iter == 3, f"Expected 3 unique candidates (parameter space size), got {actual_n_iter}"

    def test_search_cv_with_no_X(self, y_X_factory):
        """Test search CV with X=None (forecast-only scenario)."""
        y, _ = y_X_factory(length=100, n_features=0, seed=42)
        y_train, y_test = y[:80], y[80:]

        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 7]},
            scoring=MeanAbsoluteError(),
            cv=2,
        )
        search.fit(y_train, X=None, forecasting_horizon=3)

        y_pred = search.predict(forecasting_horizon=3, X=None)
        assert len(y_pred) == 3

        search.observe(y_test[:5], X=None)
        search.rewind(y_test[:10], X=None)

    def test_search_cv_score_without_X(self, y_X_factory):
        """Test score() method with X=None."""
        y, _ = y_X_factory(length=100, n_features=0, seed=42)
        y_train, y_test = y[:80], y[80:]

        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 5]},
            scoring=MeanAbsoluteError(),
            cv=2,
        )
        search.fit(y_train, X=None, forecasting_horizon=3)

        score = search.score(y_test, X=None)
        assert isinstance(score, int | float)


class TestMultiMetric:
    """Tests for multi-metric search CV."""

    def test_multimetric_best_score_selection(self, y_X_factory):
        """Test that best_score_ corresponds to refit metric in multi-metric search."""
        y, X = y_X_factory(length=100, seed=42)
        y_train = y[:80]
        X_train = X[:80] if X is not None else None

        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 5, 10]},
            scoring={
                "mae": MeanAbsoluteError(),
                "rmse": RootMeanSquaredError(),
            },
            cv=2,
            refit="rmse",
        )
        search.fit(y_train, X_train, forecasting_horizon=3)

        expected_score = search.cv_results_["mean_test_rmse"][search.best_index_]
        assert abs(search.best_score_ - expected_score) < 1e-6

    def test_multimetric_score_returns_dict(self, y_X_factory):
        """Test that score() returns dict for multi-metric search."""
        y, X = y_X_factory(length=100, seed=42)
        y_train, y_test = y[:80], y[80:]
        X_train, X_test = (X[:80], X[80:]) if X is not None else (None, None)

        forecasting_horizon = 3
        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 5]},
            scoring={
                "mae": MeanAbsoluteError(),
                "rmse": RootMeanSquaredError(),
            },
            cv=2,
            refit="mae",
        )
        search.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)

        scores = search.score(
            y_test[:forecasting_horizon],
            X_test[:forecasting_horizon] if X_test is not None else None,
        )
        assert isinstance(scores, dict)
        assert "mae" in scores
        assert "rmse" in scores


class TestReturnTrainScore:
    """Tests for return_train_score parameter."""

    def test_return_train_score_adds_keys(self, y_X_factory):
        """Test that return_train_score=True adds train score keys."""
        y, X = y_X_factory(length=100, seed=42)
        y_train = y[:80]
        X_train = X[:80] if X is not None else None

        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 5]},
            scoring=MeanAbsoluteError(),
            cv=2,
            return_train_score=True,
        )
        search.fit(y_train, X_train, forecasting_horizon=3)

        assert "mean_train_score" in search.cv_results_
        assert "split0_train_score" in search.cv_results_
        assert "split1_train_score" in search.cv_results_


class TestErrorHandling:
    """Tests for error handling in search CV."""

    def test_error_score_nan_continues_fit(self, y_X_factory):
        """Test that error_score=np.nan continues fit even with potential errors."""
        y, X = y_X_factory(length=100, seed=42)
        y_train = y[:80]
        X_train = X[:80] if X is not None else None

        import numpy as np

        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 5, 10]},
            scoring=MeanAbsoluteError(),
            cv=2,
            error_score=np.nan,
        )
        search.fit(y_train, X_train, forecasting_horizon=3)
        assert hasattr(search, "cv_results_")


class TestScorerDirectionCorrectness:
    """Tests that search correctly selects the best model respecting scorer direction.

    Lower-is-better scorers (e.g. MAE) should select the candidate with the
    *lowest* raw error.  Scores in cv_results_ follow sklearn's sign
    convention: they are negated for lower-is-better metrics so that higher
    numeric values always indicate better performance.
    """

    def test_grid_search_selects_lowest_error(self, y_X_factory):
        """GridSearchCV with MAE must pick the candidate with lowest error, not highest."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)
        y_train = y[:80]

        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 5, 10]},
            scoring=MeanAbsoluteError(),
            cv=2,
        )
        search.fit(y_train, X=None, forecasting_horizon=3)

        # Compute actual MAE for each candidate to verify selection
        cv_results = search.cv_results_
        mean_scores = cv_results["mean_test_score"]

        # Scores should be negative (lower_is_better negation convention)
        assert np.all(mean_scores <= 0), (
            f"MAE scores in cv_results_ should be negative (sklearn sign convention), got {mean_scores}"
        )

        # best_index_ should point to the candidate with the highest
        # (least negative) score, i.e. lowest raw MAE
        assert search.best_index_ == np.argmax(mean_scores), (
            f"best_index_={search.best_index_} does not match argmax of mean_test_score={mean_scores}"
        )

        # best_score_ should be negative
        assert search.best_score_ <= 0, f"best_score_ should be negative for MAE, got {search.best_score_}"

    def test_best_score_is_negated_raw_mae(self, y_X_factory):
        """best_score_ should equal the negated raw MAE of the best candidate."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)
        y_train = y[:80]

        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 5]},
            scoring=MeanAbsoluteError(),
            cv=2,
        )
        search.fit(y_train, X=None, forecasting_horizon=3)

        # best_score_ is the mean cv score (negated MAE)
        assert search.best_score_ < 0
        # The actual MAE is the absolute value
        actual_mae = -search.best_score_
        assert actual_mae > 0

    def test_multimetric_respects_scorer_direction(self, y_X_factory):
        """Multi-metric search: each scorer should be negated per its lower_is_better tag."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)
        y_train = y[:80]

        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 5, 10]},
            scoring={
                "mae": MeanAbsoluteError(),
                "rmse": RootMeanSquaredError(),
            },
            cv=2,
            refit="mae",
        )
        search.fit(y_train, X=None, forecasting_horizon=3)

        cv_results = search.cv_results_

        # Both MAE and RMSE are lower_is_better → scores should be negative
        assert np.all(cv_results["mean_test_mae"] <= 0), (
            f"MAE scores should be negative, got {cv_results['mean_test_mae']}"
        )
        assert np.all(cv_results["mean_test_rmse"] <= 0), (
            f"RMSE scores should be negative, got {cv_results['mean_test_rmse']}"
        )

        # best_index_ (by MAE) should select the candidate with the highest
        # (least negative) MAE score → lowest raw MAE error
        mae_scores = cv_results["mean_test_mae"]
        assert search.best_index_ == np.argmax(mae_scores)

    def test_randomized_search_selects_lowest_error(self, y_X_factory):
        """RandomizedSearchCV with MAE must pick the candidate with lowest error."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)
        y_train = y[:80]

        search = RandomizedSearchCV(
            forecaster=SeasonalNaive(),
            param_distributions={"seasonality": [1, 5, 10]},
            scoring=MeanAbsoluteError(),
            cv=2,
            n_iter=3,
            random_state=42,
        )
        search.fit(y_train, X=None, forecasting_horizon=3)

        mean_scores = search.cv_results_["mean_test_score"]

        # Scores should be non-positive (negated MAE)
        assert np.all(mean_scores <= 0)

        # best_index_ should point to least negative score
        assert search.best_index_ == np.argmax(mean_scores)


class TestBestParamsConsistency:
    """Tests for best_params_, best_index_, and best_forecaster_ consistency."""

    def test_best_params_matches_best_index(self, y_X_factory):
        """best_params_ must correspond to the candidate at best_index_."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 3, 5, 10]},
            scoring=MeanAbsoluteError(),
            cv=2,
        )
        search.fit(y[:80], X=None, forecasting_horizon=3)

        # best_params_ should match the params at best_index_
        cv_params = search.cv_results_["params"][search.best_index_]
        assert search.best_params_ == cv_params

    def test_best_forecaster_uses_best_params(self, y_X_factory):
        """best_forecaster_ must be parameterized with best_params_."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 5, 10]},
            scoring=MeanAbsoluteError(),
            cv=2,
        )
        search.fit(y[:80], X=None, forecasting_horizon=3)

        best = search.best_forecaster_
        for param, value in search.best_params_.items():
            assert getattr(best, param) == value

    def test_best_score_matches_cv_results_at_best_index(self, y_X_factory):
        """best_score_ must equal mean_test_score at best_index_."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 5, 10]},
            scoring=MeanAbsoluteError(),
            cv=2,
        )
        search.fit(y[:80], X=None, forecasting_horizon=3)

        expected = search.cv_results_["mean_test_score"][search.best_index_]
        assert search.best_score_ == expected
