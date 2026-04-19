"""Systematic tests for GridSearchCV and RandomizedSearchCV using check generators."""

import numpy as np
import pytest
from scipy.stats import randint
from sklearn.base import clone

from conftest import run_checks
from yohou.interval import SplitConformalForecaster
from yohou.metrics import MeanAbsoluteError, RootMeanSquaredError
from yohou.metrics.interval import EmpiricalCoverage, IntervalScore
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
            # GridSearchCV with interval scorer (interval search)
            (
                GridSearchCV,
                {
                    "param_grid": {"point_forecaster__seasonality": [1, 5]},
                    "scoring": IntervalScore(coverage=[0.9]),
                    "cv": 2,
                    "refit": True,
                },
                {"search_type": "grid", "refit": True, "multimetric": False, "interval_scoring": True},
                ["check_search_observe_delegates", "check_search_rewind_delegates"],
            ),
            # GridSearchCV with list-of-dicts param_grid (multiple grid search spaces)
            (
                GridSearchCV,
                {
                    "param_grid": [{"seasonality": [2, 4]}, {"seasonality": [7]}],
                    "scoring": MeanAbsoluteError(),
                    "cv": 2,
                    "refit": True,
                },
                {"search_type": "grid", "refit": True, "multimetric": False},
                [],
            ),
        ],
    )
    @pytest.mark.slow
    def test_search_cv_systematic_checks(self, search_cv_class, params, tags, expected_failures, y_X_factory):
        """Run systematic checks on search CV classes using generator pattern."""
        y, X = y_X_factory(length=200, n_targets=1, n_features=2, seed=42)
        y_train, y_test = y[:180], y[180:]
        X_train, X_test = (X[:180], X[180:]) if X is not None else (None, None)

        if tags.get("interval_scoring", False):
            forecaster = SplitConformalForecaster(point_forecaster=SeasonalNaive(), calibration_size=20)
        else:
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


class TestSearchRefitFalse:
    """Tests for search CV behavior with refit=False."""

    def test_refit_false_no_best_forecaster(self, y_X_factory):
        """refit=False means best_forecaster_ is not available."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)
        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 5]},
            scoring=MeanAbsoluteError(),
            cv=2,
            refit=False,
        )
        search.fit(y[:80], X=None, forecasting_horizon=3)
        assert not hasattr(search, "best_forecaster_")

    def test_refit_false_cv_results_available(self, y_X_factory):
        """refit=False still populates cv_results_."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)
        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 5]},
            scoring=MeanAbsoluteError(),
            cv=2,
            refit=False,
        )
        search.fit(y[:80], X=None, forecasting_horizon=3)
        assert "mean_test_score" in search.cv_results_

    def test_refit_false_predict_raises(self, y_X_factory):
        """refit=False makes predict() raise AttributeError."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)
        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 5]},
            scoring=MeanAbsoluteError(),
            cv=2,
            refit=False,
        )
        search.fit(y[:80], X=None, forecasting_horizon=3)
        with pytest.raises(AttributeError):
            search.predict(forecasting_horizon=3)


class TestSearchPropertyValidation:
    """Tests for search CV properties that require fitted state."""

    def test_tags_before_fit(self):
        """__sklearn_tags__() works before fit (uses base forecaster)."""
        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 5]},
            scoring=MeanAbsoluteError(),
            cv=2,
        )
        tags = search.__sklearn_tags__()
        assert tags.forecaster_tags is not None

    def test_scoring_none_raises(self, y_X_factory):
        """scoring=None raises ValueError during fit."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)
        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 5]},
            scoring=None,
            cv=2,
        )
        with pytest.raises(ValueError, match="scoring"):
            search.fit(y[:80], X=None, forecasting_horizon=3)

    def test_callable_refit_selects_index(self, y_X_factory):
        """Callable refit selects custom best index."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)
        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 5, 10]},
            scoring=MeanAbsoluteError(),
            cv=2,
            refit=lambda results: 0,
        )
        search.fit(y[:80], X=None, forecasting_horizon=3)
        assert search.best_index_ == 0

    def test_invalid_multimetric_refit_raises(self, y_X_factory):
        """Invalid refit key for multimetric raises ValueError."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)
        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 5]},
            scoring={"mae": MeanAbsoluteError(), "rmse": RootMeanSquaredError()},
            cv=2,
            refit="nonexistent",
        )
        with pytest.raises(ValueError, match="refit"):
            search.fit(y[:80], X=None, forecasting_horizon=3)


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


class TestIntervalSearch:
    """Tests for interval forecaster tuning with interval metrics."""

    @pytest.mark.slow
    def test_grid_search_interval_scorer(self, y_X_factory):
        """GridSearchCV with interval scorer tunes SplitConformalForecaster."""
        y, X = y_X_factory(length=200, n_targets=1, n_features=0, seed=42)
        y_train = y[:180]

        search = GridSearchCV(
            forecaster=SplitConformalForecaster(
                point_forecaster=SeasonalNaive(),
                calibration_size=20,
            ),
            param_grid={"point_forecaster__seasonality": [1, 5, 10]},
            scoring=IntervalScore(coverage=[0.9]),
            cv=2,
        )
        search.fit(y_train, X=None, forecasting_horizon=3)

        assert hasattr(search, "best_params_")
        assert hasattr(search, "best_forecaster_")
        assert hasattr(search, "best_score_")

        # Predict interval from best forecaster
        y_pred = search.predict_interval(X=None, coverage_rates=[0.9])
        assert "time" in y_pred.columns
        interval_cols = [c for c in y_pred.columns if "_lower_" in c or "_upper_" in c]
        assert len(interval_cols) > 0, "Should have interval prediction columns"

    @pytest.mark.slow
    def test_grid_search_empirical_coverage_scorer(self, y_X_factory):
        """GridSearchCV with EmpiricalCoverage (higher_is_better) selects correctly."""
        y, _ = y_X_factory(length=200, n_targets=1, n_features=0, seed=42)
        y_train = y[:180]

        search = GridSearchCV(
            forecaster=SplitConformalForecaster(
                point_forecaster=SeasonalNaive(),
                calibration_size=20,
            ),
            param_grid={"point_forecaster__seasonality": [1, 5]},
            scoring=EmpiricalCoverage(coverage=[0.9]),
            cv=2,
        )
        search.fit(y_train, X=None, forecasting_horizon=3)

        # EmpiricalCoverage is NOT lower_is_better, scores should be positive
        mean_scores = search.cv_results_["mean_test_score"]
        assert np.all(mean_scores >= 0), f"EmpiricalCoverage scores should be >= 0, got {mean_scores}"

    @pytest.mark.slow
    def test_randomized_search_interval_scorer(self, y_X_factory):
        """RandomizedSearchCV works with interval scorers."""
        y, _ = y_X_factory(length=200, n_targets=1, n_features=0, seed=42)
        y_train = y[:180]

        search = RandomizedSearchCV(
            forecaster=SplitConformalForecaster(
                point_forecaster=SeasonalNaive(),
                calibration_size=20,
            ),
            param_distributions={"point_forecaster__seasonality": [1, 5, 10, 15]},
            n_iter=3,
            scoring=IntervalScore(coverage=[0.9]),
            cv=2,
            random_state=42,
        )
        search.fit(y_train, X=None, forecasting_horizon=3)

        assert hasattr(search, "best_params_")

    @pytest.mark.slow
    def test_interval_search_coverage_rates_propagated(self, y_X_factory):
        """Coverage rates from scorer propagated to forecaster.fit()."""
        y, _ = y_X_factory(length=200, n_targets=1, n_features=0, seed=42)
        y_train = y[:180]

        search = GridSearchCV(
            forecaster=SplitConformalForecaster(
                point_forecaster=SeasonalNaive(),
                calibration_size=20,
            ),
            param_grid={"point_forecaster__seasonality": [1, 5]},
            scoring=IntervalScore(coverage=[0.9, 0.95]),
            cv=2,
        )
        search.fit(y_train, X=None, forecasting_horizon=3)

        # Verify the best forecaster was refit with coverage_rates
        y_pred = search.predict_interval(X=None, coverage_rates=[0.9, 0.95])
        lower_cols = [c for c in y_pred.columns if "_lower_" in c]
        upper_cols = [c for c in y_pred.columns if "_upper_" in c]
        assert len(lower_cols) >= 2, "Should have lower bounds for both coverage rates"
        assert len(upper_cols) >= 2, "Should have upper bounds for both coverage rates"


class TestMixedMultimetric:
    """Tests for mixed point + interval multimetric search."""

    @pytest.mark.slow
    def test_mixed_multimetric_point_and_interval(self, y_X_factory):
        """Multimetric search with both point and interval scorers on a both-type forecaster."""
        y, _ = y_X_factory(length=200, n_targets=1, n_features=0, seed=42)
        y_train = y[:180]

        search = GridSearchCV(
            forecaster=SplitConformalForecaster(
                point_forecaster=SeasonalNaive(),
                calibration_size=20,
            ),
            param_grid={"point_forecaster__seasonality": [1, 5]},
            scoring={
                "mae": MeanAbsoluteError(),
                "interval_score": IntervalScore(coverage=[0.9]),
            },
            cv=2,
            refit="mae",
        )
        search.fit(y_train, X=None, forecasting_horizon=3)

        assert hasattr(search, "best_params_")
        assert "mean_test_mae" in search.cv_results_
        assert "mean_test_interval_score" in search.cv_results_

    @pytest.mark.slow
    def test_mixed_multimetric_refit_on_interval(self, y_X_factory):
        """Refit selects based on interval metric in mixed multimetric."""
        y, _ = y_X_factory(length=200, n_targets=1, n_features=0, seed=42)
        y_train = y[:180]

        search = GridSearchCV(
            forecaster=SplitConformalForecaster(
                point_forecaster=SeasonalNaive(),
                calibration_size=20,
            ),
            param_grid={"point_forecaster__seasonality": [1, 5]},
            scoring={
                "mae": MeanAbsoluteError(),
                "interval_score": IntervalScore(coverage=[0.9]),
            },
            cv=2,
            refit="interval_score",
        )
        search.fit(y_train, X=None, forecasting_horizon=3)

        expected_score = search.cv_results_["mean_test_interval_score"][search.best_index_]
        assert abs(search.best_score_ - expected_score) < 1e-6


class TestIncompatibleForecasterScorer:
    """Tests for validation of forecaster/scorer type compatibility."""

    def test_point_forecaster_with_interval_scorer_raises(self, y_X_factory):
        """Point-only forecaster with interval scorer raises ValueError."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)
        y_train = y[:80]

        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 5]},
            scoring=IntervalScore(coverage=[0.9]),
            cv=2,
        )
        with pytest.raises(ValueError, match="does not support predict_interval"):
            search.fit(y_train, X=None, forecasting_horizon=3)

    def test_point_forecaster_with_multimetric_interval_raises(self, y_X_factory):
        """Point-only forecaster with multimetric containing interval scorer raises."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)
        y_train = y[:80]

        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 5]},
            scoring={
                "mae": MeanAbsoluteError(),
                "interval_score": IntervalScore(coverage=[0.9]),
            },
            cv=2,
            refit="mae",
        )
        with pytest.raises(ValueError, match="does not support predict_interval"):
            search.fit(y_train, X=None, forecasting_horizon=3)

    def test_both_type_forecaster_with_point_scorer_ok(self, y_X_factory):
        """Both-type forecaster with point-only scorer should work."""
        y, _ = y_X_factory(length=200, n_targets=1, n_features=0, seed=42)
        y_train = y[:180]

        search = GridSearchCV(
            forecaster=SplitConformalForecaster(
                point_forecaster=SeasonalNaive(),
                calibration_size=20,
            ),
            param_grid={"point_forecaster__seasonality": [1, 5]},
            scoring=MeanAbsoluteError(),
            cv=2,
        )
        search.fit(y_train, X=None, forecasting_horizon=3)
        assert hasattr(search, "best_params_")


class TestSearchRefitCallable:
    """Tests for refit as callable."""

    def test_refit_callable_selects_best(self, y_X_factory):
        """Callable refit selects the best parameter set."""
        from yohou.metrics import MeanAbsoluteError
        from yohou.model_selection import GridSearchCV
        from yohou.point import SeasonalNaive

        y, _ = y_X_factory(length=60, n_targets=1, n_features=0)
        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 3, 7]},
            scoring=MeanAbsoluteError(),
            cv=2,
            refit=lambda cv_results: 0,
        )
        search.fit(y[:50], forecasting_horizon=3)
        assert hasattr(search, "best_forecaster_")
        y_pred = search.predict()
        assert len(y_pred) == 3


class TestSearchMultiMetricDict:
    """Tests for multi-metric scoring with dict."""

    def test_multi_metric_scoring_dict(self, y_X_factory):
        """Dict scoring with multiple metrics stores all results."""
        from yohou.metrics import MeanAbsoluteError, RootMeanSquaredError
        from yohou.model_selection import GridSearchCV
        from yohou.point import SeasonalNaive

        y, _ = y_X_factory(length=60, n_targets=1, n_features=0)
        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 3]},
            scoring={"mae": MeanAbsoluteError(), "rmse": RootMeanSquaredError()},
            cv=2,
            refit="mae",
        )
        search.fit(y[:50], forecasting_horizon=3)
        assert hasattr(search, "cv_results_")
        assert "mean_test_mae" in search.cv_results_
        assert "mean_test_rmse" in search.cv_results_


class TestSearchDelegatedProperties:
    """Tests for delegated properties from best_forecaster_."""

    def test_interval_property_delegation(self, y_X_factory):
        """interval_ property is delegated from best_forecaster_."""
        from yohou.metrics import MeanAbsoluteError
        from yohou.model_selection import GridSearchCV
        from yohou.point import SeasonalNaive

        y, _ = y_X_factory(length=60, n_targets=1, n_features=0)
        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 3]},
            scoring=MeanAbsoluteError(),
            cv=2,
        )
        search.fit(y[:50], forecasting_horizon=3)
        interval = search.interval_
        assert interval is not None

    def test_n_features_in_property_delegation(self, y_X_factory):
        """n_features_in_ property is delegated from best_forecaster_."""
        y, X = y_X_factory(length=60, n_targets=1, n_features=2)
        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 3]},
            scoring=MeanAbsoluteError(),
            cv=2,
        )
        search.fit(y[:50], X[:50], forecasting_horizon=3)
        assert search.n_features_in_ == search.best_forecaster_.n_features_in_

    def test_feature_names_in_property_delegation(self, y_X_factory):
        """feature_names_in_ property is delegated from best_forecaster_."""
        y, X = y_X_factory(length=60, n_targets=1, n_features=2)
        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 3]},
            scoring=MeanAbsoluteError(),
            cv=2,
        )
        search.fit(y[:50], X[:50], forecasting_horizon=3)
        assert search.feature_names_in_ == search.best_forecaster_.feature_names_in_


class TestSearchSelectBestIndex:
    """Tests for _select_best_index callable error paths."""

    def test_callable_refit_non_integer_raises_type_error(self, y_X_factory):
        """Callable refit returning non-integer raises TypeError."""
        y, _ = y_X_factory(length=60, n_targets=1, n_features=0)
        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 3]},
            scoring=MeanAbsoluteError(),
            cv=2,
            refit=lambda cv_results: "not_an_int",
        )
        with pytest.raises(TypeError, match="best_index_ returned is not an integer"):
            search.fit(y[:50], forecasting_horizon=3)

    def test_callable_refit_out_of_range_raises_index_error(self, y_X_factory):
        """Callable refit returning out-of-range index raises IndexError."""
        y, _ = y_X_factory(length=60, n_targets=1, n_features=0)
        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 3]},
            scoring=MeanAbsoluteError(),
            cv=2,
            refit=lambda cv_results: 9999,
        )
        with pytest.raises(IndexError, match="best_index_ index out of range"):
            search.fit(y[:50], forecasting_horizon=3)


class TestSearchGetScorersValidation:
    """Tests for _get_scorers validation branches."""

    def test_scoring_none_raises(self):
        """Scoring=None raises ValueError."""
        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1]},
            scoring=None,
            cv=2,
        )
        with pytest.raises(ValueError, match="scoring parameter cannot be None"):
            search._get_scorers()

    def test_scoring_invalid_type_raises(self):
        """Scoring with invalid type raises ValueError."""
        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1]},
            scoring="not_a_scorer",
            cv=2,
        )
        with pytest.raises(ValueError, match="scoring must be an instance of BaseScorer"):
            search._get_scorers()

    def test_check_refit_for_multimetric_invalid_refit(self):
        """Invalid refit value with multimetric scoring raises ValueError."""
        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1]},
            scoring={"mae": MeanAbsoluteError(), "rmse": RootMeanSquaredError()},
            cv=2,
            refit=True,
        )
        with pytest.raises(ValueError, match="For multi-metric scoring"):
            search._check_refit_for_multimetric({"mae": 1.0, "rmse": 2.0})


class TestSystematicChecksPanel:
    """Systematic search CV checks with panel data, covering panel prediction paths."""

    @pytest.mark.slow
    def test_grid_search_panel_systematic_checks(self, y_X_panel_factory):
        """Run systematic checks on GridSearchCV fitted on panel data."""
        y_panel, _ = y_X_panel_factory(n_groups=2, length=80, n_targets=1, n_features=0, seed=42)
        y_train = y_panel[:60]
        y_test = y_panel[60:]
        search_cv = clone(
            GridSearchCV(
                forecaster=SeasonalNaive(),
                param_grid={"seasonality": [1, 5]},
                scoring=MeanAbsoluteError(),
                cv=2,
                refit=True,
            )
        )
        search_cv.fit(y_train, forecasting_horizon=3)
        run_checks(
            search_cv,
            _yield_yohou_search_checks(search_cv, y_train, None, y_test, None),
        )


class TestClassProbaSearch:
    """Tests for search CV with class-probability forecasters."""

    def test_grid_search_class_proba_predict(self, class_proba_y_X_factory):
        """GridSearchCV with class_proba forecaster exposes predict_class_proba."""
        from sklearn.tree import DecisionTreeClassifier

        from yohou.class_proba import ClassProbaReductionForecaster
        from yohou.metrics import LogLoss

        y, X = class_proba_y_X_factory(length=100, n_targets=1, n_features=0, n_classes=3, seed=42)
        y_train, y_test = y[:80], y[80:]

        search_cv = GridSearchCV(
            forecaster=ClassProbaReductionForecaster(
                estimator=DecisionTreeClassifier(random_state=42),
            ),
            param_grid={"estimator__max_depth": [2, 5]},
            scoring=LogLoss(),
            cv=2,
            refit=True,
        )
        search_cv.fit(y_train, forecasting_horizon=3)

        y_pred = search_cv.predict_class_proba(forecasting_horizon=3)
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) == 3
        assert len(y_pred) == 3

        y_pred2 = search_cv.observe_predict_class_proba(y=y_test[:3])
        proba_cols2 = [c for c in y_pred2.columns if "_proba_" in c]
        assert len(proba_cols2) == 3


class TestSearchForecasterHas:
    """Tests for the _search_forecaster_has helper."""

    def test_returns_false_when_forecaster_type_is_none(self):
        """Check returns False for point/interval/class_proba when type is None."""
        from yohou.model_selection.search import _search_forecaster_has

        class _NoneTypeForecaster(SeasonalNaive):
            def __sklearn_tags__(self):
                tags = super().__sklearn_tags__()
                tags.forecaster_tags.forecaster_type = None
                return tags

        check_fn = _search_forecaster_has("point")
        mock_search = type(
            "MockSearch",
            (),
            {
                "refit": True,
                "best_forecaster_": _NoneTypeForecaster(),
            },
        )()
        assert check_fn(mock_search) is False
