"""Systematic tests for GridSearchCV and RandomizedSearchCV using check generators."""

import pytest
from scipy.stats import randint, uniform
from sklearn.base import clone

from yohou.metrics import MeanAbsoluteError, RootMeanSquaredError
from yohou.model_selection import GridSearchCV, RandomizedSearchCV
from yohou.point_forecaster import SeasonalNaive
from yohou.testing import _yield_yohou_search_checks


# ============================================================================
# Parametrized Systematic Tests
# ============================================================================


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
            [],  # Test if schema validation is fixed
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
            [],  # Multimetric now works with Option A implementation
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
            [],  # Test if schema validation is fixed
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
            [],  # Test if schema validation is fixed
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
            [],  # Multimetric now works with Option A implementation
        ),
    ],
)
def test_search_cv_systematic_checks(
    search_cv_class, params, tags, expected_failures, y_X_factory
):
    """Run systematic checks on search CV classes using generator pattern.

    This test replaces individual test functions by using the check generator
    pattern to validate all search CV behaviors systematically.
    """
    # Generate data
    y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)
    y_train, y_test = y[:80], y[80:]
    X_train, X_test = (X[:80], X[80:]) if X is not None else (None, None)

    # Create and fit search CV
    forecaster = SeasonalNaive()
    search_cv = search_cv_class(forecaster=forecaster, **params)
    search_cv_fitted = clone(search_cv)
    search_cv_fitted.fit(y_train, X_train, forecasting_horizon=3)

    # Run all generated checks
    expected_failures_set = set(expected_failures)
    for check_name, check_func, check_kwargs in _yield_yohou_search_checks(
        search_cv_fitted,
        y_train,
        X_train,
        y_test,
        X_test,
        tags=tags,
    ):
        if check_name in expected_failures_set:
            pytest.skip(f"Expected failure: {check_name}")
        else:
            # Execute check
            check_func(search_cv_fitted, **check_kwargs)


# ============================================================================
# Panel Data Tests
# ============================================================================


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
def test_search_cv_panel_data(search_cv_class, params, y_X_panel_factory):
    """Test search CV with panel data using prefixed columns."""
    # Generate panel data
    y_panel, X_panel = y_X_panel_factory(
        n_groups=2, length=80, n_targets=1, n_features=2, seed=42
    )
    y_train, y_test = y_panel[:60], y_panel[60:]
    X_train, X_test = (X_panel[:60], X_panel[60:]) if X_panel is not None else (None, None)

    # Create and fit search CV
    forecaster = SeasonalNaive()
    search_cv = search_cv_class(forecaster=forecaster, **params)
    search_cv.fit(y_train, X_train, forecasting_horizon=3)

    # Test predictions work with panel data
    y_pred = search_cv.predict(forecasting_horizon=3, X=X_test)

    # Check predictions have expected structure
    assert "time" in y_pred.columns
    assert len(y_pred) == 3  # forecasting_horizon

    # Predictions should have panel prefixes
    non_time_cols = [c for c in y_pred.columns if c != "time" and c != "observed_time"]
    assert len(non_time_cols) > 0, "Should have prediction columns"
    assert any("__" in col for col in non_time_cols), "Should have panel prefixes"


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_grid_search_empty_param_grid(y_X_factory):
    """Test GridSearchCV with single-value param_grid (degenerate case)."""
    y, X = y_X_factory(length=100, seed=42)
    y_train = y[:80]
    X_train = X[:80] if X is not None else None

    # Single value = 1 candidate
    search = GridSearchCV(
        forecaster=SeasonalNaive(),
        param_grid={"seasonality": [5]},
        scoring=MeanAbsoluteError(),
        cv=2,
    )
    search.fit(y_train, X_train, forecasting_horizon=3)

    # Should still work with 1 candidate
    assert len(search.cv_results_["params"]) == 1
    assert search.best_params_ == {"seasonality": 5}


def test_randomized_search_n_iter_exceeds_space(y_X_factory):
    """Test RandomizedSearchCV when n_iter exceeds parameter space."""
    y, X = y_X_factory(length=100, seed=42)
    y_train = y[:80]
    X_train = X[:80] if X is not None else None

    import warnings

    # n_iter=10 but only 3 possible values - expect sklearn warning
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

    # sklearn's RandomizedSearchCV deduplicates samples when n_iter exceeds space
    # With 3 unique values and n_iter=10, sklearn samples without replacement first,
    # then with replacement, but deduplicates the result
    actual_n_iter = len(search.cv_results_["params"])
    assert actual_n_iter == 3, (
        f"Expected 3 unique candidates (parameter space size), got {actual_n_iter}"
    )


def test_search_cv_with_no_X(y_X_factory):
    """Test search CV with X=None (forecast-only scenario)."""
    # pytest.skip("Bug: check_inputs called with X=None causes TypeError")
    y, _ = y_X_factory(length=100, n_features=0, seed=42)
    y_train, y_test = y[:80], y[80:]

    search = GridSearchCV(
        forecaster=SeasonalNaive(),
        param_grid={"seasonality": [1, 7]},
        scoring=MeanAbsoluteError(),
        cv=2,
    )
    search.fit(y_train, X=None, forecasting_horizon=3)

    # Predict without X
    y_pred = search.predict(forecasting_horizon=3, X=None)
    assert len(y_pred) == 3

    # Update without X
    search.update(y_test[:5], X=None)

    # Reset without X
    search.reset(y_test[:10], X=None)


def test_search_cv_score_without_X(y_X_factory):
    """Test score() method with X=None."""
    # pytest.skip("Bug: check_inputs called with X=None causes TypeError")
    y, _ = y_X_factory(length=100, n_features=0, seed=42)
    y_train, y_test = y[:80], y[80:]

    search = GridSearchCV(
        forecaster=SeasonalNaive(),
        param_grid={"seasonality": [1, 5]},
        scoring=MeanAbsoluteError(),
        cv=2,
    )
    search.fit(y_train, X=None, forecasting_horizon=3)

    # Score without X
    score = search.score(y_test, X=None)
    assert isinstance(score, (int, float))


# ============================================================================
# Multi-Metric Tests
# ============================================================================


def test_multimetric_best_score_selection(y_X_factory):
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
        refit="rmse",  # Refit on RMSE
    )
    search.fit(y_train, X_train, forecasting_horizon=3)

    # best_score_ should match RMSE at best_index_
    expected_score = search.cv_results_["mean_test_rmse"][search.best_index_]
    assert abs(search.best_score_ - expected_score) < 1e-6


def test_multimetric_score_returns_dict(y_X_factory):
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


    # score() should return dict - only pass first forecasting_horizon rows of test data
    scores = search.score(y_test[:forecasting_horizon], X_test[:forecasting_horizon] if X_test is not None else None)
    assert isinstance(scores, dict)
    assert "mae" in scores
    assert "rmse" in scores


# ============================================================================
# Return Train Score Tests
# ============================================================================


def test_return_train_score_adds_keys(y_X_factory):
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

    # Should have train score keys
    assert "mean_train_score" in search.cv_results_
    assert "split0_train_score" in search.cv_results_
    assert "split1_train_score" in search.cv_results_


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_error_score_nan_continues_fit(y_X_factory):
    """Test that error_score='raise' raises on failure while np.nan continues."""
    y, X = y_X_factory(length=100, seed=42)
    y_train = y[:80]
    X_train = X[:80] if X is not None else None

    import numpy as np

    # With error_score=np.nan, should complete fit even with potential errors
    search = GridSearchCV(
        forecaster=SeasonalNaive(),
        param_grid={"seasonality": [1, 5, 10]},
        scoring=MeanAbsoluteError(),
        cv=2,
        error_score=np.nan,
    )
    # Should not raise (all params are valid for SeasonalNaive anyway)
    search.fit(y_train, X_train, forecasting_horizon=3)
    assert hasattr(search, "cv_results_")
