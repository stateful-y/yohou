"""Check functions for yohou search CV classes (GridSearchCV, RandomizedSearchCV).

This module provides validation functions for hyperparameter search classes
that implement cross-validated parameter optimization for forecasters.
"""

try:
    import polars as pl
    from polars.testing import assert_frame_equal
except ImportError as e:
    raise ImportError("polars.testing is required for yohou.testing module. Install with: uv sync --group tests") from e

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from yohou.model_selection import GridSearchCV, RandomizedSearchCV
from yohou.utils import inspect_panel

__all__ = [
    "check_grid_search_exhaustive",
    "check_grid_search_param_grid_validation",
    "check_randomized_search_distributions",
    "check_randomized_search_n_iter",
    "check_randomized_search_reproducibility",
    "check_search_clone_preserves_params",
    "check_search_cv_results_structure",
    "check_search_error_score_handling",
    "check_search_fit_sets_attributes",
    "check_search_interval_predict_delegates",
    "check_search_method_availability",
    "check_search_multimetric_scoring",
    "check_search_not_fitted_error",
    "check_search_panel_data",
    "check_search_predict_delegates",
    "check_search_refit_false_no_forecaster",
    "check_search_rewind_delegates",
    "check_search_return_train_score",
    "check_search_observe_delegates",
]


def check_search_fit_sets_attributes(
    search_cv, y: pl.DataFrame, X: pl.DataFrame | None = None, forecasting_horizon: int = 3
) -> None:
    """Check fit() sets required search CV attributes.

    Validates that fit() creates all required attributes including cv_results_,
    best_forecaster_, best_params_, best_score_, best_index_, scorer_, n_splits_,
    and multimetric_.

    Parameters
    ----------
    search_cv : BaseSearchCV
        Unfitted search CV instance
    y : pl.DataFrame
        Training target data with "time" column
    X : pl.DataFrame, optional
        Training features with "time" column
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast

    Raises
    ------
    AssertionError
        If required attributes are not set after fit()

    """
    search_cv_clone = clone(search_cv)
    search_cv_clone.fit(y, X, forecasting_horizon=forecasting_horizon)

    # Check core fitted attributes
    assert hasattr(search_cv_clone, "cv_results_"), "fit() must set cv_results_ attribute"
    assert isinstance(search_cv_clone.cv_results_, dict), (
        f"cv_results_ should be dict, got {type(search_cv_clone.cv_results_)}"
    )

    assert hasattr(search_cv_clone, "best_params_"), "fit() must set best_params_ attribute"
    assert isinstance(search_cv_clone.best_params_, dict), (
        f"best_params_ should be dict, got {type(search_cv_clone.best_params_)}"
    )

    assert hasattr(search_cv_clone, "best_score_"), "fit() must set best_score_ attribute"
    assert isinstance(search_cv_clone.best_score_, int | float | np.number), (
        f"best_score_ should be numeric, got {type(search_cv_clone.best_score_)}"
    )

    assert hasattr(search_cv_clone, "best_index_"), "fit() must set best_index_ attribute"
    assert isinstance(search_cv_clone.best_index_, int | np.integer), (
        f"best_index_ should be int, got {type(search_cv_clone.best_index_)}"
    )

    assert hasattr(search_cv_clone, "scorer_"), "fit() must set scorer_ attribute"

    assert hasattr(search_cv_clone, "n_splits_"), "fit() must set n_splits_ attribute"
    assert isinstance(search_cv_clone.n_splits_, int | np.integer), (
        f"n_splits_ should be int, got {type(search_cv_clone.n_splits_)}"
    )

    assert hasattr(search_cv_clone, "multimetric_"), "fit() must set multimetric_ attribute"
    assert isinstance(search_cv_clone.multimetric_, bool), (
        f"multimetric_ should be bool, got {type(search_cv_clone.multimetric_)}"
    )

    # Check best_forecaster_ when refit=True
    if search_cv_clone.refit:
        assert hasattr(search_cv_clone, "best_forecaster_"), "fit() must set best_forecaster_ when refit=True"
        assert hasattr(search_cv_clone, "refit_time_"), "fit() must set refit_time_ when refit=True"


def check_search_not_fitted_error(search_cv, y: pl.DataFrame, X: pl.DataFrame | None = None) -> None:
    """Check accessing fitted attributes before fit() raises NotFittedError.

    Parameters
    ----------
    search_cv : BaseSearchCV
        Unfitted search CV instance
    y : pl.DataFrame
        Test target data
    X : pl.DataFrame, optional
        Test features

    Raises
    ------
    AssertionError
        If NotFittedError is not raised when accessing fitted attributes

    """
    search_cv_clone = clone(search_cv)

    # Should raise NotFittedError when checking if fitted
    try:
        check_is_fitted(search_cv_clone, "best_forecaster_")
        raise AssertionError(
            f"{search_cv_clone.__class__.__name__} should raise NotFittedError "
            f"when accessing best_forecaster_ before fit()"
        )
    except NotFittedError:
        # Expected behavior
        pass

    # Should raise NotFittedError when calling predict before fit
    try:
        search_cv_clone.predict(forecasting_horizon=1, X=X)
        raise AssertionError(
            f"{search_cv_clone.__class__.__name__} should raise NotFittedError when calling predict() before fit()"
        )
    except (NotFittedError, AttributeError):
        # Expected behavior (AttributeError if refit=False)
        pass


def check_search_cv_results_structure(
    search_cv, y: pl.DataFrame, X: pl.DataFrame | None = None, forecasting_horizon: int = 3
) -> None:
    """Check cv_results_ has required structure.

    Validates that cv_results_ contains params, mean_test_score, rank_test_score,
    and split{n}_test_score keys with correct lengths.

    Parameters
    ----------
    search_cv : BaseSearchCV
        Fitted search CV instance
    y : pl.DataFrame
        Training target data
    X : pl.DataFrame, optional
        Training features
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast

    Raises
    ------
    AssertionError
        If cv_results_ structure is invalid

    """
    search_cv_clone = clone(search_cv)
    search_cv_clone.fit(y, X, forecasting_horizon=forecasting_horizon)

    cv_results = search_cv_clone.cv_results_

    # Check required keys
    assert "params" in cv_results, "cv_results_ must have 'params' key"
    assert isinstance(cv_results["params"], list), (
        f"cv_results_['params'] should be list, got {type(cv_results['params'])}"
    )

    n_candidates = len(cv_results["params"])
    assert n_candidates > 0, "cv_results_['params'] should not be empty"

    # Check score keys (single metric)
    if not search_cv_clone.multimetric_:
        assert "mean_test_score" in cv_results, "cv_results_ must have 'mean_test_score' key"
        assert "rank_test_score" in cv_results, "cv_results_ must have 'rank_test_score' key"

        assert len(cv_results["mean_test_score"]) == n_candidates, (
            f"mean_test_score length {len(cv_results['mean_test_score'])} should match params length {n_candidates}"
        )
        assert len(cv_results["rank_test_score"]) == n_candidates, (
            f"rank_test_score length {len(cv_results['rank_test_score'])} should match params length {n_candidates}"
        )

    # Check split score keys
    n_splits = search_cv_clone.n_splits_
    if not search_cv_clone.multimetric_:
        # Single metric: split keys are split{N}_test_score
        for split_idx in range(n_splits):
            split_key = f"split{split_idx}_test_score"
            assert split_key in cv_results, f"cv_results_ must have '{split_key}' key"
            assert len(cv_results[split_key]) == n_candidates, (
                f"{split_key} length {len(cv_results[split_key])} should match params length {n_candidates}"
            )
    else:
        # Multimetric: split keys are split{N}_test_{scorer_name}
        scorer_names = list(search_cv_clone.scorer_.keys()) if hasattr(search_cv_clone.scorer_, "keys") else []
        for split_idx in range(n_splits):
            for scorer_name in scorer_names:
                split_key = f"split{split_idx}_test_{scorer_name}"
                assert split_key in cv_results, f"cv_results_ must have '{split_key}' key"
                assert len(cv_results[split_key]) == n_candidates, (
                    f"{split_key} length {len(cv_results[split_key])} should match params length {n_candidates}"
                )


def check_search_refit_false_no_forecaster(
    search_cv, y: pl.DataFrame, X: pl.DataFrame | None = None, forecasting_horizon: int = 3
) -> None:
    """Check refit=False doesn't create best_forecaster_.

    Parameters
    ----------
    search_cv : BaseSearchCV
        Unfitted search CV instance with refit=False
    y : pl.DataFrame
        Training target data
    X : pl.DataFrame, optional
        Training features
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast

    Raises
    ------
    AssertionError
        If best_forecaster_ is created when refit=False

    """
    search_cv_clone = clone(search_cv)

    # Temporarily set refit=False if not already
    original_refit = search_cv_clone.refit
    search_cv_clone.refit = False

    search_cv_clone.fit(y, X, forecasting_horizon=forecasting_horizon)

    # Should have results but no best_forecaster_
    assert hasattr(search_cv_clone, "best_params_"), "fit() should set best_params_ even when refit=False"
    assert hasattr(search_cv_clone, "best_score_"), "fit() should set best_score_ even when refit=False"
    assert not hasattr(search_cv_clone, "best_forecaster_"), "fit() should NOT set best_forecaster_ when refit=False"
    assert not hasattr(search_cv_clone, "refit_time_"), "fit() should NOT set refit_time_ when refit=False"

    # predict() should not be available
    try:
        search_cv_clone.predict(forecasting_horizon=1, X=X)
        raise AssertionError(f"{search_cv_clone.__class__.__name__} should not have predict() when refit=False")
    except AttributeError:
        # Expected behavior
        pass

    # Restore original refit value
    search_cv_clone.refit = original_refit


def check_search_predict_delegates(
    search_cv,
    y_train: pl.DataFrame,
    y_test: pl.DataFrame,
    X_train: pl.DataFrame | None = None,
    X_test: pl.DataFrame | None = None,
) -> None:
    """Check predict() delegates to best_forecaster_.predict() correctly.

    Parameters
    ----------
    search_cv : BaseSearchCV
        Fitted search CV instance
    y_train : pl.DataFrame
        Training target data
    y_test : pl.DataFrame
        Test target data
    X_train : pl.DataFrame, optional
        Training features
    X_test : pl.DataFrame, optional
        Test features

    Raises
    ------
    AssertionError
        If predict() doesn't delegate correctly

    """
    forecasting_horizon = min(3, len(y_test))

    # Make predictions from search CV
    y_pred_search = search_cv.predict(forecasting_horizon=forecasting_horizon, X=X_test)

    # Make predictions from best_forecaster_ directly
    y_pred_best = search_cv.best_forecaster_.predict(forecasting_horizon=forecasting_horizon, X=X_test)

    # Predictions should be identical
    assert_frame_equal(y_pred_search, y_pred_best, check_exact=False)


def check_search_observe_delegates(
    search_cv,
    y_train: pl.DataFrame,
    y_update: pl.DataFrame,
    X_train: pl.DataFrame | None = None,
    X_update: pl.DataFrame | None = None,
) -> None:
    """Check observe() delegates to best_forecaster_.observe() correctly.

    Parameters
    ----------
    search_cv : BaseSearchCV
        Fitted search CV instance
    y_train : pl.DataFrame
        Training target data
    y_update : pl.DataFrame
        Update target data
    X_train : pl.DataFrame, optional
        Training features
    X_update : pl.DataFrame, optional
        Update features

    Raises
    ------
    AssertionError
        If observe() doesn't delegate correctly

    """
    # Get initial observed_time
    initial_observed_time = search_cv.best_forecaster_.observed_time_

    # Observe via search CV
    search_cv.observe(y_update, X_update)

    # Check that best_forecaster_ was observed
    updated_observed_time = search_cv.best_forecaster_.observed_time_

    # observed_time should have changed
    if isinstance(initial_observed_time, dict):
        # Panel data case
        for group_name in initial_observed_time:
            assert updated_observed_time[group_name] > initial_observed_time[group_name], (
                f"observed_time for group {group_name} should increase after observe, "
                f"got {initial_observed_time[group_name]} -> {updated_observed_time[group_name]}"
            )
    else:
        # Non-panel case
        assert updated_observed_time > initial_observed_time, (
            f"observed_time should increase after observe, got {initial_observed_time} -> {updated_observed_time}"
        )


def check_search_rewind_delegates(
    search_cv,
    y_train: pl.DataFrame,
    y_reset: pl.DataFrame,
    X_train: pl.DataFrame | None = None,
    X_reset: pl.DataFrame | None = None,
) -> None:
    """Check rewind() delegates to best_forecaster_.rewind() correctly.

    Parameters
    ----------
    search_cv : BaseSearchCV
        Fitted search CV instance
    y_train : pl.DataFrame
        Training target data
    y_reset : pl.DataFrame
        Reset target data
    X_train : pl.DataFrame, optional
        Training features
    X_reset : pl.DataFrame, optional
        Reset features

    Raises
    ------
    AssertionError
        If rewind() doesn't delegate correctly

    """
    # Get initial observed_time
    initial_observed_time = search_cv.best_forecaster_.observed_time_

    # Rewind via search CV
    search_cv.rewind(y_reset, X_reset)

    # Check that best_forecaster_ was rewound
    reset_observed_time = search_cv.best_forecaster_.observed_time_

    # observed_time should have changed
    if isinstance(initial_observed_time, dict):
        # Panel data case
        for group_name in initial_observed_time:
            assert reset_observed_time[group_name] != initial_observed_time[group_name], (
                f"observed_time for group {group_name} should change after rewind"
            )
    else:
        # Non-panel case
        assert reset_observed_time != initial_observed_time, "observed_time should change after rewind"


def check_search_multimetric_scoring(
    search_cv, y: pl.DataFrame, X: pl.DataFrame | None = None, forecasting_horizon: int = 3
) -> None:
    """Check multi-metric scoring with dict scorer works correctly.

    Parameters
    ----------
    search_cv : BaseSearchCV
        Unfitted search CV instance with dict scoring
    y : pl.DataFrame
        Training target data
    X : pl.DataFrame, optional
        Training features
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast

    Raises
    ------
    AssertionError
        If multi-metric scoring doesn't work correctly

    """
    search_cv_clone = clone(search_cv)

    # Check that scoring is a dict
    if not isinstance(search_cv_clone.scoring, dict):
        raise ValueError("This check requires search_cv.scoring to be a dict")

    search_cv_clone.fit(y, X, forecasting_horizon=forecasting_horizon)

    # Check multimetric_ flag
    assert search_cv_clone.multimetric_, "multimetric_ should be True when scoring is dict"

    # Check that all metrics are in cv_results_
    for metric_name in search_cv_clone.scoring:
        mean_key = f"mean_test_{metric_name}"
        rank_key = f"rank_test_{metric_name}"

        assert mean_key in search_cv_clone.cv_results_, (
            f"cv_results_ must have '{mean_key}' key for multi-metric scoring"
        )
        assert rank_key in search_cv_clone.cv_results_, (
            f"cv_results_ must have '{rank_key}' key for multi-metric scoring"
        )

    # Check scorer_ is dict
    assert isinstance(search_cv_clone.scorer_, dict), (
        f"scorer_ should be dict for multi-metric, got {type(search_cv_clone.scorer_)}"
    )

    # Check that best_score_ matches the refit metric
    if isinstance(search_cv_clone.refit, str):
        refit_metric = search_cv_clone.refit
        expected_best_score = search_cv_clone.cv_results_[f"mean_test_{refit_metric}"][search_cv_clone.best_index_]
        assert np.isclose(search_cv_clone.best_score_, expected_best_score), (
            f"best_score_ should match mean_test_{refit_metric} at best_index_, "
            f"got {search_cv_clone.best_score_} vs {expected_best_score}"
        )


def check_search_return_train_score(
    search_cv, y: pl.DataFrame, X: pl.DataFrame | None = None, forecasting_horizon: int = 3
) -> None:
    """Check return_train_score=True adds train score keys to cv_results_.

    Parameters
    ----------
    search_cv : BaseSearchCV
        Unfitted search CV instance with return_train_score=True
    y : pl.DataFrame
        Training target data
    X : pl.DataFrame, optional
        Training features
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast

    Raises
    ------
    AssertionError
        If train scores are not in cv_results_

    """
    search_cv_clone = clone(search_cv)

    # Temporarily set return_train_score=True
    original_return_train_score = search_cv_clone.return_train_score
    search_cv_clone.return_train_score = True

    search_cv_clone.fit(y, X, forecasting_horizon=forecasting_horizon)

    cv_results = search_cv_clone.cv_results_

    # Check for train score keys
    if not search_cv_clone.multimetric_:
        assert "mean_train_score" in cv_results, "cv_results_ must have 'mean_train_score' when return_train_score=True"

        # Check split train scores
        n_splits = search_cv_clone.n_splits_
        for split_idx in range(n_splits):
            split_key = f"split{split_idx}_train_score"
            assert split_key in cv_results, f"cv_results_ must have '{split_key}' when return_train_score=True"

    # Restore original value
    search_cv_clone.return_train_score = original_return_train_score


def check_search_error_score_handling(
    search_cv, y: pl.DataFrame, X: pl.DataFrame | None = None, forecasting_horizon: int = 3
) -> None:
    """Check error_score parameter handles failing fits correctly.

    Parameters
    ----------
    search_cv : BaseSearchCV
        Unfitted search CV instance with error_score=np.nan
    y : pl.DataFrame
        Training target data
    X : pl.DataFrame, optional
        Training features
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast

    Raises
    ------
    AssertionError
        If error scores are not handled correctly

    """
    search_cv_clone = clone(search_cv)

    # Set error_score to np.nan (don't raise)
    search_cv_clone.error_score = np.nan

    # Note: This check assumes that invalid parameters will be tested
    # If all parameters are valid, this check may pass trivially
    search_cv_clone.fit(y, X, forecasting_horizon=forecasting_horizon)

    # Check that fit completed without raising
    assert hasattr(search_cv_clone, "cv_results_"), (
        "fit() should complete with error_score=np.nan even with some failing fits"
    )


def check_search_clone_preserves_params(search_cv) -> None:
    """Check sklearn clone() preserves search CV parameters.

    Parameters
    ----------
    search_cv : BaseSearchCV
        Search CV instance (fitted or unfitted)

    Raises
    ------
    AssertionError
        If clone doesn't preserve parameters

    """
    search_cv_cloned = clone(search_cv)

    # Get parameters from original and clone
    params_original = search_cv.get_params()
    params_cloned = search_cv_cloned.get_params()

    # Check that parameter names match
    assert params_original.keys() == params_cloned.keys(), (
        f"Cloned search CV should have same parameter names, "
        f"got {set(params_original.keys())} vs {set(params_cloned.keys())}"
    )

    # Check that forecaster type matches
    assert type(params_original["forecaster"]) is type(params_cloned["forecaster"]), (
        f"Cloned forecaster type should match, "
        f"got {type(params_original['forecaster'])} vs {type(params_cloned['forecaster'])}"
    )

    # Check that scorer type matches
    if params_original["scoring"] is not None:
        if isinstance(params_original["scoring"], dict):
            assert isinstance(params_cloned["scoring"], dict), "Cloned scoring should be dict when original is dict"
        else:
            assert type(params_original["scoring"]) is type(params_cloned["scoring"]), (
                f"Cloned scoring type should match, "
                f"got {type(params_original['scoring'])} vs {type(params_cloned['scoring'])}"
            )

    # Check that fitted state is NOT cloned
    if hasattr(search_cv, "cv_results_"):
        assert not hasattr(search_cv_cloned, "cv_results_"), "Cloned search CV should not have fitted attributes"


def check_grid_search_exhaustive(
    search_cv, y: pl.DataFrame, X: pl.DataFrame | None = None, forecasting_horizon: int = 3
) -> None:
    """Check GridSearchCV evaluates all parameter combinations.

    Parameters
    ----------
    search_cv : GridSearchCV
        Unfitted GridSearchCV instance
    y : pl.DataFrame
        Training target data
    X : pl.DataFrame, optional
        Training features
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast

    Raises
    ------
    AssertionError
        If not all parameter combinations are evaluated

    """

    if not isinstance(search_cv, GridSearchCV):
        raise ValueError("This check requires GridSearchCV instance")

    search_cv_clone = clone(search_cv)
    search_cv_clone.fit(y, X, forecasting_horizon=forecasting_horizon)

    # Count expected combinations
    param_grid = search_cv_clone.param_grid
    if isinstance(param_grid, dict):
        # Single grid
        expected_combinations = 1
        for param_values in param_grid.values():
            expected_combinations *= len(param_values)
    elif isinstance(param_grid, list):
        # Multiple grids
        expected_combinations = 0
        for grid in param_grid:
            grid_combinations = 1
            for param_values in grid.values():
                grid_combinations *= len(param_values)
            expected_combinations += grid_combinations
    else:
        raise ValueError(f"Invalid param_grid type: {type(param_grid)}")

    # Check actual combinations
    actual_combinations = len(search_cv_clone.cv_results_["params"])
    assert actual_combinations == expected_combinations, (
        f"GridSearchCV should evaluate {expected_combinations} combinations, got {actual_combinations}"
    )


def check_grid_search_param_grid_validation(search_cv) -> None:
    """Check param_grid format is validated (dict or list of dicts).

    Parameters
    ----------
    search_cv : GridSearchCV
        GridSearchCV instance

    Raises
    ------
    AssertionError
        If param_grid format is not validated correctly

    """

    if not isinstance(search_cv, GridSearchCV):
        raise ValueError("This check requires GridSearchCV instance")

    param_grid = search_cv.param_grid

    # Check param_grid is dict or list of dicts
    if isinstance(param_grid, dict):
        # Valid: single dict
        for key, value in param_grid.items():
            assert isinstance(key, str), f"param_grid keys should be str, got {type(key)}"
            assert isinstance(value, list | tuple), f"param_grid values should be list or tuple, got {type(value)}"
    elif isinstance(param_grid, list):
        # Valid: list of dicts
        for grid in param_grid:
            assert isinstance(grid, dict), f"param_grid list elements should be dict, got {type(grid)}"
            for key, value in grid.items():
                assert isinstance(key, str), f"param_grid keys should be str, got {type(key)}"
                assert isinstance(value, list | tuple), f"param_grid values should be list or tuple, got {type(value)}"
    else:
        raise AssertionError(f"param_grid should be dict or list of dicts, got {type(param_grid)}")


def check_randomized_search_n_iter(
    search_cv, y: pl.DataFrame, X: pl.DataFrame | None = None, forecasting_horizon: int = 3
) -> None:
    """Check n_iter controls number of parameter combinations evaluated.

    Parameters
    ----------
    search_cv : RandomizedSearchCV
        Unfitted RandomizedSearchCV instance
    y : pl.DataFrame
        Training target data
    X : pl.DataFrame, optional
        Training features
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast

    Raises
    ------
    AssertionError
        If number of combinations doesn't match n_iter

    """

    if not isinstance(search_cv, RandomizedSearchCV):
        raise ValueError("This check requires RandomizedSearchCV instance")

    search_cv_clone = clone(search_cv)
    search_cv_clone.fit(y, X, forecasting_horizon=forecasting_horizon)

    # Check actual combinations matches n_iter
    actual_combinations = len(search_cv_clone.cv_results_["params"])
    expected_combinations = search_cv_clone.n_iter

    assert actual_combinations == expected_combinations, (
        f"RandomizedSearchCV should evaluate {expected_combinations} combinations (n_iter), got {actual_combinations}"
    )


def check_randomized_search_reproducibility(
    search_cv, y: pl.DataFrame, X: pl.DataFrame | None = None, forecasting_horizon: int = 3
) -> None:
    """Check random_state produces same parameter samples.

    Parameters
    ----------
    search_cv : RandomizedSearchCV
        Unfitted RandomizedSearchCV instance with random_state set
    y : pl.DataFrame
        Training target data
    X : pl.DataFrame, optional
        Training features
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast

    Raises
    ------
    AssertionError
        If random_state doesn't produce reproducible results

    """

    if not isinstance(search_cv, RandomizedSearchCV):
        raise ValueError("This check requires RandomizedSearchCV instance")

    if search_cv.random_state is None:
        raise ValueError("This check requires random_state to be set")

    # Fit first time
    search_cv_clone1 = clone(search_cv)
    search_cv_clone1.fit(y, X, forecasting_horizon=forecasting_horizon)

    # Fit second time with same random_state
    search_cv_clone2 = clone(search_cv)
    search_cv_clone2.fit(y, X, forecasting_horizon=forecasting_horizon)

    # Check that same parameters were sampled
    params1 = search_cv_clone1.cv_results_["params"]
    params2 = search_cv_clone2.cv_results_["params"]

    assert params1 == params2, "RandomizedSearchCV should produce same parameters with same random_state"

    # Check that scores are identical
    if not search_cv_clone1.multimetric_:
        scores1 = search_cv_clone1.cv_results_["mean_test_score"]
        scores2 = search_cv_clone2.cv_results_["mean_test_score"]

        np.testing.assert_array_equal(
            scores1,
            scores2,
            err_msg="RandomizedSearchCV should produce same scores with same random_state",
        )
    else:
        # For multimetric, check all scorer scores
        scorer_names = list(search_cv_clone1.scorer_.keys()) if hasattr(search_cv_clone1.scorer_, "keys") else []
        for scorer_name in scorer_names:
            scores1 = search_cv_clone1.cv_results_[f"mean_test_{scorer_name}"]
            scores2 = search_cv_clone2.cv_results_[f"mean_test_{scorer_name}"]

            np.testing.assert_array_equal(
                scores1,
                scores2,
                err_msg=f"RandomizedSearchCV should produce same {scorer_name} scores with same random_state",
            )


def check_randomized_search_distributions(
    search_cv, y: pl.DataFrame, X: pl.DataFrame | None = None, forecasting_horizon: int = 3
) -> None:
    """Check scipy.stats distributions work for parameter sampling.

    Parameters
    ----------
    search_cv : RandomizedSearchCV
        Unfitted RandomizedSearchCV instance with scipy distributions
    y : pl.DataFrame
        Training target data
    X : pl.DataFrame, optional
        Training features
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast

    Raises
    ------
    AssertionError
        If distributions don't work correctly

    """

    if not isinstance(search_cv, RandomizedSearchCV):
        raise ValueError("This check requires RandomizedSearchCV instance")

    search_cv_clone = clone(search_cv)
    search_cv_clone.fit(y, X, forecasting_horizon=forecasting_horizon)

    # Check that fit completed successfully
    assert hasattr(search_cv_clone, "cv_results_"), (
        "RandomizedSearchCV should fit successfully with scipy distributions"
    )

    # Check that parameters were sampled
    params = search_cv_clone.cv_results_["params"]
    assert len(params) > 0, "RandomizedSearchCV should sample parameters"


def check_search_panel_data(
    search_cv,
    y_train: pl.DataFrame,
    y_test: pl.DataFrame,
    X_train: pl.DataFrame | None = None,
    X_test: pl.DataFrame | None = None,
    panel_group_names: list[str] | None = None,
) -> None:
    """Check panel_group_names parameter propagates correctly.

    Parameters
    ----------
    search_cv : BaseSearchCV
        Fitted search CV instance with panel data
    y_train : pl.DataFrame
        Training target data with panel groups
    y_test : pl.DataFrame
        Test target data with panel groups
    X_train : pl.DataFrame, optional
        Training features with panel groups
    X_test : pl.DataFrame, optional
        Test features with panel groups
    panel_group_names : list of str, optional
        Panel group names to test

    Raises
    ------
    AssertionError
        If panel_group_names doesn't propagate correctly

    """
    forecasting_horizon = min(3, len(y_test))

    # Test predict with panel_group_names
    y_pred = search_cv.predict(forecasting_horizon=forecasting_horizon, X=X_test, panel_group_names=panel_group_names)

    # Check that predictions have panel structure if expected
    if panel_group_names is not None:
        # Predictions should only include specified groups

        _, panel_groups = inspect_panel(y_pred)

        # Check that all requested groups are present
        pred_group_prefixes = set(panel_groups.keys())
        for group_name in panel_group_names:
            # Group name might be a prefix
            assert any(group_name in prefix for prefix in pred_group_prefixes), (
                f"Requested panel group '{group_name}' not found in predictions"
            )


def check_search_method_availability(
    search_cv, y: pl.DataFrame, X: pl.DataFrame | None = None, forecasting_horizon: int = 3
) -> None:
    """Check @available_if decorator logic with refit=True/False.

    Parameters
    ----------
    search_cv : BaseSearchCV
        Unfitted search CV instance
    y : pl.DataFrame
        Training target data
    X : pl.DataFrame, optional
        Training features
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast

    Raises
    ------
    AssertionError
        If method availability doesn't match refit setting

    """
    # Test with refit=True
    search_cv_refit = clone(search_cv)
    # For multimetric, preserve the original refit value (scorer name or callable)
    # For single metric, set refit=True
    if not (isinstance(search_cv.scoring, dict) and isinstance(search_cv.refit, str)):
        search_cv_refit.refit = True
    search_cv_refit.fit(y, X, forecasting_horizon=forecasting_horizon)

    # Methods should be available
    assert hasattr(search_cv_refit, "predict"), "predict() should be available when refit=True"
    assert callable(search_cv_refit.predict), "predict should be callable when refit=True"

    # Test with refit=False
    search_cv_no_refit = clone(search_cv)
    search_cv_no_refit.refit = False
    search_cv_no_refit.fit(y, X, forecasting_horizon=forecasting_horizon)

    # Methods should raise AttributeError
    try:
        search_cv_no_refit.predict(forecasting_horizon=1)
        raise AssertionError("predict() should raise AttributeError when refit=False")
    except AttributeError:
        # Expected behavior
        pass


def check_search_interval_predict_delegates(
    search_cv,
    y_train: pl.DataFrame,
    y_test: pl.DataFrame,
    X_train: pl.DataFrame | None = None,
    X_test: pl.DataFrame | None = None,
) -> None:
    """Check predict_interval() works after interval search with refit.

    Validates that the best forecaster supports ``predict_interval`` and
    returns a valid interval prediction DataFrame.

    Parameters
    ----------
    search_cv : BaseSearchCV
        Fitted search CV instance (interval scorer, refit=True).
    y_train : pl.DataFrame
        Training target data.
    y_test : pl.DataFrame
        Test target data.
    X_train : pl.DataFrame, optional
        Training features.
    X_test : pl.DataFrame, optional
        Test features.

    Raises
    ------
    AssertionError
        If predict_interval() fails or returns invalid predictions.

    """
    check_is_fitted(search_cv)

    coverage_rates = [0.9]

    y_pred = search_cv.predict_interval(X=X_test, coverage_rates=coverage_rates)

    assert isinstance(y_pred, pl.DataFrame), f"predict_interval should return pl.DataFrame, got {type(y_pred)}"
    assert "time" in y_pred.columns, "Interval predictions should have 'time' column"

    interval_cols = [c for c in y_pred.columns if "_lower_" in c or "_upper_" in c]
    assert len(interval_cols) > 0, f"Interval predictions should have _lower_/_upper_ columns, got {y_pred.columns}"
