"""Check functions for yohou splitters.

This module provides systematic validation functions for testing BaseSplitter
implementations. All check functions raise AssertionError on failure.

Organized into three categories:
1. Tag System Checks - Validate tag system correctness (3 functions)
2. Functionality Checks - Core splitter behavior validation (4 functions)
3. Parameter Validation Checks - Validate parameter constraints (1 function)
"""

import datetime
import inspect

try:
    import polars as pl
except ImportError as e:
    raise ImportError("polars is required for yohou.testing module. Install with: uv sync --group tests") from e

import numpy as np
from sklearn.base import clone

__all__ = [
    "check_splitter_n_splits_consistency",
    "check_splitter_non_overlapping_tests",
    "check_splitter_panel_data_support",
    "check_splitter_parameter_constraints",
    "check_splitter_produces_valid_indices",
    "check_splitter_tags_accessible_before_fit",
    "check_splitter_tags_match_capabilities",
    "check_splitter_tags_static_after_fit",
]


def check_splitter_tags_accessible_before_fit(splitter) -> None:
    """Check __sklearn_tags__() is callable on splitter instance.

    Tags should be accessible on instances to describe capabilities.

    Parameters
    ----------
    splitter : BaseSplitter instance
        Splitter instance (can be unfitted)

    Raises
    ------
    AssertionError
        If __sklearn_tags__() is not callable or raises an error

    """
    splitter_name = type(splitter).__name__
    assert hasattr(splitter, "__sklearn_tags__"), f"{splitter_name} must have __sklearn_tags__() method"

    # Should be callable on instance
    try:
        tags = splitter.__sklearn_tags__()
    except Exception as e:
        raise AssertionError(f"{splitter_name}.__sklearn_tags__() raised {type(e).__name__}: {e}") from e

    assert tags.estimator_type == "splitter", (
        f"{splitter_name} tags must have estimator_type='splitter', got {tags.estimator_type}"
    )


def check_splitter_tags_static_after_fit(splitter, y: pl.DataFrame, X_actual: pl.DataFrame | None = None) -> None:
    """Check tags remain unchanged after fit.

    Tags describe inherent capabilities, not fitted state.
    They should be identical before and after fitting.

    Parameters
    ----------
    splitter : BaseSplitter
        Unfitted splitter instance
    y : pl.DataFrame
        Target time series with "time" column
    X_actual : pl.DataFrame, optional
        Exogenous features

    Raises
    ------
    AssertionError
        If tags change after fitting

    """
    splitter_clone = clone(splitter)

    # Get tags before any operation
    tags_before = splitter_clone.__sklearn_tags__()

    # Run split (which validates but doesn't "fit" in sklearn sense)
    _ = list(splitter_clone.split(y, X_actual))

    # Get tags after split
    tags_after = splitter_clone.__sklearn_tags__()

    # Tags should be identical
    assert tags_before == tags_after, "Tags changed after split() - tags should be static"


def check_splitter_tags_match_capabilities(
    splitter,
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None = None,
    expected_tags: dict | None = None,
) -> None:
    """Check tag values match actual splitter behavior.

    Validates that declared tags accurately reflect what the splitter does.

    Parameters
    ----------
    splitter : BaseSplitter
        Splitter instance
    y : pl.DataFrame
        Target time series with "time" column
    X_actual : pl.DataFrame, optional
        Exogenous features
    expected_tags : dict, optional
        Expected tag values to validate (keys: splitter_type, supports_panel_data, etc.)

    Raises
    ------
    AssertionError
        If tags don't match actual capabilities

    """
    tags = splitter.__sklearn_tags__()
    splitter_tags = tags.splitter_tags

    if expected_tags is not None:
        for key, expected_value in expected_tags.items():
            actual_value = getattr(splitter_tags, key)
            assert actual_value == expected_value, f"Tag mismatch: {key}={actual_value}, expected {expected_value}"

    # get_n_splits should always work without data
    try:
        n_splits_no_data = splitter.get_n_splits(y=None, X_actual=None)
        assert isinstance(n_splits_no_data, int) and n_splits_no_data >= 0, (
            "get_n_splits() with y=None should return non-negative integer"
        )
    except ValueError as e:
        raise AssertionError(f"{type(splitter).__name__}.get_n_splits(y=None) raised ValueError: {e}") from e


def check_splitter_produces_valid_indices(splitter, y: pl.DataFrame, X_actual: pl.DataFrame | None = None) -> None:
    """Check all train/test indices are valid row positions.

    Indices should be non-negative integers within [0, len(y)).

    Parameters
    ----------
    splitter : BaseSplitter
        Splitter instance
    y : pl.DataFrame
        Target time series with "time" column
    X_actual : pl.DataFrame, optional
        Exogenous features

    Raises
    ------
    AssertionError
        If indices are out of bounds or invalid

    """
    n_samples = len(y)

    for i, (train_idx, test_idx) in enumerate(splitter.split(y, X_actual)):
        # Check train indices
        assert isinstance(train_idx, np.ndarray), f"Split {i}: train indices should be ndarray"
        assert train_idx.dtype == np.intp, f"Split {i}: train indices should have dtype intp"
        assert len(train_idx) > 0, f"Split {i}: train set cannot be empty"
        assert np.all(train_idx >= 0), f"Split {i}: train indices contain negative values"
        assert np.all(train_idx < n_samples), f"Split {i}: train indices out of bounds (>= {n_samples})"

        # Check test indices
        assert isinstance(test_idx, np.ndarray), f"Split {i}: test indices should be ndarray"
        assert test_idx.dtype == np.intp, f"Split {i}: test indices should have dtype intp"
        assert len(test_idx) > 0, f"Split {i}: test set cannot be empty"
        assert np.all(test_idx >= 0), f"Split {i}: test indices contain negative values"
        assert np.all(test_idx < n_samples), f"Split {i}: test indices out of bounds (>= {n_samples})"

        # Check temporal ordering (train should come before test)
        assert np.max(train_idx) < np.min(test_idx), (
            f"Split {i}: train indices must be before test indices (time series ordering)"
        )


def check_splitter_n_splits_consistency(splitter, y: pl.DataFrame, X_actual: pl.DataFrame | None = None) -> None:
    """Check get_n_splits() matches actual split count.

    Parameters
    ----------
    splitter : BaseSplitter
        Splitter instance
    y : pl.DataFrame
        Target time series with "time" column
    X_actual : pl.DataFrame, optional
        Exogenous features

    Raises
    ------
    AssertionError
        If reported n_splits doesn't match actual count

    """
    reported_n_splits = splitter.get_n_splits(y, X_actual)
    actual_splits = list(splitter.split(y, X_actual))
    actual_n_splits = len(actual_splits)

    assert reported_n_splits == actual_n_splits, (
        f"get_n_splits() returned {reported_n_splits} but split() yielded {actual_n_splits} splits"
    )


def check_splitter_non_overlapping_tests(splitter, y: pl.DataFrame, X_actual: pl.DataFrame | None = None) -> None:
    """Check test sets don't overlap if produces_non_overlapping_tests=True.

    Parameters
    ----------
    splitter : BaseSplitter
        Splitter instance
    y : pl.DataFrame
        Target time series with "time" column
    X_actual : pl.DataFrame, optional
        Exogenous features

    Raises
    ------
    AssertionError
        If test sets overlap when they shouldn't

    """
    tags = splitter.__sklearn_tags__()
    if not tags.splitter_tags.produces_non_overlapping_tests:
        # Skip check if overlap is allowed
        return

    splits = list(splitter.split(y, X_actual))
    test_sets = [set(test_idx) for _, test_idx in splits]

    # Check all pairs for overlap
    for i in range(len(test_sets)):
        for j in range(i + 1, len(test_sets)):
            overlap = test_sets[i] & test_sets[j]
            assert len(overlap) == 0, (
                f"Splits {i} and {j} have overlapping test sets (indices: {sorted(overlap)}), "
                f"but produces_non_overlapping_tests=True"
            )


def check_splitter_panel_data_support(
    splitter,
    y_panel: pl.DataFrame,
    X_panel: pl.DataFrame | None = None,
) -> None:
    """Check splitter handles panel data if supports_panel_data=True.

    Parameters
    ----------
    splitter : BaseSplitter
        Splitter instance
    y_panel : pl.DataFrame
        Panel data with prefixed columns following group__column naming
        (e.g., "store_1__sales")
    X_panel : pl.DataFrame, optional
        Panel exogenous features

    Raises
    ------
    AssertionError
        If splitter declares support but fails on panel data

    """
    tags = splitter.__sklearn_tags__()
    if not tags.splitter_tags.supports_panel_data:
        # Skip check if panel data not supported
        return

    # Should not raise on panel data
    try:
        splits = list(splitter.split(y_panel, X_panel))
        assert len(splits) > 0, "Panel data split produced no splits"
    except Exception as e:
        raise AssertionError(f"Splitter declares supports_panel_data=True but failed on panel data: {e}") from e


def check_splitter_parameter_constraints(
    splitter_class,
    param_name: str,
    invalid_values: list,
) -> None:
    """Check parameter constraints are enforced via sklearn validation.

    Parameters
    ----------
    splitter_class : type
        Splitter class
    param_name : str
        Parameter name to test
    invalid_values : list
        List of invalid values that should trigger ValueError

    Raises
    ------
    AssertionError
        If invalid values are accepted

    """
    # Supply each other constructor parameter its own declared default so that
    # only ``param_name`` is exercised. Parameters without a default are left
    # out; the splitter is expected to validate ``param_name`` regardless.
    init_params = inspect.signature(splitter_class.__init__).parameters
    defaults = {
        name: param.default
        for name, param in init_params.items()
        if name not in ("self", param_name)
        and param.default is not inspect.Parameter.empty
        and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }

    for invalid_value in invalid_values:
        try:
            # Create instance with invalid parameter + defaults for required params
            params = defaults.copy()
            params[param_name] = invalid_value
            splitter = splitter_class(**params)

            # sklearn validates on first method call, so try get_n_splits or split
            try:
                splitter.get_n_splits(y=None, X_actual=None)
            except ValueError as e:
                # Check if it's a parameter validation error
                error_msg = str(e)
                if param_name in error_msg or "parameter" in error_msg.lower():
                    continue  # Parameter validation worked
                # Otherwise try with actual data
                y_test = pl.DataFrame({
                    "time": [datetime.datetime(2020, 1, i) for i in range(1, 101)],
                    "value": range(100),
                })
                list(splitter.split(y_test))

            # If we reach here, invalid value was accepted
            raise AssertionError(f"{splitter_class.__name__}: invalid {param_name}={invalid_value} was accepted")

        except (ValueError, TypeError) as e:
            # Expected: parameter validation should catch this
            error_msg = str(e)
            assert param_name in error_msg or "parameter" in error_msg.lower(), (
                f"Expected validation error for {param_name}, got: {e}"
            )
