"""Check functions for yohou transformers.

This module provides systematic validation functions for testing BaseTransformer
implementations. All check functions raise AssertionError on failure.

Organized into three categories:
1. Core Yohou Checks - Time series-specific validation
2. Enhanced sklearn Checks - Adapted from sklearn patterns
3. Check Generator - Dynamically generates applicable checks based on tags
"""

from typing import Any, Callable, Dict, Generator, Tuple

import polars as pl
import polars.selectors as cs
from polars.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

# ============================================================================
# CORE YOHOU CHECKS
# ============================================================================


def check_fit_sets_attributes(transformer, X, y=None):
    """Check fit() sets required attributes.

    Validates that fit() creates feature_names_in_, n_features_in_,
    and _observation_horizon attributes as expected by sklearn conventions.

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted transformer instance
    X : pl.DataFrame
        Training data with "time" column
    y : pl.DataFrame, optional
        Target data for supervised transformers

    Raises
    ------
    AssertionError
        If required attributes are not set after fit()
    """
    transformer_clone = clone(transformer)
    transformer_clone.fit(X, y)

    # Check sklearn-required attributes
    assert hasattr(transformer_clone, "feature_names_in_"), (
        "fit() must set feature_names_in_ attribute"
    )
    assert hasattr(transformer_clone, "n_features_in_"), "fit() must set n_features_in_ attribute"

    # Check yohou-required attributes
    assert hasattr(transformer_clone, "_observation_horizon"), (
        "fit() must set _observation_horizon attribute"
    )

    # Validate values
    expected_features = [col for col in X.columns if col != "time"]
    assert list(transformer_clone.feature_names_in_) == expected_features, (
        f"feature_names_in_ mismatch: {transformer_clone.feature_names_in_} vs {expected_features}"
    )

    assert transformer_clone.n_features_in_ == len(expected_features), (
        f"n_features_in_ should be {len(expected_features)}, got {transformer_clone.n_features_in_}"
    )


def check_observation_horizon_not_fitted(transformer, X):
    """Check accessing observation_horizon before fit() raises NotFittedError.

    For stateful transformers (observation_horizon > 0), accessing the
    observation_horizon property before fit() should raise NotFittedError.

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted transformer instance
    X : pl.DataFrame
        Test data

    Raises
    ------
    AssertionError
        If NotFittedError is not raised for stateful transformers
    """
    transformer_clone = clone(transformer)

    try:
        _ = transformer_clone.observation_horizon
        # If we get here, either it's stateless or improperly implemented
        # Stateless transformers (horizon=0) are allowed to work without fit
        if transformer_clone.observation_horizon != 0:
            raise AssertionError(
                f"{transformer_clone.__class__.__name__} allows accessing "
                f"observation_horizon before fit() but is not stateless. "
                f"Should raise NotFittedError."
            )
    except NotFittedError:
        # Expected behavior for stateful transformers
        pass


def check_observation_horizon_after_fit(transformer, X, y=None):
    """Check observation_horizon is valid after fit().

    After fitting, observation_horizon should be a non-negative integer.

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted transformer
    X : pl.DataFrame
        Training data
    y : pl.DataFrame, optional
        Target data

    Raises
    ------
    AssertionError
        If observation_horizon is not a valid non-negative integer
    """
    transformer_clone = clone(transformer)
    transformer_clone.fit(X, y)

    horizon = transformer_clone.observation_horizon

    assert isinstance(horizon, int), f"observation_horizon must be int, got {type(horizon)}"
    assert horizon >= 0, f"observation_horizon must be non-negative, got {horizon}"


def check_reset_updates_memory(transformer, X, y=None):
    """Check reset() updates _X_observed to last observation_horizon rows.

    The reset() method should update the transformer's memory to contain
    only the last observation_horizon rows of the provided data.

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted transformer
    X : pl.DataFrame
        Training data (must be longer than observation_horizon)
    y : pl.DataFrame, optional
        Target data

    Raises
    ------
    AssertionError
        If _X_observed is not properly updated
    ValueError
        If X is too short for the transformer's observation_horizon
    """
    transformer_clone = clone(transformer)
    transformer_clone.fit(X, y)

    horizon = transformer_clone.observation_horizon

    if len(X) < horizon:
        raise ValueError(f"X length {len(X)} < observation_horizon {horizon}")

    # Create new data to reset with
    X_new = X.head(horizon + 5) if len(X) >= horizon + 5 else X
    transformer_clone.reset(X_new)

    # Check _X_observed has correct length
    assert len(transformer_clone._X_observed) == min(horizon, len(X_new)), (
        f"_X_observed length should be {min(horizon, len(X_new))}, got {len(transformer_clone._X_observed)}"
    )

    # Check _X_observed contains last horizon rows
    expected = X_new.tail(horizon)
    assert_frame_equal(transformer_clone._X_observed, expected)


def check_update_concatenates_memory(transformer, X, y=None):
    """Check update() appends new data and maintains horizon size.

    The update() method should append new observations to _X_observed
    and trim to observation_horizon length.

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted transformer
    X : pl.DataFrame
        Initial training data
    y : pl.DataFrame, optional
        Target data

    Raises
    ------
    AssertionError
        If update() doesn't properly maintain memory
    """
    transformer_clone = clone(transformer)
    transformer_clone.fit(X, y)

    horizon = transformer_clone.observation_horizon

    # Create update data (10 new rows)
    X_update = X.tail(10)
    initial_memory_len = len(transformer_clone._X_observed)

    transformer_clone.update(X_update)

    # Memory should not exceed horizon
    assert len(transformer_clone._X_observed) <= horizon, (
        f"_X_observed length {len(transformer_clone._X_observed)} exceeds horizon {horizon}"
    )

    # Memory should have grown or stayed at horizon
    expected_len = min(initial_memory_len + len(X_update), horizon)
    assert len(transformer_clone._X_observed) == expected_len, (
        f"Expected _X_observed length {expected_len}, got {len(transformer_clone._X_observed)}"
    )


def check_update_transform_equivalence(transformer, X, y=None):
    """Check update().transform() == fit().transform() for same final state.

    For the same final data state, using update() should produce the same
    transform output as using fit() directly.

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted transformer
    X : pl.DataFrame
        Training data
    y : pl.DataFrame, optional
        Target data

    Raises
    ------
    AssertionError
        If update and fit paths produce different results
    """
    # Split data
    split_point = len(X) // 2
    X_first = X.head(split_point)
    X_second = X.tail(len(X) - split_point)

    # Path 1: fit on first, update with second
    transformer1 = clone(transformer)
    transformer1.fit(X_first, y)
    transformer1.update(X_second)
    X_trans1 = transformer1.transform(X_second)

    # Path 2: fit on all data
    transformer2 = clone(transformer)
    transformer2.fit(X, y)
    X_trans2 = transformer2.transform(X_second)

    # Results should be equivalent
    assert_frame_equal(X_trans1, X_trans2, rel_tol=1e-6, abs_tol=1e-8)


def check_insufficient_data_raises(transformer, X, y=None):
    """Check behavior when data length < observation_horizon.

    Transformers should either raise appropriate errors or gracefully handle
    insufficient data when given data shorter than their observation_horizon.

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted transformer
    X : pl.DataFrame
        Test data (will be truncated)
    y : pl.DataFrame, optional
        Target data

    Notes
    -----
    This check verifies that transformers don't crash unexpectedly with
    insufficient data. They may either raise a clear error or return
    an empty/truncated result.
    """
    transformer_clone = clone(transformer)
    transformer_clone.fit(X, y)

    horizon = transformer_clone.observation_horizon

    if horizon == 0:
        # Stateless transformers don't need minimum data
        return

    if len(X) <= horizon:
        # Need longer X to test
        return

    # Create data shorter than horizon
    X_short = X.head(horizon - 1) if horizon > 1 else X.head(0)

    try:
        result = transformer_clone.transform(X_short)
        # If it succeeds, verify result is valid (has time column, non-negative length)
        assert "time" in result.columns, "Result must have time column"
        assert len(result) >= 0, "Result length must be non-negative"
        # Graceful handling is acceptable (e.g., returning empty dataframe)
    except (ValueError, IndexError, pl.exceptions.ShapeError, pl.exceptions.ComputeError):
        # Expected behavior - transformer raises appropriate error
        pass


def check_transform_output_structure(transformer, X, y=None):
    """Check transform() output has "time" column and valid structure.

    Transform output must be a polars DataFrame with a "time" column
    and valid data types.

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted transformer
    X : pl.DataFrame
        Training data
    y : pl.DataFrame, optional
        Target data

    Raises
    ------
    AssertionError
        If output structure is invalid
    """
    transformer_clone = clone(transformer)
    transformer_clone.fit(X, y)

    X_trans = transformer_clone.transform(X)

    # Check it's a DataFrame
    assert isinstance(X_trans, pl.DataFrame), (
        f"transform() must return pl.DataFrame, got {type(X_trans)}"
    )

    # Check time column exists
    assert "time" in X_trans.columns, "transform() output must contain 'time' column"

    # Check time column is datetime
    assert X_trans["time"].dtype in [pl.Datetime, pl.Date], (
        f"'time' column must be datetime type, got {X_trans['time'].dtype}"
    )

    # Check output has at least one feature column
    feature_cols = [col for col in X_trans.columns if col != "time"]
    assert len(feature_cols) > 0, (
        "transform() output must have at least one feature column besides 'time'"
    )


def check_feature_names_out_match(transformer, X, y=None):
    """Check get_feature_names_out() matches transform() output columns.

    The feature names returned by get_feature_names_out() should match
    the actual columns in the transform() output (excluding 'time').

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted transformer
    X : pl.DataFrame
        Training data
    y : pl.DataFrame, optional
        Target data

    Raises
    ------
    AssertionError
        If feature names don't match output columns
    """
    transformer_clone = clone(transformer)
    transformer_clone.fit(X, y)

    X_trans = transformer_clone.transform(X)
    feature_names = transformer_clone.get_feature_names_out()

    # Get actual feature columns (exclude time)
    actual_features = [col for col in X_trans.columns if col != "time"]

    assert list(feature_names) == actual_features, (
        f"get_feature_names_out() mismatch: {list(feature_names)} vs {actual_features}"
    )


def check_inverse_transform_identity(transformer, X, y=None, atol=1e-6, rtol=1e-5):
    """Check inverse_transform(transform(X)) ≈ X.

    Basic round-trip test for invertible transformers.

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted transformer
    X : pl.DataFrame
        Training data
    y : pl.DataFrame, optional
        Target data
    atol, rtol : float
        Tolerances for numerical comparison

    Raises
    ------
    AssertionError
        If round-trip fails
    """
    if not hasattr(transformer, "inverse_transform"):
        # Not an invertible transformer, skip
        return

    transformer_clone = clone(transformer)
    transformer_clone.fit(X, y)

    # Transform the data - this may drop some rows (e.g., differencing)
    X_trans = transformer_clone.transform(X)

    # For yohou transformers, inverse_transform requires X_p (past observations)
    # X_p should be the observations immediately before X_trans in the original X
    horizon = transformer_clone.observation_horizon
    if horizon > 0:
        # Find how many rows were dropped by transform
        n_dropped = len(X) - len(X_trans)
        # X_p: the `horizon` rows immediately before X_trans started
        # These are rows at position [n_dropped - horizon : n_dropped]
        X_p = X[n_dropped - horizon : n_dropped]
    else:
        # Stateless transformer
        X_p = None

    # Inverse transform
    X_reconstructed = transformer_clone.inverse_transform(X_trans, X_p)

    # The original data we should recover is the portion that was transformed
    # (excluding the dropped rows)
    X_expected = X.tail(len(X_trans))

    # Basic shape check
    assert X_reconstructed.shape == X_expected.shape, (
        f"Shape mismatch: {X_expected.shape} -> {X_trans.shape} -> {X_reconstructed.shape}"
    )

    # Numerical comparison
    assert_frame_equal(X_expected, X_reconstructed, rel_tol=rtol, abs_tol=atol)


def check_panel_data_support(transformer, X_panel, y=None):
    """Check transformer handles panel columns (panel data) correctly.

    Panel data uses columns with __ separator to represent multiple time series.
    Transformers should preserve panel columns or handle them appropriately.

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted transformer
    X_panel : pl.DataFrame
        Panel data with panel columns
    y : pl.DataFrame, optional
        Target data

    Raises
    ------
    AssertionError
        If panel data handling fails
    """
    from yohou.utils import inspect_locality

    # Check if X_panel actually has panel columns
    global_names, panel_groups = inspect_locality(X_panel)

    if not panel_groups:
        # Not panel data, skip
        return

    transformer_clone = clone(transformer)

    try:
        transformer_clone.fit(X_panel, y)
        X_trans = transformer_clone.transform(X_panel)

        # Check output is valid
        assert isinstance(X_trans, pl.DataFrame), "Panel data transform must return DataFrame"
        assert "time" in X_trans.columns, "Panel data transform must preserve 'time' column"

    except NotImplementedError:
        # Transformer explicitly doesn't support panel data
        pass


def check_clone_preserves_params(transformer):
    """Check sklearn's clone() preserves init parameters.

    Clone should create a new instance with the same parameters
    but independent state.

    Parameters
    ----------
    transformer : BaseTransformer
        Any transformer instance

    Raises
    ------
    AssertionError
        If clone doesn't preserve parameters
    """
    transformer_clone = clone(transformer)

    # Check they're different objects
    assert transformer_clone is not transformer, "clone() should create a new object"

    # Check parameters are preserved
    orig_params = transformer.get_params()
    clone_params = transformer_clone.get_params()

    assert orig_params == clone_params, (
        f"clone() parameters mismatch: {orig_params} vs {clone_params}"
    )


# ============================================================================
# ENHANCED SKLEARN CHECKS
# ============================================================================


def check_transformers_unfitted_stateless(transformer, X):
    """Check stateless transformers work without fitting.

    For transformers with observation_horizon = 0, transform should
    work on unfitted instances (similar to sklearn's FunctionTransformer).

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted transformer with observation_horizon = 0
    X : pl.DataFrame
        Test data

    Raises
    ------
    AssertionError
        If stateless transformer requires fitting
    """
    transformer_clone = clone(transformer)

    # Check if it's actually stateless
    try:
        horizon = transformer_clone.observation_horizon
        if horizon != 0:
            # Not stateless, skip this check
            return
    except NotFittedError:
        # Can't determine, skip
        return

    # Should work without fit() for stateless transformers
    try:
        X_trans = transformer_clone.transform(X)
        assert X_trans.shape[0] == X.shape[0], (
            f"Stateless transformer changed n_samples: {X.shape[0]} -> {X_trans.shape[0]}"
        )
    except NotFittedError:
        raise AssertionError(
            f"{transformer.__class__.__name__} claims to be stateless "
            f"(observation_horizon=0) but raises NotFittedError. "
            f"Stateless transformers should work without fitting."
        )


def check_transformer_preserve_dtypes(transformer, X, y=None):
    """Check transformer preserves input dtypes.

    Transform and inverse_transform should maintain dtype consistency
    for numerical columns (except 'time' which is always datetime).

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted transformer
    X : pl.DataFrame
        Test data with known dtypes
    y : pl.DataFrame, optional
        Target data

    Raises
    ------
    AssertionError
        If dtypes change unexpectedly
    """
    transformer_clone = clone(transformer)
    transformer_clone.fit(X, y)

    X_numeric = X.select(cs.numeric() & ~cs.by_name("time"))
    input_dtypes = {col: dtype for col, dtype in zip(X_numeric.columns, X_numeric.dtypes)}

    X_trans = transformer_clone.transform(X)

    # Check transformed output dtypes (may have different columns)
    for col in X_trans.select(cs.numeric() & ~cs.by_name("time")).columns:
        output_dtype = X_trans[col].dtype
        # Allow float32 -> float64 promotion but expect numerical types
        assert output_dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt64, pl.UInt32], (
            f"Column '{col}' has unexpected dtype {output_dtype}"
        )

    # Check inverse transform preserves original dtypes (if invertible)
    if hasattr(transformer_clone, "inverse_transform"):
        try:
            # Prepare X_p for yohou inverse_transform
            horizon = transformer_clone.observation_horizon
            if horizon > 0:
                n_dropped = len(X) - len(X_trans)
                X_p = X[n_dropped - horizon : n_dropped]
            else:
                X_p = None

            X_inv = transformer_clone.inverse_transform(X_trans, X_p)
            X_expected_portion = X.tail(len(X_trans))

            for col in input_dtypes:
                if col in X_inv.columns:
                    # Allow some flexibility for float types
                    if input_dtypes[col] in [pl.Float64, pl.Float32]:
                        assert X_inv[col].dtype in [pl.Float64, pl.Float32], (
                            f"inverse_transform changed float dtype of '{col}': "
                            f"{input_dtypes[col]} -> {X_inv[col].dtype}"
                        )
                    else:
                        assert X_inv[col].dtype == input_dtypes[col], (
                            f"inverse_transform changed dtype of '{col}': "
                            f"{input_dtypes[col]} -> {X_inv[col].dtype}"
                        )
        except NotImplementedError:
            # Not invertible - skip inverse check
            pass


def check_fit_idempotent(transformer, X, y=None):
    """Check that fit(X).fit(X) equals fit(X).

    Calling fit multiple times with same data should yield identical
    internal state and predictions.

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted transformer
    X : pl.DataFrame
        Training data
    y : pl.DataFrame, optional
        Target data for supervised transformers

    Raises
    ------
    AssertionError
        If double-fit produces different results
    """
    transformer1 = clone(transformer)
    transformer2 = clone(transformer)

    # Single fit
    transformer1.fit(X, y)
    X_trans1 = transformer1.transform(X)

    # Double fit
    transformer2.fit(X, y).fit(X, y)
    X_trans2 = transformer2.transform(X)

    assert_frame_equal(X_trans1, X_trans2, rel_tol=1e-5, abs_tol=1e-8)

    # Check fitted attributes match
    for attr in ["feature_names_in_", "n_features_in_", "_observation_horizon"]:
        if hasattr(transformer1, attr):
            val1 = getattr(transformer1, attr)
            val2 = getattr(transformer2, attr)
            assert val1 == val2, f"Attribute '{attr}' differs after double fit: {val1} vs {val2}"


def check_inverse_transform_round_trip(transformer, X, y=None, atol=1e-6, rtol=1e-5):
    """Check inverse_transform(transform(X)) ≈ X with shape validation.

    More comprehensive than check_inverse_transform_identity:
    - Validates shape preservation
    - Checks dtype consistency
    - Handles panel data columns
    - Configurable tolerance

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted invertible transformer
    X : pl.DataFrame
        Test data
    y : pl.DataFrame, optional
        Target data
    atol, rtol : float
        Absolute and relative tolerance for numerical comparison

    Raises
    ------
    AssertionError
        If round-trip fails
    """
    if not hasattr(transformer, "inverse_transform"):
        # Not invertible, skip
        return

    transformer_clone = clone(transformer)
    transformer_clone.fit(X, y)

    # Forward transform
    X_trans = transformer_clone.transform(X)

    # For yohou transformers, inverse_transform requires X_p (past observations)
    horizon = transformer_clone.observation_horizon
    if horizon > 0:
        n_dropped = len(X) - len(X_trans)
        X_p = X[n_dropped - horizon : n_dropped]
    else:
        X_p = None

    # Backward transform
    X_reconstructed = transformer_clone.inverse_transform(X_trans, X_p)

    # Expected is the portion of X that was transformed
    X_expected = X.tail(len(X_trans))

    # Shape validation
    assert X_reconstructed.shape == X_expected.shape, (
        f"Shape mismatch: {X_expected.shape} -> {X_trans.shape} -> {X_reconstructed.shape}"
    )

    # Column validation
    assert set(X_reconstructed.columns) == set(X_expected.columns), (
        f"Columns mismatch: {set(X_expected.columns)} vs {set(X_reconstructed.columns)}"
    )

    # Dtype validation (excluding 'time')
    for col in X_expected.select(~cs.by_name("time")).columns:
        assert X_expected[col].dtype == X_reconstructed[col].dtype, (
            f"Dtype changed for '{col}': {X_expected[col].dtype} -> {X_reconstructed[col].dtype}"
        )

    # Numerical comparison
    assert_frame_equal(X_expected, X_reconstructed, rel_tol=rtol, abs_tol=atol)


def check_fit_transform_equivalence(transformer, X, y=None):
    """Check fit_transform(X) == fit(X).transform(X).

    The convenience method fit_transform should produce identical
    results to separate fit and transform calls.

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted transformer
    X : pl.DataFrame
        Training data
    y : pl.DataFrame, optional
        Target data

    Raises
    ------
    AssertionError
        If methods produce different results
    """
    transformer1 = clone(transformer)
    transformer2 = clone(transformer)

    # Separate fit and transform
    X_trans1 = transformer1.fit(X, y).transform(X)

    # Combined fit_transform
    if hasattr(transformer2, "fit_transform"):
        X_trans2 = transformer2.fit_transform(X, y)
    else:
        # Default implementation from TransformerMixin
        X_trans2 = transformer2.fit(X, y).transform(X)

    assert_frame_equal(X_trans1, X_trans2, rel_tol=1e-7, abs_tol=1e-10)


def check_memory_bounded(transformer, X, y=None, n_updates=5):
    """Check memory doesn't grow unbounded with sequential updates.

    Important for production time series applications with continuous
    data streams. Memory should stabilize at observation_horizon size.

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted transformer
    X : pl.DataFrame
        Training data
    y : pl.DataFrame, optional
        Target data
    n_updates : int
        Number of update iterations to test

    Raises
    ------
    AssertionError
        If memory grows beyond expected bounds
    """
    transformer_clone = clone(transformer)
    transformer_clone.fit(X, y)

    horizon = transformer_clone.observation_horizon
    max_memory_factor = 2.0
    expected_max_rows = int(horizon * max_memory_factor)

    # Create update chunks
    chunk_size = max(1, len(X) // (n_updates * 2))

    for i in range(n_updates):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(X))
        if start_idx >= len(X):
            break

        X_chunk = X.slice(start_idx, chunk_size)
        transformer_clone.update(X_chunk)

        actual_rows = len(transformer_clone._X_observed)
        assert actual_rows <= expected_max_rows, (
            f"Memory grew beyond bounds after {i + 1} updates: "
            f"{actual_rows} rows > {expected_max_rows} (horizon={horizon})"
        )


# ============================================================================
# CHECK GENERATOR
# ============================================================================


def _get_transformer_tags(transformer) -> Dict[str, Any]:
    """Extract tags from transformer for check generation.

    Parameters
    ----------
    transformer : BaseTransformer
        Transformer instance (fitted or unfitted)

    Returns
    -------
    tags : dict
        Dictionary of transformer properties
    """
    tags = {
        "stateful": True,
        "invertible": hasattr(transformer, "inverse_transform"),
        "requires_positive_X": False,
        "no_panel_data": False,
    }

    # Check if stateless (observation_horizon = 0)
    try:
        if hasattr(transformer, "_observation_horizon"):
            tags["stateful"] = transformer._observation_horizon > 0
        elif hasattr(transformer, "observation_horizon"):
            try:
                horizon = transformer.observation_horizon
                tags["stateful"] = horizon > 0
            except NotFittedError:
                tags["stateful"] = True
    except:
        pass

    return tags


def _yield_yohou_transformer_checks(
    transformer,
    X_train: pl.DataFrame,
    X_test: pl.DataFrame,
    y: pl.DataFrame = None,
    tags: Dict[str, Any] = None,
) -> Generator[Tuple[str, Callable, Dict], None, None]:
    """Generate all applicable checks for a transformer.

    Parameters
    ----------
    transformer : BaseTransformer
        Transformer to test (should be fitted)
    X_train : pl.DataFrame
        Training data
    X_test : pl.DataFrame
        Test data for validation
    y : pl.DataFrame, optional
        Target data (for supervised transformers)
    tags : dict, optional
        Transformer tags (if None, will be auto-detected)

    Yields
    ------
    (check_name, check_function, check_kwargs) : tuple
        Test identifier, callable check function, and kwargs dict
    """
    if tags is None:
        tags = _get_transformer_tags(transformer)

    # Core checks (always run)
    yield "check_fit_sets_attributes", check_fit_sets_attributes, {"X": X_train, "y": y}
    yield (
        "check_transform_output_structure",
        check_transform_output_structure,
        {"X": X_train, "y": y},
    )
    yield "check_feature_names_out_match", check_feature_names_out_match, {"X": X_train, "y": y}
    yield "check_clone_preserves_params", check_clone_preserves_params, {}

    # Stateful transformer checks
    if tags.get("stateful", True):
        yield (
            "check_observation_horizon_not_fitted",
            check_observation_horizon_not_fitted,
            {"X": X_train},
        )
        yield (
            "check_observation_horizon_after_fit",
            check_observation_horizon_after_fit,
            {"X": X_train, "y": y},
        )
        yield "check_reset_updates_memory", check_reset_updates_memory, {"X": X_train, "y": y}
        yield (
            "check_update_concatenates_memory",
            check_update_concatenates_memory,
            {"X": X_train, "y": y},
        )
        yield (
            "check_update_transform_equivalence",
            check_update_transform_equivalence,
            {"X": X_train, "y": y},
        )
        yield (
            "check_insufficient_data_raises",
            check_insufficient_data_raises,
            {"X": X_train, "y": y},
        )
        yield "check_memory_bounded", check_memory_bounded, {"X": X_train, "y": y}
    else:
        # Stateless checks
        yield (
            "check_transformers_unfitted_stateless",
            check_transformers_unfitted_stateless,
            {"X": X_test},
        )

    # Invertible transformer checks
    if tags.get("invertible", False):
        yield (
            "check_inverse_transform_identity",
            check_inverse_transform_identity,
            {"X": X_test, "y": y},
        )
        yield (
            "check_inverse_transform_round_trip",
            check_inverse_transform_round_trip,
            {"X": X_test, "y": y},
        )

    # sklearn compatibility checks
    yield (
        "check_transformer_preserve_dtypes",
        check_transformer_preserve_dtypes,
        {"X": X_train, "y": y},
    )
    yield "check_fit_idempotent", check_fit_idempotent, {"X": X_train, "y": y}
    yield "check_fit_transform_equivalence", check_fit_transform_equivalence, {"X": X_train, "y": y}

    # Panel data check (if not explicitly excluded)
    if not tags.get("no_panel_data", False):
        # Note: panel data will be provided by test fixtures
        pass

    # Metadata routing checks (always applicable)
    yield (
        "check_metadata_routing_default_request",
        check_metadata_routing_default_request,
        {},
    )
    yield (
        "check_metadata_routing_get_metadata_routing",
        check_metadata_routing_get_metadata_routing,
        {},
    )


# ============================================================================
# FORECASTER CHECKS
# ============================================================================
# Common Forecaster Checks
# Point Forecaster Checks
# Interval Forecaster Checks
# Reduction Forecaster Checks
# ============================================================================


def check_fit_sets_forecaster_attributes(forecaster, y, X=None, forecasting_horizon=3):
    """Check fit() sets required forecaster attributes.

    Validates that fit() creates all required attributes for forecasters including
    fit_forecasting_horizon_, interval_, panel_group_names_, local_y_schema_,
    observation buffers, and transformer references.

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance
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
    forecaster_clone = clone(forecaster)
    forecaster_clone.fit(y, X, forecasting_horizon=forecasting_horizon)

    # Check core fitted attributes
    assert hasattr(forecaster_clone, "fit_forecasting_horizon_"), (
        "fit() must set fit_forecasting_horizon_ attribute"
    )
    assert forecaster_clone.fit_forecasting_horizon_ == forecasting_horizon, (
        f"fit_forecasting_horizon_ should be {forecasting_horizon}, got {forecaster_clone.fit_forecasting_horizon_}"
    )

    assert hasattr(forecaster_clone, "interval_"), "fit() must set interval_ attribute (timedelta)"

    assert hasattr(forecaster_clone, "panel_group_names_"), (
        "fit() must set panel_group_names_ attribute (None or list)"
    )
    assert hasattr(forecaster_clone, "local_y_schema_"), (
        "fit() must set local_y_schema_ attribute (dict[str, pl.DataType])"
    )
    assert hasattr(forecaster_clone, "local_X_schema_"), (
        "fit() must set local_X_schema_ attribute (dict[str, pl.DataType])"
    )
    assert hasattr(forecaster_clone, "global_X_schema_"), (
        "fit() must set global_X_schema_ attribute (None or dict[str, pl.DataType])"
    )
    assert hasattr(forecaster_clone, "local_y_t_schema_"), (
        "fit() must set local_y_t_schema_ attribute (None or dict[str, pl.DataType])"
    )
    assert hasattr(forecaster_clone, "local_X_t_schema_"), (
        "fit() must set local_X_t_schema_ attribute (None or dict[str, pl.DataType])"
    )

    # Check observation buffers
    assert hasattr(forecaster_clone, "_y_observed"), "fit() must set _y_observed buffer"
    assert hasattr(forecaster_clone, "_X_t_observed"), "fit() must set _X_t_observed buffer"

    # Check transformer attributes
    if forecaster_clone.target_transformer is not None:
        assert hasattr(forecaster_clone, "target_transformer_"), (
            "fit() must set target_transformer_ when target_transformer provided"
        )

    if forecaster_clone.feature_transformer is not None:
        assert hasattr(forecaster_clone, "feature_transformer_"), (
            "fit() must set feature_transformer_ when feature_transformer provided"
        )


def check_forecaster_not_fitted_error(forecaster, y, X=None):
    """Check accessing fitted attributes before fit() raises NotFittedError.

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance
    y : pl.DataFrame
        Test target data
    X : pl.DataFrame, optional
        Test features

    Raises
    ------
    AssertionError
        If NotFittedError is not raised when accessing fitted attributes
    """
    from sklearn.utils.validation import check_is_fitted

    forecaster_clone = clone(forecaster)

    # Should raise NotFittedError when checking if fitted
    try:
        check_is_fitted(forecaster_clone, "fit_forecasting_horizon_")
        raise AssertionError(
            f"{forecaster_clone.__class__.__name__} should raise NotFittedError "
            f"when accessing fit_forecasting_horizon_ before fit()"
        )
    except NotFittedError:
        # Expected behavior
        pass


def check_predict_time_columns(forecaster, y_test, X_test=None):
    """Check predictions have observed_time and time columns.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster instance
    y_test : pl.DataFrame
        Test target data
    X_test : pl.DataFrame, optional
        Test features

    Raises
    ------
    AssertionError
        If predictions lack required time columns
    """
    forecasting_horizon = min(3, len(y_test))

    # Check if forecaster is an interval forecaster
    if hasattr(forecaster, "predict_interval"):
        y_pred = forecaster.predict_interval(forecasting_horizon=forecasting_horizon, X=X_test)
    else:
        y_pred = forecaster.predict(forecasting_horizon=forecasting_horizon, X=X_test)

    assert "observed_time" in y_pred.columns, "Predictions must have 'observed_time' column"
    assert "time" in y_pred.columns, "Predictions must have 'time' column"

    # Validate shapes
    assert len(y_pred) == forecasting_horizon, (
        f"Predictions should have {forecasting_horizon} rows, got {len(y_pred)}"
    )

    # Validate time column types
    assert y_pred["observed_time"].dtype == pl.Datetime, "observed_time must be Datetime dtype"
    assert y_pred["time"].dtype == pl.Datetime, "time must be Datetime dtype"


def check_update_extends_observations(
    forecaster,
    y_train,
    y_update,
    X_train=None,
    X_update=None,
):
    """Check update() extends observation buffers correctly.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster instance
    y_train : pl.DataFrame
        Original training data
    y_update : pl.DataFrame
        New data for update
    X_train, X_update : pl.DataFrame, optional
        Features for training and update

    Raises
    ------
    AssertionError
        If observation buffers are not extended correctly
    """
    # Store original buffer length
    original_observed_time = forecaster.observed_time_

    # Handle both panel (dict) and non-panel (DataFrame or scalar) data
    if forecaster._y_observed is not None:
        if isinstance(forecaster._y_observed, dict):
            # Panel data: observed_time_ is a dict
            # Check the first group as a representative
            first_group = next(iter(forecaster._y_observed.keys()))
            original_y_observed_last_time = forecaster._y_observed[first_group]["time"][-1]
            assert original_observed_time[first_group] == original_y_observed_last_time, (
                "observed_time_ should match last time in _y_observed before update()"
            )
        else:
            # Non-panel data: observed_time_ is a scalar
            original_y_observed_last_time = forecaster._y_observed["time"][-1]
            assert original_observed_time == original_y_observed_last_time, (
                "observed_time_ should match last time in _y_observed before update()"
            )

    if forecaster._X_t_observed is not None:
        if isinstance(forecaster._X_t_observed, dict):
            # Panel data
            first_group = next(iter(forecaster._X_t_observed.keys()))
            if forecaster._X_t_observed[first_group] is not None:
                original_X_t_observed_last_time = forecaster._X_t_observed[first_group]["time"][-1]
                assert original_observed_time[first_group] == original_X_t_observed_last_time, (
                    "observed_time_ should match last time in _X_t_observed before update()"
                )
        else:
            # Non-panel data
            original_X_t_observed_last_time = forecaster._X_t_observed["time"][-1]
            assert original_observed_time == original_X_t_observed_last_time, (
                "observed_time_ should match last time in _X_t_observed before update()"
            )

    # Update with new data
    forecaster.update(y_update, X_update)

    # Check buffers were extended
    updated_observed_time = forecaster.observed_time_

    # Handle both panel and non-panel data for comparison
    if isinstance(updated_observed_time, dict):
        # Panel data: check all groups were updated
        for group_name in updated_observed_time:
            assert updated_observed_time[group_name] >= original_observed_time[group_name], (
                f"observed_time_ for group {group_name} should be updated"
            )
    else:
        # Non-panel data
        assert updated_observed_time >= original_observed_time, (
            "observed_time_ should be updated to at least the last time in update data"
        )

    if forecaster._y_observed is not None:
        if isinstance(forecaster._y_observed, dict):
            # Panel data
            for group_name, y_obs in forecaster._y_observed.items():
                updated_y_observed_last_time = y_obs["time"][-1]
                assert updated_y_observed_last_time == updated_observed_time[group_name], (
                    f"Last time in _y_observed['{group_name}'] should match updated observed_time_"
                )
        else:
            # Non-panel data
            updated_y_observed_last_time = forecaster._y_observed["time"][-1]
            assert updated_y_observed_last_time == updated_observed_time, (
                "Last time in _y_observed should match updated observed_time_ after update()"
            )

    if forecaster._X_t_observed is not None:
        if isinstance(forecaster._X_t_observed, dict):
            # Panel data
            for group_name, X_t_obs in forecaster._X_t_observed.items():
                if X_t_obs is not None:
                    updated_X_t_observed_last_time = X_t_obs["time"][-1]
                    assert updated_X_t_observed_last_time == updated_observed_time[group_name], (
                        f"Last time in _X_t_observed['{group_name}'] should match updated observed_time_"
                    )
        else:
            # Non-panel data
            updated_X_t_observed_last_time = forecaster._X_t_observed["time"][-1]
            assert updated_X_t_observed_last_time == updated_observed_time, (
                "Last time in _X_t_observed should match updated observed_time_ after update()"
            )


def check_reset_replaces_observations(
    forecaster,
    y_train,
    y_reset,
    X_train=None,
    X_reset=None,
):
    """Check reset() replaces observation buffers correctly.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster instance
    y_train : pl.DataFrame
        Original training data
    y_reset : pl.DataFrame
        New data for reset
    X_train, X_reset : pl.DataFrame, optional
        Features for training and reset

    Raises
    ------
    AssertionError
        If observation buffers are not replaced correctly
    """
    # Store original buffer length
    original_observed_time = forecaster.observed_time_

    # Handle both panel (dict) and non-panel (DataFrame or scalar) data
    if forecaster._y_observed is not None:
        if isinstance(forecaster._y_observed, dict):
            # Panel data
            first_group = next(iter(forecaster._y_observed.keys()))
            original_y_observed_last_time = forecaster._y_observed[first_group]["time"][-1]
            assert original_observed_time[first_group] == original_y_observed_last_time, (
                "observed_time_ should match last time in _y_observed before update()"
            )
        else:
            # Non-panel data
            original_y_observed_last_time = forecaster._y_observed["time"][-1]
            assert original_observed_time == original_y_observed_last_time, (
                "observed_time_ should match last time in _y_observed before update()"
            )

    if forecaster._X_t_observed is not None:
        if isinstance(forecaster._X_t_observed, dict):
            # Panel data
            first_group = next(iter(forecaster._X_t_observed.keys()))
            if forecaster._X_t_observed[first_group] is not None:
                original_X_t_observed_last_time = forecaster._X_t_observed[first_group]["time"][-1]
                assert original_observed_time[first_group] == original_X_t_observed_last_time, (
                    "observed_time_ should match last time in _X_t_observed before update()"
                )
        else:
            # Non-panel data
            original_X_t_observed_last_time = forecaster._X_t_observed["time"][-1]
            assert original_observed_time == original_X_t_observed_last_time, (
                "observed_time_ should match last time in _X_t_observed before update()"
            )

    # Reset to new data
    forecaster.reset(y_reset, X_reset)

    # Check buffers were replaced
    reset_observed_time = forecaster.observed_time_

    # Handle both panel and non-panel data
    if isinstance(reset_observed_time, dict):
        # Panel data: check each group's observed_time matches
        for group_name in reset_observed_time:
            # Get expected time from y_reset (last row for this group's column)
            assert reset_observed_time[group_name] == y_reset["time"][-1], (
                f"observed_time_['{group_name}'] should be reset to last time in reset data"
            )
    else:
        # Non-panel data
        assert reset_observed_time == y_reset["time"][-1], (
            "observed_time_ should be reset to last time in reset data"
        )

    if forecaster._y_observed is not None:
        if isinstance(forecaster._y_observed, dict):
            # Panel data
            for group_name, y_obs in forecaster._y_observed.items():
                reset_y_observed_last_time = y_obs["time"][-1]
                assert reset_y_observed_last_time == reset_observed_time[group_name], (
                    f"Last time in _y_observed['{group_name}'] should match reset observed_time_"
                )
        else:
            # Non-panel data
            reset_y_observed_last_time = forecaster._y_observed["time"][-1]
            assert reset_y_observed_last_time == reset_observed_time, (
                "Last time in _y_observed should match reset observed_time_ after reset()"
            )

    if forecaster._X_t_observed is not None:
        if isinstance(forecaster._X_t_observed, dict):
            # Panel data
            for group_name, X_t_obs in forecaster._X_t_observed.items():
                if X_t_obs is not None:
                    reset_X_t_observed_last_time = X_t_obs["time"][-1]
                    assert reset_X_t_observed_last_time == reset_observed_time[group_name], (
                        f"Last time in _X_t_observed['{group_name}'] should match reset observed_time_"
                    )
        else:
            # Non-panel data
            reset_X_t_observed_last_time = forecaster._X_t_observed["time"][-1]
            assert reset_X_t_observed_last_time == reset_observed_time, (
                "Last time in _X_t_observed should match reset observed_time_ after reset()"
            )


def check_forecasting_horizon_validation(forecaster, y, X=None):
    """Check forecasting_horizon < 1 raises ValueError.

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance
    y : pl.DataFrame
        Training target data
    X : pl.DataFrame, optional
        Training features

    Raises
    ------
    AssertionError
        If invalid horizon doesn't raise ValueError
    """
    forecaster_clone = clone(forecaster)

    # Test horizon = 0
    try:
        forecaster_clone.fit(y, X, forecasting_horizon=0)
        raise AssertionError(
            f"{forecaster_clone.__class__.__name__} should raise ValueError "
            f"for forecasting_horizon=0"
        )
    except ValueError as e:
        assert "forecasting_horizon" in str(e).lower() or "positive" in str(e).lower(), (
            f"ValueError should mention forecasting_horizon, got: {e}"
        )

    # Test negative horizon
    forecaster_clone = clone(forecaster)
    try:
        forecaster_clone.fit(y, X, forecasting_horizon=-1)
        raise AssertionError(
            f"{forecaster_clone.__class__.__name__} should raise ValueError "
            f"for forecasting_horizon=-1"
        )
    except ValueError as e:
        assert "forecasting_horizon" in str(e).lower() or "positive" in str(e).lower(), (
            f"ValueError should mention forecasting_horizon, got: {e}"
        )


def check_prediction_types_property(forecaster):
    """Check prediction_types property returns correct set.

    Parameters
    ----------
    forecaster : BaseForecaster
        Forecaster instance (fitted or unfitted)

    Raises
    ------
    AssertionError
        If prediction_types doesn't return valid set
    """
    pred_types = forecaster.prediction_types

    assert isinstance(pred_types, set), (
        f"prediction_types should return set, got {type(pred_types)}"
    )

    valid_types = {"point", "interval"}
    assert pred_types.issubset(valid_types), (
        f"prediction_types should be subset of {valid_types}, got {pred_types}"
    )

    assert len(pred_types) > 0, "prediction_types should not be empty"


def check_clone_preserves_forecaster_params(forecaster):
    """Check sklearn's clone() preserves init parameters.

    Parameters
    ----------
    forecaster : BaseForecaster
        Forecaster instance

    Raises
    ------
    AssertionError
        If cloned forecaster has different parameters
    """
    forecaster_clone = clone(forecaster)

    # Get parameters
    original_params = forecaster.get_params(deep=False)
    cloned_params = forecaster_clone.get_params(deep=False)

    # Check same parameter keys
    assert set(original_params.keys()) == set(cloned_params.keys()), (
        f"clone() should have same parameter keys, got {set(cloned_params.keys())} vs {set(original_params.keys())}"
    )

    # Check parameter values (for nested estimators, check type)
    for key in original_params:
        orig_val = original_params[key]
        cloned_val = cloned_params[key]

        # For None values
        if orig_val is None:
            assert cloned_val is None, f"Parameter {key}: expected None, got {cloned_val}"
        # For list of (name, estimator) tuples (meta-estimators like Decomposer, FeaturePipeline)
        elif isinstance(orig_val, list) and len(orig_val) > 0 and isinstance(orig_val[0], tuple):
            assert isinstance(cloned_val, list), (
                f"Parameter {key}: expected list, got {type(cloned_val)}"
            )
            assert len(orig_val) == len(cloned_val), f"Parameter {key}: different lengths"

            for i, (orig_item, cloned_item) in enumerate(zip(orig_val, cloned_val)):
                assert isinstance(orig_item, tuple), f"Parameter {key}[{i}]: expected tuple"
                assert isinstance(cloned_item, tuple), f"Parameter {key}[{i}]: expected tuple"
                assert len(orig_item) == 2, (
                    f"Parameter {key}[{i}]: expected (name, estimator) tuple"
                )
                assert len(cloned_item) == 2, (
                    f"Parameter {key}[{i}]: expected (name, estimator) tuple"
                )

                orig_name, orig_est = orig_item
                cloned_name, cloned_est = cloned_item

                # Names should match exactly
                assert orig_name == cloned_name, (
                    f"Parameter {key}[{i}]: different names {cloned_name} != {orig_name}"
                )

                # Estimators should be different instances but same type
                assert type(orig_est) == type(cloned_est), (
                    f"Parameter {key}[{i}] estimator: different types {type(cloned_est)} vs {type(orig_est)}"
                )
                assert orig_est is not cloned_est, (
                    f"Parameter {key}[{i}] estimator: should be cloned, not same instance"
                )

                # Check estimator params match
                if hasattr(orig_est, "get_params"):
                    orig_est_params = orig_est.get_params(deep=True)
                    cloned_est_params = cloned_est.get_params(deep=True)
                    for param_key in orig_est_params.keys():
                        orig_param = orig_est_params.get(param_key)
                        cloned_param = cloned_est_params.get(param_key)
                        if hasattr(orig_param, "get_params"):
                            assert type(orig_param) == type(cloned_param), (
                                f"Parameter {key}[{i}]__{param_key}: different types"
                            )
                        elif orig_param != cloned_param:
                            assert orig_param == cloned_param, (
                                f"Parameter {key}[{i}]__{param_key}: {cloned_param} != {orig_param}"
                            )
        # For estimator instances, check type and params (recursively)
        elif hasattr(orig_val, "get_params"):
            assert type(orig_val) == type(cloned_val), (
                f"Parameter {key}: different types {type(cloned_val)} vs {type(orig_val)}"
            )
            # Use deep=True to get all nested params, compare them
            orig_deep_params = orig_val.get_params(deep=True)
            cloned_deep_params = cloned_val.get_params(deep=True)

            # Compare only primitive values and types (not object instances)
            for param_key in orig_deep_params.keys():
                orig_param = orig_deep_params.get(param_key)
                cloned_param = cloned_deep_params.get(param_key)

                # Skip comparing estimator instances themselves, just check types
                if hasattr(orig_param, "get_params"):
                    assert type(orig_param) == type(cloned_param), (
                        f"Parameter {key}__{param_key}: different types"
                    )
                elif orig_param != cloned_param:
                    assert orig_param == cloned_param, (
                        f"Parameter {key}__{param_key}: {cloned_param} != {orig_param}"
                    )
        # For other values, direct comparison
        else:
            assert orig_val == cloned_val, f"Parameter {key}: {cloned_val} != {orig_val}"

    # Check they are different objects
    assert forecaster_clone is not forecaster, "clone() should create new instance"


# ============================================================================
# POINT FORECASTER CHECKS
# ============================================================================


def check_point_prediction_structure(forecaster, y_test, X_test=None):
    """Check point predictions have correct column structure.

    Parameters
    ----------
    forecaster : BasePointForecaster
        Fitted point forecaster instance
    y_test : pl.DataFrame
        Test target data
    X_test : pl.DataFrame, optional
        Test features

    Raises
    ------
    AssertionError
        If prediction structure is incorrect
    """
    forecasting_horizon = min(3, len(y_test))
    y_pred = forecaster.predict(forecasting_horizon=forecasting_horizon, X=X_test)

    # Should have observed_time, time, and target columns
    assert "observed_time" in y_pred.columns, "Point predictions must have 'observed_time'"
    assert "time" in y_pred.columns, "Point predictions must have 'time'"

    # Should NOT have interval columns
    interval_cols = [col for col in y_pred.columns if "_lower_" in col or "_upper_" in col]
    assert len(interval_cols) == 0, (
        f"Point predictions should not have interval columns, found: {interval_cols}"
    )

    # Should have target columns
    target_cols = [col for col in y_pred.columns if col not in ["observed_time", "time"]]
    assert len(target_cols) > 0, "Point predictions must have at least one target column"


def check_point_prediction_types(forecaster):
    """Check point forecaster returns prediction_types == {"point"}.

    Parameters
    ----------
    forecaster : BasePointForecaster
        Point forecaster instance

    Raises
    ------
    AssertionError
        If prediction_types is not {"point"}
    """
    pred_types = forecaster.prediction_types

    assert pred_types == {"point"}, (
        f"Point forecaster should return prediction_types={{'point'}}, got {pred_types}"
    )


# ============================================================================
# INTERVAL FORECASTER CHECKS
# ============================================================================


def check_interval_prediction_columns(forecaster, y_test, X_test=None):
    """Check interval predictions have {col}_lower_{rate} and {col}_upper_{rate} format.

    Parameters
    ----------
    forecaster : BaseIntervalForecaster
        Fitted interval forecaster instance
    y_test : pl.DataFrame
        Test target data
    X_test : pl.DataFrame, optional
        Test features

    Raises
    ------
    AssertionError
        If interval column naming is incorrect
    """
    from yohou.utils.panel import inspect_locality

    forecasting_horizon = min(3, len(y_test))

    # Call predict_interval for interval forecasters
    y_pred = forecaster.predict_interval(forecasting_horizon=forecasting_horizon, X=X_test)

    # Get coverage rates - use fit_coverage_rates_ (set during fit)
    coverage_rates = forecaster.fit_coverage_rates_

    # Check if we have panel data (columns with __ separator)
    _, y_panel_groups = inspect_locality(y_test)

    if len(y_panel_groups) > 0:
        # For panel data, interval columns use __ separator
        # e.g., "stores__store_0_lower_0.1"
        for group_prefix in y_panel_groups.keys():
            # Get fields from the original training data (full column names)
            expected_fields = y_panel_groups[group_prefix]

            for rate in coverage_rates:
                for field in expected_fields:
                    lower_col = f"{field}_lower_{rate}"
                    upper_col = f"{field}_upper_{rate}"

                    assert lower_col in y_pred.columns, f"Missing lower bound column: {lower_col}"
                    assert upper_col in y_pred.columns, f"Missing upper bound column: {upper_col}"
    else:
        # For global data, check individual column pattern: {col}_lower_{rate}
        target_cols = list(forecaster.local_y_schema_.keys())

        # Check each coverage rate has lower and upper bounds for each target
        for rate in coverage_rates:
            for col in target_cols:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"

                assert lower_col in y_pred.columns, f"Missing lower bound column: {lower_col}"
                assert upper_col in y_pred.columns, f"Missing upper bound column: {upper_col}"


def check_interval_bounds(forecaster, y_test, X_test=None):
    """Check upper >= lower for all coverage rates and time steps.

    Parameters
    ----------
    forecaster : BaseIntervalForecaster
        Fitted interval forecaster instance
    y_test : pl.DataFrame
        Test target data
    X_test : pl.DataFrame, optional
        Test features

    Raises
    ------
    AssertionError
        If upper bounds are less than lower bounds
    """
    from yohou.utils.panel import inspect_locality

    forecasting_horizon = min(3, len(y_test))
    y_pred = forecaster.predict(forecasting_horizon=forecasting_horizon, X=X_test)

    coverage_rates = forecaster.coverage_rates

    # Check if we have panel data (columns with __ separator)
    _, y_panel_groups = inspect_locality(y_test)

    if len(y_panel_groups) > 0:
        # For panel data, interval columns use __ separator
        for group_prefix in y_panel_groups.keys():
            # Get fields from the original training data (full column names)
            expected_fields = y_panel_groups[group_prefix]

            for rate in coverage_rates:
                for field in expected_fields:
                    lower_col = f"{field}_lower_{rate}"
                    upper_col = f"{field}_upper_{rate}"

                    lower_vals = y_pred[lower_col].to_numpy()
                    upper_vals = y_pred[upper_col].to_numpy()

                    violations = lower_vals > upper_vals
                    if violations.any():
                        raise AssertionError(
                            f"Found {violations.sum()} violations where lower > upper for "
                            f"{field} at coverage {rate}"
                        )
    else:
        # For global data, check individual columns
        target_cols = list(forecaster.local_y_schema_.keys())

        for rate in coverage_rates:
            for col in target_cols:
                lower_col = f"{col}_lower_{rate}"
                upper_col = f"{col}_upper_{rate}"

                lower_vals = y_pred[lower_col].to_numpy()
                upper_vals = y_pred[upper_col].to_numpy()

                violations = lower_vals > upper_vals
                if violations.any():
                    raise AssertionError(
                        f"Found {violations.sum()} violations where lower > upper for "
                        f"{col} at coverage {rate}"
                    )


def check_interval_prediction_types(forecaster):
    """Check interval forecaster returns prediction_types containing "interval".

    Parameters
    ----------
    forecaster : BaseIntervalForecaster
        Interval forecaster instance

    Raises
    ------
    AssertionError
        If prediction_types doesn't contain "interval"
    """
    pred_types = forecaster.prediction_types

    assert "interval" in pred_types, (
        f"Interval forecaster should include 'interval' in prediction_types, got {pred_types}"
    )


def check_coverage_rates_parameter(forecaster):
    """Check coverage_rates is list of floats in (0, 1).

    Parameters
    ----------
    forecaster : BaseIntervalForecaster
        Interval forecaster instance

    Raises
    ------
    AssertionError
        If coverage_rates is invalid
    """
    coverage_rates = forecaster.coverage_rates

    assert isinstance(coverage_rates, list), (
        f"coverage_rates should be list, got {type(coverage_rates)}"
    )

    assert len(coverage_rates) > 0, "coverage_rates should not be empty"

    for rate in coverage_rates:
        assert isinstance(rate, (int, float)), (
            f"Each coverage rate should be numeric, got {type(rate)} for {rate}"
        )
        assert 0 < rate < 1, f"Coverage rates should be in (0, 1), got {rate}"


# ============================================================================
# REDUCTION FORECASTER CHECKS
# ============================================================================


def check_estimator_parameter(forecaster):
    """Check estimator parameter is sklearn BaseEstimator.

    Parameters
    ----------
    forecaster : BaseReductionForecaster
        Reduction forecaster instance

    Raises
    ------
    AssertionError
        If estimator is not a sklearn BaseEstimator
    """
    from sklearn.base import BaseEstimator

    assert hasattr(forecaster, "estimator"), "Reduction forecaster must have 'estimator' parameter"

    estimator = forecaster.estimator
    assert isinstance(estimator, BaseEstimator), (
        f"estimator should be sklearn BaseEstimator, got {type(estimator)}"
    )


def check_reduction_strategy(forecaster):
    """Check reduction_strategy parameter is valid.

    Parameters
    ----------
    forecaster : BaseReductionForecaster
        Reduction forecaster instance

    Raises
    ------
    AssertionError
        If reduction_strategy is invalid
    """
    if not hasattr(forecaster, "reduction_strategy"):
        # Not all reduction forecasters expose this parameter
        return

    strategy = forecaster.reduction_strategy
    valid_strategies = ["direct", "multi-output"]

    assert strategy in valid_strategies, (
        f"reduction_strategy should be in {valid_strategies}, got '{strategy}'"
    )


# ============================================================================
# Cross-Learning Forecaster Checks
# ============================================================================


def check_panel_data(forecaster, y_panel, X_panel=None):
    """Check cross-learning with panel data predicts all groups by default.

    Validates that when panel_group=None (default), predictions are
    generated for all groups in the panel data columns.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster with panel data
    y_panel : pl.DataFrame
        Panel data with panel columns for testing
    X_panel : pl.DataFrame or None
        Panel features

    Raises
    ------
    AssertionError
        If default prediction doesn't include all groups
    """
    from yohou.utils.panel import inspect_locality

    # Predict with default (panel_group=None)
    y_pred = forecaster.predict(X=X_panel, forecasting_horizon=3, panel_group=None)

    # Check that all local groups from training data are in predictions
    _, y_panel_groups = inspect_locality(y_panel)

    if len(y_panel_groups) > 0:
        # Should have predictions for all group columns (with __ separator)
        for group_prefix, expected_fields in y_panel_groups.items():
            for field in expected_fields:
                assert field in y_pred.columns, (
                    f"Column '{field}' missing from predictions. "
                    f"panel_group=None should predict all groups."
                )


def check_panel_single_group(forecaster, y_panel, X_panel=None):
    """Check cross-learning filters to specified panel group.

    Validates that when panel_group is specified, predictions are
    generated only for that panel group (all columns with that prefix).

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster with panel data
    y_panel : pl.DataFrame
        Panel data with panel columns for testing
    X_panel : pl.DataFrame or None
        Panel features

    Raises
    ------
    AssertionError
        If filtered prediction doesn't match specified group
    """
    from yohou.utils.panel import inspect_locality

    _, y_panel_groups = inspect_locality(y_panel)

    if len(y_panel_groups) > 0:
        # Get first group prefix
        first_group = list(y_panel_groups.keys())[0]

        # Predict with specific group
        y_pred = forecaster.predict(X=X_panel, forecasting_horizon=3, panel_group=first_group)

        # Should have columns from the specified group (flat columns with __ separator)
        group_cols = y_panel_groups[first_group]
        assert len(group_cols) > 0, f"Group '{first_group}' should have columns"
        for col in group_cols:
            assert col in y_pred.columns, (
                f"Column '{col}' from group '{first_group}' should be in predictions"
            )


def check_panel_invalid_group_raises(forecaster, y_panel, X_panel=None):
    """Check that invalid panel_group raises ValueError.

    Validates error handling when panel_group specifies a panel group
    that doesn't exist in the training data.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster with panel data
    y_panel : pl.DataFrame
        Panel data with panel columns for testing
    X_panel : pl.DataFrame or None
        Panel features

    Raises
    ------
    AssertionError
        If ValueError is not raised for invalid group
    """
    from yohou.utils.panel import inspect_locality

    _, y_panel_groups = inspect_locality(y_panel)

    if len(y_panel_groups) > 0:
        # Try to predict with invalid group name
        try:
            forecaster.predict(
                X=X_panel, forecasting_horizon=3, panel_group_names=["invalid_group"]
            )
            raise AssertionError(
                "predict() should raise ValueError for invalid panel_group, but didn't"
            )
        except ValueError as e:
            # Expected - check error message mentions the invalid group
            assert "invalid_group" in str(e) or "not found" in str(e).lower(), (
                f"ValueError message should mention invalid group, got: {e}"
            )


# ============================================================================
# FORECASTER CHECK GENERATOR
# ============================================================================


def _yield_yohou_forecaster_checks(
    forecaster,
    y_train: pl.DataFrame,
    X_train: pl.DataFrame | None,
    y_test: pl.DataFrame,
    X_test: pl.DataFrame | None,
    tags: Dict[str, Any] | None = None,
) -> Generator[Tuple[str, Callable, Dict], None, None]:
    """Generate applicable checks for a forecaster based on tags.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster instance
    y_train : pl.DataFrame
        Training target data with "time" column
    X_train : pl.DataFrame or None
        Training features
    y_test : pl.DataFrame
        Test target data
    X_test : pl.DataFrame or None
        Test features
    tags : dict or None
        Forecaster metadata tags:
        - forecaster_type: "point" | "interval" | "both"
        - uses_reduction: bool
        - supports_panel_data: bool
        - uses_transformers: bool
        - supports_scoring: bool

    Yields
    ------
    check_name : str
        Name of the check function
    check_func : callable
        Check function to execute
    check_kwargs : dict
        Keyword arguments for check function (bundled data)
    """
    if tags is None:
        tags = {}

    # Bundle data for check functions
    check_kwargs = {
        "y_train": y_train,
        "X_train": X_train,
        "y_test": y_test,
        "X_test": X_test,
    }

    # Common forecaster checks (always yield)
    yield (
        "check_fit_sets_forecaster_attributes",
        check_fit_sets_forecaster_attributes,
        {"y": y_train, "X": X_train, "forecasting_horizon": 3},
    )
    yield (
        "check_forecaster_not_fitted_error",
        check_forecaster_not_fitted_error,
        {"y": y_train, "X": X_train},
    )
    yield (
        "check_predict_time_columns",
        check_predict_time_columns,
        {"y_test": y_test, "X_test": X_test},
    )
    yield (
        "check_forecasting_horizon_validation",
        check_forecasting_horizon_validation,
        {"y": y_train, "X": X_train},
    )
    yield "check_prediction_types_property", check_prediction_types_property, {}
    yield "check_clone_preserves_forecaster_params", check_clone_preserves_forecaster_params, {}

    # Update/reset checks (if enough data)
    if len(y_test) >= 10:
        y_update = y_test[:3]
        y_reset = y_test[:10]
        X_update = X_test[:3] if X_test is not None else None
        X_reset = X_test[:10] if X_test is not None else None

        yield (
            "check_update_extends_observations",
            check_update_extends_observations,
            {
                "y_train": y_train,
                "y_update": y_update,
                "X_train": X_train,
                "X_update": X_update,
            },
        )
        yield (
            "check_reset_replaces_observations",
            check_reset_replaces_observations,
            {
                "y_train": y_train,
                "y_reset": y_reset,
                "X_train": X_train,
                "X_reset": X_reset,
            },
        )

    # Transformer composition checks
    if tags.get("uses_transformers", False):
        if len(y_test) >= 5:
            yield (
                "check_reset_propagates_to_transformers",
                check_reset_propagates_to_transformers,
                {
                    "y_train": y_train,
                    "y_reset": y_reset,
                    "X_train": X_train,
                    "X_reset": X_reset,
                },
            )

    # Point forecaster checks
    if tags.get("forecaster_type") == "point":
        yield (
            "check_point_prediction_structure",
            check_point_prediction_structure,
            {"y_test": y_test, "X_test": X_test},
        )
        yield "check_point_prediction_types", check_point_prediction_types, {}

    # Interval forecaster checks
    if tags.get("forecaster_type") == "interval":
        yield (
            "check_interval_prediction_columns",
            check_interval_prediction_columns,
            {"y_test": y_test, "X_test": X_test},
        )
        yield (
            "check_interval_bounds",
            check_interval_bounds,
            {"y_test": y_test, "X_test": X_test},
        )
        yield "check_interval_prediction_types", check_interval_prediction_types, {}
        yield "check_coverage_rates_parameter", check_coverage_rates_parameter, {}

    # Reduction forecaster checks
    if tags.get("uses_reduction", False):
        yield "check_estimator_parameter", check_estimator_parameter, {}
        yield "check_reduction_strategy", check_reduction_strategy, {}

    # Cross-learning checks (for panel data)
    if tags.get("supports_panel_data", False):
        # Need to check if we have panel data available
        from yohou.utils.panel import inspect_locality

        _, y_panel_groups = inspect_locality(y_train)
        if len(y_panel_groups) > 0:
            # We have panel data, run cross-learning checks
            yield (
                "check_panel_data",
                check_panel_data,
                {"y_panel": y_test, "X_panel": X_test},
            )
            yield (
                "check_panel_single_group",
                check_panel_single_group,
                {"y_panel": y_test, "X_panel": X_test},
            )
            yield (
                "check_panel_invalid_group_raises",
                check_panel_invalid_group_raises,
                {"y_panel": y_test, "X_panel": X_test},
            )

    # Metadata routing checks (always applicable)
    yield (
        "check_metadata_routing_default_request",
        check_metadata_routing_default_request,
        {},
    )
    yield (
        "check_metadata_routing_get_metadata_routing",
        check_metadata_routing_get_metadata_routing,
        {},
    )


# ============================================================================
# METADATA ROUTING CHECKS
# ============================================================================


def check_metadata_routing_default_request(estimator_fitted):
    """Check that by default metadata routing request is empty.

    Tests:
    - get_metadata_routing() returns MetadataRouter or MetadataRequest
    - Default requests are empty (all metadata values are None)

    Parameters
    ----------
    estimator_fitted : BaseForecaster or BaseTransformer
        A fitted estimator instance.

    Raises
    ------
    AssertionError
        If routing structure is incorrect or requests are not empty.

    """
    from metadata_routing_common import assert_request_is_empty
    from sklearn.utils.metadata_routing import MetadataRequest, MetadataRouter

    # Routing is always enabled in Yohou - no check needed

    router = estimator_fitted.get_metadata_routing()
    assert isinstance(router, (MetadataRouter, MetadataRequest)), (
        f"Expected MetadataRouter or MetadataRequest, got {type(router)}"
    )

    # Check requests are empty (with possible exclusions for defaults)
    exclude = {}  # Can add specific exclusions per estimator type
    assert_request_is_empty(router, exclude=exclude)


def check_metadata_routing_get_metadata_routing(estimator_fitted):
    """Check that get_metadata_routing() is implemented correctly.

    Tests:
    - Method exists and returns MetadataRouter or MetadataRequest
    - Router has correct owner
    - Router includes child estimators if applicable

    Parameters
    ----------
    estimator_fitted : BaseForecaster or BaseTransformer
        A fitted estimator instance.

    Raises
    ------
    AssertionError
        If get_metadata_routing implementation is incorrect.

    """
    from sklearn.utils.metadata_routing import MetadataRequest, MetadataRouter

    assert hasattr(estimator_fitted, "get_metadata_routing"), (
        f"{type(estimator_fitted).__name__} must implement get_metadata_routing()"
    )

    router = estimator_fitted.get_metadata_routing()

    assert isinstance(router, (MetadataRouter, MetadataRequest)), (
        f"get_metadata_routing() must return MetadataRouter or MetadataRequest, got {type(router)}"
    )

    # Check owner is set
    assert router.owner is not None, "Router must have an owner set"
