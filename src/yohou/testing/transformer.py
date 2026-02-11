"""Check functions for yohou transformers.

This module provides systematic validation functions for testing BaseTransformer
implementations. All check functions raise AssertionError on failure.

Organized into three categories:
1. Core Yohou Checks - Time series-specific validation (11 functions)
2. Enhanced sklearn Checks - Adapted from sklearn patterns (7 functions)
3. Tag System Checks - Validate tag system correctness (3 functions)
"""

try:
    import polars as pl
    import polars.selectors as cs
    from polars.testing import assert_frame_equal
except ImportError as e:
    raise ImportError("polars.testing is required for yohou.testing module. Install with: uv sync --group tests") from e

from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

# ============================================================================
# CORE YOHOU CHECKS
# ============================================================================


def check_fit_sets_attributes(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
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
    assert hasattr(transformer_clone, "feature_names_in_"), "fit() must set feature_names_in_ attribute"
    assert hasattr(transformer_clone, "n_features_in_"), "fit() must set n_features_in_ attribute"

    # Check yohou-required attributes
    assert hasattr(transformer_clone, "_observation_horizon"), "fit() must set _observation_horizon attribute"

    # Validate values
    expected_features = [col for col in X.columns if col != "time"]
    assert list(transformer_clone.feature_names_in_) == expected_features, (
        f"feature_names_in_ mismatch: {transformer_clone.feature_names_in_} vs {expected_features}"
    )

    assert transformer_clone.n_features_in_ == len(expected_features), (
        f"n_features_in_ should be {len(expected_features)}, got {transformer_clone.n_features_in_}"
    )


def check_observation_horizon_not_fitted(transformer, X: pl.DataFrame) -> None:
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


def check_observation_horizon_after_fit(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
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


def check_reset_updates_memory(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
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


def check_update_concatenates_memory(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
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

    # Split data: fit on first 80%, update with next 10 rows
    X_train, X_temp = train_test_split(X, test_size=0.2, shuffle=False)
    X_update = X_temp.head(10)  # Take first 10 rows from remaining 20%

    transformer_clone.fit(X_train, y)

    horizon = transformer_clone.observation_horizon
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


def check_update_transform_equivalence(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
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
    X_first, X_second = train_test_split(X, test_size=0.5, shuffle=False)

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


def check_insufficient_data_raises(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
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


def check_transform_output_structure(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
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
    assert isinstance(X_trans, pl.DataFrame), f"transform() must return pl.DataFrame, got {type(X_trans)}"

    # Check time column exists
    assert "time" in X_trans.columns, "transform() output must contain 'time' column"

    # Check time column is datetime
    assert X_trans["time"].dtype in [pl.Datetime, pl.Date], (
        f"'time' column must be datetime type, got {X_trans['time'].dtype}"
    )

    # Check output has at least one feature column
    feature_cols = [col for col in X_trans.columns if col != "time"]
    assert len(feature_cols) > 0, "transform() output must have at least one feature column besides 'time'"


def check_feature_names_out_match(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
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


def check_inverse_transform_identity(
    transformer,
    X: pl.DataFrame,
    y: pl.DataFrame | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
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
    atol : float
        Absolute tolerance for numerical comparison
    rtol : float
        Relative tolerance for numerical comparison

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


def check_panel_data_support(transformer, X_panel: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
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


# ============================================================================
# ENHANCED SKLEARN CHECKS
# ============================================================================


def check_transformers_unfitted_stateless(transformer, X: pl.DataFrame) -> None:
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


def check_transformer_preserve_dtypes(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
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
    input_dtypes = {col: dtype for col, dtype in zip(X_numeric.columns, X_numeric.dtypes, strict=False)}

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
                            f"inverse_transform changed dtype of '{col}': {input_dtypes[col]} -> {X_inv[col].dtype}"
                        )
        except NotImplementedError:
            # Not invertible - skip inverse check
            pass


def check_fit_idempotent(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
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


def check_inverse_transform_round_trip(
    transformer,
    X: pl.DataFrame,
    y: pl.DataFrame | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
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
    atol : float
        Absolute tolerance for numerical comparison
    rtol : float
        Relative tolerance for numerical comparison

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


def check_fit_transform_equivalence(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
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


def check_memory_bounded(
    transformer,
    X_train: pl.DataFrame,
    X_test: pl.DataFrame,
    y: pl.DataFrame | None = None,
    n_updates: int = 5,
) -> None:
    """Check memory doesn't grow unbounded with sequential updates.

    Important for production time series applications with continuous
    data streams. Memory should stabilize at observation_horizon size.

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted transformer
    X_train : pl.DataFrame
        Training data (used for fit)
    X_test : pl.DataFrame
        Test data (used for updates - non-overlapping with training)
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
    transformer_clone.fit(X_train, y)

    horizon = transformer_clone.observation_horizon
    max_memory_factor = 2.0
    expected_max_rows = int(horizon * max_memory_factor)

    # Create update chunks from test data (non-overlapping with training)
    chunk_size = max(1, len(X_test) // n_updates)

    for i in range(n_updates):
        start_idx = i * chunk_size
        if start_idx >= len(X_test):
            break

        X_chunk = X_test.slice(start_idx, chunk_size)
        transformer_clone.update(X_chunk)

        actual_rows = len(transformer_clone._X_observed)
        assert actual_rows <= expected_max_rows, (
            f"Memory grew beyond bounds after {i + 1} updates: "
            f"{actual_rows} rows > {expected_max_rows} (horizon={horizon})"
        )


# ============================================================================
# TAG SYSTEM CHECKS
# ============================================================================


def check_tags_accessible_before_fit(transformer, X: pl.DataFrame | None = None) -> None:
    """Check __sklearn_tags__() is accessible before fit().

    Tags should be static class capabilities, not fitted state.
    They must be accessible before calling fit().

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted transformer instance
    X : pl.DataFrame, optional
        Not used, included for signature consistency

    Raises
    ------
    AssertionError
        If __sklearn_tags__() raises error or is not callable

    """
    transformer_clone = clone(transformer)

    assert hasattr(transformer_clone, "__sklearn_tags__"), (
        f"{transformer_clone.__class__.__name__} must implement __sklearn_tags__() method"
    )

    try:
        tags = transformer_clone.__sklearn_tags__()
    except Exception as e:
        raise AssertionError(
            f"{transformer_clone.__class__.__name__}.__sklearn_tags__() raised {type(e).__name__}: {e}"
        ) from e

    # Validate tag structure
    assert hasattr(tags, "transformer_tags"), "Tags must have transformer_tags attribute"
    assert hasattr(tags, "input_tags"), "Tags must have input_tags attribute"


def check_tags_static_after_fit(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
    """Check tags remain static (don't change) after fit().

    Tags represent capabilities, not fitted state. They should have
    the same values before and after fit().

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted transformer instance
    X : pl.DataFrame
        Training data
    y : pl.DataFrame, optional
        Target data

    Raises
    ------
    AssertionError
        If tags change after fit()

    """
    transformer_clone = clone(transformer)

    # Get tags before fit
    tags_before = transformer_clone.__sklearn_tags__()
    stateful_before = tags_before.transformer_tags.stateful if tags_before.transformer_tags else None
    invertible_before = tags_before.transformer_tags.invertible if tags_before.transformer_tags else None
    min_value_before = tags_before.input_tags.min_value if tags_before.input_tags else None

    # Fit the transformer
    transformer_clone.fit(X, y)

    # Get tags after fit
    tags_after = transformer_clone.__sklearn_tags__()
    stateful_after = tags_after.transformer_tags.stateful if tags_after.transformer_tags else None
    invertible_after = tags_after.transformer_tags.invertible if tags_after.transformer_tags else None
    min_value_after = tags_after.input_tags.min_value if tags_after.input_tags else None

    # Verify tags didn't change
    assert stateful_before == stateful_after, f"stateful tag changed after fit: {stateful_before} -> {stateful_after}"
    assert invertible_before == invertible_after, (
        f"invertible tag changed after fit: {invertible_before} -> {invertible_after}"
    )
    assert min_value_before == min_value_after, (
        f"min_value tag changed after fit: {min_value_before} -> {min_value_after}"
    )


def check_tags_match_capabilities(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
    """Check tags accurately reflect transformer capabilities.

    Validates that tag values match actual transformer behavior:
    - stateful tag matches presence of _observation_horizon attribute
    - invertible tag matches presence of inverse_transform method

    Parameters
    ----------
    transformer : BaseTransformer
        Fitted transformer instance
    X : pl.DataFrame
        Training data (not used, for consistency)
    y : pl.DataFrame, optional
        Target data (not used, for consistency)

    Raises
    ------
    AssertionError
        If tags don't match actual capabilities

    """
    tags = transformer.__sklearn_tags__()

    # Check stateful tag
    if tags.transformer_tags:
        has_observation_horizon = hasattr(transformer, "_observation_horizon")
        is_stateful = tags.transformer_tags.stateful

        # If transformer has observation_horizon attribute, it should be tagged as stateful
        if has_observation_horizon and not is_stateful:
            raise AssertionError(
                f"{transformer.__class__.__name__} has _observation_horizon attribute but stateful tag is False"
            )

        # Check invertible tag
        has_inverse_transform = hasattr(transformer, "inverse_transform") and callable(transformer.inverse_transform)
        is_invertible = tags.transformer_tags.invertible

        if has_inverse_transform != is_invertible:
            raise AssertionError(
                f"{transformer.__class__.__name__} inverse_transform existence ({has_inverse_transform}) "
                f"doesn't match invertible tag ({is_invertible})"
            )


def check_transformer_methods_call_check_is_fitted(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
    """Check all transformer methods (except fit) raise NotFittedError when unfitted.

    Validates that transform(), reset(), update(), update_transform(), and
    inverse_transform() methods all check fitted state and raise NotFittedError
    before operating on an unfitted transformer.

    Parameters
    ----------
    transformer : BaseTransformer
        Unfitted transformer instance
    X : pl.DataFrame
        Training/test data with "time" column (should have at least 100 rows for slicing)
    y : pl.DataFrame, optional
        Target data for supervised transformers

    Raises
    ------
    AssertionError
        If any method fails to raise NotFittedError when called on unfitted transformer

    """
    transformer_clone = clone(transformer)

    # Test that transform() raises NotFittedError when unfitted
    try:
        transformer_clone.transform(X)
        raise AssertionError(
            f"{transformer_clone.__class__.__name__}.transform() must raise NotFittedError when called on unfitted transformer"
        )
    except NotFittedError:
        pass  # Expected

    # Test that update() raises NotFittedError when unfitted
    try:
        transformer_clone.update(X)
        raise AssertionError(
            f"{transformer_clone.__class__.__name__}.update() must raise NotFittedError when called on unfitted transformer"
        )
    except NotFittedError:
        pass  # Expected

    # Test that reset() raises NotFittedError when unfitted
    try:
        transformer_clone.reset(X)
        raise AssertionError(
            f"{transformer_clone.__class__.__name__}.reset() must raise NotFittedError when called on unfitted transformer"
        )
    except NotFittedError:
        pass  # Expected

    # Test that update_transform() raises NotFittedError when unfitted
    try:
        transformer_clone.update_transform(X)
        raise AssertionError(
            f"{transformer_clone.__class__.__name__}.update_transform() must raise NotFittedError when called on unfitted transformer"
        )
    except NotFittedError:
        pass  # Expected

    # Test inverse_transform() if implemented
    if hasattr(transformer_clone, "inverse_transform"):
        # Need to fit the transformer first to be able to call transform and get X_t
        transformer_clone.fit(X, y)
        X_t = transformer_clone.transform(X)

        # Create a fresh unfitted clone
        transformer_clone2 = clone(transformer)
        try:
            transformer_clone2.inverse_transform(X_t, X_p=X)
            raise AssertionError(
                f"{transformer.__class__.__name__}.inverse_transform() must raise NotFittedError when called on unfitted transformer"
            )
        except NotFittedError:
            pass  # Expected
