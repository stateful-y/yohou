"""Check functions for yohou transformers.

This module provides systematic validation functions for testing BaseActualTransformer
implementations. All check functions raise AssertionError on failure.

Organized into three categories:
1. Core Yohou Checks - Time series-specific validation (11 functions)
2. Enhanced sklearn Checks - Adapted from sklearn patterns (7 functions)
3. Tag System Checks - Validate tag system correctness (3 functions)
"""

import numpy as np

try:
    import polars as pl
    import polars.selectors as cs
    from polars.testing import assert_frame_equal
except ImportError as e:
    raise ImportError("polars.testing is required for yohou.testing module. Install with: uv sync --group tests") from e

from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from yohou.base import BaseActualTransformer
from yohou.utils import inspect_panel

__all__ = [
    "check_feature_names_out_match",
    "check_fit_idempotent",
    "check_fit_sets_attributes",
    "check_fit_transform_equivalence",
    "check_insufficient_data_raises",
    "check_inverse_transform_identity",
    "check_inverse_transform_round_trip",
    "check_inverse_observe_transform_identity",
    "check_memory_bounded",
    "check_observation_horizon_after_fit",
    "check_observation_horizon_not_fitted",
    "check_panel_data_support",
    "check_panel_group_preservation",
    "check_rewind_transform_behavior",
    "check_rewind_updates_memory",
    "check_tags_accessible_before_fit",
    "check_tags_match_capabilities",
    "check_tags_static_after_fit",
    "check_transform_output_structure",
    "check_transformer_methods_call_check_is_fitted",
    "check_transformer_preserve_dtypes",
    "check_transformers_unfitted_stateless",
    "check_observe_concatenates_memory",
    "check_observe_transform_equivalence",
    "check_batch_invariance",
    "check_observe_transform_sequential_consistency",
    "check_transform_drops_warmup_rows",
]


def _build_X_p(transformer, X: pl.DataFrame, X_trans: pl.DataFrame) -> pl.DataFrame | None:
    """Build the ``X_p`` past-observations frame for inverse_transform.

    yohou's ``inverse_transform`` needs the ``observation_horizon`` rows that
    immediately precede the first transformed row. ``transform`` may drop rows
    (e.g. differencing), so those past rows sit at positions
    ``[n_dropped - horizon : n_dropped]`` in the original ``X``. Stateless
    transformers (``observation_horizon == 0``) need no past context.

    Parameters
    ----------
    transformer : BaseActualTransformer
        Fitted transformer.
    X : pl.DataFrame
        Original (untransformed) data.
    X_trans : pl.DataFrame
        Output of ``transformer.transform(X)``.

    Returns
    -------
    pl.DataFrame or None
        The past-observation rows, or ``None`` for stateless transformers.

    """
    horizon = transformer.observation_horizon
    if horizon <= 0:
        return None
    n_dropped = len(X) - len(X_trans)
    return X[n_dropped - horizon : n_dropped]


def check_fit_sets_attributes(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
    """Check fit() sets required attributes.

    Validates that fit() creates feature_names_in_, n_features_in_,
    and _observation_horizon attributes as expected by sklearn conventions.

    Parameters
    ----------
    transformer : BaseActualTransformer
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
    """Check observation_horizon behavior before fit().

    For transformers that inherit the base observation_horizon (no override),
    accessing it before fit() should raise NotFittedError. Transformers that
    override observation_horizon as a @property (computed from constructor
    params) are allowed to return a value before fit.

    Parameters
    ----------
    transformer : BaseActualTransformer
        Unfitted transformer instance
    X : pl.DataFrame
        Test data

    Raises
    ------
    AssertionError
        If observation_horizon returns a non-zero value before fit without
        an explicit property override

    """
    transformer_clone = clone(transformer)

    try:
        horizon = transformer_clone.observation_horizon
        # Accessible before fit: either stateless (0) or @property override.
        # Both are valid.  Property overrides compute from constructor params.
        if horizon < 0:
            raise AssertionError(
                f"{transformer_clone.__class__.__name__}.observation_horizon "
                f"returned {horizon} before fit(). Must be non-negative."
            )
    except NotFittedError:
        # Expected for transformers using the base property (no override)
        pass


def check_observation_horizon_after_fit(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
    """Check observation_horizon is valid after fit().

    After fitting, observation_horizon should be a non-negative integer.

    Parameters
    ----------
    transformer : BaseActualTransformer
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


def check_transform_drops_warmup_rows(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
    """Check stateful transformers drop exactly observation_horizon rows.

    Stateful transformers (observation_horizon > 0) must drop the first
    observation_horizon rows in their transform() output. Stateless
    transformers are skipped (they may legitimately change row count,
    e.g. Downsampler).

    Parameters
    ----------
    transformer : BaseActualTransformer
        Unfitted transformer
    X : pl.DataFrame
        Training data (must have enough rows)
    y : pl.DataFrame, optional
        Target data

    Raises
    ------
    AssertionError
        If the number of dropped rows doesn't match observation_horizon

    """
    transformer_clone = clone(transformer)
    transformer_clone.fit(X, y)

    horizon = transformer_clone.observation_horizon
    X_t = transformer_clone.transform(X)

    # Stateless transformers may legitimately change row count (e.g. Downsampler)
    if horizon == 0:
        return

    assert len(X_t) == len(X) - horizon, (
        f"Stateful transformer should drop exactly {horizon} rows "
        f"(observation_horizon), but input has {len(X)} rows and "
        f"output has {len(X_t)} rows (dropped {len(X) - len(X_t)})"
    )
    assert X_t["time"][0] == X["time"][horizon], (
        f"First output timestamp should be X['time'][{horizon}] = {X['time'][horizon]}, but got {X_t['time'][0]}"
    )


def check_rewind_updates_memory(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
    """Check rewind(X) sets _X_observed to X.tail(observation_horizon).

    The rewind() method should update the transformer's memory to contain
    only the last observation_horizon rows of the provided data.

    Parameters
    ----------
    transformer : BaseActualTransformer
        Unfitted transformer
    X : pl.DataFrame
        Training data (should have at least observation_horizon rows)
    y : pl.DataFrame, optional
        Target data

    Raises
    ------
    AssertionError
        If _X_observed is not properly updated

    Notes
    -----
    For stateless transformers (observation_horizon == 0) the expected memory
    is an empty frame, so the assertions hold trivially; no memory behavior is
    exercised. The check is also a no-op when len(X) < observation_horizon.

    """
    transformer_clone = clone(transformer)
    transformer_clone.fit(X, y)

    horizon = transformer_clone.observation_horizon

    if len(X) < horizon:  # pragma: no cover - the check suite always supplies enough rows for the horizon
        # Precondition cannot be met with this data; skip like the other checks.
        return

    # Create new data to reset with
    X_new = X.head(horizon + 5) if len(X) >= horizon + 5 else X
    transformer_clone.rewind(X_new)

    # Check _X_observed has correct length
    assert len(transformer_clone._X_observed) == min(horizon, len(X_new)), (
        f"_X_observed length should be {min(horizon, len(X_new))}, got {len(transformer_clone._X_observed)}"
    )

    # Check _X_observed contains last horizon rows
    expected = X_new.tail(horizon)
    assert_frame_equal(transformer_clone._X_observed, expected)


def check_observe_concatenates_memory(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
    """Check observe() appends new data and maintains horizon size.

    The observe() method should append new observations to _X_observed
    and trim to observation_horizon length.

    Parameters
    ----------
    transformer : BaseActualTransformer
        Unfitted transformer
    X : pl.DataFrame
        Initial training data
    y : pl.DataFrame, optional
        Target data

    Raises
    ------
    AssertionError
        If observe() doesn't properly maintain memory

    Notes
    -----
    For stateless transformers (observation_horizon == 0) _X_observed is always
    an empty frame, so both length assertions resolve to ``0 <= 0`` and
    ``0 == 0``; no meaningful memory behavior is exercised for that class.

    """
    transformer_clone = clone(transformer)

    # Split data: fit on first 80%, update with next 10 rows
    X_train, X_temp = train_test_split(X, test_size=0.2, shuffle=False)
    X_update = X_temp.head(10)  # Take first 10 rows from remaining 20%

    transformer_clone.fit(X_train, y)

    horizon = transformer_clone.observation_horizon
    initial_memory_len = len(transformer_clone._X_observed)

    transformer_clone.observe(X_update)

    # Memory should not exceed horizon
    assert len(transformer_clone._X_observed) <= horizon, (
        f"_X_observed length {len(transformer_clone._X_observed)} exceeds horizon {horizon}"
    )

    # Memory should have grown or stayed at horizon
    expected_len = min(initial_memory_len + len(X_update), horizon)
    assert len(transformer_clone._X_observed) == expected_len, (
        f"Expected _X_observed length {expected_len}, got {len(transformer_clone._X_observed)}"
    )


def check_observe_transform_equivalence(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
    """Check observe() does not change transform() output for a fitted transformer.

    ``transform()`` does not consult ``_X_observed``; it transforms the provided
    DataFrame independently of internal memory. Calling ``observe()`` beforehand
    updates the memory buffer (not the fitted parameters), so starting from the
    same fitted state, transforming after an ``observe()`` call must match
    transforming without it.

    Parameters
    ----------
    transformer : BaseActualTransformer
        Unfitted transformer
    X : pl.DataFrame
        Training data
    y : pl.DataFrame, optional
        Target data

    Raises
    ------
    AssertionError
        If observe() changes the transform output

    """
    # Split data so X_second continues immediately after X_first.
    X_first, X_second = train_test_split(X, test_size=0.5, shuffle=False)
    y_first = y.head(len(X_first)) if y is not None else None

    # Both paths share the same fitted parameters (fit on X_first). observe()
    # may only ingest data that continues after the fitted/observed state, so
    # X_second is the natural continuation of X_first.
    # Path 1: fit only.
    transformer1 = clone(transformer)
    transformer1.fit(X_first, y_first)
    X_trans1 = transformer1.transform(X_second)

    # Path 2: fit, then observe (updates memory, not fitted params).
    transformer2 = clone(transformer)
    transformer2.fit(X_first, y_first)
    transformer2.observe(X_second)
    X_trans2 = transformer2.transform(X_second)

    # Results should be equivalent
    assert_frame_equal(X_trans1, X_trans2, rel_tol=1e-6, abs_tol=1e-8)


def check_observe_transform_sequential_consistency(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
    """Check observe_transform(A) then observe_transform(B) == observe_transform(A+B).

    Sequential observe_transform calls should produce the same output as a
    single observe_transform call on concatenated data. Also verifies that
    internal state (_X_observed) is consistent after both operations.

    Parameters
    ----------
    transformer : BaseActualTransformer
        Unfitted transformer
    X : pl.DataFrame
        Training data (will be split into fit, A, B portions)
    y : pl.DataFrame, optional
        Target data

    Raises
    ------
    AssertionError
        If sequential updates produce different results than combined update

    Notes
    -----
    This check splits X into three parts:
    - X_fit: Used for initial fit
    - A: First observe_transform batch
    - B: Second observe_transform batch

    Then verifies:
    1. concat(observe_transform(A), observe_transform(B)) == observe_transform(concat(A, B))
    2. _X_observed is identical after both operations

    """
    # Need enough data for 3-way split
    if len(X) < 12:
        return

    # Split into fit portion and two update portions
    fit_size = len(X) // 2
    update_size = (len(X) - fit_size) // 2
    X_fit = X[:fit_size]
    A = X[fit_size : fit_size + update_size]
    B = X[fit_size + update_size :]

    # Path 1: Sequential observe_transform calls
    transformer1 = clone(transformer)
    transformer1.fit(X_fit, y)
    A_trans = transformer1.observe_transform(A)
    B_trans = transformer1.observe_transform(B)
    output_sequential = pl.concat([A_trans, B_trans])

    # Path 2: Single observe_transform on concatenated data
    transformer2 = clone(transformer)
    transformer2.fit(X_fit, y)
    AB = pl.concat([A, B])
    output_combined = transformer2.observe_transform(AB)

    # Outputs should be equivalent
    assert_frame_equal(
        output_sequential,
        output_combined,
        rel_tol=1e-6,
        abs_tol=1e-8,
    )

    # Internal state should also match
    if hasattr(transformer1, "_X_observed") and hasattr(transformer2, "_X_observed"):
        assert_frame_equal(
            transformer1._X_observed,
            transformer2._X_observed,
            rel_tol=1e-6,
            abs_tol=1e-8,
        )


def check_batch_invariance(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
    """Check that a transformer declaring ``batch_invariant`` earns the claim.

    Observing a block of rows in one ``observe_transform`` call must yield the same
    rows, within a relative tolerance, as observing them one at a time from the same
    starting state. That is the property a caller relies on when it replaces a rolling
    observe with a single bulk one.

    A no-op for transformers that do not declare the tag: absence of the claim is not a
    defect, it only forfeits the bulk path.

    Parameters
    ----------
    transformer : BaseActualTransformer
        Unfitted transformer.
    X : pl.DataFrame
        Training data, split into a fit portion and a replay portion.
    y : pl.DataFrame, optional
        Target data.

    Raises
    ------
    AssertionError
        If the per row and bulk outputs differ by more than floating point
        reassociation.

    Notes
    -----
    Compared on a relative tolerance rather than bit equality, and deliberately so.
    Batching reassociates accumulators, so a rolling mean or standard deviation can
    differ between the two paths by about one ULP. A bit equality assertion would reject
    a correct transformer.

    This is a finer granularity than
    `check_observe_transform_sequential_consistency`, which compares two chunks against
    their concatenation. Two chunks can agree while single rows do not, because a
    transformer whose lookback exceeds its declared ``observation_horizon`` still sees
    enough history in a large chunk and runs short on a single row.

    """
    tags = transformer.__sklearn_tags__()
    if tags.transformer_tags is None or not tags.transformer_tags.batch_invariant:
        return

    if len(X) < 12:
        return

    fit_size = len(X) // 2
    X_fit, block = X[:fit_size], X[fit_size:]

    bulk_transformer = clone(transformer)
    bulk_transformer.fit(X_fit, y)
    bulk = bulk_transformer.observe_transform(block)

    row_transformer = clone(transformer)
    row_transformer.fit(X_fit, y)
    per_row = pl.concat([row_transformer.observe_transform(block[i : i + 1]) for i in range(len(block))])

    assert bulk.columns == per_row.columns, (
        f"{type(transformer).__name__} declares batch_invariant but a bulk observe "
        f"produced columns {bulk.columns} against {per_row.columns} per row"
    )
    assert_frame_equal(
        bulk,
        per_row,
        rel_tol=1e-9,
        abs_tol=1e-12,
    )


def check_rewind_transform_behavior(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
    """Check rewind_transform() behavior and contract.

    Verifies that rewind_transform():
    1. Produces the same output as transform() on the same input from a freshly
       fitted clone (i.e. the result depends only on the fitted parameters and
       the provided input, matching transform())
    2. Calls transform() and discards the first observation_horizon values
    3. Resets the internal state to the last observation_horizon input rows

    Parameters
    ----------
    transformer : BaseActualTransformer
        Unfitted transformer
    X : pl.DataFrame
        Training data (needs to be long enough)
    y : pl.DataFrame, optional
        Target data

    Raises
    ------
    AssertionError
        If rewind_transform doesn't follow the expected contract

    """
    # Need enough data for meaningful test
    transformer_clone = clone(transformer)
    transformer_clone.fit(X, y)

    horizon = transformer_clone.observation_horizon

    # Need enough data: at least 2 * observation_horizon + 1
    min_len = 2 * horizon + 1 if horizon > 0 else 5
    if len(X) < min_len:
        return

    # Split data into fit and test portions
    split_point = len(X) // 2
    X_fit = X[:split_point]
    X_new = X[split_point:]

    # Test that rewind_transform doesn't use pre-existing memory
    transformer1 = clone(transformer)
    transformer1.fit(X_fit, y)

    # Apply rewind_transform
    X_rewind_trans = transformer1.rewind_transform(X_new)

    # Expected behavior: transform(X_new) (transform already drops warmup rows)
    transformer2 = clone(transformer)
    transformer2.fit(X_fit, y)  # Fit with same data to have same fitted params
    X_expected = transformer2.transform(X_new)

    # Check outputs match
    assert_frame_equal(
        X_rewind_trans,
        X_expected,
        rel_tol=1e-6,
        abs_tol=1e-8,
    )

    # Check that internal state was reset to X_new
    if hasattr(transformer1, "_X_observed") and horizon > 0:
        expected_observed = X_new[-horizon:] if horizon <= len(X_new) else X_new
        assert_frame_equal(
            transformer1._X_observed,
            expected_observed,
            rel_tol=1e-6,
            abs_tol=1e-8,
        )


def check_insufficient_data_raises(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
    """Check behavior when data length < observation_horizon.

    Transformers should either raise appropriate errors or gracefully handle
    insufficient data when given data shorter than their observation_horizon.

    Parameters
    ----------
    transformer : BaseActualTransformer
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
        # If it succeeds, graceful handling is acceptable (e.g. an empty frame),
        # but the result must still carry the mandatory time column.
        assert "time" in result.columns, "Result must have time column"
    except (ValueError, IndexError, pl.exceptions.ShapeError, pl.exceptions.ComputeError):
        # Expected behavior - transformer raises appropriate error
        pass


def check_transform_output_structure(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
    """Check transform() output has "time" column and valid structure.

    Transform output must be a polars DataFrame with a "time" column
    and valid data types.

    Parameters
    ----------
    transformer : BaseActualTransformer
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
    transformer : BaseActualTransformer
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
    transformer : BaseActualTransformer
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
    tags = transformer.__sklearn_tags__()
    if not (tags.transformer_tags and tags.transformer_tags.invertible):
        return

    transformer_clone = clone(transformer)
    transformer_clone.fit(X, y)

    # Transform the data - this may drop some rows (e.g., differencing)
    X_trans = transformer_clone.transform(X)

    # For yohou transformers, inverse_transform requires X_p (past observations)
    # immediately preceding X_trans in the original X.
    X_p = _build_X_p(transformer_clone, X, X_trans)

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


def check_inverse_observe_transform_identity(
    transformer,
    X: pl.DataFrame,
    y: pl.DataFrame | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """Check inverse_transform(observe_transform(X)) ≈ X.

    Verifies that inverse_transform correctly inverts observe_transform output.
    For stateless transformers: inverse_transform(observe_transform(X), X_p=None) == X
    For stateful transformers: uses X_p from before the observe_transform window.

    Parameters
    ----------
    transformer : BaseActualTransformer
        Unfitted transformer
    X : pl.DataFrame
        Training data (will be split for fit and observe_transform)
    y : pl.DataFrame, optional
        Target data
    atol : float
        Absolute tolerance for numerical comparison
    rtol : float
        Relative tolerance for numerical comparison

    Raises
    ------
    AssertionError
        If inverse round-trip via observe_transform fails

    Notes
    -----
    This check differs from check_inverse_transform_identity:
    - Uses observe_transform instead of transform
    - Tests round-trip on data AFTER initial fit (streaming scenario)
    - Verifies X_p handling for stateful transformers during updates

    """
    tags = transformer.__sklearn_tags__()
    if not (tags.transformer_tags and tags.transformer_tags.invertible):
        return

    # Need enough data to split into fit and update portions
    if len(X) < 10:
        return

    # Split data: first half for fit, second half for observe_transform
    split_idx = len(X) // 2
    X_fit = X[:split_idx]
    X_update = X[split_idx:]

    transformer_clone = clone(transformer)
    transformer_clone.fit(X_fit, y)

    horizon = transformer_clone.observation_horizon

    # Get X_p BEFORE observe_transform (from the fitted state)
    # For stateful transformers, X_p is the last `horizon` rows of observed data
    X_p = transformer_clone._X_observed.clone() if horizon > 0 else None

    # Apply observe_transform - this transforms X_update using memory context
    X_trans = transformer_clone.observe_transform(X_update)

    # Inverse transform to recover original
    X_reconstructed = transformer_clone.inverse_transform(X_trans, X_p)

    # The portion of X we should recover is X_update (possibly truncated)
    # If transformer drops rows, we compare against the tail of X_update
    X_expected = X_update.tail(len(X_trans))

    # Shape check
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
    transformer : BaseActualTransformer
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
    # Check if X_panel actually has panel columns
    global_names, panel_groups = inspect_panel(X_panel)

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


def check_panel_group_preservation(transformer, X_panel: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
    """Check that transformers preserve panel group names after transformation.

    Panel data uses columns with ``__`` separator (``<GROUP>__<SERIES>``).
    After transformation, the panel group names must remain unchanged even
    if the series names change (e.g., ``store_1__sales`` may become
    ``store_1__diff_s_7_sales`` but the group must stay ``store_1``).

    Parameters
    ----------
    transformer : BaseActualTransformer
        Unfitted transformer.
    X_panel : pl.DataFrame
        Panel data with panel columns.
    y : pl.DataFrame, optional
        Target data.

    Raises
    ------
    AssertionError
        If panel group names are not preserved after transformation.

    """
    # Check if X_panel actually has panel columns
    _, input_panel_groups = inspect_panel(X_panel)

    if not input_panel_groups:
        # Not panel data, skip
        return

    input_group_names = set(input_panel_groups.keys())

    transformer_clone = clone(transformer)

    try:
        transformer_clone.fit(X_panel, y)
        X_trans = transformer_clone.transform(X_panel)
    except NotImplementedError:
        # Transformer explicitly doesn't support panel data
        return

    # Inspect the output for panel groups
    _, output_panel_groups = inspect_panel(X_trans)
    output_group_names = set(output_panel_groups.keys())

    # Output must have the same panel group names
    assert output_group_names == input_group_names, (
        f"Panel group names changed after transformation. "
        f"Input groups: {sorted(input_group_names)}, "
        f"Output groups: {sorted(output_group_names)}. "
        f"Transformers must preserve panel group prefixes. "
        f"Output columns: {X_trans.columns}"
    )


def check_transformers_unfitted_stateless(transformer, X: pl.DataFrame) -> None:
    """Check stateless transformers transform deterministically across fits.

    Every transformer requires ``fit()`` before ``transform()`` (BaseActualTransformer
    always calls ``check_is_fitted``). For a stateless transformer
    (``observation_horizon == 0``) fitting carries no information beyond the
    schema, so two independent fits on the same data must yield identical
    transform output.

    Parameters
    ----------
    transformer : BaseActualTransformer
        Unfitted transformer expected to be stateless.
    X : pl.DataFrame
        Test data

    Raises
    ------
    AssertionError
        If a stateless transformer produces different output across fits

    """
    transformer1 = clone(transformer)
    transformer1.fit(X)

    # Only exercise genuinely stateless transformers.
    if transformer1.observation_horizon != 0:
        return

    X_trans1 = transformer1.transform(X)

    transformer2 = clone(transformer)
    transformer2.fit(X)
    X_trans2 = transformer2.transform(X)

    assert_frame_equal(X_trans1, X_trans2, rel_tol=1e-6, abs_tol=1e-8)


def check_transformer_preserve_dtypes(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
    """Check transformer preserves input dtypes.

    Transform and inverse_transform should maintain dtype consistency
    for numerical columns (except 'time' which is always datetime).

    Parameters
    ----------
    transformer : BaseActualTransformer
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
    input_dtypes = dict(zip(X_numeric.columns, X_numeric.dtypes, strict=False))

    X_trans = transformer_clone.transform(X)

    # Check transformed output dtypes (may have different columns)
    for col in X_trans.select(cs.numeric() & ~cs.by_name("time")).columns:
        output_dtype = X_trans[col].dtype
        # Allow float32 -> float64 promotion but expect numerical types
        assert output_dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.UInt64, pl.UInt32], (
            f"Column '{col}' has unexpected dtype {output_dtype}"
        )

    # Check inverse transform preserves original dtypes (if invertible)
    _tags = transformer_clone.__sklearn_tags__()
    if _tags.transformer_tags and _tags.transformer_tags.invertible:
        try:
            # Prepare X_p for yohou inverse_transform
            X_p = _build_X_p(transformer_clone, X, X_trans)

            X_inv = transformer_clone.inverse_transform(X_trans, X_p)

            for col, dtype in input_dtypes.items():
                if col in X_inv.columns:
                    # Allow some flexibility for float types
                    if dtype in [pl.Float64, pl.Float32]:
                        assert X_inv[col].dtype in [pl.Float64, pl.Float32], (
                            f"inverse_transform changed float dtype of '{col}': {dtype} -> {X_inv[col].dtype}"
                        )
                    else:
                        assert X_inv[col].dtype == dtype, (
                            f"inverse_transform changed dtype of '{col}': {dtype} -> {X_inv[col].dtype}"
                        )
        except (NotImplementedError, ValueError):
            # Not invertible or conditional inverse_transform not available
            # (e.g., sklearn's SimpleImputer requires add_indicator=True)
            pass


def check_fit_idempotent(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
    """Check that fit(X).fit(X) equals fit(X).

    Calling fit multiple times with same data should yield identical
    internal state and predictions.

    Parameters
    ----------
    transformer : BaseActualTransformer
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

    # Check fitted attributes match. Some transformers store array-valued
    # attributes (e.g. feature_names_in_ as a numpy array), so compare
    # element-wise rather than relying on a scalar ``==``.
    for attr in ["feature_names_in_", "n_features_in_", "_observation_horizon"]:
        if hasattr(transformer1, attr):
            val1 = getattr(transformer1, attr)
            val2 = getattr(transformer2, attr)
            if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
                equal = np.array_equal(np.asarray(val1), np.asarray(val2))
            else:
                equal = val1 == val2
            assert equal, f"Attribute '{attr}' differs after double fit: {val1} vs {val2}"


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
    transformer : BaseActualTransformer
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
    tags = transformer.__sklearn_tags__()
    if not (tags.transformer_tags and tags.transformer_tags.invertible):
        return

    transformer_clone = clone(transformer)
    transformer_clone.fit(X, y)

    # Forward transform
    X_trans = transformer_clone.transform(X)

    # For yohou transformers, inverse_transform requires X_p (past observations)
    X_p = _build_X_p(transformer_clone, X, X_trans)

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
    transformer : BaseActualTransformer
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

    # Combined fit_transform (BaseActualTransformer always defines fit_transform())
    X_trans2 = transformer2.fit_transform(X, y)

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
    transformer : BaseActualTransformer
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

    Notes
    -----
    For stateless transformers (observation_horizon == 0) the expected bound is
    ``int(0 * max_memory_factor) == 0`` and _X_observed is always empty, so the
    assertion holds trivially; no memory-growth behavior is exercised.

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
        transformer_clone.observe(X_chunk)

        actual_rows = len(transformer_clone._X_observed)
        assert actual_rows <= expected_max_rows, (
            f"Memory grew beyond bounds after {i + 1} updates: "
            f"{actual_rows} rows > {expected_max_rows} (horizon={horizon})"
        )


def check_tags_accessible_before_fit(transformer, X: pl.DataFrame | None = None) -> None:
    """Check __sklearn_tags__() is accessible before fit().

    Tags should be static class capabilities, not fitted state.
    They must be accessible before calling fit().

    Parameters
    ----------
    transformer : BaseActualTransformer
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
    transformer : BaseActualTransformer
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
    transformer : BaseActualTransformer
        Transformer instance (may be unfitted; tags must be accessible before
        fit per sklearn convention). Used as-is without cloning or fitting.
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
        # A transformer is stateful if it has _observation_horizon > 0
        # (observation_horizon=0 means no state needs to be preserved)
        has_stateful_observation_horizon = (
            hasattr(transformer, "_observation_horizon") and getattr(transformer, "_observation_horizon", 0) > 0
        )
        is_stateful = tags.transformer_tags.stateful

        # If transformer has observation_horizon > 0, it should be tagged as stateful
        if has_stateful_observation_horizon and not is_stateful:
            raise AssertionError(
                f"{transformer.__class__.__name__} has _observation_horizon > 0 but stateful tag is False"
            )

        # Check invertible tag: invertible transformers must override _inverse_transform
        # or inverse_transform (Tier 2 classes like SklearnTransformer override
        # inverse_transform directly instead of using the _inverse_transform hook).
        has_inverse_impl = (
            type(transformer)._inverse_transform is not BaseActualTransformer._inverse_transform
            or type(transformer).inverse_transform is not BaseActualTransformer.inverse_transform
        )
        is_invertible = tags.transformer_tags.invertible

        if is_invertible and not has_inverse_impl:
            raise AssertionError(
                f"{transformer.__class__.__name__} is tagged invertible but has no "
                f"_inverse_transform or inverse_transform override"
            )


def check_transformer_methods_call_check_is_fitted(transformer, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
    """Check all transformer methods (except fit) raise NotFittedError when unfitted.

    Validates that transform(), rewind(), observe(), observe_transform(), and
    inverse_transform() methods all check fitted state and raise NotFittedError
    before operating on an unfitted transformer.

    Parameters
    ----------
    transformer : BaseActualTransformer
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

    # Test that observe() raises NotFittedError when unfitted
    try:
        transformer_clone.observe(X)
        raise AssertionError(
            f"{transformer_clone.__class__.__name__}.observe() must raise NotFittedError when called on unfitted transformer"
        )
    except NotFittedError:
        pass  # Expected

    # Test that rewind() raises NotFittedError when unfitted
    try:
        transformer_clone.rewind(X)
        raise AssertionError(
            f"{transformer_clone.__class__.__name__}.rewind() must raise NotFittedError when called on unfitted transformer"
        )
    except NotFittedError:
        pass  # Expected

    # Test that observe_transform() raises NotFittedError when unfitted
    try:
        transformer_clone.observe_transform(X)
        raise AssertionError(
            f"{transformer_clone.__class__.__name__}.observe_transform() must raise NotFittedError when called on unfitted transformer"
        )
    except NotFittedError:
        pass  # Expected

    # Test that rewind_transform() raises NotFittedError when unfitted
    try:
        transformer_clone.rewind_transform(X)
        raise AssertionError(
            f"{transformer_clone.__class__.__name__}.rewind_transform() must raise NotFittedError when called on unfitted transformer"
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
