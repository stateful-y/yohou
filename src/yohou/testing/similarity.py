"""Check functions for yohou similarity measures.

Systematic validation functions for the ``BaseSimilarity`` estimator family
(interval-forecasting similarity measures). All check functions raise
AssertionError on failure.
"""

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

__all__ = [
    "check_similarity_metric_params_verbatim",
    "check_similarity_methods_call_check_is_fitted",
    "check_similarity_predict_matrix_shape",
    "check_similarity_to_weights_rows_reserve_mass",
]


def check_similarity_predict_matrix_shape(similarity, y_calib: pl.DataFrame, y_pred_calib: pl.DataFrame) -> None:
    """Check ``predict`` returns an ``(n_pred, n_calib)`` weight matrix.

    Parameters
    ----------
    similarity : BaseSimilarity
        Similarity instance.
    y_calib : pl.DataFrame
        Calibration target frame.
    y_pred_calib : pl.DataFrame
        Calibration prediction frame.

    Raises
    ------
    AssertionError
        If ``predict`` does not return the expected ``(n_pred, n_calib)`` shape.

    """
    name = type(similarity).__name__
    sim = clone(similarity)
    sim.fit(y_calib, y_pred_calib)

    n_pred = 2
    query = y_pred_calib.head(n_pred)
    weights = sim.predict(query)
    assert isinstance(weights, np.ndarray), f"{name}.predict must return np.ndarray, got {type(weights)}"
    assert weights.shape == (n_pred, len(y_calib)), (
        f"{name}.predict returned shape {weights.shape}, expected ({n_pred}, {len(y_calib)})"
    )


def check_similarity_to_weights_rows_reserve_mass(
    similarity, y_calib: pl.DataFrame, y_pred_calib: pl.DataFrame
) -> None:
    """Check each predicted weight row is non-negative and sums below 1.

    Pins the ``BaseSimilarity._to_weights`` / ``_reserve_mass`` invariant: a
    softmax of negative distances re-normalized as ``raw / (sum(raw) + 1)``, so
    every row reserves mass for the (hypothetical) test point.

    Parameters
    ----------
    similarity : BaseSimilarity
        Similarity instance.
    y_calib : pl.DataFrame
        Calibration target frame.
    y_pred_calib : pl.DataFrame
        Calibration prediction frame.

    Raises
    ------
    AssertionError
        If any row is negative or sums to 1 or more.

    """
    name = type(similarity).__name__
    sim = clone(similarity)
    sim.fit(y_calib, y_pred_calib)

    weights = sim.predict(y_pred_calib.head(2))
    assert np.all(weights >= 0), f"{name}.predict produced negative weights"
    row_sums = weights.sum(axis=1)
    assert np.all(row_sums < 1.0), (
        f"{name}.predict rows must sum to < 1 (reserved-mass invariant), got max row sum {row_sums.max()}"
    )


def check_similarity_metric_params_verbatim(similarity) -> None:
    """Check ``metric_params=None`` is stored verbatim (no init-mutation).

    Skipped for similarities without a ``metric_params`` parameter.

    Parameters
    ----------
    similarity : BaseSimilarity
        Similarity instance (used for its type).

    Raises
    ------
    AssertionError
        If a default-constructed instance mutates ``metric_params`` away from None.

    """
    cls = type(similarity)
    fresh = cls()
    if "metric_params" not in fresh.get_params(deep=False):
        return
    assert fresh.metric_params is None, (
        f"{cls.__name__}: default metric_params mutated from None to {fresh.metric_params!r}"
    )
    assert fresh.get_params(deep=False)["metric_params"] is None, (
        f"{cls.__name__}: get_params reports metric_params as {fresh.get_params(deep=False)['metric_params']!r}, not None"
    )


def check_similarity_methods_call_check_is_fitted(similarity, y_pred_calib: pl.DataFrame) -> None:
    """Check ``predict`` and ``observe`` raise ``NotFittedError`` before ``fit``.

    Parameters
    ----------
    similarity : BaseSimilarity
        Similarity instance.
    y_pred_calib : pl.DataFrame
        A prediction frame to pass to the unfitted methods under test.

    Raises
    ------
    AssertionError
        If ``predict`` or ``observe`` does not raise ``NotFittedError`` when unfitted.

    """
    name = type(similarity).__name__
    query = y_pred_calib.head(2)

    unfitted = clone(similarity)
    try:
        unfitted.predict(query)
        raise AssertionError(f"{name}.predict() must raise NotFittedError when unfitted")
    except NotFittedError:
        pass

    unfitted2 = clone(similarity)
    try:
        # observe(y, y_pred, X_actual=None): both slots receive query here as a
        # convenience; only the unfitted-error path is being exercised.
        unfitted2.observe(query, query)
        raise AssertionError(f"{name}.observe() must raise NotFittedError when unfitted")
    except NotFittedError:
        pass
