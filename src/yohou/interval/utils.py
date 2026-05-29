"""Utility functions for weighted quantile calculations."""

import numpy as np
import numpy.typing as npt

__all__ = ["weighted_quantile"]


def weighted_quantile(x: npt.NDArray[np.float64], q: float, weights: npt.NDArray[np.float64]) -> float:
    """Compute weighted quantile using cumulative sum approach.

    Weights are normalized to sum to 1 internally so that all quantile
    levels are computable regardless of the raw weight magnitude.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.

    q : float
        Quantile level to compute (between 0 and 1).  The function
        returns the smallest value ``x[i]`` (in sorted order) such that
        the cumulative normalized weight up to ``x[i]`` is at least
        ``1 - q``.

    weights : np.ndarray
        Weights for each value in x (must match length of x).

    Returns
    -------
    float
        The weighted quantile value.

    """
    if len(x) != len(weights):
        raise ValueError(f"x and weights must have the same length, got {len(x)} and {len(weights)}")

    w_sum = np.sum(weights)
    if w_sum == 0:
        return float("inf")

    # Normalize so the weighted CDF reaches 1, making all quantile levels computable
    w_norm = weights / w_sum
    # Sort scores ascending and reorder weights to match
    x_ordered = np.argsort(x)
    # Walk up the sorted scores until cumulative weight reaches the (1-q)-th level
    index_threshold = np.min(np.where(np.cumsum(w_norm[x_ordered]) >= 1 - q))
    return float(np.sort(x)[index_threshold])
