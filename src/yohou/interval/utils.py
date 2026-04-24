"""Utility functions for weighted quantile calculations."""

import numpy as np
import numpy.typing as npt

__all__ = ["weighted_quantile"]


def weighted_quantile(x: npt.NDArray[np.float64], q: float, weights: npt.NDArray[np.float64]) -> float:
    """Compute weighted quantile using cumulative sum approach.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.

    q : float
        Quantile to compute (between 0 and 1).

    weights : np.ndarray
        Weights for each value in x (must match length of x).

    Returns
    -------
    float
        The weighted quantile value, or infinity if insufficient weight.

    """
    if len(x) != len(weights):
        raise ValueError(f"x and weights must have the same length, got {len(x)} and {len(weights)}")
    if np.sum(weights) >= 1 - q:
        x_ordered = np.argsort(x)
        index_threshold = np.min(np.where(np.cumsum(weights[x_ordered]) >= 1 - q))
        quantile = np.sort(x)[index_threshold]

    else:
        quantile = float("inf")

    return float(quantile)
