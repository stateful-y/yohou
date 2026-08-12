"""Utility functions for weighted quantile calculations."""

import warnings

import numpy as np
import numpy.typing as npt

__all__ = ["warn_if_weights_collapsed", "weighted_quantile"]


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

    Raises
    ------
    ValueError
        If ``x`` and ``weights`` have different lengths, or if the weights
        sum to zero (no calibration mass), which would otherwise yield an
        undefined (infinite) quantile that silently corrupts interval bounds.

    """
    if len(x) != len(weights):
        raise ValueError(f"x and weights must have the same length, got {len(x)} and {len(weights)}")

    w_sum = np.sum(weights)
    if w_sum == 0:
        raise ValueError(
            "weighted_quantile received weights that sum to zero; cannot compute a "
            "weighted quantile. This usually means all similarity weights collapsed "
            "to zero for the current prediction context."
        )

    # Normalize so the weighted CDF reaches 1, making all quantile levels computable
    w_norm = weights / w_sum
    # Sort scores ascending and reorder weights to match
    x_ordered = np.argsort(x)
    # Walk up the sorted scores until cumulative weight reaches the (1-q)-th level
    index_threshold = np.min(np.where(np.cumsum(w_norm[x_ordered]) >= 1 - q))
    return float(np.sort(x)[index_threshold])


def warn_if_weights_collapsed(weights: npt.NDArray[np.float64], step: int, coverage_rate: float) -> float:
    """Warn when a weight row carries almost no calibration mass.

    A weight vector concentrated on one calibration row makes the weighted
    quantile a function of a single score, so both tails return that score and
    the interval collapses to zero width at a nominal coverage rate. That is
    never intentional, and it is silent otherwise: the emitted interval is
    well-formed, just meaningless. Guarding on the weights rather than on the
    emitted width avoids firing on the legitimate case of constant residuals,
    where a zero-width interval is the correct answer.

    Parameters
    ----------
    weights : np.ndarray
        Weight row of shape ``(n_calibration,)``.
    step : int
        Horizon step the weights belong to, named in the warning.
    coverage_rate : float
        Nominal coverage rate being computed, named in the warning.

    Returns
    -------
    float
        The effective sample size ``(sum w)^2 / sum(w^2)``, or ``0.0`` when
        the weights carry no mass at all.

    Warns
    -----
    UserWarning
        When the effective sample size falls below 2.

    """
    total = float(np.sum(weights))
    sum_of_squares = float(np.sum(np.square(weights)))
    effective_sample_size = (total**2) / sum_of_squares if sum_of_squares > 0 else 0.0

    if effective_sample_size < 2.0:
        warnings.warn(
            f"Similarity weights for step {step} at coverage rate {coverage_rate} have collapsed to "
            f"{effective_sample_size:.2f} effective calibration rows out of {len(weights)}. The interval "
            "is being read off almost a single conformity score and may be degenerate. This usually means "
            "the similarity is over-concentrating; raise its `bandwidth` to widen the neighbourhood.",
            UserWarning,
            stacklevel=3,
        )

    return effective_sample_size
