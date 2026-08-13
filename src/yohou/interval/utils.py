"""Utility functions for weighted quantile calculations."""

import warnings

import numpy as np
import numpy.typing as npt

__all__ = [
    "required_calibration_size",
    "warn_if_calibration_too_small",
    "warn_if_weights_collapsed",
    "weighted_quantile",
]


def weighted_quantile(x: npt.NDArray[np.float64], q: float, weights: npt.NDArray[np.float64]) -> float:
    """Compute a weighted conformal quantile.

    Returns the smallest value ``x[i]`` (in sorted order) whose cumulative
    weight reaches ``1 - q``.

    The weights are used as given and are **not** renormalized. Callers pass
    weights that already reserve mass for the hypothetical test point, so each
    row sums to strictly less than 1: uniform weights are ``1 / (n + 1)`` each
    and similarity weights come from
    [`BaseSimilarity._reserve_mass`][yohou.interval.base.BaseSimilarity].
    That reserved remainder is the weighted analogue of the ``(n + 1)``
    correction in unweighted split conformal, and it is what makes coverage at
    least ``1 - q``. Renormalizing to sum to 1, as this function used to do,
    cancels the reservation and costs exactly one order statistic.

    When the calibration mass never reaches ``1 - q`` the correct bound is
    unbounded. The largest observed score is returned instead, which
    under-covers; callers detect that case up front with
    :func:`warn_if_calibration_too_small` and warn.

    Parameters
    ----------
    x : np.ndarray
        Input array of values.

    q : float
        Miscoverage level (between 0 and 1). The cumulative weight target is
        ``1 - q``.

    weights : np.ndarray
        Weights for each value in x (must match length of x), already
        carrying reserved test-point mass.

    Returns
    -------
    float
        The weighted quantile value.

    Raises
    ------
    ValueError
        If ``x`` and ``weights`` have different lengths, or if the weights
        sum to zero (no calibration mass), which would otherwise yield an
        undefined quantile that silently corrupts interval bounds.

    """
    if len(x) != len(weights):
        raise ValueError(f"x and weights must have the same length, got {len(x)} and {len(weights)}")

    if np.sum(weights) == 0:
        raise ValueError(
            "weighted_quantile received weights that sum to zero; cannot compute a "
            "weighted quantile. This usually means all similarity weights collapsed "
            "to zero for the current prediction context."
        )

    # Sort scores ascending and reorder weights to match
    x_ordered = np.argsort(x)
    x_sorted = x[x_ordered]
    # Walk up the sorted scores until cumulative weight reaches the (1-q)-th level.
    reached = np.where(np.cumsum(weights[x_ordered]) >= 1 - q)[0]
    if reached.size == 0:
        # Not enough calibration mass to reach the level: the conformal bound is
        # unbounded. Return the widest finite bound available.
        return float(x_sorted[-1])
    return float(x_sorted[int(reached[0])])


def required_calibration_size(coverage_rate: float, symmetric: bool) -> int:
    """Smallest calibration count that can express a coverage rate.

    Split conformal takes the ``ceil((n + 1) * q)``-th order statistic of the
    calibration scores, which is what makes coverage at least ``q``. That index
    exists only while ``ceil((n + 1) * q) <= n``, which rearranges to
    ``n >= q / (1 - q)``. Above that rate the correct bound is unbounded, and
    any finite value the implementation returns instead under-covers.

    Asymmetric scorers split the miscoverage across two tails, so the binding
    level is ``q = 1 - alpha / 2`` rather than the coverage rate itself, and
    they need roughly twice as many scores for the same rate.

    Parameters
    ----------
    coverage_rate : float
        Nominal coverage rate.
    symmetric : bool
        Whether the conformity scorer is symmetric. Asymmetric scorers split
        the miscoverage across two tails, so each tail sits further out and
        needs roughly twice as many scores.

    Returns
    -------
    int
        Minimum number of calibration scores per value column. Returns ``0``
        when the rate imposes no constraint.

    """
    if coverage_rate >= 1.0:
        return 0

    tail_level = coverage_rate if symmetric else 1.0 - (1.0 - coverage_rate) / 2.0
    if tail_level >= 1.0:
        return 0

    # q / (1 - q) is the closed form, but it is not safe to ceil directly:
    # 0.9 / (1 - 0.9) evaluates to 9.000000000000002, which would demand 10
    # scores where 9 suffices. Start just below the closed form and step up
    # until the conformal index actually fits, which is exact by construction.
    candidate = max(int(np.floor(tail_level / (1.0 - tail_level))) - 1, 1)
    while int(np.ceil((candidate + 1) * tail_level)) > candidate:
        candidate += 1
    return candidate


def warn_if_calibration_too_small(n_scores: int, coverage_rate: float, symmetric: bool, step: int) -> bool:
    """Warn when the calibration set cannot express the requested coverage.

    Below the required count the interval is narrower than the nominal rate
    implies: an asymmetric scorer clamps at the span of the observed scores and
    stops widening, a symmetric one selects too low an order statistic. Either
    way the frame is well-formed and the columns are labelled with the rate that
    was asked for, which is what makes this worth saying out loud.

    Parameters
    ----------
    n_scores : int
        Calibration scores available per value column for this step.
    coverage_rate : float
        Nominal coverage rate being computed.
    symmetric : bool
        Whether the conformity scorer is symmetric.
    step : int
        Horizon step, named in the warning.

    Returns
    -------
    bool
        Whether a warning was emitted.

    Warns
    -----
    UserWarning
        When ``n_scores`` is below the count the rate requires.

    """
    required = required_calibration_size(coverage_rate, symmetric)
    if n_scores >= required:
        return False

    warnings.warn(
        f"Coverage rate {coverage_rate} at step {step} needs at least {required} calibration scores per "
        f"value column, but only {n_scores} are available. The interval is capped at the span of the "
        "observed conformity scores, so it is narrower than the nominal rate implies and will under-cover. "
        "Increase calibration_size, extend the history, or request a lower coverage rate.",
        UserWarning,
        stacklevel=3,
    )
    return True


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
