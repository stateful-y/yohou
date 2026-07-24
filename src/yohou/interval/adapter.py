"""Adaptive conformal inference adapters for interval forecasting."""

import numbers
from typing import Literal

from sklearn.utils.validation import check_is_fitted

from yohou.utils._compat import Interval, StrOptions, _fit_context

from .base import BaseConformalAdapter

__all__ = ["AdaptiveConformalInference"]


class AdaptiveConformalInference(BaseConformalAdapter):
    r"""Online miscoverage-level adjustment (Gibbs and Candes, 2021).

    Maintains a time-varying effective miscoverage level and updates it
    from the coverage realized on newly observed data, so a conformal
    forecaster restores its target coverage under distribution shift. For
    each tracked coverage rate the update is

    $$\alpha_{t+1} = \mathrm{clip}\big(\alpha_t + \gamma\,(\alpha^{*} -
    \mathrm{err}_t),\; \epsilon,\; 1 - \epsilon\big)$$

    where $\gamma$ is ``step_size``, $\alpha^{*} = 1 - \text{coverage rate}$
    is the target miscoverage, and $\mathrm{err}_t \in [0, 1]$ is the
    realized miscoverage of the interval that was in force. When the
    conformity scorer is asymmetric, a lower and an upper level are tracked
    separately, each targeting $\alpha^{*}/2$.

    This adapter owns one horizon step; ``SplitConformalForecaster`` clones
    one per step and supplies the miscoverage indicators (it holds the
    calibration scores and any similarity weights). The adapter is the
    level-recursion state machine only.

    Parameters
    ----------
    step_size : float, default=0.05
        Learning rate $\gamma$ of the online update. Larger values react
        faster to coverage drift but track more noisily.
    alpha_pooling : {"per_step", "shared"}, default="per_step"
        How the enclosing forecaster pools miscoverage across horizon steps
        before updating. ``"per_step"`` lets each step's level evolve
        independently; ``"shared"`` makes the forecaster pool the per-step
        indicators and feed every step the same update, yielding one shared
        trajectory. The adapter stores this so the forecaster can read it;
        the update itself is identical either way.
    epsilon : float, default=0.0
        Clips the effective level to ``[epsilon, 1 - epsilon]``. The default
        ``0.0`` reproduces the paper-exact ``[0, 1]`` clipping; a positive
        value prevents fully degenerate intervals in production.

    Attributes
    ----------
    levels_ : dict
        Current effective level per coverage rate: a float for symmetric
        scorers, or a ``(lower, upper)`` tuple for asymmetric ones.
    coverage_rates_ : list of float
        The coverage rates seeded at fit time.
    symmetric_ : bool
        Whether the tracked scorer is symmetric.

    References
    ----------
    [1] Gibbs, I., & Candes, E. (2021). "Adaptive conformal inference under
        distribution shift." Advances in Neural Information Processing
        Systems, 34, 1660-1672.

    See Also
    --------
    - [`BaseConformalAdapter`][yohou.interval.base.BaseConformalAdapter] : Abstract adapter base class.
    - [`SplitConformalForecaster`][yohou.interval.split_conformal.SplitConformalForecaster] :
        Conformal forecaster that consumes an adapter.

    Examples
    --------
    >>> from yohou.interval.adapter import AdaptiveConformalInference
    >>> adapter = AdaptiveConformalInference(step_size=0.1).fit([0.9], symmetric=True)
    >>> round(adapter.predict()[0.9], 4)
    0.1
    >>> # A miscovered observation (err=1) lowers the level, widening the interval.
    >>> _ = adapter.observe([{0.9: 1.0}])
    >>> round(adapter.predict()[0.9], 4)
    0.01
    >>> # Rewinding restores the seeded level.
    >>> _ = adapter.rewind(1)
    >>> round(adapter.predict()[0.9], 4)
    0.1

    """

    _parameter_constraints: dict = {
        "step_size": [Interval(numbers.Real, 0, None, closed="neither")],
        "alpha_pooling": [StrOptions({"per_step", "shared"})],
        "epsilon": [Interval(numbers.Real, 0, 0.5, closed="left")],
    }

    def __init__(
        self,
        step_size: float = 0.05,
        alpha_pooling: Literal["per_step", "shared"] = "per_step",
        epsilon: float = 0.0,
    ) -> None:
        self.step_size = step_size
        self.alpha_pooling = alpha_pooling
        self.epsilon = epsilon

    def _clip(self, level: float) -> float:
        """Clamp a level to ``[epsilon, 1 - epsilon]``."""
        return min(max(level, self.epsilon), 1.0 - self.epsilon)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, coverage_rates: list[float], *, symmetric: bool) -> "AdaptiveConformalInference":
        """Seed one effective level per coverage rate.

        Parameters
        ----------
        coverage_rates : list of float
            Nominal coverage rates to track. Each is seeded at its target
            miscoverage ``1 - coverage_rate`` (halved per tail for
            asymmetric scorers).
        symmetric : bool
            Whether the conformity scorer is symmetric.

        Returns
        -------
        self

        """
        self.symmetric_ = symmetric
        self.coverage_rates_ = list(coverage_rates)

        self.levels_: dict[float, object] = {}
        self._level_history: dict[float, list] = {}
        for coverage_rate in self.coverage_rates_:
            alpha_target = 1.0 - coverage_rate
            seed = alpha_target if symmetric else (alpha_target / 2.0, alpha_target / 2.0)
            self.levels_[coverage_rate] = seed
            self._level_history[coverage_rate] = [seed]

        return self

    def observe(self, errors: list[dict[float, object]]) -> "AdaptiveConformalInference":
        """Advance each level by one update per observed row.

        Parameters
        ----------
        errors : list of dict
            One entry per newly observed row, mapping each tracked coverage
            rate to its miscoverage signal (a float for symmetric scorers, a
            ``(lower, upper)`` tuple for asymmetric ones).

        Returns
        -------
        self

        """
        check_is_fitted(self, "levels_")

        for row_errors in errors:
            for coverage_rate in self.coverage_rates_:
                alpha_target = 1.0 - coverage_rate
                err = row_errors[coverage_rate]

                if self.symmetric_:
                    current = self.levels_[coverage_rate]
                    assert isinstance(current, float)
                    updated: object = self._clip(current + self.step_size * (alpha_target - float(err)))  # ty: ignore[invalid-argument-type]
                else:
                    lower, upper = self.levels_[coverage_rate]  # ty: ignore[not-iterable]
                    err_lower, err_upper = err  # ty: ignore[not-iterable]
                    target = alpha_target / 2.0
                    updated = (
                        self._clip(lower + self.step_size * (target - float(err_lower))),
                        self._clip(upper + self.step_size * (target - float(err_upper))),
                    )

                self.levels_[coverage_rate] = updated
                self._level_history[coverage_rate].append(updated)

        return self

    def predict(self) -> dict[float, object]:
        """Return the current effective level per coverage rate.

        Returns
        -------
        dict
            Maps each tracked coverage rate to its current effective level.

        """
        check_is_fitted(self, "levels_")
        return dict(self.levels_)

    def rewind(self, n_rows: int) -> "AdaptiveConformalInference":
        """Undo the level updates from the last ``n_rows`` observations.

        Parameters
        ----------
        n_rows : int
            Number of most-recently observed rows to roll back, never
            dropping below the fit-time seed.

        Returns
        -------
        self

        """
        check_is_fitted(self, "levels_")

        for coverage_rate in self.coverage_rates_:
            history = self._level_history[coverage_rate]
            for _ in range(n_rows):
                if len(history) > 1:
                    history.pop()
            self.levels_[coverage_rate] = history[-1]

        return self
