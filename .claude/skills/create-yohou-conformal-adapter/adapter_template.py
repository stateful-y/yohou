"""Template for a new conformal adapter.

Copy into ``src/yohou/interval/adapter.py`` and rename. This template shows the
full ``BaseConformalAdapter`` lifecycle with one tunable parameter. Delete the
comments and adapt the update rule to your method.
"""

import numbers

from sklearn.utils.validation import check_is_fitted

from yohou.utils._compat import Interval, _fit_context

from .base import BaseConformalAdapter

__all__ = ["MyAdapter"]


class MyAdapter(BaseConformalAdapter):
    """One-line summary of the level-adaptation rule.

    Longer description: what signal it reacts to and how the effective level
    moves. Cite the method's reference.

    Parameters
    ----------
    step_size : float, default=0.05
        Learning rate of the online update.

    Attributes
    ----------
    levels_ : dict
        Current effective level per coverage rate (float for symmetric
        scorers, ``(lower, upper)`` tuple for asymmetric).
    coverage_rates_ : list of float
        The coverage rates seeded at fit time.
    symmetric_ : bool
        Whether the tracked scorer is symmetric.

    Examples
    --------
    >>> from yohou.interval.adapter import MyAdapter  # doctest: +SKIP
    >>> adapter = MyAdapter().fit([0.9], symmetric=True)  # doctest: +SKIP
    >>> round(adapter.predict()[0.9], 4)  # doctest: +SKIP
    0.1

    """

    _parameter_constraints: dict = {
        "step_size": [Interval(numbers.Real, 0, None, closed="neither")],
    }

    def __init__(self, step_size: float = 0.05) -> None:
        # Only assign constructor args here, verbatim (sklearn contract).
        self.step_size = step_size

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, coverage_rates: list[float], *, symmetric: bool) -> "MyAdapter":
        """Seed one effective level per coverage rate."""
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

    def observe(self, errors: list[dict[float, object]]) -> "MyAdapter":
        """Advance each level by one update per observed row."""
        check_is_fitted(self, "levels_")
        for row_errors in errors:
            for coverage_rate in self.coverage_rates_:
                alpha_target = 1.0 - coverage_rate
                err = row_errors[coverage_rate]
                if self.symmetric_:
                    current = float(self.levels_[coverage_rate])  # type: ignore[arg-type]
                    # Replace with your rule. A row's zero-update sentinel has
                    # err == alpha_target, so this must be a no-op for it.
                    updated: object = current + self.step_size * (alpha_target - float(err))
                else:
                    lower, upper = self.levels_[coverage_rate]  # type: ignore[misc]
                    err_lower, err_upper = err  # type: ignore[misc]
                    target = alpha_target / 2.0
                    updated = (
                        lower + self.step_size * (target - float(err_lower)),
                        upper + self.step_size * (target - float(err_upper)),
                    )
                self.levels_[coverage_rate] = updated
                self._level_history[coverage_rate].append(updated)
        return self

    def predict(self) -> dict[float, object]:
        """Return the current effective level per coverage rate."""
        check_is_fitted(self, "levels_")
        return dict(self.levels_)

    def rewind(self, n_rows: int) -> "MyAdapter":
        """Undo the last ``n_rows`` updates, never below the seed."""
        check_is_fitted(self, "levels_")
        for coverage_rate in self.coverage_rates_:
            history = self._level_history[coverage_rate]
            for _ in range(n_rows):
                if len(history) > 1:
                    history.pop()
            self.levels_[coverage_rate] = history[-1]
        return self
