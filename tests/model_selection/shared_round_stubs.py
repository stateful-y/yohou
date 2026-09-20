"""Stub boosting estimator and adapter for testing ``validation="cv"`` without a boosting library.

``CurveRegressor`` behaves like a boosting model whose prediction after r
rounds is ``train_mean * r / 10``. Its evaluation loss at round r is the mean
absolute gap between the evaluation targets and that prediction, so the best
round depends on the evaluation data it receives: a wrong evaluation set
yields a different curve rather than passing silently.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

from yohou.model_selection import BaseEarlyStoppingAdapter


class CurveRegressor(RegressorMixin, BaseEstimator):
    """Boosting-like regressor with a controllable stopping curve."""

    def __init__(self, n_rounds: int = 60, patience: int = 5, fail_below_train_rows: int = 0, scale: float = 1.0):
        self.n_rounds = n_rounds
        self.patience = patience
        self.fail_below_train_rows = fail_below_train_rows
        self.scale = scale

    def fit(self, X, y, eval_set=None, callbacks=None, **kwargs):
        arr = np.asarray(y, dtype=float)
        if len(arr) < self.fail_below_train_rows:
            raise RuntimeError(f"CurveRegressor refuses {len(arr)} training rows")
        self.received_callbacks_ = callbacks
        self._ncols = 1 if arr.ndim == 1 else arr.shape[1]
        self.train_mean_ = float(np.nanmean(arr)) * self.scale
        self.received_eval_targets_ = None
        self.curve_ = []
        if eval_set is None:
            self.rounds_trained_ = self.n_rounds
        else:
            y_eval = np.asarray(eval_set[0][1], dtype=float).ravel()
            self.received_eval_targets_ = y_eval
            best, best_round = np.inf, 0
            for r in range(1, self.n_rounds + 1):
                loss = float(np.mean(np.abs(y_eval - self.train_mean_ * r / 10.0)))
                self.curve_.append(loss)
                if loss < best:
                    best, best_round = loss, r
                elif r - best_round >= self.patience:
                    break
            self.rounds_trained_ = len(self.curve_)
        self.rounds_used_ = self.rounds_trained_
        return self

    def predict(self, X):
        out = np.full((len(X), self._ncols), self.train_mean_ * self.rounds_used_ / 10.0)
        return out.ravel() if self._ncols == 1 else out


class CurveEarlyStoppingAdapter(BaseEarlyStoppingAdapter):
    """Adapter for `CurveRegressor`, recording every call it receives."""

    def __init__(self):
        self.calls = []

    def supports(self, estimator):
        return isinstance(estimator, CurveRegressor)

    def validate(self, estimator, fit_params=None):
        self.calls.append(("validate", estimator.get_params(), fit_params))

    def prepare_fold_fit(self, estimator):
        self.calls.append(("prepare_fold_fit", estimator.get_params()))
        return clone(estimator), {}

    def stopping_curve(self, fitted):
        return np.asarray(fitted.curve_, dtype=float), False

    def truncate(self, fitted, n_rounds):
        self.calls.append(("truncate", n_rounds))
        fitted.rounds_used_ = n_rounds

    def prepare_refit(self, estimator, n_rounds):
        self.calls.append(("prepare_refit", n_rounds))
        return clone(estimator).set_params(n_rounds=n_rounds)


class CallbackCurveEarlyStoppingAdapter(CurveEarlyStoppingAdapter):
    """`CurveEarlyStoppingAdapter` whose fold fits carry a ``callbacks`` fit parameter."""

    def prepare_fold_fit(self, estimator):
        return clone(estimator), {"callbacks": ["adapter"]}


class QuantileCurveRegressor(CurveRegressor):
    """`CurveRegressor` with the ``quantile`` parameter interval forecasters set per bound."""

    def __init__(
        self,
        quantile: float = 0.5,
        n_rounds: int = 60,
        patience: int = 5,
        fail_below_train_rows: int = 0,
        scale: float = 1.0,
    ):
        super().__init__(n_rounds=n_rounds, patience=patience, fail_below_train_rows=fail_below_train_rows, scale=scale)
        self.quantile = quantile

    def fit(self, X, y, eval_set=None, **kwargs):
        # Offset the level by the quantile so the two bounds stop at different rounds.
        self.scale = 0.8 + 0.4 * self.quantile
        return super().fit(X, y, eval_set=eval_set, **kwargs)
