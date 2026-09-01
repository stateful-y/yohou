"""Check functions for yohou reduction forecasters.

This module provides validation functions specific to reduction forecasters
(BaseReductionForecaster implementations).
"""

import inspect

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone

__all__ = [
    "check_estimator_parameter",
    "check_reduction_strategy",
    "check_validation_holdout_default_noop",
    "check_validation_holdout_fit",
    "check_validation_holdout_parameters",
]

# Holdout size the behavioral validation checks use. Together with the check
# suite's forecasting_horizon of 3 this leaves strict-mode room
# (5 >= 3 -> 3 evaluation anchors per group) on every family fixture.
_CHECK_VALIDATION_SIZE = 5
_CHECK_HORIZON = 3


class _RecordingEvalRegressor(RegressorMixin, BaseEstimator):
    """Regressor stub whose fit records the eval_set it received."""

    def fit(self, X, y, eval_set=None, sample_weight=None):
        """Record the fit inputs and learn the target mean."""
        self.received_eval_set_ = eval_set
        self.train_X_ = X
        arr = np.asarray(y, dtype=float)
        self._ncols = 1 if arr.ndim == 1 else arr.shape[1]
        self._mean = float(np.nanmean(arr))
        return self

    def predict(self, X):
        """Predict the learned mean for every row."""
        out = np.full((len(X), self._ncols), self._mean)
        return out.ravel() if self._ncols == 1 else out


class _RecordingEvalClassifier(ClassifierMixin, BaseEstimator):
    """Classifier stub whose fit records the eval_set it received."""

    def fit(self, X, y, eval_set=None, sample_weight=None):
        """Record the fit inputs and learn the class codes."""
        self.received_eval_set_ = eval_set
        self.train_X_ = X
        arr = np.asarray(y, dtype=float)
        self.classes_ = np.unique(arr.ravel())
        self._ncols = 1 if arr.ndim == 1 else arr.shape[1]
        return self

    def predict(self, X):
        """Predict the first class for every row."""
        out = np.full((len(X), self._ncols), self.classes_[0])
        return out.ravel() if self._ncols == 1 else out

    def predict_proba(self, X):
        """Predict uniform class probabilities for every row."""
        return np.tile(np.full(len(self.classes_), 1.0 / len(self.classes_)), (len(X), 1))


def _stub_for(forecaster) -> BaseEstimator:
    """Return the recording stub matching the forecaster's family."""
    tags = forecaster.__sklearn_tags__()
    forecaster_type = tags.forecaster_tags.forecaster_type if tags.forecaster_tags else frozenset()
    if forecaster_type is not None and "class_proba" in forecaster_type:
        return _RecordingEvalClassifier()
    return _RecordingEvalRegressor()


def check_estimator_parameter(forecaster) -> None:
    """Check estimator parameter is sklearn BaseEstimator.

    Parameters
    ----------
    forecaster : BaseReductionForecaster
        Reduction forecaster instance

    Raises
    ------
    AssertionError
        If estimator is not a sklearn BaseEstimator

    """
    assert hasattr(forecaster, "estimator"), "Reduction forecaster must have 'estimator' parameter"

    estimator = forecaster.estimator
    assert isinstance(estimator, BaseEstimator), f"estimator should be sklearn BaseEstimator, got {type(estimator)}"


def check_reduction_strategy(forecaster) -> None:
    """Check reduction_strategy is one of 'direct', 'dir-rec', 'multi-output'.

    The same set is enforced by ``BaseReductionForecaster._parameter_constraints``
    during ``_validate_params``; this check additionally guards against a value
    set illegally after construction (e.g. by direct attribute assignment, which
    bypasses constraint validation).

    Parameters
    ----------
    forecaster : BaseReductionForecaster
        Reduction forecaster instance

    Raises
    ------
    AssertionError
        If reduction_strategy is not one of the allowed values

    """
    if not hasattr(forecaster, "reduction_strategy"):
        # Not all reduction forecasters expose this parameter
        return

    strategy = forecaster.reduction_strategy
    valid_strategies = ["direct", "dir-rec", "multi-output"]

    assert strategy in valid_strategies, f"reduction_strategy should be in {valid_strategies}, got '{strategy}'"


def check_validation_holdout_parameters(forecaster) -> None:
    """Check the validation-holdout constructor parameters and their defaults.

    Parameters
    ----------
    forecaster : BaseReductionForecaster
        Reduction forecaster instance exposing ``validation_size``.

    Raises
    ------
    AssertionError
        If the parameters are missing from ``get_params``, their constructor
        defaults are not ``None`` / ``False``, or the instance values have
        invalid types.

    """
    params = forecaster.get_params(deep=False)
    assert "validation_size" in params, "validation_size must be a constructor parameter"
    assert "validation_overlap" in params, "validation_overlap must be a constructor parameter"

    signature = inspect.signature(type(forecaster).__init__)
    assert signature.parameters["validation_size"].default is None, "validation_size must default to None"
    assert signature.parameters["validation_overlap"].default is False, "validation_overlap must default to False"

    if params["validation_size"] is not None:
        assert isinstance(params["validation_size"], int) and params["validation_size"] >= 1
    assert isinstance(params["validation_overlap"], bool)


def check_validation_holdout_fit(
    forecaster,
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None = None,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
) -> None:
    """Check that fitting with ``validation_size`` delivers a valid eval set.

    Clones the forecaster with a recording stub estimator (keeping whatever
    transformers and strategy the instance is equipped with), fits with a
    holdout, and asserts the delivered evaluation pair has training-matching
    feature columns and the strict-mode row count, and that the post-fit
    observation state covers all provided data.

    Parameters
    ----------
    forecaster : BaseReductionForecaster
        Reduction forecaster instance exposing ``validation_size``.
    y : pl.DataFrame
        Target time series with a ``"time"`` column.
    X_actual : pl.DataFrame or None, default=None
        Feature time series aligned with ``y``.
    X_future : pl.DataFrame or None, default=None
        Known future features.
    X_forecast : pl.DataFrame or None, default=None
        External forecasts.

    Raises
    ------
    AssertionError
        If no evaluation set reaches the stub, its shape or columns diverge
        from training, or the observation state stops short of the data end.

    """
    cloned = clone(forecaster)
    cloned.set_params(
        estimator=_stub_for(forecaster),
        validation_size=_CHECK_VALIDATION_SIZE,
        validation_overlap=False,
    )
    cloned.fit(
        y=y,
        X_actual=X_actual,
        forecasting_horizon=_CHECK_HORIZON,
        X_future=X_future,
        X_forecast=X_forecast,
    )

    n_groups = len(cloned.groups_) if cloned.groups_ else 1
    expected_rows = (_CHECK_VALIDATION_SIZE - _CHECK_HORIZON + 1) * n_groups

    estimators = cloned.estimator_ if isinstance(cloned.estimator_, list) else [cloned.estimator_]
    for est in estimators:
        assert est.received_eval_set_ is not None, "no eval_set reached the estimator"
        assert len(est.received_eval_set_) == 1
        X_eval, y_eval = est.received_eval_set_[0]
        assert len(X_eval) == expected_rows, f"expected {expected_rows} evaluation rows, got {len(X_eval)}"
        assert list(X_eval.columns) == list(est.train_X_.columns), (
            "evaluation feature columns must match training feature columns"
        )
        assert len(y_eval) == expected_rows

    last_time = y["time"][-1]
    observed = cloned.observed_time_
    if isinstance(observed, dict):
        assert all(t == last_time for t in observed.values()), "observation state must end at the data end"
    else:
        assert observed == last_time, "observation state must end at the data end"


def check_validation_holdout_default_noop(
    forecaster,
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None = None,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
) -> None:
    """Check that ``validation_size=None`` delivers no eval set at all.

    Parameters
    ----------
    forecaster : BaseReductionForecaster
        Reduction forecaster instance exposing ``validation_size``.
    y : pl.DataFrame
        Target time series with a ``"time"`` column.
    X_actual : pl.DataFrame or None, default=None
        Feature time series aligned with ``y``.
    X_future : pl.DataFrame or None, default=None
        Known future features.
    X_forecast : pl.DataFrame or None, default=None
        External forecasts.

    Raises
    ------
    AssertionError
        If an eval-set argument reaches the estimator without a holdout.

    """
    cloned = clone(forecaster)
    cloned.set_params(estimator=_stub_for(forecaster), validation_size=None)
    cloned.fit(
        y=y,
        X_actual=X_actual,
        forecasting_horizon=_CHECK_HORIZON,
        X_future=X_future,
        X_forecast=X_forecast,
    )
    estimators = cloned.estimator_ if isinstance(cloned.estimator_, list) else [cloned.estimator_]
    for est in estimators:
        assert est.received_eval_set_ is None, "validation_size=None must not deliver an eval_set"
