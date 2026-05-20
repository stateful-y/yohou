"""Utilities for model evaluation and (nested) cross-validation scoring."""

from __future__ import annotations

import numbers
import time
import warnings
from contextlib import suppress
from traceback import format_exc
from typing import Any

import numpy as np
import polars as pl
from joblib import logger
from sklearn.base import clone
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    process_routing,
)

from yohou.base import BaseForecaster
from yohou.metrics.base import BaseIntervalScorer, BaseScorer
from yohou.utils._compat import _check_method_params, _num_samples, _safe_split


def _split_X_forecast(
    X_forecast: pl.DataFrame | None,
    y: pl.DataFrame,
    train_indices: np.ndarray[Any, Any],
    test_indices: np.ndarray[Any, Any],
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Split X_forecast by vintage_time range for a CV fold.

    Training receives vintages where ``vintage_time <= cutoff_time``.
    Testing receives vintages where ``cutoff_time < vintage_time <= test_end_time``.
    The cutoff is the time corresponding to the last training index and
    the test end is the time corresponding to the last test index.

    Parameters
    ----------
    X_forecast : pl.DataFrame or None
        External forecasts with ``"vintage_time"`` and ``"time"`` columns.
        If ``None``, returns ``(None, None)`` immediately.
    y : pl.DataFrame
        Target time series with ``"time"`` column, used to derive
        cutoff and test end times.
    train_indices : ndarray
        Row indices of the training set.
    test_indices : ndarray
        Row indices of the test set.

    Returns
    -------
    tuple of (pl.DataFrame or None, pl.DataFrame or None)
        ``(X_forecast_train, X_forecast_test)``. Both are ``None``
        when ``X_forecast`` is ``None``.
    """
    if X_forecast is None:
        return None, None

    cutoff_time = y["time"][int(train_indices[-1])]
    test_end_time = y["time"][int(test_indices[-1])]

    X_forecast_train = X_forecast.filter(pl.col("vintage_time") <= cutoff_time)
    X_forecast_test = X_forecast.filter(
        (pl.col("vintage_time") > cutoff_time) & (pl.col("vintage_time") <= test_end_time)
    )
    return X_forecast_train, X_forecast_test


def _check_scoring(forecaster: BaseForecaster, scoring: object) -> BaseScorer | _MultimetricScorer:
    """Check the scoring parameter.

    In addition, multimetric scoring leverages a caching mechanism to not call the same
    estimator response method multiple times. Hence, the scorer is modified to only use
    a single response method given a list of response methods and the estimator.

    Parameters
    ----------
    forecaster : BaseForecaster
        The forecaster for which the scoring will be applied.

    scoring : list, tuple or dict
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        The possibilities are:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where they keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

        See [Multimetric Grid Search](https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html) for an example.

    Returns
    -------
    scorers_dict : dict
        A dict mapping each scorer name to its validated scorer.
    """
    if isinstance(scoring, BaseScorer):
        scorers = scoring

    elif isinstance(scoring, dict):
        keys = set(scoring)
        if not all(isinstance(k, str) for k in keys):
            raise ValueError(f"Non-string types were found in the keys of the given dict. scoring={scoring!r}")
        if len(keys) == 0:
            raise ValueError(f"An empty dict was passed. {scoring!r}")

        if not all(isinstance(v, BaseScorer) for v in scoring.values()):
            raise ValueError(f"Non-scorer types were found in the values of the given dict. scoring={scoring!r}")

        scorers = scoring  # Return the dict as-is

    else:
        raise ValueError(
            "Invalid scoring. It should be an instance of `BaseScorer` or "
            "a dict with strings as keys and instances of `BaseScorer` as "
            f"values. Got {scoring}."
        )

    return scorers  # ty: ignore[invalid-return-type]


class _MultimetricScorer:
    """Callable for multimetric scoring used to avoid repeated calls.

    Avoids repeated calls
    to `predict_proba`, `predict`, and `decision_function`.

    `_MultimetricScorer` will return a dictionary of scores corresponding to
    the scorers in the dictionary. Note that `_MultimetricScorer` can be
    created with a dictionary with one key  (i.e. only one actual scorer).

    Parameters
    ----------
    scorers : dict
        Dictionary mapping names to callable scorers.

    raise_exc : bool, default=True
        Whether to raise the exception in `__call__` or not. If set to `False`
        a formatted string of the exception details is passed as result of
        the failing scorer.
    """

    def __init__(self, *, scorers: dict[str, BaseScorer], raise_exc: bool = True) -> None:
        self._scorers = scorers
        self._raise_exc = raise_exc

    def fit(self, y: pl.DataFrame, *, forecaster=None) -> _MultimetricScorer:
        """Fit all scorers that have a fit method.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series used for fitting stateful scorers.
        forecaster : BaseForecaster or None, default=None
            The fitted forecaster instance, forwarded to scorers that
            need it (e.g. for residual computation).

        Returns
        -------
        self
        """
        for scorer in self._scorers.values():
            if hasattr(scorer, "fit"):
                scorer.fit(y, forecaster=forecaster)
        return self

    def __call__(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, **params: object) -> dict[str, float | str]:
        """Evaluate predicted target values."""
        scores: dict[str, float | str] = {}

        routed_params = process_routing(self, "score", **params)

        for name, scorer in self._scorers.items():
            try:
                params = routed_params.get(name)
                if params is None:
                    raise ValueError(f"Missing routing params for scorer '{name}'")
                scores[name] = scorer(y_truth, y_pred, **params.score)  # ty: ignore[invalid-assignment]
            except Exception as e:
                if self._raise_exc:
                    raise e
                else:
                    scores[name] = format_exc()
        return scores

    def get_metadata_routing(self) -> object:
        """Get metadata routing of this object.

        Please check [Metadata Routing User Guide](https://scikit-learn.org/stable/metadata_routing.html) on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A `MetadataRouter` encapsulating routing information.
        """
        router = MetadataRouter(owner=self)
        for name, scorer in self._scorers.items():
            router.add(
                **{name: scorer},
                method_mapping=MethodMapping().add(caller="score", callee="score"),
            )
        return router


def _fit_and_score(
    forecaster: BaseForecaster,
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None,
    forecasting_horizon: int,
    *,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
    scorer: BaseScorer | _MultimetricScorer | None,
    train: np.ndarray[Any, Any],
    test: np.ndarray[Any, Any],
    verbose: int,
    parameters: dict[str, object] | None,
    fit_params: dict[str, object] | None,
    predict_func_params: dict[str, object] | None,
    score_params: dict[str, object] | None,
    return_train_score: bool = False,
    return_parameters: bool = False,
    return_n_test_samples: bool = False,
    return_times: bool = False,
    return_forecaster: bool = False,
    return_predictions: bool = False,
    predict_forecasting_horizon: int | None = None,
    predict_stride: int | None = None,
    predict_method: str | None = None,
    split_progress: tuple[int, int] | None = None,
    candidate_progress: tuple[int, int] | None = None,
    error_score: float | str = np.nan,
    coverage_rates: list[float] | None = None,
) -> dict[str, object]:
    """Fit forecaster and compute scores for a given dataset split.

    Parameters
    ----------
    forecaster : BaseForecaster
        The forecaster to fit and evaluate.
    y : pl.DataFrame
        Target time series with a ``"time"`` column.
    X_actual : pl.DataFrame or None
        Actual feature observations with a ``"time"`` column, or
        ``None``.
    forecasting_horizon : int
        Number of time steps to forecast.
    X_future : pl.DataFrame or None, default=None
        Known future features with a ``"time"`` column.
    X_forecast : pl.DataFrame or None, default=None
        External forecasts with ``"vintage_time"`` and ``"time"``
        columns.
    scorer : BaseScorer, _MultimetricScorer, or None
        Scorer (single or multi-metric) used to evaluate predictions.
        A single scorer returns a float; a ``_MultimetricScorer`` returns
        a dict mapping scorer names to floats. When ``None``, scoring
        is skipped (useful with ``return_predictions=True``).
    train : np.ndarray
        Row indices of training samples.
    test : np.ndarray
        Row indices of test samples.
    verbose : int
        Verbosity level for progress messages.
    parameters : dict or None
        Hyperparameters to set on the forecaster via ``set_params``.
    fit_params : dict or None
        Routed metadata passed to ``forecaster.fit``.
    predict_func_params : dict or None
        Routed metadata passed to the prediction function
        (``observe_predict`` or ``observe_predict_interval``).
    score_params : dict or None
        Routed metadata passed to the scorer.
    return_train_score : bool, default=False
        Whether to include training scores in the result.
    return_parameters : bool, default=False
        Whether to include the evaluated parameters in the result.
    return_n_test_samples : bool, default=False
        Whether to include the number of test samples in the result.
    return_times : bool, default=False
        Whether to include fit and score wall-clock times in the result.
    return_forecaster : bool, default=False
        Whether to include the fitted forecaster in the result.
    return_predictions : bool, default=False
        Whether to include the predictions DataFrame in the result.
    predict_forecasting_horizon : int or None, default=None
        Override forecasting horizon for ``observe_predict``.
        ``None`` uses the forecaster's fit-time default.
    predict_stride : int or None, default=None
        Override stride for rolling ``observe_predict``.
        ``None`` uses the forecaster's default.
    predict_method : str or None, default=None
        Explicit prediction method (``"predict"``,
        ``"predict_interval"``, or ``"predict_class_proba"``).
        Used when ``scorer=None`` and ``return_predictions=True``.
    split_progress : tuple of (int, int) or None, default=None
        ``(current_split, total_splits)`` for verbose logging.
    candidate_progress : tuple of (int, int) or None, default=None
        ``(current_candidate, total_candidates)`` for verbose logging.
    error_score : float or "raise", default=np.nan
        Value to assign if an error occurs during fitting.  If ``"raise"``,
        the error is re-raised.
    coverage_rates : list of float or None, default=None
        Coverage levels for interval prediction (e.g. ``[0.9, 0.95]``).
        Passed to ``forecaster.fit`` when the scorer requires intervals.

    Returns
    -------
    result : dict
        Dictionary with the following keys (presence depends on flags):

        - ``"test_scores"`` - float or dict of scorer name to float.
        - ``"train_scores"`` - same format, only if *return_train_score*.
        - ``"n_test_samples"`` - int, only if *return_n_test_samples*.
        - ``"fit_time"`` - float (seconds), only if *return_times*.
        - ``"score_time"`` - float (seconds), only if *return_times*.
        - ``"parameters"`` - dict or None, only if *return_parameters*.
        - ``"forecaster"`` - fitted forecaster, only if *return_forecaster*.
        - ``"predictions"`` - pl.DataFrame, only if *return_predictions*.
        - ``"fit_error"`` - traceback string or ``None``.

    """
    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0] + 1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += f"; {candidate_progress[0] + 1}/{candidate_progress[1]}"

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    if verbose > 9:
        pass

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_method_params(y, params=fit_params, indices=train)
    score_params = score_params if score_params is not None else {}
    score_params_test = _check_method_params(y, params=score_params, indices=test)

    if parameters is not None:
        # here we clone the parameters, since sometimes the parameters
        # themselves might be estimators, e.g. when we search over different
        # estimators in a pipeline.
        # ref: https://github.com/scikit-learn/scikit-learn/pull/26786
        forecaster = forecaster.set_params(**clone(parameters, safe=False))

    start_time = time.time()

    y_train, X_actual_train = _safe_split(forecaster, y, X_actual, train)
    y_test, X_actual_test = _safe_split(forecaster, y, X_actual, test, train)
    X_forecast_train, X_forecast_test = _split_X_forecast(X_forecast, y, train, test)

    result: dict[str, object] = {}
    test_scores: dict[str, float | str] | float | str
    train_scores: dict[str, float | str] | float | str | None = None
    fit_time: float
    score_time: float
    try:
        if coverage_rates is not None:
            fit_params["coverage_rates"] = coverage_rates
        forecaster.fit(
            y=y_train,
            X_actual=X_actual_train,
            forecasting_horizon=forecasting_horizon,
            X_future=X_future,
            X_forecast=X_forecast_train,
            **fit_params,
        )

    except Exception:  # noqa: BLE001
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, _MultimetricScorer):
                test_scores = {name: float(error_score) for name in scorer._scorers}
                if return_train_score:
                    train_scores = {name: float(error_score) for name in scorer._scorers}
            elif scorer is not None:
                test_scores = float(error_score)
                if return_train_score:
                    train_scores = float(error_score)
        result["fit_error"] = format_exc()
    else:
        result["fit_error"] = None

        fit_time = time.time() - start_time

        # Need a scorer for _predict to resolve response method
        if scorer is not None:
            y_pred = _predict(
                forecaster,
                y_test,
                X_actual_test,
                scorer,
                predict_func_params=predict_func_params,
                predict_forecasting_horizon=predict_forecasting_horizon,
                predict_stride=predict_stride,
                coverage_rates=coverage_rates,
                X_future=X_future,
                X_forecast=X_forecast_test,
            )

            test_scores = _score(
                forecaster,
                y_train,
                y_test,
                y_pred,
                scorer,
                score_params_test,
                error_score,
            )
        elif return_predictions:
            y_pred = _predict(
                forecaster,
                y_test,
                X_actual_test,
                method=predict_method or "predict",
                predict_func_params=predict_func_params,
                predict_forecasting_horizon=predict_forecasting_horizon,
                predict_stride=predict_stride,
                coverage_rates=coverage_rates,
                X_future=X_future,
                X_forecast=X_forecast_test,
            )
        else:
            y_pred = None

        score_time = time.time() - start_time - fit_time

        if return_train_score:
            if scorer is None:
                raise ValueError("return_train_score requires a scorer.")
            # forecaster is stateful and needs to be rewound to predict the past
            score_params_train = _check_method_params(y, params=score_params, indices=train)
            train_rewind = train[: -len(test)]
            test_rewind = train[-len(test) :]
            y_train_rewind, X_actual_train_rewind = _safe_split(forecaster, y_train, X_actual_train, train_rewind)
            y_train_test, X_actual_train_test = _safe_split(
                forecaster, y_train, X_actual_train, test_rewind, train_rewind
            )
            forecaster.rewind(
                y_train_rewind, X_actual=X_actual_train_rewind, X_future=X_future, X_forecast=X_forecast_train
            )
            y_pred_train = _predict(
                forecaster,
                y_train_test,
                X_actual_train_test,
                scorer,
                predict_func_params=predict_func_params,
                predict_forecasting_horizon=predict_forecasting_horizon,
                predict_stride=predict_stride,
                coverage_rates=coverage_rates,
                X_future=X_future,
                X_forecast=X_forecast_train,
            )
            train_scores = _score(
                forecaster,
                y_train_rewind,
                y_train_test,
                y_pred_train,
                scorer,
                score_params_train,
                error_score,
            )

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2 and scorer is not None:
            if isinstance(test_scores, dict):
                for scorer_name in sorted(test_scores):
                    result_msg += f" {scorer_name}: ("
                    result_msg += f"test={test_scores[scorer_name]:.3f})"
            else:
                result_msg += ", score="
                result_msg += f"{test_scores:.3f}"
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg

    if scorer is not None:
        result["test_scores"] = test_scores
    if return_train_score:
        result["train_scores"] = train_scores
    if return_n_test_samples:
        result["n_test_samples"] = _num_samples(y_test)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_forecaster:
        result["forecaster"] = forecaster
    if return_predictions:
        result["predictions"] = y_pred
    return result


_RESPONSE_METHOD_PRIORITY: dict[str, int] = {
    "predict": 0,
    "predict_interval": 1,
    "predict_class_proba": 2,
}

_OBSERVE_METHOD_MAP: dict[str, str] = {
    "predict": "observe_predict",
    "predict_interval": "observe_predict_interval",
    "predict_class_proba": "observe_predict_class_proba",
}


def _get_response_methods(scorer: BaseScorer | _MultimetricScorer) -> set[str]:
    """Get the set of response methods needed by scorer(s)."""
    if isinstance(scorer, _MultimetricScorer):
        return {s._response_method for s in scorer._scorers.values()}  # ty: ignore[unresolved-attribute]
    return {scorer._response_method}  # ty: ignore[unresolved-attribute]


def _resolve_response_method(scorer: BaseScorer | _MultimetricScorer) -> str:
    """Resolve the single response method to use for prediction.

    When multiple scorers need different prediction types, the
    highest-priority (richest) method is chosen.
    """
    methods = _get_response_methods(scorer)
    return max(methods, key=lambda m: _RESPONSE_METHOD_PRIORITY[m])


# Backward-compat shims for yohou-optuna <= 0.1.0a1
def _needs_interval_predictions(scorer: BaseScorer | _MultimetricScorer) -> bool:
    """Check if any scorer requires interval predictions."""
    return "predict_interval" in _get_response_methods(scorer)


def _needs_point_predictions(scorer: BaseScorer | _MultimetricScorer) -> bool:
    """Check if any scorer requires point predictions."""
    return "predict" in _get_response_methods(scorer)


def _collect_coverage_rates(scorer: BaseScorer | _MultimetricScorer) -> list[float] | None:
    """Collect coverage rates from interval scorers.

    Returns the union of all coverage_rates across interval scorers,
    or None to let the forecaster use its fit-time defaults.
    """
    scorers: list[BaseScorer] = list(scorer._scorers.values()) if isinstance(scorer, _MultimetricScorer) else [scorer]
    all_rates: set[float] = set()
    for s in scorers:
        if isinstance(s, BaseIntervalScorer) and s.coverage_rates is not None:
            all_rates.update(s.coverage_rates)
    return sorted(all_rates) if all_rates else None


def _validate_forecaster_scorer_compatibility(
    forecaster: BaseForecaster, scorer: BaseScorer | _MultimetricScorer
) -> None:
    """Validate that forecaster and scorer types are compatible.

    Raises ``ValueError`` when:

    * An interval scorer is used with a point-only forecaster (which lacks
      ``predict_interval`` / ``observe_predict_interval``).
    * Point-only scorers are used with an interval-only forecaster (which
      lacks ``observe_predict``).

    Parameters
    ----------
    forecaster : BaseForecaster
        The forecaster to validate.
    scorer : BaseScorer or _MultimetricScorer
        The scorer(s) to validate against.

    Raises
    ------
    ValueError
        If the forecaster/scorer combination is incompatible.
    """
    tags = forecaster.__sklearn_tags__()
    forecaster_type = getattr(tags.forecaster_tags, "forecaster_type", None)

    methods = _get_response_methods(scorer)

    if "predict_interval" in methods and forecaster_type is not None and "interval" not in forecaster_type:
        raise ValueError(
            "Scorer requires interval predictions but forecaster "
            f"(type={forecaster_type!r}) does not support predict_interval. "
            "Use an interval forecaster or one that supports both point and interval."
        )

    if methods == {"predict"} and forecaster_type is not None and "point" not in forecaster_type:
        raise ValueError(
            f"Forecaster (type={forecaster_type!r}) does not support observe_predict "
            "required by point-only scorers. "
            "Use a forecaster that supports point predictions or interval scorers."
        )

    if "predict_class_proba" in methods and (forecaster_type is None or "class_proba" not in forecaster_type):
        raise ValueError(
            "Scorer requires class-probability predictions but forecaster "
            f"(type={forecaster_type!r}) does not support predict_class_proba. "
            "Use a class_proba-type forecaster."
        )

    if forecaster_type is not None and "class_proba" in forecaster_type and (methods - {"predict_class_proba"}):
        raise ValueError(
            f"Forecaster (type={forecaster_type!r}) does not support point or interval "
            "predictions required by the scorer. Use class-probability scorers."
        )


def _predict(
    forecaster: BaseForecaster,
    y_test: pl.DataFrame,
    X_actual_test: pl.DataFrame | None,
    scorer: BaseScorer | _MultimetricScorer | None = None,
    *,
    method: str | None = None,
    predict_func_params: dict[str, object] | None = None,
    predict_forecasting_horizon: int | None = None,
    predict_stride: int | None = None,
    coverage_rates: list[float] | None = None,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Produce predictions from a fitted forecaster.

    Resolves the response method from ``scorer`` (or uses ``method``
    directly), calls the appropriate ``observe_*`` method on the
    forecaster, deduplicates overlapping prediction windows, and
    returns a clean DataFrame.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster to generate predictions from.
    y_test : pl.DataFrame
        Test target time series with a ``"time"`` column.
    X_actual_test : pl.DataFrame or None
        Test actual feature observations, or ``None``.
    scorer : BaseScorer, _MultimetricScorer, or None, default=None
        Scorer used to determine which prediction method to call.
        Ignored when ``method`` is provided.
    method : str or None, default=None
        Explicit prediction method (``"predict"``,
        ``"predict_interval"``, or ``"predict_class_proba"``).
        When provided, overrides scorer-based resolution.
        One of ``scorer`` or ``method`` must be provided.
    predict_func_params : dict or None, default=None
        Routed metadata passed to the prediction function.
    predict_forecasting_horizon : int or None, default=None
        Forecasting horizon for ``observe_predict``.  ``None`` uses the
        forecaster's fit-time default.
    predict_stride : int or None, default=None
        Stride for rolling ``observe_predict``.  ``None`` uses the
        forecaster's default (equal to forecasting horizon).
    coverage_rates : list of float or None, default=None
        Coverage rates for interval predictions.  When ``None``,
        rates are collected from the scorer automatically (if a
        scorer is provided).
    X_future : pl.DataFrame or None, default=None
        Known future features with a ``"time"`` column.
    X_forecast : pl.DataFrame or None, default=None
        External forecasts with ``"vintage_time"`` and ``"time"``
        columns.

    Returns
    -------
    pl.DataFrame
        Predictions deduplicated by ``"time"`` (keeping the last
        occurrence) and sorted by ``"time"``.
    """
    predict_func_params = {} if predict_func_params is None else dict(predict_func_params)

    # Resolve which forecaster method to call
    if method is not None:
        response_method = method
    elif scorer is not None:
        response_method = _resolve_response_method(scorer)
    else:
        raise ValueError("Either scorer or method must be provided to _predict.")
    observe_method = _OBSERVE_METHOD_MAP[response_method]

    # Inject optional predict params
    if predict_forecasting_horizon is not None:
        predict_func_params["forecasting_horizon"] = predict_forecasting_horizon
    if predict_stride is not None:
        predict_func_params["stride"] = predict_stride

    if response_method == "predict_interval":
        coverage_rates_for_predict = (
            coverage_rates
            if coverage_rates is not None
            else (_collect_coverage_rates(scorer) if scorer is not None else None)
        )
        y_pred = getattr(forecaster, observe_method)(
            y_test,
            X_actual_test,
            coverage_rates=coverage_rates_for_predict,
            X_future=X_future,
            X_forecast=X_forecast,
            **predict_func_params,
        )
    else:
        y_pred = getattr(forecaster, observe_method)(
            y_test,
            X_actual_test,
            X_future=X_future,
            X_forecast=X_forecast,
            **predict_func_params,
        )

    # observe_predict produces overlapping prediction windows when the last
    # observation chunk is smaller than stride. Deduplicate (keeping the most
    # recently informed prediction) and sort so downstream scorers receive a
    # clean, monotonically increasing time column.
    # TODO: Address this formally in scorers
    return y_pred.unique(subset=["time"], keep="last").sort("time")


def _score(
    forecaster: BaseForecaster,
    y_train: pl.DataFrame,
    y_test: pl.DataFrame,
    y_pred: pl.DataFrame,
    scorer: BaseScorer | _MultimetricScorer,
    score_params: dict[str, object] | None,
    error_score: str | float = "raise",
) -> float | dict[str, float | str] | str:
    """Compute the score(s) of a forecaster from pre-computed predictions.

    Will return a dict of floats if ``scorer`` is a ``_MultiMetricScorer``,
    otherwise a single float is returned.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster (used by stateful scorers that need it).
    y_train : pl.DataFrame
        Training target time series (used by scorers like MASE).
    y_test : pl.DataFrame
        Test target time series.
    y_pred : pl.DataFrame
        Predictions produced by ``_predict``.
    scorer : BaseScorer or _MultimetricScorer
        Scorer used to evaluate predictions.
    score_params : dict or None
        Routed metadata for the scorer.
    error_score : float or "raise", default="raise"
        Value to assign if scoring fails.
    """
    score_params = {} if score_params is None else score_params

    scores: float | dict[str, float | str] | str
    try:
        # Only fit scorer if it has a fit method (stateful scorers)
        if hasattr(scorer, "fit"):
            scorer.fit(y_train, forecaster=forecaster)
        scores = scorer(y_test, y_pred, **score_params)  # ty: ignore[invalid-assignment]

    except Exception:  # noqa: BLE001
        if isinstance(scorer, _MultimetricScorer):
            # If `_MultimetricScorer` raises exception, the `error_score`
            # parameter is equal to "raise".
            raise
        elif error_score == "raise":
            raise
        else:
            scores = str(error_score) if isinstance(error_score, str) else float(error_score)
            warnings.warn(
                (
                    "Scoring failed. The score on this train-test partition for "
                    f"these parameters will be set to {error_score}. Details: \n"
                    f"{format_exc()}"
                ),
                UserWarning,
                stacklevel=2,
            )

    # Check non-raised error messages in `_MultimetricScorer`
    if isinstance(scorer, _MultimetricScorer) and isinstance(scores, dict):
        exception_messages = [(name, str_e) for name, str_e in scores.items() if isinstance(str_e, str)]
        if exception_messages:
            # error_score != "raise"
            for name, _ in exception_messages:
                scores[name] = float(error_score) if isinstance(error_score, numbers.Number) else error_score

            details = "\n".join(f"{name}: {e}" for name, e in exception_messages)
            warnings.warn(
                (
                    "Scoring failed. The score on this train-test partition for "
                    f"these parameters will be set to {error_score}. Details: \n"
                    f"{details}"
                ),
                UserWarning,
                stacklevel=2,
            )
    error_msg = "scoring must return a number, got %s (%s) instead. (scorer=%s)"

    # Check for DataFrame return (aggregation_method != "all" scorers)

    if isinstance(scores, pl.DataFrame):
        msg = (
            "Scorers with aggregation_method != 'all' cannot be used with SearchCV. "
            "SearchCV requires scalar scores for optimization. "
            "Please set aggregation_method='all' on your scorer."
        )
        raise ValueError(msg)

    if isinstance(scores, dict):
        for name, score in scores.items():
            if isinstance(score, str):
                continue  # Already error string
            # Check for DataFrame in dict values
            if isinstance(score, pl.DataFrame):
                msg = (
                    f"Scorer '{name}' with aggregation_method != 'all' "
                    "cannot be used with SearchCV. "
                    "SearchCV requires scalar scores for optimization. "
                    f"Please set aggregation_method='all' on scorer '{name}'."
                )
                raise ValueError(msg)
            score_val: float | str = score
            item_method = getattr(score_val, "item", None)
            if item_method is not None:
                with suppress(ValueError):
                    # e.g. unwrap memmapped scalars
                    score_val = item_method()
            if not isinstance(score_val, numbers.Number):
                raise ValueError(error_msg % (score_val, type(score_val), name))
            scores[name] = float(score_val) if isinstance(score_val, int | float | numbers.Number) else score_val

        # Negate scores for lower_is_better scorers (sklearn sign convention).
        # This ensures rankdata(-scores) in _format_results correctly assigns
        # rank 1 to the best performer for both error metrics (lower raw value
        # is better) and score metrics (higher raw value is better).
        if isinstance(scorer, _MultimetricScorer):
            for name, individual_scorer in scorer._scorers.items():
                if name in scores and isinstance(scores, dict):
                    score_val = scores[name]
                    if isinstance(score_val, int | float):
                        scorer_tags = individual_scorer.__sklearn_tags__()
                        if scorer_tags.scorer_tags is not None and scorer_tags.scorer_tags.lower_is_better:
                            scores[name] = -score_val
    else:  # scalar
        if isinstance(scores, str):
            return scores  # Already error string
        item_method = getattr(scores, "item", None)
        if item_method is not None:
            with suppress(ValueError):
                # e.g. unwrap memmapped scalars
                scores = item_method()
        if not isinstance(scores, numbers.Number):
            raise ValueError(error_msg % (scores, type(scores), scorer))
        scores = float(scores)

        # Negate score for lower_is_better scorers (sklearn sign convention).
        # Error metrics (MAE, RMSE, etc.) return positive values where lower
        # is better. Negating them makes higher == better, so downstream
        # ranking via rankdata(-scores) selects the lowest-error candidate.
        if not isinstance(scorer, _MultimetricScorer):
            scorer_tags = scorer.__sklearn_tags__()
            if scorer_tags.scorer_tags is not None and scorer_tags.scorer_tags.lower_is_better:
                scores = -scores
    return scores
