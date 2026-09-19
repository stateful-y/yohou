"""Utilities for model evaluation and (nested) cross-validation scoring."""

from __future__ import annotations

import numbers
import time
import warnings
from contextlib import suppress
from dataclasses import dataclass
from traceback import format_exc
from typing import Any, NamedTuple, cast

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


def _check_scoring(scoring: object) -> BaseScorer | dict[str, BaseScorer]:
    """Check the scoring parameter.

    Parameters
    ----------
    scoring : BaseScorer or dict of {str: BaseScorer}
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        The possibilities are:

        - a single ``BaseScorer`` instance;
        - a dictionary mapping scorer names (``str``) to ``BaseScorer``
          instances for multimetric scoring.

    Returns
    -------
    BaseScorer or dict of {str: BaseScorer}
        The single scorer is returned unchanged; a dict is validated and
        returned as-is.
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

        scorers = cast(dict[str, BaseScorer], scoring)  # Return the dict as-is

    else:
        raise ValueError(
            "Invalid scoring. It should be an instance of `BaseScorer` or "
            "a dict with strings as keys and instances of `BaseScorer` as "
            f"values. Got {scoring}."
        )

    return scorers


class _MultimetricScorer:
    """Callable for multimetric scoring used to avoid repeated calls.

    Avoids repeated calls to ``observe_predict``, ``observe_predict_interval``,
    and ``observe_predict_class_proba`` by caching the single prediction call
    shared across all scorers.

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
    fold = _fit_fold(
        forecaster,
        y,
        X_actual,
        forecasting_horizon,
        X_future=X_future,
        X_forecast=X_forecast,
        scorer=scorer,
        train=train,
        test=test,
        verbose=verbose,
        parameters=parameters,
        fit_params=fit_params,
        score_params=score_params,
        return_train_score=return_train_score,
        split_progress=split_progress,
        candidate_progress=candidate_progress,
        error_score=error_score,
        coverage_rates=coverage_rates,
    )
    return _score_fold(
        fold,
        X_future=X_future,
        scorer=scorer,
        verbose=verbose,
        predict_func_params=predict_func_params,
        return_train_score=return_train_score,
        return_parameters=return_parameters,
        return_n_test_samples=return_n_test_samples,
        return_times=return_times,
        return_forecaster=return_forecaster,
        return_predictions=return_predictions,
        predict_forecasting_horizon=predict_forecasting_horizon,
        predict_stride=predict_stride,
        predict_method=predict_method,
        error_score=error_score,
        coverage_rates=coverage_rates,
    )


@dataclass
class _FoldFit:
    """The outcome of fitting one CV fold, carried to its scoring.

    Carries a fitted fold's state from `_fit_fold` to `_score_fold`.

    Notes
    -----
    `_fit_and_score` fits and scores a fold in one call. A shared-round search
    fits every fold of a candidate before scoring any of them, so the two
    halves are separate functions and this record passes between them.

    """

    forecaster: BaseForecaster
    y: pl.DataFrame
    train: np.ndarray[Any, Any]
    test: np.ndarray[Any, Any]
    y_train: pl.DataFrame
    X_actual_train: pl.DataFrame | None
    y_test: pl.DataFrame
    X_actual_test: pl.DataFrame | None
    X_forecast_train: pl.DataFrame | None
    X_forecast_test: pl.DataFrame | None
    parameters: dict[str, object] | None
    score_params: dict[str, object]
    score_params_test: dict[str, object]
    progress_msg: str
    params_msg: str
    fit_time: float
    fit_error: str | None
    test_scores: dict[str, float | str] | float | str | None = None
    train_scores: dict[str, float | str] | float | str | None = None


def _fit_fold(
    forecaster: BaseForecaster,
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None,
    forecasting_horizon: int,
    *,
    X_future: pl.DataFrame | None,
    X_forecast: pl.DataFrame | None,
    scorer: BaseScorer | _MultimetricScorer | None,
    train: np.ndarray[Any, Any],
    test: np.ndarray[Any, Any],
    verbose: int,
    parameters: dict[str, object] | None,
    fit_params: dict[str, object] | None,
    score_params: dict[str, object] | None,
    return_train_score: bool,
    split_progress: tuple[int, int] | None,
    candidate_progress: tuple[int, int] | None,
    error_score: float | str,
    coverage_rates: list[float] | None,
    validation_window: bool = False,
    extra_fit_params: dict[str, object] | None = None,
) -> _FoldFit:
    """Fit the forecaster on one CV fold's training rows.

    Parameters
    ----------
    forecaster : BaseForecaster
        The forecaster to fit.
    y : pl.DataFrame
        Target time series with a ``"time"`` column.
    X_actual : pl.DataFrame or None
        Actual feature observations, or ``None``.
    forecasting_horizon : int
        Number of time steps to forecast.
    X_future : pl.DataFrame or None
        Known future features.
    X_forecast : pl.DataFrame or None
        External forecasts with ``"vintage_time"`` and ``"time"`` columns.
    scorer : BaseScorer, _MultimetricScorer, or None
        Scorer, used here only to shape error scores.
    train : np.ndarray
        Row indices of training samples.
    test : np.ndarray
        Row indices of test samples.
    verbose : int
        Verbosity level.
    parameters : dict or None
        Hyperparameters to set on the forecaster via ``set_params``.
    fit_params : dict or None
        Routed metadata passed to ``forecaster.fit``.
    score_params : dict or None
        Routed metadata passed to the scorer.
    return_train_score : bool
        Whether train scores will be computed, which shapes error scores.
    split_progress : tuple of (int, int) or None
        ``(current_split, total_splits)`` for verbose logging.
    candidate_progress : tuple of (int, int) or None
        ``(current_candidate, total_candidates)`` for verbose logging.
    error_score : float or "raise"
        Value to assign if fitting fails, or ``"raise"``.
    coverage_rates : list of float or None
        Coverage levels passed to ``forecaster.fit``.
    validation_window : bool, default=False
        Whether to pass the fold's test rows to ``fit`` as ``y_val``,
        ``X_actual_val``, and ``X_forecast_val``.
    extra_fit_params : dict or None, default=None
        Further keyword arguments for ``forecaster.fit``, such as an
        early-stopping adapter's callbacks.

    Returns
    -------
    _FoldFit
        The fitted fold, or a record of its fit error.

    Raises
    ------
    ValueError
        If ``error_score`` is neither numeric nor ``"raise"``.
    Exception
        The fit error, when ``error_score="raise"``.

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

    params_msg = _params_message(parameters) if verbose > 1 else ""

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

    fold = _FoldFit(
        forecaster=forecaster,
        y=y,
        train=train,
        test=test,
        y_train=y_train,
        X_actual_train=X_actual_train,
        y_test=y_test,
        X_actual_test=X_actual_test,
        X_forecast_train=X_forecast_train,
        X_forecast_test=X_forecast_test,
        parameters=parameters,
        score_params=score_params,
        score_params_test=score_params_test,
        progress_msg=progress_msg,
        params_msg=params_msg,
        fit_time=0.0,
        fit_error=None,
    )
    try:
        if coverage_rates is not None:
            fit_params["coverage_rates"] = coverage_rates
        if extra_fit_params:
            fit_params = {**fit_params, **extra_fit_params}
        if validation_window:
            fit_params = {
                **fit_params,
                "y_val": y_test,
                "X_actual_val": X_actual_test,
                "X_forecast_val": X_forecast_test,
            }
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
        fold.fit_time = time.time() - start_time
        if error_score == "raise":
            raise
        _record_fit_error(fold, format_exc(), scorer, error_score, return_train_score)
    else:
        fold.fit_time = time.time() - start_time
    return fold


def _record_fit_error(
    fold: _FoldFit,
    traceback: str,
    scorer: BaseScorer | _MultimetricScorer | None,
    error_score: float | str,
    return_train_score: bool,
) -> None:
    """Mark a fold as failed and fill its scores with ``error_score``.

    Parameters
    ----------
    fold : _FoldFit
        The fold to mark. Mutated in place.
    traceback : str
        The formatted traceback of the failure.
    scorer : BaseScorer, _MultimetricScorer, or None
        Scorer, which shapes the error scores.
    error_score : float or "raise"
        The numeric error score.
    return_train_score : bool
        Whether train scores are reported.

    """
    fold.fit_error = traceback
    if isinstance(error_score, numbers.Number):
        if isinstance(scorer, _MultimetricScorer):
            fold.test_scores = {name: float(error_score) for name in scorer._scorers}
            if return_train_score:
                fold.train_scores = {name: float(error_score) for name in scorer._scorers}
        elif scorer is not None:
            fold.test_scores = float(error_score)
            if return_train_score:
                fold.train_scores = float(error_score)


def _score_fold(
    fold: _FoldFit,
    *,
    X_future: pl.DataFrame | None,
    scorer: BaseScorer | _MultimetricScorer | None,
    verbose: int,
    predict_func_params: dict[str, object] | None,
    return_train_score: bool,
    return_parameters: bool,
    return_n_test_samples: bool,
    return_times: bool,
    return_forecaster: bool,
    return_predictions: bool,
    predict_forecasting_horizon: int | None,
    predict_stride: int | None,
    predict_method: str | None,
    error_score: float | str,
    coverage_rates: list[float] | None,
) -> dict[str, object]:
    """Score a fitted fold and assemble its `_fit_and_score` result.

    Parameters
    ----------
    fold : _FoldFit
        The fold returned by `_fit_fold`.
    X_future : pl.DataFrame or None
        Known future features.
    scorer : BaseScorer, _MultimetricScorer, or None
        Scorer (single or multi-metric).
    verbose : int
        Verbosity level.
    predict_func_params : dict or None
        Routed metadata passed to the prediction function.
    return_train_score, return_parameters, return_n_test_samples, return_times, return_forecaster, return_predictions : bool
        Which entries the result carries; see `_fit_and_score`.
    predict_forecasting_horizon : int or None
        Override forecasting horizon for ``observe_predict``.
    predict_stride : int or None
        Override stride for rolling ``observe_predict``.
    predict_method : str or None
        Explicit prediction method when ``scorer`` is None.
    error_score : float or "raise"
        Value assigned to a score that cannot be computed.
    coverage_rates : list of float or None
        Coverage levels for interval prediction.

    Returns
    -------
    dict
        The `_fit_and_score` result for this fold.

    Raises
    ------
    ValueError
        If ``return_train_score`` is True and ``scorer`` is None.

    """
    forecaster = fold.forecaster
    y_train, X_actual_train = fold.y_train, fold.X_actual_train
    y_test, X_actual_test = fold.y_test, fold.X_actual_test
    X_forecast_train, X_forecast_test = fold.X_forecast_train, fold.X_forecast_test
    train, test = fold.train, fold.test

    result: dict[str, object] = {"fit_error": fold.fit_error}
    test_scores = fold.test_scores
    train_scores = fold.train_scores
    y_pred: pl.DataFrame | None = None
    fit_time = fold.fit_time
    score_time = 0.0

    if fold.fit_error is None:
        score_start = time.time()

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
                fold.score_params_test,
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

        score_time = time.time() - score_start

        if return_train_score:
            if scorer is None:
                raise ValueError("return_train_score requires a scorer.")
            window = _train_window_predictions(
                forecaster,
                y_train,
                X_actual_train,
                n_rows=len(test),
                scorer=scorer,
                predict_func_params=predict_func_params,
                predict_forecasting_horizon=predict_forecasting_horizon,
                predict_stride=predict_stride,
                coverage_rates=coverage_rates,
                X_future=X_future,
                X_forecast_train=X_forecast_train,
            )
            train_scores = _score_train_window(
                forecaster,
                window,
                scorer,
                y=fold.y,
                score_params=fold.score_params,
                train=train,
                error_score=error_score,
            )

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{fold.progress_msg}] END "
        result_msg = fold.params_msg + (";" if fold.params_msg else "")
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
        print(end_msg)  # noqa: T201

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
        result["parameters"] = fold.parameters
    if return_forecaster:
        result["forecaster"] = forecaster
    if return_predictions:
        result["predictions"] = y_pred
    return result


def _select_shared_rounds(
    curves: dict[str, list[tuple[np.ndarray, bool]]],
) -> tuple[dict[str, int], dict[str, bool]]:
    """Choose one boosting round per estimator position from the folds' stopping curves.

    For each position, the curves of every fold are averaged over the rounds
    every fold trained (1 to the shortest curve's length), and the best round of
    that average is chosen, the smallest on ties. A fold cannot be scored at a
    round it did not train, which is why the average stops at the shortest
    curve.

    Parameters
    ----------
    curves : dict of {str: list of tuple of (ndarray, bool)}
        For each position key, one ``(curve, higher_is_better)`` pair per fold.

    Returns
    -------
    rounds : dict of str to int
        The chosen round (1-based) per position, in ``curves`` order.
    at_boundary : dict of str to bool
        Per position, whether the chosen round is the last round every fold
        trained, so a later round might have been better.

    Raises
    ------
    ValueError
        If a position has no fold, a curve is empty, or the folds disagree on
        whether higher values are better.

    """
    rounds: dict[str, int] = {}
    at_boundary: dict[str, bool] = {}
    for position, fold_curves in curves.items():
        if not fold_curves:
            raise ValueError(f"Cannot choose a shared round for {position!r}: no fold has a stopping curve for it.")
        directions = {bool(higher) for _, higher in fold_curves}
        if len(directions) != 1:
            raise ValueError(
                f"Cannot choose a shared round for {position!r}: the folds disagree on the stopping "
                f"metric's direction (higher is better in some folds, lower in others)."
            )
        shortest = min(len(curve) for curve, _ in fold_curves)
        if shortest == 0:
            raise ValueError(f"Cannot choose a shared round for {position!r}: a fold's stopping curve is empty.")
        mean = np.mean(np.vstack([np.asarray(curve, dtype=float)[:shortest] for curve, _ in fold_curves]), axis=0)
        best = int(np.nanargmax(mean) if directions.pop() else np.nanargmin(mean)) + 1
        rounds[position] = best
        at_boundary[position] = best == shortest
    return rounds, at_boundary


def _merge_fit_params(fit_params: dict[str, object], extra: dict[str, object]) -> dict[str, object]:
    """Add an adapter's fit parameters to the caller's.

    Parameters
    ----------
    fit_params : dict
        The caller's routed fit parameters.
    extra : dict
        The adapter's fit parameters.

    Returns
    -------
    dict
        The merged parameters. A key present in both is allowed only when both
        values are lists, which are concatenated (for example callbacks).

    Raises
    ------
    ValueError
        If a key is present in both and the values are not both lists.

    """
    merged = dict(fit_params)
    for key, value in extra.items():
        if key not in merged:
            merged[key] = value
        elif isinstance(merged[key], list) and isinstance(value, list):
            merged[key] = [*cast(list, merged[key]), *value]
        else:
            raise ValueError(
                f"The fit parameter {key!r} is set both by the caller and by the early-stopping adapter; "
                f"only list values (such as callbacks) can be combined."
            )
    return merged


def _check_shared_round_forecaster_type(forecaster: BaseForecaster) -> None:
    """Reject a forecaster class that ``validation="cv"`` cannot use.

    Parameters
    ----------
    forecaster : BaseForecaster
        The forecaster passed to the search.

    Raises
    ------
    ValueError
        If the forecaster is not a reduction forecaster.

    """
    from yohou.base.reduction import BaseReductionForecaster

    if not isinstance(forecaster, BaseReductionForecaster):
        raise ValueError(
            f"validation='cv' requires a reduction forecaster (PointReductionForecaster, "
            f"IntervalReductionForecaster, or ClassProbaReductionForecaster), whose boosted estimators "
            f"receive each fold's test window as their evaluation set; got {forecaster.__class__.__name__}."
        )


def _check_shared_round_forecaster(forecaster: BaseForecaster) -> None:
    """Reject a forecaster configuration that ``validation="cv"`` cannot use.

    Parameters
    ----------
    forecaster : BaseForecaster
        The forecaster with a candidate's parameters set.

    Raises
    ------
    ValueError
        If the forecaster is not a reduction forecaster, has ``validation_size``
        set, or uses the ``"dir-rec"`` strategy.

    """
    _check_shared_round_forecaster_type(forecaster)
    validation_size = getattr(forecaster, "validation_size", None)
    if validation_size is not None:
        raise ValueError(
            f"validation='cv' and validation_size={validation_size} both supply the evaluation set: the "
            f"search uses each fold's test window, the forecaster would hold out its own tail. Set "
            f"validation_size=None, or use validation=None to keep the forecaster's holdout."
        )
    if getattr(forecaster, "reduction_strategy", None) == "dir-rec":
        raise ValueError(
            "validation='cv' cannot use reduction_strategy='dir-rec': each later step is trained on the "
            "earlier steps' predictions, so cutting an earlier step to its chosen round would change a "
            "later step's inputs between training and prediction. Use 'direct' or 'multi-output'."
        )


def _params_message(parameters: dict[str, object] | None) -> str:
    """Format candidate parameters for verbose progress lines, sorted by key."""
    if parameters is None:
        return ""
    return ", ".join(f"{k}={parameters[k]}" for k in sorted(parameters))


def _evaluate_candidate_shared_rounds(
    forecaster: BaseForecaster,
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None,
    forecasting_horizon: int,
    *,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
    splits: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]],
    parameters: dict[str, object] | None,
    early_stopping_adapter: Any = None,
    scorer: BaseScorer | _MultimetricScorer,
    verbose: int,
    fit_params: dict[str, object] | None,
    predict_func_params: dict[str, object] | None,
    score_params: dict[str, object] | None,
    return_train_score: bool = False,
    return_parameters: bool = False,
    return_n_test_samples: bool = False,
    return_times: bool = False,
    candidate_progress: tuple[int, int] | None = None,
    error_score: float | str = np.nan,
    coverage_rates: list[float] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Evaluate one candidate with early stopping on each fold's test window and a shared round.

    Every fold is fitted with its test window as the evaluation set and with
    the adapter's configuration that keeps every trained round. One round per
    estimator position is then chosen from the fold-average stopping curve
    (`_select_shared_rounds`), every successful fold's estimators are cut to
    those rounds, and each fold is predicted and scored as `_fit_and_score`
    would. A fold whose fit fails gets ``error_score`` and contributes no curve.

    Parameters
    ----------
    forecaster : BaseForecaster
        The unfitted reduction forecaster (not mutated; each fold uses a clone).
    y : pl.DataFrame
        Target time series with a ``"time"`` column.
    X_actual : pl.DataFrame or None
        Actual feature observations, or ``None``.
    forecasting_horizon : int
        Number of time steps to forecast.
    X_future : pl.DataFrame or None, default=None
        Known future features.
    X_forecast : pl.DataFrame or None, default=None
        External forecasts with ``"vintage_time"`` and ``"time"`` columns.
    splits : list of tuple of np.ndarray
        ``(train, test)`` row indices for every fold.
    parameters : dict or None
        The candidate's hyperparameters.
    early_stopping_adapter : BaseEarlyStoppingAdapter or None, default=None
        The adapter to use, or None to resolve a built-in one.
    scorer : BaseScorer or _MultimetricScorer
        Scorer (single or multi-metric).
    verbose : int
        Verbosity level.
    fit_params : dict or None
        Routed metadata passed to ``forecaster.fit``.
    predict_func_params : dict or None
        Routed metadata passed to the prediction function.
    score_params : dict or None
        Routed metadata passed to the scorer.
    return_train_score, return_parameters, return_n_test_samples, return_times : bool
        Which entries each fold's result carries; see `_fit_and_score`.
    candidate_progress : tuple of (int, int) or None, default=None
        ``(current_candidate, total_candidates)`` for verbose logging.
    error_score : float or "raise", default=np.nan
        Value assigned to a failed fold, or ``"raise"``.
    coverage_rates : list of float or None, default=None
        Coverage levels for interval prediction.

    Returns
    -------
    results : list of dict
        One `_fit_and_score` result per fold, in ``splits`` order.
    record : dict
        ``"rounds"`` (dict of position key to chosen round),
        ``"rounds_at_boundary"`` (bool), ``"boundary_positions"`` (list of
        position keys), and ``"curve_lengths"`` (one dict of position key to
        curve length per fold, None for a failed fold).

    Raises
    ------
    ValueError
        If the candidate's configuration cannot be used with
        ``validation="cv"``, raised before any fold is fitted.

    """
    from yohou.model_selection.early_stopping import (
        _eval_target,
        _replace_eval_target,
        _resolve_early_stopping_adapter,
    )

    configured = clone(forecaster)
    if parameters:
        configured.set_params(**clone(parameters, safe=False))
    _check_shared_round_forecaster(configured)
    adapter = _resolve_early_stopping_adapter(configured.estimator, early_stopping_adapter)
    adapter.validate(_eval_target(configured.estimator))

    folds: list[_FoldFit] = []
    curve_lengths: list[dict[str, int] | None] = []
    curves: dict[str, list[tuple[np.ndarray, bool]]] = {}
    for split_idx, (train, test) in enumerate(splits):
        fold_forecaster = clone(configured)
        prepared, adapter_fit_params = adapter.prepare_fold_fit(_eval_target(fold_forecaster.estimator))
        fold_forecaster.set_params(estimator=_replace_eval_target(fold_forecaster.estimator, prepared))
        routed = dict(fit_params or {})
        # Adapter parameters bypass per-sample slicing; a key the caller also
        # routes (such as callbacks) is combined here and overrides theirs.
        shared = {k: v for k, v in routed.items() if k in adapter_fit_params}
        fold = _fit_fold(
            fold_forecaster,
            y,
            X_actual,
            forecasting_horizon,
            X_future=X_future,
            X_forecast=X_forecast,
            scorer=scorer,
            train=train,
            test=test,
            verbose=verbose,
            parameters=None,
            fit_params=routed,
            score_params=score_params,
            return_train_score=return_train_score,
            split_progress=(split_idx, len(splits)),
            candidate_progress=candidate_progress,
            error_score=error_score,
            coverage_rates=coverage_rates,
            validation_window=True,
            extra_fit_params=_merge_fit_params(shared, adapter_fit_params),
        )
        fold.parameters = parameters
        fold.params_msg = _params_message(parameters) if verbose > 1 else ""
        folds.append(fold)
        if fold.fit_error is not None:
            curve_lengths.append(None)
            continue
        lengths: dict[str, int] = {}
        for position, estimator in fold.forecaster._fitted_estimator_positions():  # ty: ignore[unresolved-attribute]
            curve, higher_is_better = adapter.stopping_curve(estimator)
            curves.setdefault(position, []).append((curve, higher_is_better))
            lengths[position] = len(curve)
        curve_lengths.append(lengths)

    rounds: dict[str, int] = {}
    boundary: dict[str, bool] = {}
    if curves:
        rounds, boundary = _select_shared_rounds(curves)
        for fold in folds:
            if fold.fit_error is None:
                for position, estimator in fold.forecaster._fitted_estimator_positions():  # ty: ignore[unresolved-attribute]
                    adapter.truncate(estimator, rounds[position])

    # No warning here: this runs inside a joblib worker, where warnings do not
    # reach the caller. The search warns from the returned record.
    boundary_positions = [position for position, flagged in boundary.items() if flagged]

    results = [
        _score_fold(
            fold,
            X_future=X_future,
            scorer=scorer,
            verbose=verbose,
            predict_func_params=predict_func_params,
            return_train_score=return_train_score,
            return_parameters=return_parameters,
            return_n_test_samples=return_n_test_samples,
            return_times=return_times,
            return_forecaster=False,
            return_predictions=False,
            predict_forecasting_horizon=None,
            predict_stride=None,
            predict_method=None,
            error_score=error_score,
            coverage_rates=coverage_rates,
        )
        for fold in folds
    ]
    record: dict[str, object] = {
        "rounds": rounds,
        "rounds_at_boundary": bool(boundary_positions),
        "boundary_positions": boundary_positions,
        "curve_lengths": curve_lengths,
    }
    return results, record


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


# Intentional backward-compat shims for yohou-optuna <= 0.1.0a1. These thin
# wrappers over _get_response_methods are not used elsewhere in the main source
# tree, but they are a public-facing compatibility surface relied on by the
# yohou-optuna integration and its tests, so they are kept deliberately.
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
    * Any point scorer, alone or in a scoring dict with interval scorers, is
      used with a forecaster that has no point predictions (which lacks
      ``observe_predict``). Folds would otherwise be scored from interval
      predictions, which carry no point column, and every fold would fail.

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

    # A scoring dict mixing point and interval scorers resolves to interval
    # predictions, which carry no point column: without this check every fold
    # of a forecaster lacking point predictions fails while scoring.
    if (
        "predict" in methods
        and "predict_interval" in methods
        and forecaster_type is not None
        and "point" not in forecaster_type
        and "class_proba" not in forecaster_type
    ):
        point_scorers = sorted(
            name
            for name, single in getattr(scorer, "_scorers", {}).items()
            if getattr(single, "_response_method", None) == "predict"
        )
        raise ValueError(
            f"Forecaster (type={forecaster_type!r}) does not support observe_predict "
            f"required by the point scorers {point_scorers}. "
            "Use a forecaster that supports point predictions, or only interval scorers."
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
        Predictions deduplicated on ``("vintage_time", "time")`` when a
        ``"vintage_time"`` column is present (otherwise on ``"time"``),
        keeping the last occurrence per unique key, and sorted by
        ``("time", "vintage_time")``. Rows that share a ``"time"`` but
        differ in ``"vintage_time"`` are all preserved.
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
    # clean, monotonically increasing time column. Deduplicate on the full
    # vintage key so rows that differ only in vintage_time are preserved.
    # TODO: Address this formally in scorers
    dedup_subset = ["vintage_time", "time"] if "vintage_time" in y_pred.columns else ["time"]
    sort_cols = [col for col in ("time", "vintage_time") if col in y_pred.columns]
    return y_pred.unique(subset=dedup_subset, keep="last").sort(sort_cols)


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

    Returns a float for single scorers, or a dict of ``{name: float}`` if
    ``scorer`` is a ``_MultimetricScorer``.  When ``error_score != "raise"``
    and scoring raises an exception, a string traceback is returned instead
    (or stored per-metric for the multimetric case).

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
            scores[name] = float(score_val)

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


class _TrainWindow(NamedTuple):
    """Predictions over the stretch of a training window that a train score covers.

    Attributes
    ----------
    y_before : pl.DataFrame
        Training rows before the scored stretch; the forecaster was rewound to
        their end, and fitted scorers are fitted on them.
    y_scored : pl.DataFrame
        The scored training rows.
    y_pred : pl.DataFrame
        Walk-forward predictions over ``y_scored``.
    positions : np.ndarray
        Positions of the scored rows relative to the start of the training
        window.

    """

    y_before: pl.DataFrame
    y_scored: pl.DataFrame
    y_pred: pl.DataFrame
    positions: np.ndarray


def _train_window_predictions(
    forecaster: BaseForecaster,
    y_train: pl.DataFrame,
    X_actual_train: pl.DataFrame | None,
    *,
    n_rows: int,
    scorer: BaseScorer | _MultimetricScorer | None = None,
    method: str | None = None,
    predict_func_params: dict[str, object] | None = None,
    predict_forecasting_horizon: int | None = None,
    predict_stride: int | None = None,
    coverage_rates: list[float] | None = None,
    X_future: pl.DataFrame | None = None,
    X_forecast_train: pl.DataFrame | None = None,
) -> _TrainWindow | None:
    """Predict over the last ``n_rows`` training rows the forecaster learned from.

    A train score compares the test window with rows the model was fitted on. A
    forecaster that set a trailing stretch of its fit data aside (its
    ``holdout_size`` forecaster tag) did not learn from those rows, so the
    scored stretch ends before them. The fitted forecaster is rewound to the
    rows before the stretch and walked forward over it, exactly as the test
    window is predicted. No refit takes place, and the forecaster is left
    observed up to the end of the stretch.

    Positions are relative to ``y_train``, so the right rows are scored however
    the training window is placed in the full series.

    Parameters
    ----------
    forecaster : BaseForecaster
        Forecaster fitted on ``y_train``.
    y_train : pl.DataFrame
        The training window the forecaster was fitted on.
    X_actual_train : pl.DataFrame or None
        Actual features aligned with ``y_train``.
    n_rows : int
        Length of the stretch to score, normally the test window length.
    scorer : BaseScorer, _MultimetricScorer or None, default=None
        Scorer resolving the response method. Ignored when ``method`` is given.
    method : str or None, default=None
        Explicit response method (``"predict"``, ``"predict_interval"`` or
        ``"predict_class_proba"``).
    predict_func_params : dict or None, default=None
        Routed metadata passed to the prediction function.
    predict_forecasting_horizon : int or None, default=None
        Forecasting horizon for the walk-forward; ``None`` uses the fit default.
    predict_stride : int or None, default=None
        Stride for the walk-forward; ``None`` uses the forecaster default.
    coverage_rates : list of float or None, default=None
        Coverage rates for interval predictions.
    X_future : pl.DataFrame or None, default=None
        Known future features.
    X_forecast_train : pl.DataFrame or None, default=None
        External forecasts for the training window.

    Returns
    -------
    _TrainWindow or None
        The rows before the stretch, the scored rows, their predictions and
        their positions; or ``None``, after a warning, when no training rows
        would remain before the stretch.

    """
    forecaster_tags = forecaster.__sklearn_tags__().forecaster_tags
    holdout = forecaster_tags.holdout_size if forecaster_tags is not None else 0
    n_before = len(y_train) - holdout - n_rows
    if n_before <= 0:
        # Negative positions would wrap around to the end of the frame and score
        # the wrong rows, so the train score is reported as unavailable instead.
        # This typically happens on an early expanding-window fold whose training
        # window is short next to the test window plus the held-back stretch.
        warnings.warn(
            "Train score is unavailable for a fold whose training window "
            f"({len(y_train)} rows) is not larger than its test window ({n_rows} rows) "
            f"plus the forecaster's held-back rows ({holdout}); reporting NaN for that fold.",
            UserWarning,
            stacklevel=2,
        )
        return None

    before = np.arange(n_before)
    positions = np.arange(n_before, n_before + n_rows)
    y_before, X_actual_before = _safe_split(forecaster, y_train, X_actual_train, before)
    y_scored, X_actual_scored = _safe_split(forecaster, y_train, X_actual_train, positions, before)
    forecaster.rewind(y_before, X_actual=X_actual_before, X_future=X_future, X_forecast=X_forecast_train)
    y_pred = _predict(
        forecaster,
        y_scored,
        X_actual_scored,
        scorer,
        method=method,
        predict_func_params=predict_func_params,
        predict_forecasting_horizon=predict_forecasting_horizon,
        predict_stride=predict_stride,
        coverage_rates=coverage_rates,
        X_future=X_future,
        X_forecast=X_forecast_train,
    )
    return _TrainWindow(y_before, y_scored, y_pred, positions)


def _score_train_window(
    forecaster: BaseForecaster,
    window: _TrainWindow | None,
    scorer: BaseScorer | _MultimetricScorer,
    *,
    y: pl.DataFrame,
    score_params: dict[str, object] | None,
    train: np.ndarray,
    error_score: str | float = "raise",
) -> float | dict[str, float | str] | str:
    """Score the predictions of ``_train_window_predictions``.

    Parameters
    ----------
    forecaster : BaseForecaster
        The forecaster that produced the predictions.
    window : _TrainWindow or None
        The train window, or ``None`` when the train score is unavailable.
    scorer : BaseScorer or _MultimetricScorer
        Scorer(s) to evaluate.
    y : pl.DataFrame
        The full series ``train`` indexes into, used to slice ``score_params``.
    score_params : dict or None
        Per-row score parameters over ``y``; sliced to the scored rows.
    train : np.ndarray
        Absolute indices of the training window in ``y``.
    error_score : 'raise' or float, default='raise'
        Passed to ``_score``.

    Returns
    -------
    float, dict or str
        As ``_score``; NaN (per scorer for a multimetric scorer) when ``window``
        is ``None``.

    """
    if window is None:
        if isinstance(scorer, _MultimetricScorer):
            return {name: float("nan") for name in scorer._scorers}
        return float("nan")
    score_params_train = _check_method_params(y, params=score_params or {}, indices=train[window.positions])
    return _score(
        forecaster,
        window.y_before,
        window.y_scored,
        window.y_pred,
        scorer,
        score_params_train,
        error_score,
    )
