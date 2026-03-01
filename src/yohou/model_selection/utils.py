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
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import _check_method_params, _num_samples

from yohou.base import BaseForecaster
from yohou.metrics.base import BaseIntervalScorer, BaseScorer


def _check_scoring(forecastor: BaseForecaster, scoring: object) -> BaseScorer | _MultimetricScorer:
    """Check the scoring parameter.

    In addition, multimetric scoring leverages a caching mechanism to not call the same
    estimator response method multiple times. Hence, the scorer is modified to only use
    a single response method given a list of response methods and the estimator.

    Parameters
    ----------
    forecastor : forecastor instance
        The forecastor for which the scoring will be applied.

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

    return scorers  # type: ignore[return-value]


class _MultimetricScorer:
    """Callable for multimetric scoring used to avoid repeated calls
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

    def fit(self, y: pl.DataFrame) -> _MultimetricScorer:
        """Fit all scorers that have a fit method.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series used for fitting stateful scorers.

        Returns
        -------
        self
        """
        for scorer in self._scorers.values():
            if hasattr(scorer, "fit"):
                scorer.fit(y)
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
                scores[name] = scorer(y_truth, y_pred, **params.score)  # type: ignore[assignment]
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
    X: pl.DataFrame | None,
    forecasting_horizon: int,
    *,
    scorer: BaseScorer | _MultimetricScorer,
    train: np.ndarray[Any, Any],
    test: np.ndarray[Any, Any],
    verbose: int,
    parameters: dict[str, object] | None,
    fit_params: dict[str, object] | None,
    predict_params: dict[str, object] | None,
    score_params: dict[str, object] | None,
    return_train_score: bool = False,
    return_parameters: bool = False,
    return_n_test_samples: bool = False,
    return_times: bool = False,
    return_forecaster: bool = False,
    split_progress: tuple[int, int] | None = None,
    candidate_progress: tuple[int, int] | None = None,
    error_score: float | str = np.nan,
) -> dict[str, object]:
    """Fit forecaster and compute scores for a given dataset split.

    Parameters
    ----------
    forecaster : forecaster object implementing 'fit'
        The object to use to fit the data.

    y : pl.DataFrame
        Target time series.

    X : pl.DataFrame or None
        Feature time series.

    forecasting_horizon : int >= 1
        Horizon to forecast.

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

    train : array-like of shape (n_train_samples,)
        Indices of training samples.

    test : array-like of shape (n_test_samples,)
        Indices of test samples.

    verbose : int
        The verbosity level.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in forecaster fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

    parameters : dict or None
        Parameters to be set on the forecaster.

    fit_params : dict or None
        Parameters that will be passed to ``forecaster.fit``.

    predict_params : dict or None
        Parameters that will be passed to ``forecaster.predict``.

    score_params : dict or None
        Parameters that will be passed to the scorer.

    return_train_score : bool, default=False
        Whether to return the train scores.

    return_parameters : bool, default=False
        Return parameters that has been used for the forecaster.

    split_progress : {list, tuple} of int, default=None
        A list or tuple of format (<current_split_id>, <total_num_of_splits>).

    candidate_progress : {list, tuple} of int, default=None
        A list or tuple of format
        (<current_candidate_id>, <total_number_of_candidates>).

    return_n_test_samples : bool, default=False
        Whether to return the ``n_test_samples``.

    return_times : bool, default=False
        Whether to return the fit/score times.

    return_forecaster : bool, default=False
        Whether to return the fitted forecaster.

    Returns
    -------
    result : dict with the following attributes
        test_scores : dict of scorer name -> float
            Score on testing set (for all the scorers).
        train_scores : dict of scorer name -> float, optional
            Score on training set (for all the scorers).
            Only returned if `return_train_score` is True.
        n_test_samples : int
            Number of test samples.
        fit_time : float
            Time spent for fitting in seconds.
        score_time : float
            Time spent for scoring in seconds.
        parameters : dict or None
            The parameters that have been evaluated.
        forecaster : forecaster object
            The fitted forecaster.
        fit_error : str or None
            Traceback str if the fit failed, None if the fit succeeded.
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

    y_train, X_train = _safe_split(forecaster, y, X, train)
    y_test, X_test = _safe_split(forecaster, y, X, test, train)

    result: dict[str, object] = {}
    test_scores: dict[str, float | str] | float | str
    train_scores: dict[str, float | str] | float | str | None = None
    fit_time: float
    score_time: float
    try:
        forecaster.fit(y=y_train, X=X_train, forecasting_horizon=forecasting_horizon, **fit_params)

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
            else:
                test_scores = float(error_score)
                if return_train_score:
                    train_scores = float(error_score)
        result["fit_error"] = format_exc()
    else:
        result["fit_error"] = None

        fit_time = time.time() - start_time
        test_scores = _score(
            forecaster,
            y_train,
            y_test,
            X_test,
            predict_params,
            scorer,
            score_params_test,
            error_score,
        )
        score_time = time.time() - start_time - fit_time

        if return_train_score:
            # forecaster is stateful and needs to be rewound to predict the past
            score_params_train = _check_method_params(y, params=score_params, indices=train)
            train_rewind = train[: -len(test)]
            test_rewind = train[-len(test) :]
            y_train_rewind, X_train_rewind = _safe_split(forecaster, y_train, X_train, train_rewind)
            y_train_test, X_train_test = _safe_split(forecaster, y_train, X_train, test_rewind, train_rewind)
            forecaster.rewind(y_train_rewind, X_train_rewind)
            train_scores = _score(
                forecaster,
                y_train_rewind,
                y_train_test,
                X_train_test,
                predict_params,
                scorer,
                score_params_train,
                error_score,
            )

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2:
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
    return result


def _needs_interval_predictions(scorer: BaseScorer | _MultimetricScorer) -> bool:
    """Check if any scorer requires interval predictions."""
    if isinstance(scorer, _MultimetricScorer):
        return any(isinstance(s, BaseIntervalScorer) for s in scorer._scorers.values())
    return isinstance(scorer, BaseIntervalScorer)


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


def _score(
    forecaster: BaseForecaster,
    y_train: pl.DataFrame,
    y_test: pl.DataFrame,
    X_test: pl.DataFrame | None,
    predict_params: dict[str, object] | None,
    scorer: BaseScorer | _MultimetricScorer,
    score_params: dict[str, object] | None,
    error_score: str | float = "raise",
) -> float | dict[str, float | str] | str:
    """Compute the score(s) of an forecaster on a given test set.

    Will return a dict of floats if `scorer` is a _MultiMetricScorer, otherwise a single
    float is returned.
    """
    score_params = {} if score_params is None else score_params
    predict_params = {} if predict_params is None else predict_params

    scores: float | dict[str, float | str] | str
    try:
        # Detect whether interval predictions are needed
        if _needs_interval_predictions(scorer):
            coverage_rates = _collect_coverage_rates(scorer)
            y_pred = forecaster.observe_predict_interval(  # type: ignore[union-attr]
                y_test, X_test, coverage_rates=coverage_rates, **predict_params
            )
        else:
            y_pred = forecaster.observe_predict(y_test, X_test, **predict_params)  # type: ignore[arg-type]

        # observe_predict produces overlapping prediction windows when the last
        # observation chunk is smaller than stride. Deduplicate (keeping the most
        # recently informed prediction) and sort so downstream scorers receive a
        # clean, monotonically increasing time column.
        # TODO: Address this formally in scorers
        y_pred = y_pred.unique(subset=["time"], keep="last").sort("time")

        # Only fit scorer if it has a fit method (stateful scorers)
        if hasattr(scorer, "fit"):
            scorer.fit(y_train)
        scores = scorer(y_test, y_pred, **score_params)  # type: ignore[assignment]

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
