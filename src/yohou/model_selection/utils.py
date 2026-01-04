"""Utilities for model evaluation and (nested) cross-validation scoring."""

from __future__ import annotations

import numbers
import sys
import time
import warnings
from contextlib import suppress
from traceback import format_exc
from typing import Any

import numpy as np
import optuna
import polars as pl
from joblib import logger
from optuna import exceptions
from optuna import trial as trial_module
from optuna.storages._heartbeat import is_heartbeat_enabled
from optuna.study._optimize import (
    _log_failed_trial,
    _logger,
)
from optuna.study._tell import STUDY_TELL_WARNING_KEY, _tell_with_warning
from optuna.trial import TrialState
from sklearn.base import clone
from sklearn.utils import Bunch
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    _routing_enabled,
    process_routing,
)
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import _check_method_params, _num_samples

from yohou.base import BaseForecaster
from yohou.metrics.base import BaseScorer


def _check_scoring(
    forecastor: "BaseForecaster", scoring: object
) -> BaseScorer | "_MultimetricScorer":
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

        See :ref:`multimetric_grid_search` for an example.

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
            raise ValueError(
                f"Non-string types were found in the keys of the given dict. scoring={scoring!r}"
            )
        if len(keys) == 0:
            raise ValueError(f"An empty dict was passed. {scoring!r}")

        if not all(isinstance(v, BaseScorer) for v in scoring.values()):
            raise ValueError(
                f"Non-scorer types were found in the values of the given dict. scoring={scoring!r}"
            )

    else:
        raise ValueError(
            "Invalid scoring. It should be an instance of `BaseScorer` or "
            "a dict with strings as keys and instances of `BaseScorer` as "
            f"values. Got {scoring}."
        )

    return scorers


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

    def __call__(self, *args: object, **kwargs: object) -> dict[str, float | str]:
        """Evaluate predicted target values."""
        scores: dict[str, float | str] = {}

        if _routing_enabled():
            routed_params = process_routing(self, "score", **kwargs)
        else:
            # they all get the same args, and they all get them all
            routed_params = Bunch(**{name: Bunch(score=kwargs) for name in self._scorers})

        for name, scorer in self._scorers.items():
            try:
                params = routed_params.get(name)
                if params is None:
                    raise ValueError(f"Missing routing params for scorer '{name}'")
                scores[name] = scorer(*args, **params.score)
            except Exception as e:
                if self._raise_exc:
                    raise e
                else:
                    scores[name] = format_exc()
        return scores

    def get_metadata_routing(self) -> object:
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A :class:`~utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        return MetadataRouter(owner=self.__class__.__name__).add(
            **self._scorers, method_mapping="score"
        )


def _fit_and_score(
    forecaster: "BaseForecaster",
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
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

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

    except Exception:
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
            y_test,
            X_test,
            predict_params,
            scorer,
            score_params_test,
            error_score,
        )
        score_time = time.time() - start_time - fit_time

        if return_train_score:
            # TODO: forecaster is stateful and needs to be reset to predict the past
            score_params_train = _check_method_params(y, params=score_params, indices=train)
            train_reset = train[: -len(test)]
            test_reset = train[-len(test) :]
            y_train_reset, X_train_reset = _safe_split(forecaster, y_train, X_train, train_reset)
            y_train_test, X_train_test = _safe_split(
                forecaster, y_train, X_train, test_reset, train_reset
            )
            forecaster.reset(y_train_reset, X_train_reset)
            train_scores = _score(
                forecaster,
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
        print(end_msg)

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


def _score(
    forecaster: "BaseForecaster",
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
        y_pred = forecaster.update_predict(y_test, X_test, **predict_params)  # type: ignore[arg-type]
        scores = scorer(y_truth=y_test, y_pred=y_pred, **score_params)

    except Exception:
        if isinstance(scorer, _MultimetricScorer):
            # If `_MultimetricScorer` raises exception, the `error_score`
            # parameter is equal to "raise".
            raise
        else:
            if error_score == "raise":
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
                )

    # Check non-raised error messages in `_MultimetricScorer`
    if isinstance(scorer, _MultimetricScorer):
        if isinstance(scores, dict):
            exception_messages = [
                (name, str_e) for name, str_e in scores.items() if isinstance(str_e, str)
            ]
            if exception_messages:
                # error_score != "raise"
                for name, str_e in exception_messages:
                    scores[name] = (
                        float(error_score)
                        if isinstance(error_score, numbers.Number)
                        else error_score
                    )
                warnings.warn(
                    (
                        "Scoring failed. The score on this train-test partition for "
                        f"these parameters will be set to {error_score}. Details: \n"
                        f"{str_e}"
                    ),
                    UserWarning,
                )
    error_msg = "scoring must return a number, got %s (%s) instead. (scorer=%s)"
    if isinstance(scores, dict):
        for name, score in scores.items():
            if isinstance(score, str):
                continue  # Already error string
            score_val: float | str = score
            if hasattr(score_val, "item"):
                with suppress(ValueError):
                    # e.g. unwrap memmapped scalars
                    score_val = score_val.item()
            if not isinstance(score_val, numbers.Number):
                raise ValueError(error_msg % (score_val, type(score_val), name))
            scores[name] = (
                float(score_val)
                if isinstance(score_val, (int, float, numbers.Number))
                else score_val
            )
    else:  # scalar
        if isinstance(scores, str):
            return scores  # Already error string
        if hasattr(scores, "item"):
            with suppress(ValueError):
                # e.g. unwrap memmapped scalars
                scores = scores.item()
        if not isinstance(scores, numbers.Number):
            raise ValueError(error_msg % (scores, type(scores), scorer))
        scores = float(scores)
    return scores


def _run_trials_batch(
    study: "optuna.Study",
    batch_func: "optuna.study.study.ObjectiveFuncType",
    i_trial: int,
    n_trials_per_batch: int,
    catch: tuple[type[Exception], ...],
) -> trial_module.FrozenTrial:
    """Run a batch of Optuna trials."""
    if is_heartbeat_enabled(study._storage):
        optuna.storages.fail_stale_trials(study)

    trial_batch = [study.ask() for _ in range(n_trials_per_batch)]

    state: TrialState | None = None
    batch_value_or_values: Any | None = None
    func_err: Exception | KeyboardInterrupt | None = None
    func_err_fail_exc_info: Any | None = None

    try:
        batch_value_or_values, results = batch_func(trial_batch, i_trial)

    except exceptions.TrialPruned as e:
        state = TrialState.PRUNED
        func_err = e

    except (Exception, KeyboardInterrupt) as e:
        state = TrialState.FAIL
        func_err = e
        func_err_fail_exc_info = sys.exc_info()

    for i_trial, trial in enumerate(trial_batch):
        value_or_values = None
        if batch_value_or_values is not None:
            value_or_values = batch_value_or_values[i_trial]

        # `_tell_with_warning` may raise during trial post-processing.
        try:
            frozen_trial = _tell_with_warning(
                study=study,
                trial=trial,
                value_or_values=value_or_values,
                state=state,
                suppress_warning=True,
            )

        except Exception:
            frozen_trial = study._storage.get_trial(trial._trial_id)
            raise

        finally:
            if frozen_trial.state == TrialState.COMPLETE:
                study._log_completed_trial(frozen_trial)
            elif frozen_trial.state == TrialState.PRUNED:
                _logger.info("Trial {} pruned. {}".format(frozen_trial.number, str(func_err)))
            elif frozen_trial.state == TrialState.FAIL:
                if func_err is not None:
                    _log_failed_trial(
                        frozen_trial,
                        repr(func_err),
                        exc_info=func_err_fail_exc_info,
                        value_or_values=value_or_values,
                    )
                elif STUDY_TELL_WARNING_KEY in frozen_trial.system_attrs:
                    _log_failed_trial(
                        frozen_trial,
                        frozen_trial.system_attrs[STUDY_TELL_WARNING_KEY],
                        value_or_values=value_or_values,
                    )
                else:
                    assert False, "Should not reach."
            else:
                assert False, "Should not reach."

        if (
            frozen_trial.state == TrialState.FAIL
            and func_err is not None
            and not isinstance(func_err, catch)
        ):
            raise func_err

    return frozen_trial, results
