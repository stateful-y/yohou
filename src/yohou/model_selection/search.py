"""Optuna-based hyperparameter search with time series (nested) cross-validation."""

from __future__ import annotations

import gc
import numbers
import time
import warnings
from collections import defaultdict
from functools import partial
from typing import Any

import numpy as np
import optuna
import polars as pl
from joblib import effective_n_jobs
from numpy.ma import MaskedArray
from optuna import progress_bar as pbar_module
from pydantic import StrictInt
from scipy.stats import rankdata
from sklearn.base import (
    _fit_context,
    clone,
)
from sklearn.exceptions import NotFittedError
from sklearn.metrics._scorer import (
    _MultimetricScorer,
)
from sklearn.model_selection._search import _check_refit
from sklearn.model_selection._validation import (
    _aggregate_score_dicts,
    _insert_error_scores,
    _normalize_score_results,
    _warn_or_raise_about_fit_failures,
)
from sklearn.utils import Bunch
from sklearn.utils._param_validation import HasMethods, StrOptions
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    process_routing,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_method_params,
    check_is_fitted,
    indexable,
)

from yohou.base import BaseForecaster
from yohou.metrics.base import BaseScorer

from .optuna import Sampler, Storage
from .split import check_cv
from .utils import (
    _check_scoring,
    _fit_and_score,
    _run_trials_batch,
)


def _best_forecaster_has(attr):
    """Check if best_forecaster_ has the given attribute.

    This is used as the check for available_if decorator to ensure
    predict/update methods are only available when refit=True and
    the best_forecaster_ has been fitted.

    Parameters
    ----------
    attr : str
        The name of the attribute to check.

    Returns
    -------
    callable
        A function that checks if the best_forecaster_ has the attribute.
    """

    def check(self):
        # Check refit and best_forecaster_ existence
        if not self.refit:
            return False
        if not hasattr(self, "best_forecaster_"):
            return False
        # Check if best_forecaster_ has the attribute
        return hasattr(self.best_forecaster_, attr)

    return check


# TODO: Use methods from GridSearchCV without inheriting from it
# TODO: Handle predict_* parameters?
class SearchCV(BaseForecaster):
    """Exhaustive search over specified parameter values for an forecaster.

    SearchCV implements a "fit", "predict",  and "update_predict" function.

    The parameters of the forecaster used to apply these methods are optimized
    by cross-validated grid-search over the parameter's distribution.

    Parameters
    ----------
    forecaster : forecaster object
        This is assumed to implement the yohou forecaster interface.

    param_distributions : dict or list of dicts
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
        If a list of dicts is given, first a dict is sampled uniformly, and
        then a parameter is sampled using that dict as above.

    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        If `scoring` represents a single score, one can use:
        - a callable (see :ref:`scoring`) that returns a single value.

        If `scoring` represents multiple scores, one can use:
        - a dictionary with metric names as keys and callables a values.

    sampler : instance of optuna sampler


    storage : instance of optuna storage

    n_warmup_trials : int >= 0, default=0
        Number of warmup trials for which to run a randomized search.

    n_trials : int >= 0, default=5
        Number of trials for which to run the search defined by `sampler`.

    scoring : str, callable, list, tuple or dict
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        If `scoring` represents a single score, one can use:
        - a callable (see :ref:`scoring`) that returns a single value.

        If `scoring` represents multiple scores, one can use:
        - a dictionary with metric names as keys and callables a values.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    refit : bool, str, or callable, default=True
        Refit a forecaster using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a `str` denoting the
        scorer that would be used to find the best parameters for refitting
        the forecaster at the end.

        Where there are considerations other than maximum score in
        choosing a best forecaster, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given ``cv_results_``. In that
        case, the ``best_forecaster_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.

        The refitted forecaster is made available at the ``best_forecaster_``
        attribute and permits using ``predict`` and ``update_predict``directly
        on this ``SearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold time series cross validation,
        - integer, to specify the number of folds in a time series `Splitter`,
        - :class:`yohou.model_selection.Splitter` instance,
        - An iterable yielding (train, test) splits as arrays of indices.

    verbose : int
        Controls the verbosity: the higher, the more messages.

        - >1 : the computation time for each fold and parameter candidate is
          displayed;
        - >2 : the score is also displayed;
        - >3 : the fold and candidate parameter indexes are also displayed
          together with the starting time of the computation.

    pre_dispatch : int, or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in forecaster fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a polars ``DataFrame``.

    best_forecaster_ : forecaster
        Estimator that was chosen by the search, i.e. forecaster
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_forecaster

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

        This attribute is not available if ``refit`` is a function.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if
        `best_forecaster_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_forecaster_` exposes
        `n_features_in_` when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if
        `best_forecaster_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_forecaster_` exposes
        `feature_names_in_` when fit.

    """

    _required_parameters = ["forecaster", "param_distributions", "scoring"]
    _parameter_constraints: dict[str, list[object]] = {
        "forecaster": [HasMethods(["fit"])],
        "param_distributions": [dict, list],
        "sampler": [Sampler],
        "storage": [Storage, None],
        "n_warmup_trials": [numbers.Integral],
        "n_trials": [numbers.Integral],
        "scoring": [BaseScorer],
        "n_jobs": [numbers.Integral, None],
        "refit": ["boolean", str, callable],
        "cv": ["cv_object"],
        "verbose": ["verbose"],
        "pre_dispatch": [numbers.Integral, str],
        "error_score": [StrOptions({"raise"}), numbers.Real],
        "return_train_score": ["boolean"],
    }

    def __init__(
        self,
        forecaster: "BaseForecaster",
        param_distributions: dict[str, optuna.distributions.BaseDistribution],
        scoring: "BaseScorer" | "_MultimetricScorer",
        *,
        sampler: "Sampler" = Sampler(),
        storage: "Storage" | None = None,
        n_warmup_trials: int = 0,
        n_trials: int = 5,
        n_jobs: int | None = None,
        refit: bool | str | object = True,
        cv: object = None,
        verbose: int = 0,
        pre_dispatch: str | int = "2*n_jobs",
        error_score: float | str = np.nan,
        return_train_score: bool = False,
    ) -> None:
        self.forecaster = forecaster
        self.param_distributions = param_distributions
        self.scoring = scoring
        self.sampler = sampler
        self.storage = storage
        self.n_warmup_trials = n_warmup_trials
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score

    @property
    def prediction_types(self) -> set[str]:
        """Get the types of predictions this forecaster produces.

        Returns
        -------
        set of {"point", "interval"}
            Set of prediction types produced by the implicit forecaster.
            Point forecasters return {"point"}, interval forecasters return {"interval"},
            and forecasters producing both return {"point", "interval"}.

        """
        return self.forecaster.prediction_types

    def score(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params: object) -> object:
        """Return the score on the given data, if the forecaster has been refit.

        This uses the score defined by ``scoring``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        **params : dict
            Parameters to be passed to the underlying scorer(s).

        Returns
        -------
        score : float
            The score defined by ``scoring`` if provided, and the
            ``best_forecaster_.score`` method otherwise.
        """
        _check_refit(self, "score")
        check_is_fitted(self)

        _raise_for_params(params, self, "score")

        if _routing_enabled():
            score_params = process_routing(self, "score", **params).scorer["score"]
        else:
            score_params = dict()

        if self.scorer_ is None:
            raise ValueError(
                "No score function explicitly defined, "
                "and the forecaster doesn't provide one %s" % self.best_forecaster_
            )
        if isinstance(self.scorer_, dict):
            if self.multimetric_:
                scorer = self.scorer_[self.refit]
            else:
                scorer = self.scorer_
            return scorer(self.best_forecaster_, X, y, **score_params)

        # callable
        score = self.scorer_(self.best_forecaster_, X, y, **score_params)
        if self.multimetric_:
            score = score[self.refit]
        return score

    @available_if(_best_forecaster_has("predict"))  # type: ignore[untyped-decorator]
    def predict(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        cross_learning_group: str | None = None,
        predict_transformed: bool = False,
    ) -> pl.DataFrame:
        """Call predict on the forecaster with the best found parameters.

        Only available if ``refit=True`` and the underlying forecaster supports
        ``predict``.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Exogenous feature time series.

        forecasting_horizon : int >= 1 or None, default=None
            Horizon to forecast. If None, uses the fitted forecaster's
            ``fit_forecasting_horizon_``.

        cross_learning_group : str or None, default=None
            For panel data (local_group_names_ is not None):
            - If None: predict for all groups (default behavior)
            - If str: predict only for the specified group (cross-learning)
            For global data: parameter is ignored.

        predict_transformed : bool, default=False
            If ``True``, the predictions are returned in the transformed space.

        Returns
        -------
        pl.DataFrame
            Predicted time series or values for based on the forecaster with
            the best found parameters.

        """
        check_is_fitted(self)
        return self.best_forecaster_.predict(
            X=X,
            forecasting_horizon=forecasting_horizon,
            cross_learning_group=cross_learning_group,
            predict_transformed=predict_transformed,
        )

    @available_if(_best_forecaster_has("update"))  # type: ignore[untyped-decorator]
    def update(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> "SearchCV":
        """Call update on the forecaster with the best found parameters.

        Only available if ``refit=True`` and the underlying forecaster supports
        ``update``.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series for updates.

        X : pl.DataFrame or None, default=None
            Exogenous feature time series for updates.

        Returns
        -------
        self

        """
        check_is_fitted(self)

        self.best_forecaster_.update(y, X)
        return self

    @available_if(_best_forecaster_has("update_predict"))  # type: ignore[untyped-decorator]
    def update_predict(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        stride: StrictInt | None = None,
        predict_transformed: bool = False,
        **params,
    ) -> pl.DataFrame:
        """Call update_predict on the forecaster with the best found parameters.

        Only available if ``refit=True`` and the underlying forecaster supports
        ``update_predict``.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series for updates.

        X : pl.DataFrame or None, default=None
            Exogenous feature time series for updates.

        forecasting_horizon : int >= 1 or None, default=None
            Horizon to forecast recursively. If None, uses the fitted forecaster's
            ``fit_forecasting_horizon_``.

        stride : int >= 1 or None, default=None
            Stride in between two predictions. If None, uses the fitted forecaster's
            ``fit_forecasting_horizon_``.

        predict_transformed : bool, default=False
            Whether to output prediction in the transformed space.

        **params : dict
            Metadata to route to `predict()`.

        Returns
        -------
        pl.DataFrame
            Predicted time series or values for based on the forecaster with
            the best found parameters.

        """
        check_is_fitted(self)

        return self.best_forecaster_.update_predict(
            y, X, forecasting_horizon, stride, predict_transformed, **params
        )

    @available_if(_best_forecaster_has("reset"))  # type: ignore[untyped-decorator]
    def reset(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> "SearchCV":
        """Call reset on the forecaster with the best found parameters.

        Only available if ``refit=True`` and the underlying forecaster supports
        ``reset``.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        X : pl.DataFrame or None, default=None
            Exogenous feature time series.

        Returns
        -------
        self

        """
        check_is_fitted(self)

        self.best_forecaster_.reset(y, X)
        return self

    @property
    def n_features_in_(self) -> object:
        """Number of features seen during :term:`fit`.

        Only available when `refit=True`.
        """
        # For consistency with other estimators we raise a AttributeError so
        # that hasattr() fails if the search estimator isn't fitted.
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_features_in_ attribute.".format(self.__class__.__name__)
            ) from nfe

        return self.best_forecaster_.n_features_in_

    def _get_candidate_params(self, trial: optuna.Trial, i_trial: int) -> dict[str, object]:
        """Generate candidate parameters for a trial."""
        if i_trial == self.n_warmup_trials:
            self.study_.sampler = self.sampler_

        candidate_params: dict[str, Any] = {
            name: trial._suggest(name, distribution)
            for name, distribution in self.param_distributions.items()
        }

        return candidate_params

    def _optimize(
        self,
        func: object,
        callbacks: list[object],
        timeout: float | None = None,
        reseed_sampler_rng: bool = False,
        gc_after_trial: bool = False,
        show_progress_bar: bool = False,
    ) -> list[object]:
        """
        Optimize

        Parameters
        ----------
        func : callable
            A callable that implements objective function.

        timeout : float or None, default=None
            Stop study after the given number of second(s). :obj:`None` represents no limit in
            terms of elapsed time. The study continues to create trials until the number of
            trials reaches ``n_trials``, ``timeout`` period elapses,
            :func:`~optuna.study.Study.stop` is called or, a termination signal such as
            SIGTERM or Ctrl+C is received.

        gc_after_trial : bool, default=False
            Flag to determine whether to automatically run garbage collection after each trial.
            Set to :obj:`True` to run the garbage collection, :obj:`False` otherwise.
            When it runs, it runs a full collection by internally calling :func:`gc.collect`.
            If you see an increase in memory consumption over several trials, try setting this
            flag to :obj:`True`.

        show_progress_bar : bool, default=False
            Flag to show progress bars or not. To disable progress bar, set this :obj:`False`.
            Currently, progress bar is experimental feature and disabled
            when ``n_trials`` is :obj:`None`, ``timeout`` is not :obj:`None`, and
            ``n_jobs`` :math:`\\ne 1`.
        """
        progress_bar = pbar_module._ProgressBar(
            show_progress_bar, self.effective_n_trials_, timeout
        )
        self.study_._stop_flag = False

        try:
            all_outputs: dict[str, object] = {}

            # Here we set `in_optimize_loop = True`, not at the beginning of the `_optimize()`
            # function. Because it is a thread-local object and `n_jobs` option spawns new threads.
            self.study_._thread_local.in_optimize_loop = True
            if reseed_sampler_rng:
                self.study_.sampler.reseed_rng()

            i_trial = 0
            n_trials_per_batch = effective_n_jobs(self.n_jobs)
            time_start = time.time()

            while True:
                if self.study_._stop_flag:
                    break

                if i_trial >= self.effective_n_trials_:
                    break
                i_trial += n_trials_per_batch

                if timeout is not None:
                    elapsed_seconds = time.time() - time_start
                    if elapsed_seconds >= timeout:
                        break

                try:
                    frozen_trial, outputs = _run_trials_batch(
                        self.study_, func, i_trial, n_trials_per_batch, catch=()
                    )

                    if not all_outputs:
                        all_outputs = outputs
                    else:
                        for key, value in outputs.items():
                            if isinstance(all_outputs[key], list):
                                all_outputs[key].extend(value)
                            else:
                                all_outputs[key] = np.concatenate((all_outputs[key], value))

                finally:
                    # The following line mitigates memory problems that can be occurred in some
                    # environments (e.g., services that use computing containers such as GitHub
                    # Actions). Please refer to the following PR for further details:
                    # https://github.com/optuna/optuna/pull/325.
                    if gc_after_trial:
                        gc.collect()

                if callbacks is not None:
                    for callback in callbacks:
                        callback(self.study_, frozen_trial)  # type: ignore[operator]

                if progress_bar is not None:
                    elapsed_seconds = time.time() - time_start
                    progress_bar.update(elapsed_seconds, self.study_)

            self.study_._storage.remove_session()

        finally:
            self.study_._thread_local.in_optimize_loop = False
            progress_bar.close()

        return all_outputs  # type: ignore[no-any-return]

    def _run_search(self, evaluate_candidates: object) -> dict[str, object]:
        """Repeatedly calls `evaluate_candidates` to conduct a search.

        Parameters
        ----------
        evaluate_candidates : callable
            This callback accepts:
                - a list of candidates, where each candidate is a dict of
                  parameter settings.
                - an optional `cv` parameter which can be used to e.g.
                  evaluate candidates on different dataset splits, or
                  evaluate candidates on subsampled data (as done in the
                  SucessiveHaling estimators). By default, the original `cv`
                  parameter is used, and it is available as a private
                  `_checked_cv_orig` attribute.
                - an optional `more_results` dict. Each key will be added to
                  the `cv_results_` attribute. Values should be lists of
                  length `n_candidates`

            It returns a dict of all results so far, formatted like
            ``cv_results_``.

        """

        def batch_func(trial_batch: list[optuna.Trial], i_trial: int) -> tuple[object, object]:
            """Evaluate batch of trials."""
            candidate_params = [
                self._get_candidate_params(trial, i_trial + i_trial_cand)
                for i_trial_cand, trial in enumerate(trial_batch)
            ]

            results = evaluate_candidates(candidate_params, i_trial)  # type: ignore[operator]

            batch_values = results["mean_test_score"]

            return batch_values, results

        return self._optimize(batch_func, callbacks=[])  # type: ignore[return-value]

    def _check_refit_for_multimetric(self, scores: object) -> None:
        """Check `refit` is compatible with `scores` is valid"""
        multimetric_refit_msg = (
            "For multi-metric scoring, the parameter refit must be set to a "
            "scorer key or a callable to refit a forecaster with the best "
            "parameter setting on the whole data and make the best_* "
            "attributes available for that metric. If this is not needed, "
            f"refit should be set to False explicitly. {self.refit!r} was "
            "passed."
        )

        valid_refit_dict = isinstance(self.refit, str) and self.refit in scores  # type: ignore[operator]

        if self.refit is not False and not valid_refit_dict and not callable(self.refit):
            raise ValueError(multimetric_refit_msg)

    @staticmethod
    def _select_best_index(refit: object, refit_metric: str, results: dict[str, object]) -> object:
        """Select index of the best combination of hyperparemeters."""
        if callable(refit):
            # If callable, refit is expected to return the index of the best
            # parameter set.
            best_index = refit(results)
            if not isinstance(best_index, numbers.Integral):
                raise TypeError("best_index_ returned is not an integer")
            if best_index < 0 or best_index >= len(results["params"]):  # type: ignore[operator, arg-type]
                raise IndexError("best_index_ index out of range")
        else:
            best_index = results[f"rank_test_{refit_metric}"].argmin()  # type: ignore[attr-defined]
        return best_index

    def _get_scorers(self) -> tuple[object, str]:
        """Get the scorer(s) to be used.

        This is used in ``fit`` and ``get_metadata_routing``.

        Returns
        -------
        scorers, refit_metric
        """
        refit_metric = "score"
        scorers = _check_scoring(self.forecaster, self.scoring)
        if isinstance(self.scoring, BaseScorer):
            scorers = self.scoring
        else:
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit  # type: ignore[assignment]
            scorers = _MultimetricScorer(scorers=scorers, raise_exc=(self.error_score == "raise"))

        return scorers, refit_metric

    def _get_routed_params_for_fit(self, params: dict[str, object]) -> dict[str, object]:
        """Get the parameters to be used for routing.

        This is a method instead of a snippet in ``fit`` since it's used twice,
        here in ``fit``, and in ``HalvingRandomSearchCV.fit``.
        """
        if _routing_enabled():
            routed_params = process_routing(self, "fit", **params)
        else:
            params = params.copy()
            routed_params = Bunch(
                forecaster=Bunch(fit=params),
                splitter=Bunch(split={}),
                scorer=Bunch(score={}),
            )
        return routed_params  # type: ignore[no-any-return]

    @_fit_context(  # type: ignore[untyped-decorator]
        # *SearchCV.forecaster is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: int = 1,
        **params: object,
    ) -> "SearchCV":
        """Run fit with all sets of parameters.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        X : pl.DataFrame or None, default=None
            Exogenous feature time series.

        forecasting_horizon : int >= 1, default=1
            Horizon to forecast.

        **params : dict of str -> object
            Parameters passed to the ``fit`` method of the forecaster, the scorer,
            and the CV splitter.

            If a fit parameter is an array-like whose length is equal to
            `num_samples` then it will be split across CV groups along with `X`
            and `y`. For example, the :term:`sample_weight` parameter is split
            because `len(sample_weights) = len(X)`.

        Returns
        -------
        self : object
            Instance of fitted forecaster.
        """
        # Import here to avoid circular dependency
        from yohou.utils.validation import check_inputs

        # Set forecasting horizon attribute (required by forecaster interface)
        if forecasting_horizon < 1:
            raise ValueError(
                f"`forecasting_horizon` should be a positive int. It is: {forecasting_horizon}"
            )
        self.fit_forecasting_horizon_ = forecasting_horizon

        # Set interval attribute (required by forecaster interface)
        self.interval_ = check_inputs(y, X)

        # Set panel data structure attributes (required by forecaster interface)
        self._set_local_groups(y, X)

        self.sampler_ = clone(self.sampler).instantiate().instance_
        self.warmup_sampler_ = optuna.samplers.RandomSampler()

        self.storage_ = None
        if self.storage is not None:
            self.storage_ = clone(self.storage).instantiate().instance_

        self.study_ = optuna.create_study(
            direction="minimize",
            sampler=self.warmup_sampler_,
            storage=self.storage_,
        )
        self.effective_n_trials_ = self.n_warmup_trials + self.n_trials

        scorers, refit_metric = self._get_scorers()

        # TODO: Replace with something adapted to pl time series
        y, X = indexable(y, X)
        params = _check_method_params(y, params=params)

        routed_params = self._get_routed_params_for_fit(params)

        cv_orig = check_cv(self.cv, forecasting_horizon)  # type: ignore[arg-type]
        n_splits = cv_orig.get_n_splits(y, **routed_params.splitter.split)  # type: ignore[attr-defined]

        base_forecaster = clone(self.forecaster)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)
        n_cand_per_trials = self.n_jobs or 1

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=routed_params.forecaster.fit,  # type: ignore[attr-defined]
            # TODO: Use routing?
            predict_params=routed_params.forecaster.predict,
            score_params=routed_params.scorer.score,  # type: ignore[attr-defined]
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )

        with parallel:
            all_candidate_params: list[dict[str, object]] = []
            all_out: list[object] = []
            all_more_results: dict[str, list[object]] = defaultdict(list)

            def evaluate_candidates(
                candidate_params: list[dict[str, object]],
                i_trial: int,
                more_results: dict[str, object] | None = None,
            ) -> dict[str, object]:
                """Evaluate candidate parameters via cross-validation."""
                cv = cv_orig
                n_candidates = self.n_warmup_trials + self.n_trials

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates, totalling {2} fits".format(
                            n_splits, n_candidates, n_candidates * n_splits
                        )
                    )

                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_forecaster),
                        y,
                        X,
                        forecasting_horizon,
                        train=train,
                        test=test,
                        parameters=candidate_params[i_trial_cand],
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(
                            i_trial + i_trial_cand,
                            self.effective_n_trials_,
                        ),
                        **fit_and_score_kwargs,
                    )
                    for i_trial_cand in range(n_cand_per_trials)
                    for split_idx, (train, test) in enumerate(
                        cv.split(y, **routed_params.splitter.split)  # type: ignore[attr-defined]
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_cand_per_trials * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned inconsistent results. Expected {} "
                        "splits, got {}".format(n_splits, len(out) // n_cand_per_trials)
                    )

                _warn_or_raise_about_fit_failures(out, self.error_score)

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)  # type: ignore[arg-type]

                # TODO: Rank only works batch-wise so it is not global
                results = self._format_results(
                    candidate_params,
                    n_splits,
                    out,
                    more_results,  # type: ignore[arg-type]
                )

                return results

            results = self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]["test_scores"]  # type: ignore[index]
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit  # type: ignore[assignment]

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(self.refit, refit_metric, results)
            if not callable(self.refit):
                # With a non-custom callable, we can select the best score
                # based on the best index
                self.best_score_ = results[f"mean_test_{refit_metric}"][self.best_index_]  # type: ignore[index]
            self.best_params_ = results["params"][self.best_index_]  # type: ignore[index]

        if self.refit:
            # here we clone the forecaster as well as the parameters, since
            # sometimes the parameters themselves might be forecasters, e.g.
            # when we search over different forecasters in a pipeline.
            # ref: https://github.com/scikit-learn/scikit-learn/pull/26786
            self.best_forecaster_ = clone(
                clone(base_forecaster).set_params(**clone(self.best_params_, safe=False))
            )

            refit_start_time = time.time()
            self.best_forecaster_.fit(
                y,
                X=X,
                forecasting_horizon=forecasting_horizon,
                **routed_params.forecaster.fit,  # type: ignore[attr-defined]
            )
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_forecaster_, "feature_names_in_"):
                self.feature_names_in_ = self.best_forecaster_.feature_names_in_
            if hasattr(self.best_forecaster_, "n_features_in_"):
                self.n_features_in_ = self.best_forecaster_.n_features_in_

        # Store the only scorer not as a dict for single metric evaluation
        if isinstance(scorers, _MultimetricScorer):
            self.scorer_ = scorers._scorers
        else:
            self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    def _format_results(
        self,
        candidate_params: list[dict[str, object]],
        n_splits: int,
        out: list[object],
        more_results: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Format cross-validation results into cv_results_ dictionary."""
        n_candidates = len(candidate_params)
        out = _aggregate_score_dicts(out)

        results = dict(more_results or {})
        for key, val in results.items():
            # each value is a list (as per evaluate_candidate's convention)
            # we convert it to an array for consistency with the other keys
            results[key] = np.asarray(val)

        def _store(
            key_name: str,
            array: object,
            weights: object = None,
            splits: bool = False,
            rank: bool = False,
        ) -> None:
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by splits, then by parameters
            # We want `array` to have `n_candidates` rows and `n_splits` cols.
            array = np.array(array, dtype=np.float64).reshape(n_candidates, n_splits)
            if splits:
                for split_idx in range(n_splits):
                    # Uses closure to alter the results
                    results["split%d_%s" % (split_idx, key_name)] = array[:, split_idx]  # type: ignore[index]

            array_means = np.average(array, axis=1, weights=weights)
            results["mean_%s" % key_name] = array_means

            if key_name.startswith(("train_", "test_")) and np.any(~np.isfinite(array_means)):
                warnings.warn(
                    (
                        f"One or more of the {key_name.split('_')[0]} scores "
                        f"are non-finite: {array_means}"
                    ),
                    category=UserWarning,
                )

            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(
                np.average(
                    (array - array_means[:, np.newaxis]) ** 2,
                    axis=1,
                    weights=weights,
                )
            )
            results["std_%s" % key_name] = array_stds

            if rank:
                # When the fit/scoring fails `array_means` contains NaNs, we
                # will exclude them from the ranking process and consider them
                # as tied with the worst performers.
                if np.isnan(array_means).all():
                    # All fit/scoring routines failed.
                    rank_result = np.ones_like(array_means, dtype=np.int32)
                else:
                    min_array_means = np.nanmin(array_means) - 1
                    array_means = np.nan_to_num(array_means, nan=min_array_means)
                    rank_result = rankdata(-array_means, method="min").astype(np.int32, copy=False)
                results["rank_%s" % key_name] = rank_result

        _store("fit_time", out["fit_time"])  # type: ignore[call-overload]
        _store("score_time", out["score_time"])  # type: ignore[call-overload]
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results: dict[str, object] = defaultdict(
            partial(
                MaskedArray,
                np.empty(
                    n_candidates,
                ),
                mask=True,
                dtype=object,
            )
        )
        for cand_idx, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_idx] = value  # type: ignore[index]

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results["params"] = candidate_params

        test_scores_dict = _normalize_score_results(out["test_scores"])  # type: ignore[call-overload]

        for scorer_name in test_scores_dict:
            # Computed the (weighted) mean and std for test scores alone
            _store(
                "test_%s" % scorer_name,
                test_scores_dict[scorer_name],
                splits=True,
                rank=True,
                weights=None,
            )

        if self.return_train_score:
            train_scores_dict = _normalize_score_results(out["train_scores"])  # type: ignore[call-overload]
            for scorer_name in train_scores_dict:
                _store(
                    "train_%s" % scorer_name,
                    train_scores_dict[scorer_name],
                    splits=True,
                    rank=False,
                    weights=None,
                )

        return results

    def get_metadata_routing(self) -> object:
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self)
        router.add(
            forecaster=self.forecaster,
            method_mapping=MethodMapping()
            .add(caller="fit", callee="fit")
            .add(caller="predict", callee="predict"),
        )

        scorer, _ = self._get_scorers()
        router.add(
            scorer=scorer,
            method_mapping=MethodMapping()
            .add(caller="score", callee="score")
            .add(caller="fit", callee="score"),
        )
        router.add(
            splitter=self.cv,
            method_mapping=MethodMapping().add(caller="fit", callee="split"),
        )
        return router
