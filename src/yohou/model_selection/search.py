"""Grid and randomized hyperparameter search with time series cross-validation."""

from __future__ import annotations

import numbers
import time
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import numpy as np
from numpy.ma import MaskedArray
from scipy.stats import rankdata
from sklearn.base import (
    MetaEstimatorMixin,
    _fit_context,
    clone,
)
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.model_selection._validation import (
    _aggregate_score_dicts,
    _insert_error_scores,
    _normalize_score_results,
    _warn_or_raise_about_fit_failures,
)
from sklearn.utils._array_api import xpx
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
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
from yohou.utils import validate_forecaster_data

from .split import check_cv
from .utils import (
    _check_scoring,
    _fit_and_score,
    _MultimetricScorer,
)


def _search_forecaster_has(attr):
    """Check if SearchCV.best_forecaster_ has a given attribute or prediction type.

    This is used as the check for available_if decorator to ensure
    predict/update methods are only available when refit=True and
    the best_forecaster_ supports the required functionality.

    For special attributes like "point" or "interval", checks if the prediction type
    is supported via tags. Otherwise, checks if the attribute exists on best_forecaster_.

    Parameters
    ----------
    attr : str
        The name of the attribute or prediction type to check.
        Special values: "point", "interval" check forecaster_type in tags.
        Other values check hasattr(best_forecaster_, attr).

    Returns
    -------
    callable
        A function that checks if the best_forecaster_ has the attribute.

    Examples
    --------
    >>> # Check for point prediction support
    >>> def predict(self, forecasting_horizon=1):
    ...     return self.best_forecaster_.predict(forecasting_horizon)
    >>> predict = available_if(_search_forecaster_has("predict"))(predict)
    >>>
    >>> # Check for specific method
    >>> def reset(self, y, X=None):
    ...     return self.best_forecaster_.reset(y, X)
    >>> reset = available_if(_search_forecaster_has("reset"))(reset)
    """

    def check(self):
        """Check if the forecaster has the required attribute or prediction type."""
        # Check refit and best_forecaster_ existence
        if not self.refit:
            return False

        # Check if best_forecaster_ was set
        if not hasattr(self, "best_forecaster_"):
            return False

        # Check forecaster_type tags for prediction type checks
        if attr in {"point", "interval"}:
            tags = self.best_forecaster_.__sklearn_tags__()
            forecaster_type = getattr(tags.forecaster_tags, "forecaster_type", None)
            if forecaster_type == "both":
                return True
            return forecaster_type == attr

        # Otherwise check attribute existence
        return hasattr(self.best_forecaster_, attr)

    return check


def _yield_masked_array_for_each_param(candidate_params):
    """Yield masked arrays for each parameter in candidate_params.

    Parameters
    ----------
    candidate_params : list of dict
        List of parameter dictionaries.

    Yields
    ------
    param_name : str
        Parameter name.
    param_values : MaskedArray
        Masked array of parameter values (masked where parameter is not present).
    """
    all_param_names = {key for params in candidate_params for key in params}

    for param_name in sorted(all_param_names):
        param_values = []
        for params in candidate_params:
            param_values.append(params.get(param_name, MaskedArray.fill_value))

        param_array = MaskedArray(
            param_values,
            mask=[params.get(param_name) is MaskedArray.fill_value for params in candidate_params],
            dtype=object,
        )
        yield param_name, param_array


class BaseSearchCV(BaseForecaster, MetaEstimatorMixin, metaclass=ABCMeta):
    """Abstract base class for hyperparameter search with time series cross-validation.

    Warning: This class should not be used directly. Use derived classes instead.
    """

    _parameter_constraints: dict = {
        "forecaster": [HasMethods(["fit", "predict"])],
        "scoring": [None, callable, dict],
        "n_jobs": [numbers.Integral, None],
        "refit": ["boolean", str, callable],
        "cv": [numbers.Integral, HasMethods(["split", "get_n_splits"]), None],
        "verbose": ["verbose"],
        "pre_dispatch": [numbers.Integral, str],
        "error_score": [StrOptions({"raise"}), numbers.Real],
        "return_train_score": ["boolean"],
    }

    @abstractmethod
    def __init__(
        self,
        forecaster,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        self.forecaster = forecaster
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score

    def __sklearn_tags__(self):
        """Get tags from best_forecaster_ if available."""
        if hasattr(self, "best_forecaster_"):
            return self.best_forecaster_.__sklearn_tags__()
        # Return tags from the base forecaster during initialization
        return self.forecaster.__sklearn_tags__()

    @property
    def n_features_in_(self):
        """Number of features seen during fit."""
        check_is_fitted(self)
        return self.best_forecaster_.n_features_in_

    @property
    def feature_names_in_(self):
        """Names of features seen during fit."""
        check_is_fitted(self)
        return self.best_forecaster_.feature_names_in_

    def _check_refit_for_multimetric(self, scores):
        """Check that refit parameter is valid for multimetric scoring.

        Parameters
        ----------
        scores : dict
            Dictionary of scorer names to scores.

        Raises
        ------
        ValueError
            If refit is not valid for multimetric scoring.
        """
        if not (
            isinstance(self.refit, str)
            or isinstance(self.refit, (bool, type(None)))
            or callable(self.refit)
        ):
            raise ValueError(
                f"For multi-metric scoring, refit parameter must be a str, bool, None, or callable, "
                f"got {self.refit}."
            )

        if isinstance(self.refit, str):
            if self.refit not in scores:
                raise ValueError(
                    f"For multi-metric scoring, refit={self.refit!r} is not "
                    f"a valid scorer name. Available scorers: {sorted(scores)}."
                )

    @staticmethod
    def _select_best_index(refit, refit_metric, results):
        """Select the index of the best parameter combination.

        Parameters
        ----------
        refit : bool, str, or callable
            Refit parameter from constructor.
        refit_metric : str
            Name of the metric to use for refitting.
        results : dict
            Cross-validation results dictionary.

        Returns
        -------
        best_index : int
            Index of the best parameter combination.
        """
        if callable(refit):
            # If callable, refit is expected to return the index of the best
            # parameter set.
            best_index = refit(results)
            if not isinstance(best_index, numbers.Integral):
                raise TypeError("best_index_ returned is not an integer")
            if best_index < 0 or best_index >= len(results["params"]):
                raise IndexError("best_index_ index out of range")
        else:
            best_index = results[f"rank_test_{refit_metric}"].argmin()
        return best_index

    def _get_scorers(self):
        """Get the scorer(s) to be used.

        This is used in fit and get_metadata_routing.

        Returns
        -------
        scorers : BaseScorer or _MultimetricScorer
            Scorer to use.
        refit_metric : str
            Name of the metric to use for refitting.
        """
        refit_metric = "score"

        if self.scoring is None:
            raise ValueError("scoring parameter cannot be None")

        refit_metric = "score"  # Default for single metric

        if callable(self.scoring):
            if not isinstance(self.scoring, BaseScorer):
                raise ValueError(
                    "scoring must be an instance of BaseScorer or a dict of BaseScorer instances"
                )
            scorers = self.scoring
            # Single metric, default name is "score"
        elif isinstance(self.scoring, dict):
            # Multi-metric scoring
            scorers_dict = _check_scoring(self.forecaster, self.scoring)
            self._check_refit_for_multimetric(self.scoring)
            refit_metric = self.refit if isinstance(self.refit, str) else "score"
            scorers = _MultimetricScorer(
                scorers=scorers_dict, raise_exc=(self.error_score == "raise")
            )
        else:
            raise ValueError(
                "scoring must be an instance of BaseScorer or a dict of BaseScorer instances"
            )

        return scorers, refit_metric

    def _get_routed_params_for_fit(self, params):
        """Get the parameters to be used for routing.

        This is a method instead of a snippet in fit since it's used twice,
        here in fit, and potentially in subclasses.

        Parameters
        ----------
        params : dict
            Parameters passed to fit.

        Returns
        -------
        routed_params : Bunch
            Routed parameters for forecaster, scorer, and splitter.
        """
        if _routing_enabled():
            routed_params = process_routing(self, "fit", **params)
        else:
            from sklearn.utils import Bunch

            # Legacy behavior without metadata routing
            params = params.copy()
            routed_params = Bunch(
                forecaster=Bunch(fit=params),
                splitter=Bunch(split={}),
                scorer=Bunch(score={}),
            )

        return routed_params

    @abstractmethod
    def _run_search(self, evaluate_candidates):
        """Execute the search strategy.

        Subclasses implement specific search strategies (grid, random, etc.).

        Parameters
        ----------
        evaluate_candidates : callable
            This callback accepts:
                - a list of candidates, where each candidate is a dict of
                  parameter settings.
                - an optional `cv` parameter which can be used to e.g.
                  evaluate candidates on different dataset splits.
                - an optional `more_results` dict. Each key will be added to
                  the `cv_results_` attribute. Values should be lists of
                  length `n_candidates`.

            It returns a dict of all results so far, formatted like `cv_results_`.

        Examples
        --------
        GridSearchCV implements::

            def _run_search(self, evaluate_candidates):
                evaluate_candidates(ParameterGrid(self.param_grid))

        RandomizedSearchCV implements::

            def _run_search(self, evaluate_candidates):
                evaluate_candidates(
                    ParameterSampler(
                        self.param_distributions,
                        self.n_iter,
                        random_state=self.random_state
                    )
                )
        """
        raise NotImplementedError("_run_search not implemented.")

    def _format_results(self, candidate_params, n_splits, out, more_results=None):
        """Format the cv_results_ dictionary.

        Parameters
        ----------
        candidate_params : list of dict
            List of parameter dictionaries.
        n_splits : int
            Number of cross-validation splits.
        out : list of dict
            List of fit/score results from _fit_and_score.
        more_results : dict, optional
            Additional results to include in cv_results_.

        Returns
        -------
        results : dict
            Formatted cv_results_ dictionary.
        """
        n_candidates = len(candidate_params)

        # _aggregate_score_dicts returns a dict with keys: fit_time, score_time, test_scores, train_scores
        out = _aggregate_score_dicts(out)

        test_scores = _normalize_score_results(out["test_scores"])
        if self.return_train_score:
            train_scores = _normalize_score_results(out["train_scores"])

        results = {}

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """Store scores/times to cv_results_."""
            # When test scores are below 0, smaller is better
            array = np.array(array, dtype=np.float64).reshape(n_candidates, n_splits)
            if splits:
                for split_idx in range(n_splits):
                    results[f"split{split_idx}_{key_name}"] = array[:, split_idx]

            array_means = np.average(array, axis=1, weights=weights)
            results[f"mean_{key_name}"] = array_means

            if key_name.startswith(("train_", "test_")) and np.any(~np.isfinite(array_means)):
                warnings.warn(
                    f"One or more of the {key_name.split('_')[0]} scores "
                    "are non-finite: "
                    f"{array_means}",
                    category=UserWarning,
                )

            # Weighted std is not directly available in np
            # However, std = sqrt(mean(abs(x - mean(x))^2))
            # The weighted variance is then:
            # var = mean(w * (x - mean(x))^2)
            if weights is not None:
                array_stds = np.sqrt(
                    np.average((array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights)
                )
            else:
                array_stds = np.std(array, axis=1)
            results[f"std_{key_name}"] = array_stds

            if rank:
                # When the fit/scoring fails, array_means contains NaNs, we
                # will exclude them from the ranking process and consider them
                # as tied with the worst performers.
                if np.isnan(array_means).all():
                    # All fit/scoring routines failed
                    rank_result = np.ones_like(array_means, dtype=np.int32)
                else:
                    min_array_means = np.nanmin(array_means) - 1
                    array_means_ranked = xpx.nan_to_num(array_means, fill_value=min_array_means)
                    rank_result = rankdata(-array_means_ranked, method="min").astype(np.int32)
                results[f"rank_{key_name}"] = rank_result

        _store("fit_time", out["fit_time"])
        _store("score_time", out["score_time"])
        # Use one MaskedArray for all the test scores per scorer
        # because it's more memory efficient than storing multiple
        # MaskedArrays.
        test_sample_counts = np.array(out.get("n_test_samples", []), dtype=np.int32)
        if len(test_sample_counts) > 0:
            test_sample_counts = test_sample_counts.reshape(n_candidates, n_splits)
        
        # Store test scores - test_scores is a dict with scorer names as keys
        for scorer_name in test_scores:
            _store(
                f"test_{scorer_name}",
                test_scores[scorer_name],
                splits=True,
                rank=True,
                weights=test_sample_counts if len(test_sample_counts) > 0 else None,
            )
        
        if self.return_train_score:
            for scorer_name in train_scores:
                _store(f"train_{scorer_name}", train_scores[scorer_name], splits=True)

        # Store parameters
        results["params"] = candidate_params
        for param_name, param_values in _yield_masked_array_for_each_param(candidate_params):
            results[f"param_{param_name}"] = param_values

        if more_results is not None:
            for key, value in more_results.items():
                results[key] = value

        return results

    def get_metadata_routing(self):
        """Get metadata routing for this object.

        Returns
        -------
        routing : MetadataRouter
            A MetadataRouter encapsulating routing information.
        """
        router = MetadataRouter(owner=self)

        # Add forecaster routing
        router.add(
            forecaster=self.forecaster,
            method_mapping=MethodMapping()
            .add(caller="fit", callee="fit")
            .add(caller="predict", callee="predict")
            .add(caller="predict_interval", callee="predict_interval")
            .add(caller="update", callee="update")
            .add(caller="update_predict", callee="update_predict")
            .add(caller="update_predict_interval", callee="update_predict_interval")
            .add(caller="reset", callee="reset")
            .add(caller="score", callee="predict"),
        )

        # Add scorer routing
        scorers, _ = self._get_scorers()
        # Always add as "scorer" regardless of single or multi-metric
        # _MultimetricScorer handles routing internally
        router.add(
            scorer=scorers,
            method_mapping=MethodMapping().add(caller="fit", callee="score"),
        )

        # Add CV splitter routing (if applicable)
        router.add(
            splitter=self.cv,
            method_mapping=MethodMapping().add(caller="fit", callee="split"),
        )

        return router

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y, X=None, forecasting_horizon=1, **params):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with "time" column.

        X : pl.DataFrame, optional
            Exogenous feature time series with "time" column.

        forecasting_horizon : int, default=1
            Number of time steps to forecast.

        **params : dict of str -> object
            Parameters passed to the fit method of the forecaster, the scorer,
            and the CV splitter. Use metadata routing to control parameter flow.

        Returns
        -------
        self : BaseSearchCV
            Fitted search object.
        """
        _raise_for_params(params, self, "fit")

        # Validate input data - just check inputs, don't use validate_forecaster_data since it expects a forecaster
        from yohou.utils.validation import check_inputs, check_time_column, check_panel_internal_consistency, check_panel_groups_match
        if y is None:
            raise ValueError("`y` cannot be None")
        check_time_column(y)
        check_panel_internal_consistency(y, "y")
        if X is not None:
            check_time_column(X)
            check_panel_internal_consistency(X, "X")
            check_panel_groups_match(y, X)
        interval = check_inputs(y, X)

        scorers, refit_metric = self._get_scorers()

        y, X = indexable(y, X)
        params = _check_method_params(y, params=params)

        routed_params = self._get_routed_params_for_fit(params)

        # Get CV splitter
        cv_orig = check_cv(self.cv, forecasting_horizon)
        n_splits = cv_orig.get_n_splits(y, X, **routed_params.splitter.split)

        base_forecaster = clone(self.forecaster)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=routed_params.forecaster.fit,
            predict_params=routed_params.forecaster.predict,
            score_params=routed_params.scorer.score,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                """Evaluate candidate parameter settings using cross-validation."""
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        f"Fitting {n_splits} folds for each of {n_candidates} candidates,"
                        f" totalling {n_candidates * n_splits} fits"
                    )

                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_forecaster),
                        y,
                        X,
                        forecasting_horizon,
                        train=train,
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for cand_idx, parameters in enumerate(candidate_params)
                    for split_idx, (train, test) in enumerate(
                        cv.split(y, X, **routed_params.splitter.split)
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        f"cv.split and cv.get_n_splits returned "
                        f"inconsistent results. Expected {n_candidates * n_splits} "
                        f"splits, got {len(out)}"
                    )

                _warn_or_raise_about_fit_failures(out, self.error_score)

                # For callable self.scoring, the return type is only known after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]["test_scores"]
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callable scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(self.refit, refit_metric, results)
            if not callable(self.refit):
                # With a non-custom callable, we can select the best score
                # based on the best index
                self.best_score_ = results[f"mean_test_{refit_metric}"][self.best_index_]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # Clone the forecaster and parameters for refitting
            self.best_forecaster_ = clone(base_forecaster).set_params(
                **clone(self.best_params_, safe=False)
            )

            refit_start_time = time.time()
            # Don't pass y/X in routed params - they're positional
            fit_params_filtered = {k: v for k, v in routed_params.forecaster.fit.items() if k not in ['y', 'X']}
            self.best_forecaster_.fit(y, X, forecasting_horizon, **fit_params_filtered)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            # Fit scorers on the training data if they have a fit method
            if isinstance(scorers, _MultimetricScorer):
                scorers.fit(y)
            elif hasattr(scorers, "fit"):
                scorers.fit(y)

        # Store the scorer (not as dict for single metric evaluation)
        if isinstance(scorers, _MultimetricScorer):
            self.scorer_ = scorers._scorers
        else:
            self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    @available_if(_search_forecaster_has("predict"))
    def predict(self, forecasting_horizon=None, X=None, panel_group_names=None, **params):
        """Make predictions using the best forecaster.

        Parameters
        ----------
        forecasting_horizon : int, optional
            Number of time steps to forecast. If None, uses the horizon from fit.

        X : pl.DataFrame, optional
            Exogenous feature time series.

        panel_group_names : list of str, optional
            Panel group names to predict for.

        **params : dict
            Parameters passed to the predict method.

        Returns
        -------
        y_pred : pl.DataFrame
            Predicted time series.
        """
        check_is_fitted(self)
        _raise_for_params(params, self, "predict")
        return self.best_forecaster_.predict(
            forecasting_horizon=forecasting_horizon,
            X=X,
            panel_group_names=panel_group_names,
            **params,
        )

    @available_if(_search_forecaster_has("interval"))
    def predict_interval(
        self,
        forecasting_horizon=None,
        X=None,
        coverage_rates=None,
        panel_group_names=None,
        **params,
    ):
        """Make interval predictions using the best forecaster.

        Parameters
        ----------
        forecasting_horizon : int, optional
            Number of time steps to forecast.

        X : pl.DataFrame, optional
            Exogenous feature time series.

        coverage_rates : list of float, optional
            Coverage rates for prediction intervals.

        panel_group_names : list of str, optional
            Panel group names to predict for.

        **params : dict
            Parameters passed to the predict_interval method.

        Returns
        -------
        y_pred : pl.DataFrame
            Predicted intervals.
        """
        check_is_fitted(self)
        _raise_for_params(params, self, "predict_interval")
        return self.best_forecaster_.predict_interval(
            forecasting_horizon=forecasting_horizon,
            X=X,
            coverage_rates=coverage_rates,
            panel_group_names=panel_group_names,
            **params,
        )

    @available_if(_search_forecaster_has("update"))
    def update(self, y, X=None, panel_group_names=None):
        """Update the best forecaster with new observations.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.

        X : pl.DataFrame, optional
            New exogenous features.

        panel_group_names : list of str, optional
            Panel group names to update.

        Returns
        -------
        self : BaseSearchCV
            Updated search object.
        """
        check_is_fitted(self)
        self.best_forecaster_.update(y, X, panel_group_names)
        return self

    @available_if(_search_forecaster_has("reset"))
    def reset(self, y, X=None, panel_group_names=None):
        """Reset the best forecaster observation horizon.

        Parameters
        ----------
        y : pl.DataFrame
            Target observations to reset to.

        X : pl.DataFrame, optional
            Exogenous features to reset to.

        panel_group_names : list of str, optional
            Panel group names to reset.

        Returns
        -------
        self : BaseSearchCV
            Reset search object.
        """
        check_is_fitted(self)
        self.best_forecaster_.reset(y, X, panel_group_names)
        return self

    @available_if(_search_forecaster_has("update_predict"))
    def update_predict(
        self, y, X=None, panel_group_names=None, predict_transformed=False, **params
    ):
        """Update the best forecaster and make predictions.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.

        X : pl.DataFrame, optional
            New exogenous features and future features.

        panel_group_names : list of str, optional
            Panel group names to update and predict.

        predict_transformed : bool, default=False
            Whether to return transformed predictions.

        **params : dict
            Parameters passed to the update_predict method.

        Returns
        -------
        y_pred : pl.DataFrame
            Predicted time series.
        """
        check_is_fitted(self)
        _raise_for_params(params, self, "update_predict")
        return self.best_forecaster_.update_predict(
            y, X, panel_group_names, predict_transformed, **params
        )

    @available_if(_search_forecaster_has("update_predict_interval"))
    def update_predict_interval(
        self, y, X=None, coverage_rates=None, panel_group_names=None, **params
    ):
        """Update the best forecaster and make interval predictions.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.

        X : pl.DataFrame, optional
            New exogenous features and future features.

        coverage_rates : list of float, optional
            Coverage rates for prediction intervals.

        panel_group_names : list of str, optional
            Panel group names to update and predict.

        **params : dict
            Parameters passed to the update_predict_interval method.

        Returns
        -------
        y_pred : pl.DataFrame
            Predicted intervals.
        """
        check_is_fitted(self)
        _raise_for_params(params, self, "update_predict_interval")
        return self.best_forecaster_.update_predict_interval(
            y, X, coverage_rates, panel_group_names, **params
        )

    def score(self, y, X=None, **params):
        """Score the best forecaster on the given data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        X : pl.DataFrame, optional
            Exogenous features.

        **params : dict
            Parameters passed to the scorer.

        Returns
        -------
        score : float or dict
            Score or dictionary of scores.
        """
        check_is_fitted(self)
        _raise_for_params(params, self, "score")

        # Make predictions (don't pass score params to predict)
        y_pred = self.best_forecaster_.predict(X=X)

        # Score using the fitted scorer
        if isinstance(self.scorer_, dict):
            scores = {}
            for name, scorer in self.scorer_.items():
                scores[name] = scorer(y, y_pred, **params)
            return scores
        else:
            return self.scorer_(y, y_pred, **params)


class GridSearchCV(BaseSearchCV):
    """Exhaustive search over specified parameter values for a forecaster.

    GridSearchCV implements a "fit" method that evaluates all parameter
    combinations specified in param_grid using cross-validation.
    The parameters of the forecaster are optimized by cross-validated
    grid-search over a parameter grid.

    Parameters
    ----------
    forecaster : BaseForecaster
        A forecaster object that implements the yohou forecaster interface.

    param_grid : dict or list of dict
        Dictionary with parameter names (str) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored.

    scoring : BaseScorer or dict of BaseScorer, optional
        Strategy to evaluate the performance of the cross-validated model.
        If a BaseScorer, a single score is computed. If a dict, multiple
        scores are computed and refit parameter determines which is used
        for refitting.

    n_jobs : int, optional
        Number of jobs to run in parallel. None means 1.
        -1 means using all processors.

    refit : bool, str, or callable, default=True
        Refit a forecaster using the best found parameters on the whole dataset.
        For multiple metric evaluation, this needs to be a str denoting the
        scorer that would be used to find the best parameters for refitting.
        If callable, refit(cv_results_) should return the selected best_index_.

    cv : int, BaseSplitter, or iterable, optional
        Determines the cross-validation splitting strategy.
        If int, ExpandingWindowSplitter with specified n_splits is used.
        If None, ExpandingWindowSplitter with 5 splits is used.

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs dispatched during parallel execution.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in forecaster fitting.
        If set to 'raise', the error is raised.

    return_train_score : bool, default=False
        If False, the cv_results_ attribute will not include training scores.

    Attributes
    ----------
    cv_results_ : dict of numpy masked arrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas DataFrame. Contains all cross-validation results.

    best_forecaster_ : BaseForecaster
        Forecaster that was chosen by the search, i.e. forecaster which gave
        highest score (or smallest loss if specified) on the left out data.
        Only available if refit=True.

    best_score_ : float
        Mean cross-validated score of the best_forecaster.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the cv_results_ arrays) which corresponds to the best
        candidate parameter setting.

    scorer_ : BaseScorer or dict
        Scorer function(s) used on the held out data to choose the best
        parameters for the model.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best forecaster on the whole dataset.
        Only present if refit=True.

    multimetric_ : bool
        Whether multiple metrics were passed to scoring.

    Examples
    --------
    >>> from yohou.point_forecaster import PointReductionForecaster
    >>> from yohou.model_selection import GridSearchCV
    >>> from yohou.metrics import MeanAbsoluteError
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> # Create sample data
    >>> dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
    >>> y = pl.DataFrame({"time": dates, "value": range(100)})
    >>> # Define parameter grid
    >>> param_grid = {
    ...     "estimator__alpha": [0.1, 1.0, 10.0],
    ...     "feature_transformer__lags": [[1], [1, 2]],
    ... }
    >>> # Create search object
    >>> search = GridSearchCV(
    ...     forecaster=PointReductionForecaster(),
    ...     param_grid=param_grid,
    ...     scoring=MeanAbsoluteError(),
    ...     cv=3,
    ... )
    >>> # Fit and find best parameters
    >>> search.fit(y, forecasting_horizon=5)  # doctest: +SKIP
    >>> search.best_params_  # doctest: +SKIP
    """

    _parameter_constraints: dict = {
        **BaseSearchCV._parameter_constraints,
        "param_grid": [dict, list],
    }

    def __init__(
        self,
        forecaster,
        param_grid,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        super().__init__(
            forecaster=forecaster,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.param_grid = param_grid

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid."""
        evaluate_candidates(ParameterGrid(self.param_grid))


class RandomizedSearchCV(BaseSearchCV):
    """Randomized search over specified parameter distributions for a forecaster.

    RandomizedSearchCV implements a "fit" method that samples n_iter
    parameter settings from the specified distributions using cross-validation.
    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions.

    Parameters
    ----------
    forecaster : BaseForecaster
        A forecaster object that implements the yohou forecaster interface.

    param_distributions : dict
        Dictionary with parameter names (str) as keys and distributions
        or lists of parameters to try. Distributions must provide a rvs
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring : BaseScorer or dict of BaseScorer, optional
        Strategy to evaluate the performance of the cross-validated model.
        If a BaseScorer, a single score is computed. If a dict, multiple
        scores are computed and refit parameter determines which is used
        for refitting.

    n_jobs : int, optional
        Number of jobs to run in parallel. None means 1.
        -1 means using all processors.

    refit : bool, str, or callable, default=True
        Refit a forecaster using the best found parameters on the whole dataset.
        For multiple metric evaluation, this needs to be a str denoting the
        scorer that would be used to find the best parameters for refitting.
        If callable, refit(cv_results_) should return the selected best_index_.

    cv : int, BaseSplitter, or iterable, optional
        Determines the cross-validation splitting strategy.
        If int, ExpandingWindowSplitter with specified n_splits is used.
        If None, ExpandingWindowSplitter with 5 splits is used.

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs dispatched during parallel execution.

    random_state : int or None, default=None
        Pseudo random number generator state used for random uniform sampling.
        Pass an int for reproducible output across multiple function calls.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in forecaster fitting.
        If set to 'raise', the error is raised.

    return_train_score : bool, default=False
        If False, the cv_results_ attribute will not include training scores.

    Attributes
    ----------
    cv_results_ : dict of numpy masked arrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas DataFrame. Contains all cross-validation results.

    best_forecaster_ : BaseForecaster
        Forecaster that was chosen by the search, i.e. forecaster which gave
        highest score (or smallest loss if specified) on the left out data.
        Only available if refit=True.

    best_score_ : float
        Mean cross-validated score of the best_forecaster.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the cv_results_ arrays) which corresponds to the best
        candidate parameter setting.

    scorer_ : BaseScorer or dict
        Scorer function(s) used on the held out data to choose the best
        parameters for the model.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best forecaster on the whole dataset.
        Only present if refit=True.

    multimetric_ : bool
        Whether multiple metrics were passed to scoring.

    Examples
    --------
    >>> from yohou.point_forecaster import PointReductionForecaster
    >>> from yohou.model_selection import RandomizedSearchCV
    >>> from yohou.metrics import MeanAbsoluteError
    >>> from scipy.stats import uniform
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> # Create sample data
    >>> dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
    >>> y = pl.DataFrame({"time": dates, "value": range(100)})
    >>> # Define parameter distributions
    >>> param_distributions = {
    ...     "estimator__alpha": uniform(0.01, 10.0),
    ...     "feature_transformer__lags": [[1], [1, 2], [1, 2, 3]],
    ... }
    >>> # Create search object
    >>> search = RandomizedSearchCV(
    ...     forecaster=PointReductionForecaster(),
    ...     param_distributions=param_distributions,
    ...     n_iter=20,
    ...     scoring=MeanAbsoluteError(),
    ...     cv=3,
    ...     random_state=42,
    ... )
    >>> # Fit and find best parameters
    >>> search.fit(y, forecasting_horizon=5)  # doctest: +SKIP
    >>> search.best_params_  # doctest: +SKIP
    """

    _parameter_constraints: dict = {
        **BaseSearchCV._parameter_constraints,
        "param_distributions": [dict, list],
        "n_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        forecaster,
        param_distributions,
        *,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
    ):
        super().__init__(
            forecaster=forecaster,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _run_search(self, evaluate_candidates):
        """Sample n_iter candidates from param_distributions."""
        evaluate_candidates(
            ParameterSampler(self.param_distributions, self.n_iter, random_state=self.random_state)
        )
