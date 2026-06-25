"""Grid and randomized hyperparameter search with time series cross-validation."""

from __future__ import annotations

import numbers
import time
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Any, cast

import numpy as np
import polars as pl
from numpy.ma import MaskedArray
from scipy.stats import rankdata
from sklearn.base import (
    MetaEstimatorMixin,
    clone,
)
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    process_routing,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    check_is_fitted,
    indexable,
)

from yohou.base import BaseForecaster
from yohou.metrics.base import BaseScorer
from yohou.utils import validate_search_data
from yohou.utils._compat import (
    HasMethods,
    Interval,
    StrOptions,
    _aggregate_score_dicts,
    _check_method_params,
    _fit_context,
    _insert_error_scores,
    _normalize_score_results,
    _raise_for_params,
    _warn_or_raise_about_fit_failures,
)

from .split import check_cv
from .utils import (
    _check_scoring,
    _collect_coverage_rates,
    _fit_and_score,
    _MultimetricScorer,
    _resolve_response_method,
    _validate_forecaster_scorer_compatibility,
)

__all__ = ["BaseSearchCV", "GridSearchCV", "RandomizedSearchCV"]


def _search_forecaster_has(attr):
    """Check if SearchCV.best_forecaster_ has a given attribute or prediction type.

    This function is used as the check for sklearn.utils.metaestimators.available_if
    decorator to ensure predict/observe methods are only available when refit=True and
    the best_forecaster_ supports the required functionality.

    The function performs forecaster-specific checks:
    - For special attributes "point" or "interval", it checks the forecaster_type
      via __sklearn_tags__ to determine prediction type support
    - For other attributes (methods like "rewind", "observe"), it checks attribute
      existence via hasattr(best_forecaster_, attr)

    This enables conditional method delegation based on the underlying forecaster's
    capabilities, ensuring that SearchCV only exposes methods that the selected
    best_forecaster_ can handle.

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
        The returned function accepts a SearchCV instance and returns bool.

    Examples
    --------
    >>> # Check for point prediction support via forecaster_type tag
    >>> def predict(self, forecasting_horizon=1):
    ...     return self.best_forecaster_.predict(forecasting_horizon)
    >>> predict = available_if(_search_forecaster_has("point"))(predict)
    >>>
    >>> # Check for specific method existence
    >>> def rewind(self, y, X_actual=None):
    ...     return self.best_forecaster_.rewind(y, X_actual)
    >>> rewind = available_if(_search_forecaster_has("rewind"))(rewind)
    >>>
    >>> # Check for interval prediction support
    >>> def predict_interval(self, forecasting_horizon=1, coverage_rates=None):
    ...     return self.best_forecaster_.predict_interval(...)
    >>> predict_interval = available_if(_search_forecaster_has("interval"))(predict_interval)

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
        if attr in {"point", "interval", "class_proba"}:
            tags = self.best_forecaster_.__sklearn_tags__()
            forecaster_type = getattr(tags.forecaster_tags, "forecaster_type", None)
            if forecaster_type is None:
                return False
            return attr in forecaster_type

        # Otherwise check attribute existence
        return hasattr(self.best_forecaster_, attr)

    return check


def _yield_masked_array_for_each_param(candidate_params):
    """Yield masked arrays for each parameter in candidate_params.

    This function creates masked arrays for storing parameter values in cv_results_.
    For each unique parameter name across all candidates, it generates a masked array
    where entries are masked if that parameter was not present in the candidate dict.

    This allows cv_results_ to represent parameter grids where not all parameters
    are present in every candidate (e.g., when using multiple param_grid dicts).
    The masked array approach is memory-efficient compared to storing multiple
    separate arrays.

    Parameters
    ----------
    candidate_params : list of dict
        List of parameter dictionaries, one per candidate parameter setting.
        Each dict maps parameter names to values.

    Yields
    ------
    param_name : str
        Parameter name (sorted alphabetically for consistent ordering).
    param_values : MaskedArray
        Masked array of parameter values with shape (n_candidates,).
        Entries are masked (True) where parameter is not present in candidate,
        unmasked (False) where parameter exists.

    Examples
    --------
    >>> candidate_params = [
    ...     {"alpha": 1.0, "l1_ratio": 0.5},
    ...     {"alpha": 10.0, "l1_ratio": 0.7},
    ...     {"alpha": 100.0},  # l1_ratio not present
    ... ]
    >>> for param_name, param_values in _yield_masked_array_for_each_param(candidate_params):
    ...     print(f"{param_name}: {param_values}")
    alpha: [1.0 10.0 100.0]
    l1_ratio: [0.5 0.7 --]

    """
    all_param_names = {key for params in candidate_params for key in params}

    for param_name in sorted(all_param_names):
        param_values = []
        mask = []
        for params in candidate_params:
            if param_name in params:
                param_values.append(params[param_name])
                mask.append(False)
            else:
                param_values.append(None)  # Placeholder for masked value
                mask.append(True)

        param_array = MaskedArray(
            param_values,
            mask=mask,
            dtype=object,
        )
        yield param_name, param_array


class BaseSearchCV(BaseForecaster, MetaEstimatorMixin, metaclass=ABCMeta):
    """Abstract base class for hyperparameter search with cross-validation.

    Warning: This class should not be used directly. Use derived classes
    ``GridSearchCV`` and ``RandomizedSearchCV`` instead.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table::

            +-----------+------------+-------------------+---+-----------------+
            |param_alpha|param_gamma |param_degree       |...|rank_test_score  |
            +===========+============+===================+===+=================+
            |  1.0      |     --     |         --        |...|        2        |
            +-----------+------------+-------------------+---+-----------------+
            |  10.0     |     --     |         --        |...|        1        |
            +-----------+------------+-------------------+---+-----------------+
            |  100.0    |     --     |         --        |...|        3        |
            +-----------+------------+-------------------+---+-----------------+
            |   --      |    0.1     |         2         |...|        5        |
            +-----------+------------+-------------------+---+-----------------+
            |   --      |    0.2     |         3         |...|        4        |
            +-----------+------------+-------------------+---+-----------------+

        will be represented by a cv_results_ dict of::

            {
            'param_alpha': masked_array(data=[1.0, 10.0, 100.0, --, --],
                                        mask=[False False False  True  True]...),
            'param_gamma': masked_array(data=[--, --, --, 0.1, 0.2],
                                       mask=[ True  True  True False False]...),
            'param_degree': masked_array(data=[--, --, --, 2, 3],
                                         mask=[ True  True  True False False]...),
            'split0_test_score'  : [0.80, 0.70, 0.60, 0.75, 0.85],
            'split1_test_score'  : [0.82, 0.75, 0.65, 0.72, 0.80],
            'mean_test_score'    : [0.81, 0.725, 0.625, 0.735, 0.825],
            'std_test_score'     : [0.01, 0.025, 0.025, 0.015, 0.025],
            'rank_test_score'    : [2, 3, 5, 4, 1],
            'split0_train_score' : [0.85, 0.80, 0.75, 0.82, 0.90],
            'split1_train_score' : [0.87, 0.82, 0.77, 0.84, 0.92],
            'mean_train_score'   : [0.86, 0.81, 0.76, 0.83, 0.91],
            'std_train_score'    : [0.01, 0.01, 0.01, 0.01, 0.01],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49, 0.55],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01, 0.02],
            'mean_score_time'    : [0.01, 0.06, 0.04, 0.04, 0.05],
            'std_score_time'     : [0.00, 0.00, 0.00, 0.01, 0.00],
            'params'             : [{'alpha': 1.0}, {'alpha': 10.0}, ...],
            }

        NOTE: The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. (``'split0_test_mae'``, ``'mean_train_rmse'`` etc.)

    best_forecaster_ : BaseForecaster
        Forecaster that was chosen by the search, i.e. forecaster which gave
        highest score (or smallest loss if specified) on the left out data.
        Not available if ``refit=False``.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_forecaster_.

        Follows sklearn's sign convention: for ``lower_is_better`` scorers
        (e.g. MAE, RMSE) the value is **negated** so that higher always
        means better.  The raw metric value is ``-best_score_``.

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

    scorer_ : BaseScorer or dict
        Scorer function(s) used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best forecaster on the whole dataset.

        This is present only if ``refit`` is not False.

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    n_features_in_ : int
        Number of features seen during ``fit``. Only defined if
        ``best_forecaster_`` is defined (see the documentation for the ``refit``
        parameter for more details) and that ``best_forecaster_`` exposes
        ``n_features_in_`` when fit.

    feature_names_in_ : ndarray of shape (``n_features_in_``,)
        Names of features seen during ``fit``. Only defined if
        ``best_forecaster_`` is defined (see the documentation for the ``refit``
        parameter for more details) and that ``best_forecaster_`` exposes
        ``feature_names_in_`` when fit.

    See Also
    --------
    - [`GridSearchCV`][yohou.model_selection.search.GridSearchCV] : Exhaustive search over specified parameter values.
    - [`RandomizedSearchCV`][yohou.model_selection.search.RandomizedSearchCV] : Randomized search over parameter distributions.

    """

    _parameter_constraints: dict = {
        "forecaster": [HasMethods(["fit", "predict"])],
        "scoring": [callable, dict],
        "n_jobs": [numbers.Integral, None],
        "refit": ["boolean", str, callable],
        "cv": [numbers.Integral, HasMethods(["split", "get_n_splits"]), None],
        "verbose": ["verbose"],
        "pre_dispatch": [numbers.Integral, str],
        "error_score": [StrOptions({"raise"}), numbers.Real],
        "return_train_score": ["boolean"],
    }

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Merge parameter constraints from all classes in the MRO."""
        super().__init_subclass__(**kwargs)
        merged: dict = {}
        for klass in reversed(cls.__mro__):
            own = klass.__dict__.get("_parameter_constraints")
            if own and isinstance(own, dict):
                merged.update(own)
        cls._parameter_constraints = merged

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

    @property
    def interval_(self):
        """Time interval detected during fit."""
        check_is_fitted(self)
        return self.best_forecaster_.interval_

    @property
    def groups_(self):
        """Panel group names detected during fit."""
        check_is_fitted(self)
        return self.best_forecaster_.groups_

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
        multimetric_refit_msg = (
            "For multi-metric scoring, the parameter refit must be set to a "
            "scorer key or a callable to refit a forecaster with the best "
            "parameter setting on the whole data and make the best_* "
            "attributes available for that metric. If this is not needed, "
            f"refit should be set to False explicitly. {self.refit!r} was "
            "passed."
        )

        valid_refit_dict = isinstance(self.refit, str) and self.refit in scores

        if self.refit is not False and not valid_refit_dict and not callable(self.refit):
            raise ValueError(multimetric_refit_msg)

    @staticmethod
    def _select_best_index(refit, refit_metric, results):
        """Select the index of the best parameter combination.

        This method implements the logic for choosing which parameter setting
        is "best" based on the refit parameter:

        - If refit is callable: Call refit(results) which must return an integer
          index into the results arrays. This allows custom selection logic.
        - Otherwise: Use argmin on rank_test_{refit_metric}, which selects the
          parameter setting with the best (rank 1) mean test score for that metric.

        The ranking approach (argmin of rank) handles ties consistently and ensures
        that lower rank numbers (better performance) are selected.

        Parameters
        ----------
        refit : bool, str, or callable
            Refit parameter from constructor. If callable, should accept cv_results_
            dict and return integer index.
        refit_metric : str
            Name of the metric to use for refitting (e.g., 'score', 'mae', 'rmse').
            Used to construct the key 'rank_test_{refit_metric}' in results dict.
        results : dict
            Cross-validation results dictionary (cv_results_) containing
            'rank_test_{refit_metric}' and 'params' keys.

        Returns
        -------
        best_index : int
            Index of the best parameter combination. This index can be used to
            access results['params'][best_index] for the best parameters.

        Raises
        ------
        TypeError
            If callable refit returns non-integer.
        IndexError
            If callable refit returns out-of-range index.

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
        if self.scoring is None:
            raise ValueError("scoring parameter cannot be None")

        refit_metric = "score"  # Default for single metric

        if callable(self.scoring):
            if not isinstance(self.scoring, BaseScorer):
                raise ValueError("scoring must be an instance of BaseScorer, got a callable that is not a BaseScorer")
            scorers = self.scoring
            # Single metric, default name is "score"
        elif isinstance(self.scoring, dict):
            # Multi-metric scoring
            # _check_scoring returns the dict directly when given a dict
            scorers_dict = cast(dict[str, BaseScorer], _check_scoring(self.scoring))
            self._check_refit_for_multimetric(self.scoring)
            refit_metric = self.refit if isinstance(self.refit, str) else "score"
            scorers = _MultimetricScorer(scorers=scorers_dict, raise_exc=(self.error_score == "raise"))
        else:
            raise ValueError("scoring must be an instance of BaseScorer or a dict of BaseScorer instances")

        return scorers, refit_metric

    def _get_routed_params_for_fit(self, params):
        """Get the parameters to be used for routing.

        This is extracted into a method so that subclasses can override
        routing behaviour if needed.

        Parameters
        ----------
        params : dict
            Parameters passed to fit.

        Returns
        -------
        routed_params : Bunch
            Routed parameters for forecaster, scorer, and splitter.
        """
        routed_params = process_routing(self, "fit", **params)
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
                    ParameterSampler(self.param_distributions, self.n_iter, random_state=self.random_state)
                )
        """
        raise NotImplementedError("_run_search not implemented.")

    def _format_results(self, candidate_params, n_splits, out, more_results=None):
        """Format the cv_results_ dictionary.

        This method aggregates results from parallel _fit_and_score calls into
        the cv_results_ dictionary structure with comprehensive statistics:

        - Per-split scores: split{i}_test_{scorer} for each fold
        - Aggregated scores: mean_test_{scorer}, std_test_{scorer}
        - Rankings: rank_test_{scorer} (1=best, higher=worse)
        - Timing: mean_fit_time, std_fit_time, mean_score_time, std_score_time
        - Parameters: param_{name} as masked arrays
        - Train scores: split{i}_train_{scorer}, mean_train_{scorer}, std_train_{scorer}
          (if return_train_score=True)

        For multi-metric scoring, scores for each metric are stored with keys
        ending in the scorer name (e.g., 'mean_test_mae', 'rank_test_rmse').

        The aggregation uses weighted averaging when test_sample_counts vary
        across folds, ensuring that folds with more samples contribute
        proportionally more to the mean score.

        Parameters
        ----------
        candidate_params : list of dict
            List of parameter dictionaries, one per candidate.
        n_splits : int
            Number of cross-validation splits.
        out : list of dict
            List of fit/score results from _fit_and_score, length n_candidates * n_splits.
            Each dict contains: test_scores, train_scores, fit_time, score_time, n_test_samples.
        more_results : dict, optional
            Additional results to include in cv_results_. Keys are column names,
            values are lists of length n_candidates.

        Returns
        -------
        results : dict
            Formatted cv_results_ dictionary with keys including:
            - 'params': List of parameter dicts
            - 'param_{name}': Masked array for each parameter
            - 'split{i}_test_{scorer}': Per-fold test scores
            - 'mean_test_{scorer}': Mean test score
            - 'std_test_{scorer}': Standard deviation of test scores
            - 'rank_test_{scorer}': Ranking (1=best)
            - 'mean_fit_time', 'std_fit_time': Fit timing statistics
            - 'mean_score_time', 'std_score_time': Score timing statistics
            - Train score keys (if return_train_score=True)

        """
        n_candidates = len(candidate_params)

        # _aggregate_score_dicts returns a dict with keys:
        # fit_time, score_time, test_scores, train_scores
        out = _aggregate_score_dicts(out)

        test_scores = _normalize_score_results(out["test_scores"])
        if self.return_train_score:
            train_scores = _normalize_score_results(out["train_scores"])

        results = {}

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """Store scores/times to cv_results_."""
            # Scores for lower_is_better metrics are negated in _score() so
            # that higher numeric values always indicate better performance
            # (matching sklearn's sign convention).  rankdata(-values) then
            # correctly assigns rank 1 to the best candidate.
            array = np.array(array, dtype=np.float64).reshape(n_candidates, n_splits)
            if splits:
                for split_idx in range(n_splits):
                    results[f"split{split_idx}_{key_name}"] = array[:, split_idx]

            array_means = np.average(array, axis=1, weights=weights)
            results[f"mean_{key_name}"] = array_means

            if key_name.startswith(("train_", "test_")) and np.any(~np.isfinite(array_means)):
                warnings.warn(
                    f"One or more of the {key_name.split('_')[0]} scores are non-finite: {array_means}",
                    stacklevel=2,
                    category=UserWarning,
                )

            # Weighted std is not directly available in np
            # However, std = sqrt(mean(abs(x - mean(x))^2))
            # The weighted variance is then:
            # var = sum(w * (x - mean(x))^2) / sum(w)
            if weights is not None:
                array_stds = np.sqrt(np.average((array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights))
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
                    array_means_ranked = np.nan_to_num(array_means, nan=min_array_means)
                    rank_result = rankdata(-array_means_ranked, method="min").astype(np.int32)
                results[f"rank_{key_name}"] = rank_result

        _store("fit_time", out["fit_time"])
        _store("score_time", out["score_time"])
        # Weight per-fold test scores by fold size when available.
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
            .add(caller="predict_class_proba", callee="predict_class_proba")
            .add(caller="observe_predict", callee="observe_predict")
            .add(caller="observe_predict_interval", callee="observe_predict_interval")
            .add(caller="observe_predict_class_proba", callee="observe_predict_class_proba"),
        )

        # Add scorer routing only when scoring is configured. With the
        # default scoring=None, the scorer is resolved at fit time, so there
        # is nothing to route yet and _get_scorers would raise.
        if self.scoring is not None:
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
    def fit(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: int = 1,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> BaseSearchCV:
        """Run fit with all sets of parameters.

        Performs cross-validated hyperparameter search using the specified
        parameter grid/distributions. For each parameter combination, fits
        the forecaster on each training fold and evaluates on the test fold.
        Optionally refits the best forecaster on the entire dataset if
        ``refit=True``.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Processed by the feature transformer to produce
            lags, rolling statistics, and other derived features. If
            ``None``, only target-derived features are used.
        forecasting_horizon : int, default=1
            Number of time steps to forecast into the future.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted search instance.

        """
        _raise_for_params(params, self, "fit")

        # Validate input data for search

        validate_search_data(y, X_actual)

        scorers, refit_metric = self._get_scorers()

        # Validate forecaster/scorer type compatibility (step 1)
        _validate_forecaster_scorer_compatibility(self.forecaster, scorers)

        # Determine prediction context from scorer response methods
        response_method = _resolve_response_method(scorers)
        collected_coverage_rates = _collect_coverage_rates(scorers) if response_method == "predict_interval" else None

        y, X_actual = indexable(y, X_actual)
        params = _check_method_params(y, params=params)

        routed_params = self._get_routed_params_for_fit(params)

        # Get CV splitter
        cv_orig = check_cv(self.cv, forecasting_horizon)
        n_splits = cv_orig.get_n_splits(y, X_actual, **routed_params.splitter.split)

        base_forecaster = clone(self.forecaster)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = {
            "scorer": scorers,
            "fit_params": routed_params.forecaster.fit,
            "predict_func_params": getattr(routed_params.forecaster, response_method, {}),
            "score_params": routed_params.scorer.score,
            "return_train_score": self.return_train_score,
            "return_n_test_samples": True,
            "return_times": True,
            "return_parameters": False,
            "error_score": self.error_score,
            "verbose": self.verbose,
            "coverage_rates": collected_coverage_rates,
        }
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
                    print(  # noqa: T201
                        f"Fitting {n_splits} folds for each of {n_candidates} "
                        f"candidates, totalling {n_candidates * n_splits} fits"
                    )

                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_forecaster),
                        y,
                        X_actual,
                        forecasting_horizon,
                        X_future=X_future,
                        X_forecast=X_forecast,
                        train=train,
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for cand_idx, parameters in enumerate(candidate_params)
                    for split_idx, (train, test) in enumerate(cv.split(y, X_actual, **routed_params.splitter.split))
                )

                if len(out) < 1:
                    raise ValueError("No fits were performed. Was the CV iterator empty? Were there no candidates?")
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
                results = self._format_results(all_candidate_params, n_splits, all_out, all_more_results)

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
            self.best_forecaster_ = clone(base_forecaster).set_params(**clone(self.best_params_, safe=False))

            refit_start_time = time.time()

            refit_params = dict(routed_params.forecaster.fit.items())
            if collected_coverage_rates is not None:
                refit_params["coverage_rates"] = collected_coverage_rates
            self.best_forecaster_.fit(
                y,
                X_actual,
                forecasting_horizon,
                X_future=X_future,
                X_forecast=X_forecast,
                **refit_params,
            )
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            # Fit scorers on the training data if they have a fit method
            if isinstance(scorers, _MultimetricScorer) or hasattr(scorers, "fit"):
                scorers.fit(y, forecaster=self.best_forecaster_)

        # Store the scorer (not as dict for single metric evaluation)
        if isinstance(scorers, _MultimetricScorer):
            self.scorer_ = scorers._scorers
        else:
            self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    @available_if(_search_forecaster_has("predict"))
    def predict(
        self,
        forecasting_horizon: int | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate point forecasts using the best forecaster.

        Parameters
        ----------
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future.  If ``None``,
            uses the horizon specified at fit time.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Point predictions with ``"vintage_time"``, ``"time"``, and one
            column per target variable.

        """
        check_is_fitted(self)
        _raise_for_params(params, self, "predict")
        return self.best_forecaster_.predict(
            forecasting_horizon=forecasting_horizon,
            groups=groups,
            X_future=X_future,
            X_forecast=X_forecast,
            **params,
        )

    @available_if(_search_forecaster_has("interval"))
    def predict_interval(
        self,
        forecasting_horizon: int | None = None,
        coverage_rates: list[float] | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate interval forecasts using the best forecaster.

        Parameters
        ----------
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future.  If ``None``,
            uses the horizon specified at fit time.
        coverage_rates : list of float or None, default=None
            Coverage levels for prediction intervals (e.g., ``[0.9, 0.95]``
            for 90 % and 95 % intervals).  If ``None``, defaults to the rates
            used at fit time.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Interval predictions with ``"vintage_time"``, ``"time"``, and
            lower/upper bound columns for each target at each coverage rate.

        """
        check_is_fitted(self)
        _raise_for_params(params, self, "predict_interval")
        return self.best_forecaster_.predict_interval(
            forecasting_horizon=forecasting_horizon,
            coverage_rates=coverage_rates,
            groups=groups,
            X_future=X_future,
            X_forecast=X_forecast,
            **params,
        )

    @available_if(_search_forecaster_has("observe"))
    def observe(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> BaseSearchCV:
        """Observe new data with the best forecaster without refitting.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X_actual : pl.DataFrame or None, default=None
            New actual feature observations with a ``"time"`` column
            aligned with ``y``. Forwarded to the best forecaster.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.

        Returns
        -------
        self
            The search object with updated observation buffers.

        """
        check_is_fitted(self)
        self.best_forecaster_.observe(y, X_actual, groups=groups, X_future=X_future, X_forecast=X_forecast)
        return self

    @available_if(_search_forecaster_has("rewind"))
    def rewind(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> BaseSearchCV:
        """Rewind the best forecaster observation buffers.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations to restore the observation state
            to. Must align with ``y``.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.

        Returns
        -------
        self
            The search object with rewound observation buffers.

        """
        check_is_fitted(self)
        self.best_forecaster_.rewind(y, X_actual, groups=groups, X_future=X_future, X_forecast=X_forecast)
        return self

    @available_if(_search_forecaster_has("observe_predict"))
    def observe_predict(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: int | None = None,
        groups: list[str] | None = None,
        stride: int | None = None,
        predict_transformed: bool = False,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Observe new data and generate point forecasts.

        Equivalent to calling ``observe(y, X_actual)`` then ``predict()``.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Sliced and observed incrementally at each step
            of the rolling loop.
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future.  If ``None``,
            uses the horizon specified at fit time.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.
        stride : int or None, default=None
            Step size for rolling update-predict.  If ``None``, defaults to
            ``forecasting_horizon``.
        predict_transformed : bool, default=False
            If ``True``, return predictions in the transformed space without
            applying inverse target transformation.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Point predictions with ``"vintage_time"``, ``"time"``, and one
            column per target variable.

        """
        check_is_fitted(self)
        _raise_for_params(params, self, "observe_predict")
        return self.best_forecaster_.observe_predict(
            y,
            X_actual=X_actual,
            forecasting_horizon=forecasting_horizon,
            groups=groups,
            stride=stride,
            predict_transformed=predict_transformed,
            X_future=X_future,
            X_forecast=X_forecast,
            **params,
        )

    @available_if(_search_forecaster_has("observe_predict_interval"))
    def observe_predict_interval(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: int | None = None,
        coverage_rates: list[float] | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Observe new data and generate interval forecasts.

        Equivalent to calling ``observe(y, X_actual)`` then
        ``predict_interval()``.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Sliced and observed incrementally at each step
            of the rolling loop.
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future.  If ``None``,
            uses the horizon specified at fit time.
        coverage_rates : list of float or None, default=None
            Coverage levels for prediction intervals (e.g., ``[0.9, 0.95]``
            for 90 % and 95 % intervals).  If ``None``, defaults to the rates
            used at fit time.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Interval predictions with ``"vintage_time"``, ``"time"``, and
            lower/upper bound columns for each target at each coverage rate.

        """
        check_is_fitted(self)
        _raise_for_params(params, self, "observe_predict_interval")
        return self.best_forecaster_.observe_predict_interval(
            y,
            X_actual=X_actual,
            forecasting_horizon=forecasting_horizon,
            coverage_rates=coverage_rates,
            groups=groups,
            X_future=X_future,
            X_forecast=X_forecast,
            **params,
        )

    @available_if(_search_forecaster_has("class_proba"))
    def predict_class_proba(
        self,
        forecasting_horizon: int | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate class-probability forecasts using the best forecaster.

        Parameters
        ----------
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future.  If ``None``,
            uses the horizon specified at fit time.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Class-probability predictions with ``"vintage_time"``,
            ``"time"``, and probability columns for each target.

        """
        check_is_fitted(self)
        _raise_for_params(params, self, "predict_class_proba")
        return self.best_forecaster_.predict_class_proba(
            forecasting_horizon=forecasting_horizon,
            groups=groups,
            X_future=X_future,
            X_forecast=X_forecast,
            **params,
        )

    @available_if(_search_forecaster_has("observe_predict_class_proba"))
    def observe_predict_class_proba(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: int | None = None,
        groups: list[str] | None = None,
        stride: int | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Observe new data and generate class-probability forecasts.

        Equivalent to calling ``observe(y, X_actual)`` then
        ``predict_class_proba()``.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more categorical value columns.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Sliced and observed incrementally at each step
            of the rolling loop.
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future.  If ``None``,
            uses the horizon specified at fit time.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.
        stride : int or None, default=None
            Step size for rolling update-predict.  If ``None``, defaults to
            ``forecasting_horizon``.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Class-probability predictions with ``"vintage_time"``,
            ``"time"``, and probability columns for each target.

        """
        check_is_fitted(self)
        _raise_for_params(params, self, "observe_predict_class_proba")
        return self.best_forecaster_.observe_predict_class_proba(
            y,
            X_actual=X_actual,
            forecasting_horizon=forecasting_horizon,
            groups=groups,
            stride=stride,
            X_future=X_future,
            X_forecast=X_forecast,
            **params,
        )


class GridSearchCV(BaseSearchCV):
    """Exhaustive search over specified parameter values for a forecaster.

    Important members are fit, predict, predict_interval, observe, and rewind.

    GridSearchCV implements a "fit" method that evaluates all parameter
    combinations specified in param_grid using time series cross-validation.
    The parameters of the forecaster are optimized by cross-validated
    grid-search over a parameter grid.

    It also implements "predict", "predict_interval", "observe", "rewind",
    "observe_predict", and "observe_predict_interval" if the underlying
    forecaster supports these methods and refit=True.

    Parameters
    ----------
    forecaster : BaseForecaster
        A forecaster object implementing the yohou forecaster interface
        with fit and predict methods.

    param_grid : dict or list of dict
        Dictionary with parameter names (`str`) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

        Examples::

            param_grid = {"estimator__alpha": [0.1, 1.0, 10.0]}
            param_grid = [
                {"estimator__alpha": [0.1, 1.0], "estimator__l1_ratio": [0.5, 0.7]},
                {"estimator__alpha": [10.0, 100.0]},
            ]

    scoring : BaseScorer or dict of {str: BaseScorer}
        Strategy to evaluate the performance of the cross-validated model
        on the test set.  Required; although the constructor default is
        ``None``, passing ``None`` raises ``ValueError`` at ``fit`` time.

        If a single BaseScorer instance, the same scorer is used for all
        folds and stored in cv_results_ with key 'score'.

        If a dict, keys are scorer names and values are BaseScorer instances.
        This enables multi-metric evaluation. The ``refit`` parameter must
        be set to a scorer name or False to determine which scorer to use
        for selecting the best parameters.

        Unlike sklearn, string scorer names are not supported. You must use
        yohou.metrics BaseScorer instances (e.g., MeanAbsoluteError(),
        RootMeanSquaredError()).

        Examples::

            scoring = MeanAbsoluteError()
            scoring = {"mae": MeanAbsoluteError(), "rmse": RootMeanSquaredError()}

        Note: For multi-metric evaluation with dict, cv_results_ will contain
        keys like 'mean_test_mae', 'rank_test_mae', 'mean_test_rmse', etc.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a ``joblib.parallel_backend`` context.
        ``-1`` means using all processors.

    refit : bool, str, or callable, default=True
        Refit a forecaster using the best found parameters on the whole dataset.

        For multiple metric evaluation, this needs to be a ``str`` denoting the
        scorer that would be used to find the best parameters for refitting
        the forecaster at the end.

        Where there are considerations other than maximum score in
        choosing a best forecaster, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given ``cv_results_``. In that
        case, the ``best_forecaster_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.

        The refitted forecaster is made available at the ``best_forecaster_``
        attribute and permits using ``predict`` directly on this
        ``GridSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

        Examples::

            refit = True  # Use single scorer or first scorer for multi-metric
            refit = "mae"  # For multi-metric: use 'mae' scorer to select best
            refit = False  # Don't refit, only evaluate
            refit = lambda cv_results: cv_results["rank_test_mae"].argmin()

    cv : int, BaseSplitter, or None, default=None
        Determines the cross-validation splitting strategy.

        Possible inputs for cv are:

        - None, to use the default 5-fold expanding window splitter
        - int, to specify the number of folds in an
          ``ExpandingWindowSplitter``.
        - An object to be used as a cross-validation generator (must have
          ``split`` and ``get_n_splits`` methods).

        For time series data, typical splitters are:

        - ``ExpandingWindowSplitter``: Train size increases with each fold
        - ``SlidingWindowSplitter``: Fixed train size, sliding forward

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

        - >0 : the total number of fits is displayed.
        - >1 : the score and time for each fit and parameter candidate is displayed.
        - >2 : the fold indices and scores for each candidate are displayed.
        - >10 : the parameter candidate indices are displayed.

    pre_dispatch : int or str, default='2*n_jobs'
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

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.

        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table::

            +-----------+------------+-------------------+---+-----------------+
            |param_alpha|param_gamma |param_degree       |...|rank_test_score  |
            +===========+============+===================+===+=================+
            |  1.0      |     --     |         --        |...|        2        |
            +-----------+------------+-------------------+---+-----------------+
            |  10.0     |     --     |         --        |...|        1        |
            +-----------+------------+-------------------+---+-----------------+
            |  100.0    |     --     |         --        |...|        3        |
            +-----------+------------+-------------------+---+-----------------+
            |   --      |    0.1     |         2         |...|        5        |
            +-----------+------------+-------------------+---+-----------------+
            |   --      |    0.2     |         3         |...|        4        |
            +-----------+------------+-------------------+---+-----------------+

        will be represented by a cv_results_ dict of::

            {
            'param_alpha': masked_array(data=[1.0, 10.0, 100.0, --, --],
                                        mask=[False False False  True  True]...),
            'param_gamma': masked_array(data=[--, --, --, 0.1, 0.2],
                                       mask=[ True  True  True False False]...),
            'param_degree': masked_array(data=[--, --, --, 2, 3],
                                         mask=[ True  True  True False False]...),
            'split0_test_score'  : [0.80, 0.70, 0.60, 0.75, 0.85],
            'split1_test_score'  : [0.82, 0.75, 0.65, 0.72, 0.80],
            'mean_test_score'    : [0.81, 0.725, 0.625, 0.735, 0.825],
            'std_test_score'     : [0.01, 0.025, 0.025, 0.015, 0.025],
            'rank_test_score'    : [2, 3, 5, 4, 1],
            'split0_train_score' : [0.85, 0.80, 0.75, 0.82, 0.90],
            'split1_train_score' : [0.87, 0.82, 0.77, 0.84, 0.92],
            'mean_train_score'   : [0.86, 0.81, 0.76, 0.83, 0.91],
            'std_train_score'    : [0.01, 0.01, 0.01, 0.01, 0.01],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49, 0.55],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01, 0.02],
            'mean_score_time'    : [0.01, 0.06, 0.04, 0.04, 0.05],
            'std_score_time'     : [0.00, 0.00, 0.00, 0.01, 0.00],
            'params'             : [{'alpha': 1.0}, {'alpha': 10.0}, ...],
            }

        NOTE: The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. (``'split0_test_mae'``, ``'mean_train_rmse'`` etc.)

    best_forecaster_ : BaseForecaster
        Forecaster that was chosen by the search, i.e. forecaster which gave
        highest score (or smallest loss if specified) on the left out data.
        Not available if ``refit=False``.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_forecaster_.

        Follows sklearn's sign convention: for ``lower_is_better`` scorers
        (e.g. MAE, RMSE) the value is **negated** so that higher always
        means better.  The raw metric value is ``-best_score_``.

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

    scorer_ : BaseScorer or dict
        Scorer function(s) used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best forecaster on the whole dataset.

        This is present only if ``refit`` is not False.

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    n_features_in_ : int
        Number of features seen during ``fit``. Only defined if
        ``best_forecaster_`` is defined (see the documentation for the ``refit``
        parameter for more details) and that ``best_forecaster_`` exposes
        ``n_features_in_`` when fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during ``fit``. Only defined if
        ``best_forecaster_`` is defined (see the documentation for the ``refit``
        parameter for more details) and that ``best_forecaster_`` exposes
        ``feature_names_in_`` when fit.

    See Also
    --------
    - [`RandomizedSearchCV`][yohou.model_selection.search.RandomizedSearchCV] : Randomized search over parameter distributions.
    - [`ExpandingWindowSplitter`][yohou.model_selection.split.ExpandingWindowSplitter] : Cross-validation with expanding training windows.
    - [`SlidingWindowSplitter`][yohou.model_selection.split.SlidingWindowSplitter] : Cross-validation with sliding fixed-size windows.
    - [`MeanAbsoluteError`][yohou.metrics.point.MeanAbsoluteError] : Mean absolute error scorer.
    - [`RootMeanSquaredError`][yohou.metrics.point.RootMeanSquaredError] : Root mean squared error scorer.

    Notes
    -----
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit scorer is passed in which case it is used instead.

    If n_jobs was set to a value higher than one, the data is copied for each
    point in the grid (and not n_jobs times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set ``pre_dispatch``. Then, the memory is copied only
    ``pre_dispatch`` many times. A reasonable value for ``pre_dispatch`` is
    ``2 * n_jobs``.

    Examples
    --------
    >>> from yohou.point import PointReductionForecaster
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
    >>> # Single-metric search
    >>> search = GridSearchCV(
    ...     forecaster=PointReductionForecaster(),
    ...     param_grid=param_grid,
    ...     scoring=MeanAbsoluteError(),
    ...     cv=3,
    ... )
    >>> search.fit(y, forecasting_horizon=5)  # doctest: +SKIP
    >>> search.best_params_  # doctest: +SKIP
    >>> y_pred = search.predict(forecasting_horizon=5)  # doctest: +SKIP
    >>>
    >>> # Multi-metric search
    >>> from yohou.metrics import RootMeanSquaredError
    >>> scoring = {"mae": MeanAbsoluteError(), "rmse": RootMeanSquaredError()}
    >>> search = GridSearchCV(
    ...     forecaster=PointReductionForecaster(),
    ...     param_grid=param_grid,
    ...     scoring=scoring,
    ...     refit="mae",  # Use 'mae' to select best parameters
    ...     cv=3,
    ... )
    >>> search.fit(y, forecasting_horizon=5)  # doctest: +SKIP
    >>> search.best_params_  # doctest: +SKIP
    >>> # cv_results_ contains both 'mean_test_mae' and 'mean_test_rmse'
    >>> search.cv_results_["mean_test_mae"]  # doctest: +SKIP
    >>> search.cv_results_["mean_test_rmse"]  # doctest: +SKIP

    """

    _parameter_constraints: dict = {
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
    """Randomized search on hyperparameters.

    Important members are fit, predict, predict_interval, observe, and score.

    RandomizedSearchCV implements a "fit" method that samples ``n_iter``
    parameter settings from the specified distributions using time series
    cross-validation. In contrast to GridSearchCV, not all parameter values
    are tried out, but rather a fixed number of parameter settings is sampled
    from the specified distributions. The number of parameter settings that
    are tried is given by ``n_iter``.

    It also implements "predict", "predict_interval", "observe", "rewind",
    "observe_predict", and "observe_predict_interval" if the underlying
    forecaster supports these methods and refit=True.

    Parameters
    ----------
    forecaster : BaseForecaster
        A forecaster object implementing the yohou forecaster interface
        with fit and predict methods.

    param_distributions : dict or list of dict
        Dictionary with parameter names (`str`) as keys and distributions
        or lists of parameter settings to try as values. If a list,
        it is sampled uniformly. If a list of dicts is given, first a dict
        is sampled uniformly, and then a parameter is sampled using that dict
        as above.

        Distributions are assumed to implement the ``rvs`` method for sampling
        (such as those from scipy.stats.distributions).

        Examples::

            from scipy.stats import uniform, randint

            param_distributions = {
                "estimator__alpha": uniform(0.01, 10.0),
                "estimator__l1_ratio": uniform(0.0, 1.0),
                "feature_transformer__lags": [[1], [1, 2], [1, 2, 3]],
            }

            # Alternative: List of dicts (samples dict first, then parameters)
            param_distributions = [
                {"estimator__alpha": uniform(0.01, 1.0), "estimator__fit_intercept": [True]},
                {"estimator__alpha": uniform(1.0, 10.0), "estimator__fit_intercept": [False]},
            ]

    n_iter : int, default=10
        Number of parameter settings that are sampled. ``n_iter`` trades
        off runtime vs quality of the solution.

        Higher ``n_iter`` will provide better coverage of the parameter space
        but increase computation time linearly.

    scoring : BaseScorer or dict of {str: BaseScorer}
        Strategy to evaluate the performance of the cross-validated model
        on the test set.  Required; although the constructor default is
        ``None``, passing ``None`` raises ``ValueError`` at ``fit`` time.

        If a single BaseScorer instance, the same scorer is used for all
        folds and stored in cv_results_ with key 'score'.

        If a dict, keys are scorer names and values are BaseScorer instances.
        This enables multi-metric evaluation. The ``refit`` parameter must
        be set to a scorer name or False to determine which scorer to use
        for selecting the best parameters.

        Unlike sklearn, string scorer names are not supported. You must use
        yohou.metrics BaseScorer instances (e.g., MeanAbsoluteError(),
        RootMeanSquaredError()).

        Examples::

            scoring = MeanAbsoluteError()
            scoring = {"mae": MeanAbsoluteError(), "rmse": RootMeanSquaredError()}

        Note: For multi-metric evaluation with dict, cv_results_ will contain
        keys like 'mean_test_mae', 'rank_test_mae', 'mean_test_rmse', etc.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a ``joblib.parallel_backend`` context.
        ``-1`` means using all processors.

    refit : bool, str, or callable, default=True
        Refit a forecaster using the best found parameters on the whole dataset.

        For multiple metric evaluation, this needs to be a ``str`` denoting the
        scorer that would be used to find the best parameters for refitting
        the forecaster at the end.

        Where there are considerations other than maximum score in
        choosing a best forecaster, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given ``cv_results_``. In that
        case, the ``best_forecaster_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.

        The refitted forecaster is made available at the ``best_forecaster_``
        attribute and permits using ``predict`` directly on this
        ``RandomizedSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

        Examples::

            refit = True  # Use single scorer or first scorer for multi-metric
            refit = "mae"  # For multi-metric: use 'mae' scorer to select best
            refit = False  # Don't refit, only evaluate
            refit = lambda cv_results: cv_results["rank_test_mae"].argmin()

    cv : int, BaseSplitter, or None, default=None
        Determines the cross-validation splitting strategy.

        Possible inputs for cv are:

        - None, to use the default 5-fold expanding window splitter
        - int, to specify the number of folds in an
          ``ExpandingWindowSplitter``.
        - An object to be used as a cross-validation generator (must have
          ``split`` and ``get_n_splits`` methods).

        For time series data, typical splitters are:

        - ``ExpandingWindowSplitter``: Train size increases with each fold
        - ``SlidingWindowSplitter``: Fixed train size, sliding forward

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

        - >0 : the total number of fits is displayed.
        - >1 : the score and time for each fit and parameter candidate is displayed.
        - >2 : the fold indices and scores for each candidate are displayed.
        - >10 : the parameter candidate indices are displayed.

    pre_dispatch : int or str, default='2*n_jobs'
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

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in forecaster fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.

        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table::

            +-----------+------------+-------------------+---+-----------------+
            |param_alpha|param_gamma |param_degree       |...|rank_test_score  |
            +===========+============+===================+===+=================+
            |  1.0      |     --     |         --        |...|        2        |
            +-----------+------------+-------------------+---+-----------------+
            |  10.0     |     --     |         --        |...|        1        |
            +-----------+------------+-------------------+---+-----------------+
            |  100.0    |     --     |         --        |...|        3        |
            +-----------+------------+-------------------+---+-----------------+
            |   --      |    0.1     |         2         |...|        5        |
            +-----------+------------+-------------------+---+-----------------+
            |   --      |    0.2     |         3         |...|        4        |
            +-----------+------------+-------------------+---+-----------------+

        will be represented by a cv_results_ dict of::

            {
            'param_alpha': masked_array(data=[1.0, 10.0, 100.0, --, --],
                                        mask=[False False False  True  True]...),
            'param_gamma': masked_array(data=[--, --, --, 0.1, 0.2],
                                       mask=[ True  True  True False False]...),
            'param_degree': masked_array(data=[--, --, --, 2, 3],
                                         mask=[ True  True  True False False]...),
            'split0_test_score'  : [0.80, 0.70, 0.60, 0.75, 0.85],
            'split1_test_score'  : [0.82, 0.75, 0.65, 0.72, 0.80],
            'mean_test_score'    : [0.81, 0.725, 0.625, 0.735, 0.825],
            'std_test_score'     : [0.01, 0.025, 0.025, 0.015, 0.025],
            'rank_test_score'    : [2, 3, 5, 4, 1],
            'split0_train_score' : [0.85, 0.80, 0.75, 0.82, 0.90],
            'split1_train_score' : [0.87, 0.82, 0.77, 0.84, 0.92],
            'mean_train_score'   : [0.86, 0.81, 0.76, 0.83, 0.91],
            'std_train_score'    : [0.01, 0.01, 0.01, 0.01, 0.01],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49, 0.55],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01, 0.02],
            'mean_score_time'    : [0.01, 0.06, 0.04, 0.04, 0.05],
            'std_score_time'     : [0.00, 0.00, 0.00, 0.01, 0.00],
            'params'             : [{'alpha': 1.0}, {'alpha': 10.0}, ...],
            }

        NOTE: The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. (``'split0_test_mae'``, ``'mean_train_rmse'`` etc.)

    best_forecaster_ : BaseForecaster
        Forecaster that was chosen by the search, i.e. forecaster which gave
        highest score (or smallest loss if specified) on the left out data.
        Not available if ``refit=False``.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_forecaster_.

        Follows sklearn's sign convention: for ``lower_is_better`` scorers
        (e.g. MAE, RMSE) the value is **negated** so that higher always
        means better.  The raw metric value is ``-best_score_``.

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

    scorer_ : BaseScorer or dict
        Scorer function(s) used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best forecaster on the whole dataset.

        This is present only if ``refit`` is not False.

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    n_features_in_ : int
        Number of features seen during ``fit``. Only defined if
        ``best_forecaster_`` is defined (see the documentation for the ``refit``
        parameter for more details) and that ``best_forecaster_`` exposes
        ``n_features_in_`` when fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during ``fit``. Only defined if
        ``best_forecaster_`` is defined (see the documentation for the ``refit``
        parameter for more details) and that ``best_forecaster_`` exposes
        ``feature_names_in_`` when fit.

    See Also
    --------
    - [`GridSearchCV`][yohou.model_selection.search.GridSearchCV] : Exhaustive search over specified parameter values.
    - [`ExpandingWindowSplitter`][yohou.model_selection.split.ExpandingWindowSplitter] : Cross-validation with expanding training windows.
    - [`SlidingWindowSplitter`][yohou.model_selection.split.SlidingWindowSplitter] : Cross-validation with sliding fixed-size windows.
    - [`MeanAbsoluteError`][yohou.metrics.point.MeanAbsoluteError] : Mean absolute error scorer.
    - [`RootMeanSquaredError`][yohou.metrics.point.RootMeanSquaredError] : Root mean squared error scorer.

    Notes
    -----
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit scorer is passed in which case it is used instead.

    If n_jobs was set to a value higher than one, the data is copied for each
    parameter setting (and not n_jobs times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set ``pre_dispatch``. Then, the memory is copied only
    ``pre_dispatch`` many times. A reasonable value for ``pre_dispatch`` is
    ``2 * n_jobs``.

    RandomizedSearchCV is particularly useful when the parameter space is
    large or when evaluating each parameter setting is expensive. By sampling
    a fixed number of settings, you can control the computational budget while
    still exploring the parameter space effectively. For high-dimensional
    parameter spaces, random search can be more efficient than grid search at
    finding good parameter settings.

    Examples
    --------
    >>> from yohou.point import PointReductionForecaster
    >>> from yohou.model_selection import RandomizedSearchCV
    >>> from yohou.metrics import MeanAbsoluteError
    >>> from scipy.stats import uniform, randint
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
    >>> # Single-metric search
    >>> search = RandomizedSearchCV(
    ...     forecaster=PointReductionForecaster(),
    ...     param_distributions=param_distributions,
    ...     n_iter=20,
    ...     scoring=MeanAbsoluteError(),
    ...     cv=3,
    ...     random_state=42,
    ... )
    >>> search.fit(y, forecasting_horizon=5)  # doctest: +SKIP
    >>> search.best_params_  # doctest: +SKIP
    >>> y_pred = search.predict(forecasting_horizon=5)  # doctest: +SKIP
    >>>
    >>> # Multi-metric search with custom refit strategy
    >>> from yohou.metrics import RootMeanSquaredError
    >>> scoring = {"mae": MeanAbsoluteError(), "rmse": RootMeanSquaredError()}
    >>> # Custom refit: Choose parameters with best MAE, but also consider RMSE
    >>> def refit_strategy(cv_results):
    ...     # Find candidates where MAE rank is in top 5
    ...     mae_ranks = cv_results["rank_test_mae"]
    ...     top_mae_mask = mae_ranks <= 5
    ...     # Among those, pick the one with best RMSE
    ...     rmse_scores = cv_results["mean_test_rmse"]
    ...     rmse_scores_masked = np.ma.array(rmse_scores, mask=~top_mae_mask)
    ...     return np.ma.argmax(rmse_scores_masked)  # Higher RMSE score is better
    >>> search = RandomizedSearchCV(
    ...     forecaster=PointReductionForecaster(),
    ...     param_distributions=param_distributions,
    ...     n_iter=20,
    ...     scoring=scoring,
    ...     refit=refit_strategy,
    ...     cv=3,
    ...     random_state=42,
    ... )
    >>> search.fit(y, forecasting_horizon=5)  # doctest: +SKIP
    >>> search.best_params_  # doctest: +SKIP
    >>> # cv_results_ contains both metrics
    >>> search.cv_results_["mean_test_mae"]  # doctest: +SKIP
    >>> search.cv_results_["mean_test_rmse"]  # doctest: +SKIP

    """

    _parameter_constraints: dict = {
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
        evaluate_candidates(ParameterSampler(self.param_distributions, self.n_iter, random_state=self.random_state))
