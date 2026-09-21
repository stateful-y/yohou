"""Early-stopping adapters for searches that stop on each fold's test window."""

from __future__ import annotations

import abc
import importlib
import numbers
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

from yohou.base.reduction import _eval_target
from yohou.utils._modules import _loaded_module

__all__ = [
    "BaseEarlyStoppingAdapter",
    "CatBoostEarlyStoppingAdapter",
    "HistGradientBoostingEarlyStoppingAdapter",
    "LightGBMEarlyStoppingAdapter",
    "XGBoostEarlyStoppingAdapter",
]


class BaseEarlyStoppingAdapter(BaseEstimator, metaclass=abc.ABCMeta):
    """Base class for translating early stopping between a search and a boosting library.

    Subclass it to use ``validation="cv"`` with an estimator that no built-in
    adapter supports, and pass the instance as the search's
    ``early_stopping_adapter``. Every method receives the estimator that
    receives the evaluation set: the estimator itself, or the final step when
    the forecaster's estimator is a ``sklearn.pipeline.Pipeline``.

    See Also
    --------
    - [`LightGBMEarlyStoppingAdapter`][yohou.model_selection.LightGBMEarlyStoppingAdapter] : Adapter for LightGBM models.
    - [`XGBoostEarlyStoppingAdapter`][yohou.model_selection.XGBoostEarlyStoppingAdapter] : Adapter for XGBoost models.
    - [`CatBoostEarlyStoppingAdapter`][yohou.model_selection.CatBoostEarlyStoppingAdapter] : Adapter for CatBoost models.
    - [`HistGradientBoostingEarlyStoppingAdapter`][yohou.model_selection.HistGradientBoostingEarlyStoppingAdapter] : Adapter for scikit-learn histogram gradient boosting models.

    Notes
    -----
    The search knows nothing about any boosting library: everything
    library-specific lives in an adapter. The built-in adapters touch library
    internals where the libraries offer no public alternative, and those
    accesses are confined to the adapter module.

    """

    @abc.abstractmethod
    def supports(self, estimator: BaseEstimator) -> bool:
        """Return whether this adapter handles the estimator.

        Used when the search resolves an adapter automatically, and to check
        an adapter passed explicitly, which is used for every candidate. It
        must not import a library the caller has not imported.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted estimator that receives the evaluation set.

        Returns
        -------
        bool
            True when the adapter handles this estimator.

        """

    @abc.abstractmethod
    def validate(self, estimator: BaseEstimator, fit_params: dict[str, Any] | None = None) -> None:
        """Reject configurations the shared-round mode cannot honour.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted estimator that receives the evaluation set.
        fit_params : dict or None, default=None
            The metadata the caller routes to ``fit``, for configurations set
            at fit time rather than on the estimator.

        Raises
        ------
        ValueError
            If the configuration cannot be used with ``validation="cv"``.

        """

    @abc.abstractmethod
    def prepare_fold_fit(self, estimator: BaseEstimator) -> tuple[BaseEstimator, dict[str, Any]]:
        """Configure an estimator for a fold fit that trains every round up to its ceiling.

        The returned estimator, fitted with an evaluation set and the returned
        fit parameters, evaluates that set after every round and never stops
        early, so its stopping curve covers the whole round ceiling and
        `truncate` can cut it to any round. The metric and its direction stay
        those the original configuration would stop on.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted estimator that receives the evaluation set. Not
            mutated.

        Returns
        -------
        estimator : BaseEstimator
            An unfitted, configured clone.
        fit_params : dict
            Extra keyword arguments for the estimator's ``fit``.

        """

    @abc.abstractmethod
    def stopping_curve(self, fitted: BaseEstimator) -> tuple[np.ndarray, bool]:
        """Read the per-round evaluation metric early stopping used.

        Parameters
        ----------
        fitted : BaseEstimator
            An estimator prepared by `prepare_fold_fit` and fitted with an
            evaluation set.

        Returns
        -------
        curve : np.ndarray
            One value per trained round, on the evaluation set.
        higher_is_better : bool
            Whether larger values are better.

        """

    @abc.abstractmethod
    def truncate(self, fitted: BaseEstimator, n_rounds: int) -> None:
        """Make plain ``predict`` (and ``predict_proba``) use only the first rounds.

        Parameters
        ----------
        fitted : BaseEstimator
            A fitted estimator. Mutated in place.
        n_rounds : int
            Number of rounds to keep, at least 1 and at most the number of
            trained rounds.

        Raises
        ------
        ValueError
            If ``n_rounds`` is not between 1 and the number of trained rounds.

        """

    @abc.abstractmethod
    def prepare_refit(self, estimator: BaseEstimator, n_rounds: int) -> BaseEstimator:
        """Configure an estimator to train a fixed number of rounds with early stopping off.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted estimator that receives the evaluation set. Not
            mutated.
        n_rounds : int
            The round ceiling.

        Returns
        -------
        BaseEstimator
            An unfitted, configured clone that fits without an evaluation set.

        """


def _check_rounds(fitted_rounds: int, n_rounds: int, library: str) -> None:
    """Reject a truncation outside the trained rounds.

    Parameters
    ----------
    fitted_rounds : int
        Number of rounds the model holds.
    n_rounds : int
        Requested round count.
    library : str
        Library name for the error message.

    Raises
    ------
    ValueError
        If ``n_rounds`` is not between 1 and ``fitted_rounds``.

    """
    if not isinstance(n_rounds, numbers.Integral) or not 1 <= n_rounds <= fitted_rounds:
        raise ValueError(f"Cannot truncate the {library} model to {n_rounds} rounds: it holds {fitted_rounds} rounds.")


# LightGBM treats every name in each group as the same parameter
# (lightgbm.basic._ConfigAliases, LightGBM 4.7.0).
_LGBM_EARLY_STOPPING_ALIASES = ("early_stopping_round", "early_stopping_rounds", "early_stopping", "n_iter_no_change")
_LGBM_METRIC_ALIASES = ("metric", "metrics", "metric_types")
_LGBM_ROUND_ALIASES = (
    "n_estimators",
    "num_iterations",
    "num_iteration",
    "n_iter",
    "num_tree",
    "num_trees",
    "num_round",
    "num_rounds",
    "nrounds",
    "num_boost_round",
    "max_iter",
)


class _StoppingMetricRecorder:
    """LightGBM callback that records which metric a fold fit is judged on.

    It never stops training. On each fit's first iteration it stores, on the
    booster being trained, the first metric of the first validation set and
    LightGBM's direction for it, which `LightGBMEarlyStoppingAdapter.stopping_curve` reads
    back. Everything it records lives on the booster, so one instance can
    serve several fits at once (the per-step fits of one forecaster,
    sequential or threaded).
    """

    def __init__(self):
        # Instance attributes, not class attributes: lightgbm.train assigns a
        # default order with ``cb.__dict__.setdefault``, which ignores class
        # attributes.
        self.order = 30
        self.before_iteration = False

    def __call__(self, env: Any) -> None:
        """Store the evaluated metric and its direction on the booster at the first iteration."""
        if env.iteration != env.begin_iteration:
            return
        results = [r for r in env.evaluation_result_list if r[0] != "training"]
        if not results:
            raise ValueError(
                "The LightGBM early-stopping adapter needs an evaluation set other than the training data."
            )
        dataset_name, metric_name, _, higher_is_better = results[0][:4]
        env.model._yohou_stopping_metric = (dataset_name, metric_name)
        env.model._yohou_stopping_higher_better = bool(higher_is_better)


def _metric_names(value: Any) -> list[str]:
    """Return the metric names one LightGBM metric setting contributes.

    Parameters
    ----------
    value : Any
        A ``metric`` alias or an ``eval_metric``: None, a name, a
        comma-separated string of names, a custom callable, or a sequence
        of those.

    Returns
    -------
    list of str
        One name per configured metric. A callable is named by its
        ``__name__``, since LightGBM evaluates it alongside the rest.

    """
    if value is None:
        return []
    if isinstance(value, str):
        return [name.strip() for name in value.split(",") if name.strip()]
    if callable(value):
        return [getattr(value, "__name__", "custom")]
    if isinstance(value, list | tuple):
        return [name for item in value for name in _metric_names(item)]
    return [text] if (text := str(value).strip()) else []


def _lgbm_metrics(params: dict[str, Any], fit_params: dict[str, Any] | None = None) -> list[str]:
    """Return the metric names a LightGBM fit is configured to evaluate.

    Parameters
    ----------
    params : dict
        The estimator's ``get_params()``.
    fit_params : dict or None, default=None
        The metadata the caller routes to ``fit``.

    Returns
    -------
    list of str
        Every distinct name, in first-seen order, under any ``metric`` alias
        and under a fit-time ``eval_metric``.

    Notes
    -----
    Only explicitly configured metrics are counted. LightGBM also evaluates
    the objective's own default metric unless ``metric="None"``, which this
    check does not attempt to resolve.

    """
    sources = [params.get(alias) for alias in _LGBM_METRIC_ALIASES]
    sources.append((fit_params or {}).get("eval_metric"))
    names = [name for value in sources for name in _metric_names(value)]
    return list(dict.fromkeys(names))


class LightGBMEarlyStoppingAdapter(BaseEarlyStoppingAdapter):
    """Early-stopping adapter for LightGBM's scikit-learn estimators.

    Handles every ``lightgbm.LGBMModel`` (``LGBMRegressor``,
    ``LGBMClassifier``, ``LGBMRanker``). Fold fits remove the estimator's
    ``early_stopping_round`` and train every one of ``n_estimators`` rounds,
    recording the metric on the evaluation set after each. The stopping curve
    is the first metric of the evaluation set, so more than one metric, set on
    the estimator or as a fit-time ``eval_metric`` list, is accepted only with
    ``first_metric_only=True``: LightGBM's own early stopping otherwise
    compares every metric.

    ``boosting_type="dart"`` is rejected: dart rescales earlier trees as it
    adds new ones, so a model cut to k rounds is not the model trained for k.

    See Also
    --------
    - [`BaseEarlyStoppingAdapter`][yohou.model_selection.BaseEarlyStoppingAdapter] : The adapter contract.

    """

    def supports(self, estimator: BaseEstimator) -> bool:
        """Return whether the estimator is a LightGBM model.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted estimator that receives the evaluation set.

        Returns
        -------
        bool
            True for any ``lightgbm.LGBMModel``.

        """
        lightgbm = _loaded_module("lightgbm")
        return lightgbm is not None and isinstance(estimator, lightgbm.LGBMModel)

    def validate(self, estimator: BaseEstimator, fit_params: dict[str, Any] | None = None) -> None:
        """Reject dart boosting and several metrics without ``first_metric_only``.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted LightGBM estimator.
        fit_params : dict or None, default=None
            The metadata the caller routes to ``fit``. Every name under
            ``eval_metric`` counts towards the metric check.

        Raises
        ------
        ValueError
            If ``boosting_type`` (or its ``boosting``/``boost`` aliases) is
            ``"dart"``, or if more than one metric is configured and
            ``first_metric_only`` is not True.

        """
        params = estimator.get_params()
        if "dart" in (params.get("boosting_type"), params.get("boosting"), params.get("boost")):
            raise ValueError(
                "validation='cv' cannot use LightGBM with boosting_type='dart': dart rescales earlier "
                "trees as it adds new ones, so a model cut to fewer rounds is not the model trained for "
                "that many. Use boosting_type='gbdt', or validation=None."
            )
        metrics = _lgbm_metrics(params, fit_params)
        if len(metrics) > 1 and not params.get("first_metric_only"):
            raise ValueError(
                f"validation='cv' cannot use LightGBM with the metrics {metrics!r} and "
                f"first_metric_only={params.get('first_metric_only')!r}: the stopping curve is the first "
                f"metric only, while LightGBM's early stopping compares every metric unless "
                f"first_metric_only=True. Set first_metric_only=True, keep a single metric, or use "
                f"validation=None."
            )

    def prepare_fold_fit(self, estimator: BaseEstimator) -> tuple[BaseEstimator, dict[str, Any]]:
        """Remove LightGBM's early stopping and record the evaluated metric instead.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted LightGBM estimator. Not mutated.

        Returns
        -------
        estimator : BaseEstimator
            A clone with its early-stopping parameters removed.
        fit_params : dict
            ``{"callbacks": [callback]}``.

        """
        params = estimator.get_params()
        prepared = clone(estimator).set_params(**{a: None for a in _LGBM_EARLY_STOPPING_ALIASES if a in params})
        return prepared, {"callbacks": [_StoppingMetricRecorder()]}

    def stopping_curve(self, fitted: BaseEstimator) -> tuple[np.ndarray, bool]:
        """Read the metric the stopping callback compared, per round.

        Parameters
        ----------
        fitted : BaseEstimator
            A LightGBM estimator prepared by `prepare_fold_fit` and fitted.

        Returns
        -------
        curve : np.ndarray
            The metric on the evaluation set, one value per trained round.
        higher_is_better : bool
            LightGBM's direction for that metric.

        Raises
        ------
        ValueError
            If the estimator was not fitted through `prepare_fold_fit`.

        """
        model: Any = fitted
        booster = model.booster_
        metric = getattr(booster, "_yohou_stopping_metric", None)
        if metric is None:
            raise ValueError(
                "This LightGBM model was not fitted with the callbacks from prepare_fold_fit, "
                "so its stopping curve is unknown."
            )
        dataset_name, metric_name = metric
        curve = np.asarray(model.evals_result_[dataset_name][metric_name], dtype=float)
        return curve, bool(booster._yohou_stopping_higher_better)

    def truncate(self, fitted: BaseEstimator, n_rounds: int) -> None:
        """Make ``predict`` use the first ``n_rounds`` trees.

        Parameters
        ----------
        fitted : BaseEstimator
            A fitted LightGBM estimator. Mutated in place.
        n_rounds : int
            Number of rounds to keep.

        Raises
        ------
        ValueError
            If ``n_rounds`` is not between 1 and the number of trained rounds.

        """
        model: Any = fitted
        booster = model.booster_
        _check_rounds(booster.current_iteration(), n_rounds, "LightGBM")
        # LGBMModel.predict falls back to the booster's best_iteration when no
        # num_iteration is passed.
        booster.best_iteration = int(n_rounds)

    def prepare_refit(self, estimator: BaseEstimator, n_rounds: int) -> BaseEstimator:
        """Set the round ceiling and remove early stopping.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted LightGBM estimator. Not mutated.
        n_rounds : int
            The round ceiling.

        Returns
        -------
        BaseEstimator
            A configured clone.

        """
        params = estimator.get_params()
        updates: dict[str, Any] = {a: None for a in _LGBM_EARLY_STOPPING_ALIASES if a in params}
        updates.update({a: int(n_rounds) for a in _LGBM_ROUND_ALIASES if params.get(a) is not None})
        updates["n_estimators"] = int(n_rounds)
        return clone(estimator).set_params(**updates)


# XGBoost's EarlyStopping maximizes these metrics when `maximize` is None
# (xgboost.callback.EarlyStopping._update_rounds, XGBoost 3.4.1).
_XGB_MAXIMIZE_PREFIXES = ("auc", "aucpr", "pre", "pre@", "map", "ndcg", "auc@", "aucpr@", "map@", "ndcg@")


class XGBoostEarlyStoppingAdapter(BaseEarlyStoppingAdapter):
    """Early-stopping adapter for XGBoost's scikit-learn estimators.

    Handles every ``xgboost.XGBModel``. Fold fits clear
    ``early_stopping_rounds`` and replace any ``EarlyStopping`` callback with
    one that cannot trigger, so every one of ``n_estimators`` rounds is
    trained. The stopping curve is the metric XGBoost's ``EarlyStopping``
    compares: the last metric of the last evaluation set, unless an
    ``EarlyStopping`` callback names others, with that callback's
    ``maximize`` setting.

    ``booster="dart"`` is rejected: a cut model is not the model trained for
    that many rounds.

    See Also
    --------
    - [`BaseEarlyStoppingAdapter`][yohou.model_selection.BaseEarlyStoppingAdapter] : The adapter contract.

    """

    @staticmethod
    def _early_stopping_callbacks(callbacks: list[Any] | None) -> list[Any]:
        """Return the XGBoost ``EarlyStopping`` callbacks among ``callbacks``."""
        xgboost = _loaded_module("xgboost")
        if xgboost is None:
            return []
        return [cb for cb in callbacks or [] if isinstance(cb, xgboost.callback.EarlyStopping)]

    def supports(self, estimator: BaseEstimator) -> bool:
        """Return whether the estimator is an XGBoost model.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted estimator that receives the evaluation set.

        Returns
        -------
        bool
            True for any ``xgboost.XGBModel``.

        """
        xgboost = _loaded_module("xgboost")
        return xgboost is not None and isinstance(estimator, xgboost.XGBModel)

    def validate(self, estimator: BaseEstimator, fit_params: dict[str, Any] | None = None) -> None:
        """Reject dart boosting.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted XGBoost estimator.
        fit_params : dict or None, default=None
            The metadata the caller routes to ``fit``. Unused.

        Raises
        ------
        ValueError
            If ``booster="dart"``.

        """
        if estimator.get_params().get("booster") == "dart":
            raise ValueError(
                "validation='cv' cannot use XGBoost with booster='dart': dart rescales earlier trees as "
                "it adds new ones, so a model cut to fewer rounds is not the model trained for that many. "
                "Use booster='gbtree', or validation=None."
            )

    def prepare_fold_fit(self, estimator: BaseEstimator) -> tuple[BaseEstimator, dict[str, Any]]:
        """Disable early stopping while keeping the metric and direction it would use.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted XGBoost estimator. Not mutated.

        Returns
        -------
        estimator : BaseEstimator
            A clone without ``early_stopping_rounds``, whose ``EarlyStopping``
            callbacks are replaced by copies with patience above the round
            ceiling and ``save_best=False``.
        fit_params : dict
            An empty dict.

        """
        prepared = clone(estimator)
        updates: dict[str, Any] = {"early_stopping_rounds": None}
        callbacks = prepared.get_params().get("callbacks")
        stopping = self._early_stopping_callbacks(callbacks)
        if stopping:
            xgboost = _loaded_module("xgboost")
            never_stops = (prepared.get_params().get("n_estimators") or 100) + 1
            updates["callbacks"] = [
                xgboost.callback.EarlyStopping(
                    rounds=never_stops,
                    metric_name=cb.metric_name,
                    data_name=cb.data,
                    maximize=cb.maximize,
                    save_best=False,
                )
                if cb in stopping
                else cb
                for cb in callbacks
            ]
        return prepared.set_params(**updates), {}

    def stopping_curve(self, fitted: BaseEstimator) -> tuple[np.ndarray, bool]:
        """Read the metric XGBoost's early stopping compared, per round.

        Parameters
        ----------
        fitted : BaseEstimator
            A fitted XGBoost estimator.

        Returns
        -------
        curve : np.ndarray
            The metric on the evaluation set, one value per trained round.
        higher_is_better : bool
            The ``maximize`` setting of an ``EarlyStopping`` callback, or
            XGBoost's metric-name rule when none is set.

        """
        model: Any = fitted
        evals = model.evals_result()
        callbacks = self._early_stopping_callbacks(model.get_params().get("callbacks"))
        callback = callbacks[0] if callbacks else None
        data_name = getattr(callback, "data", None) or list(evals)[-1]
        metric_name = getattr(callback, "metric_name", None) or list(evals[data_name])[-1]
        curve = np.asarray(evals[data_name][metric_name], dtype=float)
        maximize = getattr(callback, "maximize", None)
        if maximize is None:
            maximize = metric_name != "mape" and any(metric_name.startswith(p) for p in _XGB_MAXIMIZE_PREFIXES)
        return curve, bool(maximize)

    def truncate(self, fitted: BaseEstimator, n_rounds: int) -> None:
        """Make ``predict`` use the first ``n_rounds`` rounds.

        Parameters
        ----------
        fitted : BaseEstimator
            A fitted XGBoost estimator. Mutated in place.
        n_rounds : int
            Number of rounds to keep.

        Raises
        ------
        ValueError
            If ``n_rounds`` is not between 1 and the number of trained rounds.

        """
        model: Any = fitted
        booster = model.get_booster()
        _check_rounds(booster.num_boosted_rounds(), n_rounds, "XGBoost")
        # XGBModel.predict uses iteration_range (0, best_iteration + 1) when
        # the booster carries a best_iteration attribute.
        booster.set_attr(best_iteration=str(int(n_rounds) - 1))

    def prepare_refit(self, estimator: BaseEstimator, n_rounds: int) -> BaseEstimator:
        """Set ``n_estimators`` and remove early stopping.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted XGBoost estimator. Not mutated.
        n_rounds : int
            The round ceiling.

        Returns
        -------
        BaseEstimator
            A configured clone.

        """
        prepared = clone(estimator)
        updates: dict[str, Any] = {"n_estimators": int(n_rounds), "early_stopping_rounds": None}
        callbacks = prepared.get_params().get("callbacks")
        if callbacks:
            stopping = self._early_stopping_callbacks(callbacks)
            updates["callbacks"] = [cb for cb in callbacks if cb not in stopping] or None
        return prepared.set_params(**updates)


_CATBOOST_ROUND_ALIASES = ("iterations", "n_estimators", "num_boost_round", "num_trees")
_CATBOOST_STOPPING_PARAMS = ("early_stopping_rounds", "od_type", "od_wait", "od_pval")


class CatBoostEarlyStoppingAdapter(BaseEarlyStoppingAdapter):
    """Early-stopping adapter for CatBoost regressors and classifiers.

    Fold fits remove ``early_stopping_rounds`` and the overfitting-detector
    parameters and set ``use_best_model=False`` (CatBoost otherwise shrinks
    the model to its best iteration whenever an evaluation set is given), so
    every one of ``iterations`` rounds is trained and kept. The stopping curve
    is ``eval_metric`` (or ``loss_function`` when no evaluation metric is set)
    on the evaluation set.

    CatBoost derives its default ``learning_rate`` from ``iterations``, so a
    refit with a different round count would train a different model; an
    explicit ``learning_rate`` is therefore required.

    See Also
    --------
    - [`BaseEarlyStoppingAdapter`][yohou.model_selection.BaseEarlyStoppingAdapter] : The adapter contract.

    """

    def supports(self, estimator: BaseEstimator) -> bool:
        """Return whether the estimator is a CatBoost regressor or classifier.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted estimator that receives the evaluation set.

        Returns
        -------
        bool
            True for ``CatBoostRegressor`` and ``CatBoostClassifier``.

        """
        catboost = _loaded_module("catboost")
        return catboost is not None and isinstance(estimator, catboost.CatBoostRegressor | catboost.CatBoostClassifier)

    def validate(self, estimator: BaseEstimator, fit_params: dict[str, Any] | None = None) -> None:
        """Require an explicit learning rate.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted CatBoost estimator.
        fit_params : dict or None, default=None
            The metadata the caller routes to ``fit``. Unused.

        Raises
        ------
        ValueError
            If ``learning_rate`` is not set.

        """
        if estimator.get_params().get("learning_rate") is None:
            raise ValueError(
                "validation='cv' requires an explicit learning_rate on CatBoost estimators: CatBoost "
                "derives its default learning rate from iterations, so the refit, which trains the "
                "chosen number of rounds, would use a different learning rate than the folds did and "
                "train a different model. Set learning_rate."
            )

    def prepare_fold_fit(self, estimator: BaseEstimator) -> tuple[BaseEstimator, dict[str, Any]]:
        """Remove the overfitting detector and keep every trained tree.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted CatBoost estimator. Not mutated.

        Returns
        -------
        estimator : BaseEstimator
            A copy without early-stopping parameters and with
            ``use_best_model=False``.
        fit_params : dict
            An empty dict.

        """
        return self._without_stopping(estimator, {"use_best_model": False}), {}

    @staticmethod
    def _without_stopping(estimator: BaseEstimator, updates: dict[str, Any]) -> BaseEstimator:
        """Rebuild a CatBoost estimator without its overfitting-detector parameters.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted CatBoost estimator. Not mutated.
        updates : dict
            Parameters to set on the rebuilt estimator.

        Returns
        -------
        BaseEstimator
            A new estimator of the same type, without the detector's parameters
            and with ``updates`` applied.

        Notes
        -----
        CatBoost's ``get_params`` lists only the parameters that were set, so
        rebuilding without the detector's keys removes them rather than passing
        None through to the library.

        """
        kept = {k: v for k, v in clone(estimator).get_params().items() if k not in _CATBOOST_STOPPING_PARAMS}
        return type(estimator)(**{**kept, **updates})

    def stopping_curve(self, fitted: BaseEstimator) -> tuple[np.ndarray, bool]:
        """Read the evaluation metric on the evaluation set, per round.

        Parameters
        ----------
        fitted : BaseEstimator
            A fitted CatBoost estimator.

        Returns
        -------
        curve : np.ndarray
            The metric on the evaluation set, one value per trained round.
        higher_is_better : bool
            Whether CatBoost maximizes that metric.

        Raises
        ------
        ValueError
            If the model holds no validation results for its metric.

        """
        model: Any = fitted
        params = model.get_all_params()
        metric = params.get("eval_metric") or params.get("loss_function")
        evals = model.get_evals_result()
        validation_keys = [k for k in evals if k.startswith("validation")]
        if not validation_keys:
            raise ValueError("This CatBoost model was fitted without an evaluation set, so it has no stopping curve.")
        results = evals[validation_keys[-1]]
        name = (
            metric
            if metric in results
            else next((k for k in results if k.split(":")[0] == str(metric).split(":")[0]), None)
        )
        if name is None:
            raise ValueError(f"CatBoost recorded no validation values for {metric!r}; got {sorted(results)}.")
        # CatBoost exposes metric direction only through this private module.
        catboost_core = importlib.import_module("catboost._catboost")
        return np.asarray(results[name], dtype=float), bool(catboost_core.is_maximizable_metric(name))

    def truncate(self, fitted: BaseEstimator, n_rounds: int) -> None:
        """Remove every tree after the first ``n_rounds``.

        Parameters
        ----------
        fitted : BaseEstimator
            A fitted CatBoost estimator. Mutated in place.
        n_rounds : int
            Number of rounds to keep.

        Raises
        ------
        ValueError
            If ``n_rounds`` is not between 1 and the number of trained rounds.

        """
        model: Any = fitted
        _check_rounds(model.tree_count_, n_rounds, "CatBoost")
        if n_rounds < model.tree_count_:
            model.shrink(ntree_end=int(n_rounds))

    def prepare_refit(self, estimator: BaseEstimator, n_rounds: int) -> BaseEstimator:
        """Set the round count and remove the overfitting detector.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted CatBoost estimator. Not mutated.
        n_rounds : int
            The round ceiling.

        Returns
        -------
        BaseEstimator
            A configured clone.

        """
        params = estimator.get_params()
        round_param = next((a for a in _CATBOOST_ROUND_ALIASES if params.get(a) is not None), "iterations")
        return self._without_stopping(estimator, {round_param: int(n_rounds), "use_best_model": False})


class HistGradientBoostingEarlyStoppingAdapter(BaseEarlyStoppingAdapter):
    """Early-stopping adapter for scikit-learn's histogram gradient boosting.

    ``HistGradientBoostingRegressor`` and ``HistGradientBoostingClassifier``
    only record a stopping curve when early stopping is switched on, and they
    reject an evaluation set outright when it is switched off. Fold fits
    therefore set ``early_stopping=True`` with ``n_iter_no_change`` above
    ``max_iter``, whatever the candidate carried: the patience can never expire,
    so every round up to the ceiling is trained and ``validation_score_`` holds
    the whole curve.

    scikit-learn records a *score*, so the curve is higher-is-better whatever
    the configured ``scoring``. Its first entry is the score before any
    iteration, which the curve drops so that entry i is the model after round
    i+1, matching every other adapter.

    Truncation slices the fitted predictor list, the only route the library
    offers: it exposes no truncated-prediction argument. ``n_iter_`` follows
    the cut because it is derived from that list.

    See Also
    --------
    - [`BaseEarlyStoppingAdapter`][yohou.model_selection.BaseEarlyStoppingAdapter] : The adapter contract.

    """

    def supports(self, estimator: BaseEstimator) -> bool:
        """Return whether the estimator is a histogram gradient boosting model.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted estimator that receives the evaluation set.

        Returns
        -------
        bool
            True for ``HistGradientBoostingRegressor`` and
            ``HistGradientBoostingClassifier``.

        """
        # scikit-learn is a hard dependency, so this import always resolves;
        # it is local to keep the module's import graph flat.
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        return isinstance(estimator, HistGradientBoostingRegressor | HistGradientBoostingClassifier)

    def validate(self, estimator: BaseEstimator, fit_params: dict[str, Any] | None = None) -> None:
        """Accept every configuration; nothing here blocks the shared-round mode.

        In particular an ``early_stopping`` of ``False`` or ``"auto"`` is not
        rejected: `prepare_fold_fit` switches it on, which is what the mode
        promises to do with whatever stopping settings the candidate carries.
        The holdout path, which has no adapter to correct the estimator, is the
        one that rejects those values.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted estimator that receives the evaluation set.
        fit_params : dict or None, default=None
            The metadata the caller routes to ``fit``. Unused.

        """

    def prepare_fold_fit(self, estimator: BaseEstimator) -> tuple[BaseEstimator, dict[str, Any]]:
        """Return a clone that trains every round and records the curve.

        Parameters
        ----------
        estimator : BaseEstimator
            The candidate's estimator.

        Returns
        -------
        BaseEstimator
            A clone with early stopping on and a patience above the ceiling.
        dict
            No extra fit parameters; the evaluation set is delivered by the
            forecaster in the ``X_val`` dialect.

        """
        prepared = clone(estimator)
        max_iter = int(prepared.get_params()["max_iter"])
        # Patience above the ceiling cannot expire, so switching early stopping
        # on buys the curve without ever shortening the fit.
        prepared.set_params(early_stopping=True, n_iter_no_change=max_iter + 1)
        return prepared, {}

    def stopping_curve(self, fitted: BaseEstimator) -> tuple[np.ndarray, bool]:
        """Return the validation curve and its direction.

        Parameters
        ----------
        fitted : BaseEstimator
            A fitted histogram gradient boosting estimator.

        Returns
        -------
        np.ndarray
            ``validation_score_`` without its pre-iteration baseline entry.
        bool
            Always True: scikit-learn stores a score, not a loss.

        Raises
        ------
        ValueError
            If the model recorded no validation curve, because its fit
            received no evaluation set and ``validation_fraction`` is None.

        """
        model: Any = fitted
        curve = np.asarray(getattr(model, "validation_score_", []), dtype=float)
        if curve.size <= 1:
            raise ValueError(
                "This scikit-learn model recorded no validation curve: its fit received no "
                "evaluation set (X_val, y_val) and validation_fraction is None."
            )
        # Entry 0 is the score before any iteration, which is not a round any
        # model here can express.
        return curve[1:], True

    def truncate(self, fitted: BaseEstimator, n_rounds: int) -> None:
        """Cut the fitted model to its first ``n_rounds`` rounds, in place.

        Parameters
        ----------
        fitted : BaseEstimator
            A fitted histogram gradient boosting estimator. Mutated in place.
        n_rounds : int
            Number of rounds to keep.

        Raises
        ------
        ValueError
            If ``n_rounds`` is not between 1 and the number of trained rounds.

        """
        model: Any = fitted
        _check_rounds(len(model._predictors), n_rounds, "scikit-learn")
        # The library offers no truncated-prediction argument, so the predictor
        # list is cut directly. ``n_iter_`` is derived from it and follows.
        model._predictors = model._predictors[: int(n_rounds)]

    def prepare_refit(self, estimator: BaseEstimator, n_rounds: int) -> BaseEstimator:
        """Return a clone that trains exactly ``n_rounds`` rounds with no validation set.

        Parameters
        ----------
        estimator : BaseEstimator
            The candidate's estimator.
        n_rounds : int
            The shared round count chosen across folds.

        Returns
        -------
        BaseEstimator
            A clone with ``max_iter=n_rounds`` and early stopping off.

        """
        prepared = clone(estimator)
        prepared.set_params(max_iter=int(n_rounds), early_stopping=False)
        return prepared


_BUILTIN_ADAPTERS: tuple[type[BaseEarlyStoppingAdapter], ...] = (
    LightGBMEarlyStoppingAdapter,
    XGBoostEarlyStoppingAdapter,
    CatBoostEarlyStoppingAdapter,
    HistGradientBoostingEarlyStoppingAdapter,
)


def _replace_eval_target(estimator: BaseEstimator, target: BaseEstimator) -> BaseEstimator:
    """Return ``estimator`` with its evaluation-set receiver replaced.

    Parameters
    ----------
    estimator : BaseEstimator
        A forecaster's ``estimator``. Not mutated.
    target : BaseEstimator
        The replacement for the estimator itself, or for a ``Pipeline``'s
        final step.

    Returns
    -------
    BaseEstimator
        ``target``, or a clone of the pipeline ending in ``target``.

    """
    if isinstance(estimator, Pipeline):
        pipeline = clone(estimator)
        pipeline.steps[-1] = (pipeline.steps[-1][0], target)
        return pipeline
    return target


def _resolve_early_stopping_adapter(
    estimator: BaseEstimator, adapter: BaseEarlyStoppingAdapter | None = None
) -> BaseEarlyStoppingAdapter:
    """Return the adapter to use for a forecaster's estimator.

    Parameters
    ----------
    estimator : BaseEstimator
        A forecaster's ``estimator`` (a ``Pipeline`` resolves on its final step).
    adapter : BaseEarlyStoppingAdapter or None, default=None
        An explicit adapter, returned unchanged when it supports the estimator.

    Returns
    -------
    BaseEarlyStoppingAdapter
        The explicit adapter, or the first built-in adapter supporting the
        estimator.

    Raises
    ------
    ValueError
        If no adapter is given and no built-in adapter supports the estimator,
        or if the given adapter's ``supports`` rejects it.

    """
    target = _eval_target(estimator)
    if adapter is not None:
        # An explicit adapter is used for every candidate, so a grid that swaps
        # the estimator can hand it one it was not written for. It would then
        # translate another library's parameters and fail far from the cause,
        # or read a meaningless curve, so its own `supports` is consulted here.
        if not adapter.supports(target):
            raise ValueError(
                f"early_stopping_adapter={adapter.__class__.__name__}() does not support "
                f"{target.__class__.__name__}, and an explicit adapter is used for every "
                f"candidate. Leave early_stopping_adapter=None to choose a built-in adapter "
                f"per candidate, or pass an adapter whose supports() accepts this estimator."
            )
        return adapter
    for adapter_cls in _BUILTIN_ADAPTERS:
        candidate = adapter_cls()
        if candidate.supports(target):
            return candidate
    raise ValueError(
        f"validation='cv' has no early-stopping adapter for {target.__class__.__name__}. Built-in adapters "
        f"cover LightGBM, XGBoost, CatBoost, and scikit-learn histogram gradient boosting "
        f"estimators; for another estimator, subclass "
        f"BaseEarlyStoppingAdapter and pass an instance as early_stopping_adapter."
    )
