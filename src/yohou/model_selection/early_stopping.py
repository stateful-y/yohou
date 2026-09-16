"""Early-stopping adapters for searches that stop on each fold's test window.

A search with ``validation="cv"`` fits every fold with that fold's test window
as the evaluation set, chooses one boosting round count per fitted estimator
from the fold-average stopping curve, scores every fold at that round, and
refits with it. The search itself knows nothing about any boosting library:
everything library-specific lives in an adapter.

An adapter answers six questions about the estimator that receives the
evaluation set (the estimator itself, or a ``Pipeline``'s final step):

- does it handle this estimator (`supports`),
- can the mode honour its configuration (`validate`),
- how to fit a fold up to the round ceiling without stopping early
  (`prepare_fold_fit`),
- what the per-round stopping metric was, and which direction is better
  (`stopping_curve`),
- how to make plain ``predict`` use only the first k rounds (`truncate`),
- how to fit on all data with a fixed round count and no early stopping
  (`prepare_refit`).

The built-in adapters touch library internals where the libraries offer no
public alternative. Those accesses are confined to this module.
"""

from __future__ import annotations

import abc
import importlib
import numbers
import sys
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

__all__ = [
    "BaseEarlyStoppingAdapter",
    "CatBoostAdapter",
    "LightGBMAdapter",
    "XGBoostAdapter",
]


class BaseEarlyStoppingAdapter(BaseEstimator, abc.ABC):
    """Base class for translating early stopping between a search and a boosting library.

    Subclass it to use ``validation="cv"`` with an estimator that no built-in
    adapter supports, and pass the instance as the search's
    ``early_stopping_adapter``. Every method receives the estimator that
    receives the evaluation set: the estimator itself, or the final step when
    the forecaster's estimator is a ``sklearn.pipeline.Pipeline``.

    See Also
    --------
    - [`LightGBMAdapter`][yohou.model_selection.LightGBMAdapter] : Adapter for LightGBM models.
    - [`XGBoostAdapter`][yohou.model_selection.XGBoostAdapter] : Adapter for XGBoost models.
    - [`CatBoostAdapter`][yohou.model_selection.CatBoostAdapter] : Adapter for CatBoost models.

    """

    @abc.abstractmethod
    def supports(self, estimator: BaseEstimator) -> bool:
        """Return whether this adapter handles the estimator.

        Used when the search resolves an adapter automatically. It must not
        import a library the caller has not imported.

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
    def validate(self, estimator: BaseEstimator) -> None:
        """Reject configurations the shared-round mode cannot honour.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted estimator that receives the evaluation set.

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


def _loaded_module(name: str) -> Any:
    """Return an already imported library module, without importing it.

    An estimator of a library can only exist once that library is imported, so
    an absent module means the estimator cannot belong to it.

    Parameters
    ----------
    name : str
        Top-level module name.

    Returns
    -------
    module or None
        The module, or None when it has not been imported.

    """
    return sys.modules.get(name)


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
    LightGBM's direction for it, which `LightGBMAdapter.stopping_curve` reads
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


class LightGBMAdapter(BaseEarlyStoppingAdapter):
    """Early-stopping adapter for LightGBM's scikit-learn estimators.

    Handles every ``lightgbm.LGBMModel`` (``LGBMRegressor``,
    ``LGBMClassifier``, ``LGBMRanker``). Fold fits remove the estimator's
    ``early_stopping_round`` and train every one of ``n_estimators`` rounds,
    recording the metric on the evaluation set after each. The stopping curve
    is the first metric of the evaluation set, as LightGBM compares with
    ``first_metric_only=True``.

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

    def validate(self, estimator: BaseEstimator) -> None:
        """Reject dart boosting.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted LightGBM estimator.

        Raises
        ------
        ValueError
            If ``boosting_type`` (or its ``boosting`` alias) is ``"dart"``.

        """
        params = estimator.get_params()
        if "dart" in (params.get("boosting_type"), params.get("boosting"), params.get("boost")):
            raise ValueError(
                "validation='cv' cannot use LightGBM with boosting_type='dart': dart rescales earlier "
                "trees as it adds new ones, so a model cut to fewer rounds is not the model trained for "
                "that many. Use boosting_type='gbdt', or validation=None."
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


class XGBoostAdapter(BaseEarlyStoppingAdapter):
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
    def _early_stopping_callbacks(estimator: BaseEstimator) -> list[Any]:
        """Return the estimator's XGBoost ``EarlyStopping`` callbacks."""
        xgboost = _loaded_module("xgboost")
        callbacks = estimator.get_params().get("callbacks") or []
        if xgboost is None:
            return []
        return [cb for cb in callbacks if isinstance(cb, xgboost.callback.EarlyStopping)]

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

    def validate(self, estimator: BaseEstimator) -> None:
        """Reject dart boosting.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted XGBoost estimator.

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
        stopping = self._early_stopping_callbacks(prepared)
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
        callbacks = self._early_stopping_callbacks(fitted)
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
            stopping = self._early_stopping_callbacks(prepared)
            updates["callbacks"] = [cb for cb in callbacks if cb not in stopping] or None
        return prepared.set_params(**updates)


_CATBOOST_ROUND_ALIASES = ("iterations", "n_estimators", "num_boost_round", "num_trees")
_CATBOOST_STOPPING_PARAMS = ("early_stopping_rounds", "od_type", "od_wait", "od_pval")


class CatBoostAdapter(BaseEarlyStoppingAdapter):
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

    def validate(self, estimator: BaseEstimator) -> None:
        """Require an explicit learning rate.

        Parameters
        ----------
        estimator : BaseEstimator
            The unfitted CatBoost estimator.

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


_BUILTIN_ADAPTERS: tuple[type[BaseEarlyStoppingAdapter], ...] = (LightGBMAdapter, XGBoostAdapter, CatBoostAdapter)


def _eval_target(estimator: BaseEstimator) -> BaseEstimator:
    """Return the estimator that receives the evaluation set.

    Parameters
    ----------
    estimator : BaseEstimator
        A forecaster's ``estimator``.

    Returns
    -------
    BaseEstimator
        The estimator itself, or a ``Pipeline``'s final step.

    """
    if isinstance(estimator, Pipeline):
        return estimator.steps[-1][1]
    return estimator


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
        An explicit adapter, returned unchanged.

    Returns
    -------
    BaseEarlyStoppingAdapter
        The explicit adapter, or the first built-in adapter supporting the
        estimator.

    Raises
    ------
    ValueError
        If no adapter is given and no built-in adapter supports the estimator.

    """
    if adapter is not None:
        return adapter
    target = _eval_target(estimator)
    for adapter_cls in _BUILTIN_ADAPTERS:
        candidate = adapter_cls()
        if candidate.supports(target):
            return candidate
    raise ValueError(
        f"validation='cv' has no early-stopping adapter for {target.__class__.__name__}. Built-in adapters "
        f"cover LightGBM, XGBoost, and CatBoost estimators; for another estimator, subclass "
        f"BaseEarlyStoppingAdapter and pass an instance as early_stopping_adapter."
    )
