"""Cross-validation functions for evaluating forecasters."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.utils.validation import indexable

from yohou.base import BaseForecaster
from yohou.metrics.base import BaseScorer

from .split import BaseSplitter, check_cv
from .utils import (
    _check_scoring,
    _collect_coverage_rates,
    _fit_and_score,
    _MultimetricScorer,
    _resolve_response_method,
    _validate_forecaster_scorer_compatibility,
)


def cross_validate(
    forecaster: BaseForecaster,
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None = None,
    forecasting_horizon: int = 1,
    *,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
    scoring: BaseScorer | dict[str, BaseScorer],
    cv: int | BaseSplitter | None = 5,
    predict_forecasting_horizon: int | None = None,
    predict_stride: int | None = None,
    n_jobs: int | None = None,
    verbose: int = 0,
    pre_dispatch: str | int = "2*n_jobs",
    return_train_score: bool = False,
    return_forecaster: bool = False,
    return_indices: bool = False,
    error_score: float | str = np.nan,
) -> pl.DataFrame | dict[str, Any]:
    """Evaluate a forecaster by cross-validation and return test scores and timings.

    Parameters
    ----------
    forecaster : BaseForecaster
        The forecaster to evaluate.
    y : pl.DataFrame
        Target time series with a ``"time"`` column.
    X_actual : pl.DataFrame or None, default=None
        Actual feature observations with a ``"time"`` column.
    forecasting_horizon : int, default=1
        Number of time steps to forecast.
    X_future : pl.DataFrame or None, default=None
        Known future features with a ``"time"`` column.
    X_forecast : pl.DataFrame or None, default=None
        External forecasts with ``"vintage_time"`` and ``"time"``
        columns.
    scoring : BaseScorer or dict of str to BaseScorer
        Scorer (single or multi-metric) used to evaluate predictions.
    cv : int, BaseSplitter, or None, default=5
        Cross-validation splitting strategy. ``None`` or int creates
        an ``ExpandingWindowSplitter``.
    predict_forecasting_horizon : int or None, default=None
        Override forecasting horizon for ``observe_predict``.
        ``None`` uses the forecaster's fit-time default.
    predict_stride : int or None, default=None
        Override stride for ``observe_predict``.
        ``None`` uses the forecaster's default.
    n_jobs : int or None, default=None
        Number of parallel jobs (``None`` means 1).
    verbose : int, default=0
        Verbosity level.
    pre_dispatch : str or int, default="2*n_jobs"
        Controls the number of pre-dispatched jobs for parallel
        execution.
    return_train_score : bool, default=False
        Whether to include training scores.
    return_forecaster : bool, default=False
        Whether to include fitted forecasters.
    return_indices : bool, default=False
        Whether to include train/test indices per fold.
    error_score : float or "raise", default=np.nan
        Value to assign if an error occurs during fitting.

    Returns
    -------
    pl.DataFrame or dict
        When ``return_forecaster`` and ``return_indices`` are both
        ``False`` (the default), returns a ``pl.DataFrame`` with one
        row per fold containing columns ``split`` (int), ``fit_time``
        (float), ``score_time`` (float), and score columns.

        Single scorer: ``test_score`` (and ``train_score`` if
        ``return_train_score=True``).

        Multi-metric: ``test_{name}`` (and ``train_{name}``)
        for each scorer name.

        When ``return_forecaster`` or ``return_indices`` is ``True``,
        returns a dict with ``"results"`` (the DataFrame),
        ``"forecaster"`` (list of fitted forecasters), and/or
        ``"indices"`` (dict with ``"train"`` and ``"test"`` lists
        of ``np.ndarray``).
    """
    scorers: BaseScorer | _MultimetricScorer
    if isinstance(scoring, dict):
        # _check_scoring validates the dict keys/values, then we wrap
        _check_scoring(forecaster, scoring)
        scorers = _MultimetricScorer(scorers=cast(dict[str, BaseScorer], scoring), raise_exc=(error_score == "raise"))
    else:
        scorers = _check_scoring(forecaster, scoring)

    _validate_forecaster_scorer_compatibility(forecaster, scorers)

    response_method = _resolve_response_method(scorers)
    collected_coverage_rates = _collect_coverage_rates(scorers) if response_method == "predict_interval" else None

    y, X_actual = indexable(y, X_actual)

    cv_obj = check_cv(cv, forecasting_horizon)
    splits = list(cv_obj.split(y, X_actual))
    n_splits = len(splits)

    base_forecaster = clone(forecaster)

    out = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)(
        delayed(_fit_and_score)(
            clone(base_forecaster),
            y,
            X_actual,
            forecasting_horizon,
            X_future=X_future,
            X_forecast=X_forecast,
            scorer=scorers,
            train=train,
            test=test,
            verbose=verbose,
            parameters=None,
            fit_params=None,
            predict_func_params=None,
            score_params=None,
            return_train_score=return_train_score,
            return_times=True,
            return_forecaster=return_forecaster,
            predict_forecasting_horizon=predict_forecasting_horizon,
            predict_stride=predict_stride,
            error_score=error_score,
            coverage_rates=collected_coverage_rates,
            split_progress=(split_idx, n_splits),
        )
        for split_idx, (train, test) in enumerate(splits)
    )

    results: dict[str, Any] = {}

    # Aggregate fit_time and score_time
    results["split"] = list(range(n_splits))
    results["fit_time"] = [r["fit_time"] for r in out]
    results["score_time"] = [r["score_time"] for r in out]

    # Aggregate scores
    first_scores = out[0]["test_scores"]
    if isinstance(first_scores, dict):
        for name in first_scores:
            results[f"test_{name}"] = [r["test_scores"][name] for r in out]
        if return_train_score:
            for name in first_scores:
                results[f"train_{name}"] = [r["train_scores"][name] for r in out]
    else:
        results["test_score"] = [r["test_scores"] for r in out]
        if return_train_score:
            results["train_score"] = [r["train_scores"] for r in out]

    results_df = pl.DataFrame(results)

    if return_forecaster or return_indices:
        ret: dict[str, Any] = {"results": results_df}
        if return_forecaster:
            ret["forecaster"] = [r["forecaster"] for r in out]
        if return_indices:
            ret["indices"] = {
                "train": [train for train, _ in splits],
                "test": [test for _, test in splits],
            }
        return ret

    return results_df


def cross_val_score(
    forecaster: BaseForecaster,
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None = None,
    forecasting_horizon: int = 1,
    *,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
    scoring: BaseScorer,
    cv: int | BaseSplitter | None = 5,
    predict_forecasting_horizon: int | None = None,
    predict_stride: int | None = None,
    n_jobs: int | None = None,
    verbose: int = 0,
    pre_dispatch: str | int = "2*n_jobs",
    error_score: float | str = np.nan,
) -> pl.DataFrame:
    """Evaluate a forecaster by cross-validation and return test scores.

    Parameters
    ----------
    forecaster : BaseForecaster
        The forecaster to evaluate.
    y : pl.DataFrame
        Target time series with a ``"time"`` column.
    X_actual : pl.DataFrame or None, default=None
        Actual feature observations with a ``"time"`` column.
    forecasting_horizon : int, default=1
        Number of time steps to forecast.
    X_future : pl.DataFrame or None, default=None
        Known future features with a ``"time"`` column.
    X_forecast : pl.DataFrame or None, default=None
        External forecasts with ``"vintage_time"`` and ``"time"``
        columns.
    scoring : BaseScorer
        Single scorer used to evaluate predictions.  Dicts of scorers
        are rejected; use ``cross_validate`` for multi-metric
        evaluation.
    cv : int, BaseSplitter, or None, default=5
        Cross-validation splitting strategy.
    predict_forecasting_horizon : int or None, default=None
        Override forecasting horizon for ``observe_predict``.
    predict_stride : int or None, default=None
        Override stride for ``observe_predict``.
    n_jobs : int or None, default=None
        Number of parallel jobs.
    verbose : int, default=0
        Verbosity level.
    pre_dispatch : str or int, default="2*n_jobs"
        Controls pre-dispatched jobs for parallel execution.
    error_score : float or "raise", default=np.nan
        Value to assign if an error occurs during fitting.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``split`` (int, 0-indexed fold
        identifier) and ``score`` (float, test score per fold).
    """
    if isinstance(scoring, dict):
        raise ValueError(
            "cross_val_score does not accept dict scoring. Use cross_validate for multi-metric evaluation."
        )

    cv_results = cast(
        pl.DataFrame,
        cross_validate(
            forecaster,
            y,
            X_actual=X_actual,
            forecasting_horizon=forecasting_horizon,
            X_future=X_future,
            X_forecast=X_forecast,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            predict_forecasting_horizon=predict_forecasting_horizon,
            predict_stride=predict_stride,
            error_score=error_score,
        ),
    )
    return cv_results.select("split", pl.col("test_score").alias("score"))


def cross_val_predict(
    forecaster: BaseForecaster,
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None = None,
    forecasting_horizon: int = 1,
    *,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
    cv: int | BaseSplitter | None = 5,
    predict_forecasting_horizon: int | None = None,
    predict_stride: int | None = None,
    coverage_rates: list[float] | None = None,
    n_jobs: int | None = None,
    verbose: int = 0,
    pre_dispatch: str | int = "2*n_jobs",
    method: str = "predict",
) -> pl.DataFrame:
    """Generate cross-validated predictions for each fold.

    For each CV fold, the forecaster is fitted on the training data
    and predictions are produced on the test data.  Predictions from
    all folds are concatenated into a single DataFrame with a
    ``"split"`` column identifying the originating fold.

    Parameters
    ----------
    forecaster : BaseForecaster
        The forecaster to evaluate.
    y : pl.DataFrame
        Target time series with a ``"time"`` column.
    X_actual : pl.DataFrame or None, default=None
        Actual feature observations with a ``"time"`` column.
    forecasting_horizon : int, default=1
        Number of time steps to forecast.
    X_future : pl.DataFrame or None, default=None
        Known future features with a ``"time"`` column.
    X_forecast : pl.DataFrame or None, default=None
        External forecasts with ``"vintage_time"`` and ``"time"``
        columns.
    cv : int, BaseSplitter, or None, default=5
        Cross-validation splitting strategy.
    predict_forecasting_horizon : int or None, default=None
        Override forecasting horizon for ``observe_predict``.
    predict_stride : int or None, default=None
        Override stride for ``observe_predict``.
    coverage_rates : list of float or None, default=None
        Coverage rates for interval predictions.  Only used when
        ``method="predict_interval"``.
    n_jobs : int or None, default=None
        Number of parallel jobs.
    verbose : int, default=0
        Verbosity level.
    pre_dispatch : str or int, default="2*n_jobs"
        Controls pre-dispatched jobs for parallel execution.
    method : str, default="predict"
        Prediction method: ``"predict"``, ``"predict_interval"``,
        or ``"predict_class_proba"``.

    Returns
    -------
    pl.DataFrame
        Concatenated predictions from all folds with an integer
        ``"split"`` column identifying the originating fold.
    """
    valid_methods = {"predict", "predict_interval", "predict_class_proba"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method!r}.")

    y, X_actual = indexable(y, X_actual)

    cv_obj = check_cv(cv, forecasting_horizon)
    splits = list(cv_obj.split(y, X_actual))
    n_splits = len(splits)

    base_forecaster = clone(forecaster)

    out = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)(
        delayed(_fit_and_score)(
            clone(base_forecaster),
            y,
            X_actual,
            forecasting_horizon,
            X_future=X_future,
            X_forecast=X_forecast,
            scorer=None,
            train=train,
            test=test,
            verbose=verbose,
            parameters=None,
            fit_params=None,
            predict_func_params=None,
            score_params=None,
            return_predictions=True,
            predict_method=method,
            predict_forecasting_horizon=predict_forecasting_horizon,
            predict_stride=predict_stride,
            coverage_rates=coverage_rates,
            error_score="raise",
            split_progress=(split_idx, n_splits),
        )
        for split_idx, (train, test) in enumerate(splits)
    )

    dfs = []
    for split_idx, result in enumerate(out):
        pred = result["predictions"]
        dfs.append(pred.with_columns(pl.lit(split_idx).alias("split")))

    return pl.concat(dfs)
