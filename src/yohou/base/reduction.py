"""Base class for reduction-based forecasters."""

import abc
import inspect
import numbers
import warnings
from collections.abc import Callable
from typing import Any, Literal
from typing import cast as typing_cast

import numpy as np
import polars as pl
import polars.selectors as cs
from pydantic import StrictInt
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.metadata_routing import MetadataRouter, MethodMapping
from sklearn.utils.parallel import Parallel, delayed

from yohou.base.forecaster import BaseForecaster
from yohou.base.transformer import BaseTransformer
from yohou.utils import Tags, cast, tabularize
from yohou.utils._compat import HasMethods, Interval, StrOptions
from yohou.utils.weighting import combine_weight_vectors, resolve_weight_to_array

__all__ = ["BaseReductionForecaster"]


class BaseReductionForecaster(BaseForecaster, metaclass=abc.ABCMeta):
    """Base class for forecasters using reduction to supervised learning.

    Converts the time series forecasting task to a tabular one.

    Parameters
    ----------
    estimator : instance of `BaseEstimator`, default=LinearRegression()
        Estimator used to fit the tabularized data.
    reduction_strategy : {"direct", "dir-rec", "multi-output"}, default="multi-output"
        Reduction strategy to use.
    target_as_feature : {"transformed", "raw"} or None, default="transformed"
        Controls whether the target is included as a feature.
        ``"transformed"`` includes the transformed target, ``"raw"``
        includes the raw target, and ``None`` uses only exogenous features.
    target_transformer : instance of `BaseTransformer` or None, default=None
        Transformer used to transform the target time series into the new target.
    feature_transformer : instance of `BaseTransformer` or None, default=None
        Transformer used to transform the target time series into features.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data. See `BaseForecaster` for details.
    step_feature_alignment : {"all", "matched", "cumulative"}, default="all"
        Controls which step-indexed feature columns each direct estimator
        sees. Only affects the ``"direct"`` strategy.

        - ``"all"``: every estimator receives all step columns
          (``*_step_1..H``). Backward compatible, maximum information.
        - ``"matched"``: estimator for step h receives only ``*_step_h``
          columns. Cleanest signal, no cross-horizon leakage.
        - ``"cumulative"``: estimator for step h receives columns
          ``*_step_1..h``. All information up to horizon h.
    nan_handling : {"drop", "pass"}, default="pass"
        How to handle NaN values in tabularized data.
        ``"pass"`` leaves NaN in place (suitable for estimators that
        handle NaN natively, such as tree-based models). ``"drop"``
        removes any training instance where X or y contains NaN before
        fitting the estimator, and emits a warning with the count of
        dropped rows. At predict time, returns NaN predictions for any
        time step whose features contain NaN.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel for the ``"direct"`` strategy
        (fitting and predicting H independent models). ``None`` means 1
        unless in a ``joblib.parallel_backend`` context. ``-1`` means
        using all processors. Has no effect for ``"multi-output"`` or
        ``"dir-rec"`` strategies.

    Notes
    -----
    Reduction strategies:

    - **Multi-output**: A single model predicts all H horizon steps
      simultaneously. Simple and fast, but assumes the same model
      structure is appropriate for every step.
    - **Direct**: H independent models, one per horizon step. Each
      model specialises in its own step, avoiding error accumulation
      from recursive prediction but ignoring inter-step dependencies.
    - **Dir-Rec** (direct-recursive hybrid): H models are fitted
      sequentially. Model h predicts step h using the original features
      augmented with in-sample predictions from models 1 to h-1. This
      combines the specialised per-step training of the direct
      strategy with inter-step information flow.

    For direct and dir-rec strategies, ``estimator_`` becomes a
    ``list[BaseEstimator]`` of length H (one per horizon step) instead
    of a single estimator.

    All strategies can be applied recursively for multi-step forecasting
    beyond the fit horizon by specifying a larger forecasting horizon
    during prediction.

    See Also
    --------
    - [`PointReductionForecaster`][yohou.point.reduction.PointReductionForecaster] : Point forecaster using reduction.
    - [`IntervalReductionForecaster`][yohou.interval.reduction.IntervalReductionForecaster] : Interval forecaster using reduction.
    """

    _parameter_constraints: dict = {
        **BaseForecaster._parameter_constraints,
        "estimator": [HasMethods(["fit", "predict"])],
        "reduction_strategy": [StrOptions({"direct", "dir-rec", "multi-output"})],
        "step_feature_alignment": [StrOptions({"all", "matched", "cumulative"})],
        "nan_handling": [StrOptions({"drop", "pass"})],
        "n_jobs": [Interval(numbers.Integral, -1, None, closed="left"), None],
    }

    def __init__(
        self,
        estimator: BaseEstimator = LinearRegression(),
        reduction_strategy: Literal["direct", "dir-rec", "multi-output"] = "multi-output",
        target_as_feature: Literal["transformed", "raw"] | None = "transformed",
        target_transformer: BaseTransformer | None = None,
        feature_transformer: BaseTransformer | None = None,
        panel_strategy: Literal["global", "multivariate"] = "global",
        step_feature_alignment: Literal["all", "matched", "cumulative"] = "all",
        nan_handling: Literal["drop", "pass"] = "pass",
        n_jobs: int | None = None,
    ):
        BaseForecaster.__init__(
            self,
            target_as_feature=target_as_feature,
            target_transformer=target_transformer,
            feature_transformer=feature_transformer,
            panel_strategy=panel_strategy,
        )

        self.estimator = estimator
        self.reduction_strategy = reduction_strategy
        self.step_feature_alignment = step_feature_alignment
        self.nan_handling = nan_handling
        self.n_jobs = n_jobs

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None

        # Mark as using reduction
        tags.forecaster_tags.uses_reduction = True

        # Mark as supporting time_weight
        tags.forecaster_tags.supports_time_weight = True

        # Mark as supporting vintage_weight
        tags.forecaster_tags.supports_vintage_weight = True

        return tags

    def _process_fit_weights(
        self,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        time_weight: Callable | pl.DataFrame | dict | None,
        sample_weight_alignment: str,
        forecasting_horizon: int,
        vintage_weight: Callable | pl.DataFrame | dict | None = None,
    ) -> np.ndarray | None:
        """Convert time_weight and vintage_weight to sklearn sample_weight.

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target time series (global or panel data).
        time_weight : callable, pl.DataFrame, dict, or None
            Time weighting specification.
        sample_weight_alignment : str
            Strategy for aligning time weights to tabularized samples.
        forecasting_horizon : int
            Number of forecast steps (determines tabularization window).
        vintage_weight : callable, pl.DataFrame, dict, or None, default=None
            Vintage weighting specification.

        Returns
        -------
        np.ndarray or None
            Sample weights array matching tabularized data rows, or None if
            both time_weight and vintage_weight are None.

        """
        if time_weight is None and vintage_weight is None:
            return None

        if self.groups_ is None:
            # Global data: y_t is DataFrame
            assert isinstance(y_t, pl.DataFrame)
            sample_weights, _ = self._compute_sample_weights_one(
                y_t=y_t,
                time_weight=time_weight,
                sample_weight_alignment=sample_weight_alignment,
                forecasting_horizon=forecasting_horizon,
                group_name=None,
                vintage_weight=vintage_weight,
            )
        else:
            # Panel data: y_t is dict, stack weights
            assert isinstance(y_t, dict)
            sample_weights_list = []
            for panel_group_name in self.groups_:
                y_t_local = y_t[panel_group_name]
                weights_local, _ = self._compute_sample_weights_one(
                    y_t=y_t_local,
                    time_weight=time_weight,
                    sample_weight_alignment=sample_weight_alignment,
                    forecasting_horizon=forecasting_horizon,
                    group_name=panel_group_name,
                    vintage_weight=vintage_weight,
                )
                sample_weights_list.append(weights_local)
            sample_weights = np.concatenate(sample_weights_list)

        return sample_weights

    def _compute_sample_weights_one(
        self,
        y_t: pl.DataFrame,
        time_weight: Callable | pl.DataFrame | dict | None,
        sample_weight_alignment: str,
        forecasting_horizon: int,
        group_name: str | None,
        vintage_weight: Callable | pl.DataFrame | dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute sample weights for one time series (global or local).

        Resolves ``time_weight`` (with alignment strategy) and
        ``vintage_weight`` (direct lookup), combines them
        multiplicatively, and normalizes so ``sum = n_samples``.

        Parameters
        ----------
        y_t : pl.DataFrame
            Transformed target time series with "time" column.
        time_weight : callable, pl.DataFrame, dict, or None
            Time weighting specification.
        sample_weight_alignment : str
            Strategy for aligning time weights to tabularized samples.
        forecasting_horizon : int
            Number of forecast steps.
        group_name : str or None
            Panel group name (for panel-aware callables), or None for global data.
        vintage_weight : callable, pl.DataFrame, dict, or None, default=None
            Vintage weighting specification. Resolved via direct lookup
            at observation time (no alignment strategy).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(sample_weights, nonzero_mask)`` where ``nonzero_mask`` is a
            boolean array (``True`` for non-zero weight samples).

        """
        n_samples = len(y_t) - forecasting_horizon
        time_series = y_t["time"]

        # Resolve time_weight with alignment strategy
        tw_aligned = None
        if time_weight is not None:
            weights_array = resolve_weight_to_array(
                time_weight,
                time_series,
                join_column="time",
                group_name=group_name,
            )

            if sample_weight_alignment == "first_step":
                aligned_indices = np.arange(1, n_samples + 1)
                tw_aligned = weights_array[aligned_indices]

            elif sample_weight_alignment == "mean_step":
                tw_aligned = np.array([
                    weights_array[i + 1 : i + forecasting_horizon + 1].mean() for i in range(n_samples)
                ])

            elif sample_weight_alignment == "weighted_mean_step":
                horizon_decay = np.exp(-np.arange(forecasting_horizon) * 0.5)
                horizon_decay = horizon_decay / horizon_decay.sum()
                tw_aligned = np.array([
                    np.sum(weights_array[i + 1 : i + forecasting_horizon + 1] * horizon_decay) for i in range(n_samples)
                ])

            elif sample_weight_alignment == "max_weight_step":
                tw_aligned = np.array([
                    weights_array[i + 1 : i + forecasting_horizon + 1].max() for i in range(n_samples)
                ])

            elif sample_weight_alignment == "min_weight_step":
                tw_aligned = np.array([
                    weights_array[i + 1 : i + forecasting_horizon + 1].min() for i in range(n_samples)
                ])

            else:
                raise ValueError(
                    f"Invalid sample_weight_alignment: {sample_weight_alignment}. "
                    f"Must be 'first_step', 'mean_step', 'weighted_mean_step', "
                    f"'max_weight_step', or 'min_weight_step'."
                )

        # Resolve vintage_weight via direct lookup (no alignment)
        vw_aligned = None
        if vintage_weight is not None:
            vw_array = resolve_weight_to_array(
                vintage_weight,
                time_series,
                join_column="time",
                group_name=group_name,
            )
            # Direct lookup: sample i's vintage is time_series[i]
            vw_aligned = vw_array[:n_samples]

        # Combine and normalize
        sample_weights = combine_weight_vectors(tw_aligned, vw_aligned, n=n_samples)
        if sample_weights is None:  # pragma: no cover
            # Both were None (shouldn't reach here since caller checks)
            sample_weights = np.ones(n_samples)

        nonzero_mask = sample_weights > 0.0
        return sample_weights, nonzero_mask

    def _get_tabularized_dataset(
        self,
        y_t: pl.DataFrame,
        X_t: pl.DataFrame,
        forecasting_horizon: int,
        y_columns: list[str] | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Convert transformed time series to tabular supervised learning format.

        Creates feature matrix (X_tab) and target matrix (y_tab) suitable for training
        sklearn regressors. Target columns are lagged and renamed to indicate forecast
        steps (lag_1 → step_1 for 1-step-ahead prediction, etc.).

        Parameters
        ----------
        y_t : pl.DataFrame
            Transformed target time series.
        X_t : pl.DataFrame
            Transformed feature matrix (may include lagged y_t).
        forecasting_horizon : int
            Number of steps to forecast (determines how many lag features needed).
        y_columns : list of str or None, default=None
            Target column names. If None, uses all columns from local_y_t_schema_.

        Returns
        -------
        X_tab : pl.DataFrame
            Feature matrix for supervised learning. Excludes "time" column and
            truncates last forecasting_horizon rows (no targets available).
        y_tab : pl.DataFrame
            Target matrix with columns for each (target, step) combination.
            Columns follow pattern: {target}_step_{1}, {target}_step_{2}, ...

        Notes
        -----
        Lag-to-step renaming convention:
        - Input: y with lag_0, lag_1, lag_2, ..., lag_H features
        - For forecasting_horizon=3:
            - lag_1 → step_1 (1-step-ahead target)
            - lag_2 → step_2 (2-step-ahead target)
            - lag_3 → step_3 (3-step-ahead target)
            - lag_0 is the most recent observation (not a target)

        This convention makes it clear that we're predicting future values, not
        explaining historical ones.

        See Also
        --------
        `tabularize` : Creates lagged features.

        """
        # Use provided y_columns or fall back to all columns from local_y_t_schema_
        if y_columns is None:
            y_columns = list(self.local_y_t_schema_.keys())

        X_tab = X_t.select(~cs.by_name("time"))[:-forecasting_horizon]
        y_tab = tabularize(
            y_t.select(~cs.by_name("time")),
            lags=list(range(1 + forecasting_horizon)),
        ).rename({
            f"{col}_lag_{lag}": f"{col}_step_{forecasting_horizon - lag}"
            for lag in range(1 + forecasting_horizon)
            for col in y_columns
        })[[f"{col}_step_{step}" for step in range(1, 1 + forecasting_horizon) for col in y_columns]]

        return X_tab, y_tab

    def _estimator_fit_one(
        self,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        X_t: pl.DataFrame | dict[str, pl.DataFrame] | None,
        forecasting_horizon: StrictInt,
        time_weight: Callable | pl.DataFrame | dict | None = None,
        sample_weight_alignment: str = "first_step",
        vintage_weight: Callable | pl.DataFrame | dict | None = None,
        estimator_params: dict[str, Any] | None = None,
        estimator_fit_params: dict[str, Any] | None = None,
    ) -> BaseEstimator | list[BaseEstimator]:
        """Dispatch estimator fitting to the strategy-specific method.

        Routes to `_estimator_fit_multi_output`, `_estimator_fit_direct`,
        or `_estimator_fit_dir_rec` based on ``self.reduction_strategy``.

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target time series.
        X_t : pl.DataFrame or dict[str, pl.DataFrame] or None
            Transformed feature matrix.
        forecasting_horizon : int
            Number of steps to forecast.
        time_weight : callable, pl.DataFrame, dict, or None, default=None
            Time weighting function, DataFrame, or dict to weight samples.
        sample_weight_alignment : str, default="first_step"
            Strategy for aligning time weights to tabularized samples.
        vintage_weight : callable, pl.DataFrame, dict, or None, default=None
            Vintage weighting specification.
        estimator_params : dict or None
            Additional parameters to pass to the estimator's set_params method.
        estimator_fit_params : dict or None
            Additional parameters to pass to the estimator's fit method.

        Returns
        -------
        BaseEstimator or list[BaseEstimator]
            For ``"multi-output"``: a single fitted estimator.
            For ``"direct"`` or ``"dir-rec"``: a list of H fitted estimators
            (one per horizon step).

        See Also
        --------
        `_estimator_fit_multi_output` : Multi-output strategy.
        `_estimator_fit_direct` : Direct strategy.
        `_estimator_fit_dir_rec` : Dir-Rec (direct-recursive) strategy.

        """
        if self.reduction_strategy == "direct":
            return self._estimator_fit_direct(
                y_t,
                X_t,
                forecasting_horizon,
                time_weight=time_weight,
                sample_weight_alignment=sample_weight_alignment,
                vintage_weight=vintage_weight,
                estimator_params=estimator_params,
                estimator_fit_params=estimator_fit_params,
            )
        if self.reduction_strategy == "dir-rec":
            return self._estimator_fit_dir_rec(
                y_t,
                X_t,
                forecasting_horizon,
                time_weight=time_weight,
                sample_weight_alignment=sample_weight_alignment,
                vintage_weight=vintage_weight,
                estimator_params=estimator_params,
                estimator_fit_params=estimator_fit_params,
            )
        return self._estimator_fit_multi_output(
            y_t,
            X_t,
            forecasting_horizon,
            time_weight=time_weight,
            sample_weight_alignment=sample_weight_alignment,
            vintage_weight=vintage_weight,
            estimator_params=estimator_params,
            estimator_fit_params=estimator_fit_params,
        )

    def _get_stacked_tabularized_data(
        self,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        X_t: pl.DataFrame | dict[str, pl.DataFrame] | None,
        forecasting_horizon: int,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Tabularize and stack data for fitting (handles both standard and panel).

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target time series.
        X_t : pl.DataFrame or dict[str, pl.DataFrame] or None
            Transformed feature matrix.
        forecasting_horizon : int
            Number of steps to forecast.

        Returns
        -------
        X_tab : pl.DataFrame
            Stacked feature matrix.
        y_tab : pl.DataFrame
            Stacked target matrix with all horizon steps.

        """
        if self.groups_ is None:
            assert isinstance(y_t, pl.DataFrame)
            assert isinstance(X_t, pl.DataFrame)
            return self._get_tabularized_dataset(y_t, X_t, forecasting_horizon)

        assert isinstance(y_t, dict)
        assert isinstance(X_t, dict)
        X_tab_list, y_tab_list = [], []
        for panel_group_name in self.groups_:
            y_t_local = y_t[panel_group_name]
            X_t_local = X_t[panel_group_name]
            y_columns = [c for c in y_t_local.columns if c != "time"]
            X_tab_local, y_tab_local = self._get_tabularized_dataset(
                y_t_local,
                X_t_local,
                forecasting_horizon,
                y_columns=y_columns,
            )
            X_tab_list.append(X_tab_local)
            y_tab_list.append(y_tab_local)
        return pl.concat(X_tab_list), pl.concat(y_tab_list)

    def _validate_and_prepare_fit(
        self,
        X_tab: pl.DataFrame,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        time_weight: Callable | pl.DataFrame | dict | None,
        sample_weight_alignment: str,
        forecasting_horizon: int,
        vintage_weight: Callable | pl.DataFrame | dict | None = None,
    ) -> np.ndarray | None:
        """Validate training data and compute sample weights.

        Parameters
        ----------
        X_tab : pl.DataFrame
            Feature matrix.
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target (for weight computation).
        time_weight : callable, pl.DataFrame, dict, or None
            Time weighting specification.
        sample_weight_alignment : str
            Alignment strategy for time weights.
        forecasting_horizon : int
            Number of forecast steps.
        vintage_weight : callable, pl.DataFrame, dict, or None, default=None
            Vintage weighting specification.

        Returns
        -------
        np.ndarray or None
            Sample weights, or None.

        Raises
        ------
        ValueError
            If training dataset is empty.

        """
        if len(X_tab) == 0:
            raise ValueError(
                "Training dataset is empty (0 samples). This typically occurs when "
                "the feature transformer reduces the data size below the minimum "
                "required for the forecasting horizon. Please check your "
                "transformer settings and ensure sufficient data length."
            )
        return self._process_fit_weights(
            y_t=y_t,
            time_weight=time_weight,
            sample_weight_alignment=sample_weight_alignment,
            forecasting_horizon=forecasting_horizon,
            vintage_weight=vintage_weight,
        )

    def _apply_nan_handling(
        self,
        X_tab: pl.DataFrame,
        y_tab: pl.DataFrame | pl.Series,
        sample_weight: np.ndarray | None,
        *,
        context: str = "",
    ) -> tuple[pl.DataFrame, pl.DataFrame | pl.Series, np.ndarray | None]:
        """Remove rows containing NaN/null from tabularized training data.

        When ``nan_handling="drop"``, removes any row where X_tab or y_tab
        contains at least one null value. Filters sample_weight in lockstep.
        Emits a warning reporting the number of dropped rows.

        When ``nan_handling="pass"``, returns inputs unchanged.

        Parameters
        ----------
        X_tab : pl.DataFrame
            Feature matrix.
        y_tab : pl.DataFrame or pl.Series
            Target matrix or series.
        sample_weight : np.ndarray or None
            Sample weights (filtered in lockstep if rows are dropped).
        context : str, default=""
            Additional context for the warning message (e.g., " (step 3)").

        Returns
        -------
        X_tab : pl.DataFrame
            Filtered feature matrix.
        y_tab : pl.DataFrame or pl.Series
            Filtered target.
        sample_weight : np.ndarray or None
            Filtered sample weights.

        """
        if self.nan_handling == "pass":
            return X_tab, y_tab, sample_weight

        # For null: check all columns. For NaN: only float columns.
        null_free = X_tab.select(pl.all_horizontal(pl.all().is_not_null())).to_series()
        float_cols = X_tab.select(cs.float())
        if float_cols.width > 0:
            nan_free = float_cols.select(pl.all_horizontal(pl.all().is_not_nan())).to_series()
            x_ok = null_free & nan_free
        else:
            x_ok = null_free

        if isinstance(y_tab, pl.Series):
            y_ok = y_tab.is_not_null()
            if y_tab.dtype.is_float():
                y_ok = y_ok & y_tab.is_not_nan()
        else:
            y_null_free = y_tab.select(pl.all_horizontal(pl.all().is_not_null())).to_series()
            y_float_cols = y_tab.select(cs.float())
            if y_float_cols.width > 0:
                y_nan_free = y_float_cols.select(pl.all_horizontal(pl.all().is_not_nan())).to_series()
                y_ok = y_null_free & y_nan_free
            else:
                y_ok = y_null_free

        mask = x_ok & y_ok
        n_total = len(mask)
        n_dropped = n_total - mask.sum()

        if n_dropped > 0:
            if n_dropped == n_total:
                raise ValueError(
                    f"All {n_total} training instances contain NaN{context}. "
                    f"Cannot fit with nan_handling='drop' and 0 samples remaining."
                )
            pct = 100 * n_dropped / n_total
            warnings.warn(
                f"NaN handling dropped {n_dropped} of {n_total} training instances ({pct:.1f}%){context}.",
                stacklevel=3,
            )
            X_tab = X_tab.filter(mask)
            y_tab = y_tab.filter(mask)
            if sample_weight is not None:
                sample_weight = sample_weight[mask.to_numpy()]

        return X_tab, y_tab, sample_weight

    def _features_have_nan(self, X_tab: pl.DataFrame) -> bool:
        """Check if a feature DataFrame contains any NaN or null values.

        Only used when ``nan_handling="drop"`` to decide whether the
        estimator can be called safely at predict time.
        """
        null_free = X_tab.select(pl.all_horizontal(pl.all().is_not_null())).to_series()
        float_cols = X_tab.select(cs.float())
        if float_cols.width > 0:
            nan_free = float_cols.select(pl.all_horizontal(pl.all().is_not_nan())).to_series()
            return not (null_free & nan_free).all()
        return not null_free.all()

    def _nan_predict_result(self, n_rows: int = 1) -> np.ndarray:
        """Return a NaN array shaped like a multi-output prediction."""
        assert self.local_y_t_schema_ is not None
        n_outputs = self.fit_forecasting_horizon_ * len(self.local_y_t_schema_)
        return np.full((n_rows, n_outputs), np.nan)

    @staticmethod
    def _resolve_sample_weight_params(
        estimator: BaseEstimator,
        sample_weight: np.ndarray,
    ) -> dict[str, Any]:
        """Resolve how to pass sample_weight to the estimator's fit method.

        Handles plain estimators, sklearn ``Pipeline`` (configuring
        metadata routing on the last step), and meta-estimators that
        accept ``**kwargs``.

        Parameters
        ----------
        estimator : BaseEstimator
            The (cloned) estimator about to be fitted. May be mutated
            in place (metadata routing configuration on Pipeline steps).
        sample_weight : np.ndarray
            Sample weights array.

        Returns
        -------
        dict[str, Any]
            Keyword arguments to merge into the ``fit`` call.

        Raises
        ------
        ValueError
            If the estimator cannot accept ``sample_weight``.

        """
        fit_sig = inspect.signature(estimator.fit)  # ty: ignore[unresolved-attribute]

        # 1. Explicit sample_weight parameter
        if "sample_weight" in fit_sig.parameters:
            return {"sample_weight": sample_weight}

        # 2. Pipeline: configure metadata routing on the last step
        if isinstance(estimator, Pipeline):
            last_step = estimator.steps[-1][1]
            last_sig = inspect.signature(last_step.fit)
            if "sample_weight" not in last_sig.parameters:
                raise ValueError(
                    f"Pipeline's final step {last_step.__class__.__name__} does not support "
                    f"sample_weight parameter. Cannot use time_weight/vintage_weight for training."
                )
            last_step.set_fit_request(sample_weight=True)
            for _, step in estimator.steps[:-1]:
                if step != "passthrough":
                    step.set_fit_request(sample_weight=False)
            return {"sample_weight": sample_weight}

        # 3. VAR_KEYWORD fallback (**kwargs / **fit_params)
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in fit_sig.parameters.values())
        if has_var_keyword:
            return {"sample_weight": sample_weight}

        raise ValueError(
            f"Estimator {estimator.__class__.__name__} does not support "
            f"sample_weight parameter. Cannot use time_weight/vintage_weight for training."
        )

    def _fit_single_estimator(
        self,
        X_tab: pl.DataFrame,
        y_tab: pl.DataFrame | pl.Series,
        sample_weight: np.ndarray | None,
        estimator_params: dict[str, Any] | None = None,
        estimator_fit_params: dict[str, Any] | None = None,
    ) -> BaseEstimator:
        """Clone, configure, and fit a single estimator instance.

        Parameters
        ----------
        X_tab : pl.DataFrame
            Feature matrix.
        y_tab : pl.DataFrame or pl.Series
            Target (DataFrame for multi-output, Series for single-output).
        sample_weight : np.ndarray or None
            Sample weights.
        estimator_params : dict or None
            Parameters to pass to set_params.
        estimator_fit_params : dict or None
            Additional parameters for the fit call.

        Returns
        -------
        BaseEstimator
            Fitted estimator.

        """
        estimator = clone(self.estimator).set_params(**(estimator_params or {}))

        fit_params = estimator_fit_params or {}
        if sample_weight is not None:
            fit_params = {**fit_params, **self._resolve_sample_weight_params(estimator, sample_weight)}

        estimator.fit(X_tab, y_tab, **fit_params)
        return estimator

    def _estimator_fit_multi_output(
        self,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        X_t: pl.DataFrame | dict[str, pl.DataFrame] | None,
        forecasting_horizon: StrictInt,
        time_weight: Callable | pl.DataFrame | dict | None = None,
        sample_weight_alignment: str = "first_step",
        vintage_weight: Callable | pl.DataFrame | dict | None = None,
        estimator_params: dict[str, Any] | None = None,
        estimator_fit_params: dict[str, Any] | None = None,
    ) -> BaseEstimator:
        """Fit a single multi-output estimator on tabularized time series data.

        A single model predicts all H horizon steps simultaneously. The
        target matrix has shape ``(n_samples, H * n_targets)``.

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target time series.
        X_t : pl.DataFrame or dict[str, pl.DataFrame] or None
            Transformed feature matrix.
        forecasting_horizon : int
            Number of steps to forecast.
        time_weight : callable, pl.DataFrame, dict, or None, default=None
            Time weighting function, DataFrame, or dict to weight samples.
        sample_weight_alignment : str, default="first_step"
            Strategy for aligning time weights to tabularized samples.
        vintage_weight : callable, pl.DataFrame, dict, or None, default=None
            Vintage weighting specification.
        estimator_params : dict or None
            Additional parameters to pass to the estimator's set_params method.
        estimator_fit_params : dict or None
            Additional parameters to pass to the estimator's fit method.

        Returns
        -------
        BaseEstimator
            Fitted sklearn regressor.

        See Also
        --------
        `_get_tabularized_dataset` : Creates supervised learning matrices.
        `_estimator_predict_multi_output` : Uses fitted model for prediction.

        """
        X_tab, y_tab = self._get_stacked_tabularized_data(y_t, X_t, forecasting_horizon)
        sample_weight = self._validate_and_prepare_fit(
            X_tab,
            y_t,
            time_weight,
            sample_weight_alignment,
            forecasting_horizon,
            vintage_weight=vintage_weight,
        )
        X_tab, y_tab, sample_weight = self._apply_nan_handling(X_tab, y_tab, sample_weight)
        return self._fit_single_estimator(
            X_tab,
            y_tab,
            sample_weight,
            estimator_params,
            estimator_fit_params,
        )

    def _filter_step_features(
        self,
        X_tab: pl.DataFrame,
        step: int,
    ) -> pl.DataFrame:
        """Filter step-indexed feature columns for a direct estimator.

        When ``step_feature_alignment`` is ``"all"`` (default), returns
        ``X_tab`` unchanged. For ``"matched"``, keeps only step columns
        matching the given step number. For ``"cumulative"``, keeps step
        columns from 1 through the given step number. Non-step columns
        are always kept.

        Parameters
        ----------
        X_tab : pl.DataFrame
            Feature matrix containing observation features and possibly
            step-indexed columns from X_future/X_forecast.
        step : int
            1-based horizon step index.

        Returns
        -------
        pl.DataFrame
            Filtered feature matrix.

        """
        if self.step_feature_alignment == "all" or not self._step_column_names_:
            return X_tab

        step_cols_in_tab = [c for c in X_tab.columns if c in self._step_column_names_]
        if not step_cols_in_tab:
            return X_tab

        if self.step_feature_alignment == "matched":
            keep_suffix = f"_step_{step}"
            drop = [c for c in step_cols_in_tab if not c.endswith(keep_suffix)]
        else:
            # cumulative: keep _step_1 .. _step_{step}
            keep_suffixes = {f"_step_{s}" for s in range(1, step + 1)}
            drop = [c for c in step_cols_in_tab if not any(c.endswith(s) for s in keep_suffixes)]

        return X_tab.drop(drop) if drop else X_tab

    def _estimator_fit_direct(
        self,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        X_t: pl.DataFrame | dict[str, pl.DataFrame] | None,
        forecasting_horizon: StrictInt,
        time_weight: Callable | pl.DataFrame | dict | None = None,
        sample_weight_alignment: str = "first_step",
        vintage_weight: Callable | pl.DataFrame | dict | None = None,
        estimator_params: dict[str, Any] | None = None,
        estimator_fit_params: dict[str, Any] | None = None,
    ) -> list[BaseEstimator]:
        """Fit H independent estimators, one per horizon step.

        Each model ``h`` is trained to predict step ``h`` only
        (single-output regression). The feature matrix is the same for
        all models; only the target column differs.

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target time series.
        X_t : pl.DataFrame or dict[str, pl.DataFrame] or None
            Transformed feature matrix.
        forecasting_horizon : int
            Number of steps to forecast.
        time_weight : callable, pl.DataFrame, dict, or None, default=None
            Time weighting function, DataFrame, or dict to weight samples.
        sample_weight_alignment : str, default="first_step"
            Strategy for aligning time weights to tabularized samples.
        vintage_weight : callable, pl.DataFrame, dict, or None, default=None
            Vintage weighting specification.
        estimator_params : dict or None
            Additional parameters to pass to each estimator's set_params.
        estimator_fit_params : dict or None
            Additional parameters to pass to each estimator's fit.

        Returns
        -------
        list[BaseEstimator]
            List of H fitted estimators, one per horizon step.

        See Also
        --------
        `_estimator_predict_direct` : Uses fitted models for prediction.

        """
        X_tab, y_tab = self._get_stacked_tabularized_data(y_t, X_t, forecasting_horizon)
        sample_weight = self._validate_and_prepare_fit(
            X_tab,
            y_t,
            time_weight,
            sample_weight_alignment,
            forecasting_horizon,
            vintage_weight=vintage_weight,
        )

        y_columns = (
            list(self.local_y_t_schema_.keys())
            if self.groups_ is None
            else [c for c in next(iter(y_t.values())).columns if c != "time"]
            if isinstance(y_t, dict)
            else list(self.local_y_t_schema_.keys())
        )

        def _fit_step(step: int) -> BaseEstimator:
            """Fit a single estimator for horizon step."""
            step_col_names = [f"{col}_step_{step + 1}" for col in y_columns]
            y_step: pl.DataFrame | pl.Series = y_tab.select(step_col_names)
            if y_step.shape[1] == 1:
                y_step = y_step.to_series()
            X_tab_step = self._filter_step_features(X_tab, step + 1)
            X_tab_step, y_step, sw_step = self._apply_nan_handling(
                X_tab_step, y_step, sample_weight, context=f" (step {step + 1})"
            )
            return self._fit_single_estimator(
                X_tab_step,
                y_step,
                sw_step,
                estimator_params,
                estimator_fit_params,
            )

        estimators: list[BaseEstimator] = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_step)(step) for step in range(forecasting_horizon)
        )
        return estimators

    def _estimator_fit_dir_rec(
        self,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        X_t: pl.DataFrame | dict[str, pl.DataFrame] | None,
        forecasting_horizon: StrictInt,
        time_weight: Callable | pl.DataFrame | dict | None = None,
        sample_weight_alignment: str = "first_step",
        vintage_weight: Callable | pl.DataFrame | dict | None = None,
        estimator_params: dict[str, Any] | None = None,
        estimator_fit_params: dict[str, Any] | None = None,
    ) -> list[BaseEstimator]:
        """Fit H estimators sequentially with recursive feature augmentation.

        Model ``h`` predicts step ``h`` using the original features
        augmented with in-sample predictions from models
        ``1, 2, ..., h-1``. This combines the direct strategy's
        per-step specialization with recursive information flow.

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target time series.
        X_t : pl.DataFrame or dict[str, pl.DataFrame] or None
            Transformed feature matrix.
        forecasting_horizon : int
            Number of steps to forecast.
        time_weight : callable, pl.DataFrame, dict, or None, default=None
            Time weighting function, DataFrame, or dict to weight samples.
        sample_weight_alignment : str, default="first_step"
            Strategy for aligning time weights to tabularized samples.
        vintage_weight : callable, pl.DataFrame, dict, or None, default=None
            Vintage weighting specification.
        estimator_params : dict or None
            Additional parameters to pass to each estimator's set_params.
        estimator_fit_params : dict or None
            Additional parameters to pass to each estimator's fit.

        Returns
        -------
        list[BaseEstimator]
            List of H fitted estimators with progressively augmented features.

        See Also
        --------
        `_estimator_predict_dir_rec` : Uses fitted models for prediction.

        """
        X_tab, y_tab = self._get_stacked_tabularized_data(y_t, X_t, forecasting_horizon)
        sample_weight = self._validate_and_prepare_fit(
            X_tab,
            y_t,
            time_weight,
            sample_weight_alignment,
            forecasting_horizon,
            vintage_weight=vintage_weight,
        )
        X_tab, y_tab, sample_weight = self._apply_nan_handling(X_tab, y_tab, sample_weight)
        assert isinstance(y_tab, pl.DataFrame)

        self._dir_rec_n_original_features_ = X_tab.shape[1]

        y_columns = (
            list(self.local_y_t_schema_.keys())
            if self.groups_ is None
            else [c for c in next(iter(y_t.values())).columns if c != "time"]
            if isinstance(y_t, dict)
            else list(self.local_y_t_schema_.keys())
        )

        estimators: list[BaseEstimator] = []
        X_aug = X_tab.clone()  # Progressively augmented feature matrix
        for step in range(forecasting_horizon):
            step_col_names = [f"{col}_step_{step + 1}" for col in y_columns]
            y_step: pl.DataFrame | pl.Series = y_tab.select(step_col_names)
            if y_step.shape[1] == 1:
                y_step = y_step.to_series()
            est = self._fit_single_estimator(
                X_aug,
                y_step,
                sample_weight,
                estimator_params,
                estimator_fit_params,
            )
            estimators.append(est)

            # Augment features with in-sample predictions for next step
            if step < forecasting_horizon - 1:
                preds = est.predict(X_aug)  # ty: ignore[unresolved-attribute]
                if preds.ndim == 1:
                    preds = preds.reshape(-1, 1)
                X_aug = X_aug.with_columns([pl.Series(f"__aug_{step}_{j}", preds[:, j]) for j in range(preds.shape[1])])

        return estimators

    def _estimator_predict_one(
        self,
        estimator: BaseEstimator | list[BaseEstimator],
        groups: list[str],
    ) -> pl.DataFrame:
        """Dispatch estimator prediction to the strategy-specific method.

        Routes to `_estimator_predict_multi_output`,
        `_estimator_predict_direct`, or `_estimator_predict_dir_rec`
        based on ``self.reduction_strategy``.

        Parameters
        ----------
        estimator : BaseEstimator or list[BaseEstimator]
            For ``"multi-output"``: a single fitted estimator.
            For ``"direct"`` or ``"dir-rec"``: a list of H fitted estimators.
        groups : list of str
            Panel group names to predict for.

        Returns
        -------
        pl.DataFrame
            Predictions for the forecasting horizon.

        See Also
        --------
        `_estimator_predict_multi_output` : Multi-output strategy.
        `_estimator_predict_direct` : Direct strategy.
        `_estimator_predict_dir_rec` : Dir-Rec (direct-recursive) strategy.

        """
        if self.reduction_strategy == "direct":
            assert isinstance(estimator, list)
            return self._estimator_predict_direct(
                typing_cast(list[BaseEstimator], estimator),
                groups,
            )
        if self.reduction_strategy == "dir-rec":
            assert isinstance(estimator, list)
            return self._estimator_predict_dir_rec(
                typing_cast(list[BaseEstimator], estimator),
                groups,
            )
        assert isinstance(estimator, BaseEstimator)
        return self._estimator_predict_multi_output(estimator, groups)

    def _get_predict_features(
        self,
        panel_group_name: str | None = None,
    ) -> pl.DataFrame:
        """Extract the last-row feature vector for prediction.

        Parameters
        ----------
        panel_group_name : str or None
            If None, uses global ``_X_t_observed``. Otherwise, uses
            the panel group's DataFrame.

        Returns
        -------
        pl.DataFrame
            Feature row for prediction.

        """
        assert self._X_t_observed is not None
        assert self.local_X_t_schema_ is not None
        if panel_group_name is None:
            assert isinstance(self._X_t_observed, pl.DataFrame)
            X_t = self._X_t_observed.tail(1).select(~cs.by_name("time"))
        else:
            assert isinstance(self._X_t_observed, dict)
            X_t_dict = typing_cast(dict[str, pl.DataFrame], self._X_t_observed)
            X_t = X_t_dict[panel_group_name].tail(1).select(~cs.by_name("time"))
        return X_t.select(list(self.local_X_t_schema_.keys()))

    def _reshape_predictions(
        self,
        y_tab_pred: np.ndarray,
        panel_group_name: str | None = None,
    ) -> pl.DataFrame:
        """Reshape raw prediction array into a polars DataFrame.

        Parameters
        ----------
        y_tab_pred : np.ndarray
            Raw prediction output from estimator.predict.
        panel_group_name : str or None
            If not None, re-prefix columns for panel data.

        Returns
        -------
        pl.DataFrame
            Predictions with proper column names and dtypes.

        """
        assert self.local_y_t_schema_ is not None
        y_cols = list(self.local_y_t_schema_.keys())
        y_pred = pl.DataFrame(
            y_tab_pred.reshape(self.fit_forecasting_horizon_, len(y_cols)),
            schema=y_cols,
        )
        y_pred = cast(y_pred, self.local_y_t_schema_)
        if panel_group_name is not None:
            y_pred = y_pred.rename({col: f"{panel_group_name}__{col}" for col in y_cols})
        return y_pred

    def _estimator_predict_multi_output(
        self,
        estimator: BaseEstimator,
        groups: list[str],
    ) -> pl.DataFrame:
        """Generate predictions using a fitted multi-output estimator.

        Parameters
        ----------
        estimator : BaseEstimator
            Fitted scikit-learn estimator.
        groups : list of str
            Panel group names to predict for.

        Returns
        -------
        pl.DataFrame
            Predictions for the forecasting horizon.

        """
        if self.groups_ is None:
            X_tab = self._get_predict_features()
            if self.nan_handling == "drop" and self._features_have_nan(X_tab):
                y_tab_pred = self._nan_predict_result()
            else:
                y_tab_pred = estimator.predict(X_tab)  # ty: ignore[unresolved-attribute]
            return self._reshape_predictions(y_tab_pred)

        y_pred_dict = {}
        for panel_group_name in groups:
            X_tab = self._get_predict_features(panel_group_name)
            if self.nan_handling == "drop" and self._features_have_nan(X_tab):
                y_tab_pred = self._nan_predict_result()
            else:
                y_tab_pred = estimator.predict(X_tab)
            y_pred_dict[panel_group_name] = self._reshape_predictions(y_tab_pred, panel_group_name)
        return pl.concat(list(y_pred_dict.values()), how="horizontal")

    def _estimator_predict_direct(
        self,
        estimators: list[BaseEstimator],
        groups: list[str],
    ) -> pl.DataFrame:
        """Generate predictions using H independent direct estimators.

        Each estimator predicts a single horizon step. Results are
        stacked row-wise to form the full forecast.

        Parameters
        ----------
        estimators : list[BaseEstimator]
            H fitted estimators, one per horizon step.
        groups : list of str
            Panel group names to predict for.

        Returns
        -------
        pl.DataFrame
            Predictions for the forecasting horizon.

        """
        assert self.local_y_t_schema_ is not None
        y_cols = list(self.local_y_t_schema_.keys())
        n_targets = len(y_cols)

        def _predict_step(est: BaseEstimator, X_tab: pl.DataFrame) -> np.ndarray:
            """Predict a single horizon step."""
            if self.nan_handling == "drop" and self._features_have_nan(X_tab):
                return np.full(n_targets, np.nan)
            pred = est.predict(X_tab)  # ty: ignore[unresolved-attribute]
            return np.atleast_1d(pred.ravel())[:n_targets]

        if self.groups_ is None:
            X_tab = self._get_predict_features()
            rows: list[np.ndarray] = Parallel(n_jobs=self.n_jobs)(
                delayed(_predict_step)(est, self._filter_step_features(X_tab, step + 1))
                for step, est in enumerate(estimators)
            )
            y_pred_arr = np.vstack(rows)
            y_pred = pl.DataFrame(y_pred_arr, schema=y_cols)
            return cast(y_pred, self.local_y_t_schema_)

        y_pred_dict = {}
        for panel_group_name in groups:
            X_tab = self._get_predict_features(panel_group_name)
            rows = Parallel(n_jobs=self.n_jobs)(
                delayed(_predict_step)(est, self._filter_step_features(X_tab, step + 1))
                for step, est in enumerate(estimators)
            )
            y_pred_arr = np.vstack(rows)
            y_pred_local = pl.DataFrame(y_pred_arr, schema=y_cols)
            y_pred_local = cast(y_pred_local, self.local_y_t_schema_)
            y_pred_local = y_pred_local.rename({col: f"{panel_group_name}__{col}" for col in y_cols})
            y_pred_dict[panel_group_name] = y_pred_local
        return pl.concat(list(y_pred_dict.values()), how="horizontal")

    def _estimator_predict_dir_rec(
        self,
        estimators: list[BaseEstimator],
        groups: list[str],
    ) -> pl.DataFrame:
        """Generate predictions using H dir-rec estimators with feature augmentation.

        Model 1 predicts on original features. Model h predicts on
        original features augmented with predictions from models 1..h-1.

        Parameters
        ----------
        estimators : list[BaseEstimator]
            H fitted estimators with progressively augmented features.
        groups : list of str
            Panel group names to predict for.

        Returns
        -------
        pl.DataFrame
            Predictions for the forecasting horizon.

        """
        assert self.local_y_t_schema_ is not None
        y_cols = list(self.local_y_t_schema_.keys())
        n_targets = len(y_cols)

        if self.groups_ is None:
            X_tab = self._get_predict_features()
            X_aug = X_tab.clone()
            rows = []
            for i, est in enumerate(estimators):
                if self.nan_handling == "drop" and self._features_have_nan(X_aug):
                    pred = np.full(n_targets, np.nan)
                else:
                    pred = est.predict(X_aug)  # ty: ignore[unresolved-attribute]
                    pred = np.atleast_1d(pred.ravel())
                rows.append(pred[:n_targets])
                # Augment features for next model
                X_aug = X_aug.with_columns([pl.Series(f"__aug_{i}_{j}", [v]) for j, v in enumerate(pred)])
            y_pred_arr = np.vstack(rows)
            y_pred = pl.DataFrame(y_pred_arr, schema=y_cols)
            return cast(y_pred, self.local_y_t_schema_)

        y_pred_dict = {}
        for panel_group_name in groups:
            X_tab = self._get_predict_features(panel_group_name)
            X_aug = X_tab.clone()
            rows = []
            for i, est in enumerate(estimators):
                if self.nan_handling == "drop" and self._features_have_nan(X_aug):
                    pred = np.full(n_targets, np.nan)
                else:
                    pred = est.predict(X_aug)
                    pred = np.atleast_1d(pred.ravel())
                rows.append(pred[:n_targets])
                X_aug = X_aug.with_columns([pl.Series(f"__aug_{i}_{j}", [v]) for j, v in enumerate(pred)])
            y_pred_arr = np.vstack(rows)
            y_pred_local = pl.DataFrame(y_pred_arr, schema=y_cols)
            y_pred_local = cast(y_pred_local, self.local_y_t_schema_)
            y_pred_local = y_pred_local.rename({col: f"{panel_group_name}__{col}" for col in y_cols})
            y_pred_dict[panel_group_name] = y_pred_local
        return pl.concat(list(y_pred_dict.values()), how="horizontal")

    def get_metadata_routing(self) -> MetadataRouter:
        """Get metadata routing including wrapped estimator.

        BaseReductionForecaster is a router because it wraps a sklearn estimator.
        It needs to forward metadata (like time_weight) from the forecaster's
        fit() method to the wrapped estimator's fit() method.

        Returns
        -------
        router : MetadataRouter
            Router that forwards to transformers (from parent) and wrapped estimator.
        """
        # Get parent routing (for target_transformer, feature_transformer)
        router = super().get_metadata_routing()

        # Add wrapped sklearn estimator routing
        if hasattr(self, "estimator") and self.estimator is not None:
            router.add(
                estimator=self.estimator,
                method_mapping=MethodMapping().add(caller="fit", callee="fit"),
            )

        return router
