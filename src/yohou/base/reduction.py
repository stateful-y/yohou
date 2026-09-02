"""Base class for reduction-based forecasters."""

import abc
import inspect
import numbers
import warnings
from typing import Any, Literal
from typing import cast as typing_cast

import numpy as np
import polars as pl
import polars.selectors as cs
from pydantic import StrictInt
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils.metadata_routing import MetadataRouter, MethodMapping
from sklearn.utils.parallel import Parallel, delayed

from yohou.base.forecast_transformer import BaseForecastTransformer
from yohou.base.forecaster import BaseForecaster
from yohou.base.step_transformer import BaseStepTransformer, _is_step_indexed, _step_index
from yohou.base.transformer import BaseActualTransformer
from yohou.base.utils import _derive_step_columns, _observe_transformers_one
from yohou.utils import Tags, cast, tabularize
from yohou.utils._compat import HasMethods, Interval, StrOptions
from yohou.utils.panel import get_group_df
from yohou.weighting import BaseWeighter
from yohou.weighting.weighters import _combine_weight_vectors, _resolve_weighter_to_array

__all__ = ["BaseReductionForecaster"]


def _predict_direct_step(
    estimator: BaseEstimator,
    X_tab: pl.DataFrame,
    n_targets: int,
    row_ok: np.ndarray,
) -> np.ndarray:
    """Predict one horizon step for every observation row given.

    Module level and free of any reference to the forecaster: a task dispatched to a
    worker carries this function by name, the one estimator for its step, and the feature
    rows, and nothing else.

    One call covers every panel group. Under ``panel_strategy="global"`` all groups share
    the estimator for a given step, so their feature rows stack into a single call rather
    than one call per group.

    Parameters
    ----------
    estimator : BaseEstimator
        Fitted estimator for this horizon step.
    X_tab : pl.DataFrame
        Feature rows, one per observation unit (one per panel group, or a single row when
        the data is not a panel).
    n_targets : int
        Target columns to take from the prediction.
    row_ok : np.ndarray
        Boolean mask, ``True`` for rows safe to feed to the estimator. Rows masked out
        yield NaN. Resolved by the caller, so this function needs neither the
        ``nan_handling`` mode nor the check itself.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_rows, n_targets)``, NaN in the rows ``row_ok`` excluded.

    """
    out = np.full((X_tab.height, n_targets), np.nan)
    if row_ok.all():
        kept = X_tab
    elif row_ok.any():
        kept = X_tab.filter(pl.Series(row_ok))
    else:
        return out
    pred = np.asarray(estimator.predict(kept))  # ty: ignore[unresolved-attribute]
    out[row_ok] = pred.reshape(kept.height, -1)[:, :n_targets]
    return out


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
    target_transformer : instance of `BaseActualTransformer` or None, default=None
        Transformer used to transform the target time series into the new target.
    actual_transformer : instance of `BaseActualTransformer` or None, default=None
        Transformer used to transform the feature time series into features.
    forecast_transformer : instance of `BaseForecastTransformer` or None, default=None
        Transformer applied to ``X_forecast`` before step columns are derived,
        so the step columns reaching the estimator are built from transformed
        values. Must be forecast-kind (vintage-indexed); an actual-kind
        transformer is rejected. ``None`` leaves ``X_forecast`` untouched.
    step_transformer : BaseStepTransformer or None, default=None
        Transformer applied to the derived ``{base}_step_1..H`` frame after
        step columns are built from ``X_future``/``X_forecast`` and before they
        join the design matrix. Reduces or rescales along the horizon axis.
        ``None`` leaves the step columns as derived.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data. See `BaseForecaster` for details.
    step_feature_alignment : {"all", "matched", "cumulative"}, default="all"
        Controls which step-indexed feature columns each direct estimator
        sees. Only the ``"direct"`` strategy applies this parameter; setting it
        to a non-default value on any other strategy emits a ``UserWarning`` at
        fit and changes nothing.

        - ``"all"``: every estimator receives all step columns
          (``*_step_1..H``). Backward compatible, maximum information.
        - ``"matched"``: estimator for step h receives only ``*_step_h``
          columns. Cleanest signal, no cross-horizon leakage.
        - ``"cumulative"``: estimator for step h receives columns
          ``*_step_1..h``. All information up to horizon h.

        Filtering applies identically whatever the step columns were derived
        from (``X_future``, ``X_forecast``, or an ``X_forecast`` routed through a
        ``forecast_transformer``) and under either panel strategy. A non-default
        alignment that cannot recognize any step column in the feature matrix
        raises ``RuntimeError`` rather than falling back to ``"all"``, so a
        filter that fails to apply cannot be mistaken for one that had nothing
        to filter.

        The other strategies are excluded for different reasons.
        ``"multi-output"`` *cannot* filter: one estimator predicts every
        horizon from a single feature vector, reading a different step column
        per output, so it needs them all. ``"dir-rec"`` *could* filter (before
        each step's feature augmentation) but does not; that is a deliberate
        scope decision rather than a structural limit.
    training_stride : int, default=1
        Keep one tabularized training instance every ``training_stride`` rows,
        tail-anchored: the most recent instance is always kept and kept
        origins sit ``training_stride`` rows apart counting back from it. The
        default 1 keeps every instance. Combined with data whose last row sits
        on a production origin, a stride of one day in rows trains only on
        instances whose origin matches the production decision cadence. The
        mask applies to the feature matrix, the target matrix, and
        ``sample_weight`` in lockstep, before ``nan_handling``, and on panel
        data it is built per group and stacked in group order.
    nan_handling : {"drop", "pass"}, default="pass"
        How to handle NaN values in tabularized data.
        ``"pass"`` leaves NaN in place (suitable for estimators that
        handle NaN natively, such as tree-based models). ``"drop"``
        removes any training instance where X or y contains NaN before
        fitting the estimator, and emits a warning with the count of
        dropped rows. At predict time, returns NaN predictions for any
        time step whose features contain NaN.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel over the H independent models of
        the ``"direct"`` strategy, at fit and at predict. ``None`` means 1
        unless in a ``joblib.parallel_backend`` context. ``-1`` means using
        all processors. Has no effect for ``"multi-output"`` or ``"dir-rec"``
        strategies.

        Raising it above 1 pays at fit, where a task is a whole estimator
        training. At predict whether it pays depends on the estimator: a task
        is one call over the stacked observation rows, so a gradient-boosted
        model on a wide feature matrix gains from the dispatch while a linear
        model spends more on dispatch than the inference costs. Left at the
        default, prediction runs serially and dispatches nothing.
    time_weighter : instance of `BaseWeighter` or None, default=None
        Weighter producing per-timestamp weights for the target time
        axis. Converted to sklearn ``sample_weight`` for training using
        ``sample_weight_alignment``.
    vintage_weighter : instance of `BaseWeighter` or None, default=None
        Weighter keyed on the observation timestamp of each training sample
        (the timestamp from which that forecast window was generated).
        Combined multiplicatively with ``time_weighter``. No alignment
        strategy is applied.
    sample_weight_alignment : {"first_step", "mean_step", \
"weighted_mean_step", "max_weight_step", "min_weight_step"}, \
default="first_step"
        How to collapse the per-step ``time_weighter`` weights of a
        sample's forecast window into a single ``sample_weight``.
        ``"first_step"`` uses the one-step-ahead weight; ``"mean_step"``,
        ``"max_weight_step"``, and ``"min_weight_step"`` aggregate across
        the horizon; ``"weighted_mean_step"`` uses an exponentially
        decaying weighting of the horizon steps.

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
    during prediction, unless ``X_forecast`` was provided at fit time, in
    which case a ``ValueError`` is raised; use
    [ForecastedFeatureForecaster][yohou.compose.ForecastedFeatureForecaster]
    for that case.

    Validation holdout (``validation_size``, ``validation_overlap``):

    These two parameters are declared by the families that expose them
    ([`PointReductionForecaster`][yohou.point.reduction.PointReductionForecaster],
    [`ClassProbaReductionForecaster`][yohou.class_proba.reduction.ClassProbaReductionForecaster],
    and
    [`IntervalReductionForecaster`][yohou.interval.reduction.IntervalReductionForecaster]);
    this class implements the machinery they drive.

    Early stopping itself (rounds, metric, callbacks) is configured on the
    estimator, never by yohou. Because those libraries do not refit after
    stopping, the tail's information is spent on the stopping decision; to
    train on everything, refit with ``validation_size=None`` and the
    discovered iteration count. Inside a search CV, each fold's inner fit
    holds out the tail of its own training window, shrinking effective
    training data accordingly.

    A ``Pipeline`` estimator is fitted in two phases: its transformer steps
    are fitted on the training rows, the evaluation features are pushed
    through them, and its final step is fitted directly with an ``eval_set``
    in the same transformed space it trains in. ``estimator_`` remains a
    fitted ``Pipeline``.

    Fitting raises ``ValueError`` when:

    - the estimator's ``fit`` (or, for a ``Pipeline``, its final step's fit)
      accepts neither ``eval_set`` nor ``**kwargs``, or the pipeline ends in
      ``"passthrough"``;
    - the estimator is a ``sklearn.multioutput`` wrapper (a multi-column
      ``eval_set`` target cannot be routed per sub-estimator);
    - the head left after the split cannot build one training row;
    - ``validation_size`` is smaller than ``forecasting_horizon`` while
      ``validation_overlap=False``;
    - a class_proba target class appears only inside the tail;
    - a raw ``eval_set`` is also passed through fit ``**params``;
    - the transformed head is too short to anchor the evaluation window
      (a transformer consumed the boundary rows as warmup).

    See Also
    --------
    - [`PointReductionForecaster`][yohou.point.reduction.PointReductionForecaster] : Point forecaster using reduction.
    - [`IntervalReductionForecaster`][yohou.interval.reduction.IntervalReductionForecaster] : Interval forecaster using reduction.
    """

    # Declared, not assigned: each subclass fits its own shape (one estimator, a list of
    # H, or a dict keyed by quantile), but the multi-origin paths below read it from the
    # base. A bare annotation leaves `check_is_fitted` seeing an unfitted instance.
    estimator_: Any

    # Declared per family, never here: BaseForecaster merges
    # _parameter_constraints across the MRO and a subclass cannot remove an
    # inherited key.
    validation_size: int | None
    validation_overlap: bool

    _parameter_constraints: dict = {
        **BaseForecaster._parameter_constraints,
        "estimator": [HasMethods(["fit", "predict"])],
        "reduction_strategy": [StrOptions({"direct", "dir-rec", "multi-output"})],
        "step_feature_alignment": [StrOptions({"all", "matched", "cumulative"})],
        "training_stride": [Interval(numbers.Integral, 1, None, closed="left")],
        # validation_size/validation_overlap are declared by the families that
        # expose them (point, class_proba). Declaring them here would leak them
        # into every subclass through the MRO merge in
        # BaseForecaster.__init_subclass__, which cannot be undone downstream.
        "nan_handling": [StrOptions({"drop", "pass"})],
        "n_jobs": [Interval(numbers.Integral, -1, None, closed="left"), None],
        "time_weighter": [BaseWeighter, None],
        "vintage_weighter": [BaseWeighter, None],
        "sample_weight_alignment": [
            StrOptions({
                "first_step",
                "mean_step",
                "weighted_mean_step",
                "max_weight_step",
                "min_weight_step",
            })
        ],
    }

    def __init__(
        self,
        estimator: BaseEstimator = LinearRegression(),
        *,
        reduction_strategy: Literal["direct", "dir-rec", "multi-output"] = "multi-output",
        target_transformer: BaseActualTransformer | None = None,
        actual_transformer: BaseActualTransformer | None = None,
        forecast_transformer: BaseForecastTransformer | None = None,
        step_transformer: BaseStepTransformer | None = None,
        target_as_feature: Literal["transformed", "raw"] | None = "transformed",
        panel_strategy: Literal["global", "multivariate"] = "global",
        step_feature_alignment: Literal["all", "matched", "cumulative"] = "all",
        training_stride: int = 1,
        nan_handling: Literal["drop", "pass"] = "pass",
        n_jobs: int | None = None,
        time_weighter: BaseWeighter | None = None,
        vintage_weighter: BaseWeighter | None = None,
        sample_weight_alignment: str = "first_step",
    ):
        BaseForecaster.__init__(
            self,
            target_as_feature=target_as_feature,
            target_transformer=target_transformer,
            actual_transformer=actual_transformer,
            forecast_transformer=forecast_transformer,
            step_transformer=step_transformer,
            panel_strategy=panel_strategy,
        )

        self.estimator = estimator
        self.reduction_strategy = reduction_strategy
        self.step_feature_alignment = step_feature_alignment
        self.training_stride = training_stride
        # validation_size/validation_overlap are assigned by the families that
        # expose them (point, class_proba), so a family that does not accept
        # them carries no attribute sklearn's get_params cannot see.
        self.nan_handling = nan_handling
        self.n_jobs = n_jobs
        self.time_weighter = time_weighter
        self.vintage_weighter = vintage_weighter
        self.sample_weight_alignment = sample_weight_alignment

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

    def _warn_inapplicable_step_alignment(self) -> None:
        """Warn when the chosen strategy will not apply ``step_feature_alignment``.

        Only ``"direct"`` filters step columns per estimator, so a non-default
        value on any other strategy is inert. Silently accepting it reads as
        configured behaviour that never happens.

        Warns rather than raises: an inapplicable value is a no-op, not an
        error, so a search over ``reduction_strategy`` crossed with
        ``step_feature_alignment`` explores a smaller effective grid instead of
        failing on every non-direct cell.

        Call once per ``fit``, after parameter validation.
        """
        if self.step_feature_alignment == "all" or self.reduction_strategy == "direct":
            return

        if self.reduction_strategy == "multi-output":
            reason = (
                "a single estimator predicts every horizon from one feature vector, "
                "so it reads a different step column per output and needs them all"
            )
        else:
            reason = (
                "dir-rec augments its feature matrix with earlier steps' predictions "
                "and is excluded from step filtering by design, not by structure"
            )

        warnings.warn(
            f"step_feature_alignment={self.step_feature_alignment!r} has no effect with "
            f"reduction_strategy={self.reduction_strategy!r}, because {reason}. Only "
            f'reduction_strategy="direct" applies step_feature_alignment; this fit '
            f'proceeds as though it were "all".',
            UserWarning,
            # warn <- _warn_inapplicable_step_alignment <- fit <- sklearn's
            # _fit_context wrapper <- user.
            stacklevel=4,
        )

    def _process_fit_weights(
        self,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        forecasting_horizon: int,
    ) -> np.ndarray | None:
        """Convert ``time_weighter``/``vintage_weighter`` to sklearn sample_weight.

        Reads the constructor-resident ``self.time_weighter``,
        ``self.vintage_weighter``, and ``self.sample_weight_alignment``.

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target time series (global or panel data).
        forecasting_horizon : int
            Number of forecast steps (determines tabularization window).

        Returns
        -------
        np.ndarray or None
            Sample weights array matching tabularized data rows, or None if
            both ``time_weighter`` and ``vintage_weighter`` are None.

        """
        if self.time_weighter is None and self.vintage_weighter is None:
            return None

        if self.groups_ is None:
            # Global data: y_t is DataFrame
            assert isinstance(y_t, pl.DataFrame)
            sample_weights = self._compute_sample_weights_one(
                y_t=y_t,
                forecasting_horizon=forecasting_horizon,
                group_name=None,
            )
        else:
            # Panel data: y_t is dict, stack weights
            assert isinstance(y_t, dict)
            sample_weights_list = []
            for panel_group_name in self.groups_:
                y_t_local = y_t[panel_group_name]
                weights_local = self._compute_sample_weights_one(
                    y_t=y_t_local,
                    forecasting_horizon=forecasting_horizon,
                    group_name=panel_group_name,
                )
                sample_weights_list.append(weights_local)
            sample_weights = np.concatenate(sample_weights_list)

        return sample_weights

    def _compute_sample_weights_one(
        self,
        y_t: pl.DataFrame,
        forecasting_horizon: int,
        group_name: str | None,
    ) -> np.ndarray:
        """Compute sample weights for one time series (global or local).

        Resolves ``self.time_weighter`` (with alignment strategy) and
        ``self.vintage_weighter`` (direct lookup), combines them
        multiplicatively, and normalizes so ``sum = n_samples``.

        Parameters
        ----------
        y_t : pl.DataFrame
            Transformed target time series with "time" column.
        forecasting_horizon : int
            Number of forecast steps.
        group_name : str or None
            Panel group name (for panel-aware weighters), or None for global data.

        Returns
        -------
        np.ndarray
            Per-sample weights aligned to the tabularized training rows.

        """
        n_samples = len(y_t) - forecasting_horizon
        time_series = y_t["time"]
        sample_weight_alignment = self.sample_weight_alignment

        # Resolve time_weighter with alignment strategy
        tw_aligned = None
        if self.time_weighter is not None:
            weights_array = _resolve_weighter_to_array(
                self.time_weighter,
                time_series,
                group_name=group_name,
                name="time weight",
            )

            if sample_weight_alignment == "first_step":
                aligned_indices = np.arange(1, n_samples + 1)
                tw_aligned = weights_array[aligned_indices]
            else:
                # Sliding windows over steps 1..H for each of the n_samples rows.
                windows = np.lib.stride_tricks.sliding_window_view(
                    weights_array[1 : n_samples + forecasting_horizon],
                    window_shape=forecasting_horizon,
                )[:n_samples]
                if sample_weight_alignment == "mean_step":
                    tw_aligned = windows.mean(axis=1)
                elif sample_weight_alignment == "weighted_mean_step":
                    horizon_decay = np.exp(-np.arange(forecasting_horizon) * 0.5)
                    horizon_decay = horizon_decay / horizon_decay.sum()
                    tw_aligned = windows @ horizon_decay
                elif sample_weight_alignment == "max_weight_step":
                    tw_aligned = windows.max(axis=1)
                else:
                    # min_weight_step (only remaining StrOptions value)
                    tw_aligned = windows.min(axis=1)

        # Resolve vintage_weighter via direct lookup (no alignment)
        vw_aligned = None
        if self.vintage_weighter is not None:
            vw_array = _resolve_weighter_to_array(
                self.vintage_weighter,
                time_series,
                group_name=group_name,
                name="vintage weight",
            )
            # Direct lookup: sample i's vintage is time_series[i]
            vw_aligned = vw_array[:n_samples]

        # Combine and normalize. At least one weighter is set (the public
        # caller short-circuits when both are None), so this is never None.
        sample_weights = _combine_weight_vectors(tw_aligned, vw_aligned, n=n_samples)
        assert sample_weights is not None
        return sample_weights

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
        - General mapping: lag_{H - k} becomes step_k for k in 1..H.
        - For forecasting_horizon=3:
            - lag_2 → step_1 (1-step-ahead target)
            - lag_1 → step_2 (2-step-ahead target)
            - lag_0 → step_3 (3-step-ahead target)
            - lag_3 (the oldest lag) maps to step_0 and is not selected as a
              target

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
        estimator_params: dict[str, Any] | None = None,
        estimator_fit_params: dict[str, Any] | None = None,
        eval_data: tuple[pl.DataFrame, pl.DataFrame] | None = None,
    ) -> BaseEstimator | list[BaseEstimator]:
        """Dispatch estimator fitting to the strategy-specific method.

        Routes to `_estimator_fit_multi_output`, `_estimator_fit_direct`,
        or `_estimator_fit_dir_rec` based on ``self.reduction_strategy``.
        Sample weighting is read from the constructor-resident
        ``self.time_weighter`` / ``self.vintage_weighter``.

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target time series.
        X_t : pl.DataFrame or dict[str, pl.DataFrame] or None
            Transformed feature matrix.
        forecasting_horizon : int
            Number of steps to forecast.
        estimator_params : dict or None
            Additional parameters to pass to the estimator's set_params method.
        estimator_fit_params : dict or None
            Additional parameters to pass to the estimator's fit method.
        eval_data : tuple or None
            The stacked validation-holdout ``(X_tab_eval, y_tab_eval)``
            pair, or None when ``validation_size`` is unset.

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
                estimator_params=estimator_params,
                estimator_fit_params=estimator_fit_params,
                eval_data=eval_data,
            )
        if self.reduction_strategy == "dir-rec":
            return self._estimator_fit_dir_rec(
                y_t,
                X_t,
                forecasting_horizon,
                estimator_params=estimator_params,
                estimator_fit_params=estimator_fit_params,
                eval_data=eval_data,
            )
        return self._estimator_fit_multi_output(
            y_t,
            X_t,
            forecasting_horizon,
            estimator_params=estimator_params,
            estimator_fit_params=estimator_fit_params,
            eval_data=eval_data,
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
        if X_t is None:
            raise ValueError(
                "Cannot tabularize: no feature matrix is available. Set "
                "target_as_feature to include the target as a feature, or "
                "provide exogenous features."
            )

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
        forecasting_horizon: int,
    ) -> np.ndarray | None:
        """Validate training data and compute sample weights.

        Reads constructor-resident weighters via :meth:`_process_fit_weights`.

        Parameters
        ----------
        X_tab : pl.DataFrame
            Feature matrix.
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target (for weight computation).
        forecasting_horizon : int
            Number of forecast steps.

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
                "the actual transformer reduces the data size below the minimum "
                "required for the forecasting horizon. Please check your "
                "transformer settings and ensure sufficient data length."
            )
        return self._process_fit_weights(
            y_t=y_t,
            forecasting_horizon=forecasting_horizon,
        )

    def _training_stride_mask(
        self,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        forecasting_horizon: int,
    ) -> np.ndarray | None:
        """Build the tail-anchored keep mask over tabularized instances.

        Instance ``i`` of a series with ``n`` instances is kept when
        ``i % k == (n - 1) % k``, so the most recent instance is always kept
        and kept origins sit ``k`` rows apart counting back from it. Tail
        anchoring is the point: the data tail is what upstream preparation
        aligns to the production origin, while the head depends on the
        configured training window and carries no anchor.

        Returns ``None`` when ``training_stride == 1`` so callers skip the
        filter entirely. On panel data one mask is built per group and
        concatenated in ``groups_`` order, matching how
        ``_get_stacked_tabularized_data`` and ``_process_fit_weights`` stack
        frames and weights.
        """
        if self.training_stride == 1:
            return None
        k = self.training_stride

        def one(n_instances: int) -> np.ndarray:
            """Tail-anchored keep mask for one series of ``n_instances`` rows."""
            return np.arange(n_instances) % k == (n_instances - 1) % k

        if self.groups_ is None:
            assert isinstance(y_t, pl.DataFrame)
            return one(len(y_t) - forecasting_horizon)
        assert isinstance(y_t, dict)
        return np.concatenate([one(len(y_t[g]) - forecasting_horizon) for g in self.groups_])

    def _apply_training_stride(
        self,
        X_tab: pl.DataFrame,
        y_tab: pl.DataFrame,
        sample_weight: np.ndarray | None,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        forecasting_horizon: int,
    ) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray | None]:
        """Filter tabularized data to strided instances, weights in lockstep.

        Runs once per estimator-fit call, before any ``nan_handling``, so the
        NaN-drop statistics reflect only kept instances and every reduction
        strategy sees the same strided dataset.
        """
        mask = self._training_stride_mask(y_t, forecasting_horizon)
        if mask is None:
            return X_tab, y_tab, sample_weight

        keep = pl.Series(mask)
        X_tab = X_tab.filter(keep)
        y_tab = y_tab.filter(keep)
        if sample_weight is not None:
            sample_weight = sample_weight[mask]

        if len(X_tab) == 0:
            # Unreachable when the pre-stride dataset is non-empty: the mask is
            # tail-anchored, so the last instance is always kept. Guarded anyway
            # so a future mask change cannot fail downstream in silence.
            raise ValueError(
                f"Training dataset is empty (0 samples) after applying "
                f"training_stride={self.training_stride}. Check that the input "
                f"series is long enough for the forecasting horizon and the stride."
            )
        return X_tab, y_tab, sample_weight

    def _apply_nan_handling(
        self,
        X_tab: pl.DataFrame,
        y_tab: pl.DataFrame | pl.Series,
        sample_weight: np.ndarray | None,
        *,
        context: str = "",
        is_validation: bool = False,
    ) -> tuple[pl.DataFrame, pl.DataFrame | pl.Series, np.ndarray | None]:
        """Remove rows containing NaN/null from a tabularized dataset.

        Applies to training rows and, via ``is_validation``, to the
        validation holdout's evaluation rows.

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
        is_validation : bool, default=False
            Whether these rows are the validation holdout's evaluation rows
            rather than training rows. Evaluation rows are never thinned by
            ``training_stride``, so messages about them name them as
            validation rows and omit the stride hint.

        Returns
        -------
        X_tab : pl.DataFrame
            Filtered feature matrix.
        y_tab : pl.DataFrame or pl.Series
            Filtered target.
        sample_weight : np.ndarray or None
            Filtered sample weights.

        Raises
        ------
        ValueError
            If ``nan_handling="drop"`` and every instance contains NaN,
            leaving 0 samples remaining.

        """
        if self.nan_handling == "pass":
            return X_tab, y_tab, sample_weight

        x_ok = self._compute_x_ok_mask(X_tab)

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
            noun = "validation" if is_validation else "training"
            if n_dropped == n_total:
                # training_stride never thins evaluation rows, so the hint
                # would misattribute their count on the validation path.
                stride_hint = (
                    f" The {n_total} instances are the ones kept by "
                    f"training_stride={self.training_stride}; a coarser stride "
                    f"leaves fewer instances to survive NaN handling."
                    if self.training_stride > 1 and not is_validation
                    else ""
                )
                raise ValueError(
                    f"All {n_total} {noun} instances contain NaN{context}. "
                    f"Cannot fit with nan_handling='drop' and 0 samples remaining."
                    f"{stride_hint}"
                )
            pct = 100 * n_dropped / n_total
            warnings.warn(
                f"NaN handling dropped {n_dropped} of {n_total} {noun} instances ({pct:.1f}%){context}.",
                stacklevel=3,
            )
            X_tab = X_tab.filter(mask)
            y_tab = y_tab.filter(mask)
            if sample_weight is not None:
                sample_weight = sample_weight[mask.to_numpy()]

        return X_tab, y_tab, sample_weight

    @staticmethod
    def _compute_x_ok_mask(X_tab: pl.DataFrame) -> pl.Series:
        """Compute a per-row mask of feature rows free of nulls and NaNs.

        For nulls, every column is checked. For NaNs, only float columns
        are checked. The returned boolean Series is ``True`` for rows that
        are safe to feed to the estimator.
        """
        null_free = X_tab.select(pl.all_horizontal(pl.all().is_not_null())).to_series()
        float_cols = X_tab.select(cs.float())
        if float_cols.width > 0:
            nan_free = float_cols.select(pl.all_horizontal(pl.all().is_not_nan())).to_series()
            return null_free & nan_free
        return null_free

    def _features_have_nan(self, X_tab: pl.DataFrame) -> bool:
        """Check if a feature DataFrame contains any NaN or null values.

        Only used when ``nan_handling="drop"`` to decide whether the
        estimator can be called safely at predict time.
        """
        return not self._compute_x_ok_mask(X_tab).all()

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
            for step_name, step in estimator.steps[:-1]:
                if step != "passthrough":
                    try:
                        step.set_fit_request(sample_weight=False)
                    except (TypeError, AttributeError) as exc:
                        warnings.warn(
                            f"Could not disable sample_weight routing on Pipeline step "
                            f"{step_name!r} ({step.__class__.__name__}): {exc}. This step "
                            f"may receive sample_weight unexpectedly.",
                            stacklevel=2,
                        )
            return {"sample_weight": sample_weight}

        # 3. VAR_KEYWORD fallback (**kwargs / **fit_params)
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in fit_sig.parameters.values())
        if has_var_keyword:
            return {"sample_weight": sample_weight}

        raise ValueError(
            f"Estimator {estimator.__class__.__name__} does not support "
            f"sample_weight parameter. Cannot use time_weight/vintage_weight for training."
        )

    @staticmethod
    def _eval_set_target(estimator: BaseEstimator) -> tuple[BaseEstimator, str]:
        """Return the estimator that receives ``eval_set``, and how to name it.

        For a ``Pipeline`` this is the final step: `_fit_pipeline_with_eval_set`
        fits that step directly, because sklearn hands fit parameters to steps
        untransformed and, under metadata routing, rejects the
        ``<step>__<param>`` form outright.

        Parameters
        ----------
        estimator : BaseEstimator
            The estimator ``validation_size`` will deliver an evaluation set to.

        Returns
        -------
        target : BaseEstimator
            The estimator whose ``fit`` receives ``eval_set``.
        label : str
            Prefix naming that estimator in error messages.

        Raises
        ------
        ValueError
            If the estimator is a ``Pipeline`` whose final step is
            ``"passthrough"``, which cannot be fitted at all.

        """
        if isinstance(estimator, Pipeline):
            final = estimator.steps[-1][1]
            if isinstance(final, str):
                raise ValueError(
                    "validation_size cannot be used with a Pipeline whose final step "
                    "is 'passthrough': there is no estimator to deliver an eval_set to. "
                    "End the pipeline with an estimator, or leave validation_size=None."
                )
            return final, "Pipeline's final step "
        return estimator, ""

    @staticmethod
    def _check_eval_set_support(estimator: BaseEstimator) -> None:
        """Check that the estimator's fit can accept an ``eval_set`` argument.

        For a ``Pipeline``, the check applies to its final step, the only step
        that is given an evaluation set; ``**kwargs`` on ``Pipeline.fit``
        itself does not count as support.

        Parameters
        ----------
        estimator : BaseEstimator
            The estimator ``validation_size`` will deliver an evaluation set to.

        Raises
        ------
        ValueError
            If the estimator (or, for a ``Pipeline``, its final step) is a
            ``sklearn.multioutput`` wrapper (a multi-column evaluation target
            cannot be routed per sub-estimator), its fit signature has neither
            an ``eval_set`` parameter nor ``**kwargs``, or it is a ``Pipeline``
            ending in ``"passthrough"``.

        """
        target, label = BaseReductionForecaster._eval_set_target(estimator)

        if isinstance(target, MultiOutputRegressor | MultiOutputClassifier):
            raise ValueError(
                f"validation_size cannot be used with {label}{target.__class__.__name__}: "
                f"the wrapper fits one sub-estimator per target column and cannot "
                f"route a multi-column eval_set target per sub-estimator. Use an "
                f"estimator with native multi-output support, or the 'direct' "
                f"reduction strategy with a single target."
            )

        fit_sig = inspect.signature(target.fit)  # ty: ignore[unresolved-attribute]
        if "eval_set" in fit_sig.parameters:
            return
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in fit_sig.parameters.values()):
            return
        raise ValueError(
            f"{label or 'Estimator '}{target.__class__.__name__} does not support an eval_set "
            f"fit parameter, so validation_size cannot deliver an evaluation set to it. "
            f"Use an estimator whose fit accepts eval_set (e.g. LightGBM, XGBoost, "
            f"CatBoost), or leave validation_size=None."
        )

    def _fit_pipeline_with_eval_set(
        self,
        estimator: Pipeline,
        X_tab: pl.DataFrame,
        y_tab: pl.DataFrame | pl.Series,
        sample_weight: np.ndarray | None,
        fit_params: dict[str, Any],
        eval_data: tuple[pl.DataFrame, pl.DataFrame | pl.Series],
    ) -> Pipeline:
        """Fit a ``Pipeline`` in two phases so its final step evaluates in its own space.

        ``Pipeline.fit`` transforms the training data on its way through the
        steps but hands fit parameters to steps untouched, so an evaluation set
        passed that way would be scored in raw space against a model trained in
        transformed space. Under metadata routing (which yohou enables) the
        ``<step>__<param>`` form is rejected outright. So the transformer prefix
        is fitted on the training rows here, the evaluation features are pushed
        through it, and the final step is fitted directly with both in the same
        space. Fitting the prefix on the training rows only is also what keeps
        the holdout leak-free.

        Parameters
        ----------
        estimator : Pipeline
            The (cloned) pipeline about to be fitted.
        X_tab : pl.DataFrame
            Training feature matrix.
        y_tab : pl.DataFrame or pl.Series
            Training target.
        sample_weight : np.ndarray or None
            Training sample weights, delivered straight to the final step
            because this path bypasses ``Pipeline.fit`` and its routing.
        fit_params : dict
            Additional fit parameters for the final step.
        eval_data : tuple
            The ``(X_eval, y_eval)`` pair, in the pipeline's input space.

        Returns
        -------
        Pipeline
            The fitted pipeline, reassembled from the fitted prefix and final
            step so it still reports as fitted and predicts.

        Raises
        ------
        ValueError
            If the final step cannot accept ``eval_set`` (or ``sample_weight``
            when weights apply).

        """
        # eval_set support (final step, passthrough, wrapper) was checked by
        # `_prepare_validation_fit` before any state changed; eval_data is only
        # ever non-None downstream of it.
        final_name, final_estimator = estimator.steps[-1]
        prefix_steps = estimator.steps[:-1]
        X_eval, y_eval = eval_data

        if prefix_steps:
            prefix = Pipeline(prefix_steps, memory=estimator.memory, verbose=estimator.verbose)
            X_train_t = prefix.fit_transform(X_tab, y_tab)
            X_eval_t = prefix.transform(X_eval)
            prefix_steps = prefix.steps
        else:
            X_train_t, X_eval_t = X_tab, X_eval

        final_fit_params = {**fit_params, "eval_set": [(X_eval_t, y_eval)]}
        if sample_weight is not None:
            try:
                final_fit_params.update(self._resolve_sample_weight_params(final_estimator, sample_weight))
            except ValueError as exc:
                if isinstance(final_estimator, Pipeline):
                    raise
                # Same wording as the non-holdout Pipeline path, so the two
                # fit paths report the identical failure identically.
                raise ValueError(
                    f"Pipeline's final step {final_estimator.__class__.__name__} does not "
                    f"support sample_weight parameter. Cannot use time_weight/vintage_weight "
                    f"for training."
                ) from exc

        final_estimator.fit(X_train_t, y_tab, **final_fit_params)

        return Pipeline(
            [*prefix_steps, (final_name, final_estimator)],
            memory=estimator.memory,
            verbose=estimator.verbose,
        )

    def _validate_validation_split(self, y: pl.DataFrame, forecasting_horizon: int) -> None:
        """Validate ``validation_size`` against the data and horizon.

        Parameters
        ----------
        y : pl.DataFrame
            Full target time series (before the head/tail split).
        forecasting_horizon : int
            Number of steps to forecast.

        Raises
        ------
        ValueError
            If ``validation_size < forecasting_horizon`` in strict mode, or
            the head left after the split cannot build one training row.

        """
        n = self.validation_size
        assert n is not None
        if not self.validation_overlap and n < forecasting_horizon:
            raise ValueError(
                f"validation_size={n} is smaller than forecasting_horizon="
                f"{forecasting_horizon}: no evaluation row's target window fits "
                f"inside the holdout. Increase validation_size to at least "
                f"{forecasting_horizon}, or set validation_overlap=True to "
                f"evaluate boundary rows whose targets partially overlap the "
                f"training data."
            )
        head = y.height - n
        min_head = forecasting_horizon + 1
        if head < min_head:
            raise ValueError(
                f"validation_size={n} leaves {max(head, 0)} head rows out of "
                f"{y.height}, but at least {min_head} are needed to build one "
                f"training row at forecasting_horizon={forecasting_horizon}. "
                f"Reduce validation_size or provide more data."
            )

    def _split_validation_tail(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None,
    ) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame, pl.DataFrame | None]:
        """Split raw inputs into a training head and a validation tail.

        The tail is the last ``validation_size`` rows of ``y``;
        ``X_actual`` is split at the tail's first timestamp so both frames
        stay aligned.

        Parameters
        ----------
        y : pl.DataFrame
            Full target time series.
        X_actual : pl.DataFrame or None
            Full feature time series.

        Returns
        -------
        y_head, X_head, y_tail, X_tail
            The split frames; the X halves are None when ``X_actual`` is.

        """
        n = self.validation_size
        assert n is not None
        y_head, y_tail = y[:-n], y[-n:]
        X_head = X_tail = None
        if X_actual is not None:
            boundary = y_tail["time"][0]
            X_head = X_actual.filter(pl.col("time") < boundary)
            X_tail = X_actual.filter(pl.col("time") >= boundary)
        return y_head, X_head, y_tail, X_tail

    def _prepare_validation_fit(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None,
        forecasting_horizon: int,
        params: dict[str, Any],
    ) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame, pl.DataFrame | None]:
        """Run the fail-fast validation-holdout checks and split the raw inputs.

        Parameters
        ----------
        y : pl.DataFrame
            Full target time series.
        X_actual : pl.DataFrame or None
            Full feature time series.
        forecasting_horizon : int
            Number of steps to forecast.
        params : dict
            The fit ``**params``, checked for a conflicting raw ``eval_set``.

        Returns
        -------
        y_head, X_head, y_tail, X_tail
            The raw split, as returned by `_split_validation_tail`.

        Raises
        ------
        ValueError
            On any invalid validation-holdout configuration.

        """
        if "eval_set" in params:
            raise ValueError(
                "fit received a raw eval_set through **params while "
                "validation_size is set. The evaluation set is built internally "
                "from the held-out tail; remove the eval_set fit parameter or "
                "set validation_size=None."
            )
        self._check_eval_set_support(self.estimator)
        self._validate_validation_split(y, forecasting_horizon)
        return self._split_validation_tail(y, X_actual)

    def _observe_validation_tail(
        self,
        y_tail: pl.DataFrame,
        X_tail: pl.DataFrame | None,
        forecasting_horizon: int,
        X_future: pl.DataFrame | None,
        X_forecast: pl.DataFrame | None,
    ) -> tuple[
        pl.DataFrame | dict[str, pl.DataFrame],
        pl.DataFrame | dict[str, pl.DataFrame],
    ]:
        """Walk the validation tail through the observe path, capturing transforms.

        Delegates the state update to the same ``observe()`` machinery every
        other caller uses (`_observe_standard_capture` /
        `_observe_panel_capture`), then joins the step columns the evaluation
        matrix needs. This pass is the only observation of the tail: fit must
        not observe it again.

        Parameters
        ----------
        y_tail : pl.DataFrame
            The held-out raw target rows (encoded for class_proba).
        X_tail : pl.DataFrame or None
            The held-out raw feature rows.
        forecasting_horizon : int
            Number of steps to forecast.
        X_future : pl.DataFrame or None
            Known future features, as passed to fit.
        X_forecast : pl.DataFrame or None
            External forecasts, as passed to fit.

        Returns
        -------
        y_t_tail : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed tail target rows (per group on panel data).
        X_t_tail : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed tail feature rows including step-derived columns
            (per group on panel data).

        """
        # Derived before the observe pass, which updates the stored raws:
        # step columns for every tail row, resolved as-of each row's time with
        # the fitted forecast and step transformers applied.
        step_columns = None
        if self._step_column_names_:
            X_future_eff = X_future if X_future is not None else self._X_future_raw_
            X_forecast_eff = self._transform_X_forecast(X_forecast) if X_forecast is not None else self._X_forecast_t_
            step_columns = _derive_step_columns(
                X_future_eff,
                X_forecast_eff,
                y_tail["time"],
                forecasting_horizon,
                self.interval_,
                step_transform=self._transform_X_step,
            )

        if self.groups_ is None:
            y_t_tail, X_t_cap = self._observe_standard_capture(y_tail, X_tail, X_future, X_forecast)
            X_t_tail = self._join_tail_step_columns(X_t_cap, step_columns, y_t_tail)
            return y_t_tail, X_t_tail

        groups = self.groups_
        y_t_tails, X_t_caps = self._observe_panel_capture(y_tail, X_tail, groups, X_future, X_forecast)

        step_schema_per_group = getattr(self, "_step_schema_per_group_", None)
        X_t_tails: dict[str, pl.DataFrame] = {}
        for panel_group_name in groups:
            step_local = None
            if step_columns is not None and step_schema_per_group is not None:
                # Panel splits the wide step frame per group; mirrors
                # `_bulk_origin_features` rather than inventing a new rule.
                available = set(step_columns.columns)
                step_schema = {
                    k: v
                    for k, v in step_schema_per_group.items()
                    if k in available or f"{panel_group_name}__{k}" in available
                }
                step_local = get_group_df(step_columns, panel_group_name, step_schema)
            X_t_tails[panel_group_name] = self._join_tail_step_columns(
                X_t_caps[panel_group_name], step_local, y_t_tails[panel_group_name]
            )

        return y_t_tails, X_t_tails

    @staticmethod
    def _join_tail_step_columns(
        X_t: pl.DataFrame | None,
        step_columns: pl.DataFrame | None,
        y_t: pl.DataFrame,
    ) -> pl.DataFrame:
        """Attach derived step columns to the transformed tail features.

        Parameters
        ----------
        X_t : pl.DataFrame or None
            Transformed tail features, or None when the forecaster has no
            feature frame of its own.
        step_columns : pl.DataFrame or None
            Step columns derived for the tail rows, or None when the
            forecaster has none.
        y_t : pl.DataFrame
            Transformed tail target, used to select the step rows when it is
            the only source of times.

        Returns
        -------
        pl.DataFrame
            The tail feature matrix the evaluation set is built from.

        """
        if step_columns is None:
            assert X_t is not None
            return X_t
        if X_t is not None:
            return X_t.join(step_columns, on="time", how="left")
        return step_columns.join(y_t.select("time"), on="time", how="semi")

    def _build_validation_eval_set(
        self,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        X_t: pl.DataFrame | dict[str, pl.DataFrame] | None,
        y_t_tail: pl.DataFrame | dict[str, pl.DataFrame],
        X_t_tail: pl.DataFrame | dict[str, pl.DataFrame],
        forecasting_horizon: int,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Tabularize the boundary window into the evaluation pair.

        In strict mode (``validation_overlap=False``) the evaluation anchors
        are exactly those whose full target window lies in the tail:
        ``validation_size - forecasting_horizon + 1`` rows, starting from the
        last head row. With overlap, the ``forecasting_horizon - 1``
        boundary anchors whose targets straddle the split are added,
        yielding ``validation_size`` rows.

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed head target (from ``_pre_fit``).
        X_t : pl.DataFrame or dict[str, pl.DataFrame] or None
            Transformed head features (from ``_pre_fit``).
        y_t_tail : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed tail target (from `_observe_validation_tail`).
        X_t_tail : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed tail features (from `_observe_validation_tail`).
        forecasting_horizon : int
            Number of steps to forecast.

        Returns
        -------
        X_tab_eval : pl.DataFrame
            Evaluation feature matrix, stacked per group.
        y_tab_eval : pl.DataFrame
            Evaluation target matrix, stacked per group.

        """
        # A reduction forecaster always reports requires_exogenous=True, and
        # BaseForecaster rejects target_as_feature=None without X_actual for
        # those, so a fit that reaches here always has a feature matrix.
        # The transformed head height was validated by
        # `_build_validation_eval_data` before the tail was observed.
        assert X_t is not None
        head_rows = 1 if not self.validation_overlap else forecasting_horizon

        def one(
            y_t_head_local: pl.DataFrame,
            X_t_head_local: pl.DataFrame,
            y_t_tail_local: pl.DataFrame,
            X_t_tail_local: pl.DataFrame,
            y_columns: list[str] | None,
        ) -> tuple[pl.DataFrame, pl.DataFrame]:
            """Tabularize one group's boundary window into its eval pair."""
            window_y = pl.concat(
                [y_t_head_local[-head_rows:], y_t_tail_local.select(y_t_head_local.columns)],
                how="vertical",
            )
            window_X = pl.concat(
                [X_t_head_local[-head_rows:], X_t_tail_local.select(X_t_head_local.columns)],
                how="vertical",
            )
            return self._get_tabularized_dataset(window_y, window_X, forecasting_horizon, y_columns=y_columns)

        if self.groups_ is None:
            assert isinstance(y_t, pl.DataFrame)
            assert isinstance(X_t, pl.DataFrame)
            assert isinstance(y_t_tail, pl.DataFrame)
            assert isinstance(X_t_tail, pl.DataFrame)
            return one(y_t, X_t, y_t_tail, X_t_tail, y_columns=None)

        assert isinstance(y_t, dict)
        assert isinstance(X_t, dict)
        assert isinstance(y_t_tail, dict)
        assert isinstance(X_t_tail, dict)
        X_tab_list, y_tab_list = [], []
        for panel_group_name in self.groups_:
            y_columns = [c for c in y_t[panel_group_name].columns if c != "time"]
            X_tab_local, y_tab_local = one(
                y_t[panel_group_name],
                X_t[panel_group_name],
                y_t_tail[panel_group_name],
                X_t_tail[panel_group_name],
                y_columns=y_columns,
            )
            X_tab_list.append(X_tab_local)
            y_tab_list.append(y_tab_local)
        return pl.concat(X_tab_list), pl.concat(y_tab_list)

    def _build_validation_eval_data(
        self,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        X_t: pl.DataFrame | dict[str, pl.DataFrame] | None,
        y_tail: pl.DataFrame,
        X_tail: pl.DataFrame | None,
        forecasting_horizon: int,
        X_future: pl.DataFrame | None,
        X_forecast: pl.DataFrame | None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Observe the tail and build the evaluation pair, in one call.

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed head target (from ``_pre_fit``).
        X_t : pl.DataFrame or dict[str, pl.DataFrame] or None
            Transformed head features (from ``_pre_fit``).
        y_tail : pl.DataFrame
            The held-out raw target rows (encoded for class_proba).
        X_tail : pl.DataFrame or None
            The held-out raw feature rows.
        forecasting_horizon : int
            Number of steps to forecast.
        X_future : pl.DataFrame or None
            Known future features, as passed to fit.
        X_forecast : pl.DataFrame or None
            External forecasts, as passed to fit.

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame]
            The stacked ``(X_tab_eval, y_tab_eval)`` pair.

        Raises
        ------
        ValueError
            If the transformed head is too short to anchor the evaluation
            window (a transformer consumed the boundary rows as warmup);
            raised before any tail row is observed.

        """
        # Validate the transformed head BEFORE the tail is observed, so this
        # failure mode joins the others that raise before any observation
        # state changes (the tail observation is irreversible).
        head_rows = 1 if not self.validation_overlap else forecasting_horizon
        heads = [y_t] if self.groups_ is None else [y_t[g] for g in self.groups_]
        for head_frame in heads:
            assert isinstance(head_frame, pl.DataFrame)
            if head_frame.height < head_rows:
                raise ValueError(
                    f"Cannot build the validation evaluation set: the transformed "
                    f"head has {head_frame.height} rows but {head_rows} are "
                    f"needed to anchor the evaluation window. A transformer "
                    f"consumed the boundary rows as warmup; provide more data or "
                    f"reduce validation_size."
                )

        y_t_tail, X_t_tail = self._observe_validation_tail(y_tail, X_tail, forecasting_horizon, X_future, X_forecast)
        return self._build_validation_eval_set(y_t, X_t, y_t_tail, X_t_tail, forecasting_horizon)

    @staticmethod
    def _select_step_target(frame: pl.DataFrame, step_col_names: list[str]) -> pl.DataFrame | pl.Series:
        """Select one step's target columns, collapsing a single column to a series.

        Parameters
        ----------
        frame : pl.DataFrame
            Tabularized target matrix (training or evaluation).
        step_col_names : list of str
            The step's target column names.

        Returns
        -------
        pl.DataFrame or pl.Series
            The step target, as a series when it has a single column so it
            matches what single-output estimators expect.

        """
        step_target: pl.DataFrame | pl.Series = frame.select(step_col_names)
        if step_target.shape[1] == 1:
            step_target = step_target.to_series()
        return step_target

    @staticmethod
    def _augment_with_predictions(X: pl.DataFrame, estimator: BaseEstimator, step: int) -> pl.DataFrame:
        """Append one dir-rec step's predictions as ``__aug_{step}_*`` columns.

        Parameters
        ----------
        X : pl.DataFrame
            Feature matrix to augment (training or evaluation).
        estimator : BaseEstimator
            The step's fitted estimator.
        step : int
            Zero-based step index, used in the appended column names.

        Returns
        -------
        pl.DataFrame
            The matrix with the prediction columns appended.

        """
        preds = estimator.predict(X)  # ty: ignore[unresolved-attribute]
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return X.with_columns([pl.Series(f"__aug_{step}_{j}", preds[:, j]) for j in range(preds.shape[1])])

    def _fit_single_estimator(
        self,
        X_tab: pl.DataFrame,
        y_tab: pl.DataFrame | pl.Series,
        sample_weight: np.ndarray | None,
        estimator_params: dict[str, Any] | None = None,
        estimator_fit_params: dict[str, Any] | None = None,
        eval_data: tuple[pl.DataFrame, pl.DataFrame | pl.Series] | None = None,
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
        eval_data : tuple or None
            The ``(X_eval, y_eval)`` validation-holdout pair, delivered as
            the estimator's ``eval_set`` fit argument. Shaped exactly like
            ``(X_tab, y_tab)``. A ``Pipeline`` estimator is fitted in two
            phases so its final step evaluates in the transformed space; see
            `_fit_pipeline_with_eval_set`.

        Returns
        -------
        BaseEstimator
            Fitted estimator.

        """
        estimator = clone(self.estimator).set_params(**(estimator_params or {}))
        fit_params = estimator_fit_params or {}

        if eval_data is not None and isinstance(estimator, Pipeline):
            return self._fit_pipeline_with_eval_set(estimator, X_tab, y_tab, sample_weight, fit_params, eval_data)

        if sample_weight is not None:
            fit_params = {**fit_params, **self._resolve_sample_weight_params(estimator, sample_weight)}
        if eval_data is not None:
            fit_params = {**fit_params, "eval_set": [tuple(eval_data)]}

        estimator.fit(X_tab, y_tab, **fit_params)
        return estimator

    def _estimator_fit_multi_output(
        self,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        X_t: pl.DataFrame | dict[str, pl.DataFrame] | None,
        forecasting_horizon: StrictInt,
        estimator_params: dict[str, Any] | None = None,
        estimator_fit_params: dict[str, Any] | None = None,
        eval_data: tuple[pl.DataFrame, pl.DataFrame] | None = None,
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
        estimator_params : dict or None
            Additional parameters to pass to the estimator's set_params method.
        estimator_fit_params : dict or None
            Additional parameters to pass to the estimator's fit method.
        eval_data : tuple or None
            The stacked validation-holdout ``(X_tab_eval, y_tab_eval)``
            pair, delivered whole (full-width target) as ``eval_set``.

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
            forecasting_horizon,
        )
        X_tab, y_tab, sample_weight = self._apply_training_stride(X_tab, y_tab, sample_weight, y_t, forecasting_horizon)
        X_tab, y_tab, sample_weight = self._apply_nan_handling(X_tab, y_tab, sample_weight)
        eval_pair = None
        if eval_data is not None:
            X_eval, y_eval, _ = self._apply_nan_handling(eval_data[0], eval_data[1], None, is_validation=True)
            eval_pair = (X_eval, y_eval)
        return self._fit_single_estimator(
            X_tab,
            y_tab,
            sample_weight,
            estimator_params,
            estimator_fit_params,
            eval_data=eval_pair,
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

        ``_step_column_names_`` holds every column the step stage produced, which
        after a ``step_transformer`` includes horizon-agnostic summaries
        (``temp_step_mean``). Those carry no step index to align against and
        describe the whole block, so they must reach every per-step estimator;
        filtering applies only to members that actually carry a step index. The
        set cannot simply be narrowed at the source, because the observe-path
        column swap needs the full post-transform set to avoid leaving stale
        columns behind.

        Step columns are recognized through
        [`_is_step_column`][yohou.base.forecaster.BaseForecaster._is_step_column],
        which accepts both the panel-wide and the local spelling. ``X_tab`` is a
        stacked per-group matrix under ``panel_strategy="global"``, so it carries
        local names while ``_step_column_names_`` holds panel-wide ones; matching
        against the panel-wide set alone recognizes nothing there and silently
        degrades ``matched`` and ``cumulative`` to ``all``.

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

        Raises
        ------
        RuntimeError
            If a non-default alignment is requested and step columns were derived
            at fit, but no column of ``X_tab`` is recognized as one.

        """
        if self.step_feature_alignment == "all" or not self._step_column_names_:
            return X_tab

        step_cols_in_tab = [c for c in X_tab.columns if self._is_step_column(c)]
        if not step_cols_in_tab:
            # Unreachable by construction: the fit that derived step columns also
            # recorded both of their spellings, and X_tab is built from those same
            # frames. Raising rather than returning X_tab unchanged is the point.
            # The passthrough that used to stand here is what let a naming drift
            # between the recorded names and the tabular ones disable filtering in
            # silence, so every per-step model trained on every step's columns while
            # the configuration said otherwise. A wrong model that reports itself as
            # the right one is worse than a failed fit.
            raise RuntimeError(
                f"step_feature_alignment={self.step_feature_alignment!r} cannot be applied: "
                f"{len(self._step_column_names_)} step column(s) were derived at fit, but none "
                f"of the {X_tab.width} tabularized feature columns is recognized as one. This "
                f"is an internal naming mismatch between the recorded step column names and the "
                f"feature matrix; it is not caused by the data or by this parameter. Please "
                f"report it, with the panel_strategy and the exogenous inputs used."
            )

        # Only horizon-indexed columns are candidates. A step_transformer emits
        # whole-block summaries (temp_step_mean, wx_step_c0) describing every step
        # at once, with no index to align against, so they must reach every
        # per-step estimator. An empty result here is ordinary rather than the
        # naming drift the guard above reports: it means the slot reduced every
        # block away.
        indexed = [c for c in step_cols_in_tab if _is_step_indexed(c)]
        if not indexed:
            return X_tab

        # "matched" keeps only this step; "cumulative" keeps _step_1 .. _step_{step}.
        keep = {step} if self.step_feature_alignment == "matched" else set(range(1, step + 1))

        drop = [c for c in indexed if _step_index(c) not in keep]

        return X_tab.drop(drop) if drop else X_tab

    def _estimator_fit_direct(
        self,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        X_t: pl.DataFrame | dict[str, pl.DataFrame] | None,
        forecasting_horizon: StrictInt,
        estimator_params: dict[str, Any] | None = None,
        estimator_fit_params: dict[str, Any] | None = None,
        eval_data: tuple[pl.DataFrame, pl.DataFrame] | None = None,
    ) -> list[BaseEstimator]:
        """Fit H independent estimators, one per horizon step.

        Each model ``h`` predicts step ``h`` only. With a single target
        column this is single-output regression; with multiple target
        columns it is multi-output regression per step. The feature matrix
        is the same for all models; only the target column(s) differ.

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target time series.
        X_t : pl.DataFrame or dict[str, pl.DataFrame] or None
            Transformed feature matrix.
        forecasting_horizon : int
            Number of steps to forecast.
        estimator_params : dict or None
            Additional parameters to pass to each estimator's set_params.
        estimator_fit_params : dict or None
            Additional parameters to pass to each estimator's fit.
        eval_data : tuple or None
            The stacked validation-holdout pair; each step's estimator
            receives its own step's evaluation targets, with the same
            step-feature filtering applied to the evaluation features.

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
            forecasting_horizon,
        )
        X_tab, y_tab, sample_weight = self._apply_training_stride(X_tab, y_tab, sample_weight, y_t, forecasting_horizon)

        if self.groups_ is None:
            y_columns = list(self.local_y_t_schema_.keys())
        else:
            assert isinstance(y_t, dict)
            y_columns = [c for c in next(iter(y_t.values())).columns if c != "time"]

        def _fit_step(step: int) -> BaseEstimator:
            """Fit a single estimator for horizon step."""
            step_col_names = [f"{col}_step_{step + 1}" for col in y_columns]
            y_step = self._select_step_target(y_tab, step_col_names)
            X_tab_step = self._filter_step_features(X_tab, step + 1)
            X_tab_step, y_step, sw_step = self._apply_nan_handling(
                X_tab_step, y_step, sample_weight, context=f" (step {step + 1})"
            )
            eval_pair = None
            if eval_data is not None:
                y_eval_step = self._select_step_target(eval_data[1], step_col_names)
                X_eval_step = self._filter_step_features(eval_data[0], step + 1)
                X_eval_step, y_eval_step, _ = self._apply_nan_handling(
                    X_eval_step, y_eval_step, None, context=f" (step {step + 1})", is_validation=True
                )
                eval_pair = (X_eval_step, y_eval_step)
            return self._fit_single_estimator(
                X_tab_step,
                y_step,
                sw_step,
                estimator_params,
                estimator_fit_params,
                eval_data=eval_pair,
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
        estimator_params: dict[str, Any] | None = None,
        estimator_fit_params: dict[str, Any] | None = None,
        eval_data: tuple[pl.DataFrame, pl.DataFrame] | None = None,
    ) -> list[BaseEstimator]:
        """Fit H estimators sequentially with recursive feature augmentation.

        Model ``h`` predicts step ``h`` using the original features
        augmented with in-sample predictions from models
        ``1, 2, ..., h-1``. This combines the direct strategy's
        per-step specialization with recursive information flow.

        NaN handling is applied once on the full tabularized dataset before
        fitting begins (contrast with the ``"direct"`` strategy, which
        applies it per step).

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target time series.
        X_t : pl.DataFrame or dict[str, pl.DataFrame] or None
            Transformed feature matrix.
        forecasting_horizon : int
            Number of steps to forecast.
        estimator_params : dict or None
            Additional parameters to pass to each estimator's set_params.
        estimator_fit_params : dict or None
            Additional parameters to pass to each estimator's fit.
        eval_data : tuple or None
            The stacked validation-holdout pair; evaluation features are
            augmented with earlier-step predictions in lockstep with the
            training features.

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
            forecasting_horizon,
        )
        X_tab, y_tab, sample_weight = self._apply_training_stride(X_tab, y_tab, sample_weight, y_t, forecasting_horizon)
        X_tab, y_tab, sample_weight = self._apply_nan_handling(X_tab, y_tab, sample_weight)
        assert isinstance(y_tab, pl.DataFrame)

        self._dir_rec_n_original_features_ = X_tab.shape[1]

        if self.groups_ is None:
            y_columns = list(self.local_y_t_schema_.keys())
        else:
            assert isinstance(y_t, dict)
            y_columns = [c for c in next(iter(y_t.values())).columns if c != "time"]

        X_eval_aug: pl.DataFrame | None = None
        y_eval_tab: pl.DataFrame | None = None
        if eval_data is not None:
            X_eval, y_eval, _ = self._apply_nan_handling(eval_data[0], eval_data[1], None, is_validation=True)
            assert isinstance(y_eval, pl.DataFrame)
            X_eval_aug = X_eval.clone()
            y_eval_tab = y_eval

        estimators: list[BaseEstimator] = []
        X_aug = X_tab.clone()  # Progressively augmented feature matrix
        for step in range(forecasting_horizon):
            step_col_names = [f"{col}_step_{step + 1}" for col in y_columns]
            y_step = self._select_step_target(y_tab, step_col_names)
            eval_pair = None
            if X_eval_aug is not None:
                assert y_eval_tab is not None
                eval_pair = (X_eval_aug, self._select_step_target(y_eval_tab, step_col_names))
            est = self._fit_single_estimator(
                X_aug,
                y_step,
                sample_weight,
                estimator_params,
                estimator_fit_params,
                eval_data=eval_pair,
            )
            estimators.append(est)

            # Augment features with in-sample predictions for next step
            if step < forecasting_horizon - 1:
                X_aug = self._augment_with_predictions(X_aug, est, step)
                if X_eval_aug is not None:
                    # Evaluation rows carry the same feed-forward augmentation
                    # as training rows, so step k's eval features match its
                    # training features column for column.
                    X_eval_aug = self._augment_with_predictions(X_eval_aug, est, step)

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
        # `estimator` is constrained by duck typing (``HasMethods(["fit", "predict"])``), so a
        # sklearn-compatible estimator that does not subclass ``BaseEstimator`` is legal:
        # CatBoost and XGBoost's native APIs are both in that position. Only the
        # not-a-list narrowing matters here, matching the two branches above.
        assert not isinstance(estimator, list), (
            "the multi-output strategy fits a single estimator, but a list was stored; "
            "this is an internal inconsistency in the fitted state"
        )
        return self._estimator_predict_multi_output(typing_cast(BaseEstimator, estimator), groups)

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
        drop_nan = self.nan_handling == "drop"

        # One feature row per observation unit: the single row for non-panel data, or one
        # row per panel group stacked. Under panel_strategy="global" every group shares
        # `estimators[step]`, so a step is one call over the stacked rows rather than one
        # call per group. That is the same arithmetic with an order of magnitude fewer
        # estimator calls, and it removes the per-call validation overhead that dominates
        # cheap estimators.
        # `groups_` is populated only under ``panel_strategy="global"``: `_pre_fit` routes
        # ``"multivariate"`` to the standard path, which leaves it None. That is what makes
        # one call per step sound, because every group then shares `estimators[step]`. A
        # future panel strategy that populated `groups_` without that sharing would batch
        # rows belonging to different models and be wrong in silence, so pin the invariant
        # rather than leave it implicit.
        assert self.groups_ is None or self.panel_strategy == "global", (
            "batched prediction assumes every panel group shares the step's estimator, "
            f"which panel_strategy={self.panel_strategy!r} does not guarantee"
        )

        panel = self.groups_ is not None
        X_tab = (
            pl.concat([self._get_predict_features(g) for g in groups], how="vertical")
            if panel
            else self._get_predict_features()
        )

        # Step filtering depends only on column names, which are the local schema's and so
        # shared across groups: filter once per step, not once per group per step.
        step_frames = [self._filter_step_features(X_tab, step + 1) for step in range(len(estimators))]
        row_masks = [
            (self._compute_x_ok_mask(frame).to_numpy() if drop_nan else np.ones(frame.height, dtype=bool))
            for frame in step_frames
        ]

        # `n_jobs` pays off only when a step's inference costs more than dispatching it,
        # which depends on the estimator: a gradient-boosted model over a wide frame is
        # milliseconds and gains, a linear model is well under and loses. It defaults to
        # 1, so dispatch is opt-in.
        step_preds: list[np.ndarray] = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_direct_step)(est, frame, n_targets, mask)
            for est, frame, mask in zip(estimators, step_frames, row_masks, strict=True)
        )

        if not panel:
            y_pred = pl.DataFrame(np.vstack([p[0] for p in step_preds]), schema=y_cols)
            return cast(y_pred, self.local_y_t_schema_)

        y_pred_dict = {}
        for row, panel_group_name in enumerate(groups):
            y_pred_arr = np.vstack([step_pred[row] for step_pred in step_preds])
            y_pred_local = pl.DataFrame(y_pred_arr, schema=y_cols)
            y_pred_local = cast(y_pred_local, self.local_y_t_schema_)
            y_pred_local = y_pred_local.rename({col: f"{panel_group_name}__{col}" for col in y_cols})
            y_pred_dict[panel_group_name] = y_pred_local
        return pl.concat(list(y_pred_dict.values()), how="horizontal")

    def _bulk_origin_features(
        self,
        *,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None,
        groups: list[str],
        X_future: pl.DataFrame | None,
        X_forecast: pl.DataFrame | None,
    ) -> tuple[list[pl.DataFrame], list[Any], list[Any]]:
        """Transform every calibration origin's features in one pass per group.

        The rolling path observes one row at a time, and ``observe_transform``
        concatenates its buffer with the incoming rows, transforms the whole window,
        then keeps only the new part, so observing a single row still transforms the
        whole observation horizon behind it. Transforming the block once instead does
        that work for every origin at once.

        Sound only when every transformer reports ``batch_invariant``; the caller checks
        that. The first origin is not produced here: it precedes any calibration observe
        and takes its features from the fitted state.

        Parameters
        ----------
        y : pl.DataFrame
            The calibration block, one row per origin after the first.
        X_actual : pl.DataFrame or None
            Actual features aligned with ``y``.
        groups : list of str
            Panel group names, or empty for a standard forecaster.
        X_future : pl.DataFrame or None
            Known future features.
        X_forecast : pl.DataFrame or None
            External forecasts.

        Returns
        -------
        X_tab_per_origin : list of pl.DataFrame
            One feature frame per origin, each with one row per group.
        y_observed_per_origin : list
            The ``_y_observed`` each origin's inverse transform must see.
        observed_time_per_origin : list
            The ``observed_time_`` each origin's time columns must be built from.

        """
        panel = self.groups_ is not None
        n = y.height

        # Step columns are a pure function of the observation times, so they are derived
        # once for the whole block exactly as the rolling loop's precomputed branch does.
        step_columns = _derive_step_columns(
            X_future,
            self._transform_X_forecast(X_forecast),
            y["time"],
            self.fit_forecasting_horizon_,
            self.interval_,
        )

        assert self.local_X_t_schema_ is not None
        feature_order = list(self.local_X_t_schema_.keys())
        group_names = groups if panel else [None]

        per_group_rows: dict[Any, pl.DataFrame] = {}
        for group_name in group_names:
            if panel:
                y_local = get_group_df(df=y, group_name=group_name, schema=self.local_y_schema_)
                X_local = None
                if X_actual is not None and self.local_X_actual_schema_ is not None:
                    X_local = get_group_df(df=X_actual, group_name=group_name, schema=self._build_X_actual_schema())
                target_transformer = (
                    self.target_transformer_[group_name]
                    if isinstance(self.target_transformer_, dict)
                    else self.target_transformer_
                )
                actual_transformer = (
                    self.actual_transformer_[group_name]
                    if isinstance(self.actual_transformer_, dict)
                    else self.actual_transformer_
                )
            else:
                y_local, X_local = y, X_actual
                # Only panel fits keep a per-group dict, and this branch is the non-panel one.
                assert not isinstance(self.target_transformer_, dict)
                assert not isinstance(self.actual_transformer_, dict)
                target_transformer = self.target_transformer_
                actual_transformer = self.actual_transformer_

            X_t_local = _observe_transformers_one(
                y_local, X_local, target_transformer, actual_transformer, self.target_as_feature
            )

            if step_columns is not None:
                # Panel splits the wide step frame per group; standard takes it whole.
                # Mirrors `_observe_with_precomputed_steps_panel` and its standard
                # counterpart rather than inventing a third rule.
                step_schema_per_group = getattr(self, "_step_schema_per_group_", None)
                if panel and step_schema_per_group is not None:
                    available = set(step_columns.columns)
                    step_schema = {
                        k: v
                        for k, v in step_schema_per_group.items()
                        if k in available or f"{group_name}__{k}" in available
                    }
                    step_local = get_group_df(step_columns, group_name, step_schema).select(~cs.by_name("time"))
                elif not panel:
                    step_local = step_columns.select(~cs.by_name("time"))
                else:
                    step_local = None

                if step_local is not None:
                    X_t_local = (
                        pl.concat([X_t_local, step_local], how="horizontal") if X_t_local is not None else step_local
                    )

            assert X_t_local is not None
            per_group_rows[group_name] = X_t_local.select(feature_order)

        X_tab_per_origin = [
            pl.concat([per_group_rows[g][i : i + 1] for g in group_names], how="vertical") for i in range(n)
        ]

        # `_y_observed` and `observed_time_` are what the inverse transform and the time
        # columns read, and both advance with the origin. Reconstructed by slicing rather
        # than by rolling, which is the whole point of not rolling.
        observed_time_per_origin: list[Any] = []
        y_observed_per_origin: list[Any] = []
        horizon = self.observation_horizon
        for i in range(n):
            if panel:
                observed_time_per_origin.append(dict.fromkeys(groups, y["time"][i]))
            else:
                observed_time_per_origin.append(y["time"][i])

            if horizon <= 0:
                y_observed_per_origin.append(dict.fromkeys(groups) if panel else None)
                continue

            if panel:
                assert isinstance(self._y_observed, dict)
                per_origin: dict[str, pl.DataFrame | None] = {}
                for group_name in groups:
                    stored = self._y_observed[group_name]
                    local = get_group_df(df=y[: i + 1], group_name=group_name, schema=self.local_y_schema_)
                    combined = pl.concat([stored, local]) if stored is not None else local
                    per_origin[group_name] = combined[-horizon:]
                y_observed_per_origin.append(per_origin)
            else:
                stored = typing_cast(pl.DataFrame | None, self._y_observed)
                combined = pl.concat([stored, y[: i + 1]]) if stored is not None else y[: i + 1]
                y_observed_per_origin.append(combined[-horizon:])

        return X_tab_per_origin, y_observed_per_origin, observed_time_per_origin

    def _observe_predict_bulk_origins(
        self,
        *,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None,
        groups: list[str],
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        predict_transformed: bool = False,
    ) -> pl.DataFrame:
        """Replay every origin without rolling, transforming and predicting in bulk.

        Produces the same frame as ``observe_predict`` at ``stride=1`` for a frozen
        ``direct`` forecaster, within floating point reassociation. Two per origin costs
        disappear: the transformer chain runs once over the whole block rather than once
        per origin, and the estimator issues H calls rather than H per origin.

        Not bit identical, and deliberately so. Batching a rolling accumulator
        reassociates it, so rolling statistic columns can move by about one ULP while
        nothing else moves at all. Callers comparing the two paths must use a relative
        tolerance.

        Requires every transformer to report ``batch_invariant``; the caller checks that
        with `_chains_are_batch_invariant` and falls back to the rolling path otherwise.

        Parameters
        ----------
        y : pl.DataFrame
            The calibration block.
        X_actual : pl.DataFrame or None
            Actual features aligned with ``y``.
        groups : list of str
            Panel group names to operate on.
        X_future : pl.DataFrame or None, default=None
            Known future features.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts.
        predict_transformed : bool, default=False
            Return predictions in transformed space rather than original scale.

        Returns
        -------
        pl.DataFrame
            The concatenated per-origin predictions.

        Raises
        ------
        ValueError
            If the reduction strategy is not ``"direct"``.

        See Also
        --------
        `_bulk_origin_features` : Builds every origin's feature row in one pass.
        `_observe_predict_batched_origins` : The rolling-observe, batched-predict path.

        """
        if self.reduction_strategy != "direct":
            raise ValueError(
                f"bulk origin replay is implemented for reduction_strategy='direct', not {self.reduction_strategy!r}"
            )

        panel = self.groups_ is not None

        # The first origin precedes any calibration observe: its features are the fitted
        # state, not something the block produces.
        first_X_tab = (
            pl.concat([self._get_predict_features(g) for g in groups], how="vertical")
            if panel
            else self._get_predict_features()
        )
        first_y_observed = self._y_observed
        first_observed_time = self.observed_time_

        rest_X_tab, rest_y_observed, rest_observed_time = self._bulk_origin_features(
            y=y, X_actual=X_actual, groups=groups, X_future=X_future, X_forecast=X_forecast
        )

        X_tab_per_origin = [first_X_tab, *rest_X_tab]
        y_observed_per_origin = [first_y_observed, *rest_y_observed]
        observed_time_per_origin = [first_observed_time, *rest_observed_time]

        frames = self._estimator_predict_direct_multi(
            typing_cast(list[BaseEstimator], self.estimator_), groups, X_tab_per_origin
        )

        saved_y, saved_time = self._y_observed, self.observed_time_
        out: list[pl.DataFrame] = []
        try:
            for frame, y_observed, observed_time in zip(
                frames, y_observed_per_origin, observed_time_per_origin, strict=True
            ):
                self._y_observed = y_observed
                self.observed_time_ = observed_time
                y_t, y_inv = self._predict(groups=groups, y_pred_step=self._add_time_columns(frame))
                out.append(y_t if predict_transformed else y_inv)
        except BaseException:
            self._y_observed, self.observed_time_ = saved_y, saved_time
            raise

        # The rolling path observes its way through the block and ends having observed
        # all of it. This path reconstructs each origin by slicing and never advances
        # the buffer, so it has to land on that same end state deliberately: the last
        # origin's state, which is the one that has seen every row.
        #
        # Restoring the pre-replay state instead leaves the forecaster rewound by the
        # whole block, and the next observe stitches a stale window onto fresh
        # forecasts. The two then sit `len(y)` apart on the time axis, which surfaces
        # as an inconsistent-interval error from whichever transformer inverts first.
        self._y_observed = y_observed_per_origin[-1]
        self.observed_time_ = observed_time_per_origin[-1]

        return pl.concat(out)

    def _observe_predict_batched_origins(
        self,
        *,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None,
        groups: list[str],
        stride: int,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        predict_transformed: bool = False,
    ) -> pl.DataFrame:
        """Roll over origins, then predict all of them in one pass per horizon step.

        Equivalent to ``observe_predict`` for a frozen ``direct`` forecaster, and
        produces the same frame. The difference is where the estimator work happens: the
        loop records each origin's feature row instead of predicting, and a single
        batched pass afterwards issues H estimator calls over every origin's rows rather
        than H per origin.

        Only the estimator calls move. The per-origin inverse transform and frame
        assembly still run once per origin, through `_predict`, so nothing is
        reimplemented here and a target transformer whose inverse depends on per-origin
        ``_y_observed`` stays correct.

        Parameters
        ----------
        y : pl.DataFrame
            Target observations to roll through.
        X_actual : pl.DataFrame or None
            Actual features aligned with ``y``.
        groups : list of str
            Panel group names to operate on.
        stride : int
            Rows observed between successive origins.
        X_future : pl.DataFrame or None, default=None
            Known future features.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts.
        predict_transformed : bool, default=False
            Return predictions in transformed space rather than original scale.

        Returns
        -------
        pl.DataFrame
            The same concatenated per-origin predictions ``observe_predict`` returns.

        Raises
        ------
        ValueError
            If the reduction strategy is not ``"direct"``. The other strategies either
            hold one estimator for all steps or make each step depend on the previous,
            so neither has per-step calls to batch across origins.

        See Also
        --------
        `_estimator_predict_direct_multi` : The batched estimator pass.

        """
        if self.reduction_strategy != "direct":
            raise ValueError(
                "batched multi-origin prediction is implemented for "
                f"reduction_strategy='direct', not {self.reduction_strategy!r}"
            )

        panel = self.groups_ is not None

        def capture(groups: list[str], **_: Any) -> dict[str, Any]:
            """Record what this origin would have predicted from, and predict nothing.

            ``_y_observed`` and ``observed_time_`` are rebound to fresh objects by every
            observe rather than mutated in place, so holding the current reference keeps
            this origin's values without copying them.
            """
            X_tab = (
                pl.concat([self._get_predict_features(g) for g in groups], how="vertical")
                if panel
                else self._get_predict_features()
            )
            return {
                "X_tab": X_tab,
                "y_observed": self._y_observed,
                "observed_time": self.observed_time_,
            }

        def assemble(collected: list[dict[str, Any]]) -> pl.DataFrame:
            """Predict every origin in one batched estimator pass and inverse each.

            The estimator call is what batching buys: one pass per horizon step over all
            origins instead of one per origin. Everything after it stays per-origin,
            because the inverse transform and the time columns both read state that
            belongs to the origin the row came from.
            """
            frames = self._estimator_predict_direct_multi(
                typing_cast(list[BaseEstimator], self.estimator_),
                groups,
                [state["X_tab"] for state in collected],
            )
            saved_y, saved_time = self._y_observed, self.observed_time_
            out: list[pl.DataFrame] = []
            try:
                for state, frame in zip(collected, frames, strict=True):
                    # The inverse and the time columns are both origin-dependent, so the
                    # origin's own state is restored around each call rather than reusing
                    # whatever the loop left behind.
                    self._y_observed = state["y_observed"]
                    self.observed_time_ = state["observed_time"]
                    y_pred_step = self._add_time_columns(frame)
                    y_t, y_inv = self._predict(groups=groups, y_pred_step=y_pred_step)
                    out.append(y_t if predict_transformed else y_inv)
            finally:
                self._y_observed, self.observed_time_ = saved_y, saved_time
            return pl.concat(out)

        return self._observe_predict_loop(
            predict_fn=capture,
            y=y,
            X_actual=X_actual,
            X_future=X_future,
            X_forecast=X_forecast,
            groups=groups,
            stride=stride,
            reduce_fn=assemble,
        )

    def _estimator_predict_direct_multi(
        self,
        estimators: list[BaseEstimator],
        groups: list[str],
        X_tab_per_origin: list[pl.DataFrame],
    ) -> list[pl.DataFrame]:
        """Predict many origins with one estimator call per horizon step.

        The single-origin path issues H calls per origin, each over one row per panel
        group. When a caller holds many origins whose feature rows are already known and
        the fitted estimators are shared across them, those calls collapse: H calls over
        every origin's rows stacked produce the same numbers, because a step's estimator
        is applied row-wise.

        The saving is entirely per-call overhead, since the number of rows predicted is
        unchanged. A tree estimator's cost at these widths is dominated by fixed per
        call overhead rather than by the rows themselves, so folding many small calls
        into one large one is close to free.

        Parameters
        ----------
        estimators : list[BaseEstimator]
            H fitted estimators, one per horizon step.
        groups : list of str
            Panel group names, in the order the rows of each origin's frame follow.
        X_tab_per_origin : list of pl.DataFrame
            One feature frame per origin, each with one row per entry of ``groups``
            (or a single row when the forecaster is not fitted on panel data).

        Returns
        -------
        list of pl.DataFrame
            One prediction frame per origin, in the same shape and column order that
            `_estimator_predict_direct` returns for a single origin.

        See Also
        --------
        `_estimator_predict_direct` : The single-origin path this batches over.

        """
        assert self.local_y_t_schema_ is not None
        y_cols = list(self.local_y_t_schema_.keys())
        n_targets = len(y_cols)
        drop_nan = self.nan_handling == "drop"
        panel = self.groups_ is not None

        # Same invariant the single-origin path pins: batching rows across groups is
        # sound only because every group shares the step's estimator.
        assert self.groups_ is None or self.panel_strategy == "global", (
            "batched prediction assumes every panel group shares the step's estimator, "
            f"which panel_strategy={self.panel_strategy!r} does not guarantee"
        )

        n_origins = len(X_tab_per_origin)
        if n_origins == 0:
            return []
        rows_per_origin = X_tab_per_origin[0].height
        assert all(f.height == rows_per_origin for f in X_tab_per_origin), (
            "every origin must contribute the same number of feature rows for the "
            "stacked predictions to be sliced back apart by position"
        )

        X_all = pl.concat(X_tab_per_origin, how="vertical")

        # Step filtering depends only on column names, which are shared across origins
        # as well as groups, so it runs once per step rather than once per origin.
        step_preds: list[np.ndarray] = []
        for step, estimator in enumerate(estimators):
            frame = self._filter_step_features(X_all, step + 1)
            mask = self._compute_x_ok_mask(frame).to_numpy() if drop_nan else np.ones(frame.height, dtype=bool)
            step_preds.append(_predict_direct_step(estimator, frame, n_targets, mask))

        out: list[pl.DataFrame] = []
        for origin in range(n_origins):
            base = origin * rows_per_origin
            if not panel:
                y_pred = pl.DataFrame(np.vstack([p[base] for p in step_preds]), schema=y_cols)
                out.append(cast(y_pred, self.local_y_t_schema_))
                continue

            y_pred_dict = {}
            for row, panel_group_name in enumerate(groups):
                y_pred_arr = np.vstack([step_pred[base + row] for step_pred in step_preds])
                y_pred_local = pl.DataFrame(y_pred_arr, schema=y_cols)
                y_pred_local = cast(y_pred_local, self.local_y_t_schema_)
                y_pred_local = y_pred_local.rename({col: f"{panel_group_name}__{col}" for col in y_cols})
                y_pred_dict[panel_group_name] = y_pred_local
            out.append(pl.concat(list(y_pred_dict.values()), how="horizontal"))
        return out

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
        It routes fit-method metadata (e.g. custom fit params passed via
        ``**params``) to the wrapped sklearn estimator and to inherited
        transformers. Time and vintage weighting are resolved internally from
        the ``time_weighter`` / ``vintage_weighter`` constructor parameters,
        not via metadata routing.

        Returns
        -------
        router : MetadataRouter
            Router that forwards to transformers (from parent) and wrapped estimator.
        """
        # Get parent routing (for target_transformer, actual_transformer)
        router = super().get_metadata_routing()

        # Add wrapped sklearn estimator routing
        if hasattr(self, "estimator") and self.estimator is not None:
            router.add(
                estimator=self.estimator,
                method_mapping=MethodMapping().add(caller="fit", callee="fit"),
            )

        return router
