"""Implementation of reduction-based interval forecasters."""

from collections.abc import Callable
from typing import Literal

import numpy as np
import polars as pl
import polars.selectors as cs
from pydantic import StrictFloat, StrictInt
from sklearn.base import BaseEstimator
from sklearn.linear_model import QuantileRegressor
from sklearn.multioutput import MultiOutputRegressor

from yohou.base import BaseReductionForecaster, BaseTransformer
from yohou.utils._compat import HasMethods, StrOptions, _fit_context

from .base import BaseIntervalForecaster

__all__ = ["IntervalReductionForecaster"]


class IntervalReductionForecaster(BaseReductionForecaster, BaseIntervalForecaster):
    """Interval forecaster using sklearn estimators on tabularized time series.

    Converts the time series interval forecasting task to a tabular one.

    Parameters
    ----------
    estimator : BaseEstimator, default=MultiOutputRegressor(QuantileRegressor())
        Quantile estimator used to fit the tabularized data.
    reduction_strategy : {"direct", "dir-rec", "multi-output"}, default="multi-output"
        Strategy for multi-step forecasting.
    target_as_feature : {"transformed", "raw"} or None, default="transformed"
        Controls whether the target is included as a feature.
        ``"transformed"`` includes the transformed target, ``"raw"``
        includes the raw target, and ``None`` uses only exogenous features.
    feature_transformer : BaseTransformer or None, default=None
        Transformer used to transform the feature time series into features.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data. See `BaseForecaster` for details.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel for the ``"direct"`` strategy
        (fitting and predicting H independent models). ``None`` means 1
        unless in a ``joblib.parallel_backend`` context. ``-1`` means
        using all processors. Has no effect for ``"multi-output"`` or
        ``"dir-rec"`` strategies.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.interval import IntervalReductionForecaster
    >>>
    >>> # Create simple time series data
    >>> df = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         start=datetime(2021, 1, 1), end=datetime(2021, 1, 10), interval="1d", eager=True
    ...     ),
    ...     "value": [10.0, 12.0, 15.0, 14.0, 16.0, 18.0, 20.0, 19.0, 21.0, 23.0],
    ... })
    >>>
    >>> # Split into train/test
    >>> train = df[:8]
    >>>
    >>> # Create and fit interval forecaster
    >>> forecaster = IntervalReductionForecaster()
    >>> _ = forecaster.fit(y=train, forecasting_horizon=1, coverage_rates=[0.1, 0.5, 0.9])
    >>>
    >>> # Generate prediction intervals
    >>> y_pred = forecaster.predict_interval(forecasting_horizon=1, coverage_rates=[0.1, 0.5, 0.9])
    >>> len(y_pred)
    1
    >>> # Check that prediction has lower and upper bounds for each coverage rate
    >>> "value_lower_0.1" in y_pred.columns
    True
    >>> "value_upper_0.9" in y_pred.columns
    True

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

    For direct and dir-rec strategies, each value in ``estimator_``
    becomes a ``list[BaseEstimator]`` of length H (one per horizon
    step) instead of a single estimator.

    All strategies can be applied recursively for multi-step forecasting
    beyond the fit horizon by specifying a larger forecasting horizon
    during prediction.

    This forecaster uses quantile regression to produce prediction intervals.
    For each coverage rate alpha, it predicts:

    - Lower bound: (1 - alpha)/2 quantile
    - Upper bound: (1 + alpha)/2 quantile

    The intervals naturally adapt to heteroscedastic data where uncertainty
    varies over time.

    Multi-quantile estimators (e.g. CatBoost with ``MultiQuantile`` loss)
    are also supported.  When detected, a single model is fitted for all
    quantiles simultaneously, which can be significantly faster than the
    default approach of fitting separate lower/upper models per coverage
    rate.

    See Also
    --------
    `SplitConformalForecaster` : Conformal prediction intervals.
    `PointReductionForecaster` : Point forecasts without intervals.

    """

    _parameter_constraints: dict = {
        **BaseReductionForecaster._parameter_constraints,
        **BaseIntervalForecaster._parameter_constraints,
        "estimator": [HasMethods(["fit", "predict"])],
        "reduction_strategy": [StrOptions({"direct", "dir-rec", "multi-output"})],
    }

    def __init__(
        self,
        estimator: BaseEstimator = MultiOutputRegressor(QuantileRegressor()),
        reduction_strategy: Literal["direct", "dir-rec", "multi-output"] = "multi-output",
        target_as_feature: Literal["transformed", "raw"] | None = "transformed",
        feature_transformer: BaseTransformer | None = None,
        n_jobs: int | None = None,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ):
        BaseReductionForecaster.__init__(
            self,
            estimator=estimator,
            reduction_strategy=reduction_strategy,
            target_as_feature=target_as_feature,
            feature_transformer=feature_transformer,
            n_jobs=n_jobs,
            panel_strategy=panel_strategy,
        )

        BaseIntervalForecaster.__init__(
            self,
            feature_transformer=feature_transformer,
            target_as_feature=target_as_feature,
            panel_strategy=panel_strategy,
        )

    def _detect_multiquantile_loss(self) -> str | None:
        """Detect a multi-quantile loss function on the estimator.

        Checks whether the wrapped estimator (or any nested sub-estimator)
        exposes a ``loss_function`` parameter whose value starts with
        ``"MultiQuantile"`` (the convention used by CatBoost).

        When the estimator is wrapped (e.g. inside
        ``MultiOutputRegressor``), the deep parameter name is returned
        (e.g. ``"estimator__loss_function"``) so that ``set_params``
        can reach the nested parameter.

        Returns
        -------
        str or None
            The parameter name (e.g. ``"loss_function"`` or
            ``"estimator__loss_function"``) if detected, ``None``
            otherwise.

        """
        params = self.estimator.get_params(deep=True)
        for param_name, value in params.items():
            if (
                param_name.split("__")[-1] == "loss_function"
                and isinstance(value, str)
                and value.startswith("MultiQuantile")
            ):
                return param_name
        return None

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        coverage_rates: list[StrictFloat] | None = None,
        time_weight: Callable | pl.DataFrame | None = None,
        sample_weight_alignment: str = "first_step",
        **params,
    ) -> "IntervalReductionForecaster":
        """Fit the forecaster to historical data.

        Tabularizes the time series, fits the wrapped sklearn estimator,
        and calibrates prediction intervals from residuals.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        forecasting_horizon : int, default=1
            Number of time steps to forecast into the future.
        coverage_rates : list of float or None, default=None
            Coverage levels for prediction intervals (e.g., ``[0.9, 0.95]``
            for 90 % and 95 % intervals).  If ``None``, defaults to
            ``[0.95]``.
        time_weight : callable, pl.DataFrame, or None, default=None
            Per-timestep weights for fitting.  If ``pl.DataFrame``, must
            have a ``"time"`` and a ``"weight"`` column.  If callable,
            receives a ``pl.DataFrame`` and returns weights.  If ``None``,
            uniform weighting is applied.
        sample_weight_alignment : str, default="first_step"
            Strategy for converting ``time_weight`` to sklearn
            ``sample_weight`` across forecast horizons.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted forecaster instance.

        Notes
        -----
        **Time Weight Formats**:

        The ``time_weight`` parameter accepts three formats for specifying time-based
        importance of training samples:

        1. **DataFrame**: Must have "time" column matching y timestamps, plus:

           - Global weights: Single "weight" column applies to all series
           - Panel weights: Columns named "{group}_weight" (e.g., "store_1_weight")
             for group-specific weighting. Falls back to "weight" column if
             group-specific column missing.

        2. **Callable (single-argument)**: ``f(time: pl.Series) -> pl.Series``

           Applied uniformly to all series. Returns pl.Series with same length
           as input, containing non-negative weight values.

        3. **Callable (panel-aware)**: ``f(time: pl.Series, group_name: str) -> pl.Series``

           Enables group-specific weight generation. Signature detected via
           ``inspect.signature()`` parameter count (2 params = panel-aware).
           For global data, group_name will be None.

        **Sample Weight Alignment Strategies**:

        After tabularization, training samples no longer correspond 1:1 with original
        time points. Each sample predicts a window of future steps [t+1, ..., t+H].
        The alignment strategy determines how weights from this window are aggregated:

        - ``"first_step"``: Use weight at first forecast step (t+1)

          Example: For H=3, sample at t=10 uses weight at t=11

          Best for: Emphasizing immediate forecasts

        - ``"mean_step"``: Average weight across all horizon steps

          Example: For H=3, sample at t=10 uses mean(weight[t=11:t=13])

          Best for: Equal importance across forecast horizon, robust to noise

        - ``"weighted_mean_step"``: Exponentially weighted mean (near-term emphasized)

          Example: For H=3, sample at t=10 uses weighted average favoring t=11 over t=13

          Best for: Gradual decay in importance with forecast distance

        - ``"max_weight_step"``: Maximum weight across horizon steps

          Example: For H=3, sample at t=10 uses max(weight[t=11:t=13])

          Best for: Capturing seasonal peaks where any step may be critical

        - ``"min_weight_step"``: Minimum weight across horizon steps

          Example: For H=3, sample at t=10 uses min(weight[t=11:t=13])

          Best for: Conservative weighting, only high if all steps important

        **Weight Validation**:

        - Weights must be non-negative and finite (no NaN/inf)
        - Sum must be non-zero
        - Estimator must support ``sample_weight`` parameter in fit()

        **Interval Estimation**:

        This forecaster trains separate quantile regressors for upper and lower bounds.
        The same time weights and alignment strategy are applied to both estimators,
        ensuring consistent importance weighting across the prediction interval.

        """
        forecasting_horizon, self.fit_coverage_rates_ = self._validate_fit_params(forecasting_horizon, coverage_rates)

        y_t, X_t = BaseIntervalForecaster._pre_fit(
            self,
            y=y,
            X=X,
            forecasting_horizon=forecasting_horizon,
        )

        # Detect multi-quantile estimator (e.g. CatBoost ``MultiQuantile`` loss).
        # When present, a single model is fitted for all coverage-rate quantiles
        # simultaneously instead of 2N separate lower/upper models.
        multiquantile_param = self._detect_multiquantile_loss()

        if multiquantile_param is not None:
            all_quantiles = sorted({q for cr in self.fit_coverage_rates_ for q in ((1.0 - cr) / 2.0, (1.0 + cr) / 2.0)})
            q_to_idx = {q: i for i, q in enumerate(all_quantiles)}
            self._multiquantile_map_ = {
                cr: (q_to_idx[(1.0 - cr) / 2.0], q_to_idx[(1.0 + cr) / 2.0]) for cr in self.fit_coverage_rates_
            }

            alpha_str = ",".join(str(q) for q in all_quantiles)
            estimator = self._estimator_fit_one(
                y_t,
                X_t,
                forecasting_horizon,
                time_weight=time_weight,
                sample_weight_alignment=sample_weight_alignment,
                estimator_params={multiquantile_param: f"MultiQuantile:alpha={alpha_str}"},
            )
            self.estimator_ = {"_multiquantile": estimator}
        else:
            estimator_param_names = list(self.estimator.get_params(deep=True))
            quantile_param_names = [
                param_name for param_name in estimator_param_names if param_name.split("__")[-1] == "quantile"
            ]

            if len(quantile_param_names) > 1:
                raise ValueError(
                    f"Found multiple quantile parameters in estimator: {quantile_param_names}. "
                    f"IntervalReductionForecaster requires exactly one quantile parameter. "
                    f"Please use an estimator with a single quantile parameter."
                )

            if len(quantile_param_names) == 0:
                raise ValueError(
                    f"No quantile parameter found in estimator. "
                    f"IntervalReductionForecaster requires an estimator with a 'quantile' "
                    f"parameter (e.g., QuantileRegressor) or a multi-quantile loss function "
                    f"(e.g., CatBoost ``loss_function='MultiQuantile:alpha=...'``). "
                    f"Available parameters: {estimator_param_names}"
                )

            quantile_param_name = quantile_param_names[0]

            estimators = {}
            for coverage_rate in self.fit_coverage_rates_:
                estimator_params_lower = {
                    quantile_param_name: (1.0 - coverage_rate) / 2.0,
                }
                estimator_lower = self._estimator_fit_one(
                    y_t,
                    X_t,
                    forecasting_horizon,
                    time_weight=time_weight,
                    sample_weight_alignment=sample_weight_alignment,
                    estimator_params=estimator_params_lower,
                )

                estimator_params_upper = {
                    quantile_param_name: (1.0 + coverage_rate) / 2.0,
                }
                estimator_upper = self._estimator_fit_one(
                    y_t,
                    X_t,
                    forecasting_horizon,
                    time_weight=time_weight,
                    sample_weight_alignment=sample_weight_alignment,
                    estimator_params=estimator_params_upper,
                )

                estimators[f"coverage_rate_{coverage_rate}_lower"] = estimator_lower
                estimators[f"coverage_rate_{coverage_rate}_upper"] = estimator_upper

            self.estimator_ = estimators
        return self

    def _predict_one(
        self,
        panel_group_names: list[str],
        coverage_rates: list[StrictFloat] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Predicts `_fit_forecasting_horizon` steps from the observation horizon.

        Parameters
        ----------
        panel_group_names : list of str
            Panel group names to predict for.
        coverage_rates : list of float
            Coverage rates for the prediction intervals.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        if coverage_rates is None:
            coverage_rates = self.fit_coverage_rates_

        if "_multiquantile" in self.estimator_:
            return self._predict_one_multiquantile(panel_group_names, coverage_rates)

        y_pred = pl.DataFrame()
        for coverage_rate in coverage_rates:
            estimator_lower = self.estimator_[f"coverage_rate_{coverage_rate}_lower"]
            estimator_upper = self.estimator_[f"coverage_rate_{coverage_rate}_upper"]

            # Predict lower and upper bounds
            y_pred_lower = self._estimator_predict_one(estimator_lower, panel_group_names=panel_group_names)
            y_pred_upper = self._estimator_predict_one(estimator_upper, panel_group_names=panel_group_names)

            # Enforce monotonic bounds (lower <= upper) by column-wise min/max
            # This handles cases where QuantileRegressor predictions cross
            lower_cols = y_pred_lower.columns
            upper_cols = y_pred_upper.columns
            for lower_col, upper_col in zip(lower_cols, upper_cols, strict=False):
                lower_vals = y_pred_lower.get_column(lower_col)
                upper_vals = y_pred_upper.get_column(upper_col)
                # Compute true lower and upper
                true_lower = pl.when(lower_vals <= upper_vals).then(lower_vals).otherwise(upper_vals)
                true_upper = pl.when(lower_vals <= upper_vals).then(upper_vals).otherwise(lower_vals)
                y_pred_lower = y_pred_lower.with_columns(true_lower.alias(lower_col))
                y_pred_upper = y_pred_upper.with_columns(true_upper.alias(upper_col))

            # Rename columns to include coverage rate
            # Use actual column names (prefixed for panel data, unprefixed for global)
            lower_rename = {col: f"{col}_lower_{coverage_rate}" for col in y_pred_lower.columns}
            upper_rename = {col: f"{col}_upper_{coverage_rate}" for col in y_pred_upper.columns}

            # Rename columns (works for both global and panel data)
            y_pred_lower = y_pred_lower.rename(lower_rename)
            y_pred_upper = y_pred_upper.rename(upper_rename)

            # Merge lower and upper bounds
            if y_pred.shape[1] == 0:
                # First iteration: concatenate lower and upper bounds
                y_pred = pl.concat([y_pred_lower, y_pred_upper], how="horizontal")
            else:
                # Subsequent iterations: concatenate with existing predictions
                y_pred = pl.concat([y_pred, y_pred_lower, y_pred_upper], how="horizontal")

        y_pred = self._add_time_columns(y_pred)

        return y_pred

    def _predict_one_multiquantile(
        self,
        panel_group_names: list[str],
        coverage_rates: list[StrictFloat],
    ) -> pl.DataFrame:
        """Predict intervals from a multi-quantile estimator.

        Called when the wrapped estimator natively predicts multiple
        quantiles in a single ``predict`` call (e.g. CatBoost with
        ``MultiQuantile`` loss).  The single-model prediction is split
        into per-coverage-rate lower / upper bound columns.

        Parameters
        ----------
        panel_group_names : list of str
            Panel group names to predict for.
        coverage_rates : list of float
            Coverage rates for the prediction intervals.

        Returns
        -------
        pl.DataFrame
            Predicted intervals with time columns.

        """
        estimator = self.estimator_["_multiquantile"]
        target_cols = list(self.local_y_t_schema_.keys())

        if self.panel_group_names_ is None:
            # Global data
            assert isinstance(self._X_t_observed, pl.DataFrame)
            X_t = self._X_t_observed[[-1]].select(~cs.by_name("time"))
            assert self.local_X_t_schema_ is not None
            X_tab = X_t.select(list(self.local_X_t_schema_.keys())).to_numpy()
            y_raw = estimator.predict(X_tab)  # type: ignore[union-attr, unresolved-attribute]  # ty:ignore[unresolved-attribute]

            result_cols: list[pl.Series] = []
            for coverage_rate in coverage_rates:
                lower_idx, upper_idx = self._multiquantile_map_[coverage_rate]
                for col in target_cols:
                    lower_vals = y_raw[:, lower_idx].astype(np.float64)
                    upper_vals = y_raw[:, upper_idx].astype(np.float64)
                    true_lower = np.minimum(lower_vals, upper_vals)
                    true_upper = np.maximum(lower_vals, upper_vals)
                    result_cols.append(pl.Series(f"{col}_lower_{coverage_rate}", true_lower))
                    result_cols.append(pl.Series(f"{col}_upper_{coverage_rate}", true_upper))

            y_pred = pl.DataFrame(result_cols)
        else:
            # Panel data
            panel_frames: list[pl.DataFrame] = []
            for group_name in panel_group_names:
                assert isinstance(self._X_t_observed, dict)
                X_t_group = self._X_t_observed[group_name][[-1]].select(~cs.by_name("time"))
                X_tab = X_t_group.select(list(self.local_X_t_schema_.keys())).to_numpy()
                y_raw = estimator.predict(X_tab)

                group_cols: list[pl.Series] = []
                group_y_cols = list(self.local_y_t_schema_.keys())
                for coverage_rate in coverage_rates:
                    lower_idx, upper_idx = self._multiquantile_map_[coverage_rate]
                    for col in group_y_cols:
                        lower_vals = y_raw[:, lower_idx].astype(np.float64)
                        upper_vals = y_raw[:, upper_idx].astype(np.float64)
                        true_lower = np.minimum(lower_vals, upper_vals)
                        true_upper = np.maximum(lower_vals, upper_vals)
                        group_cols.append(pl.Series(f"{group_name}__{col}_lower_{coverage_rate}", true_lower))
                        group_cols.append(pl.Series(f"{group_name}__{col}_upper_{coverage_rate}", true_upper))
                panel_frames.append(pl.DataFrame(group_cols))
            y_pred = pl.concat(panel_frames, how="horizontal")

        y_pred = self._add_time_columns(y_pred)
        return y_pred
