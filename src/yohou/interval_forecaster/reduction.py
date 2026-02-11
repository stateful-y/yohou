"""Implementation of reduction-based interval forecasters."""

from collections.abc import Callable
from typing import Literal

import polars as pl
from pydantic import StrictFloat, StrictInt
from sklearn.base import BaseEstimator, _fit_context
from sklearn.linear_model import QuantileRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils._param_validation import HasMethods, StrOptions

from yohou.base import BaseReductionForecaster, BaseTransformer

from .base import BaseIntervalForecaster


class IntervalReductionForecaster(BaseReductionForecaster, BaseIntervalForecaster):
    """Interval forecaster using sklearn estimators on tabularized time series.

    Converts the time series interval forecasting task to a tabular one.

    Parameters
    ----------
    estimator : BaseEstimator, default=MultiOutputRegressor(QuantileRegressor())
        Quantile estimator used to fit the tabularized data.
    reduction_strategy : {"direct", "multi-output"}, default="multi-output"
        Strategy for multi-step forecasting.
    input_features : {"X", "y_t|X", "y|X"}, default="y_t|X"
        Defines how the feature or the input to the ``feature_transformer``
        if passed is built.
    feature_transformer : BaseTransformer or None, default=None
        Transformer used to transform the `input_features` time series into features.
    update_strategy : {"average", "constant"}, default="average"
        How to update intervals with new observations.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.interval_forecaster import IntervalReductionForecaster
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
    - Direct: Separate model for each horizon step; predicts directly from inputs.
    - Multi-output: Single model predicts all horizon steps simultaneously.

    All models can be applied recursively for multi-step forecasting by specifying
    the forecasting horizon during prediction.

    This forecaster uses quantile regression to produce prediction intervals.
    For each coverage rate α, it predicts:

    - Lower bound: (1 - α)/2 quantile
    - Upper bound: (1 + α)/2 quantile

    The intervals naturally adapt to heteroscedastic data where uncertainty
    varies over time.

    See Also
    --------
    SplitConformalForecaster : Conformal prediction intervals
    PointReductionForecaster : Point forecasts without intervals

    """

    _parameter_constraints: dict = {
        **BaseReductionForecaster._parameter_constraints,
        **BaseIntervalForecaster._parameter_constraints,
        "estimator": [HasMethods(["fit", "predict"])],
        "reduction_strategy": [StrOptions({"direct", "multi-output"})],
    }

    def __init__(
        self,
        estimator: BaseEstimator = MultiOutputRegressor(QuantileRegressor()),
        reduction_strategy: Literal["direct", "multi-output"] = "multi-output",
        input_features: Literal["X", "y_t|X", "y|X"] = "y_t|X",
        feature_transformer: BaseTransformer | None = None,
    ):
        BaseReductionForecaster.__init__(
            self,
            estimator=estimator,
            reduction_strategy=reduction_strategy,
            input_features=input_features,
            feature_transformer=feature_transformer,
        )

        BaseIntervalForecaster.__init__(
            self,
            feature_transformer=feature_transformer,
        )

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
        """Fits the forecaster and returns it.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame or None, default=None
            Exogenous feature time series.
        forecasting_horizon : int > 1, default=1
            Horizon to forecast.
        coverage_rates : list of float or None, default=None
            Coverage rates for the prediction intervals. If None, uses ``[0.95]``.
        time_weight : callable, pl.DataFrame, or None, default=None
            Time-based weights for training samples. Converted to sklearn
            ``sample_weight`` parameter during model fitting. See Notes for format details.
        sample_weight_alignment : {"first_step", "mean_step", "weighted_mean_step", "max_weight_step", "min_weight_step"}, default="first_step"
            Strategy for aligning time weights to tabularized training samples.
            See Notes for detailed explanation of each strategy.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self

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
                f"IntervalReductionForecaster requires an estimator with a 'quantile' parameter "
                f"(e.g., QuantileRegressor). Available parameters: {estimator_param_names}"
            )

        quantile_param_name = quantile_param_names[0]

        estimators = {}

        # TODO: Support CatBoost multiquantile
        for coverage_rate in self.fit_coverage_rates_:
            # Fit lower bound estimator (lower quantile)
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

            # Fit upper bound estimator (upper quantile)
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
        coverage_rates: list[StrictFloat],
    ) -> pl.DataFrame:
        """Predicts `_fit_forecasting_horizon` steps from the observation horizon.

        Parameters
        ----------
        panel_group_names : list of str
            Panel group names to predict for.
        coverage_rates : list of float
            Coverage rates for the prediction intervals.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        y_pred = pl.DataFrame()
        for coverage_rate in coverage_rates:
            estimator_lower = self.estimator_[f"coverage_rate_{coverage_rate}_lower"]
            estimator_upper = self.estimator_[f"coverage_rate_{coverage_rate}_upper"]

            # Predict lower and upper bounds
            y_pred_lower = self._estimator_predict_one(estimator_lower, panel_group_names=panel_group_names)
            y_pred_upper = self._estimator_predict_one(estimator_upper, panel_group_names=panel_group_names)

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
