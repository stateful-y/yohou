"""Implementation of reduction-based point forecasters."""

from collections.abc import Callable
from typing import Literal

import polars as pl
from pydantic import StrictInt
from sklearn.base import BaseEstimator, _fit_context
from sklearn.linear_model import LinearRegression
from sklearn.utils._param_validation import HasMethods, StrOptions

from yohou.base import BaseReductionForecaster, BaseTransformer

from .base import BasePointForecaster


class PointReductionForecaster(BaseReductionForecaster, BasePointForecaster):
    """Point forecaster using sklearn estimators on tabularized time series.

    Converts the time series point forecasting task to a tabular one.

    Parameters
    ----------
    estimator : BaseEstimator, default=LinearRegression()
        Point estimator used to fit the tabularized data.
    reduction_strategy : {"direct", "multi-output"}, default="multi-output"
        Strategy for multi-step forecasting.
    target_transformer : BaseTransformer or None, default=None
        Transformer for target preprocessing.
    feature_transformer : BaseTransformer or None, default=None
        Transformer for feature engineering (typically LagTransformer).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.point_forecaster import PointReductionForecaster
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
    >>> # Create and fit forecaster
    >>> forecaster = PointReductionForecaster()
    >>> _ = forecaster.fit(y=train, forecasting_horizon=1)
    >>>
    >>> # Generate one-step prediction
    >>> y_pred = forecaster.predict(forecasting_horizon=1)
    >>> len(y_pred)
    1
    >>> sorted(y_pred.columns)
    ['observed_time', 'time', 'value']

    Notes
    -----
    Reduction strategies:
    - Direct: Separate model for each horizon step; predicts directly from inputs.
    - Multi-output: Single model predicts all horizon steps simultaneously.

    All models can be applied recursively for multi-step forecasting by specifying
    the forecasting horizon during prediction.

    See Also
    --------
    BaseReductionForecaster : Base class for reduction forecasters
    LagTransformer : Create lagged features for reduction strategies

    """

    _parameter_constraints: dict = {
        **BaseReductionForecaster._parameter_constraints,
        **BasePointForecaster._parameter_constraints,
        "estimator": [HasMethods(["fit", "predict"])],
        "reduction_strategy": [StrOptions({"direct", "multi-output"})],
    }

    _supports_panel = True

    def __init__(
        self,
        estimator: BaseEstimator = LinearRegression(),
        reduction_strategy: Literal["direct", "multi-output"] = "multi-output",
        target_transformer: BaseTransformer | None = None,
        feature_transformer: BaseTransformer | None = None,
    ) -> None:
        BaseReductionForecaster.__init__(
            self,
            estimator=estimator,
            reduction_strategy=reduction_strategy,
        )

        BasePointForecaster.__init__(
            self,
            target_transformer=target_transformer,
            feature_transformer=feature_transformer,
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        time_weight: Callable | pl.DataFrame | None = None,
        sample_weight_alignment: str = "first_step",
        **params,
    ) -> "PointReductionForecaster":
        """Fits the forecaster and returns it.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame or None, default=None
            Exogenous feature time series.
        forecasting_horizon : int >= 1, default=1
            Horizon to forecast.
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

        """
        forecasting_horizon = self._validate_fit_params(forecasting_horizon)

        y_t, X_t = BasePointForecaster._pre_fit(
            self,
            y=y,
            X=X,
            forecasting_horizon=forecasting_horizon,
        )

        self.estimator_ = self._estimator_fit_one(
            y_t,
            X_t,
            forecasting_horizon,
            time_weight=time_weight,
            sample_weight_alignment=sample_weight_alignment,
            estimator_fit_params=params,
        )

        return self

    def _predict_one(
        self,
        panel_group_names: list[str],
    ) -> pl.DataFrame:
        """Predicts `_fit_forecasting_horizon` steps from the observation horizon.

        Parameters
        ----------
        panel_group_names : list of str
            Panel group names to predict for.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        y_pred = self._estimator_predict_one(self.estimator_, panel_group_names=panel_group_names)
        y_pred = self._add_time_columns(y_pred)

        return y_pred
