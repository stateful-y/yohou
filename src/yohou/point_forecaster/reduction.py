"""Implementation of reduction-based point forecasters."""

from typing import Callable

import polars as pl
from pydantic import StrictInt
from sklearn.base import BaseEstimator, _fit_context
from sklearn.linear_model import LinearRegression
from sklearn.utils._param_validation import HasMethods, StrOptions
from typing_extensions import Literal

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
    ...         start=datetime(2021, 1, 1),
    ...         end=datetime(2021, 1, 10),
    ...         interval="1d",
    ...         eager=True
    ...     ),
    ...     "value": [10.0, 12.0, 15.0, 14.0, 16.0, 18.0, 20.0, 19.0, 21.0, 23.0]
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
            ``sample_weight`` parameter during model fitting, giving more importance
            to observations at specific times (e.g., recent data, seasonal peaks).
            
            Accepts three formats:
            
            - **DataFrame**: Must have "time" column matching y timestamps, plus:
              
              - Global weights: Single "weight" column applies to all series
              - Panel weights: Columns named "{group}_weight" (e.g., "store_1_weight")
                for group-specific weighting. Falls back to "weight" column if
                group-specific column missing.
            
            - **Callable (single-argument)**: Function accepting time series:
              ``f(time: pl.Series) -> pl.Series``
              
              Applied uniformly to all series. Returns pl.Series with same length
              as input, containing non-negative weight values.
            
            - **Callable (panel-aware)**: Function accepting time series and group name:
              ``f(time: pl.Series, group_name: str) -> pl.Series``
              
              Enables group-specific weight generation. Signature detected via
              ``inspect.signature()`` parameter count (2 params = panel-aware).
              For global data, group_name will be None.
            
            - **None**: Equal weighting for all training samples (standard behavior).
            
            **Alignment with tabularized data**: After tabularization, rows no longer
            correspond 1:1 with original times. The ``sample_weight_alignment``
            parameter controls how original time-based weights map to tabularized
            training samples (see below).
            
            **Validation**: Weights must be non-negative and finite (no NaN/inf).
            Sum must be non-zero. Estimator must support sample_weight parameter.
        sample_weight_alignment : {"first_step", "mid_step", "last_step"}, default="first_step"
            Strategy for aligning time weights to tabularized training samples:
            
            - "first_step": Weight from first prediction target time (default)
            - "mid_step": Weight from middle of prediction window
            - "last_step": Weight from last prediction target time
            
            Example: For forecasting_horizon=5, each training sample predicts
            steps [t+1, t+2, ..., t+5]. Alignment determines which step's weight
            is used: first_step uses t+1, mid_step uses t+3, last_step uses t+5.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self

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
