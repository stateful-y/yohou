"""Implementation of reduction-based point forecasters."""

from collections.abc import Callable
from typing import Literal

import polars as pl
from pydantic import StrictInt
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from yohou.base import BaseReductionForecaster, BaseTransformer
from yohou.utils._compat import HasMethods, StrOptions, _fit_context

from .base import BasePointForecaster

__all__ = ["PointReductionForecaster"]


class PointReductionForecaster(BaseReductionForecaster, BasePointForecaster):
    """Point forecaster using sklearn estimators on tabularized time series.

    Converts the time series point forecasting task to a tabular one.

    Parameters
    ----------
    estimator : BaseEstimator, default=LinearRegression()
        Point estimator used to fit the tabularized data.
    reduction_strategy : {"direct", "dir-rec", "multi-output"}, default="multi-output"
        Strategy for multi-step forecasting.
    target_transformer : BaseTransformer or None, default=None
        Transformer for target preprocessing.
    feature_transformer : BaseTransformer or None, default=None
        Transformer for feature engineering (typically LagTransformer).
    target_as_feature : {"transformed", "raw"} or None, default="transformed"
        Whether to include the target variable as a feature for reduction.
        If ``"transformed"``, the transformed target is used. If ``"raw"``,
        the raw target is used. If ``None``, the target is not included as
        a feature.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data. See `BaseForecaster` for details.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel for the ``"direct"`` strategy
        (fitting and predicting H independent models). ``None`` means 1
        unless in a ``joblib.parallel_backend`` context. ``-1`` means
        using all processors. Has no effect for ``"multi-output"`` or
        ``"dir-rec"`` strategies.
    step_feature_alignment : {"all", "matched", "cumulative"}, default="all"
        Controls which step-indexed feature columns each direct estimator
        sees. Only affects the ``"direct"`` strategy.

        - ``"all"``: every estimator receives all step columns.
        - ``"matched"``: estimator for step h receives only ``*_step_h``.
        - ``"cumulative"``: estimator for step h receives ``*_step_1..h``.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.point import PointReductionForecaster
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
    ['time', 'value', 'vintage_time']

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
    `BaseReductionForecaster` : Base class for reduction forecasters.
    `LagTransformer` : Create lagged features for reduction strategies.

    """

    _parameter_constraints: dict = {
        **BaseReductionForecaster._parameter_constraints,
        **BasePointForecaster._parameter_constraints,
        "estimator": [HasMethods(["fit", "predict"])],
        "reduction_strategy": [StrOptions({"direct", "dir-rec", "multi-output"})],
    }

    _supports_panel = True

    def __init__(
        self,
        estimator: BaseEstimator = LinearRegression(),
        reduction_strategy: Literal["direct", "dir-rec", "multi-output"] = "multi-output",
        target_transformer: BaseTransformer | None = None,
        feature_transformer: BaseTransformer | None = None,
        target_as_feature: Literal["transformed", "raw"] | None = "transformed",
        step_feature_alignment: Literal["all", "matched", "cumulative"] = "all",
        n_jobs: int | None = None,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ) -> None:
        BaseReductionForecaster.__init__(
            self,
            estimator=estimator,
            reduction_strategy=reduction_strategy,
            target_as_feature=target_as_feature,
            step_feature_alignment=step_feature_alignment,
            n_jobs=n_jobs,
            panel_strategy=panel_strategy,
        )

        BasePointForecaster.__init__(
            self,
            target_transformer=target_transformer,
            feature_transformer=feature_transformer,
            target_as_feature=target_as_feature,
            panel_strategy=panel_strategy,
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        time_weight: Callable | pl.DataFrame | dict | None = None,
        vintage_weight: Callable | pl.DataFrame | dict | None = None,
        sample_weight_alignment: str = "first_step",
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> "PointReductionForecaster":
        """Fit the forecaster to historical data.

        Tabularizes the time series and fits the wrapped sklearn estimator.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X_actual : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        forecasting_horizon : int, default=1
            Number of time steps to forecast into the future.
        time_weight : callable, pl.DataFrame, dict, or None, default=None
            Per-timestep weights for fitting.  Accepts a callable
            ``f(time_series) -> pl.Series``, a panel-aware callable
            ``f(time_series, group_name) -> pl.Series``, a DataFrame
            with ``"time"`` and ``"weight"`` columns, or a
            ``{datetime_or_str: float}`` dict (``"*"`` key sets default).
        vintage_weight : callable, pl.DataFrame, dict, or None, default=None
            Per-vintage weights for fitting.  Same formats as
            ``time_weight``.  Resolved via direct lookup at observation
            time (no alignment strategy). Combined multiplicatively
            with ``time_weight``.
        sample_weight_alignment : str, default="first_step"
            Strategy for converting ``time_weight`` to sklearn
            ``sample_weight`` across forecast horizons. Does not apply
            to ``vintage_weight`` (which uses direct lookup).
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column. Deterministic
            values available for past and future dates. Bypasses the
            feature transformer.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns. Bypasses the feature transformer.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted forecaster instance.

        """
        forecasting_horizon = self._validate_fit_params(forecasting_horizon)

        y_t, X_t = self._pre_fit(
            y=y,
            X=X_actual,
            forecasting_horizon=forecasting_horizon,
            X_future=X_future,
            X_forecast=X_forecast,
        )

        self.estimator_ = self._estimator_fit_one(
            y_t,
            X_t,
            forecasting_horizon,
            time_weight=time_weight,
            sample_weight_alignment=sample_weight_alignment,
            vintage_weight=vintage_weight,
            estimator_fit_params=params,
        )

        return self

    def _predict_one(
        self,
        groups: list[str],
        **params,
    ) -> pl.DataFrame:
        """Predicts `_fit_forecasting_horizon` steps from the observation horizon.

        Parameters
        ----------
        groups : list of str
            Panel group names to predict for.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        y_pred = self._estimator_predict_one(self.estimator_, groups=groups)
        y_pred = self._add_time_columns(y_pred)

        return y_pred
