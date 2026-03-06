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
    ['observed_time', 'time', 'value']

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
        n_jobs: int | None = None,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ) -> None:
        BaseReductionForecaster.__init__(
            self,
            estimator=estimator,
            reduction_strategy=reduction_strategy,
            target_as_feature=target_as_feature,
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
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        time_weight: Callable | pl.DataFrame | None = None,
        sample_weight_alignment: str = "first_step",
        **params,
    ) -> "PointReductionForecaster":
        """Fit the forecaster to historical data.

        Tabularizes the time series and fits the wrapped sklearn estimator.

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

        """
        forecasting_horizon = self._validate_fit_params(forecasting_horizon)

        y_t, X_t = self._pre_fit(
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
        **params,
    ) -> pl.DataFrame:
        """Predicts `_fit_forecasting_horizon` steps from the observation horizon.

        Parameters
        ----------
        panel_group_names : list of str
            Panel group names to predict for.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        y_pred = self._estimator_predict_one(self.estimator_, panel_group_names=panel_group_names)
        y_pred = self._add_time_columns(y_pred)

        return y_pred
