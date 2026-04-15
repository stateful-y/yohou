"""Base classes for interval forecasters and similarity measures."""

import abc
from typing import Any, Literal

import numpy as np
import polars as pl
import polars.selectors as cs
from pydantic import StrictFloat, StrictInt
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseForecaster, BaseTransformer
from yohou.utils import INTERVAL, Tags, cast, select_panel_columns, validate_forecaster_data

__all__ = ["BaseIntervalForecaster", "BaseSimilarity"]


class BaseSimilarity(BaseEstimator, metaclass=abc.ABCMeta):
    """Base class for similarity measures used in interval forecasting.

    Similarity measures assign weights to calibration residuals based
    on how similar past prediction contexts are to the current one.

    Notes
    -----
    Used by ``SplitConformalForecaster`` to produce adaptive (locally
    weighted) prediction intervals.  When ``similarity=None``, uniform
    weights are used.

    See Also
    --------
    `DistanceSimilarity` : Distance-based similarity measure.
    `SplitConformalForecaster` : Conformal forecaster that uses similarities.

    """

    _parameter_constraints: dict = {}

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with similarity-specific attributes.

        """
        tags = Tags(estimator_type="similarity", requires_fit=True)

        # Most similarity measures are symmetric and require predictions
        assert tags.similarity_tags is not None
        tags.similarity_tags.symmetric = True
        tags.similarity_tags.requires_predictions = True
        tags.similarity_tags.produces_weights = True

        return tags

    @property
    def discarded_time_stamps(self) -> None:
        """Get discarded timestamps (placeholder property).

        Returns
        -------
        None

        """
        return None

    @abc.abstractmethod
    def fit(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> "BaseSimilarity":
        """Fit the similarity measure.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        y_pred : pl.DataFrame
            Point predictions.

        X : pl.DataFrame or None, default=None
            Exogenous features.

        Returns
        -------
        self

        """

    @abc.abstractmethod
    def observe(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> "BaseSimilarity":
        """Observe new data and update the similarity measure.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.

        y_pred : pl.DataFrame
            New predictions.

        X : pl.DataFrame or None, default=None
            New exogenous features.

        Returns
        -------
        self

        """

    @abc.abstractmethod
    def predict(
        self,
        y_pred: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating[Any]]]:
        """Compute similarity weights for predictions.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Predictions to compute similarities for.

        X : pl.DataFrame or None, default=None
            Exogenous features.

        Returns
        -------
        np.ndarray
            Similarity weights.

        """


class BaseIntervalForecaster(BaseForecaster, metaclass=abc.ABCMeta):
    """Base class for interval forecasters.

    Parameters
    ----------
    feature_transformer : instance of `BaseTransformer` or None, default=None
        Transformer used to transform the feature time series into features.
    target_as_feature : {"transformed", "raw"} or None, default="transformed"
        Controls whether the target is included as a feature.
        ``"transformed"`` includes the transformed target, ``"raw"``
        includes the raw target, and ``None`` uses only exogenous features.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data. See `BaseForecaster` for details.

    Attributes
    ----------
    fit_coverage_rates_ : list of float
        Coverage rates used during fit.

    Notes
    -----
    Interval forecasters produce prediction intervals at specified
    coverage rates.  The ``forecaster_type`` tag is ``INTERVAL``
    (or ``POINT_INTERVAL`` if point predictions are also available).

    See Also
    --------
    `SplitConformalForecaster` : Conformal interval forecaster.
    `IntervalReductionForecaster` : ML-based interval forecaster.
    `BasePointForecaster` : Base class for point forecasters.

    """

    _parameter_constraints: dict = {
        **BaseForecaster._parameter_constraints,
    }

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        tags.forecaster_tags.forecaster_type = INTERVAL
        return tags

    def __init__(
        self,
        feature_transformer: BaseTransformer | None = None,
        target_as_feature: Literal["transformed", "raw"] | None = "transformed",
        panel_strategy: Literal["global", "multivariate"] = "global",
    ) -> None:
        super().__init__(
            feature_transformer=feature_transformer,
            target_transformer=None,
            target_as_feature=target_as_feature,
            panel_strategy=panel_strategy,
        )

    @abc.abstractmethod
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        coverage_rates: list[float] | None = None,
        **params,
    ) -> "BaseIntervalForecaster":
        """Fit the forecaster to historical data.

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
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted forecaster instance.

        Raises
        ------
        ValueError
            If ``forecasting_horizon`` < 1, ``coverage_rates`` not in (0, 1],
            or if ``y`` / ``X`` have invalid structure.

        """

    def _validate_fit_params(
        self,
        forecasting_horizon: StrictInt,
        coverage_rates: list[StrictFloat] | None = None,
    ) -> tuple[StrictInt, list[StrictFloat]]:
        """Validate fit parameters.

        Parameters
        ----------
        forecasting_horizon : int
            Forecasting horizon to validate.
        coverage_rates : list of float or None
            Coverage rates to validate. If None, uses [0.95].

        Returns
        -------
        tuple of (int, list of float)
            Validated forecasting horizon and coverage rates.

        Raises
        ------
        ValueError
            If forecasting_horizon < 1 or coverage_rates not in (0, 1].

        """
        if forecasting_horizon < 1:
            raise ValueError(f"forecasting_horizon must be >= 1, got {forecasting_horizon}")

        if coverage_rates is None:
            coverage_rates = [0.95]

        # Validate coverage rates
        for rate in coverage_rates:
            if not (0 < rate <= 1):
                raise ValueError(f"All coverage_rates must be in (0, 1], got {rate}")

        return forecasting_horizon, coverage_rates

    def _validate_predict_params(
        self,
        forecasting_horizon: StrictInt | None,
        coverage_rates: list[StrictFloat] | None = None,
    ) -> tuple[StrictInt, list[StrictFloat]]:
        """Validate and return predict parameters.

        Parameters
        ----------
        forecasting_horizon : int or None
            Forecasting horizon to validate. If None, uses fit_forecasting_horizon_.
        coverage_rates : list of float or None
            Coverage rates to validate. If None, uses fit_coverage_rates_.

        Returns
        -------
        tuple of (int, list of float)
            Validated forecasting horizon and coverage rates.

        Raises
        ------
        ValueError
            If forecasting_horizon < 1 or coverage_rates not in (0, 1].

        """
        if forecasting_horizon is None:
            forecasting_horizon = self.fit_forecasting_horizon_
        if coverage_rates is None:
            # fit_coverage_rates_ is set by concrete subclasses during fit().
            coverage_rates = self.fit_coverage_rates_  # ty: ignore[unresolved-attribute]
        return self._validate_fit_params(forecasting_horizon, coverage_rates)

    def predict_interval(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        coverage_rates: list[float] | None = None,
        strategy: Literal["mean", "median", "point"] | None = None,
        panel_group_names: list[str] | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate interval forecasts.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future.  If ``None``,
            uses the horizon specified at fit time.
        coverage_rates : list of float or None, default=None
            Coverage levels for prediction intervals (e.g., ``[0.9, 0.95]``
            for 90 % and 95 % intervals).  If ``None``, defaults to the rates
            used at fit time.
        strategy : {"mean", "median", "point"} or None, default=None
            Strategy for deriving point predictions from prediction intervals
            during recursive multi-step forecasting:

            - ``"mean"``: use the mean of the interval bounds
            - ``"median"``: use the median of the interval bounds
            - ``"point"``: use the point forecast directly (if available)

            If ``None``, defaults to ``"mean"``.
        panel_group_names : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Interval predictions with ``"observed_time"``, ``"time"``, and
            lower/upper bound columns for each target at each coverage rate.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the forecaster has not been fitted yet.
        ValueError
            If ``X`` has invalid structure, ``coverage_rates`` not in (0, 1],
            or ``panel_group_names`` contains names not seen during fit.

        """
        check_is_fitted(self, ["panel_group_names_", "local_y_schema_", "fit_forecasting_horizon_"])
        _, X, panel_group_names = validate_forecaster_data(
            self,
            y=None,
            X=X,
            reset=False,
            panel_group_names=panel_group_names,
        )

        forecasting_horizon, coverage_rates = self._validate_predict_params(forecasting_horizon, coverage_rates)

        y_columns = list(self.local_y_schema_.keys())
        if panel_group_names is not None:
            y_columns = [f"{panel_group}__{col}" for panel_group in panel_group_names for col in self.local_y_schema_]

            if X is not None:
                X = select_panel_columns(
                    X,
                    self.panel_group_names_,
                    include_global=True,
                )

        def step_fn(forecaster, groups):
            """Produce one interval-prediction block."""
            y_pred_step, y_pred_step_inv = forecaster._predict(groups, coverage_rates=coverage_rates)
            return y_pred_step_inv, y_pred_step_inv

        def derive_observation_fn(forecaster, y_pred_step_inv, X, step):
            """Derive observation from interval bounds."""
            time = y_pred_step_inv.select(cs.by_name("time"))

            y_data: dict[str, Any] = {"time": time["time"]}
            for col in y_columns:
                lower_cols = [c for c in y_pred_step_inv.columns if c.startswith(f"{col}_lower_")]
                upper_cols = [c for c in y_pred_step_inv.columns if c.startswith(f"{col}_upper_")]

                all_bound_cols = lower_cols + upper_cols

                if strategy == "point" and col in y_pred_step_inv.columns:
                    y_data[col] = y_pred_step_inv[col]
                elif strategy == "median":
                    y_data[col] = y_pred_step_inv.select(
                        pl.median_horizontal(all_bound_cols)  # ty: ignore[unresolved-attribute]
                    ).to_series()
                else:
                    y_data[col] = y_pred_step_inv.select(pl.mean_horizontal(all_bound_cols)).to_series()

            y = pl.DataFrame(y_data)

            if panel_group_names is not None:
                cast_schema = {}
                for group_name in panel_group_names:
                    for col_name, dtype in forecaster.local_y_schema_.items():
                        cast_schema[f"{group_name}__{col_name}"] = dtype
            else:
                cast_schema = forecaster.local_y_schema_

            y = cast(y.select(~cs.by_name("time")), cast_schema)
            y = pl.concat([y_data["time"].to_frame(), y], how="horizontal")

            X_slice = None
            if X is not None:
                X_slice = X.join(y.select("time"), on="time", how="semi")

                if len(X_slice) != len(y):
                    raise ValueError(
                        f"Missing X for future steps. Needed {len(y)} rows, but X slice has {len(X_slice)} rows."
                    )

            return y, X_slice

        return self._recursive_predict(
            X=X,
            forecasting_horizon=forecasting_horizon,
            panel_group_names=panel_group_names,
            step_fn=step_fn,
            derive_observation_fn=derive_observation_fn,
        )

    def observe_predict_interval(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        coverage_rates: list[float] | None = None,
        strategy: Literal["mean", "median", "point"] | None = None,
        panel_group_names: list[str] | None = None,
        stride: StrictInt | None = None,
        **params,
    ) -> pl.DataFrame:
        """Alternate recursive predict_interval and observe.

        Equivalent to calling ``observe(y, X)`` then
        ``predict_interval(X)``.  Returns interval predictions.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X : pl.DataFrame or None, default=None
            Exogenous features with a ``"time"`` column matching ``y``.
            If ``None``, no exogenous features are used.
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future.  If ``None``,
            uses the horizon specified at fit time.
        coverage_rates : list of float or None, default=None
            Coverage levels for prediction intervals (e.g., ``[0.9, 0.95]``
            for 90 % and 95 % intervals).  If ``None``, defaults to the rates
            used at fit time.
        strategy : {"mean", "median", "point"} or None, default=None
            Strategy for deriving point predictions from prediction intervals
            during recursive multi-step forecasting:

            - ``"mean"``: use the mean of the interval bounds
            - ``"median"``: use the median of the interval bounds
            - ``"point"``: use the point forecast directly (if available)

            If ``None``, defaults to ``"mean"``.
        panel_group_names : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.
        stride : int or None, default=None
            Step size for rolling update-predict.  If ``None``, defaults to
            ``forecasting_horizon``.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Interval predictions with ``"observed_time"``, ``"time"``, and
            lower/upper bound columns for each target at each coverage rate.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the forecaster has not been fitted yet.
        ValueError
            If ``y`` / ``X`` have invalid structure, ``coverage_rates`` not in
            (0, 1], or ``panel_group_names`` contains names not seen during fit.

        """
        check_is_fitted(self, ["panel_group_names_", "local_y_schema_", "fit_forecasting_horizon_"])
        y, X, panel_group_names = validate_forecaster_data(
            self,
            y=y,
            X=X,
            reset=False,
            panel_group_names=panel_group_names,
        )

        forecasting_horizon, _ = self._validate_predict_params(forecasting_horizon, coverage_rates)

        if stride is None:
            stride = self.fit_forecasting_horizon_

        return self._observe_predict_loop(
            predict_fn=self.predict_interval,
            y=y,
            X=X,
            panel_group_names=panel_group_names,
            stride=stride,
            forecasting_horizon=forecasting_horizon,
            coverage_rates=coverage_rates,
            strategy=strategy,
            **params,
        )

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

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        raise NotImplementedError()
