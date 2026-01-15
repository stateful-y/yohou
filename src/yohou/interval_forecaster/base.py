"""Base classes for interval forecasters and similarity measures."""

import abc
from copy import deepcopy
from typing import Any, Literal, override

import numpy as np
import polars as pl
import polars.selectors as cs
from pydantic import StrictFloat, StrictInt
from sklearn.base import BaseEstimator

from yohou.base import BaseForecaster, BaseTransformer, Tags
from yohou.utils import select_panel_columns
from yohou.utils.polars import cast
from yohou.utils.validation import validate_data


class BaseSimilarity(BaseEstimator, metaclass=abc.ABCMeta):
    """Base class for similarity measures used in interval forecasting."""

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with similarity-specific attributes.

        """
        tags = Tags(estimator_type="similarity", requires_fit=True)

        # Most similarity measures are symmetric and require predictions
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
        raise NotImplementedError()

    @abc.abstractmethod
    def update(
        self,
        y: pl.DataFrame,
        y_pred: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> "BaseSimilarity":
        """Update the similarity measure with new observations.

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
        raise NotImplementedError()

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
        raise NotImplementedError()


class BaseIntervalForecaster(BaseForecaster, metaclass=abc.ABCMeta):
    """Base class for conformal forecasters.

    Parameters
    ----------
    feature_transformer : instance of `BaseTransformer` or None, default=None
        Transformer used to transform the `input_features` time series into features.
    input_features: "X" | "y_t|X" | "y|X", default="y_t|X"
        Defines how the feature or the input to the ``feature_transformer``
         if passed is built.

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
        tags.forecaster_tags.forecaster_type = "interval"
        return tags

    def __init__(
        self,
        feature_transformer: BaseTransformer | None = None,
        input_features: Literal["X", "y_t|X", "y|X"] = "y_t|X",
    ) -> None:
        super().__init__(
            feature_transformer=feature_transformer,
            target_transformer=None,
            input_features=input_features,
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
        """Fits the forecaster and returns it.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame or None, default=None
            Exogenous feature time series.
        forecasting_horizon : int >= 1, default=1
            Horizon to forecast.
        coverage_rates : list of floats or None, default=None
            Coverage rates for the prediction intervals. If None, uses ``[0.95]``.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self

        """
        raise NotImplementedError()

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
            coverage_rates = self.fit_coverage_rates_
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
        """Predicts an interval according to coverage rates.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Exogenous feature time series.
        forecasting_horizon : int >= 1 or None, default=None
            Horizon to forecast. If None, uses ``fit_forecasting_horizon_``.
        coverage_rates : list of floats or None, default=None
            Coverage rates for the prediction intervals. If None, uses ``fit_coverage_rates_``.
        strategy : {"mean", "median", "point"} or None, default=None
            Strategy for updating with new point observations:
            - "mean": use the mean of the interval bounds as point observation
            - "median": use the median of the interval bounds as point observation
            - "point": use the point forecast directly (if available)
            If None, defaults to "mean".
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data:
            - If None: predict for all groups
            - If list of str: predict only for the specified panel groups
            Parameter is ignored if the forecaster was not fitted on panel data.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Predicted time series with interval bounds.

        """
        _, X, panel_group_names = validate_data(
            self,
            y=None,
            X=X,
            reset=False,
            panel_group_names=panel_group_names,
            check_continuity=False,
        )

        forecasting_horizon, coverage_rates = self._validate_predict_params(
            forecasting_horizon, coverage_rates
        )

        forecaster = deepcopy(self)

        y_columns = list(forecaster.local_y_schema_.keys())
        if panel_group_names is not None:
            y_columns = [
                f"{panel_group}__{col}"
                for panel_group in panel_group_names
                for col in forecaster.local_y_schema_.keys()
            ]

            # Filter X
            if X is not None:
                X = select_panel_columns(
                    X,
                    self.panel_group_names_,
                    include_global=True,
                )

        y_pred = pl.DataFrame()
        for step in range(1, forecasting_horizon + 1, self.fit_forecasting_horizon_):
            y_pred_step, y_pred_step_inv = BaseForecaster._predict(
                forecaster, panel_group_names, coverage_rates=coverage_rates
            )
            y_pred = pl.concat([y_pred, y_pred_step_inv])

            if step + self.fit_forecasting_horizon_ <= forecasting_horizon:
                time = y_pred_step.select(cs.by_name("time"))

                # Global data case
                y_data = {"time": time["time"]}
                for col in y_columns:
                    # Find all coverage rates for this column
                    lower_cols = [
                        c for c in y_pred_step_inv.columns if c.startswith(f"{col}_lower_")
                    ]
                    upper_cols = [
                        c for c in y_pred_step_inv.columns if c.startswith(f"{col}_upper_")
                    ]

                    all_bound_cols = lower_cols + upper_cols

                    if strategy == "point" and col in y_pred_step_inv.columns:
                        y_data[col] = y_pred_step_inv[col]
                    elif strategy == "median":
                        y_data[col] = y_pred_step_inv.select(
                            pl.median_horizontal(all_bound_cols)
                        ).to_series()
                    else:
                        y_data[col] = y_pred_step_inv.select(
                            pl.mean_horizontal(all_bound_cols)
                        ).to_series()

                y = pl.DataFrame(y_data)

                # Cast to match expected schema (mean/median returns Float64)
                # Build prefixed schema for panel data
                if panel_group_names is not None:
                    cast_schema = {}
                    for group_name in panel_group_names:
                        for col, dtype in forecaster.local_y_schema_.items():
                            cast_schema[f"{group_name}__{col}"] = dtype
                else:
                    cast_schema = forecaster.local_y_schema_

                y = cast(y.select(~cs.by_name("time")), cast_schema)
                y = pl.concat([y_data["time"].to_frame(), y], how="horizontal")

                X_slice = None
                if X is not None:
                    X_slice = X.join(y.select("time"), on="time", how="semi")

                    if len(X_slice) != len(y):
                        raise ValueError(
                            f"Missing X for future steps. Needed {len(y)} rows, "
                            f"but X slice has {len(X_slice)} rows."
                        )

                forecaster.update(y, X_slice)

        y_pred = y_pred.with_columns(observed_time=y_pred["observed_time"][0])

        if forecasting_horizon % self.fit_forecasting_horizon_:
            end = (
                self.fit_forecasting_horizon_ - forecasting_horizon % self.fit_forecasting_horizon_
            )
            y_pred = y_pred[:-end]

        return y_pred

    def update_predict_interval(
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
        """Alternate recursive `predict` and `update`.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series for updates.
        X : pl.DataFrame or None, default=None
            Feature time series for predictions.
        forecasting_horizon : int >= 1 or None, default=None
            Horizon to forecast recursively. If None, uses ``fit_forecasting_horizon_``.
        coverage_rates : list of floats or None, default=None
            Coverage rates for the prediction intervals. If None, uses ``fit_coverage_rates_``
        strategy : {"mean", "median", "point"} or None, default=None
            Strategy for updating with new point observations:
            - "mean": use the mean of the interval bounds as point observation
            - "median": use the median of the interval bounds as point observation
            - "point": use the point forecast directly (if available)
            If None, defaults to "mean".
        panel_group_names : list of str or None, default=None
            Group prefixes for panel data:
            - If None: predict for all groups
            - If list of str: update and predict only for the specified panel groups
            Parameter is ignored if the forecaster was not fitted on panel data.
        stride : int >= 1 or None, default=None
            Number of new observations to use for each update. If None, uses
            ``fit_forecasting_horizon_``.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        y, X, panel_group_names = validate_data(
            self,
            y=y,
            X=X,
            reset=False,
            panel_group_names=panel_group_names,
            check_continuity=True,
        )

        forecasting_horizon, _ = self._validate_predict_params(forecasting_horizon, coverage_rates)

        if stride is None:
            stride = self.fit_forecasting_horizon_

        # Initial prediction with predict_transformed parameter
        y_pred_i = self.predict_interval(
            X=X,
            forecasting_horizon=forecasting_horizon,
            coverage_rates=coverage_rates,
            panel_group_names=panel_group_names,
            strategy=strategy,
            **params,
        )

        y_pred = y_pred_i
        for i in range(0, len(y), stride):
            y_slice = y[i : i + stride]

            X_slice = None
            if X is not None:
                # Filter X to match y_slice times
                # Use semi-join to f ilter X rows that have matching times in y_slice
                X_slice = X.join(y_slice.select("time"), on="time", how="semi")

            self.update(y=y_slice, X=X_slice, panel_group_names=panel_group_names)

            X_future = None
            if X is not None:
                # Filter X to start after the last observed time
                # This ensures predict() gets features aligned with the forecast horizon
                last_time = y_slice["time"][-1]
                X_future = X.filter(pl.col("time") > last_time)

            y_pred_i = self.predict_interval(
                X=X_future,
                forecasting_horizon=forecasting_horizon,
                coverage_rates=coverage_rates,
                panel_group_names=panel_group_names,
                strategy=strategy,
                **params,
            )

            y_pred = pl.concat([y_pred, y_pred_i])

        return y_pred

    @override
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
        raise NotImplementedError()
