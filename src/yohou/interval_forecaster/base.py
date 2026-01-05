"""Base classes for interval forecasters and similarity measures."""

import abc
from copy import deepcopy
from typing import Any

import numpy as np
import polars as pl
import polars.selectors as cs
from pydantic import StrictInt
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseForecaster, BaseTransformer
from yohou.utils import filter_panel_columns


class BaseSimilarity(BaseEstimator, metaclass=abc.ABCMeta):
    """Base class for similarity measures used in interval forecasting."""

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
    coverage_rates: list of floats, default=[0.05]
        List of miscoverage levels to generate intervals for.

    """

    @property
    def prediction_types(self) -> set[str]:
        """Get the prediction types this forecaster produces.

        Returns
        -------
        set of str
            {"interval"}

        """
        return {"interval"}

    def __init__(
        self,
        coverage_rates: list[float],
        update_strategy: str,
        feature_transformer: BaseTransformer | None = None,
    ) -> None:
        super().__init__(
            feature_transformer=feature_transformer,
            target_transformer=None,
        )
        self.coverage_rates = coverage_rates
        self.update_strategy = update_strategy

    @abc.abstractmethod
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
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

        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self

        """
        raise NotImplementedError()

    def update(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> "BaseIntervalForecaster":
        """Updates the forecaster with more recent data and
        returns it.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        X : pl.DataFrame or None
            Exogenous feature time series.


        Returns
        -------
        self

        """
        # Check if y contains point data (original target) vs interval data (predictions)
        # For panel data, columns are prefixed (e.g., "x__a"), so check with prefixes
        if self.local_group_names_ is not None:
            # Panel data: check if any group's columns exist
            y_contains_points = any(
                f"{group}__{col}" in y.columns
                for group in self.local_group_names_
                for col in self.local_y_schema_.keys()
            )
        else:
            # Global data: direct column name check
            y_contains_points = set(list(self.local_y_schema_.keys())) <= set(y.columns)

        if "point" in self.prediction_types or y_contains_points:
            # Point data: select time and target columns
            if self.local_group_names_ is not None:
                # Panel data: select prefixed columns
                target_cols = [
                    f"{group}__{col}"
                    for group in self.local_group_names_
                    for col in self.local_y_schema_.keys()
                ]
            else:
                # Global data: select unprefixed columns
                target_cols = list(self.local_y_schema_.keys())
            y = y.select(["time"] + target_cols)

        else:
            time = y.select(cs.by_name("time"))

            match self.update_strategy:
                case "average":
                    if self.local_group_names_ is not None:
                        y_groups_list = []
                        for local_group_name in self.local_group_names_:
                            # Select columns for this group
                            group_cols = [c for c in y.columns if c.startswith(f"{local_group_name}__")]
                            y_local = y.select(group_cols)

                            # Build expressions using actual column names (with prefixes)
                            # For each unprefixed col in schema, find matching prefixed columns
                            y_local = y_local.select(
                                [
                                    pl.concat_list(
                                        [
                                            f"{local_group_name}__{col}_lower_{coverage_rate}"
                                            for coverage_rate in self.coverage_rates
                                        ]
                                        + [
                                            f"{local_group_name}__{col}_upper_{coverage_rate}"
                                            for coverage_rate in self.coverage_rates
                                        ]
                                    )
                                    .list.mean()
                                    .alias(f"{local_group_name}__{col}")
                                    for col in list(self.local_y_schema_.keys())
                                ]
                            )

                            y_groups_list.append(y_local)

                        y = pl.concat(y_groups_list, how="horizontal")

                    else:
                        y = y.select(
                            [
                                pl.concat_list(
                                    [
                                        f"{col}_lower_{coverage_rate}"
                                        for coverage_rate in self.coverage_rates
                                    ]
                                    + [
                                        f"{col}_upper_{coverage_rate}"
                                        for coverage_rate in self.coverage_rates
                                    ]
                                )
                                .list.mean()
                                .alias(col)
                                for col in list(self.local_y_schema_.keys())
                            ]
                        )

                case "constant":
                    y_old = self._y_observed[[-1]].select(~cs.by_name("time"))
                    y = pl.concat([y_old] * len(time))

            y = pl.concat([time, y], how="horizontal")

        BaseForecaster.update(self, y, X)

        return self

    def predict(
        self,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        panel_group: str | None = None,
        predict_transformed: bool = False,
        **params,
    ) -> pl.DataFrame:
        """Predicts the model forecasting horizon from the observation horizon.

        Parameters
        ----------
        X : pl.DataFrame or None, default=None
            Exogenous feature time series.

        forecasting_horizon : int >= 1 or None, default=None
            Horizon to forecast. If None, uses ``fit_forecasting_horizon_``.

        panel_group : str or None, default=None
            For panel data (local_group_names_ is not None):
            - If None: predict for all groups (default behavior)
            - If str: predict only for the specified group (cross-learning)
            For global data: parameter is ignored.

        predict_transformed : bool, default=False
            If ``True``, the predictions are returned in the transformed space.

        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Predicted time series with interval bounds.

        """
        check_is_fitted(self, "fit_forecasting_horizon_")

        # Use fit_forecasting_horizon_ as default
        if forecasting_horizon is None:
            forecasting_horizon = self.fit_forecasting_horizon_

        # Validate panel_group only if provided
        if panel_group is not None and (
            self.local_group_names_ is None or panel_group not in self.local_group_names_
        ):
            raise ValueError(
                f"Group {panel_group} not found in local groups: {self.local_group_names_}"
            )

        # Handle panel data: predict all panel groups if panel_group=None
        # For now, just predict all groups together (default behavior)
        # TODO: Implement individual group predictions if needed

        forecaster = deepcopy(self)

        if self.local_group_names_ is not None and panel_group is not None:
            # Filter _y_observed
            if forecaster._y_observed is not None:
                forecaster._y_observed = filter_panel_columns(
                    forecaster._y_observed,
                    panel_group,
                    self.local_group_names_,
                    include_global=False,
                )

            # Filter X
            if X is not None:
                X = filter_panel_columns(
                    X,
                    panel_group,
                    self.local_group_names_,
                    include_global=True,
                )

        y_pred = pl.DataFrame()
        for step in range(1, forecasting_horizon + 1, self.fit_forecasting_horizon_):
            y_pred_step, y_pred_step_inv = BaseForecaster._predict(forecaster, panel_group)

            # Choose which version to accumulate based on predict_transformed
            if predict_transformed:
                y_pred = pl.concat([y_pred, y_pred_step])
            else:
                y_pred = pl.concat([y_pred, y_pred_step_inv])

            if step + self.fit_forecasting_horizon_ <= forecasting_horizon:
                time = y_pred_step.select(cs.by_name("time"))

                # Compute midpoints from intervals for recursive update
                if self.local_group_names_ is None:
                    # Global data case
                    y_data = {"time": time["time"]}
                    for col in list(self.local_y_schema_.keys()):
                        # Find all coverage rates for this column
                        lower_cols = [
                            c for c in y_pred_step_inv.columns if c.startswith(f"{col}_lower_")
                        ]
                        upper_cols = [
                            c for c in y_pred_step_inv.columns if c.startswith(f"{col}_upper_")
                        ]

                        if lower_cols and upper_cols:
                            # Use the first coverage rate to compute midpoint
                            lower_col = sorted(lower_cols)[0]
                            upper_col = sorted(upper_cols)[0]
                            y_data[col] = (
                                y_pred_step_inv[lower_col] + y_pred_step_inv[upper_col]
                            ) / 2
                    y = pl.DataFrame(y_data)
                else:
                    # Panel data case: compute midpoints for each group
                    y_parts = [time]
                    for local_group_name in self.local_group_names_:
                        # Select columns for this group
                        group_cols = [c for c in y_pred_step_inv.columns if c.startswith(f"{local_group_name}__")]
                        y_local_pred = y_pred_step_inv.select(group_cols)

                        y_local_data = {}
                        for col in list(self.local_y_schema_.keys()):
                            # Find coverage rate columns for this target (with prefix)
                            lower_cols = [
                                c for c in y_local_pred.columns if c.startswith(f"{local_group_name}__{col}_lower_")
                            ]
                            upper_cols = [
                                c for c in y_local_pred.columns if c.startswith(f"{local_group_name}__{col}_upper_")
                            ]

                            if lower_cols and upper_cols:
                                # Use the first coverage rate to compute midpoint
                                lower_col = sorted(lower_cols)[0]
                                upper_col = sorted(upper_cols)[0]
                                # Store with prefix to avoid duplicate column names
                                y_local_data[f"{local_group_name}__{col}"] = (
                                    y_local_pred[lower_col] + y_local_pred[upper_col]
                                ) / 2

                        y_parts.append(pl.DataFrame(y_local_data))

                    y = pl.concat(y_parts, how="horizontal")

                X_slice = None
                if X is not None:
                    start_idx = step - 1
                    end_idx = start_idx + self.fit_forecasting_horizon_
                    X_slice = X[start_idx:end_idx]

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
