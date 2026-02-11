"""Base class for reduction-based forecasters."""

import abc
import inspect
from collections.abc import Callable
from typing import Any, Literal
from typing import cast as typing_cast

import numpy as np
import polars as pl
import polars.selectors as cs
from pydantic import StrictInt
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression
from sklearn.utils.metadata_routing import MetadataRouter, MethodMapping

from yohou.base.forecaster import BaseForecaster
from yohou.base.transformer import BaseTransformer
from yohou.utils import Tags, cast, tabularize, validate_callable_signature

__all__ = ["BaseReductionForecaster"]


class BaseReductionForecaster(BaseForecaster, metaclass=abc.ABCMeta):
    """Base class for forecasters using reduction to supervised learning.

    Converts the time series forecasting task to a tabular one.

    Parameters
    ----------
    estimator : instance of `BaseEstimator`, default=LinearRegression()
        Estimator used to fit the tabularized data.
    reduction_strategy : {"direct", "multi-output"}, default="multi-output"
        Reduction strategy to use.
    target_as_feature : {"transformed", "raw"} or None, default="transformed"
        Controls whether the target is included as a feature.
        ``"transformed"`` includes the transformed target, ``"raw"``
        includes the raw target, and ``None`` uses only exogenous features.
    target_transformer : instance of `BaseTransformer` or None, default=None
        Transformer used to transform the target time series into the new target.
    feature_transformer : instance of `BaseTransformer` or None, default=None
        Transformer used to transform the target time series into features.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data. See `BaseForecaster` for details.

    Notes
    -----
    Reduction strategies:
    - Direct: Separate model for each horizon step; predicts directly from inputs.
    - Multi-output: Single model predicts all horizon steps simultaneously.

    All models can be applied recursively for multi-step forecasting by specifying
    the forecasting horizon during prediction.

    See Also
    --------
    [point.PointReductionForecaster][] : Point forecaster using
    reduction
    [interval.IntervalReductionForecaster][] : Interval forecaster
    using reduction
    """

    def __init__(
        self,
        estimator: BaseEstimator = LinearRegression(),
        reduction_strategy: Literal["direct", "multi-output"] = "multi-output",
        target_as_feature: Literal["transformed", "raw"] | None = "transformed",
        target_transformer: BaseTransformer | None = None,
        feature_transformer: BaseTransformer | None = None,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ):
        BaseForecaster.__init__(
            self,
            target_as_feature=target_as_feature,
            target_transformer=target_transformer,
            feature_transformer=feature_transformer,
            panel_strategy=panel_strategy,
        )

        self.estimator = estimator
        self.reduction_strategy = reduction_strategy

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

        return tags

    def _process_time_weight_to_sample_weight(
        self,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        time_weight: Callable | pl.DataFrame | None,
        sample_weight_alignment: str,
        forecasting_horizon: int,
    ) -> np.ndarray | None:
        """Convert time_weight to sklearn sample_weight for tabularized data.

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target time series (global or panel data).
        time_weight : callable or pl.DataFrame or None
            Time weighting specification.
        sample_weight_alignment : {"first_step", "mean_step", "weighted_mean_step", "max_weight_step", "min_weight_step"}
            Strategy for aligning time weights to tabularized samples.
        forecasting_horizon : int
            Number of forecast steps (determines tabularization window).

        Returns
        -------
        np.ndarray or None
            Sample weights array matching tabularized data rows, or None if
            time_weight is None.

        """
        if time_weight is None:
            return None

        if self.panel_group_names_ is None:
            # Global data: y_t is DataFrame
            assert isinstance(y_t, pl.DataFrame)
            sample_weights = self._compute_sample_weights_one(
                y_t=y_t,
                time_weight=time_weight,
                sample_weight_alignment=sample_weight_alignment,
                forecasting_horizon=forecasting_horizon,
                group_name=None,
            )
        else:
            # Panel data: y_t is dict, stack weights
            assert isinstance(y_t, dict)
            sample_weights_list = []
            for panel_group_name in self.panel_group_names_:
                y_t_local = y_t[panel_group_name]
                weights_local = self._compute_sample_weights_one(
                    y_t=y_t_local,
                    time_weight=time_weight,
                    sample_weight_alignment=sample_weight_alignment,
                    forecasting_horizon=forecasting_horizon,
                    group_name=panel_group_name,
                )
                sample_weights_list.append(weights_local)
            sample_weights = np.concatenate(sample_weights_list)

        return sample_weights

    def _compute_sample_weights_one(
        self,
        y_t: pl.DataFrame,
        time_weight: Callable | pl.DataFrame,
        sample_weight_alignment: str,
        forecasting_horizon: int,
        group_name: str | None,
    ) -> np.ndarray:
        """Compute sample weights for one time series (global or local).

        Parameters
        ----------
        y_t : pl.DataFrame
            Transformed target time series with "time" column.
        time_weight : callable or pl.DataFrame
            Time weighting specification.
        sample_weight_alignment : {"first_step", "mean_step", "weighted_mean_step", "max_weight_step", "min_weight_step"}
            Strategy for aligning time weights to tabularized samples.
        forecasting_horizon : int
            Number of forecast steps.
        group_name : str or None
            Panel group name (for panel-aware callables), or None for global data.

        Returns
        -------
        np.ndarray
            Sample weights for this time series.

        """
        # Get time column from y_t
        time_series = y_t["time"]

        # Compute raw time weights
        if callable(time_weight):
            # Check signature: 1 param (global) or 2 params (panel-aware)
            n_params = validate_callable_signature(time_weight)
            tw = typing_cast("Callable[..., pl.Series]", time_weight)
            weights_series = tw(time_series) if n_params == 1 else tw(time_series, group_name)
        else:
            # DataFrame format
            # Try group-specific column first, fall back to global "weight"
            if group_name is not None:
                weight_col = f"{group_name}_weight"
                if weight_col in time_weight.columns:
                    weights_df = time_weight.select(["time", weight_col])
                    weights_df = weights_df.rename({weight_col: "weight"})
                elif "weight" in time_weight.columns:
                    # Silent fallback to global weight
                    weights_df = time_weight.select(["time", "weight"])
                else:
                    raise ValueError(
                        f"time_weight DataFrame must have either '{weight_col}' or "
                        f"'weight' column for panel group '{group_name}'"
                    )
            else:
                # Global data: use "weight" column
                if "weight" not in time_weight.columns:
                    raise ValueError("time_weight DataFrame must have 'weight' column for global data")
                weights_df = time_weight.select(["time", "weight"])

            # Join with y_t to align weights
            weights_joined = y_t.select(["time"]).join(weights_df, on="time", how="left")
            weights_series = weights_joined["weight"]

        # Align weights to tabularized samples
        # Tabularized data has n_samples = len(y_t) - forecasting_horizon
        # Each sample i corresponds to a window ending at time index i
        n_samples = len(y_t) - forecasting_horizon

        # Determine sample weights based on alignment strategy
        if sample_weight_alignment == "first_step":
            # Weight from first prediction target time
            # Sample i predicts steps [i+1, ..., i+forecasting_horizon]
            # First step is at index i+1
            aligned_indices = np.arange(1, n_samples + 1)
            sample_weights = weights_series[aligned_indices].to_numpy()

        elif sample_weight_alignment == "mean_step":
            # Average weight across all prediction horizon steps
            # Sample i uses mean(weights[i+1], weights[i+2], ..., weights[i+H])
            sample_weights = []
            for i in range(n_samples):
                horizon_weights = weights_series[i + 1 : i + forecasting_horizon + 1].to_numpy()
                sample_weights.append(horizon_weights.mean())
            sample_weights = np.array(sample_weights)

        elif sample_weight_alignment == "weighted_mean_step":
            # Exponentially weighted mean across prediction horizon
            # Recent steps get more weight than distant ones
            sample_weights = []
            # Create exponential decay weights for horizon steps (most recent = highest)
            horizon_decay = np.exp(-np.arange(forecasting_horizon) * 0.5)
            horizon_decay = horizon_decay / horizon_decay.sum()  # Normalize

            for i in range(n_samples):
                horizon_weights = weights_series[i + 1 : i + forecasting_horizon + 1].to_numpy()
                # Weighted average: closer steps more important
                sample_weights.append(np.sum(horizon_weights * horizon_decay))
            sample_weights = np.array(sample_weights)

        elif sample_weight_alignment == "max_weight_step":
            # Maximum weight across prediction horizon
            # Sample i uses max(weights[i+1], weights[i+2], ..., weights[i+H])
            sample_weights = []
            for i in range(n_samples):
                horizon_weights = weights_series[i + 1 : i + forecasting_horizon + 1].to_numpy()
                sample_weights.append(horizon_weights.max())
            sample_weights = np.array(sample_weights)

        elif sample_weight_alignment == "min_weight_step":
            # Minimum weight across prediction horizon
            # Sample i uses min(weights[i+1], weights[i+2], ..., weights[i+H])
            sample_weights = []
            for i in range(n_samples):
                horizon_weights = weights_series[i + 1 : i + forecasting_horizon + 1].to_numpy()
                sample_weights.append(horizon_weights.min())
            sample_weights = np.array(sample_weights)

        else:
            raise ValueError(
                f"Invalid sample_weight_alignment: {sample_weight_alignment}. "
                f"Must be 'first_step', 'mean_step', 'weighted_mean_step', "
                f"'max_weight_step', or 'min_weight_step'."
            )

        # Normalize to preserve sample count (sklearn convention)
        # sum(sample_weights) = n_samples
        weight_sum = sample_weights.sum()
        if weight_sum == 0:
            raise ValueError("Sum of time weights is zero, cannot normalize")
        sample_weights = sample_weights * (n_samples / weight_sum)

        return sample_weights

    def _get_tabularized_dataset(
        self,
        y_t: pl.DataFrame,
        X_t: pl.DataFrame,
        forecasting_horizon: int,
        y_columns: list[str] | None = None,
    ) -> tuple[
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ]:
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
        X_tab : np.ndarray of shape (n_samples, n_features)
            Feature matrix for supervised learning. Excludes "time" column and
            truncates last forecasting_horizon rows (no targets available).
        y_tab : np.ndarray of shape (n_samples, forecasting_horizon * n_targets)
            Target matrix with columns for each (target, step) combination.
            Columns follow pattern: {target}_step_{1}, {target}_step_{2}, ...

        Notes
        -----
        Lag-to-step renaming convention:
        - Input: y with lag_0, lag_1, lag_2, ..., lag_H features
        - For forecasting_horizon=3:
            - lag_1 → step_1 (1-step-ahead target)
            - lag_2 → step_2 (2-step-ahead target)
            - lag_3 → step_3 (3-step-ahead target)
            - lag_0 is the most recent observation (not a target)

        This convention makes it clear that we're predicting future values, not
        explaining historical ones.

        See Also
        --------
        [utils.tabularization.tabularize][] : Creates lagged features

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

        return X_tab.to_numpy(), y_tab.to_numpy()

    def _estimator_fit_one(
        self,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        X_t: pl.DataFrame | dict[str, pl.DataFrame] | None,
        forecasting_horizon: StrictInt,
        time_weight: Callable | pl.DataFrame | None = None,
        sample_weight_alignment: str = "first_step",
        estimator_params: dict[str, Any] | None = None,
        estimator_fit_params: dict[str, Any] | None = None,
    ) -> BaseEstimator:
        """Fit an sklearn estimator on tabularized time series data.

        Converts time series to supervised learning format and trains the estimator.
        Handles both global (single series) and local (panel data) cases, stacking
        panel data vertically for training a single global model.

        Parameters
        ----------
        y_t : pl.DataFrame
            Transformed target time series.
        X_t : pl.DataFrame
            Transformed feature matrix.
        forecasting_horizon : int
            Number of steps to forecast.
        time_weight : callable or pl.DataFrame or None, default=None
            Time weighting function or DataFrame to weight samples.
            Converted to sample_weight during tabularization.
        sample_weight_alignment : {"first_step", "mean_step", "weighted_mean_step", "max_weight_step", "min_weight_step"}, default="first_step"
            Strategy for aligning time weights to tabularized samples.
        estimator_params : dict
            Additional parameters to pass to the estimator's set_params method.
        estimator_fit_params : dict
            Additional parameters to pass to the estimator's fit method.

        Returns
        -------
        BaseEstimator
            Fitted sklearn regressor.

        Notes
        -----
        For panel data (panel_group_names_ is not None):
        - Unnests each panel column
        - Tabularizes each local time series separately
        - Stacks all series vertically (X_tab = vstack of all local X_tabs)
        - Trains single model across all series (global model)

        For global data:
        - Directly tabularizes and fits

        This enables "global forecasting" where patterns learned from multiple
        series can benefit predictions for all series.

        See Also
        --------
        `_get_tabularized_dataset` : Creates supervised learning matrices
        `_estimator_predict_one` : Uses fitted model for prediction

        """
        estimator = clone(self.estimator).set_params(**(estimator_params or {}))

        if self.panel_group_names_ is None:
            # Global time series
            assert isinstance(y_t, pl.DataFrame)
            assert isinstance(X_t, pl.DataFrame)
            X_tab, y_tab = self._get_tabularized_dataset(
                y_t,
                X_t,
                forecasting_horizon,
            )

        else:
            # Panel data: stack all series
            # y_t and X_t are dicts mapping group_name to DataFrames
            assert isinstance(y_t, dict)
            assert isinstance(X_t, dict)
            X_tab_list, y_tab_list = [], []
            for panel_group_name in self.panel_group_names_:
                # Get DataFrames for this group (already have unprefixed columns)
                y_t_local = y_t[panel_group_name]
                X_t_local = X_t[panel_group_name]

                # Get column names (excluding "time") for tabularization
                y_columns = [c for c in y_t_local.columns if c != "time"]

                # Pass the group's DataFrame to tabularize
                X_tab_local, y_tab_local = self._get_tabularized_dataset(
                    y_t_local,
                    X_t_local,
                    forecasting_horizon,
                    y_columns=y_columns,
                )

                X_tab_list.append(X_tab_local)
                y_tab_list.append(y_tab_local)

            X_tab = np.vstack(X_tab_list)
            y_tab = np.vstack(y_tab_list)

        # Process time_weight to sample_weight
        sample_weight = self._process_time_weight_to_sample_weight(
            y_t=y_t,
            time_weight=time_weight,
            sample_weight_alignment=sample_weight_alignment,
            forecasting_horizon=forecasting_horizon,
        )

        # Check if estimator supports sample_weight
        if sample_weight is not None:
            fit_signature = inspect.signature(estimator.fit)
            if "sample_weight" not in fit_signature.parameters:
                raise ValueError(
                    f"Estimator {estimator.__class__.__name__} does not support "
                    f"sample_weight parameter. Cannot use time_weight for training."
                )

        # Fit estimator with sample_weight if provided
        fit_params = estimator_fit_params or {}
        if sample_weight is not None:
            fit_params = {**fit_params, "sample_weight": sample_weight}

        if len(X_tab) == 0:
            raise ValueError(
                "Training dataset is empty (0 samples). This typically occurs when "
                "the feature transformer reduces the data size below the minimum "
                "required for the forecasting horizon. Please check your "
                "transformer settings and ensure sufficient data length."
            )

        estimator.fit(X_tab, y_tab, **fit_params)

        return estimator

    def _estimator_predict_one(
        self,
        estimator: BaseEstimator,
        panel_group_names: list[str],
    ) -> pl.DataFrame:
        """Generate predictions using fitted estimator on tabularized data.

        Parameters
        ----------
        estimator : BaseEstimator
            Fitted scikit-learn estimator.
        panel_group_names : list of str
            Panel group names to predict for.

        Returns
        -------
        pl.DataFrame
            Predictions for the forecasting horizon.

        """
        # Non-panel data
        if self.panel_group_names_ is None:
            # Global data: _X_t_observed is a DataFrame
            assert self._X_t_observed is not None
            assert isinstance(self._X_t_observed, pl.DataFrame)
            X_t = self._X_t_observed[[-1]].select(~cs.by_name("time"))
            assert self.local_X_t_schema_ is not None
            X_tab = X_t.select(list(self.local_X_t_schema_.keys())).to_numpy()
            assert self.local_y_t_schema_ is not None
            y_tab_pred = estimator.predict(X_tab)  # type: ignore[attr-defined]
            y_pred = pl.DataFrame(
                y_tab_pred.reshape(self.fit_forecasting_horizon_, len(list(self.local_y_t_schema_.keys()))),
                schema=list(self.local_y_t_schema_.keys()),
            )
            # Cast to preserve dtypes from transformed target schema
            y_pred = cast(y_pred, self.local_y_t_schema_)

        # Panel data
        else:
            # Panel data: _X_t_observed is a dict (one DataFrame per group)
            # Each DataFrame has unprefixed columns (after get_group_df applied)
            assert self._X_t_observed is not None
            y_pred_dict = {}
            for panel_group_name in panel_group_names:
                # Get X_t for this group (already unprefixed)
                X_t_group = self._X_t_observed[panel_group_name][[-1]].select(~cs.by_name("time"))

                # Use transformed schema to get feature order
                X_tab = X_t_group.select(list(self.local_X_t_schema_.keys())).to_numpy()

                # Get y columns from transformed schema (unprefixed)
                group_y_cols = list(self.local_y_t_schema_.keys())

                y_tab_pred = estimator.predict(X_tab)
                y_pred_local = pl.DataFrame(
                    y_tab_pred.reshape(self.fit_forecasting_horizon_, len(group_y_cols)),
                    schema=group_y_cols,
                )
                # Cast to preserve dtypes from transformed target schema
                y_pred_local = cast(y_pred_local, self.local_y_t_schema_)

                # Re-prefix column names for concatenation (e.g., "a" → "x__a")
                y_pred_local = y_pred_local.rename({col: f"{panel_group_name}__{col}" for col in group_y_cols})

                y_pred_dict[panel_group_name] = y_pred_local

            y_pred = pl.concat(list(y_pred_dict.values()), how="horizontal")

        return y_pred

    def get_metadata_routing(self) -> MetadataRouter:
        """Get metadata routing including wrapped estimator.

        BaseReductionForecaster is a router because it wraps a sklearn estimator.
        It needs to forward metadata (like time_weight) from the forecaster's
        fit() method to the wrapped estimator's fit() method.

        Returns
        -------
        router : MetadataRouter
            Router that forwards to transformers (from parent) and wrapped estimator.
        """
        # Get parent routing (for target_transformer, feature_transformer)
        router = super().get_metadata_routing()

        # Add wrapped sklearn estimator routing
        if hasattr(self, "estimator") and self.estimator is not None:
            router.add(
                estimator=self.estimator,
                method_mapping=MethodMapping().add(caller="fit", callee="fit"),
            )

        return router
