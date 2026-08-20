"""Abstract base class for trend and seasonality forecasters."""

import numbers
from datetime import datetime
from typing import Literal
from typing import cast as typing_cast

import numpy as np
import polars as pl
from pydantic import StrictInt
from sklearn.base import RegressorMixin, clone
from sklearn.pipeline import Pipeline

from yohou.base import BaseActualTransformer
from yohou.point import BasePointForecaster
from yohou.utils._compat import StrOptions
from yohou.utils.panel import get_group_df
from yohou.utils.tags import Tags
from yohou.utils.validation import interval_to_timedelta


class _BaseTrendForecaster(BasePointForecaster):
    """Abstract base class for trend forecasters.

    Provides common infrastructure for trend-based forecasting methods,
    including data validation and a multi-step prediction interface (one
    block of ``fit_forecasting_horizon_`` steps per call).

    Parameters
    ----------
    target_transformer : BaseActualTransformer, optional
        Transformer applied to target before forecasting.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data.  See `BaseForecaster` for details.

    """

    _parameter_constraints: dict = {
        **BasePointForecaster._parameter_constraints,
    }

    def __init__(
        self,
        target_transformer: BaseActualTransformer | None = None,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ):
        """Initialize _BaseTrendForecaster.

        Parameters
        ----------
        target_transformer : BaseActualTransformer, optional
            Transformer for target variable.
        panel_strategy : {"global", "multivariate"}, default="global"
            How to handle panel data.  See `BaseForecaster` for details.

        """
        super().__init__(
            target_transformer=target_transformer,
            target_as_feature=None,
            panel_strategy=panel_strategy,
        )

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        tags.forecaster_tags.requires_exogenous = False
        return tags

    def _pre_fit(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> tuple[pl.DataFrame | dict[str, pl.DataFrame], pl.DataFrame | dict[str, pl.DataFrame] | None]:
        """Preprocess and transform inputs before fitting.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X_actual : pl.DataFrame or None, default=None
            Features time series.
        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.
        X_future : pl.DataFrame or None, default=None
            Known future features.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts. See ``fit()`` for full parameter
            description.

        Returns
        -------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target.
        X_t : pl.DataFrame or dict[str, pl.DataFrame] or None
            Transformed features.

        Notes
        -----
        Sets ``self._first_observed_time`` to the first observed datetime (a
        dict keyed by group name in panel mode) as a side effect; it is read
        by ``_get_time_indices`` to compute relative time indices on every
        predict call and restored by ``rewind``.

        """
        y_t, X_t = super()._pre_fit(
            y=y,
            X_actual=X_actual,
            forecasting_horizon=forecasting_horizon,
            X_future=X_future,
            X_forecast=X_forecast,
        )

        # Panel data
        if self.groups_ is not None:
            assert isinstance(y_t, dict)
            self._first_observed_time = {group: y_t[group]["time"][0] for group in self.groups_}

        # Non-panel data
        else:
            assert isinstance(y_t, pl.DataFrame)
            self._first_observed_time = y_t["time"][0]

        return y_t, X_t

    def rewind(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> "_BaseTrendForecaster":
        """Rewinds the forecaster by rewinding the observation horizon.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X_actual : pl.DataFrame or None
            Actual feature observations to restore the observation
            state to. Must align with ``y``.
        groups : list of str or None, default=None
            Group prefixes for panel data:
            - If None: rewinds observation state for all fitted groups
            - If list of str: rewinds only the specified panel groups
            Parameter is ignored if the forecaster was not fitted on panel data.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns. Vintage times do not need to align exactly with
            observation times; the latest vintage at or before each
            observation time is selected automatically (as-of matching).

        Returns
        -------
        self

        """
        super().rewind(y=y, X_actual=X_actual, groups=groups, X_future=X_future, X_forecast=X_forecast)

        if groups is None:
            groups = self.groups_

        # Panel data: each group uses its own transformer's observation
        # horizon, not the first group's (groups may differ).
        if groups is not None:
            first_observed_time = {
                group: get_group_df(df=y, group_name=group, schema=self.local_y_schema_)["time"][
                    self._group_target_observation_horizon(group)
                ]
                for group in groups
            }
            self._first_observed_time |= first_observed_time

        # Non-panel data
        else:
            self._first_observed_time = y["time"][self._group_target_observation_horizon(None)]

        return self

    def _group_target_observation_horizon(self, group: str | None) -> int:
        """Observation horizon of the target transformer for one group.

        Returns the group's own transformer horizon in panel mode (where
        ``target_transformer_`` is a dict), the scalar transformer's horizon in
        non-panel mode, or 0 when there is no target transformer.
        """
        if self.target_transformer_ is None:
            return 0
        if isinstance(self.target_transformer_, dict):
            # The dict form only exists in panel mode, where group is a real name.
            assert group is not None
            transformer_dict = typing_cast("dict[str, BaseActualTransformer | None]", self.target_transformer_)
            transformer = transformer_dict[group]
            if transformer is None:
                return 0
            return transformer.observation_horizon
        return self.target_transformer_.observation_horizon

    def _get_time_indices(
        self, forecasting_horizon: int | None = None, panel_group_name: str | None = None
    ) -> pl.Series:
        """Generate monotonically increasing integer time-step indices.

        Indices are measured relative to ``_first_observed_time`` and continue
        from the current position (the number of steps observed so far).

        Parameters
        ----------
        forecasting_horizon : int or None, default=None
            Number of steps to predict. When an int, the method returns the
            prediction indices ``[current_time_index, current_time_index +
            forecasting_horizon)``. When ``None``, it returns the historical
            training-time indices ``[0, current_time_index)`` used during
            ``_fit_estimator``.
        panel_group_name : str or None
            Panel group name for which to get time indices.

        Returns
        -------
        pl.Series
            Time step indices. If ``forecasting_horizon`` is an int, the
            prediction indices ``[current_time_index, current_time_index +
            forecasting_horizon)``; if ``None``, the historical training
            indices ``[0, current_time_index)``.

        """
        if panel_group_name is not None:
            assert isinstance(self.observed_time_, dict)
            assert isinstance(self._first_observed_time, dict)
            observed_time = self.observed_time_[panel_group_name]
            first_observed_time = self._first_observed_time[panel_group_name]

        else:
            assert isinstance(self.observed_time_, datetime)
            assert isinstance(self._first_observed_time, datetime)
            observed_time = self.observed_time_
            first_observed_time = self._first_observed_time

        # Number of steps from first_observed_time to observed_time, inclusive
        # of both endpoints (matching datetime_range(...).len()). For fixed
        # intervals this is computed arithmetically to avoid materialising a
        # Series proportional to the observation history.
        step = interval_to_timedelta(self.interval_)
        if step is not None and step.total_seconds() > 0:
            elapsed = (observed_time - first_observed_time).total_seconds()
            current_time_index = int(elapsed // step.total_seconds()) + 1
        else:
            current_time_index = pl.datetime_range(
                start=first_observed_time,
                end=observed_time,
                interval=self.interval_,
                eager=True,
            ).len()

        if forecasting_horizon is not None:
            indices = pl.arange(
                current_time_index,
                current_time_index + forecasting_horizon,
                eager=True,
            )

        else:
            indices = pl.arange(
                0,
                current_time_index,
                eager=True,
            )

        return indices

    def _prepare_data(self, y: pl.DataFrame | dict[str, pl.DataFrame], panel_group_name: str | None = None) -> tuple:
        """Prepare time indices and target array for a single group.

        Parameters
        ----------
        y : pl.DataFrame or dict[str, pl.DataFrame]
            Target data for a single group (DataFrame) or all groups (dict).
        panel_group_name : str or None
            Panel group name for which to get time indices.

        Returns
        -------
        tuple
            (X_time_indices, y_array) ready for model fitting.
        """
        X_time_indices = self._get_time_indices(panel_group_name=panel_group_name).to_numpy().reshape(-1, 1)
        if panel_group_name is not None:
            assert isinstance(y, dict)
            y_group = y[panel_group_name]
            assert isinstance(y_group, pl.DataFrame)
            y_array = y_group.select([col for col in y_group.columns if col != "time"]).to_numpy()
        else:
            assert isinstance(y, pl.DataFrame)
            y_array = y.select([col for col in y.columns if col != "time"]).to_numpy()

        return X_time_indices, y_array

    def _fit_estimator(self, estimator: RegressorMixin | Pipeline, y_t: pl.DataFrame | dict[str, pl.DataFrame]) -> None:
        """Fit the underlying estimator on prepared time series data.

        Parameters
        ----------
        estimator : RegressorMixin or Pipeline
            The sklearn estimator or pipeline to fit.
        y_t : pl.DataFrame or dict of str to pl.DataFrame
            Transformed target time series, either a single DataFrame or a dict of panel-group DataFrames.

        Notes
        -----
        Panel data is always handled with a single pooled estimator: all
        group time series are vertically stacked into one training set
        before fitting, regardless of ``panel_strategy``.

        """
        # Non-panel data
        if self.groups_ is None:
            X_time_indices, y_array = self._prepare_data(y_t)
            self.estimator_ = clone(estimator)
            self.estimator_.fit(X_time_indices, y_array)

        # Panel data: pooled estimator (global strategy)
        else:
            X_time_indices, y_array = [], []
            for panel_group_name in self.groups_:
                X_time_indices_group, y_array_group = self._prepare_data(y_t, panel_group_name=panel_group_name)
                X_time_indices.append(X_time_indices_group)
                y_array.append(y_array_group)

            # Stack all groups
            X_time_indices = np.vstack(X_time_indices)
            y_array = np.vstack(y_array)

            self.estimator_ = clone(estimator)
            self.estimator_.fit(X_time_indices, y_array)

    def _predict_one(
        self,
        groups: list[str],
        **params,
    ) -> pl.DataFrame:
        """Predicts `fit_forecasting_horizon_` steps from the observation horizon.

        Parameters
        ----------
        groups : list of str
            Panel group names to predict for.

        **params : dict
            Accepted for signature compatibility and ignored: this transformer
            routes no metadata to nested estimators.

        Returns
        -------
        pl.DataFrame
            Predicted time series with ``vintage_time``, ``time``, and one
            value column per target variable. Panel output has
            group-prefixed value columns (``group__col``) with the
            ``vintage_time`` / ``time`` pair present once.

        """
        y_t_columns = list(self.local_y_t_schema_.keys())

        # Non-panel data
        if self.groups_ is None:
            X_time_indices_pred = self._get_time_indices(self.fit_forecasting_horizon_).to_numpy().reshape(-1, 1)
            y_pred_array = self.estimator_.predict(X_time_indices_pred)
            y_pred_array = y_pred_array.reshape(-1, len(y_t_columns))

            # Convert to polars DataFrame with correct column names
            y_pred = pl.DataFrame({col: y_pred_array[:, i] for i, col in enumerate(y_t_columns)})

        # Panel data
        else:
            y_pred = []
            for panel_group_name in groups:
                X_time_indices_pred = (
                    self
                    ._get_time_indices(self.fit_forecasting_horizon_, panel_group_name=panel_group_name)
                    .to_numpy()
                    .reshape(-1, 1)
                )

                # Use pooled estimator for all groups
                estimator_group = self.estimator_

                # Predict using model
                y_pred_array = estimator_group.predict(X_time_indices_pred)
                y_pred_array = y_pred_array.reshape(-1, len(y_t_columns))

                # Convert to polars DataFrame with unprefixed column names
                y_pred_group = pl.DataFrame({
                    f"{panel_group_name}__{col}": y_pred_array[:, i] for i, col in enumerate(y_t_columns)
                })
                y_pred.append(y_pred_group)

            y_pred = pl.concat(y_pred, how="horizontal")

        y_pred = self._add_time_columns(y_pred)

        return y_pred


class _BaseSeasonalityForecaster(_BaseTrendForecaster):
    """Abstract base class for seasonality forecasters.

    Provides common infrastructure for pattern-based and Fourier-based
    seasonality forecasting, including time-to-phase conversion, phase
    tracking, and data validation.

    Parameters
    ----------
    seasonality : int or float
        Length of seasonal cycle (number of time steps).
        ``PatternSeasonalityForecaster`` requires an integer period, while
        ``FourierSeasonalityForecaster`` accepts a float (e.g. ``365.25``).
    target_transformer : BaseActualTransformer, optional
        Transformer applied to target before forecasting.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data.  See `BaseForecaster` for details.

    """

    _parameter_constraints: dict = {
        "seasonality": [numbers.Real],
        "target_transformer": [BaseActualTransformer, None],
        "panel_strategy": [StrOptions({"global", "multivariate"})],
    }

    def __init__(
        self,
        seasonality: float,
        target_transformer: BaseActualTransformer | None = None,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ):
        """Initialize _BaseSeasonalityForecaster.

        Parameters
        ----------
        seasonality : int or float
            Length of seasonal cycle.
        target_transformer : BaseActualTransformer, optional
            Transformer for target variable.
        panel_strategy : {"global", "multivariate"}, default="global"
            How to handle panel data.  See `BaseForecaster` for details.

        """
        super().__init__(target_transformer=target_transformer, panel_strategy=panel_strategy)
        self.seasonality = seasonality

    def _validate_sufficient_data(self, y: pl.DataFrame | dict[str, pl.DataFrame]) -> None:
        """Validate that y has at least one complete seasonal cycle.

        Parameters
        ----------
        y : pl.DataFrame or dict[str, pl.DataFrame]
            Target time series (DataFrame for global data, dict for panel data).

        Raises
        ------
        ValueError
            If y has fewer than seasonality rows.

        """
        # Handle panel data (dict of DataFrames)
        if isinstance(y, dict):
            for group_name, group_df in y.items():
                assert isinstance(group_df, pl.DataFrame)
                if len(group_df) < self.seasonality:
                    raise ValueError(
                        f"Insufficient data for group '{group_name}': need at least "
                        f"{self.seasonality} observations (one seasonal cycle), got {len(group_df)}"
                    )
        # Handle global data (single DataFrame)
        elif len(y) < self.seasonality:
            raise ValueError(
                f"Insufficient data: need at least {self.seasonality} observations (one seasonal cycle), got {len(y)}"
            )
