"""Implementation of seasonal naive forecasters."""

import numbers
from typing import Literal

import polars as pl
import polars.selectors as cs
from pydantic import StrictInt

from yohou.utils._compat import Interval, _fit_context

from ..utils.tags import Tags
from .base import BasePointForecaster

__all__ = ["MeanSeasonalNaive", "SeasonalNaive"]


class SeasonalNaive(BasePointForecaster):
    """Seasonal naive forecaster that repeats values from previous season.

    Parameters
    ----------
    seasonality : int, default=1
        The seasonal period length. For example, 7 for weekly seasonality
        in daily data, or 12 for monthly seasonality in monthly data.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data. See `BaseForecaster` for details.

    Attributes
    ----------
    interval_ : str
        Detected time interval of the training data.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.point import SeasonalNaive
    >>>
    >>> df = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         start=datetime(2021, 1, 1),
    ...         end=datetime(2021, 1, 10),
    ...         interval="1d",
    ...         eager=True,
    ...     ),
    ...     "value": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0],
    ... })
    >>> forecaster = SeasonalNaive(seasonality=3)
    >>> _ = forecaster.fit(y=df, forecasting_horizon=3)
    >>> y_pred = forecaster.predict(forecasting_horizon=3)
    >>> len(y_pred)
    3

    Notes
    -----
    Predictions repeat the last ``seasonality`` observed values
    cyclically.  For example, with ``seasonality=7`` the forecast for
    each day equals the observation from the same weekday in the last
    observed week.

    See Also
    --------
    `MeanSeasonalNaive` : Averages multiple past seasonal cycles.
    `PointReductionForecaster` : ML-based point forecaster.

    """

    _parameter_constraints: dict = {
        **BasePointForecaster._parameter_constraints,
        "seasonality": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        seasonality: StrictInt = 1,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ):
        BasePointForecaster.__init__(
            self,
            feature_transformer=None,
            target_transformer=None,
            target_as_feature=None,
            panel_strategy=panel_strategy,
        )

        self.seasonality = seasonality

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        tags.forecaster_tags.ignores_exogenous = True
        tags.forecaster_tags.stateful = True
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> "BasePointForecaster":
        """Fit the forecaster to historical data.

        Stores the last ``seasonality`` observations for naive repetition.

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
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted forecaster instance.

        """
        self._observation_horizon = self.seasonality

        BasePointForecaster.fit(
            self,
            y=y,
            X=X,
            forecasting_horizon=forecasting_horizon,
            **params,
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
        # Non-panel data
        if self.groups_ is None:
            assert isinstance(self._y_observed, pl.DataFrame)
            y_pred = self._y_observed.select(~cs.by_name("time"))
            if self.fit_forecasting_horizon_ > self.seasonality:
                # Number of full repetitions needed
                n_repeats = int((self.fit_forecasting_horizon_ + self.seasonality - 1) // self.seasonality)
                y_pred = pl.concat([y_pred] * n_repeats)

            y_pred = y_pred.head(self.fit_forecasting_horizon_)

        # Panel data
        else:
            assert isinstance(self._y_observed, dict)
            y_pred = []
            for panel_group_name in groups:
                y_group = self._y_observed[panel_group_name]
                assert isinstance(y_group, pl.DataFrame)
                y_pred_group = y_group.select(~cs.by_name("time"))

                if self.fit_forecasting_horizon_ > self.seasonality:
                    # Number of full repetitions needed
                    n_repeats = (self.fit_forecasting_horizon_ + self.seasonality - 1) // self.seasonality
                    y_pred_group = pl.concat([y_pred_group] * n_repeats)

                y_pred_group = y_pred_group.head(self.fit_forecasting_horizon_)

                # Rename columns to add panel prefix
                y_pred_group = y_pred_group.rename({col: f"{panel_group_name}__{col}" for col in y_pred_group.columns})

                y_pred.append(y_pred_group)

            # Concatenate horizontally (side by side)
            y_pred = pl.concat(y_pred, how="horizontal")

        y_pred = self._add_time_columns(y_pred)

        return y_pred


class MeanSeasonalNaive(BasePointForecaster):
    """Seasonal naive forecaster that averages values across past seasons.

    Instead of repeating only the last seasonal cycle (as ``SeasonalNaive``
    does), this forecaster averages the same position across ``n_seasons``
    past cycles.  For example, with ``seasonality=7`` and ``n_seasons=3``,
    the forecast for Monday is the mean of the last three observed Mondays.

    Parameters
    ----------
    seasonality : int, default=1
        The seasonal period length. For example, 7 for weekly seasonality
        in daily data, or 12 for monthly seasonality in monthly data.
    n_seasons : int, default=1
        Number of past seasonal cycles to average over. When set to 1, the
        behaviour is identical to ``SeasonalNaive``.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data. See ``BaseForecaster`` for details.

    Attributes
    ----------
    interval_ : str
        Detected time interval of the training data.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.point import MeanSeasonalNaive
    >>>
    >>> df = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         start=datetime(2021, 1, 1),
    ...         end=datetime(2021, 1, 12),
    ...         interval="1d",
    ...         eager=True,
    ...     ),
    ...     "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    ... })
    >>> forecaster = MeanSeasonalNaive(seasonality=3, n_seasons=2)
    >>> _ = forecaster.fit(y=df, forecasting_horizon=3)
    >>> y_pred = forecaster.predict(forecasting_horizon=3)
    >>> len(y_pred)
    3

    Notes
    -----
    The forecaster stores the last ``seasonality * n_seasons`` observations.
    These are reshaped into ``n_seasons`` groups of ``seasonality`` values
    and the arithmetic mean is computed per position.  The resulting pattern
    is repeated cyclically to fill the forecasting horizon.

    When ``n_seasons=1`` the output is identical to ``SeasonalNaive`` and
    the original column dtype is preserved.  When ``n_seasons > 1`` the
    averaging produces ``Float64`` columns.

    See Also
    --------
    `SeasonalNaive` : Repeats the last seasonal cycle without averaging.
    `PointReductionForecaster` : ML-based point forecaster.

    """

    _parameter_constraints: dict = {
        **BasePointForecaster._parameter_constraints,
        "seasonality": [Interval(numbers.Integral, 1, None, closed="left")],
        "n_seasons": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        seasonality: StrictInt = 1,
        n_seasons: StrictInt = 1,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ):
        BasePointForecaster.__init__(
            self,
            feature_transformer=None,
            target_transformer=None,
            target_as_feature=None,
            panel_strategy=panel_strategy,
        )

        self.seasonality = seasonality
        self.n_seasons = n_seasons

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        tags.forecaster_tags.ignores_exogenous = True
        tags.forecaster_tags.stateful = True
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> "BasePointForecaster":
        """Fit the forecaster to historical data.

        Stores the last ``seasonality * n_seasons`` observations for
        averaging across seasonal cycles.

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
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted forecaster instance.

        Raises
        ------
        ValueError
            If ``y`` has fewer rows than ``seasonality * n_seasons``.

        """
        self._observation_horizon = self.seasonality * self.n_seasons

        BasePointForecaster.fit(
            self,
            y=y,
            X=X,
            forecasting_horizon=forecasting_horizon,
            **params,
        )

        return self

    def _compute_mean_pattern(self, y_values: pl.DataFrame) -> pl.DataFrame:
        """Compute the mean seasonal pattern from observed values.

        Parameters
        ----------
        y_values : pl.DataFrame
            Observed values (excluding the "time" column) with exactly
            ``seasonality * n_seasons`` rows.

        Returns
        -------
        pl.DataFrame
            DataFrame with ``seasonality`` rows containing the averaged
            pattern.

        """
        if self.n_seasons == 1:
            return y_values

        result = {}
        for col in y_values.columns:
            values = y_values[col].to_numpy().reshape(self.n_seasons, self.seasonality)
            result[col] = values.mean(axis=0).tolist()
        return pl.DataFrame(result)

    def _predict_one(
        self,
        groups: list[str],
        **params,
    ) -> pl.DataFrame:
        """Predict ``fit_forecasting_horizon_`` steps from the observation horizon.

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
        # Non-panel data
        if self.groups_ is None:
            assert isinstance(self._y_observed, pl.DataFrame)
            y_values = self._y_observed.select(~cs.by_name("time"))
            y_pred = self._compute_mean_pattern(y_values)

            if self.fit_forecasting_horizon_ > self.seasonality:
                n_repeats = int((self.fit_forecasting_horizon_ + self.seasonality - 1) // self.seasonality)
                y_pred = pl.concat([y_pred] * n_repeats)

            y_pred = y_pred.head(self.fit_forecasting_horizon_)

        # Panel data
        else:
            assert isinstance(self._y_observed, dict)
            y_pred = []
            for panel_group_name in groups:
                y_group = self._y_observed[panel_group_name]
                assert isinstance(y_group, pl.DataFrame)
                y_values = y_group.select(~cs.by_name("time"))
                y_pred_group = self._compute_mean_pattern(y_values)

                if self.fit_forecasting_horizon_ > self.seasonality:
                    n_repeats = (self.fit_forecasting_horizon_ + self.seasonality - 1) // self.seasonality
                    y_pred_group = pl.concat([y_pred_group] * n_repeats)

                y_pred_group = y_pred_group.head(self.fit_forecasting_horizon_)

                # Rename columns to add panel prefix
                y_pred_group = y_pred_group.rename({col: f"{panel_group_name}__{col}" for col in y_pred_group.columns})

                y_pred.append(y_pred_group)

            # Concatenate horizontally (side by side)
            y_pred = pl.concat(y_pred, how="horizontal")

        y_pred = self._add_time_columns(y_pred)

        return y_pred
