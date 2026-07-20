"""Implementation of seasonal naive forecasters."""

import numbers
from typing import Literal

import polars as pl
import polars.selectors as cs
from pydantic import StrictInt

from yohou.utils._compat import Interval

from ..utils.tags import Tags
from .base import BasePointForecaster

__all__ = ["MeanSeasonalNaive", "SeasonalNaive"]


def _tile_to_horizon(pattern: pl.DataFrame, horizon: int, seasonality: int) -> pl.DataFrame:
    """Repeat a seasonal pattern cyclically to fill the forecasting horizon.

    Parameters
    ----------
    pattern : pl.DataFrame
        Seasonal pattern (value columns only, no ``"time"`` column).
    horizon : int
        Number of rows to produce.
    seasonality : int
        Seasonal period length of ``pattern``.

    Returns
    -------
    pl.DataFrame
        ``pattern`` repeated and trimmed to ``horizon`` rows.

    """
    if horizon > seasonality:
        n_repeats = (horizon + seasonality - 1) // seasonality
        pattern = pl.concat([pattern] * n_repeats)
    return pattern.head(horizon)


def _build_panel_prediction(
    y_observed: dict[str, pl.DataFrame],
    groups: list[str],
    horizon: int,
    seasonality: int,
    get_pattern_fn,
) -> pl.DataFrame:
    """Build panel predictions by tiling a per-group seasonal pattern.

    Parameters
    ----------
    y_observed : dict[str, pl.DataFrame]
        Observed values per panel group.
    groups : list of str
        Panel group names to predict for.
    horizon : int
        Forecasting horizon (number of rows per group).
    seasonality : int
        Seasonal period length.
    get_pattern_fn : callable
        Maps a group's observed DataFrame (value columns only, no ``"time"``)
        to its seasonal pattern DataFrame.

    Returns
    -------
    pl.DataFrame
        Horizontally concatenated, group-prefixed predictions.

    """
    y_pred = []
    for panel_group_name in groups:
        y_group = y_observed[panel_group_name]
        assert isinstance(y_group, pl.DataFrame)
        pattern = get_pattern_fn(y_group.select(~cs.by_name("time")))
        y_pred_group = _tile_to_horizon(pattern, horizon, seasonality)
        y_pred_group = y_pred_group.rename({col: f"{panel_group_name}__{col}" for col in y_pred_group.columns})
        y_pred.append(y_pred_group)
    return pl.concat(y_pred, how="horizontal")


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
    - [`MeanSeasonalNaive`][yohou.point.naive.MeanSeasonalNaive] : Averages multiple past seasonal cycles.
    - [`PointReductionForecaster`][yohou.point.reduction.PointReductionForecaster] : ML-based point forecaster.

    """

    _parameter_constraints: dict = {
        "panel_strategy": BasePointForecaster._parameter_constraints["panel_strategy"],
        "seasonality": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        seasonality: StrictInt = 1,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ):
        BasePointForecaster.__init__(
            self,
            actual_transformer=None,
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
        tags.forecaster_tags.requires_exogenous = False
        tags.forecaster_tags.stateful = True
        return tags

    @property
    def _observation_horizon(self) -> int:
        """Size of the rolling observation buffer.

        ``SeasonalNaive`` retains the last ``seasonality`` rows so that
        ``_predict_one`` can tile the full seasonal pattern over the
        forecast horizon.
        """
        return self.seasonality

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
            pattern = self._y_observed.select(~cs.by_name("time"))
            y_pred = _tile_to_horizon(pattern, self.fit_forecasting_horizon_, self.seasonality)

        # Panel data
        else:
            assert isinstance(self._y_observed, dict)
            y_pred = _build_panel_prediction(
                self._y_observed,
                groups,
                self.fit_forecasting_horizon_,
                self.seasonality,
                lambda values: values,
            )

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
    - [`SeasonalNaive`][yohou.point.naive.SeasonalNaive] : Repeats the last seasonal cycle without averaging.
    - [`PointReductionForecaster`][yohou.point.reduction.PointReductionForecaster] : ML-based point forecaster.

    """

    _parameter_constraints: dict = {
        "panel_strategy": BasePointForecaster._parameter_constraints["panel_strategy"],
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
            actual_transformer=None,
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
        tags.forecaster_tags.requires_exogenous = False
        tags.forecaster_tags.stateful = True
        return tags

    @property
    def _observation_horizon(self) -> int:
        """Size of the rolling observation buffer.

        ``MeanSeasonalNaive`` retains the last ``seasonality * n_seasons``
        rows so that ``_compute_mean_pattern`` can average each seasonal
        position across the ``n_seasons`` most recent cycles.
        """
        return self.seasonality * self.n_seasons

    def _compute_mean_pattern(self, y_values: pl.DataFrame) -> pl.DataFrame:
        """Compute the mean seasonal pattern from observed values.

        Parameters
        ----------
        y_values : pl.DataFrame
            Observed values (excluding the "time" column) with exactly
            ``seasonality * n_seasons`` rows. The caller guarantees this row
            count by slicing ``_y_observed`` to ``observation_horizon``;
            ``fit`` raises ``ValueError`` first when the training data is
            shorter than ``seasonality * n_seasons``.

        Returns
        -------
        pl.DataFrame
            DataFrame with ``seasonality`` rows containing the averaged
            pattern.

        """
        if self.n_seasons == 1:
            return y_values

        return (
            y_values
            .with_row_index("_pos")
            .with_columns((pl.col("_pos") % self.seasonality).alias("_pos"))
            .group_by("_pos", maintain_order=True)
            .agg(pl.exclude("_pos").mean())
            .sort("_pos")
            .drop("_pos")
        )

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
            pattern = self._compute_mean_pattern(y_values)
            y_pred = _tile_to_horizon(pattern, self.fit_forecasting_horizon_, self.seasonality)

        # Panel data
        else:
            assert isinstance(self._y_observed, dict)
            y_pred = _build_panel_prediction(
                self._y_observed,
                groups,
                self.fit_forecasting_horizon_,
                self.seasonality,
                self._compute_mean_pattern,
            )

        y_pred = self._add_time_columns(y_pred)

        return y_pred
