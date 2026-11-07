"""Implementation of seasonal naive forecasters."""

import numbers
from typing import Literal

import polars as pl
import polars.selectors as cs
from pydantic import StrictInt
from sklearn.base import clone

from yohou.utils._compat import HasMethods, Interval, StrOptions
from yohou.utils.validation import add_interval

from ..utils.tags import Tags
from .base import BasePointForecaster

__all__ = ["MeanSeasonalNaive", "SeasonalNaive", "SegmentProfileForecaster"]


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

    When ``n_seasons=1`` the output is identical to ``SeasonalNaive``.
    Forecasts are cast to each column's training dtype, as for every yohou
    forecaster, so with ``n_seasons > 1`` an integer target gives averages
    rounded to the nearest integer (half to even).

    See Also
    --------
    - [`SeasonalNaive`][yohou.point.naive.SeasonalNaive] : Repeats the last seasonal cycle without averaging.
    - [`SegmentProfileForecaster`][yohou.point.naive.SegmentProfileForecaster] : Average per calendar segment, for local-time and holiday patterns.
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


#: Internal columns of the long frame profiles are computed from: each row's segment
#: label, the value column a value came from, and the value. Prefixed so that no
#: user column can collide with them.
_LABEL = "__yohou_segment_label__"
_COLUMN = "__yohou_column__"
_VALUE = "__yohou_value__"


class SegmentProfileForecaster(BasePointForecaster):
    """Forecast each future timestamp from the profile of its calendar segment.

    Each forecast timestamp is labelled by ``segmenter``, and its forecast is the
    mean or median of the remembered values carrying the same label. Because the
    forecast timestamps are labelled themselves, a profile keyed by local hour,
    weekday, month or holiday stays aligned across daylight saving, months of
    unequal length and holidays, unlike a profile keyed by row position.

    Parameters
    ----------
    segmenter : BaseSegmenter or None, default=None
        Segmenter that labels each timestamp. ``None`` puts every row in one
        segment, so every forecast is the pooled statistic (a mean or median
        forecaster).
    method : {"mean", "median"}, default="mean"
        Statistic of each label's values.
    n_recent : int or None, default=None
        If set, each label uses only its ``n_recent`` most recent finite values.
        Missing values are excluded before counting. ``n_recent=1`` is a
        calendar-aware seasonal naive forecaster.
    window_size : int or None, default=None
        Maximum number of most recent rows remembered. ``None`` caps the memory at
        the number of training rows.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data. See ``BaseForecaster`` for details. Each
        ``group__column`` gets its own profiles under both strategies.

    Attributes
    ----------
    segmenter_ : BaseSegmenter or None
        Fitted clone of ``segmenter``.
    interval_ : str
        Detected time interval of the training data.

    Notes
    -----
    Only finite values (neither null nor NaN) enter a profile. A forecast
    timestamp whose label is null (outside every segment), or whose label has no
    finite remembered value, gets the statistic of all finite remembered values of
    that column. A column with no finite remembered value at all is forecast as
    null, while other columns are forecast normally. No warning is emitted at
    predict time. Profiles are computed in Float64, and forecasts are then cast to
    each column's training dtype, as for every yohou forecaster, so an integer
    target gives forecasts rounded to the nearest integer (half to even).

    The memory holds up to ``window_size`` rows. ``observe`` adds rows and drops
    the oldest beyond that cap. ``rewind`` replaces the memory with the rows given
    and accepts fewer rows than the cap; ``observe`` then refills it. Fitting on
    fewer rows than an integer ``window_size`` raises ``ValueError``. With
    ``window_size=None``, the cap is the training length at each fit, so each
    cross-validation fold has its own cap.

    There is no ``target_transformer``: the memory holds untransformed values. To
    remove a trend first, use the forecaster as a component of a
    [`DecompositionPipeline`][yohou.compose.decomposition_pipeline.DecompositionPipeline].

    See Also
    --------
    - [`PatternSeasonalityForecaster`][yohou.stationarity.seasonality.PatternSeasonalityForecaster] : Seasonal profile keyed by row position.
    - [`MeanSeasonalNaive`][yohou.point.naive.MeanSeasonalNaive] : Average of the last seasons, keyed by row position.
    - [`CalendarSegmenter`][yohou.segmenting.segmenters.CalendarSegmenter] : Segments by calendar feature, optionally in local time.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> from yohou.segmenting import CalendarSegmenter
    >>> times = [datetime(2024, 1, 1) + timedelta(hours=h) for h in range(72)]
    >>> y = pl.DataFrame({"time": times, "load": [float(t.hour) for t in times]})
    >>> forecaster = SegmentProfileForecaster(CalendarSegmenter("hour"))
    >>> _ = forecaster.fit(y, forecasting_horizon=3)
    >>> forecaster.predict()["load"].to_list()
    [0.0, 1.0, 2.0]

    """

    _parameter_constraints: dict = {
        "panel_strategy": BasePointForecaster._parameter_constraints["panel_strategy"],
        "segmenter": [HasMethods(["compute_labels"]), None],
        "method": [StrOptions({"mean", "median"})],
        "n_recent": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "window_size": [Interval(numbers.Integral, 1, None, closed="left"), None],
    }

    def __init__(
        self,
        segmenter=None,
        method: Literal["mean", "median"] = "mean",
        n_recent: StrictInt | None = None,
        window_size: StrictInt | None = None,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ):
        BasePointForecaster.__init__(
            self,
            actual_transformer=None,
            target_transformer=None,
            target_as_feature=None,
            panel_strategy=panel_strategy,
        )

        self.segmenter = segmenter
        self.method = method
        self.n_recent = n_recent
        self.window_size = window_size

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
        """Memory cap: ``window_size``, or the training length when it is None.

        During a memory update the cap is lowered to the rows available, so a short
        ``rewind`` keeps what it is given instead of raising.
        """
        cap = self.window_size if self.window_size is not None else getattr(self, "_training_rows", 0)
        rows_in_update = getattr(self, "_rows_in_update", None)
        return cap if rows_in_update is None else min(cap, rows_in_update)

    @_observation_horizon.setter
    def _observation_horizon(self, value: int) -> None:
        """Store the training length, the cap used when ``window_size`` is None."""
        self._training_rows = value

    def fit(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> "SegmentProfileForecaster":
        """Fit the forecaster: remember up to ``window_size`` training rows.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column.
        X_actual : pl.DataFrame or None, default=None
            Ignored. Present for API compatibility.
        forecasting_horizon : int, default=1
            Number of steps to forecast.
        X_future : pl.DataFrame or None, default=None
            Ignored. Present for API compatibility.
        X_forecast : pl.DataFrame or None, default=None
            Ignored. Present for API compatibility.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted forecaster instance.

        Raises
        ------
        ValueError
            If an integer ``window_size`` exceeds the number of training rows, or
            the segmenter cannot label the training timestamps.

        """
        # The cap must be known before the base fit fills the memory in _pre_fit.
        self._observation_horizon = len(y)
        return super().fit(
            y,
            X_actual=X_actual,
            forecasting_horizon=forecasting_horizon,
            X_future=X_future,
            X_forecast=X_forecast,
            **params,
        )

    def _update_y_X_t_observed_standard(
        self,
        y: pl.DataFrame,
        X_t: pl.DataFrame | None,
        observation_horizon: int,
    ) -> None:
        """Update the memory, keeping at most the rows given."""
        super()._update_y_X_t_observed_standard(y, X_t, min(observation_horizon, len(y)))

    def _update_y_X_t_observed_from_dicts(
        self,
        y: dict[str, pl.DataFrame],
        X_t: dict[str, pl.DataFrame] | None,
        groups: list[str],
    ) -> None:
        """Update the panel memory, keeping at most the rows given per group."""
        # The base method reads self.observation_horizon itself, so the cap is
        # lowered for the duration of the update.
        self._rows_in_update = min(len(y[group]) for group in groups)
        try:
            super()._update_y_X_t_observed_from_dicts(y, X_t, groups)
        finally:
            del self._rows_in_update

    def _fit(
        self,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        X_t: pl.DataFrame | dict[str, pl.DataFrame] | None,
        forecasting_horizon: StrictInt,
    ) -> None:
        """Check the cap and fit the segmenter on the training timestamps."""
        if self.window_size is not None and self.window_size > self._training_rows:
            raise ValueError(
                f"SegmentProfileForecaster window_size={self.window_size} exceeds the "
                f"{self._training_rows} training rows"
            )
        time = y_t["time"] if isinstance(y_t, pl.DataFrame) else next(iter(y_t.values()))["time"]
        self.segmenter_ = None if self.segmenter is None else clone(self.segmenter).fit(time)
        if self.segmenter_ is not None:
            # Raise now, not at the first predict, if the segmenter cannot label this data.
            self.segmenter_.compute_labels(time)

    def _labels(self, time: pl.Series) -> pl.Series:
        """Segment label of each timestamp; one shared label without a segmenter."""
        if self.segmenter_ is None:
            return pl.Series(_LABEL, [True] * len(time))
        return self.segmenter_.compute_labels(time).alias(_LABEL)

    def _profile_forecast(self, memory: pl.DataFrame, future_time: pl.Series) -> pl.DataFrame:
        """Forecast every value column of ``memory`` at ``future_time`` from its label profiles."""
        value_cols = [c for c in memory.columns if c != "time"]
        value = pl.col(_VALUE)
        stat = value.mean() if self.method == "mean" else value.median()

        finite = (
            memory
            .with_columns([pl.col(c).cast(pl.Float64) for c in value_cols])
            .with_columns(self._labels(memory["time"]))
            .unpivot(index=["time", _LABEL], on=value_cols, variable_name=_COLUMN, value_name=_VALUE)
            .filter(value.is_not_null() & value.is_not_nan())
        )
        pooled = dict(finite.group_by(_COLUMN).agg(stat).iter_rows())

        in_segment = finite.filter(pl.col(_LABEL).is_not_null()).sort("time")
        if self.n_recent is not None:
            in_segment = in_segment.group_by([_COLUMN, _LABEL], maintain_order=True).tail(self.n_recent)
        by_label = in_segment.group_by([_COLUMN, _LABEL]).agg(stat)

        future = pl.DataFrame(self._labels(future_time))
        columns = {}
        for col_name in value_cols:
            table = by_label.filter(pl.col(_COLUMN) == col_name).select(_LABEL, _VALUE)
            # Null and unseen labels miss the join and take the pooled statistic;
            # a column with no finite value has no pooled statistic and stays null.
            columns[col_name] = future.join(table, on=_LABEL, how="left", maintain_order="left")[_VALUE].fill_null(
                pl.lit(pooled.get(col_name), dtype=pl.Float64)
            )
        return pl.DataFrame(columns, schema=dict.fromkeys(value_cols, pl.Float64))

    def _predict_one(
        self,
        groups: list[str],
        **params,
    ) -> pl.DataFrame:
        """Predict ``fit_forecasting_horizon_`` steps from the label profiles.

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
        # The forecast timestamps are labelled before y_pred exists, so they are
        # computed here with the same call _add_time_columns uses.
        if self.groups_ is None:
            assert isinstance(self._y_observed, pl.DataFrame)
            observed_time, time_dtype = self.observed_time_, self._y_observed["time"].dtype
        else:
            assert isinstance(self._y_observed, dict)
            observed_time = self.observed_time_[next(iter(self.observed_time_))]
            time_dtype = next(iter(self._y_observed.values()))["time"].dtype
        future_time = pl.Series(
            "time",
            [add_interval(observed_time, self.interval_, n=n) for n in range(1, self.fit_forecasting_horizon_ + 1)],
        ).cast(time_dtype)

        if self.groups_ is None:
            y_pred = self._profile_forecast(self._y_observed, future_time)
        else:
            y_pred = pl.concat(
                [
                    self._profile_forecast(self._y_observed[group], future_time).rename(
                        lambda col, group=group: f"{group}__{col}"
                    )
                    for group in groups
                ],
                how="horizontal",
            )

        return self._add_time_columns(y_pred)
