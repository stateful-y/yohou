"""Pattern-based seasonality forecasting implementation."""

from typing import Literal

import polars as pl
import polars.selectors as cs
from pydantic import StrictInt
from sklearn.base import _fit_context

from .base import _BaseSeasonalityForecaster


class SeasonalityForecaster(_BaseSeasonalityForecaster):
    """Forecast using seasonal pattern extraction and repetition.

    Learns seasonal patterns from historical data and repeats them into the
    future. Suitable for time series with strong periodic behavior.

    Parameters
    ----------
    seasonality : int
        Seasonal period length (e.g., 12 for monthly data with yearly seasonality,
        7 for daily data with weekly seasonality).
    method : {"naive", "average", "median"}, default="average"
        Method for aggregating seasonal patterns:
        - "naive": Use last complete cycle
        - "average": Mean across all cycles
        - "median": Median across all cycles (robust to outliers)
    target_transformer : BaseTransformer, optional
        Transformer for target variable.

    Attributes
    ----------
    seasonal_pattern_ : pl.DataFrame
        Learned seasonal pattern (length = seasonality).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.point_forecaster import SeasonalityForecaster
    >>>
    >>> # Create time series with monthly seasonality
    >>> pattern = [10, 12, 15, 13, 11, 9, 8, 10, 12, 15, 13, 11]
    >>> y = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         start=datetime(2020, 1, 1),
    ...         end=datetime(2022, 12, 1),
    ...         interval="1mo",
    ...         eager=True
    ...     ),
    ...     "value": pattern * 3
    ... })
    >>>
    >>> # Fit seasonal forecaster
    >>> forecaster = SeasonalityForecaster(seasonality=12, method="average")
    >>> forecaster.fit(y, forecasting_horizon=6)
    SeasonalityForecaster(seasonality=12)
    >>>
    >>> # Forecast next 6 months
    >>> y_pred = forecaster.predict(forecasting_horizon=6)

    Notes
    -----
    - Requires at least 2 complete seasonal cycles for "average"/"median" methods
    - "naive" method only requires 1 complete cycle
    - Works best with detrended data (consider using with differencing transformers)

    """

    _parameter_constraints: dict = {
        **_BaseSeasonalityForecaster._parameter_constraints,
        "method": [str],
    }

    def __init__(
        self,
        seasonality: StrictInt,
        method: Literal["naive", "average", "median"] = "average",
        target_transformer=None,
    ):
        """Initialize SeasonalityForecaster.

        Parameters
        ----------
        seasonality : int
            Length of seasonal cycle.
        method : {"naive", "average", "median"}, default="average"
            Aggregation method for multiple cycles.
        target_transformer : BaseTransformer, optional
            Transformer for target variable.

        """
        super().__init__(seasonality=seasonality, target_transformer=target_transformer)
        self.method = method

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> "SeasonalityForecaster":
        """Fit seasonal pattern from historical data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with "time" column.
        X : pl.DataFrame, optional
            Exogenous features (currently not used, reserved for future).
        forecasting_horizon : int, default=1
            Number of steps ahead to forecast.
        **params : dict
            Additional metadata (routed via sklearn's metadata routing).

        Returns
        -------
        self
            Fitted forecaster.

        Raises
        ------
        ValueError
            If insufficient data for specified method.

        """
        # Pre-fit: validate inputs, apply target transformer, set attributes
        y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)

        # Validate sufficient data for seasonality
        self._validate_method_requirements(y_t)

        # Extract seasonal pattern
        self.seasonal_pattern_ = self._extract_pattern(y_t)

        return self

    def _validate_method_requirements(self, y: pl.DataFrame) -> None:
        """Validate sufficient data for the specified method.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.

        Raises
        ------
        ValueError
            If insufficient data for method.

        """
        min_required = self.seasonality
        if self.method in ["average", "median"]:
            min_required = 2 * self.seasonality

        if len(y) < min_required:
            raise ValueError(
                f"Insufficient data for method='{self.method}': "
                f"need at least {min_required} observations "
                f"({min_required // self.seasonality} complete cycles), got {len(y)}"
            )

    def _extract_pattern(self, y: pl.DataFrame) -> pl.DataFrame:
        """Extract seasonal pattern from data.

        Parameters
        ----------
        y : pl.DataFrame
            Transformed target time series (without "time" column after _pre_fit).

        Returns
        -------
        pl.DataFrame
            Seasonal pattern with length = seasonality.

        """
        # Calculate number of complete cycles
        n_cycles = len(y) // self.seasonality

        if self.method == "naive":
            # Return last complete cycle
            start_idx = (n_cycles - 1) * self.seasonality
            end_idx = n_cycles * self.seasonality
            pattern = y[start_idx:end_idx]

        else:
            # Reshape into cycles and aggregate
            # Truncate to complete cycles only
            truncated_length = n_cycles * self.seasonality
            y_truncated = y[:truncated_length]

            # Add cycle and position indices
            cycle_indices = [i // self.seasonality for i in range(truncated_length)]
            position_indices = [i % self.seasonality for i in range(truncated_length)]

            y_with_indices = y_truncated.with_columns(
                pl.Series("cycle", cycle_indices),
                pl.Series("position", position_indices),
            )

            # Group by position and aggregate
            if self.method == "average":
                pattern = y_with_indices.group_by("position").agg(
                    [pl.col(c).mean() for c in y.columns if c != "time"]
                )
            else:  # median
                pattern = y_with_indices.group_by("position").agg(
                    [pl.col(c).median() for c in y.columns if c != "time"]
                )

            # Sort by position to maintain order
            pattern = pattern.sort("position").select(cs.all().exclude(["position", "cycle"]))

        return pattern

    def _predict_from_pattern(self, forecasting_horizon: int, **params) -> pl.DataFrame:
        """Generate predictions from stored seasonal pattern.

        Parameters
        ----------
        forecasting_horizon : int
            Number of steps to predict.
        **params : dict
            Additional parameters.

        Returns
        -------
        pl.DataFrame
            Predictions for next forecasting_horizon steps.

        """
        # Get phase indices for predictions
        phases = self._get_time_indices(forecasting_horizon) % self.seasonality

        # Look up values from pattern
        predictions = {}
        for col_name in self.seasonal_pattern_.columns:
            if col_name == "time":
                continue

            # Extract values at specified phases
            pattern_values = self.seasonal_pattern_[col_name].to_list()
            pred_values = [pattern_values[phase] for phase in phases.to_list()]
            predictions[col_name] = pred_values

        y_pred = pl.DataFrame(predictions)
        return self._add_time_columns(y_pred)
