"""Seasonality forecaster implementation."""

import numbers
from typing import Literal

import numpy as np
import polars as pl
import polars.selectors as cs
from pydantic import StrictInt
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from yohou.utils._compat import Interval, StrOptions

from .base import _BaseSeasonalityForecaster

__all__ = ["FourierSeasonalityForecaster", "PatternSeasonalityForecaster"]


class PatternSeasonalityForecaster(_BaseSeasonalityForecaster):
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
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data.  See `BaseForecaster` for details.

    Attributes
    ----------
    seasonal_pattern_ : pl.DataFrame
        Learned seasonal pattern (length = seasonality).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.stationarity import PatternSeasonalityForecaster
    >>>
    >>> # Create time series with monthly seasonality
    >>> pattern = [10, 12, 15, 13, 11, 9, 8, 10, 12, 15, 13, 11]
    >>> y = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         start=datetime(2020, 1, 1), end=datetime(2022, 12, 1), interval="1mo", eager=True
    ...     ),
    ...     "value": pattern * 3,
    ... })
    >>>
    >>> # Fit seasonal forecaster
    >>> forecaster = PatternSeasonalityForecaster(seasonality=12, method="average")
    >>> forecaster.fit(y, forecasting_horizon=6)
    PatternSeasonalityForecaster(seasonality=12)
    >>>
    >>> # Forecast next 6 months
    >>> y_pred = forecaster.predict(forecasting_horizon=6)

    See Also
    --------
    `FourierSeasonalityForecaster` : Fourier-based seasonality for smooth curves.
    `PolynomialTrendForecaster` : Polynomial trend estimation.
    `DecompositionPipeline` : Combines trend + seasonality + residual forecasters.

    Notes
    -----
    - Requires at least 2 complete seasonal cycles for "average"/"median" methods
    - "naive" method only requires 1 complete cycle
    - Works best with detrended data (consider using with differencing transformers)

    """

    _parameter_constraints: dict = {
        **_BaseSeasonalityForecaster._parameter_constraints,
        "method": [StrOptions({"naive", "average", "median"})],
    }

    def __init__(
        self,
        seasonality: StrictInt,
        method: Literal["naive", "average", "median"] = "average",
        target_transformer=None,
        panel_strategy="global",
    ):
        super().__init__(seasonality=seasonality, target_transformer=target_transformer, panel_strategy=panel_strategy)
        self.method = method

    def _fit(
        self,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        X_t: pl.DataFrame | dict[str, pl.DataFrame] | None,
        forecasting_horizon: StrictInt,
    ) -> None:
        """Extract seasonal pattern from transformed data.

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target time series.
        X_t : pl.DataFrame or dict[str, pl.DataFrame] or None
            Transformed features (unused).
        forecasting_horizon : int
            Number of steps ahead to forecast.

        Raises
        ------
        ValueError
            If insufficient data for specified method.

        """
        # Validate sufficient data for seasonality
        self._validate_method_requirements(y_t)

        # Extract seasonal pattern
        self.seasonal_pattern_ = self._extract_pattern(y_t)

    def _validate_method_requirements(self, y_t: pl.DataFrame | dict[str, pl.DataFrame]) -> None:
        """Validate sufficient data for the specified method.

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target time series.

        Raises
        ------
        ValueError
            If insufficient data for method.

        """
        min_required = self.seasonality
        if self.method in ["average", "median"]:
            min_required = 2 * self.seasonality

        # Handle panel data (dict of DataFrames)
        if isinstance(y_t, dict):
            for panel_group_name, y_t_group in y_t.items():
                assert isinstance(y_t_group, pl.DataFrame)
                if len(y_t_group) < min_required:
                    raise ValueError(
                        f"Insufficient data for group '{panel_group_name}' with method='{self.method}': "
                        f"need at least {min_required} observations "
                        f"({min_required // self.seasonality} complete cycles), got {len(y_t_group)}"
                    )
        # Handle global data (single DataFrame)
        elif len(y_t) < min_required:
            raise ValueError(
                f"Insufficient data for method='{self.method}': "
                f"need at least {min_required} observations "
                f"({min_required // self.seasonality} complete cycles), got {len(y_t)}"
            )

    def _extract_pattern(self, y_t: pl.DataFrame | dict[str, pl.DataFrame]) -> pl.DataFrame:
        """Extract seasonal pattern from data.

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target time series.

        Returns
        -------
        pl.DataFrame
            Seasonal pattern with length = seasonality.

        """
        # Non-panel data
        if self.groups_ is None:
            assert isinstance(y_t, pl.DataFrame)
            patterns = self._extract_pattern_one(y_t)

        # Panel data with pooled pattern (concatenate all groups)
        else:
            # Concatenate all panel group data vertically
            # In panel mode, y_t is a dict and subscript returns DataFrame
            assert isinstance(y_t, dict)
            all_groups_data = [y_t[group_name] for group_name in self.groups_]
            y_t_pooled = pl.concat(all_groups_data, how="vertical")
            patterns = self._extract_pattern_one(y_t_pooled)

        return patterns

    def _extract_pattern_one(self, y_t: pl.DataFrame) -> pl.DataFrame:
        """Extract seasonal pattern from a single DataFrame.

        Parameters
        ----------
        y_t : pl.DataFrame
            Transformed target time series.

        Returns
        -------
        pl.DataFrame
            Seasonal pattern with length = seasonality.

        """
        # Calculate number of complete cycles
        n_cycles = len(y_t) // self.seasonality

        if self.method == "naive":
            # Return last complete cycle
            start_idx = (n_cycles - 1) * self.seasonality
            end_idx = n_cycles * self.seasonality
            pattern = y_t[start_idx:end_idx]

        else:
            # Reshape into cycles and aggregate
            # Truncate to complete cycles only
            truncated_length = n_cycles * self.seasonality
            assert isinstance(truncated_length, int)
            y_truncated = y_t[:truncated_length]

            # Add cycle and position indices
            cycle_indices = [i // self.seasonality for i in range(truncated_length)]
            position_indices = [i % self.seasonality for i in range(truncated_length)]

            y_with_indices = y_truncated.with_columns(
                pl.Series("cycle", cycle_indices),
                pl.Series("position", position_indices),
            )

            # Group by position and aggregate
            if self.method == "average":
                pattern = y_with_indices.group_by("position").agg([
                    pl.col(c).mean() for c in y_t.columns if c != "time"
                ])
            else:  # median
                pattern = y_with_indices.group_by("position").agg([
                    pl.col(c).median() for c in y_t.columns if c != "time"
                ])

            # Sort by position to maintain order
            pattern = pattern.sort("position").select(cs.all().exclude(["position", "cycle"]))

        return pattern

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
        y_t_columns = list(self.local_y_t_schema_.keys())

        # Non-panel data
        if self.groups_ is None:
            # Get phase indices for predictions
            X_phases = self._get_time_indices(self.fit_forecasting_horizon_) % self.seasonality

            # Look up values from pattern
            assert isinstance(self.seasonal_pattern_, pl.DataFrame)
            y_pred = {}
            for col_name in self.seasonal_pattern_.columns:
                if col_name == "time":
                    continue

                # Extract values at specified phases
                pattern_values = self.seasonal_pattern_[col_name].to_list()
                pred_values = [pattern_values[phase] for phase in X_phases.to_list()]
                y_pred[col_name] = pred_values

            y_pred = pl.DataFrame(y_pred)

        # Panel data
        else:
            y_pred = []
            for panel_group_name in groups:
                # Get phase indices for this group
                X_phases = (
                    self._get_time_indices(self.fit_forecasting_horizon_, panel_group_name=panel_group_name)
                    % self.seasonality
                )

                # Get shared pooled pattern
                pattern = self.seasonal_pattern_

                # Look up values from pattern
                y_pred_group = {}
                for col_name in y_t_columns:
                    # Extract values at specified phases
                    pattern_values = pattern[col_name].to_list()
                    pred_values = [pattern_values[phase] for phase in X_phases.to_list()]
                    y_pred_group[f"{panel_group_name}__{col_name}"] = pred_values

                y_pred.append(pl.DataFrame(y_pred_group))

            y_pred = pl.concat(y_pred, how="horizontal")

        return self._add_time_columns(y_pred)


class FourierSeasonalityForecaster(_BaseSeasonalityForecaster):
    """Forecast using Fourier series representation of seasonality.

    Represents seasonality using Fourier series with specified harmonics,
    fitted via ElasticNet regression. More flexible than pattern-based
    methods and can handle non-integer seasonality.

    Parameters
    ----------
    seasonality : float
        Seasonal period length (can be non-integer, e.g., 365.25 for yearly).
    harmonics : list of int, default=[1]
        List of Fourier harmonics to use (e.g., [1, 2, 3] uses first 3 harmonics).
    estimator : RegressorMixin, default=ElasticNet()
        Regression model used to fit Fourier coefficients.
    target_transformer : BaseTransformer, optional
        Transformer for target variable.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data.  See `BaseForecaster` for details.

    Attributes
    ----------
    estimator_ : Pipeline or dict[str, Pipeline]
        Fitted sklearn Pipeline with a fourier feature transformer and the provided
        a clone of the `estimator` model.
    harmonics_ : list of int
        Effective list of harmonics used for Fourier features.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> from datetime import datetime
    >>> from yohou.stationarity import FourierSeasonalityForecaster
    >>>
    >>> # Create time series with sinusoidal seasonality
    >>> time_range = pl.datetime_range(
    ...     start=datetime(2020, 1, 1), end=datetime(2020, 2, 29), interval="1d", eager=True
    ... )
    >>> y = pl.DataFrame({
    ...     "time": time_range,
    ...     "value": [np.sin(2 * np.pi * i / 12) for i in range(len(time_range))],
    ... })
    >>>
    >>> # Fit Fourier seasonality forecaster
    >>> forecaster = FourierSeasonalityForecaster(seasonality=12, harmonics=[1, 2, 3])
    >>> forecaster.fit(y, forecasting_horizon=30)  # doctest: +ELLIPSIS
    FourierSeasonalityForecaster(...)
    >>>
    >>> # Forecast next 30 days
    >>> y_pred = forecaster.predict(forecasting_horizon=30)

    See Also
    --------
    `PatternSeasonalityForecaster` : Pattern-based seasonality for discrete cycles.
    `PolynomialTrendForecaster` : Polynomial trend estimation.
    `DecompositionPipeline` : Combines trend + seasonality + residual forecasters.

    Notes
    -----
    - Handles non-integer seasonality (e.g., 365.25 days/year)
    - Produces smooth seasonal curves
    - Can represent multiple seasonalities by using more harmonics
    - Unlike pattern-based methods, representation is continuous and differentiable

    """

    _parameter_constraints: dict = {
        **_BaseSeasonalityForecaster._parameter_constraints,
        "harmonics": [list, None],
        "alpha": [Interval(numbers.Real, 0, None, closed="left")],
        "l1_ratio": [Interval(numbers.Real, 0, 1, closed="both")],
    }

    def __init__(
        self,
        seasonality: float,
        harmonics: list[int] | None = None,
        estimator: RegressorMixin = ElasticNet(),
        target_transformer=None,
        panel_strategy="global",
    ):
        super().__init__(
            seasonality=int(seasonality),
            target_transformer=target_transformer,
            panel_strategy=panel_strategy,
        )

        self.seasonality = seasonality
        self.harmonics = harmonics
        self.estimator = estimator

    def _build_fourier_features(self, X_time_indices: np.ndarray) -> np.ndarray:
        """Construct Fourier feature matrix.

        Parameters
        ----------
        X_time_indices : np.ndarray
            Time step indices.

        Returns
        -------
        np.ndarray
            Shape (n_samples, 2 * len(harmonics)) with sin/cos features.

        """
        features = []

        for k in self.harmonics_:
            features.append(np.sin(2 * np.pi * k * X_time_indices / self.seasonality))
            features.append(np.cos(2 * np.pi * k * X_time_indices / self.seasonality))
        return np.column_stack(features)

    def _fit(
        self,
        y_t: pl.DataFrame | dict[str, pl.DataFrame],
        X_t: pl.DataFrame | dict[str, pl.DataFrame] | None,
        forecasting_horizon: StrictInt,
    ) -> None:
        """Fit Fourier series model to transformed data.

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target time series.
        X_t : pl.DataFrame or dict[str, pl.DataFrame] or None
            Transformed features (unused).
        forecasting_horizon : int
            Number of steps ahead to forecast.

        Raises
        ------
        ValueError
            If harmonics list is empty, contains non-positive integers,
            or exceeds the Nyquist limit.

        """
        # Domain-specific validation: harmonics must be positive and not exceed
        # seasonality/2 (Nyquist limit)

        if self.harmonics is not None and not self.harmonics:
            raise ValueError("harmonics list cannot be empty")

        self.harmonics_ = self.harmonics if self.harmonics else [1]

        if any(h < 1 for h in self.harmonics_):
            raise ValueError("All harmonics must be positive integers")
        max_harmonic = max(self.harmonics_)
        if max_harmonic > self.seasonality / 2:
            raise ValueError(
                f"Maximum harmonic ({max_harmonic}) cannot exceed seasonality/2 "
                f"({self.seasonality / 2:.1f}) due to Nyquist sampling theorem."
            )

        # Validate sufficient data (at least one cycle)
        self._validate_sufficient_data(y_t)

        estimator = Pipeline([
            ("poly_features", FunctionTransformer(func=self._build_fourier_features)),
            ("regressor", clone(self.estimator)),
        ])

        self._fit_estimator(estimator, y_t)
