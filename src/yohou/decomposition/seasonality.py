"""Seasonality forecaster implementation."""

import numbers
from typing import Literal

import numpy as np
import polars as pl
import polars.selectors as cs
from pydantic import StrictInt
from sklearn.base import RegressorMixin, _fit_context, clone
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils._param_validation import Interval

from .base import _BaseSeasonalityForecaster


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
    model_panel : bool, default=False
        Strategy for panel data modeling.
        - If False (default): Pooled strategy - aggregates patterns across all
        panel groups to create a single shared seasonal pattern. Lower memory
        usage. Best when series have similar seasonality or limited per-series data.
        - If True: Per-group strategy - extracts separate seasonal pattern for
        each panel group. Captures group-specific seasonal dynamics. Higher
        memory usage. Best when series have distinct seasonal patterns and
        sufficient per-group data.
        Ignored for global (non-panel) data.

    Attributes
    ----------
    seasonal_pattern_ : pl.DataFrame or dict[str, pl.DataFrame]
        Learned seasonal pattern (length = seasonality).
        DataFrame for global data or model_panel=False (pooled).
        Dict of DataFrames for model_panel=True (per-group patterns).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.point_forecaster import PatternSeasonalityForecaster
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
    >>> forecaster = PatternSeasonalityForecaster(seasonality=12, method="average")
    >>> forecaster.fit(y, forecasting_horizon=6)
    PatternSeasonalityForecaster(seasonality=12)
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
        "model_panel": [bool],
    }

    def __init__(
        self,
        seasonality: StrictInt,
        method: Literal["naive", "average", "median"] = "average",
        target_transformer=None,
        model_panel: bool = False,
    ):
        super().__init__(
            seasonality=seasonality, target_transformer=target_transformer, model_panel=model_panel
        )
        self.method = method

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> "PatternSeasonalityForecaster":
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
                if len(y_t_group) < min_required:
                    raise ValueError(
                        f"Insufficient data for group '{panel_group_name}' with method='{self.method}': "
                        f"need at least {min_required} observations "
                        f"({min_required // self.seasonality} complete cycles), got {len(y_t_group)}"
                    )
        # Handle global data (single DataFrame)
        else:
            if len(y_t) < min_required:
                raise ValueError(
                    f"Insufficient data for method='{self.method}': "
                    f"need at least {min_required} observations "
                    f"({min_required // self.seasonality} complete cycles), got {len(y_t)}"
                )

    def _extract_pattern(
        self, y_t: pl.DataFrame | dict[str, pl.DataFrame]
    ) -> pl.DataFrame | dict[str, pl.DataFrame]:
        """Extract seasonal pattern from data.

        Parameters
        ----------
        y_t : pl.DataFrame or dict[str, pl.DataFrame]
            Transformed target time series.

        Returns
        -------
        pl.DataFrame or dict[str, pl.DataFrame]
            Seasonal pattern with length = seasonality.
            Returns dict for panel data with model_panel=True,
            DataFrame otherwise (global data or pooled panel).

        """
        # Non-panel data
        if self.panel_group_names_ is None:
            patterns = self._extract_pattern_one(y_t)

        # Panel data with per-group patterns
        elif self.model_panel:
            patterns = {}
            for panel_group_name in self.panel_group_names_:
                y_t_group = y_t[panel_group_name]
                patterns[panel_group_name] = self._extract_pattern_one(y_t_group)

        # Panel data with pooled pattern (concatenate all groups)
        else:
            # Concatenate all panel group data vertically
            all_groups_data = [y_t[group_name] for group_name in self.panel_group_names_]
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
                pattern = y_with_indices.group_by("position").agg(
                    [pl.col(c).mean() for c in y_t.columns if c != "time"]
                )
            else:  # median
                pattern = y_with_indices.group_by("position").agg(
                    [pl.col(c).median() for c in y_t.columns if c != "time"]
                )

            # Sort by position to maintain order
            pattern = pattern.sort("position").select(cs.all().exclude(["position", "cycle"]))

        return pattern

    def _predict_one(
        self,
        panel_group_names: list[str],
    ) -> pl.DataFrame:
        """Predicts `_fit_forecasting_horizon` steps from the observation horizon.

        Parameters
        ----------
        panel_group_names : list of str
            Panel group names to predict for.

        Returns
        -------
        pl.DataFrame
            Predicted time series.

        """
        y_t_columns = list(self.local_y_t_schema_.keys())

        # Non-panel data
        if self.panel_group_names_ is None:
            # Get phase indices for predictions
            X_phases = self._get_time_indices(self.fit_forecasting_horizon_) % self.seasonality

            # Look up values from pattern
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
            for panel_group_name in panel_group_names:
                # Get phase indices for this group
                X_phases = (
                    self._get_time_indices(
                        self.fit_forecasting_horizon_, panel_group_name=panel_group_name
                    )
                    % self.seasonality
                )

                # Get pattern for this group (either per-group or shared)
                if isinstance(self.seasonal_pattern_, dict):
                    pattern = self.seasonal_pattern_[panel_group_name]
                else:
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
    harmonics : list of int, default=[1, 2, 3]
        List of Fourier harmonics to use (e.g., [1, 2, 3] uses first 3 harmonics).
    estimator : RegressorMixin, default=ElasticNet()
        Regression model used to fit Fourier coefficients.
    target_transformer : BaseTransformer, optional
        Transformer for target variable.
    model_panel : bool, default=False
        Strategy for panel data modeling.
        - If False (default): Pooled strategy - treats all panel groups as
        independent samples for a single global model. Shares parameters
        across series. Lower memory usage. Best when series have similar
        patterns or limited per-series data.
        - If True: Per-group strategy - fits separate model for each panel
        group. Captures group-specific dynamics. Higher memory usage.
        Best when series have distinct patterns and sufficient per-group data.
        Ignored for global (non-panel) data.

    Attributes
    ----------
    estimator_ : Pipeline or dict[str, Pipeline]
        Fitted sklearn Pipeline with a fourier feature transformer and the provided
        a clone of the `estimator` model.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> from datetime import datetime
    >>> from yohou.point_forecaster import FourierSeasonalityForecaster
    >>>
    >>> # Create time series with sinusoidal seasonality
    >>> time_range = pl.datetime_range(
    ...     start=datetime(2020, 1, 1),
    ...     end=datetime(2020, 2, 29),
    ...     interval="1d",
    ...     eager=True
    ... )
    >>> y = pl.DataFrame({
    ...     "time": time_range,
    ...     "value": [np.sin(2 * np.pi * i / 12) for i in range(len(time_range))]
    ... })
    >>>
    >>> # Fit Fourier seasonality forecaster
    >>> forecaster = FourierSeasonalityForecaster(seasonality=12, harmonics=[1, 2, 3])
    >>> forecaster.fit(y, forecasting_horizon=30)
    FourierSeasonalityForecaster(seasonality=12)
    >>>
    >>> # Forecast next 30 days
    >>> y_pred = forecaster.predict(forecasting_horizon=30)

    Notes
    -----
    - Handles non-integer seasonality (e.g., 365.25 days/year)
    - Produces smooth seasonal curves
    - Can represent multiple seasonalities by using more harmonics
    - Unlike pattern-based methods, representation is continuous and differentiable

    """

    _parameter_constraints: dict = {
        **_BaseSeasonalityForecaster._parameter_constraints,
        "harmonics": [list],
        "alpha": [Interval(numbers.Real, 0, None, closed="left")],
        "l1_ratio": [Interval(numbers.Real, 0, 1, closed="both")],
    }

    def __init__(
        self,
        seasonality: float,
        harmonics: list[int] | None = None,
        estimator: RegressorMixin = ElasticNet(),
        target_transformer=None,
        model_panel: bool = False,
    ):
        super().__init__(
            seasonality=int(seasonality),
            target_transformer=target_transformer,
            model_panel=model_panel,
        )

        self.seasonality = seasonality
        self.harmonics = harmonics if harmonics is not None else [1, 2, 3]
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

        for k in self.harmonics:
            features.append(np.sin(2 * np.pi * k * X_time_indices / self.seasonality))
            features.append(np.cos(2 * np.pi * k * X_time_indices / self.seasonality))
        return np.column_stack(features)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        **params,
    ) -> "FourierSeasonalityForecaster":
        """Fit Fourier series model to historical data.

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

        """
        # Domain-specific validation: harmonics must be positive and not exceed
        # seasonality/2 (Nyquist limit)
        if not self.harmonics:
            raise ValueError("harmonics list cannot be empty")
        if any(h < 1 for h in self.harmonics):
            raise ValueError("All harmonics must be positive integers")
        max_harmonic = max(self.harmonics)
        if max_harmonic > self.seasonality / 2:
            raise ValueError(
                f"Maximum harmonic ({max_harmonic}) cannot exceed seasonality/2 "
                f"({self.seasonality / 2:.1f}) due to Nyquist sampling theorem."
            )

        # Pre-fit: validate inputs, apply target transformer, set attributes
        y_t, X_t = self._pre_fit(y=y, X=X, forecasting_horizon=forecasting_horizon)

        # Validate sufficient data (at least one cycle)
        self._validate_sufficient_data(y_t)

        estimator = Pipeline(
            [
                ("poly_features", FunctionTransformer(func=self._build_fourier_features)),
                ("regressor", clone(self.estimator)),
            ]
        )

        self._fit_estimator(estimator, y_t)

        return self
