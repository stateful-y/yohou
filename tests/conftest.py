"""Global pytest fixtures for yohou transformer testing.

This file contains:
1. Dummy transformer classes (SimpleTransformer, StatelessTransformer, etc.)
2. Data generation fixtures (base_time_series, time_series_factory, panel_time_series_factory)
3. Transformer registry fixture with metadata and expected failures
4. Edge case dataset factories
"""

from datetime import datetime, timedelta

import polars as pl
import polars.selectors as cs
import pytest
from sklearn.exceptions import NotFittedError

from yohou.base import BaseTransformer
from yohou.interval_forecaster.base import BaseIntervalForecaster
from yohou.point_forecaster.base import BasePointForecaster

# ============================================================================
# DUMMY TRANSFORMER CLASSES
# ============================================================================


class SimpleTransformer(BaseTransformer):
    """Identity transformer with configurable observation horizon.

    Used for testing composition classes, parameter passing, and
    basic transformer contract compliance.
    """

    def __init__(self, observation_horizon=1, add_constant=0.0):
        self.observation_horizon = observation_horizon
        self.add_constant = add_constant

    @property
    def observation_horizon(self):
        if not hasattr(self, "_observation_horizon"):
            raise NotFittedError("Call fit before accessing observation_horizon")
        return self._observation_horizon

    @observation_horizon.setter
    def observation_horizon(self, value):
        self._observation_horizon = value

    def fit(self, X, y=None):
        BaseTransformer.fit(self, X, y)
        return self

    def transform(self, X):
        return X.select([pl.col("time"), (cs.numeric() & ~cs.by_name("time")) + self.add_constant])

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class StatelessTransformer(BaseTransformer):
    """Transformer with observation_horizon=0 (works without fitting)."""

    def __init__(self, multiplier=2.0):
        self.multiplier = multiplier
        self._observation_horizon = 0

    @property
    def observation_horizon(self):
        return 0

    def fit(self, X, y=None):
        # Set minimal attributes for sklearn compatibility
        self.feature_names_in_ = [col for col in X.columns if col != "time"]
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        return X.select([pl.col("time"), (cs.numeric() & ~cs.by_name("time")) * self.multiplier])

    def get_feature_names_out(self, input_features=None):
        return [col for col in input_features if col != "time"] if input_features else []


class InvertibleTransformer(BaseTransformer):
    """Transformer with perfect inverse_transform."""

    def __init__(self, observation_horizon=2, offset=10.0):
        self.observation_horizon = observation_horizon
        self.offset = offset

    @property
    def observation_horizon(self):
        if not hasattr(self, "_observation_horizon"):
            raise NotFittedError("Call fit before accessing observation_horizon")
        return self._observation_horizon

    @observation_horizon.setter
    def observation_horizon(self, value):
        self._observation_horizon = value

    def transform(self, X):
        return X.select([pl.col("time"), (cs.numeric() & ~cs.by_name("time")) + self.offset])

    def inverse_transform(self, X):
        return X.select([pl.col("time"), (cs.numeric() & ~cs.by_name("time")) - self.offset])

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class PanelAwareTransformer(BaseTransformer):
    """Transformer explicitly handling panel data columns."""

    def __init__(self, observation_horizon=1):
        self.observation_horizon = observation_horizon

    @property
    def observation_horizon(self):
        if not hasattr(self, "_observation_horizon"):
            raise NotFittedError("Call fit before accessing observation_horizon")
        return self._observation_horizon

    @observation_horizon.setter
    def observation_horizon(self, value):
        self._observation_horizon = value

    def transform(self, X):
        # Preserve panel columns
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


# ============================================================================
# DATA GENERATION FIXTURES
# ============================================================================


@pytest.fixture
def time_series_factory():
    """Factory fixture for creating custom time_series data.

    Returns a callable that generates polars DataFrames with:
    - "time" column (datetime)
    - Configurable number of numeric feature columns
    - Deterministic data based on seed
    """

    def _make(length=50, n_components=2, seed=42, min_length=None):
        if min_length and length < min_length:
            raise ValueError(f"length {length} < min_length {min_length}")

        # Use datetime_range to generate time series
        # end = start + (length - 1) * interval
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
            interval="1s",
            eager=True,
        )
        rng = pl.Series(range(length)).cast(pl.Float64)
        features = {f"feature_{i}": rng + (i * 100) for i in range(n_components)}
        return pl.DataFrame({"time": time, **features})

    return _make


@pytest.fixture
def X_factory(time_series_factory):
    """Alias for time_series_factory for consistency with y_X_factory naming.

    This fixture generates exogenous features (X) for time series models.
    Use this when you need only X data without y targets.

    .. note::
        This is the preferred name for consistency. time_series_factory
        is maintained for backward compatibility.

    """
    return time_series_factory


@pytest.fixture(scope="session")
def base_time_series():
    """Session-scoped base time_series (immutable, reused for performance).

    Creates a standard 100-row, 3-feature dataset that is cached for
    the entire test session to improve performance. Tests should not
    modify this fixture.
    """
    length = 100
    n_features = 3

    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1, 0, 0, length - 1),
        interval="1s",
        eager=True,
    )
    rng = pl.Series(range(length)).cast(pl.Float64)
    features = {f"feature_{i}": rng + (i * 100) for i in range(n_features)}
    return pl.DataFrame({"time": time, **features})


@pytest.fixture
def time_series_train_test_factory():
    """Factory for generating continuous train/test splits.

    Returns a function that creates X_train and X_test where X_test
    continues chronologically from where X_train ends (no overlap, no gap).

    Parameters
    ----------
    train_length : int
        Number of observations in training set
    test_length : int
        Number of observations in test set
    n_components : int, default=2
        Number of feature columns
    seed : int, default=42
        Random seed (currently unused, but kept for API consistency)

    Returns
    -------
    X_train, X_test : tuple of pl.DataFrame
        Continuous train and test time series
    """

    def _make(train_length, test_length, n_components=2, seed=42):
        total_length = train_length + test_length

        # Generate continuous time series
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1) + timedelta(seconds=total_length - 1),
            interval="1s",
            eager=True,
        )
        rng = pl.Series(range(total_length)).cast(pl.Float64)
        features = {f"feature_{i}": rng + (i * 100) for i in range(n_components)}
        df = pl.DataFrame({"time": time, **features})

        # Split into train and test
        X_train = df.head(train_length)
        X_test = df.tail(test_length)

        return X_train, X_test

    return _make


@pytest.fixture
def panel_time_series_factory():
    """Factory for panel data with both global and local columns.

    Creates a DataFrame with:
    - "time" column (datetime)
    - Global columns (shared across all panels)
    - Panel-specific columns with `__` separator

    Parameters
    ----------
    length : int, default=50
        Number of time steps.
    n_series : int, default=3
        Number of series per group.
    n_groups : int, default=1
        Number of distinct panel groups.
        - If n_groups=1: Creates "panel__series_0", "panel__series_1", ... (backward compatible)
        - If n_groups>1: Creates "group0__series_0", "group1__series_0", ... (distinct groups)
    n_global : int, default=0
        Number of global features (shared across all panels).
    seed : int, default=42
        Random seed for reproducibility.

    Examples
    --------
    Single group (backward compatible):
        y = panel_time_series_factory(length=100, n_series=3)
        # columns: ["time", "panel__series_0", "panel__series_1", "panel__series_2"]

    Multiple distinct groups:
        y = panel_time_series_factory(length=100, n_series=2, n_groups=3)
        # columns: ["time", "group0__series_0", "group0__series_1",
        #           "group1__series_0", "group1__series_1",
        #           "group2__series_0", "group2__series_1"]
    """

    def _make(length=50, n_series=3, n_groups=1, n_global=0, seed=42):
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
            interval="1s",
            eager=True,
        )

        # Build panel data (columns with __ separator)
        panel_data = {}
        if n_groups == 1:
            # Backward compatible: single "panel" group
            for i in range(n_series):
                panel_data[f"panel__series_{i}"] = range(i * 10, length + (i * 10))
        else:
            # Multiple distinct groups
            for group_idx in range(n_groups):
                for series_idx in range(n_series):
                    col_name = f"group{group_idx}__series_{series_idx}"
                    panel_data[col_name] = range(
                        (group_idx * n_series + series_idx) * 10,
                        length + (group_idx * n_series + series_idx) * 10,
                    )

        # Add global columns (shared features across all panels)
        global_features = {}
        for i in range(n_global):
            global_features[f"global_{i}"] = pl.Series(
                range(i * 100, length + (i * 100)), dtype=pl.Float64
            )

        # Combine: time + global columns + panel columns
        result = pl.DataFrame({"time": time, **global_features, **panel_data})

        return result

    return _make


@pytest.fixture
def edge_case_datasets_factory():
    """Factory for edge case test datasets.

    Returns a callable that generates datasets for testing edge cases:
    - empty: Empty DataFrame with correct schema
    - single_row: Single observation
    - exact_horizon: Exactly observation_horizon rows
    """

    def _make(observation_horizon=1):
        return {
            "empty": pl.DataFrame({"time": [], "feature": []}),
            "single_row": pl.DataFrame(
                {
                    "time": [datetime(2021, 1, 1)],
                    "feature": [1.0],
                }
            ),
            "exact_horizon": pl.DataFrame(
                {
                    "time": pl.datetime_range(
                        start=datetime(2021, 1, 1),
                        end=datetime(2021, 1, 1, 0, 0, observation_horizon - 1),
                        interval="1s",
                        eager=True,
                    ),
                    "feature": pl.Series(range(observation_horizon), dtype=pl.Float64),
                }
            ),
        }

    return _make


# ============================================================================
# TRANSFORMER CONFIGURATION FIXTURES
# ============================================================================


@pytest.fixture
def dummy_transformers():
    """Minimal transformers for composition testing.

    Returns a dictionary of dummy transformer instances that can be
    used to test composition classes (FeaturePipeline, FeatureUnion, etc.)
    """
    return {
        "simple": SimpleTransformer(observation_horizon=1),
        "stateless": StatelessTransformer(),
        "invertible": InvertibleTransformer(observation_horizon=2),
        "panel_aware": PanelAwareTransformer(observation_horizon=1),
    }


@pytest.fixture
def transformer_registry():
    """Registry of transformers with metadata for parametrized tests.

    Returns a dictionary mapping transformer names to configuration dicts
    containing:
    - transformer: An instance of the transformer
    - expected_failed_checks: List of check names expected to fail
    """
    from yohou.preprocessing import (
        LagTransformer,
        SeasonalDifferencing,
        SeasonalLogDifferencing,
    )

    return {
        "SeasonalDifferencing": {
            "transformer": SeasonalDifferencing(
                seasonality=1
            ),  # seasonality=1 for first difference
            "expected_failed_checks": [],
        },
        "SeasonalLogDifferencing": {
            "transformer": SeasonalLogDifferencing(
                seasonality=1
            ),  # seasonality=1 for first difference
            "expected_failed_checks": [],
        },
        "LagTransformer": {
            "transformer": LagTransformer(lag=[1, 2]),
            "expected_failed_checks": [
                "check_inverse_transform_identity",
                "check_inverse_transform_round_trip",
            ],
        },
    }


# ============================================================================
# FORECASTER FIXTURES
# ============================================================================


@pytest.fixture
def y_X_factory():
    """Factory for generating (y, X) tuples.

    Returns a callable that generates time series data for forecaster testing.
    """
    from datetime import datetime, timedelta

    import numpy as np

    def _factory(length=100, n_targets=2, n_features=3, seed=42, panel=False, n_groups=2):
        """Generate forecaster test data.

        Parameters
        ----------
        length : int
            Number of time steps
        n_targets : int
            Number of target features
        n_features : int
            Number of features (0 for None)
        seed : int
            Random seed
        panel : bool
            Whether to create panel data with columns using __ separator
        n_groups : int, default=2
            Number of panel groups when panel=True. Each target/feature will
            be replicated with group suffixes: y_0__group_0, y_0__group_1, etc.

        Returns
        -------
        y : pl.DataFrame
            Target data with "time" column
        X : pl.DataFrame or None
            Features with "time" column
        """
        rng = np.random.default_rng(seed)

        time = pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16) + timedelta(seconds=length - 1),
            interval="1s",
            eager=True,
        )

        if panel:
            # Generate panel data with __ separator
            y = pl.DataFrame({"time": time})
            for i in range(n_targets):
                base_values = rng.random(length)
                for group_idx in range(n_groups):
                    # Add slight variation per group
                    variation = group_idx * 0.1
                    col_name = f"y_{i}__group_{group_idx}"
                    y = y.with_columns(pl.Series(col_name, base_values + variation))

            X = None
            if n_features > 0:
                X = pl.DataFrame({"time": time})
                for i in range(n_features):
                    base_values = rng.random(length)
                    for group_idx in range(n_groups):
                        # Add slight variation per group
                        variation = group_idx * 0.05
                        col_name = f"X_{i}__group_{group_idx}"
                        X = X.with_columns(pl.Series(col_name, base_values + variation))
        else:
            # Generate regular (non-panel) data
            y = pl.DataFrame({"time": time})
            for i in range(n_targets):
                y = y.with_columns(pl.Series(f"y_{i}", rng.random(length)))

            X = None
            if n_features > 0:
                X = pl.DataFrame({"time": time})
                for i in range(n_features):
                    X = X.with_columns(pl.Series(f"X_{i}", rng.random(length)))

        return y, X

    return _factory


@pytest.fixture
def y_X_panel_factory():
    """Factory for generating panel data (y, X) tuples for testing.

    Returns a callable that generates panel time series data with group prefixes.
    Each target/feature is replicated across groups with naming pattern:
    <name>__group_0, <name>__group_1, etc.
    """
    from datetime import datetime, timedelta

    import numpy as np

    def _factory(n_groups=2, length=100, n_targets=2, n_features=2, seed=42):
        """Generate panel data for forecaster testing.

        Parameters
        ----------
        n_groups : int, default=2
            Number of panel groups to create
        length : int, default=100
            Number of time steps
        n_targets : int, default=2
            Number of target variables per group
        n_features : int, default=2
            Number of features per group (0 for X=None)
        seed : int, default=42
            Random seed for reproducibility

        Returns
        -------
        y : pl.DataFrame
            Panel target data with "time" column and prefixed targets
        X : pl.DataFrame or None
            Panel features with "time" column and prefixed features
        """
        rng = np.random.default_rng(seed)

        time = pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16) + timedelta(seconds=length - 1),
            interval="1s",
            eager=True,
        )

        # Generate panel y data
        y = pl.DataFrame({"time": time})
        for i in range(n_targets):
            base_values = rng.random(length) * 100  # Scale for better visibility
            for group_idx in range(n_groups):
                # Add group-specific offset
                variation = group_idx * 10 + rng.normal(0, 1, length)
                col_name = f"y_{i}__group_{group_idx}"
                y = y.with_columns(pl.Series(col_name, base_values + variation))

        # Generate panel X data
        X = None
        if n_features > 0:
            X = pl.DataFrame({"time": time})
            for i in range(n_features):
                base_values = rng.random(length) * 50
                for group_idx in range(n_groups):
                    # Add group-specific offset
                    variation = group_idx * 5 + rng.normal(0, 0.5, length)
                    col_name = f"X_{i}__group_{group_idx}"
                    X = X.with_columns(pl.Series(col_name, base_values + variation))

        return y, X

    return _factory


@pytest.fixture
def forecaster_registry():
    """Registry of forecasters with metadata and expected failures."""
    from sklearn.linear_model import Ridge

    from yohou.interval_forecaster import SplitConformalForecaster
    from yohou.point_forecaster import NaiveForecaster, PointReductionForecaster

    return {
        "NaiveForecaster": {
            "forecaster": NaiveForecaster(seasonality=1),
            "expected_failed_checks": [],
        },
        "PointReductionForecaster": {
            "forecaster": PointReductionForecaster(estimator=Ridge()),
            "expected_failed_checks": [],
        },
        "SplitConformalForecaster": {
            "forecaster": SplitConformalForecaster(
                point_forecaster=NaiveForecaster(seasonality=1),
                calibration_size=0.2,
            ),
            "expected_failed_checks": ["check_reset_propagates_to_transformers"],
        },
    }


@pytest.fixture
def panel_X_factory():
    """Factory for panel data X with panel columns."""

    def _make(length=50, n_panels=2, n_features=2):
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
            interval="1s",
            eager=True,
        )

        # Create panel columns with __ separator
        panel_data = {}
        for i in range(n_panels):
            for j in range(n_features):
                panel_data[f"panel_{i}__feature_{j}"] = range(
                    i * 100 + j * 10, length + (i * 100 + j * 10)
                )

        # Create DataFrame with panel columns
        panel_df = pl.DataFrame(panel_data)

        return pl.concat([pl.DataFrame({"time": time}), panel_df], how="horizontal")

    return _make


@pytest.fixture
def pattern_factory():
    """Generate time series with specific patterns for testing.

    Returns
    -------
    callable
        Factory function with signature:
        pattern_factory(pattern_type, length, **kwargs) -> pl.DataFrame

    """
    from datetime import datetime, timedelta

    import numpy as np

    def _factory(
        pattern_type: str,
        length: int = 100,
        amplitude: float = 1.0,
        noise_level: float = 0.1,
        seed: int | None = None,
    ) -> pl.DataFrame:
        """Generate time series with specific pattern.

        Parameters
        ----------
        pattern_type : str
            One of: "linear_trend", "exponential_trend", "seasonal",
            "trend_seasonal", "step_change", "random_walk"
        length : int
            Number of time steps
        amplitude : float
            Pattern amplitude
        noise_level : float
            Amount of random noise (std dev)
        seed : int | None
            Random seed for reproducibility

        Returns
        -------
        pl.DataFrame
            Time series with "time" and "target" columns

        """
        rng = np.random.default_rng(seed)
        t = np.arange(length)
        time = [datetime(2020, 1, 1) + timedelta(days=int(i)) for i in t]

        if pattern_type == "linear_trend":
            values = amplitude * t
        elif pattern_type == "exponential_trend":
            values = amplitude * np.exp(0.01 * t)
        elif pattern_type == "seasonal":
            values = amplitude * np.sin(2 * np.pi * t / 12)
        elif pattern_type == "trend_seasonal":
            values = amplitude * t + amplitude * np.sin(2 * np.pi * t / 12)
        elif pattern_type == "step_change":
            values = np.where(t < length // 2, 0, amplitude)
        elif pattern_type == "random_walk":
            values = np.cumsum(rng.normal(0, amplitude, length))
        else:
            raise ValueError(f"Unknown pattern_type: {pattern_type}")

        # Add noise
        if noise_level > 0:
            values += rng.normal(0, noise_level * amplitude, length)

        return pl.DataFrame({"time": time, "target": values})

    return _factory


class DummyPointForecaster(BasePointForecaster):
    """Minimal point forecaster for composition testing."""

    def __init__(self, constant=0.0):
        super().__init__()
        self.constant = constant

    def fit(self, y, X=None, forecasting_horizon=1):
        super().fit(y, X, forecasting_horizon)
        return self

    def _predict_one(self):
        # Return constant prediction
        return pl.DataFrame(
            {
                col: [self.constant] * self.fit_forecasting_horizon_
                for col in list(self.local_y_schema_.keys())
            }
        )


class DummyIntervalForecaster(BaseIntervalForecaster):
    """Minimal interval forecaster for composition testing."""

    def __init__(self, width=1.0):
        super().__init__()
        self.width = width

    def fit(self, y, X=None, forecasting_horizon=1):
        super().fit(y, X, forecasting_horizon)
        return self

    def _predict_one(self):
        # Return constant interval
        data = {}
        for col in list(self.local_y_schema_.keys()):
            data[f"{col}_lower"] = [-self.width] * self.fit_forecasting_horizon_
            data[f"{col}_upper"] = [self.width] * self.fit_forecasting_horizon_

        return pl.DataFrame(data)
