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

    def fit(self, X, y=None):
        self.reset(X)
        return self

    def transform(self, X):
        return X.select([pl.col("time"), (cs.numeric() & ~cs.by_name("time")) + self.offset])

    def inverse_transform(self, X):
        return X.select([pl.col("time"), (cs.numeric() & ~cs.by_name("time")) - self.offset])

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class PanelAwareTransformer(BaseTransformer):
    """Transformer explicitly handling panel data struct columns."""

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

    def fit(self, X, y=None):
        self.reset(X)
        return self

    def transform(self, X):
        # Preserve struct columns (panel data)
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

    def _make(length=50, n_features=2, seed=42, min_length=None):
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
        features = {f"feature_{i}": rng + (i * 100) for i in range(n_features)}
        return pl.DataFrame({"time": time, **features})

    return _make


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
def panel_time_series_factory():
    """Factory for panel data with both global and local (struct) columns.

    Creates a DataFrame with:
    - "time" column (datetime)
    - Global columns (shared across all panels)
    - "panel" struct column (panel-specific time series)

    This tests transformers handling mixed global/local data.
    """

    def _make(length=50, n_series=3, n_global=2, seed=42):
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
            interval="1s",
            eager=True,
        )

        # Build panel data (local/struct columns)
        series_data = {}
        for i in range(n_series):
            series_data[f"series_{i}"] = range(i * 10, length + (i * 10))

        schema = {f"series_{i}": pl.Float64 for i in range(n_series)}

        # Create DataFrame with struct column
        panel_df = pl.DataFrame(
            {"panel": pl.DataFrame(series_data)}, schema={"panel": pl.Struct(schema)}
        )

        # Add global columns (shared features across all panels)
        global_features = {}
        for i in range(n_global):
            global_features[f"global_{i}"] = pl.Series(
                range(i * 100, length + (i * 100)), dtype=pl.Float64
            )

        # Combine: time + global columns + panel struct
        result = pl.DataFrame({"time": time, **global_features})
        result = pl.concat([result, panel_df], how="horizontal")

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
    used to test composition classes (Pipeline, FeatureUnion, etc.)
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
    - tags: Dictionary of transformer properties
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
            "tags": {"invertible": True, "stateful": True},
        },
        "SeasonalLogDifferencing": {
            "transformer": SeasonalLogDifferencing(
                seasonality=1
            ),  # seasonality=1 for first difference
            "expected_failed_checks": [],
            "tags": {"invertible": True, "requires_positive_X": True, "stateful": True},
        },
        "LagTransformer": {
            "transformer": LagTransformer(lag=[1, 2]),
            "expected_failed_checks": [
                "check_inverse_transform_identity",
                "check_inverse_transform_round_trip",
            ],
            "tags": {"invertible": False, "stateful": True},
        },
    }


# ============================================================================
# FORECASTER FIXTURES
# ============================================================================


@pytest.fixture
def y_X_factory():
    """Factory for generating (y, X_post, X_ante) tuples.

    Returns a callable that generates time series data for forecaster testing.
    """
    from datetime import datetime, timedelta

    import numpy as np

    def _factory(
        length=100, n_y_features=2, n_X_post_features=3, n_X_ante_features=2, seed=42, panel=False
    ):
        """Generate forecaster test data.

        Parameters
        ----------
        length : int
            Number of time steps
        n_y_features : int
            Number of target features
        n_X_post_features : int
            Number of ex-ante features (0 for None)
        n_X_ante_features : int
            Number of ex-post features (0 for None)
        seed : int
            Random seed
        panel : bool
            Whether to create panel data with struct columns

        Returns
        -------
        y : pl.DataFrame
            Target data with "time" column
        X_post : pl.DataFrame or None
            Ex-ante features with "time" column
        X_ante : pl.DataFrame or None
            Ex-post features with "time" column
        """
        rng = np.random.default_rng(seed)

        time = pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16) + timedelta(seconds=length - 1),
            interval="1s",
            eager=True,
        )

        # Generate y
        y = pl.DataFrame({"time": time})
        for i in range(n_y_features):
            y = y.with_columns(pl.Series(f"y_{i}", rng.random(length)))

        # Generate X_post
        X_post = None
        if n_X_post_features > 0:
            X_post = pl.DataFrame({"time": time})
            for i in range(n_X_post_features):
                X_post = X_post.with_columns(pl.Series(f"X_post_{i}", rng.random(length)))

        # Generate X_ante
        X_ante = None
        if n_X_ante_features > 0:
            X_ante = pl.DataFrame({"time": time})
            for i in range(n_X_ante_features):
                X_ante = X_ante.with_columns(pl.Series(f"X_ante_{i}", rng.random(length)))

        if panel:
            # TODO: Convert to struct columns for panel data
            # This would require implementing panel data conversion
            pass

        return y, X_post, X_ante

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
            "tags": {
                "forecaster_type": "point",
                "uses_reduction": False,
                "supports_panel_data": True,
                "uses_transformers": False,
            },
            "expected_failed_checks": [],
        },
        "PointReductionForecaster": {
            "forecaster": PointReductionForecaster(estimator=Ridge()),
            "tags": {
                "forecaster_type": "point",
                "uses_reduction": True,
                "supports_panel_data": True,
                "uses_transformers": True,
            },
            "expected_failed_checks": [],
        },
        "SplitConformalForecaster": {
            "forecaster": SplitConformalForecaster(
                point_forecaster=NaiveForecaster(seasonality=1),
                calibration_size=0.2,
            ),
            "tags": {
                "forecaster_type": "interval",
                "uses_reduction": False,
                "supports_panel_data": True,
                "uses_transformers": False,
            },
            "expected_failed_checks": ["check_reset_propagates_to_transformers"],
        },
    }
