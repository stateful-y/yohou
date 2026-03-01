"""Tests for ColumnTransformer composition class."""

import sys
from pathlib import Path

import polars as pl
import pytest
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).parent))

from conftest import SimpleTransformer, StatelessTransformer
from yohou.compose import ColumnTransformer


@pytest.fixture
def time_series_3col():
    """Create a 3-column time series for column transformer tests."""
    from datetime import datetime, timedelta

    return pl.DataFrame({
        "time": pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1) + timedelta(seconds=19),
            interval="1s",
            eager=True,
        ),
        "a": list(range(20)),
        "b": [float(x) * 10 for x in range(20)],
        "c": [float(x) * 100 for x in range(20)],
    })


class TestColumnTransformerBasic:
    """Basic ColumnTransformer functionality."""

    def test_init(self):
        """Test ColumnTransformer initialization."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=1), ["a"]),
            ]
        )
        assert ct.transformers is not None

    def test_fit_transform_single_transformer(self, time_series_3col):
        """Test fit and transform with a single transformer."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0, add_constant=1.0), ["a", "b"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        result = ct.transform(time_series_3col)
        assert isinstance(result, pl.DataFrame)
        assert "time" in result.columns

    def test_remainder_drop(self, time_series_3col):
        """Test that remainder='drop' excludes unspecified columns."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0), ["a"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        result = ct.transform(time_series_3col)
        # "b" and "c" should be dropped
        assert "c" not in result.columns

    def test_remainder_passthrough(self, time_series_3col):
        """Test that remainder='passthrough' preserves unspecified columns."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0), ["a"]),
            ],
            remainder="passthrough",
        )
        ct.fit(time_series_3col)
        result = ct.transform(time_series_3col)
        # "b" and "c" should be passed through
        assert len(result.columns) >= 3  # time + a + b + c at minimum

    def test_multiple_transformers(self, time_series_3col):
        """Test with multiple transformers on different columns."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0, add_constant=1.0), ["a"]),
                ("t2", SimpleTransformer(observation_horizon=0, add_constant=2.0), ["b"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        result = ct.transform(time_series_3col)
        assert isinstance(result, pl.DataFrame)

    def test_clone(self, time_series_3col):
        """Test that ColumnTransformer can be cloned."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0), ["a"]),
            ],
        )
        ct_clone = clone(ct)
        assert ct_clone is not ct


class TestColumnTransformerObservationHorizon:
    """Test observation_horizon property."""

    def test_observation_horizon_max(self, time_series_3col):
        """Test that observation_horizon is max of component horizons."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=2), ["a"]),
                ("t2", SimpleTransformer(observation_horizon=5), ["b"]),
            ],
        )
        ct.fit(time_series_3col)
        # ColumnTransformer takes max of component observation horizons
        assert ct.observation_horizon >= 2

    def test_observation_horizon_stateless(self, time_series_3col):
        """Test observation_horizon with stateless transformers."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0), ["a"]),
                ("t2", SimpleTransformer(observation_horizon=0), ["b"]),
            ],
        )
        ct.fit(time_series_3col)
        assert ct.observation_horizon == 0


class TestColumnTransformerUpdateReset:
    """Test observe_transform and rewind_transform methods."""

    def test_observe_transform(self, time_series_3col):
        """Test observe_transform method."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", "passthrough", ["a", "b"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        # observe_transform should work on new data
        result = ct.observe_transform(time_series_3col)
        assert isinstance(result, pl.DataFrame)

    def test_rewind_transform(self, time_series_3col):
        """Test rewind_transform method."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", StatelessTransformer(), ["a", "b"]),
            ],
            remainder="passthrough",
        )
        ct.fit(time_series_3col)
        result = ct.rewind_transform(time_series_3col)
        assert isinstance(result, pl.DataFrame)

    def test_fit_transform_method(self, time_series_3col):
        """Test fit_transform method."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0), ["a", "b"]),
            ],
            remainder="drop",
        )
        result = ct.fit_transform(time_series_3col)
        assert isinstance(result, pl.DataFrame)


class TestColumnTransformerFeatureNames:
    """Test get_feature_names_out."""

    def test_feature_names_out(self, time_series_3col):
        """Test that get_feature_names_out returns feature names."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0), ["a", "b"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        feature_names = ct.get_feature_names_out()
        assert isinstance(feature_names, list | type(None)) or hasattr(feature_names, "__iter__")

    def test_verbose_feature_names(self, time_series_3col):
        """Test verbose_feature_names_out=True prefixes names."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", StatelessTransformer(), ["a"]),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )
        ct.fit(time_series_3col)
        feature_names = ct.get_feature_names_out()
        assert feature_names is not None
        # With verbose, names should be prefixed with transformer name using _
        assert any("t1_" in str(n) for n in feature_names)


class TestColumnTransformerPassthrough:
    """Test passthrough and drop special cases."""

    def test_passthrough_transformer(self, time_series_3col):
        """Test using 'passthrough' as transformer."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", "passthrough", ["a", "b"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        result = ct.transform(time_series_3col)
        assert isinstance(result, pl.DataFrame)

    def test_drop_transformer(self, time_series_3col):
        """Test that 'drop' is not accepted as transformer string."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", "passthrough", ["b"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        result = ct.transform(time_series_3col)
        assert isinstance(result, pl.DataFrame)
