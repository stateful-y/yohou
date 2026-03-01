"""Unit tests for base.py standalone functions.

Tests for:
- _fit_transform_transformers_one
- _build_feature_input
- _observe_transformers_one
- _rewind_transformers_one
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from yohou.base.utils import (
    _build_feature_input,
    _fit_transform_transformers_one,
    _observe_transformers_one,
    _rewind_transformers_one,
)
from yohou.stationarity.transformers import SeasonalDifferencing

sys.path.insert(0, str(Path(__file__).parent))
from conftest import SimpleTransformer as SimpleTransformerClass


@pytest.fixture
def SimpleTransformer():
    """Fixture that returns the SimpleTransformer class for instantiation in tests."""
    return SimpleTransformerClass


def make_exog_data(length, n_features):
    """Create exogenous data with unique column names to avoid conflicts with y data."""
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
        interval="1s",
        eager=True,
    )
    features = {f"exog_{i}": list(range(i * 100, i * 100 + length)) for i in range(n_features)}
    return pl.DataFrame({"time": time, **features})


class TestBuildFeatureInput:
    """Tests for _build_feature_input with various target_as_feature values."""

    def test_y_t_X_no_exog(self, time_series_factory):
        """Test with target_as_feature='transformed' and no exogenous features."""
        y = time_series_factory(length=50, n_components=2)
        y_t = y.select([pl.col("time"), pl.col("feature_0") + 10])

        result = _build_feature_input(y=y, y_t=y_t, X=None, target_as_feature="transformed", feature_transformer=None)

        assert result.equals(y_t)

    def test_y_t_X_with_exog(self, time_series_factory):
        """Test with target_as_feature='transformed' and exogenous features."""
        y = time_series_factory(length=50, n_components=2)
        y_t = y.select([pl.col("time"), pl.col("feature_0") + 10])
        X = make_exog_data(50, 3)

        result = _build_feature_input(y=y, y_t=y_t, X=X, target_as_feature="transformed", feature_transformer=None)

        assert len(result.columns) == 1 + 1 + 3
        assert "time" in result.columns
        assert "feature_0" in result.columns
        assert "exog_0" in result.columns

    def test_y_X_no_exog(self, time_series_factory):
        """Test with target_as_feature='raw' and no exogenous features."""
        y = time_series_factory(length=50, n_components=2)
        y_t = y.select([pl.col("time"), pl.col("feature_0") + 10])

        result = _build_feature_input(y=y, y_t=y_t, X=None, target_as_feature="raw", feature_transformer=None)

        assert result.equals(y)

    def test_y_X_with_exog(self, time_series_factory):
        """Test with target_as_feature='raw' and exogenous features."""
        y = time_series_factory(length=50, n_components=2)
        y_t = y.select([pl.col("time"), pl.col("feature_0") + 10])
        X = make_exog_data(50, 3)

        result = _build_feature_input(y=y, y_t=y_t, X=X, target_as_feature="raw", feature_transformer=None)

        assert len(result.columns) == 1 + 2 + 3

    def test_X_only_with_exog(self, time_series_factory):
        """Test with target_as_feature=None and exogenous features."""
        y = time_series_factory(length=50, n_components=2)
        y_t = y.select([pl.col("time"), pl.col("feature_0") + 10])
        X = make_exog_data(50, 3)

        result = _build_feature_input(y=y, y_t=y_t, X=X, target_as_feature=None, feature_transformer=None)

        assert result.equals(X)

    def test_X_only_no_exog_no_transformer(self, time_series_factory):
        """Test with target_as_feature=None, no exog, and no transformer."""
        y = time_series_factory(length=50, n_components=2)
        y_t = y.select([pl.col("time"), pl.col("feature_0") + 10])

        result = _build_feature_input(y=y, y_t=y_t, X=None, target_as_feature=None, feature_transformer=None)

        assert result is None

    def test_X_only_no_exog_with_transformer(self, time_series_factory, SimpleTransformer):
        """Test with target_as_feature=None, no exog, but has transformer."""
        y = time_series_factory(length=50, n_components=2)
        y_t = y.select([pl.col("time"), pl.col("feature_0") + 10])
        transformer = SimpleTransformer(observation_horizon=2)

        with pytest.raises(ValueError, match="target_as_feature=None requires X to be provided"):
            _build_feature_input(y=y, y_t=y_t, X=None, target_as_feature=None, feature_transformer=transformer)

    def test_invalid_target_as_feature(self, time_series_factory):
        """Test raises error for invalid target_as_feature value."""
        y = time_series_factory(length=50, n_components=2)
        y_t = y

        with pytest.raises(ValueError, match="Invalid target_as_feature="):
            _build_feature_input(y=y, y_t=y_t, X=None, target_as_feature="invalid", feature_transformer=None)


class TestFitTransformTransformersOne:
    """Tests for _fit_transform_transformers_one with various transformer configs."""

    def test_no_transformers(self, time_series_factory):
        """Test with no transformers passes data through unchanged."""
        y = time_series_factory(length=50, n_components=2)
        X = make_exog_data(50, 3)

        y_t, X_t, target_tf, feature_tf = _fit_transform_transformers_one(
            y=y, X=X, target_transformer=None, feature_transformer=None, target_as_feature="transformed"
        )

        assert y_t.equals(y)
        assert len(X_t.columns) == 1 + 2 + 3
        assert target_tf is None
        assert feature_tf is None

    def test_target_only(self, time_series_factory, SimpleTransformer):
        """Test with only target transformer."""
        y = time_series_factory(length=50, n_components=2)
        X = make_exog_data(50, 3)
        target_transformer = SimpleTransformer(observation_horizon=5, add_constant=10.0)

        y_t, X_t, target_tf, feature_tf = _fit_transform_transformers_one(
            y=y,
            X=X,
            target_transformer=target_transformer,
            feature_transformer=None,
            target_as_feature="transformed",
        )

        assert not y_t.equals(y)
        assert y_t.select(pl.col("feature_0"))[0, 0] == y.select(pl.col("feature_0"))[0, 0] + 10
        assert target_tf is not None
        assert target_tf.observation_horizon == 5
        assert feature_tf is None

    def test_feature_only(self, time_series_factory, SimpleTransformer):
        """Test with only feature transformer."""
        y = time_series_factory(length=50, n_components=2)
        X = make_exog_data(50, 3)
        feature_transformer = SimpleTransformer(observation_horizon=3, add_constant=5.0)

        y_t, X_t, target_tf, feature_tf = _fit_transform_transformers_one(
            y=y,
            X=X,
            target_transformer=None,
            feature_transformer=feature_transformer,
            target_as_feature="transformed",
        )

        assert y_t.equals(y[3:])
        assert X_t is not None
        assert feature_tf is not None
        assert feature_tf.observation_horizon == 3
        assert target_tf is None

    def test_both_transformers(self, time_series_factory, SimpleTransformer):
        """Test with both target and feature transformers."""
        y = time_series_factory(length=50, n_components=2)
        X = make_exog_data(50, 3)
        target_transformer = SimpleTransformer(observation_horizon=5, add_constant=10.0)
        feature_transformer = SimpleTransformer(observation_horizon=3, add_constant=5.0)

        y_t, X_t, target_tf, feature_tf = _fit_transform_transformers_one(
            y=y,
            X=X,
            target_transformer=target_transformer,
            feature_transformer=feature_transformer,
            target_as_feature="transformed",
        )

        assert target_tf is not None
        assert feature_tf is not None
        assert target_tf.observation_horizon == 5
        assert feature_tf.observation_horizon == 3
        assert len(y_t) == len(y) - 3
        assert X_t is not None

    def test_clones_transformers(self, time_series_factory, SimpleTransformer):
        """Test that transformers are cloned (originals not mutated)."""
        y = time_series_factory(length=50, n_components=2)
        target_transformer = SimpleTransformer(observation_horizon=5)

        with pytest.raises((NotFittedError, AttributeError), match="feature_names_in_|not fitted"):
            _ = target_transformer.feature_names_in_

        y_t, X_t, target_tf, feature_tf = _fit_transform_transformers_one(
            y=y,
            X=None,
            target_transformer=target_transformer,
            feature_transformer=None,
            target_as_feature="transformed",
        )

        assert hasattr(target_tf, "feature_names_in_")
        with pytest.raises((NotFittedError, AttributeError), match="feature_names_in_|not fitted"):
            _ = target_transformer.feature_names_in_

    def test_target_as_feature_raw(self, time_series_factory, SimpleTransformer):
        """Test with target_as_feature='raw'."""
        y = time_series_factory(length=50, n_components=2)
        X = make_exog_data(50, 3)
        target_transformer = SimpleTransformer(observation_horizon=5, add_constant=10.0)
        feature_transformer = SimpleTransformer(observation_horizon=3, add_constant=5.0)

        y_t, X_t, target_tf, feature_tf = _fit_transform_transformers_one(
            y=y,
            X=X,
            target_transformer=target_transformer,
            feature_transformer=feature_transformer,
            target_as_feature="raw",
        )

        assert target_tf is not None
        assert feature_tf is not None

    def test_target_as_feature_none(self, time_series_factory, SimpleTransformer):
        """Test with target_as_feature=None."""
        y = time_series_factory(length=50, n_components=2)
        X = make_exog_data(50, 3)
        target_transformer = SimpleTransformer(observation_horizon=5, add_constant=10.0)
        feature_transformer = SimpleTransformer(observation_horizon=3, add_constant=5.0)

        y_t, X_t, target_tf, feature_tf = _fit_transform_transformers_one(
            y=y,
            X=X,
            target_transformer=target_transformer,
            feature_transformer=feature_transformer,
            target_as_feature=None,
        )

        assert target_tf is not None
        assert feature_tf is not None
        assert len(X_t.columns) == len(X.columns)


class TestObserveTransformersOne:
    """Tests for _observe_transformers_one with various transformer configs."""

    def test_no_transformers(self, time_series_factory):
        """Test with no transformers still builds feature input."""
        y = time_series_factory(length=10, n_components=2)
        X = make_exog_data(10, 3)

        X_t = _observe_transformers_one(
            y=y, X=X, target_transformer=None, feature_transformer=None, target_as_feature="transformed"
        )

        assert X_t is not None
        assert len(X_t.columns) == 1 + 2 + 3

    def test_target_only(self, time_series_factory, SimpleTransformer):
        """Test with only target transformer."""
        y = time_series_factory(length=50, n_components=2)
        X = make_exog_data(50, 3)

        target_transformer = SimpleTransformer(observation_horizon=5, add_constant=10.0)
        target_transformer.fit(y[:40])

        y_new = y[40:45]
        X_new = X[40:45]

        X_t = _observe_transformers_one(
            y=y_new,
            X=X_new,
            target_transformer=target_transformer,
            feature_transformer=None,
            target_as_feature="transformed",
        )

        assert X_t is not None

    def test_feature_only(self, time_series_factory, SimpleTransformer):
        """Test with only feature transformer."""
        y = time_series_factory(length=50, n_components=2)
        X = make_exog_data(50, 3)

        feature_transformer = SimpleTransformer(observation_horizon=3, add_constant=5.0)
        X_feat_in = pl.concat([y, X.select(pl.exclude("time"))], how="horizontal")
        feature_transformer.fit(X_feat_in[:40])

        y_new = y[40:45]
        X_new = X[40:45]

        X_t = _observe_transformers_one(
            y=y_new,
            X=X_new,
            target_transformer=None,
            feature_transformer=feature_transformer,
            target_as_feature="transformed",
        )

        assert X_t is not None

    def test_both_transformers(self, time_series_factory, SimpleTransformer):
        """Test with both target and feature transformers."""
        y = time_series_factory(length=50, n_components=2)
        X = make_exog_data(50, 3)

        target_transformer = SimpleTransformer(observation_horizon=5, add_constant=10.0)
        target_transformer.fit(y[:40])

        y_t = target_transformer.transform(y[:40])
        X_feat_in = pl.concat([y_t, X[:40].select(pl.exclude("time"))], how="horizontal")
        feature_transformer = SimpleTransformer(observation_horizon=3, add_constant=5.0)
        feature_transformer.fit(X_feat_in)

        y_new = y[40:45]
        X_new = X[40:45]

        X_t = _observe_transformers_one(
            y=y_new,
            X=X_new,
            target_transformer=target_transformer,
            feature_transformer=feature_transformer,
            target_as_feature="transformed",
        )

        assert X_t is not None


class TestRewindTransformersOne:
    """Tests for _rewind_transformers_one with various transformer configs."""

    def test_no_transformers(self, time_series_factory):
        """Test with no transformers builds feature input from last rows."""
        y = time_series_factory(length=50, n_components=2)
        X = make_exog_data(50, 3)

        X_t = _rewind_transformers_one(
            y=y,
            X=X,
            target_transformer=None,
            feature_transformer=None,
            observation_horizon=5,
            target_as_feature="transformed",
        )

        assert X_t is not None

    def test_target_only(self, time_series_factory, SimpleTransformer):
        """Test with only target transformer."""
        y = time_series_factory(length=50, n_components=2)
        X = make_exog_data(50, 3)

        target_transformer = SimpleTransformer(observation_horizon=5, add_constant=10.0)
        target_transformer.fit(y[:40])

        observation_horizon = 5
        X_t = _rewind_transformers_one(
            y=y,
            X=X,
            target_transformer=target_transformer,
            feature_transformer=None,
            observation_horizon=observation_horizon,
            target_as_feature="transformed",
        )

        assert X_t is not None
        assert len(X_t) > 0

    def test_feature_only(self, time_series_factory, SimpleTransformer):
        """Test with only feature transformer."""
        y = time_series_factory(length=50, n_components=2)
        X = make_exog_data(50, 3)

        X_feat_in = pl.concat([y, X.select(pl.exclude("time"))], how="horizontal")
        feature_transformer = SimpleTransformer(observation_horizon=3, add_constant=5.0)
        feature_transformer.fit(X_feat_in[:40])

        observation_horizon = 5
        X_t = _rewind_transformers_one(
            y=y,
            X=X,
            target_transformer=None,
            feature_transformer=feature_transformer,
            observation_horizon=observation_horizon,
            target_as_feature="transformed",
        )

        assert X_t is not None
        assert len(X_t) == 1

    def test_both_transformers(self, time_series_factory, SimpleTransformer):
        """Test with both transformers.

        NOTE: This test currently fails due to a bug in _rewind_transformers_one.
        The function calls _build_feature_input with full y, but y_t (transformed y)
        has only observation_horizon rows. When concatenating y_t with X (which has full length),
        polars fills missing rows with null values, causing check_interval_consistency to fail.

        Potential fix: Pass y[-observation_horizon:] to _build_feature_input instead of full y.
        """
        y = time_series_factory(length=20, n_components=2)
        X = make_exog_data(20, 3)

        target_transformer = SimpleTransformer(observation_horizon=2, add_constant=10.0)
        target_transformer.fit(y)

        y_t = target_transformer.transform(y)
        X_feat_in = pl.concat([y_t, X.select(pl.exclude("time"))], how="horizontal")
        feature_transformer = SimpleTransformer(observation_horizon=2, add_constant=5.0)
        feature_transformer.fit(X_feat_in)

        observation_horizon = 5

        X_t = _rewind_transformers_one(
            y=y,
            X=X,
            target_transformer=target_transformer,
            feature_transformer=feature_transformer,
            observation_horizon=observation_horizon,
            target_as_feature="transformed",
        )

        assert X_t is not None
        assert len(X_t) == 1

    def test_stateful_transformer(self, time_series_factory):
        """Test with stateful SeasonalDifferencing transformer."""
        y = time_series_factory(length=50, n_components=1)

        target_transformer = SeasonalDifferencing(seasonality=5)
        target_transformer.fit(y[:40])

        observation_horizon = 10
        X_t = _rewind_transformers_one(
            y=y,
            X=None,
            target_transformer=target_transformer,
            feature_transformer=None,
            observation_horizon=observation_horizon,
            target_as_feature="transformed",
        )

        assert X_t is not None
        assert len(X_t) == observation_horizon

    def test_insufficient_data(self, time_series_factory, SimpleTransformer):
        """Test with insufficient data raises appropriate error."""
        y = time_series_factory(length=10, n_components=2)

        target_transformer = SimpleTransformer(observation_horizon=5, add_constant=10.0)
        target_transformer.fit(y)

        observation_horizon = 15

        with pytest.raises((ValueError, IndexError)):
            _rewind_transformers_one(
                y=y,
                X=None,
                target_transformer=target_transformer,
                feature_transformer=None,
                observation_horizon=observation_horizon,
                target_as_feature="transformed",
            )
