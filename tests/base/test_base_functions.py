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

        result = _build_feature_input(
            y=y, y_t=y_t, X_actual=None, target_as_feature="transformed", actual_transformer=None
        )

        assert result.equals(y_t)

    def test_y_t_X_with_exog(self, time_series_factory):
        """Test with target_as_feature='transformed' and exogenous features."""
        y = time_series_factory(length=50, n_components=2)
        y_t = y.select([pl.col("time"), pl.col("feature_0") + 10])
        X_actual = make_exog_data(50, 3)

        result = _build_feature_input(
            y=y, y_t=y_t, X_actual=X_actual, target_as_feature="transformed", actual_transformer=None
        )

        assert len(result.columns) == 1 + 1 + 3
        assert "time" in result.columns
        assert "feature_0" in result.columns
        assert "exog_0" in result.columns

    def test_y_X_no_exog(self, time_series_factory):
        """Test with target_as_feature='raw' and no exogenous features."""
        y = time_series_factory(length=50, n_components=2)
        y_t = y.select([pl.col("time"), pl.col("feature_0") + 10])

        result = _build_feature_input(y=y, y_t=y_t, X_actual=None, target_as_feature="raw", actual_transformer=None)

        assert result.equals(y)

    def test_y_X_with_exog(self, time_series_factory):
        """Test with target_as_feature='raw' and exogenous features."""
        y = time_series_factory(length=50, n_components=2)
        y_t = y.select([pl.col("time"), pl.col("feature_0") + 10])
        X_actual = make_exog_data(50, 3)

        result = _build_feature_input(y=y, y_t=y_t, X_actual=X_actual, target_as_feature="raw", actual_transformer=None)

        assert len(result.columns) == 1 + 2 + 3

    def test_X_only_with_exog(self, time_series_factory):
        """Test with target_as_feature=None and exogenous features."""
        y = time_series_factory(length=50, n_components=2)
        y_t = y.select([pl.col("time"), pl.col("feature_0") + 10])
        X_actual = make_exog_data(50, 3)

        result = _build_feature_input(y=y, y_t=y_t, X_actual=X_actual, target_as_feature=None, actual_transformer=None)

        assert result.equals(X_actual)

    def test_X_only_no_exog_no_transformer(self, time_series_factory):
        """Test with target_as_feature=None, no exog, and no transformer."""
        y = time_series_factory(length=50, n_components=2)
        y_t = y.select([pl.col("time"), pl.col("feature_0") + 10])

        result = _build_feature_input(y=y, y_t=y_t, X_actual=None, target_as_feature=None, actual_transformer=None)

        assert result is None

    def test_X_only_no_exog_with_transformer(self, time_series_factory, SimpleTransformer):
        """Test with target_as_feature=None, no exog, but has transformer."""
        y = time_series_factory(length=50, n_components=2)
        y_t = y.select([pl.col("time"), pl.col("feature_0") + 10])
        transformer = SimpleTransformer(observation_horizon=2)

        with pytest.raises(ValueError, match="target_as_feature=None requires X_actual to be provided"):
            _build_feature_input(y=y, y_t=y_t, X_actual=None, target_as_feature=None, actual_transformer=transformer)

    def test_invalid_target_as_feature(self, time_series_factory):
        """Test raises error for invalid target_as_feature value."""
        y = time_series_factory(length=50, n_components=2)
        y_t = y

        with pytest.raises(ValueError, match="Invalid target_as_feature="):
            _build_feature_input(y=y, y_t=y_t, X_actual=None, target_as_feature="invalid", actual_transformer=None)


class TestFitTransformTransformersOne:
    """Tests for _fit_transform_transformers_one with various transformer configs."""

    def test_no_transformers(self, time_series_factory):
        """Test with no transformers passes data through unchanged."""
        y = time_series_factory(length=50, n_components=2)
        X_actual = make_exog_data(50, 3)

        y_t, X_t, target_tf, feature_tf = _fit_transform_transformers_one(
            y=y, X_actual=X_actual, target_transformer=None, actual_transformer=None, target_as_feature="transformed"
        )

        assert y_t.equals(y)
        assert len(X_t.columns) == 1 + 2 + 3
        assert target_tf is None
        assert feature_tf is None

    def test_target_only(self, time_series_factory, SimpleTransformer):
        """Test with only target transformer."""
        y = time_series_factory(length=50, n_components=2)
        X_actual = make_exog_data(50, 3)
        target_transformer = SimpleTransformer(observation_horizon=5, add_constant=10.0)

        y_t, X_t, target_tf, feature_tf = _fit_transform_transformers_one(
            y=y,
            X_actual=X_actual,
            target_transformer=target_transformer,
            actual_transformer=None,
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
        X_actual = make_exog_data(50, 3)
        actual_transformer = SimpleTransformer(observation_horizon=3, add_constant=5.0)

        y_t, X_t, target_tf, feature_tf = _fit_transform_transformers_one(
            y=y,
            X_actual=X_actual,
            target_transformer=None,
            actual_transformer=actual_transformer,
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
        X_actual = make_exog_data(50, 3)
        target_transformer = SimpleTransformer(observation_horizon=5, add_constant=10.0)
        actual_transformer = SimpleTransformer(observation_horizon=3, add_constant=5.0)

        y_t, X_t, target_tf, feature_tf = _fit_transform_transformers_one(
            y=y,
            X_actual=X_actual,
            target_transformer=target_transformer,
            actual_transformer=actual_transformer,
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
            X_actual=None,
            target_transformer=target_transformer,
            actual_transformer=None,
            target_as_feature="transformed",
        )

        assert hasattr(target_tf, "feature_names_in_")
        with pytest.raises((NotFittedError, AttributeError), match="feature_names_in_|not fitted"):
            _ = target_transformer.feature_names_in_

    def test_target_as_feature_raw(self, time_series_factory, SimpleTransformer):
        """Test with target_as_feature='raw'."""
        y = time_series_factory(length=50, n_components=2)
        X_actual = make_exog_data(50, 3)
        target_transformer = SimpleTransformer(observation_horizon=5, add_constant=10.0)
        actual_transformer = SimpleTransformer(observation_horizon=3, add_constant=5.0)

        y_t, X_t, target_tf, feature_tf = _fit_transform_transformers_one(
            y=y,
            X_actual=X_actual,
            target_transformer=target_transformer,
            actual_transformer=actual_transformer,
            target_as_feature="raw",
        )

        assert target_tf is not None
        assert feature_tf is not None

    def test_target_as_feature_none(self, time_series_factory, SimpleTransformer):
        """Test with target_as_feature=None."""
        y = time_series_factory(length=50, n_components=2)
        X_actual = make_exog_data(50, 3)
        target_transformer = SimpleTransformer(observation_horizon=5, add_constant=10.0)
        actual_transformer = SimpleTransformer(observation_horizon=3, add_constant=5.0)

        y_t, X_t, target_tf, feature_tf = _fit_transform_transformers_one(
            y=y,
            X_actual=X_actual,
            target_transformer=target_transformer,
            actual_transformer=actual_transformer,
            target_as_feature=None,
        )

        assert target_tf is not None
        assert feature_tf is not None
        assert len(X_t.columns) == len(X_actual.columns)


class TestObserveTransformersOne:
    """Tests for _observe_transformers_one with various transformer configs."""

    def test_no_transformers(self, time_series_factory):
        """Test with no transformers still builds feature input."""
        y = time_series_factory(length=10, n_components=2)
        X_actual = make_exog_data(10, 3)

        X_t = _observe_transformers_one(
            y=y, X_actual=X_actual, target_transformer=None, actual_transformer=None, target_as_feature="transformed"
        )

        assert X_t is not None
        assert len(X_t.columns) == 1 + 2 + 3

    def test_target_only(self, time_series_factory, SimpleTransformer):
        """Test with only target transformer."""
        y = time_series_factory(length=50, n_components=2)
        X_actual = make_exog_data(50, 3)

        target_transformer = SimpleTransformer(observation_horizon=5, add_constant=10.0)
        target_transformer.fit(y[:40])

        y_new = y[40:45]
        X_new = X_actual[40:45]

        X_t = _observe_transformers_one(
            y=y_new,
            X_actual=X_new,
            target_transformer=target_transformer,
            actual_transformer=None,
            target_as_feature="transformed",
        )

        # 1 time + 2 transformed target features + 3 exog columns over the window.
        assert len(X_t.columns) == 1 + 2 + 3
        assert len(X_t) == 5
        # Target transformer adds 10 to feature_0; exog passes through unchanged.
        assert X_t["feature_0"][0] == y_new["feature_0"][0] + 10.0
        assert X_t["exog_0"][0] == X_new["exog_0"][0]

    def test_feature_only(self, time_series_factory, SimpleTransformer):
        """Test with only feature transformer."""
        y = time_series_factory(length=50, n_components=2)
        X_actual = make_exog_data(50, 3)

        actual_transformer = SimpleTransformer(observation_horizon=3, add_constant=5.0)
        X_feat_in = pl.concat([y, X_actual.select(pl.exclude("time"))], how="horizontal")
        actual_transformer.fit(X_feat_in[:40])

        y_new = y[40:45]
        X_new = X_actual[40:45]

        X_t = _observe_transformers_one(
            y=y_new,
            X_actual=X_new,
            target_transformer=None,
            actual_transformer=actual_transformer,
            target_as_feature="transformed",
        )

        assert len(X_t.columns) == 1 + 2 + 3
        assert len(X_t) == 5
        # Feature transformer adds 5 to every numeric column (target and exog).
        assert X_t["feature_0"][0] == y_new["feature_0"][0] + 5.0
        assert X_t["exog_0"][0] == X_new["exog_0"][0] + 5.0

    def test_both_transformers(self, time_series_factory, SimpleTransformer):
        """Test with both target and feature transformers."""
        y = time_series_factory(length=50, n_components=2)
        X_actual = make_exog_data(50, 3)

        target_transformer = SimpleTransformer(observation_horizon=5, add_constant=10.0)
        target_transformer.fit(y[:40])

        y_t = target_transformer.transform(y[:40])
        X_feat_in = pl.concat([y_t, X_actual[:40].select(pl.exclude("time"))], how="horizontal")
        actual_transformer = SimpleTransformer(observation_horizon=3, add_constant=5.0)
        actual_transformer.fit(X_feat_in)

        y_new = y[40:45]
        X_new = X_actual[40:45]

        X_t = _observe_transformers_one(
            y=y_new,
            X_actual=X_new,
            target_transformer=target_transformer,
            actual_transformer=actual_transformer,
            target_as_feature="transformed",
        )

        assert len(X_t.columns) == 1 + 2 + 3
        assert len(X_t) == 5
        # Target adds 10 then feature adds 5 to the target column (+15 total);
        # exog only sees the feature transformer (+5).
        assert X_t["feature_0"][0] == y_new["feature_0"][0] + 15.0
        assert X_t["exog_0"][0] == X_new["exog_0"][0] + 5.0


class TestRewindTransformersOne:
    """Tests for _rewind_transformers_one with various transformer configs."""

    def test_no_transformers(self, time_series_factory):
        """Test with no transformers builds feature input from last rows."""
        y = time_series_factory(length=50, n_components=2)
        X_actual = make_exog_data(50, 3)

        X_t = _rewind_transformers_one(
            y=y,
            X_actual=X_actual,
            target_transformer=None,
            actual_transformer=None,
            observation_horizon=5,
            target_as_feature="transformed",
        )

        # 1 time + 2 raw target features + 3 exog columns; no transformer means
        # the feature input is the full concatenation, ending at the last row.
        assert len(X_t.columns) == 1 + 2 + 3
        assert X_t["time"][-1] == y["time"][-1]
        assert X_t["feature_0"][-1] == y["feature_0"][-1]

    def test_target_only(self, time_series_factory, SimpleTransformer):
        """Test with only target transformer."""
        y = time_series_factory(length=50, n_components=2)
        X_actual = make_exog_data(50, 3)

        target_transformer = SimpleTransformer(observation_horizon=5, add_constant=10.0)
        target_transformer.fit(y[:40])

        observation_horizon = 5
        X_t = _rewind_transformers_one(
            y=y,
            X_actual=X_actual,
            target_transformer=target_transformer,
            actual_transformer=None,
            observation_horizon=observation_horizon,
            target_as_feature="transformed",
        )

        assert X_t is not None
        assert len(X_t) > 0

    def test_feature_only(self, time_series_factory, SimpleTransformer):
        """Test with only feature transformer."""
        y = time_series_factory(length=50, n_components=2)
        X_actual = make_exog_data(50, 3)

        X_feat_in = pl.concat([y, X_actual.select(pl.exclude("time"))], how="horizontal")
        actual_transformer = SimpleTransformer(observation_horizon=3, add_constant=5.0)
        actual_transformer.fit(X_feat_in[:40])

        observation_horizon = 5
        X_t = _rewind_transformers_one(
            y=y,
            X_actual=X_actual,
            target_transformer=None,
            actual_transformer=actual_transformer,
            observation_horizon=observation_horizon,
            target_as_feature="transformed",
        )

        assert X_t is not None
        assert len(X_t) == 1

    def test_both_transformers(self, time_series_factory, SimpleTransformer):
        """Both transformers rewind to a single aligned feature row."""
        y = time_series_factory(length=20, n_components=2)
        X_actual = make_exog_data(20, 3)

        target_transformer = SimpleTransformer(observation_horizon=2, add_constant=10.0)
        target_transformer.fit(y)

        y_t = target_transformer.transform(y)
        X_feat_in = pl.concat([y_t, X_actual.select(pl.exclude("time"))], how="horizontal")
        actual_transformer = SimpleTransformer(observation_horizon=2, add_constant=5.0)
        actual_transformer.fit(X_feat_in)

        observation_horizon = 5

        X_t = _rewind_transformers_one(
            y=y,
            X_actual=X_actual,
            target_transformer=target_transformer,
            actual_transformer=actual_transformer,
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
            X_actual=None,
            target_transformer=target_transformer,
            actual_transformer=None,
            observation_horizon=observation_horizon,
            target_as_feature="transformed",
        )

        assert X_t is not None
        assert len(X_t) == observation_horizon

    def test_zero_observation_horizon(self, time_series_factory, SimpleTransformer):
        """observation_horizon == 0 rewinds over all rows, not an empty slice.

        With negative slicing, y[:-0] is empty and y[-0:] is the full frame,
        which inverts the rewind/observe windows. The explicit split index keeps
        the rewind window as the full frame and the observe window empty.
        """
        y = time_series_factory(length=20, n_components=2)
        X_actual = make_exog_data(20, 3)

        target_transformer = SimpleTransformer(observation_horizon=0, add_constant=10.0)
        target_transformer.fit(y)

        X_t = _rewind_transformers_one(
            y=y,
            X_actual=X_actual,
            target_transformer=target_transformer,
            actual_transformer=None,
            observation_horizon=0,
            target_as_feature="transformed",
        )

        # Zero horizon implies an empty observation window, not the full frame
        # that the buggy y[-0:] slice would have produced.
        assert X_t is not None
        assert len(X_t) == 0

    def test_insufficient_data(self, time_series_factory, SimpleTransformer):
        """Test with insufficient data raises appropriate error."""
        y = time_series_factory(length=10, n_components=2)

        target_transformer = SimpleTransformer(observation_horizon=5, add_constant=10.0)
        target_transformer.fit(y)

        observation_horizon = 15

        with pytest.raises((ValueError, IndexError)):
            _rewind_transformers_one(
                y=y,
                X_actual=None,
                target_transformer=target_transformer,
                actual_transformer=None,
                observation_horizon=observation_horizon,
                target_as_feature="transformed",
            )

    def test_feature_output_without_time_column_uses_tail(self, time_series_factory):
        """A feature transformer that drops 'time' falls back to the last row."""
        from unittest.mock import MagicMock

        y = time_series_factory(length=20, n_components=1)
        X_actual = make_exog_data(20, 2)

        actual_transformer = MagicMock()
        actual_transformer.observation_horizon = 2
        # rewind_transform output has no 'time' column.
        actual_transformer.rewind_transform.return_value = pl.DataFrame({"f": [1.0, 2.0, 3.0]})

        X_t = _rewind_transformers_one(
            y=y,
            X_actual=X_actual,
            target_transformer=None,
            actual_transformer=actual_transformer,
            observation_horizon=5,
            target_as_feature=None,
        )

        # No time column to align on, so the last row is taken.
        assert X_t.height == 1
        assert X_t["f"][0] == 3.0

    def test_feature_output_missing_last_time_uses_tail(self, time_series_factory):
        """When the latest observation is dropped, alignment falls back to the tail."""
        from unittest.mock import MagicMock

        y = time_series_factory(length=20, n_components=1)
        X_actual = make_exog_data(20, 2)
        last_time = y["time"][-1]

        # rewind_transform keeps 'time' but never includes last_time, so the
        # timestamp filter is empty and the tail(1) fallback is used.
        earlier = [last_time - timedelta(days=2), last_time - timedelta(days=1)]
        actual_transformer = MagicMock()
        actual_transformer.observation_horizon = 2
        actual_transformer.rewind_transform.return_value = pl.DataFrame({"time": earlier, "f": [1.0, 2.0]})

        X_t = _rewind_transformers_one(
            y=y,
            X_actual=X_actual,
            target_transformer=None,
            actual_transformer=actual_transformer,
            observation_horizon=5,
            target_as_feature=None,
        )

        assert X_t.height == 1
        assert X_t["f"][0] == 2.0
