"""Tests for compose utility functions."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import polars as pl
import polars.testing as plt
import pytest

from yohou.base import BaseActualTransformer


class TestHstack:
    """Tests for _hstack horizontal stacking with time-intersection alignment."""

    @pytest.fixture()
    def time_df(self):
        """Create a simple time series DataFrame."""
        return pl.DataFrame({
            "time": pl.datetime_range(
                start=datetime(2021, 1, 1),
                end=datetime(2021, 1, 1) + timedelta(days=9),
                interval="1d",
                eager=True,
            ),
            "a": list(range(10)),
        })

    def test_single_df_no_trim(self, time_df):
        """A single DataFrame passes through unchanged."""
        from yohou.compose.utils import _hstack

        result = _hstack(
            Xs=[time_df],
            column_names=[["a"]],
        )
        assert result.shape == time_df.shape
        assert result.columns == ["time", "a"]

    def test_equal_horizons(self, time_df):
        """Two DataFrames with equal observation horizons stack without trimming."""
        from yohou.compose.utils import _hstack

        df2 = time_df.rename({"a": "b"})
        result = _hstack(
            Xs=[time_df, df2],
            column_names=[["a"], ["b"]],
        )
        assert result.columns == ["time", "a", "b"]
        assert result.shape[0] == 10

    def test_children_disagreeing_on_row_order_still_align(self, time_df):
        """A child that emits the shared rows in a different order still aligns by index.

        Row order is not part of the contract: a child may re-group its rows (as
        PerVintageActualTransformer does via partition_by). Features must attach
        to the row bearing their own index, not to whatever row shares their
        position.
        """
        from yohou.compose.utils import _hstack

        reversed_df = time_df.rename({"a": "b"}).reverse()
        result = _hstack(
            Xs=[time_df, reversed_df],
            column_names=[["a"], ["b"]],
        )

        # b carries the same values as a, so on every row a == b once aligned by time.
        assert result["a"].to_list() == result["b"].to_list()

    def test_forecast_kind_children_disagreeing_on_row_order_still_align(self):
        """The same guarantee holds for a two-axis (vintage_time, time) index."""
        from yohou.compose.utils import _hstack

        t = datetime(2020, 1, 1)
        index = {
            "vintage_time": [t, t],
            "time": [t + timedelta(days=2), t + timedelta(days=3)],
        }
        a = pl.DataFrame({**index, "a": [10.0, 20.0]})
        b = pl.DataFrame({**index, "b": [30.0, 40.0]}).reverse()

        result = _hstack(Xs=[a, b], column_names=[["a"], ["b"]]).sort("time")

        assert result["a"].to_list() == [10.0, 20.0]
        assert result["b"].to_list() == [30.0, 40.0]

    def test_different_lengths_align_to_time_intersection(self, time_df):
        """DataFrames with different row counts align on their shared timestamps.

        ``_hstack`` keeps only the intersection of ``"time"`` values across all
        inputs, so a DataFrame missing its leading rows (e.g. because a
        transformer dropped warmup rows) trims the others down to the shared
        range rather than producing nulls.
        """
        from yohou.compose.utils import _hstack

        # df1 starts at row 2 (rows 0-1 absent); df2 has all 10 rows.
        df1 = time_df[2:]  # 8 rows, has "time" and "a"
        df2 = time_df.rename({"a": "b"})  # 10 rows, has "time" and "b"

        result = _hstack(
            Xs=[df1, df2],
            column_names=[["a"], ["b"]],
        )
        # Intersection of df1 (rows 2-9) and df2 (rows 0-9) is rows 2-9 → 8 rows.
        assert result.shape[0] == 8
        assert result.columns == ["time", "a", "b"]
        assert result["time"][0] == datetime(2021, 1, 3)

    def test_renames_columns(self, time_df):
        """Columns are renamed according to column_names parameter."""
        from yohou.compose.utils import _hstack

        result = _hstack(
            Xs=[time_df],
            column_names=[["renamed_a"]],
        )
        assert "renamed_a" in result.columns
        assert "a" not in result.columns

    def test_time_column_aligned_to_intersection_start(self, time_df):
        """Time column starts at the first timestamp shared by all inputs."""
        from yohou.compose.utils import _hstack

        # df1 has all 10 rows; df2 is missing its first 2 rows.
        df2 = time_df[2:].rename({"a": "b"})
        result = _hstack(
            Xs=[time_df, df2],
            column_names=[["a"], ["b"]],
        )
        # Shared range starts where df2 starts (row 2 → 2021-01-03).
        expected_start = datetime(2021, 1, 3)
        assert result["time"][0] == expected_start


class TestObserveTransformOne:
    """Tests for _observe_transform_one."""

    def test_calls_observe_transform_when_available(self):
        """Should call observe_transform on transformers that support it."""
        from yohou.compose.utils import _observe_transform_one

        expected = pl.DataFrame({"time": [1], "a": [10]})
        transformer = MagicMock()
        transformer.observe_transform.return_value = expected
        X = pl.DataFrame({"time": [1], "a": [5]})

        result = _observe_transform_one(transformer, X, y=None, weight=None, params={"observe_transform": {}})

        transformer.observe_transform.assert_called_once()
        plt.assert_frame_equal(result, expected)

    def test_falls_back_to_transform_without_observe(self):
        """Should use transform() for transformers without observe_transform."""
        from yohou.compose.utils import _observe_transform_one

        expected = pl.DataFrame({"time": [1], "a": [10]})
        transformer = MagicMock(spec=["transform"])
        transformer.transform.return_value = expected
        X = pl.DataFrame({"time": [1], "a": [5]})

        result = _observe_transform_one(transformer, X, y=None, weight=None, params={})

        transformer.transform.assert_called_once_with(X)
        plt.assert_frame_equal(result, expected)

    def test_applies_weight(self):
        """Weight scales feature columns but leaves the datetime time column intact."""
        from yohou.compose.utils import _observe_transform_one

        times = [datetime(2021, 1, 1), datetime(2021, 1, 2)]
        transformed = pl.DataFrame({"time": times, "a": [10.0, 20.0]})
        transformer = MagicMock()
        transformer.observe_transform.return_value = transformed
        X = pl.DataFrame({"time": times, "a": [5.0, 10.0]})

        result = _observe_transform_one(transformer, X, y=None, weight=0.5, params={"observe_transform": {}})

        expected = pl.DataFrame({"time": times, "a": [5.0, 10.0]})
        plt.assert_frame_equal(result, expected)
        # The time column keeps its datetime dtype; only "a" is scaled.
        assert result.schema["time"] == pl.Datetime

    def test_no_weight_returns_unmodified(self):
        """None weight returns transformed output as-is."""
        from yohou.compose.utils import _observe_transform_one

        transformed = pl.DataFrame({"a": [10.0, 20.0]})
        transformer = MagicMock()
        transformer.observe_transform.return_value = transformed
        X = pl.DataFrame({"a": [5.0, 10.0]})

        result = _observe_transform_one(transformer, X, y=None, weight=None, params={"observe_transform": {}})

        plt.assert_frame_equal(result, transformed)


class TestRewindTransformOne:
    """Tests for _rewind_transform_one."""

    def test_calls_rewind_transform_when_available(self):
        """Should call rewind_transform on transformers that support it."""
        from yohou.compose.utils import _rewind_transform_one

        expected = pl.DataFrame({"time": [1], "a": [10]})
        transformer = MagicMock()
        transformer.rewind_transform.return_value = expected
        X = pl.DataFrame({"time": [1], "a": [5]})

        result = _rewind_transform_one(transformer, X, y=None, weight=None, params={"rewind_transform": {}})

        transformer.rewind_transform.assert_called_once()
        plt.assert_frame_equal(result, expected)

    def test_falls_back_to_transform_without_rewind(self):
        """Should use transform() for transformers without rewind_transform."""
        from yohou.compose.utils import _rewind_transform_one

        expected = pl.DataFrame({"time": [1], "a": [10]})
        transformer = MagicMock(spec=["transform"])
        transformer.transform.return_value = expected
        X = pl.DataFrame({"time": [1], "a": [5]})

        result = _rewind_transform_one(transformer, X, y=None, weight=None, params={})

        transformer.transform.assert_called_once_with(X)
        plt.assert_frame_equal(result, expected)

    def test_applies_weight(self):
        """Weight scales feature columns but leaves the datetime time column intact."""
        from yohou.compose.utils import _rewind_transform_one

        times = [datetime(2021, 1, 1), datetime(2021, 1, 2)]
        transformed = pl.DataFrame({"time": times, "a": [10.0, 20.0]})
        transformer = MagicMock()
        transformer.rewind_transform.return_value = transformed
        X = pl.DataFrame({"time": times, "a": [5.0, 10.0]})

        result = _rewind_transform_one(transformer, X, y=None, weight=0.5, params={"rewind_transform": {}})

        expected = pl.DataFrame({"time": times, "a": [5.0, 10.0]})
        plt.assert_frame_equal(result, expected)
        # The time column keeps its datetime dtype; only "a" is scaled.
        assert result.schema["time"] == pl.Datetime


class _StatefulWithoutMethods(BaseActualTransformer):
    """A BaseActualTransformer that has lost its stateful observe/rewind methods.

    Used to verify the guards reject a broken stateful transformer rather
    than silently falling back to a stateless ``transform``.
    """

    @property
    def observe_transform(self):
        raise AttributeError("observe_transform removed")

    @property
    def rewind_transform(self):
        raise AttributeError("rewind_transform removed")

    def _fit(self, X, y=None):
        return None

    def _transform(self, X):
        return X

    def get_feature_names_out(self, input_features=None):
        return ["a"]


class TestStatefulFallbackGuards:
    """Stateful BaseTransformers missing their methods must error, not fall back."""

    def test_observe_transform_one_rejects_broken_stateful(self):
        """A BaseActualTransformer without observe_transform raises AttributeError."""
        from yohou.compose.utils import _observe_transform_one

        X = pl.DataFrame({"time": [1], "a": [1.0]})
        with pytest.raises(AttributeError, match="no 'observe_transform' method"):
            _observe_transform_one(_StatefulWithoutMethods(), X, y=None, weight=None, params={})

    def test_rewind_transform_one_rejects_broken_stateful(self):
        """A BaseActualTransformer without rewind_transform raises AttributeError."""
        from yohou.compose.utils import _rewind_transform_one

        X = pl.DataFrame({"time": [1], "a": [1.0]})
        with pytest.raises(AttributeError, match="no 'rewind_transform' method"):
            _rewind_transform_one(_StatefulWithoutMethods(), X, y=None, weight=None, params={})
