"""Tests for compose utility functions."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import polars as pl
import polars.testing as plt
import pytest


class TestHstack:
    """Tests for _hstack horizontal stacking with observation horizon alignment."""

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
        """Single DataFrame with observation_horizon=1 should pass through."""
        from yohou.compose.utils import _hstack

        result = _hstack(
            Xs=[time_df],
            column_names=[["a"]],
            observation_horizons=[1],
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
            observation_horizons=[1, 1],
        )
        assert result.columns == ["time", "a", "b"]
        assert result.shape[0] == 10

    def test_different_horizons_trims_shorter(self, time_df):
        """DataFrames from transformers with different horizons get aligned.

        A transformer with observation_horizon=3 outputs N-2 rows (warmup),
        while horizon=1 outputs all N rows. _hstack aligns by trimming the
        lower-horizon outputs' leading rows.
        """
        from yohou.compose.utils import _hstack

        # Simulate: horizon=3 transformer already consumed 2 warmup rows → 8 rows
        df1 = time_df[2:]  # 8 rows, has "time" and "a"
        # Simulate: horizon=1 transformer kept all rows → 10 rows
        df2 = time_df.rename({"a": "b"})  # 10 rows, has "time" and "b"

        result = _hstack(
            Xs=[df1, df2],
            column_names=[["a"], ["b"]],
            observation_horizons=[3, 1],
        )
        # Max horizon=3. df1 trimmed by 3-3=0 → 8 rows. df2 trimmed by 3-1=2 → 8 rows.
        assert result.shape[0] == 8
        assert result.columns == ["time", "a", "b"]

    def test_renames_columns(self, time_df):
        """Columns are renamed according to column_names parameter."""
        from yohou.compose.utils import _hstack

        result = _hstack(
            Xs=[time_df],
            column_names=[["renamed_a"]],
            observation_horizons=[1],
        )
        assert "renamed_a" in result.columns
        assert "a" not in result.columns

    def test_time_column_aligned_to_max_horizon(self, time_df):
        """Time column from first DataFrame aligned to max horizon difference."""
        from yohou.compose.utils import _hstack

        # df1: horizon=1, all 10 rows
        # df2: horizon=3, 8 rows (2 warmup consumed)
        df2 = time_df[2:].rename({"a": "b"})
        result = _hstack(
            Xs=[time_df, df2],
            column_names=[["a"], ["b"]],
            observation_horizons=[1, 3],
        )
        # Max horizon=3. df1 trimmed by 3-1=2 → starts at row 2.
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
        """Weight multiplies the transformed output."""
        from yohou.compose.utils import _observe_transform_one

        transformed = pl.DataFrame({"a": [10.0, 20.0]})
        transformer = MagicMock()
        transformer.observe_transform.return_value = transformed
        X = pl.DataFrame({"a": [5.0, 10.0]})

        result = _observe_transform_one(transformer, X, y=None, weight=0.5, params={"observe_transform": {}})

        expected = pl.DataFrame({"a": [5.0, 10.0]})
        plt.assert_frame_equal(result, expected)

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
        """Weight multiplies the rewind-transformed output."""
        from yohou.compose.utils import _rewind_transform_one

        transformed = pl.DataFrame({"a": [10.0, 20.0]})
        transformer = MagicMock()
        transformer.rewind_transform.return_value = transformed
        X = pl.DataFrame({"a": [5.0, 10.0]})

        result = _rewind_transform_one(transformer, X, y=None, weight=0.5, params={"rewind_transform": {}})

        expected = pl.DataFrame({"a": [5.0, 10.0]})
        plt.assert_frame_equal(result, expected)
