"""Unit tests for plotting utilities."""

import polars as pl
import pytest

from yohou.plotting._utils import (
    _normalize_y_pred,
    resolve_color_palette,
    validate_plotting_params,
)
from yohou.utils import validate_plotting_data


class TestResolveColorPalette:
    """Tests for resolve_color_palette utility."""

    def test_none_returns_default(self):
        """None palette falls back to default colour sequence."""
        result = resolve_color_palette(None, 3)
        assert len(result) == 3
        assert all(isinstance(c, str) for c in result)

    def test_custom_palette_exact(self):
        """Custom palette with exact count returns as-is."""
        palette = ["red", "green", "blue"]
        result = resolve_color_palette(palette, 3)
        assert result == palette

    def test_custom_palette_cycles(self):
        """Custom palette cycles when n > len(palette)."""
        palette = ["red", "blue"]
        result = resolve_color_palette(palette, 4)
        assert result == ["red", "blue", "red", "blue"]

    def test_zero_returns_empty(self):
        """Requesting zero colours returns empty list."""
        assert resolve_color_palette(None, 0) == []
        assert resolve_color_palette(["red"], 0) == []


class TestValidatePlottingParams:
    """Tests for validate_plotting_params utility."""

    def test_valid_kind(self):
        """Valid kind passes silently."""
        validate_plotting_params(kind="line", valid_kinds={"line", "bar"})

    def test_invalid_kind(self):
        """Invalid kind raises ValueError."""
        with pytest.raises(ValueError, match="kind must be one of"):
            validate_plotting_params(kind="scatter", valid_kinds={"line", "bar"})

    def test_kind_none_skips(self):
        """kind=None skips validation."""
        validate_plotting_params(kind=None, valid_kinds={"line", "bar"})

    def test_facet_n_cols_valid(self):
        """Valid facet_n_cols passes silently."""
        validate_plotting_params(facet_n_cols=2)

    def test_facet_n_cols_zero(self):
        """Zero facet_n_cols raises ValueError."""
        with pytest.raises(ValueError, match="facet_n_cols must be >= 1"):
            validate_plotting_params(facet_n_cols=0)

    def test_n_bins_valid(self):
        """Valid n_bins passes silently."""
        validate_plotting_params(n_bins=10)

    def test_n_bins_zero(self):
        """Zero n_bins raises ValueError."""
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            validate_plotting_params(n_bins=0)


class TestNormalizeYPred:
    """Tests for _normalize_y_pred utility."""

    def test_dataframe_input(self):
        """DataFrame input is wrapped in dict with default key."""
        df = pl.DataFrame({"time": [1, 2], "y": [10, 20]})
        result = _normalize_y_pred(df)
        assert isinstance(result, dict)
        assert "Forecast" in result
        assert result["Forecast"].equals(df)

    def test_dataframe_custom_name(self):
        """DataFrame input uses custom default_name."""
        df = pl.DataFrame({"time": [1, 2], "y": [10, 20]})
        result = _normalize_y_pred(df, default_name="Model A")
        assert "Model A" in result

    def test_dict_passthrough(self):
        """Dict input is returned as-is."""
        d = {"A": pl.DataFrame({"time": [1], "y": [10]})}
        result = _normalize_y_pred(d)
        assert result is d

    def test_invalid_type(self):
        """Invalid type raises TypeError."""
        with pytest.raises(TypeError, match="y_pred must be pl.DataFrame or dict"):
            _normalize_y_pred([1, 2, 3])


class TestValidatePlottingData:
    """Tests for validate_plotting_data utility."""

    def test_basic_validation(self):
        """Valid DataFrame returns numeric columns."""
        df = pl.DataFrame({"time": [1, 2, 3], "y": [10.0, 20.0, 30.0]})
        result = validate_plotting_data(df, exclude=["time"])
        assert result == ["y"]

    def test_not_polars(self):
        """Non-Polars input raises TypeError."""
        with pytest.raises(TypeError, match="Expected pl.DataFrame"):
            validate_plotting_data({"time": [1, 2, 3]})

    def test_empty_df(self):
        """Empty DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="at least 1 rows"):
            validate_plotting_data(pl.DataFrame())

    def test_no_time_column(self):
        """Missing 'time' column raises ValueError."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        with pytest.raises(ValueError, match="time"):
            validate_plotting_data(df)

    def test_integer_time_column(self):
        """Integer 'time' column is accepted (skip interval check)."""
        df = pl.DataFrame({"time": [1, 2, 3], "y": [10.0, 20.0, 30.0]})
        result = validate_plotting_data(df, exclude=["time"])
        assert result == ["y"]

    def test_columns_string(self):
        """String column spec resolves correctly."""
        df = pl.DataFrame({"time": [1, 2], "y": [10, 20], "z": [5, 15]})
        result = validate_plotting_data(df, columns="y")
        assert result == ["y"]

    def test_columns_list(self):
        """List column spec resolves correctly."""
        df = pl.DataFrame({"time": [1, 2], "y": [10, 20], "z": [5, 15]})
        result = validate_plotting_data(df, columns=["y", "z"])
        assert result == ["y", "z"]

    def test_panel_columns(self):
        """Panel group names resolve panel columns."""
        df = pl.DataFrame({
            "time": [1, 2],
            "sales__a": [10, 20],
            "sales__b": [30, 40],
        })
        result = validate_plotting_data(df, panel_group_names=["sales"])
        assert set(result) == {"sales__a", "sales__b"}

    def test_panel_columns_with_member(self):
        """Panel group + member filter."""
        df = pl.DataFrame({
            "time": [1, 2],
            "sales__a": [10, 20],
            "sales__b": [30, 40],
        })
        result = validate_plotting_data(df, panel_group_names=["sales"], columns="a")
        assert result == ["sales__a"]

    def test_min_rows(self):
        """Custom min_rows is respected."""
        df = pl.DataFrame({"time": [1], "y": [10.0]})
        with pytest.raises(ValueError, match="at least 2 rows"):
            validate_plotting_data(df, min_rows=2)
