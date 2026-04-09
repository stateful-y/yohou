"""Unit tests for plotting utilities."""

import pytest

pytest.importorskip("plotly", reason="plotting extra not installed")


import polars as pl
import pytest

from yohou.plotting._utils import (
    LegendTracker,
    _group_panel_columns,
    _normalize_y_pred,
    get_color_sequence,
    resolve_color_palette,
    resolve_panel_columns,
)
from yohou.utils import validate_plotting_data, validate_plotting_params


class TestGetColorSequence:
    """Tests for get_color_sequence utility."""

    def test_no_args_returns_full_palette(self):
        """Calling with no arguments returns complete default palette."""
        colors = get_color_sequence()
        assert isinstance(colors, list)
        assert len(colors) == 12
        assert all(isinstance(c, str) for c in colors)

    def test_specific_count(self):
        """Requesting a specific count returns that many colors."""
        colors = get_color_sequence(5)
        assert len(colors) == 5

    def test_exceeding_palette_cycles(self):
        """Requesting more colors than palette size cycles through."""
        colors = get_color_sequence(25)
        assert len(colors) == 25
        full_palette = get_color_sequence()
        assert colors[0] == full_palette[0]
        assert colors[12] == full_palette[0]


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


class TestResolvePanelColumns:
    """Tests for resolve_panel_columns utility."""

    @pytest.fixture
    def panel_df(self):
        """Panel DataFrame with two groups."""
        return pl.DataFrame({
            "time": [1, 2],
            "sales__a": [10, 20],
            "sales__b": [30, 40],
            "traffic__a": [50, 60],
        })

    def test_string_columns_coerced_to_list(self, panel_df):
        """String columns parameter converted to list internally."""
        result = resolve_panel_columns(panel_df, columns="a")
        assert "sales__a" in result
        assert "traffic__a" in result

    def test_panel_group_with_member_filter(self, panel_df):
        """Panel group and member filter together resolve correctly."""
        result = resolve_panel_columns(panel_df, panel_group_names=["sales"], columns=["b"])
        assert result == ["sales__b"]

    def test_nonexistent_group_raises(self, panel_df):
        """Nonexistent panel group name raises ValueError."""
        with pytest.raises(ValueError, match="No panel columns found for groups"):
            resolve_panel_columns(panel_df, panel_group_names=["nonexistent"])

    def test_nonexistent_member_raises(self, panel_df):
        """Nonexistent member with valid group raises ValueError."""
        with pytest.raises(ValueError, match="No panel columns found for groups"):
            resolve_panel_columns(panel_df, panel_group_names=["sales"], columns=["z"])


class TestGroupPanelColumns:
    """Tests for _group_panel_columns utility."""

    def test_groups_and_members(self):
        """Groups panel columns by prefix and extracts member names."""
        groups, members = _group_panel_columns(["T3__a", "T3__b", "T4__a"])
        assert groups == {"T3": ["T3__a", "T3__b"], "T4": ["T4__a"]}
        assert members == ["a", "b"]

    def test_plain_column_fallback(self):
        """Column without __ uses full name as both key and member."""
        groups, members = _group_panel_columns(["plain_col"])
        assert groups == {"plain_col": ["plain_col"]}
        assert members == ["plain_col"]

    def test_mixed_columns(self):
        """Mix of panel and plain columns groups correctly."""
        groups, members = _group_panel_columns(["T__x", "plain"])
        assert groups == {"T": ["T__x"], "plain": ["plain"]}
        assert set(members) == {"x", "plain"}


class TestLegendTracker:
    """Tests for LegendTracker utility."""

    def test_first_call_returns_true(self):
        """First call for a name returns True."""
        tracker = LegendTracker()
        assert tracker.should_show("sales") is True

    def test_second_call_returns_false(self):
        """Second call for the same name returns False."""
        tracker = LegendTracker()
        tracker.should_show("sales")
        assert tracker.should_show("sales") is False

    def test_independent_names(self):
        """Different names are tracked independently."""
        tracker = LegendTracker()
        assert tracker.should_show("sales") is True
        assert tracker.should_show("demand") is True
        assert tracker.should_show("sales") is False
        assert tracker.should_show("demand") is False

    def test_show_legend_false_always_returns_false(self):
        """When show_legend=False, should_show always returns False."""
        tracker = LegendTracker(show_legend=False)
        assert tracker.should_show("sales") is False
        assert tracker.should_show("demand") is False

    def test_show_legend_true_default(self):
        """Default show_legend=True allows first-seen tracking."""
        tracker = LegendTracker(show_legend=True)
        assert tracker.should_show("a") is True
        assert tracker.should_show("a") is False


class TestResolveColorPaletteEdgeCases:
    """Edge cases for resolve_color_palette (lines 493-494)."""

    def test_empty_list_raises(self):
        """Passing an empty list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            resolve_color_palette([], 3)


class TestResamplerImportErrors:
    """Tests for plotly-resampler ImportError fallback paths."""

    def _block_resampler(self):
        """Return a mock import that blocks plotly_resampler sub-imports."""
        import builtins
        from unittest.mock import patch

        orig = builtins.__import__

        def _mock(name, *a, **kw):
            if "plotly_resampler" in name:
                raise ImportError("plotly_resampler is not installed")
            return orig(name, *a, **kw)

        return patch("builtins.__import__", side_effect=_mock)

    def test_create_figure_widget_no_resampler_raises(self):
        """_create_figure with resampler='widget' raises when plotly-resampler missing."""
        from yohou.plotting._utils import _create_figure

        with self._block_resampler(), pytest.raises(ImportError, match="plotly-resampler"):
            _create_figure(resampler="widget")

    def test_create_figure_bool_no_resampler_raises(self):
        """_create_figure with resampler=True raises when plotly-resampler missing."""
        from yohou.plotting._utils import _create_figure

        with self._block_resampler(), pytest.raises(ImportError, match="plotly-resampler"):
            _create_figure(resampler=True)

    def test_create_subplots_widget_no_resampler_raises(self):
        """_create_subplots with resampler='widget' raises when plotly-resampler missing."""
        from yohou.plotting._utils import _create_subplots

        with self._block_resampler(), pytest.raises(ImportError, match="plotly-resampler"):
            _create_subplots(resampler="widget", rows=1, cols=1)

    def test_create_subplots_bool_no_resampler_raises(self):
        """_create_subplots with resampler=True raises when plotly-resampler missing."""
        from yohou.plotting._utils import _create_subplots

        with self._block_resampler(), pytest.raises(ImportError, match="plotly-resampler"):
            _create_subplots(resampler=True, rows=1, cols=1)

    def test_fill_trace_kwargs_no_resampler_returns_empty(self):
        """_fill_trace_kwargs returns empty dict when plotly-resampler missing."""
        import plotly.graph_objects as go

        from yohou.plotting._utils import _fill_trace_kwargs

        with self._block_resampler():
            result = _fill_trace_kwargs(go.Figure())
        assert result == {}


class TestResamplerSuccessPaths:
    """Tests for the resampler success paths when plotly-resampler is installed."""

    _has_resampler = pytest.importorskip("plotly_resampler", reason="plotly-resampler not installed") is not None

    def test_create_figure_resampler(self):
        """_create_figure(resampler=True) returns a FigureResampler."""
        from plotly_resampler import FigureResampler

        from yohou.plotting._utils import _create_figure

        fig = _create_figure(resampler=True)
        assert isinstance(fig, FigureResampler)

    def test_create_subplots_resampler(self):
        """_create_subplots(resampler=True) returns a FigureResampler."""
        from plotly_resampler import FigureResampler

        from yohou.plotting._utils import _create_subplots

        fig = _create_subplots(resampler=True, rows=1, cols=1)
        assert isinstance(fig, FigureResampler)
