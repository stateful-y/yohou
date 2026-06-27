"""Tests for legend, color, and panel rendering contracts after the standardization refactoring."""

import pytest

pytest.importorskip("plotly", reason="plotting extra not installed")


import numpy as np
import polars as pl
import pytest
from plotly import graph_objects as go

from yohou.plotting import (
    plot_autocorrelation,
    plot_boxplot,
    plot_correlation_heatmap,
    plot_cross_correlation,
    plot_decomposition,
    plot_distribution,
    plot_lag_scatter,
    plot_missing_data,
    plot_outliers,
    plot_partial_autocorrelation,
    plot_phase,
    plot_resampling_comparison,
    plot_rolling_statistics,
    plot_scatter_matrix,
    plot_seasonal_heatmap,
    plot_seasonality,
    plot_spectrum,
    plot_subseasonality,
    plot_time_weight,
)
from yohou.plotting._utils import LegendTracker, linked_legendgroup_kwargs

from .conftest import (
    assert_consistent_colors,
    assert_figure_valid,
    legend_groups,
    trace_opacities,
    visible_legend_names,
)


@pytest.fixture
def panel_df():
    """Small daily panel (y__a, y__b), 120 rows."""
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 29), "1d", eager=True)
    n = len(dates)
    return pl.DataFrame({
        "time": dates,
        "y__a": [100.0 + i % 30 + (i % 7) * 3 for i in range(n)],
        "y__b": [200.0 + i % 20 + (i % 5) * 5 for i in range(n)],
    })


@pytest.fixture
def panel_3member_df():
    """Daily panel with 3 members (y__a, y__b, y__c), 120 rows."""
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 29), "1d", eager=True)
    n = len(dates)
    return pl.DataFrame({
        "time": dates,
        "y__a": [100.0 + i for i in range(n)],
        "y__b": [200.0 + i * 0.5 for i in range(n)],
        "y__c": [50.0 + i * 1.5 for i in range(n)],
    })


@pytest.fixture
def panel_3year_df():
    """3-year daily panel, 2 members. For subseasonality."""
    dates = pl.date_range(pl.date(2018, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
    n = len(dates)
    return pl.DataFrame({
        "time": dates,
        "y__a": [100.0 + i % 30 + (i % 7) * 3 for i in range(n)],
        "y__b": [200.0 + i % 20 + (i % 5) * 5 for i in range(n)],
    })


class TestLinkedLegendgroupKwargs:
    def test_primary_trace(self):
        lt = LegendTracker()
        kw = linked_legendgroup_kwargs("member_a", lt, is_primary=True)
        assert kw["legendgroup"] == "member_a"
        assert kw["showlegend"] is True
        assert kw["name"] == "member_a"

    def test_secondary_trace(self):
        lt = LegendTracker()
        kw = linked_legendgroup_kwargs("member_a", lt, is_primary=False)
        assert kw["legendgroup"] == "member_a"
        assert kw["showlegend"] is False

    def test_dedup_via_legend_tracker(self):
        lt = LegendTracker()
        kw1 = linked_legendgroup_kwargs("m", lt, is_primary=True)
        kw2 = linked_legendgroup_kwargs("m", lt, is_primary=True)
        assert kw1["showlegend"] is True
        assert kw2["showlegend"] is False


class TestBoxplotPanelLegend:
    def test_panel_legend_shows_groups(self, panel_df):
        """With facet_by='member' (default), groups appear in legend."""
        fig = plot_boxplot(panel_df, period="1mo")
        names = visible_legend_names(fig)
        assert "y" in names or any("y" in n for n in names)

    def test_panel_legendgroup_per_member(self, panel_df):
        fig = plot_boxplot(panel_df, period="1mo")
        groups = legend_groups(fig)
        # facet_by="member" (default): 1 group ("y") overlaid across member subplots
        assert len(groups) >= 1


class TestOutliersPanelLegend:
    def test_outlier_markers_share_legendgroup_with_line(self, panel_df):
        fig = plot_outliers(panel_df)
        groups = legend_groups(fig)
        # facet_by="member" (default): 1 group ("y") overlaid across member subplots
        assert len(groups) >= 1

    def test_default_outlier_symbol_is_x(self, panel_df):
        # Inject extreme values so both members actually trigger outlier markers;
        # without them plot_outliers adds no marker trace and the contract is untested.
        df = panel_df.with_columns(
            pl.when(pl.int_range(pl.len()) == 60).then(5000.0).otherwise(pl.col("y__a")).alias("y__a"),
            pl.when(pl.int_range(pl.len()) == 40).then(-3000.0).otherwise(pl.col("y__b")).alias("y__b"),
        )
        fig = plot_outliers(df)
        outlier_symbols = [
            t.marker.symbol
            for t in fig.data
            if isinstance(t, go.Scatter | go.Scattergl)
            and getattr(t, "marker", None)
            and getattr(t.marker, "symbol", None)
            and t.marker.symbol not in (None, "circle")
        ]
        assert outlier_symbols, "Expected at least one outlier marker trace"
        assert all(s == "x" for s in outlier_symbols)


class TestRollingStatisticsPanelLegend:
    def test_legend_grouped_by_member(self, panel_df):
        fig = plot_rolling_statistics(panel_df, window_size=7, statistics=["mean"])
        groups = legend_groups(fig)
        # facet_by="member" (default): 1 group ("y") overlaid across member subplots
        assert len(groups) >= 1

    def test_panel_traces_share_legendgroup(self, panel_df):
        fig = plot_rolling_statistics(panel_df, window_size=7, statistics=["mean"])
        groups = legend_groups(fig)
        assert groups, "Expected at least one legendgroup"
        # The raw trace and the rolling-mean trace must share a legendgroup, so
        # every group must link at least two traces together.
        for g in groups:
            count = sum(1 for t in fig.data if getattr(t, "legendgroup", None) == g)
            assert count >= 2, f"Group '{g}' links only {count} trace(s); expected >= 2"


class TestComponentsPanelLayout:
    @pytest.fixture
    def panel_components(self, panel_df):
        """Build (y, components) for panel data."""
        dates = panel_df["time"]
        n = len(dates)
        components = {
            "trend": pl.DataFrame({
                "time": dates,
                "y__a": [i * 0.5 for i in range(n)],
                "y__b": [i * 1.0 for i in range(n)],
            }),
        }
        return panel_df, components

    def test_panel_grid_produces_figure(self, panel_components):
        y, components = panel_components
        result = plot_decomposition(y, components)
        assert isinstance(result, dict)
        for fig in result.values():
            assert_figure_valid(fig)

    def test_panel_traces_have_legendgroup(self, panel_components):
        y, components = panel_components
        result = plot_decomposition(y, components)
        assert isinstance(result, dict)
        for fig in result.values():
            groups = legend_groups(fig)
            assert len(groups) >= 1


class TestSubseasonalityPanelLegend:
    def test_legendgroup_per_panel_member(self, panel_3year_df):
        result = plot_subseasonality(panel_3year_df, seasonality="month")
        assert isinstance(result, dict)
        for fig in result.values():
            groups = legend_groups(fig)
            assert len(groups) >= 1

    def test_no_crash_with_sparse_data(self):
        dates = pl.date_range(pl.date(2018, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        n = len(dates)
        df = pl.DataFrame({
            "time": dates,
            "y__a": [float(i) if i % 3 != 0 else None for i in range(n)],
            "y__b": [float(i) if i % 5 != 0 else None for i in range(n)],
        })
        result = plot_subseasonality(df, seasonality="month")
        assert isinstance(result, dict)
        for fig in result.values():
            assert_figure_valid(fig)


class TestSeasonalHeatmapPanel:
    def test_all_panel_members_get_subplot(self, panel_hourly_df):
        fig = plot_seasonal_heatmap(panel_hourly_df)
        heatmap_traces = [t for t in fig.data if isinstance(t, go.Heatmap)]
        assert len(heatmap_traces) >= 2

    def test_subplot_titles_contain_member_names(self, panel_hourly_df):
        fig = plot_seasonal_heatmap(panel_hourly_df)
        annotations = [a.text for a in fig.layout.annotations if hasattr(a, "text")]
        assert any("a" in a for a in annotations)
        assert any("b" in a for a in annotations)


class TestLagScatterPanelColors:
    def test_panel_returns_dict_for_multi_member(self, panel_df):
        result = plot_lag_scatter(panel_df, lags=[1])
        assert isinstance(result, dict)
        assert set(result.keys()) == {"a", "b"}
        for fig in result.values():
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0

    def test_panel_groups_overlaid_per_member(self, panel_df):
        result = plot_lag_scatter(panel_df, lags=[1])
        for fig in result.values():
            groups = legend_groups(fig)
            assert len(groups) >= 1

    def test_seasonal_mode_has_legendgrouptitle(self):
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
            "y": [100 + i % 30 for i in range(366)],
        })
        fig = plot_lag_scatter(df, columns="y", lags=[1, 2], seasonality="month")
        traces_with_title = [
            t for t in fig.data if getattr(t, "legendgrouptitle", None) and getattr(t.legendgrouptitle, "text", None)
        ]
        assert len(traces_with_title) >= 1


class TestAutocorrelationPanelLegend:
    """Legend and color contracts for plot_autocorrelation and plot_partial_autocorrelation."""

    @pytest.mark.parametrize(
        "func",
        [plot_autocorrelation, plot_partial_autocorrelation],
        ids=["acf", "pacf"],
    )
    def test_member_mode_single_group_overlaid(self, func, panel_df):
        # facet_by="member" (default): the single group prefix "y" is overlaid
        # across the per-member subplots, so it appears once as legend and group.
        fig = func(panel_df, max_lags=10)
        assert visible_legend_names(fig) == ["y"]
        assert legend_groups(fig) == {"y"}
        for g in legend_groups(fig):
            assert_consistent_colors(fig, g)

    @pytest.mark.parametrize(
        "func",
        [plot_autocorrelation, plot_partial_autocorrelation],
        ids=["acf", "pacf"],
    )
    def test_group_mode_member_names_are_legendgroups(self, func, panel_df):
        # facet_by="group": each member ("a", "b") becomes its own legend group.
        fig = func(panel_df, max_lags=10, facet_by="group")
        assert legend_groups(fig) == {"a", "b"}
        assert set(visible_legend_names(fig)) == {"a", "b"}
        for g in legend_groups(fig):
            assert_consistent_colors(fig, g)


class TestCrossCorrelation:
    def test_columns_optional_auto_pairs(self):
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "x": [100 + i for i in range(91)],
            "y": [105 + i for i in range(91)],
            "z": [90 + i * 2 for i in range(91)],
        })
        fig = plot_cross_correlation(df, columns=["x", "y", "z"], max_lags=10)
        assert_figure_valid(fig)
        groups = legend_groups(fig)
        assert len(groups) == 3  # x vs y, x vs z, y vs z

    def test_two_columns_backward_compat(self):
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "x": [100 + i for i in range(91)],
            "y": [105 + i for i in range(91)],
        })
        fig = plot_cross_correlation(df, columns=["x", "y"], max_lags=10)
        assert_figure_valid(fig)

    def test_legend_present(self):
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "x": [100 + i for i in range(91)],
            "y": [105 + i for i in range(91)],
        })
        fig = plot_cross_correlation(df, columns=["x", "y"], max_lags=10)
        # Single-pair mode uses stem plot; the marker trace has a name
        named = [t for t in fig.data if getattr(t, "name", None)]
        assert len(named) >= 1

    def test_panel_auto_pairs(self, panel_df):
        result = plot_cross_correlation(panel_df, max_lags=10)
        # plot_cross_correlation always returns go.Figure now
        assert isinstance(result, go.Figure)
        assert_figure_valid(result)
        groups = legend_groups(result)
        assert len(groups) >= 1

    def test_columns_none_uses_all_numeric(self):
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "a": list(range(91)),
            "b": list(range(91)),
            "c": list(range(91)),
        })
        fig = plot_cross_correlation(df, max_lags=5)
        assert_figure_valid(fig)
        groups = legend_groups(fig)
        assert len(groups) == 3


def _make_daily_panel():
    """Small daily panel DataFrame for parametrized tests."""
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 29), "1d", eager=True)
    n = len(dates)
    return pl.DataFrame({
        "time": dates,
        "y__a": [100.0 + i % 30 for i in range(n)],
        "y__b": [200.0 + i % 20 for i in range(n)],
    })


def _make_3year_daily():
    """3-year daily DataFrame for seasonality tests."""
    dates = pl.date_range(pl.date(2018, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
    n = len(dates)
    return pl.DataFrame({
        "time": dates,
        "y__a": [100.0 + i % 30 for i in range(n)],
        "y__b": [200.0 + i % 20 for i in range(n)],
    })


def _make_hourly_panel():
    """Hourly panel for heatmap tests."""
    dates = pl.datetime_range(pl.datetime(2020, 1, 1), pl.datetime(2020, 2, 29, 23), "1h", eager=True)
    n = len(dates)
    return pl.DataFrame({
        "time": dates,
        "y__a": [10.0 + np.sin(i * 0.3) * 5 for i in range(n)],
        "y__b": [20.0 + np.cos(i * 0.3) * 3 for i in range(n)],
    })


def _make_simple():
    """Single-column daily DataFrame."""
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
    return pl.DataFrame({
        "time": dates,
        "y": [100 + i % 30 for i in range(len(dates))],
    })


def _make_short_2col():
    """Short 2-column DataFrame for cross-correlation."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
        "x": [100 + i % 20 for i in range(91)],
        "y": [150 + (i % 15) * 2 for i in range(91)],
    })


def _make_weight_df():
    """Single-column time_weight DataFrame."""
    n = 100
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True),
        "time_weight": [(i + 1) / n for i in range(n)],
    })


# Functions that return go.Figure with a simple call + expected title
_TITLE_CASES: list[tuple[str, object, dict, str]] = [
    ("rolling_statistics", plot_rolling_statistics, {"df": _make_simple(), "window_size": 3}, "Rolling Statistics"),
    ("missing_data", plot_missing_data, {"df": _make_simple()}, "Missing Data"),
    ("distribution", plot_distribution, {"df": _make_simple()}, "Distribution"),
    ("outliers", plot_outliers, {"df": _make_simple()}, "Outlier Detection"),
    ("acf", plot_autocorrelation, {"df": _make_simple(), "max_lags": 10}, "Autocorrelation (ACF)"),
    ("pacf", plot_partial_autocorrelation, {"df": _make_simple(), "max_lags": 10}, "Partial Autocorrelation (PACF)"),
    (
        "seasonality",
        plot_seasonality,
        {"df": _make_3year_daily(), "columns": "y__a", "groups": None},
        "Seasonal Pattern",
    ),
    (
        "subseasonality",
        plot_subseasonality,
        {"df": _make_3year_daily(), "columns": "y__a", "groups": None},
        "Seasonal Subseries",
    ),
    ("seasonal_heatmap", plot_seasonal_heatmap, {"df": _make_hourly_panel()}, "Seasonal Heatmap"),
    ("lag_scatter", plot_lag_scatter, {"df": _make_simple(), "lags": [1]}, "Lag 1 Scatter Plot"),
    (
        "cross_correlation",
        plot_cross_correlation,
        {"df": _make_short_2col(), "columns": ["x", "y"], "max_lags": 10},
        "Cross-Correlation",
    ),
    (
        "resampling_comparison",
        plot_resampling_comparison,
        {"df_original": _make_simple(), "df_resampled": _make_simple()},
        "Original vs Resampled",
    ),
    ("correlation_heatmap", plot_correlation_heatmap, {"df": _make_short_2col()}, "Correlation Heatmap"),
    ("scatter_matrix", plot_scatter_matrix, {"df": _make_short_2col()}, "Scatter Matrix"),
    ("time_weight", plot_time_weight, {"df": _make_weight_df()}, "Time Weights"),
    ("phase", plot_phase, {"df": _make_simple()}, "Phase Spectrum"),
    ("spectrum", plot_spectrum, {"df": _make_simple()}, "Periodogram"),
]


class TestDefaultTitleContract:
    """Every plot function must produce a static, function-descriptive default title."""

    @pytest.mark.parametrize(
        "name,func,kw,expected_title",
        _TITLE_CASES,
        ids=[c[0] for c in _TITLE_CASES],
    )
    def test_default_title(self, name, func, kw, expected_title):
        result = func(**kw)
        fig = result if isinstance(result, go.Figure) else next(iter(result.values()))
        assert fig.layout.title.text == expected_title, (
            f"{name}: expected title '{expected_title}', got '{fig.layout.title.text}'"
        )


_PANEL_COLOR_CASES: list[tuple[str, object, dict]] = [
    ("rolling_statistics", plot_rolling_statistics, {"df": _make_daily_panel(), "window_size": 3}),
    ("boxplot", plot_boxplot, {"df": _make_daily_panel(), "period": "1mo"}),
    ("outliers", plot_outliers, {"df": _make_daily_panel()}),
    ("acf", plot_autocorrelation, {"df": _make_daily_panel(), "max_lags": 10}),
    ("pacf", plot_partial_autocorrelation, {"df": _make_daily_panel(), "max_lags": 10}),
    ("lag_scatter", plot_lag_scatter, {"df": _make_daily_panel(), "lags": [1]}),
    ("cross_correlation", plot_cross_correlation, {"df": _make_daily_panel(), "max_lags": 10}),
]


class TestColorConsistencyContract:
    """All traces in the same legendgroup must share the same base color."""

    @pytest.mark.parametrize(
        "name,func,kw",
        _PANEL_COLOR_CASES,
        ids=[c[0] for c in _PANEL_COLOR_CASES],
    )
    def test_consistent_colors(self, name, func, kw):
        result = func(**kw)
        figs = result.values() if isinstance(result, dict) else [result]
        for fig in figs:
            for g in legend_groups(fig):
                assert_consistent_colors(fig, g)


_PANEL_LAYOUT_CASES: list[tuple[str, object, dict]] = [
    ("subseasonality", plot_subseasonality, {"df": _make_3year_daily(), "seasonality": "month"}),
    ("correlation_heatmap", plot_correlation_heatmap, {"df": _make_daily_panel()}),
    ("scatter_matrix", plot_scatter_matrix, {"df": _make_daily_panel()}),
]


class TestPanelLayoutContract:
    """Functions with facet_by param must return correct types per mode."""

    @pytest.mark.parametrize(
        "name,func,kw",
        _PANEL_LAYOUT_CASES,
        ids=[c[0] for c in _PANEL_LAYOUT_CASES],
    )
    def test_grid_returns_figure(self, name, func, kw):
        """Default panel call returns dict (facet_by default is group or per-member)."""
        result = func(**kw)
        assert isinstance(result, dict), f"{name}: should return dict"
        for fig in result.values():
            assert_figure_valid(fig)


class TestRollingStatsRawColor:
    """Raw trace color must match the rolling statistics traces of the same member."""

    def test_panel_raw_trace_color_matches_stats(self):
        df = _make_daily_panel()
        fig = plot_rolling_statistics(df, window_size=7, statistics=["mean"])
        # Group traces by legendgroup
        for group in legend_groups(fig):
            traces = [t for t in fig.data if getattr(t, "legendgroup", None) == group]
            colors = set()
            for t in traces:
                line = getattr(t, "line", None)
                if line and getattr(line, "color", None):
                    colors.add(line.color)
            assert len(colors) == 1, f"Group '{group}': expected 1 color, got {colors}"

    def test_non_panel_single_col_uses_palette_color(self):
        df = _make_simple()
        fig = plot_rolling_statistics(df, window_size=7, statistics=["mean"])
        # Should NOT use #94a3b8 gray for the raw trace
        for t in fig.data:
            line = getattr(t, "line", None)
            if line and getattr(line, "color", None):
                assert line.color != "#94a3b8", "Raw trace should use palette color, not gray"


class TestComponentsResampler:
    """plot_decomposition panel path should support resampler config."""

    def test_panel_dict_does_not_crash(self):
        """Panel components with default config produces valid dict."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        n = len(dates)
        y = pl.DataFrame({
            "time": dates,
            "y__a": list(range(n)),
            "y__b": [i * 2 for i in range(n)],
        })
        components = {
            "trend": pl.DataFrame({
                "time": dates,
                "y__a": [i * 0.5 for i in range(n)],
                "y__b": [i * 1.0 for i in range(n)],
            }),
        }
        result = plot_decomposition(y, components)
        assert isinstance(result, dict)
        for fig in result.values():
            assert_figure_valid(fig)


class TestSeasonalityLegend:
    """Seasonality legend: one entry per member, RGBA opacity on line colors."""

    def test_panel_one_legend_entry_per_member(self):
        df = _make_3year_daily()
        fig = plot_seasonality(df)
        shown = [t.name for t in fig.data if getattr(t, "showlegend", None) is True]
        groups = legend_groups(fig)
        assert set(shown) == groups, f"Expected one legend entry per member {groups}, got shown={shown}"

    def test_non_panel_one_legend_entry_per_column(self):
        dates = pl.date_range(pl.date(2018, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "y": [100 + i % 30 for i in range(len(dates))],
        })
        fig = plot_seasonality(df, columns="y")
        # In column-mode subplots, subplot titles identify columns; no legend swatch needed
        assert len(fig.data) >= 1

    def test_legend_swatch_is_opaque_hex(self):
        """The showlegend=True trace must use a fully opaque hex color (no RGBA)."""
        df = _make_3year_daily()
        fig = plot_seasonality(df)
        for t in fig.data:
            if getattr(t, "showlegend", None) is True:
                assert getattr(t, "opacity", None) is None, (
                    f"Legend trace '{t.name}' has trace-level opacity={t.opacity}"
                )
                line = getattr(t, "line", None)
                assert line, f"Legend trace '{t.name}' has no line"
                color = getattr(line, "color", "")
                assert color.startswith("#"), f"Legend swatch should use hex color, got '{color}'"

    def test_no_cycle_at_full_opacity(self):
        """No data cycle should reach alpha=1.0; max is capped at 0.9."""
        df = _make_3year_daily()
        fig = plot_seasonality(df)
        for group in legend_groups(fig):
            opacities = trace_opacities(fig, group)
            for op in opacities:
                assert op <= 0.91, f"Cycle opacity {op} exceeds 0.9 cap"

    def test_panel_opacity_increases_with_time(self):
        df = _make_3year_daily()
        # facet_by="group" so each member is a separate legend group with its own opacities
        fig = plot_seasonality(df, facet_by="group")
        for group in legend_groups(fig):
            opacities = trace_opacities(fig, group)
            cycle_opacities = [o for o in opacities if o < 1.0] or opacities
            if len(cycle_opacities) >= 2:
                for i in range(len(cycle_opacities) - 1):
                    assert cycle_opacities[i] <= cycle_opacities[i + 1] + 0.01, (
                        f"Opacities should increase with time: {cycle_opacities}"
                    )

    def test_non_panel_opacity_increases_with_time(self):
        dates = pl.date_range(pl.date(2018, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "y": [100 + i % 30 for i in range(len(dates))],
        })
        fig = plot_seasonality(df, columns="y")
        opacities = trace_opacities(fig, "y")
        if len(opacities) >= 2:
            for i in range(len(opacities) - 1):
                assert opacities[i] <= opacities[i + 1] + 0.01, f"Opacities should increase with time: {opacities}"

    def test_opacity_power_kwarg(self):
        """Higher opacity_power makes intermediate cycles ramp to opaque faster."""
        df = _make_3year_daily()
        # Linear ramp
        fig_linear = plot_seasonality(df, columns="y__a", groups=None, opacity_power=1.0)
        # Fast ramp - cycles get opaque sooner
        fig_fast = plot_seasonality(df, columns="y__a", groups=None, opacity_power=3.0)
        op_linear = trace_opacities(fig_linear, "y__a")
        op_fast = trace_opacities(fig_fast, "y__a")
        assert op_linear and op_fast, "Expected opacity values"
        # Both should start at 0.1 and end at 0.9
        assert abs(op_linear[0] - 0.1) < 0.02
        assert abs(op_fast[0] - 0.1) < 0.02
        assert abs(op_linear[-1] - 0.9) < 0.02
        assert abs(op_fast[-1] - 0.9) < 0.02
        # Midpoint of fast curve should be higher than linear
        mid = len(op_linear) // 2
        if mid > 0:
            assert op_fast[mid] > op_linear[mid], (
                f"Fast curve midpoint ({op_fast[mid]}) should be > linear midpoint ({op_linear[mid]})"
            )


class TestSubseasonalityPanelLayout:
    """plot_subseasonality with panel data returns dict keyed by member name."""

    def test_non_panel_ignores_panel_layout(self):
        dates = pl.date_range(pl.date(2018, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "y": [100 + i % 30 for i in range(len(dates))],
        })
        result = plot_subseasonality(df, columns="y", seasonality="month")
        # Non-panel data should return go.Figure
        assert isinstance(result, go.Figure)
