"""Dedicated panel-data tests for the plotting module.

Tests deeper panel scenarios not covered by per-module tests:
- Multi-group filtering via ``groups``
- Three-group panels
- DataFrames with null values
- Subplot count validation
"""

import pytest

pytest.importorskip("plotly", reason="plotting extra not installed")


import plotly.graph_objects as go
import polars as pl
import pytest

from yohou.plotting import (
    plot_autocorrelation,
    plot_boxplot,
    plot_forecast,
    plot_lag_scatter,
    plot_missing_data,
    plot_partial_autocorrelation,
    plot_phase,
    plot_rolling_statistics,
    plot_seasonality,
    plot_spectrum,
    plot_subseasonality,
    plot_time_series,
    plot_time_weight,
)


class TestPanelTimeSeries:
    """Panel tests for plot_time_series."""

    def test_two_groups(self, panel_daily_df):
        """Two-group panel produces figure with data for both groups."""
        fig = plot_time_series(panel_daily_df, groups=["y"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_three_groups(self, panel_monthly_3groups):
        """Three-group panel produces traces for all groups."""
        fig = plot_time_series(panel_monthly_3groups, groups=["sales"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3

    def test_columns_filter_member(self, panel_monthly_3groups):
        """Columns filter selects specific members within a panel group."""
        fig = plot_time_series(
            panel_monthly_3groups,
            groups=["sales"],
            columns=["a"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_columns_filter_multiple_members(self, panel_monthly_3groups):
        """Columns filter with multiple members."""
        fig = plot_time_series(
            panel_monthly_3groups,
            groups=["sales"],
            columns=["a", "c"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2


class TestPanelRollingStatistics:
    """Panel tests for plot_rolling_statistics."""

    def test_three_groups(self, panel_monthly_3groups):
        """Three groups with rolling mean should produce faceted figure."""
        fig = plot_rolling_statistics(
            panel_monthly_3groups,
            window_size=3,
            statistics="mean",
            groups=["sales"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3

    def test_filter_by_group_name(self, panel_two_groups_df):
        """Filtering to one group prefix should only show that group's panels."""
        fig_temp = plot_rolling_statistics(
            panel_two_groups_df,
            window_size=3,
            statistics="mean",
            groups=["temp"],
        )
        fig_wind = plot_rolling_statistics(
            panel_two_groups_df,
            window_size=3,
            statistics="mean",
            groups=["wind"],
        )
        assert isinstance(fig_temp, go.Figure)
        assert isinstance(fig_wind, go.Figure)
        # Both should produce figures; data counts may differ but both valid
        assert len(fig_temp.data) > 0
        assert len(fig_wind.data) > 0

    def test_columns_filter_member(self, panel_two_groups_df):
        """Columns filter selects specific member within a group."""
        fig = plot_rolling_statistics(
            panel_two_groups_df,
            window_size=3,
            statistics="mean",
            groups=["temp"],
            columns=["city_a"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestPanelBoxplot:
    """Panel tests for plot_boxplot."""

    def test_two_groups(self, panel_daily_df):
        """Two-group boxplot panel."""
        fig = plot_boxplot(
            panel_daily_df,
            period="1mo",
            groups=["y"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_three_groups(self, panel_monthly_3groups):
        """Three-group boxplot panel."""
        fig = plot_boxplot(
            panel_monthly_3groups,
            period="1mo",
            groups=["sales"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestPanelMissingData:
    """Panel tests for plot_missing_data."""

    def test_with_nulls(self, panel_with_nulls):
        """Panel with actual missing values."""
        fig = plot_missing_data(
            panel_with_nulls,
            kind="heatmap",
            groups=["y"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_bars_kind(self, panel_with_nulls):
        """Panel missing data with bars kind."""
        fig = plot_missing_data(
            panel_with_nulls,
            kind="bars",
            groups=["y"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_matrix_kind(self, panel_with_nulls):
        """Panel missing data with matrix kind."""
        fig = plot_missing_data(
            panel_with_nulls,
            kind="matrix",
            groups=["y"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestPanelACF:
    """Panel tests for plot_autocorrelation."""

    def test_three_groups(self, panel_monthly_3groups):
        """Three-group ACF panel."""
        fig = plot_autocorrelation(
            panel_monthly_3groups,
            max_lags=5,
            groups=["sales"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_filter_groups(self, panel_two_groups_df):
        """ACF filtered to single group prefix."""
        fig = plot_autocorrelation(
            panel_two_groups_df,
            max_lags=10,
            groups=["temp"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_columns_filter_member(self, panel_monthly_3groups):
        """Columns filter selects specific member within ACF panel."""
        fig_all = plot_autocorrelation(
            panel_monthly_3groups,
            max_lags=5,
            groups=["sales"],
        )
        fig_one = plot_autocorrelation(
            panel_monthly_3groups,
            max_lags=5,
            groups=["sales"],
            columns=["a"],
        )
        # Fewer traces when filtering to one member
        assert len(fig_one.data) < len(fig_all.data)


class TestPanelPACF:
    """Panel tests for plot_partial_autocorrelation."""

    def test_two_groups(self, panel_daily_df):
        """Two-group PACF panel."""
        fig = plot_partial_autocorrelation(
            panel_daily_df,
            max_lags=10,
            groups=["y"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestPanelSeasonality:
    """Panel tests for plot_seasonality."""

    def test_three_groups(self, panel_monthly_3groups):
        """Three-group seasonality panel."""
        fig = plot_seasonality(
            panel_monthly_3groups,
            seasonality="month",
            groups=["sales"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_filter_groups(self, panel_two_groups_df):
        """Seasonality filtered to one group prefix."""
        fig = plot_seasonality(
            panel_two_groups_df,
            seasonality="month",
            groups=["wind"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestPanelLagScatter:
    """Panel tests for plot_lag_scatter."""

    def test_multi_lag(self, panel_daily_df):
        """Two-group lag scatter with multiple lags."""
        result = plot_lag_scatter(
            panel_daily_df,
            lags=[1, 7],
            groups=["y"],
        )
        assert isinstance(result, dict)
        for fig in result.values():
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0

    def test_three_groups(self, panel_monthly_3groups):
        """Three-group lag scatter."""
        result = plot_lag_scatter(
            panel_monthly_3groups,
            lags=1,
            groups=["sales"],
        )
        assert isinstance(result, dict)
        assert len(result) == 3
        for fig in result.values():
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0


class TestPanelSpectrum:
    """Panel tests for plot_spectrum."""

    def test_two_groups(self, panel_daily_df):
        """Two-group spectrum panel."""
        fig = plot_spectrum(
            panel_daily_df,
            groups=["y"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_log_scale(self, panel_daily_df):
        """Spectrum with log scale on panel data."""
        fig = plot_spectrum(
            panel_daily_df,
            log_scale=True,
            groups=["y"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_columns_filter_member(self, panel_daily_df):
        """Columns filter selects specific member within spectrum panel."""
        fig = plot_spectrum(
            panel_daily_df,
            groups=["y"],
            columns=["a"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestPanelForecast:
    """Panel tests for plot_forecast."""

    def test_panel_forecast(self, panel_monthly_df):
        """Panel forecast with y_test and y_pred both having panel columns."""
        y_test = panel_monthly_df.head(6)
        y_pred = panel_monthly_df.tail(6)
        fig = plot_forecast(y_test, y_pred, groups=["y"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_panel_with_train(self, panel_monthly_df):
        """Panel forecast including training data."""
        y_train = panel_monthly_df.head(6)
        y_test = panel_monthly_df.tail(6)
        y_pred = panel_monthly_df.tail(6)
        fig = plot_forecast(y_test, y_pred, y_train=y_train, groups=["y"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_multivariate_panel_per_subplot_legend(self, panel_two_groups_df):
        """Multivariate panel assigns per-member colours and deduplicates legend."""
        y_test = panel_two_groups_df.tail(30)
        y_pred = panel_two_groups_df.tail(30)
        y_train = panel_two_groups_df.head(61)
        fig = plot_forecast(y_test, y_pred, y_train=y_train)
        assert isinstance(fig, go.Figure)
        # Each (legendgroup, name) pair should appear at most once with showlegend.
        shown_pairs = [(tr.legendgroup, tr.name) for tr in fig.data if tr.showlegend]
        assert len(shown_pairs) == len(set(shown_pairs)), f"Duplicate legend entries: {shown_pairs}"

    def test_multivariate_panel_member_names_in_legend(self, panel_two_groups_df):
        """Group names appear in legend groups for multivariate panels (facet_by=member)."""
        y_test = panel_two_groups_df.tail(30)
        y_pred = panel_two_groups_df.tail(30)
        fig = plot_forecast(y_test, y_pred)
        # With facet_by="member", groups (temp, wind) appear as legend groups
        legend_groups = {tr.legendgroup for tr in fig.data if tr.legendgroup}
        assert "temp" in legend_groups
        assert "wind" in legend_groups


class TestPanelTimeWeight:
    """Panel tests for plot_time_weight."""

    def test_auto_detect_panels(self):
        """Panel time weights auto-detected from weight__group columns."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        n = len(dates)
        df = pl.DataFrame({
            "time": dates,
            "weight__store_1": [1.0 / (n - i) for i in range(n)],
            "weight__store_2": [float(i + 1) / n for i in range(n)],
        })
        fig = plot_time_weight(df, weight_column="weight")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2


# Cross-module: invalid groups


class TestPanelInvalidGroups:
    """Verify ValueError for non-existent groups."""

    def test_rolling_statistics_invalid_group(self, panel_daily_df):
        """Rolling statistics with non-existent group raises ValueError."""
        with pytest.raises(ValueError, match="No panel columns found"):
            plot_rolling_statistics(
                panel_daily_df,
                window_size=3,
                statistics="mean",
                groups=["nonexistent"],
            )

    def test_acf_invalid_group(self, panel_daily_df):
        """ACF with non-existent group raises ValueError."""
        with pytest.raises(ValueError, match="No panel columns found"):
            plot_autocorrelation(
                panel_daily_df,
                max_lags=5,
                groups=["nonexistent"],
            )

    def test_invalid_member(self, panel_daily_df):
        """Non-existent member postfix raises ValueError."""
        with pytest.raises(ValueError, match="No panel columns found"):
            plot_time_series(
                panel_daily_df,
                groups=["y"],
                columns=["nonexistent"],
            )


class TestPanelSubseasonality:
    """Panel tests for plot_subseasonality."""

    def test_two_groups(self, panel_daily_df):
        """Two-member panel subseasonality returns a dict keyed by member name."""
        result = plot_subseasonality(panel_daily_df, seasonality="month")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"a", "b"}
        for fig in result.values():
            assert len(fig.data) > 0

    def test_three_groups(self, panel_monthly_3groups):
        """Three-member panel subseasonality returns dict of figures."""
        result = plot_subseasonality(panel_monthly_3groups, seasonality="month")
        assert isinstance(result, dict)
        assert len(result) == 3
        for fig in result.values():
            assert len(fig.data) > 0

    def test_mean_line_color_matches_series(self):
        """Mean line color matches its series color, not a single hardcoded color."""
        dates = pl.date_range(pl.date(2018, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        n = len(dates)
        df = pl.DataFrame({
            "time": dates,
            "g1__a": [100 + i % 30 for i in range(n)],
            "g2__a": [200 + i % 20 for i in range(n)],
        })
        # Single member "a" with 2 groups → returns go.Figure
        fig = plot_subseasonality(df, seasonality="quarter", show_mean=True)
        series_colors = []
        mean_colors = []
        for trace in fig.data:
            if trace.mode == "lines+markers":
                series_colors.append(trace.line.color)
            elif trace.mode == "lines" and trace.line.dash == "dash":
                mean_colors.append(trace.line.color)
        # There must be at least 2 distinct series (g1, g2)
        assert len(set(series_colors)) >= 2
        # Mean colors should not all be the same (they match their series)
        assert len(set(mean_colors)) >= 2


class TestPanelPhase:
    """Panel tests for plot_phase."""

    def test_two_groups(self, panel_daily_df):
        """Two-group panel phase plot produces a valid figure."""
        fig = plot_phase(panel_daily_df, groups=["y"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_columns_filter_member(self, panel_daily_df):
        """Columns filter selects specific member within phase panel."""
        fig = plot_phase(panel_daily_df, groups=["y"], columns=["a"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
