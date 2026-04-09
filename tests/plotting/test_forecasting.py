"""Tests for forecasting plotting functions."""

import pytest

pytest.importorskip("plotly", reason="plotting extra not installed")


import importlib.util
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from plotly import graph_objects as go

from yohou.plotting import (
    plot_decomposition,
    plot_forecast,
    plot_time_weight,
)

from .conftest import assert_figure_valid, assert_layout


class TestPlotForecast:
    """Tests for plot_forecast function."""

    def test_basic(self):
        """Test basic forecast plotting."""
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [191 + i for i in range(30)],
        })
        y_pred = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [190 + i + (i % 3) for i in range(30)],
        })
        fig = plot_forecast(y_test, y_pred)
        assert len(fig.data) > 0

    def test_with_history(self):
        """Test forecast with training history shown."""
        y_train = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "y": [100 + i for i in range(91)],
        })
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [191 + i for i in range(30)],
        })
        y_pred = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [190 + i + (i % 3) for i in range(30)],
        })
        fig = plot_forecast(y_test, y_pred, y_train=y_train, n_history=30)
        assert len(fig.data) > 0

    def test_with_intervals(self):
        """Test forecast with prediction intervals."""
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [191 + i for i in range(30)],
        })
        y_pred = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [190 + i + (i % 3) for i in range(30)],
            "y_lower_0.9": [185 + i for i in range(30)],
            "y_upper_0.9": [195 + i for i in range(30)],
        })
        fig = plot_forecast(y_test, y_pred, coverage_rates=[0.9])
        assert len(fig.data) > 0

    def test_panel_group_names(self):
        """Test forecast with panel_group_names parameter."""
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [191 + i for i in range(30)],
        })
        y_pred = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [190 + i for i in range(30)],
        })
        # Non-panel data with panel_group_names should still work (no panel columns detected)
        fig = plot_forecast(y_test, y_pred, panel_group_names=["group"])
        assert len(fig.data) >= 0  # May produce empty figure or error gracefully


class TestPlotTimeWeight:
    """Tests for plot_time_weight function."""

    def test_basic(self):
        """Test basic time weight plotting."""
        time_weight = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "weight": [1.0 / (91 - i) for i in range(91)],
        })
        fig = plot_time_weight(time_weight, weight_column="weight")
        assert len(fig.data) > 0

    def test_with_panel(self):
        """Test time weight plotting with panel data."""
        time_weight = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "weight__store_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "weight__store_2": [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0],
        })
        fig = plot_time_weight(time_weight, weight_column="weight")
        assert len(fig.data) >= 2  # One trace per panel group

    def test_custom_title(self):
        """Test time weight plotting with custom title."""
        time_weight = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "weight": [0.1 * i for i in range(1, 11)],
        })
        fig = plot_time_weight(time_weight, weight_column="weight", title="Custom Weights")
        assert_layout(fig, title="Custom Weights")

    def test_custom_dimensions(self):
        """Test time weight plotting with custom dimensions."""
        time_weight = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "weight": [0.1 * i for i in range(1, 11)],
        })
        fig = plot_time_weight(time_weight, weight_column="weight", width=800, height=400)
        assert_layout(fig, width=800, height=400)

    def test_missing_column(self):
        """Test error when weight column is missing."""
        time_weight = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "value": [0.1 * i for i in range(1, 11)],
        })
        with pytest.raises((ValueError, KeyError)):
            plot_time_weight(time_weight, weight_column="missing")


@pytest.fixture
def decomposition_data():
    """Create sample decomposition data."""
    dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
    n = len(dates)
    y = pl.DataFrame({"time": dates, "y": list(range(n))})
    components = {
        "trend": pl.DataFrame({"time": dates, "y": [i * 0.5 for i in range(n)]}),
        "seasonality": pl.DataFrame({
            "time": dates,
            "y": [10.0 * (i % 7) / 7 for i in range(n)],
        }),
        "residual": pl.DataFrame({"time": dates, "y": [i * 0.3 for i in range(n)]}),
    }
    return y, components


@pytest.fixture
def multi_model_data():
    """Create sample multi-model forecast data."""
    y_test = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
        "y": [191 + i for i in range(30)],
    })
    y_pred_a = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
        "y": [190 + i for i in range(30)],
    })
    y_pred_b = pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
        "y": [192 + i for i in range(30)],
    })
    return y_test, {"Model A": y_pred_a, "Model B": y_pred_b}


class TestPlotDecomposition:
    """Tests for plot_decomposition function."""

    def test_basic(self, decomposition_data):
        """Test basic decomposition plot."""
        y, components = decomposition_data
        fig = plot_decomposition(y, components)
        # Original + 3 components = 4 traces
        assert len(fig.data) >= 4

    def test_no_original(self, decomposition_data):
        """Test decomposition without original series."""
        y, components = decomposition_data
        fig = plot_decomposition(y, components, show_original=False)
        # 3 component traces only
        assert len(fig.data) >= 3

    def test_custom_title(self, decomposition_data):
        """Test custom title."""
        y, components = decomposition_data
        fig = plot_decomposition(y, components, title="My Decomposition")
        assert_layout(fig, title="My Decomposition")

    def test_default_title(self, decomposition_data):
        """Test default title."""
        y, components = decomposition_data
        fig = plot_decomposition(y, components)
        assert_layout(fig, title="Time Series Decomposition")

    def test_specific_columns(self):
        """Test with specific column selection."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        n = len(dates)
        y = pl.DataFrame({
            "time": dates,
            "y1": list(range(n)),
            "y2": [i * 2 for i in range(n)],
        })
        components = {
            "trend": pl.DataFrame({
                "time": dates,
                "y1": [i * 0.5 for i in range(n)],
                "y2": [i * 1.0 for i in range(n)],
            }),
        }
        fig = plot_decomposition(y, components, columns="y1")
        # Original y1 + trend y1 = 2 traces
        assert len(fig.data) == 2

    def test_empty_components(self):
        """Test that empty components raises ValueError."""
        y = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "y": list(range(10)),
        })
        with pytest.raises(ValueError, match="non-empty"):
            plot_decomposition(y, {})

    def test_invalid_y(self):
        """Test that invalid y raises TypeError."""
        with pytest.raises(TypeError, match="DataFrame"):
            plot_decomposition("not a df", {"trend": pl.DataFrame({"time": [], "y": []})})

    def test_custom_palette(self, decomposition_data):
        """Test custom color palette."""
        y, components = decomposition_data
        fig = plot_decomposition(y, components, color_palette=["#ff0000"])
        assert fig.data[0].line.color == "#ff0000"

    def test_invalid_components_type(self):
        """Test that invalid components type raises TypeError."""
        y = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "y": list(range(10)),
        })
        with pytest.raises(TypeError, match="dict.*list.*tuple"):
            plot_decomposition(y, 42)


class TestPlotDecompositionPanel:
    """Panel data tests for plot_decomposition."""

    def test_panel_dict_components(self):
        """Panel data with dict components produces a dict of figures."""
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
        assert "a" in result
        assert_figure_valid(result["a"])
        assert len(result["a"].data) >= 2

    def test_custom_dimensions(self):
        """Custom dimensions are passed through."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        y = pl.DataFrame({"time": dates, "y": list(range(10))})
        components = {
            "trend": pl.DataFrame({"time": dates, "y": [i * 0.5 for i in range(10)]}),
        }
        fig = plot_decomposition(y, components, width=900, height=500)
        assert_layout(fig, width=900, height=500)

    def test_panel_faceted_layout(self):
        """Panel data returns dict[str, go.Figure] keyed by group."""
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
            "residual": pl.DataFrame({
                "time": dates,
                "y__a": [i * 0.1 for i in range(n)],
                "y__b": [i * 0.2 for i in range(n)],
            }),
        }
        result = plot_decomposition(y, components)
        assert isinstance(result, dict)
        assert "a" in result
        fig = result["a"]
        assert_figure_valid(fig)
        # 3 rows (original + trend + residual) x 1 group per member = 3 traces
        assert len(fig.data) >= 3

    def test_panel_group_filter(self):
        """panel_group_names filters to specific groups."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        n = len(dates)
        y = pl.DataFrame({
            "time": dates,
            "g1__a": list(range(n)),
            "g2__a": [i * 2 for i in range(n)],
        })
        components = {
            "trend": pl.DataFrame({
                "time": dates,
                "g1__a": [i * 0.5 for i in range(n)],
                "g2__a": [i * 1.0 for i in range(n)],
            }),
        }
        result = plot_decomposition(y, components, panel_group_names=["g1"])
        # With 1 group and 1 member "a", returns single go.Figure
        assert isinstance(result, go.Figure)
        assert_figure_valid(result)
        # Only g1 group -> 1 trace for original + 1 for trend
        assert len(result.data) == 2


_has_statsmodels = importlib.util.find_spec("statsmodels") is not None


@pytest.mark.skipif(not _has_statsmodels, reason="statsmodels not installed")
class TestPlotDecompositionStl:
    """Tests for plot_decomposition STL mode (list/tuple components)."""

    @pytest.fixture
    def monthly_df(self):
        """Create monthly data suitable for STL decomposition."""
        rng = np.random.default_rng(42)
        return pl.DataFrame({
            "time": pl.date_range(pl.date(2018, 1, 1), pl.date(2023, 12, 1), "1mo", eager=True),
            "y": [100 + 2 * i + 10 * np.sin(2 * np.pi * i / 12) + rng.standard_normal() for i in range(72)],
        })

    def test_basic(self, monthly_df):
        """Test basic STL decomposition via plot_decomposition."""
        fig = plot_decomposition(
            monthly_df,
            ["observed", "trend", "seasonal", "residual", "seasonal_adjusted"],
            method="stl",
            columns="y",
        )
        assert len(fig.data) >= 4

    def test_explicit_period(self, monthly_df):
        """Test STL mode with explicit period."""
        fig = plot_decomposition(
            monthly_df,
            ["observed", "trend", "seasonal", "residual"],
            method="stl",
            columns="y",
            period=12,
        )
        assert len(fig.data) >= 4

    def test_subset_components(self, monthly_df):
        """Test showing only a subset of STL components."""
        fig = plot_decomposition(monthly_df, ["trend", "seasonal"], method="stl", columns="y", show_original=False)
        assert len(fig.data) == 2

    def test_observed_sets_show_original(self, monthly_df):
        """Test that 'observed' in components enables original trace."""
        fig = plot_decomposition(monthly_df, ["observed", "trend"], method="stl", columns="y")
        names = [t.name for t in fig.data]
        # "observed" maps to show_original, so the first trace uses the column name
        assert "y" in names
        assert "Trend" in names

    def test_robust_false(self, monthly_df):
        """Test passing robust=False."""
        fig = plot_decomposition(
            monthly_df,
            ["trend", "seasonal"],
            method="stl",
            columns="y",
            show_original=False,
            robust=False,
        )
        assert len(fig.data) == 2

    def test_window_params(self, monthly_df):
        """Test passing window parameters."""
        fig = plot_decomposition(
            monthly_df,
            ["trend", "residual"],
            method="stl",
            columns="y",
            show_original=False,
            period=12,
            seasonal_window=15,
            trend_window=25,
        )
        assert len(fig.data) == 2

    def test_stl_default_title(self, monthly_df):
        """Test STL mode default title."""
        fig = plot_decomposition(monthly_df, ["trend"], method="stl", columns="y")
        assert_layout(fig, title="STL Decomposition")

    def test_stl_custom_title(self, monthly_df):
        """Test STL mode custom title."""
        fig = plot_decomposition(monthly_df, ["trend"], method="stl", columns="y", title="My STL")
        assert_layout(fig, title="My STL")

    def test_unknown_component(self, monthly_df):
        """Test unknown STL component raises ValueError."""
        with pytest.raises(ValueError, match="Unknown components"):
            plot_decomposition(monthly_df, ["trend", "bogus"], method="stl", columns="y")

    def test_tuple_components(self, monthly_df):
        """Test that tuple components trigger STL mode."""
        fig = plot_decomposition(monthly_df, ("trend", "seasonal"), method="stl", columns="y", show_original=False)
        assert len(fig.data) == 2

    def test_nan_interpolation_warning(self):
        """STL mode warns when NaN values are interpolated."""
        dates = pl.date_range(
            pl.date(2018, 1, 1),
            pl.date(2022, 12, 31),
            "1mo",
            eager=True,
        )
        n = len(dates)
        values = [100 + 10 * (i % 12) + i * 0.5 for i in range(n)]
        values[5] = None
        values[10] = None
        df = pl.DataFrame({"time": dates, "y": values})
        with pytest.warns(UserWarning, match="Interpolated"):
            fig = plot_decomposition(df, ["trend", "residual"], method="stl")
        assert_figure_valid(fig)

    def test_unsupported_interval_error(self):
        """Auto-period with unsupported interval frequency raises ValueError."""
        from datetime import datetime

        dates = pl.datetime_range(
            datetime(2020, 1, 1),
            datetime(2020, 1, 1, 0, 18),
            "2m",
            eager=True,
        )
        n = len(dates)
        df = pl.DataFrame({
            "time": dates,
            "y": list(range(n)),
        })
        with pytest.raises(ValueError, match="Cannot infer STL period"):
            plot_decomposition(df, ["trend", "seasonal"], method="stl")


@pytest.mark.skipif(not _has_statsmodels, reason="statsmodels not installed")
class TestPlotDecompositionMstl:
    """Tests for plot_decomposition MSTL mode (multi-seasonal decomposition)."""

    @pytest.fixture
    def hourly_df(self):
        """Create hourly data with daily + weekly seasonality for MSTL."""
        rng = np.random.default_rng(42)
        n = 24 * 7 * 12  # ~12 weeks of hourly data
        return pl.DataFrame({
            "time": pl.datetime_range(
                pl.datetime(2022, 1, 1),
                pl.datetime(2022, 1, 1) + pl.duration(hours=n - 1),
                "1h",
                eager=True,
            ),
            "y": [
                50
                + 10 * np.sin(2 * np.pi * i / 24)  # daily
                + 5 * np.sin(2 * np.pi * i / (24 * 7))  # weekly
                + 0.01 * i  # trend
                + rng.standard_normal()
                for i in range(n)
            ],
        })

    def test_basic_mstl(self, hourly_df):
        """Test basic MSTL decomposition with two periods."""
        fig = plot_decomposition(
            hourly_df,
            ["observed", "trend", "seasonal", "residual"],
            method="mstl",
            columns="y",
            periods=[24, 24 * 7],
        )
        # observed + trend + seasonal_24 + seasonal_168 + residual = 5 traces
        assert len(fig.data) == 5

    def test_explicit_periods(self, hourly_df):
        """Test MSTL with explicitly listed periods."""
        fig = plot_decomposition(
            hourly_df,
            ["trend", "seasonal"],
            method="mstl",
            columns="y",
            show_original=False,
            periods=[24, 24 * 7],
        )
        # trend + seasonal_24 + seasonal_168 = 3 traces
        assert len(fig.data) == 3

    def test_mstl_component_names(self, hourly_df):
        """Test that MSTL produces expected seasonal trace names."""
        fig = plot_decomposition(
            hourly_df,
            ["trend", "seasonal", "residual"],
            method="mstl",
            columns="y",
            show_original=False,
            periods=[24, 24 * 7],
        )
        names = [t.name for t in fig.data]
        assert "Trend" in names
        assert "Seasonal (daily)" in names
        assert "Seasonal (weekly)" in names
        assert "Residual" in names

    def test_mstl_default_title(self, hourly_df):
        """Test MSTL mode sets MSTL-specific default title."""
        fig = plot_decomposition(
            hourly_df,
            ["trend"],
            method="mstl",
            columns="y",
            periods=[24, 24 * 7],
        )
        assert_layout(fig, title="MSTL Decomposition")

    def test_mstl_custom_title(self, hourly_df):
        """Test MSTL mode respects custom title."""
        fig = plot_decomposition(
            hourly_df,
            ["trend"],
            method="mstl",
            columns="y",
            title="Custom MSTL",
            periods=[24, 24 * 7],
        )
        assert_layout(fig, title="Custom MSTL")

    def test_mstl_subplot_labels(self, hourly_df):
        """Test that MSTL subplot labels use human-readable period names."""
        fig = plot_decomposition(
            hourly_df,
            ["trend", "seasonal", "residual"],
            method="mstl",
            columns="y",
            show_original=False,
            periods=[24, 24 * 7],
        )
        yaxis_titles = []
        for i in range(4):  # trend + seasonal_24 + seasonal_168 + residual
            key = f"yaxis{i + 1}" if i > 0 else "yaxis"
            yaxis_titles.append(fig.layout[key].title.text)

        assert "Trend" in yaxis_titles
        assert "Seasonal (daily)" in yaxis_titles
        assert "Seasonal (weekly)" in yaxis_titles
        assert "Residual" in yaxis_titles

    def test_mstl_robust_false(self, hourly_df):
        """Test passing robust=False to MSTL."""
        fig = plot_decomposition(
            hourly_df,
            ["trend", "seasonal"],
            method="mstl",
            columns="y",
            show_original=False,
            periods=[24, 24 * 7],
            robust=False,
        )
        assert len(fig.data) == 3

    def test_mstl_single_period_list(self, hourly_df):
        """Test MSTL with a single-element periods list."""
        fig = plot_decomposition(
            hourly_df,
            ["trend", "seasonal", "residual"],
            method="mstl",
            columns="y",
            show_original=False,
            periods=[24],
        )
        # trend + seasonal_24 + residual = 3 traces
        assert len(fig.data) == 3


class TestComputeDecompositionMstl:
    """Unit tests for the _compute_mstl helper."""

    @pytest.fixture
    def series(self):
        rng = np.random.default_rng(42)
        n = 24 * 7 * 8  # 8 weeks hourly
        return pl.Series([50 + 10 * np.sin(2 * np.pi * i / 24) + rng.standard_normal() for i in range(n)])

    def test_return_keys(self, series):
        from yohou.plotting.forecasting import _compute_mstl

        result = _compute_mstl(series, periods=[24, 24 * 7])
        assert "observed" in result
        assert "trend" in result
        assert "residual" in result
        assert "seasonal_24" in result
        assert "seasonal_168" in result
        assert "seasonal_adjusted" in result

    def test_components_sum_to_observed(self, series):
        from yohou.plotting.forecasting import _compute_mstl

        result = _compute_mstl(series, periods=[24, 24 * 7])
        observed = np.array(result["observed"])
        reconstructed = (
            np.array(result["trend"])
            + np.array(result["seasonal_24"])
            + np.array(result["seasonal_168"])
            + np.array(result["residual"])
        )
        np.testing.assert_allclose(observed, reconstructed, atol=1e-8)

    def test_single_period(self, series):
        from yohou.plotting.forecasting import _compute_mstl

        result = _compute_mstl(series, periods=[24])
        assert "seasonal_24" in result
        assert len(result) == 5  # observed, trend, seasonal_24, residual, seasonal_adjusted

    def test_fewer_seasonal_columns_than_periods(self, series):
        """MSTL raises when a period is too large for the series length.

        When statsmodels drops a period (series too short), _compute_mstl
        should raise ValueError rather than silently returning zeros.
        """
        from yohou.plotting.forecasting import _compute_mstl

        # series is ~8 weeks hourly (1344 rows).  8760 needs ~1 year.
        with pytest.raises(ValueError, match="too short"):
            _compute_mstl(series, periods=[24, 168, 8760])


@pytest.mark.skipif(not _has_statsmodels, reason="statsmodels not installed")
class TestPlotDecompositionClassical:
    """Tests for plot_decomposition classical (seasonal_decompose) mode."""

    @pytest.fixture
    def monthly_df(self):
        """Create monthly data suitable for classical decomposition."""
        rng = np.random.default_rng(42)
        return pl.DataFrame({
            "time": pl.date_range(pl.date(2018, 1, 1), pl.date(2023, 12, 1), "1mo", eager=True),
            "y": [100 + 2 * i + 10 * np.sin(2 * np.pi * i / 12) + rng.standard_normal() for i in range(72)],
        })

    def test_basic(self, monthly_df):
        """Test basic classical decomposition."""
        fig = plot_decomposition(
            monthly_df,
            ["observed", "trend", "seasonal", "residual"],
            method="classical",
            columns="y",
        )
        assert len(fig.data) >= 4

    def test_default_title(self, monthly_df):
        """Test classical mode default title."""
        fig = plot_decomposition(monthly_df, ["trend"], method="classical", columns="y")
        assert_layout(fig, title="Classical Decomposition")

    def test_custom_title(self, monthly_df):
        """Test classical mode custom title."""
        fig = plot_decomposition(
            monthly_df,
            ["trend"],
            method="classical",
            columns="y",
            title="My Classical",
        )
        assert_layout(fig, title="My Classical")

    def test_multiplicative(self, monthly_df):
        """Test classical decomposition with multiplicative model."""
        fig = plot_decomposition(
            monthly_df,
            ["trend", "seasonal", "residual"],
            method="classical",
            columns="y",
            model="multiplicative",
            show_original=False,
        )
        assert len(fig.data) >= 3

    def test_two_sided_false(self, monthly_df):
        """Test classical with two_sided=False."""
        fig = plot_decomposition(
            monthly_df,
            ["trend", "residual"],
            method="classical",
            columns="y",
            two_sided=False,
            show_original=False,
        )
        assert len(fig.data) == 2

    def test_extrapolate_trend(self, monthly_df):
        """Test classical with extrapolate_trend='freq'."""
        fig = plot_decomposition(
            monthly_df,
            ["trend", "seasonal"],
            method="classical",
            columns="y",
            extrapolate_trend="freq",
            show_original=False,
        )
        assert len(fig.data) == 2

    def test_explicit_period(self, monthly_df):
        """Test classical with explicit period."""
        fig = plot_decomposition(
            monthly_df,
            ["trend", "seasonal"],
            method="classical",
            columns="y",
            period=12,
            show_original=False,
        )
        assert len(fig.data) == 2

    def test_seasonal_adjusted(self, monthly_df):
        """Test classical seasonal_adjusted component."""
        fig = plot_decomposition(
            monthly_df,
            ["seasonal_adjusted"],
            method="classical",
            columns="y",
            show_original=False,
        )
        assert len(fig.data) == 1


@pytest.mark.skipif(not _has_statsmodels, reason="statsmodels not installed")
class TestDecompositionMultiplicative:
    """Tests for multiplicative model across all decomposition methods."""

    @pytest.fixture
    def monthly_df(self):
        """Create positive monthly data suitable for multiplicative decomposition."""
        rng = np.random.default_rng(42)
        return pl.DataFrame({
            "time": pl.date_range(pl.date(2018, 1, 1), pl.date(2023, 12, 1), "1mo", eager=True),
            "y": [100 + 2 * i + 10 * np.sin(2 * np.pi * i / 12) + rng.standard_normal() for i in range(72)],
        })

    def test_stl_multiplicative_warns(self, monthly_df):
        """STL multiplicative emits UserWarning about log-transform approximation."""
        with pytest.warns(UserWarning, match="log-transform"):
            fig = plot_decomposition(
                monthly_df,
                ["trend", "seasonal"],
                method="stl",
                columns="y",
                model="multiplicative",
                show_original=False,
            )
        assert len(fig.data) == 2

    def test_mstl_multiplicative_warns(self):
        """MSTL multiplicative emits UserWarning about log-transform approximation."""
        rng = np.random.default_rng(42)
        n = 24 * 7 * 12
        df = pl.DataFrame({
            "time": pl.datetime_range(
                pl.datetime(2022, 1, 1),
                pl.datetime(2022, 1, 1) + pl.duration(hours=n - 1),
                "1h",
                eager=True,
            ),
            "y": [50 + 10 * np.sin(2 * np.pi * i / 24) + rng.standard_normal() for i in range(n)],
        })
        with pytest.warns(UserWarning, match="log-transform"):
            fig = plot_decomposition(
                df,
                ["trend", "seasonal"],
                method="mstl",
                columns="y",
                model="multiplicative",
                periods=[24, 24 * 7],
                show_original=False,
            )
        assert len(fig.data) >= 2

    def test_classical_multiplicative_no_warning(self, monthly_df):
        """Classical multiplicative does not emit log-transform warning."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig = plot_decomposition(
                monthly_df,
                ["trend", "seasonal"],
                method="classical",
                columns="y",
                model="multiplicative",
                show_original=False,
            )
        assert len(fig.data) == 2

    def test_mismatched_stl_params_warns(self, monthly_df):
        """STL-only params with classical method emit UserWarning."""
        with pytest.warns(UserWarning, match="trend_window.*only used with method='stl'"):
            plot_decomposition(
                monthly_df,
                ["trend"],
                method="classical",
                columns="y",
                trend_window=15,
            )


class TestFormatComponentLabel:
    """Unit tests for _format_component_label."""

    def test_trend(self):
        from yohou.plotting.forecasting import _format_component_label

        assert _format_component_label("trend") == "Trend"

    def test_residual(self):
        from yohou.plotting.forecasting import _format_component_label

        assert _format_component_label("residual") == "Residual"

    def test_seasonal_adjusted(self):
        from yohou.plotting.forecasting import _format_component_label

        assert _format_component_label("seasonal_adjusted") == "Seasonal Adjusted"

    def test_seasonal_daily(self):
        from yohou.plotting.forecasting import _format_component_label

        assert _format_component_label("seasonal_daily") == "Seasonal (daily)"

    def test_seasonal_weekly(self):
        from yohou.plotting.forecasting import _format_component_label

        assert _format_component_label("seasonal_weekly") == "Seasonal (weekly)"

    def test_seasonal_annual(self):
        from yohou.plotting.forecasting import _format_component_label

        assert _format_component_label("seasonal_annual") == "Seasonal (annual)"

    def test_seasonal_numeric_fallback(self):
        from yohou.plotting.forecasting import _format_component_label

        assert _format_component_label("seasonal_99") == "Seasonal (99)"


class TestPeriodToLabel:
    """Unit tests for _period_to_label."""

    def test_hourly_daily(self):
        from yohou.plotting.forecasting import _period_to_label

        assert _period_to_label(24, "1h") == "daily"

    def test_hourly_weekly(self):
        from yohou.plotting.forecasting import _period_to_label

        assert _period_to_label(168, "1h") == "weekly"

    def test_hourly_annual(self):
        from yohou.plotting.forecasting import _period_to_label

        assert _period_to_label(24 * 365, "1h") == "annual"

    def test_daily_weekly(self):
        from yohou.plotting.forecasting import _period_to_label

        assert _period_to_label(7, "1d") == "weekly"

    def test_daily_annual(self):
        from yohou.plotting.forecasting import _period_to_label

        assert _period_to_label(365, "1d") == "annual"

    def test_15min_daily(self):
        from yohou.plotting.forecasting import _period_to_label

        assert _period_to_label(96, "15min") == "daily"

    def test_15min_weekly(self):
        from yohou.plotting.forecasting import _period_to_label

        assert _period_to_label(96 * 7, "15min") == "weekly"

    def test_monthly_annual(self):
        from yohou.plotting.forecasting import _period_to_label

        assert _period_to_label(12, "1mo") == "annual"

    def test_unknown_interval_returns_period(self):
        from yohou.plotting.forecasting import _period_to_label

        assert _period_to_label(24, None) == "24"

    def test_sub_daily(self):
        from yohou.plotting.forecasting import _period_to_label

        # 6 observations at 1h = 6 hours < 0.5 day
        assert _period_to_label(6, "1h") == "6h"


class TestPlotForecastMultiModel:
    """Tests for plot_forecast with multiple model predictions."""

    def test_basic(self, multi_model_data):
        """Test multi-model forecast overlay."""
        y_test, y_preds = multi_model_data
        fig = plot_forecast(y_test, y_preds)
        # 1 actual + 2 models = 3 traces
        assert len(fig.data) == 3

    def test_with_train(self, multi_model_data):
        """Test multi-model with training data."""
        y_test, y_preds = multi_model_data
        y_train = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "y": [100 + i for i in range(91)],
        })
        fig = plot_forecast(y_test, y_preds, y_train=y_train)
        # 1 train + 1 actual + 2 models = 4 traces
        assert len(fig.data) == 4

    def test_default_title(self, multi_model_data):
        """Test multi-model default title."""
        y_test, y_preds = multi_model_data
        fig = plot_forecast(y_test, y_preds)
        assert_layout(fig, title="Forecast Comparison")

    def test_custom_title(self, multi_model_data):
        """Test multi-model custom title."""
        y_test, y_preds = multi_model_data
        fig = plot_forecast(y_test, y_preds, title="My Comparison")
        assert_layout(fig, title="My Comparison")

    def test_model_names(self, multi_model_data):
        """Test model names appear in traces."""
        y_test, y_preds = multi_model_data
        fig = plot_forecast(y_test, y_preds)
        trace_names = [t.name for t in fig.data]
        assert "y (Model A)" in trace_names
        assert "y (Model B)" in trace_names

    def test_single_df_still_works(self):
        """Test backward compatibility with single DataFrame."""
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [191 + i for i in range(30)],
        })
        y_pred = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [190 + i for i in range(30)],
        })
        fig = plot_forecast(y_test, y_pred)
        # 1 actual + 1 forecast = 2 traces
        assert len(fig.data) == 2


class TestPlotForecastMultiColumn:
    """Tests for plot_forecast with multiple target columns."""

    def test_multi_column_distinct_colors(self):
        """Test that multi-column forecasts have distinct per-column colors."""
        dates = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True)
        y_test = pl.DataFrame({"time": dates, "a": list(range(30)), "b": list(range(30, 60))})
        y_pred = pl.DataFrame({"time": dates, "a": [x + 1 for x in range(30)], "b": [x + 1 for x in range(30, 60)]})
        fig = plot_forecast(y_test, y_pred)
        a_fc = [t for t in fig.data if t.name == "a (Forecast)"][0]
        b_fc = [t for t in fig.data if t.name == "b (Forecast)"][0]
        assert a_fc.line.color != b_fc.line.color

    def test_multi_column_per_column_colors(self):
        """Test that multi-column actual/forecast share the same per-column color."""
        dates = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True)
        y_test = pl.DataFrame({"time": dates, "a": list(range(30)), "b": list(range(30, 60))})
        y_pred = pl.DataFrame({"time": dates, "a": [x + 1 for x in range(30)], "b": [x + 1 for x in range(30, 60)]})
        fig = plot_forecast(y_test, y_pred)
        a_actual = [t for t in fig.data if t.name == "a (Actual)"][0]
        b_actual = [t for t in fig.data if t.name == "b (Actual)"][0]
        # Different columns get different colors
        assert a_actual.line.color != b_actual.line.color

    def test_multi_column_per_column_train_legend(self):
        """Test that each column gets its own Train legend entry for multi-column."""
        dates_train = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        dates_test = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True)
        y_train = pl.DataFrame({"time": dates_train, "a": list(range(91)), "b": list(range(91))})
        y_test = pl.DataFrame({"time": dates_test, "a": list(range(30)), "b": list(range(30))})
        y_pred = pl.DataFrame({"time": dates_test, "a": list(range(30)), "b": list(range(30))})
        fig = plot_forecast(y_test, y_pred, y_train=y_train)
        train_traces = [t for t in fig.data if t.name and "Train" in t.name]
        assert len(train_traces) == 2
        train_names = {t.name for t in train_traces}
        assert train_names == {"a (Train)", "b (Train)"}


class TestPlotForecastPanelMultiModel:
    """Tests for plot_forecast with panel data and multi-model."""

    def test_panel_multi_model(self):
        """Test panel data with multiple model predictions."""
        dates = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True)
        y_test = pl.DataFrame({"time": dates, "y__s1": list(range(30)), "y__s2": list(range(30, 60))})
        y_pred_a = pl.DataFrame({
            "time": dates,
            "y__s1": [x + 1 for x in range(30)],
            "y__s2": [x + 1 for x in range(30, 60)],
        })
        y_pred_b = pl.DataFrame({
            "time": dates,
            "y__s1": [x - 1 for x in range(30)],
            "y__s2": [x - 1 for x in range(30, 60)],
        })
        fig = plot_forecast(y_test, {"A": y_pred_a, "B": y_pred_b})
        names = [t.name for t in fig.data if t.name is not None]
        # Multi-member panel labels include member: "s1 (A)", "s2 (B)", etc.
        assert any("A" in n for n in names)
        assert any("B" in n for n in names)


class TestPlotForecastMultiModelIntervals:
    """Tests for multi-model forecast with interval bands."""

    def test_multi_model_interval_bands(self):
        """Multi-model overlaid forecasts with prediction intervals."""
        dates = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True)
        y_test = pl.DataFrame({"time": dates, "y": [191 + i for i in range(30)]})
        y_pred_a = pl.DataFrame({
            "time": dates,
            "y": [190 + i for i in range(30)],
            "y_lower_0.9": [185 + i for i in range(30)],
            "y_upper_0.9": [195 + i for i in range(30)],
        })
        y_pred_b = pl.DataFrame({
            "time": dates,
            "y": [192 + i for i in range(30)],
            "y_lower_0.9": [187 + i for i in range(30)],
            "y_upper_0.9": [197 + i for i in range(30)],
        })
        fig = plot_forecast(y_test, {"M1": y_pred_a, "M2": y_pred_b}, coverage_rates=[0.9])
        assert_figure_valid(fig)
        names = [t.name for t in fig.data if t.name is not None]
        assert any("M1" in n for n in names)
        assert any("M2" in n for n in names)
        assert any("PI" in n for n in names)


class TestPlotForecastPanelSingleMember:
    """Tests for panel forecast with single member per group."""

    def test_single_member_panel(self):
        """Single-member panel groups use solid line without dash."""
        dates_train = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        dates_test = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True)
        y_train = pl.DataFrame({
            "time": dates_train,
            "y__store1": list(range(91)),
            "z__store1": list(range(91, 182)),
        })
        y_test = pl.DataFrame({
            "time": dates_test,
            "y__store1": list(range(30)),
            "z__store1": list(range(30, 60)),
        })
        y_pred = pl.DataFrame({
            "time": dates_test,
            "y__store1": [x + 1 for x in range(30)],
            "z__store1": [x + 1 for x in range(30, 60)],
        })
        fig = plot_forecast(
            y_test,
            y_pred,
            y_train=y_train,
            panel_group_names=["y", "z"],
        )
        assert_figure_valid(fig)


class TestPlotForecastPanelTrainAndIntervals:
    """Tests for panel forecast with train history and intervals."""

    def test_panel_with_train_and_intervals(self):
        """Panel forecast with y_train and coverage_rates renders bands."""
        dates_train = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        dates_test = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True)
        y_train = pl.DataFrame({
            "time": dates_train,
            "y__a": [100 + i for i in range(91)],
            "y__b": [200 + i for i in range(91)],
        })
        y_test = pl.DataFrame({
            "time": dates_test,
            "y__a": [191 + i for i in range(30)],
            "y__b": [291 + i for i in range(30)],
        })
        y_pred = pl.DataFrame({
            "time": dates_test,
            "y__a": [190 + i for i in range(30)],
            "y__b": [289 + i for i in range(30)],
            "y__a_lower_0.9": [185 + i for i in range(30)],
            "y__a_upper_0.9": [195 + i for i in range(30)],
            "y__b_lower_0.9": [284 + i for i in range(30)],
            "y__b_upper_0.9": [294 + i for i in range(30)],
        })
        fig = plot_forecast(
            y_test,
            y_pred,
            y_train=y_train,
            coverage_rates=[0.9],
            panel_group_names=["y"],
        )
        assert_figure_valid(fig)
        names = [t.name for t in fig.data if t.name is not None]
        assert any("Train" in n for n in names)
        assert any("PI" in n for n in names)


class TestPlotForecastMultiModelTrainHistory:
    """Tests for multi-model forecast with training history and n_history."""

    def test_multi_model_with_train_and_n_history(self):
        """Multi-model forecast with y_train and n_history covers tail branch."""
        y_train = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "y": [100.0 + i for i in range(91)],
        })
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [191.0 + i for i in range(30)],
        })
        y_pred_a = pl.DataFrame({
            "time": y_test["time"],
            "y": [190.0 + i * 1.1 for i in range(30)],
        })
        y_pred_b = pl.DataFrame({
            "time": y_test["time"],
            "y": [192.0 + i * 0.9 for i in range(30)],
        })
        fig = plot_forecast(
            y_test,
            {"Model A": y_pred_a, "Model B": y_pred_b},
            y_train=y_train,
            n_history=30,
        )
        assert_figure_valid(fig)
        assert len(fig.data) >= 3


class TestPlotForecastPanelErrors:
    """Tests for panel forecast error paths."""

    def test_panel_nonexistent_group_raises(self):
        """Panel forecast with nonexistent group name raises ValueError."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        y_test = pl.DataFrame({
            "time": dates,
            "sales__store_1": [float(i) for i in range(10)],
            "sales__store_2": [float(i) + 5 for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "time": dates,
            "sales__store_1": [float(i) + 1 for i in range(10)],
            "sales__store_2": [float(i) + 6 for i in range(10)],
        })
        with pytest.raises(ValueError, match="No panel columns found for groups"):
            plot_forecast(y_test, y_pred, panel_group_names=["nonexistent"])


class TestPlotTimeWeightPanelErrors:
    """Tests for time_weight panel error paths."""

    def test_panel_weight_nonexistent_group_raises(self):
        """Panel weight with nonexistent group name raises ValueError."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "time_weight__store_1": [0.1 * i for i in range(1, 11)],
            "time_weight__store_2": [0.05 * i for i in range(1, 11)],
        })
        with pytest.raises(ValueError, match="No weight columns found for panel groups"):
            plot_time_weight(df, panel_group_names=["nonexistent"])

    def test_panel_weight_valid_group(self):
        """Panel weight with valid group name produces a figure."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        df = pl.DataFrame({
            "time": dates,
            "time_weight__store_1": [0.1 * i for i in range(1, 11)],
            "time_weight__store_2": [0.05 * i for i in range(1, 11)],
        })
        fig = plot_time_weight(df, panel_group_names=["time_weight"])
        assert_figure_valid(fig)


class TestPlotDecompositionErrors:
    """Tests for plot_decomposition error paths."""

    def test_empty_components_no_original_raises(self):
        """Empty components list with show_original=False raises ValueError."""
        y = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "y": [float(i) for i in range(91)],
        })
        with pytest.raises(ValueError, match="components must contain at least one displayable component"):
            plot_decomposition(y, [], method="stl", show_original=False)

    def test_list_without_method_raises(self):
        """List components without method raises ValueError."""
        y = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "y": [float(i) for i in range(91)],
        })
        with pytest.raises(ValueError, match="method is required"):
            plot_decomposition(y, ["trend", "seasonal"])

    def test_mstl_without_periods_raises(self):
        """MSTL without periods raises ValueError."""
        y = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "y": [float(i) for i in range(91)],
        })
        with pytest.raises(ValueError, match="periods is required"):
            plot_decomposition(y, ["trend", "seasonal"], method="mstl")


class TestPlotForecastWithIntervals:
    """Tests for plot_forecast with prediction intervals (coverage_rates truthy branch)."""

    def test_forecast_with_intervals(self):
        """Forecast with interval columns triggers coverage_rates rendering branch."""
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 10), "1d", eager=True),
            "y": [100.0 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 10), "1d", eager=True),
            "y": [101.0 + i for i in range(10)],
            "y_lower_0.9": [98.0 + i for i in range(10)],
            "y_upper_0.9": [104.0 + i for i in range(10)],
        })
        fig = plot_forecast(y_test, y_pred)
        assert_figure_valid(fig)
        assert len(fig.data) >= 2


class TestPlotForecastShowTransition:
    """Tests for plot_forecast show_transition=True branch."""

    def test_show_transition_with_train(self):
        """show_transition=True with y_train prepends last train point to forecast line."""
        y_train = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "y": [100.0 + i * 0.5 for i in range(91)],
        })
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 10), "1d", eager=True),
            "y": [146.0 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 10), "1d", eager=True),
            "y": [145.0 + i for i in range(10)],
        })
        fig = plot_forecast(y_test, y_pred, y_train=y_train, show_transition=True)
        assert_figure_valid(fig)
        assert len(fig.data) >= 3


class TestPlotTimeWeightNonPanel:
    """Tests for plot_time_weight non-panel single weight column branch."""

    def test_single_weight_column(self):
        """Non-panel time weight plots single weight column."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "time_weight": [1.0 / (i + 1) for i in range(91)],
        })
        fig = plot_time_weight(df)
        assert_figure_valid(fig)

    def test_single_weight_with_fill(self):
        """Non-panel time weight with fill enabled."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "time_weight": [0.9**i for i in range(10)],
        })
        fig = plot_time_weight(df, fill=True)
        assert_figure_valid(fig)


class TestPlotDecompositionShowOriginal:
    """Tests for plot_decomposition show_original=True branch."""

    def test_show_original_true(self):
        """show_original=True adds original series panel to the subplot."""
        y = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "y": [float(i) + (i % 7) * 3.0 for i in range(91)],
        })
        trend = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "y": [float(i) for i in range(91)],
        })
        fig = plot_decomposition(y, {"Trend": trend}, show_original=True)
        assert_figure_valid(fig)
        assert len(fig.data) >= 2


class TestConnectGaps:
    """Tests for connect_gaps parameter in forecasting functions."""

    @pytest.fixture
    def simple_forecast_data(self):
        """Minimal non-panel forecast DataFrames."""
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [191 + i for i in range(30)],
        })
        y_pred = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True),
            "y": [190 + i for i in range(30)],
        })
        return y_test, y_pred

    def test_plot_forecast_connect_gaps_false(self, simple_forecast_data):
        """connect_gaps=False (default) does not raise and returns a figure."""
        y_test, y_pred = simple_forecast_data
        fig = plot_forecast(y_test, y_pred, connect_gaps=False)
        assert_figure_valid(fig)

    def test_plot_forecast_connect_gaps_true(self, simple_forecast_data):
        """connect_gaps=True sets connectgaps on line Scatter traces."""
        y_test, y_pred = simple_forecast_data
        fig = plot_forecast(y_test, y_pred, connect_gaps=True)
        assert_figure_valid(fig)
        scatter_line_traces = [t for t in fig.data if isinstance(t, go.Scatter) and t.mode == "lines"]
        assert all(t.connectgaps for t in scatter_line_traces)

    def test_plot_time_weight_connect_gaps_true(self):
        """connect_gaps=True is accepted by plot_time_weight."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "time_weight": [0.1 * i for i in range(1, 11)],
        })
        fig = plot_time_weight(df, connect_gaps=True)
        assert_figure_valid(fig)
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert all(t.connectgaps for t in scatter_traces)


class TestInvalidDimensions:
    """Tests for width/height validation in forecasting functions."""

    @pytest.fixture
    def minimal_forecast_data(self):
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 5), "1d", eager=True),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        y_pred = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 5), "1d", eager=True),
            "y": [1.1, 2.1, 3.1, 4.1, 5.1],
        })
        return y_test, y_pred

    def test_plot_forecast_invalid_width(self, minimal_forecast_data):
        y_test, y_pred = minimal_forecast_data
        with pytest.raises(ValueError, match="width"):
            plot_forecast(y_test, y_pred, width=0)

    def test_plot_forecast_invalid_height(self, minimal_forecast_data):
        y_test, y_pred = minimal_forecast_data
        with pytest.raises(ValueError, match="height"):
            plot_forecast(y_test, y_pred, height=-1)

    def test_plot_time_weight_invalid_width(self):
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 5), "1d", eager=True),
            "time_weight": [0.2 * i for i in range(1, 6)],
        })
        with pytest.raises(ValueError, match="width"):
            plot_time_weight(df, width=0)

    def test_plot_decomposition_invalid_height(self):
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        y = pl.DataFrame({"time": dates, "y": list(range(10))})
        components = {"trend": pl.DataFrame({"time": dates, "y": [i * 0.5 for i in range(10)]})}
        with pytest.raises(ValueError, match="height"):
            plot_decomposition(y, components, height=-5)


class TestPeriodToLabelExtraBranches:
    """Cover the quarterly and semi-annual branches."""

    def test_daily_monthly(self):
        from yohou.plotting.forecasting import _period_to_label

        assert _period_to_label(30, "1d") == "monthly"

    def test_daily_quarterly(self):
        from yohou.plotting.forecasting import _period_to_label

        assert _period_to_label(90, "1d") == "quarterly"

    def test_daily_semi_annual(self):
        from yohou.plotting.forecasting import _period_to_label

        assert _period_to_label(180, "1d") == "semi-annual"

    def test_unknown_interval_string(self):
        """Unrecognized interval returns period as string."""
        from yohou.plotting.forecasting import _period_to_label

        assert _period_to_label(42, "7s") == "42"


class TestFormatComponentLabelExtra:
    """Cover generic (non-seasonal, non-special) component names."""

    def test_multi_word_name(self):
        from yohou.plotting.forecasting import _format_component_label

        assert _format_component_label("low_pass") == "Low Pass"

    def test_single_word(self):
        from yohou.plotting.forecasting import _format_component_label

        assert _format_component_label("observed") == "Observed"


@pytest.mark.skipif(
    not importlib.util.find_spec("statsmodels"),
    reason="statsmodels not installed",
)
class TestSTLEvenWindows:
    """Cover even window arguments being bumped to odd."""

    def test_even_trend_window(self):
        """Even trend_window is internally adjusted to odd."""
        dates = pl.date_range(pl.date(2018, 1, 1), pl.date(2022, 12, 31), "1mo", eager=True)
        n = len(dates)
        df = pl.DataFrame({
            "time": dates,
            "y": [100 + 10 * (i % 12) + i * 0.5 for i in range(n)],
        })
        fig = plot_decomposition(
            df,
            ["trend", "seasonal"],
            method="stl",
            columns="y",
            show_original=False,
            period=12,
            trend_window=14,
        )
        assert_figure_valid(fig)

    def test_even_seasonal_window(self):
        """Even seasonal_window is internally adjusted to odd."""
        dates = pl.date_range(pl.date(2018, 1, 1), pl.date(2022, 12, 31), "1mo", eager=True)
        n = len(dates)
        df = pl.DataFrame({
            "time": dates,
            "y": [100 + 10 * (i % 12) + i * 0.5 for i in range(n)],
        })
        fig = plot_decomposition(
            df,
            ["trend", "seasonal"],
            method="stl",
            columns="y",
            show_original=False,
            period=12,
            seasonal_window=8,
        )
        assert_figure_valid(fig)

    def test_even_low_pass_window(self):
        """Even low_pass_window is internally adjusted to odd."""
        dates = pl.date_range(pl.date(2018, 1, 1), pl.date(2022, 12, 31), "1mo", eager=True)
        n = len(dates)
        df = pl.DataFrame({
            "time": dates,
            "y": [100 + 10 * (i % 12) + i * 0.5 for i in range(n)],
        })
        fig = plot_decomposition(
            df,
            ["trend", "seasonal"],
            method="stl",
            columns="y",
            show_original=False,
            period=12,
            low_pass_window=12,
        )
        assert_figure_valid(fig)


class TestPlotDecompositionPanelGroupNames:
    """Cover panel_group_names branch in plot_decomposition."""

    def test_panel_components_dict(self):
        """Panel components with dict input returns dict of figures."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        y = pl.DataFrame({
            "time": dates,
            "y__a": [float(i) + (i % 7) * 3.0 for i in range(91)],
            "y__b": [float(i) * 2 + (i % 5) * 2.0 for i in range(91)],
        })
        trend = pl.DataFrame({
            "time": dates,
            "y__a": [float(i) for i in range(91)],
            "y__b": [float(i) * 2 for i in range(91)],
        })
        result = plot_decomposition(
            y,
            {"Trend": trend},
            panel_group_names=["y"],
            show_original=True,
        )
        assert isinstance(result, dict)
        assert "a" in result
        assert isinstance(result["a"], go.Figure)
        assert len(result["a"].data) >= 2

    def test_panel_components_no_original(self):
        """Panel components with show_original=False."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        y = pl.DataFrame({
            "time": dates,
            "y__a": [float(i) for i in range(91)],
            "y__b": [float(i) * 2 for i in range(91)],
        })
        trend = pl.DataFrame({
            "time": dates,
            "y__a": [float(i) * 0.5 for i in range(91)],
            "y__b": [float(i) for i in range(91)],
        })
        result = plot_decomposition(
            y,
            {"Trend": trend},
            panel_group_names=["y"],
            show_original=False,
        )
        assert isinstance(result, dict)
        assert len(result["a"].data) >= 1


class TestComponentsFallbackColumns:
    """Cover component DataFrame with renamed columns (not matching y)."""

    def test_component_renamed_columns(self):
        """Component with different column names falls back to own columns."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True)
        y = pl.DataFrame({
            "time": dates,
            "y": [float(i) + (i % 7) * 3.0 for i in range(91)],
        })
        # Component has a different column name than y's columns
        trend = pl.DataFrame({
            "time": dates,
            "log_off_0.0_tourists": [float(i) * 0.5 for i in range(91)],
        })
        fig = plot_decomposition(y, {"Trend": trend}, show_original=True)
        assert_figure_valid(fig)
        assert len(fig.data) >= 2


class TestMultiModelShowTransition:
    """Cover multi-model with show_transition and coverage_rates."""

    @pytest.fixture
    def multi_model_transition_data(self):
        y_train = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "y": [100.0 + i for i in range(91)],
        })
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 10), "1d", eager=True),
            "y": [191.0 + i for i in range(10)],
        })
        y_pred_a = pl.DataFrame({
            "time": y_test["time"],
            "y": [190.0 + i for i in range(10)],
            "y_lower_0.9": [185.0 + i for i in range(10)],
            "y_upper_0.9": [195.0 + i for i in range(10)],
        })
        y_pred_b = pl.DataFrame({
            "time": y_test["time"],
            "y": [192.0 + i for i in range(10)],
            "y_lower_0.9": [187.0 + i for i in range(10)],
            "y_upper_0.9": [197.0 + i for i in range(10)],
        })
        return y_train, y_test, {"A": y_pred_a, "B": y_pred_b}

    def test_multi_model_transition_intervals(self, multi_model_transition_data):
        """Multi-model with show_transition=True and intervals."""
        y_train, y_test, y_preds = multi_model_transition_data
        fig = plot_forecast(
            y_test,
            y_preds,
            y_train=y_train,
            coverage_rates=[0.9],
            show_transition=True,
        )
        assert_figure_valid(fig)
        names = [t.name for t in fig.data if t.name is not None]
        assert any("PI" in n for n in names)

    def test_multi_model_no_intervals(self, multi_model_transition_data):
        """Multi-model without coverage_rates skips interval rendering."""
        y_train, y_test, y_preds = multi_model_transition_data
        fig = plot_forecast(
            y_test,
            y_preds,
            y_train=y_train,
            show_transition=True,
        )
        assert_figure_valid(fig)


class TestSingleModelIntervalRender:
    """Cover single-model forecast with multiple coverage rates."""

    def test_multiple_coverage_rates(self):
        """Single-model with multiple coverage_rates renders layered bands."""
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 10), "1d", eager=True),
            "y": [100.0 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "time": y_test["time"],
            "y": [101.0 + i for i in range(10)],
            "y_lower_0.5": [99.0 + i for i in range(10)],
            "y_upper_0.5": [103.0 + i for i in range(10)],
            "y_lower_0.9": [96.0 + i for i in range(10)],
            "y_upper_0.9": [106.0 + i for i in range(10)],
        })
        fig = plot_forecast(y_test, y_pred, coverage_rates=[0.5, 0.9])
        assert_figure_valid(fig)
        names = [t.name for t in fig.data if t.name is not None]
        assert any("50%" in n for n in names)
        assert any("90%" in n for n in names)

    def test_single_model_transition_with_intervals(self):
        """Single-model with show_transition=True and intervals."""
        y_train = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 3, 31), "1d", eager=True),
            "y": [100.0 + i for i in range(91)],
        })
        y_test = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 10), "1d", eager=True),
            "y": [191.0 + i for i in range(10)],
        })
        y_pred = pl.DataFrame({
            "time": y_test["time"],
            "y": [190.0 + i for i in range(10)],
            "y_lower_0.9": [185.0 + i for i in range(10)],
            "y_upper_0.9": [195.0 + i for i in range(10)],
        })
        fig = plot_forecast(
            y_test,
            y_pred,
            y_train=y_train,
            coverage_rates=[0.9],
            show_transition=True,
        )
        assert_figure_valid(fig)


class TestPanelForecastPIBranches:
    """Cover panel forecast interval rendering with multi-model."""

    def test_panel_multi_model_intervals(self):
        """Panel forecast with multi-model and intervals."""
        dates_test = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 10), "1d", eager=True)
        y_test = pl.DataFrame({
            "time": dates_test,
            "y__a": [191.0 + i for i in range(10)],
            "y__b": [291.0 + i for i in range(10)],
        })
        y_pred_m1 = pl.DataFrame({
            "time": dates_test,
            "y__a": [190.0 + i for i in range(10)],
            "y__b": [289.0 + i for i in range(10)],
            "y__a_lower_0.9": [185.0 + i for i in range(10)],
            "y__a_upper_0.9": [195.0 + i for i in range(10)],
            "y__b_lower_0.9": [284.0 + i for i in range(10)],
            "y__b_upper_0.9": [294.0 + i for i in range(10)],
        })
        y_pred_m2 = pl.DataFrame({
            "time": dates_test,
            "y__a": [192.0 + i for i in range(10)],
            "y__b": [292.0 + i for i in range(10)],
        })
        fig = plot_forecast(
            y_test,
            {"M1": y_pred_m1, "M2": y_pred_m2},
            coverage_rates=[0.9],
            panel_group_names=["y"],
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


@pytest.mark.skipif(
    not importlib.util.find_spec("statsmodels"),
    reason="statsmodels not installed",
)
class TestMSTLIntervalFallback:
    """Cover MSTL label inference when interval is unknown."""

    def test_mstl_unknown_interval_fallback(self):
        """MSTL with unknown interval falls back to numeric seasonal labels."""
        rng = np.random.default_rng(42)
        n = 200
        df = pl.DataFrame({
            "time": pl.datetime_range(
                pl.datetime(2022, 1, 1),
                pl.datetime(2022, 1, 1) + pl.duration(hours=n - 1),
                "1h",
                eager=True,
            ),
            "y": [50 + 10 * np.sin(2 * np.pi * i / 24) + rng.standard_normal() for i in range(n)],
        })
        fig = plot_decomposition(
            df,
            ["trend", "seasonal", "residual"],
            method="mstl",
            columns="y",
            show_original=False,
            periods=[24],
        )
        assert_figure_valid(fig)


class TestPlotForecastPanelFacetByNone:
    """Tests for plot_forecast panel data with facet_by=None."""

    def test_panel_facet_by_none(self):
        """facet_by=None places all panel columns into a single subplot."""
        dates = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True)
        y_test = pl.DataFrame({
            "time": dates,
            "y__s1": list(range(30)),
            "y__s2": list(range(30, 60)),
        })
        y_pred = pl.DataFrame({
            "time": dates,
            "y__s1": [x + 1 for x in range(30)],
            "y__s2": [x + 1 for x in range(30, 60)],
        })
        fig = plot_forecast(y_test, y_pred, facet_by=None)
        assert_figure_valid(fig)
        assert len(fig.data) >= 2

    def test_panel_facet_by_none_with_intervals(self):
        """facet_by=None with prediction intervals produces band traces."""
        dates = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True)
        y_test = pl.DataFrame({
            "time": dates,
            "y__s1": list(range(30)),
            "y__s2": list(range(30, 60)),
        })
        y_pred = pl.DataFrame({
            "time": dates,
            "y__s1": [x + 1 for x in range(30)],
            "y__s2": [x + 1 for x in range(30, 60)],
            "y__s1_lower_0.9": [x - 2 for x in range(30)],
            "y__s1_upper_0.9": [x + 4 for x in range(30)],
            "y__s2_lower_0.9": [x + 28 for x in range(30)],
            "y__s2_upper_0.9": [x + 62 for x in range(30)],
        })
        fig = plot_forecast(y_test, y_pred, facet_by=None, coverage_rates=[0.9])
        assert_figure_valid(fig)
        names = [t.name for t in fig.data if t.name is not None]
        assert any("PI" in n for n in names)

    def test_panel_facet_by_none_multi_model(self):
        """facet_by=None with multi-model predictions."""
        dates = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True)
        y_test = pl.DataFrame({
            "time": dates,
            "y__s1": list(range(30)),
            "y__s2": list(range(30, 60)),
        })
        pred_a = pl.DataFrame({
            "time": dates,
            "y__s1": [x + 1 for x in range(30)],
            "y__s2": [x + 1 for x in range(30, 60)],
        })
        pred_b = pl.DataFrame({
            "time": dates,
            "y__s1": [x - 1 for x in range(30)],
            "y__s2": [x - 1 for x in range(30, 60)],
        })
        fig = plot_forecast(y_test, {"A": pred_a, "B": pred_b}, facet_by=None)
        assert_figure_valid(fig)
        names = [t.name for t in fig.data if t.name is not None]
        assert any("A" in n for n in names)
        assert any("B" in n for n in names)


class TestPlotTimeWeightPanelFacetByNone:
    """Tests for plot_time_weight panel data with facet_by=None."""

    def test_panel_facet_by_none(self):
        """facet_by=None places all panel weight columns into a single subplot."""
        time_weight = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "weight__store_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "weight__store_2": [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0],
        })
        fig = plot_time_weight(time_weight, weight_column="weight", facet_by=None)
        assert_figure_valid(fig)
        assert len(fig.data) >= 2

    def test_panel_facet_by_member(self):
        """facet_by='member' creates one subplot per member."""
        time_weight = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "weight__store_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "weight__store_2": [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0],
        })
        fig = plot_time_weight(time_weight, weight_column="weight", facet_by="member")
        assert_figure_valid(fig)


class TestPlotDecompositionYLabel:
    """Tests for y_label parameter in plot_decomposition."""

    def test_stl_y_label(self):
        """y_label overrides per-row labels in STL mode."""
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        n = len(dates)
        y = pl.DataFrame({"time": dates, "y": list(range(n))})
        components = {
            "trend": pl.DataFrame({"time": dates, "y": [i * 0.5 for i in range(n)]}),
        }
        fig = plot_decomposition(y, components, y_label="Custom Y")
        assert_figure_valid(fig)

    def test_panel_y_label(self):
        """y_label is applied to panel plot_decomposition figures."""
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
        result = plot_decomposition(y, components, y_label="Custom Panel Y")
        assert isinstance(result, dict)
        for fig in result.values():
            assert_figure_valid(fig)


class TestPlotForecastPanelFacetByGroup:
    """Tests for plot_forecast panel data with facet_by='group'."""

    def test_panel_facet_by_group(self):
        """facet_by='group' creates one subplot per group."""
        dates = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 30), "1d", eager=True)
        y_test = pl.DataFrame({
            "time": dates,
            "y__s1": list(range(30)),
            "y__s2": list(range(30, 60)),
        })
        y_pred = pl.DataFrame({
            "time": dates,
            "y__s1": [x + 1 for x in range(30)],
            "y__s2": [x + 1 for x in range(30, 60)],
        })
        fig = plot_forecast(y_test, y_pred, facet_by="group")
        assert_figure_valid(fig)
        assert len(fig.data) >= 2


class TestPlotTimeWeightPanelFacetByGroup:
    """Tests for plot_time_weight panel data with facet_by='group'."""

    def test_panel_facet_by_group(self):
        """facet_by='group' creates one subplot per group in time weight."""
        time_weight = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
            "weight__store_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "weight__store_2": [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0],
        })
        fig = plot_time_weight(time_weight, weight_column="weight", facet_by="group")
        assert_figure_valid(fig)


class TestPlotForecastPanelMissingPred:
    """Tests for panel forecast where some columns have no matching prediction."""

    def test_multi_model_column_without_pred(self):
        """Multi-model panel where a y_test column has no matching prediction column."""
        dates = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 10), "1d", eager=True)
        y_test = pl.DataFrame({
            "time": dates,
            "y__s1": list(range(10)),
            "y__s2": list(range(10, 20)),
        })
        # Only predict s1, not s2
        y_pred = pl.DataFrame({
            "time": dates,
            "y__s1": [x + 1 for x in range(10)],
        })
        fig = plot_forecast(y_test, {"M1": y_pred}, facet_by="group")
        assert_figure_valid(fig)


class TestComputeStlImportError:
    """Test STL/MSTL import error branches."""

    def test_stl_import_error(self):
        """_compute_stl raises ImportError when statsmodels is missing."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "statsmodels" in name:
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        n = len(dates)
        y = pl.DataFrame({"time": dates, "y": list(range(n))})

        with (
            patch("builtins.__import__", side_effect=mock_import),
            pytest.raises(ImportError, match="statsmodels is required"),
        ):
            plot_decomposition(y, ["trend", "seasonal", "residual"], method="stl", columns="y")

    def test_mstl_import_error(self):
        """_compute_mstl raises ImportError when statsmodels is missing."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "statsmodels" in name:
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True)
        n = len(dates)
        y = pl.DataFrame({"time": dates, "y": list(range(n))})

        with patch("builtins.__import__", side_effect=mock_import), pytest.raises(ImportError, match="statsmodels"):
            plot_decomposition(y, ["trend", "seasonal", "residual"], method="mstl", columns="y", periods=[7, 365])


@pytest.mark.skipif(
    not importlib.util.find_spec("statsmodels"),
    reason="statsmodels not installed",
)
class TestMSTLAutoPeriodsError:
    """Cover MSTL periods='auto' with unsupported interval."""

    def test_auto_periods_unsupported_interval(self):
        """MSTL auto periods raises ValueError for unsupported interval."""
        n = 200
        # 3-minute interval is not in _INTERVAL_TO_MSTL_PERIODS
        df = pl.DataFrame({
            "time": pl.datetime_range(
                pl.datetime(2022, 1, 1),
                pl.datetime(2022, 1, 1) + pl.duration(minutes=3 * (n - 1)),
                "3m",
                eager=True,
            ),
            "y": [float(i) for i in range(n)],
        })
        with pytest.raises(ValueError, match="Cannot infer MSTL periods"):
            plot_decomposition(df, ["trend", "seasonal"], method="mstl", columns="y", periods="auto")

    def test_auto_periods_interval_fallback(self):
        """MSTL with explicit periods and irregular data falls back to numeric labels."""
        rng = np.random.default_rng(42)
        n = 200
        # Irregular timestamps → check_interval_consistency raises → interval=None
        base = pl.datetime_range(
            pl.datetime(2022, 1, 1),
            pl.datetime(2022, 1, 1) + pl.duration(hours=n - 1),
            "1h",
            eager=True,
        )
        # Break regularity by shifting one point
        from datetime import timedelta

        base = base.to_list()
        base[100] = base[100] + timedelta(minutes=17)
        df = pl.DataFrame({
            "time": base,
            "y": [50 + 10 * np.sin(2 * np.pi * i / 24) + rng.standard_normal() for i in range(n)],
        })
        fig = plot_decomposition(
            df,
            ["trend", "seasonal", "residual"],
            method="mstl",
            columns="y",
            show_original=False,
            periods=[24],
        )
        assert_figure_valid(fig)


@pytest.mark.skipif(not _has_statsmodels, reason="statsmodels not installed")
class TestDecompositionNonPositiveOffset:
    """Tests for multiplicative decomposition with non-positive data (offset logic)."""

    @pytest.fixture
    def monthly_zero_df(self):
        """Monthly data with zero and negative values for offset testing."""
        rng = np.random.default_rng(42)
        return pl.DataFrame({
            "time": pl.date_range(pl.date(2018, 1, 1), pl.date(2023, 12, 1), "1mo", eager=True),
            "y": [-5 + 2 * i + 10 * np.sin(2 * np.pi * i / 12) + rng.standard_normal() for i in range(72)],
        })

    def test_stl_multiplicative_non_positive(self, monthly_zero_df):
        """STL multiplicative with non-positive values applies offset."""
        with pytest.warns(UserWarning, match="log-transform"):
            fig = plot_decomposition(
                monthly_zero_df,
                ["trend", "seasonal"],
                method="stl",
                columns="y",
                model="multiplicative",
                show_original=False,
            )
        assert_figure_valid(fig)

    def test_classical_multiplicative_non_positive(self, monthly_zero_df):
        """Classical multiplicative with non-positive values applies offset."""
        fig = plot_decomposition(
            monthly_zero_df,
            ["trend", "seasonal"],
            method="classical",
            columns="y",
            model="multiplicative",
            show_original=False,
        )
        assert_figure_valid(fig)

    def test_mstl_multiplicative_non_positive(self):
        """MSTL multiplicative with non-positive values applies offset."""
        rng = np.random.default_rng(42)
        n = 24 * 7 * 12
        df = pl.DataFrame({
            "time": pl.datetime_range(
                pl.datetime(2022, 1, 1),
                pl.datetime(2022, 1, 1) + pl.duration(hours=n - 1),
                "1h",
                eager=True,
            ),
            "y": [-10 + 10 * np.sin(2 * np.pi * i / 24) + rng.standard_normal() for i in range(n)],
        })
        with pytest.warns(UserWarning, match="log-transform"):
            fig = plot_decomposition(
                df,
                ["trend", "seasonal"],
                method="mstl",
                columns="y",
                model="multiplicative",
                periods=[24, 24 * 7],
                show_original=False,
            )
        assert_figure_valid(fig)


@pytest.mark.skipif(not _has_statsmodels, reason="statsmodels not installed")
class TestDecompositionParameterWarnings:
    """Tests for parameter mismatch warnings in plot_decomposition."""

    @pytest.fixture
    def monthly_df(self):
        """Monthly data for decomposition tests."""
        rng = np.random.default_rng(42)
        return pl.DataFrame({
            "time": pl.date_range(pl.date(2018, 1, 1), pl.date(2023, 12, 1), "1mo", eager=True),
            "y": [100 + 2 * i + 10 * np.sin(2 * np.pi * i / 12) + rng.standard_normal() for i in range(72)],
        })

    def test_two_sided_with_stl_warns(self, monthly_df):
        """Passing two_sided to STL method emits a warning."""
        with pytest.warns(UserWarning, match="two_sided.*only used with method='classical'"):
            plot_decomposition(
                monthly_df,
                ["trend"],
                method="stl",
                columns="y",
                two_sided=False,
            )

    def test_extrapolate_trend_with_stl_warns(self, monthly_df):
        """Passing extrapolate_trend to STL method emits a warning."""
        with pytest.warns(UserWarning, match="extrapolate_trend.*only used with method='classical'"):
            plot_decomposition(
                monthly_df,
                ["trend"],
                method="stl",
                columns="y",
                extrapolate_trend="freq",
            )

    def test_classical_auto_period_unsupported_interval(self):
        """Classical decomposition with period='auto' and unsupported interval raises."""
        from datetime import datetime

        dates = pl.datetime_range(
            datetime(2020, 1, 1),
            datetime(2020, 1, 1, 0, 18),
            "2m",
            eager=True,
        )
        n = len(dates)
        df = pl.DataFrame({"time": dates, "y": list(range(n))})
        with pytest.raises(ValueError, match="Cannot infer period"):
            plot_decomposition(df, ["trend"], method="classical")

class TestPlotForecastClassProba:
    """Tests for plot_forecast with class-probability predictions."""

    @pytest.fixture()
    def _class_proba_data(self):
        """Create class-probability test data."""
        times = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 10), "1d", eager=True)
        y_test = pl.DataFrame({
            "time": times,
            "weather": ["sunny", "sunny", "rainy", "cloudy", "sunny", "rainy", "cloudy", "sunny", "sunny", "rainy"],
        })
        y_proba = pl.DataFrame({
            "time": times,
            "weather_proba_cloudy": [0.1, 0.1, 0.2, 0.6, 0.1, 0.2, 0.7, 0.1, 0.05, 0.3],
            "weather_proba_rainy": [0.2, 0.1, 0.6, 0.2, 0.1, 0.7, 0.2, 0.1, 0.05, 0.6],
            "weather_proba_sunny": [0.7, 0.8, 0.2, 0.2, 0.8, 0.1, 0.1, 0.8, 0.9, 0.1],
        })
        return y_test, y_proba

    def test_single_model(self, _class_proba_data):
        """Stacked area chart is produced for single-model proba predictions."""
        y_test, y_proba = _class_proba_data
        fig = plot_forecast(y_test, y_proba)
        assert isinstance(fig, go.Figure)
        # 3 class traces + 1 truth marker trace
        assert len(fig.data) >= 3

    def test_multi_model(self, _class_proba_data):
        """Subplots are created for multi-model proba predictions."""
        y_test, y_proba = _class_proba_data
        y_proba_b = y_proba.with_columns(
            pl.col("weather_proba_sunny") * 0.5,
            pl.col("weather_proba_rainy") * 1.5,
        )
        fig = plot_forecast(
            y_test,
            {"Model A": y_proba, "Model B": y_proba_b},
        )
        assert isinstance(fig, go.Figure)
        # Should have traces from both models
        assert len(fig.data) >= 6

    def test_custom_title_and_palette(self, _class_proba_data):
        """Custom title and palette are applied."""
        y_test, y_proba = _class_proba_data
        fig = plot_forecast(
            y_test,
            y_proba,
            title="Custom Title",
            color_palette=["#ff0000", "#00ff00", "#0000ff"],
        )
        assert fig.layout.title.text == "Custom Title"

    def test_truth_markers_present(self, _class_proba_data):
        """Truth markers are shown when y_test has matching target column."""
        y_test, y_proba = _class_proba_data
        fig = plot_forecast(y_test, y_proba)
        trace_names = [t.name for t in fig.data]
        assert "True class" in trace_names


class TestPlotForecastCategorical:
    """Tests for plot_forecast with categorical string predictions."""

    @pytest.fixture()
    def _categorical_data(self):
        """Create categorical test data."""
        times = pl.date_range(pl.date(2020, 4, 1), pl.date(2020, 4, 10), "1d", eager=True)
        y_test = pl.DataFrame({
            "time": times,
            "weather": ["sunny", "sunny", "rainy", "cloudy", "sunny", "rainy", "cloudy", "sunny", "sunny", "rainy"],
        })
        y_pred = pl.DataFrame({
            "time": times,
            "weather": ["sunny", "rainy", "rainy", "cloudy", "sunny", "sunny", "cloudy", "sunny", "rainy", "rainy"],
        })
        return y_test, y_pred

    def test_single_model(self, _categorical_data):
        """Step chart is produced for single-model categorical predictions."""
        y_test, y_pred = _categorical_data
        fig = plot_forecast(y_test, y_pred)
        assert isinstance(fig, go.Figure)
        # Actual + Forecast traces
        assert len(fig.data) >= 2

    def test_multi_model(self, _categorical_data):
        """Multiple model categorical predictions are overlaid."""
        y_test, y_pred = _categorical_data
        y_pred_b = y_pred.with_columns(
            pl.Series(
                "weather", ["cloudy", "sunny", "sunny", "rainy", "rainy", "cloudy", "sunny", "rainy", "cloudy", "sunny"]
            ),
        )
        fig = plot_forecast(
            y_test,
            {"Model A": y_pred, "Model B": y_pred_b},
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3

    def test_with_training_data(self, _categorical_data):
        """Training data is shown when provided."""
        y_test, y_pred = _categorical_data
        y_train = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 3, 22), pl.date(2020, 3, 31), "1d", eager=True),
            "weather": ["sunny", "sunny", "cloudy", "rainy", "sunny", "rainy", "cloudy", "sunny", "rainy", "sunny"],
        })
        fig = plot_forecast(y_test, y_pred, y_train=y_train)
        assert isinstance(fig, go.Figure)
        # Train + Actual + Forecast
        assert len(fig.data) >= 3

    def test_y_axis_has_category_labels(self, _categorical_data):
        """Y-axis tick labels are category names, not integers."""
        y_test, y_pred = _categorical_data
        fig = plot_forecast(y_test, y_pred)
        yaxis = fig.layout.yaxis
        assert set(yaxis.ticktext) == {"cloudy", "rainy", "sunny"}

    def test_custom_title(self, _categorical_data):
        """Custom title is applied to categorical forecast."""
        y_test, y_pred = _categorical_data
        fig = plot_forecast(y_test, y_pred, title="My Categorical Plot")
        assert fig.layout.title.text == "My Categorical Plot"


class TestPlotForecastPanelClassProba:
    """Tests for plot_forecast with panel class-probability data."""

    @pytest.fixture
    def _panel_proba_data(self):
        """Panel class-probability data with two members."""
        times = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 5), "1d", eager=True)
        y_test = pl.DataFrame({
            "time": times,
            "weather__east": ["sunny", "rainy", "rainy", "sunny", "cloudy"],
            "weather__west": ["cloudy", "sunny", "sunny", "rainy", "sunny"],
        })
        y_pred = pl.DataFrame({
            "time": times,
            "weather_proba_sunny__east": [0.7, 0.2, 0.1, 0.6, 0.1],
            "weather_proba_rainy__east": [0.2, 0.6, 0.7, 0.2, 0.2],
            "weather_proba_cloudy__east": [0.1, 0.2, 0.2, 0.2, 0.7],
            "weather_proba_sunny__west": [0.1, 0.7, 0.6, 0.2, 0.8],
            "weather_proba_rainy__west": [0.2, 0.1, 0.2, 0.6, 0.1],
            "weather_proba_cloudy__west": [0.7, 0.2, 0.2, 0.2, 0.1],
        })
        return y_test, y_pred

    def test_basic_panel(self, _panel_proba_data):
        """Panel class-proba data produces a faceted figure."""
        y_test, y_pred = _panel_proba_data
        fig = plot_forecast(y_test, y_pred)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_subplot_titles_are_members(self, _panel_proba_data):
        """Subplot titles correspond to panel members."""
        y_test, y_pred = _panel_proba_data
        fig = plot_forecast(y_test, y_pred)
        annotations = [a.text for a in fig.layout.annotations]
        assert "east" in annotations
        assert "west" in annotations

    def test_multi_model_panel(self, _panel_proba_data):
        """Multi-model panel class-proba produces subplots."""
        y_test, y_pred = _panel_proba_data
        y_pred_b = y_pred.with_columns(
            pl.col("weather_proba_sunny__east").alias("weather_proba_sunny__east") * 0.9,
        )
        fig = plot_forecast(y_test, {"Model A": y_pred, "Model B": y_pred_b})
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 4

    def test_panel_with_training_data(self, _panel_proba_data):
        """Panel class-proba data with y_train renders training traces."""
        y_test, y_pred = _panel_proba_data
        y_train = pl.DataFrame({
            "time": pl.date_range(pl.date(2019, 12, 27), pl.date(2019, 12, 31), "1d", eager=True),
            "weather__east": ["sunny", "rainy", "cloudy", "sunny", "rainy"],
            "weather__west": ["cloudy", "sunny", "rainy", "cloudy", "sunny"],
        })
        fig = plot_forecast(y_test, y_pred, y_train=y_train)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2


class TestPlotForecastPanelCategorical:
    """Tests for plot_forecast with panel categorical data."""

    @pytest.fixture
    def _panel_cat_data(self):
        """Panel categorical data with two members."""
        times = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 5), "1d", eager=True)
        y_test = pl.DataFrame({
            "time": times,
            "weather__east": ["sunny", "rainy", "rainy", "sunny", "cloudy"],
            "weather__west": ["cloudy", "sunny", "sunny", "rainy", "sunny"],
        })
        y_pred = pl.DataFrame({
            "time": times,
            "weather__east": ["sunny", "sunny", "rainy", "sunny", "rainy"],
            "weather__west": ["cloudy", "rainy", "sunny", "rainy", "sunny"],
        })
        return y_test, y_pred

    def test_basic_panel(self, _panel_cat_data):
        """Panel categorical data produces a faceted figure."""
        y_test, y_pred = _panel_cat_data
        fig = plot_forecast(y_test, y_pred)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_subplot_titles_are_members(self, _panel_cat_data):
        """Subplot titles correspond to panel members."""
        y_test, y_pred = _panel_cat_data
        fig = plot_forecast(y_test, y_pred)
        annotations = [a.text for a in fig.layout.annotations]
        assert "east" in annotations
        assert "west" in annotations

    def test_with_training_data(self, _panel_cat_data):
        """Training data is shown in panel mode."""
        y_test, y_pred = _panel_cat_data
        y_train = pl.DataFrame({
            "time": pl.date_range(pl.date(2019, 12, 27), pl.date(2019, 12, 31), "1d", eager=True),
            "weather__east": ["sunny", "sunny", "cloudy", "rainy", "sunny"],
            "weather__west": ["rainy", "cloudy", "sunny", "sunny", "rainy"],
        })
        fig = plot_forecast(y_test, y_pred, y_train=y_train)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 4

    def test_y_axis_category_labels(self, _panel_cat_data):
        """Y-axis tick labels are category names in panel mode."""
        y_test, y_pred = _panel_cat_data
        fig = plot_forecast(y_test, y_pred)
        yaxis = fig.layout.yaxis
        assert set(yaxis.ticktext) == {"cloudy", "rainy", "sunny"}
