"""Regression tests for the 2026-06-15 QA remediation of the plotting module.

Each test pins a behavior that was changed while closing a specific QA finding,
so the fix is not silently reverted by future refactors.
"""

import pytest

pytest.importorskip("plotly", reason="plotting extra not installed")

import numpy as np
import polars as pl

from yohou.plotting import (
    plot_calibration,
    plot_cv_results_scatter,
    plot_forecast,
    plot_missing_data,
    plot_outliers,
    plot_phase,
    plot_resampling_comparison,
    plot_rolling_statistics,
    plot_splits,
)
from yohou.plotting._utils import resolve_panel_columns
from yohou.plotting.diagnostics import _compute_ccf
from yohou.plotting.forecasting import _compute_classical


def _series_df(values):
    n = len(values)
    return pl.DataFrame({
        "time": pl.datetime_range(
            pl.datetime(2020, 1, 1),
            pl.datetime(2020, 1, 1) + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        ),
        "y": values,
    })


class TestOutlierColor:
    """outlier-color-param-never-used."""

    def test_outlier_markers_use_outlier_color(self):
        vals = [1.0] * 20 + [100.0] + [1.0] * 5
        df = _series_df(vals)
        fig = plot_outliers(df, columns="y", outlier_color="#00FF00", method="zscore", threshold=2.0)
        marker_colors = [t.marker.color for t in fig.data if t.mode == "markers"]
        assert "#00FF00" in marker_colors


class TestRollingLineOpacity:
    """rolling-stats-nonpanel-hardcoded-opacity."""

    def test_original_series_uses_line_opacity(self):
        df = _series_df([float(i % 7) for i in range(60)])
        fig = plot_rolling_statistics(df, columns="y", line_opacity=0.85, show_original=True)
        opacities = [t.opacity for t in fig.data if t.opacity is not None]
        assert pytest.approx(0.85) in opacities


class TestCcfNormalization:
    """ccf-values-outside-minus1-plus1."""

    def test_ccf_within_unit_interval(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=80)
        y = np.roll(x, 5) + 0.05 * rng.normal(size=80)
        ccf = _compute_ccf(x, y, 30)
        assert len(ccf) == 61
        assert all(-1.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in ccf)

    def test_ccf_zero_lag_is_normalized_correlation(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=100)
        y = 2.0 * x + 1.0  # perfectly correlated
        ccf = _compute_ccf(x, y, 10)
        assert ccf[10] == pytest.approx(1.0, abs=1e-6)


class TestMissingDataMatrixTimeAggregation:
    """matrix-kind-ignores-time-aggregation."""

    def test_matrix_and_heatmap_agree_under_time_aggregation(self):
        dates = pl.datetime_range(
            pl.datetime(2020, 1, 1),
            pl.datetime(2020, 1, 10, 23),
            interval="1h",
            eager=True,
        )
        vals = [None if i % 24 == 0 else float(i) for i in range(len(dates))]
        df = pl.DataFrame({"time": dates, "y": vals})
        fig_heatmap = plot_missing_data(df, kind="heatmap", time_aggregation="1d")
        fig_matrix = plot_missing_data(df, kind="matrix", time_aggregation="1d")
        z_heatmap = list(fig_heatmap.data[0].z)
        z_matrix = list(fig_matrix.data[0].z)
        assert z_matrix == z_heatmap
        # Aggregation collapsed 240 hourly points into 10 daily cells.
        assert len(z_matrix[0]) == 10


class TestPhasePanelColors:
    """plot-phase-panel-colors-indexed-by-entity-idx-not-column-idx."""

    def test_panel_palette_does_not_index_out_of_range(self):
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 4, 9), "1d", eager=True)
        n = len(dates)
        df = pl.DataFrame({
            "time": dates,
            "sales__a": np.sin(np.arange(n) * 0.2).tolist(),
            "sales__b": np.cos(np.arange(n) * 0.2).tolist(),
            "demand__a": np.sin(np.arange(n) * 0.3).tolist(),
            "demand__b": np.cos(np.arange(n) * 0.3).tolist(),
        })
        # Two groups overlaid per member facet; palette of two distinct colors.
        fig = plot_phase(df, groups=[], color_palette=["#111111", "#222222"])
        line_colors = {t.line.color for t in fig.data if t.line.color is not None}
        assert {"#111111", "#222222"}.issubset(line_colors)


class TestResamplingPanelColors:
    """resampling-panel-resolve-color-inside-render-closure."""

    def test_panel_members_get_distinct_colors(self):
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 2, 9), "1d", eager=True)
        n = len(dates)
        original = pl.DataFrame({
            "time": dates,
            "sales__a": list(range(n)),
            "sales__b": list(range(n, 2 * n)),
        })
        resampled = original.group_by_dynamic("time", every="1w").agg(
            pl.col("sales__a").mean(), pl.col("sales__b").mean()
        )
        # facet_by="group" overlays the two members in one facet, so they must
        # receive distinct palette colours (entity_idx based, not a single color).
        fig = plot_resampling_comparison(
            original, resampled, groups=[], facet_by="group", color_palette=["#abcdef", "#fedcba"]
        )
        line_colors = {t.line.color for t in fig.data if t.line.color is not None}
        assert {"#abcdef", "#fedcba"}.issubset(line_colors)


class TestCalibrationClassProbaParams:
    """calibration-class-proba-drops-four-public-params."""

    def test_reference_color_is_forwarded(self):
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 20), "1d", eager=True)
        n = len(dates)
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, n)
        y_pred = pl.DataFrame({
            "time": dates,
            "target_proba_yes": p.tolist(),
            "target_proba_no": (1 - p).tolist(),
        })
        truth = ["yes" if pi > 0.5 else "no" for pi in p]
        y_truth = pl.DataFrame({"time": dates, "target": truth})
        fig = plot_calibration(y_pred, y_truth, target="target", reference_color="#123456")
        ref_colors = [t.line.color for t in fig.data if t.line.color is not None]
        assert "#123456" in ref_colors


class TestCalibrationKdeWarning:
    """kde-failure-silently-omits-trace (distribution shares the helper)."""

    def test_zero_variance_scores_warn(self):
        from yohou.metrics import MeanAbsoluteError
        from yohou.plotting import plot_score_distribution

        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        n = len(dates)
        y_truth = pl.DataFrame({"time": dates, "y": [5.0] * n})
        y_pred = pl.DataFrame({"time": dates, "y": [5.0] * n})
        with pytest.warns(UserWarning, match="KDE"):
            plot_score_distribution(MeanAbsoluteError(), y_truth, y_pred, kind="kde")


class TestForecastVintageTimeExcluded:
    """vintage-time-excluded-from-test-value-cols."""

    def test_vintage_time_not_plotted_as_series(self):
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True)
        n = len(dates)
        y_test = pl.DataFrame({
            "time": dates,
            "vintage_time": dates,
            "y": [float(i) for i in range(n)],
        })
        y_pred = pl.DataFrame({
            "time": dates,
            "vintage_time": dates,
            "y": [float(i) + 0.5 for i in range(n)],
        })
        fig = plot_forecast(y_test, y_pred)
        for trace in fig.data:
            assert "vintage" not in (trace.name or "").lower()


class TestSplitsEmptyError:
    """plot-splits-empty-splits-silent-empty-figure."""

    def test_zero_splits_raises(self):
        from yohou.model_selection import BaseSplitter

        class _EmptySplitter(BaseSplitter):
            def split(self, y, X_actual=None):  # noqa: ARG002
                return iter(())

            def _iter_test_indices(self, y, X_actual=None):  # noqa: ARG002
                return iter(())

            def get_n_splits(self, y=None, X_actual=None):  # noqa: ARG002
                return 0

        y = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 5), "1d", eager=True),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        with pytest.raises(ValueError, match="no splits"):
            plot_splits(y, _EmptySplitter())


class TestCvResultsNoMeanKeys:
    """plot-cv-results-scorer-autodetect-unguarded-index."""

    def test_missing_mean_test_keys_raises_clear_error(self):
        cv_results = {"param_alpha": [0.1, 1.0]}
        with pytest.raises(ValueError, match="mean_test"):
            plot_cv_results_scatter(cv_results, param_name="alpha")


class TestResolvePanelColumnsNoPanelMessage:
    """resolve-panel-columns-error-message-misleading-when-no-panel."""

    def test_no_panel_columns_distinct_message(self):
        df = pl.DataFrame({"time": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="no panel"):
            resolve_panel_columns(df)


class TestClassicalMultiplicativeNonPositive:
    """classical-decomp-multiplicative-offset-mismatch."""

    def test_observed_matches_original_and_adjusted_consistent(self):
        t = np.arange(48)
        vals = (np.sin(2 * np.pi * t / 12) * 5 + 0.1 * t).tolist()  # has negatives
        with pytest.warns(UserWarning, match="log-transform"):
            out = _compute_classical(pl.Series("y", vals), period=12, model="multiplicative")
        observed = np.array(out["observed"])
        seasonal = np.array(out["seasonal"])
        adjusted = np.array(out["seasonal_adjusted"])
        assert np.nanmax(np.abs(observed - np.array(vals))) == pytest.approx(0.0)
        assert np.nanmax(np.abs(adjusted - observed / seasonal)) == pytest.approx(0.0)
