"""Tests for panel data boundary conditions.

Step 28 of codebase quality plan: verify forecasters, transformers, and
utilities handle degenerate panel data setups (single group, mixed locality,
empty group names, etc.).
"""

from __future__ import annotations

import polars as pl

from yohou.point.naive import SeasonalNaive
from yohou.utils.panel import get_group_df, inspect_panel


class TestInspectPanelBoundaries:
    """Boundary conditions for inspect_panel utility."""

    def test_single_panel_group(self, y_X_panel_factory):
        """A single panel group should produce exactly one group entry."""
        y, _ = y_X_panel_factory(n_groups=1, length=50, n_targets=2, n_features=0)
        global_cols, panels = inspect_panel(y)
        assert len(panels) == 1, f"Expected 1 group, got {len(panels)}"
        assert len(global_cols) == 0

    def test_many_panel_groups(self, y_X_panel_factory):
        """Many groups should all be detected."""
        y, _ = y_X_panel_factory(n_groups=10, length=50, n_targets=1, n_features=0)
        _, panels = inspect_panel(y)
        assert len(panels) == 10, f"Expected 10 groups, got {len(panels)}"

    def test_no_panel_columns(self, y_X_factory):
        """Non-panel data should return empty panel dict."""
        y, _ = y_X_factory(length=50, n_targets=2, n_features=0, panel=False)
        global_cols, panels = inspect_panel(y)
        assert len(panels) == 0
        assert len(global_cols) == 2  # y_0, y_1

    def test_mixed_global_and_panel_columns(self, panel_time_series_factory):
        """Mixed global + panel columns should be separated correctly."""
        df = panel_time_series_factory(length=50, n_series=2, n_groups=2, n_global=3)
        global_cols, panels = inspect_panel(df)
        assert len(global_cols) == 3, f"Expected 3 global columns, got {len(global_cols)}"
        assert len(panels) == 2, f"Expected 2 panel groups, got {len(panels)}"


class TestGetGroupDfBoundaries:
    """Boundary conditions for get_group_df utility."""

    def test_extract_single_group(self, y_X_panel_factory):
        """Extracting a single group should produce unprefixed columns."""
        y, _ = y_X_panel_factory(n_groups=3, length=50, n_targets=1, n_features=0)
        # get_group_df expects a schema with unprefixed column names
        unprefixed_schema = {"y_0": pl.Float64}
        group_df = get_group_df(y, "group_0", unprefixed_schema)
        # Should have time + unprefixed target columns
        assert "time" in group_df.columns
        assert all("__" not in col for col in group_df.columns)

    def test_extract_all_groups_same_length(self, y_X_panel_factory):
        """Each extracted group should have the same length as the original."""
        y, _ = y_X_panel_factory(n_groups=3, length=50, n_targets=1, n_features=0)
        unprefixed_schema = {"y_0": pl.Float64}
        for i in range(3):
            group_df = get_group_df(y, f"group_{i}", unprefixed_schema)
            assert len(group_df) == 50


class TestPanelForecasterBoundaries:
    """Panel forecasters with boundary panel configurations."""

    def test_single_group_fit_predict(self, y_X_factory):
        """Forecaster should work with a single panel group."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, panel=True, n_groups=1)
        forecaster = SeasonalNaive(seasonality=1)
        forecaster.fit(y[:80], forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)
        assert len(y_pred) == 5

    def test_single_group_observe_predict(self, y_X_factory):
        """observe_predict should work with a single panel group."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, panel=True, n_groups=1)
        forecaster = SeasonalNaive(seasonality=1)
        forecaster.fit(y[:80], forecasting_horizon=5)
        y_pred = forecaster.observe_predict(y[80:85])
        # observe_predict returns forecasting_horizon predictions
        assert len(y_pred) >= 5

    def test_predict_subset_of_groups(self, y_X_factory):
        """Predicting a subset of groups should only return that subset's columns."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, panel=True, n_groups=3)
        forecaster = SeasonalNaive(seasonality=5)
        forecaster.fit(y[:80], forecasting_horizon=5)
        y_pred = forecaster.predict(
            forecasting_horizon=5,
            panel_group_names=["group_0"],
        )
        # Should only have time + observed_time + group_0 columns
        non_time_cols = [c for c in y_pred.columns if c not in ("time", "observed_time")]
        assert all(c.startswith("group_0__") for c in non_time_cols), (
            f"Expected only group_0 columns, got {non_time_cols}"
        )
