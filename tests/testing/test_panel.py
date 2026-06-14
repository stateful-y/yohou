"""Tests for yohou.testing.panel check functions."""

from yohou.point.naive import SeasonalNaive
from yohou.testing.panel import (
    check_panel_data,
    check_panel_invalid_group_raises,
    check_panel_single_group,
)


class TestPanelChecks:
    """Tests for panel data check functions."""

    def test_check_panel_data(self, y_X_factory):
        """Test check_panel_data passes for valid panel forecaster."""
        # Create panel data
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42, panel=True)

        forecaster = SeasonalNaive(seasonality=12)
        forecaster.fit(y[:40], X[:40], forecasting_horizon=3)

        # Should not raise
        check_panel_data(forecaster, y[:40])

    def test_check_panel_data_non_panel(self, y_X_factory):
        """Test check handles non-panel data correctly."""
        # Create non-panel data (global only)
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42, panel=False)

        forecaster = SeasonalNaive(seasonality=12)
        forecaster.fit(y[:40], X[:40], forecasting_horizon=3)

        # Should not raise - check handles non-panel case
        check_panel_data(forecaster, y[:40])

    def test_check_panel_single_group(self, y_X_factory):
        """Test check_panel_single_group passes for valid panel forecaster."""
        # Create panel data
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42, panel=True)

        forecaster = SeasonalNaive(seasonality=12)
        forecaster.fit(y[:40], X[:40], forecasting_horizon=3)

        # Should not raise
        check_panel_single_group(forecaster, y[:40])

    def test_check_panel_invalid_group_raises(self, y_X_factory):
        """Test check_panel_invalid_group_raises passes for valid panel forecaster."""
        # Create panel data
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42, panel=True)

        forecaster = SeasonalNaive(seasonality=12)
        forecaster.fit(y[:40], X[:40], forecasting_horizon=3)

        # Should not raise
        check_panel_invalid_group_raises(forecaster, y[:40])
