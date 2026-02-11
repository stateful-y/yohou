"""Tests for deterministic reproducibility of stochastic components.

Step 29 of codebase quality plan: verify that identical seeds produce
identical results across forecasters, search, and data factories.
"""

from __future__ import annotations

from yohou.point.naive import SeasonalNaive


class TestFactoryReproducibility:
    """Data generation fixtures should be deterministic given the same seed."""

    def test_y_X_factory_same_seed_identical(self, y_X_factory):
        """Two calls with the same seed produce identical DataFrames."""
        y1, X1 = y_X_factory(length=50, n_targets=2, n_features=3, seed=123)
        y2, X2 = y_X_factory(length=50, n_targets=2, n_features=3, seed=123)
        assert y1.equals(y2), "y should be identical for same seed"
        assert X1.equals(X2), "X should be identical for same seed"

    def test_y_X_factory_different_seed_different(self, y_X_factory):
        """Two calls with different seeds produce different DataFrames."""
        y1, _ = y_X_factory(length=50, n_targets=2, n_features=3, seed=1)
        y2, _ = y_X_factory(length=50, n_targets=2, n_features=3, seed=2)
        assert not y1.equals(y2), "y should differ for different seeds"

    def test_panel_factory_same_seed_identical(self, y_X_panel_factory):
        """Panel factory is deterministic with same seed."""
        y1, X1 = y_X_panel_factory(n_groups=2, length=50, seed=99)
        y2, X2 = y_X_panel_factory(n_groups=2, length=50, seed=99)
        assert y1.equals(y2), "Panel y should be identical for same seed"
        assert X1.equals(X2), "Panel X should be identical for same seed"


class TestForecasterReproducibility:
    """Deterministic forecasters should produce identical results across runs."""

    def test_naive_forecaster_deterministic(self, y_X_factory):
        """SeasonalNaive (deterministic) produces identical predictions."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        f1 = SeasonalNaive(seasonality=5)
        f1.fit(y[:80], forecasting_horizon=5)
        pred1 = f1.predict(forecasting_horizon=5)

        f2 = SeasonalNaive(seasonality=5)
        f2.fit(y[:80], forecasting_horizon=5)
        pred2 = f2.predict(forecasting_horizon=5)

        # Drop time columns for value comparison
        v1 = pred1.drop("time", "observed_time")
        v2 = pred2.drop("time", "observed_time")
        assert v1.equals(v2), "Deterministic forecaster should produce identical predictions"

    def test_observe_predict_deterministic(self, y_X_factory):
        """observe_predict gives same results when called with same data."""
        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

        f1 = SeasonalNaive(seasonality=5)
        f1.fit(y[:80], forecasting_horizon=5)
        pred1 = f1.observe_predict(y[80:85])

        f2 = SeasonalNaive(seasonality=5)
        f2.fit(y[:80], forecasting_horizon=5)
        pred2 = f2.observe_predict(y[80:85])

        v1 = pred1.drop("time", "observed_time")
        v2 = pred2.drop("time", "observed_time")
        assert v1.equals(v2), "observe_predict should be deterministic"
