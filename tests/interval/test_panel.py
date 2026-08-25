"""Tests for cross-learning functionality in interval forecasters."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone

from conftest import run_checks
from yohou.interval import IntervalReductionForecaster, SplitConformalForecaster
from yohou.point import SeasonalNaive
from yohou.testing import _yield_yohou_forecaster_checks


class TestIntervalReductionPanelChecks:
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "forecaster,tags,expected_failures",
        [
            (
                IntervalReductionForecaster(),
                {"forecaster_type": frozenset({"interval"}), "uses_reduction": True, "supports_panel_data": True},
                [],
            ),
        ],
    )
    def test_interval_reduction_panel_checks(self, forecaster, tags, expected_failures, panel_time_series_factory):
        """Run systematic cross-learning checks on IntervalReductionForecaster with panel data."""
        y = panel_time_series_factory(length=100, n_series=3)
        y_train, y_test = y[:80], y[80:]

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_actual=None, forecasting_horizon=3, coverage_rates=[0.1, 0.5, 0.9])

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(forecaster_fitted, y_train, None, y_test, None),
            expected_failures=set(expected_failures),
        )


class TestIntervalReductionPanelBehavior:
    def test_panel_interval_global_data(self, time_series_factory):
        """A global-fitted interval forecaster predicts with ``groups=None`` and rejects groups.

        The default ``groups=None`` path returns the flat interval columns, and
        passing any explicit ``groups`` on a globally-fitted forecaster must
        raise rather than silently ignore the argument.
        """
        y = time_series_factory(length=50, n_components=1)
        y_train = y[:40]

        forecaster = IntervalReductionForecaster()

        forecaster.fit(y=y_train, X_actual=None, forecasting_horizon=3, coverage_rates=[0.1, 0.5, 0.9])

        y_pred = forecaster.predict_interval(forecasting_horizon=3, groups=None)
        assert "feature_0_lower_0.1" in y_pred.columns
        assert len(y_pred) == 3

        with pytest.raises(ValueError, match="fitted on global data"):
            forecaster.predict_interval(forecasting_horizon=3, groups=["feature_0"])


class TestSplitConformalPanelChecks:
    """Systematic checks for SplitConformalForecaster on panel data.

    The existing systematic run for this class uses ``n_targets=1``, so until
    now no generated check had ever seen it hold more than one value column.
    That is why a pooled calibration quantile survived: every green check was
    measured on data that could not expose it.
    """

    @pytest.mark.slow
    def test_split_conformal_panel_checks(self, panel_time_series_factory):
        # Sized so the default 0.95 rate is expressible at every horizon step:
        # an asymmetric scorer needs 39 scores there, and step 3 sees
        # calibration_size - 2. At the previous 30 the sweep ran in a regime the
        # forecaster itself warns about, which is not what these checks are for.
        y = panel_time_series_factory(length=220, n_series=3)
        y_train, y_test = y[:180], y[180:]

        forecaster = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
        )
        forecaster.fit(y_train, forecasting_horizon=3)

        run_checks(
            forecaster,
            _yield_yohou_forecaster_checks(forecaster, y_train, None, y_test, None),
            expected_failures=set(),
        )


class TestMultiVariablePanelCalibration:
    """Calibration is keyed by value column, which is finer than by group."""

    def test_four_columns_get_four_independent_calibrations(self):
        """Two groups times two variables is four calibrations, not two."""
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(160)]

        def series(scale: float, seed: int) -> list[float]:
            rng = np.random.default_rng(seed)
            return [scale * (10.0 + 5.0 * np.sin(2 * np.pi * i / 7)) + rng.normal(0, 0.5 * scale) for i in range(160)]

        y = pl.DataFrame({
            "time": dates,
            "a__sales": series(1.0, 1),
            "a__returns": series(10.0, 2),
            "b__sales": series(100.0, 3),
            "b__returns": series(1000.0, 4),
        })
        columns = ["a__sales", "a__returns", "b__sales", "b__returns"]

        forecaster = SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=7), calibration_size=40).fit(
            y[:140], forecasting_horizon=1, coverage_rates=[0.9]
        )

        intervals = forecaster.predict_interval(forecasting_horizon=1, coverage_rates=[0.9])
        widths = {c: float(intervals[f"{c}_upper_0.9"][0] - intervals[f"{c}_lower_0.9"][0]) for c in columns}

        # Each width tracks its own column's scale, so consecutive columns
        # differ by roughly the 10x scale step between them.
        for finer, coarser in zip(columns, columns[1:], strict=False):
            ratio = widths[coarser] / widths[finer]
            assert 3.0 < ratio < 30.0, f"{coarser} should be about 10x {finer}, got {widths}"

        assert len(set(widths.values())) == 4, f"expected four distinct calibrations, got {widths}"
