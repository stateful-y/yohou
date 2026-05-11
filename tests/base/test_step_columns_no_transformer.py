"""Tests for step column injection when transformers produce no output.

Covers the code paths in standard.py and panel.py where _X_t_observed is None
but step columns exist (e.g. SeasonalNaive fitted with X_future/X_forecast).
"""

from datetime import datetime, timedelta

import polars as pl

from yohou.point import SeasonalNaive


def _make_data(length=50, forecasting_horizon=3):
    """Build standard y, X_future, X_forecast for step column tests."""
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
        interval="1s",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": list(range(length))})

    time_ext = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=length + forecasting_horizon - 1),
        interval="1s",
        eager=True,
    )
    X_future = pl.DataFrame({
        "time": time_ext,
        "temp": [float(i % 10) for i in range(len(time_ext))],
    })

    vintages = []
    for t_idx in range(length):
        vt = time[t_idx]
        for step in range(1, forecasting_horizon + 1):
            ft = vt + timedelta(seconds=step)
            vintages.append({"vintage_time": vt, "time": ft, "rain": float(t_idx + step)})
    X_forecast = pl.DataFrame(vintages)

    return y, X_future, X_forecast


def _make_panel_data(length=50, forecasting_horizon=3):
    """Build panel y and X_future for step column tests."""
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
        interval="1s",
        eager=True,
    )
    y = pl.DataFrame({
        "time": time,
        "grp_a__value": list(range(length)),
        "grp_b__value": list(range(10, length + 10)),
    })

    time_ext = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=length + forecasting_horizon - 1),
        interval="1s",
        eager=True,
    )
    X_future = pl.DataFrame({
        "time": time_ext,
        "grp_a__temp": [float(i % 7) for i in range(len(time_ext))],
        "grp_b__temp": [float(i % 5) for i in range(len(time_ext))],
    })

    return y, X_future


class TestStandardStepColumnsNoTransformer:
    """Step columns with a forecaster that has no transformer output (target_as_feature=None)."""

    def test_observe_re_derives_step_columns(self):
        """observe() re-derives step columns when _X_t_observed was None."""
        y, X_future, X_forecast = _make_data(length=50, forecasting_horizon=3)
        f = SeasonalNaive(seasonality=3)
        f.fit(y[:40], forecasting_horizon=3, X_future=X_future, X_forecast=X_forecast)

        f.observe(y[40:42], X_future=X_future, X_forecast=X_forecast)
        y_pred = f.predict(forecasting_horizon=3)

        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) == 3

    def test_rewind_re_derives_step_columns(self):
        """rewind() re-derives step columns when _X_t_observed was None."""
        y, X_future, X_forecast = _make_data(length=50, forecasting_horizon=3)
        f = SeasonalNaive(seasonality=3)
        f.fit(y[:40], forecasting_horizon=3, X_future=X_future, X_forecast=X_forecast)

        f.observe(y[40:42], X_future=X_future, X_forecast=X_forecast)
        f.rewind(y[:40], X_future=X_future, X_forecast=X_forecast)
        y_pred = f.predict(forecasting_horizon=3)

        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) == 3

    def test_observe_predict_loop_with_step_columns(self):
        """observe_predict uses precomputed step columns when transformers yield None."""
        y, X_future, _ = _make_data(length=50, forecasting_horizon=3)
        f = SeasonalNaive(seasonality=3)
        f.fit(y[:40], forecasting_horizon=3, X_future=X_future)

        y_pred = f.observe_predict(y[40:46], stride=3, X_future=X_future)

        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) >= 3


class TestPanelStepColumnsNoTransformer:
    """Panel step columns with a forecaster that has no transformer output."""

    def test_panel_fit_with_step_columns_no_X_t(self):
        """Panel fit with X_future when transformers produce no X_t."""
        y, X_future = _make_panel_data(length=50, forecasting_horizon=3)
        f = SeasonalNaive(seasonality=3)
        f.fit(y[:40], forecasting_horizon=3, X_future=X_future)

        y_pred = f.predict(forecasting_horizon=3)
        assert isinstance(y_pred, pl.DataFrame)
        assert "grp_a__value" in y_pred.columns
        assert len(y_pred) == 3

    def test_panel_observe_with_step_columns_no_X_t(self):
        """Panel observe re-derives step columns when _X_t_observed[group] was None."""
        y, X_future = _make_panel_data(length=50, forecasting_horizon=3)
        f = SeasonalNaive(seasonality=3)
        f.fit(y[:40], forecasting_horizon=3, X_future=X_future)

        f.observe(y[40:42], X_future=X_future)
        y_pred = f.predict(forecasting_horizon=3)
        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) == 3

    def test_panel_observe_predict_with_step_columns_no_X_t(self):
        """Panel observe_predict with precomputed step columns and no transformer output."""
        y, X_future = _make_panel_data(length=50, forecasting_horizon=3)
        f = SeasonalNaive(seasonality=3)
        f.fit(y[:40], forecasting_horizon=3, X_future=X_future)

        y_pred = f.observe_predict(y[40:46], stride=3, X_future=X_future)
        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) >= 3
