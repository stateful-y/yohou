"""End-to-end tests for heterogeneous-cadence X_forecast resolution.

A single X_forecast frame may carry channels issued on different schedules. Each
base column resolves against its own newest applicable vintage, at fit and at
serve, densified before step-column derivation so the frame-wide as-of in
windowing still serves the whole row. These tests exercise the behaviour through
real forecasters (with and without a forecast_transformer) and pin the
coherence, backward-compatibility, and leakage contracts.
"""

import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.dummy import DummyRegressor

from yohou.base.utils import _densify_forecast_vintages
from yohou.point import PointReductionForecaster

H = 3


def _d(day: int) -> datetime:
    return datetime(2021, 1, 1) + timedelta(days=day)


def _y(days: range) -> pl.DataFrame:
    return pl.DataFrame({"time": [_d(k) for k in days], "value": np.asarray([float(k) for k in days])})


def _ignore_warnings():
    warnings.simplefilter("ignore")


class TestPerColumnResolution:
    """A slower channel resolves against its own newest vintage."""

    @staticmethod
    def _mixed_forecast() -> pl.DataFrame:
        """``fast`` issued every day 0..9; ``slow`` only at day 6.

        The day-6 slow vintage covers days 7, 8, 9, so it is still current for
        an observation at day 8, even though newer fast-only vintages exist.
        """
        rows = []
        for v in range(10):
            for step in range(1, H + 1):
                rows.append({
                    "vintage_time": _d(v),
                    "time": _d(v + step),
                    "fast": float(v + step),
                    "slow": float(100 + v + step) if v == 6 else None,
                })
        return pl.DataFrame(
            rows,
            schema={
                "vintage_time": pl.Datetime,
                "time": pl.Datetime,
                "fast": pl.Float64,
                "slow": pl.Float64,
            },
        )

    def test_slow_channel_contributes_at_fit(self):
        """Training rows near the slow vintage carry non-null slow step columns."""
        forecaster = PointReductionForecaster(estimator=DummyRegressor(), nan_handling="pass")
        with warnings.catch_warnings():
            _ignore_warnings()
            forecaster.fit(y=_y(range(10)), forecasting_horizon=H, X_forecast=self._mixed_forecast())

        # The transformed cache is dense: the slow channel is populated for the
        # vintages its day-6 issue still covers, not collapsed to null.
        dense = forecaster._X_forecast_t_
        covered = dense.filter(pl.col("slow").is_not_null())
        assert covered.height > 0, "the slow channel must survive densification at fit"

    def test_slow_channel_contributes_at_serve(self):
        """Predicting at day 8 fills the slow step columns from the day-6 vintage."""
        forecaster = PointReductionForecaster(estimator=DummyRegressor(), nan_handling="pass")
        mixed = self._mixed_forecast()
        with warnings.catch_warnings():
            _ignore_warnings()
            forecaster.fit(y=_y(range(7)), forecasting_horizon=H, X_forecast=mixed)
            # Observe up to day 8 with the mixed frame available.
            forecaster.observe(y=_y(range(7, 9)), X_forecast=mixed)
            pred = forecaster.predict(X_forecast=mixed)

        assert pred.height > 0
        # The prediction ran without raising; the dense cache carried slow.
        assert forecaster._X_forecast_t_.filter(pl.col("slow").is_not_null()).height > 0


class TestNoTransformerResolution:
    """Densification runs with no forecast_transformer, on standard and panel.

    This is the case the first review found unguarded: a plain forecaster has no
    forecast_transformer, so a resolution wired only into the transformer path
    would skip it and collapse mixed cadence.
    """

    @staticmethod
    def _standard_mixed() -> pl.DataFrame:
        rows = []
        for v in range(10):
            for step in range(1, H + 1):
                rows.append({
                    "vintage_time": _d(v),
                    "time": _d(v + step),
                    "fast": float(v + step),
                    "slow": float(100 + step) if v == 6 else None,
                })
        return pl.DataFrame(
            rows,
            schema={
                "vintage_time": pl.Datetime,
                "time": pl.Datetime,
                "fast": pl.Float64,
                "slow": pl.Float64,
            },
        )

    def test_standard_no_transformer_resolves_slow(self):
        """No forecast_transformer set: slow still resolves per column."""
        forecaster = PointReductionForecaster(estimator=DummyRegressor(), nan_handling="pass")
        assert forecaster.forecast_transformer is None
        with warnings.catch_warnings():
            _ignore_warnings()
            forecaster.fit(y=_y(range(10)), forecasting_horizon=H, X_forecast=self._standard_mixed())
        assert forecaster._X_forecast_t_.filter(pl.col("slow").is_not_null()).height > 0

    def test_panel_no_transformer_resolves_slow(self):
        """Panel dispatch with no transformer densifies per group__column."""
        forecaster = PointReductionForecaster(estimator=DummyRegressor(), nan_handling="pass")
        # Two groups; y carries group__value columns.
        y = pl.DataFrame({
            "time": [_d(k) for k in range(10)],
            "g1__value": np.arange(10, dtype=float),
            "g2__value": np.arange(10, dtype=float) + 50,
        })
        rows = []
        for v in range(10):
            for step in range(1, H + 1):
                rows.append({
                    "vintage_time": _d(v),
                    "time": _d(v + step),
                    "g1__fast": float(v + step),
                    "g2__fast": float(v + step),
                    "g1__slow": float(step) if v == 6 else None,
                    "g2__slow": float(step) if v == 6 else None,
                })
        schema = {"vintage_time": pl.Datetime, "time": pl.Datetime}
        schema.update(dict.fromkeys(["g1__fast", "g2__fast", "g1__slow", "g2__slow"], pl.Float64))
        X_forecast = pl.DataFrame(rows, schema=schema)
        with warnings.catch_warnings():
            _ignore_warnings()
            forecaster.fit(y=y, forecasting_horizon=H, X_forecast=X_forecast)
        dense = forecaster._X_forecast_t_
        assert dense.filter(pl.col("g1__slow").is_not_null()).height > 0
        assert dense.filter(pl.col("g2__slow").is_not_null()).height > 0


class TestCoherenceAndCompatibility:
    """Vintage coherence, backward compatibility, and the leakage bound."""

    def test_newer_short_range_vintage_not_backfilled_from_older(self):
        """A column's steps come from a single vintage, never spliced.

        ``wind`` is issued long-range at day 0 (covering days 1..5) and short at
        day 1 (covering days 2, 3). For an observation resolving to day 1, step 3
        (day 4) must be null, not backfilled from the day-0 vintage that reaches
        it. Cell-wise filling would splice day 0's value into day 1's trajectory.
        """
        rows = []
        for step in range(1, 6):
            rows.append({"vintage_time": _d(0), "time": _d(step), "wind": 900.0})
        for step in (1, 2):
            rows.append({"vintage_time": _d(1), "time": _d(1 + step), "wind": 910.0})
        frame = pl.DataFrame(rows, schema={"vintage_time": pl.Datetime, "time": pl.Datetime, "wind": pl.Float64})

        dense = _densify_forecast_vintages(frame)
        day1 = dense.filter(pl.col("vintage_time") == _d(1)).sort("time")
        # Day-1 vintage covers days 2 and 3 (from its own issue); day 4 is null,
        # not 900.0 from the day-0 vintage.
        vals = dict(zip(day1["time"].to_list(), day1["wind"].to_list(), strict=True))
        assert vals[_d(2)] == 910.0
        assert vals[_d(3)] == 910.0
        assert _d(4) not in vals or vals[_d(4)] is None

    def test_uniform_cadence_is_unchanged(self):
        """A frame where every vintage carries every column is untouched."""
        rows = []
        for v in range(8):
            for step in range(1, H + 1):
                rows.append({
                    "vintage_time": _d(v),
                    "time": _d(v + step),
                    "a": float(v + step),
                    "b": float(v * 2 + step),
                })
        frame = pl.DataFrame(
            rows,
            schema={
                "vintage_time": pl.Datetime,
                "time": pl.Datetime,
                "a": pl.Float64,
                "b": pl.Float64,
            },
        )
        dense = _densify_forecast_vintages(frame).sort("vintage_time", "time")
        assert dense.equals(frame.sort("vintage_time", "time"))

    def test_resolution_never_crosses_the_observation_time(self):
        """Densification only ever fills from vintages at or before each vintage."""
        # slow issued at day 5 only; a vintage at day 2 must not borrow it.
        rows = []
        for v in (2, 5):
            rows.append({"vintage_time": _d(v), "time": _d(v + 1), "slow": float(v) if v == 5 else None})
        frame = pl.DataFrame(rows, schema={"vintage_time": pl.Datetime, "time": pl.Datetime, "slow": pl.Float64})
        dense = _densify_forecast_vintages(frame)
        day2 = dense.filter(pl.col("vintage_time") == _d(2))
        assert day2["slow"].drop_nulls().len() == 0, "a later vintage must not fill an earlier one"


@pytest.mark.parametrize("with_transformer", [False, True])
def test_mixed_cadence_prediction_runs(with_transformer):
    """A forecaster fit and served on a mixed-cadence frame predicts cleanly."""
    from yohou.compose import PerVintageActualTransformer
    from yohou.preprocessing import StandardScaler

    transformer = None
    if with_transformer:
        transformer = PerVintageActualTransformer(transformer=StandardScaler())

    rows = []
    for v in range(12):
        for step in range(1, H + 1):
            rows.append({
                "vintage_time": _d(v),
                "time": _d(v + step),
                "fast": float(v + step),
                "slow": float(100 + step) if v % 4 == 0 else None,
            })
    mixed = pl.DataFrame(
        rows,
        schema={
            "vintage_time": pl.Datetime,
            "time": pl.Datetime,
            "fast": pl.Float64,
            "slow": pl.Float64,
        },
    )
    forecaster = PointReductionForecaster(
        estimator=DummyRegressor(), nan_handling="pass", forecast_transformer=transformer
    )
    with warnings.catch_warnings():
        _ignore_warnings()
        forecaster.fit(y=_y(range(12)), forecasting_horizon=H, X_forecast=mixed)
        pred = forecaster.predict(X_forecast=mixed)
    assert pred.height > 0
