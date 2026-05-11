"""Integration test for the Tier 1 _fit() hook + _observation_horizon property pattern.

Verifies that a custom forecaster using only ``_observation_horizon``,
``_fit()``, and ``_predict_one()`` can be fitted and used end to end.
"""

from __future__ import annotations

import numbers
from typing import Literal

import numpy as np
import polars as pl
import pytest
from sklearn.utils._param_validation import Interval

from yohou.point.base import BasePointForecaster
from yohou.utils.tags import Tags


class _WindowMeanForecaster(BasePointForecaster):
    """Minimal Tier 1 forecaster that predicts the mean of recent values."""

    _parameter_constraints: dict = {
        **BasePointForecaster._parameter_constraints,
        "window": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        window: int = 5,
        panel_strategy: Literal["global", "multivariate"] = "global",
    ):
        super().__init__(
            feature_transformer=None,
            target_transformer=None,
            target_as_feature=None,
            panel_strategy=panel_strategy,
        )
        self.window = window

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        tags.forecaster_tags.requires_exogenous = False
        tags.forecaster_tags.stateful = True
        return tags

    @property
    def _observation_horizon(self) -> int:
        return self.window

    def _fit(self, y_t, X_t, forecasting_horizon):
        if isinstance(y_t, dict):
            self.means_ = {}
            for group, df in y_t.items():
                vals = df.select(pl.exclude("time")).to_numpy()
                self.means_[group] = np.mean(vals[-self.window :], axis=0)
        else:
            vals = y_t.select(pl.exclude("time")).to_numpy()
            self.mean_ = np.mean(vals[-self.window :], axis=0)

    def _predict_one(self, groups, **params):
        if self.groups_ is None:
            cols = [c for c in self.local_y_schema_ if c != "time"]
            return pl.DataFrame(
                {col: [float(self.mean_[i])] * self.fit_forecasting_horizon_ for i, col in enumerate(cols)}
            )
        results = {}
        for group in groups:
            cols = [c for c in self.local_y_schema_ if c != "time"]
            results[group] = pl.DataFrame(
                {col: [float(self.means_[group][i])] * self.fit_forecasting_horizon_ for i, col in enumerate(cols)}
            )
        return results


class TestTier1FitHook:
    """Verify the Tier 1 (simple) forecaster pattern works end to end."""

    @pytest.fixture()
    def daily_series(self):
        rng = np.random.default_rng(42)
        n = 100
        return pl.DataFrame(
            {
                "time": pl.datetime_range(
                    pl.lit("2020-01-01").str.to_datetime(),
                    pl.lit("2020-01-01").str.to_datetime() + pl.duration(days=n - 1),
                    interval="1d",
                    eager=True,
                ),
                "value": rng.normal(10, 1, n).tolist(),
            }
        )

    def test_fit_sets_attributes(self, daily_series):
        m = _WindowMeanForecaster(window=7)
        m.fit(daily_series[:80], forecasting_horizon=5)

        assert hasattr(m, "mean_")
        assert hasattr(m, "fit_forecasting_horizon_")
        assert m.fit_forecasting_horizon_ == 5

    def test_observation_horizon_from_property(self):
        m = _WindowMeanForecaster(window=10)
        assert m.observation_horizon == 10

    def test_observation_horizon_changes_with_param(self):
        m = _WindowMeanForecaster(window=3)
        assert m.observation_horizon == 3
        m.set_params(window=20)
        assert m.observation_horizon == 20

    def test_fit_and_check_is_fitted(self, daily_series):
        from sklearn.utils.validation import check_is_fitted

        m = _WindowMeanForecaster(window=5)
        m.fit(daily_series[:80], forecasting_horizon=3)
        check_is_fitted(m)

    def test_clone_preserves_params(self):
        from sklearn.base import clone

        m = _WindowMeanForecaster(window=12)
        m2 = clone(m)
        assert m2.window == 12
        assert m2.observation_horizon == 12
