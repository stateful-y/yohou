"""Coverage for X_future payloads wider than the historic test envelope.

Nothing in the suite used X_future above 2 value columns at a horizon of 6, and
nothing used it at a horizon of 7 or more. That is exactly the regime in which
step expansion is cheap, which is why a payload blowing up to hundreds of
columns went unnoticed. These tests move the envelope.
"""

from datetime import datetime

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from yohou.point import PointReductionForecaster

H = 48
N_FEATURES = 5
N_STEP_COLUMNS = N_FEATURES * H


@pytest.fixture(scope="module")
def wide_data():
    """A daily series plus a 5-column event-like X_future covering every target."""
    time = pl.datetime_range(datetime(2021, 1, 1), datetime(2022, 12, 31), interval="1d", eager=True)
    rng = np.random.default_rng(7)
    y = pl.DataFrame({"time": time, "value": rng.standard_normal(len(time)).cumsum()})

    future_time = pl.datetime_range(datetime(2021, 1, 1), datetime(2023, 6, 30), interval="1d", eager=True)
    X_future = pl.DataFrame({
        "time": future_time,
        **{f"feat_{i}": rng.standard_normal(len(future_time)) for i in range(N_FEATURES)},
    })
    return y, X_future


class TestWideXFuture:
    """A wide X_future fits, predicts, and filters as documented."""

    def test_wide_x_future_fits_and_predicts(self, wide_data):
        """5 value columns at H=48 produce 240 step columns and a usable forecast."""
        y, X_future = wide_data
        forecaster = PointReductionForecaster(estimator=LinearRegression())
        forecaster.fit(y=y, forecasting_horizon=H, X_future=X_future)

        assert len(forecaster._step_column_names_) == N_STEP_COLUMNS
        assert len(forecaster.predict()) == H

    def test_matched_drops_the_other_steps(self, wide_data):
        """Each direct estimator sees its own 5 step columns, not the other 235."""
        y, X_future = wide_data
        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            reduction_strategy="direct",
            step_feature_alignment="matched",
        )
        forecaster.fit(y=y, forecasting_horizon=H, X_future=X_future)

        all_steps = PointReductionForecaster(
            estimator=LinearRegression(),
            reduction_strategy="direct",
            step_feature_alignment="all",
        )
        all_steps.fit(y=y, forecasting_horizon=H, X_future=X_future)

        n_non_step = all_steps.estimator_[0].n_features_in_ - N_STEP_COLUMNS
        for estimator in forecaster.estimator_:
            assert estimator.n_features_in_ == N_FEATURES + n_non_step

    def test_cumulative_grows_with_horizon(self, wide_data):
        """Step h sees steps 1..h, so the feature count rises monotonically."""
        y, X_future = wide_data
        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            reduction_strategy="direct",
            step_feature_alignment="cumulative",
        )
        forecaster.fit(y=y, forecasting_horizon=H, X_future=X_future)

        counts = [estimator.n_features_in_ for estimator in forecaster.estimator_]
        assert counts == sorted(counts)
        assert counts[0] < counts[-1]
        assert counts[-1] - counts[0] == N_STEP_COLUMNS - N_FEATURES
