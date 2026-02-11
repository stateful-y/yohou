from datetime import datetime

import numpy as np
import polars as pl
import polars.testing
import pytest
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from yohou.point_forecaster import SeasonalNaive
from yohou.testing import _yield_yohou_forecaster_checks

length = 22

y = pl.DataFrame({
    "time": pl.datetime_range(
        start=datetime(2021, 12, 16),
        end=datetime(2021, 12, 16, 0, 0, length - 1),
        interval="1s",
        eager=True,
    ),
    "a": range(length),
    "b": range(10, length + 10),
})

X = pl.DataFrame({
    "time": pl.datetime_range(
        start=datetime(2021, 12, 16),
        end=datetime(2021, 12, 16, 0, 0, length - 1),
        interval="1s",
        eager=True,
    ),
    "c": range(length),
    "d": range(10, length + 10),
    "e": range(20, length + 20),
})


@pytest.mark.parametrize(
    "forecaster,expected_failures",
    [
        (
            SeasonalNaive(seasonality=1),
            [],
        ),
        (
            SeasonalNaive(seasonality=7),
            [],
        ),
    ],
)
def test_seasonal_naive_checks(forecaster, expected_failures, y_X_factory):
    """Run systematic checks on SeasonalNaive forecaster."""
    # Generate data
    y, X = y_X_factory(length=200, seed=42)
    y_train, y_test = y[:160], y[160:]
    X_train, X_test = (X[:160], X[160:]) if X is not None else (None, None)

    # Fit forecaster
    forecaster_fitted = clone(forecaster)
    forecaster_fitted.fit(y_train, X_train, forecasting_horizon=3)

    # Run all generated checks
    expected_failures_set = set(expected_failures)
    for check_name, check_func, check_kwargs in _yield_yohou_forecaster_checks(
        forecaster_fitted,
        y_train,
        X_train,
        y_test,
        X_test,
    ):
        if check_name in expected_failures_set:
            pytest.skip(f"Expected failure: {check_name}")
        else:
            check_func(forecaster_fitted, **check_kwargs)


@pytest.mark.parametrize(
    "seasonality, fit_forecasting_horizon, predict_forecasting_horizon, expected_a",
    [
        (1, 1, 5, [16, 16, 16, 16, 16]),
        (1, 3, 5, [16, 16, 16, 16, 16]),
        (1, 3, 2, [16, 16]),
        (2, 1, 5, [15, 16, 15, 16, 15]),
        (2, 3, 5, [15, 16, 15, 16, 15]),
        (2, 3, 2, [15, 16]),
        (8, 1, 5, [9, 10, 11, 12, 13]),
        (8, 3, 5, [9, 10, 11, 12, 13]),
        (8, 3, 2, [9, 10]),
    ],
)
def test_predict(seasonality, fit_forecasting_horizon, predict_forecasting_horizon, expected_a):
    """Test specific prediction behavior of SeasonalNaive."""
    y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.2, shuffle=False)
    forecaster = SeasonalNaive(seasonality=seasonality)

    forecaster.fit(y=y_train, X=X_train, forecasting_horizon=fit_forecasting_horizon)

    y_pred = forecaster.predict(forecasting_horizon=predict_forecasting_horizon)

    expected_y_pred = pl.DataFrame({
        "observed_time": [y_train["time"][-1]] * predict_forecasting_horizon,
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16, 0, 0, len(y_train)),
            end=datetime(2021, 12, 16, 0, 0, len(y_train) + predict_forecasting_horizon - 1),
            interval="1s",
            eager=True,
        ),
        "a": expected_a,
        "b": np.array(expected_a) + 10,
    })
    pl.testing.assert_frame_equal(y_pred, expected_y_pred)
