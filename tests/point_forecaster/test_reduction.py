import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from yohou.point_forecaster import PointReductionForecaster

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from estimator_checks import _yield_yohou_forecaster_checks

length = 22

time = pl.DataFrame(
    {
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 0, 0, length - 1),
            interval="1s",
            eager=True,
        ),
    }
)
y = pl.DataFrame(
    {
        "a": range(length),
        "b": range(10, length + 10),
    },
    schema={
        "a": pl.Float64,
        "b": pl.Float64,
    },
)
y = pl.concat([time, y], how="horizontal")

X_ante = pl.DataFrame(
    {
        "c": range(length),
        "d": range(10, length + 10),
        "e": range(20, length + 20),
    },
    schema={
        "c": pl.Float64,
        "d": pl.Float64,
        "e": pl.Float64,
    },
)
X_ante = pl.concat([time, X_ante], how="horizontal")

y_train, y_test, X_ante_train, X_ante_test = train_test_split(
    y, X_ante, test_size=0.2, shuffle=False
)


@pytest.mark.parametrize(
    "fit_forecasting_horizon, predict_forecasting_horizon, expected_a",
    [
        (1, 5, [17, 17.4, 17.56, 17.624, 17.6496]),
        (3, 5, [17, 18, 19, 18.2, 19.2]),
        (3, 2, [17, 18]),
    ],
)
def test_predict(fit_forecasting_horizon, predict_forecasting_horizon, expected_a):
    forecaster = PointReductionForecaster()

    forecaster.fit(y=y_train, X_ante=X_ante_train, forecasting_horizon=fit_forecasting_horizon)

    y_pred = forecaster.predict(forecasting_horizon=predict_forecasting_horizon)

    expected_y_pred = pl.DataFrame(
        {
            "observed_time": [y_train["time"][-1]] * predict_forecasting_horizon,
            "time": pl.datetime_range(
                start=datetime(2021, 12, 16, 0, 0, len(y_train)),
                end=datetime(2021, 12, 16, 0, 0, len(y_train) + predict_forecasting_horizon - 1),
                interval="1s",
                eager=True,
            ),
            "a": expected_a,
            "b": np.array(expected_a) + 10,
        },
        schema={
            "observed_time": pl.Datetime(time_unit="us", time_zone=None),
            "time": pl.Datetime(time_unit="us", time_zone=None),
            "a": pl.Float64,
            "b": pl.Float64,
        },
    )
    pl.testing.assert_frame_equal(y_pred, expected_y_pred)


@pytest.mark.parametrize(
    "fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a",
    [
        (1, 5, 1, [22, 22.4, 22.56, 22.624, 22.6496]),
        (3, 5, 2, [22, 23, 24, 23.2, 24.2]),
        (3, 2, 1, [22, 23]),
    ],
)
def test_update_predict(fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a):
    forecaster = PointReductionForecaster()

    forecaster.fit(y=y_train, X_ante=X_ante_train, forecasting_horizon=fit_forecasting_horizon)

    y_pred = forecaster.update_predict(
        y=y_test,
        X_ante=X_ante_test,
        forecasting_horizon=predict_forecasting_horizon,
        stride=stride,
    )

    # Get the last set of predictions (after all updates)
    y_pred = y_pred.tail(predict_forecasting_horizon)

    expected_y_pred = pl.DataFrame(
        {
            "observed_time": [y_test["time"][-1]] * predict_forecasting_horizon,
            "time": pl.datetime_range(
                start=datetime(2021, 12, 16, 0, 0, len(y_train) + len(y_test)),
                end=datetime(
                    2021, 12, 16, 0, 0, len(y_train) + len(y_test) + predict_forecasting_horizon - 1
                ),
                interval="1s",
                eager=True,
            ),
            "a": expected_a,
            "b": np.array(expected_a) + 10,
        },
        schema={
            "observed_time": pl.Datetime(time_unit="us", time_zone=None),
            "time": pl.Datetime(time_unit="us", time_zone=None),
            "a": pl.Float64,
            "b": pl.Float64,
        },
    )
    pl.testing.assert_frame_equal(y_pred, expected_y_pred)


y_struct = pl.DataFrame(
    {
        "x": pl.DataFrame(
            {
                "a": range(length),
                "b": range(10, length + 10),
            }
        ),
        "y": pl.DataFrame(
            {
                "a": range(10, length + 10),
                "b": range(20, length + 20),
            }
        ),
    },
    schema={
        "x": pl.Struct({"a": pl.Float64, "b": pl.Float64}),
        "y": pl.Struct({"a": pl.Float64, "b": pl.Float64}),
    },
)
y_struct = pl.concat([time, y_struct], how="horizontal")

X_ante_struct = pl.DataFrame(
    {
        "x": pl.DataFrame(
            {
                "c": range(length),
            }
        ),
        "y": pl.DataFrame(
            {
                "c": range(10, length + 10),
            }
        ),
        "d": range(10, length + 10),
        "e": range(20, length + 20),
    },
    schema={
        "x": pl.Struct({"c": pl.Float64}),
        "y": pl.Struct({"c": pl.Float64}),
        "d": pl.Float64,
        "e": pl.Float64,
    },
)
X_ante_struct = pl.concat([time, X_ante_struct], how="horizontal")

y_train_struct, y_test_struct, X_ante_train_struct, X_ante_test_struct = train_test_split(
    y_struct, X_ante_struct, test_size=0.2, shuffle=False
)


@pytest.mark.parametrize(
    "fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a",
    [
        # Note: struct columns produce different predictions than flat columns
        # due to how the reduction forecaster handles grouped features
        (1, 5, 1, [22.0, 22.666667, 23.111111, 23.407407, 23.604938]),
        (3, 5, 2, [22.0, 23.0, 24.0, 24.0, 25.0]),
        (3, 2, 1, [22.0, 23.0]),
    ],
)
def test_update_predict_global(
    fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a
):
    forecaster = PointReductionForecaster()

    forecaster.fit(
        y=y_train_struct,
        X_ante=X_ante_train_struct,
        forecasting_horizon=fit_forecasting_horizon,
    )

    y_pred = forecaster.update_predict(
        y=y_test_struct,
        X_ante=X_ante_test_struct,
        forecasting_horizon=predict_forecasting_horizon,
        stride=stride,
    )

    # Get the last set of predictions (after all updates)
    y_pred = y_pred.tail(predict_forecasting_horizon)

    expected_y_pred = pl.DataFrame(
        {
            "observed_time": [y_test_struct["time"][-1]] * predict_forecasting_horizon,
            "time": pl.datetime_range(
                start=datetime(2021, 12, 16, 0, 0, len(y_train_struct) + len(y_test_struct)),
                end=datetime(
                    2021,
                    12,
                    16,
                    0,
                    0,
                    len(y_train_struct) + len(y_test_struct) + predict_forecasting_horizon - 1,
                ),
                interval="1s",
                eager=True,
            ),
            "x": pl.DataFrame(
                {
                    "a": expected_a,
                    "b": np.array(expected_a) + 10,
                },
                schema={"a": pl.Float64, "b": pl.Float64},
            ),
            "y": pl.DataFrame(
                {
                    "a": np.array(expected_a) + 10,
                    "b": np.array(expected_a) + 20,
                },
                schema={"a": pl.Float64, "b": pl.Float64},
            ),
        },
        schema={
            "observed_time": pl.Datetime(time_unit="us", time_zone=None),
            "time": pl.Datetime(time_unit="us", time_zone=None),
            "x": pl.Struct({"a": pl.Float64, "b": pl.Float64}),
            "y": pl.Struct({"a": pl.Float64, "b": pl.Float64}),
        },
    )
    pl.testing.assert_frame_equal(y_pred, expected_y_pred)


# ============================================================================
# Check generator tests
# ============================================================================


@pytest.mark.parametrize(
    "forecaster,tags,expected_failures",
    [
        (
            PointReductionForecaster(),
            {"forecaster_type": "point", "uses_reduction": True},
            [],
        ),
        (
            PointReductionForecaster(estimator=LinearRegression()),
            {"forecaster_type": "point", "uses_reduction": True},
            [],
        ),
    ],
)
def test_point_reduction_checks(forecaster, tags, expected_failures, y_X_factory):
    """Run systematic checks on PointReductionForecaster."""
    # Generate data
    y, X_ante, X_post = y_X_factory(length=100, seed=42)
    y_train, y_test = y[:80], y[80:]
    X_ante_train, X_ante_test = X_ante[:80], X_ante[80:]
    X_post_train, X_post_test = (X_post[:80], X_post[80:]) if X_post is not None else (None, None)

    # Fit forecaster
    forecaster_fitted = clone(forecaster)
    forecaster_fitted.fit(y_train, X_ante_train, X_post_train, forecasting_horizon=3)

    # Run all generated checks
    expected_failures_set = set(expected_failures)
    for check_name, check_func, check_kwargs in _yield_yohou_forecaster_checks(
        forecaster_fitted,
        y_train,
        X_ante_train,
        X_post_train,
        y_test,
        X_ante_test,
        X_post_test,
        tags=tags,
    ):
        if check_name in expected_failures_set:
            pytest.skip(f"Expected failure: {check_name}")
        else:
            check_func(forecaster_fitted, **check_kwargs)


# ============================================================================
# Analytical tests with LinearRegression
# ============================================================================


def test_linear_regression_perfect_linear_trend():
    """Test PointReductionForecaster with LinearRegression on perfect linear trend.

    With a perfect linear trend y = mx + b and default lag-1 features, LinearRegression
    should produce exact predictions since the relationship is exactly linear.
    """
    # Create perfect linear trend: y = 2*t + 10
    length = 50
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1, 0, 0, length - 1),
        interval="1s",
        eager=True,
    )
    y = pl.DataFrame(
        {
            "time": time,
            "value": [2.0 * i + 10.0 for i in range(length)],
        }
    )

    # Train/test split
    y_train, y_test = y[:40], y[40:]

    # Create forecaster with default feature transformer (LagTransformer(lag=[1])) and LinearRegression
    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
    )

    # Fit on training data with horizon=1 (one-step-ahead forecasting)
    forecaster.fit(y_train, X_ante=None, X_post=None, forecasting_horizon=1)

    # Predict one step ahead (exact prediction for linear trend with AR(1) structure)
    y_pred = forecaster.predict(forecasting_horizon=1)

    # Expected value: continue the linear trend
    expected_value = 2.0 * 40 + 10.0  # y_41 = 2*40 + 10 = 90

    # Check prediction is very close (numerical precision tolerance)
    predicted_value = y_pred["value"][0]
    np.testing.assert_allclose(predicted_value, expected_value, rtol=1e-5, atol=1e-5)


def test_linear_regression_ar1_process():
    """Test PointReductionForecaster with LinearRegression on AR(1) process.

    For an AR(1) process y_t = phi * y_{t-1} + c, LinearRegression with default lag=1
    should recover the exact parameters and produce exact one-step-ahead predictions.
    """
    # Create AR(1) process: y_t = 0.8 * y_{t-1} + 5
    phi = 0.8
    c = 5.0
    length = 50

    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1, 0, 0, length - 1),
        interval="1s",
        eager=True,
    )

    # Generate AR(1) series
    values = [10.0]  # Initial value
    for i in range(1, length):
        values.append(phi * values[-1] + c)

    y = pl.DataFrame(
        {
            "time": time,
            "value": values,
        }
    )

    # Train/test split
    y_train, y_test = y[:40], y[40:]

    # Create forecaster with default feature transformer (LagTransformer(lag=[1])) and LinearRegression
    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
    )

    # Fit on training data with horizon=1
    forecaster.fit(y_train, X_ante=None, X_post=None, forecasting_horizon=1)

    # Check fitted coefficients are close to true values
    # The default LagTransformer creates lag=1 features
    # LinearRegression learns: y = coef * lag1 + intercept
    # Should have: coef ≈ phi, intercept ≈ c
    fitted_estimator = forecaster.estimator_
    np.testing.assert_allclose(fitted_estimator.coef_[0], phi, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fitted_estimator.intercept_, c, rtol=1e-10, atol=1e-10)

    # Predict one step ahead (should be exact given the model structure)
    y_pred = forecaster.predict(forecasting_horizon=1)

    # Expected value: phi * last_observed + c
    expected_value = phi * y_train["value"][-1] + c

    # Check prediction is exact
    predicted_value = y_pred["value"][0]
    np.testing.assert_allclose(predicted_value, expected_value, rtol=1e-10, atol=1e-10)
