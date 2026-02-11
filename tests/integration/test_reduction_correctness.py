"""Analytical correctness integration tests for reduction forecasters.

This module verifies that PointReductionForecaster produces numerically exact
predictions for time series with known closed-form solutions. Tests cover:

1. Tabularization verification (feature/target structure)
2. Constant series (exact constant predictions)
3. Linear trends (exact recursive predictions)
4. AR(1) processes (closed-form multi-step forecasts)
5. AR(p) processes (coefficient recovery and predictions)
6. Exogenous features (known feature contributions)
7. Horizon mismatches (fit_horizon ≠ predict_horizon)

All tests use `atol=1e-10` for exact cases (no noise) to ensure numerical precision.
"""

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer
from yohou.utils.tabularization import tabularize


@pytest.mark.integration
@pytest.mark.parametrize(
    "length,lags",
    [
        (10, [1]),
        (10, [1, 2]),
        (10, [1, 2, 3, 5]),
        (20, [1, 2, 3]),
        (20, [1, 3, 7]),
        (50, [1]),
        (50, [1, 5, 10]),
        (100, [1]),
        (100, [1, 2, 3, 4, 5]),
        (100, [1, 5, 10, 20]),
    ],
)
def test_tabularization_shape(linear_series, length, lags):
    """Verify tabularization produces correct number of rows after dropping NaN lags."""
    y = linear_series(slope=1.0, intercept=0.0, length=length)

    result = tabularize(df_time_series=y, lags=lags)

    # Number of rows should be length - max(lags)
    expected_rows = length - max(lags)
    assert len(result) == expected_rows, f"Expected {expected_rows} rows, got {len(result)}"


@pytest.mark.integration
@pytest.mark.parametrize("lags", [[1], [1, 2], [1, 2, 3, 5]])
def test_tabularization_feature_names(linear_series, lags):
    """Verify feature column names match {col}_lag_{i} pattern."""
    y = linear_series(slope=1.0, intercept=0.0, length=100)

    result = tabularize(df_time_series=y, lags=lags)

    # Expected feature names: value_lag_1, value_lag_2, ...
    expected_features = [f"value_lag_{lag}" for lag in lags]
    feature_cols = [c for c in result.columns if c != "time"]

    # Sort both for comparison
    assert sorted(feature_cols) == sorted(expected_features), (
        f"Feature names mismatch: expected {expected_features}, got {feature_cols}"
    )


@pytest.mark.integration
@pytest.mark.parametrize("lags", [[1], [1, 2], [1, 2, 3]])
def test_tabularization_output_columns(linear_series, lags):
    """Verify output has time column and lagged columns, not original value."""
    y = linear_series(slope=1.0, intercept=0.0, length=50)

    result = tabularize(df_time_series=y, lags=lags)

    # Should have "time" column
    assert "time" in result.columns, "Missing 'time' column in output"

    # Should NOT have original "value" column
    assert "value" not in result.columns, "Original 'value' column should be excluded from output"

    # Should have lagged columns
    for lag in lags:
        col_name = f"value_lag_{lag}"
        assert col_name in result.columns, f"Missing column {col_name}"


@pytest.mark.integration
def test_tabularization_values_constant(constant_series):
    """Verify tabularized lagged values are correct for constant series."""
    value = 42.0
    y = constant_series(value=value, length=20)

    result = tabularize(df_time_series=y, lags=[1, 2, 3])

    # All lagged values should be the constant
    for col in ["value_lag_1", "value_lag_2", "value_lag_3"]:
        actual = result[col].to_numpy()
        expected = np.full(len(result), value)
        np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.integration
def test_tabularization_values_linear(linear_series):
    """Verify tabularized lagged values are correct for linear series."""
    slope = 2.0
    intercept = 10.0
    length = 20
    y = linear_series(slope=slope, intercept=intercept, length=length)

    result = tabularize(df_time_series=y, lags=[1, 2])

    # After dropping first 2 rows (max lag = 2), result starts at original index 2
    # Row 0 of result corresponds to original index 2:
    #   value_lag_1 = y[1] = slope*1 + intercept
    #   value_lag_2 = y[0] = slope*0 + intercept
    first_lag_1 = result["value_lag_1"][0]
    first_lag_2 = result["value_lag_2"][0]

    expected_lag_1 = slope * 1 + intercept  # y at original index 1
    expected_lag_2 = slope * 0 + intercept  # y at original index 0

    np.testing.assert_allclose(first_lag_1, expected_lag_1, atol=1e-10)
    np.testing.assert_allclose(first_lag_2, expected_lag_2, atol=1e-10)


@pytest.mark.integration
@pytest.mark.parametrize("value", [0.0, 1.0, 42.0, -10.5, 1e6])
@pytest.mark.parametrize("forecasting_horizon", [1, 2, 5, 10])
def test_constant_series_exact_prediction(constant_series, value, forecasting_horizon):
    """Verify constant series predictions are exactly constant."""
    y = constant_series(value=value, length=100)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
    )
    forecaster.fit(y[:80], forecasting_horizon=forecasting_horizon)

    y_pred = forecaster.predict()
    pred_values = y_pred["value"].to_numpy()

    expected = np.full(forecasting_horizon, value)
    np.testing.assert_allclose(pred_values, expected, atol=1e-10)


@pytest.mark.integration
@pytest.mark.parametrize("value", [5.0, 100.0])
def test_constant_series_observe_predict(constant_series, value):
    """Verify constant series observe then predict maintains constant."""
    y = constant_series(value=value, length=100)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
    )
    forecaster.fit(y[:50], forecasting_horizon=3)

    # Observe new constant data, then predict
    forecaster.observe(y[50:80])
    y_pred = forecaster.predict()
    pred_values = y_pred["value"].to_numpy()

    expected = np.full(3, value)
    np.testing.assert_allclose(pred_values, expected, atol=1e-10)


@pytest.mark.integration
@pytest.mark.parametrize("slope", [0.5, 1.0, 2.0, 100.0])
@pytest.mark.parametrize("forecasting_horizon", [1, 5, 10])
def test_linear_trend_prediction(linear_series, analytical_forecast, slope, forecasting_horizon):
    """Verify linear trend predictions match analytical forecast."""
    intercept = 10.0
    train_length = 80
    y = linear_series(slope=slope, intercept=intercept, length=100)

    forecaster = PointReductionForecaster(
        estimator=Ridge(alpha=0.0),  # Ridge with alpha=0 == LinearRegression
    )
    forecaster.fit(y[:train_length], forecasting_horizon=forecasting_horizon)

    y_pred = forecaster.predict()
    pred_values = y_pred["value"].to_numpy()

    # Analytical forecast: y = slope * t + intercept
    # Last training index is train_length - 1, next is train_length
    expected = analytical_forecast(
        series_type="linear",
        params={"slope": slope, "intercept": intercept},
        horizon=forecasting_horizon,
        last_index=train_length - 1,
    )

    np.testing.assert_allclose(pred_values, expected, atol=1e-10)


@pytest.mark.integration
@pytest.mark.parametrize("slope", [1.0, 5.0])
def test_linear_trend_horizon_1_exact(linear_series, slope):
    """Verify horizon=1 predicts exact next value for linear trend."""
    intercept = 0.0
    train_length = 50
    y = linear_series(slope=slope, intercept=intercept, length=60)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
    )
    forecaster.fit(y[:train_length], forecasting_horizon=1)

    y_pred = forecaster.predict()
    pred_value = y_pred["value"][0]

    # Next value at t=train_length
    expected = slope * train_length + intercept

    np.testing.assert_allclose(pred_value, expected, atol=1e-10)


@pytest.mark.integration
@pytest.mark.parametrize("slope", [2.0, 10.0])
def test_linear_trend_recursive_horizon_5(linear_series, analytical_forecast, slope):
    """Verify horizon=5 recursive prediction for linear trend."""
    intercept = 5.0
    train_length = 50
    y = linear_series(slope=slope, intercept=intercept, length=70)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
    )
    forecaster.fit(y[:train_length], forecasting_horizon=5)

    y_pred = forecaster.predict()
    pred_values = y_pred["value"].to_numpy()

    # Each step should add slope to previous
    expected = analytical_forecast(
        series_type="linear",
        params={"slope": slope, "intercept": intercept},
        horizon=5,
        last_index=train_length - 1,
    )

    np.testing.assert_allclose(pred_values, expected, atol=1e-9)


@pytest.mark.integration
@pytest.mark.parametrize("slope", [1.0, 3.0])
def test_linear_trend_fit_horizon_5_predict_horizon_10(linear_series, analytical_forecast, slope):
    """Verify fit with horizon=5, predict with horizon=10 (recursive 2× application)."""
    intercept = 0.0
    train_length = 50
    y = linear_series(slope=slope, intercept=intercept, length=70)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
    )
    forecaster.fit(y[:train_length], forecasting_horizon=5)

    # Predict with horizon=10 (larger than fit horizon)
    y_pred = forecaster.predict(forecasting_horizon=10)
    pred_values = y_pred["value"].to_numpy()

    # Should still follow linear trend
    expected = analytical_forecast(
        series_type="linear",
        params={"slope": slope, "intercept": intercept},
        horizon=10,
        last_index=train_length - 1,
    )

    np.testing.assert_allclose(pred_values, expected, atol=1e-8)


@pytest.mark.integration
@pytest.mark.parametrize("phi", [0.3, 0.8, 0.99, -0.5])
@pytest.mark.parametrize("forecasting_horizon", [1, 3, 5])
def test_ar1_prediction_accuracy(ar1_series, analytical_forecast, phi, forecasting_horizon):
    """Verify AR(1) predictions match analytical closed-form forecast."""
    c = 2.0
    train_length = 100
    y = ar1_series(phi=phi, c=c, length=120, noise_std=0.0)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
        feature_transformer=LagTransformer(lag=[1]),
    )
    forecaster.fit(y[:train_length], forecasting_horizon=forecasting_horizon)

    y_pred = forecaster.predict()
    pred_values = y_pred["value"].to_numpy()

    # Analytical forecast: y_{t+k} = phi^k * y_t + c * (1 - phi^k) / (1 - phi)
    last_obs = y["value"][train_length - 1]
    expected = analytical_forecast(
        series_type="ar1",
        params={"phi": phi, "c": c},
        horizon=forecasting_horizon,
        last_obs=last_obs,
    )

    # Use slightly larger tolerance for recursive predictions
    np.testing.assert_allclose(pred_values, expected, atol=1e-8, rtol=1e-6)


@pytest.mark.integration
@pytest.mark.parametrize("phi", [0.5, 0.9])
def test_ar1_fitted_coefficients(ar_p_series, phi):
    """Verify fitted AR(1) coefficients match true parameters."""
    c = 1.0
    # Use ar_p_series (starts at c, not stationary mean) so data is non-constant
    y = ar_p_series(coeffs=[phi], c=c, length=200, noise_std=0.0)

    # No feature_transformer needed: internal tabularization creates
    # features = y_t, target = y_{t+1} = phi*y_t + c
    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
    )
    forecaster.fit(y[:150], forecasting_horizon=1)

    # Check fitted coefficients
    fitted_phi = forecaster.estimator_.coef_[0][0]  # First target, first feature
    fitted_c = forecaster.estimator_.intercept_[0]  # First target

    np.testing.assert_allclose(fitted_phi, phi, atol=1e-3)
    np.testing.assert_allclose(fitted_c, c, atol=1e-2)


@pytest.mark.integration
@pytest.mark.parametrize("phi", [0.7, -0.3])
def test_ar1_horizon_1_exact(ar1_series, phi):
    """Verify horizon=1 AR(1) prediction is exact (y_{t+1} = phi*y_t + c)."""
    c = 3.0
    train_length = 100
    y = ar1_series(phi=phi, c=c, length=110, noise_std=0.0)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
        feature_transformer=LagTransformer(lag=[1]),
    )
    forecaster.fit(y[:train_length], forecasting_horizon=1)

    y_pred = forecaster.predict()
    pred_value = y_pred["value"][0]

    # Exact next step: y_{t+1} = phi * y_t + c
    last_obs = y["value"][train_length - 1]
    expected = phi * last_obs + c

    np.testing.assert_allclose(pred_value, expected, atol=1e-9)


@pytest.mark.integration
@pytest.mark.parametrize("phi", [0.6, 0.95])
@pytest.mark.parametrize("horizon", [2, 3, 5])
def test_ar1_multistep_recursive(ar1_series, analytical_forecast, phi, horizon):
    """Verify multi-step AR(1) forecast via recursion."""
    c = 1.5
    train_length = 80
    y = ar1_series(phi=phi, c=c, length=100, noise_std=0.0)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
        feature_transformer=LagTransformer(lag=[1]),
    )
    forecaster.fit(y[:train_length], forecasting_horizon=horizon)

    y_pred = forecaster.predict()
    pred_values = y_pred["value"].to_numpy()

    last_obs = y["value"][train_length - 1]
    expected = analytical_forecast(
        series_type="ar1",
        params={"phi": phi, "c": c},
        horizon=horizon,
        last_obs=last_obs,
    )

    np.testing.assert_allclose(pred_values, expected, atol=1e-7, rtol=1e-5)


@pytest.mark.integration
def test_ar2_fitted_coefficients(ar_p_series):
    """Verify fitted AR(2) coefficients match derived one-step-ahead parameters.

    With LagTransformer(lag=[1, 2]), features are [y_{t-1}, y_{t-2}] and
    the target is y_{t+1}. For AR(2) y_t = phi_1*y_{t-1} + phi_2*y_{t-2} + c:
        y_{t+1} = (phi_1^2 + phi_2)*y_{t-1} + phi_1*phi_2*y_{t-2} + c*(1+phi_1)
    """
    coeffs = [0.5, 0.3]
    c = 2.0
    y = ar_p_series(coeffs=coeffs, c=c, length=300, noise_std=0.0)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
        feature_transformer=LagTransformer(lag=[1, 2]),
    )
    forecaster.fit(y[:250], forecasting_horizon=1)

    # Check fitted coefficients: one-step-ahead relationship through LagTransformer
    fitted_coeffs = forecaster.estimator_.coef_[0]  # First target
    fitted_c = forecaster.estimator_.intercept_[0]

    # Due to one-step shift from LagTransformer + internal tabularization:
    # effective coef_1 = phi_1^2 + phi_2, coef_2 = phi_1*phi_2
    phi_1, phi_2 = coeffs
    expected_coeffs = np.array([phi_1**2 + phi_2, phi_1 * phi_2])
    expected_c = c * (1 + phi_1)

    np.testing.assert_allclose(fitted_coeffs, expected_coeffs, atol=1e-2)
    np.testing.assert_allclose(fitted_c, expected_c, atol=1e-1)


@pytest.mark.integration
def test_ar2_horizon_1_exact(ar_p_series):
    """Verify horizon=1 AR(2) prediction is exact."""
    coeffs = [0.5, 0.3]
    c = 2.0
    train_length = 200
    y = ar_p_series(coeffs=coeffs, c=c, length=210, noise_std=0.0)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
        feature_transformer=LagTransformer(lag=[1, 2]),
    )
    forecaster.fit(y[:train_length], forecasting_horizon=1)

    y_pred = forecaster.predict()
    pred_value = y_pred["value"][0]

    # Exact: y_{t+1} = 0.5*y_t + 0.3*y_{t-1} + 2.0
    y_t = y["value"][train_length - 1]
    y_t_minus_1 = y["value"][train_length - 2]
    expected = coeffs[0] * y_t + coeffs[1] * y_t_minus_1 + c

    np.testing.assert_allclose(pred_value, expected, atol=1e-8)


@pytest.mark.integration
@pytest.mark.parametrize("horizon", [2, 3, 5])
def test_ar2_multistep_prediction(ar_p_series, horizon):
    """Verify multi-step AR(2) predictions (horizon > 1)."""
    coeffs = [0.5, 0.3]
    c = 2.0
    train_length = 200
    y = ar_p_series(coeffs=coeffs, c=c, length=220, noise_std=0.0)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
        feature_transformer=LagTransformer(lag=[1, 2]),
    )
    forecaster.fit(y[:train_length], forecasting_horizon=horizon)

    y_pred = forecaster.predict()
    pred_values = y_pred["value"].to_numpy()

    # Manually compute expected via recursion
    y_values = y["value"].to_numpy()
    expected = []

    # Start with last two observed values
    history = [y_values[train_length - 2], y_values[train_length - 1]]

    for _ in range(horizon):
        y_next = coeffs[0] * history[-1] + coeffs[1] * history[-2] + c
        expected.append(y_next)
        history.append(y_next)

    expected = np.array(expected)

    np.testing.assert_allclose(pred_values, expected, atol=1e-7)


@pytest.mark.integration
def test_ar3_fitted_coefficients(ar_p_series):
    """Verify fitted AR(3) coefficients match derived one-step-ahead parameters.

    With LagTransformer(lag=[1, 2, 3]), features are [y_{t-1}, y_{t-2}, y_{t-3}]
    and the target is y_{t+1}. For AR(3):
        y_{t+1} = (phi_1^2+phi_2)*y_{t-1} + (phi_1*phi_2+phi_3)*y_{t-2}
                + phi_1*phi_3*y_{t-3} + c*(1+phi_1)
    """
    coeffs = [0.4, 0.3, 0.2]
    c = 1.0
    y = ar_p_series(coeffs=coeffs, c=c, length=400, noise_std=0.0)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
        feature_transformer=LagTransformer(lag=[1, 2, 3]),
    )
    forecaster.fit(y[:350], forecasting_horizon=1)

    fitted_coeffs = forecaster.estimator_.coef_[0]
    fitted_c = forecaster.estimator_.intercept_[0]

    # Due to one-step shift from LagTransformer + internal tabularization:
    phi_1, phi_2, phi_3 = coeffs
    expected_coeffs = np.array([
        phi_1**2 + phi_2,
        phi_1 * phi_2 + phi_3,
        phi_1 * phi_3,
    ])
    expected_c = c * (1 + phi_1)

    np.testing.assert_allclose(fitted_coeffs, expected_coeffs, atol=1e-2)
    np.testing.assert_allclose(fitted_c, expected_c, atol=1e-1)


@pytest.mark.integration
@pytest.mark.parametrize("forecasting_horizon", [1, 3, 5])
def test_exogenous_linear_combination_exact(forecasting_horizon):
    """Verify exact prediction with exogenous features using lagged relationship.

    Uses y_{t+1} = 2*x1_t + 3*x2_t + 5 so that features at time t
    (including x1_t, x2_t) can predict y at time t+1.
    Fits with forecasting_horizon=1 and predicts recursively.
    """
    from datetime import datetime, timedelta

    # Generate data: y_{t+1} = 2*x1_t + 3*x2_t + 5 (lagged relationship)
    length = 100
    start = datetime(2020, 1, 1)
    time = [start + timedelta(days=i) for i in range(length)]

    np.random.seed(42)
    x1 = np.random.randn(length)
    x2 = np.random.randn(length)
    y_values = np.zeros(length)
    y_values[0] = 5.0  # Initial value (no x at t=-1)
    for t in range(1, length):
        y_values[t] = 2.0 * x1[t - 1] + 3.0 * x2[t - 1] + 5.0

    y = pl.DataFrame({
        "time": time,
        "value": y_values,
    }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    X = pl.DataFrame({
        "time": time,
        "x1": x1,
        "x2": x2,
    }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    train_length = 80

    # Generate future X values (known ex-ante)
    np.random.seed(100)
    x1_future = np.random.randn(forecasting_horizon)
    x2_future = np.random.randn(forecasting_horizon)

    time_future = [start + timedelta(days=train_length + i) for i in range(forecasting_horizon)]
    X_future = pl.DataFrame({
        "time": time_future,
        "x1": x1_future,
        "x2": x2_future,
    }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    # Fit with forecasting_horizon=1 for recursive multi-step prediction
    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
    )
    forecaster.fit(y[:train_length], X[:train_length], forecasting_horizon=1)

    # Predict recursively with known future X
    y_pred = forecaster.predict(X=X_future, forecasting_horizon=forecasting_horizon)
    pred_values = y_pred["value"].to_numpy()

    # Expected: first step uses last training X, subsequent use X_future
    expected = np.zeros(forecasting_horizon)
    expected[0] = 2.0 * x1[train_length - 1] + 3.0 * x2[train_length - 1] + 5.0
    for k in range(1, forecasting_horizon):
        expected[k] = 2.0 * x1_future[k - 1] + 3.0 * x2_future[k - 1] + 5.0

    np.testing.assert_allclose(pred_values, expected, atol=1e-9)


@pytest.mark.integration
def test_exogenous_with_ar_component():
    """Verify exogenous + AR(1) combination: y_{t+1} = 0.8*y_t + 1.5*x_t + 2.

    Uses a lagged exogenous relationship so that features at time t
    (y_t, x_t) can exactly predict y_{t+1}.
    """
    from datetime import datetime, timedelta

    # Generate y_{t+1} = 0.8*y_t + 1.5*x_t + 2 (lagged exogenous)
    length = 150
    start = datetime(2020, 1, 1)
    time = [start + timedelta(days=i) for i in range(length)]

    np.random.seed(777)
    x = np.random.randn(length)

    y_values = np.zeros(length)
    y_values[0] = 10.0  # Initial value

    for t in range(1, length):
        y_values[t] = 0.8 * y_values[t - 1] + 1.5 * x[t - 1] + 2.0

    y = pl.DataFrame({
        "time": time,
        "value": y_values,
    }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    X = pl.DataFrame({
        "time": time,
        "x": x,
    }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    train_length = 120

    # No feature_transformer: features = (y_t, x_t), target = y_{t+1}
    # Model learns: y_{t+1} = 0.8*y_t + 1.5*x_t + 2.0 (exact)
    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
    )
    forecaster.fit(y[:train_length], X[:train_length], forecasting_horizon=1)

    # Predict (uses last training features: y_119, x_119)
    y_pred = forecaster.predict()
    pred_value = y_pred["value"][0]

    # Expected: 0.8 * y[train_length-1] + 1.5 * x[train_length-1] + 2.0
    expected = 0.8 * y_values[train_length - 1] + 1.5 * x[train_length - 1] + 2.0

    np.testing.assert_allclose(pred_value, expected, atol=1e-8)


@pytest.mark.integration
def test_exogenous_polynomial_features():
    """Verify exogenous with polynomial features: y_{t+1} = x_t + x_t^2 + 1.

    Uses a lagged relationship so features at time t (including x_t, x_t^2)
    can exactly predict y_{t+1}. Fits with forecasting_horizon=1 and
    predicts recursively for 5 steps.
    """
    from datetime import datetime, timedelta

    # Generate y_{t+1} = x_t + x_t^2 + 1 (lagged relationship)
    length = 100
    start = datetime(2020, 1, 1)
    time = [start + timedelta(days=i) for i in range(length)]

    np.random.seed(555)
    x = np.random.randn(length)
    y_values = np.zeros(length)
    y_values[0] = 1.0  # Initial value (no x at t=-1)
    for t in range(1, length):
        y_values[t] = x[t - 1] + x[t - 1] ** 2 + 1.0

    y = pl.DataFrame({
        "time": time,
        "value": y_values,
    }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    X = pl.DataFrame({
        "time": time,
        "x": x,
    }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    train_length = 80
    predict_horizon = 5

    # Generate future X
    np.random.seed(666)
    x_future = np.random.randn(predict_horizon)
    time_future = [start + timedelta(days=train_length + i) for i in range(predict_horizon)]

    X_future = pl.DataFrame({
        "time": time_future,
        "x": x_future,
    }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    # Add polynomial features manually
    X_train = X[:train_length].with_columns((pl.col("x") ** 2).alias("x_squared"))
    X_future_poly = X_future.with_columns((pl.col("x") ** 2).alias("x_squared"))

    # Fit with forecasting_horizon=1 for recursive multi-step prediction
    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
    )
    forecaster.fit(y[:train_length], X_train, forecasting_horizon=1)

    # Predict recursively
    y_pred = forecaster.predict(X=X_future_poly, forecasting_horizon=predict_horizon)
    pred_values = y_pred["value"].to_numpy()

    # Expected: first step uses last training X, subsequent use X_future
    expected = np.zeros(predict_horizon)
    expected[0] = x[train_length - 1] + x[train_length - 1] ** 2 + 1.0
    for k in range(1, predict_horizon):
        expected[k] = x_future[k - 1] + x_future[k - 1] ** 2 + 1.0

    np.testing.assert_allclose(pred_values, expected, atol=1e-9)


@pytest.mark.integration
@pytest.mark.parametrize(
    "fit_fh,predict_fh",
    [
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 5),
        (1, 7),
        (1, 10),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 5),
        (2, 7),
        (2, 10),
        (5, 1),
        (5, 2),
        (5, 3),
        (5, 5),
        (5, 7),
        (5, 10),
    ],
)
def test_horizon_mismatch_output_shape(linear_series, fit_fh, predict_fh):
    """Verify output shape is correct for all fit_fh × predict_fh combinations."""
    y = linear_series(slope=1.0, intercept=0.0, length=100)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
    )
    forecaster.fit(y[:80], forecasting_horizon=fit_fh)

    y_pred = forecaster.predict(forecasting_horizon=predict_fh)

    assert len(y_pred) == predict_fh, f"Expected {predict_fh} predictions, got {len(y_pred)}"


@pytest.mark.integration
@pytest.mark.parametrize(
    "fit_fh,predict_fh",
    [
        (1, 1),
        (1, 3),
        (1, 5),
        (2, 1),
        (2, 5),
        (2, 10),
        (5, 1),
        (5, 3),
        (5, 10),
    ],
)
def test_horizon_mismatch_linear_correctness(linear_series, analytical_forecast, fit_fh, predict_fh):
    """Verify analytical correctness for horizon mismatch on linear data."""
    slope = 2.0
    intercept = 5.0
    train_length = 80
    y = linear_series(slope=slope, intercept=intercept, length=120)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
    )
    forecaster.fit(y[:train_length], forecasting_horizon=fit_fh)

    y_pred = forecaster.predict(forecasting_horizon=predict_fh)
    pred_values = y_pred["value"].to_numpy()

    # Analytical forecast
    expected = analytical_forecast(
        series_type="linear",
        params={"slope": slope, "intercept": intercept},
        horizon=predict_fh,
        last_index=train_length - 1,
    )

    # Slightly larger tolerance due to recursive application
    np.testing.assert_allclose(pred_values, expected, atol=1e-8)


@pytest.mark.integration
@pytest.mark.parametrize("predict_fh", [2, 5, 7, 10])
def test_horizon_mismatch_fit_1_predict_multiple(linear_series, analytical_forecast, predict_fh):
    """Verify fit with horizon=1, predict with larger horizons."""
    slope = 3.0
    intercept = 10.0
    train_length = 50
    y = linear_series(slope=slope, intercept=intercept, length=80)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
    )
    forecaster.fit(y[:train_length], forecasting_horizon=1)

    y_pred = forecaster.predict(forecasting_horizon=predict_fh)
    pred_values = y_pred["value"].to_numpy()

    expected = analytical_forecast(
        series_type="linear",
        params={"slope": slope, "intercept": intercept},
        horizon=predict_fh,
        last_index=train_length - 1,
    )

    np.testing.assert_allclose(pred_values, expected, atol=1e-8)


@pytest.mark.integration
@pytest.mark.parametrize("predict_fh", [1, 2, 3])
def test_horizon_mismatch_fit_5_predict_smaller(linear_series, analytical_forecast, predict_fh):
    """Verify fit with horizon=5, predict with smaller horizons."""
    slope = 1.0
    intercept = 0.0
    train_length = 60
    y = linear_series(slope=slope, intercept=intercept, length=80)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
    )
    forecaster.fit(y[:train_length], forecasting_horizon=5)

    y_pred = forecaster.predict(forecasting_horizon=predict_fh)
    pred_values = y_pred["value"].to_numpy()

    expected = analytical_forecast(
        series_type="linear",
        params={"slope": slope, "intercept": intercept},
        horizon=predict_fh,
        last_index=train_length - 1,
    )

    np.testing.assert_allclose(pred_values, expected, atol=1e-9)


@pytest.mark.integration
@pytest.mark.parametrize(
    "fit_fh,predict_fh",
    [
        (1, 5),
        (2, 7),
        (3, 10),
        (5, 15),
    ],
)
def test_horizon_mismatch_constant_series(constant_series, fit_fh, predict_fh):
    """Verify horizon mismatch maintains constant for constant series."""
    value = 50.0
    y = constant_series(value=value, length=100)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
    )
    forecaster.fit(y[:80], forecasting_horizon=fit_fh)

    y_pred = forecaster.predict(forecasting_horizon=predict_fh)
    pred_values = y_pred["value"].to_numpy()

    expected = np.full(predict_fh, value)
    np.testing.assert_allclose(pred_values, expected, atol=1e-10)


@pytest.mark.integration
def test_zero_slope_linear(linear_series):
    """Verify zero-slope linear series (equivalent to constant)."""
    y = linear_series(slope=0.0, intercept=42.0, length=100)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
    )
    forecaster.fit(y[:80], forecasting_horizon=5)

    y_pred = forecaster.predict()
    pred_values = y_pred["value"].to_numpy()

    expected = np.full(5, 42.0)
    np.testing.assert_allclose(pred_values, expected, atol=1e-10)


@pytest.mark.integration
def test_negative_slope_linear(linear_series, analytical_forecast):
    """Verify negative slope linear trend."""
    slope = -2.0
    intercept = 100.0
    train_length = 40
    y = linear_series(slope=slope, intercept=intercept, length=60)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
    )
    forecaster.fit(y[:train_length], forecasting_horizon=5)

    y_pred = forecaster.predict()
    pred_values = y_pred["value"].to_numpy()

    expected = analytical_forecast(
        series_type="linear",
        params={"slope": slope, "intercept": intercept},
        horizon=5,
        last_index=train_length - 1,
    )

    np.testing.assert_allclose(pred_values, expected, atol=1e-10)


@pytest.mark.integration
def test_ar1_near_unit_root(ar1_series, analytical_forecast):
    """Verify AR(1) with phi very close to 1 (near unit root)."""
    phi = 0.999
    c = 0.1
    train_length = 200
    y = ar1_series(phi=phi, c=c, length=220, noise_std=0.0)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
        feature_transformer=LagTransformer(lag=[1]),
    )
    forecaster.fit(y[:train_length], forecasting_horizon=3)

    y_pred = forecaster.predict()
    pred_values = y_pred["value"].to_numpy()

    last_obs = y["value"][train_length - 1]
    expected = analytical_forecast(
        series_type="ar1",
        params={"phi": phi, "c": c},
        horizon=3,
        last_obs=last_obs,
    )

    np.testing.assert_allclose(pred_values, expected, atol=1e-6, rtol=1e-4)


@pytest.mark.integration
def test_ar1_negative_phi_oscillation(ar_p_series):
    """Verify AR(1) with negative phi produces oscillating predictions."""
    phi = -0.7
    c = 5.0
    # Use ar_p_series (starts at c=5, far from stationary mean ≈2.94)
    # with short training so the series hasn't converged
    train_length = 10
    y = ar_p_series(coeffs=[phi], c=c, length=25, noise_std=0.0)

    # No feature_transformer: internal tabularization creates
    # features = y_t, target = y_{t+1} = phi*y_t + c
    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
    )
    forecaster.fit(y[:train_length], forecasting_horizon=4)

    y_pred = forecaster.predict()
    pred_values = y_pred["value"].to_numpy()

    # Check that predictions oscillate (sign changes)
    diffs = np.diff(pred_values)
    sign_changes = np.sum(diffs[:-1] * diffs[1:] < 0)

    # With negative phi, we expect oscillations (at least 1 sign change in 3 diffs)
    assert sign_changes >= 1, "Expected oscillating predictions with negative phi"


@pytest.mark.integration
@pytest.mark.parametrize("intercept", [0.0, 10.0, -5.0, 100.0])
def test_linear_different_intercepts(linear_series, analytical_forecast, intercept):
    """Verify linear trends with various intercepts."""
    slope = 1.0
    train_length = 50
    y = linear_series(slope=slope, intercept=intercept, length=70)

    forecaster = PointReductionForecaster(
        estimator=LinearRegression(),
    )
    forecaster.fit(y[:train_length], forecasting_horizon=5)

    y_pred = forecaster.predict()
    pred_values = y_pred["value"].to_numpy()

    expected = analytical_forecast(
        series_type="linear",
        params={"slope": slope, "intercept": intercept},
        horizon=5,
        last_index=train_length - 1,
    )

    np.testing.assert_allclose(pred_values, expected, atol=1e-10)
