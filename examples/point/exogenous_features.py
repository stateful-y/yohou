# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Exogenous Features (X_actual, X_future, X_forecast)",
    "description": "Build a forecasting model with actual observations, known-future indicators, and multi-vintage external forecasts on synthetic electricity price data.",
    "category": "tutorial",
    "companion": "pages/user-guide/exogenous-tutorial.md",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Exogenous Features: X_actual, X_future, X_forecast

        In this notebook, we will build a forecasting model that uses all three
        types of exogenous features on synthetic electricity price data. We will
        fit with actual temperature, holiday calendars, and weather forecasts,
        then produce predictions from two different forecast vintages and run
        a walk-forward evaluation.

        **Prerequisites:** Basic familiarity with yohou's fit/predict API
        ([Quickstart](/examples/quickstart/)).
        """
    )
    return


@app.cell(hide_code=True)
def _():
    from datetime import datetime, timedelta

    import numpy as np
    import polars as pl
    from sklearn.ensemble import HistGradientBoostingRegressor

    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        HistGradientBoostingRegressor,
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        datetime,
        np,
        pl,
        plot_forecast,
        plot_time_series,
        timedelta,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Create the Synthetic Data

        We create hourly electricity prices with a known linear relationship:
        `price = 50 + 2·temperature + 10·is_holiday + noise`. This makes
        the model's job transparent.
        """
    )
    return


@app.cell
def _(datetime, np, pl, timedelta):
    rng = np.random.default_rng(42)
    n = 200
    H = 6

    times = pl.Series(
        "time", [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
    )
    t = np.arange(n, dtype=float)
    actual_temp = 15.0 + 5.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, n)

    # X_actual: realized temperature readings
    X_actual = pl.DataFrame({"time": times, "temperature": actual_temp})

    # X_future: holiday indicator (Sundays = 1.0)
    holidays = [
        1.0 if (datetime(2024, 1, 1) + timedelta(hours=i)).weekday() == 6 else 0.0
        for i in range(n)
    ]
    X_future = pl.DataFrame({"time": times, "is_holiday": holidays})

    # y: price with known linear relationship
    price = 50.0 + 2.0 * actual_temp + 10.0 * np.array(holidays) + rng.normal(0, 0.1, n)
    y = pl.DataFrame({"time": times, "price": price})

    print(f"Dataset: {n} hourly observations, forecast horizon H={H}")
    y.head()
    return H, X_actual, X_future, actual_temp, holidays, n, rng, t, times, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Create Weather Forecasts (X_forecast)

        External forecasts carry a `vintage_time` column. We create one vintage
        per training observation, each covering the next H steps. The forecasts
        have a small systematic bias (0.5°C) compared to actuals.
        """
    )
    return


@app.cell
def _(H, actual_temp, n, np, pl, rng, times):
    forecast_rows = []
    for i in range(H, n):
        for step in range(1, H + 1):
            if i + step < n:
                forecast_rows.append({
                    "vintage_time": times[i],
                    "time": times[i + step],
                    "wx_temp": float(
                        actual_temp[i + step] + 0.5 + rng.normal(0, 0.3)
                    ),
                })
    X_forecast = pl.DataFrame(forecast_rows)
    print(f"X_forecast: {len(X_forecast)} rows, {X_forecast['vintage_time'].n_unique()} vintages")
    X_forecast.head()
    return (X_forecast,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Fit the Forecaster

        We use `PointReductionForecaster` with the `"direct"` strategy and
        `HistGradientBoostingRegressor` (handles nulls from partial forecast
        coverage). All three exogenous types go to `fit()`.
        """
    )
    return


@app.cell
def _(
    H,
    HistGradientBoostingRegressor,
    LagTransformer,
    PointReductionForecaster,
    X_actual,
    X_forecast,
    X_future,
    y,
):
    train_size = 160
    y_train, y_test = y[:train_size], y[train_size:]
    X_actual_train = X_actual[:train_size]
    X_actual_test = X_actual[train_size:]

    forecaster = PointReductionForecaster(
        estimator=HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=42),
        feature_transformer=LagTransformer([1, 2, 3]),
        reduction_strategy="direct",
    )

    forecaster.fit(
        y=y_train,
        X_actual=X_actual_train,
        forecasting_horizon=H,
        X_future=X_future,
        X_forecast=X_forecast,
    )

    print(f"Fitted with {len(forecaster._step_column_names_)} step columns")
    print(f"Step columns: {sorted(forecaster._step_column_names_)}")
    return X_actual_test, X_actual_train, forecaster, train_size, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice that both `is_holiday` and `wx_temp` were converted to
        step-indexed columns (`is_holiday_step_1..6`, `wx_temp_step_1..6`).
        These sit alongside the lag features from `LagTransformer` in the
        internal feature matrix.

        ## 4. Predict with Multiple Vintages

        We create two weather forecast vintages at the test boundary: one
        accurate (small bias) and one deliberately wrong (large bias).
        Each `predict()` call swaps step columns temporarily without
        mutating the forecaster's state.
        """
    )
    return


@app.cell
def _(H, actual_temp, np, pl, times, train_size):
    last_obs = times[train_size - 1]
    test_times = [times[train_size + i] for i in range(H)]

    X_forecast_accurate = pl.DataFrame({
        "vintage_time": [last_obs] * H,
        "time": test_times,
        "wx_temp": [
            float(actual_temp[train_size + i] + 0.1 + np.random.default_rng(100).normal(0, 0.1))
            for i in range(H)
        ],
    })

    X_forecast_biased = pl.DataFrame({
        "vintage_time": [last_obs] * H,
        "time": test_times,
        "wx_temp": [
            float(actual_temp[train_size + i] + 5.0 + np.random.default_rng(200).normal(0, 0.1))
            for i in range(H)
        ],
    })

    print("Created two test vintages (accurate + biased)")
    return X_forecast_accurate, X_forecast_biased, last_obs, test_times


@app.cell
def _(X_forecast_accurate, X_forecast_biased, forecaster, np):
    pred_accurate = forecaster.predict(X_forecast=X_forecast_accurate)
    pred_biased = forecaster.predict(X_forecast=X_forecast_biased)

    print("Accurate vintage prices:", [f"{v:.1f}" for v in pred_accurate["price"].to_list()])
    print("Biased vintage prices:  ", [f"{v:.1f}" for v in pred_biased["price"].to_list()])
    print(f"\nMax difference: {np.max(np.abs(pred_accurate['price'].to_numpy() - pred_biased['price'].to_numpy())):.2f}")
    return pred_accurate, pred_biased


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The predictions differ because the weather forecasts differ. The
        accurate vintage produces prices closer to truth. Importantly, calling
        `predict()` again with the same vintage returns identical results,
        confirming no internal state mutation occurred.
        """
    )
    return


@app.cell
def _(X_forecast_accurate, forecaster, np, pred_accurate):
    # Verify that calling predict with the same vintage again gives identical results
    pred_again = forecaster.predict(X_forecast=X_forecast_accurate)

    np.testing.assert_array_almost_equal(
        pred_again["price"].to_numpy(),
        pred_accurate["price"].to_numpy(),
        decimal=10,
    )
    print("Repeated predict with same vintage: identical results confirmed")
    return (pred_again,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. Walk-Forward Evaluation

        The `observe_predict` loop handles all three parameter types.
        `X_actual` is observed at each step (through the transformer),
        `X_future` and `X_forecast` provide step-indexed columns for
        prediction. We build forecast vintages covering the test period.
        """
    )
    return


@app.cell
def _(
    H,
    MeanAbsoluteError,
    X_actual_test,
    X_future,
    actual_temp,
    forecaster,
    n,
    np,
    pl,
    rng,
    times,
    train_size,
    y_test,
    y_train,
):
    import copy

    # Build X_forecast vintages covering the test period
    test_forecast_rows = []
    for _i in range(train_size, n):
        for _step in range(1, H + 1):
            if _i + _step < n:
                test_forecast_rows.append({
                    "vintage_time": times[_i],
                    "time": times[_i + _step],
                    "wx_temp": float(
                        actual_temp[_i + _step] + 0.5 + rng.normal(0, 0.3)
                    ),
                })
    X_forecast_test = pl.DataFrame(test_forecast_rows)

    eval_forecaster = copy.deepcopy(forecaster)
    preds = eval_forecaster.observe_predict(
        y=y_test,
        X_actual=X_actual_test,
        X_future=X_future,
        X_forecast=X_forecast_test,
        stride=H,
    )

    scorer = MeanAbsoluteError()
    scorer.fit(y_train)
    n_preds = min(len(preds), len(y_test))
    score = scorer.score(y_test[:n_preds], preds[:n_preds])
    print(f"Walk-forward MAE: {score:.4f}")
    print(f"Predictions generated: {len(preds)} rows")
    return n_preds, preds, score, scorer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6. Visualize the Results

        We compare the accurate and biased vintage predictions against
        the actual test prices.
        """
    )
    return


@app.cell
def _(plot_forecast, pred_accurate, y_test, y_train):
    plot_forecast(
        y_test[:6],
        pred_accurate,
        y_train=y_train[-48:],
        title="Electricity Price Forecast (Accurate Weather Vintage)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        We built a forecasting model that:

        - Uses temperature readings (`X_actual`) for lag-based features
        - Incorporates holiday calendars (`X_future`) as step-indexed features
        - Accepts weather forecast vintages (`X_forecast`) with `vintage_time`
        - Produces different predictions for different vintages without state mutation
        - Runs walk-forward evaluation with proper data separation

        **Next steps:**
        [Multi-Vintage Forecasting](/examples/point/multi_vintage_forecasting/) ·
        [About Exogenous Features](/pages/user-guide/exogenous-features/)
        """
    )
    return


if __name__ == "__main__":
    app.run()
