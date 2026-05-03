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
    "title": "How to Produce Multi-Vintage Predictions",
    "description": "Generate multiple predictions from different weather forecast vintages without refitting, using the X_forecast predict-time override.",
    "category": "how-to",
    "companion": "pages/user-guide/exogenous-howto.md",
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
        # How to Produce Multi-Vintage Predictions

        This notebook shows how to generate multiple predictions from
        different external forecast vintages at the same observation point.
        Each `predict()` call swaps step columns temporarily without
        changing the forecaster's internal state.

        **Prerequisites:** Familiarity with `X_actual`, `X_future`,
        `X_forecast` parameters
        ([Tutorial](/examples/point/exogenous_features/)).
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
    from yohou.plotting import plot_forecast
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
        timedelta,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Prepare Data and Fit

        Set up synthetic electricity prices with temperature dependence
        and fit a forecaster with all three exogenous types.
        """
    )
    return


@app.cell
def _(
    HistGradientBoostingRegressor,
    LagTransformer,
    PointReductionForecaster,
    datetime,
    np,
    pl,
    timedelta,
):
    rng = np.random.default_rng(42)
    n, H = 300, 12
    times = pl.Series("time", [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)])
    t = np.arange(n, dtype=float)
    actual_temp = 15.0 + 5.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, n)

    X_actual = pl.DataFrame({"time": times, "temperature": actual_temp})
    holidays = [1.0 if (datetime(2024, 1, 1) + timedelta(hours=i)).weekday() == 6 else 0.0 for i in range(n)]
    X_future = pl.DataFrame({"time": times, "is_holiday": holidays})
    price = 50.0 + 2.0 * actual_temp + 10.0 * np.array(holidays) + rng.normal(0, 0.1, n)
    y = pl.DataFrame({"time": times, "price": price})

    # Training X_forecast
    forecast_rows = []
    for i in range(H, 250):
        for step in range(1, H + 1):
            if i + step < n:
                forecast_rows.append({
                    "vintage_time": times[i],
                    "time": times[i + step],
                    "wx_temp": float(actual_temp[i + step] + 0.5 + rng.normal(0, 0.3)),
                })

    train_size = 250
    forecaster = PointReductionForecaster(
        estimator=HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=42),
        feature_transformer=LagTransformer([1, 2, 3]),
        reduction_strategy="direct",
    )
    forecaster.fit(
        y=y[:train_size],
        X_actual=X_actual[:train_size],
        forecasting_horizon=H,
        X_future=X_future,
        X_forecast=pl.DataFrame(forecast_rows),
    )
    print(f"Fitted: observed_time_ = {forecaster.observed_time_}")
    return (
        H,
        X_actual,
        X_future,
        actual_temp,
        forecaster,
        holidays,
        n,
        rng,
        t,
        times,
        train_size,
        y,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Create Multiple Weather Vintages

        Simulate five forecast vintages with decreasing bias (later
        vintages are more accurate). All vintages share the same
        observation point.
        """
    )
    return


@app.cell
def _(H, actual_temp, np, pl, times, train_size):
    last_obs = times[train_size - 1]
    test_times = [times[train_size + i] for i in range(H)]

    vintage_configs = [
        ("06:00", 3.0),
        ("07:00", 2.0),
        ("08:00", 1.0),
        ("09:00", 0.5),
        ("09:30", 0.1),
    ]

    vintages = {}
    for _name, _bias in vintage_configs:
        _seed = hash(_name) % (2**31)
        _local_rng = np.random.default_rng(_seed)
        vintages[_name] = pl.DataFrame({
            "vintage_time": [last_obs] * H,
            "time": test_times,
            "wx_temp": [
                float(actual_temp[train_size + i] + _bias + _local_rng.normal(0, 0.2))
                for i in range(H)
            ],
        })

    print(f"Created {len(vintages)} vintages at observation point {last_obs}")
    return last_obs, test_times, vintage_configs, vintages


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Predict with Each Vintage

        Call `predict()` once per vintage. The forecaster swaps step
        columns and restores them after each call.
        """
    )
    return


@app.cell
def _(forecaster, np, pl, vintages, y, train_size, H):
    predictions = {}
    for _name, _X_fc in vintages.items():
        _pred = forecaster.predict(X_forecast=_X_fc)
        predictions[_name] = _pred

    # Compare MAE against actual test prices
    y_test_h = y[train_size : train_size + H]
    actual_prices = y_test_h["price"].to_numpy()

    results = []
    for _name, _pred in predictions.items():
        mae = float(np.mean(np.abs(_pred["price"].to_numpy() - actual_prices)))
        results.append({"vintage": _name, "mae": f"{mae:.3f}"})

    comparison = pl.DataFrame(results)
    print("MAE by vintage (lower is better):")
    comparison
    return actual_prices, comparison, predictions, y_test_h


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Later vintages (smaller bias) produce lower error. The 09:30
        vintage is closest to reality.

        ## 4. Verify State Preservation

        Confirm that the five `predict()` calls did not change the
        forecaster's internal state.
        """
    )
    return


@app.cell
def _(forecaster, last_obs, np, predictions):
    # Bare predict should return the stored baseline
    pred_baseline = forecaster.predict()

    # Re-run 06:00 vintage: same result as before
    pred_rerun = forecaster.predict(X_forecast=vintages["06:00"])
    np.testing.assert_array_almost_equal(
        pred_rerun["price"].to_numpy(),
        predictions["06:00"]["price"].to_numpy(),
        decimal=10,
    )

    assert forecaster.observed_time_ == last_obs
    print("State preserved: observed_time_ unchanged, predictions reproducible")
    return pred_baseline, pred_rerun


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. Observe and Predict Again

        In a production loop, observe new data, then predict with the
        latest vintage.
        """
    )
    return


@app.cell
def _(H, X_actual, X_future, forecaster, np, pl, times, train_size, vintages, y):
    # Observe 12 new hours
    n_new = 12
    forecaster.observe(
        y=y[train_size : train_size + n_new],
        X_actual=X_actual[train_size : train_size + n_new],
    )
    print(f"After observe: observed_time_ = {forecaster.observed_time_}")

    # Create fresh vintage at the new observation point
    new_obs = forecaster.observed_time_
    new_test_times = [times[train_size + n_new + i] for i in range(H)]
    rng_new = np.random.default_rng(999)

    new_vintage = pl.DataFrame({
        "vintage_time": [new_obs] * H,
        "time": new_test_times,
        "wx_temp": [
            float(15.0 + 5.0 * np.sin(2 * np.pi * float(train_size + n_new + i) / 24) + 0.1 + rng_new.normal(0, 0.2))
            for i in range(H)
        ],
    })

    pred_new = forecaster.predict(X_forecast=new_vintage)
    print(f"New prediction: {len(pred_new)} steps from {pred_new['time'].min()}")
    return n_new, new_obs, new_vintage, pred_new, rng_new


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        This notebook demonstrated the production multi-vintage workflow:

        1. Fit with historical X_actual, X_future, and X_forecast
        2. Create multiple forecast vintages at the observation point
        3. Call `predict(X_forecast=...)` per vintage (no state mutation)
        4. Observe new data and repeat

        **See also:**
        [About Exogenous Features](/pages/user-guide/exogenous-features/) ·
        [Exogenous Tutorial](/examples/point/exogenous_features/)
        """
    )
    return


if __name__ == "__main__":
    app.run()
