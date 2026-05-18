# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "scikit-learn",
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Exogenous Features (X_actual, X_future, X_forecast)",
    "description": "Build a forecasting model with actual observations, known-future indicators, and multi-vintage external forecasts on synthetic electricity price data.",
    "category": "tutorial",
    "companion": "/pages/tutorials/exogenous-features/",
    "section": "getting-started",
    "api_references": ["PointReductionForecaster", "make_exogenous_regression"],
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
        a walk-forward evaluation using
        [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/).

        **Prerequisites:** Basic familiarity with yohou's fit/predict API
        ([Quickstart](/examples/quickstart/)).
        """
    )


@app.cell(hide_code=True)
def _():
    import numpy as np
    import polars as pl
    from sklearn.ensemble import HistGradientBoostingRegressor

    from yohou.datasets import make_exogenous_regression
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        HistGradientBoostingRegressor,
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        make_exogenous_regression,
        np,
        pl,
        plot_forecast,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Load the Synthetic Data

        `make_exogenous_regression()` generates hourly electricity prices
        with a known linear relationship:
        `price = 50 + 2·temperature + 10·is_holiday + noise`, plus
        weather forecasts with a small systematic bias.
        """
    )


@app.cell
def _(make_exogenous_regression):
    data = make_exogenous_regression(n_samples=200, forecasting_horizon=6)
    y = data.y
    X_actual = data.X_actual
    X_future = data.X_future
    X_forecast = data.X_forecast
    H = 6

    print(f"Dataset: {len(y)} hourly observations, forecast horizon H={H}")
    print(f"X_forecast: {len(X_forecast)} rows, {X_forecast['vintage_time'].n_unique()} vintages")
    y.head()
    return H, X_actual, X_forecast, X_future, data, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Fit the Forecaster

        We use [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) with the `"direct"` strategy and
        `HistGradientBoostingRegressor` (handles nulls from partial forecast
        coverage). All three exogenous types go to `fit()`.
        """
    )


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
    from yohou.model_selection import train_test_split

    y_train, y_test, X_actual_train, X_actual_test = train_test_split(
        y, X_actual, test_size=40
    )
    train_size = len(y_train)

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
        These sit alongside the lag features from [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/) in the
        internal feature matrix.

        ## 3. Predict with Multiple Vintages

        We create two weather forecast vintages at the test boundary: one
        accurate (small bias) and one deliberately wrong (large bias).
        Each `predict()` call swaps step columns temporarily without
        mutating the forecaster's state.
        """
    )


@app.cell
def _(H, X_actual, np, pl, train_size, y):
    _times = y["time"]
    _actual_temp = X_actual["temperature"].to_numpy()
    last_obs = _times[train_size - 1]
    test_times = [_times[train_size + i] for i in range(H)]

    X_forecast_accurate = pl.DataFrame({
        "vintage_time": [last_obs] * H,
        "time": test_times,
        "wx_temp": [
            float(_actual_temp[train_size + i] + 0.1 + np.random.default_rng(100).normal(0, 0.1)) for i in range(H)
        ],
    })

    X_forecast_biased = pl.DataFrame({
        "vintage_time": [last_obs] * H,
        "time": test_times,
        "wx_temp": [
            float(_actual_temp[train_size + i] + 5.0 + np.random.default_rng(200).normal(0, 0.1)) for i in range(H)
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
    print(
        f"\nMax difference: {np.max(np.abs(pred_accurate['price'].to_numpy() - pred_biased['price'].to_numpy())):.2f}"
    )
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
        ## 4. Walk-Forward Evaluation

        The `observe_predict` loop handles all three parameter types.
        `X_actual` is observed at each step (through the transformer),
        `X_future` and `X_forecast` provide step-indexed columns for
        prediction. We build forecast vintages covering the test period.
        """
    )


@app.cell
def _(
    H,
    MeanAbsoluteError,
    X_actual,
    X_actual_test,
    X_forecast,
    X_future,
    forecaster,
    np,
    pl,
    train_size,
    y,
    y_test,
    y_train,
):
    import copy

    # Build X_forecast vintages covering the test period
    _times = y["time"]
    _actual_temp = X_actual["temperature"].to_numpy()
    _rng = np.random.default_rng(42)
    _n = len(y)
    test_forecast_rows = []
    for _i in range(train_size, _n):
        for _step in range(1, H + 1):
            if _i + _step < _n:
                test_forecast_rows.append({
                    "vintage_time": _times[_i],
                    "time": _times[_i + _step],
                    "wx_temp": float(_actual_temp[_i + _step] + 0.5 + _rng.normal(0, 0.3)),
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
        ## 5. Visualize the Results

        We compare the accurate and biased vintage predictions against
        the actual test prices.
        """
    )


@app.cell
def _(plot_forecast, pred_accurate, y_test, y_train):
    plot_forecast(
        y_test[:6],
        pred_accurate,
        y_train=y_train[-48:],
        title="Electricity Price Forecast (Accurate Weather Vintage)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [Exogenous Features tutorial](/pages/tutorials/exogenous-features/) for the companion walkthrough
        - [Multi-Vintage Forecasting](/examples/forecasting-models/multi_vintage_forecasting/) for production multi-vintage workflows
        - [About Exogenous Features](/pages/explanation/exogenous-features/) for conceptual background
        """
    )


if __name__ == "__main__":
    app.run()
