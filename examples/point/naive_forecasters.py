"""Naive Forecasters as Baselines.

Demonstrates SeasonalNaive for baseline forecasting with update/predict workflow.
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
async def _():
    import sys

    if "pyodide" in sys.modules:
        import micropip

        await micropip.install(["plotly", "scikit-learn", "yohou"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Naive Forecasters as Baselines

    Every forecasting project needs a **baseline** to judge model quality.
    Naive methods repeat past patterns with no learning, yet they are surprisingly
    hard to beat on many datasets.

    ## What You'll Learn

    - Using `SeasonalNaive` to forecast by repeating the last seasonal cycle
    - The `update` / `predict` workflow for rolling evaluation
    - Comparing different seasonality periods
    - Visualizing forecasts with `plot_forecast`

    ## Prerequisites

    Basic understanding of seasonality in time series.
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.model_selection import train_test_split

    from yohou.datasets import load_air_passengers
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import SeasonalNaive

    return (
        MeanAbsoluteError,
        SeasonalNaive,
        load_air_passengers,
        pl,
        plot_forecast,
        plot_time_series,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load and Split Data

    We begin by loading the Air Passengers dataset and splitting it into training and test sets for evaluating the forecasters.
    """)


@app.cell
def _(load_air_passengers, train_test_split):
    y = (
        load_air_passengers()
        .rename({"Passengers": "passengers"})
    )
    y_train, y_test = train_test_split(y, test_size=0.2, shuffle=False)
    forecasting_horizon = 12

    print(f"Train: {len(y_train)} obs, Test: {len(y_test)} obs")
    return forecasting_horizon, y, y_test, y_train


@app.cell
def _(plot_time_series, y):
    plot_time_series(y, title="Air Passengers")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. SeasonalNaive Forecaster

    `SeasonalNaive` repeats the last `seasonality` observations cyclically.
    With `seasonality=12` on monthly data, it predicts each month
    using the same month from the previous year.
    """)


@app.cell
def _(SeasonalNaive, forecasting_horizon, y_train):
    naive = SeasonalNaive(seasonality=12)
    naive.fit(y_train, forecasting_horizon=forecasting_horizon)

    y_pred_naive = naive.predict(forecasting_horizon=forecasting_horizon)
    print(f"Predicted {len(y_pred_naive)} steps ahead")
    y_pred_naive.head()
    return naive, y_pred_naive


@app.cell
def _(MeanAbsoluteError, plot_forecast, y_pred_naive, y_test, y_train):
    mae = MeanAbsoluteError()
    y_test_h = y_test.head(len(y_pred_naive))
    mae.fit(y_train)
    score = mae.score(y_test_h, y_pred_naive)
    print(f"Seasonal Naive MAE: {score:.2f}")

    plot_forecast(
        y_test_h,
        y_pred_naive,
        y_train=y_train,
        title="Seasonal Naive Forecast (seasonality=12)",
    )
    return mae, score, y_test_h


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Comparing Seasonality Periods

    The `seasonality` parameter controls how far back the forecaster looks.
    Let's compare `seasonality=1` (repeat last value), `6`, and `12`.
    """)


@app.cell
def _(MeanAbsoluteError, SeasonalNaive, forecasting_horizon, y_test, y_train):
    results = {}
    for period in [1, 6, 12]:
        model = SeasonalNaive(seasonality=period)
        model.fit(y_train, forecasting_horizon=forecasting_horizon)
        _pred = model.predict(forecasting_horizon=forecasting_horizon)

        scorer = MeanAbsoluteError()
        y_t = y_test.head(len(_pred))
        scorer.fit(y_t)
        results[f"seasonality={period}"] = {
            "mae": scorer.score(y_t, _pred),
            "pred": _pred,
        }
        print(f"seasonality={period:>2d}  MAE: {results[f'seasonality={period}']['mae']:.2f}")
    return (results,)


@app.cell
def _(plot_forecast, results, y_test, y_train):
    preds_dict = {name: r["pred"] for name, r in results.items()}
    plot_forecast(
        y_test.head(12),
        preds_dict,
        y_train=y_train,
        title="Comparing Seasonality Periods",
    )
    return (preds_dict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Update/Predict Workflow

    The `update` method adds new observations without refitting.
    Combined with `predict`, this enables **rolling evaluation** -
    simulating real-world deployment where new data arrives incrementally.
    """)


@app.cell
def _(MeanAbsoluteError, SeasonalNaive, forecasting_horizon, pl, y_test, y_train):
    naive_rolling = SeasonalNaive(seasonality=12)
    naive_rolling.fit(y_train, forecasting_horizon=forecasting_horizon)

    step = forecasting_horizon
    rolling_preds = []

    for i in range(0, len(y_test), step):
        chunk = y_test.slice(i, min(step, len(y_test) - i))
        if len(chunk) == 0:
            break
        _rpred = naive_rolling.predict(forecasting_horizon=len(chunk))
        rolling_preds.append(_rpred)
        naive_rolling.observe(chunk)

    all_preds = pl.concat(rolling_preds)
    y_test_aligned = y_test.head(len(all_preds))

    mae_rolling = MeanAbsoluteError()
    mae_rolling.fit(y_train)
    score_rolling = mae_rolling.score(y_test_aligned, all_preds)

    print(f"Rolling evaluation: {len(rolling_preds)} windows")
    print(f"Rolling MAE: {score_rolling:.2f}")
    return all_preds, mae_rolling, naive_rolling, score_rolling


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - `SeasonalNaive` repeats last season's values - a strong baseline for seasonal data
    - `seasonality=12` matches monthly patterns (12 months = 1 year)
    - `update()` adds new observations without refitting the model
    - Always compare ML forecasters against naive baselines to prove added value
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Reduction forecasting**: See `reduction_forecaster.py` for ML-based approach
    - **Scoring**: See `metrics/` for comprehensive evaluation metrics
    - **Cross-validation**: See `model_selection/` for temporal CV strategies
    """)


if __name__ == "__main__":
    app.run()
