"""Interval Forecasting with CatBoost MultiQuantile.

Demonstrates IntervalReductionForecaster using CatBoost's native
multi-quantile loss to fit all quantiles in a single model, instead
of training separate lower/upper models per coverage rate.
"""

import marimo

__generated_with = "0.19.11"
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

        await micropip.install(["plotly", "scikit-learn", "catboost", "yohou"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Interval Forecasting with CatBoost MultiQuantile

    CatBoost supports a `MultiQuantile` loss function that predicts **all
    quantiles in a single model**, avoiding the 2N-model overhead of
    standard quantile regression (where N = number of coverage rates).

    `IntervalReductionForecaster` automatically detects this loss and
    activates the optimised code path.

    ## What You'll Learn

    - Using `CatBoostRegressor` with `MultiQuantile` loss
    - The speed advantage of multi-quantile vs separate quantile models
    - Evaluating interval quality with yohou metrics

    ## Prerequisites

    Familiarity with `IntervalReductionForecaster` (see `interval_reduction.py`).
    """)
    return


@app.cell(hide_code=True)
def _():
    import time

    import polars as pl
    from catboost import CatBoostRegressor
    from sklearn.linear_model import QuantileRegressor
    from sklearn.multioutput import MultiOutputRegressor

    from yohou.datasets import load_air_passengers
    from yohou.interval import IntervalReductionForecaster
    from yohou.metrics import EmpiricalCoverage, IntervalScore, MeanIntervalWidth
    from yohou.plotting import plot_forecast
    from yohou.preprocessing import LagTransformer

    return (
        CatBoostRegressor,
        EmpiricalCoverage,
        IntervalReductionForecaster,
        IntervalScore,
        LagTransformer,
        MeanIntervalWidth,
        MultiOutputRegressor,
        QuantileRegressor,
        load_air_passengers,
        pl,
        plot_forecast,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data
    """)
    return


@app.cell
def _(load_air_passengers):
    y = load_air_passengers().rename({"Passengers": "passengers"})

    split_idx = int(len(y) * 0.8)
    y_train = y.head(split_idx)
    y_test = y.tail(len(y) - split_idx)
    forecasting_horizon = len(y_test)

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    return forecasting_horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. CatBoost MultiQuantile Forecaster

    Pass `CatBoostRegressor(loss_function='MultiQuantile:alpha=...')` to
    `IntervalReductionForecaster`.  The `alpha` values in the loss function
    are **ignored**, the forecaster rewrites them at fit time to match the
    requested `coverage_rates`.  Any placeholder value is fine.
    """)
    return


@app.cell
def _(
    CatBoostRegressor,
    IntervalReductionForecaster,
    LagTransformer,
    forecasting_horizon,
    time,
    y_train,
):
    coverage_rates = [0.5, 0.9]

    catboost_fc = IntervalReductionForecaster(
        estimator=CatBoostRegressor(
            iterations=200,
            depth=4,
            learning_rate=0.1,
            loss_function="MultiQuantile:alpha=0.5",  # placeholder
            verbose=0,
        ),
        feature_transformer=LagTransformer(lag=list(range(1, 13))),
    )

    t0 = time.perf_counter()
    catboost_fc.fit(
        y_train,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage_rates,
    )
    elapsed_mq = time.perf_counter() - t0

    y_pred_mq = catboost_fc.predict_interval(
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage_rates,
    )

    print(f"MultiQuantile fit time: {elapsed_mq:.2f}s (single model)")
    print(f"Prediction columns: {y_pred_mq.columns}")
    return catboost_fc, coverage_rates, elapsed_mq, y_pred_mq


@app.cell
def _(coverage_rates, plot_forecast, y_pred_mq, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_mq,
        y_train=y_train,
        coverage_rates=coverage_rates,
        title="CatBoost MultiQuantile Intervals",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Compare with Standard Quantile Regression

    The standard path trains **2 × len(coverage_rates)** separate models.
    With CatBoost `MultiQuantile`, only **one** model is needed.
    """)
    return


@app.cell
def _(
    IntervalReductionForecaster,
    LagTransformer,
    coverage_rates,
    forecasting_horizon,
    time,
    y_train,
):
    standard_fc = IntervalReductionForecaster(
        feature_transformer=LagTransformer(lag=list(range(1, 13))),
    )

    t0_std = time.perf_counter()
    standard_fc.fit(
        y_train,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage_rates,
    )
    elapsed_std = time.perf_counter() - t0_std

    y_pred_std = standard_fc.predict_interval(
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage_rates,
    )

    print(f"Standard quantile fit time: {elapsed_std:.2f}s ({2 * len(coverage_rates)} models)")
    return elapsed_std, y_pred_std


@app.cell
def _(elapsed_mq, elapsed_std, mo):
    mo.md(
        f"""
        **Timing comparison** (this run):\n
        | Approach | Models | Fit time |
        |---|---|---|
        | CatBoost MultiQuantile | 1 | {elapsed_mq:.2f}s |
        | Standard QuantileRegressor | {4} | {elapsed_std:.2f}s |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Evaluate Interval Quality
    """)
    return


@app.cell
def _(
    EmpiricalCoverage,
    IntervalScore,
    MeanIntervalWidth,
    coverage_rates,
    mo,
    y_pred_mq,
    y_pred_std,
    y_test,
    y_train,
):
    rows = []
    for label, y_pred in [("MultiQuantile", y_pred_mq), ("Standard", y_pred_std)]:
        for scorer_cls in [EmpiricalCoverage, IntervalScore, MeanIntervalWidth]:
            scorer = scorer_cls(coverage_rates=coverage_rates)
            scorer.fit(y_train)
            score = scorer.score(y_test, y_pred)
            rows.append(f"| {label} | {scorer_cls.__name__} | {score:.4f} |")

    table = "| Approach | Metric | Score |\n|---|---|---|\n" + "\n".join(rows)
    mo.md(table)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - Set `loss_function='MultiQuantile:alpha=...'` on `CatBoostRegressor`
    - `IntervalReductionForecaster` detects this and fits **one** model
    - The alpha placeholder is overwritten: just specify `coverage_rates`
    - Multi-quantile is faster when many coverage rates are needed
    - Interval quality is comparable to separate quantile models
    """)
    return


if __name__ == "__main__":
    app.run()
