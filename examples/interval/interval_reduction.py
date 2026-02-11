"""Interval Reduction Forecasting with Quantile Regression.

Demonstrates IntervalReductionForecaster for native quantile-based intervals.
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

        await micropip.install(["plotly", "scikit-learn", "yohou"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Interval Reduction Forecasting

    `IntervalReductionForecaster` produces prediction intervals **directly**
    via quantile regression, without a separate calibration step.
    Each coverage rate trains lower/upper quantile models.

    ## What You'll Learn

    - Using `IntervalReductionForecaster` with quantile regressors
    - Specifying `coverage_rates` at fit and predict time
    - Comparing quantile vs. conformal interval approaches
    - Evaluating interval quality

    ## Prerequisites

    Understanding of `PointReductionForecaster` and prediction intervals.
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import QuantileRegressor
    from sklearn.multioutput import MultiOutputRegressor

    from yohou.datasets import load_air_passengers
    from yohou.interval import IntervalReductionForecaster
    from yohou.metrics import EmpiricalCoverage, IntervalScore, MeanIntervalWidth
    from yohou.plotting import plot_forecast
    from yohou.preprocessing import LagTransformer

    return (
        EmpiricalCoverage,
        IntervalReductionForecaster,
        IntervalScore,
        LagTransformer,
        MeanIntervalWidth,
        load_air_passengers,
        pl,
        plot_forecast,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We load the Air Passengers dataset and split it into training and test sets.
    """)
    return


@app.cell
def _(load_air_passengers):
    y = (
        load_air_passengers()
        .rename({"Passengers": "passengers"})
    )

    split_idx = int(len(y) * 0.8)
    y_train = y.head(split_idx)
    y_test = y.tail(len(y) - split_idx)
    forecasting_horizon = len(y_test)

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    return forecasting_horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. IntervalReductionForecaster Basics

    The default estimator is `MultiOutputRegressor(QuantileRegressor())`.
    `coverage_rates` are specified at **fit time** to train the needed quantile models.
    """)
    return


@app.cell
def _(
    IntervalReductionForecaster,
    LagTransformer,
    forecasting_horizon,
    y_train,
):
    coverage_rates = [0.5, 0.9]

    interval_fc = IntervalReductionForecaster(
        feature_transformer=LagTransformer(lag=list(range(1, 13))),
    )

    interval_fc.fit(
        y_train,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage_rates,
    )

    y_pred_int = interval_fc.predict_interval(
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage_rates,
    )

    print(f"Prediction columns: {y_pred_int.columns}")
    y_pred_int.head()
    return coverage_rates, y_pred_int


@app.cell
def _(coverage_rates, plot_forecast, y_pred_int, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_int,
        y_train=y_train,
        coverage_rates=coverage_rates,
        title="Quantile Regression Intervals",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Evaluating Interval Quality

    We assess how well the prediction intervals capture the true values using interval-specific metrics.
    """)
    return


@app.cell
def _(
    EmpiricalCoverage,
    IntervalScore,
    MeanIntervalWidth,
    coverage_rates,
    y_pred_int,
    y_test,
    y_train,
):
    for _scorer_cls in [EmpiricalCoverage, IntervalScore, MeanIntervalWidth]:
        scorer = _scorer_cls(coverage_rates=coverage_rates)
        scorer.fit(y_train)
        score = scorer.score(y_test, y_pred_int)
        print(f"{_scorer_cls.__name__}: {score}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Multiple Coverage Rates

    You can train and predict with as many coverage rates as needed.
    Each rate adds lower/upper columns to the prediction.
    """)
    return


@app.cell
def _(
    IntervalReductionForecaster,
    LagTransformer,
    forecasting_horizon,
    y_train,
):
    many_rates = [0.5, 0.8, 0.9, 0.95]

    interval_fc_many = IntervalReductionForecaster(
        feature_transformer=LagTransformer(lag=list(range(1, 13))),
    )
    interval_fc_many.fit(
        y_train,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=many_rates,
    )

    y_pred_many = interval_fc_many.predict_interval(
        forecasting_horizon=forecasting_horizon,
        coverage_rates=many_rates,
    )

    print(f"Columns for {len(many_rates)} coverage rates:")
    for col in y_pred_many.columns:
        print(f"  {col}")
    return many_rates, y_pred_many


@app.cell
def _(many_rates, plot_forecast, y_pred_many, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_many,
        y_train=y_train,
        coverage_rates=many_rates,
        title="Multiple Coverage Rates",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - `IntervalReductionForecaster` uses quantile regression for native intervals
    - `coverage_rates` are specified at **fit time** (models for each quantile)
    - No separate calibration set needed (unlike conformal prediction)
    - More coverage rates = more quantile models to train
    - Evaluate with `EmpiricalCoverage`, `IntervalScore`, `MeanIntervalWidth`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Conformal intervals**: See `conformal_forecasting.py` for distribution-free intervals
    - **Calibration plots**: Use `plot_calibration` from `yohou.plotting`
    - **Scoring**: See `metrics/` for comprehensive interval metrics
    """)
    return


if __name__ == "__main__":
    app.run()
