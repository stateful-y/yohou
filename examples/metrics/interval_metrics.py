# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Interval Metrics for Prediction Interval Evaluation.

Demonstrates interval scorers: EmpiricalCoverage, IntervalScore, MeanIntervalWidth,
PinballLoss, and CalibrationError.
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Interval Metrics for Prediction Interval Evaluation

    Prediction intervals need specialized metrics beyond point accuracy.
    Yohou provides 5 interval scorers that evaluate coverage, width, and calibration.

    ## What You'll Learn

    - `EmpiricalCoverage`: Does the interval contain the actual value at the expected rate?
    - `IntervalScore` (Winkler score): Penalizes wide intervals and miscoverage
    - `MeanIntervalWidth`: Average interval width (narrower = better)
    - `PinballLoss`: Quantile-specific loss
    - `CalibrationError`: Gap between nominal and empirical coverage
    - Aggregation across coverage rates with `"coveragewise"`

    ## Prerequisites

    Understanding of prediction intervals from `interval/` examples.
    """)

@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import QuantileRegressor
    from sklearn.multioutput import MultiOutputRegressor

    from yohou.datasets import fetch_tourism_monthly
    from yohou.interval import IntervalReductionForecaster
    from yohou.metrics import (
        CalibrationError,
        EmpiricalCoverage,
        IntervalScore,
        MeanIntervalWidth,
        PinballLoss,
    )
    from yohou.plotting import plot_calibration, plot_forecast
    from yohou.preprocessing import LagTransformer

    return (
        CalibrationError,
        EmpiricalCoverage,
        IntervalReductionForecaster,
        IntervalScore,
        LagTransformer,
        MeanIntervalWidth,
        MultiOutputRegressor,
        PinballLoss,
        QuantileRegressor,
        fetch_tourism_monthly,
        pl,
        plot_calibration,
        plot_forecast,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Generate Interval Predictions

    We fit an interval forecaster and produce prediction intervals to evaluate with the metrics below.
    """)

@app.cell
def _(
    IntervalReductionForecaster,
    LagTransformer,
    MultiOutputRegressor,
    QuantileRegressor,
    fetch_tourism_monthly,
):
    y = (
        fetch_tourism_monthly()
        .frame.select("time", "T1__tourists").drop_nulls()
        .rename({"T1__tourists": "tourists"})
    )

    y_train = y.head(115)
    y_test = y.tail(29)
    fh = len(y_test)
    coverage_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 0.95]

    interval_forecaster = IntervalReductionForecaster(
        estimator=MultiOutputRegressor(QuantileRegressor(solver="highs")),
        feature_transformer=LagTransformer(lag=list(range(1, 13))),
    )
    interval_forecaster.fit(
        y_train, forecasting_horizon=fh, coverage_rates=coverage_rates
    )
    y_pred_int = interval_forecaster.predict_interval(
        forecasting_horizon=fh, coverage_rates=coverage_rates
    )

    # Drop observed_time: not needed for metrics/plotting
    if "observed_time" in y_pred_int.columns:
        y_pred_int = y_pred_int.drop("observed_time")

    print(f"Coverage rates: {coverage_rates}")
    print(f"Prediction columns: {y_pred_int.columns}")
    return coverage_rates, y_pred_int, y_test, y_train

@app.cell
def _(coverage_rates, plot_forecast, y_pred_int, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_int,
        y_train=y_train,
        coverage_rates=coverage_rates,
        title="Intervals for Evaluation",
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. EmpiricalCoverage

    Measures the fraction of actual values falling within the interval.
    A 90% interval should contain ~90% of observations.
    """)

@app.cell
def _(EmpiricalCoverage, coverage_rates, y_pred_int, y_test, y_train):
    ec = EmpiricalCoverage(coverage_rates=coverage_rates)
    ec.fit(y_train)
    coverage_result = ec.score(y_test, y_pred_int)
    print("Empirical Coverage:")
    print("  Nominal → Empirical")
    for rate in coverage_rates:
        print(f"  {rate:.0%}     → result included")
    print(f"\n  Result: {coverage_result}")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. IntervalScore (Winkler Score)

    The interval score rewards narrow intervals and penalizes miscoverage.
    Lower is better. Combines both sharpness and calibration.
    """)

@app.cell
def _(IntervalScore, coverage_rates, y_pred_int, y_test, y_train):
    is_scorer = IntervalScore(coverage_rates=coverage_rates)
    is_scorer.fit(y_train)
    is_result = is_scorer.score(y_test, y_pred_int)
    print(f"Interval Score: {is_result}")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. MeanIntervalWidth

    Simply the average width of the prediction interval.
    Narrower is better **given adequate coverage**.
    """)

@app.cell
def _(MeanIntervalWidth, coverage_rates, y_pred_int, y_test, y_train):
    miw = MeanIntervalWidth(coverage_rates=coverage_rates)
    miw.fit(y_train)
    width_result = miw.score(y_test, y_pred_int)
    print(f"Mean Interval Width: {width_result}")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. PinballLoss and CalibrationError

    We examine two additional metrics that assess quantile accuracy and coverage reliability.
    """)

@app.cell
def _(CalibrationError, PinballLoss, coverage_rates, y_pred_int, y_test, y_train):
    pb = PinballLoss(coverage_rates=coverage_rates)
    pb.fit(y_train)
    pinball_result = pb.score(y_test, y_pred_int)
    print(f"Pinball Loss: {pinball_result}")

    ce = CalibrationError(coverage_rates=coverage_rates)
    ce.fit(y_train)
    cal_result = ce.score(y_test, y_pred_int)
    print(f"Calibration Error: {cal_result}")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Aggregation Methods

    Like point scorers, interval scorers support aggregation.
    `"coveragewise"` gives per-coverage-rate scores.
    """)

@app.cell
def _(EmpiricalCoverage, coverage_rates, y_pred_int, y_test, y_train):
    ec_cw = EmpiricalCoverage(
        coverage_rates=coverage_rates,
        aggregation_method="coveragewise",
    )
    ec_cw.fit(y_train)
    cw_result = ec_cw.score(y_test, y_pred_int)
    print("Coverage-wise Empirical Coverage:")
    print(cw_result)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Calibration Plot

    `plot_calibration` shows nominal vs. empirical coverage, ideal is the diagonal.
    """)

@app.cell
def _(coverage_rates, plot_calibration, y_pred_int, y_test):
    plot_calibration(
        y_pred_int,
        y_test,
        coverage_rates=coverage_rates,
        columns="tourists",
        title="Calibration: Nominal vs Empirical Coverage",
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - `EmpiricalCoverage`: Checks if intervals achieve target coverage
    - `IntervalScore`: Combined sharpness + coverage (lower = better)
    - `MeanIntervalWidth`: Sharpness only (narrower = better)
    - `PinballLoss`: Quantile-specific loss
    - `CalibrationError`: Gap between nominal and realized coverage
    - Use `aggregation_method="coveragewise"` for per-rate breakdown
    - `plot_calibration` visualizes calibration quality
    """)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Point metrics**: See `point_metrics.py` for forecast accuracy metrics
    - **Model selection**: See `model_selection/` for CV with scoring
    - **Time weighting**: See `examples/time_weighted_forecasting.py`
    """)

if __name__ == "__main__":
    app.run()
