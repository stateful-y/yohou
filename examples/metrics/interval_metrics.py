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
    "title": "Interval Metrics",
    "description": "Evaluate prediction intervals with EmpiricalCoverage, IntervalScore, MeanIntervalWidth, PinballLoss, and CalibrationError across coverage levels.",
}
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

    - [`EmpiricalCoverage`](/pages/api/generated/yohou.metrics.interval.EmpiricalCoverage/): Does the interval contain the actual value at the expected rate?
    - [`IntervalScore`](/pages/api/generated/yohou.metrics.interval.IntervalScore/) (Winkler score): Penalizes wide intervals and miscoverage
    - [`MeanIntervalWidth`](/pages/api/generated/yohou.metrics.interval.MeanIntervalWidth/): Average interval width (narrower = better)
    - [`PinballLoss`](/pages/api/generated/yohou.metrics.interval.PinballLoss/): Quantile-specific loss
    - [`CalibrationError`](/pages/api/generated/yohou.metrics.interval.CalibrationError/): Gap between nominal and empirical coverage
    - Aggregation across coverage rates with `"coveragewise"`

    ## Prerequisites

    Understanding of prediction intervals from `interval/` examples.
    """)


@app.cell(hide_code=True)
def _():
    from sklearn.linear_model import QuantileRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.multioutput import MultiOutputRegressor

    from copy import deepcopy

    from yohou.datasets import fetch_tourism_monthly
    from yohou.interval import IntervalReductionForecaster
    from yohou.metrics import (
        CalibrationError,
        EmpiricalCoverage,
        IntervalScore,
        MeanIntervalWidth,
        PinballLoss,
    )
    from yohou.plotting import (
        plot_calibration,
        plot_forecast,
        plot_score_heatmap,
        plot_score_per_vintage,
        plot_score_time_series,
        plot_time_series,
    )
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
        deepcopy,
        fetch_tourism_monthly,
        plot_calibration,
        plot_forecast,
        plot_score_heatmap,
        plot_score_per_vintage,
        plot_score_time_series,
        plot_time_series,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Generate Interval Predictions

    We fit an [`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.reduction.IntervalReductionForecaster/) that wraps a `QuantileRegressor` with
    a [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/), then produce prediction intervals at multiple coverage
    rates. These intervals serve as input for all the interval metrics below.
    """)


@app.cell
def _(
    IntervalReductionForecaster,
    LagTransformer,
    MultiOutputRegressor,
    QuantileRegressor,
    fetch_tourism_monthly,
    train_test_split,
):
    y = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})

    y_train, y_test = train_test_split(y, test_size=0.2, shuffle=False)
    fh = len(y_test)
    coverage_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 0.95]

    interval_forecaster = IntervalReductionForecaster(
        estimator=MultiOutputRegressor(QuantileRegressor(solver="highs")),
        feature_transformer=LagTransformer(lag=list(range(1, 13))),
    )
    interval_forecaster.fit(y_train, forecasting_horizon=fh, coverage_rates=coverage_rates)
    y_pred_int = interval_forecaster.predict_interval(forecasting_horizon=fh, coverage_rates=coverage_rates)

    # Drop vintage_time: not needed for metrics/plotting
    if "vintage_time" in y_pred_int.columns:
        y_pred_int = y_pred_int.drop("vintage_time")

    print(f"Coverage rates: {coverage_rates}")
    print(f"Prediction columns: {y_pred_int.columns}")
    return coverage_rates, y, y_pred_int, y_test, y_train


@app.cell
def _(plot_time_series, y):
    plot_time_series(y, title="Tourism Monthly")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) renders the prediction intervals as shaded bands around the
    point forecast, with one band per coverage rate. This gives a visual check
    that the intervals are reasonable before computing numeric metrics.
    """)


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
    ec = EmpiricalCoverage(coverage_rates=[0.9])
    ec.fit(y_train)
    coverage_result = ec.score(y_test, y_pred_int)
    print("Empirical Coverage:")
    print("  Nominal → Empirical")
    for rate in coverage_rates:
        print(f"  {rate:.0%}     → result included")
    print(f"\n  Result: {coverage_result}")


@app.cell
def _(
    EmpiricalCoverage,
    coverage_rates,
    plot_score_time_series,
    y_pred_int,
    y_test,
):
    plot_score_time_series(
        EmpiricalCoverage(coverage_rates=coverage_rates),
        y_test,
        y_pred_int,
        title="Empirical Coverage for an Expected Coverage of 90% per Timestep",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. IntervalScore (Winkler Score)

    The interval score rewards narrow intervals and penalizes miscoverage.
    Lower is better. Combines both sharpness and calibration.
    """)


@app.cell
def _(IntervalScore, y_pred_int, y_test, y_train):
    is_scorer = IntervalScore(coverage_rates=[0.9])
    is_scorer.fit(y_train)
    is_result = is_scorer.score(y_test, y_pred_int)
    print(f"Interval Score: {is_result}")


@app.cell
def _(
    IntervalScore,
    coverage_rates,
    plot_score_time_series,
    y_pred_int,
    y_test,
):
    plot_score_time_series(
        IntervalScore(coverage_rates=coverage_rates),
        y_test,
        y_pred_int,
        title="Interval Score per Timestep",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. MeanIntervalWidth

    Simply the average width of the prediction interval.
    Narrower is better **given adequate coverage**.
    """)


@app.cell
def _(MeanIntervalWidth, y_pred_int, y_test, y_train):
    miw = MeanIntervalWidth(coverage_rates=[0.9])
    miw.fit(y_train)
    width_result = miw.score(y_test, y_pred_int)
    print(f"Mean Interval Width: {width_result}")


@app.cell
def _(
    MeanIntervalWidth,
    coverage_rates,
    plot_score_time_series,
    y_pred_int,
    y_test,
):
    plot_score_time_series(
        MeanIntervalWidth(coverage_rates=coverage_rates),
        y_test,
        y_pred_int,
        title="Mean Interval Width per Timestep",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. PinballLoss and CalibrationError

    [`PinballLoss`](/pages/api/generated/yohou.metrics.interval.PinballLoss/) measures quantile-specific accuracy: it penalizes under-prediction
    and over-prediction asymmetrically according to the quantile level. Lower
    values indicate better-calibrated quantile estimates.

    [`CalibrationError`](/pages/api/generated/yohou.metrics.interval.CalibrationError/) computes the absolute gap between nominal coverage rates
    and empirical coverage. A perfectly calibrated model has zero calibration
    error at every coverage level.
    """)


@app.cell
def _(
    CalibrationError,
    PinballLoss,
    coverage_rates,
    y_pred_int,
    y_test,
    y_train,
):
    pb = PinballLoss(coverage_rates=[0.9])
    pb.fit(y_train)
    pinball_result = pb.score(y_test, y_pred_int)
    print(f"Pinball Loss: {pinball_result}")

    ce = CalibrationError(coverage_rates=coverage_rates)
    ce.fit(y_train)
    cal_result = ce.score(y_test, y_pred_int)
    print(f"Calibration Error: {cal_result}")


@app.cell
def _(PinballLoss, coverage_rates, plot_score_time_series, y_pred_int, y_test):
    plot_score_time_series(
        PinballLoss(coverage_rates=coverage_rates),
        y_test,
        y_pred_int,
        title="Pinball Loss per Timestep",
    )


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

    [`plot_calibration`](/pages/api/generated/yohou.plotting.evaluation.plot_calibration/) plots nominal coverage (x-axis) against empirical coverage
    (y-axis). A well-calibrated forecaster produces points along the diagonal.
    Deviations above the diagonal indicate over-coverage (intervals too wide),
    while deviations below indicate under-coverage (intervals too narrow).
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

    - [`EmpiricalCoverage`](/pages/api/generated/yohou.metrics.interval.EmpiricalCoverage/): Checks if intervals achieve target coverage
    - [`IntervalScore`](/pages/api/generated/yohou.metrics.interval.IntervalScore/): Combined sharpness + coverage (lower = better)
    - [`MeanIntervalWidth`](/pages/api/generated/yohou.metrics.interval.MeanIntervalWidth/): Sharpness only (narrower = better)
    - [`PinballLoss`](/pages/api/generated/yohou.metrics.interval.PinballLoss/): Quantile-specific loss
    - [`CalibrationError`](/pages/api/generated/yohou.metrics.interval.CalibrationError/): Gap between nominal and realized coverage
    - Use `aggregation_method="coveragewise"` for per-rate breakdown
    - [`plot_calibration`](/pages/api/generated/yohou.plotting.evaluation.plot_calibration/) visualizes calibration quality
    """)




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-vintage Scoring

    The `observe_predict_interval` method with `stride=1` produces one
    interval forecast per observation point, creating multiple *vintages*.
    Each vintage represents a different forecast origin, so you can analyse
    how interval quality evolves as the model absorbs more data.
    """)
    return


@app.cell
def _(coverage_rates, deepcopy, fh, interval_forecaster, y_test):
    _vintage_model = deepcopy(interval_forecaster)
    y_pred_vintages = _vintage_model.observe_predict_interval(
        y=y_test,
        stride=1,
        forecasting_horizon=fh,
        coverage_rates=coverage_rates,
    )
    print(f"Vintages: {y_pred_vintages['vintage_time'].n_unique()}")
    y_pred_vintages.head(10)
    return (y_pred_vintages,)


@app.cell
def _(IntervalScore, y_train):
    vintage_scorer = IntervalScore()
    vintage_scorer.fit(y_train)
    return (vintage_scorer,)


@app.cell
def _(vintage_scorer, plot_score_per_vintage, y_pred_vintages, y_test):
    plot_score_per_vintage(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="Interval Score per Vintage",
        y_label="Interval Score",
        height=380,
    )
    return


@app.cell
def _(vintage_scorer, plot_score_heatmap, y_pred_vintages, y_test):
    plot_score_heatmap(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="Score Heatmap (Step x Vintage)",
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Point metrics**: See [`point_metrics.py`](/examples/metrics/point_metrics/) for forecast accuracy metrics
    - **Model selection**: See [Model Selection](/examples/#model-selection) for CV with scoring
    - **Time weighting**: See `examples/time_weighted_forecasting.py`
    """)


if __name__ == "__main__":
    app.run()
