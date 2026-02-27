# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.19.11"
__gallery__ = {
    "title": "Interval Reduction Forecasting",
    "description": "Build prediction intervals directly via quantile regression with IntervalReductionForecaster, comparing multiple coverage rates and scoring metrics.",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Interval Reduction Forecasting

    [`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.reduction.IntervalReductionForecaster/) produces prediction intervals **directly**
    via quantile regression, without a separate calibration step.
    Each coverage rate trains lower/upper quantile models.

    ## What You'll Learn

    - Using [`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.reduction.IntervalReductionForecaster/) with quantile regressors
    - Specifying `coverage_rates` at fit and predict time
    - Comparing quantile vs. conformal interval approaches
    - Evaluating interval quality

    ## Prerequisites

    Understanding of [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) and prediction intervals.
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.model_selection import train_test_split

    from yohou.datasets import fetch_tourism_monthly
    from yohou.interval import IntervalReductionForecaster
    from yohou.metrics import EmpiricalCoverage, IntervalScore, MeanIntervalWidth
    from yohou.plotting import plot_forecast, plot_model_comparison_bar
    from yohou.preprocessing import LagTransformer

    return (
        EmpiricalCoverage,
        IntervalReductionForecaster,
        IntervalScore,
        LagTransformer,
        MeanIntervalWidth,
        fetch_tourism_monthly,
        pl,
        plot_forecast,
        plot_model_comparison_bar,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We load the Monthly Tourism dataset and split it into training and test sets.
    """)


@app.cell
def _(fetch_tourism_monthly, train_test_split):
    y = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})

    y_train, y_test = train_test_split(y, test_size=0.2, shuffle=False)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) renders the quantile regression intervals as shaded
    bands. Each `coverage_rates` entry produces a separate band.
    """)


@app.cell
def _(coverage_rates, plot_forecast, y_pred_int, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_int,
        y_train=y_train,
        coverage_rates=coverage_rates,
        title="Quantile Regression Intervals",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Evaluating Interval Quality

    We assess how well the prediction intervals capture the true values using interval-specific metrics.
    """)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Multiple Coverage Rates

    You can train and predict with as many coverage rates as needed.
    Each rate adds lower/upper columns to the prediction.
    """)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) overlays all four coverage bands, showing how wider
    rates produce broader intervals.
    """)


@app.cell
def _(many_rates, plot_forecast, y_pred_many, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_many,
        y_train=y_train,
        coverage_rates=many_rates,
        title="Multiple Coverage Rates",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Interval Width per Coverage Rate

    How much wider do intervals get as coverage increases?
    We score [`MeanIntervalWidth`](/pages/api/generated/yohou.metrics.interval.MeanIntervalWidth/) once per rate and compare them
    side-by-side with [`plot_model_comparison_bar`](/pages/api/generated/yohou.plotting.evaluation.plot_model_comparison_bar/).
    """)


@app.cell
def _(MeanIntervalWidth, many_rates, plot_model_comparison_bar, y_pred_many, y_test, y_train):
    width_per_rate = {}
    for rate in many_rates:
        _width_scorer = MeanIntervalWidth(coverage_rates=[rate])
        _width_scorer.fit(y_train)
        width_per_rate[f"{rate:.0%}"] = _width_scorer.score(y_test, y_pred_many)

    plot_model_comparison_bar(
        {"IntervalReduction": width_per_rate},
        group_by="scorer",
        title="Mean Interval Width by Coverage Rate",
        y_label="Width",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - [`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.reduction.IntervalReductionForecaster/) uses quantile regression for native intervals
    - `coverage_rates` are specified at **fit time** (models for each quantile)
    - No separate calibration set needed (unlike conformal prediction)
    - More coverage rates = more quantile models to train
    - Evaluate with [`EmpiricalCoverage`](/pages/api/generated/yohou.metrics.interval.EmpiricalCoverage/), [`IntervalScore`](/pages/api/generated/yohou.metrics.interval.IntervalScore/), [`MeanIntervalWidth`](/pages/api/generated/yohou.metrics.interval.MeanIntervalWidth/)
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Calibration plots**: Use [`plot_calibration`](/pages/api/generated/yohou.plotting.evaluation.plot_calibration/) from `yohou.plotting`
    - **Scoring**: See [Metrics](/examples/#metrics) for comprehensive interval metrics
    """)


if __name__ == "__main__":
    app.run()
