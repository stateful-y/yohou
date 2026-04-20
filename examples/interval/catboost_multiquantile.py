# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "catboost",
#     "scikit-learn",
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Interval Forecasting with CatBoost MultiQuantile

    CatBoost supports a `MultiQuantile` loss function that predicts **all
    quantiles in a single model**, avoiding the 2N-model overhead of
    standard quantile regression (where N = number of coverage rates).

    [`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.reduction.IntervalReductionForecaster/) automatically detects this loss and
    activates the optimised code path.

    ## What You'll Learn

    - Using `CatBoostRegressor` with `MultiQuantile` loss
    - The speed advantage of multi-quantile vs separate quantile models
    - How `reduction_strategy` interacts with multi-quantile loss
    - Evaluating interval quality with yohou metrics

    ## Prerequisites

    Familiarity with [`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.reduction.IntervalReductionForecaster/).
    """)
    return


@app.cell(hide_code=True)
def _():
    import time

    from catboost import CatBoostRegressor
    from sklearn.model_selection import train_test_split

    from copy import deepcopy

    from yohou.datasets import fetch_tourism_monthly
    from yohou.interval import IntervalReductionForecaster
    from yohou.metrics import EmpiricalCoverage, IntervalScore, MeanIntervalWidth
    from yohou.plotting import (
        plot_forecast,
        plot_score_per_step,
        plot_score_per_vintage,
    )
    from yohou.preprocessing import LagTransformer

    return (
        CatBoostRegressor,
        EmpiricalCoverage,
        IntervalReductionForecaster,
        IntervalScore,
        LagTransformer,
        MeanIntervalWidth,
        deepcopy,
        fetch_tourism_monthly,
        plot_forecast,
        plot_score_per_step,
        plot_score_per_vintage,
        time,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We load the Monthly Tourism dataset (series T1) and split it into training
    and test sets for interval forecasting.
    """)
    return


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
    ## 2. CatBoost MultiQuantile Forecaster

    Pass `CatBoostRegressor(loss_function='MultiQuantile:alpha=...')` to
    [`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.reduction.IntervalReductionForecaster/).  The `alpha` values in the loss function
    are **ignored**, the forecaster rewrites them at fit time to match the
    requested `coverage_rates`.  Any placeholder value is fine.

    Because CatBoost's `MultiQuantile` loss produces a 2D output (one
    column per quantile), it cannot be wrapped in `MultiOutputRegressor`
    and requires `forecasting_horizon=1` with recursive prediction.

    > **Note on `reduction_strategy`**: Because the multi-quantile loss
    > already produces a vector of quantiles as its native output,
    > it is always fitted with `forecasting_horizon=1` and predicts
    > recursively. This means `reduction_strategy` does not change the
    > single-model advantage here - you still get one model for all quantiles.
    > For per-step specialisation with CatBoost intervals, train separate
    > `CatBoostRegressor(loss_function='Quantile:alpha=...')` models using
    > `reduction_strategy="direct"` instead.
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
        forecasting_horizon=1,
        coverage_rates=coverage_rates,
    )
    elapsed_mq = time.perf_counter() - t0

    y_pred_mq = catboost_fc.predict_interval(
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage_rates,
    )

    print(f"MultiQuantile fit time: {elapsed_mq:.2f}s (single model)")
    print(f"Prediction columns: {y_pred_mq.columns}")
    return coverage_rates, y_pred_mq


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) visualises the prediction intervals alongside the training
    history and test data. The `coverage_rates` parameter controls which
    interval bands are shown.
    """)
    return


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

    The default estimator is `MultiOutputRegressor(QuantileRegressor())`, let's compare it with our `MultiOutputRegressor(CatBoostRegressor())`.
    """)
    return


@app.cell
def _(
    IntervalReductionForecaster,
    LagTransformer,
    coverage_rates,
    forecasting_horizon,
    plot_forecast,
    time,
    y_pred_mq,
    y_test,
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

    y_pred_std = standard_fc.predict_interval(
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage_rates,
    )

    plot_forecast(
        y_test,
        {"CatBoost with MultiQuantile loss": y_pred_mq, "Standard quantile regression": y_pred_std},
        y_train=y_train,
        coverage_rates=coverage_rates,
        title="CatBoost MultiQuantile Intervals",
    )
    return (y_pred_std,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Evaluate Interval Quality

    We evaluate both approaches using [`EmpiricalCoverage`](/pages/api/generated/yohou.metrics.interval.EmpiricalCoverage/) (does the interval
    contain the true value at the target rate?), [`IntervalScore`](/pages/api/generated/yohou.metrics.interval.IntervalScore/) (penalises
    width and miscoverage), and [`MeanIntervalWidth`](/pages/api/generated/yohou.metrics.interval.MeanIntervalWidth/) (average band width).
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
    ## Multi-vintage Scoring

    The `observe_predict_interval` method with `stride=1` produces one
    interval forecast per observation point, creating multiple *vintages*.
    Each vintage represents a different forecast origin, so you can analyse
    how interval quality evolves as the model absorbs more data.
    """)
    return


@app.cell
def _(catboost_fc, coverage_rates, deepcopy, forecasting_horizon, y_test):
    _vintage_model = deepcopy(catboost_fc)
    y_pred_vintages = _vintage_model.observe_predict_interval(
        y=y_test,
        stride=1,
        forecasting_horizon=forecasting_horizon,
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
def _(vintage_scorer, plot_score_per_step, y_pred_vintages, y_test):
    plot_score_per_step(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="Interval Score per Step",
        y_label="Interval Score",
        height=380,
    )
    return


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

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - Set `loss_function='MultiQuantile:alpha=...'` on `CatBoostRegressor`
    - [`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.reduction.IntervalReductionForecaster/) detects this and fits **one** model
    - The alpha placeholder is overwritten: just specify `coverage_rates`
    - Multi-quantile is faster when many coverage rates are needed
    - Interval quality is comparable to separate quantile models
    - `reduction_strategy` does not change multi-quantile behaviour (always recursive with H=1)

    ## Next Steps

    - **Standard interval reduction**: See [`interval_reduction.py`](/examples/interval/interval_reduction/) for `reduction_strategy` comparison
    - **Point CatBoost**: See [`catboost_forecasting.py`](/examples/point/catboost_forecasting/) for point forecasting with CatBoost
    - **Reduction strategies**: See [`reduction_strategies.py`](/examples/point/reduction_strategies/) for multi-output vs direct vs dir-rec
    """)
    return


if __name__ == "__main__":
    app.run()
