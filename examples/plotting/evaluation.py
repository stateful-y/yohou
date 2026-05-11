# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.23.1"
__gallery__ = {
    "title": "Evaluation Plots",
    "description": "Residual diagnostics, per-timestep score tracking, model comparison bar charts, calibration plots, and score distribution analysis for forecast evaluation.",
    "category": "how-to",
    "companion": "/pages/explanation/visualization/#evaluating-model-quality",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():

    from yohou.datasets import fetch_tourism_monthly
    from yohou.interval import SplitConformalForecaster
    from yohou.metrics import (
        MeanAbsoluteError,
        MeanAbsolutePercentageError,
        RootMeanSquaredError,
    )
    from yohou.plotting import (
        plot_calibration,
        plot_group_scores,
        plot_residuals,
        plot_score_distribution,
        plot_score_heatmap,
        plot_score_per_step,
        plot_score_per_vintage,
        plot_score_summary,
        plot_score_time_series,
    )
    from yohou.point import PointReductionForecaster, SeasonalNaive

    return (
        MeanAbsoluteError,
        MeanAbsolutePercentageError,
        PointReductionForecaster,
        RootMeanSquaredError,
        SeasonalNaive,
        SplitConformalForecaster,
        fetch_tourism_monthly,
        plot_calibration,
        plot_group_scores,
        plot_residuals,
        plot_score_distribution,
        plot_score_heatmap,
        plot_score_per_step,
        plot_score_per_vintage,
        plot_score_summary,
        plot_score_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Evaluation Plots

    ## What You'll Learn

    - Diagnosing residual patterns with [`plot_residuals`](/pages/api/generated/yohou.plotting.evaluation.plot_residuals/)
    - Tracking per-timestep errors with [`plot_score_time_series`](/pages/api/generated/yohou.plotting.evaluation.plot_score_time_series/)
    - Examining error distributions with [`plot_score_distribution`](/pages/api/generated/yohou.plotting.evaluation.plot_score_distribution/)
    - Analysing horizon degradation with [`plot_score_per_step`](/pages/api/generated/yohou.plotting.evaluation.plot_score_per_step/)
    - Comparing models with [`plot_score_summary`](/pages/api/generated/yohou.plotting.evaluation.plot_score_summary/)
    - Tracking accuracy across vintages with [`plot_score_per_vintage`](/pages/api/generated/yohou.plotting.evaluation.plot_score_per_vintage/)
    - Inspecting step-vintage interactions with [`plot_score_heatmap`](/pages/api/generated/yohou.plotting.evaluation.plot_score_heatmap/)
    - Per-group panel scoring with [`plot_group_scores`](/pages/api/generated/yohou.plotting.evaluation.plot_group_scores/)
    - Assessing interval calibration with [`plot_calibration`](/pages/api/generated/yohou.plotting.evaluation.plot_calibration/)

    ## Prerequisites

    Familiarity with yohou's fit/predict API and scoring system (see
    `examples/quickstart.py` and `examples/scoring.py`).
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prepare Models and Predictions

    We load the Monthly Tourism dataset via [`fetch_tourism_monthly`](/pages/api/generated/yohou.datasets._fetchers.fetch_tourism_monthly/), then fit two
    point forecasters: [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/) (repeats the last seasonal cycle) and
    [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) (uses an sklearn regressor with lag features).
    Their predictions on the held-out test set are used throughout this notebook.
    """)


@app.cell
def _(PointReductionForecaster, SeasonalNaive, fetch_tourism_monthly):
    tourism = (
        fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    )
    fh = 24
    y_train = tourism.head(len(tourism) - fh)
    y_test = tourism.tail(fh)

    naive = SeasonalNaive(seasonality=12)
    naive.fit(y_train, forecasting_horizon=fh)
    y_pred_naive = naive.predict(forecasting_horizon=fh)

    reduction = PointReductionForecaster()
    reduction.fit(y_train, forecasting_horizon=fh)
    y_pred_reduction = reduction.predict(forecasting_horizon=fh)
    return y_pred_naive, y_pred_reduction, y_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Residual Diagnostics

    [`plot_residuals`](/pages/api/generated/yohou.plotting.evaluation.plot_residuals/) creates a 4-panel layout: residuals over time,
    residuals vs fitted, histogram, and Q-Q plot. Residuals are computed
    internally as `y_truth - y_pred`.
    """)


@app.cell
def _(plot_residuals, y_pred_naive, y_test):
    plot_residuals(
        y_pred_naive,
        y_test,
        title="Seasonal Naive - Residual Diagnostics",
    )


@app.cell
def _(plot_residuals, y_pred_reduction, y_test):
    plot_residuals(
        y_pred_reduction,
        y_test,
        title="Reduction Forecaster - Residual Diagnostics",
    )


@app.cell
def _(plot_residuals, y_pred_naive, y_test):
    plot_residuals(
        y_pred_naive,
        y_test,
        columns="tourists",
        title="Naive Residuals - Single Column",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Score Time Series

    [`plot_score_time_series`](/pages/api/generated/yohou.plotting.evaluation.plot_score_time_series/) evaluates per-timestep scorer values. Supports
    single and multi-model comparisons, different scorers, and extensive
    styling parameters.
    """)


@app.cell
def _(MeanAbsoluteError, plot_score_time_series, y_pred_naive, y_test):
    plot_score_time_series(
        MeanAbsoluteError(),
        y_test,
        y_pred_naive,
        title="MAE Over Time - Seasonal Naive",
    )


@app.cell
def _(
    MeanAbsoluteError,
    plot_score_time_series,
    y_pred_naive,
    y_pred_reduction,
    y_test,
):
    plot_score_time_series(
        MeanAbsoluteError(),
        y_test,
        {"Naive": y_pred_naive, "Reduction": y_pred_reduction},
        title="MAE Over Time - Model Comparison",
    )


@app.cell
def _(
    RootMeanSquaredError,
    plot_score_time_series,
    y_pred_naive,
    y_pred_reduction,
    y_test,
):
    plot_score_time_series(
        RootMeanSquaredError(),
        y_test,
        {"Naive": y_pred_naive, "Reduction": y_pred_reduction},
        show_legend=True,
        show_markers=True,
        title="RMSE Over Time - With Markers",
    )


@app.cell
def _(
    MeanAbsoluteError,
    RootMeanSquaredError,
    plot_score_time_series,
    y_pred_naive,
    y_test,
):
    plot_score_time_series(
        {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()},
        y_test,
        y_pred_naive,
        title="Multi-Scorer Over Time - Naive",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Score Distribution

    [`plot_score_distribution`](/pages/api/generated/yohou.plotting.evaluation.plot_score_distribution/) renders histograms (or KDE) of per-timestep
    scores. Toggle **kind**, **show_mean**, **show_zero**, and **n_bins**.
    """)


@app.cell
def _(MeanAbsoluteError, plot_score_distribution, y_pred_naive, y_test):
    plot_score_distribution(
        MeanAbsoluteError(),
        y_test,
        y_pred_naive,
        kind="histogram",
        n_bins=10,
        title="MAE Distribution - Histogram (default)",
    )


@app.cell
def _(MeanAbsoluteError, plot_score_distribution, y_pred_naive, y_test):
    plot_score_distribution(
        MeanAbsoluteError(),
        y_test,
        y_pred_naive,
        kind="kde",
        title="MAE Distribution - KDE",
    )


@app.cell
def _(
    MeanAbsoluteError,
    plot_score_distribution,
    y_pred_naive,
    y_pred_reduction,
    y_test,
):
    plot_score_distribution(
        MeanAbsoluteError(),
        y_test,
        {"Naive": y_pred_naive, "Reduction": y_pred_reduction},
        kind="both",
        n_bins=15,
        show_mean=True,
        show_zero=False,
        title="MAE Distribution - Both, Custom Bins, No Zero Line",
    )


@app.cell
def _(
    RootMeanSquaredError,
    plot_score_distribution,
    y_pred_naive,
    y_pred_reduction,
    y_test,
):
    plot_score_distribution(
        RootMeanSquaredError(),
        y_test,
        {"Naive": y_pred_naive, "Reduction": y_pred_reduction},
        kind="histogram",
        n_bins=20,
        title="RMSE Distribution - 20 Bins",
    )


@app.cell
def _(
    MeanAbsoluteError,
    RootMeanSquaredError,
    plot_score_distribution,
    y_pred_naive,
    y_test,
):
    plot_score_distribution(
        {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError()},
        y_test,
        y_pred_naive,
        kind="histogram",
        title="Multi-Scorer Distribution",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Score per Horizon

    [`plot_score_per_step`](/pages/api/generated/yohou.plotting.evaluation.plot_score_per_step/) shows how forecast quality degrades at each
    step of the prediction window. Switch between **line** and **bar**
    kind, and overlay a **trend** line.
    """)


@app.cell
def _(MeanAbsoluteError, plot_score_per_step, y_pred_naive, y_test):
    plot_score_per_step(
        MeanAbsoluteError(),
        y_test,
        y_pred_naive,
        kind="line",
        title="MAE by Horizon - Line",
    )


@app.cell
def _(MeanAbsoluteError, plot_score_per_step, y_pred_naive, y_test):
    plot_score_per_step(
        MeanAbsoluteError(),
        y_test,
        y_pred_naive,
        kind="bar",
        title="MAE by Horizon - Bar",
    )


@app.cell
def _(
    MeanAbsoluteError,
    plot_score_per_step,
    y_pred_naive,
    y_pred_reduction,
    y_test,
):
    plot_score_per_step(
        MeanAbsoluteError(),
        y_test,
        {"Naive": y_pred_naive, "Reduction": y_pred_reduction},
        kind="line",
        show_trend=True,
        title="MAE by Horizon - Multi-Model with Trend",
    )


@app.cell
def _(
    RootMeanSquaredError,
    plot_score_per_step,
    y_pred_naive,
    y_pred_reduction,
    y_test,
):
    plot_score_per_step(
        RootMeanSquaredError(),
        y_test,
        {"Naive": y_pred_naive, "Reduction": y_pred_reduction},
        kind="bar",
        title="RMSE by Horizon - Bar Comparison",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Model Comparison Summary

    [`plot_score_summary`](/pages/api/generated/yohou.plotting.evaluation.plot_score_summary/) produces a
    grouped bar chart that collapses horizon steps into a single aggregate
    score per model per scorer.
    """)


@app.cell
def _(
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    RootMeanSquaredError,
    plot_score_summary,
    y_pred_naive,
    y_pred_reduction,
    y_test,
):
    plot_score_summary(
        {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError(), "MAPE": MeanAbsolutePercentageError()},
        y_test,
        {"Naive": y_pred_naive, "Reduction": y_pred_reduction},
        title="Model Comparison",
    )


@app.cell
def _(
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    RootMeanSquaredError,
    plot_score_summary,
    y_pred_naive,
    y_pred_reduction,
    y_test,
):
    plot_score_summary(
        {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError(), "MAPE": MeanAbsolutePercentageError()},
        y_test,
        {"Naive": y_pred_naive, "Reduction": y_pred_reduction},
        sort_ascending=True,
        title="Model Comparison - Sorted Ascending",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Calibration Plot

    [`plot_calibration`](/pages/api/generated/yohou.plotting.evaluation.plot_calibration/) compares nominal coverage rates (the requested interval
    widths) against empirical coverage (the fraction of test points actually
    falling inside each interval). Points near the diagonal indicate that the
    interval forecaster's uncertainty estimates are well calibrated. A
    [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/) wrapping [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/) generates the
    prediction intervals used here.
    """)


@app.cell
def _(SeasonalNaive, SplitConformalForecaster, fetch_tourism_monthly):
    tourism_calib = (
        fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    )
    fh_calib = 12
    y_train_calib = tourism_calib.head(len(tourism_calib) - fh_calib)
    y_test_calib = tourism_calib.tail(fh_calib)

    conformal_calib = SplitConformalForecaster(
        point_forecaster=SeasonalNaive(seasonality=12),
    )
    conformal_calib.fit(y_train_calib, forecasting_horizon=fh_calib)

    coverage_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 0.95]
    y_pred_calib = conformal_calib.predict_interval(
        forecasting_horizon=fh_calib,
        coverage_rates=coverage_rates,
    )
    return coverage_rates, y_pred_calib, y_test_calib


@app.cell
def _(coverage_rates, plot_calibration, y_pred_calib, y_test_calib):
    plot_calibration(
        y_pred_calib,
        y_test_calib,
        coverage_rates=coverage_rates,
        columns="tourists",
        title="Calibration - Multiple Coverage Rates",
    )


@app.cell
def _(plot_calibration, y_pred_calib, y_test_calib):
    plot_calibration(
        y_pred_calib,
        y_test_calib,
        coverage_rates=[0.5, 0.8, 0.9, 0.95],
        columns="tourists",
        x_label="Nominal",
        y_label="Empirical",
        title="Calibration - Custom Axis Labels",
    )


@app.cell
def _(plot_calibration, y_pred_calib, y_test_calib):
    plot_calibration(
        y_pred_calib,
        y_test_calib,
        coverage_rates=[0.5, 0.8, 0.9, 0.95],
        columns="tourists",
        reference_dash="dot",
        reference_color="#6366f1",
        title="Calibration - Custom Reference Line",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Score per Vintage

    [`plot_score_per_vintage`](/pages/api/generated/yohou.plotting.evaluation.plot_score_per_vintage/) tracks how forecast accuracy evolves across
    successive forecast origins (vintages). This requires multi-vintage predictions
    produced by [`observe_predict`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster.observe_predict/).
    """)


@app.cell
def _(SeasonalNaive, fetch_tourism_monthly):
    from copy import deepcopy

    tourism_v = (
        fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    )
    fh_v = 12
    y_train_v = tourism_v.head(len(tourism_v) - fh_v)
    y_test_v = tourism_v.tail(fh_v)

    naive_v = SeasonalNaive(seasonality=12)
    naive_v.fit(y_train_v, forecasting_horizon=fh_v)
    y_pred_v = deepcopy(naive_v).observe_predict(y_test_v, forecasting_horizon=fh_v, stride=1)
    return y_pred_v, y_test_v


@app.cell
def _(MeanAbsoluteError, plot_score_per_vintage, y_pred_v, y_test_v):
    plot_score_per_vintage(
        MeanAbsoluteError(),
        y_test_v,
        y_pred_v,
        kind="line",
        show_trend=True,
        title="MAE per Vintage - Line with Trend",
    )


@app.cell
def _(MeanAbsoluteError, plot_score_per_vintage, y_pred_v, y_test_v):
    plot_score_per_vintage(
        MeanAbsoluteError(),
        y_test_v,
        y_pred_v,
        kind="bar",
        title="MAE per Vintage - Bar",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Score Heatmap

    [`plot_score_heatmap`](/pages/api/generated/yohou.plotting.evaluation.plot_score_heatmap/) renders a 2D heatmap showing scores across two
    forecast dimensions (e.g. horizon step vs vintage). Each cell encodes
    the score for that specific combination, revealing patterns such as
    accuracy degrading at longer horizons or later vintages.
    """)


@app.cell
def _(MeanAbsoluteError, plot_score_heatmap, y_pred_v, y_test_v):
    plot_score_heatmap(
        MeanAbsoluteError(),
        y_test_v,
        y_pred_v,
        x_dim="step",
        y_dim="vintage",
        title="MAE Heatmap - Step vs Vintage",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Group Scores

    [`plot_group_scores`](/pages/api/generated/yohou.plotting.evaluation.plot_group_scores/) breaks down scorer performance by panel group
    (columns with ``group__member`` naming). Supports `kind="bar"` for
    aggregate comparisons, `kind="box"` for score distributions, and
    `kind="heatmap"` for a 2D overview.
    """)


@app.cell
def _(PointReductionForecaster, SeasonalNaive, fetch_tourism_monthly):
    tourism_panel = fetch_tourism_monthly(n_series=4).frame.drop_nulls()
    fh_panel = 12
    y_train_panel = tourism_panel.head(len(tourism_panel) - fh_panel)
    y_test_panel = tourism_panel.tail(fh_panel)

    naive_panel = SeasonalNaive(seasonality=12)
    naive_panel.fit(y_train_panel, forecasting_horizon=fh_panel)
    y_pred_panel_naive = naive_panel.predict(forecasting_horizon=fh_panel)

    red_panel = PointReductionForecaster()
    red_panel.fit(y_train_panel, forecasting_horizon=fh_panel)
    y_pred_panel_red = red_panel.predict(forecasting_horizon=fh_panel)
    return y_pred_panel_naive, y_pred_panel_red, y_test_panel


@app.cell
def _(
    MeanAbsoluteError,
    plot_group_scores,
    y_pred_panel_naive,
    y_pred_panel_red,
    y_test_panel,
):
    plot_group_scores(
        MeanAbsoluteError(),
        y_test_panel,
        {"Naive": y_pred_panel_naive, "Reduction": y_pred_panel_red},
        kind="bar",
        title="Groupwise MAE - Bar",
    )


@app.cell
def _(MeanAbsoluteError, plot_group_scores, y_pred_panel_naive, y_test_panel):
    plot_group_scores(
        MeanAbsoluteError(),
        y_test_panel,
        y_pred_panel_naive,
        kind="box",
        title="Groupwise MAE Distribution - Box",
    )


@app.cell
def _(
    MeanAbsoluteError,
    plot_group_scores,
    y_pred_panel_naive,
    y_pred_panel_red,
    y_test_panel,
):
    plot_group_scores(
        MeanAbsoluteError(),
        y_test_panel,
        {"Naive": y_pred_panel_naive, "Reduction": y_pred_panel_red},
        kind="heatmap",
        title="Groupwise MAE - Heatmap",
    )
