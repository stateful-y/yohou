# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Evaluation Plots",
    "description": "Residual diagnostics, per-timestep score tracking, model comparison bar charts, calibration plots, and score distribution analysis for forecast evaluation.",
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
        plot_model_comparison_bar,
        plot_residuals,
        plot_score_distribution,
        plot_score_per_horizon,
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
        plot_model_comparison_bar,
        plot_residuals,
        plot_score_distribution,
        plot_score_per_horizon,
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
    - Analysing horizon degradation with [`plot_score_per_horizon`](/pages/api/generated/yohou.plotting.evaluation.plot_score_per_horizon/)
    - Comparing models side-by-side with [`plot_model_comparison_bar`](/pages/api/generated/yohou.plotting.evaluation.plot_model_comparison_bar/)
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
    return y_pred_naive, y_pred_reduction, y_test, y_train


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Score per Horizon

    [`plot_score_per_horizon`](/pages/api/generated/yohou.plotting.evaluation.plot_score_per_horizon/) shows how forecast quality degrades at each
    step of the prediction window. Switch between **line** and **bar**
    kind, and overlay a **trend** line.
    """)


@app.cell
def _(MeanAbsoluteError, plot_score_per_horizon, y_pred_naive, y_test):
    plot_score_per_horizon(
        MeanAbsoluteError(),
        y_test,
        y_pred_naive,
        kind="line",
        title="MAE by Horizon - Line",
    )


@app.cell
def _(MeanAbsoluteError, plot_score_per_horizon, y_pred_naive, y_test):
    plot_score_per_horizon(
        MeanAbsoluteError(),
        y_test,
        y_pred_naive,
        kind="bar",
        title="MAE by Horizon - Bar",
    )


@app.cell
def _(
    MeanAbsoluteError,
    plot_score_per_horizon,
    y_pred_naive,
    y_pred_reduction,
    y_test,
):
    plot_score_per_horizon(
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
    plot_score_per_horizon,
    y_pred_naive,
    y_pred_reduction,
    y_test,
):
    plot_score_per_horizon(
        RootMeanSquaredError(),
        y_test,
        {"Naive": y_pred_naive, "Reduction": y_pred_reduction},
        kind="bar",
        title="RMSE by Horizon - Bar Comparison",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Model Comparison Bar Chart

    [`plot_model_comparison_bar`](/pages/api/generated/yohou.plotting.evaluation.plot_model_comparison_bar/) creates a grouped bar chart from a dictionary
    mapping model names to scorer-name/score-value pairs. Use **group_by** to
    switch between grouping bars by `"scorer"` (default) or `"model"`,
    **orientation** to flip to horizontal bars, and **sort_by** to order bars
    by a specific model's scores.
    """)


@app.cell(hide_code=True)
def _(
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    RootMeanSquaredError,
    y_pred_naive,
    y_pred_reduction,
    y_test,
    y_train,
):
    mae = MeanAbsoluteError().fit(y_train)
    rmse = RootMeanSquaredError().fit(y_train)
    mape = MeanAbsolutePercentageError().fit(y_train)

    score_results = {
        "Naive": {
            "MAE": mae.score(y_test, y_pred_naive),
            "RMSE": rmse.score(y_test, y_pred_naive),
            "MAPE": mape.score(y_test, y_pred_naive),
        },
        "Reduction": {
            "MAE": mae.score(y_test, y_pred_reduction),
            "RMSE": rmse.score(y_test, y_pred_reduction),
            "MAPE": mape.score(y_test, y_pred_reduction),
        },
    }
    return (score_results,)


@app.cell
def _(plot_model_comparison_bar, score_results):
    plot_model_comparison_bar(
        score_results,
        group_by="scorer",
        title="Model Comparison - Group by Scorer (default)",
    )


@app.cell
def _(plot_model_comparison_bar, score_results):
    plot_model_comparison_bar(
        score_results,
        group_by="model",
        title="Model Comparison - Group by Model",
    )


@app.cell
def _(plot_model_comparison_bar, score_results):
    plot_model_comparison_bar(
        score_results,
        group_by="scorer",
        orientation="horizontal",
        title="Model Comparison - Horizontal Bars",
    )


@app.cell
def _(plot_model_comparison_bar, score_results):
    plot_model_comparison_bar(
        score_results,
        group_by="scorer",
        sort_by="Naive",
        ascending=False,
        title="Model Comparison - Sorted by Naive Descending",
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
    ## Key Takeaways

    - **plot_residuals** takes `y_pred` and `y_truth`, computing residuals internally; produces a 4-panel diagnostic layout for single columns or faceted residuals for multiple
    - **plot_score_time_series** reveals temporal patterns in forecast errors; pass a dict for multi-model comparison
    - **plot_score_distribution** supports `kind="histogram"`, `"kde"`, or `"both"`; `show_mean` and `show_zero` add reference lines
    - **plot_score_per_horizon** shows error degradation as horizon increases; `show_trend=True` overlays a linear fit
    - **plot_model_comparison_bar** accepts `group_by="scorer"` or `"model"` and supports `orientation="horizontal"`
    - **plot_calibration** compares nominal vs empirical coverage; points near the diagonal indicate well-calibrated intervals

    ## Next Steps

    - **Forecasting**: See [`examples/plotting/forecasting_visualization.py`](/examples/plotting/forecasting_visualization/) for forecast plots, intervals, and comparison modes
    - **Cross-validation**: See [`examples/plotting/model_selection.py`](/examples/plotting/model_selection/) for train/test split visualization
    - **Similarity**: See `examples/plotting/similarity_heatmap.py` for distance-based interval weights
    - **Signal processing**: See [`examples/plotting/signal_processing.py`](/examples/plotting/signal_processing/) for spectrum and phase analysis
    """)


if __name__ == "__main__":
    app.run()
