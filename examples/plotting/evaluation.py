"""Model Evaluation Plots.

Demonstrates residual diagnostics, score time series, score distributions,
per-horizon analysis, model comparison bars, and calibration plots.

Datasets: air_passengers
Demonstrates: plot_residual_time_series, plot_score_time_series, plot_score_distribution,
    plot_score_per_horizon, plot_model_comparison_bar, plot_calibration
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
    import sys as _sys

    if "pyodide" in _sys.modules:
        import micropip

        await micropip.install(["plotly", "scikit-learn", "yohou"])
    return


@app.cell(hide_code=True)
def _():
    import polars as pl

    from yohou.datasets import load_air_passengers, load_sunspots
    from yohou.interval import SplitConformalForecaster
    from yohou.metrics import (
        MeanAbsoluteError,
        MeanAbsolutePercentageError,
        RootMeanSquaredError,
    )
    from yohou.plotting import (
        plot_calibration,
        plot_model_comparison_bar,
        plot_residual_time_series,
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
        load_air_passengers,
        pl,
        plot_calibration,
        plot_model_comparison_bar,
        plot_residual_time_series,
        plot_score_distribution,
        plot_score_per_horizon,
        plot_score_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Evaluation Plots

    ## What You'll Learn

    - Diagnosing residual patterns with `plot_residual_time_series`
    - Tracking per-timestep errors with `plot_score_time_series`
    - Examining error distributions with `plot_score_distribution`
    - Analysing horizon degradation with `plot_score_per_horizon`
    - Comparing models side-by-side with `plot_model_comparison_bar`
    - Assessing interval calibration with `plot_calibration`

    ## Prerequisites

    Familiarity with yohou's fit/predict API and scoring system (see
    `examples/quickstart.py` and `examples/scoring.py`).
    """)
    return


@app.cell
def _(PointReductionForecaster, SeasonalNaive, load_air_passengers):
    air = load_air_passengers()
    fh = 24
    y_train = air.head(len(air) - fh)
    y_test = air.tail(fh)

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

    `plot_residual_time_series` creates a 4-panel layout: residuals over time,
    residuals vs fitted, histogram, and Q-Q plot. Residuals are computed
    internally as `y_truth - y_pred`.
    """)
    return


@app.cell
def _(plot_residual_time_series, y_pred_naive, y_test):
    plot_residual_time_series(
        y_pred_naive,
        y_test,
        title="Seasonal Naive -- Residual Diagnostics",
    )
    return


@app.cell
def _(plot_residual_time_series, y_pred_reduction, y_test):
    plot_residual_time_series(
        y_pred_reduction,
        y_test,
        title="Reduction Forecaster -- Residual Diagnostics",
    )
    return


@app.cell
def _(plot_residual_time_series, y_pred_naive, y_test):
    plot_residual_time_series(
        y_pred_naive,
        y_test,
        columns="Passengers",
        title="Naive Residuals -- Single Column",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Score Time Series

    `plot_score_time_series` evaluates per-timestep scorer values. Supports
    single and multi-model comparisons, different scorers, and extensive
    styling via kwargs.
    """)
    return


@app.cell
def _(MeanAbsoluteError, plot_score_time_series, y_pred_naive, y_test):
    plot_score_time_series(
        MeanAbsoluteError(),
        y_test,
        y_pred_naive,
        title="MAE Over Time -- Seasonal Naive",
    )
    return


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
        title="MAE Over Time -- Model Comparison",
    )
    return


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
        title="RMSE Over Time -- With Markers",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Score Distribution

    `plot_score_distribution` renders histograms (or KDE) of per-timestep
    scores. Toggle **kind**, **show_mean**, **show_zero**, and **n_bins**.
    """)
    return


@app.cell
def _(MeanAbsoluteError, plot_score_distribution, y_pred_naive, y_test):
    plot_score_distribution(
        MeanAbsoluteError(),
        y_test,
        y_pred_naive,
        kind="histogram",
        n_bins=10,
        title="MAE Distribution -- Histogram (default)",
    )
    return


@app.cell
def _(MeanAbsoluteError, plot_score_distribution, y_pred_naive, y_test):
    plot_score_distribution(
        MeanAbsoluteError(),
        y_test,
        y_pred_naive,
        kind="kde",
        title="MAE Distribution -- KDE",
    )
    return


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
        title="MAE Distribution -- Both, Custom Bins, No Zero Line",
    )
    return


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
        title="RMSE Distribution -- 20 Bins",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Score per Horizon

    `plot_score_per_horizon` shows how forecast quality degrades at each
    step of the prediction window. Switch between **line** and **bar**
    kind, and overlay a **trend** line.
    """)
    return


@app.cell
def _(MeanAbsoluteError, plot_score_per_horizon, y_pred_naive, y_test):
    plot_score_per_horizon(
        MeanAbsoluteError(),
        y_test,
        y_pred_naive,
        kind="line",
        title="MAE by Horizon -- Line",
    )
    return


@app.cell
def _(MeanAbsoluteError, plot_score_per_horizon, y_pred_naive, y_test):
    plot_score_per_horizon(
        MeanAbsoluteError(),
        y_test,
        y_pred_naive,
        kind="bar",
        title="MAE by Horizon -- Bar",
    )
    return


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
        title="MAE by Horizon -- Multi-Model with Trend",
    )
    return


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
        title="RMSE by Horizon -- Bar Comparison",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Model Comparison Bar Chart

    `plot_model_comparison_bar` creates a grouped bar chart from a dict of
    model-scorer results. Vary **group_by**, **orientation**, and **sort_by**.
    """)
    return


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
        title="Model Comparison -- Group by Scorer (default)",
    )
    return


@app.cell
def _(plot_model_comparison_bar, score_results):
    plot_model_comparison_bar(
        score_results,
        group_by="model",
        title="Model Comparison -- Group by Model",
    )
    return


@app.cell
def _(plot_model_comparison_bar, score_results):
    plot_model_comparison_bar(
        score_results,
        group_by="scorer",
        orientation="horizontal",
        title="Model Comparison -- Horizontal Bars",
    )
    return


@app.cell
def _(plot_model_comparison_bar, score_results):
    plot_model_comparison_bar(
        score_results,
        group_by="scorer",
        sort_by="Naive",
        ascending=False,
        title="Model Comparison -- Sorted by Naive Descending",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Calibration Plot

    `plot_calibration` assesses whether prediction intervals are well-calibrated.
    Points near the diagonal indicate accurate uncertainty quantification.
    """)
    return


@app.cell
def _(SeasonalNaive, SplitConformalForecaster, load_air_passengers, pl):
    air_calib = load_air_passengers()
    fh_calib = 12
    y_train_calib = air_calib.head(len(air_calib) - fh_calib)
    y_test_calib = air_calib.tail(fh_calib)

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
        columns="Passengers",
        title="Calibration -- Multiple Coverage Rates",
    )
    return


@app.cell
def _(plot_calibration, y_pred_calib, y_test_calib):
    plot_calibration(
        y_pred_calib,
        y_test_calib,
        coverage_rates=[0.5, 0.8, 0.9, 0.95],
        columns="Passengers",
        x_label="Nominal",
        y_label="Empirical",
        title="Calibration -- Custom Axis Labels",
    )
    return


@app.cell
def _(plot_calibration, y_pred_calib, y_test_calib):
    plot_calibration(
        y_pred_calib,
        y_test_calib,
        coverage_rates=[0.5, 0.8, 0.9, 0.95],
        columns="Passengers",
        reference_dash="dot",
        reference_color="#6366f1",
        title="Calibration -- Custom Reference Line",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **plot_residual_time_series** takes `y_pred` and `y_truth`, computing residuals internally; produces a 4-panel diagnostic layout for single columns or faceted residuals for multiple
    - **plot_score_time_series** reveals temporal patterns in forecast errors; pass a dict for multi-model comparison
    - **plot_score_distribution** supports `kind="histogram"`, `"kde"`, or `"both"`; `show_mean` and `show_zero` add reference lines
    - **plot_score_per_horizon** shows error degradation as horizon increases; `show_trend=True` overlays a linear fit
    - **plot_model_comparison_bar** accepts `group_by="scorer"` or `"model"` and supports `orientation="horizontal"`
    - **plot_calibration** compares nominal vs empirical coverage; points near the diagonal indicate well-calibrated intervals

    ## Next Steps

    - **Forecasting**: See `examples/plotting/forecasting_visualization.py` for forecast plots, intervals, and comparison modes
    - **Cross-validation**: See `examples/plotting/model_selection.py` for train/test split visualization
    - **Similarity**: See `examples/plotting/similarity_heatmap.py` for distance-based interval weights
    - **Signal processing**: See `examples/plotting/signal_processing.py` for spectrum and phase analysis
    """)
    return


if __name__ == "__main__":
    app.run()
