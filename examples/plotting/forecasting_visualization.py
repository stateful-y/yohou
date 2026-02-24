# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///
"""Forecast Visualization and Comparison.

Demonstrates forecast plots, decomposition visualization, and time weight
plots with varied parameter combinations.

Datasets: tourism_monthly, sunspots
Demonstrates: plot_forecast, plot_components, plot_time_weight
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)

@app.cell(hide_code=True)
def _():
    import polars as pl

    from yohou.compose import DecompositionPipeline
    from yohou.datasets import fetch_sunspot, fetch_tourism_monthly
    from yohou.interval import SplitConformalForecaster
    from yohou.plotting import (
        plot_components,
        plot_forecast,
        plot_time_weight,
    )
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.stationarity import (
        PatternSeasonalityForecaster,
        PolynomialTrendForecaster,
    )
    from yohou.utils.weighting import (
        exponential_decay_weight,
        linear_decay_weight,
    )

    return (
        DecompositionPipeline,
        PatternSeasonalityForecaster,
        PointReductionForecaster,
        PolynomialTrendForecaster,
        SeasonalNaive,
        SplitConformalForecaster,
        exponential_decay_weight,
        linear_decay_weight,
        fetch_tourism_monthly,
        fetch_sunspot,
        pl,
        plot_components,
        plot_forecast,
        plot_time_weight,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Forecast Visualization and Comparison

    ## What You'll Learn

    - Plotting forecasts from single and multiple models with `plot_forecast`
    - Visualizing decomposition components with `plot_components`
    - Rendering time weight functions with `plot_time_weight`

    ## Prerequisites

    Basic familiarity with yohou's fit/predict API (see `examples/quickstart.py`).
    """)

@app.cell
def _(fetch_tourism_monthly):
    tourism = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    fh = 12
    y_train = tourism.head(len(tourism) - fh)
    y_test = tourism.tail(fh)
    return fh, y_test, y_train

@app.cell
def _(PointReductionForecaster, SeasonalNaive, fh, y_train):
    naive = SeasonalNaive(seasonality=12)
    naive.fit(y_train, forecasting_horizon=fh)
    y_pred_naive = naive.predict(forecasting_horizon=fh)

    reduction = PointReductionForecaster()
    reduction.fit(y_train, forecasting_horizon=fh)
    y_pred_reduction = reduction.predict(forecasting_horizon=fh)
    return y_pred_naive, y_pred_reduction

@app.cell
def _(SeasonalNaive, SplitConformalForecaster, fh, fetch_sunspot):
    sunspots = fetch_sunspot().frame
    ss_train = sunspots.head(len(sunspots) - fh)
    ss_test = sunspots.tail(fh)

    conformal = SplitConformalForecaster(
        point_forecaster=SeasonalNaive(seasonality=132),
    )
    conformal.fit(ss_train, forecasting_horizon=fh)
    y_pred_conformal = conformal.predict_interval(
        forecasting_horizon=fh,
        coverage_rates=[0.9],
    )
    return ss_test, ss_train, y_pred_conformal

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Single Forecast Plot

    `plot_forecast` displays actuals vs predictions. Pass **y_train** to include
    history, and use **n_history** to trim the visible window.
    """)

@app.cell
def _(plot_forecast, y_pred_naive, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_naive,
        y_train=y_train,
        title="Seasonal Naive -- Full History",
    )

@app.cell
def _(plot_forecast, y_pred_naive, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_naive,
        y_train=y_train,
        n_history=50,
        title="Seasonal Naive -- Last 50 Steps of History",
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Multi-Model Forecast

    Pass a **dict** of predictions to overlay multiple models. For interval
    forecasters, include **coverage_rates** to display prediction bands.
    Toggle **show_transition** to control the training/test boundary marker.
    """)

@app.cell
def _(plot_forecast, y_pred_naive, y_pred_reduction, y_test):
    plot_forecast(
        y_test,
        {"Naive": y_pred_naive, "Reduction": y_pred_reduction},
        title="Multi-Model Comparison (Overlay)",
    )

@app.cell
def _(plot_forecast, ss_test, ss_train, y_pred_conformal):
    plot_forecast(
        ss_test,
        y_pred_conformal,
        y_train=ss_train,
        coverage_rates=[0.9],
        n_history=120,
        title="Conformal Forecaster -- 90% PI on Sunspot",
    )

@app.cell
def _(plot_forecast, y_pred_naive, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_naive,
        y_train=y_train,
        show_transition=False,
        title="Seasonal Naive -- No Transition Marker",
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Decomposition Visualization

    `plot_components` displays a fitted `DecompositionPipeline`'s components
    as stacked subplots. Toggle **show_original** and pass a subset of components.
    """)

@app.cell
def _(
    DecompositionPipeline,
    PatternSeasonalityForecaster,
    PolynomialTrendForecaster,
    fh,
    y_train,
):
    decomp = DecompositionPipeline(
        forecasters=[
            ("trend", PolynomialTrendForecaster(degree=2)),
            ("seasonality", PatternSeasonalityForecaster(seasonality=12)),
        ],
        store_residuals=True,
    )
    decomp.fit(y_train, forecasting_horizon=fh)

    decomp_components = {}
    for _name, _fc, *_ in decomp.forecasters_:
        decomp_components[_name] = _fc.predict(forecasting_horizon=fh)
    return (decomp_components,)

@app.cell
def _(decomp_components, fh, plot_components, y_test):
    plot_components(
        y_test.tail(fh),
        decomp_components,
        show_original=True,
        title="Decomposition -- Trend + Seasonality with Original",
    )

@app.cell
def _(decomp_components, fh, plot_components, y_train):
    plot_components(
        y_train.tail(fh),
        decomp_components,
        show_original=False,
        title="Decomposition -- Components Only",
    )

@app.cell
def _(decomp_components, fh, plot_components, y_test):
    plot_components(
        y_test.tail(fh),
        {"trend": decomp_components["trend"]},
        title="Decomposition -- Trend Only",
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Time Weight Visualization

    `plot_time_weight` renders weighting functions over time. Toggle the
    **fill** kwarg for an area chart and adjust **fill_opacity**.
    """)

@app.cell(hide_code=True)
def _(exponential_decay_weight, linear_decay_weight, pl, y_train):
    _times = y_train["time"]
    _exp_fn = exponential_decay_weight(half_life=24)
    _lin_fn = linear_decay_weight()

    exp_weight_df = pl.DataFrame({"time": _times, "time_weight": _exp_fn(_times)})
    lin_weight_df = pl.DataFrame({"time": _times, "time_weight": _lin_fn(_times)})
    return exp_weight_df, lin_weight_df

@app.cell
def _(exp_weight_df, plot_time_weight):
    plot_time_weight(
        exp_weight_df,
        fill=True,
        title="Exponential Decay Weight (half_life=24)",
    )

@app.cell
def _(lin_weight_df, plot_time_weight):
    plot_time_weight(
        lin_weight_df,
        fill=False,
        title="Linear Decay Weight -- Line Only",
    )

@app.cell
def _(lin_weight_df, plot_time_weight):
    plot_time_weight(
        lin_weight_df,
        fill=True,
        fill_opacity=0.5,
        title="Linear Decay Weight -- Semi-Transparent Fill",
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **plot_forecast** handles single models, multi-model dicts, and prediction intervals via `coverage_rates`; `n_history` trims the visible training window
    - **plot_components** stacks components vertically; `show_original` toggles whether the raw series appears as the first panel
    - **plot_time_weight** visualizes weighting schemes; `fill=True` shows the area under the curve, `fill_opacity` controls transparency

    ## Next Steps

    - **Model evaluation**: See `examples/plotting/evaluation.py` for residual diagnostics, scoring distributions, and calibration plots
    - **Exploration**: See `examples/plotting/exploration.py` for rolling statistics and missing data audits
    - **Cross-validation**: See `examples/plotting/model_selection.py` for CV split visualization and hyperparameter search results
    """)

if __name__ == "__main__":
    app.run()
