# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Decomposition Variations.

Demonstrates DecompositionPipeline configurations: two-component,
three-component, with target_transformer, and panel decomposition.
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
    # Decomposition Variations

    `DecompositionPipeline` fits sequential forecasters where each
    subsequent forecaster models the residual of the previous one.
    Final predictions are **summed** across all components.

    ## What You'll Learn

    - Two-component: trend + residual
    - Three-component: trend + seasonality + residual
    - `target_transformer` for pre-transformation (e.g., log)
    - `store_residuals=True` for inspecting intermediate residuals
    - Panel data decomposition
    """)
    return

@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import DecompositionPipeline
    from yohou.datasets import fetch_sunspot, fetch_tourism_quarterly
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer
    from yohou.stationarity import (
        LogTransformer,
        PatternSeasonalityForecaster,
        PolynomialTrendForecaster,
    )

    return (
        DecompositionPipeline,
        LagTransformer,
        LogTransformer,
        MeanAbsoluteError,
        PatternSeasonalityForecaster,
        PointReductionForecaster,
        PolynomialTrendForecaster,
        Ridge,
        SeasonalNaive,
        fetch_sunspot,
        fetch_tourism_quarterly,
        pl,
        plot_forecast,
        plot_time_series,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Univariate Data
    """)
    return

@app.cell
def _(fetch_sunspot, mo, pl):
    _raw = fetch_sunspot().frame
    sunspots = _raw.group_by_dynamic("time", every="1mo").agg(pl.col("sunspot_number").mean())
    _split = int(len(sunspots) * 0.85)
    y_train = sunspots.head(_split)
    y_test = sunspots.tail(len(sunspots) - _split)
    horizon = len(y_test)
    mo.md(f"**Sunspots**: Train={len(y_train)}, Test={len(y_test)}")
    return horizon, sunspots, y_test, y_train

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Two-Component: Trend + Residual
    """)
    return

@app.cell
def _(
    DecompositionPipeline,
    LagTransformer,
    PointReductionForecaster,
    PolynomialTrendForecaster,
    Ridge,
    horizon,
    plot_forecast,
    y_test,
    y_train,
):
    fc_two = DecompositionPipeline(
        forecasters=[
            ("trend", PolynomialTrendForecaster(degree=2)),
            (
                "residual",
                PointReductionForecaster(
                    estimator=Ridge(alpha=1.0),
                    feature_transformer=LagTransformer(lag=[1, 12]),
                ),
            ),
        ],
        store_residuals=True,
    )
    fc_two.fit(y_train, forecasting_horizon=horizon)
    y_pred_two = fc_two.predict(forecasting_horizon=horizon)

    plot_forecast(
        y_test,
        y_pred_two,
        y_train=y_train,
        n_history=60,
        title="Two-Component: Quadratic Trend + Residual",
    )
    return fc_two, y_pred_two

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Three-Component: Trend + Seasonality + Residual
    """)
    return

@app.cell
def _(
    DecompositionPipeline,
    LagTransformer,
    PatternSeasonalityForecaster,
    PointReductionForecaster,
    PolynomialTrendForecaster,
    Ridge,
    horizon,
    plot_forecast,
    y_test,
    y_train,
):
    fc_three = DecompositionPipeline(
        forecasters=[
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("season", PatternSeasonalityForecaster(seasonality=132)),
            (
                "residual",
                PointReductionForecaster(
                    estimator=Ridge(alpha=1.0),
                    feature_transformer=LagTransformer(lag=[1, 12]),
                ),
            ),
        ],
    )
    fc_three.fit(y_train, forecasting_horizon=horizon)
    y_pred_three = fc_three.predict(forecasting_horizon=horizon)

    plot_forecast(
        y_test,
        y_pred_three,
        y_train=y_train,
        n_history=60,
        title="Three-Component: Trend + Season + Residual",
    )
    return fc_three, y_pred_three

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. With target_transformer (Log Scale)

    Apply a log transformation before decomposition, then invert after
    prediction. Useful for data with multiplicative trends.
    """)
    return

@app.cell
def _(
    DecompositionPipeline,
    LagTransformer,
    LogTransformer,
    PointReductionForecaster,
    PolynomialTrendForecaster,
    Ridge,
    horizon,
    plot_forecast,
    y_test,
    y_train,
):
    fc_log = DecompositionPipeline(
        forecasters=[
            ("trend", PolynomialTrendForecaster(degree=1)),
            (
                "residual",
                PointReductionForecaster(
                    estimator=Ridge(alpha=1.0),
                    feature_transformer=LagTransformer(lag=[1, 12]),
                ),
            ),
        ],
        target_transformer=LogTransformer(offset=1.0),
    )
    fc_log.fit(y_train, forecasting_horizon=horizon)
    y_pred_log = fc_log.predict(forecasting_horizon=horizon)

    plot_forecast(
        y_test,
        y_pred_log,
        y_train=y_train,
        n_history=60,
        title="Decomposition with Log Target Transformer",
    )
    return fc_log, y_pred_log

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Compare All Variations
    """)
    return

@app.cell
def _(MeanAbsoluteError, mo, y_pred_log, y_pred_three, y_pred_two, y_test, y_train):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _mae_two = float(_scorer.score(y_test, y_pred_two))
    _mae_three = float(_scorer.score(y_test, y_pred_three))
    _mae_log = float(_scorer.score(y_test, y_pred_log))

    mo.md(
        f"| Variation | MAE |\n"
        f"|-----------|-----|\n"
        f"| Trend + Residual | {_mae_two:.2f} |\n"
        f"| Trend + Season + Residual | {_mae_three:.2f} |\n"
        f"| Log + Trend + Residual | {_mae_log:.2f} |"
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Panel Data Decomposition

    Apply the same pipeline to panel data: each group gets its own
    trend and residual model.
    """)
    return

@app.cell
def _(
    DecompositionPipeline,
    LagTransformer,
    PointReductionForecaster,
    PolynomialTrendForecaster,
    Ridge,
    fetch_tourism_quarterly,
    plot_forecast,
):
    _tourism = fetch_tourism_quarterly().frame
    # Select 8 series with uniform length for a manageable panel demo
    _tourist_cols = [f"T{i}__tourists" for i in range(3, 11)]
    _tourism = _tourism.select("time", *_tourist_cols).drop_nulls()
    _split_p = int(len(_tourism) * 0.8)
    _y_train_p = _tourism.head(_split_p)
    _y_test_p = _tourism.tail(len(_tourism) - _split_p)
    _horizon_p = len(_y_test_p)

    _fc_panel = DecompositionPipeline(
        forecasters=[
            ("trend", PolynomialTrendForecaster(degree=1)),
            (
                "residual",
                PointReductionForecaster(
                    estimator=Ridge(alpha=1.0),
                    feature_transformer=LagTransformer(lag=[1, 4]),
                ),
            ),
        ],
    )
    _fc_panel.fit(_y_train_p, forecasting_horizon=_horizon_p)
    _y_pred_panel = _fc_panel.predict(forecasting_horizon=_horizon_p)

    plot_forecast(
        _y_test_p,
        _y_pred_panel,
        y_train=_y_train_p,
        n_history=8,
        panel_group_names=["T3", "T4", "T5"],
        title="Panel Decomposition: Trend + Residual",
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Predictions are summed** across all components in `DecompositionPipeline`
    - **`store_residuals=True`**: stores intermediate residuals for inspection
    - **`target_transformer`**: applied before decomposition, inverted after prediction
    - **Panel data**: each group gets independent decomposition
    - **Common patterns**:
      - Trend + Residual (simple noise after detrending)
      - Trend + Seasonality + Residual (captures periodic patterns)
      - Log transform + Decomposition (multiplicative to additive conversion)

    ## Next Steps

    - **Pipeline composition**: See `examples/compose/pipeline_composition.py`
    - **Stationarity transforms**: See `examples/stationarity/stationarity_transforms.py`
    - **Panel stationarity**: See `examples/stationarity/panel_stationarity.py`
    """)
    return

if __name__ == "__main__":
    app.run()
