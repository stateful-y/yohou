# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Panel Data Stationarity.

Demonstrates per-group decomposition (trend + seasonality) and stationarity
transformations on panel time series.
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
    # Panel Data Stationarity

    Stationarity transformations and decomposition work per-group on
    panel data. Each group gets its own trend, seasonality, and residual
    extraction.

    ## What You'll Learn

    - Per-group decomposition with `DecompositionPipeline`
    - Trend removal: `PolynomialTrendForecaster(degree=1)` on panel data
    - Seasonal differencing on panel data
    - Inspecting per-group residuals
    """)

@app.cell(hide_code=True)
def _():
    import polars as pl

    from yohou.compose import DecompositionPipeline
    from yohou.datasets import fetch_tourism_quarterly
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer
    from yohou.stationarity import (
        PatternSeasonalityForecaster,
        PolynomialTrendForecaster,
        SeasonalDifferencing,
    )
    from yohou.utils.panel import inspect_locality

    return (
        DecompositionPipeline,
        LagTransformer,
        MeanAbsoluteError,
        PatternSeasonalityForecaster,
        PointReductionForecaster,
        PolynomialTrendForecaster,
        SeasonalDifferencing,
        SeasonalNaive,
        inspect_locality,
        fetch_tourism_quarterly,
        pl,
        plot_forecast,
        plot_time_series,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load and Explore Panel Data
    """)

@app.cell
def _(inspect_locality, fetch_tourism_quarterly, mo):
    _bunch = fetch_tourism_quarterly()
    # Select 8 series with uniform length for a manageable panel demo
    _selected = [f"T{i}__tourists" for i in range(3, 11)]
    tourism = _bunch.frame.select("time", *_selected).drop_nulls()
    _globals, groups = inspect_locality(tourism)
    _split = int(len(tourism) * 0.8)
    y_train = tourism.head(_split)
    y_test = tourism.tail(len(tourism) - _split)
    horizon = len(y_test)

    mo.md(
        f"**Groups** ({len(groups)} panel groups): {list(groups.keys())}\n\n"
        f"**Train**: {len(y_train)} quarters, **Test**: {len(y_test)} quarters"
    )
    return groups, horizon, tourism, y_test, y_train

@app.cell
def _(plot_time_series, tourism):
    plot_time_series(tourism, title="Tourism Quarterly: All Panel Groups")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Seasonal Differencing on Panel Data

    `SeasonalDifferencing(seasonality=4)` removes quarterly seasonality
    from each state independently.
    """)

@app.cell
def _(SeasonalDifferencing, plot_time_series, tourism):
    sd = SeasonalDifferencing(seasonality=4)
    sd.fit(tourism)
    tourism_diff = sd.transform(tourism)
    plot_time_series(tourism_diff, title="Seasonal Differencing (lag=4): All Groups")
    return sd, tourism_diff

@app.cell
def _(mo, sd, tourism, tourism_diff):
    _inv = sd.inverse_transform(tourism_diff, tourism.head(sd.seasonality))
    # inverse_transform reconstructs the original, except for the first 4 rows
    _overlap = tourism.tail(len(_inv))
    _err = (
        _inv.drop("time")
        .to_numpy()
        - _overlap.drop("time").to_numpy()
    )
    _max_err = abs(_err).max()
    mo.md(
        f"**Inverse transform check**: max error = {_max_err:.2e}\n\n"
        f"The seasonal differencing is perfectly invertible (within floating-point precision)."
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. DecompositionPipeline on Panel Data

    Each group gets its own linear trend + residual model.
    """)

@app.cell
def _(
    DecompositionPipeline,
    LagTransformer,
    PointReductionForecaster,
    PolynomialTrendForecaster,
    horizon,
    y_test,
    y_train,
):
    from sklearn.linear_model import Ridge

    fc_decomp = DecompositionPipeline(
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
        store_residuals=True,
    )
    fc_decomp.fit(y_train, forecasting_horizon=horizon)
    y_pred_decomp = fc_decomp.predict(forecasting_horizon=horizon)
    return Ridge, fc_decomp, y_pred_decomp

@app.cell
def _(plot_forecast, y_pred_decomp, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_decomp,
        y_train=y_train,
        n_history=12,
        panel_group_names=["T1", "T2", "T3"],
        title="DecompositionPipeline: Trend + Residual (Panel)",
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Trend + Seasonality + Residual

    Add a `PatternSeasonalityForecaster` between trend and residual
    for a three-component decomposition.
    """)

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
            ("season", PatternSeasonalityForecaster(seasonality=4)),
            (
                "residual",
                PointReductionForecaster(
                    estimator=Ridge(alpha=1.0),
                    feature_transformer=LagTransformer(lag=[1, 4]),
                ),
            ),
        ],
    )
    fc_three.fit(y_train, forecasting_horizon=horizon)
    _y_pred_three = fc_three.predict(forecasting_horizon=horizon)

    plot_forecast(
        y_test,
        _y_pred_three,
        y_train=y_train,
        n_history=12,
        panel_group_names=["T1", "T2", "T3"],
        title="Trend + Seasonality + Residual (Panel)",
    )
    return (fc_three,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Compare Approaches

    Score each approach per group to see decomposition benefits.
    """)

@app.cell
def _(MeanAbsoluteError, SeasonalNaive, groups, horizon, mo, pl, y_pred_decomp, y_test, y_train):
    _naive = SeasonalNaive(seasonality=4)
    _naive.fit(y_train, forecasting_horizon=horizon)
    _y_pred_naive = _naive.predict(forecasting_horizon=horizon)

    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _rows = []
    for _state in sorted(groups.keys()):
        _s_naive = float(_scorer.score(y_test, _y_pred_naive, panel_group_names=[_state]))
        _s_decomp = float(_scorer.score(y_test, y_pred_decomp, panel_group_names=[_state]))
        _rows.append({
            "Group": _state,
            "SeasonalNaive MAE": round(_s_naive, 1),
            "Decomposition MAE": round(_s_decomp, 1),
        })

    mo.ui.table(pl.DataFrame(_rows))

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Per-group decomposition**: `DecompositionPipeline` fits trend, seasonality, and residual separately for each panel group
    - **Seasonal differencing**: `SeasonalDifferencing(seasonality=4)` operates per-group with invertible transforms
    - **Use `PolynomialTrendForecaster(degree=1)`** for linear trends (no separate `LinearTrendForecaster`)
    - **Three-component pipelines** (trend + season + residual) can improve on simpler baselines
    - **Groupwise scoring** reveals groups where decomposition helps vs hurts

    ## Next Steps

    - **Panel pipelines**: See `examples/compose/panel_pipelines.py`
    - **Stationarity transforms**: See `examples/stationarity/stationarity_transforms.py`
    - **Decomposition details**: See `examples/stationarity/decomposition.py`
    """)

if __name__ == "__main__":
    app.run()
