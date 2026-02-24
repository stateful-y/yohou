# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Composition Patterns for Panel Data.

Demonstrates how all composition meta-estimators (ColumnForecaster,
DecompositionPipeline, FeaturePipeline, FeatureUnion, ColumnTransformer)
work with panel time series, including per-group model specialisation.
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
    # Composition Patterns for Panel Data

    Yohou's composition meta-estimators automatically handle panel data by
    decomposing the panel into groups, applying per-group transformations
    and models, and reassembling the results. This notebook demonstrates
    all major composition patterns with panel data.

    ## What You'll Learn

    - Global model: single `PointReductionForecaster` across all groups
    - `ColumnForecaster`: per-group specialised models
    - `FeaturePipeline` + `FeatureUnion` inside a panel forecaster
    - `DecompositionPipeline`: automatic per-group trend/season extraction
    - Nested pipelines for panel data
    - Comparing global vs local models with groupwise scoring

    ## Prerequisites

    Panel data conventions (`__` separator) and basic forecasting
    (see `examples/quickstart.py`).
    """)
    return

@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import (
        ColumnForecaster,
        DecompositionPipeline,
        FeaturePipeline,
        FeatureUnion,
    )
    from yohou.datasets import fetch_dominick
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer, RollingStatisticsTransformer, StandardScaler
    from yohou.stationarity import PatternSeasonalityForecaster, PolynomialTrendForecaster
    from yohou.utils.panel import inspect_locality

    return (
        ColumnForecaster,
        DecompositionPipeline,
        FeaturePipeline,
        FeatureUnion,
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        PolynomialTrendForecaster,
        PatternSeasonalityForecaster,
        Ridge,
        RollingStatisticsTransformer,
        SeasonalNaive,
        StandardScaler,
        inspect_locality,
        fetch_dominick,
        pl,
        plot_forecast,
        plot_time_series,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Panel Data Recap
    """)
    return

@app.cell
def _(inspect_locality, fetch_dominick, mo):
    _dom_full = fetch_dominick().frame
    # Select 9 series with complete data (no nulls) for a manageable panel demo
    _profit_cols = ["T7__profit", "T11__profit", "T12__profit", "T13__profit", "T15__profit", "T19__profit", "T22__profit", "T23__profit", "T24__profit"]
    store = _dom_full.select("time", *_profit_cols)
    _globals, groups = inspect_locality(store)
    sales_cols = [c for c in store.columns if c.endswith("__profit")]

    _split = int(len(store) * 0.85)
    y_train = store.head(_split).select("time", *sales_cols)
    y_test = store.tail(len(store) - _split).select("time", *sales_cols)
    horizon = len(y_test)

    mo.md(
        f"**Groups**: {list(groups.keys())}\n\n"
        f"**Train**: {len(y_train)} rows, **Test**: {len(y_test)} rows\n\n"
        f"9 panel groups from Dominick dataset"
    )
    return groups, horizon, sales_cols, store, y_test, y_train

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Global Model

    A single `PointReductionForecaster` fits one model per group
    automatically when it detects panel data.
    """)
    return

@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, horizon, y_test, y_train):
    fc_global = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 7]),
    )
    fc_global.fit(y_train, forecasting_horizon=horizon)
    y_pred_global = fc_global.predict(forecasting_horizon=horizon)

    return fc_global, y_pred_global

@app.cell
def _(plot_forecast, y_pred_global, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_global,
        y_train=y_train,
        n_history=30,
        panel_group_names=["T7", "T11", "T12"],
        title="Global Model: First 3 Series",
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. ColumnForecaster: Per-Group Models

    `ColumnForecaster` assigns different forecasters to different column
    subsets. With panel data, group each store-item combination with its
    own specialised model.
    """)
    return

@app.cell
def _(
    ColumnForecaster,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SeasonalNaive,
    horizon,
    sales_cols,
    y_test,
    y_train,
):
    # Ridge for first 3 groups, SeasonalNaive for next 3, Ridge(alpha=10) for last 3
    _g1_cols = sales_cols[:3]
    _g2_cols = sales_cols[3:6]
    _g3_cols = sales_cols[6:9]

    fc_column = ColumnForecaster(
        forecasters=[
            (
                "group_1",
                PointReductionForecaster(
                    estimator=Ridge(alpha=1.0),
                    feature_transformer=LagTransformer(lag=[1, 7]),
                ),
                _g1_cols,
            ),
            ("group_2", SeasonalNaive(seasonality=7), _g2_cols),
            (
                "group_3",
                PointReductionForecaster(
                    estimator=Ridge(alpha=10.0),
                    feature_transformer=LagTransformer(lag=[1, 7, 14]),
                ),
                _g3_cols,
            ),
        ],
    )
    fc_column.fit(y_train, forecasting_horizon=horizon)
    y_pred_column = fc_column.predict(forecasting_horizon=horizon)

    return fc_column, y_pred_column

@app.cell
def _(plot_forecast, y_pred_column, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_column,
        y_train=y_train,
        n_history=30,
        panel_group_names=["T7", "T11", "T12"],
        title="ColumnForecaster: Different Model Per Group",
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. FeaturePipeline + FeatureUnion

    Build rich feature sets by combining lag features and rolling statistics
    in parallel via `FeatureUnion`, then use as `feature_transformer`.
    """)
    return

@app.cell
def _(
    FeatureUnion,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    RollingStatisticsTransformer,
    horizon,
    plot_forecast,
    y_test,
    y_train,
):
    _union = FeatureUnion(
        transformer_list=[
            ("lags", LagTransformer(lag=[1, 7])),
            ("rolling", RollingStatisticsTransformer(window_size=7, statistics=["mean", "std"])),
        ],
    )

    fc_union = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=_union,
    )
    fc_union.fit(y_train, forecasting_horizon=horizon)
    _y_pred_union = fc_union.predict(forecasting_horizon=horizon)

    plot_forecast(
        y_test,
        _y_pred_union,
        y_train=y_train,
        n_history=30,
        panel_group_names=["T7", "T13"],
        title="FeatureUnion (Lags + Rolling) on Panel Data",
    )
    return (fc_union,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. DecompositionPipeline

    Decompose each panel group into trend + residual automatically.
    The pipeline fits a separate trend model per group.
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
    fc_decomp = DecompositionPipeline(
        forecasters=[
            ("trend", PolynomialTrendForecaster(degree=1)),
            (
                "residual",
                PointReductionForecaster(
                    estimator=Ridge(alpha=1.0),
                    feature_transformer=LagTransformer(lag=[1, 7]),
                ),
            ),
        ],
        store_residuals=True,
    )
    fc_decomp.fit(y_train, forecasting_horizon=horizon)
    _y_pred_decomp = fc_decomp.predict(forecasting_horizon=horizon)

    plot_forecast(
        y_test,
        _y_pred_decomp,
        y_train=y_train,
        n_history=30,
        panel_group_names=["T7", "T13"],
        title="DecompositionPipeline (Trend + Residual) on Panel Data",
    )
    return (fc_decomp,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Nested Pipeline

    Combine decomposition with feature engineering: remove trend first,
    then use `FeatureUnion` for lag + rolling features on the residual.
    """)
    return

@app.cell
def _(
    DecompositionPipeline,
    FeatureUnion,
    LagTransformer,
    PointReductionForecaster,
    PolynomialTrendForecaster,
    Ridge,
    RollingStatisticsTransformer,
    horizon,
    plot_forecast,
    y_test,
    y_train,
):
    _union_nested = FeatureUnion(
        transformer_list=[
            ("lags", LagTransformer(lag=[1, 7])),
            ("rolling", RollingStatisticsTransformer(window_size=7, statistics="mean")),
        ],
    )

    fc_nested = DecompositionPipeline(
        forecasters=[
            ("trend", PolynomialTrendForecaster(degree=1)),
            (
                "residual",
                PointReductionForecaster(
                    estimator=Ridge(alpha=1.0),
                    feature_transformer=_union_nested,
                ),
            ),
        ],
    )
    fc_nested.fit(y_train, forecasting_horizon=horizon)
    _y_pred_nested = fc_nested.predict(forecasting_horizon=horizon)

    plot_forecast(
        y_test,
        _y_pred_nested,
        y_train=y_train,
        n_history=30,
        panel_group_names=["T7", "T24"],
        title="Nested Pipeline on Panel Data",
    )
    return (fc_nested,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Compare Global vs Local

    Score each approach with groupwise aggregation to see which groups
    benefit from specialised models.
    """)
    return

@app.cell
def _(MeanAbsoluteError, groups, mo, pl, y_pred_column, y_pred_global, y_test, y_train):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)

    _rows = []
    for _group in sorted(groups.keys()):
        _s_global = _scorer.score(y_test, y_pred_global, panel_group_names=[_group])
        _s_column = _scorer.score(y_test, y_pred_column, panel_group_names=[_group])
        _rows.append({
            "Group": _group,
            "Global MAE": round(float(_s_global), 1),
            "ColumnForecaster MAE": round(float(_s_column), 1),
        })

    comparison = pl.DataFrame(_rows)
    mo.ui.table(comparison)
    return (comparison,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **All composition meta-estimators** work with panel data automatically
    - **Global model** (single forecaster): fits one model per group with shared hyperparameters
    - **ColumnForecaster**: assigns different forecasters to different column subsets for per-group specialisation
    - **FeatureUnion**: parallel feature engineering applied per-group inside a panel forecaster
    - **DecompositionPipeline**: per-group trend/season extraction with automatic reassembly
    - **Nested pipelines**: combine decomposition with feature engineering seamlessly
    - **Groupwise scoring** reveals which groups benefit from local vs global models

    ## Next Steps

    - **Panel point forecasting**: See `examples/point/panel_forecasting.py` for deeper panel patterns
    - **Panel intervals**: See `examples/interval/panel_intervals.py` for prediction intervals on panel data
    - **Panel preprocessing**: See `examples/preprocessing/panel_preprocessing.py` for transformer-level panel handling
    """)
    return

if __name__ == "__main__":
    app.run()
