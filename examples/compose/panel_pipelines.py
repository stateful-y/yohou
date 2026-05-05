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
    "title": "Panel Pipelines",
    "description": "Combine ColumnForecaster, FeaturePipeline, FeatureUnion, and DecompositionPipeline on panel data with per-group scoring on KDD Cup air quality.",
}
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

    - Global model: single [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) across all groups
    - [`ColumnForecaster`](/pages/api/generated/yohou.compose.column_forecaster.ColumnForecaster/): per-group specialised models
    - [`FeaturePipeline`](/pages/api/generated/yohou.compose.feature_pipeline.FeaturePipeline/) + [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/) inside a panel forecaster
    - [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/): automatic per-group trend/season extraction
    - Nested pipelines for panel data
    - Comparing global vs local models with groupwise scoring

    ## Prerequisites

    Panel data conventions (`__` separator) and basic forecasting
    (see `examples/quickstart.py`).
    """)


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
    from yohou.datasets import fetch_kdd_cup
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import train_test_split
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer, RollingStatisticsTransformer, StandardScaler
    from yohou.stationarity import PatternSeasonalityForecaster, PolynomialTrendForecaster
    from yohou.utils.panel import inspect_panel

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
        inspect_panel,
        fetch_kdd_cup,
        pl,
        plot_forecast,
        plot_time_series,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Panel Data Recap

    We load the KDD Cup 2018 air quality dataset with 3 Beijing stations,
    each monitoring 6 pollutants. The `station__measurement` naming
    convention is automatically detected as panel groups.
    """)


@app.cell
def _(inspect_panel, fetch_kdd_cup, mo, train_test_split):
    _bunch = fetch_kdd_cup(n_groups=3)
    store = _bunch.frame.drop_nulls().tail(300)
    _globals, groups = inspect_panel(store)
    target_cols = [c for c in store.columns if c != "time"]

    y_train, y_test = train_test_split(store, test_size=0.15)
    horizon = len(y_test)

    mo.md(
        f"**Groups**: {list(groups.keys())}\n\n"
        f"**Train**: {len(y_train)} rows, **Test**: {len(y_test)} rows\n\n"
        f"3 stations × 6 measurements from KDD Cup 2018"
    )
    return groups, horizon, target_cols, store, y_test, y_train


@app.cell
def _(plot_time_series, store):
    plot_time_series(store, title="KDD Cup 2018: Air Quality Panel")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Global Model

    A single [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) fits one model per group
    automatically when it detects panel data.
    """)


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, horizon, y_test, y_train):
    fc_global = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 24]),
    )
    fc_global.fit(y_train, forecasting_horizon=horizon)
    y_pred_global = fc_global.predict(forecasting_horizon=horizon)

    return fc_global, y_pred_global


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) shows the global model predictions for the first
    two panel groups.
    """)


@app.cell
def _(plot_forecast, y_pred_global, y_test, y_train):
    _groups = sorted({c.split("__")[0] for c in y_train.columns if "__" in c})
    plot_forecast(
        y_test,
        y_pred_global,
        y_train=y_train,
        n_history=48,
        groups=_groups[:2],
        title="Global Model: First 2 Stations",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. ColumnForecaster: Per-Group Models

    [`ColumnForecaster`](/pages/api/generated/yohou.compose.column_forecaster.ColumnForecaster/) assigns different forecasters to different column
    subsets. With panel data, group each station's measurements with its
    own specialised model.
    """)


@app.cell
def _(
    ColumnForecaster,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SeasonalNaive,
    groups,
    horizon,
    target_cols,
    y_test,
    y_train,
):
    # Assign a different forecaster to each station
    _group_names = sorted(groups.keys())
    _g1_cols = [c for c in target_cols if c.startswith(f"{_group_names[0]}__")]
    _g2_cols = [c for c in target_cols if c.startswith(f"{_group_names[1]}__")]
    _g3_cols = [c for c in target_cols if c.startswith(f"{_group_names[2]}__")]

    fc_column = ColumnForecaster(
        forecasters=[
            (
                "station_1_ridge",
                PointReductionForecaster(
                    estimator=Ridge(alpha=1.0),
                    feature_transformer=LagTransformer(lag=[1, 24]),
                ),
                _g1_cols,
            ),
            ("station_2_naive", SeasonalNaive(seasonality=24), _g2_cols),
            (
                "station_3_ridge",
                PointReductionForecaster(
                    estimator=Ridge(alpha=10.0),
                    feature_transformer=LagTransformer(lag=[1, 24, 48]),
                ),
                _g3_cols,
            ),
        ],
    )
    fc_column.fit(y_train, forecasting_horizon=horizon)
    y_pred_column = fc_column.predict(forecasting_horizon=horizon)

    return fc_column, y_pred_column


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) shows the [`ColumnForecaster`](/pages/api/generated/yohou.compose.column_forecaster.ColumnForecaster/) predictions, where each
    station is handled by a different forecaster.
    """)


@app.cell
def _(plot_forecast, y_pred_column, y_test, y_train):
    _groups = sorted({c.split("__")[0] for c in y_train.columns if "__" in c})
    plot_forecast(
        y_test,
        y_pred_column,
        y_train=y_train,
        n_history=48,
        groups=_groups[:2],
        title="ColumnForecaster: Different Model Per Station",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. FeaturePipeline + FeatureUnion

    Build rich feature sets by combining lag features and rolling statistics
    in parallel via [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/), then use as `feature_transformer`.
    """)


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
            ("lags", LagTransformer(lag=[1, 24])),
            ("rolling", RollingStatisticsTransformer(window_size=24, statistics=["mean", "std"])),
        ],
    )

    fc_union = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=_union,
    )
    fc_union.fit(y_train, forecasting_horizon=horizon)
    _y_pred_union = fc_union.predict(forecasting_horizon=horizon)

    _groups = sorted({c.split("__")[0] for c in y_train.columns if "__" in c})
    plot_forecast(
        y_test,
        _y_pred_union,
        y_train=y_train,
        n_history=48,
        groups=_groups[:2],
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
                    feature_transformer=LagTransformer(lag=[1, 24]),
                ),
            ),
        ],
        store_residuals=True,
    )
    fc_decomp.fit(y_train, forecasting_horizon=horizon)
    _y_pred_decomp = fc_decomp.predict(forecasting_horizon=horizon)

    _groups = sorted({c.split("__")[0] for c in y_train.columns if "__" in c})
    plot_forecast(
        y_test,
        _y_pred_decomp,
        y_train=y_train,
        n_history=48,
        groups=_groups[:2],
        title="DecompositionPipeline (Trend + Residual) on Panel Data",
    )
    return (fc_decomp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Nested Pipeline

    Combine decomposition with feature engineering: remove trend first,
    then use [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/) for lag + rolling features on the residual.
    """)


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
            ("lags", LagTransformer(lag=[1, 24])),
            ("rolling", RollingStatisticsTransformer(window_size=24, statistics="mean")),
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

    _groups = sorted({c.split("__")[0] for c in y_train.columns if "__" in c})
    plot_forecast(
        y_test,
        _y_pred_nested,
        y_train=y_train,
        n_history=48,
        groups=[_groups[0], _groups[-1]],
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


@app.cell
def _(MeanAbsoluteError, groups, mo, pl, y_pred_column, y_pred_global, y_test, y_train):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)

    _rows = []
    for _group in sorted(groups.keys()):
        _s_global = _scorer.score(y_test, y_pred_global, groups=[_group])
        _s_column = _scorer.score(y_test, y_pred_column, groups=[_group])
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

    - **Panel point forecasting**: See [`examples/point/panel_forecasting.py`](/examples/point/panel_forecasting/) for deeper panel patterns
    - **Panel intervals**: See [`examples/interval/panel_intervals.py`](/examples/interval/panel_intervals/) for prediction intervals on panel data
    - **Panel preprocessing**: See [`examples/preprocessing/panel_preprocessing.py`](/examples/preprocessing/panel_preprocessing/) for transformer-level panel handling
    """)


if __name__ == "__main__":
    app.run()
