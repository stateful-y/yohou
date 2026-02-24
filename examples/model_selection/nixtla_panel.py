# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "plotly",
#     "scikit-learn",
#     "statsforecast",
#     "yohou",
# ]
# ///
"""Nixtla Forecasters on Panel Data.

Demonstrates yohou-nixtla forecasters with panel time series,
comparing global vs per-group models.
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
    # Nixtla Forecasters on Panel Data

    All `yohou-nixtla` forecasters natively handle panel data via the
    `__` (double underscore) column separator convention. Under the hood,
    yohou converts to Nixtla's long format with `unique_id`.

    ## What You'll Learn

    - Stats forecasters (AutoARIMA, AutoETS) on panel data
    - Per-group scoring and comparison
    - Cross-validation with panel splits

    > **Note**: For ML-based reduction forecasters, use `yohou.point.PointReductionForecaster`
    > with any scikit-learn regressor directly.

    **Requires**: `yohou-nixtla` package
    """)
    return

@app.cell(hide_code=True)
def _():
    import polars as pl

    from yohou.datasets import fetch_tourism_quarterly
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection.split import ExpandingWindowSplitter
    from yohou.plotting import plot_forecast
    from yohou.utils.panel import inspect_locality

    return (
        ExpandingWindowSplitter,
        MeanAbsoluteError,
        fetch_tourism_quarterly,
        pl,
        plot_forecast,
        inspect_locality,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Explore Panel Data
    """)
    return

@app.cell
def _(inspect_locality, fetch_tourism_quarterly, mo, pl):
    tourism = fetch_tourism_quarterly().frame.select("time", *[f"T{i}__tourists" for i in range(3, 11)]).drop_nulls()
    _split = int(len(tourism) * 0.8)
    y_train = tourism.head(_split)
    y_test = tourism.tail(len(tourism) - _split)
    horizon = len(y_test)

    _global_cols, _groups = inspect_locality(tourism.drop("time"))
    _group_names = list(_groups.keys())

    mo.md(
        f"**Australian Tourism**: {len(tourism)} quarters, "
        f"{len(tourism.columns) - 1} series\n\n"
        f"**Panel groups**: {', '.join(_group_names[:4])}...\n\n"
        f"**Train**: {len(y_train)}, **Test**: {len(y_test)}\n\n"
        f"**Columns**: {[c for c in tourism.columns if c != 'time']}"
    )
    return horizon, tourism, y_test, y_train

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Statistical Forecasters on Panel Data

    Nixtla's stats models fit **one model per series** internally.
    """)
    return

@app.cell
def _():
    from yohou_nixtla.stats import (
        AutoARIMAForecaster,
        AutoETSForecaster,
        SeasonalNaiveForecaster,
    )

    return AutoARIMAForecaster, AutoETSForecaster, SeasonalNaiveForecaster

@app.cell
def _(
    AutoARIMAForecaster,
    AutoETSForecaster,
    SeasonalNaiveForecaster,
    horizon,
    y_train,
):
    stats_panel = {
        "SeasonalNaive": SeasonalNaiveForecaster(season_length=4),
        "AutoARIMA": AutoARIMAForecaster(season_length=4),
        "AutoETS": AutoETSForecaster(season_length=4),
    }
    stats_panel_preds = {}
    for _name, _fc in stats_panel.items():
        _fc.fit(y_train, forecasting_horizon=horizon)
        stats_panel_preds[_name] = _fc.predict(forecasting_horizon=horizon)
    return stats_panel, stats_panel_preds

@app.cell
def _(MeanAbsoluteError, mo, pl, stats_panel_preds, y_test, y_train):
    _scorer = MeanAbsoluteError().fit(y_train)
    _rows = []
    for _name, _pred in stats_panel_preds.items():
        _mae = float(_scorer.score(y_test, _pred))
        _rows.append({"Model": _name, "Type": "Stats", "MAE (aggregate)": round(_mae, 2)})
    stats_panel_results = pl.DataFrame(_rows)
    mo.ui.table(stats_panel_results)
    return (stats_panel_results,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Per-Group Comparison

    Score each model per panel group to see where models differ.
    """)
    return

@app.cell
def _(
    MeanAbsoluteError,
    inspect_locality,
    mo,
    pl,
    stats_panel_preds,
    y_test,
    y_train,
):
    _scorer = MeanAbsoluteError(aggregation_method="componentwise").fit(y_train)
    _all_preds = {**stats_panel_preds}
    _rows = []
    _groups = list(inspect_locality(y_test.drop("time"))[1].keys())
    for _name, _pred in _all_preds.items():
        _score_df = _scorer.score(y_test, _pred)
        _scores = _score_df.drop("time")
        for _col in _scores.columns:
            _mae = float(_scores[_col].mean())
            _rows.append({
                "Model": _name,
                "Group": _col,
                "MAE": round(_mae, 2),
            })
    group_comparison = pl.DataFrame(_rows)
    mo.ui.table(group_comparison)
    return (group_comparison,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Forecast Visualisation
    """)
    return

@app.cell
def _(plot_forecast, stats_panel_preds, y_test, y_train):
    plot_forecast(
        y_test,
        stats_panel_preds["AutoARIMA"],
        y_train=y_train,
        n_history=8,
        title="AutoARIMA on Panel Data",
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Cross-Validation
    """)
    return

@app.cell
def _(
    AutoARIMAForecaster,
    ExpandingWindowSplitter,
    MeanAbsoluteError,
    horizon,
    mo,
    y_train,
):
    from yohou.model_selection import GridSearchCV

    _gs = GridSearchCV(
        forecaster=AutoARIMAForecaster(season_length=4),
        param_grid={"season_length": [2, 4]},
        scoring=MeanAbsoluteError(),
        cv=ExpandingWindowSplitter(n_splits=3),
        refit=True,
    )
    _gs.fit(y_train, forecasting_horizon=horizon)
    mo.md(
        f"**Best `season_length`**: {_gs.best_params_['season_length']}\n\n"
        f"**Best MAE (negated)**: {_gs.best_score_:.4f}"
    )
    return (GridSearchCV,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - All `yohou-nixtla` forecasters handle panel data automatically
    - The `__` column separator is converted to Nixtla's `unique_id` format
    - Stats models fit **one model per series** (local by default)
    - `season_length` should match the data periodicity (4 for quarterly)
    - `freq` is auto-inferred from the time column
    - works with GridSearchCV for hyperparameter tuning

    ## Next Steps

    - **Nixtla forecasters (univariate)**: See `examples/model_selection/nixtla_forecasters.py`
    - **Panel forecasting**: See `examples/point/panel_forecasting.py`
    - **Panel cross-validation**: See `examples/model_selection/panel_cross_validation.py`
    """)
    return

if __name__ == "__main__":
    app.run()
