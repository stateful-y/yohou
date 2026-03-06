# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
#     "yohou-nixtla",
# ]
# ///

import marimo

__generated_with = "0.19.11"
__gallery__ = {
    "title": "Nixtla Panel",
    "description": "Fit Nixtla statistical forecasters (AutoARIMA, AutoETS) on Tourism Quarterly panel data with per-group scoring and expanding-window cross-validation.",
}
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
    > with any Scikit-Learn regressor directly.

    **Requires**: `yohou-nixtla` package
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.model_selection import train_test_split

    from yohou.datasets import fetch_tourism_quarterly
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection.split import ExpandingWindowSplitter
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.utils.panel import inspect_panel

    return (
        ExpandingWindowSplitter,
        MeanAbsoluteError,
        fetch_tourism_quarterly,
        pl,
        plot_forecast,
        plot_time_series,
        inspect_panel,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Explore Panel Data

    We load the Tourism Quarterly dataset with multiple panel groups and
    inspect its structure using [`inspect_panel`](/pages/api/generated/yohou.utils.panel.inspect_panel/). Each column follows
    the `GROUP__MEMBER` naming convention that yohou uses to identify
    separate time series within the same DataFrame.
    """)


@app.cell
def _(inspect_panel, fetch_tourism_quarterly, mo, pl, train_test_split):
    tourism = fetch_tourism_quarterly().frame.select("time", *[f"T{i}__tourists" for i in range(3, 11)]).drop_nulls()
    y_train, y_test = train_test_split(tourism, test_size=0.2, shuffle=False)
    horizon = len(y_test)

    _global_cols, _groups = inspect_panel(tourism.drop("time"))
    _group_names = list(_groups.keys())

    mo.md(
        f"**Australian Tourism**: {len(tourism)} quarters, "
        f"{len(tourism.columns) - 1} series\n\n"
        f"**Panel groups**: {', '.join(_group_names[:4])}...\n\n"
        f"**Train**: {len(y_train)}, **Test**: {len(y_test)}\n\n"
        f"**Columns**: {[c for c in tourism.columns if c != 'time']}"
    )
    return horizon, tourism, y_test, y_train


@app.cell
def _(plot_time_series, tourism):
    plot_time_series(tourism, title="Australian Tourism Quarterly (Panel)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Statistical Forecasters on Panel Data

    Nixtla's stats models fit **one model per series** internally.
    """)


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


@app.cell
def _(
    MeanAbsoluteError,
    inspect_panel,
    mo,
    pl,
    stats_panel_preds,
    y_test,
    y_train,
):
    _scorer = MeanAbsoluteError(aggregation_method="componentwise").fit(y_train)
    _all_preds = {**stats_panel_preds}
    _rows = []
    _groups = list(inspect_panel(y_test.drop("time"))[1].keys())
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

    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) renders the AutoARIMA predictions across all panel
    groups. With panel data, the plot creates one trace per group,
    making it easy to compare forecast quality across series.
    """)


@app.cell
def _(plot_forecast, stats_panel_preds, y_test, y_train):
    plot_forecast(
        y_test,
        stats_panel_preds["AutoARIMA"],
        y_train=y_train,
        n_history=8,
        title="AutoARIMA on Panel Data",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Cross-Validation

    All Nixtla forecasters work with yohou's [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/). Here we
    search over `season_length` using an [`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.split.ExpandingWindowSplitter/) to
    find the best seasonal periodicity for AutoARIMA on panel data.
    """)


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

    - **Nixtla forecasters (univariate)**: See [`nixtla_forecasters.py`](/examples/point/nixtla_forecasters/)
    - **Panel forecasting**: See [`examples/point/panel_forecasting.py`](/examples/point/panel_forecasting/)
    - **Panel cross-validation**: See [`examples/model_selection/panel_cross_validation.py`](/examples/model_selection/panel_cross_validation/)
    """)


if __name__ == "__main__":
    app.run()
