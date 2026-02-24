# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Panel Point Forecasting.

Demonstrates global vs local models, ColumnForecaster specialisation,
groupwise scoring, and selective group training on panel data.
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
    # Panel Point Forecasting

    When panel data is passed to a forecaster, each group (`__`-separated
    column prefix) is modelled independently. This notebook compares
    global models, per-group specialisation with `ColumnForecaster`,
    and selective group operations.

    ## What You'll Learn

    - Global model: one hyperparameter set for all groups
    - `ColumnForecaster`: assign different models per group
    - Selective training/prediction with `panel_group_names`
    - Groupwise scoring to identify weak groups
    """)
    return

@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeRegressor

    from yohou.compose import ColumnForecaster
    from yohou.datasets import fetch_dominick
    from yohou.metrics import MeanAbsoluteError, RootMeanSquaredError
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer
    from yohou.utils.panel import inspect_locality

    return (
        ColumnForecaster,
        DecisionTreeRegressor,
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        RootMeanSquaredError,
        SeasonalNaive,
        inspect_locality,
        fetch_dominick,
        pl,
        plot_forecast,
        plot_time_series,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Panel Data
    """)
    return

@app.cell
def _(inspect_locality, fetch_dominick, mo, pl):
    _bunch = fetch_dominick()
    # Select 9 series with complete data (no nulls) for a manageable panel demo
    _selected = ["T7__profit", "T11__profit", "T12__profit", "T13__profit", "T15__profit", "T19__profit", "T22__profit", "T23__profit", "T24__profit"]
    store = _bunch.frame.select("time", *_selected).drop_nulls()
    _globals, groups = inspect_locality(store)
    _target_cols = [c for c in store.columns if c.endswith("__profit")]
    y = store.select("time", *_target_cols)

    _split = int(len(y) * 0.85)
    y_train = y.head(_split)
    y_test = y.tail(len(y) - _split)
    horizon = len(y_test)

    mo.md(
        f"**Panel groups**: {list(groups.keys())}\n\n"
        f"**Target columns**: {_target_cols}\n\n"
        f"**Train**: {len(y_train)} weeks, **Test**: {len(y_test)} weeks, "
        f"**Horizon**: {horizon}"
    )
    return groups, horizon, store, y, y_test, y_train

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Global Model

    A single `PointReductionForecaster` applies the same model template
    to every group. Each group gets its own fitted parameters, but shares
    the same hyperparameters.
    """)
    return

@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, horizon, y_train):
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
        panel_group_names=["T7", "T13", "T22"],
        title="Global Ridge Model: Selected Groups",
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. ColumnForecaster: Per-Group Specialisation

    Assign different model families to different store groups.
    """)
    return

@app.cell
def _(
    ColumnForecaster,
    DecisionTreeRegressor,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SeasonalNaive,
    horizon,
    y,
    y_train,
):
    _g1_cols = ["T7__profit", "T11__profit", "T12__profit"]
    _g2_cols = ["T13__profit", "T15__profit", "T19__profit"]
    _g3_cols = ["T22__profit", "T23__profit", "T24__profit"]

    fc_column = ColumnForecaster(
        forecasters=[
            (
                "group_1_ridge",
                PointReductionForecaster(
                    estimator=Ridge(alpha=1.0),
                    feature_transformer=LagTransformer(lag=[1, 7]),
                ),
                _g1_cols,
            ),
            ("group_2_naive", SeasonalNaive(seasonality=7), _g2_cols),
            (
                "group_3_tree",
                PointReductionForecaster(
                    estimator=DecisionTreeRegressor(max_depth=5),
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
        panel_group_names=["T7", "T13", "T22"],
        title="ColumnForecaster: Per-Group Specialisation",
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Selective Group Prediction

    Use `panel_group_names` to train or predict on a subset of groups.
    Useful when new groups appear or you want group-specific analysis.
    """)
    return

@app.cell
def _(fc_global, mo, plot_forecast, y_pred_global, y_test, y_train):
    # Predict only T7-T12 groups
    y_pred_s1 = fc_global.predict(
        forecasting_horizon=len(y_test),
        panel_group_names=["T7", "T11", "T12"],
    )
    mo.md(
        f"**Selective prediction columns**: {y_pred_s1.columns}\n\n"
        f"Only T7, T11, T12 groups are predicted; other groups are omitted."
    )
    return (y_pred_s1,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Groupwise Scoring

    Score each group individually to identify which groups perform best
    and worst under each model.
    """)
    return

@app.cell
def _(MeanAbsoluteError, groups, mo, pl, y_pred_column, y_pred_global, y_test, y_train):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _rows = []
    for _group in sorted(groups.keys()):
        _s_global = float(_scorer.score(y_test, y_pred_global))
        _s_column = float(_scorer.score(y_test, y_pred_column))
        _rows.append({
            "Group": _group,
            "Global MAE": round(_s_global, 2),
            "ColumnForecaster MAE": round(_s_column, 2),
        })

    comparison_df = pl.DataFrame(_rows)
    mo.ui.table(comparison_df)
    return (comparison_df,)

@app.cell
def _(MeanAbsoluteError, RootMeanSquaredError, mo, y_pred_global, y_test, y_train):
    # Aggregation modes: "groupwise", "componentwise", "timewise", "all"
    _mae_groupwise = MeanAbsoluteError(aggregation_method="groupwise")
    _rmse_groupwise = RootMeanSquaredError(aggregation_method="groupwise")

    _mae_groupwise.fit(y_train)
    _rmse_groupwise.fit(y_train)
    _s_mae = _mae_groupwise.score(y_test, y_pred_global)
    _s_rmse = _rmse_groupwise.score(y_test, y_pred_global)

    mo.md(
        f"**Groupwise MAE columns**: {_s_mae.columns}\n\n"
        f"**Groupwise RMSE columns**: {_s_rmse.columns}\n\n"
        "Groupwise aggregation returns one score per group (aggregated over time)."
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Global models** apply the same hyperparameters across all groups (each group still fitted independently)
    - **ColumnForecaster** enables per-group model specialisation (different algorithms, different hyperparameters)
    - **`panel_group_names`** parameter enables selective fit/predict/score on group subsets
    - **Groupwise scoring** reveals which groups benefit from specialised models
    - Always compare against a simple baseline (`SeasonalNaive`) per group

    ## Next Steps

    - **Panel intervals**: See `examples/interval/panel_intervals.py`
    - **Aggregation modes**: See `examples/metrics/aggregation_modes.py`
    - **Panel cross-validation**: See `examples/model_selection/panel_cross_validation.py`
    """)
    return

if __name__ == "__main__":
    app.run()
