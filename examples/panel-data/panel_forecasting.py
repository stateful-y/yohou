# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.23.1"
__gallery__ = {
    "title": "How to Forecast Panel Data with ColumnForecaster",
    "description": "Apply a shared forecasting model across multiple series in a panel dataset using ColumnForecaster with the __ column separator convention.",
    "category": "how-to",
    "companion": "/pages/how-to/panel-data/",
    "section": "panel-data",
}
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
    global models, per-group specialisation with [`ColumnForecaster`](/pages/api/generated/yohou.compose.column_forecaster.ColumnForecaster/),
    and selective group operations.

    ## 1. Prepare Panel Data

    We load the KDD Cup 2018 air quality dataset with 3 Beijing stations
    each monitoring 6 pollutants. [`inspect_panel`](/pages/api/generated/yohou.utils.panel.inspect_panel/) discovers the panel
    groups from the `__` separator in column names. The data is split
    85/15 into train and test sets.
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeRegressor

    from yohou.compose import ColumnForecaster
    from yohou.datasets import fetch_kdd_cup
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import train_test_split
    from yohou.plotting import (
        plot_forecast,
        plot_group_scores,
        plot_score_time_series,
        plot_time_series,
    )
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer
    from yohou.utils.panel import inspect_panel

    return (
        ColumnForecaster,
        DecisionTreeRegressor,
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        SeasonalNaive,
        deepcopy,
        fetch_kdd_cup,
        inspect_panel,
        plot_forecast,
        plot_group_scores,
        plot_score_time_series,
        plot_time_series,
        train_test_split,
    )


@app.cell
def _(fetch_kdd_cup, inspect_panel, mo, plot_time_series, train_test_split):
    _bunch = fetch_kdd_cup(n_groups=3)
    _df = _bunch.frame.drop_nulls().tail(300)
    _globals, groups = inspect_panel(_df)
    _target_cols = [c for c in _df.columns if c != "time"]
    y = _df.select("time", *_target_cols)

    y_train, y_test = train_test_split(y, test_size=0.15)
    horizon = len(y_test)

    mo.vstack([
        mo.md(
            f"**Panel groups**: {list(groups.keys())}\n\n"
            f"**Target columns**: {len(_target_cols)} pollutant series\n\n"
            f"**Train**: {len(y_train)} hours, **Test**: {len(y_test)} hours, "
            f"**Horizon**: {horizon}"
        ),
        plot_time_series(y, title="KDD Cup 2018: Air Quality (3 Stations)"),
    ])
    return groups, horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Panel Reduction Forecasting

    A single [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) applies the same model template
    to every group. Each group gets its own fitted parameters, but shares
    the same hyperparameters.
    """)


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, horizon, y_train):
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
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) with `groups` and `n_history` shows predictions
    for selected groups in a faceted layout, trimmed to the last 48 time steps.
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
        title="Global Ridge Model: Selected Stations",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. ColumnForecaster: Per-Group Specialisation

    Assign different model families to different station groups.
    """)


@app.cell
def _(
    ColumnForecaster,
    DecisionTreeRegressor,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SeasonalNaive,
    groups,
    horizon,
    y_train,
):
    _group_names = sorted(groups.keys())
    _g1_cols = groups[_group_names[0]]
    _g2_cols = groups[_group_names[1]]
    _g3_cols = groups[_group_names[2]]

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
                "station_3_tree",
                PointReductionForecaster(
                    estimator=DecisionTreeRegressor(max_depth=5),
                    feature_transformer=LagTransformer(lag=[1, 24, 48]),
                ),
                _g3_cols,
            ),
        ],
    )
    fc_column.fit(y_train, forecasting_horizon=horizon)
    y_pred_column = fc_column.predict(forecasting_horizon=horizon)
    return (y_pred_column,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) for the [`ColumnForecaster`](/pages/api/generated/yohou.compose.column_forecaster.ColumnForecaster/) lets you visually compare how
    per-group model specialisation affects predictions for the same groups.
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
        title="ColumnForecaster: Per-Group Specialisation",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Selective Group Operations

    `predict`, `observe`, and `rewind` all accept a `groups`
    parameter.  This lets you predict a subset of groups, observe new
    data for only the groups that have reported so far (e.g. some stations
    report with different delays), or rewind specific groups without
    touching the rest.

    Below we demonstrate this on the first station: predict from the
    training window, then observe new data and predict again (the
    forecast origin moves forward), and finally rewind back and predict
    once more (the origin returns to where it was).
    """)


@app.cell
def _(deepcopy, fc_global, groups, mo, plot_forecast, y_test, y_train):

    _fc = deepcopy(fc_global)
    _group_name = sorted(groups.keys())[0]
    _group = [_group_name]
    _group_cols = groups[_group_name]
    _horizon = 12

    # 1) Predict from training window
    _y_pred_before = _fc.predict(forecasting_horizon=_horizon, groups=_group)

    # 2) Observe the first half of test data for this station only
    _half = len(y_test) // 2
    _y_obs = y_test.select("time", *_group_cols).head(_half)
    _fc.observe(_y_obs, groups=_group)
    _y_pred_after_obs = _fc.predict(forecasting_horizon=_horizon, groups=_group)

    # 3) Rewind back: the forecast origin returns to the training window
    _fc.rewind(y_test.select("time", *_group_cols), groups=_group)
    _y_pred_after_rwd = _fc.predict(forecasting_horizon=_horizon, groups=_group)

    mo.vstack([
        mo.md("**After fit**: forecast starts right after training data"),
        plot_forecast(
            y_test,
            _y_pred_before,
            y_train=y_train,
            groups=_group,
            n_history=24,
            title=f"{_group_name}: Predict from Training Window",
        ),
        mo.md(f"**After observe({_half} rows)**: forecast origin moves forward"),
        plot_forecast(
            y_test,
            _y_pred_after_obs,
            y_train=y_train,
            groups=_group,
            n_history=24,
            title=f"{_group_name}: Predict After Observe",
        ),
        mo.md("**After rewind**: forecast origin returns to the training window"),
        plot_forecast(
            y_test,
            _y_pred_after_rwd,
            y_train=y_train,
            groups=_group,
            n_history=24,
            title=f"{_group_name}: Predict After Rewind",
        ),
    ])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Groupwise Scoring

    Use [`plot_score_time_series`](/pages/api/generated/yohou.plotting.evaluation.plot_score_time_series/) with `groups` to visualise
    per-group error over time.  Each group gets its own subplot, making
    it easy to spot which groups are well-served by the global model
    and which would benefit from specialisation.

    [`plot_group_scores`](/pages/api/generated/yohou.plotting.evaluation.plot_group_scores/) summarises the MAE per model
    broken down by group as a bar chart, box distribution, or heatmap.
    """)


@app.cell
def _(
    MeanAbsoluteError,
    plot_score_time_series,
    y_pred_column,
    y_pred_global,
    y_test,
    y_train,
):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _groups = sorted({c.split("__")[0] for c in y_train.columns if "__" in c})
    plot_score_time_series(
        _scorer,
        y_test,
        {"Global Ridge": y_pred_global, "ColumnForecaster": y_pred_column},
        groups=_groups[:2],
        title="MAE Over Time by Station",
    )


@app.cell
def _(
    MeanAbsoluteError,
    plot_group_scores,
    y_pred_column,
    y_pred_global,
    y_test,
):
    plot_group_scores(
        MeanAbsoluteError(),
        y_test,
        {"Global Ridge": y_pred_global, "ColumnForecaster": y_pred_column},
        title="Groupwise MAE: Global vs ColumnForecaster",
        y_label="MAE",
    )


@app.cell
def _(MeanAbsoluteError, plot_group_scores, y_pred_global, y_test):
    plot_group_scores(
        MeanAbsoluteError(),
        y_test,
        y_pred_global,
        kind="box",
        title="Global Ridge - MAE Distribution by Group",
    )


@app.cell
def _(
    MeanAbsoluteError,
    plot_group_scores,
    y_pred_column,
    y_pred_global,
    y_test,
):
    plot_group_scores(
        MeanAbsoluteError(),
        y_test,
        {"Global Ridge": y_pred_global, "ColumnForecaster": y_pred_column},
        kind="heatmap",
        title="Groupwise MAE - Heatmap",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Panel intervals**: See [`examples/panel-data/panel_intervals.py`](/examples/panel_intervals/)
    - **Aggregation modes**: See [`examples/evaluation-search/aggregation_modes.py`](/examples/aggregation_modes/)
    - **Panel cross-validation**: See [`examples/panel-data/panel_cross_validation.py`](/examples/panel_cross_validation/)
    """)


if __name__ == "__main__":
    app.run()
