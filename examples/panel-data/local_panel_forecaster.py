# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.19.11"
__gallery__ = {
    "title": "How to Configure LocalPanelForecaster",
    "description": "Wrap any forecaster with LocalPanelForecaster for fully independent per-group clones, parallel fitting via n_jobs, and selective group operations.",
    "category": "how-to",
    "section": "panel-data",
    "companion": "/pages/how-to/panel-data/",
    "api_references": ["LocalPanelForecaster", "inspect_panel", "fetch_tourism_quarterly", "train_test_split", "plot_time_series", "PointReductionForecaster", "LagTransformer", "plot_forecast", "SeasonalNaive", "ColumnForecaster"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LocalPanelForecaster

    [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster/) wraps any forecaster and fits **completely
    independent clones** per panel group via `sklearn.base.clone()`.
    Each clone sees standard (non-panel) data with group prefixes stripped,
    and predictions are reassembled back into prefixed-column format.

    ## 1. Prepare Panel Data

    Australian Tourism: quarterly trips for 8 series.  Each series is a
    panel group with the `__` separator (e.g. `T3__tourists`).
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

    from sklearn.linear_model import Ridge

    from yohou.compose import LocalPanelForecaster
    from yohou.datasets import fetch_tourism_quarterly
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import train_test_split
    from yohou.plotting import (
        plot_forecast,
        plot_score_per_vintage,
        plot_score_time_series,
        plot_time_series,
    )
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer
    from yohou.utils.panel import inspect_panel

    return (
        LagTransformer,
        LocalPanelForecaster,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        SeasonalNaive,
        deepcopy,
        fetch_tourism_quarterly,
        inspect_panel,
        plot_forecast,
        plot_score_per_vintage,
        plot_score_time_series,
        plot_time_series,
        train_test_split,
    )


@app.cell
def _(inspect_panel, fetch_tourism_quarterly, mo, train_test_split):
    _tourism_full = fetch_tourism_quarterly().frame
    # Select T3-T10 (same length, 88 rows after drop_nulls)
    _tourist_cols = [f"T{i}__tourists" for i in range(3, 11)]
    tourism = _tourism_full.select("time", *_tourist_cols).drop_nulls()
    _globals, groups = inspect_panel(tourism)
    y = tourism.select("time", *[c for c in tourism.columns if c.endswith("__tourists")])

    y_train, y_test = train_test_split(y, test_size=0.2)
    horizon = len(y_test)

    mo.md(
        f"**Panel groups**: {sorted(groups.keys())}\n\n"
        f"**Train**: {len(y_train)} quarters | **Test**: {len(y_test)} quarters | "
        f"**Horizon**: {horizon}"
    )
    return groups, horizon, tourism, y, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_time_series`](/pages/api/generated/yohou.plotting.exploration.plot_time_series/) shows all 8 quarterly tourism series
    before splitting into training and test sets.
    """)


@app.cell
def _(plot_time_series, y):
    plot_time_series(y, title="Tourism Quarterly: Quarterly Tourists by Series")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Basic Usage

    Wrap a forecaster in [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster/).  Each panel group gets a
    fresh `clone()` that is fitted independently on that group's data.
    """)


@app.cell
def _(LagTransformer, LocalPanelForecaster, PointReductionForecaster, Ridge, horizon, y_train):
    fc_local = LocalPanelForecaster(
        forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 2, 4]),
        ),
    )
    fc_local.fit(y_train, forecasting_horizon=horizon)
    y_pred_local = fc_local.predict(forecasting_horizon=horizon)
    return fc_local, y_pred_local


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) displays the per-group forecasts produced by the
    [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster/). Each group has its own independently fitted model.
    """)


@app.cell
def _(plot_forecast, y_pred_local, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_local,
        y_train=y_train,
        n_history=12,
        groups=["T3", "T4", "T5"],
        title="LocalPanelForecaster: Top 3 Series",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Accessing Per-Group Clones

    After fitting, `forecasters_` holds a dict mapping group names to
    their fitted clones.  You can inspect individual models.
    """)


@app.cell
def _(fc_local, mo):
    _rows = []
    for _name, _fc in sorted(fc_local.forecasters_.items()):
        _coefs = _fc.estimator_.coef_
        _rows.append(f"- **{_name}**: {len(_coefs)} coefficients")

    mo.md(f"**Fitted clones** ({len(fc_local.forecasters_)}):\n\n" + "\n".join(_rows))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Selective Group Operations

    Use `groups` to predict, observe, or rewind only a subset
    of groups.  This is useful when new data arrives for specific groups
    or you want group-specific analysis.
    """)


@app.cell
def _(fc_local, horizon, mo):
    y_pred_top3 = fc_local.predict(
        forecasting_horizon=horizon,
        groups=["T3", "T4", "T5"],
    )
    mo.md(
        f"**Selective prediction columns**: {sorted(y_pred_top3.columns)}\n\n"
        "Only the requested groups are predicted; others are omitted."
    )
    return (y_pred_top3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Observe-Predict Workflow

    [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster/) supports the observe-predict cycle for
    rolling evaluation.  New observations update only the specified
    groups' clones.
    """)


@app.cell
def _(fc_local, horizon, mo, y_test):
    _half = len(y_test) // 2
    _y_first_half = y_test.head(_half)
    _y_second_half = y_test.tail(len(y_test) - _half)

    fc_local.observe(_y_first_half)
    y_pred_updated = fc_local.predict(forecasting_horizon=len(_y_second_half))

    mo.md(
        f"Observed {_half} new rows → predicted next {len(_y_second_half)} steps.\n\n"
        f"**Updated prediction columns**: {sorted(c for c in y_pred_updated.columns if c != 'time')}"
    )
    return (y_pred_updated,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Comparison: Local vs Global

    Compare [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster/) (independent clones) against the
    default `panel_strategy="global"` (shared hyperparameters, per-group
    transformers) visually and with per-timestep scoring.
    """)


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, horizon, fetch_tourism_quarterly, train_test_split):
    _tourism = fetch_tourism_quarterly().frame
    _tourist_cols = [f"T{i}__tourists" for i in range(3, 11)]
    _y2 = _tourism.select("time", *_tourist_cols).drop_nulls()
    y_train2, y_test2 = train_test_split(_y2, test_size=0.2)

    fc_global = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 2, 4]),
        panel_strategy="global",
    )
    fc_global.fit(y_train2, forecasting_horizon=horizon)
    y_pred_global = fc_global.predict(forecasting_horizon=horizon)
    return fc_global, y_pred_global, y_test2, y_train2


@app.cell
def _(
    LagTransformer,
    LocalPanelForecaster,
    PointReductionForecaster,
    Ridge,
    horizon,
    y_train2,
):
    fc_local2 = LocalPanelForecaster(
        forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 2, 4]),
        ),
    )
    fc_local2.fit(y_train2, forecasting_horizon=horizon)
    y_pred_local2 = fc_local2.predict(forecasting_horizon=horizon)
    return fc_local2, y_pred_local2


@app.cell
def _(plot_forecast, y_pred_global, y_pred_local2, y_test2, y_train2):
    plot_forecast(
        y_test2,
        {"Global": y_pred_global, "Local": y_pred_local2},
        y_train=y_train2,
        n_history=12,
        groups=["T3", "T4", "T5"],
        title="Local vs Global: Forecast Comparison",
    )


@app.cell
def _(MeanAbsoluteError, plot_score_time_series, y_pred_global, y_pred_local2, y_test2, y_train2):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train2)
    plot_score_time_series(
        _scorer,
        y_test2,
        {"Global": y_pred_global, "Local": y_pred_local2},
        groups=["T3", "T4", "T5"],
        title="Local vs Global: MAE Over Time",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Wrapping Different Forecaster Types

    [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster/) works with any forecaster, including
    simple baselines like [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/).
    """)


@app.cell
def _(LocalPanelForecaster, MeanAbsoluteError, SeasonalNaive, horizon, mo, y_test2, y_train2):
    fc_naive_local = LocalPanelForecaster(
        forecaster=SeasonalNaive(seasonality=4),
    )
    fc_naive_local.fit(y_train2, forecasting_horizon=horizon)
    _y_pred_naive = fc_naive_local.predict(forecasting_horizon=horizon)

    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train2)
    _naive_mae = round(float(_scorer.score(y_test2, _y_pred_naive)), 1)

    mo.md(
        f"**SeasonalNaive (local)** overall MAE: {_naive_mae}\n\n"
        "Each state gets its own `SeasonalNaive` clone with `seasonality=4` "
        "(quarterly pattern)."
    )
    return (fc_naive_local,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-vintage Scoring

    The `observe_predict` method with `stride=1` produces one forecast per
    observation point, creating multiple *vintages*. Each vintage represents
    a different forecast origin, so you can analyse how accuracy evolves as
    the model absorbs more data.
    """)


@app.cell
def _(deepcopy, fc_global, horizon, y_test2):
    _vintage_model = deepcopy(fc_global)
    y_pred_vintages = _vintage_model.observe_predict(
        y=y_test2,
        stride=1,
        forecasting_horizon=horizon,
    )
    print(f"Vintages: {y_pred_vintages['vintage_time'].n_unique()}")
    y_pred_vintages.head(10)
    return (y_pred_vintages,)


@app.cell
def _(MeanAbsoluteError, y_train2):
    vintage_scorer = MeanAbsoluteError()
    vintage_scorer.fit(y_train2)
    return (vintage_scorer,)


@app.cell
def _(vintage_scorer, plot_score_per_vintage, y_pred_vintages, y_test2):
    plot_score_per_vintage(
        vintage_scorer,
        y_test2,
        y_pred_vintages,
        title="MAE per Forecast Vintage",
        y_label="MAE",
        height=380,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Panel strategy overview**: See `examples/panel_reduction.py`
    - **Per-group specialisation**: See [`examples/point/panel_forecasting.py`](/examples/panel-data/panel_forecasting/) for [`ColumnForecaster`](/pages/api/generated/yohou.compose.column_forecaster.ColumnForecaster/)
    - **Composition patterns**: See [`examples/compose/panel_pipelines.py`](/examples/panel-data/panel_pipelines/)
    - **Panel intervals**: See [`examples/interval/panel_intervals.py`](/examples/panel-data/panel_intervals/)
    """)


if __name__ == "__main__":
    app.run()
