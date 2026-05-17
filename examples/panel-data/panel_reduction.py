# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "How to Use LocalPanelForecaster for Per-Series Models",
    "description": "Fit a separate reduction model for each panel series using LocalPanelForecaster and compare with a shared ColumnForecaster.",
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
    # Panel Reduction Forecasting

    Panel data contains multiple related time series identified by a common
    naming convention.  Yohou uses a `__` (double underscore) separator to
    distinguish panel groups from series names.

    This notebook demonstrates how [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) handles panel
    data through different `panel_strategy` options, plus the fully
    independent [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster/) approach.

    ## 1. Panel Data Conventions

    Panel columns follow the `<group>__<series>` naming pattern.
    [`inspect_panel`](/pages/api/generated/yohou.utils.panel.inspect_panel/) discovers groups automatically. We load the KDD Cup
    2018 air quality dataset, a multivariate panel with 3 Beijing
    stations, each monitoring 6 pollutants.
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import LocalPanelForecaster
    from yohou.datasets import fetch_kdd_cup
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import train_test_split
    from yohou.plotting import (
        plot_forecast,
        plot_score_heatmap,
        plot_score_per_vintage,
        plot_score_time_series,
        plot_time_series,
    )
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer
    from yohou.utils.panel import inspect_panel

    return (
        LagTransformer,
        LocalPanelForecaster,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        deepcopy,
        fetch_kdd_cup,
        inspect_panel,
        pl,
        plot_forecast,
        plot_score_heatmap,
        plot_score_per_vintage,
        plot_score_time_series,
        plot_time_series,
        train_test_split,
    )


@app.cell
def _(fetch_kdd_cup, inspect_panel, mo):
    _bunch = fetch_kdd_cup(n_groups=3)
    store = _bunch.frame.drop_nulls().tail(300)

    global_cols, panel_groups = inspect_panel(store)

    mo.md(
        f"**Dataset shape**: {store.shape}\n\n"
        f"**Global columns** (shared): `{global_cols}`\n\n"
        f"**Panel groups** ({len(panel_groups)}):\n\n"
        + "\n".join(f"- `{name}`: {cols}" for name, cols in sorted(panel_groups.items()))
    )
    return panel_groups, store


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_time_series`](/pages/api/generated/yohou.plotting.exploration.plot_time_series/) renders all panel columns in one figure. Each `__`-prefixed
    series gets its own trace, giving an overview of pollutant levels across stations.
    """)


@app.cell
def _(plot_time_series, store):
    plot_time_series(
        store,
        title="KDD Cup 2018: Air Quality Panel",
    )


@app.cell
def _(mo, store, train_test_split):
    _target_cols = [c for c in store.columns if c != "time"]
    y = store.select("time", *_target_cols)
    y_train, y_test = train_test_split(y, test_size=0.15)
    horizon = len(y_test)

    mo.md(
        f"**Targets**: {len(_target_cols)} pollutant columns\n\n"
        f"**Train**: {len(y_train)} rows | **Test**: {len(y_test)} rows | "
        f"**Horizon**: {horizon}"
    )
    return horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Strategy: Global (Default)

    With `panel_strategy="global"` (the default), each panel group gets
    **independent observation buffers and transformers** but shares the
    **same model hyperparameters**.  The forecaster detects groups via the
    `__` separator, strips prefixes, fits per-group transformers, and
    pools the transformed data into a single estimator.

    This is efficient when groups share similar dynamics and you want
    one set of hyperparameters to govern all groups.
    """)


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, horizon, y_train):
    fc_global = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 24]),
        panel_strategy="global",  # default, shown explicitly for clarity
    )
    fc_global.fit(y_train, forecasting_horizon=horizon)
    y_pred_global = fc_global.predict(forecasting_horizon=horizon)
    return (y_pred_global,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) with `groups` shows predictions for selected
    groups in a faceted layout, with training history trimmed to the last 48 steps.
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
        title="Global Strategy: One Model, Per-Group State",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Strategy: Multivariate

    With `panel_strategy="multivariate"`, the forecaster **skips panel
    detection entirely**.  All `__`-prefixed columns are treated as
    ordinary multivariate columns.  A single transformer and model see
    the full wide DataFrame, enabling **cross-group feature interactions**.

    This is useful when groups are correlated and you want the model to
    learn from inter-group relationships (e.g. co-located pollution monitors,
    spatially correlated air quality patterns).
    """)


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, horizon, y_train):
    fc_multivariate = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 24]),
        panel_strategy="multivariate",
    )
    fc_multivariate.fit(y_train, forecasting_horizon=horizon)
    y_pred_multivariate = fc_multivariate.predict(forecasting_horizon=horizon)
    return (y_pred_multivariate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) for the multivariate strategy shows the same groups,
    letting you compare predictions against the global strategy above.
    """)


@app.cell
def _(plot_forecast, y_pred_multivariate, y_test, y_train):
    _groups = sorted({c.split("__")[0] for c in y_train.columns if "__" in c})
    plot_forecast(
        y_test,
        y_pred_multivariate,
        y_train=y_train,
        n_history=48,
        groups=_groups[:2],
        title="Multivariate Strategy: Cross-Group Features",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Strategy: Local (Independent Clones)

    [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster/) wraps any forecaster and fits **completely
    independent clones** per panel group.  Unlike `panel_strategy="global"`
    which shares hyperparameters across groups, each clone has its own
    parameters.  This is the right choice when groups are heterogeneous.
    """)


@app.cell
def _(
    LagTransformer,
    LocalPanelForecaster,
    PointReductionForecaster,
    Ridge,
    horizon,
    y_train,
):
    fc_local = LocalPanelForecaster(
        forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 24]),
        ),
    )
    fc_local.fit(y_train, forecasting_horizon=horizon)
    y_pred_local = fc_local.predict(forecasting_horizon=horizon)
    return (y_pred_local,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) for the local strategy shows fully independent per-group
    clones, which may capture group-specific dynamics better.
    """)


@app.cell
def _(plot_forecast, y_pred_local, y_test, y_train):
    _groups = sorted({c.split("__")[0] for c in y_train.columns if "__" in c})
    plot_forecast(
        y_test,
        y_pred_local,
        y_train=y_train,
        n_history=48,
        groups=_groups[:2],
        title="Local Strategy: Independent Per-Group Clones",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Strategy Comparison

    Compare all three strategies using per-group MAE (stepwise+vintagewise aggregation
    produces one score per group, averaged across timesteps).
    """)


@app.cell
def _(
    MeanAbsoluteError,
    mo,
    panel_groups,
    pl,
    y_pred_global,
    y_pred_local,
    y_pred_multivariate,
    y_test,
    y_train,
):
    _scorer_sv = MeanAbsoluteError(aggregation_method=["stepwise", "vintagewise"])
    _scorer_sv.fit(y_train)

    _scores_global = _scorer_sv.score(y_test, y_pred_global)
    _scores_multi = _scorer_sv.score(y_test, y_pred_multivariate)
    _scores_local = _scorer_sv.score(y_test, y_pred_local)

    _rows = []
    for _group in sorted(panel_groups.keys()):
        _group_cols = [c for c in _scores_global.columns if c.startswith(f"{_group}__")]
        if _group_cols:
            _rows.append({
                "Group": _group,
                "Global MAE": round(sum(_scores_global[c].item() for c in _group_cols) / len(_group_cols), 2),
                "Multivariate MAE": round(sum(_scores_multi[c].item() for c in _group_cols) / len(_group_cols), 2),
                "Local MAE": round(sum(_scores_local[c].item() for c in _group_cols) / len(_group_cols), 2),
            })

    comparison = pl.DataFrame(_rows)
    mo.ui.table(comparison)


@app.cell
def _(
    MeanAbsoluteError,
    plot_score_time_series,
    y_pred_global,
    y_pred_local,
    y_pred_multivariate,
    y_test,
    y_train,
):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)

    plot_score_time_series(
        _scorer,
        y_test,
        {
            "Global": y_pred_global,
            "Multivariate": y_pred_multivariate,
            "Local": y_pred_local,
        },
        title="MAE Over Time: Strategy Comparison",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.5 Reduction Strategies on Panel Data

    The `reduction_strategy` parameter works with all panel strategies.
    Below we compare multi-output (default) vs direct on the global
    panel strategy. Each strategy trains on the pooled panel data
    but differs in **how** multi-step targets are handled.
    """)


@app.cell
def _(
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    horizon,
    mo,
    y_test,
    y_train,
):
    _reduction_scores = {}
    for _strat in ["multi-output", "direct"]:
        _fc = PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 24]),
            panel_strategy="global",
            reduction_strategy=_strat,
        )
        _fc.fit(y_train, forecasting_horizon=horizon)
        _pred = _fc.predict(forecasting_horizon=horizon)
        _mae = MeanAbsoluteError()
        _mae.fit(y_train)
        _reduction_scores[_strat] = _mae.score(y_test.head(len(_pred)), _pred)

    mo.ui.table(
        [{"reduction_strategy": k, "MAE": f"{v:.2f}"} for k, v in _reduction_scores.items()],
        label="MAE by reduction_strategy (global panel)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## When to Use Each Strategy

    | Strategy | Parameter | Best For |
    |---|---|---|
    | **Global** | `panel_strategy="global"` (default) | Homogeneous groups sharing similar dynamics |
    | **Multivariate** | `panel_strategy="multivariate"` | Correlated groups with cross-group interactions |
    | **Local** | `LocalPanelForecaster(forecaster=...)` | Heterogeneous groups needing fully independent models |

    **Rules of thumb:**

    - Start with `"global"` (simplest, good default)
    - Try `"multivariate"` if groups are strongly correlated (e.g. nearby monitoring stations)
    - Use [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster/) when groups differ significantly (e.g. different cities, sensor types)
    - Combine with [`ColumnForecaster`](/pages/api/generated/yohou.compose.column_forecaster.ColumnForecaster/) to assign different model families per group

    ## Next Steps

    - **Reduction strategies**: See [`reduction_strategies.py`](/examples/getting-started/reduction_strategies/) for multi-output vs direct vs dir-rec comparison
    - **LocalPanelForecaster deep dive**: See [`examples/compose/local_panel_forecaster.py`](/examples/panel-data/local_panel_forecaster/)
    - **Per-group specialisation**: See [`examples/point/panel_forecasting.py`](/examples/panel-data/panel_forecasting/)
    - **Composition patterns**: See [`examples/compose/panel_pipelines.py`](/examples/panel-data/panel_pipelines/)
    - **Panel intervals**: See [`examples/interval/panel_intervals.py`](/examples/panel-data/panel_intervals/)
    - **Panel cross-validation**: See [`examples/model_selection/panel_cross_validation.py`](/examples/panel-data/panel_cross_validation/)
    """)


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
def _(deepcopy, fc_global, horizon, y_test):
    _vintage_model = deepcopy(fc_global)
    y_pred_vintages = _vintage_model.observe_predict(
        y=y_test,
        stride=1,
        forecasting_horizon=horizon,
    )
    print(f"Vintages: {y_pred_vintages['vintage_time'].n_unique()}")
    y_pred_vintages.head(10)
    return (y_pred_vintages,)


@app.cell
def _(MeanAbsoluteError, y_train):
    vintage_scorer = MeanAbsoluteError()
    vintage_scorer.fit(y_train)
    return (vintage_scorer,)


@app.cell
def _(vintage_scorer, plot_score_per_vintage, y_pred_vintages, y_test):
    plot_score_per_vintage(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="MAE per Forecast Vintage",
        y_label="MAE",
        height=380,
    )


@app.cell
def _(vintage_scorer, plot_score_heatmap, y_pred_vintages, y_test):
    plot_score_heatmap(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="Score Heatmap (Step x Vintage)",
    )


if __name__ == "__main__":
    app.run()
