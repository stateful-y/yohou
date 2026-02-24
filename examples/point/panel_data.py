# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Panel Data Forecasting.

Demonstrates panel data conventions, panel_strategy parameter,
and how different strategies handle multi-group time series.
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
    # Panel Data Forecasting

    Panel data contains multiple related time series identified by a common
    naming convention.  Yohou uses a `__` (double underscore) separator to
    distinguish panel groups from series names.

    ## What You'll Learn

    - Panel data conventions (`<group>__<series>`) and inspection utilities
    - `panel_strategy="global"` (default): shared model, per-group state
    - `panel_strategy="multivariate"`: treat panel as wide multivariate data
    - `LocalPanelForecaster`: fully independent per-group clones
    - When to use each strategy
    - Groupwise scoring to compare strategies
    """)
    return

@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import LocalPanelForecaster
    from yohou.datasets import fetch_dominick
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer
    from yohou.utils.panel import inspect_locality

    return (
        LagTransformer,
        LocalPanelForecaster,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        inspect_locality,
        fetch_dominick,
        pl,
        plot_forecast,
        plot_time_series,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Panel Data Conventions

    Panel columns follow the `<group>__<series>` naming pattern.
    `inspect_locality` discovers groups automatically.
    """)
    return

@app.cell
def _(inspect_locality, fetch_dominick, mo, pl):
    _bunch = fetch_dominick()
    # Select 6 series with complete data (no nulls) for a manageable panel demo
    _selected = ["T7__profit", "T11__profit", "T12__profit", "T13__profit", "T15__profit", "T19__profit"]
    store = _bunch.frame.select("time", *_selected).drop_nulls()

    global_cols, panel_groups = inspect_locality(store)

    mo.md(
        f"**Dataset shape**: {store.shape}\n\n"
        f"**Global columns** (shared): `{global_cols}`\n\n"
        f"**Panel groups** ({len(panel_groups)}):\n\n"
        + "\n".join(f"- `{name}`: {cols}" for name, cols in sorted(panel_groups.items()))
    )
    return panel_groups, store

@app.cell
def _(plot_time_series, store):
    plot_time_series(
        store,
        title="Dominick: All Panel Groups",
    )
    return

@app.cell
def _(mo, store):
    _target_cols = [c for c in store.columns if c.endswith("__profit")]
    y = store.select("time", *_target_cols)
    _split = int(len(y) * 0.85)
    y_train = y.head(_split)
    y_test = y.tail(len(y) - _split)
    horizon = len(y_test)

    mo.md(
        f"**Targets**: {len(_target_cols)} profit columns\n\n"
        f"**Train**: {len(y_train)} rows | **Test**: {len(y_test)} rows | "
        f"**Horizon**: {horizon}"
    )
    return horizon, y, y_test, y_train

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
    return

@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, horizon, y_train):
    fc_global = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 7]),
        panel_strategy="global",  # default, shown explicitly for clarity
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
        title="Global Strategy: One Model, Per-Group State",
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Strategy: Multivariate

    With `panel_strategy="multivariate"`, the forecaster **skips panel
    detection entirely**.  All `__`-prefixed columns are treated as
    ordinary multivariate columns.  A single transformer and model see
    the full wide DataFrame, enabling **cross-group feature interactions**.

    This is useful when groups are correlated and you want the model to
    learn from inter-group relationships (e.g. spatial spillover effects,
    cannibalization between stores).
    """)
    return

@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, horizon, y_train):
    fc_multivariate = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 7]),
        panel_strategy="multivariate",
    )
    fc_multivariate.fit(y_train, forecasting_horizon=horizon)
    y_pred_multivariate = fc_multivariate.predict(forecasting_horizon=horizon)
    return fc_multivariate, y_pred_multivariate

@app.cell
def _(plot_forecast, y_pred_multivariate, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_multivariate,
        y_train=y_train,
        n_history=30,
        panel_group_names=["T7", "T11", "T12"],
        title="Multivariate Strategy: Cross-Group Features",
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Strategy: Local (Independent Clones)

    `LocalPanelForecaster` wraps any forecaster and fits **completely
    independent clones** per panel group.  Unlike `panel_strategy="global"`
    which shares hyperparameters across groups, each clone has its own
    parameters.  This is the right choice when groups are heterogeneous.
    """)
    return

@app.cell
def _(LagTransformer, LocalPanelForecaster, PointReductionForecaster, Ridge, horizon, y_train):
    fc_local = LocalPanelForecaster(
        forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 7]),
        ),
    )
    fc_local.fit(y_train, forecasting_horizon=horizon)
    y_pred_local = fc_local.predict(forecasting_horizon=horizon)
    return fc_local, y_pred_local

@app.cell
def _(plot_forecast, y_pred_local, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_local,
        y_train=y_train,
        n_history=30,
        panel_group_names=["T7", "T11", "T12"],
        title="Local Strategy: Independent Per-Group Clones",
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Strategy Comparison

    Compare all three strategies using per-group MAE (timewise aggregation
    produces one score per group, averaged across timesteps).
    """)
    return

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
    _scorer_timewise = MeanAbsoluteError(aggregation_method="timewise")
    _scorer_timewise.fit(y_train)

    _scores_global = _scorer_timewise.score(y_test, y_pred_global)
    _scores_multi = _scorer_timewise.score(y_test, y_pred_multivariate)
    _scores_local = _scorer_timewise.score(y_test, y_pred_local)

    _rows = []
    for _group in sorted(panel_groups.keys()):
        _col = f"{_group}__profit"
        if _col in _scores_global.columns:
            _rows.append({
                "Group": _group,
                "Global MAE": round(_scores_global[_col].item(), 2),
                "Multivariate MAE": round(_scores_multi[_col].item(), 2),
                "Local MAE": round(_scores_local[_col].item(), 2),
            })

    comparison = pl.DataFrame(_rows)
    mo.ui.table(comparison)
    return (comparison,)

@app.cell
def _(MeanAbsoluteError, mo, y_pred_global, y_pred_local, y_pred_multivariate, y_test, y_train):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _overall_global = round(float(_scorer.score(y_test, y_pred_global)), 2)
    _overall_multi = round(float(_scorer.score(y_test, y_pred_multivariate)), 2)
    _overall_local = round(float(_scorer.score(y_test, y_pred_local)), 2)

    mo.md(
        f"**Overall MAE** (aggregated across all groups and timesteps):\n\n"
        f"| Strategy | MAE |\n"
        f"|---|---|\n"
        f"| Global | {_overall_global} |\n"
        f"| Multivariate | {_overall_multi} |\n"
        f"| Local | {_overall_local} |"
    )
    return

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
    - Try `"multivariate"` if groups are strongly correlated (e.g. nearby stores)
    - Use `LocalPanelForecaster` when groups differ significantly (e.g. different products, regions)
    - Combine with `ColumnForecaster` to assign different model families per group

    ## Next Steps

    - **LocalPanelForecaster deep dive**: See `examples/compose/local_panel_forecaster.py`
    - **Per-group specialisation**: See `examples/point/panel_forecasting.py`
    - **Composition patterns**: See `examples/compose/panel_pipelines.py`
    - **Panel intervals**: See `examples/interval/panel_intervals.py`
    - **Panel cross-validation**: See `examples/model_selection/panel_cross_validation.py`
    """)
    return

if __name__ == "__main__":
    app.run()
