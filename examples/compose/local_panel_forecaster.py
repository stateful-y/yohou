"""LocalPanelForecaster: Independent Per-Group Forecasting.

Demonstrates LocalPanelForecaster for fitting completely independent
forecaster clones per panel group, with parallel execution and
selective group operations.
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
async def _():
    import sys as _sys

    if "pyodide" in _sys.modules:
        import micropip

        await micropip.install(["plotly", "scikit-learn", "yohou"])
    _ = None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LocalPanelForecaster

    `LocalPanelForecaster` wraps any forecaster and fits **completely
    independent clones** per panel group via `sklearn.base.clone()`.
    Each clone sees standard (non-panel) data with group prefixes stripped,
    and predictions are reassembled back into prefixed-column format.

    ## What You'll Learn

    - Basic usage: wrap any forecaster for per-group independence
    - Parallel fitting with `n_jobs`
    - Selective group prediction, observe, and rewind
    - Comparison with `panel_strategy="global"` (shared model)
    - Interval forecasting with `LocalPanelForecaster`
    - Accessing individual per-group clones

    ## When to Use

    Use `LocalPanelForecaster` when panel groups are **heterogeneous** and
    a single pooled model cannot capture group-specific dynamics.  Each
    clone gets its own fitted parameters, no parameter sharing.

    For homogeneous groups, the default `panel_strategy="global"` is
    simpler and often sufficient.
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeRegressor

    from yohou.compose import LocalPanelForecaster
    from yohou.datasets import fetch_tourism_quarterly
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer
    from yohou.utils.panel import inspect_locality

    return (
        DecisionTreeRegressor,
        LagTransformer,
        LocalPanelForecaster,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
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
    ## 1. Prepare Panel Data

    Australian Tourism: quarterly trips for 8 states.  Each state is a
    panel group with the `__` separator (e.g. `T1__tourists`).
    """)
    return


@app.cell
def _(inspect_locality, fetch_tourism_quarterly, mo):
    _tourism_full = fetch_tourism_quarterly().frame
    # Select first 8 series for a manageable panel demo
    _tourist_cols = [c for c in _tourism_full.columns if c.endswith("__tourists")][:8]
    tourism = _tourism_full.select("time", *_tourist_cols)
    _globals, groups = inspect_locality(tourism)
    y = tourism.select("time", *[c for c in tourism.columns if c.endswith("__tourists")])

    _split = int(len(y) * 0.80)
    y_train = y.head(_split)
    y_test = y.tail(len(y) - _split)
    horizon = len(y_test)

    mo.md(
        f"**Panel groups**: {sorted(groups.keys())}\n\n"
        f"**Train**: {len(y_train)} quarters | **Test**: {len(y_test)} quarters | "
        f"**Horizon**: {horizon}"
    )
    return groups, horizon, tourism, y, y_test, y_train


@app.cell
def _(plot_time_series, y):
    plot_time_series(y, title="Tourism Quarterly: Quarterly Tourists by Series")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Basic Usage

    Wrap a forecaster in `LocalPanelForecaster`.  Each panel group gets a
    fresh `clone()` that is fitted independently on that group's data.
    """)
    return


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


@app.cell
def _(plot_forecast, y_pred_local, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_local,
        y_train=y_train,
        n_history=12,
        panel_group_names=["T1", "T2", "T3"],
        title="LocalPanelForecaster: Top 3 Series",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Accessing Per-Group Clones

    After fitting, `forecasters_` holds a dict mapping group names to
    their fitted clones.  You can inspect individual models.
    """)
    return


@app.cell
def _(fc_local, mo):
    _rows = []
    for _name, _fc in sorted(fc_local.forecasters_.items()):
        _coefs = _fc.estimator_.coef_
        _rows.append(f"- **{_name}**: {len(_coefs)} coefficients")

    mo.md(
        f"**Fitted clones** ({len(fc_local.forecasters_)}):\n\n"
        + "\n".join(_rows)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Selective Group Operations

    Use `panel_group_names` to predict, observe, or rewind only a subset
    of groups.  This is useful when new data arrives for specific groups
    or you want group-specific analysis.
    """)
    return


@app.cell
def _(fc_local, horizon, mo):
    y_pred_top3 = fc_local.predict(
        forecasting_horizon=horizon,
        panel_group_names=["T1", "T2", "T3"],
    )
    mo.md(
        f"**Selective prediction columns**: {sorted(y_pred_top3.columns)}\n\n"
        "Only the requested groups are predicted; others are omitted."
    )
    return (y_pred_top3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Observe–Predict Workflow

    `LocalPanelForecaster` supports the observe–predict cycle for
    rolling evaluation.  New observations update only the specified
    groups' clones.
    """)
    return


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

    Compare `LocalPanelForecaster` (independent clones) against the
    default `panel_strategy="global"` (shared hyperparameters, per-group
    transformers) using groupwise MAE.
    """)
    return


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, horizon, fetch_tourism_quarterly):
    _tourism = fetch_tourism_quarterly().frame
    _tourist_cols = [c for c in _tourism.columns if c.endswith("__tourists")][:8]
    _y2 = _tourism.select("time", *_tourist_cols)
    _split2 = int(len(_y2) * 0.80)
    y_train2 = _y2.head(_split2)
    y_test2 = _y2.tail(len(_y2) - _split2)

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
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    groups,
    horizon,
    mo,
    pl,
    y_pred_global,
    y_test2,
    y_train2,
):
    fc_local2 = LocalPanelForecaster(
        forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 2, 4]),
        ),
    )
    fc_local2.fit(y_train2, forecasting_horizon=horizon)
    _y_pred_local2 = fc_local2.predict(forecasting_horizon=horizon)

    _scorer = MeanAbsoluteError(aggregation_method="timewise")
    _scorer.fit(y_train2)
    _scores_global = _scorer.score(y_test2, y_pred_global)
    _scores_local = _scorer.score(y_test2, _y_pred_local2)

    _rows = []
    for _group in sorted(groups.keys()):
        _col = f"{_group}__tourists"
        if _col in _scores_global.columns:
            _rows.append({
                "State": _group,
                "Global MAE": round(_scores_global[_col].item(), 1),
                "Local MAE": round(_scores_local[_col].item(), 1),
            })

    comparison = pl.DataFrame(_rows)
    mo.ui.table(comparison)
    return comparison, fc_local2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Wrapping Different Forecaster Types

    `LocalPanelForecaster` works with any forecaster, including
    simple baselines like `SeasonalNaive`.
    """)
    return


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
    ## Key Takeaways

    | Feature | `panel_strategy="global"` | `LocalPanelForecaster` |
    |---|---|---|
    | **Model sharing** | Shared hyperparameters | Fully independent clones |
    | **Transformers** | Per-group (automatic) | Per-group (inside each clone) |
    | **Cross-group learning** | Pooled estimator | None |
    | **Parallel fitting** | Sequential | `n_jobs` parameter |
    | **Selective operations** | `panel_group_names` on all methods | `panel_group_names` on all methods |
    | **Best for** | Homogeneous groups | Heterogeneous groups |

    ## Next Steps

    - **Panel strategy overview**: See `examples/panel_data.py`
    - **Per-group specialisation**: See `examples/point/panel_forecasting.py` for `ColumnForecaster`
    - **Composition patterns**: See `examples/compose/panel_pipelines.py`
    - **Panel intervals**: See `examples/interval/panel_intervals.py`
    """)
    return


if __name__ == "__main__":
    app.run()
