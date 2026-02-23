"""Panel Data Preprocessing.

Shows how transformers work with panel time series automatically and
demonstrates manual per-group extraction with get_group_df / dict_to_panel.
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Panel Data Preprocessing

    Yohou transformers detect panel columns (containing `__`) and apply
    operations per-group automatically. This notebook shows the automatic
    path, then the manual per-group workflow using `get_group_df` and
    `dict_to_panel`.

    ## What You'll Learn

    - Automatic panel-aware transformation with `StandardScaler`, rolling statistics, imputation
    - Manual per-group extraction: `get_group_df()` and `dict_to_panel()`
    - Inspecting panel structure with `inspect_locality()`
    - Custom per-group preprocessing pipelines
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl

    from yohou.datasets import fetch_tourism_quarterly
    from yohou.plotting import plot_time_series
    from yohou.preprocessing import (
        LagTransformer,
        RollingStatisticsTransformer,
        SimpleTimeImputer,
        StandardScaler,
    )
    from yohou.utils.panel import dict_to_panel, get_group_df, inspect_locality

    return (
        LagTransformer,
        RollingStatisticsTransformer,
        SimpleTimeImputer,
        StandardScaler,
        dict_to_panel,
        get_group_df,
        inspect_locality,
        fetch_tourism_quarterly,
        pl,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Inspect Panel Structure
    """)
    return


@app.cell
def _(inspect_locality, fetch_tourism_quarterly, mo, pl):
    _bunch = fetch_tourism_quarterly()
    # Select first 8 series for a manageable panel demo
    _selected = [f"T{i}__tourists" for i in range(1, 9)]
    tourism = _bunch.frame.select("time", *_selected).drop_nulls()
    _globals, panel_groups = inspect_locality(tourism)

    mo.md(
        f"**Shape**: {tourism.shape}\n\n"
        f"**Global columns**: {_globals}\n\n"
        f"**Panel groups**: {list(panel_groups.keys())}\n\n"
        f"**Columns per group**: {[len(v) for v in panel_groups.values()]}\n\n"
        f"**All panel columns**: {[c for c in tourism.columns if c != 'time']}"
    )
    return panel_groups, tourism


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Automatic Panel-Aware Transformers

    When a transformer receives panel data, it detects the `__` separator
    and processes each group independently while preserving the naming
    convention.
    """)
    return


@app.cell
def _(StandardScaler, mo, tourism):
    scaler = StandardScaler()
    scaler.fit(tourism)
    tourism_scaled = scaler.transform(tourism)

    mo.md(
        f"**Scaled columns**: {tourism_scaled.columns}\n\n"
        f"Column naming preserved: all panel columns still contain `__`.\n\n"
        f"Mean of scaled target (T1): "
        f"{tourism_scaled['T1__tourists'].mean():.4f} (should be ~0.0)"
    )
    return scaler, tourism_scaled


@app.cell
def _(RollingStatisticsTransformer, tourism):
    rolling = RollingStatisticsTransformer(window_size=4, statistics=["mean", "std"])
    rolling.fit(tourism)
    tourism_rolling = rolling.transform(tourism)
    tourism_rolling.head()
    return rolling, tourism_rolling


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Rolling statistics are computed per-group and the output columns
    retain the panel naming convention. Check the output columns to see
    the generated feature names.
    """)
    return


@app.cell
def _(mo, tourism_rolling):
    mo.md(
        f"**Output columns**: {tourism_rolling.columns}\n\n"
        f"**Rows**: {len(tourism_rolling)} "
        f"(reduced from original due to window_size=4 observation horizon)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Lag Transformer on Panel Data
    """)
    return


@app.cell
def _(LagTransformer, mo, tourism):
    lag_tf = LagTransformer(lag=[1, 4])
    lag_tf.fit(tourism)
    tourism_lagged = lag_tf.transform(tourism)

    mo.md(
        f"**Lag output columns**: {tourism_lagged.columns}\n\n"
        f"Lags applied per group: each panel column gets `_lag_1` and `_lag_4` variants."
    )
    return lag_tf, tourism_lagged


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Manual Per-Group Extraction

    Use `get_group_df` to extract a single group as a standard DataFrame
    with unprefixed column names. Useful for custom per-group analysis.
    """)
    return


@app.cell
def _(get_group_df, mo, pl, tourism):
    _schema = {"tourists": pl.Float64}
    t1_df = get_group_df(tourism, "T1", _schema)

    mo.md(
        f"**T1 group shape**: {t1_df.shape}\n\n"
        f"**Columns**: {t1_df.columns}\n\n"
        f"Note: column is now `tourists` (unprefixed), not `T1__tourists`."
    )
    return (t1_df,)


@app.cell
def _(t1_df, plot_time_series):
    plot_time_series(t1_df, title="T1: Tourists (Extracted Group)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. dict_to_panel: Reassemble Panel Data

    After per-group processing, recombine results into a single panel
    DataFrame using `dict_to_panel`.
    """)
    return


@app.cell
def _(StandardScaler, dict_to_panel, get_group_df, mo, pl, tourism):
    _schema = {"tourists": pl.Float64}
    _groups = {}
    for _group in ["T1", "T2", "T3"]:
        _df = get_group_df(tourism, _group, _schema)
        _sc = StandardScaler()
        _sc.fit(_df)
        _groups[_group] = _sc.transform(_df)

    panel_again = dict_to_panel(_groups)
    mo.md(
        f"**Reassembled panel columns**: {panel_again.columns}\n\n"
        f"**Shape**: {panel_again.shape}\n\n"
        "The `dict_to_panel` function re-prefixes columns with `<group>__`."
    )
    return (panel_again,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Handling Missing Values in Panel Data

    `SimpleTimeImputer` fills missing values per-group when working
    with panel data.
    """)
    return


@app.cell
def _(SimpleTimeImputer, mo, pl, tourism):
    # Inject some nulls for demonstration
    _mask = pl.Series([i % 7 == 0 for i in range(len(tourism))])
    _tourism_missing = tourism.with_columns(
        pl.when(_mask).then(None).otherwise(pl.col("T1__tourists")).alias("T1__tourists"),
        pl.when(_mask)
        .then(None)
        .otherwise(pl.col("T2__tourists"))
        .alias("T2__tourists"),
    )
    _null_count = _tourism_missing.null_count().drop("time")

    imputer = SimpleTimeImputer(method="linear")
    imputer.fit(_tourism_missing)
    _tourism_filled = imputer.transform(_tourism_missing)
    _null_after = _tourism_filled.null_count().drop("time")

    mo.md(
        f"**Nulls before**: {_null_count.row(0)}\n\n"
        f"**Nulls after**: {_null_after.row(0)}\n\n"
        "Linear interpolation applied per group."
    )
    return (imputer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Automatic panel handling**: All transformers process panel groups independently
    - **Column naming preserved**: `StandardScaler`, `LagTransformer`, `RollingStatisticsTransformer` all maintain the `__` convention
    - **`inspect_locality()`**: Explores panel structure (global vs group columns)
    - **`get_group_df()`**: Extracts one group with unprefixed column names
    - **`dict_to_panel()`**: Reassembles per-group DataFrames into a panel DataFrame
    - **Custom workflows**: Extract → process → reassemble for per-group specialisation

    ## Next Steps

    - **Panel stationarity**: See `examples/stationarity/panel_stationarity.py`
    - **Panel pipelines**: See `examples/compose/panel_pipelines.py`
    - **Panel forecasting**: See `examples/point/panel_forecasting.py`
    """)
    return


if __name__ == "__main__":
    app.run()
