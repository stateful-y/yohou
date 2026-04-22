# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Panel Preprocessing",
    "description": "Automatic panel-aware transformation (StandardScaler, rolling stats, imputation) plus manual per-group workflows with get_group_df and dict_to_panel.",
    "category": "how-to",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Panel Data Preprocessing

    Yohou transformers detect panel columns (containing `__`) and apply
    operations per-group automatically. This notebook shows the automatic
    path, then the manual per-group workflow using [`get_group_df`](/pages/api/generated/yohou.utils.panel.get_group_df/) and
    [`dict_to_panel`](/pages/api/generated/yohou.utils.panel.dict_to_panel/).

    This notebook shows how to perform automatic panel-aware transformation (StandardScaler, rolling stats, imputation) plus manual per-group workflows with get_group_df and dict_to_panel.
    """)


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
    from yohou.utils.panel import dict_to_panel, get_group_df, inspect_panel

    return (
        LagTransformer,
        RollingStatisticsTransformer,
        SimpleTimeImputer,
        StandardScaler,
        dict_to_panel,
        fetch_tourism_quarterly,
        get_group_df,
        inspect_panel,
        pl,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Inspect Panel Structure

    [`inspect_panel`](/pages/api/generated/yohou.utils.panel.inspect_panel/) splits the columns of a DataFrame into global columns
    (no `__` separator) and panel groups (columns prefixed with
    `<group>__<member>`). This is the starting point for any panel-aware
    workflow.
    """)


@app.cell
def _(fetch_tourism_quarterly, inspect_panel, mo):
    _bunch = fetch_tourism_quarterly()
    # Select 8 series with uniform length for a manageable panel demo
    _selected = [f"T{i}__tourists" for i in range(3, 11)]
    tourism = _bunch.frame.select("time", *_selected).drop_nulls()
    _globals, panel_groups = inspect_panel(tourism)

    mo.md(
        f"**Shape**: {tourism.shape}\n\n"
        f"**Global columns**: {_globals}\n\n"
        f"**Panel groups**: {list(panel_groups.keys())}\n\n"
        f"**Columns per group**: {[len(v) for v in panel_groups.values()]}"
    )
    return (tourism,)


@app.cell
def _(plot_time_series, tourism):
    plot_time_series(tourism, title="Panel Tourism Data (8 Groups)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Automatic Panel-Aware Transformers

    When a transformer receives panel data, it detects the `__` separator
    and processes each group independently while preserving the naming
    convention.
    """)


@app.cell
def _(StandardScaler, plot_time_series, tourism):
    scaler = StandardScaler()
    scaler.fit(tourism)
    tourism_scaled = scaler.transform(tourism)
    plot_time_series(tourism_scaled, title="StandardScaler: Panel Data (z-scored per group)")


@app.cell
def _(RollingStatisticsTransformer, plot_time_series, tourism):
    rolling = RollingStatisticsTransformer(window_size=4, statistics=["mean", "std"])
    rolling.fit(tourism)
    tourism_rolling = rolling.transform(tourism)
    plot_time_series(tourism_rolling, title="Rolling Mean & Std (window=4, per group)")
    return (tourism_rolling,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Lag Transformer on Panel Data

    [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/) creates lagged copies of every panel column
    independently. The output columns follow the pattern
    `<group>__<member>_lag_<k>`, preserving the panel naming convention so
    downstream forecasters can still identify groups.
    """)


@app.cell
def _(LagTransformer, plot_time_series, tourism):
    lag_tf = LagTransformer(lag=[1, 4])
    lag_tf.fit(tourism)
    tourism_lagged = lag_tf.transform(tourism)
    plot_time_series(tourism_lagged, title="Lag Transformer (lag=[1, 4], per group)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Manual Per-Group Extraction

    Use [`get_group_df`](/pages/api/generated/yohou.utils.panel.get_group_df/) to extract a single group as a standard DataFrame
    with unprefixed column names. Useful for custom per-group analysis.
    """)


@app.cell
def _(get_group_df, mo, pl, tourism):
    _schema = {"tourists": pl.Float64}
    t3_df = get_group_df(tourism, "T3", _schema)

    mo.md(
        f"**T3 group shape**: {t3_df.shape}\n\n"
        f"**Columns**: {t3_df.columns}\n\n"
        f"Note: column is now `tourists` (unprefixed), not `T3__tourists`."
    )
    return (t3_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_time_series`](/pages/api/generated/yohou.plotting.exploration.plot_time_series/) renders the extracted group as a standard (non-panel)
    time series, since the column names no longer contain a `__` prefix.
    """)


@app.cell
def _(plot_time_series, t3_df):
    plot_time_series(t3_df, title="T3: Tourists (Extracted Group)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. dict_to_panel: Reassemble Panel Data

    After per-group processing, recombine results into a single panel
    DataFrame using [`dict_to_panel`](/pages/api/generated/yohou.utils.panel.dict_to_panel/).
    """)


@app.cell
def _(StandardScaler, dict_to_panel, get_group_df, pl, plot_time_series, tourism):
    _schema = {"tourists": pl.Float64}
    _groups = {}
    for _group in ["T3", "T4", "T5"]:
        _df = get_group_df(tourism, _group, _schema)
        _sc = StandardScaler()
        _sc.fit(_df)
        _groups[_group] = _sc.transform(_df)

    panel_again = dict_to_panel(_groups)
    plot_time_series(panel_again, title="Reassembled Panel (T3, T4, T5 z-scored)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Handling Missing Values in Panel Data

    [`SimpleTimeImputer`](/pages/api/generated/yohou.preprocessing.imputation.SimpleTimeImputer/) fills missing values per-group when working
    with panel data.
    """)


@app.cell
def _(SimpleTimeImputer, pl, plot_time_series, tourism):
    # Inject some nulls for demonstration
    _mask = pl.Series([i % 7 == 0 for i in range(len(tourism))])
    _tourism_missing = tourism.with_columns(
        pl.when(_mask).then(None).otherwise(pl.col("T3__tourists")).alias("T3__tourists"),
        pl.when(_mask).then(None).otherwise(pl.col("T4__tourists")).alias("T4__tourists"),
    )

    imputer = SimpleTimeImputer(method="linear")
    imputer.fit(_tourism_missing)
    _tourism_filled = imputer.transform(_tourism_missing)

    # Show T3 before and after imputation
    _combined = (
        _tourism_filled
        .select("time", "T3__tourists")
        .rename({"T3__tourists": "imputed"})
        .join(
            _tourism_missing.select("time", "T3__tourists").rename({"T3__tourists": "with gaps"}),
            on="time",
        )
    )
    plot_time_series(_combined, title="T3: Panel Imputation (linear, per group)")



if __name__ == "__main__":
    app.run()
