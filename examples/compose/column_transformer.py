# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Column-Wise Feature Transformation.

Demonstrates ColumnTransformer for applying different transforms to different
columns, including panel-aware column selection.
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
    # Column-Wise Feature Transformation

    When forecasting with multivariate data, different columns often need
    different preprocessing, demands from different regions should be scaled,
    and some features may pass through untouched.
    `ColumnTransformer` applies **distinct transformers to distinct columns**
    in a single step.

    ## What You'll Learn

    - Building a `ColumnTransformer` with named transformer-column pairs
    - Remainder handling: `"drop"` vs `"passthrough"`
    - Inspecting output feature names
    - Combining `ColumnTransformer` with `PointReductionForecaster`
    - Using `ColumnTransformer` inside a panel forecaster for automatic per-group application

    ## Prerequisites

    Familiarity with `PointReductionForecaster` and basic transformers
    (see `examples/point/reduction_forecaster.py`).
    """)

@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import ColumnTransformer
    from yohou.datasets import fetch_dominick, fetch_electricity_demand
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer, StandardScaler
    from yohou.stationarity import LogTransformer
    from yohou.utils.panel import inspect_locality

    return (
        ColumnTransformer,
        LagTransformer,
        LogTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        StandardScaler,
        inspect_locality,
        fetch_dominick,
        fetch_electricity_demand,
        pl,
        plot_forecast,
        plot_time_series,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Multivariate Data

    The Electricity Demand dataset has five state-level demand columns with
    very different dynamics: `vic__demand`, `nsw__demand`, `sa__demand`, etc.
    We use Victoria as the target and other states as exogenous features.
    """)

@app.cell
def _(fetch_electricity_demand, pl):
    _elec = fetch_electricity_demand().frame
    # Downsample to daily for manageable size (drop trailing all-null days)
    vic_daily = _elec.group_by_dynamic("time", every="1d").agg(
        pl.col("vic__demand").mean().alias("Demand"),
        pl.col("nsw__demand").mean().alias("NSW_Demand"),
        pl.col("sa__demand").mean().alias("SA_Demand"),
    ).drop_nulls()

    split_idx = int(len(vic_daily) * 0.85)
    y_train = vic_daily.head(split_idx).select("time", "Demand")
    X_train = vic_daily.head(split_idx).select("time", "NSW_Demand", "SA_Demand")
    y_test = vic_daily.tail(len(vic_daily) - split_idx).select("time", "Demand")
    X_test = vic_daily.tail(len(vic_daily) - split_idx).select("time", "NSW_Demand", "SA_Demand")

    y_train.head()
    return X_test, X_train, split_idx, vic_daily, y_test, y_train

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Build a ColumnTransformer

    Each tuple is `(name, transformer, columns)`. We scale NSW_Demand with
    `StandardScaler`, pass SA_Demand through unchanged, and drop any remaining
    columns.
    """)

@app.cell
def _(ColumnTransformer, StandardScaler):
    ct = ColumnTransformer(
        transformers=[
            ("scale_nsw", StandardScaler(), ["NSW_Demand"]),
            ("sa", "passthrough", ["SA_Demand"]),
        ],
        remainder="drop",
    )
    ct
    return (ct,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. ColumnTransformer Inside a Forecaster

    Pass the `ColumnTransformer` as `feature_transformer` to
    `PointReductionForecaster`. The forecaster calls `.fit_transform()` on `X`
    at fit time and `.transform()` on `X` at predict time.
    """)

@app.cell
def _(
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    ct,
    X_test,
    X_train,
    y_test,
    y_train,
):
    forecaster_ct = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=ct,
    )

    forecasting_horizon = len(y_test)
    forecaster_ct.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)
    y_pred_ct = forecaster_ct.predict(X_test, forecasting_horizon=forecasting_horizon)
    return forecaster_ct, forecasting_horizon, y_pred_ct

@app.cell
def _(plot_forecast, y_pred_ct, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_ct,
        y_train=y_train,
        n_history=60,
        title="Forecast with ColumnTransformer (NSW scaled, SA passthrough)",
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Remainder Handling

    Setting `remainder="passthrough"` keeps all columns not explicitly listed.
    Compare with `remainder="drop"` (the default) to see the effect on
    feature availability.
    """)

@app.cell
def _(ColumnTransformer, StandardScaler, mo):
    ct_pass = ColumnTransformer(
        transformers=[
            ("scale_nsw", StandardScaler(), ["NSW_Demand"]),
        ],
        remainder="passthrough",
    )
    mo.md(
        f"**With `remainder='passthrough'`**: SA_Demand is kept automatically.\n\n"
        f"Transformer configuration: {ct_pass}"
    )
    return (ct_pass,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Inspecting Feature Names

    After fitting, `get_feature_names_out()` reveals the output column names,
    which include the transformer name prefix when `verbose_feature_names_out=True`
    (the default).
    """)

@app.cell
def _(ColumnTransformer, StandardScaler, X_train, mo):
    ct_verbose = ColumnTransformer(
        transformers=[
            ("scale_nsw", StandardScaler(), ["NSW_Demand"]),
            ("sa", "passthrough", ["SA_Demand"]),
        ],
        verbose_feature_names_out=True,
    )
    ct_verbose.fit(X_train)
    _names = ct_verbose.get_feature_names_out()
    mo.md(f"**Feature names (verbose=True)**: {_names}")
    return (ct_verbose,)

@app.cell
def _(ColumnTransformer, StandardScaler, X_train, mo):
    _ct_short = ColumnTransformer(
        transformers=[
            ("scale_nsw", StandardScaler(), ["NSW_Demand"]),
            ("sa", "passthrough", ["SA_Demand"]),
        ],
        verbose_feature_names_out=False,
    )
    _ct_short.fit(X_train)
    _names = _ct_short.get_feature_names_out()
    mo.md(f"**Feature names (verbose=False)**: {_names}")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. With Panel Data

    When a `ColumnTransformer` is used as the `feature_transformer` inside a
    panel-aware forecaster, _yohou automatically applies it per group_.
    The forecaster decomposes the panel into individual groups, applies the
    transformer to each group's unprefixed columns, and reassembles the result.

    Here we use the Dominick dataset (9 panel groups).
    """)

@app.cell
def _(inspect_locality, fetch_dominick, mo, pl):
    _dom_full = fetch_dominick().frame
    _profit_cols = ["T7__profit", "T11__profit", "T12__profit", "T13__profit", "T15__profit", "T19__profit", "T22__profit", "T23__profit", "T24__profit"]
    store = _dom_full.select("time", *_profit_cols)
    _globals, _groups = inspect_locality(store)
    mo.md(
        f"**Panel groups**: {len(_groups)} groups\n\n"
        f"**First group columns**: {list(_groups.values())[0]}"
    )

    # Panel data: y contains all `__profit` columns, no separate X needed
    _split = int(len(store) * 0.9)
    y_train_panel = store.head(_split).select(
        "time", *[c for c in store.columns if c.endswith("__profit")]
    )
    y_test_panel = store.tail(len(store) - _split).select(
        "time", *[c for c in store.columns if c.endswith("__profit")]
    )

    y_train_panel.head()
    return store, y_test_panel, y_train_panel

@app.cell
def _(
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    plot_forecast,
    y_test_panel,
    y_train_panel,
):
    forecaster_panel = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 7]),
    )

    _horizon = min(len(y_test_panel), 14)
    forecaster_panel.fit(y_train_panel, forecasting_horizon=_horizon)
    _y_pred_panel = forecaster_panel.predict(forecasting_horizon=_horizon)

    plot_forecast(
        y_test_panel,
        _y_pred_panel,
        y_train=y_train_panel,
        n_history=30,
        panel_group_names=["T7", "T11", "T12"],
        title="Panel Forecast: First 3 Groups",
    )
    return (forecaster_panel,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **ColumnTransformer** applies distinct transformers to distinct columns in a single step
    - **Remainder handling** controls what happens to unlisted columns: `"drop"` (default) discards them, `"passthrough"` keeps them
    - **verbose_feature_names_out** prefixes output names with transformer names for traceability
    - **Panel integration** is automatic: when used inside a forecaster, ColumnTransformer is applied per group to unprefixed columns
    - **Forecaster composition**: Pass ColumnTransformer as `feature_transformer` to `PointReductionForecaster`

    ## Next Steps

    - **Parallel features**: See `examples/compose/feature_union.py` for combining transformers in parallel
    - **Pipeline composition**: See `examples/compose/pipeline_composition.py` for nesting ColumnTransformer in larger pipelines
    - **Panel pipelines**: See `examples/compose/panel_pipelines.py` for comprehensive panel composition patterns
    """)

if __name__ == "__main__":
    app.run()
