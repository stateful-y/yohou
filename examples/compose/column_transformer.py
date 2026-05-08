# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Column Transformer",
    "description": "Route columns through distinct transformers with ColumnTransformer, including remainder handling and automatic panel-aware column detection.",
}
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
    [`ColumnTransformer`](/pages/api/generated/yohou.compose.column_transformer.ColumnTransformer/) applies **distinct transformers to distinct columns**
    in a single step.

    ## What You'll Learn

    - Building a [`ColumnTransformer`](/pages/api/generated/yohou.compose.column_transformer.ColumnTransformer/) with named transformer-column pairs
    - Remainder handling: `"drop"` vs `"passthrough"`
    - Inspecting output feature names
    - Combining [`ColumnTransformer`](/pages/api/generated/yohou.compose.column_transformer.ColumnTransformer/) with [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/)
    - Using [`ColumnTransformer`](/pages/api/generated/yohou.compose.column_transformer.ColumnTransformer/) inside a panel forecaster for automatic per-group application

    ## Prerequisites

    Familiarity with [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) and basic transformers
    (see `examples/point/reduction_forecaster.py`).
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import ColumnTransformer
    from yohou.datasets import fetch_dominick, fetch_electricity_demand
    from yohou.model_selection import train_test_split
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer, StandardScaler
    from yohou.utils.panel import inspect_panel

    return (
        ColumnTransformer,
        LagTransformer,
        PointReductionForecaster,
        Ridge,
        StandardScaler,
        fetch_dominick,
        fetch_electricity_demand,
        inspect_panel,
        pl,
        plot_forecast,
        plot_time_series,
        train_test_split,
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
def _(fetch_electricity_demand, pl, train_test_split):
    _elec = fetch_electricity_demand().frame
    # Downsample to daily for manageable size (drop trailing all-null days)
    vic_daily = (
        _elec
        .group_by_dynamic("time", every="1d")
        .agg(
            pl.col("vic__demand").mean().alias("Demand"),
            pl.col("nsw__demand").mean().alias("NSW_Demand"),
            pl.col("sa__demand").mean().alias("SA_Demand"),
        )
        .drop_nulls()
    )

    _train_df, _test_df = train_test_split(vic_daily, test_size=0.15)
    y_train = _train_df.select("time", "Demand")
    X_actual_train = _train_df.select("time", "NSW_Demand", "SA_Demand")
    y_test = _test_df.select("time", "Demand")
    X_actual_test = _test_df.select("time", "NSW_Demand", "SA_Demand")

    return X_actual_test, X_actual_train, vic_daily, y_test, y_train


@app.cell
def _(plot_time_series, vic_daily):
    plot_time_series(vic_daily, title="Daily Victorian Electricity Demand")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Build a ColumnTransformer

    Each tuple is `(name, transformer, columns)`. We scale NSW_Demand with
    [`StandardScaler`](/pages/api/generated/yohou.preprocessing.sklearn_wrappers.StandardScaler/), pass SA_Demand through unchanged, and drop any remaining
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


@app.cell
def _(X_actual_train, ct, pl, plot_time_series):
    _ct_fitted = ct.fit(X_actual_train)
    _X_transformed = _ct_fitted.transform(X_actual_train)
    _combined = pl.concat(
        [
            X_actual_train.rename({"NSW_Demand": "NSW (raw)", "SA_Demand": "SA (raw)"}),
            _X_transformed.drop("time").rename({
                c: c.split("__", 1)[-1] + " (scaled)" for c in _X_transformed.columns if c != "time"
            }),
        ],
        how="horizontal",
    )
    plot_time_series(_combined, title="Raw vs Scaled Features")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. ColumnTransformer Inside a Forecaster

    Pass the [`ColumnTransformer`](/pages/api/generated/yohou.compose.column_transformer.ColumnTransformer/) as `feature_transformer` to
    [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/). The forecaster calls `.fit_transform()` on `X_actual`
    at fit time and `.transform()` on `X_actual` at predict time.
    """)


@app.cell
def _(PointReductionForecaster, Ridge, X_actual_train, ct, y_test, y_train):
    forecaster_ct = PointReductionForecaster(
        estimator=Ridge(alpha=1e-3),
        feature_transformer=ct,
    )

    forecasting_horizon = len(y_test)
    forecaster_ct.fit(y_train, X_actual_train, forecasting_horizon=forecasting_horizon)
    y_pred_ct = forecaster_ct.predict(forecasting_horizon=forecasting_horizon)
    return (y_pred_ct,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) visualises how the [`ColumnTransformer`](/pages/api/generated/yohou.compose.column_transformer.ColumnTransformer/)-equipped
    forecaster performs on the held-out test data.
    """)


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
        f"**With `remainder='passthrough'`**: SA_Demand is kept automatically.\n\nTransformer configuration: {ct_pass}"
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Inspecting Feature Names

    After fitting, `get_feature_names_out()` reveals the output column names,
    which include the transformer name prefix when `verbose_feature_names_out=True`
    (the default).
    """)


@app.cell
def _(ColumnTransformer, StandardScaler, X_actual_train, mo):
    ct_verbose = ColumnTransformer(
        transformers=[
            ("scale_nsw", StandardScaler(), ["NSW_Demand"]),
            ("sa", "passthrough", ["SA_Demand"]),
        ],
        verbose_feature_names_out=True,
    )
    ct_verbose.fit(X_actual_train)
    _names = ct_verbose.get_feature_names_out()
    mo.md(f"**Feature names (verbose=True)**: {_names}")


@app.cell
def _(ColumnTransformer, StandardScaler, X_actual_train, mo):
    _ct_short = ColumnTransformer(
        transformers=[
            ("scale_nsw", StandardScaler(), ["NSW_Demand"]),
            ("sa", "passthrough", ["SA_Demand"]),
        ],
        verbose_feature_names_out=False,
    )
    _ct_short.fit(X_actual_train)
    _names = _ct_short.get_feature_names_out()
    mo.md(f"**Feature names (verbose=False)**: {_names}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. With Panel Data

    When a [`ColumnTransformer`](/pages/api/generated/yohou.compose.column_transformer.ColumnTransformer/) is used as the `feature_transformer` inside a
    panel-aware forecaster, _yohou automatically applies it per group_.
    The forecaster decomposes the panel into individual groups, applies the
    transformer to each group's unprefixed columns, and reassembles the result.

    Here we use the Dominick dataset (9 panel groups).
    """)


@app.cell
def _(fetch_dominick, inspect_panel, mo, train_test_split):
    _dom_full = fetch_dominick().frame
    _profit_cols = [
        "T7__profit",
        "T11__profit",
        "T12__profit",
        "T13__profit",
        "T15__profit",
        "T19__profit",
        "T22__profit",
        "T23__profit",
        "T24__profit",
    ]
    store = _dom_full.select("time", *_profit_cols)
    _globals, _groups = inspect_panel(store)
    mo.md(f"**Panel groups**: {len(_groups)} groups\n\n**First group columns**: {list(_groups.values())[0]}")

    # Panel data: y contains all `__profit` columns, no separate X_actual needed
    y_train_panel, y_test_panel = train_test_split(store, test_size=0.1)

    y_train_panel.head()
    return y_test_panel, y_train_panel


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
        groups=["T7", "T11", "T12"],
        title="Panel Forecast: First 3 Groups",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **ColumnTransformer** applies distinct transformers to distinct columns in a single step
    - **Remainder handling** controls what happens to unlisted columns: `"drop"` (default) discards them, `"passthrough"` keeps them
    - **verbose_feature_names_out** prefixes output names with transformer names for traceability
    - **Panel integration** is automatic: when used inside a forecaster, ColumnTransformer is applied per group to unprefixed columns
    - **Forecaster composition**: Pass ColumnTransformer as `feature_transformer` to [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/)

    ## Next Steps

    - **Parallel features**: See [`examples/compose/feature_union.py`](/examples/compose/feature_union/) for combining transformers in parallel
    - **Pipeline composition**: See [`examples/compose/pipeline_composition.py`](/examples/compose/pipeline_composition/) for nesting ColumnTransformer in larger pipelines
    - **Panel pipelines**: See [`examples/compose/panel_pipelines.py`](/examples/compose/panel_pipelines/) for comprehensive panel composition patterns
    """)


if __name__ == "__main__":
    app.run()
