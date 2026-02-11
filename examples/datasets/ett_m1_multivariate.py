"""ETT-M1 Multivariate Forecasting.

Demonstrates multivariate forecasting on the ETT-M1 electricity
transformer dataset using ForecastedFeatureForecaster and exogenous
features.
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
    # ETT-M1 Multivariate Forecasting

    The Electricity Transformer Temperature (ETT-M1) dataset contains
    the oil temperature target (`OT`) and several load/temperature
    covariates sampled at 15-minute intervals. This notebook demonstrates
    multivariate forecasting using exogenous features.

    ## What You'll Learn

    - Dataset exploration: target (OT) and covariates
    - Univariate baseline: target-only forecasting
    - Multivariate: using covariates as exogenous features (X)
    - `ForecastedFeatureForecaster`: chain target and feature forecasters
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import ForecastedFeatureForecaster
    from yohou.datasets import load_ett_m1
    from yohou.metrics import MeanAbsoluteError, RootMeanSquaredError
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        ForecastedFeatureForecaster,
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        RootMeanSquaredError,
        load_ett_m1,
        pl,
        plot_forecast,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Explore the Dataset
    """)
    return


@app.cell
def _(load_ett_m1, mo):
    ett = load_ett_m1()
    # Resample to hourly for tractable computation
    ett_hourly = ett.group_by_dynamic("time", every="1h").agg(
        pl.col("OT").mean(),
        pl.col("HUFL").mean(),
        pl.col("HULL").mean(),
        pl.col("MUFL").mean(),
        pl.col("MULL").mean(),
        pl.col("LUFL").mean(),
        pl.col("LULL").mean(),
    )
    mo.md(
        f"**ETT-M1**: {len(ett)} rows at 15-min intervals\n\n"
        f"**Resampled to hourly**: {len(ett_hourly)} rows\n\n"
        f"**Target**: OT (oil temperature)\n\n"
        f"**Covariates**: HUFL, HULL, MUFL, MULL, LUFL, LULL"
    )
    return ett, ett_hourly


@app.cell
def _(ett_hourly, plot_time_series):
    plot_time_series(ett_hourly.select("time", "OT"), title="ETT-M1: Oil Temperature (Hourly)")
    return


@app.cell
def _(ett_hourly, plot_time_series):
    plot_time_series(
        ett_hourly.select("time", "HUFL", "HULL", "MUFL"),
        title="ETT-M1: Load Covariates (Hourly)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Train/Test Split
    """)
    return


@app.cell
def _(ett_hourly, mo, pl):
    _split = int(len(ett_hourly) * 0.85)
    _covariates = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]

    y_train = ett_hourly.head(_split).select("time", "OT")
    y_test = ett_hourly.tail(len(ett_hourly) - _split).select("time", "OT")
    X_train = ett_hourly.head(_split).select("time", *_covariates)
    X_test = ett_hourly.tail(len(ett_hourly) - _split).select("time", *_covariates)
    horizon = len(y_test)

    mo.md(
        f"**Train**: {len(y_train)} hours, **Test**: {len(y_test)} hours\n\n"
        f"**y columns**: {y_train.columns}\n\n"
        f"**X columns**: {X_train.columns}"
    )
    return X_test, X_train, horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Univariate Baseline (Target Only)
    """)
    return


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, horizon, y_test, y_train):
    fc_univariate = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        target_transformer=LagTransformer(lag=[1, 24]),
    )
    fc_univariate.fit(y_train, forecasting_horizon=horizon)
    y_pred_uni = fc_univariate.predict(forecasting_horizon=horizon)
    return fc_univariate, y_pred_uni


@app.cell
def _(plot_forecast, y_pred_uni, y_test, y_train):
    plot_forecast(
        y_test, y_pred_uni, y_train=y_train, n_history=168,
        title="Univariate Baseline (Target Only)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Multivariate with Known Exogenous Features

    When test-time covariates are available (known future values), pass
    them as `X` to both `fit()` and `predict()`.
    """)
    return


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, X_test, X_train, horizon, y_test, y_train):
    fc_multi = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        target_transformer=LagTransformer(lag=[1, 24]),
    )
    fc_multi.fit(y_train, X_train, forecasting_horizon=horizon)
    y_pred_multi = fc_multi.predict(X_test, forecasting_horizon=horizon)
    return fc_multi, y_pred_multi


@app.cell
def _(plot_forecast, y_pred_multi, y_test, y_train):
    plot_forecast(
        y_test, y_pred_multi, y_train=y_train, n_history=168,
        title="Multivariate (Known Exogenous)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. ForecastedFeatureForecaster

    When covariates are NOT known at prediction time, forecast them
    separately and use those forecasts as inputs.
    """)
    return


@app.cell
def _(
    ForecastedFeatureForecaster,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    X_train,
    horizon,
    y_test,
    y_train,
):
    fc_ff = ForecastedFeatureForecaster(
        target_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            target_transformer=LagTransformer(lag=[1, 24]),
        ),
        feature_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            target_transformer=LagTransformer(lag=[1, 24]),
        ),
        strategy="predicted",
    )
    fc_ff.fit(y_train, X_train, forecasting_horizon=horizon)
    y_pred_ff = fc_ff.predict(forecasting_horizon=horizon)
    return fc_ff, y_pred_ff


@app.cell
def _(plot_forecast, y_pred_ff, y_test, y_train):
    plot_forecast(
        y_test, y_pred_ff, y_train=y_train, n_history=168,
        title="ForecastedFeatureForecaster (strategy='predicted')",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Compare Approaches
    """)
    return


@app.cell
def _(MeanAbsoluteError, RootMeanSquaredError, mo, pl, y_pred_ff, y_pred_multi, y_pred_uni, y_test):
    _mae = MeanAbsoluteError()
    _rmse = RootMeanSquaredError()
    _preds = {
        "Univariate": y_pred_uni,
        "Multivariate (known X)": y_pred_multi,
        "ForecastedFeature": y_pred_ff,
    }
    _rows = []
    for _name, _pred in _preds.items():
        _m = float(_mae.score(y_test, _pred).drop("time").to_series().mean())
        _r = float(_rmse.score(y_test, _pred).drop("time").to_series().mean())
        _rows.append({"Method": _name, "MAE": round(_m, 3), "RMSE": round(_r, 3)})

    mo.ui.table(pl.DataFrame(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **ETT-M1**: Oil temperature with 6 load/temperature covariates
    - **Known exogenous** (X at test time): Best accuracy when future values are available
    - **ForecastedFeatureForecaster**: Handles unknown-at-test-time covariates by chaining forecasters
    - **`strategy="predicted"`**: Uses forecasted features during prediction
    - **Hourly resampling**: Reduces computation while preserving temporal patterns

    ## Next Steps

    - **Forecasted feature strategies**: See `examples/compose/forecasted_feature_advanced.py`
    - **Feature union**: See `examples/compose/feature_union.py`
    - **Pipeline composition**: See `examples/compose/pipeline_composition.py`
    """)
    return


if __name__ == "__main__":
    app.run()
