# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Hospital Multivariate Forecasting.

Demonstrates multivariate forecasting on the Hospital patient-count
dataset using ForecastedFeatureForecaster and exogenous features.
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
    # Hospital Multivariate Forecasting

    The Hospital dataset contains 767 monthly patient count series
    related to medical products (2000-2006). This notebook picks one
    series as the target and uses other series as exogenous covariates,
    demonstrating multivariate forecasting.

    ## What You'll Learn

    - Dataset exploration: target and covariate series
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
    from yohou.datasets import fetch_hospital
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
        fetch_hospital,
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
def _(fetch_hospital, mo):
    _all = fetch_hospital().frame
    # Use T1 as target, T2-T4 as covariates (rename to non-panel columns)
    hospital = _all.select(
        "time",
        pl.col("T1__patients").alias("patients"),
        pl.col("T2__patients").alias("cov_1"),
        pl.col("T3__patients").alias("cov_2"),
        pl.col("T4__patients").alias("cov_3"),
    )
    mo.md(
        f"**Hospital**: {len(_all)} rows, {len(_all.columns) - 1} series\n\n"
        f"**Selected**: {len(hospital)} months\n\n"
        f"**Target**: patients (from T1)\n\n"
        f"**Covariates**: cov_1 (T2), cov_2 (T3), cov_3 (T4)"
    )
    return (hospital,)

@app.cell
def _(hospital, plot_time_series):
    plot_time_series(hospital.select("time", "patients"), title="Hospital: Target Patient Counts (T1)")
    return

@app.cell
def _(hospital, plot_time_series):
    plot_time_series(
        hospital.select("time", "cov_1", "cov_2", "cov_3"),
        title="Hospital: Covariate Series (T2-T4)",
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Train/Test Split
    """)
    return

@app.cell
def _(hospital, mo, pl):
    _split = int(len(hospital) * 0.85)
    _covariates = ["cov_1", "cov_2", "cov_3"]

    y_train = hospital.head(_split).select("time", "patients")
    y_test = hospital.tail(len(hospital) - _split).select("time", "patients")
    X_train = hospital.head(_split).select("time", *_covariates)
    X_test = hospital.tail(len(hospital) - _split).select("time", *_covariates)
    horizon = len(y_test)

    mo.md(
        f"**Train**: {len(y_train)} months, **Test**: {len(y_test)} months\n\n"
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
        feature_transformer=LagTransformer(lag=[1, 12]),
    )
    fc_univariate.fit(y_train, forecasting_horizon=horizon)
    y_pred_uni = fc_univariate.predict(forecasting_horizon=horizon)
    return fc_univariate, y_pred_uni

@app.cell
def _(plot_forecast, y_pred_uni, y_test, y_train):
    plot_forecast(
        y_test, y_pred_uni, y_train=y_train, n_history=24,
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
        feature_transformer=LagTransformer(lag=[1, 12]),
    )
    fc_multi.fit(y_train, X_train, forecasting_horizon=horizon)
    y_pred_multi = fc_multi.predict(X_test, forecasting_horizon=horizon)
    return fc_multi, y_pred_multi

@app.cell
def _(plot_forecast, y_pred_multi, y_test, y_train):
    plot_forecast(
        y_test, y_pred_multi, y_train=y_train, n_history=24,
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
            feature_transformer=LagTransformer(lag=[1, 12]),
        ),
        feature_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 12]),
        ),
        strategy="predicted",
        split_ratio=0.8,
    )
    fc_ff.fit(y_train, X_train, forecasting_horizon=6)
    y_pred_ff = fc_ff.predict(forecasting_horizon=horizon)
    return fc_ff, y_pred_ff

@app.cell
def _(plot_forecast, y_pred_ff, y_test, y_train):
    plot_forecast(
        y_test, y_pred_ff, y_train=y_train, n_history=24,
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
def _(MeanAbsoluteError, RootMeanSquaredError, mo, pl, y_pred_ff, y_pred_multi, y_pred_uni, y_test, y_train):
    _mae = MeanAbsoluteError()
    _rmse = RootMeanSquaredError()
    _mae.fit(y_train)
    _rmse.fit(y_train)
    _preds = {
        "Univariate": y_pred_uni,
        "Multivariate (known X)": y_pred_multi,
        "ForecastedFeature": y_pred_ff,
    }
    _rows = []
    for _name, _pred in _preds.items():
        _m = float(_mae.score(y_test, _pred))
        _r = float(_rmse.score(y_test, _pred))
        _rows.append({"Method": _name, "MAE": round(_m, 3), "RMSE": round(_r, 3)})

    mo.ui.table(pl.DataFrame(_rows))
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Hospital dataset**: 767 monthly patient count series; here T1 is target, T2-T4 are covariates
    - **Known exogenous** (X at test time): Best accuracy when future values are available
    - **ForecastedFeatureForecaster**: Handles unknown-at-test-time covariates by chaining forecasters
    - **`strategy="predicted"`**: Uses forecasted features during prediction
    - **Monthly frequency**: Seasonal lags of 12 capture annual patterns

    ## Next Steps

    - **Forecasted feature strategies**: See `examples/compose/forecasted_feature_advanced.py`
    - **Feature union**: See `examples/compose/feature_union.py`
    - **Pipeline composition**: See `examples/compose/pipeline_composition.py`
    """)
    return

if __name__ == "__main__":
    app.run()
