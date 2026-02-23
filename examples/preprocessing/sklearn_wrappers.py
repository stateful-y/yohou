"""Sklearn-Compatible Scalers and Transformers.

Demonstrates yohou's sklearn wrapper classes (StandardScaler, MinMaxScaler,
RobustScaler, PowerTransformer, etc.) for time series preprocessing, including
inverse transforms and panel data integration.
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
    # Sklearn-Compatible Scalers and Transformers

    Yohou wraps popular sklearn preprocessing classes so they work natively
    with polars DataFrames and the `"time"` column convention. Each wrapper
    strips `"time"` before calling the underlying sklearn logic and re-attaches
    it afterwards, preserving full invertibility.

    ## What You'll Learn

    - Using `StandardScaler`, `MinMaxScaler`, `RobustScaler` for scaling
    - `PowerTransformer` (Box-Cox, Yeo-Johnson) for variance stabilisation
    - `PolynomialFeatures` for interaction terms
    - Inverse transforms for back-transformation
    - Integrating scalers as `target_transformer` and `feature_transformer` in forecasters
    - Panel data: automatic per-group scaling when used inside a panel forecaster

    ## Prerequisites

    Basic familiarity with sklearn preprocessing and `PointReductionForecaster`
    (see `examples/quickstart.py`).
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_dominick, fetch_tourism_monthly
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import (
        LagTransformer,
        MinMaxScaler,
        PolynomialFeatures,
        PowerTransformer,
        RobustScaler,
        StandardScaler,
    )

    return (
        LagTransformer,
        MeanAbsoluteError,
        MinMaxScaler,
        PointReductionForecaster,
        PolynomialFeatures,
        PowerTransformer,
        Ridge,
        RobustScaler,
        StandardScaler,
        fetch_dominick,
        fetch_tourism_monthly,
        pl,
        plot_forecast,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data
    """)
    return


@app.cell
def _(fetch_tourism_monthly):
    df = fetch_tourism_monthly().frame.select("time", "T1__tourists").rename({"T1__tourists": "Passengers"})
    _split = int(len(df) * 0.85)
    y_train = df.head(_split).select("time", "Passengers")
    y_test = df.tail(len(df) - _split).select("time", "Passengers")
    y_train.head()
    return df, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. StandardScaler

    Centres each column to zero mean and unit variance. This is the most
    common scaler for regression models.
    """)
    return


@app.cell
def _(StandardScaler, mo, y_train):
    std_scaler = StandardScaler()
    y_scaled = std_scaler.fit_transform(y_train)
    _stats = y_scaled.select("Passengers").describe()
    mo.md(
        f"**After StandardScaler**: "
        f"mean = {y_scaled['Passengers'].mean():.4f}, "
        f"std = {y_scaled['Passengers'].std():.4f}"
    )
    return std_scaler, y_scaled


@app.cell
def _(mo, std_scaler, y_scaled, y_train):
    _y_inv = std_scaler.inverse_transform(y_scaled)
    _max_err = (
        _y_inv.select("Passengers").to_series()
        - y_train.select("Passengers").to_series()
    ).abs().max()
    mo.md(f"**Round-trip reconstruction error**: {_max_err:.2e}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. MinMaxScaler

    Scales each column to the range `[0, 1]` (or a custom `feature_range`).
    Useful when the model is sensitive to feature magnitudes.
    """)
    return


@app.cell
def _(MinMaxScaler, mo, y_train):
    mm_scaler = MinMaxScaler()
    _y_mm = mm_scaler.fit_transform(y_train)
    mo.md(
        f"**After MinMaxScaler**: "
        f"min = {_y_mm['Passengers'].min():.4f}, "
        f"max = {_y_mm['Passengers'].max():.4f}"
    )
    return (mm_scaler,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. RobustScaler

    Uses median and IQR instead of mean and std, making it robust to outliers.
    """)
    return


@app.cell
def _(RobustScaler, mo, y_train):
    rob_scaler = RobustScaler()
    _y_rob = rob_scaler.fit_transform(y_train)
    mo.md(
        f"**After RobustScaler**: "
        f"median = {_y_rob['Passengers'].median():.4f}, "
        f"IQR-scaled std = {_y_rob['Passengers'].std():.4f}"
    )
    return (rob_scaler,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Compare Scalers Visually
    """)
    return


@app.cell
def _(MinMaxScaler, RobustScaler, StandardScaler, pl, plot_time_series, y_train):
    _std = StandardScaler().fit_transform(y_train).rename({"Passengers": "StandardScaler"})
    _mm = MinMaxScaler().fit_transform(y_train).rename({"Passengers": "MinMaxScaler"})
    _rob = RobustScaler().fit_transform(y_train).rename({"Passengers": "RobustScaler"})
    _combined = pl.concat(
        [
            _std.select("time", "StandardScaler"),
            _mm.select("MinMaxScaler"),
            _rob.select("RobustScaler"),
        ],
        how="horizontal",
    )
    plot_time_series(_combined, title="Scaler Comparison on Monthly Tourism")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. PowerTransformer

    `PowerTransformer` applies Box-Cox or Yeo-Johnson transformations to make
    data more Gaussian-like.  Box-Cox requires strictly positive values;
    Yeo-Johnson works with any data.
    """)
    return


@app.cell
def _(PowerTransformer, mo, y_train):
    pw_transformer = PowerTransformer(method="box-cox")
    _y_pw = pw_transformer.fit_transform(y_train)
    mo.md(
        f"**After PowerTransformer (Box-Cox)**: "
        f"skewness = {_y_pw['Passengers'].skew():.4f}"
    )
    return (pw_transformer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. PolynomialFeatures

    Creates interaction and polynomial terms. Useful as a `feature_transformer`
    to enrich the feature space before a linear regression.
    """)
    return


@app.cell
def _(PolynomialFeatures, mo, y_train):
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    _y_poly = poly.fit_transform(y_train)
    _names = poly.get_feature_names_out()
    mo.md(
        f"**PolynomialFeatures (degree=2)**: {len(_names)} output columns\n\n"
        f"Feature names: {_names}"
    )
    return (poly,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Inside a Forecaster

    Pass a scaler as `target_transformer` to automatically scale before
    fitting and inverse-scale predictions. Here we compare raw predictions
    with StandardScaler-transformed predictions.
    """)
    return


@app.cell
def _(
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    StandardScaler,
    mo,
    y_test,
    y_train,
):
    _horizon = len(y_test)

    # Baseline: no target scaling
    _fc_raw = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 6, 12]),
    )
    _fc_raw.fit(y_train, forecasting_horizon=_horizon)
    _y_pred_raw = _fc_raw.predict(forecasting_horizon=_horizon)

    # With StandardScaler on target
    fc_scaled = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        target_transformer=StandardScaler(),
        feature_transformer=LagTransformer(lag=[1, 6, 12]),
    )
    fc_scaled.fit(y_train, forecasting_horizon=_horizon)
    y_pred_scaled = fc_scaled.predict(forecasting_horizon=_horizon)

    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _mae_raw = float(_scorer.score(y_test, _y_pred_raw))
    _mae_scaled = float(_scorer.score(y_test, y_pred_scaled))
    mo.md(
        f"**MAE without scaler**: {_mae_raw:.2f}\n\n"
        f"**MAE with StandardScaler**: {_mae_scaled:.2f}"
    )
    return fc_scaled, y_pred_scaled


@app.cell
def _(plot_forecast, y_pred_scaled, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_scaled,
        y_train=y_train,
        n_history=36,
        title="Forecast with StandardScaler Target Transform",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. With Panel Data

    When a scaler is used as `target_transformer` inside a panel-aware
    forecaster, yohou fits a **separate scaler per panel group**. Each
    group's data is scaled independently.
    """)
    return


@app.cell
def _(
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    StandardScaler,
    fetch_dominick,
    plot_forecast,
):
    _panel = fetch_dominick().frame.select(
        "time", "T1__profit", "T2__profit", "T3__profit",
        "T4__profit", "T5__profit", "T6__profit",
        "T7__profit", "T8__profit", "T9__profit",
    )
    _split = int(len(_panel) * 0.9)
    _profit_cols = [c for c in _panel.columns if c.endswith("__profit")]
    _y_train_p = _panel.head(_split).select("time", *_profit_cols)
    _y_test_p = _panel.tail(len(_panel) - _split).select("time", *_profit_cols)

    _fc_panel = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        target_transformer=StandardScaler(),
        feature_transformer=LagTransformer(lag=[1, 4]),
    )
    _horizon_p = min(len(_y_test_p), 8)
    _fc_panel.fit(_y_train_p, forecasting_horizon=_horizon_p)
    _y_pred_p = _fc_panel.predict(forecasting_horizon=_horizon_p)

    plot_forecast(
        _y_test_p,
        _y_pred_p,
        y_train=_y_train_p,
        n_history=20,
        panel_group_names=["T1", "T2"],
        title="Panel Forecast with Per-Group StandardScaler",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Sklearn wrappers** (StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, etc.) work natively with polars DataFrames
    - The `"time"` column is automatically stripped before sklearn logic and re-attached afterwards
    - All scalers support **inverse_transform** for exact back-transformation
    - **PowerTransformer** (Box-Cox / Yeo-Johnson) stabilises variance and reduces skewness
    - **PolynomialFeatures** enriches the feature space with interactions and higher-order terms
    - When used as `target_transformer` in a panel forecaster, a **separate scaler is fitted per group**
    - Combine scalers with `ColumnTransformer` when different columns need different scaling

    ## Next Steps

    - **Column-wise transforms**: See `examples/compose/column_transformer.py` for applying different scalers to different columns
    - **Custom transforms**: See `examples/preprocessing/function_transformer.py` for wrapping arbitrary polars operations
    - **Stationarity transforms**: See `examples/stationarity/` for decomposition-based transforms (LogTransformer, BoxCoxTransformer, etc.)
    """)
    return


if __name__ == "__main__":
    app.run()
