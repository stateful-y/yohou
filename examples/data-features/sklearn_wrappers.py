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
    "title": "How to Use Scikit-learn Scalers",
    "description": "Wrap sklearn scalers (StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, PolynomialFeatures) for polars DataFrames with inverse transforms.",
    "category": "how-to",
    "section": "data-features",
    "companion": "/pages/how-to/use-preprocessing-transformers/",
    "api_references": ["MinMaxScaler", "PointReductionForecaster", "PolynomialFeatures", "PowerTransformer", "RobustScaler", "StandardScaler", "plot_score_per_vintage", "plot_score_time_series"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sklearn-Compatible Scalers and Transformers

    Yohou wraps popular sklearn preprocessing classes so they work natively
    with polars DataFrames and the `"time"` column convention. Each wrapper
    strips `"time"` before calling the underlying sklearn logic and re-attaches
    it afterwards, preserving full invertibility.

    **Prerequisites:** Basic familiarity with sklearn preprocessing and [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/)
    (see `examples/quickstart.py`).
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_dominick, fetch_tourism_monthly
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import train_test_split
    from yohou.plotting import (
        plot_forecast,
        plot_score_per_vintage,
        plot_score_time_series,
        plot_time_series,
    )
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
        deepcopy,
        fetch_dominick,
        fetch_tourism_monthly,
        pl,
        plot_forecast,
        plot_score_per_vintage,
        plot_score_time_series,
        plot_time_series,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We load a single monthly tourism series and split it into training and
    test sets with [`train_test_split`](/pages/api/generated/yohou.model_selection.split.train_test_split/) to preserve temporal
    ordering.
    """)


@app.cell
def _(fetch_tourism_monthly, plot_time_series, train_test_split):
    df = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    y_train, y_test = train_test_split(df, test_size=0.15)
    plot_time_series(y_train, title="Training Data")
    return y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. StandardScaler

    Centres each column to zero mean and unit variance. This is the most
    common scaler for regression models.
    """)


@app.cell
def _(StandardScaler, plot_time_series, y_train):
    std_scaler = StandardScaler()
    y_scaled = std_scaler.fit_transform(y_train)
    plot_time_series(y_scaled, title="StandardScaler")
    return std_scaler, y_scaled


@app.cell
def _(plot_time_series, std_scaler, y_scaled, y_train):
    _y_inv = std_scaler.inverse_transform(y_scaled)
    _combined = y_train.rename({"tourists": "original"}).join(
        _y_inv.rename({"tourists": "Inverted scaled"}),
        on="time",
    )
    plot_time_series(_combined, title="RobustScaler: Original vs Scaled")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. MinMaxScaler

    Scales each column to the range `[0, 1]` (or a custom `feature_range`).
    Useful when the model is sensitive to feature magnitudes.
    """)


@app.cell
def _(MinMaxScaler, plot_time_series, y_train):
    mm_scaler = MinMaxScaler()
    _y_mm = mm_scaler.fit_transform(y_train)
    plot_time_series(_y_mm, title="MinMaxScaler")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. RobustScaler

    Uses median and IQR instead of mean and std, making it robust to outliers.
    """)


@app.cell
def _(RobustScaler, plot_time_series, y_train):
    rob_scaler = RobustScaler()
    _y_rob = rob_scaler.fit_transform(y_train)
    plot_time_series(_y_rob, title="RobustScaler")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Compare Scalers Visually

    [`plot_time_series`](/pages/api/generated/yohou.plotting.exploration.plot_time_series/) can overlay multiple columns in a single chart. Here
    we horizontally concatenate the outputs of three scalers so we can
    compare their shapes and ranges side-by-side.
    """)


@app.cell
def _(
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
    pl,
    plot_time_series,
    y_train,
):
    _std = StandardScaler().fit_transform(y_train).rename({"tourists": "StandardScaler"})
    _mm = MinMaxScaler().fit_transform(y_train).rename({"tourists": "MinMaxScaler"})
    _rob = RobustScaler().fit_transform(y_train).rename({"tourists": "RobustScaler"})
    _combined = pl.concat(
        [
            _std.select("time", "StandardScaler"),
            _mm.select("MinMaxScaler"),
            _rob.select("RobustScaler"),
        ],
        how="horizontal",
    )
    plot_time_series(_combined, title="Scaler Comparison on Monthly Tourism")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. PowerTransformer

    [`PowerTransformer`](/pages/api/generated/yohou.preprocessing.sklearn_wrappers.PowerTransformer/) applies Box-Cox or Yeo-Johnson transformations to make
    data more Gaussian-like.  Box-Cox requires strictly positive values;
    Yeo-Johnson works with any data.
    """)


@app.cell
def _(PowerTransformer, plot_time_series, y_train):
    pw_transformer = PowerTransformer(method="box-cox")
    _y_pw = pw_transformer.fit_transform(y_train)
    _combined = y_train.rename({"tourists": "original"}).join(
        _y_pw.rename({"tourists": "Box-Cox"}),
        on="time",
    )
    plot_time_series(_combined, title="PowerTransformer (Box-Cox): Original vs Transformed")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. PolynomialFeatures

    Creates interaction and polynomial terms. Useful as a `feature_transformer`
    to enrich the feature space before a linear regression.
    """)


@app.cell
def _(PolynomialFeatures, plot_time_series, y_train):
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    _y_poly = poly.fit_transform(y_train)
    plot_time_series(_y_poly, title="PolynomialFeatures (degree=2)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Inside a Forecaster

    Pass a scaler as `target_transformer` to automatically scale before
    fitting and inverse-scale predictions. Here we compare raw predictions
    with StandardScaler-transformed predictions.
    """)


@app.cell
def _(
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    StandardScaler,
    plot_score_time_series,
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
    plot_score_time_series(
        _scorer,
        y_test,
        {"No Scaler": _y_pred_raw, "StandardScaler": y_pred_scaled},
        title="MAE Over Time: No Scaler vs StandardScaler",
    )
    return (y_pred_scaled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) overlays the predicted values against the test actuals.
    The `n_history=36` parameter shows the last 36 training observations for
    visual context.
    """)


@app.cell
def _(plot_forecast, y_pred_scaled, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_scaled,
        y_train=y_train,
        n_history=36,
        title="Forecast with StandardScaler Target Transform",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. With Panel Data

    When a scaler is used as `target_transformer` inside a panel-aware
    forecaster, yohou fits a **separate scaler per panel group**. Each
    group's data is scaled independently.
    """)


@app.cell
def _(
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    StandardScaler,
    fetch_dominick,
    plot_forecast,
    train_test_split,
):
    _panel = fetch_dominick().frame.select(
        "time",
        "T7__profit",
        "T11__profit",
        "T12__profit",
        "T13__profit",
        "T15__profit",
        "T19__profit",
        "T22__profit",
        "T23__profit",
        "T24__profit",
    )
    _profit_cols = [c for c in _panel.columns if c.endswith("__profit")]
    _selected = _panel.select("time", *_profit_cols)
    _y_train_p, _y_test_p = train_test_split(_selected, test_size=0.1)

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
        groups=["T7", "T11"],
        title="Panel Forecast with Per-Group StandardScaler",
    )


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
def _(deepcopy, fc_scaled, y_test):
    _vintage_model = deepcopy(fc_scaled)
    y_pred_vintages = _vintage_model.observe_predict(
        y=y_test,
        stride=1,
        forecasting_horizon=len(y_test),
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Column-wise transforms**: See [`examples/data-features/column_transformer.py`](/examples/column_transformer/) for applying different scalers to different columns
    - **Custom transforms**: See [`examples/data-features/function_transformer.py`](/examples/function_transformer/) for wrapping arbitrary polars operations
    - **Stationarity transforms**: See `examples/stationarity/` for decomposition-based transforms (LogTransformer, BoxCoxTransformer, etc.)
    """)


if __name__ == "__main__":
    app.run()
