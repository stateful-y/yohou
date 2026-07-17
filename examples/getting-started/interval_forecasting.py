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
    "title": "Interval Forecasting",
    "description": "Wrap a point forecaster with SplitConformalForecaster to produce 95% prediction intervals with statistical coverage guarantees.",
    "category": "tutorial",
    "companion": "/pages/tutorials/interval-forecasting/",
    "section": "getting-started",
    "api_references": ["EmpiricalCoverage", "FeaturePipeline", "PointReductionForecaster", "SeasonalDifferencing", "SplitConformalForecaster"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Interval Forecasting

        In this notebook, we will wrap a point forecaster with
        [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/)
        to produce prediction intervals with coverage guarantees for continuous
        targets. The split conformal method reserves a calibration set to estimate
        residual quantiles, then constructs intervals guaranteed to contain future
        observations at the target rate as calibration data grows.

        **Prerequisites:** Completed [Getting Started](/pages/tutorials/getting-started/).
        """
    )


@app.cell(hide_code=True)
def _():
    from sklearn.linear_model import Ridge

    from yohou.compose import FeaturePipeline
    from yohou.datasets import fetch_tourism_monthly
    from yohou.interval import SplitConformalForecaster
    from yohou.metrics import EmpiricalCoverage
    from yohou.plotting import plot_forecast
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer
    from yohou.stationarity import SeasonalDifferencing

    return (
        EmpiricalCoverage,
        FeaturePipeline,
        LagTransformer,
        PointReductionForecaster,
        Ridge,
        SeasonalDifferencing,
        SplitConformalForecaster,
        fetch_tourism_monthly,
        plot_forecast,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 1. Load and Setup")


@app.cell
def _(fetch_tourism_monthly):
    from yohou.model_selection import train_test_split

    bunch = fetch_tourism_monthly()
    y = (
        bunch.frame
        .select("time", "T1__tourists")
        .rename({"T1__tourists": "tourists"})
        .drop_nulls()
    )
    forecasting_horizon = 12
    y_train, y_test = train_test_split(y, test_size=forecasting_horizon)
    print(f"Series: {len(y)} months  |  Train: {len(y_train)}  |  Test: {len(y_test)}")
    y.tail()
    return bunch, forecasting_horizon, y, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 2. Configure Calibration Size")


@app.cell(hide_code=True)
def _(mo):
    calibration_slider = mo.ui.slider(
        start=12,
        stop=60,
        step=6,
        value=24,
        label="calibration_size (months held out from training for calibration)",
    )
    calibration_slider


@app.cell
def _(
    FeaturePipeline,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SeasonalDifferencing,
    SplitConformalForecaster,
    calibration_slider,
    forecasting_horizon,
    y_train,
):
    calibration_size = calibration_slider.value
    point_forecaster = PointReductionForecaster(
        estimator=Ridge(),
        target_transformer=SeasonalDifferencing(seasonality=12),
        actual_transformer=FeaturePipeline([
            ("lags", LagTransformer(lag=[1, 2, 3, 12])),
        ]),
    )
    conformal = SplitConformalForecaster(
        point_forecaster=point_forecaster,
        calibration_size=calibration_size,
    )
    conformal.fit(y_train, forecasting_horizon=forecasting_horizon)
    print(f"Fitted conformal forecaster (calibration_size={calibration_size})")
    return calibration_size, conformal, point_forecaster


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 3. Predict Intervals")


@app.cell
def _(conformal, forecasting_horizon):
    y_pred_int = conformal.predict_interval(forecasting_horizon=forecasting_horizon)
    print("Interval columns:", y_pred_int.columns)
    y_pred_int
    return (y_pred_int,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 4. Visualize and Check Coverage")


@app.cell
def _(EmpiricalCoverage, plot_forecast, y_pred_int, y_test, y_train):
    fig = plot_forecast(y_test, y_pred_int, y_train=y_train[-24:])
    cov_scorer = EmpiricalCoverage()
    cov_scorer.fit(y_train)
    empirical_cov = cov_scorer.score(y_test, y_pred_int)
    print(f"Target coverage:    0.95")
    print(f"Empirical coverage: {empirical_cov:.3f}")
    fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [Interval Forecasting](/pages/tutorials/interval-forecasting/) for the full guide
        - [Interval Forecasting How-To](/pages/how-to/interval-forecasting/) for related techniques
        """
    )
    return


if __name__ == "__main__":
    app.run()
