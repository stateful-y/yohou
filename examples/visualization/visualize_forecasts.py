# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.23.1"
__gallery__ = {
    "title": "How to Visualize Forecasts",
    "description": "Plot point forecasts, compare multiple models, render prediction interval bands, inspect residual diagnostics, and check interval calibration.",
    "category": "how-to",
    "companion": "/pages/how-to/visualize-forecasts/",
    "section": "visualization",
    "api_references": ["PointReductionForecaster", "SplitConformalForecaster", "plot_calibration", "plot_forecast", "plot_residuals"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_electricity_demand
    from yohou.interval import SplitConformalForecaster
    from yohou.model_selection import train_test_split
    from yohou.plotting import plot_calibration, plot_forecast, plot_residuals
    from yohou.point import PointReductionForecaster

    return (
        PointReductionForecaster,
        RandomForestRegressor,
        Ridge,
        SplitConformalForecaster,
        fetch_electricity_demand,
        plot_calibration,
        plot_forecast,
        plot_residuals,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How to Visualize Forecasts

    This notebook demonstrates how to plot forecast output against actuals
    using the `yohou.plotting` module. We cover point forecasts, multi-model
    comparison, prediction interval bands, residual diagnostics, and
    calibration checks.
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Data and Fit a Baseline
    """)


@app.cell
def _(PointReductionForecaster, Ridge, fetch_electricity_demand, train_test_split):
    data = fetch_electricity_demand()
    y = data.frame.select("time", "qun__demand").drop_nulls().rename({"qun__demand": "demand"})

    y_train, y_test = train_test_split(y, test_size=24)

    forecaster = PointReductionForecaster(estimator=Ridge())
    forecaster.fit(y_train, forecasting_horizon=24)
    y_pred = forecaster.predict()
    return data, forecaster, y, y_pred, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Plot a Point Forecast

    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) accepts test actuals and predictions. Pass `y_train`
    to show historical context.
    """)


@app.cell
def _(plot_forecast, y_pred, y_test, y_train):
    fig_point = plot_forecast(y_test, y_pred, y_train=y_train)
    fig_point
    return (fig_point,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Compare Multiple Models

    Pass a dict of prediction DataFrames to overlay several models in one
    chart.
    """)


@app.cell
def _(
    PointReductionForecaster,
    RandomForestRegressor,
    plot_forecast,
    y_pred,
    y_test,
    y_train,
):
    forecaster_rf = PointReductionForecaster(estimator=RandomForestRegressor())
    forecaster_rf.fit(y_train, forecasting_horizon=24)
    y_pred_rf = forecaster_rf.predict()

    fig_compare = plot_forecast(
        y_test,
        {"Ridge": y_pred, "RandomForest": y_pred_rf},
        y_train=y_train,
    )
    fig_compare
    return fig_compare, forecaster_rf, y_pred_rf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Plot Prediction Intervals

    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) renders shaded bands when `y_pred` contains interval
    columns (named `{target}__lower_{rate}` and `{target}__upper_{rate}`).
    """)


@app.cell
def _(
    PointReductionForecaster,
    Ridge,
    SplitConformalForecaster,
    plot_forecast,
    y_test,
    y_train,
):
    cp = SplitConformalForecaster(
        point_forecaster=PointReductionForecaster(estimator=Ridge()),
    )
    cp.fit(y_train, forecasting_horizon=24)
    y_pred_interval = cp.predict_interval(coverage_rates=[0.90])

    fig_interval = plot_forecast(y_test, y_pred_interval, y_train=y_train)
    fig_interval
    return cp, fig_interval, y_pred_interval


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Inspect Residuals

    [`plot_residuals`](/pages/api/generated/yohou.plotting.evaluation.plot_residuals/) produces a diagnostic view (residuals over time,
    residuals vs fitted, histogram, Q-Q plot). A well-specified model has
    residuals centred at zero with no time-dependent patterns.
    """)


@app.cell
def _(plot_residuals, y_pred, y_test):
    fig_resid = plot_residuals(y_pred, y_test)
    fig_resid
    return (fig_resid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Check Calibration

    [`plot_calibration`](/pages/api/generated/yohou.plotting.evaluation.plot_calibration/) compares nominal coverage rates against empirical
    coverage. Points close to the diagonal indicate a well-calibrated
    forecaster; points below indicate under-coverage (intervals too narrow).
    """)


@app.cell
def _(plot_calibration, y_pred_interval, y_test):
    fig_cal = plot_calibration(y_pred_interval, y_test, coverage_rates=[0.90])
    fig_cal
    return (fig_cal,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## See Also

    - [How to Visualize Forecasts](/pages/how-to/visualize-forecasts/) (companion page)
    - [How to Visualize and Compare Model Scores](/pages/how-to/visualize-scores/)
    - [`yohou.plotting` API Reference](/pages/api/plotting/)
    """)


if __name__ == "__main__":
    app.run()
