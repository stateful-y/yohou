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
    "title": "Observe-Predict Workflow",
    "description": "Incrementally observe new data and predict without refitting, including rewind for memory management and selective panel group update operations.",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Observe-Predict Workflow

    In production forecasting you rarely retrain from scratch each time new
    data arrives. Instead you **observe** new data (update memory/state) and
    then **predict** using the already-fitted model. This is much cheaper than
    refitting and enables streaming evaluation.

    ## What You'll Learn

    - `observe(y, X)`: push new observations into the forecaster's memory
    - `predict(forecasting_horizon)`: generate forecasts recursively from observed state
    - `observe_predict(y, X)`: atomic observe + predict in a single call
    - `rewind(y, X)`: reset state to a specific window without refitting
    - Panel data: selective group observation with `panel_group_names`

    ## Prerequisites

    Familiarity with [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) and `fit/predict`
    (see `examples/quickstart.py`).
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    from yohou.datasets import fetch_dominick, fetch_tourism_monthly
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        LagTransformer,
        PointReductionForecaster,
        Ridge,
        fetch_dominick,
        fetch_tourism_monthly,
        pl,
        plot_forecast,
        plot_time_series,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We split into train / calibration / test. The model is fitted on train,
    then we use the calibration set to demonstrate `observe` and `predict`
    incrementally.
    """)


@app.cell
def _(fetch_tourism_monthly, plot_time_series, train_test_split):
    df = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    _n = len(df)
    train_end = int(_n * 0.6)
    cal_end = int(_n * 0.85)

    y_train, _rest = train_test_split(df, train_size=train_end, shuffle=False)
    y_cal, y_test = train_test_split(_rest, train_size=cal_end - train_end, shuffle=False)

    plot_time_series(df, title="Monthly Tourism (T1)")
    return y_cal, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Fit the Model

    Train on the initial training window.
    """)


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, y_train):
    forecaster = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 6, 12]),
    )
    forecasting_horizon = 6
    forecaster.fit(y_train, forecasting_horizon=forecasting_horizon)
    return forecaster, forecasting_horizon


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Observe New Data

    `observe()` appends new time steps to the forecaster's memory **without
    refitting the model**. This updates the observation buffer used by
    transformers and the forecaster's internal time tracking.
    """)


@app.cell
def _(forecaster, mo, y_cal):
    # Feed the first 6 months of calibration data
    _y_observe_1 = y_cal.head(6)
    forecaster.observe(_y_observe_1)

    mo.md(
        f"**Observed time after first observe**: "
        f"`{forecaster.observed_time_}`\n\n"
        f"The model now knows about 6 additional months without refitting."
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Predict After Observing

    Now `predict()` produces forecasts starting from the newly observed
    position, 6 months ahead of where the original training ended.
    """)


@app.cell
def _(forecaster, forecasting_horizon, pl, plot_forecast, y_cal, y_train):
    y_pred_after_obs = forecaster.predict(forecasting_horizon=forecasting_horizon)
    _y_truth = y_cal.slice(6, forecasting_horizon)
    _y_history = pl.concat([y_train, y_cal.head(6)])
    plot_forecast(
        _y_truth,
        y_pred_after_obs,
        y_train=_y_history,
        n_history=24,
        title="Predict After Observing 6 Months",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Observe-Predict in One Call

    `observe_predict()` is an atomic combination of `observe()` + `predict()`.
    This is the most common pattern in rolling evaluation loops.
    """)


@app.cell
def _(
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    forecasting_horizon,
    pl,
    plot_forecast,
    y_cal,
    y_train,
):
    # Fresh forecaster for clean demo
    fc_op = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 6, 12]),
    )
    fc_op.fit(y_train, forecasting_horizon=forecasting_horizon)

    # Observe first 6 months and predict in one call
    y_pred_op = fc_op.observe_predict(
        y_cal.head(6),
        forecasting_horizon=forecasting_horizon,
    )
    _y_truth_op = y_cal.slice(6, forecasting_horizon)
    _y_history_op = pl.concat([y_train, y_cal.head(6)])
    plot_forecast(
        _y_truth_op,
        y_pred_op,
        y_train=y_train,
        n_history=24,
        title="Observe-Predict in One Call",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Rewind

    `rewind()` resets the forecaster's observation state to a specific data
    window **without refitting**. This is useful for backtesting: rewind to
    a point in the past and re-evaluate without retraining.
    """)


@app.cell
def _(
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    forecasting_horizon,
    mo,
    pl,
    plot_forecast,
    y_cal,
    y_train,
):
    fc_rw = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 6, 12]),
    )
    fc_rw.fit(y_train, forecasting_horizon=forecasting_horizon)

    # Observe full calibration set
    fc_rw.observe(y_cal)
    _time_after_obs = fc_rw.observed_time_

    # Rewind to just the first half of calibration data
    _half = len(y_cal) // 2
    _rewind_data = y_cal.head(_half)
    fc_rw.rewind(_rewind_data)
    _time_after_rewind = fc_rw.observed_time_

    # Predict from the rewound position
    y_pred_rw = fc_rw.predict(forecasting_horizon=forecasting_horizon)
    _y_truth_rw = y_cal.slice(_half, forecasting_horizon)
    _y_history_rw = pl.concat([y_train, y_cal.head(_half)])

    mo.vstack([
        mo.md(
            f"**After observing all cal data**: `{_time_after_obs}`\n\n"
            f"**After rewind to half**: `{_time_after_rewind}`\n\n"
            f"The forecaster state is now as if we had only observed "
            f"the first {_half} calibration months. Predictions start "
            f"from the rewound position."
        ),
        plot_forecast(
            _y_truth_rw,
            y_pred_rw,
            y_train=_y_history_rw,
            n_history=24,
            title="Predict After Rewind",
        ),
    ])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Panel Data: Selective Observation

    With panel data, you can observe and predict for **specific groups only**
    using `panel_group_names`. This is useful when different groups receive
    new data at different times.
    """)


@app.cell
def _(
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    fetch_dominick,
    mo,
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
    _y_train_p, _y_rest_p = train_test_split(_selected, test_size=0.2, shuffle=False)
    _y_cal_p = _y_rest_p.head(int(len(_panel) * 0.1))

    _fc_panel = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 4]),
    )
    _horizon_p = 4
    _fc_panel.fit(_y_train_p, forecasting_horizon=_horizon_p)

    # Observe ONLY T7-T12 groups (simulate partial data arrival)
    _fc_panel.observe(
        _y_cal_p.head(4),
        panel_group_names=["T7", "T11", "T12"],
    )

    # Predict for T7-T12 only (other groups still at old position)
    _y_pred_s1 = _fc_panel.predict(
        forecasting_horizon=_horizon_p,
        panel_group_names=["T7", "T11", "T12"],
    )

    mo.md(
        f"**Predicted columns**: {[c for c in _y_pred_s1.columns if c != 'time' and c != 'observed_time']}\n\n"
        f"Only T7, T11, T12 groups were observed and predicted, other groups remain at their previous state."
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **`observe(y, X)`** appends new data to the forecaster's memory without refitting (cheap, incremental)
    - **`predict(forecasting_horizon)`** generates forecasts from the current observed state
    - **`observe_predict(y, X)`** is the atomic combination: the workhorse of rolling evaluation
    - **`rewind(y, X)`** resets state to a specific window without refitting (useful for backtesting)
    - **Panel selective observation**: Use `panel_group_names` to observe/predict subsets of groups independently
    - Observations update transformer state (buffers) but **do not refit the model**

    ## Next Steps

    - **Cross-validation**: See `examples/cross_validation.py` for automated rolling-origin evaluation with [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/)
    - **Scoring**: See `examples/scoring.py` for evaluating forecast quality with point and interval metrics
    - **Panel forecasting**: See [`examples/point/panel_forecasting.py`](/examples/point/panel_forecasting/) for comprehensive panel workflows
    """)


if __name__ == "__main__":
    app.run()
