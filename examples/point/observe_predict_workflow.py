# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Observe-Predict Workflow.

Demonstrates the observe/predict cycle for rolling-origin evaluation and
online forecasting, including observe_predict, rewind, and panel data.
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
    # Observe-Predict Workflow

    In production forecasting you rarely retrain from scratch each time new
    data arrives. Instead you **observe** new data (update memory/state) and
    then **predict** using the already-fitted model. This is much cheaper than
    refitting and enables streaming evaluation.

    ## What You'll Learn

    - `observe(y, X)`: push new observations into the forecaster's memory
    - `predict(forecasting_horizon)`: generate forecasts from observed state
    - `observe_predict(y, X)`: atomic observe + predict in a single call
    - `rewind(y, X)`: reset state to a specific window without refitting
    - Rolling-origin evaluation loop using observe/predict
    - Panel data: selective group observation with `panel_group_names`

    ## Prerequisites

    Familiarity with `PointReductionForecaster` and `fit/predict`
    (see `examples/quickstart.py`).
    """)

@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_dominick, fetch_tourism_monthly
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
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

    We split into train / calibration / test. The model is fitted on train,
    then we use the calibration set to demonstrate `observe` and `predict`
    incrementally.
    """)

@app.cell
def _(fetch_tourism_monthly):
    df = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    _n = len(df)
    train_end = int(_n * 0.6)
    cal_end = int(_n * 0.85)

    y_train = df.head(train_end).select("time", "tourists")
    y_cal = df[train_end:cal_end].select("time", "tourists")
    y_test = df.tail(_n - cal_end).select("time", "tourists")

    y_train.tail(3)
    return cal_end, df, train_end, y_cal, y_test, y_train

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
def _(forecaster, forecasting_horizon, mo):
    y_pred_after_obs = forecaster.predict(forecasting_horizon=forecasting_horizon)
    mo.md(
        f"**Prediction starts at**: `{y_pred_after_obs['time'].min()}`\n\n"
        f"**Prediction ends at**: `{y_pred_after_obs['time'].max()}`\n\n"
        f"Predictions start right after the observed data, not the original training end."
    )
    return (y_pred_after_obs,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Observe-Predict in One Call

    `observe_predict()` is an atomic combination of `observe()` + `predict()`.
    This is the most common pattern in rolling evaluation loops.
    """)

@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, forecasting_horizon, y_cal, y_train):
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
    y_pred_op
    return fc_op, y_pred_op

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Rolling-Origin Evaluation Loop

    In practice, you iterate through time: observe a batch, predict the next
    window, score, and repeat. This produces a sequence of forecasts that
    can be compared against actuals.
    """)

@app.cell
def _(
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    forecasting_horizon,
    mo,
    pl,
    y_cal,
    y_train,
):
    _fc_roll = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 6, 12]),
    )
    _fc_roll.fit(y_train, forecasting_horizon=forecasting_horizon)

    _scorer = MeanAbsoluteError()
    _scorer.fit(y_cal)
    _all_preds = []
    _scores = []

    # Roll through calibration set in steps of forecasting_horizon
    for _start in range(0, len(y_cal) - forecasting_horizon + 1, forecasting_horizon):
        _y_batch = y_cal[_start : _start + forecasting_horizon]
        _y_pred = _fc_roll.observe_predict(
            _y_batch,
            forecasting_horizon=forecasting_horizon,
        )
        _all_preds.append(_y_pred)

        # Score if we have truth for the prediction window
        _pred_end = _start + 2 * forecasting_horizon
        if _pred_end <= len(y_cal):
            _y_truth = y_cal[_start + forecasting_horizon : _pred_end]
            _s = float(_scorer.score(_y_truth, _y_pred))
            _scores.append(_s)

    _preds_combined = pl.concat(_all_preds, how="vertical")
    mo.md(
        f"**Rolling evaluation**: {len(_all_preds)} windows\n\n"
        f"**MAE per window**: {[round(s, 1) for s in _scores]}\n\n"
        f"**Mean MAE**: {sum(_scores) / len(_scores):.1f}" if _scores else
        f"**Rolling evaluation**: {len(_all_preds)} windows (no complete truth windows for scoring)"
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Rewind

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
    y_cal,
    y_train,
):
    _fc_rw = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 6, 12]),
    )
    _fc_rw.fit(y_train, forecasting_horizon=forecasting_horizon)

    # Observe full calibration set
    _fc_rw.observe(y_cal)
    _time_after_obs = _fc_rw.observed_time_

    # Rewind to just the first half of calibration data
    _rewind_data = y_cal.head(len(y_cal) // 2)
    _fc_rw.rewind(_rewind_data)
    _time_after_rewind = _fc_rw.observed_time_

    mo.md(
        f"**After observing all cal data**: `{_time_after_obs}`\n\n"
        f"**After rewind to half**: `{_time_after_rewind}`\n\n"
        f"The forecaster state is now as if we had only observed "
        f"the first {len(y_cal) // 2} calibration months."
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Panel Data: Selective Observation

    With panel data, you can observe and predict for **specific groups only**
    using `panel_group_names`. This is useful when different groups receive
    new data at different times.
    """)

@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, fetch_dominick, mo):
    _panel = fetch_dominick().frame.select(
        "time", "T7__profit", "T11__profit", "T12__profit",
        "T13__profit", "T15__profit", "T19__profit",
        "T22__profit", "T23__profit", "T24__profit",
    )
    _profit_cols = [c for c in _panel.columns if c.endswith("__profit")]
    _split = int(len(_panel) * 0.8)
    _cal_split = int(len(_panel) * 0.9)

    _y_train_p = _panel.head(_split).select("time", *_profit_cols)
    _y_cal_p = _panel[_split:_cal_split].select("time", *_profit_cols)

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
    - **Rolling evaluation**: Loop `observe_predict` to evaluate forecasts at each time step
    - **Panel selective observation**: Use `panel_group_names` to observe/predict subsets of groups independently
    - Observations update transformer state (buffers) but **do not refit the model**

    ## Next Steps

    - **Cross-validation**: See `examples/cross_validation.py` for automated rolling-origin evaluation with `GridSearchCV`
    - **Scoring**: See `examples/scoring.py` for evaluating forecast quality with point and interval metrics
    - **Panel forecasting**: See `examples/point/panel_forecasting.py` for comprehensive panel workflows
    """)

if __name__ == "__main__":
    app.run()
