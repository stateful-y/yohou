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
    - `observe_predict_interval(y, X)`: observe + predict prediction intervals
    - `observe_predict_class_proba(y, X)`: observe + predict class probabilities
    - `observe_predict(y, X)` on a class-proba forecaster: observe + predict hard labels
    - `rewind(y, X)`: reset state to a specific window without refitting
    - Panel data: selective group observation with `panel_group_names`

    > **Note**: The observe/predict API is **independent of `reduction_strategy`**.
    > Whether you use `"multi-output"`, `"direct"`, or `"dir-rec"`, the observe,
    > predict, and rewind methods work identically. Only model training internals change.

    ## Prerequisites

    Familiarity with [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) and `fit/predict`
    (see `examples/quickstart.py`).
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    from yohou.class_proba import ClassProbaReductionForecaster
    from yohou.datasets import (
        fetch_air_quality_classification,
        fetch_dominick,
        fetch_sunspot,
        fetch_tourism_monthly,
    )
    from yohou.interval import SplitConformalForecaster
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer

    return (
        ClassProbaReductionForecaster,
        DecisionTreeClassifier,
        LagTransformer,
        PointReductionForecaster,
        Ridge,
        SeasonalNaive,
        SplitConformalForecaster,
        fetch_air_quality_classification,
        fetch_dominick,
        fetch_sunspot,
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
    ## 7. Prediction Intervals

    [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/)
    (and other interval forecasters) support `observe()` and
    `predict_interval()`.  After observing calibration data, the next
    prediction produces lower/upper bounds at the requested coverage rate.

    The output DataFrame contains `"time"` and a pair of columns per
    coverage level: `{target}_upper_{rate}` / `{target}_lower_{rate}`.
    """)


@app.cell
def _(SeasonalNaive, SplitConformalForecaster, fetch_sunspot, train_test_split):
    sunspots = fetch_sunspot().frame
    ss_train, _ss_rest = train_test_split(sunspots, test_size=0.3, shuffle=False)
    ss_cal, ss_test = train_test_split(_ss_rest, test_size=0.5, shuffle=False)
    ss_fh = 12

    conformal = SplitConformalForecaster(
        point_forecaster=SeasonalNaive(seasonality=132),
    )
    conformal.fit(ss_train, forecasting_horizon=ss_fh)

    print(f"Train: {len(ss_train)}, Cal: {len(ss_cal)}, Test: {len(ss_test)}")
    return conformal, ss_cal, ss_fh, ss_test, ss_train


@app.cell
def _(conformal, pl, plot_forecast, ss_cal, ss_fh, ss_test, ss_train):
    # Observe calibration data, then predict intervals from the final state
    conformal.observe(ss_cal)
    y_pred_interval = conformal.predict_interval(
        forecasting_horizon=ss_fh,
        coverage_rates=[0.9],
    )

    _y_truth_int = ss_test.head(ss_fh)
    _y_history_int = pl.concat([ss_train, ss_cal])
    plot_forecast(
        _y_truth_int,
        y_pred_interval,
        y_train=_y_history_int,
        coverage_rates=[0.9],
        n_history=60,
        title="observe_predict_interval - 90% Prediction Interval",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Class-Probability Forecasting (Soft Classification)

    [`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.reduction.ClassProbaReductionForecaster/)
    supports `observe()` and `predict_class_proba()`.  After observing
    calibration data, it returns the full **probability distribution** over
    classes for the next horizon.

    The output DataFrame contains `"time"`, `"observed_time"`, and one
    column per class: `{target}_proba_{class_label}` (float values summing
    to 1.0 at each time step).  [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/)
    auto-detects these `_proba_` columns and renders a stacked area chart.
    """)


@app.cell
def _(
    ClassProbaReductionForecaster,
    DecisionTreeClassifier,
    LagTransformer,
    fetch_air_quality_classification,
    train_test_split,
):
    cls_data = fetch_air_quality_classification()
    cls_y, cls_X = cls_data.y, cls_data.X
    cls_train_end = len(cls_y) - 400
    cls_cal_end = len(cls_y) - 200

    cls_y_train, _cls_rest_y = train_test_split(cls_y, train_size=cls_train_end, shuffle=False)
    cls_y_cal, cls_y_test = train_test_split(_cls_rest_y, train_size=cls_cal_end - cls_train_end, shuffle=False)
    cls_X_train, _cls_rest_X = train_test_split(cls_X, train_size=cls_train_end, shuffle=False)
    cls_X_cal, cls_X_test = train_test_split(_cls_rest_X, train_size=cls_cal_end - cls_train_end, shuffle=False)
    cls_fh = 24

    cls_forecaster = ClassProbaReductionForecaster(
        estimator=DecisionTreeClassifier(random_state=42),
        feature_transformer=LagTransformer(lag=[1, 2, 3, 6, 12, 24]),
    )
    cls_forecaster.fit(cls_y_train, cls_X_train, forecasting_horizon=cls_fh)

    print(f"Classes: {cls_data.classes}")
    print(f"Train: {len(cls_y_train)}, Cal: {len(cls_y_cal)}, Test: {len(cls_y_test)}")
    return cls_X_cal, cls_X_test, cls_fh, cls_forecaster, cls_y_cal, cls_y_test, cls_y_train


@app.cell
def _(cls_X_cal, cls_X_test, cls_fh, cls_forecaster, cls_y_cal, cls_y_test, plot_forecast):
    # Observe calibration data, then predict class probabilities from the final state
    cls_forecaster.observe(cls_y_cal, X=cls_X_cal)
    cls_y_proba = cls_forecaster.predict_class_proba(
        X=cls_X_test,
        forecasting_horizon=cls_fh,
    )

    print("Probability columns:", [c for c in cls_y_proba.columns if "_proba_" in c])
    plot_forecast(
        cls_y_test.head(cls_fh),
        cls_y_proba,
        title="predict_class_proba - Stacked Area",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Hard-Label Prediction on a Class-Proba Forecaster

    Calling `predict()` on a
    [`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.reduction.ClassProbaReductionForecaster/)
    returns the **argmax class** at each time step instead of the full
    probability distribution.  This is the hard-label counterpart.

    The output DataFrame contains `"time"`, `"observed_time"`, and one
    `String`-typed column per target with the most-likely class name.
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) auto-detects the categorical dtype
    and renders a step chart.
    """)


@app.cell
def _(cls_X_test, cls_fh, cls_forecaster, cls_y_test, cls_y_train, plot_forecast):
    # Predict from the already-observed state (hard labels via predict)
    cls_y_labels = cls_forecaster.predict(
        X=cls_X_test,
        forecasting_horizon=cls_fh,
    )

    print("Hard-label dtypes:", cls_y_labels.dtypes)
    plot_forecast(
        cls_y_test.head(cls_fh),
        cls_y_labels,
        y_train=cls_y_train,
        n_history=50,
        title="predict (hard labels) - Step Chart",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Panel Data: Selective Observation

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

    | Method | Forecaster Type | Returns |
    |--------|----------------|--------|
    | `observe_predict()` | Point | Numeric predictions |
    | `observe_predict()` | Class-proba | Argmax hard labels (String) |
    | `observe_predict_interval()` | Interval | Lower/upper bounds per coverage rate |
    | `observe_predict_class_proba()` | Class-proba | Probability per class (float) |

    - **`observe(y, X)`** appends new data to the forecaster's memory without refitting (cheap, incremental)
    - **`predict(forecasting_horizon)`** generates forecasts from the current observed state
    - **`observe_predict(y, X)`** is the atomic combination: the workhorse of rolling evaluation
    - **`observe_predict_interval(y, X)`** returns prediction intervals with `coverage_rates`
    - **`observe_predict_class_proba(y, X)`** returns full probability distributions over classes
    - **`rewind(y, X)`** resets state to a specific window without refitting (useful for backtesting)
    - **Panel selective observation**: Use `panel_group_names` to observe/predict subsets of groups independently
    - Observations update transformer state (buffers) but **do not refit the model**

    ## Next Steps

    - **Reduction strategies**: See [`reduction_strategies.py`](/examples/point/reduction_strategies/) for multi-output, direct, and dir-rec comparison
    - **Interval forecasting**: See [`interval_metrics.py`](/examples/metrics/interval_metrics/) for scoring prediction intervals
    - **Classification forecasting**: See [`class_proba_forecaster.py`](/examples/point/class_proba_forecaster/) for the full classification workflow
    - **Classification metrics**: See [`class_proba_metrics.py`](/examples/metrics/class_proba_metrics/) for LogLoss, BrierScore, and Accuracy
    - **Panel forecasting**: See [`panel_forecasting.py`](/examples/point/panel_forecasting/) for comprehensive panel workflows
    """)


if __name__ == "__main__":
    app.run()
