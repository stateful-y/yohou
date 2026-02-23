"""Pedestrian Counts Forecasting.

Panel forecasting workflow for the Pedestrian Counts dataset: per-sensor
evaluation, rolling observe-predict, and selective group observation on
hourly panel data.
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
    # Pedestrian Counts Forecasting

    The Pedestrian Counts dataset contains hourly pedestrian counts from
    Melbourne sensors (20 by default). We work with a subset of sensors to
    demonstrate a realistic **panel forecasting** scenario on high-frequency
    data.

    ## What You'll Learn

    - Forecasting pedestrian counts across panel groups (sensors)
    - Per-sensor evaluation and comparison
    - Rolling observe-predict on hourly panel data
    - Selective group observation (asynchronous sensor data arrival)

    ## Prerequisites

    Familiarity with panel data conventions and `PointReductionForecaster`
    (see `examples/datasets/walmart_sales.py` for dataset exploration,
    `examples/quickstart.py` for forecasting basics).
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_pedestrian_counts
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer
    from yohou.utils.panel import inspect_locality

    return (
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        fetch_pedestrian_counts,
        inspect_locality,
        pl,
        plot_forecast,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load and Inspect
    """)
    return


@app.cell
def _(fetch_pedestrian_counts, inspect_locality, mo):
    _all = fetch_pedestrian_counts().frame
    # Select first 6 sensors for a manageable panel
    _cols = ["time"] + [c for c in _all.columns if c != "time"][:6]
    pedestrian = _all.select(_cols)
    _globals, groups = inspect_locality(pedestrian)

    mo.md(
        f"**Shape**: {pedestrian.shape}\n\n"
        f"**Panel groups**: {len(groups)} sensors\n\n"
        f"**Groups**: {list(groups.keys())}\n\n"
        f"**Columns per group**: {list(groups.values())[0]}\n\n"
        f"**Time range**: {pedestrian['time'].min()} to {pedestrian['time'].max()}\n\n"
        f"Each sensor has a single `count` column (hourly pedestrian counts)."
    )
    return groups, pedestrian


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Forecast Pedestrian Counts

    With hourly data, we use lags that capture both short-term (lag 1)
    and daily (lag 24) patterns. We forecast a 24-hour horizon.
    """)
    return


@app.cell
def _(mo, pedestrian):
    _count_cols = [c for c in pedestrian.columns if c.endswith("__count")]
    _split = int(len(pedestrian) * 0.8)
    y_train = pedestrian.head(_split).select("time", *_count_cols)
    y_test = pedestrian.tail(len(pedestrian) - _split).select("time", *_count_cols)

    mo.md(
        f"**Count columns**: {_count_cols}\n\n"
        f"**Train**: {len(y_train)} hours, **Test**: {len(y_test)} hours"
    )
    return y_test, y_train


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, y_test, y_train):
    fc = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 24]),
    )
    _horizon = min(len(y_test), 48)
    fc.fit(y_train, forecasting_horizon=_horizon)
    y_pred = fc.predict(forecasting_horizon=_horizon)
    return fc, y_pred


@app.cell
def _(groups, plot_forecast, y_pred, y_test, y_train):
    _group_names = list(groups.keys())[:3]
    plot_forecast(
        y_test.head(len(y_pred)),
        y_pred,
        y_train=y_train,
        n_history=48,
        panel_group_names=_group_names,
        title="Hourly Pedestrian Count Forecast: First 3 Sensors",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. All Sensors Forecast

    Forecasting all 6 sensors simultaneously to compare predictions
    across locations.
    """)
    return


@app.cell
def _(groups, plot_forecast, y_pred, y_test, y_train):
    plot_forecast(
        y_test.head(len(y_pred)),
        y_pred,
        y_train=y_train,
        n_history=48,
        panel_group_names=list(groups.keys()),
        title="Hourly Pedestrian Count Forecast: All 6 Sensors",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Per-Sensor Scoring
    """)
    return


@app.cell
def _(MeanAbsoluteError, groups, mo, y_pred, y_test, y_train):
    _scores = {}
    _y_test_trimmed = y_test.head(len(y_pred))
    for _group in groups:
        _scorer = MeanAbsoluteError(panel_group_names=[_group])
        _scorer.fit(y_train)
        _s = _scorer.score(_y_test_trimmed, y_pred)
        _scores[_group] = round(float(_s), 1)

    _lines = [f"| {name} | {score} |" for name, score in sorted(_scores.items(), key=lambda x: x[1])]
    mo.md(
        "**Pedestrian Count MAE per Sensor**\n\n"
        "| Sensor | MAE |\n|--------|-----|\n" + "\n".join(_lines)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Rolling Observe-Predict

    Roll forward in 24-hour steps, observing then predicting the next
    24 hours at each step.
    """)
    return


@app.cell
def _(LagTransformer, MeanAbsoluteError, PointReductionForecaster, Ridge, mo, pedestrian):
    _count_cols = [c for c in pedestrian.columns if c.endswith("__count")]
    _train_end = int(len(pedestrian) * 0.5)
    _y_early = pedestrian.head(_train_end).select("time", *_count_cols)
    _y_rest = pedestrian.tail(len(pedestrian) - _train_end).select("time", *_count_cols)

    _fc_roll = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 24]),
    )
    _step = 24
    _fc_roll.fit(_y_early, forecasting_horizon=_step)

    _scorer = MeanAbsoluteError()
    _scorer.fit(_y_early)
    _window_scores = []

    for _i in range(0, min(len(_y_rest) - _step + 1, _step * 10), _step):
        _batch = _y_rest[_i : _i + _step]
        _y_pred_roll = _fc_roll.observe_predict(
            _batch,
            forecasting_horizon=_step,
        )
        _truth_end = _i + 2 * _step
        if _truth_end <= len(_y_rest):
            _truth = _y_rest[_i + _step : _truth_end]
            _s = _scorer.score(_truth, _y_pred_roll)
            _window_scores.append(round(float(_s), 1))

    mo.md(
        f"**Rolling windows**: {len(_window_scores)} scored\n\n"
        f"**MAE per window**: {_window_scores}\n\n"
        f"**Average MAE**: {sum(_window_scores) / len(_window_scores):.1f}"
        if _window_scores else
        f"**Rolling evaluation completed** (insufficient truth for scoring)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Selective Sensor Observation

    Simulate a scenario where Sensor T1 data arrives before the others.
    """)
    return


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, mo, pedestrian):
    _count_cols = [c for c in pedestrian.columns if c.endswith("__count")]
    _split = int(len(pedestrian) * 0.8)
    _y_tr = pedestrian.head(_split).select("time", *_count_cols)
    _y_new = pedestrian[_split : _split + 24].select("time", *_count_cols)

    _fc_sel = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 24]),
    )
    _fc_sel.fit(_y_tr, forecasting_horizon=24)

    # Observe only Sensor T1
    _fc_sel.observe(_y_new, panel_group_names=["T1"])
    _time_t1 = _fc_sel.observed_time_["T1"]

    # Other sensors still at old position
    _time_t2 = "unchanged (not yet observed)"

    mo.md(
        f"**Sensor T1 observed time**: `{_time_t1}`\n\n"
        f"**Sensor T2 observed time**: `{_time_t2}`\n\n"
        f"Other sensors remain at their previous position until new data arrives."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Hourly panel forecasting**: Lags `[1, 24]` capture short-term and daily patterns
    - **Per-sensor scoring** identifies which locations are harder to predict
    - **Rolling observe-predict** works the same way as with standard data but across all groups
    - **Selective observation** supports asynchronous data arrival (Sensor T1 before others)
    - Regularisation (Ridge) helps prevent overfitting across many sensors

    ## Next Steps

    - **Dataset exploration**: See `examples/datasets/walmart_sales.py` for EDA on this dataset
    - **Australian Tourism**: See `examples/datasets/australian_tourism_forecasting.py` for quarterly panel forecasting
    - **Panel concepts**: See `examples/panel_data.py` for the `__` naming convention
    """)
    return


if __name__ == "__main__":
    app.run()
