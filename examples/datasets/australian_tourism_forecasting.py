# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Tourism Quarterly Forecasting.

Panel forecasting workflow for the Tourism Quarterly dataset: fit, predict,
observe-predict, scoring, and per-group analysis across quarterly tourism series.
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
    # Tourism Quarterly Forecasting

    The Tourism Quarterly dataset contains 427 quarterly tourism series
    from the Monash forecasting competition. We select 8 same-length series
    (T3-T10) for a manageable panel, demonstrating **panel forecasting**.

    ## What You'll Learn

    - Fitting a panel forecaster on multiple tourism time series simultaneously
    - Predicting and evaluating per-group forecasts
    - Rolling observe-predict evaluation on panel data
    - Selective observation of individual groups
    - Scoring panel forecasts with per-group and aggregated metrics

    ## Prerequisites

    Familiarity with panel data concepts and `PointReductionForecaster`
    (see `examples/datasets/australian_tourism.py` for dataset exploration,
    `examples/quickstart.py` for forecasting basics).
    """)

@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_tourism_quarterly
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
        fetch_tourism_quarterly,
        inspect_locality,
        pl,
        plot_forecast,
        plot_time_series,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load and Inspect Panel Structure
    """)

@app.cell
def _(fetch_tourism_quarterly, inspect_locality, mo):
    _all = fetch_tourism_quarterly().frame
    # Select 8 same-length series (T3-T10) to avoid NaN from uneven lengths
    _value_cols = [c for c in _all.columns if c != "time"]
    _cols = ["time"] + _value_cols[2:10]
    tourism = _all.select(_cols).drop_nulls()
    _globals, groups = inspect_locality(tourism)

    mo.md(
        f"**Shape**: {tourism.shape}\n\n"
        f"**Panel groups**: {len(groups)} series\n\n"
        f"**Groups**: {list(groups.keys())}\n\n"
        f"**Columns per group**: {list(groups.values())[0]}\n\n"
        f"**Time range**: {tourism['time'].min()} to {tourism['time'].max()}"
    )
    return groups, tourism

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Train-Test Split
    """)

@app.cell
def _(mo, tourism):
    _trip_cols = [c for c in tourism.columns if c.endswith("__tourists")]
    _split = int(len(tourism) * 0.8)
    y_train = tourism.head(_split).select("time", *_trip_cols)
    y_test = tourism.tail(len(tourism) - _split).select("time", *_trip_cols)

    mo.md(
        f"**Train**: {len(y_train)} quarters "
        f"({y_train['time'].min()} to {y_train['time'].max()})\n\n"
        f"**Test**: {len(y_test)} quarters "
        f"({y_test['time'].min()} to {y_test['time'].max()})"
    )
    return y_test, y_train

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Fit a Panel Forecaster

    A single `PointReductionForecaster` fits a **separate model per panel
    group** automatically. Each series gets its own Ridge regression trained
    on lag features derived from that series only.
    """)

@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, y_test, y_train):
    forecaster = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 4]),
    )
    forecasting_horizon = len(y_test)
    forecaster.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred = forecaster.predict(forecasting_horizon=forecasting_horizon)
    return forecaster, forecasting_horizon, y_pred

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Visualise Selected Series

    Use `panel_group_names` to view forecasts for specific series.
    """)

@app.cell
def _(plot_forecast, y_pred, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred,
        y_train=y_train,
        n_history=20,
        panel_group_names=["T3", "T4", "T5"],
        title="Tourism Forecasts: Series T3-T5",
    )

@app.cell
def _(plot_forecast, y_pred, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred,
        y_train=y_train,
        n_history=20,
        panel_group_names=["T6", "T7", "T8"],
        title="Tourism Forecasts: Series T6-T8",
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Per-Group Scoring

    Score each series independently to identify which are harder to forecast.
    """)

@app.cell
def _(MeanAbsoluteError, groups, mo, y_pred, y_test, y_train):
    _scores = {}
    for _group in groups:
        _scorer = MeanAbsoluteError(panel_group_names=[_group])
        _scorer.fit(y_train)
        _s = _scorer.score(y_test, y_pred)
        _scores[_group] = round(float(_s), 1)

    _sorted = sorted(_scores.items(), key=lambda x: x[1])
    _lines = [f"| {name} | {score} |" for name, score in _sorted]
    mo.md(
        "| Series | MAE |\n|--------|-----|\n" + "\n".join(_lines)
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Rolling Observe-Predict

    Simulate a production workflow: fit on early data, then roll forward
    quarter by quarter using `observe_predict`.
    """)

@app.cell
def _(
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    mo,
    pl,
    tourism,
):
    _trip_cols = [c for c in tourism.columns if c.endswith("__tourists")]
    _train_end = int(len(tourism) * 0.6)
    _y_early = tourism.head(_train_end).select("time", *_trip_cols)
    _y_rest = tourism.tail(len(tourism) - _train_end).select("time", *_trip_cols)

    _fc_roll = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 4]),
    )
    _step = 4  # quarterly steps
    _fc_roll.fit(_y_early, forecasting_horizon=_step)

    _scorer = MeanAbsoluteError()
    _scorer.fit(_y_early)
    _all_scores = []

    for _i in range(0, len(_y_rest) - _step + 1, _step):
        _batch = _y_rest[_i : _i + _step]
        _y_pred_roll = _fc_roll.observe_predict(
            _batch,
            forecasting_horizon=_step,
        )
        # Score if truth is available
        _truth_end = _i + 2 * _step
        if _truth_end <= len(_y_rest):
            _truth = _y_rest[_i + _step : _truth_end]
            _s = _scorer.score(_truth, _y_pred_roll)
            _all_scores.append(round(float(_s), 1))

    mo.md(
        f"**Rolling windows evaluated**: {len(_all_scores)}\n\n"
        f"**MAE per window**: {_all_scores}\n\n"
        f"**Average MAE**: {sum(_all_scores) / len(_all_scores):.1f}"
        if _all_scores else
        "**Rolling windows**: completed (no full truth windows for scoring)"
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Selective Group Observation

    In practice, data for different series may arrive at different times.
    Use `panel_group_names` to observe only the groups that have new data.
    """)

@app.cell
def _(
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    mo,
    plot_forecast,
    tourism,
):
    _trip_cols = [c for c in tourism.columns if c.endswith("__tourists")]
    _split = int(len(tourism) * 0.8)
    _y_tr = tourism.head(_split).select("time", *_trip_cols)
    _y_new = tourism[_split : _split + 4].select("time", *_trip_cols)

    _fc_sel = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 4]),
    )
    _fc_sel.fit(_y_tr, forecasting_horizon=4)

    # Observe only three selected series
    _three = ["T3", "T4", "T5"]
    _fc_sel.observe(_y_new, panel_group_names=_three)

    # Predict only those three series
    _y_pred_sel = _fc_sel.predict(
        forecasting_horizon=4,
        panel_group_names=_three,
    )
    plot_forecast(
        tourism.tail(len(tourism) - _split - 4).head(4).select("time", *_trip_cols),
        _y_pred_sel,
        panel_group_names=_three,
        title="Selective Observation: Series T3-T5 Only",
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - A single `PointReductionForecaster` automatically handles **panel data** (one model per group)
    - Use `panel_group_names` in `plot_forecast` to view specific groups
    - **Per-group scoring** reveals which groups are harder to forecast
    - **Rolling observe-predict** simulates production deployment without refitting
    - **Selective observation** supports asynchronous data arrival across groups
    - Quarterly seasonality is captured by `LagTransformer(lag=[1, 4])` (1-quarter and 1-year lags)

    ## Next Steps

    - **Dataset exploration**: See `examples/datasets/australian_tourism.py` for EDA on this dataset
    - **Panel data concepts**: See `examples/panel_data.py` for panel naming conventions
    - **Pedestrian forecasting**: See `examples/datasets/pedestrian_counts_forecasting.py` for another panel forecasting example
    """)

if __name__ == "__main__":
    app.run()
