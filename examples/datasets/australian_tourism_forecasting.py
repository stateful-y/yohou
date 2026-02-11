"""Australian Tourism Forecasting.

Panel forecasting workflow for the Australian Tourism dataset: fit, predict,
observe-predict, scoring, and per-group analysis across 8 Australian states.
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
    # Australian Tourism Forecasting

    The Australian Tourism dataset contains quarterly tourism trip counts
    for 8 Australian states over 20 years (1998-2017). With its strong
    seasonality and cross-state correlations, it is an ideal showcase
    for **panel forecasting**.

    ## What You'll Learn

    - Fitting a panel forecaster on 8 state-level time series simultaneously
    - Predicting and evaluating per-group forecasts
    - Rolling observe-predict evaluation on panel data
    - Selective observation of individual states
    - Scoring panel forecasts with per-group and aggregated metrics

    ## Prerequisites

    Familiarity with panel data concepts and `PointReductionForecaster`
    (see `examples/datasets/australian_tourism.py` for dataset exploration,
    `examples/quickstart.py` for forecasting basics).
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import load_australian_tourism
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
        inspect_locality,
        load_australian_tourism,
        pl,
        plot_forecast,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load and Inspect Panel Structure
    """)
    return


@app.cell
def _(inspect_locality, load_australian_tourism, mo):
    tourism = load_australian_tourism()
    _globals, groups = inspect_locality(tourism)

    mo.md(
        f"**Shape**: {tourism.shape}\n\n"
        f"**Panel groups**: {len(groups)} states\n\n"
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
    return


@app.cell
def _(mo, tourism):
    _trip_cols = [c for c in tourism.columns if c.endswith("__trips")]
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
    group** automatically. Each state gets its own Ridge regression trained
    on lag features derived from that state's time series only.
    """)
    return


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, y_test, y_train):
    forecaster = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        target_transformer=LagTransformer(lag=[1, 4]),
    )
    forecasting_horizon = len(y_test)
    forecaster.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred = forecaster.predict(forecasting_horizon=forecasting_horizon)
    return forecaster, forecasting_horizon, y_pred


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Visualise Selected States

    Use `panel_group_names` to view forecasts for specific states.
    """)
    return


@app.cell
def _(plot_forecast, y_pred, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred,
        y_train=y_train,
        n_history=20,
        panel_group_names=["new_south_wales", "victoria", "queensland"],
        title="Tourism Forecasts: Eastern States",
    )
    return


@app.cell
def _(plot_forecast, y_pred, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred,
        y_train=y_train,
        n_history=20,
        panel_group_names=["western_australia", "tasmania", "northern_territory"],
        title="Tourism Forecasts: Western and Northern States",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Per-Group Scoring

    Score each state independently to identify which states are harder to
    forecast.
    """)
    return


@app.cell
def _(MeanAbsoluteError, groups, mo, y_pred, y_test):
    _scorer = MeanAbsoluteError()
    _scores = {}
    for _group in groups:
        _s = _scorer.score(
            y_test, y_pred,
            panel_group_names=[_group],
        )
        _val = _s.drop("time").to_series().mean()
        _scores[_group] = round(float(_val), 1)

    _sorted = sorted(_scores.items(), key=lambda x: x[1])
    _lines = [f"| {name} | {score} |" for name, score in _sorted]
    mo.md(
        "| State | MAE |\n|-------|-----|\n" + "\n".join(_lines)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Rolling Observe-Predict

    Simulate a production workflow: fit on early data, then roll forward
    quarter by quarter using `observe_predict`.
    """)
    return


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
    _trip_cols = [c for c in tourism.columns if c.endswith("__trips")]
    _train_end = int(len(tourism) * 0.6)
    _y_early = tourism.head(_train_end).select("time", *_trip_cols)
    _y_rest = tourism.tail(len(tourism) - _train_end).select("time", *_trip_cols)

    _fc_roll = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        target_transformer=LagTransformer(lag=[1, 4]),
    )
    _step = 4  # quarterly steps
    _fc_roll.fit(_y_early, forecasting_horizon=_step)

    _scorer = MeanAbsoluteError()
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
            _all_scores.append(round(float(_s.drop("time").to_series().mean()), 1))

    mo.md(
        f"**Rolling windows evaluated**: {len(_all_scores)}\n\n"
        f"**MAE per window**: {_all_scores}\n\n"
        f"**Average MAE**: {sum(_all_scores) / len(_all_scores):.1f}"
        if _all_scores else
        f"**Rolling windows**: completed (no full truth windows for scoring)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Selective Group Observation

    In practice, data for different states may arrive at different times.
    Use `panel_group_names` to observe only the groups that have new data.
    """)
    return


@app.cell
def _(
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    mo,
    plot_forecast,
    tourism,
):
    _trip_cols = [c for c in tourism.columns if c.endswith("__trips")]
    _split = int(len(tourism) * 0.8)
    _y_tr = tourism.head(_split).select("time", *_trip_cols)
    _y_new = tourism[_split : _split + 4].select("time", *_trip_cols)

    _fc_sel = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        target_transformer=LagTransformer(lag=[1, 4]),
    )
    _fc_sel.fit(_y_tr, forecasting_horizon=4)

    # Observe only eastern states
    _eastern = ["new_south_wales", "victoria", "queensland"]
    _fc_sel.observe(_y_new, panel_group_names=_eastern)

    # Predict only eastern states
    _y_pred_east = _fc_sel.predict(
        forecasting_horizon=4,
        panel_group_names=_eastern,
    )
    plot_forecast(
        tourism.tail(len(tourism) - _split - 4).head(4).select("time", *_trip_cols),
        _y_pred_east,
        panel_group_names=_eastern,
        title="Selective Observation: Eastern States Only",
    )
    return


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
    - **Walmart forecasting**: See `examples/datasets/walmart_forecasting.py` for another panel forecasting example
    """)
    return


if __name__ == "__main__":
    app.run()
