"""Walmart Sales Forecasting.

Panel forecasting workflow for the Walmart Sales dataset: multi-target
forecasting (sales + ratings), per-branch evaluation, and observe-predict
with selective groups.
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
    # Walmart Sales Forecasting

    The Walmart Sales dataset contains daily total sales and customer ratings
    for 3 branches (A, B, C) over 89 days. With its short history, multiple
    targets per group, and daily frequency, it demonstrates a realistic
    **multi-target panel forecasting** scenario.

    ## What You'll Learn

    - Forecasting multiple targets (sales and ratings) across panel groups
    - Choosing which target columns to forecast
    - Per-branch evaluation and comparison
    - Rolling observe-predict on daily panel data
    - Handling short time series with appropriate lag selection

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

    from yohou.datasets import load_walmart_sales
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
        load_walmart_sales,
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
def _(inspect_locality, load_walmart_sales, mo):
    walmart = load_walmart_sales().fill_null(strategy="forward")
    _globals, groups = inspect_locality(walmart)

    mo.md(
        f"**Shape**: {walmart.shape}\n\n"
        f"**Panel groups**: {len(groups)} branches\n\n"
        f"**Groups**: {list(groups.keys())}\n\n"
        f"**Columns per group**: {list(groups.values())[0]}\n\n"
        f"**Time range**: {walmart['time'].min()} to {walmart['time'].max()}\n\n"
        f"Each branch has `total` (daily sales) and `rating` (customer rating) columns."
    )
    return groups, walmart


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Forecast Sales Only

    With only 89 daily observations, we keep lags short and forecast a
    modest horizon. First, let's forecast the `total` (sales) column
    for all branches.
    """)
    return


@app.cell
def _(mo, walmart):
    _sales_cols = [c for c in walmart.columns if c.endswith("__total")]
    _split = int(len(walmart) * 0.8)
    y_train_sales = walmart.head(_split).select("time", *_sales_cols)
    y_test_sales = walmart.tail(len(walmart) - _split).select("time", *_sales_cols)

    mo.md(
        f"**Sales columns**: {_sales_cols}\n\n"
        f"**Train**: {len(y_train_sales)} days, **Test**: {len(y_test_sales)} days"
    )
    return y_test_sales, y_train_sales


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, y_test_sales, y_train_sales):
    fc_sales = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 7]),
    )
    _horizon = len(y_test_sales)
    fc_sales.fit(y_train_sales, forecasting_horizon=_horizon)
    y_pred_sales = fc_sales.predict(forecasting_horizon=_horizon)
    return fc_sales, y_pred_sales


@app.cell
def _(plot_forecast, y_pred_sales, y_test_sales, y_train_sales):
    plot_forecast(
        y_test_sales,
        y_pred_sales,
        y_train=y_train_sales,
        n_history=30,
        panel_group_names=["branch_a", "branch_b", "branch_c"],
        title="Daily Sales Forecast: All Branches",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Forecast Ratings

    Now forecast customer ratings separately. This demonstrates that each
    target column can have quite different characteristics.
    """)
    return


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, plot_forecast, walmart):
    _rating_cols = [c for c in walmart.columns if c.endswith("__rating")]
    _split = int(len(walmart) * 0.8)
    _y_train_r = walmart.head(_split).select("time", *_rating_cols)
    _y_test_r = walmart.tail(len(walmart) - _split).select("time", *_rating_cols)

    _fc_rating = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 7]),
    )
    _horizon = len(_y_test_r)
    _fc_rating.fit(_y_train_r, forecasting_horizon=_horizon)
    _y_pred_r = _fc_rating.predict(forecasting_horizon=_horizon)

    plot_forecast(
        _y_test_r,
        _y_pred_r,
        y_train=_y_train_r,
        n_history=30,
        panel_group_names=["branch_a", "branch_b", "branch_c"],
        title="Customer Rating Forecast: All Branches",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Per-Branch Scoring
    """)
    return


@app.cell
def _(MeanAbsoluteError, groups, mo, y_pred_sales, y_test_sales, y_train_sales):
    _scores = {}
    for _group in groups:
        _scorer = MeanAbsoluteError(panel_group_names=[_group])
        _scorer.fit(y_train_sales)
        _s = _scorer.score(y_test_sales, y_pred_sales)
        _scores[_group] = round(float(_s), 1)

    _lines = [f"| {name} | {score} |" for name, score in sorted(_scores.items(), key=lambda x: x[1])]
    mo.md(
        "**Sales MAE per Branch**\n\n"
        "| Branch | MAE |\n|--------|-----|\n" + "\n".join(_lines)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Rolling Observe-Predict

    With only 89 days, the rolling window is short. We roll forward in
    weekly steps, observing and predicting 7 days at a time.
    """)
    return


@app.cell
def _(LagTransformer, MeanAbsoluteError, PointReductionForecaster, Ridge, mo, walmart):
    _sales_cols = [c for c in walmart.columns if c.endswith("__total")]
    _train_end = int(len(walmart) * 0.5)
    _y_early = walmart.head(_train_end).select("time", *_sales_cols)
    _y_rest = walmart.tail(len(walmart) - _train_end).select("time", *_sales_cols)

    _fc_roll = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 7]),
    )
    _step = 7
    _fc_roll.fit(_y_early, forecasting_horizon=_step)

    _scorer = MeanAbsoluteError()
    _scorer.fit(_y_early)
    _window_scores = []

    for _i in range(0, len(_y_rest) - _step + 1, _step):
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
    ## 6. Selective Branch Observation

    Simulate a scenario where Branch A data arrives first.
    """)
    return


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, mo, walmart):
    _sales_cols = [c for c in walmart.columns if c.endswith("__total")]
    _split = int(len(walmart) * 0.8)
    _y_tr = walmart.head(_split).select("time", *_sales_cols)
    _y_new = walmart[_split : _split + 7].select("time", *_sales_cols)

    _fc_sel = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 7]),
    )
    _fc_sel.fit(_y_tr, forecasting_horizon=7)

    # Observe only Branch A
    _fc_sel.observe(_y_new, panel_group_names=["branch_a"])
    _time_a = _fc_sel.observed_time_["branch_a"]

    # Branch B/C still at old position (not in observed_time_ after selective observe)
    _time_b = "unchanged (not yet observed)"

    mo.md(
        f"**Branch A observed time**: `{_time_a}`\n\n"
        f"**Branch B observed time**: `{_time_b}`\n\n"
        f"Branches B and C remain at their previous position until new data arrives."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Multi-target panel forecasting**: Forecast different target columns (sales, ratings) independently
    - **Short time series**: Use modest lags (`[1, 7]` for daily) to avoid overfitting on limited data
    - **Per-branch scoring** identifies which locations are harder to predict
    - **Rolling observe-predict** works the same way as with standard data but across all groups
    - **Selective observation** supports asynchronous data arrival (Branch A before B/C)
    - With 89 days and 3 branches x 2 targets, the dataset totals only 534 data points per target: regularisation (Ridge) helps prevent overfitting

    ## Next Steps

    - **Dataset exploration**: See `examples/datasets/walmart_sales.py` for EDA on this dataset
    - **Australian Tourism**: See `examples/datasets/australian_tourism_forecasting.py` for quarterly panel forecasting
    - **Panel concepts**: See `examples/panel_data.py` for the `__` naming convention
    """)
    return


if __name__ == "__main__":
    app.run()
