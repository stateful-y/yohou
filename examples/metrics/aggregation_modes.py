# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Scorer Aggregation Modes.

Demonstrates timewise, componentwise, groupwise, coveragewise, and all
aggregation modes for point and interval scorers on panel data.
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
    # Scorer Aggregation Modes

    Yohou scorers support multiple aggregation strategies that control
    how per-timestep, per-component, per-group, and per-coverage-rate
    errors are combined.

    ## What You'll Learn

    - `"timewise"`: aggregate over time, keep components/groups
    - `"componentwise"`: aggregate over components, keep time
    - `"groupwise"`: aggregate over panel groups, keep time
    - `"coveragewise"`: aggregate over coverage rates (interval scorers)
    - `"all"`: scalar output (default)
    - `panel_group_weight`: weight groups differently during aggregation
    """)

@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_dominick
    from yohou.interval import SplitConformalForecaster
    from yohou.metrics import (
        EmpiricalCoverage,
        MeanAbsoluteError,
        MeanIntervalWidth,
        RootMeanSquaredError,
    )
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer
    from yohou.utils.panel import inspect_locality

    return (
        EmpiricalCoverage,
        LagTransformer,
        MeanAbsoluteError,
        MeanIntervalWidth,
        PointReductionForecaster,
        Ridge,
        RootMeanSquaredError,
        SplitConformalForecaster,
        inspect_locality,
        fetch_dominick,
        pl,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data and Predictions
    """)

@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, inspect_locality, fetch_dominick, mo):
    _full = fetch_dominick().frame
    _selected = ["T7__profit", "T11__profit", "T12__profit", "T13__profit", "T15__profit", "T19__profit", "T22__profit", "T23__profit", "T24__profit"]
    store = _full.select("time", *_selected)
    _globals, groups = inspect_locality(store)
    _target_cols = [c for c in store.columns if c.endswith("__profit")]
    y = store.select("time", *_target_cols)

    _split = int(len(y) * 0.85)
    y_train = y.head(_split)
    y_test = y.tail(len(y) - _split)
    horizon = len(y_test)

    fc = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 7]),
    )
    fc.fit(y_train, forecasting_horizon=horizon)
    y_pred = fc.predict(forecasting_horizon=horizon)

    mo.md(
        f"**Groups**: {list(groups.keys())}\n\n"
        f"**Horizon**: {horizon} days"
    )
    return fc, groups, horizon, store, y, y_pred, y_test, y_train

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. `"all"`: Scalar Aggregation (Default)

    A single scalar value: all dimensions (time, components, groups) are
    reduced.
    """)

@app.cell
def _(MeanAbsoluteError, mo, y_pred, y_test, y_train):
    _scorer_all = MeanAbsoluteError(aggregation_method="all")
    _scorer_all.fit(y_train)
    _score_all = _scorer_all.score(y_test, y_pred)
    mo.md(
        f"**Aggregation `'all'`**:\n\n"
        f"MAE (all): {float(_score_all):.2f}"
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. `"timewise"`: Score Per Component

    Aggregate over time, producing one score per component/group column.
    Result has one row.
    """)

@app.cell
def _(MeanAbsoluteError, mo, y_pred, y_test, y_train):
    _scorer_tw = MeanAbsoluteError(aggregation_method="timewise")
    _scorer_tw.fit(y_train)
    score_timewise = _scorer_tw.score(y_test, y_pred)
    mo.md(
        f"**Aggregation `'timewise'`**:\n\n"
        f"Result shape: {score_timewise.shape}, one row, one col per group\n\n"
        f"Columns: {score_timewise.columns}"
    )
    return (score_timewise,)

@app.cell
def _(score_timewise):
    score_timewise

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. `"componentwise"`: Score Per Timestep

    Aggregate over components/groups for each timestep. Result has
    one row per timestep.
    """)

@app.cell
def _(MeanAbsoluteError, mo, y_pred, y_test, y_train):
    _scorer_cw = MeanAbsoluteError(aggregation_method="componentwise")
    _scorer_cw.fit(y_train)
    score_componentwise = _scorer_cw.score(y_test, y_pred)
    mo.md(
        f"**Aggregation `'componentwise'`**:\n\n"
        f"Result shape: {score_componentwise.shape}, one row per timestep\n\n"
        f"Columns: {score_componentwise.columns}"
    )
    return (score_componentwise,)

@app.cell
def _(score_componentwise):
    score_componentwise.head()

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. `"groupwise"`: Aggregate Within Groups

    With panel data, aggregate within each panel group separately.
    Groups that share the same prefix are combined.
    """)

@app.cell
def _(MeanAbsoluteError, mo, y_pred, y_test, y_train):
    _scorer_gw = MeanAbsoluteError(aggregation_method="groupwise")
    _scorer_gw.fit(y_train)
    score_groupwise = _scorer_gw.score(y_test, y_pred)
    mo.md(
        f"**Aggregation `'groupwise'`**:\n\n"
        f"Result shape: {score_groupwise.shape}\n\n"
        f"Columns: {score_groupwise.columns}"
    )
    return (score_groupwise,)

@app.cell
def _(score_groupwise):
    score_groupwise.head()

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. `panel_group_weight`: Weighted Group Aggregation

    Weight certain groups more heavily in the aggregation.
    """)

@app.cell
def _(MeanAbsoluteError, mo, y_pred, y_test, y_train):
    # Weight T1-T3 groups 3x more than others
    _weights = {
        "T1": 3.0,
        "T2": 3.0,
        "T3": 3.0,
        "T4": 1.0,
        "T5": 1.0,
        "T6": 1.0,
        "T7": 1.0,
        "T8": 1.0,
        "T9": 1.0,
    }
    _scorer_weighted = MeanAbsoluteError(
        aggregation_method="all", panel_group_weight=_weights
    )
    _scorer_unweighted = MeanAbsoluteError(aggregation_method="all")

    _scorer_weighted.fit(y_train)
    _scorer_unweighted.fit(y_train)
    _s_w = _scorer_weighted.score(y_test, y_pred)
    _s_u = _scorer_unweighted.score(y_test, y_pred)

    mo.md(
        f"**Unweighted MAE**: {float(_s_u):.2f}\n\n"
        f"**Weighted MAE** (store_1 x3): {float(_s_w):.2f}\n\n"
        "Weighted scores shift toward the heavily-weighted groups."
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Multiple Aggregation Modes

    Pass a list of modes to get partially-reduced output.
    """)

@app.cell
def _(MeanAbsoluteError, mo, y_pred, y_test, y_train):
    _scorer_multi = MeanAbsoluteError(aggregation_method=["timewise", "groupwise"])
    _scorer_multi.fit(y_train)
    _score_multi = _scorer_multi.score(y_test, y_pred)
    mo.md(
        f"**`['timewise', 'groupwise']`**: aggregated over time AND group\n\n"
        f"Result shape: {_score_multi.shape}\n\n"
        f"Columns: {_score_multi.columns}"
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. `"coveragewise"`: Interval Scorer Aggregation

    Interval scorers have an extra `"coveragewise"` dimension that
    aggregates across coverage rates.
    """)

@app.cell
def _(
    EmpiricalCoverage,
    LagTransformer,
    MeanIntervalWidth,
    PointReductionForecaster,
    Ridge,
    SplitConformalForecaster,
    horizon,
    mo,
    y_test,
    y_train,
):
    _coverage_rates = [0.8, 0.9, 0.95]

    _fc_conf = SplitConformalForecaster(
        point_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 7]),
        ),
        calibration_size=horizon + 10,
    )
    _fc_conf.fit(y_train, forecasting_horizon=horizon, coverage_rates=_coverage_rates)
    _y_pred_int = _fc_conf.predict_interval(
        forecasting_horizon=horizon, coverage_rates=_coverage_rates
    )

    _cov_all = EmpiricalCoverage(aggregation_method="all")
    _cov_per_rate = EmpiricalCoverage(aggregation_method=["timewise", "componentwise"])
    _width_per_rate = MeanIntervalWidth(aggregation_method=["timewise", "componentwise"])

    _cov_all.fit(y_train)
    _cov_per_rate.fit(y_train)
    _width_per_rate.fit(y_train)
    _s_all = _cov_all.score(y_test, _y_pred_int)
    _s_per_rate = _cov_per_rate.score(y_test, _y_pred_int)
    _s_w_per_rate = _width_per_rate.score(y_test, _y_pred_int)

    mo.md(
        f"**Coverage rates**: {_coverage_rates}\n\n"
        f"**`'all'`** coverage: {float(_s_all):.3f}\n\n"
        f"**Per-rate coverage**: {_s_per_rate}\n\n"
        f"**Per-rate width**: {_s_w_per_rate}"
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    | Mode | Aggregates Over | Result |
    |------|----------------|--------|
    | `"all"` | time + components + groups + coverage | scalar |
    | `"timewise"` | time | one row, cols = components/groups |
    | `"componentwise"` | components | rows = timesteps, single value col |
    | `"groupwise"` | groups | rows = timesteps, cols = unique groups |
    | `"coveragewise"` | coverage rates (interval only) | reduces coverage dimension |

    - **`panel_group_weight`**: Dictionary of group-name weights for biased aggregation
    - **Lists of modes**: Combine multiple aggregation dimensions
    - All modes work with both point and interval scorers

    ## Next Steps

    - **Time-weighted scoring**: See `examples/metrics/time_weighted_scoring.py`
    - **Point metrics**: See `examples/metrics/point_metrics.py`
    - **Interval metrics**: See `examples/metrics/interval_metrics.py`
    """)

if __name__ == "__main__":
    app.run()
