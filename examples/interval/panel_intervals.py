# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Panel Prediction Intervals.

Demonstrates SplitConformalForecaster and IntervalReductionForecaster
on panel time series with per-group calibration and coverage analysis.
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
    # Panel Prediction Intervals

    Interval forecasters automatically produce prediction intervals for
    each panel group. This notebook demonstrates conformal and quantile
    regression approaches on panel data.

    ## What You'll Learn

    - `SplitConformalForecaster` with per-group calibration
    - `IntervalReductionForecaster` on panel data
    - Per-group coverage analysis
    - Comparing interval width across groups
    """)
    return

@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_tourism_quarterly
    from yohou.interval import IntervalReductionForecaster, SplitConformalForecaster
    from yohou.metrics import EmpiricalCoverage, MeanIntervalWidth
    from yohou.plotting import plot_forecast
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer
    from yohou.utils.panel import inspect_locality

    return (
        EmpiricalCoverage,
        IntervalReductionForecaster,
        LagTransformer,
        MeanIntervalWidth,
        PointReductionForecaster,
        Ridge,
        SplitConformalForecaster,
        inspect_locality,
        fetch_tourism_quarterly,
        pl,
        plot_forecast,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Panel Data
    """)
    return

@app.cell
def _(inspect_locality, fetch_tourism_quarterly, mo):
    _bunch = fetch_tourism_quarterly()
    # Select T3-T10 (same length series, 88 rows after drop_nulls)
    _selected = [f"T{i}__tourists" for i in range(3, 11)]
    tourism = _bunch.frame.select("time", *_selected).drop_nulls()
    _globals, groups = inspect_locality(tourism)
    _split = int(len(tourism) * 0.8)
    y_train = tourism.head(_split)
    y_test = tourism.tail(len(tourism) - _split)
    horizon = len(y_test)
    coverage_rates = [0.9]

    mo.md(
        f"**Groups**: {list(groups.keys())}\n\n"
        f"**Train**: {len(y_train)} quarters, **Test**: {len(y_test)} quarters\n\n"
        f"**Coverage target**: {coverage_rates}"
    )
    return coverage_rates, groups, horizon, tourism, y_test, y_train

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Split Conformal Forecaster on Panel Data

    `SplitConformalForecaster` calibrates per-group: each panel group
    gets its own conformal quantile based on its residual distribution.
    """)
    return

@app.cell
def _(
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SplitConformalForecaster,
    coverage_rates,
    horizon,
    y_train,
):
    fc_conformal = SplitConformalForecaster(
        point_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 4]),
        ),
        calibration_size=horizon + 5,
    )
    fc_conformal.fit(y_train, forecasting_horizon=horizon, coverage_rates=coverage_rates)
    y_pred_conf = fc_conformal.predict_interval(
        forecasting_horizon=horizon, coverage_rates=coverage_rates
    )
    return fc_conformal, y_pred_conf

@app.cell
def _(coverage_rates, plot_forecast, y_pred_conf, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_conf,
        y_train=y_train,
        n_history=8,
        coverage_rates=coverage_rates,
        panel_group_names=["T3", "T4", "T5"],
        title="Split Conformal: Panel (90% Interval)",
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Interval Reduction Forecaster on Panel Data

    `IntervalReductionForecaster` uses quantile regression to produce
    prediction intervals. Each panel group gets independent quantile
    estimates.
    """)
    return

@app.cell
def _(IntervalReductionForecaster, coverage_rates, horizon, y_train):
    fc_interval = IntervalReductionForecaster()
    fc_interval.fit(y_train, forecasting_horizon=horizon, coverage_rates=coverage_rates)
    y_pred_interval = fc_interval.predict_interval(
        forecasting_horizon=horizon, coverage_rates=coverage_rates
    )
    return fc_interval, y_pred_interval

@app.cell
def _(coverage_rates, plot_forecast, y_pred_interval, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_interval,
        y_train=y_train,
        n_history=8,
        coverage_rates=coverage_rates,
        panel_group_names=["T3", "T4", "T5"],
        title="Interval Reduction: Panel (90% Interval)",
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Per-Group Coverage Analysis

    Check whether each group achieves the target coverage rate and
    compare interval widths.
    """)
    return

@app.cell
def _(
    EmpiricalCoverage,
    MeanIntervalWidth,
    groups,
    mo,
    pl,
    y_pred_conf,
    y_pred_interval,
    y_test,
    y_train,
):
    _cov_scorer = EmpiricalCoverage()
    _width_scorer = MeanIntervalWidth()

    _cov_scorer.fit(y_train)
    _width_scorer.fit(y_train)

    _rows = []
    for _state in sorted(groups.keys()):
        _cov_c = float(_cov_scorer.score(y_test, y_pred_conf, panel_group_names=[_state]))
        _cov_i = float(_cov_scorer.score(y_test, y_pred_interval, panel_group_names=[_state]))
        _w_c = float(_width_scorer.score(y_test, y_pred_conf, panel_group_names=[_state]))
        _w_i = float(_width_scorer.score(y_test, y_pred_interval, panel_group_names=[_state]))

        _rows.append({
            "Group": _state,
            "Conformal Coverage": round(_cov_c, 3),
            "Reduction Coverage": round(_cov_i, 3),
            "Conformal Width": round(_w_c, 1),
            "Reduction Width": round(_w_i, 1),
        })

    _results = pl.DataFrame(_rows)
    mo.ui.table(_results)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Per-group calibration**: Both conformal and quantile regression methods calibrate independently per group
    - **Coverage varies by group**: Some groups may have wider/narrower intervals depending on their variance
    - **`SplitConformalForecaster`** wraps any point forecaster with conformal calibration
    - **`IntervalReductionForecaster`** uses quantile regression directly
    - Use `EmpiricalCoverage` and `MeanIntervalWidth` for per-group coverage/width analysis

    ## Next Steps

    - **Aggregation modes**: See `examples/metrics/aggregation_modes.py` for coveragewise scoring
    - **Conformal variations**: See `examples/interval/conformal_forecasting.py`
    - **Conformity scorers**: See `examples/metrics/conformity_scorers.py`
    """)
    return

if __name__ == "__main__":
    app.run()
