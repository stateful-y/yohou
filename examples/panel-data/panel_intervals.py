# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "How to Forecast Panel Prediction Intervals",
    "description": "Combine conformal and quantile regression intervals on panel data with per-group coverage analysis, calibration plots, and groupwise interval scoring.",
    "category": "how-to",
    "section": "panel-data",
    "companion": "/pages/how-to/interval-forecasting/",
    "api_references": ["fetch_kdd_cup", "inspect_panel", "train_test_split", "SplitConformalForecaster", "PointReductionForecaster", "LagTransformer", "plot_forecast", "IntervalReductionForecaster"],
}
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

    ## 1. Prepare Panel Data

    We load the KDD Cup 2018 air quality dataset with 3 Beijing stations,
    each monitoring 6 pollutants. Each station is a panel group with
    6 measurement members.
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_kdd_cup
    from yohou.interval import IntervalReductionForecaster, SplitConformalForecaster
    from yohou.metrics import EmpiricalCoverage, IntervalScore, MeanIntervalWidth
    from yohou.model_selection import train_test_split
    from yohou.plotting import plot_forecast, plot_score_per_vintage
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer
    from yohou.utils.panel import inspect_panel

    return (
        EmpiricalCoverage,
        IntervalReductionForecaster,
        IntervalScore,
        LagTransformer,
        MeanIntervalWidth,
        PointReductionForecaster,
        Ridge,
        SplitConformalForecaster,
        deepcopy,
        fetch_kdd_cup,
        inspect_panel,
        pl,
        plot_forecast,
        plot_score_per_vintage,
        train_test_split,
    )


@app.cell
def _(fetch_kdd_cup, inspect_panel, mo, train_test_split):
    _bunch = fetch_kdd_cup(n_groups=3)
    aq = _bunch.frame.drop_nulls().tail(300)
    _globals, groups = inspect_panel(aq)
    y_train, y_test = train_test_split(aq, test_size=0.15)
    horizon = len(y_test)
    coverage_rates = [0.9]

    mo.md(
        f"**Groups**: {list(groups.keys())}\n\n"
        f"**Train**: {len(y_train)} hours, **Test**: {len(y_test)} hours\n\n"
        f"**Coverage target**: {coverage_rates}"
    )
    return coverage_rates, groups, horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Split Conformal Forecaster on Panel Data

    [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/) calibrates per-group: each panel group
    gets its own conformal quantile based on its residual distribution.
    """)


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
            actual_transformer=LagTransformer(lag=[1, 4]),
        ),
        calibration_size=horizon + 5,
    )
    fc_conformal.fit(y_train, forecasting_horizon=horizon, coverage_rates=coverage_rates)
    y_pred_conf = fc_conformal.predict_interval(forecasting_horizon=horizon, coverage_rates=coverage_rates)
    _y_point = fc_conformal.predict(forecasting_horizon=horizon)
    y_pred_conf = y_pred_conf.hstack(_y_point.drop("time", "vintage_time"))
    return (y_pred_conf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) renders per-group prediction intervals. Use
    `groups` to select which groups to display.
    """)


@app.cell
def _(coverage_rates, plot_forecast, y_pred_conf, y_test, y_train):
    _groups = sorted({c.split("__")[0] for c in y_train.columns if "__" in c})
    plot_forecast(
        y_test,
        y_pred_conf,
        y_train=y_train,
        n_history=48,
        coverage_rates=coverage_rates,
        groups=_groups[:2],
        title="Split Conformal: Panel (90% Interval)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Interval Reduction Forecaster on Panel Data

    [`IntervalReductionForecaster`](/pages/api/generated/yohou.interval.reduction.IntervalReductionForecaster/) uses quantile regression to produce
    prediction intervals. Each panel group gets independent quantile
    estimates.
    """)


@app.cell
def _(IntervalReductionForecaster, coverage_rates, horizon, y_train):
    fc_interval = IntervalReductionForecaster()
    fc_interval.fit(y_train, forecasting_horizon=horizon, coverage_rates=coverage_rates)
    y_pred_interval = fc_interval.predict_interval(forecasting_horizon=horizon, coverage_rates=coverage_rates)
    return (y_pred_interval,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) for the quantile reduction approach shows the same
    groups, allowing visual comparison of interval width and shape.
    """)


@app.cell
def _(coverage_rates, plot_forecast, y_pred_interval, y_test, y_train):
    _groups = sorted({c.split("__")[0] for c in y_train.columns if "__" in c})
    plot_forecast(
        y_test,
        y_pred_interval,
        y_train=y_train,
        n_history=48,
        coverage_rates=coverage_rates,
        groups=_groups[:2],
        title="Interval Reduction: Panel (90% Interval)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Per-Group Coverage Analysis

    Check whether each group achieves the target coverage rate and
    compare interval widths.
    """)


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
        _cov_c = float(_cov_scorer.score(y_test, y_pred_conf, groups=[_state]))
        _cov_i = float(_cov_scorer.score(y_test, y_pred_interval, groups=[_state]))
        _w_c = float(_width_scorer.score(y_test, y_pred_conf, groups=[_state]))
        _w_i = float(_width_scorer.score(y_test, y_pred_interval, groups=[_state]))

        _rows.append({
            "Station": _state,
            "Conformal Coverage": round(_cov_c, 3),
            "Reduction Coverage": round(_cov_i, 3),
            "Conformal Width": round(_w_c, 1),
            "Reduction Width": round(_w_i, 1),
        })

    _results = pl.DataFrame(_rows)
    mo.ui.table(_results)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-vintage Scoring

    The `observe_predict_interval` method with `stride=1` produces one
    interval forecast per observation point, creating multiple *vintages*.
    Each vintage represents a different forecast origin, so you can analyse
    how interval quality evolves as the model absorbs more data.
    """)


@app.cell
def _(coverage_rates, deepcopy, fc_conformal, horizon, y_test):
    _vintage_model = deepcopy(fc_conformal)
    y_pred_vintages = _vintage_model.observe_predict_interval(
        y=y_test,
        stride=1,
        forecasting_horizon=horizon,
        coverage_rates=coverage_rates,
    )
    print(f"Vintages: {y_pred_vintages['vintage_time'].n_unique()}")
    y_pred_vintages.head(10)
    return (y_pred_vintages,)


@app.cell
def _(IntervalScore, y_train):
    vintage_scorer = IntervalScore()
    vintage_scorer.fit(y_train)
    return (vintage_scorer,)


@app.cell
def _(vintage_scorer, plot_score_per_vintage, y_pred_vintages, y_test):
    plot_score_per_vintage(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="Interval Score per Vintage",
        y_label="Interval Score",
        height=380,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Aggregation modes**: See [`examples/evaluation-search/aggregation_modes.py`](/examples/aggregation_modes/) for coveragewise scoring
    - **Conformity scorers**: See [`examples/evaluation-search/conformity_scorers.py`](/examples/conformity_scorers/)
    """)


if __name__ == "__main__":
    app.run()
