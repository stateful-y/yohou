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
    "title": "Interval Search",
    "description": "Tune interval forecaster parameters directly with interval metrics in GridSearchCV, including mixed point+interval multimetric search.",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hyperparameter Search for Interval Forecasters

    Tune interval forecaster parameters using interval metrics
    **directly** in [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/).

    ## What You'll Learn

    - Scoring with interval metrics (`IntervalScore`, `EmpiricalCoverage`) during CV
    - Mixed multimetric search: point + interval metrics together
    - How `GridSearchCV` automatically routes `coverage_rates` to the forecaster
    - Comparing calibration parameters with interval-aware selection
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_sunspot
    from yohou.interval import SplitConformalForecaster
    from yohou.metrics import (
        EmpiricalCoverage,
        IntervalScore,
        MeanAbsoluteError,
        MeanIntervalWidth,
    )
    from yohou.metrics.conformity import AbsoluteResidual, Residual
    from yohou.model_selection import GridSearchCV
    from yohou.model_selection.split import ExpandingWindowSplitter
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        AbsoluteResidual,
        EmpiricalCoverage,
        ExpandingWindowSplitter,
        GridSearchCV,
        IntervalScore,
        LagTransformer,
        MeanAbsoluteError,
        MeanIntervalWidth,
        PointReductionForecaster,
        Residual,
        Ridge,
        SplitConformalForecaster,
        fetch_sunspot,
        pl,
        plot_forecast,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We load the Sunspots dataset, aggregate to monthly, and split
    into training and test sets.
    """)


@app.cell
def _(fetch_sunspot, mo, pl):
    _raw = fetch_sunspot().frame
    ss = _raw.group_by_dynamic("time", every="1mo").agg(pl.col("sunspot_number").mean())
    n_test = 24
    y_train = ss.head(len(ss) - n_test)
    y_test = ss.tail(n_test)
    horizon = n_test

    mo.md(f"**Sunspots**: {len(ss)} rows | **Train**: {len(y_train)} | **Test**: {len(y_test)}")
    return horizon, ss, y_test, y_train


@app.cell
def _(plot_time_series, ss):
    plot_time_series(ss, title="Monthly Sunspot Numbers")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Search with Interval Metrics

    Pass an interval scorer like [`IntervalScore`](/pages/api/generated/yohou.metrics.interval.IntervalScore/)
    to `GridSearchCV`. The search automatically:

    1. Collects `coverage_rates` from all interval scorers
    2. Passes them to `forecaster.fit()` and `observe_predict_interval()`
    3. Scores each fold with interval predictions

    No manual `predict_interval` call is needed during the search.
    """)


@app.cell
def _(
    AbsoluteResidual,
    ExpandingWindowSplitter,
    GridSearchCV,
    IntervalScore,
    LagTransformer,
    PointReductionForecaster,
    Residual,
    Ridge,
    SplitConformalForecaster,
    horizon,
    y_train,
):
    _base = SplitConformalForecaster(
        point_forecaster=PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(lag=[1, 12]),
        ),
        calibration_size=50,
        conformity_scorer=Residual(),
    )
    gs_interval = GridSearchCV(
        forecaster=_base,
        param_grid={
            "calibration_size": [30, 50, 80, 120],
            "conformity_scorer": [Residual(), AbsoluteResidual()],
        },
        scoring=IntervalScore(coverage_rates=[0.9]),
        refit=True,
        cv=ExpandingWindowSplitter(n_splits=3),
    )
    gs_interval.fit(y_train, forecasting_horizon=horizon)
    return (gs_interval,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Results

    The search used `IntervalScore` to rank parameters. Lower (more negative)
    interval score means tighter intervals with good coverage.
    """)


@app.cell
def _(gs_interval, mo, pl):
    _raw = gs_interval.cv_results_
    _safe = {
        k: [
            str(v) if hasattr(v, "__class__") and not isinstance(v, (int, float, str, bool, type(None))) else v
            for v in vals
        ]
        if isinstance(vals, list)
        else vals
        for k, vals in _raw.items()
    }
    _results = pl.DataFrame(_safe)
    _cols = [c for c in _results.columns if "param_" in c or "mean_test" in c or "rank_test" in c]
    mo.ui.table(_results.select(_cols))


@app.cell
def _(gs_interval, mo):
    mo.md(f"""
    **Best params**: {gs_interval.best_params_}

    **Best IntervalScore (negated)**: {gs_interval.best_score_:.4f}
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Predict and Visualize

    The best forecaster was refitted on all training data with the
    `coverage_rates` collected from the scorer.
    """)


@app.cell
def _(gs_interval, horizon, plot_forecast, y_test, y_train):
    y_pred_interval = gs_interval.predict_interval(
        forecasting_horizon=horizon,
        coverage_rates=[0.9],
    )
    plot_forecast(
        y_test,
        y_pred_interval,
        y_train=y_train,
        coverage_rates=[0.9],
        n_history=50,
        title="Best Interval Forecaster (Interval Score Search)",
    )
    return (y_pred_interval,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Mixed Multimetric: Point + Interval

    You can combine point and interval scorers in a single search.
    `GridSearchCV` handles this transparently:

    - **Interval scorers** receive interval predictions (`_lower_`/`_upper_` columns)
    - **Point scorers** receive midpoint estimates derived from the tightest interval

    Use `refit` to select which metric determines the best model.
    """)


@app.cell
def _(
    AbsoluteResidual,
    EmpiricalCoverage,
    ExpandingWindowSplitter,
    GridSearchCV,
    IntervalScore,
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Residual,
    Ridge,
    SplitConformalForecaster,
    horizon,
    y_train,
):
    _base_mixed = SplitConformalForecaster(
        point_forecaster=PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(lag=[1, 12]),
        ),
        calibration_size=50,
        conformity_scorer=Residual(),
    )
    gs_mixed = GridSearchCV(
        forecaster=_base_mixed,
        param_grid={
            "calibration_size": [30, 50, 80],
            "conformity_scorer": [Residual(), AbsoluteResidual()],
        },
        scoring={
            "interval_score": IntervalScore(coverage_rates=[0.9]),
            "coverage": EmpiricalCoverage(coverage_rates=[0.9]),
            "mae": MeanAbsoluteError(),
        },
        refit="interval_score",
        cv=ExpandingWindowSplitter(n_splits=3),
    )
    gs_mixed.fit(y_train, forecasting_horizon=horizon)
    return (gs_mixed,)


@app.cell
def _(gs_mixed, mo, pl):
    _raw_m = gs_mixed.cv_results_
    _safe_m = {
        k: [
            str(v) if hasattr(v, "__class__") and not isinstance(v, (int, float, str, bool, type(None))) else v
            for v in vals
        ]
        if isinstance(vals, list)
        else vals
        for k, vals in _raw_m.items()
    }
    _results_m = pl.DataFrame(_safe_m)
    _cols_m = [c for c in _results_m.columns if "param_" in c or "mean_test" in c or "rank_test" in c]
    mo.ui.table(_results_m.select(_cols_m))


@app.cell
def _(gs_mixed, mo):
    mo.md(f"""
    **Best params (by interval score)**: {gs_mixed.best_params_}

    **Best IntervalScore**: {gs_mixed.best_score_:.4f}
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Evaluate Interval Width

    After selecting the best model, evaluate coverage and width on the
    held-out test set.
    """)


@app.cell
def _(
    EmpiricalCoverage,
    MeanIntervalWidth,
    gs_mixed,
    horizon,
    mo,
    y_test,
    y_train,
):
    y_pred_mixed = gs_mixed.predict_interval(
        forecasting_horizon=horizon,
        coverage_rates=[0.9],
    )
    _cov = EmpiricalCoverage(coverage_rates=[0.9])
    _width = MeanIntervalWidth(coverage_rates=[0.9])
    _cov.fit(y_train)
    _width.fit(y_train)

    mo.md(f"""
    - **Empirical coverage (90%)**: {float(_cov.score(y_test, y_pred_mixed)):.3f}
    - **Mean interval width**: {float(_width.score(y_test, y_pred_mixed)):.1f}
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Interval metrics work directly** in `GridSearchCV` - no workaround needed
    - `coverage_rates` are automatically collected from interval scorers and routed
    - **Mixed multimetric** search combines point and interval metrics in one call
    - Point scorers receive midpoint estimates `(lower + upper) / 2` from the tightest interval
    - Use `refit` to pick which metric selects the best configuration

    ## Next Steps

    - [`multi_metric_search.py`](/examples/model_selection/multi_metric_search/) for more on multi-metric strategies
    - [`optuna_search.py`](/examples/model_selection/optuna_search/) for Optuna-based search
    - [`conformal_conformity_scorers.py`](/examples/interval/conformal_conformity_scorers/) for conformity scorer details
    """)


if __name__ == "__main__":
    app.run()
