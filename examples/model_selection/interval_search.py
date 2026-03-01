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
    "description": "Tune SplitConformalForecaster parameters (calibration_size, conformity_scorer) with GridSearchCV, then evaluate interval coverage and width separately.",
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

    Tune interval forecaster parameters using GridSearchCV.

    ## What You'll Learn

    - Searching over `calibration_size` and `conformity_scorer`
    - GridSearchCV uses **point predictions** internally for scoring
    - Evaluating interval quality separately after search
    - Balancing coverage vs interval width
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    from yohou.datasets import fetch_sunspot
    from yohou.interval import SplitConformalForecaster
    from yohou.metrics import (
        EmpiricalCoverage,
        MeanAbsoluteError,
        MeanIntervalWidth,
        RootMeanSquaredError,
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
        LagTransformer,
        MeanAbsoluteError,
        MeanIntervalWidth,
        PointReductionForecaster,
        Residual,
        Ridge,
        RootMeanSquaredError,
        SplitConformalForecaster,
        fetch_sunspot,
        pl,
        plot_forecast,
        plot_time_series,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We load the Sunspots dataset, aggregate to monthly resolution, and split
    into training and test sets. The test set length defines the forecasting
    horizon used throughout the search.
    """)


@app.cell
def _(fetch_sunspot, mo, pl, train_test_split):
    _raw = fetch_sunspot().frame
    ss = _raw.group_by_dynamic("time", every="1mo").agg(pl.col("sunspot_number").mean())
    y_train, _y_rest = train_test_split(ss, test_size=0.15, shuffle=False)
    y_test = _y_rest.head(24)
    horizon = len(y_test)

    mo.md(f"**Sunspots**: {len(ss)} observations\n\n**Train**: {len(y_train)}, **Test**: {len(y_test)}")
    return horizon, ss, y_test, y_train


@app.cell
def _(plot_time_series, ss):
    plot_time_series(ss, title="Monthly Sunspot Numbers")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Search over Calibration Parameters

    [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) scores using **point predictions** from
    `observe_predict`. We use point metrics (MAE, RMSE) for
    the search, then evaluate interval quality separately.
    """)


@app.cell
def _(
    AbsoluteResidual,
    ExpandingWindowSplitter,
    GridSearchCV,
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Residual,
    Ridge,
    RootMeanSquaredError,
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
    interval_gs = GridSearchCV(
        forecaster=_base,
        param_grid={
            "calibration_size": [30, 50, 80, 120],
            "conformity_scorer": [Residual(), AbsoluteResidual()],
        },
        scoring={
            "mae": MeanAbsoluteError(),
            "rmse": RootMeanSquaredError(),
        },
        refit="mae",
        cv=ExpandingWindowSplitter(n_splits=3),
    )
    interval_gs.fit(y_train, forecasting_horizon=horizon)
    return (interval_gs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Search Results

    The `cv_results_` dictionary contains per-parameter-combination scores
    for each metric. We extract the parameter columns and score columns into
    a table to compare configurations.
    """)


@app.cell
def _(interval_gs, mo, pl):
    _raw = interval_gs.cv_results_
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
def _(interval_gs, mo):
    mo.md(f"""
    **Best params**: {interval_gs.best_params_}\n\n"
        f"**Best MAE (negated)**: {interval_gs.best_score_:.4f}
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Evaluate Interval Quality

    After selecting the best model via point metrics, evaluate the
    interval predictions using coverage and width metrics.
    """)


@app.cell
def _(
    EmpiricalCoverage,
    MeanIntervalWidth,
    horizon,
    interval_gs,
    mo,
    y_test,
    y_train,
):
    coverage_rates = [0.9, 0.95]
    y_pred_interval = interval_gs.predict_interval(
        forecasting_horizon=horizon,
        coverage_rates=coverage_rates,
    )
    _cov_scorer_90 = EmpiricalCoverage(coverage_rates=[0.9])
    _cov_scorer_95 = EmpiricalCoverage(coverage_rates=[0.95])
    _width_scorer_90 = MeanIntervalWidth(coverage_rates=[0.9])
    _cov_scorer_90.fit(y_train)
    _cov_scorer_95.fit(y_train)
    _width_scorer_90.fit(y_train)

    _metrics = {
        "Coverage 90%": round(float(_cov_scorer_90.score(y_test, y_pred_interval)), 3),
        "Coverage 95%": round(float(_cov_scorer_95.score(y_test, y_pred_interval)), 3),
        "Mean Width 90%": round(float(_width_scorer_90.score(y_test, y_pred_interval)), 1),
    }
    mo.md("\n".join(f"- **{k}**: {v}" for k, v in _metrics.items()))
    return coverage_rates, y_pred_interval


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) renders the best interval forecaster's prediction bands
    alongside the actual test values. The shaded regions represent the
    prediction intervals at each requested coverage rate.
    """)


@app.cell
def _(coverage_rates, plot_forecast, y_pred_interval, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_interval,
        y_train=y_train,
        coverage_rates=coverage_rates,
        n_history=50,
        title="Best Interval Forecaster (GridSearchCV)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Point Forecast from Best Interval Model

    Interval forecasters also produce point predictions.
    """)


@app.cell
def _(horizon, interval_gs, plot_forecast, y_test, y_train):
    y_pred_point = interval_gs.predict(forecasting_horizon=horizon)
    plot_forecast(
        y_test,
        y_pred_point,
        y_train=y_train,
        n_history=50,
        title="Point Prediction from Best Interval Forecaster",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - GridSearchCV scores interval forecasters using **point predictions** via `observe_predict`
    - Use **point metrics** (MAE, RMSE) for the search scoring
    - Evaluate **interval quality** separately after search using `predict_interval`
    - Tune `calibration_size` and `conformity_scorer` for conformal forecasters
    - Larger calibration sets tend to improve coverage but reduce training data

    ## Next Steps

    - **Multi-metric search**: See [`examples/model_selection/multi_metric_search.py`](/examples/model_selection/multi_metric_search/)
    - **Optuna search**: See [`examples/model_selection/optuna_search.py`](/examples/model_selection/optuna_search/)
    - **Conformity scorers**: See [`examples/interval/conformal_conformity_scorers.py`](/examples/interval/conformal_conformity_scorers/)
    """)


if __name__ == "__main__":
    app.run()
