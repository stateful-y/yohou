"""Interval Forecaster Hyperparameter Search.

Demonstrates GridSearchCV with interval forecasters, tuning
calibration_size, conformity_scorer, and quantile levels.
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
    # Hyperparameter Search for Interval Forecasters

    Tune interval forecaster parameters using GridSearchCV.

    ## What You'll Learn

    - Searching over `calibration_size` and `conformity_scorer`
    - GridSearchCV uses **point predictions** internally for scoring
    - Evaluating interval quality separately after search
    - Balancing coverage vs interval width
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

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
    from yohou.plotting import plot_forecast
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
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data
    """)
    return


@app.cell
def _(fetch_sunspot, mo):
    ss = fetch_sunspot().frame
    _split = int(len(ss) * 0.85)
    y_train = ss.head(_split)
    y_test = ss.tail(len(ss) - _split)
    horizon = min(len(y_test), 50)

    mo.md(
        f"**Sunspots**: {len(ss)} observations\n\n"
        f"**Train**: {len(y_train)}, **Test**: {len(y_test)}"
    )
    return horizon, ss, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Search over Calibration Parameters

    `GridSearchCV` scores using **point predictions** from
    `observe_predict`. We use point metrics (MAE, RMSE) for
    the search, then evaluate interval quality separately.
    """)
    return


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
    """)
    return


@app.cell
def _(interval_gs, mo, pl):
    _raw = interval_gs.cv_results_
    _safe = {k: [str(v) if hasattr(v, '__class__') and not isinstance(v, (int, float, str, bool, type(None))) else v for v in vals] if isinstance(vals, list) else vals for k, vals in _raw.items()}
    _results = pl.DataFrame(_safe)
    _cols = [c for c in _results.columns if "param_" in c or "mean_test" in c or "rank_test" in c]
    mo.ui.table(_results.select(_cols))
    return


@app.cell
def _(interval_gs, mo):
    mo.md(
        f"**Best params**: {interval_gs.best_params_}\n\n"
        f"**Best MAE (negated)**: {interval_gs.best_score_:.4f}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Evaluate Interval Quality

    After selecting the best model via point metrics, evaluate the
    interval predictions using coverage and width metrics.
    """)
    return


@app.cell
def _(
    EmpiricalCoverage,
    MeanIntervalWidth,
    horizon,
    interval_gs,
    mo,
    pl,
    y_test,
    y_train,
):
    _coverage_rates = [0.9, 0.95]
    y_pred_interval = interval_gs.predict_interval(
        forecasting_horizon=horizon,
        coverage_rates=_coverage_rates,
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
    return (y_pred_interval,)


@app.cell
def _(plot_forecast, y_pred_interval, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_interval,
        y_train=y_train,
        n_history=50,
        title="Best Interval Forecaster (GridSearchCV)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Point Forecast from Best Interval Model

    Interval forecasters also produce point predictions.
    """)
    return


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
    return (y_pred_point,)


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

    - **Multi-metric search**: See `examples/model_selection/multi_metric_search.py`
    - **Optuna search**: See `examples/model_selection/optuna_search.py`
    - **Conformity scorers**: See `examples/interval/conformal_conformity_scorers.py`
    """)
    return


if __name__ == "__main__":
    app.run()
