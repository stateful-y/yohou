"""Optuna Hyperparameter Optimisation.

Demonstrates yohou-optuna integration for Bayesian hyperparameter search
with OptunaSearchCV.
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

        await micropip.install(["plotly", "scikit-learn", "yohou", "optuna"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optuna Hyperparameter Optimisation

    Bayesian hyperparameter search using `OptunaSearchCV` from the
    `yohou-optuna` workspace package.

    ## What You'll Learn

    - Setting up `OptunaSearchCV` with parameter distributions
    - Defining continuous, integer, and categorical distributions
    - Running a study with time series cross-validation
    - Inspecting optimisation history and trials
    - Comparing with GridSearchCV

    **Requires**: `yohou-optuna` package (install with `uv pip install yohou-optuna`)
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl
    from optuna.distributions import CategoricalDistribution, FloatDistribution
    from sklearn.linear_model import ElasticNet, Ridge

    from yohou.datasets import load_air_passengers
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection.split import ExpandingWindowSplitter
    from yohou.plotting import plot_cv_results_scatter, plot_forecast
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer
    from yohou_optuna import OptunaSearchCV, Sampler

    return (
        CategoricalDistribution,
        ElasticNet,
        ExpandingWindowSplitter,
        FloatDistribution,
        LagTransformer,
        MeanAbsoluteError,
        OptunaSearchCV,
        PointReductionForecaster,
        Ridge,
        Sampler,
        load_air_passengers,
        pl,
        plot_cv_results_scatter,
        plot_forecast,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data
    """)
    return


@app.cell
def _(load_air_passengers, mo):
    ap = load_air_passengers()
    _split = int(len(ap) * 0.85)
    y_train = ap.head(_split).select("time", "Passengers")
    y_test = ap.tail(len(ap) - _split).select("time", "Passengers")
    horizon = len(y_test)

    mo.md(f"**Train**: {len(y_train)} months, **Test**: {len(y_test)} months")
    return ap, horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Define Parameter Distributions

    Use `optuna.distributions` to define the search space. Unlike
    GridSearchCV's exhaustive grid, Optuna samples intelligently
    using a Tree-structured Parzen Estimator (TPE) by default.
    """)
    return


@app.cell
def _(CategoricalDistribution, FloatDistribution):
    param_distributions = {
        "estimator__alpha": FloatDistribution(0.001, 50.0, log=True),
        "estimator__l1_ratio": FloatDistribution(0.0, 1.0),
    }
    param_distributions_ridge = {
        "estimator__alpha": FloatDistribution(0.001, 50.0, log=True),
    }
    lag_distributions = {
        "feature_transformer__lag": CategoricalDistribution(
            [[1, 12], [1, 6, 12], [1, 3, 6, 12]]
        ),
    }
    return lag_distributions, param_distributions, param_distributions_ridge


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. OptunaSearchCV with TPE Sampler
    """)
    return


@app.cell
def _(
    ElasticNet,
    ExpandingWindowSplitter,
    LagTransformer,
    MeanAbsoluteError,
    OptunaSearchCV,
    PointReductionForecaster,
    Sampler,
    horizon,
    param_distributions,
    y_train,
):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    optuna_search = OptunaSearchCV(
        forecaster=PointReductionForecaster(
            estimator=ElasticNet(),
            feature_transformer=LagTransformer(lag=[1, 12]),
        ),
        param_distributions=param_distributions,
        scoring=MeanAbsoluteError(),
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
        n_trials=15,
        cv=ExpandingWindowSplitter(n_splits=3),
        refit=True,
    )
    optuna_search.fit(y_train, forecasting_horizon=horizon)
    return optuna, optuna_search


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Inspect Results
    """)
    return


@app.cell
def _(mo, optuna_search):
    mo.md(
        f"**Best params**: {optuna_search.best_params_}\n\n"
        f"**Best score (negated MAE)**: {optuna_search.best_score_:.4f}\n\n"
        f"**Number of trials**: {len(optuna_search.cv_results_['params'])}"
    )
    return


@app.cell
def _(plot_cv_results_scatter, optuna_search):
    plot_cv_results_scatter(optuna_search.cv_results_, "estimator__alpha")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Trial History

    Each trial records the parameters sampled and resulting score.
    """)
    return


@app.cell
def _(mo, optuna_search, pl):
    _trials = []
    for _i, _params in enumerate(optuna_search.cv_results_["params"]):
        _trials.append({
            "Trial": _i,
            "alpha": round(_params.get("estimator__alpha", 0), 4),
            "l1_ratio": round(_params.get("estimator__l1_ratio", 0), 4),
            "Mean Test Score": round(
                float(optuna_search.cv_results_["mean_test_score"][_i]), 4
            ),
        })
    mo.ui.table(pl.DataFrame(_trials))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Best Model Forecast
    """)
    return


@app.cell
def _(horizon, optuna_search, plot_forecast, y_test, y_train):
    y_pred_optuna = optuna_search.predict(forecasting_horizon=horizon)
    plot_forecast(
        y_test,
        y_pred_optuna,
        y_train=y_train,
        n_history=36,
        title="Best Forecaster (OptunaSearchCV)",
    )
    return (y_pred_optuna,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Ridge with Lag Search

    Combine continuous hyperparameter search with categorical
    choices like lag configurations.
    """)
    return


@app.cell
def _(
    ExpandingWindowSplitter,
    LagTransformer,
    MeanAbsoluteError,
    OptunaSearchCV,
    PointReductionForecaster,
    Ridge,
    Sampler,
    horizon,
    lag_distributions,
    mo,
    optuna,
    param_distributions_ridge,
    y_train,
):
    _combined = {**param_distributions_ridge, **lag_distributions}
    optuna_lag_search = OptunaSearchCV(
        forecaster=PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(lag=[1, 12]),
        ),
        param_distributions=_combined,
        scoring=MeanAbsoluteError(),
        sampler=Sampler(sampler=optuna.samplers.TPESampler, seed=42),
        n_trials=10,
        cv=ExpandingWindowSplitter(n_splits=3),
        refit=True,
    )
    optuna_lag_search.fit(y_train, forecasting_horizon=horizon)

    mo.md(
        f"**Best params**: {optuna_lag_search.best_params_}\n\n"
        f"**Best score**: {optuna_lag_search.best_score_:.4f}"
    )
    return (optuna_lag_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - `OptunaSearchCV` uses Bayesian optimization (TPE) for efficient search
    - Define distributions with `FloatDistribution`, `IntDistribution`, `CategoricalDistribution`
    - `Sampler` wraps Optuna samplers with reproducible seeds
    - Supports the same multi-metric scoring as GridSearchCV
    - `n_trials` controls search budget (more trials = better exploration)
    - Mix continuous and categorical distributions in one search

    ## Next Steps

    - **Multi-metric search**: See `examples/model_selection/multi_metric_search.py`
    - **Interval search**: See `examples/model_selection/interval_search.py`
    - **Hyperparameter search basics**: See `examples/model_selection/hyperparameter_search.py`
    """)
    return


if __name__ == "__main__":
    app.run()
