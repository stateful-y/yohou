# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
#     "yohou-nixtla",
# ]
# ///
"""Nixtla Statistical and Neural Forecasters.

Demonstrates yohou-nixtla integration across statistical and neural forecaster
families with side-by-side comparison on tourism monthly data.
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
    # Nixtla Statistical and Neural Forecasters

    The `yohou-nixtla` package wraps forecasters from Nixtla's
    `statsforecast` and `neuralforecast` libraries into yohou's sklearn-compatible API.

    ## What You'll Learn

    - **Statistical**: AutoARIMA, AutoETS, HoltWinters, Theta family
    - **Neural**: NBEATS, NHITS, MLP (requires PyTorch)
    - Side-by-side comparison across families
    - Using Nixtla forecasters with GridSearchCV

    > **Note**: For ML-based reduction forecasters, use `yohou.point.PointReductionForecaster`
    > with any scikit-learn regressor directly.

    **Requires**: `yohou-nixtla` package (install with `uv pip install yohou-nixtla`)
    """)

@app.cell(hide_code=True)
def _():
    import polars as pl

    from yohou.datasets import fetch_tourism_monthly
    from yohou.metrics import MeanAbsoluteError, RootMeanSquaredError
    from yohou.plotting import plot_forecast

    return (
        MeanAbsoluteError,
        RootMeanSquaredError,
        fetch_tourism_monthly,
        pl,
        plot_forecast,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data
    """)

@app.cell
def _(fetch_tourism_monthly, mo):
    tourism = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    _split = int(len(tourism) * 0.85)
    y_train = tourism.head(_split)
    y_test = tourism.tail(len(tourism) - _split)
    horizon = len(y_test)

    mo.md(f"**Train**: {len(y_train)} months, **Test**: {len(y_test)} months")
    return tourism, horizon, y_test, y_train

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Statistical Forecasters

    Wrappers around `statsforecast` models. Each model is fitted per
    series with classical statistical methods.

    Key parameters:
    - `season_length`: Seasonal period (12 for monthly data)
    - `freq`: Frequency string (auto-inferred if not set)
    """)

@app.cell
def _():
    from yohou_nixtla.stats import (
        AutoARIMAForecaster,
        AutoETSForecaster,
        AutoThetaForecaster,
        HoltWintersForecaster,
        SeasonalNaiveForecaster,
    )

    return (
        AutoARIMAForecaster,
        AutoETSForecaster,
        AutoThetaForecaster,
        HoltWintersForecaster,
        SeasonalNaiveForecaster,
    )

@app.cell
def _(
    AutoARIMAForecaster,
    AutoETSForecaster,
    AutoThetaForecaster,
    HoltWintersForecaster,
    SeasonalNaiveForecaster,
    horizon,
    y_train,
):
    stats_models = {
        "SeasonalNaive": SeasonalNaiveForecaster(season_length=12),
        "AutoARIMA": AutoARIMAForecaster(season_length=12),
        "AutoETS": AutoETSForecaster(season_length=12),
        "AutoTheta": AutoThetaForecaster(season_length=12),
        "HoltWinters": HoltWintersForecaster(season_length=12),
    }
    stats_preds = {}
    for _name, _fc in stats_models.items():
        _fc.fit(y_train, forecasting_horizon=horizon)
        stats_preds[_name] = _fc.predict(forecasting_horizon=horizon)
    return stats_models, stats_preds

@app.cell
def _(MeanAbsoluteError, mo, pl, stats_preds, y_test, y_train):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _rows = []
    for _name, _pred in stats_preds.items():
        _mae = float(_scorer.score(y_test, _pred))
        _rows.append({"Model": _name, "Family": "Stats", "MAE": round(_mae, 2)})
    stats_results = pl.DataFrame(_rows)
    mo.ui.table(stats_results)
    return (stats_results,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Neural Forecasters

    Wrappers around `neuralforecast` models. These use deep learning
    and require PyTorch.

    Key parameters:
    - `input_size`: Lookback window length
    - `max_steps`: Training iterations (keep low for demos)
    """)

@app.cell
def _():
    from yohou_nixtla.neural import NBEATSForecaster, NHITSForecaster

    return NBEATSForecaster, NHITSForecaster

@app.cell
def _(NBEATSForecaster, NHITSForecaster, horizon, y_train):
    neural_models = {
        "NBEATS": NBEATSForecaster(input_size=24, max_steps=50),
        "NHITS": NHITSForecaster(input_size=24, max_steps=50),
    }
    neural_preds = {}
    for _name, _fc in neural_models.items():
        _fc.fit(y_train, forecasting_horizon=horizon)
        neural_preds[_name] = _fc.predict(forecasting_horizon=horizon)
    return neural_models, neural_preds

@app.cell
def _(MeanAbsoluteError, mo, neural_preds, pl, y_test, y_train):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _rows = []
    for _name, _pred in neural_preds.items():
        _mae = float(_scorer.score(y_test, _pred))
        _rows.append({"Model": _name, "Family": "Neural", "MAE": round(_mae, 2)})
    neural_results = pl.DataFrame(_rows)
    mo.ui.table(neural_results)
    return (neural_results,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Cross-Family Comparison
    """)

@app.cell
def _(mo, neural_results, pl, stats_results):
    all_results = pl.concat([stats_results, neural_results])
    all_results_sorted = all_results.sort("MAE")
    mo.ui.table(all_results_sorted)
    return all_results, all_results_sorted

@app.cell
def _(neural_preds, plot_forecast, stats_preds, y_test, y_train):
    # Show best from each family
    _best_stats = min(stats_preds, key=lambda k: float(stats_preds[k].drop("time", "observed_time").to_series().mean()))
    _best_neural = min(neural_preds, key=lambda k: float(neural_preds[k].drop("time", "observed_time").to_series().mean()))
    plot_forecast(
        y_test,
        stats_preds[_best_stats],
        y_train=y_train,
        n_history=36,
        title=f"Best Stats: {_best_stats}",
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Nixtla with GridSearchCV

    All Nixtla forecasters work with yohou's GridSearchCV.
    """)

@app.cell
def _(
    AutoARIMAForecaster,
    MeanAbsoluteError,
    horizon,
    mo,
    y_train,
):
    from yohou.model_selection import GridSearchCV
    from yohou.model_selection.split import ExpandingWindowSplitter

    _gs = GridSearchCV(
        forecaster=AutoARIMAForecaster(season_length=12),
        param_grid={
            "season_length": [6, 12],
        },
        scoring=MeanAbsoluteError(),
        cv=ExpandingWindowSplitter(n_splits=3),
        refit=True,
    )
    _gs.fit(y_train, forecasting_horizon=horizon)
    mo.md(
        f"**Best `season_length`**: {_gs.best_params_['season_length']}\n\n"
        f"**Best MAE (negated)**: {_gs.best_score_:.4f}"
    )
    return ExpandingWindowSplitter, GridSearchCV

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - `yohou-nixtla` provides a unified API for statistical and neural forecasters
    - **Stats** (10): Classical methods, fast, no feature engineering needed
    - **Neural** (5): Deep learning models, require PyTorch, slow but powerful
    - For ML-based reduction, use `yohou.point.PointReductionForecaster` with any sklearn regressor
    - All Nixtla forecasters are **point-only** (`forecaster_type="point"`)
    - `freq` is auto-inferred from data when not specified
    - Works with `GridSearchCV`, `RandomizedSearchCV`, and `OptunaSearchCV`

    ## Next Steps

    - **Nixtla + panel data**: See `examples/model_selection/nixtla_panel.py`
    - **Optuna search**: See `examples/model_selection/optuna_search.py`
    - **Hyperparameter search**: See `examples/model_selection/hyperparameter_search.py`
    """)

if __name__ == "__main__":
    app.run()
