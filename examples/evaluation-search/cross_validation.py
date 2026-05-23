# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.19.11"
__gallery__ = {
    "title": "Cross-Validation for Time Series",
    "description": "Evaluate forecasters with cross_val_score, cross_validate, and cross_val_predict using temporal splitters.",
    "category": "how-to",
    "section": "evaluation-search",
    "companion": "/pages/how-to/evaluate-forecast-accuracy/",
    "api_references": ["cross_validate", "cross_val_score", "cross_val_predict", "ExpandingWindowSplitter"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cross-Validation for Time Series

    Yohou provides three functions for cross-validated evaluation of forecasters:

    - [`cross_val_score`](/pages/api/generated/yohou.model_selection.validation.cross_val_score/): returns a DataFrame with `split` and `score` columns (simplest entry point).
    - [`cross_validate`](/pages/api/generated/yohou.model_selection.validation.cross_validate/): returns a DataFrame with scores and timings, or a dict when extra outputs are requested.
    - [`cross_val_predict`](/pages/api/generated/yohou.model_selection.validation.cross_val_predict/): returns out-of-fold predictions as a DataFrame.

    **Prerequisites:** Familiarity with splitters (see `cv_splitters.py`) and scorers (see `point_metrics.py`).
    """)


@app.cell(hide_code=True)
def _():
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_electricity_demand
    from yohou.metrics import MeanAbsoluteError, RootMeanSquaredError
    from yohou.model_selection import (
        ExpandingWindowSplitter,
        cross_val_predict,
        cross_val_score,
        cross_validate,
    )
    from yohou.point import PointReductionForecaster, SeasonalNaive

    return (
        ExpandingWindowSplitter,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        RootMeanSquaredError,
        SeasonalNaive,
        cross_val_predict,
        cross_val_score,
        cross_validate,
        fetch_electricity_demand,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Setup

    Load the dataset and define the forecaster and cross-validation splitter used throughout.
    """)


@app.cell
def _(ExpandingWindowSplitter, PointReductionForecaster, Ridge, fetch_electricity_demand):
    data = fetch_electricity_demand()
    y = data.frame.select("time", "vic__demand").drop_nulls().head(3000)

    forecaster = PointReductionForecaster(estimator=Ridge())
    cv = ExpandingWindowSplitter(n_splits=5, test_size=14)
    fh = 14

    print(f"Series length: {len(y)}, Forecasting horizon: {fh}")
    return cv, fh, forecaster, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Quick Evaluation with `cross_val_score`

    `cross_val_score` is the simplest way to get per-fold scores. It returns a DataFrame with `split` and `score` columns.
    """)


@app.cell
def _(MeanAbsoluteError, cross_val_score, cv, fh, forecaster, y):
    scores = cross_val_score(
        forecaster,
        y,
        scoring=MeanAbsoluteError(),
        cv=cv,
        forecasting_horizon=fh,
    )
    print(scores)
    print(f"Mean: {scores['score'].mean():.2f} (+/- {scores['score'].std():.2f})")
    return (scores,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Detailed Results with `cross_validate`

    `cross_validate` returns a DataFrame with scores, fit times, and score times.
    With a single scorer the score column is `test_score`.
    """)


@app.cell
def _(MeanAbsoluteError, cross_validate, cv, fh, forecaster, y):
    single_results = cross_validate(
        forecaster,
        y,
        scoring=MeanAbsoluteError(),
        cv=cv,
        forecasting_horizon=fh,
    )
    print(f"Columns: {single_results.columns}")
    print(single_results)
    return (single_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Multi-Metric Evaluation

    Pass a dictionary of scorers to evaluate multiple metrics at once.
    The DataFrame columns follow the pattern `test_{name}` for each scorer name.
    """)


@app.cell
def _(MeanAbsoluteError, RootMeanSquaredError, cross_validate, cv, fh, forecaster, y):
    multi_results = cross_validate(
        forecaster,
        y,
        scoring={
            "mae": MeanAbsoluteError(),
            "rmse": RootMeanSquaredError(),
        },
        cv=cv,
        forecasting_horizon=fh,
    )
    print(f"Columns: {multi_results.columns}")
    print(multi_results)
    return (multi_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Baseline Comparison with `cross_val_score`

    Compare a model against a naive baseline to confirm it adds value.
    """)


@app.cell
def _(MeanAbsoluteError, SeasonalNaive, cross_val_score, cv, fh, forecaster, y):
    model_scores = cross_val_score(
        forecaster,
        y,
        scoring=MeanAbsoluteError(),
        cv=cv,
        forecasting_horizon=fh,
    )

    baseline_scores = cross_val_score(
        SeasonalNaive(seasonality=7),
        y,
        scoring=MeanAbsoluteError(),
        cv=cv,
        forecasting_horizon=fh,
    )

    print(f"Model MAE:    {model_scores['score'].mean():.2f} (+/- {model_scores['score'].std():.2f})")
    print(f"Baseline MAE: {baseline_scores['score'].mean():.2f} (+/- {baseline_scores['score'].std():.2f})")
    return baseline_scores, model_scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Out-of-Fold Predictions with `cross_val_predict`

    `cross_val_predict` returns a DataFrame of predictions from each fold,
    with a `split` column identifying which fold produced each prediction.
    This is useful for visualization and model blending (stacking).
    """)


@app.cell
def _(cross_val_predict, cv, fh, forecaster, y):
    predictions = cross_val_predict(
        forecaster,
        y,
        cv=cv,
        forecasting_horizon=fh,
    )
    print(f"Predictions shape: {predictions.shape}")
    print(f"Columns: {predictions.columns}")
    predictions.head(10)
    return (predictions,)


if __name__ == "__main__":
    app.run()
