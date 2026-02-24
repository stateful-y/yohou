# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Point Metrics for Forecast Evaluation.

Demonstrates all point scoring metrics and aggregation methods.
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
    # Point Metrics for Forecast Evaluation

    Yohou provides a comprehensive set of point forecast metrics, all following
    sklearn's scorer API with `fit` / `score`. Each metric supports flexible
    aggregation across time, components, and panel groups.

    ## What You'll Learn

    - All 8 point scorers: MAE, MSE, RMSE, MedianAE, MAPE, sMAPE, RMSSE, MASE
    - Aggregation methods: `"timewise"`, `"componentwise"`, `"groupwise"`, `"all"`
    - Scaled metrics that require training data for normalization
    - Time-weighted scoring
    - Visualizing scores with `plot_score_time_series` and `plot_model_comparison_bar`

    ## Prerequisites

    Basic understanding of forecast error metrics.
    """)
    return

@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_tourism_monthly
    from yohou.metrics import (
        MeanAbsoluteError,
        MeanAbsolutePercentageError,
        MeanAbsoluteScaledError,
        MeanSquaredError,
        MedianAbsoluteError,
        RootMeanSquaredError,
        RootMeanSquaredScaledError,
        SymmetricMeanAbsolutePercentageError,
    )
    from yohou.plotting import plot_model_comparison_bar, plot_score_time_series
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer

    return (
        LagTransformer,
        MeanAbsoluteError,
        MeanAbsolutePercentageError,
        MeanAbsoluteScaledError,
        MeanSquaredError,
        MedianAbsoluteError,
        PointReductionForecaster,
        Ridge,
        RootMeanSquaredError,
        RootMeanSquaredScaledError,
        SeasonalNaive,
        SymmetricMeanAbsolutePercentageError,
        fetch_tourism_monthly,
        pl,
        plot_model_comparison_bar,
        plot_score_time_series,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Generate Forecasts for Evaluation

    We fit a simple forecaster and generate predictions to use as input for the metric functions.
    """)
    return

@app.cell
def _(
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SeasonalNaive,
    fetch_tourism_monthly,
):
    y = (
        fetch_tourism_monthly()
        .frame.select("time", "T1__tourists").drop_nulls()
        .rename({"T1__tourists": "passengers"})
    )

    y_train = y.head(120)
    y_test = y.tail(24)
    fh = len(y_test)

    # Two models to compare
    naive = SeasonalNaive(seasonality=12)
    naive.fit(y_train, forecasting_horizon=fh)
    y_pred_naive = naive.predict(forecasting_horizon=fh)

    ridge_fc = PointReductionForecaster(
        estimator=Ridge(),
        feature_transformer=LagTransformer(lag=list(range(1, 13))),
    )
    ridge_fc.fit(y_train, forecasting_horizon=fh)
    y_pred_ridge = ridge_fc.predict(forecasting_horizon=fh)

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    print(f"Naive predictions: {len(y_pred_naive)}")
    print(f"Ridge predictions: {len(y_pred_ridge)}")
    return y_pred_naive, y_pred_ridge, y_test, y_train

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Basic Scorers

    All scorers follow the same pattern: instantiate → `fit(y_train)` → `score(y_test, y_pred)`.
    """)
    return

@app.cell
def _(
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MedianAbsoluteError,
    RootMeanSquaredError,
    SymmetricMeanAbsolutePercentageError,
    y_pred_naive,
    y_pred_ridge,
    y_test,
    y_train,
):
    scorers = [
        ("MAE", MeanAbsoluteError()),
        ("MSE", MeanSquaredError()),
        ("RMSE", RootMeanSquaredError()),
        ("MedianAE", MedianAbsoluteError()),
        ("MAPE", MeanAbsolutePercentageError()),
        ("sMAPE", SymmetricMeanAbsolutePercentageError()),
    ]

    print(f"{'Metric':>10s}  {'Naive':>10s}  {'Ridge':>10s}")
    print("-" * 35)
    for _name, _scorer in scorers:
        _scorer.fit(y_train)
        _s_naive = _scorer.score(y_test, y_pred_naive)
        _s_ridge = _scorer.score(y_test, y_pred_ridge)
        print(f"{_name:>10s}  {_s_naive:>10.2f}  {_s_ridge:>10.2f}")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Scaled Metrics (MASE, RMSSE)

    Scaled metrics normalize errors by the in-sample naive forecast error.
    They require `fit(y_train)` to compute the scaling factor.
    A score < 1 means the model outperforms the naive baseline.
    """)
    return

@app.cell
def _(
    MeanAbsoluteScaledError,
    RootMeanSquaredScaledError,
    y_pred_naive,
    y_pred_ridge,
    y_test,
    y_train,
):
    for _name, _scorer_cls in [
        ("MASE", MeanAbsoluteScaledError),
        ("RMSSE", RootMeanSquaredScaledError),
    ]:
        _scorer = _scorer_cls(seasonality=12)
        _scorer.fit(y_train)  # fit on TRAIN data for scaling
        _s_naive = _scorer.score(y_test, y_pred_naive)
        _s_ridge = _scorer.score(y_test, y_pred_ridge)
        print(f"{_name}: Naive={_s_naive:.3f}, Ridge={_s_ridge:.3f}")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Aggregation Methods

    By default `aggregation_method="all"` returns a single scalar.
    Choose `"timewise"` or `"componentwise"` for more granular results.
    """)
    return

@app.cell
def _(MeanAbsoluteError, y_pred_ridge, y_test, y_train):
    # Timewise: one score per time step
    mae_tw = MeanAbsoluteError(aggregation_method="timewise")
    mae_tw.fit(y_train)
    scores_tw = mae_tw.score(y_test, y_pred_ridge)
    print("Timewise MAE (first 5 steps):")
    print(scores_tw.head())
    return

@app.cell
def _(MeanAbsoluteError, y_pred_ridge, y_test, y_train):
    # Componentwise: one score per target column
    mae_cw = MeanAbsoluteError(aggregation_method="componentwise")
    mae_cw.fit(y_train)
    scores_cw = mae_cw.score(y_test, y_pred_ridge)
    print(f"Componentwise MAE: {scores_cw}")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Visualizing Scores

    We plot the metric results to compare forecaster performance visually.
    """)
    return

@app.cell
def _(
    MeanAbsoluteError,
    plot_score_time_series,
    y_pred_naive,
    y_pred_ridge,
    y_test,
    y_train,
):
    mae_scorer = MeanAbsoluteError(aggregation_method="timewise")
    mae_scorer.fit(y_train)

    plot_score_time_series(
        mae_scorer,
        y_test,
        {"Naive": y_pred_naive, "Ridge": y_pred_ridge},
        title="MAE Over Time",
    )
    return

@app.cell
def _(
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    RootMeanSquaredError,
    plot_model_comparison_bar,
    y_pred_naive,
    y_pred_ridge,
    y_test,
    y_train,
):
    # Build results dict for bar chart
    results = {}
    for _model_name, _y_pred in [("Naive", y_pred_naive), ("Ridge", y_pred_ridge)]:
        _model_scores = {}
        for _scorer_name, _scorer_cls in [
            ("MAE", MeanAbsoluteError),
            ("RMSE", RootMeanSquaredError),
            ("MAPE", MeanAbsolutePercentageError),
        ]:
            _s = _scorer_cls()
            _s.fit(y_train)
            _model_scores[_scorer_name] = _s.score(y_test, _y_pred)
        results[_model_name] = _model_scores

    plot_model_comparison_bar(results, title="Model Comparison")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - All point scorers follow `fit()` → `score()` pattern
    - Basic metrics (MAE, MSE, RMSE, MAPE, sMAPE, MedianAE) fit on test data
    - Scaled metrics (MASE, RMSSE) fit on **training data** for normalization
    - `aggregation_method` controls granularity: `"all"`, `"timewise"`, `"componentwise"`
    - Use `plot_score_time_series` for temporal error analysis
    - Use `plot_model_comparison_bar` for multi-model comparison
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Interval metrics**: See `interval_metrics.py` for interval scoring
    - **Cross-validation**: See `model_selection/` for temporal CV with scoring
    - **Time weighting**: See `examples/time_weighted_forecasting.py`
    """)
    return

if __name__ == "__main__":
    app.run()
