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
    "title": "How to Score Class-Probability Forecasts",
    "description": "Evaluate categorical forecasts with LogLoss, BrierScore, and Accuracy. Covers per-timestep scoring, aggregation modes, and reliability diagrams.",
    "category": "how-to",
    "section": "evaluation-search",
    "companion": "/pages/how-to/class-probability-forecasting/",
    "api_references": ["Accuracy", "BrierScore", "ClassProbaReductionForecaster", "LogLoss", "plot_calibration", "plot_score_time_series"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Class-Probability Metrics

    Yohou provides three dedicated metrics for evaluating categorical
    probability forecasts. All follow sklearn's `fit` / `score` API and
    support flexible aggregation across time steps, components, and panel
    groups.

    **Prerequisites:** Basic understanding of classification metrics and probability calibration.
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    from yohou.class_proba import ClassProbaReductionForecaster
    from yohou.datasets import fetch_air_quality_classification
    from yohou.metrics import Accuracy, BrierScore, LogLoss
    from yohou.model_selection import train_test_split
    from yohou.plotting import (
        plot_calibration,
        plot_forecast,
        plot_score_time_series,
        plot_time_series,
    )
    from yohou.preprocessing import LagTransformer

    return (
        Accuracy,
        BrierScore,
        ClassProbaReductionForecaster,
        DecisionTreeClassifier,
        LagTransformer,
        LogLoss,
        RandomForestClassifier,
        deepcopy,
        fetch_air_quality_classification,
        plot_calibration,
        plot_forecast,
        plot_score_time_series,
        plot_time_series,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Generate Forecasts

    We use [`fetch_air_quality_classification`](/pages/api/generated/yohou.datasets._fetchers.fetch_air_quality_classification/)
    to derive a 4-class air quality target from the KDD Cup 2018 PM2.5 data,
    then fit two models of differing capacity: a Decision Tree and a
    Random Forest. Both produce probability forecasts via rolling
    `observe_predict_class_proba()`.
    """)


@app.cell
def _(fetch_air_quality_classification, train_test_split):
    data = fetch_air_quality_classification()
    y, X_actual = data.y, data.X_actual

    y_train, y_test, X_actual_train, X_actual_test = train_test_split(
        y,
        X_actual,
        test_size=200,
    )

    print(f"Classes: {data.classes}")
    print(f"Train: {len(y_train)} obs | Test: {len(y_test)} obs")
    return X_actual_test, X_actual_train, data, y, y_test, y_train, X_actual


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dataset Overview

    The features are 5 pollutant time series measured hourly. Visualizing
    them helps understand the temporal patterns the models must learn.
    """)


@app.cell
def _(X_actual, plot_time_series):
    plot_time_series(X_actual, title="Air Quality - Pollutant Features")


@app.cell
def _(data, y):
    target_col = data.target_names[0]
    counts = y.group_by(target_col).len().sort("len", descending=True)
    print("Class distribution:")
    counts
    return counts, target_col


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Categorical Target Over Time

    Visualizing the target as a step chart reveals temporal patterns:
    transitions between air quality classes and how long each state persists.
    """)


@app.cell
def _(plot_forecast, y, y_train):
    plot_forecast(
        y,
        y,
        y_train=y_train,
        n_history=200,
        title="Air Quality Target Over Time",
    )


@app.cell
def _(
    ClassProbaReductionForecaster,
    DecisionTreeClassifier,
    LagTransformer,
    RandomForestClassifier,
    X_actual_test,
    X_actual_train,
    deepcopy,
    y_test,
    y_train,
):
    fh = 24

    dt = ClassProbaReductionForecaster(
        estimator=DecisionTreeClassifier(random_state=42),
        actual_transformer=LagTransformer(lag=[1, 2, 3, 6, 12, 24]),
    )
    dt.fit(y_train, X_actual_train, forecasting_horizon=fh)
    dt_hard = deepcopy(dt)
    y_proba_dt = dt.observe_predict_class_proba(y=y_test, X_actual=X_actual_test).sort("time")

    rf = ClassProbaReductionForecaster(
        estimator=RandomForestClassifier(n_estimators=50, random_state=42),
        actual_transformer=LagTransformer(lag=[1, 2, 3, 6, 12, 24]),
    )
    rf.fit(y_train, X_actual_train, forecasting_horizon=fh)
    rf_hard = deepcopy(rf)
    y_proba_rf = rf.observe_predict_class_proba(y=y_test, X_actual=X_actual_test).sort("time")

    print(f"DT predictions: {len(y_proba_dt)} rows")
    print(f"RF predictions: {len(y_proba_rf)} rows")
    return dt, dt_hard, fh, rf, rf_hard, y_proba_dt, y_proba_rf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hard-Label Forecast Comparison

    Before looking at probabilities, let's compare the hard class predictions
    (argmax of probabilities) from both models against the actual classes.
    """)


@app.cell
def _(X_actual_test, dt_hard, fh, plot_forecast, rf_hard, y_test):
    y_pred_dt = dt_hard.observe_predict(y=y_test, X_actual=X_actual_test).sort("time")
    y_pred_rf = rf_hard.observe_predict(y=y_test, X_actual=X_actual_test).sort("time")
    plot_forecast(
        y_test,
        {"Decision Tree": y_pred_dt, "Random Forest": y_pred_rf},
        title="Categorical Forecast Comparison (Hard Labels)",
    )
    return y_pred_dt, y_pred_rf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Multi-Model Probability Forecast

    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/)
    auto-detects `_proba_` columns and renders stacked area charts.
    Passing a dict of predictions creates one subplot per model for
    side-by-side comparison.
    """)


@app.cell
def _(plot_forecast, y_proba_dt, y_proba_rf, y_test):
    plot_forecast(
        y_test,
        {"Decision Tree": y_proba_dt, "Random Forest": y_proba_rf},
        title="Probability Forecast Comparison",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Log Loss

    [`LogLoss`](/pages/api/generated/yohou.metrics.class_proba.LogLoss/) computes cross-entropy loss between predicted probabilities
    and the true class distribution. Lower values indicate better-calibrated
    predictions. Probabilities are clipped to avoid infinite loss.
    """)


@app.cell
def _(LogLoss, y_proba_dt, y_proba_rf, y_test):
    ll = LogLoss()
    ll.fit(y_test)

    ll_dt = ll.score(y_test, y_proba_dt)
    ll_rf = ll.score(y_test, y_proba_rf)

    print(f"Log Loss (Decision Tree):  {ll_dt:.4f}")
    print(f"Log Loss (Random Forest):  {ll_rf:.4f}")
    return ll, ll_dt, ll_rf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Per-Timestep Scores

    Set `aggregation=["stepwise", "vintagewise"]` to get a score for each component. This
    reveals when predictions are most uncertain.
    """)


@app.cell
def _(LogLoss, y_proba_rf, y_test):
    ll_sv = LogLoss(aggregation_method=["stepwise", "vintagewise"])
    ll_sv.fit(y_test)
    scores_time = ll_sv.score(y_test, y_proba_rf)

    print("Per-timestep Log Loss (first 10):")
    scores_time.head(10)
    return ll_sv, scores_time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Brier Score

    [`BrierScore`](/pages/api/generated/yohou.metrics.class_proba.BrierScore/) measures the mean squared error between predicted
    probabilities and one-hot truth vectors. It ranges from 0 (perfect)
    to 2 (worst case) for multi-class problems.
    """)


@app.cell
def _(BrierScore, y_proba_dt, y_proba_rf, y_test):
    bs = BrierScore()
    bs.fit(y_test)

    bs_dt = bs.score(y_test, y_proba_dt)
    bs_rf = bs.score(y_test, y_proba_rf)

    print(f"Brier Score (Decision Tree):  {bs_dt:.4f}")
    print(f"Brier Score (Random Forest):  {bs_rf:.4f}")
    return bs, bs_dt, bs_rf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Componentwise Aggregation

    For multi-target datasets, `aggregation="componentwise"` returns one
    score per target column.
    """)


@app.cell
def _(BrierScore, y_proba_rf, y_test):
    bs_comp = BrierScore(aggregation_method="componentwise")
    bs_comp.fit(y_test)
    scores_comp = bs_comp.score(y_test, y_proba_rf)

    print("Componentwise Brier Score:")
    scores_comp
    return bs_comp, scores_comp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Accuracy

    [`Accuracy`](/pages/api/generated/yohou.metrics.class_proba.Accuracy/) converts probabilities to class labels via argmax
    and computes the fraction of correct predictions. Higher is better.
    """)


@app.cell
def _(Accuracy, y_proba_dt, y_proba_rf, y_test):
    acc = Accuracy()
    acc.fit(y_test)

    acc_dt = acc.score(y_test, y_proba_dt)
    acc_rf = acc.score(y_test, y_proba_rf)

    print(f"Accuracy (Decision Tree):  {acc_dt:.4f}")
    print(f"Accuracy (Random Forest):  {acc_rf:.4f}")
    return acc, acc_dt, acc_rf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Model Comparison Summary

    Let's compare all three metrics side-by-side.
    """)


@app.cell
def _(acc_dt, acc_rf, bs_dt, bs_rf, ll_dt, ll_rf, mo):
    _table = mo.ui.table(
        [
            {"Metric": "Log Loss", "Decision Tree": f"{ll_dt:.4f}", "Random Forest": f"{ll_rf:.4f}", "Better": "lower"},
            {
                "Metric": "Brier Score",
                "Decision Tree": f"{bs_dt:.4f}",
                "Random Forest": f"{bs_rf:.4f}",
                "Better": "lower",
            },
            {
                "Metric": "Accuracy",
                "Decision Tree": f"{acc_dt:.4f}",
                "Random Forest": f"{acc_rf:.4f}",
                "Better": "higher",
            },
        ],
        selection=None,
    )
    mo.md(f"""
    ### Results

    {_table}
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Visualize Predictions

    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) shows how predicted class probabilities evolve
    over time. Diamond markers indicate the true class at each time step.
    """)


@app.cell
def _(plot_forecast, y_proba_rf, y_test):
    plot_forecast(
        y_test,
        y_proba_rf,
        title="Random Forest - Class Probabilities vs Actuals",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Reliability Diagram

    [`plot_calibration`](/pages/api/generated/yohou.plotting.evaluation.plot_calibration/) compares predicted probabilities to observed
    frequencies. A well-calibrated model has points near the diagonal. This
    helps diagnose whether your model is over- or under-confident.
    """)


@app.cell
def _(plot_calibration, y_proba_rf, y_test):
    plot_calibration(
        y_proba_rf,
        y_test,
        n_bins=8,
        title="Random Forest Calibration",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Score Over Time

    [`plot_score_time_series`](/pages/api/generated/yohou.plotting.evaluation.plot_score_time_series/)
    shows how each metric evolves across time steps. Passing a dict of
    predictions overlays both models on the same axes for easy comparison.
    """)


@app.cell
def _(LogLoss, plot_score_time_series, y_proba_dt, y_proba_rf, y_test):
    plot_score_time_series(
        LogLoss(),
        y_test,
        {"Decision Tree": y_proba_dt, "Random Forest": y_proba_rf},
        title="Log Loss Over Time - Model Comparison",
    )


@app.cell
def _(BrierScore, plot_score_time_series, y_proba_dt, y_proba_rf, y_test):
    plot_score_time_series(
        BrierScore(),
        y_test,
        {"Decision Tree": y_proba_dt, "Random Forest": y_proba_rf},
        title="Brier Score Over Time - Model Comparison",
    )


@app.cell
def _(Accuracy, plot_score_time_series, y_proba_dt, y_proba_rf, y_test):
    plot_score_time_series(
        Accuracy(),
        y_test,
        {"Decision Tree": y_proba_dt, "Random Forest": y_proba_rf},
        title="Accuracy Over Time - Model Comparison",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - [`class_proba_forecaster.py`](/examples/point/class_proba_forecaster/) - Full class-probability forecasting walkthrough
    - [`point_metrics.py`](/examples/evaluation-search/point_metrics/) - Point forecast evaluation metrics
    - [`aggregation_modes.py`](/examples/evaluation-search/aggregation_modes/) - Deep dive into aggregation modes
    - [Metrics](/examples/#metrics) - All metric examples
    """)


if __name__ == "__main__":
    app.run()
