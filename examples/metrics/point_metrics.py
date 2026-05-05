# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.23.1"
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
    - Aggregation methods: `["stepwise", "vintagewise"]`, `"componentwise"`, `"groupwise"`, `"all"`
    - Scaled metrics that require training data for normalization
    - Visualizing scores with [`plot_score_time_series`](/pages/api/generated/yohou.plotting.evaluation.plot_score_time_series/) and [`plot_score_per_step`](/pages/api/generated/yohou.plotting.evaluation.plot_score_per_step/)

    ## Prerequisites

    Basic understanding of forecast error metrics.
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

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
    from yohou.model_selection import train_test_split
    from yohou.plotting import (
        plot_forecast,
        plot_score_per_vintage,
        plot_score_summary,
        plot_score_time_series,
        plot_time_series,
    )
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
        deepcopy,
        fetch_tourism_monthly,
        plot_forecast,
        plot_score_per_vintage,
        plot_score_summary,
        plot_score_time_series,
        plot_time_series,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Generate Forecasts for Evaluation

    We fit two forecasters on the Tourism Monthly dataset: a [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/)
    baseline and a [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) with a `Ridge` regressor. Both
    produce predictions over the same test horizon, giving us two sets of
    forecasts to compare across all metrics.
    """)


@app.cell
def _(
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SeasonalNaive,
    fetch_tourism_monthly,
    train_test_split,
):
    y = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})

    y_train, y_test = train_test_split(y, test_size=0.2)
    fh = len(y_test)

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
    return fh, ridge_fc, y, y_pred_naive, y_pred_ridge, y_test, y_train


@app.cell
def _(plot_time_series, y):
    plot_time_series(y, title="Tourism Monthly")


@app.cell
def _(plot_forecast, y_pred_naive, y_pred_ridge, y_test, y_train):
    plot_forecast(
        y_test,
        {"Naive": y_pred_naive, "Ridge": y_pred_ridge},
        y_train=y_train,
        title="Forecasts for Evaluation",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. MeanAbsoluteError (MAE)

    The average absolute difference between prediction and actual value.
    Easy to interpret: measured in the same units as the target.
    """)


@app.cell
def _(MeanAbsoluteError, y_pred_naive, y_pred_ridge, y_test, y_train):
    mae = MeanAbsoluteError()
    mae.fit(y_train)
    print(f"MAE  Naive: {mae.score(y_test, y_pred_naive):.2f}")
    print(f"MAE  Ridge: {mae.score(y_test, y_pred_ridge):.2f}")


@app.cell
def _(
    MeanAbsoluteError,
    plot_score_time_series,
    y_pred_naive,
    y_pred_ridge,
    y_test,
):
    plot_score_time_series(
        MeanAbsoluteError(),
        y_test,
        {"Naive": y_pred_naive, "Ridge": y_pred_ridge},
        title="MAE per Timestep",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. MeanSquaredError (MSE) and RootMeanSquaredError (RMSE)

    MSE penalizes large errors more heavily (squared). RMSE is its square
    root, bringing units back to the original scale.
    """)


@app.cell
def _(
    MeanSquaredError,
    RootMeanSquaredError,
    y_pred_naive,
    y_pred_ridge,
    y_test,
    y_train,
):
    mse = MeanSquaredError()
    mse.fit(y_train)
    print(f"MSE   Naive: {mse.score(y_test, y_pred_naive):.2f}")
    print(f"MSE   Ridge: {mse.score(y_test, y_pred_ridge):.2f}")

    rmse = RootMeanSquaredError()
    rmse.fit(y_train)
    print(f"RMSE  Naive: {rmse.score(y_test, y_pred_naive):.2f}")
    print(f"RMSE  Ridge: {rmse.score(y_test, y_pred_ridge):.2f}")


@app.cell
def _(
    RootMeanSquaredError,
    plot_score_time_series,
    y_pred_naive,
    y_pred_ridge,
    y_test,
):
    plot_score_time_series(
        RootMeanSquaredError(),
        y_test,
        {"Naive": y_pred_naive, "Ridge": y_pred_ridge},
        title="RMSE per Timestep",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. MedianAbsoluteError (MedianAE)

    The median of absolute errors which is more robust to outliers than MAE.
    """)


@app.cell
def _(MedianAbsoluteError, y_pred_naive, y_pred_ridge, y_test, y_train):
    medae = MedianAbsoluteError()
    medae.fit(y_train)
    print(f"MedianAE  Naive: {medae.score(y_test, y_pred_naive):.2f}")
    print(f"MedianAE  Ridge: {medae.score(y_test, y_pred_ridge):.2f}")


@app.cell
def _(
    MedianAbsoluteError,
    plot_score_time_series,
    y_pred_naive,
    y_pred_ridge,
    y_test,
):
    plot_score_time_series(
        MedianAbsoluteError(),
        y_test,
        {"Naive": y_pred_naive, "Ridge": y_pred_ridge},
        title="MedianAE per Timestep",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. MeanAbsolutePercentageError (MAPE) and sMAPE

    MAPE expresses error as a percentage of the actual value.
    sMAPE (symmetric MAPE) avoids the asymmetry of standard MAPE by
    normalizing by the average of actual and predicted values.
    """)


@app.cell
def _(
    MeanAbsolutePercentageError,
    SymmetricMeanAbsolutePercentageError,
    y_pred_naive,
    y_pred_ridge,
    y_test,
    y_train,
):
    mape = MeanAbsolutePercentageError()
    mape.fit(y_train)
    print(f"MAPE   Naive: {mape.score(y_test, y_pred_naive):.4f}")
    print(f"MAPE   Ridge: {mape.score(y_test, y_pred_ridge):.4f}")

    smape = SymmetricMeanAbsolutePercentageError()
    smape.fit(y_train)
    print(f"sMAPE  Naive: {smape.score(y_test, y_pred_naive):.4f}")
    print(f"sMAPE  Ridge: {smape.score(y_test, y_pred_ridge):.4f}")


@app.cell
def _(
    MeanAbsolutePercentageError,
    plot_score_time_series,
    y_pred_naive,
    y_pred_ridge,
    y_test,
):
    plot_score_time_series(
        MeanAbsolutePercentageError(),
        y_test,
        {"Naive": y_pred_naive, "Ridge": y_pred_ridge},
        title="MAPE per Timestep",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Scaled Metrics (MASE, RMSSE)

    Scaled metrics normalize errors by the in-sample naive forecast error.
    They require `fit(y_train)` to compute the scaling factor.
    A score < 1 means the model outperforms the naive baseline.
    """)


@app.cell
def _(
    MeanAbsoluteScaledError,
    RootMeanSquaredScaledError,
    y_pred_naive,
    y_pred_ridge,
    y_test,
    y_train,
):
    mase = MeanAbsoluteScaledError(seasonality=12)
    mase.fit(y_train)
    print(f"MASE   Naive: {mase.score(y_test, y_pred_naive):.3f}")
    print(f"MASE   Ridge: {mase.score(y_test, y_pred_ridge):.3f}")

    rmsse = RootMeanSquaredScaledError(seasonality=12)
    rmsse.fit(y_train)
    print(f"RMSSE  Naive: {rmsse.score(y_test, y_pred_naive):.3f}")
    print(f"RMSSE  Ridge: {rmsse.score(y_test, y_pred_ridge):.3f}")


@app.cell
def _(
    MeanAbsoluteScaledError,
    plot_score_time_series,
    y_pred_naive,
    y_pred_ridge,
    y_test,
):
    plot_score_time_series(
        MeanAbsoluteScaledError(seasonality=12),
        y_test,
        {"Naive": y_pred_naive, "Ridge": y_pred_ridge},
        title="MASE per Timestep",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Aggregation Methods

    By default `aggregation_method="all"` returns a single scalar.
    Choose `["stepwise", "vintagewise"]` or `"componentwise"` for more granular results.
    """)


@app.cell
def _(MeanAbsoluteError, y_pred_ridge, y_test, y_train):
    mae_sv = MeanAbsoluteError(aggregation_method=["stepwise", "vintagewise"])
    mae_sv.fit(y_train)
    scores_tw = mae_sv.score(y_test, y_pred_ridge)
    print("Stepwise+vintagewise MAE (first 5 steps):")
    print(scores_tw.head())


@app.cell
def _(MeanAbsoluteError, y_pred_ridge, y_test, y_train):
    mae_cw = MeanAbsoluteError(aggregation_method="componentwise")
    mae_cw.fit(y_train)
    scores_cw = mae_cw.score(y_test, y_pred_ridge)
    print(f"Componentwise MAE: {scores_cw}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Model Comparison Summary

    [`plot_score_summary`](/pages/api/generated/yohou.plotting.evaluation.plot_score_summary/) takes
    scorer(s), ground truth, and a dict of predictions, then renders a grouped bar chart
    making it easy to spot which model performs best on each metric.
    """)


@app.cell
def _(
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    RootMeanSquaredError,
    plot_score_summary,
    y_pred_naive,
    y_pred_ridge,
    y_test,
):
    plot_score_summary(
        {"MAE": MeanAbsoluteError(), "RMSE": RootMeanSquaredError(), "MAPE": MeanAbsolutePercentageError()},
        y_test,
        {"Naive": y_pred_naive, "Ridge": y_pred_ridge},
        title="Model Comparison",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Classification Accuracy (Hard Classification)

    When forecasting **categorical outcomes** (e.g., weather, state transitions),
    [`Accuracy`](/pages/api/generated/yohou.metrics.class_proba.Accuracy/) measures
    the fraction of time steps where the predicted class matches the true class.

    > **Caution**: Accuracy can be misleading on **imbalanced** datasets.  A model
    > that always predicts the majority class can score high without learning
    > anything useful. For imbalanced problems, prefer *soft* metrics like
    > [`LogLoss`](/pages/api/generated/yohou.metrics.class_proba.LogLoss/) or
    > [`BrierScore`](/pages/api/generated/yohou.metrics.class_proba.BrierScore/),
    > which evaluate the full probability distribution and penalize overconfident
    > wrong predictions.
    """)


@app.cell(hide_code=True)
def _():
    from sklearn.tree import DecisionTreeClassifier

    from yohou.class_proba import ClassProbaReductionForecaster
    from yohou.datasets import fetch_air_quality_classification
    from yohou.metrics import Accuracy

    return (
        Accuracy,
        ClassProbaReductionForecaster,
        DecisionTreeClassifier,
        fetch_air_quality_classification,
    )


@app.cell
def _(
    ClassProbaReductionForecaster,
    DecisionTreeClassifier,
    LagTransformer,
    fetch_air_quality_classification,
    train_test_split,
):
    cls_data = fetch_air_quality_classification()
    cls_y, cls_X = cls_data.y, cls_data.X_actual
    cls_y_train, cls_y_test, cls_X_train, cls_X_test = train_test_split(
        cls_y,
        cls_X,
        test_size=200,
    )
    cls_fh = 24

    cls_forecaster = ClassProbaReductionForecaster(
        estimator=DecisionTreeClassifier(random_state=42),
        feature_transformer=LagTransformer(lag=[1, 2, 3, 6, 12, 24]),
    )
    cls_forecaster.fit(cls_y_train, cls_X_train, forecasting_horizon=cls_fh)

    # predict() returns hard class labels (argmax of probabilities)
    cls_y_pred_labels = cls_forecaster.predict(
        forecasting_horizon=cls_fh,
    )
    # predict_class_proba() returns the full probability distribution
    cls_y_proba = cls_forecaster.predict_class_proba(
        forecasting_horizon=cls_fh,
    )

    print(f"Classes: {cls_data.classes}")
    print("\nHard predictions (predict):")
    print(cls_y_pred_labels)
    print("\nSoft predictions (predict_class_proba):")
    print(cls_y_proba)
    return cls_y_pred_labels, cls_y_proba, cls_y_test, cls_y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Categorical Target Over Time

    `plot_forecast` auto-detects categorical columns and renders a step chart.
    Here we visualize the training and test target side by side.
    """)


@app.cell
def _(cls_y_test, cls_y_train, plot_forecast):
    plot_forecast(
        cls_y_test,
        cls_y_test,
        y_train=cls_y_train,
        n_history=100,
        title="Air Quality Target (Categorical Time Series)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Categorical Forecast vs Actual

    The hard-label predictions from `predict()` are compared against the
    true classes. Dashed lines show the forecast, solid lines the actual.
    """)


@app.cell
def _(cls_y_pred_labels, cls_y_test, plot_forecast):
    plot_forecast(
        cls_y_test,
        cls_y_pred_labels,
        title="Categorical Forecast vs Actual",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Probability Forecast

    The full probability distribution from `predict_class_proba()` is shown
    as a stacked area chart. Diamond markers indicate the true class.
    """)


@app.cell
def _(cls_y_proba, cls_y_test, plot_forecast):
    plot_forecast(
        cls_y_test,
        cls_y_proba,
        title="Class Probability Forecast",
    )


@app.cell
def _(Accuracy, cls_y_proba, cls_y_test):
    cls_y_truth = cls_y_test.head(len(cls_y_proba))

    acc_all = Accuracy()
    acc_all.fit(cls_y_truth)
    print(f"Accuracy (scalar): {acc_all.score(cls_y_truth, cls_y_proba):.4f}")

    # Stepwise+vintagewise: aggregate across time dimensions
    acc_sv = Accuracy(aggregation_method=["stepwise", "vintagewise"])
    acc_sv.fit(cls_y_truth)
    print("\nPer-timestep accuracy:")
    print(acc_sv.score(cls_y_truth, cls_y_proba))
    return (cls_y_truth,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Accuracy Over Time

    Per-timestep accuracy for the 24-hour forecast window. A score of 1.0
    means the argmax prediction matched the true class at that step.
    """)


@app.cell
def _(Accuracy, cls_y_proba, cls_y_truth, plot_score_time_series):
    plot_score_time_series(
        Accuracy(),
        cls_y_truth,
        cls_y_proba,
        title="Accuracy per Timestep",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Hard vs Soft**: `Accuracy` scores 1.0 for a correct prediction and 0.0
    > otherwise, regardless of confidence. A model that predicts the right class
    > with 51% probability gets the same Accuracy as one that predicts with 99%.
    > For calibration-aware evaluation, see the [soft classification metrics](/examples/metrics/class_proba_metrics/)
    > (`LogLoss`, `BrierScore`).
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - All point scorers follow `fit()` → `score()` pattern
    - Basic metrics (MAE, MSE, RMSE, MAPE, sMAPE, MedianAE) need no training data
    - Scaled metrics (MASE, RMSSE) fit on **training data** for normalization
    - `aggregation_method` controls granularity: `"all"`, `["stepwise", "vintagewise"]`, `"componentwise"`
    - [`Accuracy`](/pages/api/generated/yohou.metrics.class_proba.Accuracy/) evaluates hard classification correctness (use with care on imbalanced data)
    - Use [`plot_score_time_series`](/pages/api/generated/yohou.plotting.evaluation.plot_score_time_series/) for temporal error analysis
    - Use [`plot_score_summary`](/pages/api/generated/yohou.plotting.evaluation.plot_score_summary/) for multi-model comparison
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-vintage Scoring

    The `observe_predict` method with `stride=1` produces one forecast per
    observation point, creating multiple *vintages*. Each vintage represents
    a different forecast origin, so you can analyse how accuracy evolves as
    the model absorbs more data.
    """)


@app.cell
def _(deepcopy, fh, ridge_fc, y_test):
    _vintage_model = deepcopy(ridge_fc)
    y_pred_vintages = _vintage_model.observe_predict(
        y=y_test,
        stride=1,
        forecasting_horizon=fh,
    )
    print(f"Vintages: {y_pred_vintages['vintage_time'].n_unique()}")
    y_pred_vintages.head(10)
    return (y_pred_vintages,)


@app.cell
def _(MeanAbsoluteError, y_train):
    vintage_scorer = MeanAbsoluteError()
    vintage_scorer.fit(y_train)
    return (vintage_scorer,)


@app.cell
def _(plot_score_per_vintage, vintage_scorer, y_pred_vintages, y_test):
    plot_score_per_vintage(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="MAE per Forecast Vintage",
        y_label="MAE",
        height=380,
    )


@app.cell
def _(plot_score_time_series, vintage_scorer, y_pred_vintages, y_test):
    plot_score_time_series(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="Per-timestep MAE across Vintages",
        y_label="MAE",
        height=500,
        facet_by="member",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Interval metrics**: See [`interval_metrics.py`](/examples/metrics/interval_metrics/) for interval scoring
    - **Cross-validation**: See [Model Selection](/examples/#model-selection) for temporal CV with scoring
    - **Time weighting**: See [`time_weighted_scoring.py`](/examples/metrics/time_weighted_scoring/)
    - **Classification metrics**: See [`class_proba_metrics.py`](/examples/metrics/class_proba_metrics/) for soft classification metrics (LogLoss, BrierScore) and reliability diagrams
    """)


if __name__ == "__main__":
    app.run()
