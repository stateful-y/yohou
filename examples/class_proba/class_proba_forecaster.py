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
    "title": "How to Forecast Class Probabilities",
    "description": "Use ClassProbaReductionForecaster to produce calibrated probability forecasts and evaluate them with Brier score, log loss, and accuracy.",
    "category": "how-to",
    "companion": "pages/explanation/class-probability-forecasting",
    "section": "forecasting-models",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Class-Probability Forecasting

    This notebook demonstrates **class-probability forecasting**, that is, predicting the
    probability distribution over categorical outcomes at future time steps.

    ## What You'll Learn

    - How [`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.reduction.ClassProbaReductionForecaster/) converts categorical time series into a classification problem
    - Obtaining probability predictions with `predict_class_proba()` and class labels with `predict()`
    - Evaluating predictions with [`LogLoss`](/pages/api/generated/yohou.metrics.class_proba.LogLoss/), [`BrierScore`](/pages/api/generated/yohou.metrics.class_proba.BrierScore/), and [`Accuracy`](/pages/api/generated/yohou.metrics.class_proba.Accuracy/)
    - Visualizing class probabilities with [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/)
    - Using the observe-predict workflow for rolling evaluation

    We use [`fetch_air_quality_classification`](/pages/api/generated/yohou.datasets._fetchers.fetch_air_quality_classification/),
    which derives a 4-class air quality target (good / moderate / unhealthy / hazardous)
    from the KDD Cup 2018 PM2.5 data, with 5 pollutant features.

    ## Prerequisites

    Basic familiarity with sklearn's fit/predict API and classification concepts.
    """)


@app.cell(hide_code=True)
def _():
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
    ## 1. Load the Data

    We use [`fetch_air_quality_classification`](/pages/api/generated/yohou.datasets._fetchers.fetch_air_quality_classification/)
    to derive a categorical air quality target from KDD Cup 2018 PM2.5 data.
    The target has 4 WHO-based classes (good, moderate, unhealthy, hazardous)
    and 5 pollutant features (PM10, NO2, CO, O3, SO2) at hourly intervals.
    """)


@app.cell
def _(fetch_air_quality_classification):
    data = fetch_air_quality_classification()
    y, X_actual = data.y, data.X_actual

    print(f"Classes: {data.classes}")
    print(f"Dataset: {len(y)} observations from {y['time'].min()} to {y['time'].max()}")
    y.head(10)
    return X_actual, data, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Explore the Features

    The exogenous features are 5 pollutant measurements (PM10, NO2, CO, O3, SO2)
    measured hourly. These serve as known-in-advance inputs to the forecaster.
    """)


@app.cell
def _(X_actual, plot_time_series):
    plot_time_series(X_actual, title="Pollutant Features Over Time")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Target Class Distribution

    The target variable is a WHO-based air quality category derived from PM2.5
    concentration. Let's see how the classes are distributed over time.
    """)


@app.cell
def _(data, y):

    target_col = data.target_names[0]
    counts = y.group_by(target_col).len().sort("len", descending=True)
    print("Class distribution:")
    counts


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Categorical Target Over Time

    Visualizing the categorical target as a step chart reveals temporal
    patterns: transitions between classes and how long each state persists.
    """)


@app.cell
def _(plot_time_series, y):
    plot_time_series(
        y.tail(200),
        title="Air Quality Target Over Time",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Train/Test Split

    We hold out the last 200 hours for testing.
    """)


@app.cell
def _(X_actual, train_test_split, y):
    y_train, y_test, X_actual_train, X_actual_test = train_test_split(y, X_actual, test_size=200)
    forecasting_horizon = 24

    print(f"Training: {len(y_train)} obs")
    print(f"Test: {len(y_test)} obs")
    return X_actual_test, X_actual_train, forecasting_horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Fit a Class-Probability Forecaster

    [`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.reduction.ClassProbaReductionForecaster/) wraps any sklearn classifier (here, a Decision Tree) and
    converts the time series problem into tabular classification:

    1. **Encode**: String labels are mapped to numeric codes
    2. **Tabularize**: Lag features are created from past values
    3. **Fit**: The classifier trains on the tabular data
    4. **Predict**: Recursive probability forecasting
    """)


@app.cell
def _(
    ClassProbaReductionForecaster,
    DecisionTreeClassifier,
    LagTransformer,
    X_actual_train,
    forecasting_horizon,
    y_train,
):
    forecaster = ClassProbaReductionForecaster(
        estimator=DecisionTreeClassifier(random_state=42),
        feature_transformer=LagTransformer(lag=[1, 2, 3, 6, 12, 24]),
    )
    forecaster.fit(y_train, X_actual_train, forecasting_horizon=forecasting_horizon)

    print(f"Discovered classes: {forecaster.classes_}")
    return (forecaster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Probability Predictions

    `predict_class_proba()` returns one column per class, with probabilities
    summing to 1 at each time step. `predict()` returns the most likely class.
    """)


@app.cell
def _(forecaster, forecasting_horizon):
    y_proba = forecaster.predict_class_proba(
        forecasting_horizon=forecasting_horizon,
    )
    print("Probability predictions (first 12 steps):")
    y_proba
    return (y_proba,)


@app.cell
def _(forecaster, forecasting_horizon):
    y_pred = forecaster.predict(
        forecasting_horizon=forecasting_horizon,
    )
    print("Class label predictions:")
    y_pred
    return (y_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Visualize Forecasts

    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/)
    auto-detects class-probability and categorical prediction columns.

    - **Probability predictions** are rendered as stacked area charts showing
      how the probability mass shifts across classes over the forecast horizon.
      Diamond markers highlight the true class from `y_test`.
    - **Categorical predictions** (from `predict()`) are rendered as step
      charts comparing predicted vs actual class labels.
    """)


@app.cell
def _(plot_forecast, y_proba, y_test):
    plot_forecast(
        y_test,
        y_proba,
        title="Probability Forecast (Stacked Area)",
    )


@app.cell
def _(plot_forecast, y_pred, y_test):
    plot_forecast(
        y_test,
        y_pred,
        title="Categorical Forecast (Step Chart)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Evaluate with Metrics

    Yohou provides three class-probability metrics:

    - [`LogLoss`](/pages/api/generated/yohou.metrics.class_proba.LogLoss/): Measures how well predicted probabilities match the true distribution (lower is better)
    - [`BrierScore`](/pages/api/generated/yohou.metrics.class_proba.BrierScore/): Mean squared error between predicted probabilities and one-hot truth (lower is better)
    - [`Accuracy`](/pages/api/generated/yohou.metrics.class_proba.Accuracy/): Fraction of correct argmax predictions (higher is better)
    """)


@app.cell
def _(Accuracy, BrierScore, LogLoss, y_proba, y_test):
    y_truth_slice = y_test.head(len(y_proba))

    log_loss = LogLoss().fit(y_truth_slice)
    brier = BrierScore().fit(y_truth_slice)
    accuracy = Accuracy().fit(y_truth_slice)

    print(f"Log Loss:    {log_loss.score(y_truth_slice, y_proba):.4f}")
    print(f"Brier Score: {brier.score(y_truth_slice, y_proba):.4f}")
    print(f"Accuracy:    {accuracy.score(y_truth_slice, y_proba):.4f}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Rolling Observe-Predict

    The `observe_predict_class_proba()` method performs rolling evaluation:
    observe new data, then predict the next horizon. This simulates real-world
    deployment where predictions are updated as new observations arrive.
    """)


@app.cell
def _(X_actual_test, forecaster, y_test):
    y_rolling_proba = forecaster.observe_predict_class_proba(
        y=y_test,
        X_actual=X_actual_test,
    ).sort("time")
    print(f"Rolling predictions: {len(y_rolling_proba)} rows")
    return (y_rolling_proba,)


@app.cell
def _(y_rolling_proba):
    y_rolling_proba


@app.cell
def _(plot_forecast, y_rolling_proba, y_test):
    plot_forecast(
        y_test,
        y_rolling_proba,
        title="Rolling Probability Forecast",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Score Evolution Over Time

    Per-timestep LogLoss and BrierScore reveal when the model is most
    uncertain. Spikes indicate time steps where predictions deviated
    most from reality.

    [`plot_score_time_series`](/pages/api/generated/yohou.plotting.evaluation.plot_score_time_series/)
    shows how forecast quality varies across time steps. Spikes reveal periods
    where the model struggles, helping diagnose whether errors are random or
    systematic.
    """)


@app.cell
def _(LogLoss, plot_score_time_series, y_rolling_proba, y_test):
    plot_score_time_series(
        LogLoss(),
        y_test,
        y_rolling_proba,
        title="Log Loss Over Time (Rolling Predictions)",
    )


@app.cell
def _(BrierScore, plot_score_time_series, y_rolling_proba, y_test):
    plot_score_time_series(
        BrierScore(),
        y_test,
        y_rolling_proba,
        title="Brier Score Over Time (Rolling Predictions)",
    )


@app.cell
def _(Accuracy, plot_score_time_series, y_rolling_proba, y_test):
    plot_score_time_series(
        Accuracy(),
        y_test,
        y_rolling_proba,
        title="Accuracy Over Time (Rolling Predictions)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Calibration Plot

    [`plot_calibration`](/pages/api/generated/yohou.plotting.evaluation.plot_calibration/)
    automatically detects class-probability columns and renders a calibration
    plot. Points near the diagonal indicate good calibration; points below
    mean the model is overconfident for that class.
    """)


@app.cell
def _(plot_calibration, y_rolling_proba, y_test):
    plot_calibration(
        y_rolling_proba,
        y_test,
        n_bins=8,
        title="Calibration Plot (Rolling Predictions)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - Try different classifiers: `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`
    - Experiment with `reduction_strategy="direct"` for independent step classifiers
    - Add more lag features with [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/)
    - Explore [Metrics](/examples/#metrics) for more evaluation options
    - See [`reduction_forecaster.py`](/examples/point/reduction_forecaster/) for the regression equivalent
    """)


if __name__ == "__main__":
    app.run()
