# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Class-Probability Forecasting

    This notebook demonstrates **class-probability forecasting** - predicting the
    probability distribution over categorical outcomes at future time steps.

    ## What You'll Learn

    - How [`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.reduction.ClassProbaReductionForecaster/) converts categorical time series into a classification problem
    - Obtaining probability predictions with `predict_class_proba()` and class labels with `predict()`
    - Evaluating predictions with [`LogLoss`](/pages/api/generated/yohou.metrics.class_proba.LogLoss/), [`BrierScore`](/pages/api/generated/yohou.metrics.class_proba.BrierScore/), and [`Accuracy`](/pages/api/generated/yohou.metrics.class_proba.Accuracy/)
    - Visualizing class probabilities with [`plot_class_probabilities`](/pages/api/generated/yohou.plotting.evaluation.plot_class_probabilities/)
    - Using the observe-predict workflow for rolling evaluation

    ## Prerequisites

    Basic familiarity with sklearn's fit/predict API and classification concepts.
    """)
    return


@app.cell(hide_code=True)
def _():
    from sklearn.tree import DecisionTreeClassifier

    from yohou.class_proba import ClassProbaReductionForecaster
    from yohou.datasets import make_weather_classification
    from yohou.metrics import Accuracy, BrierScore, LogLoss
    from yohou.plotting import plot_class_probabilities, plot_time_series
    from yohou.preprocessing import LagTransformer

    return (
        Accuracy,
        BrierScore,
        ClassProbaReductionForecaster,
        DecisionTreeClassifier,
        LagTransformer,
        LogLoss,
        make_weather_classification,
        plot_class_probabilities,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load the Data

    We use [`make_weather_classification`](/pages/api/generated/yohou.datasets._generators.make_weather_classification/) to generate a synthetic weather dataset
    with three classes: sunny, rainy, and cloudy. The target is driven by
    seasonal temperature and humidity features.
    """)
    return


@app.cell
def _(make_weather_classification):
    data = make_weather_classification(length=365, seed=42)
    y, X = data.y, data.X

    print(f"Classes: {data.classes}")
    print(f"Dataset: {len(y)} observations from {y['time'].min()} to {y['time'].max()}")
    y.head(10)
    return X, data, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Train/Test Split

    We hold out the last 60 days for testing.
    """)
    return


@app.cell
def _(X, y):
    split_point = len(y) - 60
    y_train, y_test = y[:split_point], y[split_point:]
    X_train, X_test = X[:split_point], X[split_point:]
    forecasting_horizon = 7

    print(f"Training: {len(y_train)} obs")
    print(f"Test: {len(y_test)} obs")
    return X_test, X_train, forecasting_horizon, split_point, y_test, y_train


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
    return


@app.cell
def _(
    ClassProbaReductionForecaster,
    DecisionTreeClassifier,
    LagTransformer,
    X_train,
    forecasting_horizon,
    y_train,
):
    forecaster = ClassProbaReductionForecaster(
        estimator=DecisionTreeClassifier(random_state=42),
        feature_transformer=LagTransformer(lag=[1, 2, 3, 7]),
    )
    forecaster.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)

    print(f"Discovered classes: {forecaster.classes_}")
    return (forecaster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Probability Predictions

    `predict_class_proba()` returns one column per class, with probabilities
    summing to 1 at each time step. `predict()` returns the most likely class.
    """)
    return


@app.cell
def _(X_test, forecaster, forecasting_horizon):
    y_proba = forecaster.predict_class_proba(
        X=X_test[:forecasting_horizon],
        forecasting_horizon=forecasting_horizon,
    )
    print("Probability predictions:")
    y_proba
    return (y_proba,)


@app.cell
def _(X_test, forecaster, forecasting_horizon):
    y_pred = forecaster.predict(
        X=X_test[:forecasting_horizon],
        forecasting_horizon=forecasting_horizon,
    )
    print("Class label predictions:")
    y_pred
    return (y_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Visualize Probabilities

    [`plot_class_probabilities`](/pages/api/generated/yohou.plotting.evaluation.plot_class_probabilities/) shows how predicted probabilities evolve over
    the forecast horizon. With `y_truth`, the true class is overlaid as diamond
    markers at the probability assigned to the correct class.
    """)
    return


@app.cell
def _(plot_class_probabilities, y_proba):
    plot_class_probabilities(y_proba, title="7-Day Weather Probability Forecast")
    return


@app.cell
def _(plot_class_probabilities, y_proba, y_test):
    plot_class_probabilities(
        y_proba,
        y_truth=y_test,
        title="Forecast vs Actual Weather",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Evaluate with Metrics

    Yohou provides three class-probability metrics:

    - [`LogLoss`](/pages/api/generated/yohou.metrics.class_proba.LogLoss/): Measures how well predicted probabilities match the true distribution (lower is better)
    - [`BrierScore`](/pages/api/generated/yohou.metrics.class_proba.BrierScore/): Mean squared error between predicted probabilities and one-hot truth (lower is better)
    - [`Accuracy`](/pages/api/generated/yohou.metrics.class_proba.Accuracy/): Fraction of correct argmax predictions (higher is better)
    """)
    return


@app.cell
def _(Accuracy, BrierScore, LogLoss, y_proba, y_test):
    y_truth_slice = y_test.head(len(y_proba))

    log_loss = LogLoss().fit(y_truth_slice)
    brier = BrierScore().fit(y_truth_slice)
    accuracy = Accuracy().fit(y_truth_slice)

    print(f"Log Loss:    {log_loss.score(y_truth_slice, y_proba):.4f}")
    print(f"Brier Score: {brier.score(y_truth_slice, y_proba):.4f}")
    print(f"Accuracy:    {accuracy.score(y_truth_slice, y_proba):.4f}")
    return accuracy, brier, log_loss, y_truth_slice


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Rolling Observe-Predict

    The `observe_predict_class_proba()` method performs rolling evaluation:
    observe new data, then predict the next horizon. This simulates real-world
    deployment where predictions are updated as new observations arrive.
    """)
    return


@app.cell
def _(X_test, forecaster, plot_class_probabilities, y_test):
    y_rolling_proba = forecaster.observe_predict_class_proba(
        y=y_test,
        X=X_test,
    ).sort("time")
    print(f"Rolling predictions: {len(y_rolling_proba)} rows")
    plot_class_probabilities(
        y_rolling_proba,
        y_truth=y_test,
        title="Rolling Observe-Predict Probabilities",
    )
    return (y_rolling_proba,)


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
    return


if __name__ == "__main__":
    app.run()
