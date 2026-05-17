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
    "title": "Class-Probability Forecasting",
    "description": "Forecast air quality categories using ClassProbaReductionForecaster, producing a probability distribution over four WHO air quality classes.",
    "category": "tutorial",
    "companion": "/pages/tutorials/class-proba-forecasting/",
    "section": "getting-started",
    "api_references": ["Accuracy", "BrierScore", "ClassProbaReductionForecaster", "FeaturePipeline"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Class-Probability Forecasting

        In this notebook, we will forecast air quality categories using
        [`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.reduction.ClassProbaReductionForecaster/),
        producing a probability distribution over four WHO air quality classes.
        We load a real Beijing PM2.5 dataset, fit a Random Forest classifier,
        and evaluate the probabilistic output with Brier score and accuracy.

        **Prerequisites:** Completed [Getting Started](/pages/tutorials/getting-started/).
        """
    )


@app.cell(hide_code=True)
def _():
    from sklearn.ensemble import RandomForestClassifier

    from yohou.class_proba import ClassProbaReductionForecaster
    from yohou.compose import FeaturePipeline
    from yohou.datasets import fetch_air_quality_classification
    from yohou.metrics import Accuracy, BrierScore
    from yohou.preprocessing import LagTransformer

    return (
        Accuracy,
        BrierScore,
        ClassProbaReductionForecaster,
        FeaturePipeline,
        LagTransformer,
        RandomForestClassifier,
        fetch_air_quality_classification,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 1. Load and Inspect the Data")


@app.cell
def _(fetch_air_quality_classification):
    bunch = fetch_air_quality_classification()
    y = bunch.y
    X_actual = bunch.X_actual
    print(f"Series length: {len(y)} hours")
    print(f"Classes: {bunch.classes}")
    print(f"Features: {bunch.feature_names}")
    y.head(3)
    return X_actual, bunch, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 2. Train/Test Split")


@app.cell
def _(X_actual, y):
    from yohou.model_selection import train_test_split

    forecasting_horizon = 24

    y_train, y_test, X_train, X_test = train_test_split(
        y, X_actual, test_size=forecasting_horizon
    )

    print(f"Train: {len(y_train)} hours, Test: {len(y_test)} hours")
    print("Test class distribution:")
    print(y_test["air_quality"].value_counts())
    return X_test, X_train, forecasting_horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 3. Fit ClassProbaReductionForecaster")


@app.cell
def _(
    ClassProbaReductionForecaster,
    FeaturePipeline,
    LagTransformer,
    RandomForestClassifier,
    X_train,
    forecasting_horizon,
    y_train,
):
    forecaster = ClassProbaReductionForecaster(
        estimator=RandomForestClassifier(n_estimators=50, random_state=42),
        feature_transformer=FeaturePipeline([
            ("lags", LagTransformer(lag=[1, 2, 3, 24])),
        ]),
    )
    forecaster.fit(y_train, forecasting_horizon=forecasting_horizon, X_actual=X_train)
    print("Fitted ClassProbaReductionForecaster")
    return (forecaster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 4. Predict Class Probabilities")


@app.cell
def _(X_test, forecaster, forecasting_horizon):
    y_pred_proba = forecaster.predict_class_proba(
        forecasting_horizon=forecasting_horizon,
        X_actual=X_test,
    )
    print("Probability columns:", [c for c in y_pred_proba.columns if "proba" in c])
    y_pred_proba
    return (y_pred_proba,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 5. Evaluate")


@app.cell
def _(Accuracy, BrierScore, y_pred_proba, y_test, y_train):
    brier = BrierScore()
    brier.fit(y_train)

    accuracy = Accuracy()
    accuracy.fit(y_train)

    brier_score = brier.score(y_test, y_pred_proba)
    acc_score = accuracy.score(y_test, y_pred_proba)

    print(f"Brier score: {brier_score:.3f}  (lower is better)")
    print(f"Accuracy:    {acc_score:.3f}  (higher is better)")
    return acc_score, accuracy, brier, brier_score


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [Class-Probability Forecasting](/pages/tutorials/class-proba-forecasting/) for the full guide
        - [Ensemble Forecasting](/pages/how-to/ensemble-forecasting/) for related techniques
        """
    )
    return


if __name__ == "__main__":
    app.run()
