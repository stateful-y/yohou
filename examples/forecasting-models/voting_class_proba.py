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
    "title": "How to Combine Classification Forecasters",
    "description": "Build classification ensembles with VotingClassProbaForecaster using soft and hard voting strategies.",
    "category": "how-to",
    "companion": "/pages/how-to/ensemble-forecasting/",
    "section": "forecasting-models",
    "api_references": ["Accuracy", "BrierScore", "ClassProbaReductionForecaster", "LogLoss", "VotingClassProbaForecaster"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How to Combine Classification Forecasters with VotingClassProbaForecaster

    This notebook shows how to combine multiple classification forecasters
    into a single ensemble using [`VotingClassProbaForecaster`](/pages/api/generated/yohou.ensemble.VotingClassProbaForecaster/).

    **Prerequisites:** Familiarity with class-probability forecasting. See the
    [Class-Probability Forecasting Quickstart](/examples/forecasting-models/class_proba_forecaster/).
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.tree import DecisionTreeClassifier

    from yohou.class_proba import ClassProbaReductionForecaster
    from yohou.datasets import fetch_air_quality_classification
    from yohou.ensemble import VotingClassProbaForecaster
    from yohou.metrics import Accuracy, BrierScore, LogLoss
    from yohou.plotting import plot_forecast
    from yohou.preprocessing import LagTransformer

    return (
        Accuracy,
        BrierScore,
        ClassProbaReductionForecaster,
        DecisionTreeClassifier,
        GradientBoostingClassifier,
        LagTransformer,
        LogLoss,
        LogisticRegression,
        MultiOutputClassifier,
        RandomForestClassifier,
        VotingClassProbaForecaster,
        deepcopy,
        fetch_air_quality_classification,
        plot_forecast,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load and Prepare Data

    Air quality classification with 4 categories (good / moderate / unhealthy / hazardous).
    """)


@app.cell
def _(fetch_air_quality_classification, train_test_split):
    y = fetch_air_quality_classification().y
    forecasting_horizon = 14
    y_train, y_test = train_test_split(y, test_size=forecasting_horizon, shuffle=False)

    print(f"Training: {len(y_train)} obs, Test: {len(y_test)} obs")
    print(f"Target column: {[c for c in y.columns if c != 'time']}")
    return forecasting_horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Define Base Forecasters

    Create diverse classifiers wrapped in [`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.reduction.ClassProbaReductionForecaster/).
    """)


@app.cell
def _(
    ClassProbaReductionForecaster,
    DecisionTreeClassifier,
    GradientBoostingClassifier,
    LagTransformer,
    LogisticRegression,
    MultiOutputClassifier,
    RandomForestClassifier,
):
    base_forecasters = [
        (
            "logistic",
            ClassProbaReductionForecaster(
                estimator=MultiOutputClassifier(LogisticRegression(max_iter=500)),
                actual_transformer=LagTransformer(lag=[1, 2, 3, 7]),
            ),
        ),
        (
            "tree",
            ClassProbaReductionForecaster(
                estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
                actual_transformer=LagTransformer(lag=[1, 2, 3, 7]),
            ),
        ),
        (
            "rf",
            ClassProbaReductionForecaster(
                estimator=RandomForestClassifier(n_estimators=50, random_state=42),
                actual_transformer=LagTransformer(lag=[1, 2, 3, 7]),
            ),
        ),
        (
            "gbm",
            ClassProbaReductionForecaster(
                estimator=MultiOutputClassifier(
                    GradientBoostingClassifier(n_estimators=50, random_state=42)
                ),
                actual_transformer=LagTransformer(lag=[1, 2, 3, 7]),
            ),
        ),
    ]
    return (base_forecasters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Soft Voting Ensemble

    Pass the named forecasters to [`VotingClassProbaForecaster`](/pages/api/generated/yohou.ensemble.voting_class_proba.VotingClassProbaForecaster/) with `method="soft"`.
    Soft voting averages the predicted class probabilities.
    """)


@app.cell
def _(VotingClassProbaForecaster, base_forecasters, forecasting_horizon, y_train):
    ensemble_soft = VotingClassProbaForecaster(
        forecasters=base_forecasters,
        method="soft",
    )
    ensemble_soft.fit(y_train, forecasting_horizon=forecasting_horizon)
    return (ensemble_soft,)


@app.cell
def _(deepcopy, ensemble_soft, forecasting_horizon, plot_forecast, y_test, y_train):
    y_pred_soft = deepcopy(ensemble_soft).observe_predict(y_test, forecasting_horizon=forecasting_horizon)
    plot_forecast(y_test, y_pred_soft, y_train=y_train, n_history=60, title="Soft Voting Ensemble")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Hard Voting Ensemble

    Hard voting takes the majority class from each forecaster's argmax prediction.
    """)


@app.cell
def _(VotingClassProbaForecaster, base_forecasters, forecasting_horizon, y_train):
    ensemble_hard = VotingClassProbaForecaster(
        forecasters=base_forecasters,
        method="hard",
    )
    ensemble_hard.fit(y_train, forecasting_horizon=forecasting_horizon)
    return (ensemble_hard,)


@app.cell
def _(deepcopy, ensemble_hard, forecasting_horizon, plot_forecast, y_test, y_train):
    y_pred_hard = deepcopy(ensemble_hard).observe_predict(y_test, forecasting_horizon=forecasting_horizon)
    plot_forecast(y_test, y_pred_hard, y_train=y_train, n_history=60, title="Hard Voting Ensemble")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Weighted Soft Voting

    Assign higher weight to models you expect to be better calibrated.
    """)


@app.cell
def _(VotingClassProbaForecaster, base_forecasters, forecasting_horizon, y_train):
    ensemble_weighted = VotingClassProbaForecaster(
        forecasters=base_forecasters,
        method="soft",
        weights=[0.5, 0.5, 2.0, 2.0],
    )
    ensemble_weighted.fit(y_train, forecasting_horizon=forecasting_horizon)
    return (ensemble_weighted,)


@app.cell
def _(deepcopy, ensemble_weighted, forecasting_horizon, plot_forecast, y_test, y_train):
    y_pred_weighted = deepcopy(ensemble_weighted).observe_predict(y_test, forecasting_horizon=forecasting_horizon)
    plot_forecast(y_test, y_pred_weighted, y_train=y_train, n_history=60, title="Weighted Soft Voting Ensemble")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Compare Ensemble Strategies

    Evaluate all three strategies with [`LogLoss`](/pages/api/generated/yohou.metrics.class_proba.LogLoss/), [`BrierScore`](/pages/api/generated/yohou.metrics.class_proba.BrierScore/), and [`Accuracy`](/pages/api/generated/yohou.metrics.classification.Accuracy/).
    """)


@app.cell
def _(
    Accuracy,
    BrierScore,
    LogLoss,
    deepcopy,
    ensemble_hard,
    ensemble_soft,
    ensemble_weighted,
    forecasting_horizon,
    mo,
    y_test,
    y_train,
):
    log_loss = LogLoss()
    brier = BrierScore()
    accuracy = Accuracy()

    for _scorer in [log_loss, brier, accuracy]:
        _scorer.fit(y_train)

    ensembles = {
        "Soft": ensemble_soft,
        "Hard": ensemble_hard,
        "Weighted Soft": ensemble_weighted,
    }

    rows = []
    for name, model in ensembles.items():
        y_pred = deepcopy(model).observe_predict_class_proba(y_test, forecasting_horizon=forecasting_horizon)
        rows.append(
            {
                "Strategy": name,
                "LogLoss": f"{log_loss.score(y_test, y_pred):.4f}",
                "BrierScore": f"{brier.score(y_test, y_pred):.4f}",
                "Accuracy": f"{accuracy.score(y_test, y_pred):.4f}",
            }
        )

    mo.ui.table(rows, selection=None)



@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [Ensemble Forecasting](/pages/how-to/ensemble-forecasting/) for the full guide
        - [Class-Probability Forecasting Tutorial](/pages/tutorials/class-proba-forecasting/) for related techniques
        """
    )
    return

if __name__ == "__main__":
    app.run()
