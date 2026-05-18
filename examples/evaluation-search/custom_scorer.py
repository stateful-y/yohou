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
    "title": "How to Create a Custom Scorer",
    "description": "Implement a custom point scorer with aggregation, panel support, and systematic testing.",
    "category": "how-to",
    "companion": "/pages/how-to/create-a-scorer/",
    "section": "evaluation-search",
    "api_references": ["BasePointScorer", "GridSearchCV", "PointReductionForecaster", "SeasonalNaive"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How to Create a Custom Scorer

    This notebook shows how to implement a custom point scorer by extending
    [`BasePointScorer`](/pages/api/generated/yohou.metrics.base.BasePointScorer/),
    integrating with Yohou's aggregation, panel dispatch, and
    cross-validation machinery.

    **Prerequisites:** Familiarity with the built-in scorers
    ([Point Metrics](/examples/evaluation-search/point_metrics/) · [Evaluate Forecast Accuracy](/pages/how-to/evaluate-forecast-accuracy/)).
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_tourism_monthly
    from yohou.metrics import MeanAbsoluteError
    from yohou.metrics.base import BasePointScorer
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer

    return (
        BasePointScorer,
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        SeasonalNaive,
        fetch_tourism_monthly,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Define the Scorer

    Subclass [`BasePointScorer`](/pages/api/generated/yohou.metrics.base.BasePointScorer/) and implement `score`. The method validates
    inputs, computes per-timestep raw scores, and delegates aggregation
    to the base class.
    """)


@app.cell
def _(BasePointScorer, pl):
    class MaxAbsoluteError(BasePointScorer):
        """Maximum absolute error across the forecast horizon."""

        _parameter_constraints: dict = {
            **BasePointScorer._parameter_constraints,
        }

        _metric_name = "max_ae"

        def __init__(
            self,
            aggregation_method="all",
            groups=None,
            components=None,
        ):
            super().__init__(
                aggregation_method=aggregation_method,
                groups=groups,
                components=components,
            )

        def _compute_raw_errors(self, y_truth, y_pred):
            return (y_truth - y_pred).select(pl.all().abs())

    return (MaxAbsoluteError,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Prepare Test Data

    Fit two forecasters on the Tourism Monthly dataset to produce
    predictions for scoring.
    """)


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
        .frame.select("time", "T1__tourists")
        .drop_nulls()
        .rename({"T1__tourists": "tourists"})
    )

    from yohou.model_selection import train_test_split

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

    return fh, y_pred_naive, y_pred_ridge, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Score with the Custom Metric

    Use the same `fit` / `score` pattern as built-in scorers.
    """)


@app.cell
def _(MaxAbsoluteError, y_pred_naive, y_pred_ridge, y_test, y_train):
    max_ae = MaxAbsoluteError()
    max_ae.fit(y_train)

    print(f"MaxAE  Naive: {max_ae.score(y_test, y_pred_naive):.2f}")
    print(f"MaxAE  Ridge: {max_ae.score(y_test, y_pred_ridge):.2f}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Use Aggregation Modes

    All standard aggregation modes work automatically: `"stepwise"`,
    `"vintagewise"`, `"componentwise"`, and `"all"`.
    """)


@app.cell
def _(MaxAbsoluteError, y_pred_naive, y_test, y_train):
    stepwise = MaxAbsoluteError(aggregation_method="stepwise")
    stepwise.fit(y_train)
    stepwise.score(y_test, y_pred_naive)


@app.cell
def _(MaxAbsoluteError, y_pred_naive, y_test, y_train):
    componentwise = MaxAbsoluteError(aggregation_method="componentwise")
    componentwise.fit(y_train)
    componentwise.score(y_test, y_pred_naive)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Compare Against Built-in Metrics

    The custom scorer plugs into the same comparison workflows as
    built-in metrics.
    """)


@app.cell
def _(MaxAbsoluteError, MeanAbsoluteError, y_pred_naive, y_pred_ridge, y_test, y_train):
    scorers = {
        "MAE": MeanAbsoluteError(),
        "MaxAE": MaxAbsoluteError(),
    }

    for name, scorer in scorers.items():
        scorer.fit(y_train)
        naive_score = scorer.score(y_test, y_pred_naive)
        ridge_score = scorer.score(y_test, y_pred_ridge)
        print(f"{name:<6}  Naive: {naive_score:8.2f}  Ridge: {ridge_score:8.2f}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Use in Cross-Validation

    Custom scorers work directly with [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) and
    [`RandomizedSearchCV`](/pages/api/generated/yohou.model_selection.search.RandomizedSearchCV/).
    """)


@app.cell
def _(
    LagTransformer,
    MaxAbsoluteError,
    PointReductionForecaster,
    Ridge,
    y_train,
):
    from yohou.model_selection import ExpandingWindowSplitter, GridSearchCV

    search = GridSearchCV(
        forecaster=PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(lag=list(range(1, 13))),
        ),
        param_grid={"estimator__alpha": [0.1, 1.0, 10.0]},
        scoring=MaxAbsoluteError(),
        cv=ExpandingWindowSplitter(n_splits=3, test_size=len(y_train) // 5),
    )
    search.fit(y_train, forecasting_horizon=len(y_train) // 5)

    print(f"Best alpha: {search.best_params_['estimator__alpha']}")
    print(f"Best MaxAE: {search.best_score_:.2f}")

    return (search,)



@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [Create a Custom Scorer](/pages/how-to/create-a-scorer/) for the full guide
        - [Evaluate Forecast Accuracy](/pages/how-to/evaluate-forecast-accuracy/) for related techniques
        """
    )
    return

if __name__ == "__main__":
    app.run()
