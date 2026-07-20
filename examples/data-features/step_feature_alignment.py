# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.23.5"
__gallery__ = {
    "title": "How to Align Exogenous Features Across Pipeline Steps",
    "description": "Control which step-indexed columns each direct-strategy estimator sees using the step_feature_alignment parameter of PointReductionForecaster.",
    "category": "how-to",
    "companion": "/pages/how-to/exogenous-features/",
    "section": "data-features",
    "api_references": ["PointReductionForecaster", "make_exogenous_regression"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How to Control Step Feature Alignment

    This notebook shows how to use the `step_feature_alignment` parameter of
    [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/)
    to control which step-indexed columns each direct-strategy estimator
    sees during training and prediction. The synthetic data is generated with
    [`make_exogenous_regression`](/pages/api/generated/yohou.datasets._fetchers.make_exogenous_regression/).

    **Prerequisites:** Familiarity with the direct reduction strategy and
    exogenous features
    ([Exogenous Tutorial](/examples/exogenous_features/)).
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.ensemble import HistGradientBoostingRegressor

    from yohou.datasets import make_exogenous_regression
    from yohou.metrics import MeanAbsoluteError
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        HistGradientBoostingRegressor,
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        make_exogenous_regression,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    Generate synthetic data with temperature (X_actual), holidays
    (X_future), and weather forecasts (X_forecast). Use a 12-step
    horizon to make step column differences visible.
    """)


@app.cell
def _(make_exogenous_regression):
    H = 12
    data = make_exogenous_regression(n_samples=300, forecasting_horizon=H)
    y = data.y
    X_actual = data.X_actual
    X_future = data.X_future
    X_forecast = data.X_forecast

    from yohou.model_selection import train_test_split

    y_train, y_test, X_actual_train, _ = train_test_split(
        y, X_actual, test_size=H
    )

    print(f"Horizon: {H}, Train: {len(y_train)}, Test: {len(y_test)}")
    return H, X_actual_train, X_forecast, X_future, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Compare Alignment Modes

    The `step_feature_alignment` parameter only affects the `"direct"`
    strategy, where H independent estimators are fitted (one per step).

    | Mode | Estimator for step h sees |
    |------|--------------------------|
    | `"all"` | All step columns (`*_step_1` through `*_step_H`) |
    | `"matched"` | Only its own step column (`*_step_h`) |
    | `"cumulative"` | Step columns 1 through h (`*_step_1..h`) |

    Fit three forecasters with identical configurations except for
    `step_feature_alignment`.
    """)


@app.cell
def _(
    H,
    HistGradientBoostingRegressor,
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    X_actual_train,
    X_forecast,
    X_future,
    pl,
    y_test,
    y_train,
):
    results = []
    predictions = {}

    for _mode in ["all", "matched", "cumulative"]:
        forecaster = PointReductionForecaster(
            estimator=HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=42),
            actual_transformer=LagTransformer([1, 2, 3]),
            reduction_strategy="direct",
            step_feature_alignment=_mode,
        )
        forecaster.fit(
            y=y_train,
            X_actual=X_actual_train,
            forecasting_horizon=H,
            X_future=X_future,
            X_forecast=X_forecast,
        )
        pred = forecaster.predict()
        predictions[_mode] = pred

        scorer = MeanAbsoluteError()
        scorer.fit(y_train)
        mae = scorer.score(y_test, pred)
        results.append({"mode": _mode, "mae": f"{mae:.4f}"})
        print(f"step_feature_alignment={_mode:>12s}  MAE: {mae:.4f}")

    pl.DataFrame(results)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. When to Use Each Mode

    - **"all"** (default): each estimator can learn cross-step
      patterns (e.g., step 3 weather forecast differs systematically
      from step 1). Works well when step columns carry complementary
      signal across the horizon.

    - **"matched"**: reduces feature dimensionality by giving each
      estimator only its corresponding step column. Useful when step
      columns are noisy and including irrelevant steps hurts
      generalization.

    - **"cumulative"**: a middle ground. Early-step estimators see
      fewer features (less noise), later-step estimators accumulate
      context. Useful when earlier steps provide reliable information
      that later steps can build upon.
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Inspect Feature Matrices

    Verify the difference by checking how many features each step
    estimator receives.
    """)


@app.cell
def _(
    H,
    HistGradientBoostingRegressor,
    LagTransformer,
    PointReductionForecaster,
    X_actual_train,
    X_forecast,
    X_future,
    y_train,
):
    feature_counts = {}

    for _mode in ["all", "matched", "cumulative"]:
        fc = PointReductionForecaster(
            estimator=HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=42),
            actual_transformer=LagTransformer([1, 2, 3]),
            reduction_strategy="direct",
            step_feature_alignment=_mode,
        )
        fc.fit(
            y=y_train,
            X_actual=X_actual_train,
            forecasting_horizon=H,
            X_future=X_future,
            X_forecast=X_forecast,
        )
        # Check feature count for first and last step estimators
        n_features_first = fc.estimator_[0].n_features_in_
        n_features_last = fc.estimator_[-1].n_features_in_
        feature_counts[_mode] = (n_features_first, n_features_last)
        print(f"{_mode:>12s}: step 1 has {n_features_first} features, step {H} has {n_features_last} features")



@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [Use Exogenous Features](/pages/how-to/exogenous-features/) for the full guide
        - [Compose Feature Pipelines](/pages/how-to/compose-feature-pipelines/) for related techniques
        """
    )
    return

if __name__ == "__main__":
    app.run()
