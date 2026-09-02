# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lightgbm",
#     "scikit-learn",
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "How to Enable Early Stopping",
    "description": "Hold out a validation tail with validation_size so LightGBM stops training when validation performance plateaus, for point and interval forecasters.",
    "category": "how-to",
    "section": "forecasting-models",
    "companion": "/pages/how-to/early-stopping/",
    "api_references": [
        "PointReductionForecaster",
        "IntervalReductionForecaster",
        "LagTransformer",
        "fetch_tourism_monthly",
        "plot_forecast",
    ],
}

app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How to Enable Early Stopping

    This notebook shows how to hold out a validation tail with
    `validation_size` so a LightGBM estimator stops training when its
    validation performance plateaus.

    **Prerequisites:** Familiarity with reduction forecasters
    ([View](/examples/forecasting-models/catboost_forecasting/) ·
    [Open in marimo](/examples/forecasting-models/catboost_forecasting/edit/)).

    ## 1. Prepare Data

    Load the Monthly Tourism dataset, keep one series, and split off a test
    set while preserving temporal order.
    """)


@app.cell(hide_code=True)
def _():
    from lightgbm import LGBMRegressor

    from yohou.datasets import fetch_tourism_monthly
    from yohou.interval import IntervalReductionForecaster
    from yohou.model_selection import train_test_split
    from yohou.plotting import plot_forecast
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        IntervalReductionForecaster,
        LGBMRegressor,
        LagTransformer,
        PointReductionForecaster,
        fetch_tourism_monthly,
        plot_forecast,
        train_test_split,
    )


@app.cell
def _(fetch_tourism_monthly, train_test_split):
    y = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    y_train, y_test = train_test_split(y, test_size=24)
    return y, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Configure Stopping and Hold Out a Tail

    Configure early stopping on the estimator (`early_stopping_round`), then
    set `validation_size` on the forecaster. The last 36 points of the
    training series become the evaluation set LightGBM scores each boosting
    round against.
    """)


@app.cell
def _(LGBMRegressor, LagTransformer, PointReductionForecaster, y_train):
    forecaster = PointReductionForecaster(
        estimator=LGBMRegressor(
            n_estimators=500,
            early_stopping_round=20,
            min_child_samples=5,
            verbose=-1,
        ),
        reduction_strategy="direct",
        actual_transformer=LagTransformer(lag=[1, 2, 12]),
        validation_size=36,
    )
    forecaster.fit(y=y_train, forecasting_horizon=12)
    return (forecaster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Read the Stopping Result

    With `reduction_strategy="direct"`, `estimator_` is a list with one
    fitted estimator per horizon step; each carries its own
    `best_iteration_`.
    """)


@app.cell
def _(forecaster):
    best_iterations = [est.best_iteration_ for est in forecaster.estimator_]
    best_iterations
    return (best_iterations,)


@app.cell(hide_code=True)
def _(best_iterations, mo):
    mo.md(
        f"""
        Requested boosting rounds: 500

        Best iterations per horizon step: {best_iterations}

        Every step stopped well below the maximum, so the held-out tail did
        its job.
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Forecast Past the Holdout

    The forecaster observed the held-out tail after training, so
    `predict()` continues from the end of the full training series.
    """)


@app.cell
def _(forecaster, plot_forecast, y_test, y_train):
    y_pred = forecaster.predict()
    plot_forecast(
        y_test=y_test,
        y_pred=y_pred.join(y_test, on="time", how="semi"),
        y_train=y_train,
        n_history=48,
        title="LightGBM forecast after early stopping",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Early-Stop an Interval Forecaster

    If you forecast intervals, use a quantile objective. The holdout splits
    once and every quantile estimator receives the same evaluation set,
    each stopping on its own quantile loss.
    """)


@app.cell
def _(IntervalReductionForecaster, LGBMRegressor, LagTransformer, y_train):
    interval_forecaster = IntervalReductionForecaster(
        estimator=LGBMRegressor(
            objective="quantile",
            alpha=0.5,
            n_estimators=500,
            early_stopping_round=20,
            min_child_samples=5,
            verbose=-1,
        ),
        reduction_strategy="direct",
        actual_transformer=LagTransformer(lag=[1, 2, 12]),
        validation_size=36,
    )
    interval_forecaster.fit(y=y_train, forecasting_horizon=12, coverage_rates=[0.9])
    return (interval_forecaster,)


@app.cell
def _(interval_forecaster):
    bound_best = {
        bound: [est.best_iteration_ for est in estimators]
        for bound, estimators in interval_forecaster.estimator_.items()
    }
    bound_best
    return (bound_best,)


@app.cell(hide_code=True)
def _(bound_best, mo):
    mo.md(
        f"""
        Best iterations per bound and step: {bound_best}

        Most steps stop well below the 500-round maximum. A step that reaches
        500 never triggered the 20-round patience, so its validation loss was
        still improving when the budget ran out; raise `n_estimators` for that
        chain if you want it to stop on its own.
        """
    )


if __name__ == "__main__":
    app.run()
