# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.23.1"
__gallery__ = {
    "title": "How to Create a Custom Estimator",
    "description": "Implement a LastValueForecaster from scratch, validate it with the check generator, and use it in a forecast pipeline.",
    "category": "how-to",
    "companion": "/pages/how-to/create-a-point-forecaster/",
    "section": "getting-started",
    "api_references": ["BasePointForecaster", "MeanAbsoluteError", "SeasonalNaive"],
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
        # How to Create a Custom Estimator

        This notebook implements a `LastValueForecaster` from scratch,
        validates it using the built-in check generator, and compares
        its performance against [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/).

        **Prerequisites:** Familiarity with the fit/predict API
        ([Getting Started](/pages/tutorials/getting-started/)).
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 1. Implement the Forecaster")


@app.cell
def _():
    import polars as pl
    import polars.selectors as cs

    from yohou.point.base import BasePointForecaster

    class LastValueForecaster(BasePointForecaster):
        """Repeats the last observed value for every forecast step."""

        _tags = {"ignores_exogenous": True, "stateful": True}

        @property
        def _observation_horizon(self):
            return 1

        def _predict_one(self, groups, **params):
            last_value = self._y_observed.select(~cs.by_name("time")).row(-1)[0]
            return pl.DataFrame(
                {self._y_columns[0]: [last_value] * self.fit_forecasting_horizon_}
            )

    return LastValueForecaster, cs, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 2. Fit and Predict")


@app.cell
def _(LastValueForecaster, pl):
    from yohou.datasets import fetch_sunspot
    from yohou.model_selection import train_test_split

    bunch = fetch_sunspot()
    y = bunch.frame.group_by_dynamic("time", every="1mo").agg(
        pl.col("sunspot_number").mean()
    )

    forecasting_horizon = 24
    y_train, y_test = train_test_split(y, test_size=forecasting_horizon)

    forecaster = LastValueForecaster()
    forecaster.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred = forecaster.predict(forecasting_horizon=forecasting_horizon)
    y_pred.head()

    return bunch, forecaster, forecasting_horizon, y, y_pred, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 3. Score Against a Baseline")


@app.cell
def _(y_pred, y_test, y_train):
    from yohou.metrics import MeanAbsoluteError
    from yohou.point import SeasonalNaive

    scorer = MeanAbsoluteError()
    scorer.fit(y_train)

    baseline = SeasonalNaive(seasonality=12)
    baseline.fit(y_train, forecasting_horizon=len(y_test))
    y_pred_baseline = baseline.predict(forecasting_horizon=len(y_test))

    score_custom = scorer.score(y_test, y_pred)
    score_baseline = scorer.score(y_test, y_pred_baseline)
    print(f"LastValue MAE: {score_custom:.2f}")
    print(f"SeasonalNaive MAE: {score_baseline:.2f}")

    return (
        MeanAbsoluteError,
        SeasonalNaive,
        baseline,
        score_baseline,
        score_custom,
        scorer,
        y_pred_baseline,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 4. Plot the Comparison")


@app.cell
def _(y_pred, y_pred_baseline, y_test, y_train):
    from yohou.plotting import plot_forecast

    fig = plot_forecast(
        y_train=y_train,
        y_test=y_test,
        y_pred={"LastValue": y_pred, "SeasonalNaive": y_pred_baseline},
    )
    fig

    return (fig,)



@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [Create a Point Forecaster](/pages/how-to/create-a-point-forecaster/) for the full guide
        - [Build Reduction Forecasters](/pages/how-to/build-reduction-forecasters/) for related techniques
        """
    )
    return

if __name__ == "__main__":
    app.run()
