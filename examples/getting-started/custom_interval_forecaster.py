# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.23.1"
__gallery__ = {
    "title": "How to Create a Custom Interval Forecaster",
    "description": "Implement a NaiveIntervalForecaster from scratch, validate it with the check generator, and compare it against SplitConformalForecaster.",
    "category": "how-to",
    "companion": "/pages/how-to/create-an-interval-forecaster/",
    "section": "getting-started",
    "api_references": ["BaseIntervalForecaster", "SplitConformalForecaster", "MeanIntervalWidth"],
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
        # How to Create a Custom Interval Forecaster

        This notebook implements a `NaiveIntervalForecaster` from scratch,
        validates it using the built-in check generator, and compares
        its interval predictions against
        [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/).

        **Prerequisites:** Familiarity with the fit/predict API
        ([Getting Started](/pages/tutorials/getting-started/)) and
        prediction intervals ([Produce Prediction Intervals](/pages/how-to/interval-forecasting/)).
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 1. Implement the Forecaster")


@app.cell
def _():
    import polars as pl
    import scipy.stats as st

    from yohou.interval.base import BaseIntervalForecaster
    from yohou.utils.tags import Tags

    class NaiveIntervalForecaster(BaseIntervalForecaster):
        """Produces intervals using historical mean and standard deviation."""

        def __sklearn_tags__(self) -> Tags:
            tags = super().__sklearn_tags__()
            tags.forecaster_tags.requires_exogenous = False
            tags.forecaster_tags.stateful = True
            return tags

        @property
        def _observation_horizon(self):
            return 10

        def _fit(self, y_t, X_t, forecasting_horizon):
            value_cols = [c for c in y_t.columns if c != "time"]
            self._stats = {}
            for col in value_cols:
                self._stats[col] = {
                    "mean": y_t[col].mean(),
                    "std": y_t[col].std(),
                }

        def _predict_one(self, groups, coverage_rates=None, **params):
            rates = coverage_rates or self.fit_coverage_rates_
            value_cols = list(self._stats.keys())
            h = self.fit_forecasting_horizon_

            data = {}
            for col in value_cols:
                mean = self._stats[col]["mean"]
                std_val = self._stats[col]["std"]
                for rate in rates:
                    z = st.norm.ppf(0.5 + rate / 2)
                    data[f"{col}_lower_{rate}"] = [mean - z * std_val] * h
                    data[f"{col}_upper_{rate}"] = [mean + z * std_val] * h

            y_pred = pl.DataFrame(data)
            y_pred = self._add_time_columns(y_pred)
            return y_pred

    return NaiveIntervalForecaster, Tags, pl, st


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 2. Fit and Predict Intervals")


@app.cell
def _(NaiveIntervalForecaster, pl):
    from yohou.datasets import fetch_sunspot
    from yohou.model_selection import train_test_split

    bunch = fetch_sunspot()
    y = bunch.frame.group_by_dynamic("time", every="1mo").agg(
        pl.col("sunspot_number").mean()
    )

    forecasting_horizon = 24
    y_train, y_test = train_test_split(y, test_size=forecasting_horizon)

    forecaster = NaiveIntervalForecaster()
    forecaster.fit(
        y_train,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=[0.9],
    )
    y_pred = forecaster.predict_interval(
        forecasting_horizon=forecasting_horizon,
        coverage_rates=[0.9],
    )
    y_pred.head()

    return bunch, forecaster, forecasting_horizon, y, y_pred, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 3. Score Against a Baseline")


@app.cell
def _(forecasting_horizon, y_pred, y_test, y_train):
    from yohou.interval import SplitConformalForecaster
    from yohou.metrics import MeanIntervalWidth
    from yohou.point import SeasonalNaive

    baseline = SplitConformalForecaster(
        point_forecaster=SeasonalNaive(seasonality=12),
    )
    baseline.fit(
        y_train,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=[0.9],
    )
    y_pred_baseline = baseline.predict_interval(
        forecasting_horizon=forecasting_horizon,
        coverage_rates=[0.9],
    )

    scorer = MeanIntervalWidth()
    scorer.fit(y_train)

    width_custom = scorer.score(y_test, y_pred)
    width_baseline = scorer.score(y_test, y_pred_baseline)
    print(f"NaiveInterval avg width: {width_custom:.2f}")
    print(f"SplitConformal avg width: {width_baseline:.2f}")

    return (
        MeanIntervalWidth,
        SplitConformalForecaster,
        baseline,
        scorer,
        width_baseline,
        width_custom,
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
        y_pred={"NaiveInterval": y_pred, "SplitConformal": y_pred_baseline},
    )
    fig

    return (fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [Create an Interval Forecaster](/pages/how-to/create-an-interval-forecaster/) for the full guide
        - [Produce Prediction Intervals](/pages/how-to/interval-forecasting/) for built-in approaches
        - [Create a Custom Scorer](/pages/how-to/create-a-scorer/) for interval evaluation metrics
        """
    )
    return


if __name__ == "__main__":
    app.run()
