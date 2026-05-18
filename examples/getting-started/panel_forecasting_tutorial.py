# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.23.1"
__gallery__ = {
    "title": "Panel Data Forecasting",
    "description": "Forecast multiple related time series simultaneously using the __ naming convention, LocalPanelForecaster, and per-group scoring.",
    "category": "tutorial",
    "companion": "/pages/tutorials/panel-data/",
    "section": "getting-started",
    "api_references": ["LocalPanelForecaster", "MeanAbsoluteError", "SeasonalNaive", "fetch_tourism_monthly", "inspect_panel", "plot_forecast"],
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
        # Panel Data Forecasting

        In this notebook, we will forecast multiple related time series
        simultaneously using panel data. We will load a multi-series tourism
        dataset, understand the `__` column naming convention, fit a
        [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster/) that clones a forecaster per group, and score
        the results with both aggregate and per-group metrics.

        **Prerequisites:** Completed [Getting Started](/pages/tutorials/getting-started/).
        """
    )


@app.cell(hide_code=True)
def _():
    import polars as pl

    from yohou.compose import LocalPanelForecaster
    from yohou.datasets import fetch_tourism_monthly
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast
    from yohou.point import SeasonalNaive
    from yohou.utils.panel import get_group_df, inspect_panel

    return (
        LocalPanelForecaster,
        MeanAbsoluteError,
        SeasonalNaive,
        fetch_tourism_monthly,
        get_group_df,
        inspect_panel,
        pl,
        plot_forecast,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 1. Load a Panel Dataset")


@app.cell
def _(fetch_tourism_monthly):
    bunch = fetch_tourism_monthly()
    y = bunch.frame.select(
        ["time", "T187__tourists", "T188__tourists", "T189__tourists"]
    ).drop_nulls()
    y.head()

    return bunch, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notice the column names use the `{group}__{variable}` convention.
        The text before `__` identifies the panel group (a region); the text
        after `__` is the variable name.
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 2. Inspect Panel Structure")


@app.cell
def _(inspect_panel, y):
    global_names, panel_groups = inspect_panel(y)
    panel_groups

    return global_names, panel_groups


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 3. Train/Test Split")


@app.cell
def _(y):
    from yohou.model_selection import train_test_split

    forecasting_horizon = 12
    y_train, y_test = train_test_split(y, test_size=forecasting_horizon)
    y_train.shape, y_test.shape

    return forecasting_horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 4. Fit a LocalPanelForecaster")


@app.cell
def _(LocalPanelForecaster, SeasonalNaive, forecasting_horizon, y_train):
    forecaster = LocalPanelForecaster(
        forecaster=SeasonalNaive(seasonality=12),
    )
    forecaster.fit(y_train, forecasting_horizon=forecasting_horizon)

    return (forecaster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 5. Predict")


@app.cell
def _(forecaster, forecasting_horizon):
    y_pred = forecaster.predict(forecasting_horizon=forecasting_horizon)
    y_pred.head()

    return (y_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 6. Score")


@app.cell
def _(MeanAbsoluteError, y_pred, y_test, y_train):
    scorer = MeanAbsoluteError()
    scorer.fit(y_train)
    score = scorer.score(y_test, y_pred)
    score

    return score, scorer


@app.cell
def _(MeanAbsoluteError, y_pred, y_test, y_train):
    groupwise_scorer = MeanAbsoluteError(aggregation_method="groupwise")
    groupwise_scorer.fit(y_train)
    per_group = groupwise_scorer.score(y_test, y_pred)
    per_group

    return (per_group,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 7. Plot a Single Group")


@app.cell
def _(get_group_df, panel_groups, pl, plot_forecast, y_pred, y_test, y_train):
    first_group = list(panel_groups.keys())[0]
    schema = {col.split("__")[1]: pl.Float64 for col in panel_groups[first_group]}

    fig = plot_forecast(
        y_train=get_group_df(y_train, first_group, schema),
        y_test=get_group_df(y_test, first_group, schema),
        y_pred=get_group_df(y_pred, first_group, schema),
    )
    fig

    return fig, first_group, schema


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## What We Built

        We loaded a multi-series tourism dataset, inspected its panel structure
        using the `__` naming convention, fit a [`LocalPanelForecaster`](/pages/api/generated/yohou.compose.local_panel_forecaster.LocalPanelForecaster/) that clones
        a seasonal baseline per region, scored the results with both aggregate and
        per-group metrics, and plotted forecasts for individual groups.
        """
    )



@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [Panel Data Forecasting](/pages/tutorials/panel-data/) for the full guide
        - [Panel Data How-To](/pages/how-to/panel-data/) for related techniques
        """
    )
    return

if __name__ == "__main__":
    app.run()
