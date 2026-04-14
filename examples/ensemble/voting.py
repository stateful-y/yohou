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
    "title": "How to Combine Forecasters with VotingForecaster",
    "description": "Aggregate predictions from diverse forecasters using mean, median, or weighted averaging for more robust forecasts.",
    "category": "how-to",
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
        # How to Combine Forecasters with VotingForecaster

        This notebook shows how to combine multiple forecasters into a
        single ensemble using [`VotingForecaster`](/pages/api/generated/yohou.ensemble.VotingForecaster/).
        Ensembles reduce variance by averaging out individual model errors.

        **Prerequisites:** Basic familiarity with point forecasters -
        see the [quickstart](/examples/quickstart/) for an introduction.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_sunspot
    from yohou.ensemble import VotingForecaster
    from yohou.plotting import plot_forecast, plot_model_comparison_bar
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer

    return (
        GradientBoostingRegressor,
        LagTransformer,
        PointReductionForecaster,
        RandomForestRegressor,
        Ridge,
        VotingForecaster,
        fetch_sunspot,
        pl,
        plot_forecast,
        plot_model_comparison_bar,
        SeasonalNaive,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Load and Prepare Data

        Monthly sunspot numbers with a strong ~11-year cycle.
        """
    )
    return


@app.cell
def _(fetch_sunspot, pl):
    y = fetch_sunspot().frame.group_by_dynamic("time", every="1mo").agg(
        pl.col("sunspot_number").mean()
    )
    y_train, y_test = y[:2400], y[2400:]
    forecasting_horizon = 24

    print(f"Training: {len(y_train)} obs, Test: {len(y_test)} obs")
    print(f"Forecasting horizon: {forecasting_horizon} months")
    return forecasting_horizon, y, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Define Base Forecasters

        Create diverse forecasters with different modeling approaches.
        Diversity is the key ingredient for effective ensembles.
        """
    )
    return


@app.cell
def _(
    GradientBoostingRegressor,
    LagTransformer,
    PointReductionForecaster,
    RandomForestRegressor,
    Ridge,
    SeasonalNaive,
):
    base_forecasters = [
        ("naive_12", SeasonalNaive(seasonality=12)),
        (
            "ridge",
            PointReductionForecaster(
                estimator=Ridge(),
                transformers=[LagTransformer(lags=[1, 2, 3, 6, 12])],
            ),
        ),
        (
            "rf",
            PointReductionForecaster(
                estimator=RandomForestRegressor(n_estimators=50, random_state=42),
                transformers=[LagTransformer(lags=[1, 2, 3, 6, 12])],
            ),
        ),
        (
            "gbr",
            PointReductionForecaster(
                estimator=GradientBoostingRegressor(
                    n_estimators=50, random_state=42
                ),
                transformers=[LagTransformer(lags=[1, 2, 3, 6, 12])],
            ),
        ),
    ]
    return (base_forecasters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Fit an Equal-Weight Mean Ensemble

        Pass the named forecasters to `VotingForecaster` with `method="mean"`.
        """
    )
    return


@app.cell
def _(VotingForecaster, base_forecasters, forecasting_horizon, y_train):
    ensemble_mean = VotingForecaster(
        forecasters=base_forecasters,
        method="mean",
    )
    ensemble_mean.fit(y_train, forecasting_horizon=forecasting_horizon)
    return (ensemble_mean,)


@app.cell
def _(ensemble_mean, forecasting_horizon, plot_forecast, y_test, y_train):
    y_pred_mean = ensemble_mean.predict(forecasting_horizon=forecasting_horizon)
    plot_forecast(y_train, y_pred_mean, y_test=y_test, title="Mean Ensemble")
    return (y_pred_mean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Weighted Ensemble

        Assign higher weight to models you expect to perform better.
        Weights are passed to `numpy.average` which normalizes internally.
        """
    )
    return


@app.cell
def _(VotingForecaster, base_forecasters, forecasting_horizon, y_train):
    ensemble_weighted = VotingForecaster(
        forecasters=base_forecasters,
        method="mean",
        weights=[0.5, 1.0, 2.0, 2.0],
    )
    ensemble_weighted.fit(y_train, forecasting_horizon=forecasting_horizon)
    return (ensemble_weighted,)


@app.cell
def _(
    ensemble_weighted,
    forecasting_horizon,
    plot_forecast,
    y_test,
    y_train,
):
    y_pred_weighted = ensemble_weighted.predict(
        forecasting_horizon=forecasting_horizon
    )
    plot_forecast(
        y_train, y_pred_weighted, y_test=y_test, title="Weighted Ensemble"
    )
    return (y_pred_weighted,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. Median Ensemble

        Use `method="median"` for robustness to outlier predictions.
        Weights are silently ignored with this method.
        """
    )
    return


@app.cell
def _(VotingForecaster, base_forecasters, forecasting_horizon, y_train):
    ensemble_median = VotingForecaster(
        forecasters=base_forecasters,
        method="median",
    )
    ensemble_median.fit(y_train, forecasting_horizon=forecasting_horizon)
    return (ensemble_median,)


@app.cell
def _(
    ensemble_median,
    forecasting_horizon,
    plot_forecast,
    y_test,
    y_train,
):
    y_pred_median = ensemble_median.predict(forecasting_horizon=forecasting_horizon)
    plot_forecast(
        y_train, y_pred_median, y_test=y_test, title="Median Ensemble"
    )
    return (y_pred_median,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6. Compare Ensemble Methods

        Compare all three ensembles against the individual base forecasters.
        """
    )
    return


@app.cell
def _(
    ensemble_mean,
    ensemble_median,
    ensemble_weighted,
    forecasting_horizon,
    plot_model_comparison_bar,
    y_test,
):
    from yohou.metrics import MeanAbsoluteError

    scorer = MeanAbsoluteError()

    models = {
        "Mean Ensemble": ensemble_mean,
        "Weighted Ensemble": ensemble_weighted,
        "Median Ensemble": ensemble_median,
    }

    scores = {}
    for name, model in models.items():
        y_pred = model.predict(forecasting_horizon=forecasting_horizon)
        scores[name] = scorer.score(y_test, y_pred)

    plot_model_comparison_bar(
        scores,
        metric_name="MAE",
        title="Ensemble Method Comparison",
    )
    return MeanAbsoluteError, models, scores, scorer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 7. Panel Data Ensemble

        `VotingForecaster` supports panel data automatically. Each forecaster
        handles all panel groups.
        """
    )
    return


@app.cell
def _(VotingForecaster, SeasonalNaive, pl):
    from yohou.datasets import fetch_tourism_monthly

    tourism = fetch_tourism_monthly(n_series=5).frame
    tourism_train = tourism.filter(
        pl.col("time") < pl.col("time").max() - pl.duration(days=365)
    )
    tourism_test = tourism.filter(
        pl.col("time") >= pl.col("time").max() - pl.duration(days=365)
    )

    panel_ensemble = VotingForecaster(
        forecasters=[
            ("naive_1", SeasonalNaive(seasonality=1)),
            ("naive_6", SeasonalNaive(seasonality=6)),
            ("naive_12", SeasonalNaive(seasonality=12)),
        ],
        method="mean",
    )
    panel_ensemble.fit(tourism_train, forecasting_horizon=12)
    y_pred_panel = panel_ensemble.predict(forecasting_horizon=12)

    print(f"Panel groups: {panel_ensemble.panel_group_names_}")
    print(f"Prediction shape: {y_pred_panel.shape}")
    return (
        fetch_tourism_monthly,
        panel_ensemble,
        tourism,
        tourism_test,
        tourism_train,
        y_pred_panel,
    )


@app.cell
def _(panel_ensemble, plot_forecast, tourism_test, tourism_train, y_pred_panel):
    plot_forecast(
        tourism_train,
        y_pred_panel,
        y_test=tourism_test,
        title="Panel Data Ensemble (5 Tourism Series)",
    )
    return


if __name__ == "__main__":
    app.run()
