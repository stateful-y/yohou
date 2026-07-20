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
    "title": "How to Combine Forecasters with VotingPointForecaster",
    "description": "Build point ensembles with VotingPointForecaster using mean, weighted, and median aggregation strategies.",
    "category": "how-to",
    "companion": "/pages/how-to/ensemble-forecasting/",
    "section": "forecasting-models",
    "api_references": ["PointReductionForecaster", "SeasonalNaive", "VotingPointForecaster", "plot_score_heatmap", "plot_score_per_step", "plot_score_per_vintage", "plot_score_summary", "plot_score_time_series"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How to Combine Forecasters with VotingPointForecaster

    This notebook shows how to combine multiple forecasters into a
    single ensemble using [`VotingPointForecaster`](/pages/api/generated/yohou.ensemble.VotingPointForecaster/).
    Ensembles reduce variance by averaging out individual model errors.

    **Prerequisites:** Basic familiarity with point forecasters - see the [quickstart](/examples/quickstart/) for an introduction.
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

    import polars as pl
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_sunspot, fetch_tourism_monthly
    from yohou.ensemble import VotingPointForecaster
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import train_test_split
    from yohou.plotting import (
        plot_forecast,
        plot_score_heatmap,
        plot_score_per_step,
        plot_score_per_vintage,
        plot_score_summary,
        plot_score_time_series,
    )
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer

    return (
        GradientBoostingRegressor,
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        RandomForestRegressor,
        Ridge,
        SeasonalNaive,
        VotingPointForecaster,
        deepcopy,
        fetch_sunspot,
        fetch_tourism_monthly,
        pl,
        plot_forecast,
        plot_score_heatmap,
        plot_score_per_step,
        plot_score_per_vintage,
        plot_score_summary,
        plot_score_time_series,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load and Prepare Data

    Monthly sunspot numbers with a strong ~11-year cycle.
    """)


@app.cell
def _(fetch_sunspot, pl, train_test_split):
    y = fetch_sunspot().frame.group_by_dynamic("time", every="1mo").agg(pl.col("sunspot_number").mean())
    forecasting_horizon = 24
    y_train, y_test = train_test_split(y, test_size=forecasting_horizon)

    print(f"Training: {len(y_train)} obs, Test: {len(y_test)} obs")
    print(f"Forecasting horizon: {forecasting_horizon} months")
    return forecasting_horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Define Base Forecasters

    Create diverse forecasters with different modeling approaches.
    Diversity is the key ingredient for effective ensembles.
    """)


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
                actual_transformer=LagTransformer(lag=[1, 2, 3, 6, 12]),
            ),
        ),
        (
            "rf",
            PointReductionForecaster(
                estimator=RandomForestRegressor(n_estimators=50, random_state=42),
                actual_transformer=LagTransformer(lag=[1, 2, 3, 6, 12]),
            ),
        ),
        (
            "gbr",
            PointReductionForecaster(
                estimator=GradientBoostingRegressor(n_estimators=50, random_state=42),
                actual_transformer=LagTransformer(lag=[1, 2, 3, 6, 12]),
                reduction_strategy="direct",
            ),
        ),
    ]
    return (base_forecasters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Fit an Equal-Weight Mean Ensemble

    Pass the named forecasters to [`VotingPointForecaster`](/pages/api/generated/yohou.ensemble.voting_point.VotingPointForecaster/) with `method="mean"`.
    """)


@app.cell
def _(VotingPointForecaster, base_forecasters, forecasting_horizon, y_train):
    ensemble_mean = VotingPointForecaster(
        forecasters=base_forecasters,
        method="mean",
    )
    ensemble_mean.fit(y_train, forecasting_horizon=forecasting_horizon)
    return (ensemble_mean,)


@app.cell
def _(
    deepcopy,
    ensemble_mean,
    forecasting_horizon,
    plot_forecast,
    y_test,
    y_train,
):
    y_pred_mean = deepcopy(ensemble_mean).observe_predict(y_test, forecasting_horizon=forecasting_horizon)
    plot_forecast(y_test, y_pred_mean, y_train=y_train, n_history=400, title="Mean Ensemble")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Weighted Ensemble

    Assign higher weight to models you expect to perform better.
    Weights are passed to `numpy.average` which normalizes internally.
    """)


@app.cell
def _(VotingPointForecaster, base_forecasters, forecasting_horizon, y_train):
    ensemble_weighted = VotingPointForecaster(
        forecasters=base_forecasters,
        method="mean",
        weights=[0.5, 1.0, 2.0, 2.0],
    )
    ensemble_weighted.fit(y_train, forecasting_horizon=forecasting_horizon)
    return (ensemble_weighted,)


@app.cell
def _(
    deepcopy,
    ensemble_weighted,
    forecasting_horizon,
    plot_forecast,
    y_test,
    y_train,
):
    y_pred_weighted = deepcopy(ensemble_weighted).observe_predict(y_test, forecasting_horizon=forecasting_horizon)
    plot_forecast(y_test, y_pred_weighted, y_train=y_train, n_history=400, title="Weighted Ensemble")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Median Ensemble

    Use `method="median"` for robustness to outlier predictions.
    Weights are silently ignored with this method.
    """)


@app.cell
def _(VotingPointForecaster, base_forecasters, forecasting_horizon, y_train):
    ensemble_median = VotingPointForecaster(
        forecasters=base_forecasters,
        method="median",
    )
    ensemble_median.fit(y_train, forecasting_horizon=forecasting_horizon)
    return (ensemble_median,)


@app.cell
def _(
    deepcopy,
    ensemble_median,
    forecasting_horizon,
    plot_forecast,
    y_test,
    y_train,
):
    y_pred_median = deepcopy(ensemble_median).predict(forecasting_horizon=forecasting_horizon)
    plot_forecast(y_test, y_pred_median, y_train=y_train, n_history=400, title="Median Ensemble")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Compare Ensemble Methods

    Compare all three ensembles against the individual base forecasters.
    """)


@app.cell
def _(
    MeanAbsoluteError,
    deepcopy,
    ensemble_mean,
    ensemble_median,
    ensemble_weighted,
    forecasting_horizon,
    plot_score_summary,
    y_test,
    y_train,
):
    models = {
        "Mean Ensemble": ensemble_mean,
        "Weighted Ensemble": ensemble_weighted,
        "Median Ensemble": ensemble_median,
    }

    y_preds = {}
    for name, model in models.items():
        y_preds[name] = deepcopy(model).observe_predict(y_test, forecasting_horizon=forecasting_horizon)

    plot_score_summary(
        MeanAbsoluteError(),
        y_test,
        y_preds,
        title="Ensemble Method Comparison",
    )


@app.cell
def _(
    MeanAbsoluteError,
    plot_score_per_vintage,
    y_preds,
    y_test,
):
    plot_score_per_vintage(
        MeanAbsoluteError(),
        y_test,
        y_preds,
        kind="line",
        show_trend=True,
        title="Ensemble MAE per Vintage",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Panel Data Ensemble

    [`VotingPointForecaster`](/pages/api/generated/yohou.ensemble.voting_point.VotingPointForecaster/) supports panel data automatically. Each forecaster
    handles all panel groups.
    """)


@app.cell
def _(
    SeasonalNaive,
    VotingPointForecaster,
    fetch_tourism_monthly,
    train_test_split,
):
    tourism = fetch_tourism_monthly(n_series=5).frame
    tourism_train, tourism_test = train_test_split(tourism, test_size=12 * 5)

    panel_ensemble = VotingPointForecaster(
        forecasters=[
            ("naive_1", SeasonalNaive(seasonality=1)),
            ("naive_6", SeasonalNaive(seasonality=6)),
            ("naive_12", SeasonalNaive(seasonality=12)),
        ],
        method="mean",
    )
    panel_ensemble.fit(tourism_train, forecasting_horizon=12)
    y_pred_panel = panel_ensemble.observe_predict(tourism_test, forecasting_horizon=12, groups=["T3", "T4", "T5"])

    print(f"Panel groups: {panel_ensemble.groups_}")
    print(f"Prediction shape: {y_pred_panel.shape}")
    return tourism_test, tourism_train, y_pred_panel


@app.cell
def _(plot_forecast, tourism_test, tourism_train, y_pred_panel):
    plot_forecast(
        tourism_test,
        y_pred_panel,
        y_train=tourism_train,
        groups=["T3", "T4", "T5"],
        n_history=400,
        title="Panel Data Ensemble (5 Tourism Series)",
    )


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-vintage Scoring

    The `observe_predict` method with `stride=1` produces one forecast per
    observation point, creating multiple *vintages*. Each vintage represents
    a different forecast origin, so you can analyse how accuracy evolves as
    the model absorbs more data.
    """)


@app.cell
def _(deepcopy, ensemble_mean, forecasting_horizon, y_test):
    _vintage_model = deepcopy(ensemble_mean)
    y_pred_vintages = _vintage_model.observe_predict(
        y=y_test,
        stride=1,
        forecasting_horizon=forecasting_horizon,
    )
    print(f"Vintages: {y_pred_vintages['vintage_time'].n_unique()}")
    y_pred_vintages.head(10)
    return (y_pred_vintages,)


@app.cell
def _(MeanAbsoluteError, y_train):
    vintage_scorer = MeanAbsoluteError()
    vintage_scorer.fit(y_train)
    return (vintage_scorer,)


@app.cell
def _(vintage_scorer, plot_score_heatmap, y_pred_vintages, y_test):
    plot_score_heatmap(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="Score Heatmap (Step x Vintage)",
    )


@app.cell
def _(vintage_scorer, plot_score_time_series, y_pred_vintages, y_test):
    plot_score_time_series(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="Per-timestep MAE across Vintages",
        y_label="MAE",
        height=500,
        facet_by="vintage",
    )



@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [Ensemble Forecasting](/pages/how-to/ensemble-forecasting/) for the full guide
        - [Build Reduction Forecasters](/pages/how-to/build-reduction-forecasters/) for related techniques
        """
    )
    return

if __name__ == "__main__":
    app.run()
