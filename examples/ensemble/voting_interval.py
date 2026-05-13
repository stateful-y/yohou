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
    "title": "How to Combine Interval Forecasters",
    "description": "Build interval ensembles with VotingIntervalForecaster using envelope, mean, and median aggregation strategies.",
    "category": "how-to",
    "companion": "/pages/explanation/ensemble-forecasting/",
    "section": "forecasting-models",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How to Combine Interval Forecasters with VotingIntervalForecaster

    This notebook shows how to combine multiple interval forecasters into
    a single ensemble using [`VotingIntervalForecaster`](/pages/api/generated/yohou.ensemble.VotingIntervalForecaster/).
    Aggregating intervals from diverse models can produce more robust
    coverage than any single model.

    ## Prerequisites

    Familiarity with interval forecasting. See the [conformal forecasting example](/examples/interval/conformal_forecasting/) for an introduction.
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

    import polars as pl
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_sunspot
    from yohou.ensemble import VotingIntervalForecaster
    from yohou.interval import SplitConformalForecaster
    from yohou.model_selection import train_test_split
    from yohou.plotting import plot_forecast
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer

    return (
        GradientBoostingRegressor,
        LagTransformer,
        PointReductionForecaster,
        Ridge,
        SeasonalNaive,
        SplitConformalForecaster,
        VotingIntervalForecaster,
        deepcopy,
        fetch_sunspot,
        pl,
        plot_forecast,
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
    y_train, y_test = train_test_split(y, test_size=5 * forecasting_horizon)
    coverage = [0.50, 0.80, 0.90]

    print(f"Training: {len(y_train)} obs, Test: {len(y_test)} obs")
    return coverage, forecasting_horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Define Base Interval Forecasters

    Wrap different point forecasters in `SplitConformalForecaster` to
    produce prediction intervals. Diversity in the underlying point
    model leads to better ensemble coverage.
    """)


@app.cell
def _(
    GradientBoostingRegressor,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SeasonalNaive,
    SplitConformalForecaster,
):
    base_forecasters = [
        (
            "conf_naive",
            SplitConformalForecaster(
                point_forecaster=SeasonalNaive(seasonality=12),
                calibration_size=120,
            ),
        ),
        (
            "conf_ridge",
            SplitConformalForecaster(
                point_forecaster=PointReductionForecaster(
                    estimator=Ridge(),
                    feature_transformer=LagTransformer(lag=[1, 2, 3, 6, 12]),
                ),
                calibration_size=120,
            ),
        ),
        (
            "conf_gbr",
            SplitConformalForecaster(
                point_forecaster=PointReductionForecaster(
                    estimator=GradientBoostingRegressor(n_estimators=50, random_state=42),
                    feature_transformer=LagTransformer(lag=[1, 2, 3, 6, 12]),
                    reduction_strategy="direct",
                ),
                calibration_size=120,
            ),
        ),
    ]
    return (base_forecasters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Envelope Strategy (Default)

    `method="envelope"` takes the minimum of lower bounds and the maximum
    of upper bounds across all forecasters, producing the widest
    (most conservative) interval.
    """)


@app.cell
def _(
    VotingIntervalForecaster,
    base_forecasters,
    coverage,
    forecasting_horizon,
    y_train,
):
    ensemble_envelope = VotingIntervalForecaster(
        forecasters=base_forecasters,
        method="envelope",
    )
    ensemble_envelope.fit(
        y_train,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage,
    )
    return (ensemble_envelope,)


@app.cell
def _(
    coverage,
    deepcopy,
    ensemble_envelope,
    forecasting_horizon,
    plot_forecast,
    y_test,
    y_train,
):
    y_pred_envelope = deepcopy(ensemble_envelope).observe_predict_interval(
        y_test,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage,
    )
    plot_forecast(
        y_test,
        y_pred_envelope,
        y_train=y_train,
        n_history=400,
        coverage_rates=coverage,
        title="Envelope Ensemble (Widest Coverage)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Mean Strategy

    `method="mean"` averages the lower and upper bounds separately.
    This produces a moderate interval width.
    """)


@app.cell
def _(
    VotingIntervalForecaster,
    base_forecasters,
    coverage,
    forecasting_horizon,
    y_train,
):
    ensemble_mean = VotingIntervalForecaster(
        forecasters=base_forecasters,
        method="mean",
    )
    ensemble_mean.fit(
        y_train,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage,
    )
    return (ensemble_mean,)


@app.cell
def _(
    coverage,
    deepcopy,
    ensemble_mean,
    forecasting_horizon,
    plot_forecast,
    y_test,
    y_train,
):
    y_pred_mean = deepcopy(ensemble_mean).observe_predict_interval(
        y_test,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage,
    )
    plot_forecast(
        y_test,
        y_pred_mean,
        y_train=y_train,
        n_history=400,
        coverage_rates=coverage,
        title="Mean Ensemble",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Median Strategy

    `method="median"` takes the median of lower and upper bounds.
    More robust to outlier predictions than mean averaging.
    """)


@app.cell
def _(
    VotingIntervalForecaster,
    base_forecasters,
    coverage,
    forecasting_horizon,
    y_train,
):
    ensemble_median = VotingIntervalForecaster(
        forecasters=base_forecasters,
        method="median",
    )
    ensemble_median.fit(
        y_train,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage,
    )
    return (ensemble_median,)


@app.cell
def _(
    coverage,
    deepcopy,
    ensemble_median,
    forecasting_horizon,
    plot_forecast,
    y_test,
    y_train,
):
    y_pred_median = deepcopy(ensemble_median).observe_predict_interval(
        y_test,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage,
    )
    plot_forecast(
        y_test,
        y_pred_median,
        y_train=y_train,
        n_history=400,
        coverage_rates=coverage,
        title="Median Ensemble",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Point Predictions from Interval Ensembles

    When all base forecasters also support `predict()`,
    `VotingIntervalForecaster` exposes a `predict()` method.
    Control aggregation with the `point_method` parameter.
    """)


@app.cell
def _(
    deepcopy,
    ensemble_envelope,
    forecasting_horizon,
    plot_forecast,
    y_test,
    y_train,
):
    y_pred_point = deepcopy(ensemble_envelope).observe_predict(y_test, forecasting_horizon=forecasting_horizon)
    plot_forecast(
        y_test,
        y_pred_point,
        y_train=y_train,
        n_history=400,
        title="Point Predictions from Interval Ensemble",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Weighted Mean Ensemble

    Pass `weights` to give higher influence to certain forecasters.
    Weights apply to both `method="mean"` and `point_method="mean"`.
    """)


@app.cell
def _(
    VotingIntervalForecaster,
    base_forecasters,
    coverage,
    forecasting_horizon,
    plot_forecast,
    y_test,
    y_train,
):
    ensemble_weighted = VotingIntervalForecaster(
        forecasters=base_forecasters,
        method="mean",
        weights=[0.5, 1.0, 2.0],
    )
    ensemble_weighted.fit(
        y_train,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage,
    )
    y_pred_weighted = ensemble_weighted.observe_predict_interval(
        y_test,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage,
    )
    plot_forecast(
        y_test,
        y_pred_weighted,
        y_train=y_train,
        n_history=400,
        coverage_rates=coverage,
        title="Weighted Mean Ensemble",
    )


if __name__ == "__main__":
    app.run()
