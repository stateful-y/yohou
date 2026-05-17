# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.23.1"
__gallery__ = {
    "title": "How to Choose a Forecasting Method",
    "description": "Interactive decision guide progressing from SeasonalNaive baseline through linear reduction, stationarity transforms, feature enrichment, nonlinear models, decomposition, and prediction intervals.",
    "category": "how-to",
    "companion": "/pages/how-to/choose-forecasting-method/",
    "section": "getting-started",
    "api_references": [
        "SeasonalNaive",
        "PointReductionForecaster",
        "DecompositionPipeline",
        "SplitConformalForecaster",
        "MeanAbsoluteError",
    ],
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
        # How to Choose a Forecasting Method

        This notebook walks through the decision steps for choosing a forecasting
        method, starting from a baseline and progressively adding complexity only
        when scores improve.

        **Prerequisites:** Familiarity with fit/predict/score
        ([Getting Started](/pages/tutorials/getting-started/)).
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## Setup")


@app.cell
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_tourism_monthly
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import ExpandingWindowSplitter, GridSearchCV, train_test_split
    from yohou.point import PointReductionForecaster, SeasonalNaive

    y = (
        fetch_tourism_monthly()
        .frame.select("time", "T1__tourists")
        .drop_nulls()
        .rename({"T1__tourists": "total"})
    )

    forecasting_horizon = 12
    y_train, y_test = train_test_split(y, test_size=forecasting_horizon)

    scorer = MeanAbsoluteError()
    scorer.fit(y_train)

    return (
        ExpandingWindowSplitter,
        GridSearchCV,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        SeasonalNaive,
        forecasting_horizon,
        pl,
        scorer,
        y,
        y_test,
        y_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 1. Establish a Baseline with SeasonalNaive")


@app.cell
def _(SeasonalNaive, forecasting_horizon, scorer, y_test, y_train):
    baseline = SeasonalNaive(seasonality=12)
    baseline.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred_baseline = baseline.predict(forecasting_horizon=forecasting_horizon)

    score_baseline = scorer.score(y_test, y_pred_baseline)
    print(f"SeasonalNaive MAE: {score_baseline:.2f}")

    return baseline, score_baseline, y_pred_baseline


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 2. Add a Linear Model")


@app.cell
def _(PointReductionForecaster, Ridge, forecasting_horizon, scorer, y_test, y_train):
    from yohou.preprocessing import LagTransformer

    linear = PointReductionForecaster(
        estimator=Ridge(),
        feature_transformer=LagTransformer(lag=[1, 6, 12]),
    )
    linear.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred_linear = linear.predict(forecasting_horizon=forecasting_horizon)

    score_linear = scorer.score(y_test, y_pred_linear)
    print(f"Ridge + Lags MAE: {score_linear:.2f}")

    return LagTransformer, linear, score_linear, y_pred_linear


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 3. Stabilize Non-stationary Series")


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, forecasting_horizon, scorer, y_test, y_train):
    from yohou.stationarity import SeasonalDifferencing

    stationary = PointReductionForecaster(
        estimator=Ridge(),
        feature_transformer=LagTransformer(lag=[1, 6, 12]),
        target_transformer=SeasonalDifferencing(seasonality=12),
    )
    stationary.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred_stationary = stationary.predict(forecasting_horizon=forecasting_horizon)

    score_stationary = scorer.score(y_test, y_pred_stationary)
    print(f"Ridge + Lags + SeasonalDiff MAE: {score_stationary:.2f}")

    return SeasonalDifferencing, score_stationary, stationary, y_pred_stationary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 4. Enrich Features")


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, SeasonalDifferencing, forecasting_horizon, scorer, y_test, y_train):
    from yohou.compose import FeatureUnion
    from yohou.preprocessing import RollingStatisticsTransformer

    feature_transformer = FeatureUnion(
        transformer_list=[
            ("lags", LagTransformer(lag=[1, 3, 6, 12])),
            ("rolling", RollingStatisticsTransformer(window_size=6)),
        ]
    )

    enriched = PointReductionForecaster(
        estimator=Ridge(),
        feature_transformer=feature_transformer,
        target_transformer=SeasonalDifferencing(seasonality=12),
    )
    enriched.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred_enriched = enriched.predict(forecasting_horizon=forecasting_horizon)

    score_enriched = scorer.score(y_test, y_pred_enriched)
    print(f"Ridge + Features + SeasonalDiff MAE: {score_enriched:.2f}")

    return FeatureUnion, RollingStatisticsTransformer, enriched, feature_transformer, score_enriched, y_pred_enriched


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 5. Switch to a Nonlinear Estimator")


@app.cell
def _(PointReductionForecaster, SeasonalDifferencing, feature_transformer, forecasting_horizon, scorer, y_test, y_train):
    from sklearn.ensemble import HistGradientBoostingRegressor

    nonlinear = PointReductionForecaster(
        estimator=HistGradientBoostingRegressor(max_iter=100, random_state=42),
        feature_transformer=feature_transformer,
        target_transformer=SeasonalDifferencing(seasonality=12),
        reduction_strategy="direct",
    )
    nonlinear.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred_nonlinear = nonlinear.predict(forecasting_horizon=forecasting_horizon)

    score_nonlinear = scorer.score(y_test, y_pred_nonlinear)
    print(f"HGBR + Features + SeasonalDiff MAE: {score_nonlinear:.2f}")

    return HistGradientBoostingRegressor, nonlinear, score_nonlinear, y_pred_nonlinear


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 6. Decomposition for Complex Seasonality")


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, forecasting_horizon, scorer, y_test, y_train):
    from yohou.compose import DecompositionPipeline
    from yohou.stationarity import FourierSeasonalityForecaster, PolynomialTrendForecaster

    decomp = DecompositionPipeline(
        forecasters=[
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", FourierSeasonalityForecaster(seasonality=12, harmonics=[1, 2, 3])),
            ("residual", PointReductionForecaster(estimator=Ridge(), feature_transformer=LagTransformer(lag=[1, 2, 3]))),
        ]
    )
    decomp.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred_decomp = decomp.predict(forecasting_horizon=forecasting_horizon)

    score_decomp = scorer.score(y_test, y_pred_decomp)
    print(f"DecompositionPipeline MAE: {score_decomp:.2f}")

    return DecompositionPipeline, FourierSeasonalityForecaster, PolynomialTrendForecaster, decomp, score_decomp, y_pred_decomp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 7. Add Prediction Intervals")


@app.cell
def _(nonlinear, forecasting_horizon, y_train):
    from yohou.interval import SplitConformalForecaster

    interval_forecaster = SplitConformalForecaster(
        point_forecaster=nonlinear,
        calibration_size=48,
    )
    interval_forecaster.fit(
        y_train, forecasting_horizon=forecasting_horizon, coverage_rates=[0.9]
    )
    intervals = interval_forecaster.predict_interval(coverage_rates=[0.9])
    intervals.head()

    return SplitConformalForecaster, interval_forecaster, intervals


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 8. Compare All Methods")


@app.cell
def _(
    score_baseline,
    score_decomp,
    score_enriched,
    score_linear,
    score_nonlinear,
    score_stationary,
    y_pred_baseline,
    y_pred_linear,
    y_pred_nonlinear,
    y_test,
    y_train,
):
    from yohou.plotting import plot_forecast

    print("MAE Summary:")
    print(f"  SeasonalNaive:            {score_baseline:.2f}")
    print(f"  Ridge + Lags:             {score_linear:.2f}")
    print(f"  Ridge + Lags + Diff:      {score_stationary:.2f}")
    print(f"  Ridge + Features + Diff:  {score_enriched:.2f}")
    print(f"  HGBR + Features + Diff:   {score_nonlinear:.2f}")
    print(f"  DecompositionPipeline:    {score_decomp:.2f}")

    fig = plot_forecast(
        y_train=y_train,
        y_test=y_test,
        y_pred={
            "SeasonalNaive": y_pred_baseline,
            "Ridge": y_pred_linear,
            "HGBR": y_pred_nonlinear,
        },
    )
    fig

    return (fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [How to Choose a Forecasting Method](/pages/how-to/choose-forecasting-method/) for the full decision guide
        - [Tune Forecaster Hyperparameters](/pages/how-to/tune-hyperparameters/) to optimize the chosen method
        - [Evaluate Forecast Accuracy](/pages/how-to/evaluate-forecast-accuracy/) for cross-validation and metric selection
        """
    )


if __name__ == "__main__":
    app.run()
