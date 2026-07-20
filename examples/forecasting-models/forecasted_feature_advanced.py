# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.19.11"
__gallery__ = {
    "title": "How to Use Lagged Forecasts as Features",
    "description": "Compare ForecastedFeatureForecaster strategies (actual, predicted, rewind) and split ratio tuning for chaining feature and target forecasters.",
    "category": "how-to",
    "section": "forecasting-models",
    "companion": "/pages/how-to/build-reduction-forecasters/",
    "api_references": ["ForecastedFeatureForecaster", "PointReductionForecaster", "fetch_hospital", "LagTransformer", "plot_time_series", "plot_forecast", "MeanAbsoluteError", "train_test_split"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ForecastedFeatureForecaster

    When exogenous features are not known at prediction time,
    [`ForecastedFeatureForecaster`](/pages/api/generated/yohou.compose.forecasted_feature_forecaster.ForecastedFeatureForecaster/) chains a feature forecaster with
    a target forecaster.

    ## 1. Prepare Multivariate Data

    We use the Hospital dataset with T1 as the target and T2-T4 as
    exogenous covariates. The covariates are renamed to non-panel
    column names for a standard multivariate workflow.
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import ForecastedFeatureForecaster
    from yohou.datasets import fetch_hospital
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import train_test_split
    from yohou.plotting import plot_forecast, plot_score_time_series, plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        ForecastedFeatureForecaster,
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        fetch_hospital,
        pl,
        plot_forecast,
        plot_score_time_series,
        plot_time_series,
        train_test_split,
    )


@app.cell
def _(fetch_hospital, mo, pl, train_test_split):
    _hosp = fetch_hospital().frame
    # Select 4 series and rename for multivariate analysis
    hospital = _hosp.select(
        "time",
        pl.col("T1__patients").alias("target"),
        pl.col("T2__patients").alias("feature_1"),
        pl.col("T3__patients").alias("feature_2"),
        pl.col("T4__patients").alias("feature_3"),
    )
    _train_df, _test_df = train_test_split(hospital, test_size=0.15)
    y_train = _train_df.select("time", "target")
    y_test = _test_df.select("time", "target")
    X_actual_train = _train_df.select("time", "feature_1", "feature_2", "feature_3")
    X_actual_test = _test_df.select("time", "feature_1", "feature_2", "feature_3")
    horizon = len(y_test)

    mo.md(
        f"**Hospital (monthly)**: {len(hospital)} rows\n\n"
        f"**Target**: target, **Features**: feature_1, feature_2, feature_3\n\n"
        f"**Train**: {len(y_train)}, **Test**: {len(y_test)}"
    )
    return X_actual_test, X_actual_train, hospital, horizon, y_test, y_train


@app.cell
def _(hospital, plot_time_series):
    plot_time_series(hospital, title="Hospital Monthly Patients")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Strategy: "actual"

    The target is trained on perfect-foresight features (actual values
    windowed forward). At prediction time the feature forecaster supplies the
    forecast automatically through the target's `X_forecast` channel, so you do
    not pass features to `predict`. This is simple but creates a train/serve
    mismatch (perfect at train, forecasted at serve).
    """)


@app.cell
def _(
    ForecastedFeatureForecaster,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    X_actual_train,
    horizon,
    y_train,
):
    fc_actual = ForecastedFeatureForecaster(
        target_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            actual_transformer=LagTransformer(lag=[1, 3]),
        ),
        feature_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            actual_transformer=LagTransformer(lag=[1, 3]),
        ),
        strategy="actual",
    )
    fc_actual.fit(y_train, X_actual_train, forecasting_horizon=horizon)
    y_pred_actual = fc_actual.predict(forecasting_horizon=horizon)
    return fc_actual, y_pred_actual


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Strategy: "predicted"

    The target trains on a rolling forecast of the features over a held-out
    portion, so it sees the same forecast quality at fit as at predict (no
    train/serve mismatch). In every strategy the forecast reaches the target
    through `X_forecast`; the strategies differ only in the forecast's quality.
    """)


@app.cell
def _(
    ForecastedFeatureForecaster,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    X_actual_train,
    horizon,
    y_train,
):
    fc_predicted = ForecastedFeatureForecaster(
        target_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            actual_transformer=LagTransformer(lag=[1, 3]),
        ),
        feature_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            actual_transformer=LagTransformer(lag=[1, 3]),
        ),
        strategy="predicted",
        split_ratio=0.7,
    )
    fc_predicted.fit(y_train, X_actual_train, forecasting_horizon=horizon)
    y_pred_predicted = fc_predicted.predict(forecasting_horizon=horizon)
    return fc_predicted, y_pred_predicted


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Strategy: "rewind"

    The feature forecaster's observation state is rewound before
    prediction, providing a different initialisation point.
    """)


@app.cell
def _(
    ForecastedFeatureForecaster,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    X_actual_train,
    horizon,
    y_train,
):
    fc_rewind = ForecastedFeatureForecaster(
        target_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            actual_transformer=LagTransformer(lag=[1, 3]),
        ),
        feature_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            actual_transformer=LagTransformer(lag=[1, 3]),
        ),
        strategy="rewind",
    )
    fc_rewind.fit(y_train, X_actual_train, forecasting_horizon=horizon)
    y_pred_rewind = fc_rewind.predict(forecasting_horizon=horizon)
    return fc_rewind, y_pred_rewind


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Compare Strategies

    [`plot_score_time_series`](/pages/api/generated/yohou.plotting.evaluation.plot_score_time_series/) shows per-timestep MAE for each strategy,
    and [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) overlays all three predictions against actuals.
    """)


@app.cell
def _(MeanAbsoluteError, y_pred_actual, y_pred_predicted, y_pred_rewind, y_train):
    scorer = MeanAbsoluteError()
    scorer.fit(y_train)
    strategies = {
        "actual": y_pred_actual,
        "predicted": y_pred_predicted,
        "rewind": y_pred_rewind,
    }
    return scorer, strategies


@app.cell
def _(plot_score_time_series, scorer, strategies, y_test):
    plot_score_time_series(
        scorer,
        y_test,
        strategies,
        title="Per-Timestep MAE by Strategy",
    )


@app.cell
def _(plot_forecast, strategies, y_test, y_train):
    plot_forecast(
        y_test,
        strategies,
        y_train=y_train,
        n_history=36,
        title="All Strategies vs Actuals",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. split_ratio Tuning

    `split_ratio` controls how the training data is split between
    fitting the feature forecaster and the target forecaster.
    """)


@app.cell
def _(
    ForecastedFeatureForecaster,
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    X_actual_train,
    horizon,
    mo,
    pl,
    y_test,
    y_train,
):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _rows = []
    for _ratio in [0.55, 0.65, 0.75]:
        _fc = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(
                estimator=Ridge(alpha=1.0),
                actual_transformer=LagTransformer(lag=[1, 3]),
            ),
            feature_forecaster=PointReductionForecaster(
                estimator=Ridge(alpha=1.0),
                actual_transformer=LagTransformer(lag=[1, 3]),
            ),
            strategy="predicted",
            split_ratio=_ratio,
        )
        _fc.fit(y_train, X_actual_train, forecasting_horizon=horizon)
        _pred = _fc.predict(forecasting_horizon=horizon)
        _mae = float(_scorer.score(y_test, _pred))
        _rows.append({"split_ratio": _ratio, "MAE": round(_mae, 3)})

    mo.ui.table(pl.DataFrame(_rows))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. feature_stride: Refresh the Feature Forecast Less Often

    When the feature forecaster is expensive and cannot be re-run every step in
    production, set `feature_stride` to the refresh cadence. The feature forecast is
    regenerated every `feature_stride` steps and reused in between, both at fit and at
    serve, so the target trains on features of the same age it sees in production.
    `feature_stride` takes effect through `observe_predict` (rolling), not a bare
    `predict`.
    """)


@app.cell
def _(
    ForecastedFeatureForecaster,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    X_actual_test,
    X_actual_train,
    horizon,
    y_test,
    y_train,
):
    ff_stride = ForecastedFeatureForecaster(
        target_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            actual_transformer=LagTransformer(lag=[1, 3]),
        ),
        feature_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            actual_transformer=LagTransformer(lag=[1, 3]),
        ),
        strategy="rewind",
        feature_stride=2,  # regenerate the feature forecast every 2 steps
    )
    ff_stride.fit(y_train, X_actual_train, forecasting_horizon=horizon)
    # Rolling walk-forward over the test set: observe_predict refreshes the feature
    # forecast every 2 steps and reuses it in between.
    y_pred_stride = ff_stride.observe_predict(y_test, X_actual_test, stride=horizon)
    print(f"feature_stride=2 rolling observe_predict: {y_pred_stride['vintage_time'].n_unique()} vintages")
    y_pred_stride.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Pipeline composition**: See [`examples/data-features/pipeline_composition.py`](/examples/pipeline_composition/)
    - **Feature union**: See [`examples/data-features/feature_union.py`](/examples/feature_union/)
    """)


if __name__ == "__main__":
    app.run()
