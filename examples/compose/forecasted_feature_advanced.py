# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Forecasted Feature Advanced.

Demonstrates ForecastedFeatureForecaster strategies (actual, predicted,
rewind), split_ratio tuning, and multivariate feature chains.
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ForecastedFeatureForecaster: Advanced

    When exogenous features are not known at prediction time,
    `ForecastedFeatureForecaster` chains a feature forecaster with
    a target forecaster.

    ## What You'll Learn

    - **`strategy="actual"`**: Uses real feature values during fit (default)
    - **`strategy="predicted"`**: Uses forecasted features during prediction
    - **`strategy="rewind"`**: Rewinds feature forecaster state before prediction
    - **`split_ratio`**: Controls train/validation split for feature forecaster
    - Strategy comparison on Hospital multivariate data
    """)

@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import ForecastedFeatureForecaster
    from yohou.datasets import fetch_hospital
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast
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
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Multivariate Data
    """)

@app.cell
def _(fetch_hospital, mo, pl):
    _hosp = fetch_hospital().frame
    # Select 4 series and rename for multivariate analysis
    hospital = _hosp.select(
        "time",
        pl.col("T1__patients").alias("target"),
        pl.col("T2__patients").alias("feature_1"),
        pl.col("T3__patients").alias("feature_2"),
        pl.col("T4__patients").alias("feature_3"),
    )
    _split = int(len(hospital) * 0.85)
    y_train = hospital.head(_split).select("time", "target")
    y_test = hospital.tail(len(hospital) - _split).select("time", "target")
    X_train = hospital.head(_split).select("time", "feature_1", "feature_2", "feature_3")
    X_test = hospital.tail(len(hospital) - _split).select("time", "feature_1", "feature_2", "feature_3")
    horizon = len(y_test)

    mo.md(
        f"**Hospital (monthly)**: {len(hospital)} rows\n\n"
        f"**Target**: target, **Features**: feature_1, feature_2, feature_3\n\n"
        f"**Train**: {len(y_train)}, **Test**: {len(y_test)}"
    )
    return X_test, X_train, hospital, horizon, y_test, y_train

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Strategy: "actual"

    The target forecaster is trained on actual feature values. At
    prediction time, you must provide X (or use the default forecast).
    """)

@app.cell
def _(
    ForecastedFeatureForecaster,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    X_train,
    horizon,
    y_train,
):
    fc_actual = ForecastedFeatureForecaster(
        target_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 3]),
        ),
        feature_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 3]),
        ),
        strategy="actual",
    )
    fc_actual.fit(y_train, X_train, forecasting_horizon=horizon)
    y_pred_actual = fc_actual.predict(forecasting_horizon=horizon)
    return fc_actual, y_pred_actual

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Strategy: "predicted"

    Both fit and predict use the feature forecaster's predictions.
    This avoids train-test leakage.
    """)

@app.cell
def _(
    ForecastedFeatureForecaster,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    X_train,
    horizon,
    y_train,
):
    fc_predicted = ForecastedFeatureForecaster(
        target_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 3]),
        ),
        feature_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 3]),
        ),
        strategy="predicted",
        split_ratio=0.7,
    )
    fc_predicted.fit(y_train, X_train, forecasting_horizon=horizon)
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
    X_train,
    horizon,
    y_train,
):
    fc_rewind = ForecastedFeatureForecaster(
        target_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 3]),
        ),
        feature_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 3]),
        ),
        strategy="rewind",
    )
    fc_rewind.fit(y_train, X_train, forecasting_horizon=horizon)
    y_pred_rewind = fc_rewind.predict(forecasting_horizon=horizon)
    return fc_rewind, y_pred_rewind

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Compare Strategies
    """)

@app.cell
def _(MeanAbsoluteError, mo, pl, y_pred_actual, y_pred_predicted, y_pred_rewind, y_test, y_train):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _strategies = {
        "actual": y_pred_actual,
        "predicted": y_pred_predicted,
        "rewind": y_pred_rewind,
    }
    _rows = []
    for _name, _pred in _strategies.items():
        _mae = float(_scorer.score(y_test, _pred))
        _rows.append({"Strategy": _name, "MAE": round(_mae, 3)})

    mo.ui.table(pl.DataFrame(_rows))

@app.cell
def _(plot_forecast, y_pred_predicted, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_predicted,
        y_train=y_train,
        n_history=36,
        title="ForecastedFeatureForecaster (strategy='predicted')",
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
    X_train,
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
                feature_transformer=LagTransformer(lag=[1, 3]),
            ),
            feature_forecaster=PointReductionForecaster(
                estimator=Ridge(alpha=1.0),
                feature_transformer=LagTransformer(lag=[1, 3]),
            ),
            strategy="predicted",
            split_ratio=_ratio,
        )
        _fc.fit(y_train, X_train, forecasting_horizon=horizon)
        _pred = _fc.predict(forecasting_horizon=horizon)
        _mae = float(_scorer.score(y_test, _pred))
        _rows.append({"split_ratio": _ratio, "MAE": round(_mae, 3)})

    mo.ui.table(pl.DataFrame(_rows))

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **`strategy="actual"`**: Target model trained on actual features (best if X is available at predict time)
    - **`strategy="predicted"`**: Both models use forecasted features (no future data leakage)
    - **`strategy="rewind"`**: Rewinds feature forecaster state before prediction
    - **`split_ratio`**: Higher = more data for feature forecaster, less for target forecaster
    - Choose strategy based on whether X is known at prediction time

    ## Next Steps

    - **Pipeline composition**: See `examples/compose/pipeline_composition.py`
    - **Feature union**: See `examples/compose/feature_union.py`
    """)

if __name__ == "__main__":
    app.run()
