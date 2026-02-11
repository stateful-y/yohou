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
async def _():
    import sys as _sys

    if "pyodide" in _sys.modules:
        import micropip

        await micropip.install(["plotly", "scikit-learn", "yohou"])
    return


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
    - Strategy comparison on ETT-M1 multivariate data
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import ForecastedFeatureForecaster
    from yohou.datasets import load_ett_m1
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
        load_ett_m1,
        pl,
        plot_forecast,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Multivariate Data
    """)
    return


@app.cell
def _(load_ett_m1, mo, pl):
    _ett = load_ett_m1()
    # Resample to hourly
    ett = (
        _ett.group_by_dynamic("time", every="1h")
        .agg(
            pl.col("OT").mean(),
            pl.col("HUFL").mean(),
            pl.col("HULL").mean(),
            pl.col("MUFL").mean(),
        )
    )
    _split = int(len(ett) * 0.85)
    y_train = ett.head(_split).select("time", "OT")
    y_test = ett.tail(len(ett) - _split).select("time", "OT")
    X_train = ett.head(_split).select("time", "HUFL", "HULL", "MUFL")
    X_test = ett.tail(len(ett) - _split).select("time", "HUFL", "HULL", "MUFL")
    horizon = len(y_test)

    mo.md(
        f"**ETT-M1 (hourly)**: {len(ett)} rows\n\n"
        f"**Target**: OT, **Features**: HUFL, HULL, MUFL\n\n"
        f"**Train**: {len(y_train)}, **Test**: {len(y_test)}"
    )
    return X_test, X_train, ett, horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Strategy: "actual"

    The target forecaster is trained on actual feature values. At
    prediction time, you must provide X (or use the default forecast).
    """)
    return


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
            feature_transformer=LagTransformer(lag=[1, 24]),
        ),
        feature_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 24]),
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
    return


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
            feature_transformer=LagTransformer(lag=[1, 24]),
        ),
        feature_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 24]),
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
    return


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
            feature_transformer=LagTransformer(lag=[1, 24]),
        ),
        feature_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 24]),
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
    return


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
    return


@app.cell
def _(plot_forecast, y_pred_predicted, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_predicted,
        y_train=y_train,
        n_history=168,
        title="ForecastedFeatureForecaster (strategy='predicted')",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. split_ratio Tuning

    `split_ratio` controls how the training data is split between
    fitting the feature forecaster and the target forecaster.
    """)
    return


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
                feature_transformer=LagTransformer(lag=[1, 24]),
            ),
            feature_forecaster=PointReductionForecaster(
                estimator=Ridge(alpha=1.0),
                feature_transformer=LagTransformer(lag=[1, 24]),
            ),
            strategy="predicted",
            split_ratio=_ratio,
        )
        _fc.fit(y_train, X_train, forecasting_horizon=horizon)
        _pred = _fc.predict(forecasting_horizon=horizon)
        _mae = float(_scorer.score(y_test, _pred))
        _rows.append({"split_ratio": _ratio, "MAE": round(_mae, 3)})

    mo.ui.table(pl.DataFrame(_rows))
    return


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

    - **ETT-M1 exploration**: See `examples/datasets/ett_m1_multivariate.py`
    - **Pipeline composition**: See `examples/compose/pipeline_composition.py`
    - **Feature union**: See `examples/compose/feature_union.py`
    """)
    return


if __name__ == "__main__":
    app.run()
