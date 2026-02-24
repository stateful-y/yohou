# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Forecasting with Forecasted Features.

Demonstrates ForecastedFeatureForecaster for chaining feature and target forecasting.
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Forecasting with Forecasted Features

    When exogenous features (X) are **not known in advance** at prediction time,
    you need to forecast them first. `ForecastedFeatureForecaster` chains a
    **feature forecaster** and a **target forecaster** automatically.

    ## What You'll Learn

    - When and why to use `ForecastedFeatureForecaster`
    - The three training strategies: `"actual"`, `"predicted"`, `"reset"`
    - Comparing strategies on real data
    - Using `update` for rolling evaluation

    ## Prerequisites

    Familiarity with `PointReductionForecaster`.
    """)

@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import ForecastedFeatureForecaster
    from yohou.datasets import fetch_electricity_demand
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
        fetch_electricity_demand,
        pl,
        plot_forecast,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data with Exogenous Features

    We'll forecast Victoria's electricity demand using NSW demand as an
    exogenous feature.  At prediction time, NSW demand is unknown, it
    must be forecasted too.
    """)

@app.cell
def _(fetch_electricity_demand, pl):
    raw = fetch_electricity_demand().frame

    # Resample to daily for speed (drop trailing all-null days)
    daily = (
        raw.group_by_dynamic("time", every="1d")
        .agg(
            pl.col("vic__demand").mean().alias("demand"),
            pl.col("nsw__demand").mean().alias("nsw_demand"),
        )
        .sort("time")
        .drop_nulls()
    )

    y = daily.select("time", "demand")
    X = daily.select("time", "nsw_demand")

    split_idx = len(y) - 14
    y_train, y_test = y.head(split_idx), y.tail(len(y) - split_idx)
    X_train, X_test = X.head(split_idx), X.tail(len(X) - split_idx)
    forecasting_horizon = len(y_test)

    print(f"y columns: {y.columns}, X columns: {X.columns}")
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    return (
        X,
        X_test,
        X_train,
        daily,
        forecasting_horizon,
        split_idx,
        y,
        y_test,
        y_train,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. ForecastedFeatureForecaster with "actual" Strategy

    The default `strategy="actual"` trains the target forecaster on actual X values.
    Simple, but the target model sees perfect features at train time vs. noisy
    forecasted features at predict time.
    """)

@app.cell
def _(
    ForecastedFeatureForecaster,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    X_train,
    forecasting_horizon,
    y_train,
):
    ff_actual = ForecastedFeatureForecaster(
        target_forecaster=PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(lag=[1, 2, 3]),
        ),
        feature_forecaster=PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(lag=[1, 2, 3]),
        ),
        strategy="actual",
    )

    ff_actual.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)
    y_pred_actual = ff_actual.predict(forecasting_horizon=forecasting_horizon)

    print(f"Strategy='actual', predicted {len(y_pred_actual)} steps")
    y_pred_actual.head()
    return ff_actual, y_pred_actual

@app.cell
def _(MeanAbsoluteError, y_pred_actual, y_test, y_train):
    mae_actual = MeanAbsoluteError()
    mae_actual.fit(y_train)
    score_actual = mae_actual.score(y_test, y_pred_actual)
    print(f"MAE (actual strategy): {score_actual:.2f}")
    return (mae_actual, score_actual)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Comparing Training Strategies

    - `"actual"`: Target sees real X at train: simple, fast
    - `"predicted"`: Split data, target sees forecasted X: avoids distribution shift

    Let's compare both.
    """)

@app.cell
def _(
    ForecastedFeatureForecaster,
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    X_train,
    forecasting_horizon,
    y_test,
    y_train,
):
    strategy_results = {}

    for strategy in ["actual", "predicted"]:
        ff = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(
                estimator=Ridge(),
                feature_transformer=LagTransformer(lag=[1, 2, 3]),
            ),
            feature_forecaster=PointReductionForecaster(
                estimator=Ridge(),
                feature_transformer=LagTransformer(lag=[1, 2, 3]),
            ),
            strategy=strategy,
            split_ratio=0.6,  # only used for "predicted"
        )
        ff.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)
        _pred = ff.predict(forecasting_horizon=forecasting_horizon)

        _scorer = MeanAbsoluteError()
        _scorer.fit(y_train)
        _score = _scorer.score(y_test, _pred)

        strategy_results[strategy] = {"pred": _pred, "mae": _score}
        print(f"strategy={strategy:>10s}  MAE: {_score:.2f}")
    return (strategy_results,)

@app.cell
def _(plot_forecast, strategy_results, y_test, y_train):
    preds_dict = {name: r["pred"] for name, r in strategy_results.items()}
    plot_forecast(
        y_test,
        preds_dict,
        y_train=y_train,
        title="ForecastedFeatureForecaster: Strategy Comparison",
    )
    return (preds_dict,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Update Workflow

    `ForecastedFeatureForecaster` supports `update` for both target and feature
    forecasters simultaneously.
    """)

@app.cell
def _(
    ForecastedFeatureForecaster,
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    X_test,
    X_train,
    forecasting_horizon,
    pl,
    y_test,
    y_train,
):
    ff_rolling = ForecastedFeatureForecaster(
        target_forecaster=PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(lag=[1, 2, 3]),
        ),
        feature_forecaster=PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(lag=[1, 2, 3]),
        ),
        strategy="actual",
    )
    ff_rolling.fit(y_train, X_train, forecasting_horizon=forecasting_horizon)

    step = forecasting_horizon
    rolling_preds = []

    for i in range(0, len(y_test), step):
        chunk_y = y_test.slice(i, min(step, len(y_test) - i))
        chunk_X = X_test.slice(i, min(step, len(X_test) - i))
        if len(chunk_y) == 0:
            break
        _rpred = ff_rolling.predict(forecasting_horizon=len(chunk_y))
        rolling_preds.append(_rpred)
        ff_rolling.observe(chunk_y, chunk_X)

    all_preds = pl.concat(rolling_preds)
    y_test_aligned = y_test.head(len(all_preds))

    mae_rolling = MeanAbsoluteError()
    mae_rolling.fit(y_train)
    score_rolling = mae_rolling.score(y_test_aligned, all_preds)

    print(f"Rolling evaluation: {len(rolling_preds)} windows")
    print(f"Rolling MAE: {score_rolling:.2f}")
    return all_preds, ff_rolling, mae_rolling, score_rolling

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - `ForecastedFeatureForecaster` automates the two-step forecast: features first, then target
    - Three strategies control how the target forecaster sees features during training
    - `"actual"` is simplest; `"predicted"` / `"reset"` reduce train-test distribution shift
    - `update()` passes new observations to both internal forecasters
    """)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Multi-column**: See `multi_column_forecasting.py` for `ColumnForecaster`
    - **Interval forecasting**: See `interval/` for prediction intervals
    - **Decomposition**: See `stationarity/` for `DecompositionPipeline`
    """)

if __name__ == "__main__":
    app.run()
