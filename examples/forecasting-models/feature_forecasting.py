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
    "title": "How to Build a Lag-Feature Forecaster",
    "description": "Chain feature and target forecasters with ForecastedFeatureForecaster when exogenous variables are unknown at prediction time and must be forecasted.",
    "category": "how-to",
    "section": "forecasting-models",
    "companion": "/pages/how-to/build-reduction-forecasters/",
    "api_references": ["ColumnForecaster", "ForecastedFeatureForecaster", "PointReductionForecaster"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Forecasting with Forecasted Features

    When exogenous features (X_actual) are **not known in advance** at prediction time,
    you need to forecast them first. [`ForecastedFeatureForecaster`](/pages/api/generated/yohou.compose.forecasted_feature_forecaster.ForecastedFeatureForecaster/) chains a
    **feature forecaster** and a **target forecaster** automatically.

    **Prerequisites:** Familiarity with [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/).
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import ColumnForecaster, ForecastedFeatureForecaster
    from yohou.datasets import fetch_electricity_demand
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        ColumnForecaster,
        ForecastedFeatureForecaster,
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        fetch_electricity_demand,
        pl,
        plot_forecast,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data with Exogenous Features

    We'll forecast Victoria's electricity demand using NSW demand as an
    exogenous feature plus **calendar features** (day of week, month).
    All exogenous features are forecasted by the feature forecaster.
    Calendar features follow predictable patterns, so they are easy to
    forecast accurately.
    """)


@app.cell
def _(fetch_electricity_demand, mo, pl, plot_time_series):
    raw = fetch_electricity_demand().frame

    # Resample to daily for speed (drop trailing all-null days)
    daily = (
        raw
        .group_by_dynamic("time", every="1d")
        .agg(
            pl.col("vic__demand").mean().alias("demand"),
            pl.col("nsw__demand").mean().alias("nsw_demand"),
        )
        .sort("time")
        .drop_nulls()
    )

    y = daily.select("time", "demand")

    # NSW demand + calendar features as exogenous
    X_actual = daily.select(
        "time",
        "nsw_demand",
        pl.col("time").dt.weekday().cast(pl.Float64).alias("day_of_week"),
        pl.col("time").dt.month().cast(pl.Float64).alias("month"),
    )

    from yohou.model_selection import train_test_split

    y_train, y_test, X_actual_train, X_actual_test = train_test_split(
        y, X_actual, test_size=14
    )
    forecasting_horizon = len(y_test)

    mo.vstack([
        plot_time_series(y, title="Target: VIC Demand"),
        plot_time_series(X_actual, title="Exogenous Features: NSW Demand + Calendar"),
    ])
    return X_actual_test, X_actual_train, forecasting_horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. ForecastedFeatureForecaster with "actual" Strategy

    The default `strategy="actual"` trains the target forecaster on actual X_actual values.
    Simple, but the target model sees perfect features at train time vs. noisy
    forecasted features at predict time.

    The `feature_forecaster` forecasts all exogenous columns (NSW demand and
    calendar features).  Calendar patterns are easy for the feature forecaster
    to capture, while NSW demand requires real forecasting effort.
    """)


@app.cell
def _(
    ForecastedFeatureForecaster,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    X_actual_train,
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

    ff_actual.fit(y_train, X_actual_train, forecasting_horizon=forecasting_horizon)
    y_pred_actual = ff_actual.predict(forecasting_horizon=forecasting_horizon)

    print(f"Strategy='actual', predicted {len(y_pred_actual)} steps")
    y_pred_actual.head()
    return (y_pred_actual,)


@app.cell
def _(MeanAbsoluteError, y_pred_actual, y_test, y_train):
    mae_actual = MeanAbsoluteError()
    mae_actual.fit(y_train)
    score_actual = mae_actual.score(y_test, y_pred_actual)
    print(f"MAE (actual strategy): {score_actual:.2f}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Comparing Training Strategies

    The `strategy` parameter controls what features the **target forecaster**
    sees during training:

    | Strategy | How the target trains | Trade-off |
    |----------|----------------------|----------|
    | `"actual"` | On the **real** X_actual values | Simplest and fastest, but the target sees perfect features at train time yet noisy forecasted features at test time (distribution shift) |
    | `"predicted"` | On **forecasted** X_actual from a held-out split (`split_ratio`) | No distribution shift, but both forecasters train on less data because the training set is split in two |
    | `"rewind"` | On **forecasted** X_actual produced by `observe` / `rewind` over the full training set | Best of both worlds: the target sees realistic noisy features **and** trains on the full dataset. Slower because the feature forecaster produces rolling predictions row-by-row |

    Let's compare all three.
    """)


@app.cell
def _(
    ForecastedFeatureForecaster,
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    X_actual_train,
    forecasting_horizon,
    y_test,
    y_train,
):
    strategy_results = {}

    for strategy in ["actual", "predicted", "rewind"]:
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
        ff.fit(y_train, X_actual_train, forecasting_horizon=forecasting_horizon)
        _pred = ff.predict(forecasting_horizon=forecasting_horizon)

        _scorer = MeanAbsoluteError()
        _scorer.fit(y_train)
        _score = _scorer.score(y_test, _pred)

        strategy_results[f"Forecast with {strategy} strategy"] = {"pred": _pred, "mae": _score}
        print(f"strategy={strategy:>10s}  MAE: {_score:.2f}")
    return (strategy_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) overlays predictions from all three strategies against the
    test actuals so you can visually compare `"actual"`, `"predicted"`, and
    `"rewind"`.
    """)


@app.cell
def _(plot_forecast, strategy_results, y_test, y_train):
    preds_dict = {name: r["pred"] for name, r in strategy_results.items()}
    plot_forecast(
        y_test,
        preds_dict,
        y_train=y_train,
        n_history=365,
        title="ForecastedFeatureForecaster: Strategy Comparison",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Update Workflow

    [`ForecastedFeatureForecaster`](/pages/api/generated/yohou.compose.forecasted_feature_forecaster.ForecastedFeatureForecaster/) supports `observe` for rolling evaluation.
    New observations are passed to both the target and feature forecasters
    simultaneously, updating their internal buffers without refitting.
    """)


@app.cell
def _(
    ForecastedFeatureForecaster,
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    X_actual_test,
    X_actual_train,
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
    ff_rolling.fit(y_train, X_actual_train, forecasting_horizon=forecasting_horizon)

    step = forecasting_horizon
    rolling_preds = []

    for i in range(0, len(y_test), step):
        chunk_y = y_test.slice(i, min(step, len(y_test) - i))
        chunk_X = X_actual_test.slice(i, min(step, len(X_actual_test) - i))
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Known-in-Advance Features with ColumnForecaster

    Calendar features like `day_of_week` and `month` are **known in advance**:
    we always know what day it will be tomorrow.  Forecasting them wastes
    effort and adds noise.

    The trick: use a [`ColumnForecaster`](/pages/api/generated/yohou.compose.column_forecaster.ColumnForecaster/) as the `feature_forecaster` that
    only forecasts `nsw_demand` and **drops** the calendar columns.  Pass the
    known calendar values via `X_future` at both fit and predict, so the target
    forecaster builds step columns for them and uses the known future values
    directly (no forecasting needed).
    """)


@app.cell
def _(
    ColumnForecaster,
    ForecastedFeatureForecaster,
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    X_actual_test,
    X_actual_train,
    forecasting_horizon,
    plot_forecast,
    y_test,
    y_train,
):
    ff_known = ForecastedFeatureForecaster(
        target_forecaster=PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(lag=[1, 2, 3]),
        ),
        feature_forecaster=ColumnForecaster(
            forecasters=[
                (
                    "nsw",
                    PointReductionForecaster(
                        estimator=Ridge(),
                        feature_transformer=LagTransformer(lag=[1, 2, 3]),
                    ),
                    "nsw_demand",
                ),
            ],
            remainder="drop",  # calendar columns are NOT forecasted
        ),
        strategy="actual",
    )

    # Known-ahead calendar features travel through the X_future channel at both
    # fit and predict (the target builds step columns for them at fit, then
    # consumes the known future values at predict).
    X_known_train = X_actual_train.select("time", "day_of_week", "month")
    ff_known.fit(
        y_train,
        X_actual_train,
        forecasting_horizon=forecasting_horizon,
        X_future=X_known_train,
    )

    # Pass known calendar features via X_future at predict time
    X_known = X_actual_test.select("time", "day_of_week", "month")
    y_pred_known = ff_known.predict(forecasting_horizon=forecasting_horizon, X_future=X_known)

    mae_known = MeanAbsoluteError()
    mae_known.fit(y_train)
    score_known = mae_known.score(y_test, y_pred_known)
    print(f"MAE (known-ahead calendar): {score_known:.2f}")

    plot_forecast(
        y_test,
        y_pred_known,
        y_train=y_train,
        n_history=60,
        title="Known-in-Advance Calendar Features",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Multi-column**: See [`multi_column_forecasting.py`](/examples/multi_column_forecasting/) for [`ColumnForecaster`](/pages/api/generated/yohou.compose.column_forecaster.ColumnForecaster/)
    - **Interval forecasting**: See [Forecasting Models](/pages/examples/#forecasting-models) for prediction intervals
    - **Decomposition**: See [Data & Features](/pages/examples/#data-features) for [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/)
    """)


if __name__ == "__main__":
    app.run()
