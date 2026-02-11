"""Multi-Column Forecasting with ColumnForecaster.

Demonstrates applying different forecasters to different column subsets.
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
async def _():
    import sys

    if "pyodide" in sys.modules:
        import micropip

        await micropip.install(["plotly", "scikit-learn", "yohou"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multi-Column Forecasting with ColumnForecaster

    When a dataset contains multiple target columns, you may want to apply
    **different forecasters** to each column (or column group). `ColumnForecaster`
    is yohou's answer to sklearn's `ColumnTransformer`, but for forecasters.

    ## What You'll Learn

    - Creating multivariate time series data
    - Using `ColumnForecaster` to assign forecasters to column subsets
    - Handling remainder columns with a fallback forecaster
    - Accessing fitted sub-forecasters by name
    - Scaling to many columns with the ETTm1 dataset

    ## Prerequisites

    Familiarity with `SeasonalNaive` and `PointReductionForecaster`.
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import ColumnForecaster
    from yohou.datasets import load_ett_m1, load_vic_electricity
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer

    return (
        ColumnForecaster,
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        SeasonalNaive,
        load_ett_m1,
        load_vic_electricity,
        pl,
        plot_forecast,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load Multivariate Data

    The Victoria Electricity dataset has several numeric columns.
    We'll select `Demand` and `Temperature` as targets.
    """)


@app.cell
def _(load_vic_electricity, pl):
    raw = load_vic_electricity()

    # Resample to daily to keep the example fast
    y_full = (
        raw.group_by_dynamic("time", every="1d")
        .agg(
            pl.col("Demand").mean(),
            pl.col("Temperature").mean(),
        )
        .sort("time")
    )

    # Use last 365 days
    y_full = y_full.tail(365)

    split_idx = len(y_full) - 30
    y_train = y_full.head(split_idx)
    y_test = y_full.tail(len(y_full) - split_idx)
    forecasting_horizon = len(y_test)

    print(f"Columns: {y_full.columns}")
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    y_train.head()
    return forecasting_horizon, split_idx, y_full, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. ColumnForecaster Basics

    Each entry is a `(name, forecaster, columns)` tuple.
    Here we use `SeasonalNaive` for Demand and a `PointReductionForecaster` for Temperature.
    """)


@app.cell
def _(
    ColumnForecaster,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SeasonalNaive,
    forecasting_horizon,
    y_train,
):
    col_fc = ColumnForecaster(
        forecasters=[
            ("demand", SeasonalNaive(seasonality=7), "Demand"),
            (
                "temp",
                PointReductionForecaster(
                    estimator=Ridge(),
                    feature_transformer=LagTransformer(lag=list(range(1, 8))),
                ),
                "Temperature",
            ),
        ],
    )

    col_fc.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred = col_fc.predict(forecasting_horizon=forecasting_horizon)

    print(f"Prediction columns: {y_pred.columns}")
    y_pred.head()
    return col_fc, y_pred


@app.cell
def _(MeanAbsoluteError, y_pred, y_test, y_train):
    mae = MeanAbsoluteError()
    mae.fit(y_train)
    score = mae.score(y_test, y_pred)
    print(f"Multi-column MAE (aggregated): {score:.2f}")
    return (mae, score)


@app.cell
def _(plot_forecast, y_pred, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred,
        y_train=y_train,
        title="ColumnForecaster: Demand + Temperature",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Accessing Sub-Forecasters

    After fitting, access individual forecasters by name via `named_forecasters_`.
    """)


@app.cell
def _(col_fc):
    for name, fc, cols in col_fc.forecasters_:
        print(f"{name}: {type(fc).__name__} → {cols}")

    demand_fc = col_fc.named_forecasters_["demand"]
    print(f"\nDemand forecaster seasonality: {demand_fc.seasonality}")
    return (demand_fc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Remainder Handling

    By default `remainder="drop"`, columns not assigned to any forecaster are
    excluded from predictions. Pass a forecaster to handle them automatically.
    """)


@app.cell
def _(
    ColumnForecaster,
    SeasonalNaive,
    forecasting_horizon,
    y_train,
):
    col_fc_rem = ColumnForecaster(
        forecasters=[
            ("demand", SeasonalNaive(seasonality=7), "Demand"),
        ],
        remainder=SeasonalNaive(seasonality=7),  # fallback for Temperature
    )

    col_fc_rem.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred_rem = col_fc_rem.predict(forecasting_horizon=forecasting_horizon)

    print(f"Prediction columns (with remainder): {y_pred_rem.columns}")
    print(f"Remainder columns: {col_fc_rem.remainder_cols_}")
    return col_fc_rem, y_pred_rem


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Scaling to Many Columns: ETTm1 Dataset

    The **ETTm1** dataset has 7 numeric columns (6 temperature features + 1 Oil Temperature target).
    `ColumnForecaster` handles this effortlessly, assign a dedicated forecaster
    to the target and use `remainder` for all features.
    """)


@app.cell
def _(load_ett_m1, pl):
    ett = load_ett_m1()

    # Resample to hourly (from 15-min) to keep the example fast
    ett_hourly = (
        ett.group_by_dynamic("time", every="1h")
        .agg(pl.all().exclude("time").mean())
        .sort("time")
    )

    # Use last 30 days
    ett_sub = ett_hourly.tail(30 * 24)

    ett_split = len(ett_sub) - 48  # 48 hours test
    ett_train = ett_sub.head(ett_split)
    ett_test = ett_sub.tail(len(ett_sub) - ett_split)
    ett_horizon = len(ett_test)

    print(f"ETTm1 columns: {ett_sub.columns}")
    print(f"Train: {len(ett_train)}, Test: {len(ett_test)}")
    return ett_horizon, ett_test, ett_train


@app.cell
def _(
    ColumnForecaster,
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    SeasonalNaive,
    ett_horizon,
    ett_test,
    ett_train,
):
    # Dedicated Ridge forecaster for OT (Oil Temperature target)
    # SeasonalNaive(24h) as remainder for all HUFL/HULL/MUFL/MULL/LUFL/LULL
    col_fc_ett = ColumnForecaster(
        forecasters=[
            (
                "oil_temp",
                PointReductionForecaster(
                    estimator=Ridge(),
                    feature_transformer=LagTransformer(lag=[1, 2, 3, 24]),
                ),
                "OT",
            ),
        ],
        remainder=SeasonalNaive(seasonality=24),  # daily cycle for remaining columns
    )

    col_fc_ett.fit(ett_train, forecasting_horizon=ett_horizon)
    ett_pred = col_fc_ett.predict(forecasting_horizon=ett_horizon)

    print(f"ETTm1 prediction columns: {ett_pred.columns}")
    print(f"Remainder columns: {col_fc_ett.remainder_cols_}")

    # Score
    ett_mae = MeanAbsoluteError()
    ett_mae.fit(ett_test)
    ett_score = ett_mae.score(ett_test, ett_pred)
    print(f"\nETTm1 Multi-column MAE: {ett_score:.2f}")
    return col_fc_ett, ett_pred, ett_score


@app.cell
def _(ett_pred, ett_test, ett_train, plot_forecast):
    plot_forecast(
        ett_test,
        ett_pred,
        y_train=ett_train,
        title="ETTm1: ColumnForecaster (Ridge for OT, SeasonalNaive for rest)",
        columns=["OT"],
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - `ColumnForecaster` maps different forecasters to different column subsets
    - Each `(name, forecaster, columns)` tuple defines one sub-forecaster
    - Use `remainder=` to handle unassigned columns with a fallback forecaster
    - Access fitted forecasters via `named_forecasters_["name"]`
    - Predictions are concatenated horizontally from all sub-forecasters
    - Scales naturally from 2 columns (Vic Electricity) to 7+ columns (ETTm1)
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Feature forecasting**: See `feature_forecasting.py` for `ForecastedFeatureForecaster`
    - **Decomposition**: See `stationarity/` for `DecompositionPipeline`
    - **Panel data**: See `examples/panel_data.py` for panel forecasting
    - **ETTm1 exploration**: See `examples/datasets/ett_m1.py` for detailed analysis
    """)


if __name__ == "__main__":
    app.run()
