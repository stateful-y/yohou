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
    - Scaling to many columns with the Hospital dataset

    ## Prerequisites

    Familiarity with `SeasonalNaive` and `PointReductionForecaster`.
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import ColumnForecaster
    from yohou.datasets import fetch_electricity_demand, fetch_hospital
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
        fetch_electricity_demand,
        fetch_hospital,
        pl,
        plot_forecast,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load Multivariate Data

    The Electricity Demand dataset has demand columns for several Australian states.
    We'll select Victoria and NSW demand as targets.
    """)


@app.cell
def _(fetch_electricity_demand, pl):
    raw = fetch_electricity_demand().frame

    # Resample to daily to keep the example fast
    y_full = (
        raw.group_by_dynamic("time", every="1d")
        .agg(
            pl.col("vic__demand").mean().alias("vic_demand"),
            pl.col("nsw__demand").mean().alias("nsw_demand"),
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
    Here we use `SeasonalNaive` for vic_demand and a `PointReductionForecaster` for nsw_demand.
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
            ("vic_demand", SeasonalNaive(seasonality=7), "vic_demand"),
            (
                "nsw_demand",
                PointReductionForecaster(
                    estimator=Ridge(),
                    feature_transformer=LagTransformer(lag=list(range(1, 8))),
                ),
                "nsw_demand",
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
        title="ColumnForecaster: VIC Demand + NSW Demand",
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

    vic_demand_fc = col_fc.named_forecasters_["vic_demand"]
    print(f"\nVIC demand forecaster seasonality: {vic_demand_fc.seasonality}")
    return (vic_demand_fc,)


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
            ("vic_demand", SeasonalNaive(seasonality=7), "vic_demand"),
        ],
        remainder=SeasonalNaive(seasonality=7),  # fallback for nsw_demand
    )

    col_fc_rem.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred_rem = col_fc_rem.predict(forecasting_horizon=forecasting_horizon)

    print(f"Prediction columns (with remainder): {y_pred_rem.columns}")
    print(f"Remainder columns: {col_fc_rem.remainder_cols_}")
    return col_fc_rem, y_pred_rem


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Scaling to Many Columns: Hospital Dataset

    The **Hospital** dataset has 767 monthly patient count series.
    We select 7 to demonstrate `ColumnForecaster` at scale, assigning a
    dedicated forecaster to the primary series and using `remainder` for the rest.
    """)


@app.cell
def _(fetch_hospital, pl):
    hosp = fetch_hospital().frame

    # Select first 7 series and rename to remove __ prefix for multivariate use
    _selected = [f"T{i}__patients" for i in range(1, 8)]
    _renamed = {f"T{i}__patients": f"T{i}_patients" for i in range(1, 8)}
    hosp_sub = hosp.select("time", *_selected).rename(_renamed).drop_nulls()

    hosp_split = len(hosp_sub) - 12  # 12 months test
    hosp_train = hosp_sub.head(hosp_split)
    hosp_test = hosp_sub.tail(len(hosp_sub) - hosp_split)
    hosp_horizon = len(hosp_test)

    print(f"Hospital columns: {hosp_sub.columns}")
    print(f"Train: {len(hosp_train)}, Test: {len(hosp_test)}")
    return hosp_horizon, hosp_test, hosp_train


@app.cell
def _(
    ColumnForecaster,
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    SeasonalNaive,
    hosp_horizon,
    hosp_test,
    hosp_train,
):
    # Dedicated Ridge forecaster for T1_patients (primary target)
    # SeasonalNaive(12mo) as remainder for T2..T7_patients
    col_fc_hosp = ColumnForecaster(
        forecasters=[
            (
                "primary",
                PointReductionForecaster(
                    estimator=Ridge(),
                    feature_transformer=LagTransformer(lag=[1, 2, 3, 12]),
                ),
                "T1_patients",
            ),
        ],
        remainder=SeasonalNaive(seasonality=12),  # yearly cycle for remaining columns
    )

    col_fc_hosp.fit(hosp_train, forecasting_horizon=hosp_horizon)
    hosp_pred = col_fc_hosp.predict(forecasting_horizon=hosp_horizon)

    print(f"Hospital prediction columns: {hosp_pred.columns}")
    print(f"Remainder columns: {col_fc_hosp.remainder_cols_}")

    # Score
    hosp_mae = MeanAbsoluteError()
    hosp_mae.fit(hosp_test)
    hosp_score = hosp_mae.score(hosp_test, hosp_pred)
    print(f"\nHospital Multi-column MAE: {hosp_score:.2f}")
    return col_fc_hosp, hosp_pred, hosp_score


@app.cell
def _(hosp_pred, hosp_test, hosp_train, plot_forecast):
    plot_forecast(
        hosp_test,
        hosp_pred,
        y_train=hosp_train,
        title="Hospital: ColumnForecaster (Ridge for T1, SeasonalNaive for rest)",
        columns=["T1_patients"],
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
    - Scales naturally from 2 columns (Electricity Demand) to 7+ columns (Hospital)
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Feature forecasting**: See `feature_forecasting.py` for `ForecastedFeatureForecaster`
    - **Decomposition**: See `stationarity/` for `DecompositionPipeline`
    - **Panel data**: See `examples/panel_data.py` for panel forecasting
    """)


if __name__ == "__main__":
    app.run()
