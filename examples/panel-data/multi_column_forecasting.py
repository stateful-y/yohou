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
    "title": "How to Forecast Multiple Columns Independently",
    "description": "Use ColumnForecaster to apply a point forecaster independently to each column of a multivariate time series.",
    "category": "how-to",
    "companion": "/pages/how-to/panel-data/",
    "section": "panel-data",
    "api_references": ["ColumnForecaster", "PointReductionForecaster", "SeasonalNaive"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multi-Column Forecasting with ColumnForecaster

    When a dataset contains multiple target columns, you may want to apply
    **different forecasters** to each column (or column group). [`ColumnForecaster`](/pages/api/generated/yohou.compose.column_forecaster.ColumnForecaster/)
    is yohou's answer to sklearn's [`ColumnTransformer`](/pages/api/generated/yohou.compose.column_transformer.ColumnTransformer/), but for forecasters.

    **Prerequisites:** Familiarity with [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/) and [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/).
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import ColumnForecaster
    from yohou.datasets import fetch_electricity_demand, fetch_hospital
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast, plot_time_series
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
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load Multivariate Data

    The Electricity Demand dataset has demand columns for several Australian states.
    We'll select Victoria and NSW demand as targets.
    """)


@app.cell
def _(fetch_electricity_demand, pl, plot_time_series):
    raw = fetch_electricity_demand().frame

    # Resample to daily to keep the example fast (drop trailing all-null days)
    y_full = (
        raw
        .group_by_dynamic("time", every="1d")
        .agg(
            pl.col("vic__demand").mean().alias("vic_demand"),
            pl.col("nsw__demand").mean().alias("nsw_demand"),
        )
        .sort("time")
        .drop_nulls()
    )

    # Use last 365 days
    y_full = y_full.tail(365)

    from yohou.model_selection import train_test_split

    y_train, y_test = train_test_split(y_full, test_size=30)
    forecasting_horizon = len(y_test)

    plot_time_series(y_full, title="Daily Electricity Demand (VIC + NSW)")
    return forecasting_horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. ColumnForecaster Basics

    Each entry is a `(name, forecaster, columns)` tuple.
    Here we use [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/) for vic_demand and a [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) for nsw_demand.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) displays each predicted column against its test actuals.
    When multiple target columns are present, the plot overlays them in a
    single figure.
    """)


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
    plot_forecast,
    y_test,
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

    plot_forecast(
        y_test,
        y_pred_rem,
        y_train=y_train,
        title="ColumnForecaster with Remainder (SeasonalNaive fallback)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Scaling to Many Columns: Hospital Dataset

    The **Hospital** dataset has 767 monthly patient count series.
    We select 7 to demonstrate [`ColumnForecaster`](/pages/api/generated/yohou.compose.column_forecaster.ColumnForecaster/) at scale, assigning a
    dedicated forecaster to the primary series and using `remainder` for the rest.
    """)


@app.cell
def _(fetch_hospital, plot_time_series):
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
    plot_time_series(hosp_sub, title="Hospital Patient Counts (7 Series)")
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
    return (hosp_pred,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) with `columns=["T1_patients"]` shows just the primary series
    predicted by the dedicated Ridge forecaster, while the remaining columns
    use [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/) via the `remainder` parameter.
    """)


@app.cell
def _(hosp_pred, hosp_test, hosp_train, plot_forecast):
    plot_forecast(
        hosp_test,
        hosp_pred,
        y_train=hosp_train,
        title="Hospital: ColumnForecaster (Ridge for T1, SeasonalNaive for rest)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Feature forecasting**: See [`feature_forecasting.py`](/examples/feature_forecasting/) for [`ForecastedFeatureForecaster`](/pages/api/generated/yohou.compose.forecasted_feature_forecaster.ForecastedFeatureForecaster/)
    - **Decomposition**: See [Data & Features](/pages/examples/#data-features) for [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/)
    - **Panel data**: See `examples/panel_reduction.py` for panel forecasting
    """)


if __name__ == "__main__":
    app.run()
