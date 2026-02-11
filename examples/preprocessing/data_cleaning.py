"""Data Cleaning: Imputation and Outlier Handling.

Demonstrates SimpleTimeImputer, SeasonalImputer, OutlierThresholdHandler,
and OutlierPercentileHandler.
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
    # Data Cleaning: Imputation and Outlier Handling

    Real-world time series often contain **missing values** and **outliers**.
    Yohou provides temporal-aware imputers and outlier handlers.

    ## What You'll Learn

    - `SimpleTimeImputer`: Linear, forward, backward, nearest interpolation
    - `SeasonalImputer`: Fill missing with seasonal averages
    - `OutlierThresholdHandler`: Clip/nullify by fixed thresholds
    - `OutlierPercentileHandler`: Clip/nullify by learned percentiles
    - Visualizing missing data with `plot_missing_data`

    ## Prerequisites

    None. data cleaning is often the first preprocessing step.
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl

    from yohou.datasets import load_air_passengers
    from yohou.plotting import plot_missing_data, plot_time_series
    from yohou.preprocessing import (
        OutlierPercentileHandler,
        OutlierThresholdHandler,
        SeasonalImputer,
        SimpleTimeImputer,
    )

    return (
        OutlierPercentileHandler,
        OutlierThresholdHandler,
        SeasonalImputer,
        SimpleTimeImputer,
        load_air_passengers,
        pl,
        plot_missing_data,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Create Data with Missing Values

    We construct a small synthetic time series with deliberately introduced missing values to demonstrate the cleaning capabilities.
    """)
    return


@app.cell
def _(load_air_passengers, pl):
    y = (
        load_air_passengers()
        .rename({"Passengers": "passengers"})
    )

    # Introduce missing values at random positions
    import random

    random.seed(42)
    null_indices = sorted(random.sample(range(len(y)), 15))

    y_missing = y.with_row_index("idx").with_columns(
        pl.when(pl.col("idx").is_in(null_indices))
        .then(None)
        .otherwise(pl.col("passengers"))
        .alias("passengers")
    ).drop("idx")

    null_count = y_missing["passengers"].null_count()
    print(f"Introduced {null_count} missing values")
    return null_count, null_indices, random, y, y_missing


@app.cell
def _(plot_missing_data, y_missing):
    plot_missing_data(y_missing, title="Missing Data Pattern")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. SimpleTimeImputer

    Interpolates missing values using temporal methods.
    Methods: `"linear"`, `"forward"`, `"backward"`, `"nearest"`.
    """)
    return


@app.cell
def _(SimpleTimeImputer, y_missing):
    for method in ["linear", "forward", "backward", "nearest"]:
        imputer = SimpleTimeImputer(method=method)
        imputer.fit(y_missing)
        filled = imputer.transform(y_missing)
        remaining_nulls = filled["passengers"].null_count()
        print(f"method={method:>10s}  remaining nulls: {remaining_nulls}")
    return (filled, imputer, method, remaining_nulls)


@app.cell
def _(SimpleTimeImputer, plot_time_series, y_missing):
    linear_imputer = SimpleTimeImputer(method="linear")
    linear_imputer.fit(y_missing)
    y_filled = linear_imputer.transform(y_missing)

    plot_time_series(y_filled, title="After Linear Imputation")
    return linear_imputer, y_filled


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. SeasonalImputer

    Fills missing values using the seasonal average for that position.
    `period=12` means monthly seasonality in monthly data.
    """)
    return


@app.cell
def _(SeasonalImputer, y_missing):
    seasonal_imp = SeasonalImputer(period=12)
    seasonal_imp.fit(y_missing)
    y_seasonal = seasonal_imp.transform(y_missing)

    remaining = y_seasonal["passengers"].null_count()
    print(f"Remaining nulls after seasonal imputation: {remaining}")
    return remaining, seasonal_imp, y_seasonal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Outlier Handling

    Now let's work with outliers. We'll add some extreme values to the clean data.
    """)
    return


@app.cell
def _(pl, y):
    # Add outliers
    y_outlier = y.with_row_index("idx").with_columns(
        pl.when(pl.col("idx").is_in([20, 50, 80, 110]))
        .then(pl.col("passengers") * 3)  # Spike outliers
        .when(pl.col("idx").is_in([30, 70, 100]))
        .then(pl.col("passengers") * 0.2)  # Dip outliers
        .otherwise(pl.col("passengers"))
        .alias("passengers")
    ).drop("idx")

    print("Added 4 spikes and 3 dips")
    return (y_outlier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### OutlierThresholdHandler

    Uses **fixed** thresholds. Values outside are clipped or set to null.
    """)
    return


@app.cell
def _(OutlierThresholdHandler, y_outlier):
    clip_handler = OutlierThresholdHandler(low=100, high=500, strategy="clip")
    clip_handler.fit(y_outlier)
    y_clipped = clip_handler.transform(y_outlier)

    print("Clipped to [100, 500]:")
    print(f"  Min: {y_clipped['passengers'].min()}, Max: {y_clipped['passengers'].max()}")
    return clip_handler, y_clipped


@app.cell
def _(OutlierThresholdHandler, y_outlier):
    nan_handler = OutlierThresholdHandler(low=100, high=500, strategy="nan")
    nan_handler.fit(y_outlier)
    y_nanified = nan_handler.transform(y_outlier)

    print(f"Values replaced with null: {y_nanified['passengers'].null_count()}")
    return nan_handler, y_nanified


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### OutlierPercentileHandler

    Uses **learned** percentile thresholds from the data.
    """)
    return


@app.cell
def _(OutlierPercentileHandler, y_outlier):
    pct_handler = OutlierPercentileHandler(low=5, high=95, strategy="clip")
    pct_handler.fit(y_outlier)
    y_pct = pct_handler.transform(y_outlier)

    print("Clipped to [5th, 95th] percentile:")
    print(f"  Min: {y_pct['passengers'].min():.0f}, Max: {y_pct['passengers'].max():.0f}")
    return pct_handler, y_pct


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Chaining: Outlier Removal → Imputation

    A common pattern: first replace outliers with null, then impute.
    """)
    return


@app.cell
def _(OutlierPercentileHandler, SimpleTimeImputer, y_outlier):
    # Step 1: Replace outliers with null
    step1 = OutlierPercentileHandler(low=5, high=95, strategy="nan")
    step1.fit(y_outlier)
    y_step1 = step1.transform(y_outlier)

    # Step 2: Impute the nulls
    step2 = SimpleTimeImputer(method="linear")
    step2.fit(y_step1)
    y_clean = step2.transform(y_step1)

    print(f"After outlier removal: {y_step1['passengers'].null_count()} nulls")
    print(f"After imputation: {y_clean['passengers'].null_count()} nulls")
    return step1, step2, y_clean, y_step1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - `SimpleTimeImputer`: Temporal interpolation (linear, forward, backward, nearest)
    - `SeasonalImputer`: Fills with seasonal pattern averages
    - `OutlierThresholdHandler`: Fixed-threshold clipping or nullification
    - `OutlierPercentileHandler`: Data-driven percentile-based thresholds
    - Chain outlier→imputation for robust cleaning
    - All transformers are stateless (no `observation_horizon`)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Window transformers**: See `window_transformers.py` for feature engineering
    - **Resampling**: See `resampling.py` for frequency changes
    - **Stationarity**: See `stationarity/` for trend/seasonality removal
    """)
    return


if __name__ == "__main__":
    app.run()
