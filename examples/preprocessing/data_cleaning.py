# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.19.9"
__gallery__ = {
    "title": "Data Cleaning",
    "description": "End-to-end data cleaning pipeline combining SimpleTimeImputer and SeasonalImputer for missing values with OutlierThresholdHandler for anomaly clipping.",
    "category": "how-to",
    "companion": "/pages/explanation/preprocessing/",
    "section": "data-features",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Cleaning: Imputation and Outlier Handling

    This notebook shows how to clean time series data by filling
    missing values with temporal imputers and handling outliers
    with threshold-based handlers.

    **Prerequisites:** This is a standalone preprocessing guide.
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl

    from yohou.compose import FeaturePipeline
    from yohou.datasets import fetch_tourism_monthly
    from yohou.plotting import plot_missing_data, plot_time_series
    from yohou.preprocessing import (
        OutlierPercentileHandler,
        OutlierThresholdHandler,
        SeasonalImputer,
        SimpleTimeImputer,
    )

    return (
        FeaturePipeline,
        OutlierPercentileHandler,
        OutlierThresholdHandler,
        SeasonalImputer,
        SimpleTimeImputer,
        fetch_tourism_monthly,
        pl,
        plot_missing_data,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Create Data with Missing Values

    Load a single monthly tourism series and randomly remove 15
    observations to simulate sensor dropouts or data pipeline failures.
    """)


@app.cell
def _(fetch_tourism_monthly, pl):
    y = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})

    # Introduce missing values at random positions
    import random

    random.seed(42)
    null_indices = sorted(random.sample(range(len(y)), 15))

    y_missing = (
        y
        .with_row_index("idx")
        .with_columns(
            pl.when(pl.col("idx").is_in(null_indices)).then(None).otherwise(pl.col("tourists")).alias("tourists")
        )
        .drop("idx")
    )

    null_count = y_missing["tourists"].null_count()
    print(f"Introduced {null_count} missing values")
    return null_count, null_indices, random, y, y_missing


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use [`plot_missing_data`](/pages/api/generated/yohou.plotting.exploration.plot_missing_data/) to inspect which time steps have null values.
    """)


@app.cell
def _(plot_missing_data, y_missing):
    plot_missing_data(y_missing, title="Missing Data Pattern")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Fill Gaps with SimpleTimeImputer

    Use [`SimpleTimeImputer`](/pages/api/generated/yohou.preprocessing.imputation.SimpleTimeImputer/) to fill missing values with time-aware
    interpolation. Supported methods: `"linear"`, `"forward"`, `"backward"`,
    and `"nearest"`.
    """)


@app.cell
def _(SimpleTimeImputer, y_missing):
    for method in ["linear", "forward", "backward", "nearest"]:
        imputer = SimpleTimeImputer(method=method)
        imputer.fit(y_missing)
        filled = imputer.transform(y_missing)
        remaining_nulls = filled["tourists"].null_count()
        print(f"method={method:>10s}  remaining nulls: {remaining_nulls}")
    return (filled, imputer, method, remaining_nulls)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Verify that linear imputation filled the gaps smoothly.
    """)


@app.cell
def _(SimpleTimeImputer, plot_time_series, y_missing):
    linear_imputer = SimpleTimeImputer(method="linear")
    linear_imputer.fit(y_missing)
    y_filled = linear_imputer.transform(y_missing)

    _combined = y_filled.rename({"tourists": "imputed"}).join(
        y_missing.rename({"tourists": "with gaps"}),
        on="time",
    )
    plot_time_series(_combined, title="After Linear Imputation")
    return linear_imputer, y_filled


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Fill Gaps with SeasonalImputer

    Use [`SeasonalImputer`](/pages/api/generated/yohou.preprocessing.imputation.SeasonalImputer/) to replace each missing value with the average
    at the same seasonal position. Set `period=12` for monthly data so a
    missing January is filled using Januaries from other years.
    """)


@app.cell
def _(SeasonalImputer, plot_time_series, y_missing):
    seasonal_imp = SeasonalImputer(period=12)
    seasonal_imp.fit(y_missing)
    y_seasonal = seasonal_imp.transform(y_missing)

    _combined = y_seasonal.rename({"tourists": "imputed"}).join(
        y_missing.rename({"tourists": "with gaps"}),
        on="time",
    )
    plot_time_series(_combined, title="After Seasonal Imputation (period=12)")
    return seasonal_imp, y_seasonal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Handle Outliers

    Use [`OutlierThresholdHandler`](/pages/api/generated/yohou.preprocessing.outlier.OutlierThresholdHandler/) with fixed bounds or
    [`OutlierPercentileHandler`](/pages/api/generated/yohou.preprocessing.outlier.OutlierPercentileHandler/) with data-driven percentile bounds.
    Both support `"clip"` (cap at threshold) and `"nan"` (replace with null)
    strategies.

    Inject synthetic spikes and dips to test both handlers.
    """)


@app.cell
def _(pl, y):
    # Add outliers
    y_outlier = (
        y
        .with_row_index("idx")
        .with_columns(
            pl
            .when(pl.col("idx").is_in([20, 50, 80, 110]))
            .then(pl.col("tourists") * 3)  # Spike outliers
            .when(pl.col("idx").is_in([30, 70, 100]))
            .then(pl.col("tourists") * 0.2)  # Dip outliers
            .otherwise(pl.col("tourists"))
            .alias("tourists")
        )
        .drop("idx")
    )

    print("Added 4 spikes and 3 dips")
    return (y_outlier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### OutlierThresholdHandler

    [`OutlierThresholdHandler`](/pages/api/generated/yohou.preprocessing.outlier.OutlierThresholdHandler/) clips or nullifies values that fall outside
    user-specified `low` and `high` bounds. Use this when you know the
    valid range of your data in advance.
    """)


@app.cell
def _(OutlierThresholdHandler, plot_time_series, y_outlier):
    clip_handler = OutlierThresholdHandler(low=100, high=500, strategy="clip")
    clip_handler.fit(y_outlier)
    y_clipped = clip_handler.transform(y_outlier)

    _combined = y_clipped.rename({"tourists": "clipped"}).join(
        y_outlier.rename({"tourists": "with outliers"}),
        on="time",
    )
    plot_time_series(_combined, title="OutlierThresholdHandler (clip to [100, 500])")
    return clip_handler, y_clipped


@app.cell
def _(OutlierThresholdHandler, plot_time_series, y_outlier):
    nan_handler = OutlierThresholdHandler(low=100, high=500, strategy="nan")
    nan_handler.fit(y_outlier)
    y_nanified = nan_handler.transform(y_outlier)

    _combined = y_nanified.rename({"tourists": "nullified"}).join(
        y_outlier.rename({"tourists": "with outliers"}),
        on="time",
    )
    plot_time_series(_combined, title="OutlierThresholdHandler (nan outside [100, 500])")
    return nan_handler, y_nanified


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### OutlierPercentileHandler

    [`OutlierPercentileHandler`](/pages/api/generated/yohou.preprocessing.outlier.OutlierPercentileHandler/) learns thresholds from the data at fit time.
    The `low` and `high` parameters specify the percentile values (e.g. 5 and
    95) used as bounds. This is useful when you do not have domain knowledge
    about the valid range and want the data itself to define what counts as
    extreme.
    """)


@app.cell
def _(OutlierPercentileHandler, plot_time_series, y_outlier):
    pct_handler = OutlierPercentileHandler(low=5, high=95, strategy="clip")
    pct_handler.fit(y_outlier)
    y_pct = pct_handler.transform(y_outlier)

    _combined = y_pct.rename({"tourists": "clipped"}).join(
        y_outlier.rename({"tourists": "with outliers"}),
        on="time",
    )
    plot_time_series(_combined, title="OutlierPercentileHandler (clip to [5th, 95th])")
    return pct_handler, y_pct


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Chain Outlier Removal and Imputation

    Use [`FeaturePipeline`](/pages/api/generated/yohou.compose.feature_pipeline.FeaturePipeline/) to compose an outlier handler and imputer
    into a single reusable pipeline.
    """)


@app.cell
def _(FeaturePipeline, OutlierPercentileHandler, SimpleTimeImputer, plot_time_series, y_outlier):
    pipeline = FeaturePipeline(
        steps=[
            ("outlier", OutlierPercentileHandler(low=5, high=95, strategy="nan")),
            ("impute", SimpleTimeImputer(method="linear")),
        ]
    )
    y_clean = pipeline.fit_transform(y_outlier)

    _combined = y_clean.rename({"tourists": "cleaned"}).join(
        y_outlier.rename({"tourists": "with outliers"}),
        on="time",
    )
    plot_time_series(_combined, title="FeaturePipeline: Outlier Removal + Linear Imputation")
    return pipeline, y_clean


if __name__ == "__main__":
    app.run()
