# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "How to Handle Outliers in a Forecasting Pipeline",
    "description": "Detect and clip outliers with OutlierThresholdHandler and OutlierPercentileHandler, then see how outliers affect conformal prediction intervals.",
    "category": "how-to",
    "section": "data-features",
    "companion": "/pages/how-to/handle-outliers/",
    "api_references": ["OutlierPercentileHandler", "OutlierThresholdHandler", "SplitConformalForecaster", "GammaResidual"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How to Handle Outliers in a Forecasting Pipeline

    This notebook shows how to detect and clip outliers using threshold
    and percentile handlers, then demonstrates their effect on conformal
    prediction intervals.
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import FeaturePipeline
    from yohou.datasets import fetch_sunspot
    from yohou.interval import SplitConformalForecaster
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer
    from yohou.preprocessing.outlier import OutlierPercentileHandler, OutlierThresholdHandler

    return (
        FeaturePipeline,
        LagTransformer,
        MeanAbsoluteError,
        OutlierPercentileHandler,
        OutlierThresholdHandler,
        PointReductionForecaster,
        Ridge,
        SplitConformalForecaster,
        fetch_sunspot,
        pl,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load Data and Inject Synthetic Outliers

    Load the sunspot dataset and inject a few synthetic outlier spikes to
    demonstrate the handlers.
    """)


@app.cell
def _(fetch_sunspot, pl, plot_time_series):
    data = fetch_sunspot().frame

    # Inject synthetic outliers at specific positions
    outlier_indices = [50, 120, 200]
    outlier_mask = pl.Series([i in outlier_indices for i in range(len(data))])
    data_with_outliers = data.with_columns(
        pl.when(outlier_mask)
        .then(pl.col("sunspot_number") * 5)
        .otherwise(pl.col("sunspot_number"))
        .alias("sunspot_number")
    )

    plot_time_series(data_with_outliers, title="Sunspots with Synthetic Outliers")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Clip with Fixed Thresholds

    Use [`OutlierThresholdHandler`](/pages/api/generated/yohou.preprocessing.outlier.OutlierThresholdHandler/) when you know plausible value bounds from
    domain knowledge.
    """)


@app.cell
def _(OutlierThresholdHandler, data_with_outliers, plot_time_series):
    handler_fixed = OutlierThresholdHandler(low=0.0, high=200.0)
    handler_fixed.fit(data_with_outliers)
    clipped_fixed = handler_fixed.transform(data_with_outliers)

    plot_time_series(clipped_fixed, title="After Fixed Threshold Clipping (0, 200)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Clip with Learned Percentiles

    Use [`OutlierPercentileHandler`](/pages/api/generated/yohou.preprocessing.outlier.OutlierPercentileHandler/) when bounds should be learned from the
    data distribution.
    """)


@app.cell
def _(OutlierPercentileHandler, data_with_outliers, plot_time_series):
    handler_pct = OutlierPercentileHandler(low=1, high=99)
    handler_pct.fit(data_with_outliers)
    clipped_pct = handler_pct.transform(data_with_outliers)

    plot_time_series(clipped_pct, title="After Percentile Clipping (1st to 99th)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Place Handlers in a Pipeline

    Both handlers are stateful transformers. Place them before lag features
    and stationarity transforms.
    """)


@app.cell
def _(
    FeaturePipeline,
    LagTransformer,
    MeanAbsoluteError,
    OutlierPercentileHandler,
    PointReductionForecaster,
    Ridge,
    data_with_outliers,
):
    from yohou.model_selection import train_test_split

    y_train, y_test = train_test_split(data_with_outliers, test_size=24)

    pipeline = FeaturePipeline([
        ("outlier_clip", OutlierPercentileHandler(low=1, high=99)),
        ("lags", LagTransformer(lag=[1, 6, 12])),
    ])

    forecaster = PointReductionForecaster(
        estimator=Ridge(),
        actual_transformer=pipeline,
    )
    forecaster.fit(y_train, forecasting_horizon=24)
    y_pred = forecaster.predict(forecasting_horizon=24)

    scorer = MeanAbsoluteError()
    scorer.fit(y_train)
    score = scorer.score(y_test, y_pred)
    score


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Verify: Outlier Effect on Conformal Intervals

    Outliers in the calibration set inflate interval width. Compare
    intervals with and without outlier clipping.
    """)


@app.cell
def _(
    FeaturePipeline,
    LagTransformer,
    OutlierPercentileHandler,
    PointReductionForecaster,
    Ridge,
    SplitConformalForecaster,
    data_with_outliers,
    pl,
):
    n2 = len(data_with_outliers)
    y_train2 = data_with_outliers.head(n2 - 24)

    # Without clipping
    forecaster_raw = SplitConformalForecaster(
        point_forecaster=PointReductionForecaster(
            estimator=Ridge(),
            actual_transformer=LagTransformer(lag=[1, 6, 12]),
        ),
    )
    forecaster_raw.fit(y_train2, forecasting_horizon=24)
    pred_raw = forecaster_raw.predict_interval(forecasting_horizon=24, coverage_rates=[0.9])

    # With clipping
    forecaster_clean = SplitConformalForecaster(
        point_forecaster=PointReductionForecaster(
            estimator=Ridge(),
            actual_transformer=FeaturePipeline([
                ("outlier_clip", OutlierPercentileHandler(low=1, high=99)),
                ("lags", LagTransformer(lag=[1, 6, 12])),
            ]),
        ),
    )
    forecaster_clean.fit(y_train2, forecasting_horizon=24)
    pred_clean = forecaster_clean.predict_interval(forecasting_horizon=24, coverage_rates=[0.9])

    # Compare interval widths
    raw_width = (
        pred_raw.select(pl.col("^.*upper.*$") - pl.col("^.*lower.*$")).mean().to_series()[0]
    )
    clean_width = (
        pred_clean.select(pl.col("^.*upper.*$") - pl.col("^.*lower.*$")).mean().to_series()[0]
    )
    print(f"Mean interval width without clipping: {raw_width:.1f}")
    print(f"Mean interval width with clipping:    {clean_width:.1f}")



@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [Handle Outliers](/pages/how-to/handle-outliers/) for the full guide
        - [Clean and Resample Time Series](/pages/how-to/clean-and-resample/) for related techniques
        """
    )
    return

if __name__ == "__main__":
    app.run()
