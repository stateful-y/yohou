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
    "api_references": [
        "OutlierPercentileHandler",
        "OutlierThresholdHandler",
        "CalendarSegmenter",
        "SplitConformalForecaster",
        "GammaResidual",
    ],
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
    ## 4. Handle Outliers by Time of Day

    When the normal range depends on the hour, one limit for the whole day
    misses night-time spikes. Below, hourly load runs near 60 at night and
    near 300 during the day. A reading of 150 is more than double the normal
    night level, yet a single 99th percentile sits near the daytime peak and
    leaves it alone. With a [`CalendarSegmenter`](/pages/api/generated/yohou.segmenting.CalendarSegmenter/)
    and `by_segment=True`, the handler learns one limit per hour and clips the
    03:00 reading while keeping the same value at 12:00.
    """)


@app.cell
def _(OutlierPercentileHandler, pl):
    from yohou.segmenting import CalendarSegmenter

    # Sixty days of hourly load: nights near 60, days near 300, small deterministic jitter
    _times = pl.datetime_range(pl.datetime(2024, 1, 1), pl.datetime(2024, 2, 29, 23), "1h", eager=True)
    _base = pl.Series([300.0 if 8 <= t.hour <= 18 else 60.0 for t in _times])
    _jitter = pl.Series([float((i * 7) % 11) for i in range(len(_times))])
    hourly_load = pl.DataFrame({"time": _times, "load": _base + _jitter})

    # Two new readings of 150: one at 03:00, one at 12:00
    _probe = pl.DataFrame({"time": [_times[-21], _times[-12]], "load": [150.0, 150.0]})

    _single_limit = OutlierPercentileHandler(high=99).fit(hourly_load)
    _limit_per_hour = OutlierPercentileHandler(
        high=99,
        segmenter=CalendarSegmenter("hour"),
        by_segment=True,
    ).fit(hourly_load)

    hourly_comparison = _probe.with_columns(
        _single_limit.transform(_probe)["load"].alias("single_limit"),
        _limit_per_hour.transform(_probe)["load"].alias("limit_per_hour"),
    )
    hourly_comparison


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Place Handlers in a Pipeline

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
    ## 6. Verify: Outlier Effect on Conformal Intervals

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
