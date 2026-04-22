# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "plotly",
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Time-Derived Features",
    "description": "Create calendar, holiday, Fourier, and time index features with dedicated transformers and compose them with FeatureUnion.",
    "category": "tutorial",
    "companion": "/pages/explanation/preprocessing/#time-features",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Time-Derived Features

    In this notebook we will use four stateless transformers that extract
    features from the `"time"` column:
    [`CalendarFeatureTransformer`](/pages/api/generated/yohou.preprocessing.calendar.CalendarFeatureTransformer/),
    [`HolidayFeatureTransformer`](/pages/api/generated/yohou.preprocessing.calendar.HolidayFeatureTransformer/),
    [`FourierFeatureTransformer`](/pages/api/generated/yohou.preprocessing.time_features.FourierFeatureTransformer/), and
    [`TimeIndexTransformer`](/pages/api/generated/yohou.preprocessing.time_features.TimeIndexTransformer/),
    then compose them with [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/).

    ## Prerequisites

    Basic understanding of time series feature engineering and Fourier series.
    """)
    return


@app.cell(hide_code=True)
def _():
    from datetime import date

    import polars as pl

    from yohou.compose import FeatureUnion
    from yohou.datasets import fetch_sunspot
    from yohou.plotting import plot_time_series
    from yohou.preprocessing import (
        CalendarFeatureTransformer,
        FourierFeatureTransformer,
        HolidayFeatureTransformer,
        TimeIndexTransformer,
    )

    return (
        CalendarFeatureTransformer,
        FeatureUnion,
        FourierFeatureTransformer,
        HolidayFeatureTransformer,
        TimeIndexTransformer,
        date,
        fetch_sunspot,
        pl,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We load the monthly sunspot series for the calendar, Fourier, and trend
    sections. Later we create a short daily series for holiday features.
    """)
    return


@app.cell
def _(fetch_sunspot, plot_time_series):
    y = fetch_sunspot().frame.select("time", "sunspot_number").drop_nulls()
    plot_time_series(y, title="Monthly Sunspot Number")
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. CalendarFeatureTransformer: Auto-detect

    With `features=None` (the default), the transformer detects the data
    interval and extracts all applicable features. For monthly data this
    includes `year`, `month`, `quarter`, and the boolean boundary flags,
    but excludes sub-daily features like `hour` and `minute`.
    """)
    return


@app.cell
def _(CalendarFeatureTransformer, y):
    cal_tf = CalendarFeatureTransformer()
    cal_tf.fit(y)
    y_cal = cal_tf.transform(y)
    y_cal.head(5)
    return cal_tf, y_cal


@app.cell
def _(plot_time_series, y_cal):
    plot_time_series(
        y_cal.head(1000),
        title="Extracted Calendar Features",
    )
    return


@app.cell
def _(cal_tf):
    cal_tf.applicable_features_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Select Specific Features

    Pass an explicit list to extract only the features you need.
    """)
    return


@app.cell
def _(CalendarFeatureTransformer, y):
    cal_specific = CalendarFeatureTransformer(features=["month", "quarter", "is_year_start"])
    cal_specific.fit(y)
    y_specific = cal_specific.transform(y)
    y_specific.head(5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. HolidayFeatureTransformer

    [`HolidayFeatureTransformer`](/pages/api/generated/yohou.preprocessing.calendar.HolidayFeatureTransformer/) takes a polars DataFrame of holiday dates
    and produces a binary indicator column. The holidays DataFrame must
    have a `"date"` column of `Date` or `Datetime` type.

    We create a short daily series for this section since holidays are most
    meaningful at daily granularity.
    """)
    return


@app.cell
def _(HolidayFeatureTransformer, date, pl):
    holidays = pl.DataFrame({
        "date": [
            date(2020, 1, 1),
            date(2020, 7, 4),
            date(2020, 12, 25),
        ]
    })

    X_daily = pl.DataFrame({
        "time": pl.datetime_range(
            start=date(2020, 1, 1),
            end=date(2020, 1, 10),
            interval="1d",
            eager=True,
        ),
        "value": list(range(10)),
    })

    hol_tf = HolidayFeatureTransformer(holidays=holidays)
    hol_tf.fit(X_daily)
    X_hol = hol_tf.transform(X_daily)
    X_hol
    return X_daily, holidays


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Holiday Proximity Features

    With `days_to_next=True` and `days_since_last=True`, the transformer
    adds columns that measure how far each timestamp is from the nearest
    holiday. Values are `null` when no next or previous holiday exists
    in the provided list.
    """)
    return


@app.cell
def _(HolidayFeatureTransformer, X_daily, holidays):
    hol_prox = HolidayFeatureTransformer(holidays=holidays, days_to_next=True, days_since_last=True)
    hol_prox.fit(X_daily)
    X_prox = hol_prox.transform(X_daily)
    X_prox
    return (X_prox,)


@app.cell
def _(X_prox, plot_time_series):
    plot_time_series(
        X_prox,
        columns=["holiday_indicator", "holiday_days_to_next", "holiday_days_since_last"],
        title="Holiday Indicator and Proximity",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. FourierFeatureTransformer: Single Seasonality

    Specify the seasonal period as the number of time steps. For monthly data,
    a yearly cycle is `seasonality=12`. The `harmonics` parameter controls
    which frequency multiples to include. `[1]` captures the fundamental
    frequency, `[1, 2, 3]` adds higher harmonics for sharper patterns.
    """)
    return


@app.cell
def _(FourierFeatureTransformer, y):
    fourier_yearly = FourierFeatureTransformer(seasonality=12.0, harmonics=[1, 2])
    fourier_yearly.fit(y)
    y_fourier = fourier_yearly.transform(y)
    y_fourier.head(5)
    return (y_fourier,)


@app.cell
def _(plot_time_series, y_fourier):
    plot_time_series(
        y_fourier.head(1000),
        title="Yearly Fourier Harmonics (period=12)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice the output column naming: `fourier_{seasonality}_sin_{k}` and
    `fourier_{seasonality}_cos_{k}` for each harmonic `k`. Values are
    always in the range [-1, 1].
    """)
    return


@app.cell
def _(pl, y_fourier):
    y_fourier.select(pl.col("^fourier_.*$")).describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Non-integer Seasonality

    The `seasonality` parameter accepts floats. For instance, `365.25`
    captures a yearly cycle on daily data accounting for leap years.
    The sunspot cycle is approximately 132 months (~11 years).
    """)
    return


@app.cell
def _(FourierFeatureTransformer, pl, y):
    fourier_solar = FourierFeatureTransformer(seasonality=132.0, harmonics=[1, 2, 3])
    fourier_solar.fit(y)
    y_solar = fourier_solar.transform(y)
    y_solar.select("time", pl.col("^fourier_.*$")).head(5)
    return (y_solar,)


@app.cell
def _(plot_time_series, y_solar):
    plot_time_series(
        y_solar.head(1000),
        title="Solar Cycle Fourier Harmonics (period=132)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Multiple Seasonalities with FeatureUnion

    Use [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/) to combine Fourier features at different
    periods. Each `FourierFeatureTransformer` handles one seasonal period,
    and the union concatenates their outputs.
    """)
    return


@app.cell
def _(FeatureUnion, FourierFeatureTransformer, y):
    fourier_union = FeatureUnion(
        transformer_list=[
            ("yearly", FourierFeatureTransformer(seasonality=12.0, harmonics=[1, 2])),
            ("solar", FourierFeatureTransformer(seasonality=132.0, harmonics=[1])),
        ],
    )
    fourier_union.fit(y)
    y_multi = fourier_union.transform(y)
    y_multi.head(5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. TimeIndexTransformer: Linear Index

    [`TimeIndexTransformer`](/pages/api/generated/yohou.preprocessing.time_features.TimeIndexTransformer/) converts the time column into a sequential integer
    index starting at 0. This is useful as a trend feature for
    reduction forecasters.
    """)
    return


@app.cell
def _(TimeIndexTransformer, y):
    idx_tf = TimeIndexTransformer()
    idx_tf.fit(y)
    y_idx = idx_tf.transform(y)
    y_idx.select("time", "time_index").head(5)
    return (y_idx,)


@app.cell
def _(plot_time_series, y_idx):
    plot_time_series(
        y_idx.head(1000),
        title="Time Index",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Normalized and Polynomial Index

    With `normalize=True`, the index is scaled to [0, 1] based on the fit
    data length. The `degree` parameter adds polynomial terms for
    non-linear trend modeling.
    """)
    return


@app.cell
def _(TimeIndexTransformer, y):
    poly_tf = TimeIndexTransformer(normalize=True, degree=3)
    poly_tf.fit(y)
    y_poly = poly_tf.transform(y)
    y_poly.select("time", "time_index", "time_index_2", "time_index_3").head(5)
    return (y_poly,)


@app.cell
def _(plot_time_series, y_poly):
    plot_time_series(
        y_poly.head(1000),
        title="Normalized Polynomial Time Index (degree=3)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that `time_index` is `Float64` when normalized (instead of `Int64`),
    and polynomial columns are always `Float64`.
    """)
    return


@app.cell
def _(y_poly):
    y_poly.select("time_index", "time_index_2", "time_index_3").describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Combining All with FeatureUnion

    Combine calendar, Fourier, and time index transformers into a single
    [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/) pipeline.
    """)
    return


@app.cell
def _(
    CalendarFeatureTransformer,
    FeatureUnion,
    FourierFeatureTransformer,
    TimeIndexTransformer,
    y,
):
    full_union = FeatureUnion(
        transformer_list=[
            ("calendar", CalendarFeatureTransformer(features=["month", "quarter"])),
            ("fourier_12", FourierFeatureTransformer(seasonality=12.0, harmonics=[1, 2])),
            ("fourier_132", FourierFeatureTransformer(seasonality=132.0, harmonics=[1])),
            ("trend", TimeIndexTransformer(normalize=True, degree=2)),
        ],
    )
    full_union.fit(y)
    y_full = full_union.transform(y)
    y_full.head(5)
    return
