# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "How to Handle Short Series",
    "description": "Use Fourier seasonality, simple train/test splits, and panel pooling when individual series are too short for standard approaches.",
    "category": "how-to",
    "section": "data-features",
    "companion": "/pages/how-to/handle-short-series/",
    "api_references": ["FourierSeasonalityForecaster", "ExpandingWindowSplitter", "PointReductionForecaster"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How to Handle Short Series

    Short series lack enough data for reliable seasonal estimation,
    multi-split cross validation, or scaled metrics. This notebook
    demonstrates fallback strategies for each problem.
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_tourism_monthly
    from yohou.plotting import plot_time_series

    return Ridge, fetch_tourism_monthly, pl, plot_time_series


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Create a Short Series

    Extract a single short series from the tourism dataset to simulate
    a scenario with limited history.
    """)


@app.cell
def _(fetch_tourism_monthly, pl, plot_time_series):
    full = fetch_tourism_monthly().frame
    # Take only the first 30 months of a single series
    short_series = (
        full.select("time", "T1__tourists")
        .drop_nulls()
        .rename({"T1__tourists": "tourists"})
        .head(30)
    )
    print(f"Series length: {len(short_series)} observations")
    plot_time_series(short_series, title="Short Tourism Series (30 months)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Use Fourier Seasonality Instead of Pattern Decomposition

    With fewer than 2 to 3 complete seasonal cycles, pattern based decomposition
    is unreliable. [`FourierSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.FourierSeasonalityForecaster/) fits smooth harmonic
    functions that converge with less data. Start with a low `harmonics`
    order to avoid overfitting.
    """)


@app.cell
def _(short_series):
    from yohou.stationarity.seasonality import FourierSeasonalityForecaster

    forecaster = FourierSeasonalityForecaster(seasonality=12, harmonics=[1, 2])
    forecaster.fit(short_series, forecasting_horizon=6)
    pred_fourier = forecaster.predict(forecasting_horizon=6)
    pred_fourier


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Use a Simple Train/Test Split Instead of Cross Validation

    When the series is too short for multi-split CV, use
    [`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.split.ExpandingWindowSplitter/) with `n_splits=2` for minimal splits.
    """)


@app.cell
def _(short_series):
    from yohou.model_selection import ExpandingWindowSplitter

    splitter = ExpandingWindowSplitter(n_splits=2, test_size=6)
    splits = list(splitter.split(short_series))
    print(f"Number of splits: {len(splits)}")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"Split {i}: Train size={len(train_idx)}, Test size={len(test_idx)}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Pool Short Series with panel_strategy="global"

    When individual series are short but you have many of them, pool
    information across groups. The global panel strategy trains a
    single regressor on all series simultaneously.
    """)


@app.cell
def _(Ridge, fetch_tourism_monthly):
    from yohou.compose import FeaturePipeline
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    # Load a subset of panel series (drop rows with nulls)
    panel = (
        fetch_tourism_monthly()
        .frame.select(
            "time",
            "T1__tourists",
            "T2__tourists",
            "T3__tourists",
            "T4__tourists",
            "T5__tourists",
        )
        .drop_nulls()
    )

    forecaster_global = PointReductionForecaster(
        estimator=Ridge(),
        feature_transformer=FeaturePipeline([("lags", LagTransformer(lag=[1, 2, 3]))]),
        panel_strategy="global",
    )
    forecaster_global.fit(panel, forecasting_horizon=6)
    pred_panel = forecaster_global.predict(forecasting_horizon=6)
    pred_panel


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Choose Scale Independent Metrics

    MASE and RMSSE require at least one full seasonal cycle to compute
    the naive baseline. For short series, use MAPE or WAPE instead.
    """)


@app.cell
def _():
    from yohou.metrics import MeanAbsolutePercentageError

    scorer = MeanAbsolutePercentageError()
    print(f"Scorer: {scorer}")



@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [Handle Short Series](/pages/how-to/handle-short-series/) for the full guide
        - [Handle Long Series](/pages/how-to/handle-long-series/) for related techniques
        """
    )
    return

if __name__ == "__main__":
    app.run()
