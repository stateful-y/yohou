# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "How to Handle Long Series",
    "description": "Limit history with observation_horizon, weight recent errors with exponential decay, and downsample high-frequency data.",
    "category": "how-to",
    "section": "data-features",
    "companion": "/pages/how-to/handle-long-series/",
    "api_references": ["LagTransformer", "Downsampler", "exponential_decay_weight"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How to Handle Long Series

    When a series spans decades, patterns from the distant past may be
    irrelevant. This notebook shows how to limit history, weight recent
    errors more heavily, and downsample high frequency data.
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_sunspot
    from yohou.plotting import plot_time_series

    return Ridge, fetch_sunspot, pl, plot_time_series


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load a Long Series

    The sunspot dataset spans centuries, making it a good example of data
    where distant history may hurt more than help.
    """)


@app.cell
def _(fetch_sunspot, plot_time_series):
    data = fetch_sunspot().frame
    print(f"Series length: {len(data)} observations")
    plot_time_series(data, title="Full Sunspot Series")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Limit History with observation_horizon

    Set `observation_horizon` on stateful transformers to keep only
    recent timesteps in their state. Older observations are dropped as
    new ones arrive.
    """)


@app.cell
def _(Ridge, data):
    from yohou.compose import FeaturePipeline
    from yohou.metrics import MeanAbsoluteError
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing.lags import LagTransformer

    from yohou.model_selection import train_test_split

    y_train, y_test = train_test_split(data, test_size=24)

    # Full history
    pipe_full = FeaturePipeline([("lags", LagTransformer(lag=[1, 6, 12]))])
    fc_full = PointReductionForecaster(estimator=Ridge(), feature_transformer=pipe_full)
    fc_full.fit(y_train, forecasting_horizon=24)
    pred_full = fc_full.predict(forecasting_horizon=24)

    # Limited history: keep only the last 120 observations
    pipe_limited = FeaturePipeline(
        [("lags", LagTransformer(lag=[1, 6, 12]))]
    )
    fc_limited = PointReductionForecaster(estimator=Ridge(), feature_transformer=pipe_limited)
    fc_limited.fit(y_train.tail(120), forecasting_horizon=24)
    pred_limited = fc_limited.predict(forecasting_horizon=24)

    scorer = MeanAbsoluteError()
    print(f"MAE (full history):    {scorer.score(y_test, pred_full):.2f}")
    print(f"MAE (limited history): {scorer.score(y_test, pred_limited):.2f}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Weight Recent Errors with Exponential Decay

    Instead of discarding old data entirely, weight recent observations
    more heavily. [`exponential_decay_weight`](/pages/api/generated/yohou.utils.weighting.exponential_decay_weight/) halves the weight every
    `half_life` steps.
    """)


@app.cell
def _():
    from yohou.utils.weighting import exponential_decay_weight, linear_decay_weight

    # Exponential decay: weight halves every 120 steps
    exp_weight = exponential_decay_weight(half_life=120)

    # Linear decay: weight drops to 0 at max_steps ago
    lin_weight = linear_decay_weight(max_steps=240)

    print(f"Exponential weight function: {exp_weight}")
    print(f"Linear weight function:      {lin_weight}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Downsample High Frequency Data

    When data arrives at a higher frequency than the forecast requirement,
    use [`Downsampler`](/pages/api/generated/yohou.preprocessing.resampling.Downsampler/) to aggregate. Place it at the start of the pipeline,
    before stateful transformers.
    """)


@app.cell
def _():
    from yohou.preprocessing.resampling import Downsampler

    # Aggregate to quarterly using mean
    downsampler = Downsampler(interval="91d", aggregation="mean")
    print(f"Downsampler config: interval={downsampler.interval}, aggregation={downsampler.aggregation}")



@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [Handle Long Series](/pages/how-to/handle-long-series/) for the full guide
        - [Clean and Resample Time Series](/pages/how-to/clean-and-resample/) for related techniques
        """
    )
    return

if __name__ == "__main__":
    app.run()
