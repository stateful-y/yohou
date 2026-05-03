# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.19.11"
__gallery__ = {
    "title": "Feature Union",
    "description": "Combine lag features, rolling statistics, EMA, and scaling in parallel with FeatureUnion and automatic observation horizon resolution.",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Parallel Feature Engineering with FeatureUnion

    [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/) applies multiple transformers in parallel to the
    same input and concatenates the results column-wise. The resulting
    `observation_horizon` is the **maximum** across all transformers.

    ## What You'll Learn

    - Combine lag features, rolling statistics, and scaling in parallel
    - `observation_horizon` = max of child transformers
    - `verbose_feature_names_out` for prefixed column names
    - Using [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/) as `feature_transformer` in a forecaster
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge
    from yohou.model_selection import train_test_split

    from yohou.compose import FeatureUnion
    from yohou.datasets import fetch_sunspot
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import (
        ExponentialMovingAverage,
        LagTransformer,
        RollingStatisticsTransformer,
        StandardScaler,
    )

    return (
        ExponentialMovingAverage,
        FeatureUnion,
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        RollingStatisticsTransformer,
        SeasonalNaive,
        StandardScaler,
        fetch_sunspot,
        pl,
        plot_forecast,
        plot_time_series,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load Data

    We load the Sunspot Numbers dataset and resample to monthly frequency.
    The data is split 85/15 into training and test sets.
    """)


@app.cell
def _(fetch_sunspot, mo, pl, train_test_split):
    _raw = fetch_sunspot().frame
    sunspots = _raw.group_by_dynamic("time", every="1mo").agg(pl.col("sunspot_number").mean())
    y_train, y_test = train_test_split(sunspots, test_size=0.15)
    horizon = len(y_test)
    mo.md(f"**Sunspots**: {len(sunspots)} months, **Train**: {len(y_train)}, **Test**: {len(y_test)}")
    return horizon, sunspots, y_test, y_train


@app.cell
def _(plot_time_series, sunspots):
    plot_time_series(sunspots, title="Monthly Sunspot Numbers")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Basic FeatureUnion

    Combine lag features with rolling statistics.
    """)


@app.cell
def _(FeatureUnion, LagTransformer, RollingStatisticsTransformer, mo, y_train):
    union_basic = FeatureUnion(
        transformer_list=[
            ("lags", LagTransformer(lag=[1, 6, 12])),
            ("rolling", RollingStatisticsTransformer(window_size=12, statistics=["mean", "std"])),
        ],
    )
    union_basic.fit(y_train)
    basic_features = union_basic.transform(y_train)

    mo.md(
        f"**Output columns**: {basic_features.columns}\n\n"
        f"**Output shape**: {basic_features.shape}\n\n"
        f"**observation_horizon**: {union_basic.observation_horizon} "
        f"(max of lag=12 and window=12)"
    )
    return basic_features, union_basic


@app.cell
def _(basic_features, plot_time_series):
    plot_time_series(basic_features, title="Basic FeatureUnion: Lags + Rolling Stats")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Verbose Feature Names

    When `verbose_feature_names_out=True` (default), each output column
    is prefixed with the transformer name.
    """)


@app.cell
def _(FeatureUnion, LagTransformer, RollingStatisticsTransformer, mo, y_train):
    _union_verbose = FeatureUnion(
        transformer_list=[
            ("lags", LagTransformer(lag=[1, 6])),
            ("rolling", RollingStatisticsTransformer(window_size=6, statistics="mean")),
        ],
        verbose_feature_names_out=True,
    )
    _union_verbose.fit(y_train)
    _out_verbose = _union_verbose.transform(y_train)

    _union_no_verbose = FeatureUnion(
        transformer_list=[
            ("lags", LagTransformer(lag=[1, 6])),
            ("rolling", RollingStatisticsTransformer(window_size=6, statistics="mean")),
        ],
        verbose_feature_names_out=False,
    )
    _union_no_verbose.fit(y_train)
    _out_plain = _union_no_verbose.transform(y_train)

    mo.md(
        f"**verbose=True columns**: {_out_verbose.columns}\n\n"
        f"**verbose=False columns**: {_out_plain.columns}\n\n"
        "Verbose names help disambiguate when multiple transformers produce "
        "similarly-named features."
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Three-Way Union

    Combine lags, rolling statistics, and exponential moving average.
    """)


@app.cell
def _(
    ExponentialMovingAverage,
    FeatureUnion,
    LagTransformer,
    RollingStatisticsTransformer,
    mo,
    y_train,
):
    union_three = FeatureUnion(
        transformer_list=[
            ("lags", LagTransformer(lag=[1, 6, 12])),
            ("rolling", RollingStatisticsTransformer(window_size=12, statistics=["mean", "std"])),
            ("ema", ExponentialMovingAverage(alpha=0.3)),
        ],
    )
    union_three.fit(y_train)
    three_features = union_three.transform(y_train)

    mo.md(
        f"**Three-way union output**: {three_features.shape}\n\n"
        f"**Columns**: {three_features.columns}\n\n"
        f"**observation_horizon**: {union_three.observation_horizon}"
    )
    return three_features, union_three


@app.cell
def _(plot_time_series, three_features):
    plot_time_series(three_features, title="Three-Way Union: Lags + Rolling + EMA")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. FeatureUnion in a Forecaster

    Use [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/) as the `feature_transformer` to produce rich
    feature sets for the reduction-based forecaster.
    """)


@app.cell
def _(
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    SeasonalNaive,
    horizon,
    mo,
    plot_forecast,
    union_three,
    y_test,
    y_train,
):
    fc_union = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=union_three,
    )
    fc_union.fit(y_train, forecasting_horizon=horizon)
    _y_pred_union = fc_union.predict(forecasting_horizon=horizon)

    # Baseline
    _naive = SeasonalNaive(seasonality=12)
    _naive.fit(y_train, forecasting_horizon=horizon)
    _y_pred_naive = _naive.predict(forecasting_horizon=horizon)

    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _mae_union = float(_scorer.score(y_test, _y_pred_union))
    _mae_naive = float(_scorer.score(y_test, _y_pred_naive))

    mo.md(f"**FeatureUnion + Ridge MAE**: {_mae_union:.2f}\n\n**SeasonalNaive MAE**: {_mae_naive:.2f}")
    return (fc_union,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) displays the predictions produced by the
    [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/)-powered forecaster against the test data.
    """)


@app.cell
def _(fc_union, horizon, plot_forecast, y_test, y_train):
    _y_pred = fc_union.predict(forecasting_horizon=horizon)
    plot_forecast(
        y_test,
        _y_pred,
        y_train=y_train,
        n_history=60,
        title="FeatureUnion (Lags + Rolling + EMA): Sunspots",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **[`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/)** applies transformers in parallel and concatenates outputs
    - **`observation_horizon`** = MAX across all child transformers
    - **`verbose_feature_names_out=True`** prefixes columns with transformer name
    - Use as **`feature_transformer`** in [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) for rich feature engineering
    - Combine [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/), [`RollingStatisticsTransformer`](/pages/api/generated/yohou.preprocessing.window.RollingStatisticsTransformer/), [`ExponentialMovingAverage`](/pages/api/generated/yohou.preprocessing.window.ExponentialMovingAverage/), etc.

    ## Next Steps

    - **FeaturePipeline**: See [`examples/compose/pipeline_composition.py`](/examples/compose/pipeline_composition/) for sequential pipelines
    - **Decomposition**: See [`examples/compose/decomposition_variations.py`](/examples/compose/decomposition_variations/)
    - **Panel feature union**: See [`examples/compose/panel_pipelines.py`](/examples/compose/panel_pipelines/)
    """)


if __name__ == "__main__":
    app.run()
