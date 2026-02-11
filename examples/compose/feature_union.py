"""Parallel Feature Engineering with FeatureUnion.

Demonstrates FeatureUnion to combine multiple transformers in parallel,
including observation_horizon behaviour and verbose feature names.
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
async def _():
    import sys as _sys

    if "pyodide" in _sys.modules:
        import micropip

        await micropip.install(["plotly", "scikit-learn", "yohou"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Parallel Feature Engineering with FeatureUnion

    `FeatureUnion` applies multiple transformers in parallel to the
    same input and concatenates the results column-wise. The resulting
    `observation_horizon` is the **maximum** across all transformers.

    ## What You'll Learn

    - Combine lag features, rolling statistics, and scaling in parallel
    - `observation_horizon` = max of child transformers
    - `verbose_feature_names_out` for prefixed column names
    - Using `FeatureUnion` as `feature_transformer` in a forecaster
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import FeatureUnion
    from yohou.datasets import load_sunspots
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
        load_sunspots,
        pl,
        plot_forecast,
        plot_time_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load Data
    """)
    return


@app.cell
def _(load_sunspots, mo):
    sunspots = load_sunspots()
    _split = int(len(sunspots) * 0.85)
    y_train = sunspots.head(_split)
    y_test = sunspots.tail(len(sunspots) - _split)
    horizon = len(y_test)
    mo.md(
        f"**Sunspots**: {len(sunspots)} months, "
        f"**Train**: {len(y_train)}, **Test**: {len(y_test)}"
    )
    return horizon, sunspots, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Basic FeatureUnion

    Combine lag features with rolling statistics.
    """)
    return


@app.cell
def _(FeatureUnion, LagTransformer, RollingStatisticsTransformer, mo, y_train):
    union_basic = FeatureUnion(
        transformer_list=[
            ("lags", LagTransformer(lag=[1, 6, 12])),
            ("rolling", RollingStatisticsTransformer(window_size=12, statistics=["mean", "std"])),
        ],
    )
    union_basic.fit(y_train)
    _y_features = union_basic.transform(y_train)

    mo.md(
        f"**Output columns**: {_y_features.columns}\n\n"
        f"**Output shape**: {_y_features.shape}\n\n"
        f"**observation_horizon**: {union_basic.observation_horizon} "
        f"(max of lag=12 and window=12)"
    )
    return (union_basic,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Verbose Feature Names

    When `verbose_feature_names_out=True` (default), each output column
    is prefixed with the transformer name.
    """)
    return


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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Three-Way Union

    Combine lags, rolling statistics, and exponential moving average.
    """)
    return


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
    _y_three = union_three.transform(y_train)

    mo.md(
        f"**Three-way union output**: {_y_three.shape}\n\n"
        f"**Columns**: {_y_three.columns}\n\n"
        f"**observation_horizon**: {union_three.observation_horizon}"
    )
    return (union_three,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. FeatureUnion in a Forecaster

    Use `FeatureUnion` as the `feature_transformer` to produce rich
    feature sets for the reduction-based forecaster.
    """)
    return


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
    _mae_union = float(
        _scorer.score(y_test, _y_pred_union)
    )
    _mae_naive = float(
        _scorer.score(y_test, _y_pred_naive)
    )

    mo.md(
        f"**FeatureUnion + Ridge MAE**: {_mae_union:.2f}\n\n"
        f"**SeasonalNaive MAE**: {_mae_naive:.2f}"
    )
    return (fc_union,)


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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **`FeatureUnion`** applies transformers in parallel and concatenates outputs
    - **`observation_horizon`** = MAX across all child transformers
    - **`verbose_feature_names_out=True`** prefixes columns with transformer name
    - Use as **`feature_transformer`** in `PointReductionForecaster` for rich feature engineering
    - Combine `LagTransformer`, `RollingStatisticsTransformer`, `ExponentialMovingAverage`, etc.

    ## Next Steps

    - **FeaturePipeline**: See `examples/compose/pipeline_composition.py` for sequential pipelines
    - **Decomposition**: See `examples/compose/decomposition_variations.py`
    - **Panel feature union**: See `examples/compose/panel_pipelines.py`
    """)
    return


if __name__ == "__main__":
    app.run()
