# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Custom Function-Based Transforms.

Demonstrates FunctionTransformer for wrapping arbitrary polars operations
as sklearn-compatible transformers, including stateful (auto-warmup) and
stateless patterns, inverse transforms, and check_inverse validation.
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Custom Function-Based Transforms

    `FunctionTransformer` wraps arbitrary polars operations as a full
    sklearn-compatible transformer. This gives you `fit` / `transform` /
    `inverse_transform` lifecycle methods, automatic statefulness detection,
    and seamless composition with forecasters and pipelines.

    ## What You'll Learn

    - Wrapping a stateless function (log, clip, scale)
    - Wrapping a stateful function (differencing, rolling) with auto-warmup
    - Inverse transforms and round-trip verification via `check_inverse`
    - Custom `feature_names_out` for renamed columns
    - Integrating `FunctionTransformer` inside a forecaster pipeline

    ## Prerequisites

    Working knowledge of polars expressions and `PointReductionForecaster`
    (see `examples/quickstart.py`).
    """)

@app.cell(hide_code=True)
def _():
    import numpy as np
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_tourism_monthly
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import FunctionTransformer, LagTransformer

    return (
        FunctionTransformer,
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        fetch_tourism_monthly,
        np,
        pl,
        plot_forecast,
        plot_time_series,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load Data
    """)

@app.cell
def _(fetch_tourism_monthly):
    df = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    split_idx = int(len(df) * 0.85)
    y_train = df.head(split_idx).select("time", "tourists")
    y_test = df.tail(len(df) - split_idx).select("time", "tourists")
    y_train.head()
    return df, split_idx, y_test, y_train

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Stateless Transform: Log Scale

    A simple log transform stabilises variance in multiplicative time series.
    The function receives a polars `DataFrame` **without the `"time"` column**
    and must return a DataFrame (or array) with the same number of rows.
    """)

@app.cell
def _(FunctionTransformer, np, y_train):
    log_transformer = FunctionTransformer(
        func=np.log,
        inverse_func=np.exp,
        check_inverse=False,
    )
    y_log = log_transformer.fit_transform(y_train)
    y_log.head()
    return log_transformer, y_log

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We set `check_inverse=False` because the tourism dataset is monthly data
    (variable day counts). With daily or sub-daily data you can use
    `check_inverse=True` (default) to verify `exp(log(x)) ≈ x` at `fit()` time.
    """)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Stateful Transform: Differencing

    When `func` produces leading NaN rows (e.g., `.diff()`), yohou
    auto-detects the **warmup** and sets `observation_horizon` accordingly.
    The first `observation_horizon` rows are stored as memory so that
    `inverse_transform` can recover the original values.
    """)

@app.cell
def _(FunctionTransformer, mo, pl, y_train):
    def _diff(df: pl.DataFrame) -> pl.DataFrame:
        return df.select(pl.all().diff())

    def _cumsum(df: pl.DataFrame, past: pl.DataFrame | None = None) -> pl.DataFrame:
        if past is not None:
            # Prepend past observation so cumsum starts from the original level
            combined = pl.concat([past, df])
            return combined.select(pl.all().cum_sum()).tail(len(df))
        return df.select(pl.all().cum_sum())

    diff_transformer = FunctionTransformer(
        func=_diff,
        inverse_func=_cumsum,
        check_inverse=False,
    )
    y_diff = diff_transformer.fit_transform(y_train)

    mo.md(
        f"**Original length**: {len(y_train)} rows\n\n"
        f"**After fit_transform**: {len(y_diff)} rows "
        f"(observation_horizon={diff_transformer._observation_horizon})\n\n"
        f"The first row was consumed as warmup."
    )
    return diff_transformer, y_diff

@app.cell
def _(diff_transformer, mo, y_diff, y_train):
    # inverse_transform needs X_p (warmup data) when observation_horizon > 0
    _y_recovered = diff_transformer.inverse_transform(
        y_diff, y_train.head(diff_transformer._observation_horizon)
    )
    # Compare recovered vs. original (minus warmup)
    _y_orig_tail = y_train.tail(len(y_diff))
    _max_diff = (
        _y_recovered.select("tourists").to_series()
        - _y_orig_tail.select("tourists").to_series()
    ).abs().max()
    mo.md(
        f"**Max absolute reconstruction error**: {_max_diff}\n\n"
        f"The round-trip is exact because `inverse_transform` uses stored "
        f"observation memory to recover the initial level."
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Custom Feature Names

    By default, output column names match the input. Use `feature_names_out`
    to rename:
    - `"one-to-one"`: keep original names (same as default)
    - A **callable** `(transformer, input_features) -> list[str]`
    """)

@app.cell
def _(FunctionTransformer, mo, pl, y_train):
    def _pct_change(df: pl.DataFrame) -> pl.DataFrame:
        return df.select(pl.all().pct_change())

    pct_transformer = FunctionTransformer(
        func=_pct_change,
        feature_names_out=lambda self, names: [f"{n}_pct" for n in names],
    )
    _y_pct = pct_transformer.fit_transform(y_train)
    _names = pct_transformer.get_feature_names_out()
    mo.md(f"**Output feature names**: {_names}")
    return (pct_transformer,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Keyword Arguments

    Pass extra arguments to `func` and `inverse_func` via `kw_args` and
    `inv_kw_args`.
    """)

@app.cell
def _(FunctionTransformer, mo, pl, y_train):
    def _clip(df: pl.DataFrame, *, lower: float, upper: float) -> pl.DataFrame:
        return df.select(pl.all().clip(lower, upper))

    clip_transformer = FunctionTransformer(
        func=_clip,
        kw_args={"lower": 100.0, "upper": 500.0},
    )
    _y_clipped = clip_transformer.fit_transform(y_train)
    mo.md(
        f"**Min before clip**: {y_train['tourists'].min()}\n\n"
        f"**Min after clip**: {_y_clipped['tourists'].min()}\n\n"
        f"**Max after clip**: {_y_clipped['tourists'].max()}"
    )
    return (clip_transformer,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Inside a Forecaster

    Use `FunctionTransformer` as `target_transformer` to log-transform the
    target before fitting and automatically back-transform predictions.
    """)

@app.cell
def _(
    FunctionTransformer,
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    np,
    y_test,
    y_train,
):
    forecaster = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        target_transformer=FunctionTransformer(
            func=np.log,
            inverse_func=np.exp,
            check_inverse=False,
        ),
        feature_transformer=LagTransformer(lag=[1, 6, 12]),
    )

    _horizon = len(y_test)
    forecaster.fit(y_train, forecasting_horizon=_horizon)
    y_pred = forecaster.predict(forecasting_horizon=_horizon)

    scorer = MeanAbsoluteError()
    scorer.fit(y_train)
    _score = scorer.score(y_test, y_pred)
    return forecaster, scorer, y_pred

@app.cell
def _(plot_forecast, y_pred, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred,
        y_train=y_train,
        n_history=36,
        title="Forecast with Log Target Transform",
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - `FunctionTransformer` wraps any `polars DataFrame -> DataFrame` function as an sklearn-compatible transformer
    - **func** receives the DataFrame **without** the `"time"` column; time is re-attached automatically
    - **Statefulness** is auto-detected: if `func` produces leading NaN rows, those become `observation_horizon` warmup
    - **check_inverse** verifies the round-trip `inverse_func(func(x)) ≈ x` at fit time (warns, does not raise)
    - **kw_args / inv_kw_args** pass extra keyword arguments to `func` / `inverse_func`
    - **feature_names_out** controls output column naming: `"one-to-one"` or a callable
    - Compose with forecasters via `target_transformer` or `feature_transformer` parameters

    ## Next Steps

    - **Sklearn wrappers**: See `examples/preprocessing/sklearn_wrappers.py` for built-in StandardScaler, MinMaxScaler, etc.
    - **Window transforms**: See `examples/preprocessing/window_transforms.py` for rolling and expanding windows
    - **Signal processing**: See `examples/preprocessing/signal_processing.py` for numerical filters and differentiators
    """)

if __name__ == "__main__":
    app.run()
