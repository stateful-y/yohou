# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Time-Weighted Reduction",
    "description": "Use time_weight and sample_weight_alignment to emphasise recent or seasonal training samples in PointReductionForecaster, with visualisation of weight curves and alignment strategy comparison.",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Time-Weighted Reduction Forecasting

    By default, every training sample contributes equally to the fitted
    estimator. With **time weighting** you can emphasise recent observations,
    de-emphasise stale data, or boost seasonal positions - all without
    discarding any data.

    [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) accepts a `time_weight` argument in
    `fit()` and converts it to sklearn's `sample_weight` using a configurable
    `sample_weight_alignment` strategy.

    ## What You'll Learn

    - Creating weight functions with `exponential_decay_weight`, `linear_decay_weight`, `seasonal_emphasis_weight`
    - Composing multiple weight functions with `compose_weights`
    - Visualising weights with `plot_time_weight`
    - Passing `time_weight` to `fit()`
    - Comparing the five `sample_weight_alignment` strategies
    - Tuning weight parameters with `GridSearchCV`

    ## Prerequisites

    Familiarity with [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) (see [`reduction_forecaster.py`](/examples/point/reduction_forecaster/)).
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge
    from yohou.model_selection import train_test_split

    from yohou.datasets import fetch_sunspot
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast, plot_score_per_step, plot_score_summary, plot_time_weight
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer
    from yohou.utils.weighting import (
        compose_weights,
        exponential_decay_weight,
        linear_decay_weight,
        seasonal_emphasis_weight,
    )

    return (
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        compose_weights,
        exponential_decay_weight,
        fetch_sunspot,
        linear_decay_weight,
        pl,
        plot_forecast,
        plot_score_per_step,
        plot_score_summary,
        plot_time_weight,
        seasonal_emphasis_weight,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We use the Sunspot dataset resampled to monthly frequency and split it
    into training and test sets with a 24-month forecast horizon.
    """)


@app.cell
def _(fetch_sunspot, pl, train_test_split):
    y_raw = fetch_sunspot().frame
    y = y_raw.group_by_dynamic("time", every="1mo").agg(pl.col("sunspot_number").mean())

    y_train, y_test = train_test_split(y, test_size=0.05)
    forecasting_horizon = len(y_test)

    print(f"Train: {len(y_train)}, Test: {len(y_test)}, Horizon: {forecasting_horizon}")
    return forecasting_horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Weight Functions

    Yohou provides three built-in weight generators. Each returns a callable
    `(pl.Series) -> pl.Series` that maps a time column to non-negative weights.

    | Function | Behaviour |
    |---|---|
    | `exponential_decay_weight(half_life)` | Halves weight every `half_life` steps from the most recent time |
    | `linear_decay_weight(max_steps)` | Linear ramp from 0 (oldest) to 1 (newest) |
    | `seasonal_emphasis_weight(seasonality, emphasis)` | Multiplies weight by `emphasis` for times matching the seasonal phase |

    Let's visualise each one on our training data.
    """)


@app.cell
def _(exponential_decay_weight, linear_decay_weight, pl, plot_time_weight, seasonal_emphasis_weight, y_train):
    _times = y_train.get_column("time")

    # Build a DataFrame with each weight as a column
    weight_df = pl.DataFrame({
        "time": _times,
        "time_weight": exponential_decay_weight(half_life=365)(_times).to_list(),
    })
    plot_time_weight(weight_df, title="Exponential Decay (half_life=365 days)")


@app.cell
def _(linear_decay_weight, pl, plot_time_weight, y_train):
    _times = y_train.get_column("time")
    weight_df_linear = pl.DataFrame({
        "time": _times,
        "time_weight": linear_decay_weight(max_steps=None)(_times).to_list(),
    })
    plot_time_weight(weight_df_linear, title="Linear Decay (full range)")


@app.cell
def _(pl, plot_time_weight, seasonal_emphasis_weight, y_train):
    _times = y_train.get_column("time")
    weight_df_seasonal = pl.DataFrame({
        "time": _times,
        "time_weight": seasonal_emphasis_weight(seasonality=12, emphasis=3.0)(_times).to_list(),
    })
    plot_time_weight(weight_df_seasonal, title="Seasonal Emphasis (period=12, emphasis=3x)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Composing Weights

    `compose_weights` multiplies outputs of multiple weight functions. This lets
    you combine, for example, exponential recency with seasonal emphasis:
    """)


@app.cell
def _(compose_weights, exponential_decay_weight, pl, plot_time_weight, seasonal_emphasis_weight, y_train):
    composed_fn = compose_weights(
        exponential_decay_weight(half_life=365),
        seasonal_emphasis_weight(seasonality=12, emphasis=3.0),
    )

    _times = y_train.get_column("time")
    weight_df_composed = pl.DataFrame({
        "time": _times,
        "time_weight": composed_fn(_times).to_list(),
    })
    plot_time_weight(weight_df_composed, title="Composed: Exponential Decay + Seasonal Emphasis")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Fitting with Time Weights

    Pass a weight callable (or a DataFrame with `"time"` and `"weight"` columns)
    to `fit()` via the `time_weight` parameter. The forecaster evaluates the
    function on the training time column and converts it to sklearn
    `sample_weight` during tabularization.
    """)


@app.cell
def _(
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    exponential_decay_weight,
    forecasting_horizon,
    plot_forecast,
    y_test,
    y_train,
):
    fc_uniform = PointReductionForecaster(
        estimator=Ridge(),
        feature_transformer=LagTransformer(lag=list(range(1, 13))),
    )
    fc_uniform.fit(y_train, forecasting_horizon=forecasting_horizon)
    pred_uniform = fc_uniform.predict(forecasting_horizon=forecasting_horizon)

    fc_weighted = PointReductionForecaster(
        estimator=Ridge(),
        feature_transformer=LagTransformer(lag=list(range(1, 13))),
    )
    fc_weighted.fit(
        y_train,
        forecasting_horizon=forecasting_horizon,
        time_weight=exponential_decay_weight(half_life=365),
    )
    pred_weighted = fc_weighted.predict(forecasting_horizon=forecasting_horizon)

    mae = MeanAbsoluteError()
    mae.fit(y_train)
    print(f"MAE (uniform):  {mae.score(y_test, pred_uniform):.2f}")
    print(f"MAE (weighted): {mae.score(y_test, pred_weighted):.2f}")

    plot_forecast(
        y_test,
        {"Uniform": pred_uniform, "Exp. Decay": pred_weighted},
        y_train=y_train.tail(120),
        title="Uniform vs Exponential Decay Weighting",
    )
    return pred_uniform, pred_weighted


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Sample Weight Alignment

    After tabularization each training sample covers a window of `H` future
    steps. The original per-timestep weights must be collapsed into a single
    `sample_weight` per row. The `sample_weight_alignment` parameter controls
    how:

    | Strategy | Rule |
    |---|---|
    | `"first_step"` (default) | Weight at the first forecast step `t+1` |
    | `"mean_step"` | Arithmetic mean across all `H` steps |
    | `"weighted_mean_step"` | Exponentially weighted mean favouring near steps |
    | `"max_weight_step"` | Maximum weight across `H` steps |
    | `"min_weight_step"` | Minimum weight (conservative: all steps must be important) |

    The choice matters most when the weight function changes steeply across the
    horizon window. Let's compare all five on our dataset.
    """)


@app.cell
def _(
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    exponential_decay_weight,
    forecasting_horizon,
    mo,
    y_test,
    y_train,
):
    _alignments = [
        "first_step",
        "mean_step",
        "weighted_mean_step",
        "max_weight_step",
        "min_weight_step",
    ]
    _weight_fn = exponential_decay_weight(half_life=365)
    _mae = MeanAbsoluteError()
    _mae.fit(y_train)

    alignment_rows = []
    for _align in _alignments:
        _fc = PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(lag=list(range(1, 13))),
        )
        _fc.fit(
            y_train,
            forecasting_horizon=forecasting_horizon,
            time_weight=_weight_fn,
            sample_weight_alignment=_align,
        )
        _pred = _fc.predict(forecasting_horizon=forecasting_horizon)
        _score = _mae.score(y_test, _pred)
        alignment_rows.append({"Strategy": _align, "MAE": round(_score, 2)})

    mo.ui.table(alignment_rows, label="Sample Weight Alignment Comparison")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Comparing Weight Functions

    Let's compare no weighting, exponential decay, linear decay, and seasonal
    emphasis side by side.
    """)


@app.cell
def _(
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    exponential_decay_weight,
    forecasting_horizon,
    linear_decay_weight,
    plot_score_summary,
    seasonal_emphasis_weight,
    y_test,
    y_train,
):
    _weight_configs = {
        "Uniform": None,
        "Exp. Decay (365d)": exponential_decay_weight(half_life=365),
        "Linear Decay": linear_decay_weight(max_steps=None),
        "Seasonal (12)": seasonal_emphasis_weight(seasonality=12, emphasis=3.0),
    }

    _y_preds = {}
    for _label, _wfn in _weight_configs.items():
        _fc = PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(lag=list(range(1, 13))),
        )
        _fit_kwargs = {"forecasting_horizon": forecasting_horizon}
        if _wfn is not None:
            _fit_kwargs["time_weight"] = _wfn
        _fc.fit(y_train, **_fit_kwargs)
        _y_preds[_label] = _fc.predict(forecasting_horizon=forecasting_horizon)

    plot_score_summary(
        MeanAbsoluteError(),
        y_test,
        _y_preds,
        title="Weight Function Comparison (MAE)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - `time_weight` accepts a callable, a DataFrame, or `None` (uniform)
    - Built-in generators: `exponential_decay_weight`, `linear_decay_weight`, `seasonal_emphasis_weight`
    - `compose_weights` multiplies weight functions for combined strategies
    - `sample_weight_alignment` controls how per-timestep weights map to tabularized samples
    - Weight functions and alignment strategies are **orthogonal** to `reduction_strategy`
    - Visualise weights before fitting to verify the profile makes sense

    ## Next Steps

    - **Reduction strategies**: See [`reduction_strategies.py`](/examples/point/reduction_strategies/) for multi-output vs direct vs dir-rec
    - **Feature construction**: See [`reduction_forecaster.py`](/examples/point/reduction_forecaster/) for transformers and `target_as_feature`
    - **Hyperparameter search**: Tune `half_life` or `emphasis` via [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/)
    - **Scoring with weights**: Scorers also accept `time_weight` for weighted evaluation
    """)


if __name__ == "__main__":
    app.run()
