# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Time-Weighted Scoring.

Demonstrates weighting functions (exponential, linear, seasonal) that
emphasise different time regions when evaluating forecasts.
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
    # Time-Weighted Scoring

    Yohou supports weighting forecast errors by time, allowing recent
    errors to count more, or emphasising specific seasonal periods.

    ## What You'll Learn

    - `exponential_decay_weight(half_life)`: recent-focused
    - `linear_decay_weight(max_steps)`: gradual recency bias
    - `seasonal_emphasis_weight(seasonality, emphasis)`: periodic emphasis
    - `compose_weights()`: multiply weight functions
    - Applying weights in scorer `score()` via `time_weight`
    - Panel data with time weights
    """)
    return

@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_dominick, fetch_tourism_monthly
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast, plot_time_weight
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer
    from yohou.utils.weighting import (
        compose_weights,
        exponential_decay_weight,
        linear_decay_weight,
        seasonal_emphasis_weight,
    )

    return (
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        compose_weights,
        exponential_decay_weight,
        linear_decay_weight,
        fetch_tourism_monthly,
        fetch_dominick,
        pl,
        plot_forecast,
        plot_time_weight,
        seasonal_emphasis_weight,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data and Predictions
    """)
    return

@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, fetch_tourism_monthly, mo):
    air = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "passengers"})
    _split = int(len(air) * 0.85)
    y_train = air.head(_split)
    y_test = air.tail(len(air) - _split)
    horizon = len(y_test)

    fc = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 12]),
    )
    fc.fit(y_train, forecasting_horizon=horizon)
    y_pred = fc.predict(forecasting_horizon=horizon)

    mo.md(f"**Air Passengers**: Train={len(y_train)}, Test={len(y_test)}")
    return air, fc, horizon, y_pred, y_test, y_train

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Exponential Decay Weight

    Recent observations receive the highest weight, decaying
    exponentially with a configurable `half_life`.
    """)
    return

@app.cell
def _(exponential_decay_weight, plot_time_weight, y_test):
    w_exp = exponential_decay_weight(half_life=6)
    _weights_df = y_test.select("time").with_columns(
        w_exp(y_test["time"]).alias("time_weight")
    )
    plot_time_weight(_weights_df, title="Exponential Decay (half_life=6)")
    return (w_exp,)

@app.cell
def _(MeanAbsoluteError, mo, w_exp, y_pred, y_test, y_train):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _unweighted = float(_scorer.score(y_test, y_pred))
    _weighted = float(
        _scorer.score(y_test, y_pred, time_weight=w_exp)
    )
    mo.md(
        f"**Unweighted MAE**: {_unweighted:.2f}\n\n"
        f"**Exponential-weighted MAE**: {_weighted:.2f}"
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Linear Decay Weight
    """)
    return

@app.cell
def _(linear_decay_weight, plot_time_weight, y_test):
    w_lin = linear_decay_weight(max_steps=None)
    _weights_df = y_test.select("time").with_columns(
        w_lin(y_test["time"]).alias("time_weight")
    )
    plot_time_weight(_weights_df, title="Linear Decay (full range)")
    return (w_lin,)

@app.cell
def _(MeanAbsoluteError, mo, w_lin, y_pred, y_test, y_train):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _weighted_lin = float(
        _scorer.score(y_test, y_pred, time_weight=w_lin)
    )
    mo.md(f"**Linear-weighted MAE**: {_weighted_lin:.2f}")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Seasonal Emphasis Weight

    Boost the weight at specific seasonal positions (e.g., every 12
    months for annual cycles).
    """)
    return

@app.cell
def _(plot_time_weight, seasonal_emphasis_weight, y_test):
    w_season = seasonal_emphasis_weight(seasonality=12, emphasis=3.0)
    _weights_df = y_test.select("time").with_columns(
        w_season(y_test["time"]).alias("time_weight")
    )
    plot_time_weight(_weights_df, title="Seasonal Emphasis (period=12, emphasis=3x)")
    return (w_season,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Compose Weights

    Multiply multiple weight functions together: e.g., exponential
    decay AND seasonal emphasis.
    """)
    return

@app.cell
def _(
    compose_weights,
    exponential_decay_weight,
    plot_time_weight,
    seasonal_emphasis_weight,
    y_test,
):
    w_composed = compose_weights(
        exponential_decay_weight(half_life=6),
        seasonal_emphasis_weight(seasonality=12, emphasis=2.0),
    )
    _weights_df = y_test.select("time").with_columns(
        w_composed(y_test["time"]).alias("time_weight")
    )
    plot_time_weight(_weights_df, title="Composed: Exponential + Seasonal")
    return (w_composed,)

@app.cell
def _(MeanAbsoluteError, mo, w_composed, y_pred, y_test, y_train):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _weighted_comp = float(
        _scorer.score(y_test, y_pred, time_weight=w_composed)
    )
    mo.md(f"**Composed-weighted MAE**: {_weighted_comp:.2f}")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Panel Data with Time Weights

    Time weights work with panel data: applied per-timestep across
    all groups.
    """)
    return

@app.cell
def _(
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    exponential_decay_weight,
    fetch_dominick,
    mo,
):
    _full = fetch_dominick().frame
    _selected = ["T7__profit", "T11__profit", "T12__profit", "T13__profit", "T15__profit", "T19__profit", "T22__profit", "T23__profit", "T24__profit"]
    _store = _full.select("time", *_selected)
    _target_cols = [c for c in _store.columns if c.endswith("__profit")]
    _y = _store.select("time", *_target_cols)
    _split_p = int(len(_y) * 0.85)
    _y_train_p = _y.head(_split_p)
    _y_test_p = _y.tail(len(_y) - _split_p)
    _horizon_p = len(_y_test_p)

    _fc_p = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 7]),
    )
    _fc_p.fit(_y_train_p, forecasting_horizon=_horizon_p)
    _y_pred_p = _fc_p.predict(forecasting_horizon=_horizon_p)

    _scorer = MeanAbsoluteError()
    _scorer.fit(_y_test_p)
    _w_panel = exponential_decay_weight(half_life=10)
    _unweighted_p = float(_scorer.score(_y_test_p, _y_pred_p))
    _weighted_p = float(
        _scorer.score(_y_test_p, _y_pred_p, time_weight=_w_panel)
    )

    mo.md(
        f"**Panel unweighted MAE**: {_unweighted_p:.2f}\n\n"
        f"**Panel exp-weighted MAE**: {_weighted_p:.2f}\n\n"
        "Time weights apply uniformly across all panel groups."
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **`exponential_decay_weight(half_life)`**: Most common; recent data weighted higher
    - **`linear_decay_weight(max_steps)`**: Gradual recency; `None` = full range
    - **`seasonal_emphasis_weight(seasonality, emphasis)`**: Boost specific seasonal positions
    - **`compose_weights(*fns)`**: Multiply weight functions for combined effects
    - **`time_weight`**: Pass to `scorer.score()` to weight per-timestep errors
    - **Panel data**: Weights apply uniformly to all groups (per-timestep)

    ## Next Steps

    - **Time-weighted forecasting**: See `examples/time_weighted_forecasting.py`
    - **Aggregation modes**: See `examples/metrics/aggregation_modes.py`
    - **Scoring**: See `examples/scoring.py`
    """)
    return

if __name__ == "__main__":
    app.run()
