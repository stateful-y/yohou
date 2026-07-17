# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "How to Score with Time-Weighted Metrics",
    "description": "Apply exponential decay, linear decay, and seasonal emphasis weighting to forecast evaluation, prioritising recent or periodic time steps.",
    "category": "how-to",
    "section": "evaluation-search",
    "companion": "/pages/how-to/time-weighting/",
}
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

    ## 1. Prepare Data and Predictions

    We load the Tourism Monthly dataset, fit a [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) with
    a [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) regressor, and generate predictions. The scorer will then evaluate
    these predictions with different time-weighting strategies.
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_dominick, fetch_tourism_monthly
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import train_test_split
    from yohou.plotting import (
        plot_score_per_vintage,
        plot_score_time_series,
        plot_time_series,
        plot_time_weight,
    )
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer
    from yohou.weighting import (
        ExponentialDecayWeighter,
        LinearDecayWeighter,
        CompositeWeighter,
        SeasonalEmphasisWeighter,
    )

    return (
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        CompositeWeighter,
        deepcopy,
        ExponentialDecayWeighter,
        fetch_dominick,
        fetch_tourism_monthly,
        LinearDecayWeighter,
        plot_score_per_vintage,
        plot_score_time_series,
        plot_time_series,
        plot_time_weight,
        SeasonalEmphasisWeighter,
        train_test_split,
    )


@app.cell
def _(
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    fetch_tourism_monthly,
    mo,
    train_test_split,
):
    tourism = (
        fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    )
    y_train, y_test = train_test_split(tourism, test_size=0.15)
    horizon = len(y_test)

    fc = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        actual_transformer=LagTransformer(lag=[1, 12]),
    )
    fc.fit(y_train, forecasting_horizon=horizon)
    y_pred = fc.predict(forecasting_horizon=horizon)

    mo.md(f"**Tourism Monthly**: Train={len(y_train)}, Test={len(y_test)}")
    return tourism, y_pred, y_test, y_train


@app.cell
def _(plot_time_series, tourism):
    plot_time_series(tourism, title="Tourism Monthly")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Exponential Decay Weight

    Recent observations receive the highest weight, decaying
    exponentially with a configurable `half_life`.
    """)


@app.cell
def _(ExponentialDecayWeighter, plot_time_weight, y_test):
    w_exp = ExponentialDecayWeighter(half_life=6)
    _weights_df = y_test.select("time").with_columns(w_exp.compute_weights(y_test["time"]).alias("time_weight"))
    plot_time_weight(_weights_df, title="Exponential Decay (half_life=6)")
    return (w_exp,)


@app.cell
def _(MeanAbsoluteError, mo, w_exp, y_pred, y_test, y_train):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _unweighted = float(_scorer.score(y_test, y_pred))
    _weighted = float(_scorer.set_params(time_weighter=w_exp).score(y_test, y_pred))
    mo.md(f"**Unweighted MAE**: {_unweighted:.2f}\n\n**Exponential-weighted MAE**: {_weighted:.2f}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Linear Decay Weight

    [`LinearDecayWeighter`](/pages/api/generated/yohou.weighting.weighters.LinearDecayWeighter/) assigns weights that decrease linearly from the most
    recent observation to the oldest. The optional `max_steps` parameter limits
    how far back the decay extends; when set to `None`, the full test range
    is used.
    """)


@app.cell
def _(LinearDecayWeighter, plot_time_weight, y_test):
    w_lin = LinearDecayWeighter(max_steps=None)
    _weights_df = y_test.select("time").with_columns(w_lin.compute_weights(y_test["time"]).alias("time_weight"))
    plot_time_weight(_weights_df, title="Linear Decay (full range)")
    return (w_lin,)


@app.cell
def _(MeanAbsoluteError, mo, w_lin, y_pred, y_test, y_train):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _weighted_lin = float(_scorer.set_params(time_weighter=w_lin).score(y_test, y_pred))
    mo.md(f"**Linear-weighted MAE**: {_weighted_lin:.2f}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Seasonal Emphasis Weight

    Boost the weight at specific seasonal positions (e.g., every 12
    months for annual cycles).
    """)


@app.cell
def _(plot_time_weight, SeasonalEmphasisWeighter, y_test):
    w_season = SeasonalEmphasisWeighter(seasonality=12, emphasis=3.0)
    _weights_df = y_test.select("time").with_columns(w_season.compute_weights(y_test["time"]).alias("time_weight"))
    plot_time_weight(_weights_df, title="Seasonal Emphasis (period=12, emphasis=3x)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Compose Weights

    Multiply multiple weight functions together: e.g., exponential
    decay AND seasonal emphasis.
    """)


@app.cell
def _(
    CompositeWeighter,
    ExponentialDecayWeighter,
    plot_time_weight,
    SeasonalEmphasisWeighter,
    y_test,
):
    w_composed = CompositeWeighter([
        ("decay", ExponentialDecayWeighter(half_life=6)),
        ("seasonal", SeasonalEmphasisWeighter(seasonality=12, emphasis=2.0)),
    ])
    _weights_df = y_test.select("time").with_columns(w_composed.compute_weights(y_test["time"]).alias("time_weight"))
    plot_time_weight(_weights_df, title="Composed: Exponential + Seasonal")
    return (w_composed,)


@app.cell
def _(MeanAbsoluteError, mo, w_composed, y_pred, y_test, y_train):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _weighted_comp = float(_scorer.set_params(time_weighter=w_composed).score(y_test, y_pred))
    mo.md(f"**Composed-weighted MAE**: {_weighted_comp:.2f}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Panel Data with Time Weights

    Time weights work with panel data: applied per-timestep across
    all groups.
    """)


@app.cell
def _(
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    CompositeWeighter,
    ExponentialDecayWeighter,
    fetch_dominick,
    LinearDecayWeighter,
    mo,
    plot_score_time_series,
    SeasonalEmphasisWeighter,
    train_test_split,
):
    _full = fetch_dominick().frame
    _selected = [
        "T7__profit",
        "T11__profit",
        "T12__profit",
        "T13__profit",
        "T15__profit",
        "T19__profit",
        "T22__profit",
        "T23__profit",
        "T24__profit",
    ]
    _store = _full.select("time", *_selected)
    _target_cols = [c for c in _store.columns if c.endswith("__profit")]
    _y = _store.select("time", *_target_cols)
    _y_train_p, _y_test_p = train_test_split(_y, test_size=0.15)
    _horizon_p = len(_y_test_p)

    _fc_p = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        actual_transformer=LagTransformer(lag=[1, 7]),
    )
    _fc_p.fit(_y_train_p, forecasting_horizon=_horizon_p)
    _y_pred_p = _fc_p.predict(forecasting_horizon=_horizon_p)

    _weights = [
        ("Exponential Decay (half_life=10)", ExponentialDecayWeighter(half_life=10)),
        ("Linear Decay", LinearDecayWeighter(max_steps=None)),
        ("Seasonal Emphasis (period=7, 3×)", SeasonalEmphasisWeighter(seasonality=7, emphasis=3.0)),
        (
            "Composed: Exponential + Seasonal",
            CompositeWeighter([
                ("decay", ExponentialDecayWeighter(half_life=10)),
                ("seasonal", SeasonalEmphasisWeighter(seasonality=7, emphasis=2.0)),
            ]),
        ),
    ]

    _plots = []
    for _label, _w in _weights:
        _fig = plot_score_time_series(
            MeanAbsoluteError(time_weighter=_w),
            _y_test_p,
            _y_pred_p,
            title=f"Panel MAE - {_label}",
        )
        _plots.append(_fig)

    mo.vstack(_plots)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-vintage Scoring

    The `observe_predict` method with `stride=1` produces one forecast per
    observation point, creating multiple *vintages*. Each vintage represents
    a different forecast origin, so you can analyse how accuracy evolves as
    the model absorbs more data.
    """)


@app.cell
def _(deepcopy, fc, horizon, y_test):
    _vintage_model = deepcopy(fc)
    y_pred_vintages = _vintage_model.observe_predict(
        y=y_test,
        stride=1,
        forecasting_horizon=horizon,
    )
    print(f"Vintages: {y_pred_vintages['vintage_time'].n_unique()}")
    y_pred_vintages.head(10)
    return (y_pred_vintages,)


@app.cell
def _(MeanAbsoluteError, y_train):
    vintage_scorer = MeanAbsoluteError()
    vintage_scorer.fit(y_train)
    return (vintage_scorer,)


@app.cell
def _(vintage_scorer, plot_score_per_vintage, y_pred_vintages, y_test):
    plot_score_per_vintage(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="MAE per Forecast Vintage",
        y_label="MAE",
        height=380,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Time-weighted forecasting**: See `examples/time_weighted_forecasting.py`
    - **Aggregation modes**: See [`examples/metrics/aggregation_modes.py`](/examples/evaluation-search/aggregation_modes/)
    - **Scoring**: See `examples/scoring.py`
    """)


if __name__ == "__main__":
    app.run()
