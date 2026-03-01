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
    "title": "Aggregation Modes",
    "description": "Demonstrate all scorer aggregation strategies (timewise, componentwise, groupwise, coveragewise, all) on panel data with weighted group aggregation.",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Scorer Aggregation Modes

    Yohou scorers support multiple **aggregation strategies** that control how
    per-timestep, per-component errors are reduced into summary scores.

    ## What You'll Learn

    - `"all"`: single scalar (default)
    - `"timewise"`: per-column average (aggregate over time)
    - `"componentwise"`: per-group over time (aggregate members within each panel group)
    - `"groupwise"`: per-component over time (aggregate across panel groups)
    - `"coveragewise"`: aggregate over coverage rates (interval scorers only)
    - Combining modes and `panel_group_weight` for weighted aggregation
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    from yohou.datasets import fetch_kdd_cup
    from yohou.interval import SplitConformalForecaster
    from yohou.metrics import EmpiricalCoverage, MeanAbsoluteError
    from yohou.plotting import plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer
    from yohou.utils.panel import inspect_panel

    return (
        EmpiricalCoverage,
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        SplitConformalForecaster,
        fetch_kdd_cup,
        inspect_panel,
        pl,
        plot_time_series,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data and Predictions

    We load the **KDD Cup 2018** air quality dataset, a multivariate panel
    with 3 Beijing stations, each monitoring 6 pollutants (PM2.5, PM10, NO2,
    CO, O3, SO2). This structure with **multiple members per group** is key
    to showing the difference between componentwise and groupwise aggregation.
    """)


@app.cell
def _(
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    fetch_kdd_cup,
    inspect_panel,
    mo,
    plot_time_series,
    train_test_split,
):
    _bunch = fetch_kdd_cup(n_groups=3)
    _y = _bunch.frame.drop_nulls().tail(250)

    fh = 24
    y_train, y_test = train_test_split(_y, test_size=fh, shuffle=False)

    _, groups = inspect_panel(_y)

    _fc = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 24]),
    )
    _fc.fit(y_train, forecasting_horizon=fh)
    y_pred = _fc.predict(forecasting_horizon=fh)

    mo.vstack([
        mo.md(
            f"**Panel structure**: {len(groups)} groups × "
            f"{len(next(iter(groups.values())))} members = "
            f"{sum(len(v) for v in groups.values())} columns\n\n"
            f"**Groups**: "
            f"{ {k: [c.split('__')[1] for c in v] for k, v in groups.items()} }\n\n"
            f"**Train**: {len(y_train)} hours · **Test**: {fh} hours"
        ),
        plot_time_series(_y, title="KDD Cup 2018 - Three Beijing Stations"),
    ])
    return fh, groups, y_pred, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. `"all"`: Single Scalar

    The default mode. Every dimension (time, columns, groups) is aggregated
    into a single number.
    """)


@app.cell
def _(MeanAbsoluteError, mo, y_pred, y_test, y_train):
    _scorer = MeanAbsoluteError(aggregation_method="all")
    _scorer.fit(y_train)
    _score = _scorer.score(y_test, y_pred)
    mo.md(f"**MAE (`'all'`)**: {float(_score):.2f}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. `"timewise"`: Per-Column Average

    Aggregate over time → one score per column. Shows which individual
    series are hardest to forecast.
    """)


@app.cell
def _(MeanAbsoluteError, mo, y_pred, y_test, y_train):
    _scorer_tw = MeanAbsoluteError(aggregation_method="timewise")
    _scorer_tw.fit(y_train)
    _score_tw = _scorer_tw.score(y_test, y_pred)
    mo.vstack([
        mo.md(
            f"**Shape**: {_score_tw.shape} i.e. one row, one column per series\n\n"
            "Each value is the mean absolute error for that column across all "
            "test timesteps."
        ),
        _score_tw,
    ])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. `"componentwise"`: Per-Group Over Time

    Aggregate **members within each panel group** at each timestep. Since
    each region has two series (`demand` and `temperature`), their errors are
    averaged into a single per-group score.

    This is where having **multivariate groups** matters: the six pollutant
    errors within each station are collapsed into one `{station}__score`
    column.
    """)


@app.cell
def _(MeanAbsoluteError, mo, y_pred, y_test, y_train):
    _scorer_cw = MeanAbsoluteError(aggregation_method="componentwise")
    _scorer_cw.fit(y_train)
    _score_cw = _scorer_cw.score(y_test, y_pred)
    mo.vstack([
        mo.md(
            f"**Shape**: {_score_cw.shape} i.e. "
            f"one row per timestep, one column per **group**\n\n"
            f"**Columns**: {_score_cw.columns}\n\n"
            "Note the `__score` suffix: pollutant errors within each "
            "station were averaged."
        ),
        _score_cw.head(5),
    ])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. `"groupwise"`: Per-Component Over Time

    Aggregates **across panel groups** for each component at each timestep.
    Since each station monitors 6 pollutants, the 3 station-specific errors
    for each pollutant are averaged into a single `{pollutant}` column.

    Compare with componentwise (section 4): componentwise aggregates
    *within* groups (pollutants → stations), groupwise aggregates *across*
    groups (stations → pollutants).
    """)


@app.cell
def _(MeanAbsoluteError, mo, y_pred, y_test, y_train):
    _scorer_gw = MeanAbsoluteError(aggregation_method="groupwise")
    _scorer_gw.fit(y_train)
    _score_gw = _scorer_gw.score(y_test, y_pred)
    mo.vstack([
        mo.md(
            f"**Shape**: {_score_gw.shape} i.e. "
            f"one row per timestep, one column per **component**\n\n"
            f"**Columns**: {_score_gw.columns}\n\n"
            "Compare with componentwise above: componentwise collapses "
            "pollutants within each station (18 → 3 groups), groupwise "
            "collapses stations for each pollutant (18 → 6 components)."
        ),
        _score_gw.head(5),
    ])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Combining Modes

    Pass a list to aggregate over multiple dimensions at once.
    `["timewise", "componentwise"]` removes both time and member
    dimensions, producing a single scalar which is equivalent to a flat mean
    over all errors.
    """)


@app.cell
def _(MeanAbsoluteError, mo, y_pred, y_test, y_train):
    _scorer_tc = MeanAbsoluteError(aggregation_method=["timewise", "componentwise"])
    _scorer_tc.fit(y_train)
    _score_tc = _scorer_tc.score(y_test, y_pred)
    mo.md(
        f"**`['timewise', 'componentwise']`** → scalar: "
        f"{float(_score_tc):.2f}\n\n"
        "Both time and member dimensions are aggregated, collapsing "
        "everything into a single number."
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. `panel_group_weight`: Weighted Group Aggregation

    Pass a dictionary of group weights to emphasise certain stations during
    `"all"` aggregation. This is useful when some monitoring stations are
    more critical than others.
    """)


@app.cell
def _(MeanAbsoluteError, groups, mo, y_pred, y_test, y_train):
    _group_names = sorted(groups.keys())
    _weights = {_group_names[0]: 5.0}
    _scorer_w = MeanAbsoluteError(aggregation_method="all", panel_group_weight=_weights)
    _scorer_u = MeanAbsoluteError(aggregation_method="all")

    _scorer_w.fit(y_train)
    _scorer_u.fit(y_train)

    _s_w = _scorer_w.score(y_test, y_pred)
    _s_u = _scorer_u.score(y_test, y_pred)

    mo.md(
        f"**Weights**: `{_weights}` (others default to 1.0)\n\n"
        f"**Unweighted MAE**: {float(_s_u):.2f}\n\n"
        f"**Weighted MAE**: {float(_s_w):.2f}\n\n"
        "The score shifts toward the heavily-weighted group."
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. `"coveragewise"`: Interval Scorer Aggregation

    Interval scorers score each **coverage rate** separately. This creates an
    extra dimension in the output.

    - **Without** `"coveragewise"`: scores are returned per rate (as a dict)
    - **With** `"coveragewise"`: coverage rates are averaged, collapsing that
      dimension
    """)


@app.cell
def _(
    EmpiricalCoverage,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SplitConformalForecaster,
    fh,
    mo,
    y_test,
    y_train,
):
    _coverage_rates = [0.8, 0.9, 0.95]

    _fc_int = SplitConformalForecaster(
        point_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 7]),
        ),
        calibration_size=fh + 24,
    )
    _fc_int.fit(y_train, forecasting_horizon=fh, coverage_rates=_coverage_rates)
    _y_pred_int = _fc_int.predict_interval(forecasting_horizon=fh, coverage_rates=_coverage_rates)

    _cov_per_rate = EmpiricalCoverage(aggregation_method=["timewise", "componentwise"])
    _cov_all = EmpiricalCoverage(aggregation_method="all")

    _cov_per_rate.fit(y_train)
    _cov_all.fit(y_train)

    _s_per_rate = _cov_per_rate.score(y_test, _y_pred_int)
    _s_all = _cov_all.score(y_test, _y_pred_int)

    mo.md(
        f"**Coverage rates**: {_coverage_rates}\n\n"
        "---\n\n"
        "**Without `'coveragewise'`** "
        f"(`['timewise', 'componentwise']`) → dict per rate:\n\n"
        + "\n\n".join(f"- rate {r}: {v:.3f}" for r, v in _s_per_rate.items())
        + "\n\n---\n\n"
        f"**With `'coveragewise'`** (`'all'`) → scalar: "
        f"{float(_s_all):.3f}\n\n"
        "The per-rate dict shows that wider intervals (0.95) achieve higher "
        "coverage than narrow ones (0.8), as expected. Adding `'coveragewise'` "
        "averages across rates into a single number."
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    | Mode | Aggregates Over | Result |
    |------|----------------|--------|
    | `"all"` | time + components + groups + coverage | scalar |
    | `"timewise"` | time | one row, one col per series |
    | `"componentwise"` | members within groups | rows = timesteps, cols = panel groups |
    | `"groupwise"` | panel groups (stations) | rows = timesteps, cols = components (pollutants) |
    | `"coveragewise"` | coverage rates (interval only) | collapses coverage dimension |

    - **Multivariate panel** groups are essential for seeing the difference
      between `"componentwise"` (pollutants → stations) and `"groupwise"`
      (stations → pollutants).
    - **`panel_group_weight`**: Weight groups differently during `"all"`
      aggregation.
    - **Combine modes** as a list (e.g., `["timewise", "componentwise"]`) to
      aggregate over multiple dimensions at once.
    """)


if __name__ == "__main__":
    app.run()
