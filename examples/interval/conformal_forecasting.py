# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "plotly",
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Conformal Prediction Intervals.

Demonstrates SplitConformalForecaster for distribution-free prediction intervals.
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
    # Conformal Prediction Intervals

    `SplitConformalForecaster` wraps **any** point forecaster and produces
    **distribution-free** prediction intervals with finite-sample coverage
    guarantees, using a held-out calibration set.

    ## What You'll Learn

    - Building conformal intervals around point forecasters
    - Choosing conformity scorers (`Residual`, `AbsoluteResidual`, `GammaResidual`)
    - Evaluating interval quality with `EmpiricalCoverage` and `IntervalScore`
    - Visualizing prediction intervals with `plot_forecast`

    ## Prerequisites

    Basic understanding of prediction intervals and `PointReductionForecaster`.
    """)
    return

@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_tourism_monthly
    from yohou.interval import SplitConformalForecaster
    from yohou.metrics import (
        AbsoluteResidual,
        EmpiricalCoverage,
        GammaResidual,
        IntervalScore,
        MeanIntervalWidth,
        Residual,
    )
    from yohou.plotting import plot_forecast
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer

    return (
        AbsoluteResidual,
        EmpiricalCoverage,
        GammaResidual,
        IntervalScore,
        LagTransformer,
        MeanIntervalWidth,
        PointReductionForecaster,
        Residual,
        Ridge,
        SeasonalNaive,
        SplitConformalForecaster,
        fetch_tourism_monthly,
        pl,
        plot_forecast,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We load the Monthly Tourism dataset and split it into training and test sets for calibrating and evaluating conformal intervals.
    """)
    return

@app.cell
def _(fetch_tourism_monthly, pl):
    y = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "passengers"})

    # Need enough calibration data: use 80/20 split
    split_idx = int(len(y) * 0.8)
    y_train = y.head(split_idx)
    y_test = y.tail(len(y) - split_idx)
    forecasting_horizon = min(len(y_test), 12)  # Limit horizon for calibration

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    return forecasting_horizon, y_test, y_train

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. SplitConformalForecaster

    Wraps a point forecaster and uses a calibration set to compute conformal scores.
    `calibration_size` controls how many of the most recent training observations
    are used for calibration.
    """)
    return

@app.cell
def _(
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SplitConformalForecaster,
    forecasting_horizon,
    y_train,
):
    conformal = SplitConformalForecaster(
        point_forecaster=PointReductionForecaster(
            estimator=Ridge(),
            feature_transformer=LagTransformer(lag=list(range(1, 13))),
        ),
        calibration_size=30,
    )

    conformal.fit(y_train, forecasting_horizon=forecasting_horizon)

    coverage_rates = [0.8, 0.9]
    y_pred_int = conformal.predict_interval(
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage_rates,
    )

    print(f"Prediction columns: {y_pred_int.columns}")
    y_pred_int.head()
    return coverage_rates, y_pred_int

@app.cell
def _(coverage_rates, plot_forecast, y_pred_int, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_int,
        y_train=y_train,
        coverage_rates=coverage_rates,
        title="Split Conformal Prediction Intervals",
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Evaluating Interval Quality

    - `EmpiricalCoverage`: Checks if actual coverage matches nominal (e.g., 95%)
    - `IntervalScore`: Penalizes wide intervals and miscoverage
    - `MeanIntervalWidth`: Average interval width (narrower = better, given coverage)
    """)
    return

@app.cell
def _(
    EmpiricalCoverage,
    IntervalScore,
    MeanIntervalWidth,
    coverage_rates,
    y_pred_int,
    y_test,
    y_train,
):
    for _scorer_cls in [EmpiricalCoverage, IntervalScore, MeanIntervalWidth]:
        _scorer = _scorer_cls(coverage_rates=coverage_rates)
        _scorer.fit(y_train)
        _score = _scorer.score(y_test, y_pred_int)
        print(f"{_scorer_cls.__name__}: {_score}")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Conformity Scorers

    The conformity scorer determines how prediction errors are measured:

    - `Residual()`: Raw residuals ($y - \hat{y}$): symmetric intervals
    - `AbsoluteResidual()`: $|y - \hat{y}|$: default, symmetric
    - `GammaResidual()`: $|y - \hat{y}| / \hat{y}$: scale-adaptive, wider intervals where predictions are larger
    """)
    return

@app.cell
def _(
    AbsoluteResidual,
    EmpiricalCoverage,
    GammaResidual,
    LagTransformer,
    MeanIntervalWidth,
    PointReductionForecaster,
    Residual,
    Ridge,
    SplitConformalForecaster,
    forecasting_horizon,
    y_test,
    y_train,
):
    scorer_results = {}
    for _name, _scorer_cls in [
        ("Residual", Residual),
        ("AbsoluteResidual", AbsoluteResidual),
        ("GammaResidual", GammaResidual),
    ]:
        _cf = SplitConformalForecaster(
            point_forecaster=PointReductionForecaster(
                estimator=Ridge(),
                feature_transformer=LagTransformer(lag=list(range(1, 13))),
            ),
            calibration_size=30,
            conformity_scorer=_scorer_cls(),
        )
        _cf.fit(y_train, forecasting_horizon=forecasting_horizon)
        _pred = _cf.predict_interval(
            forecasting_horizon=forecasting_horizon,
            coverage_rates=[0.9],
        )

        _cov = EmpiricalCoverage(coverage_rates=[0.9])
        _cov.fit(y_train)
        _coverage = _cov.score(y_test, _pred)

        _width = MeanIntervalWidth(coverage_rates=[0.9])
        _width.fit(y_train)
        _avg_width = _width.score(y_test, _pred)

        scorer_results[_name] = {"pred": _pred, "coverage": _coverage, "width": _avg_width}
        print(f"{_name:>20s}  coverage={_coverage}  width={_avg_width}")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Using SeasonalNaive as Base

    Conformal intervals work with **any** point forecaster, including `SeasonalNaive`.
    """)
    return

@app.cell
def _(SeasonalNaive, SplitConformalForecaster, forecasting_horizon, y_train):
    conformal_naive = SplitConformalForecaster(
        point_forecaster=SeasonalNaive(seasonality=12),
        calibration_size=24,
    )
    conformal_naive.fit(y_train, forecasting_horizon=forecasting_horizon)

    y_pred_naive_int = conformal_naive.predict_interval(
        forecasting_horizon=forecasting_horizon,
        coverage_rates=[0.9],
    )
    y_pred_naive_int.head()
    return (y_pred_naive_int,)

@app.cell
def _(plot_forecast, y_pred_naive_int, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_naive_int,
        y_train=y_train,
        coverage_rates=[0.9],
        title="Conformal Intervals on SeasonalNaive",
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - `SplitConformalForecaster` adds prediction intervals to any point forecaster
    - Coverage guarantees are distribution-free (no normality assumed)
    - `calibration_size` trades off calibration accuracy vs. training data
    - Conformity scorers control interval shape: symmetric vs. scale-adaptive
    - Evaluate with `EmpiricalCoverage`, `IntervalScore`, and `MeanIntervalWidth`
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Interval reduction**: See `interval_reduction.py` for quantile regression intervals
    - **Scoring**: See `metrics/` for comprehensive interval evaluation
    - **Calibration plots**: See `plotting/` for `plot_calibration`
    """)
    return

if __name__ == "__main__":
    app.run()
