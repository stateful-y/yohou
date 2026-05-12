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
    "title": "Conformal Prediction Intervals",
    "description": "Build distribution-free prediction intervals with SplitConformalForecaster using calibration holdouts and configurable conformity scoring functions.",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Conformal Prediction Intervals

    [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/) wraps **any** point forecaster and produces
    **distribution-free** prediction intervals with finite-sample coverage
    guarantees, using a held-out calibration set.

    ## What You'll Learn

    - Building conformal intervals around point forecasters
    - Choosing conformity scorers ([`Residual`](/pages/api/generated/yohou.metrics.conformity.Residual/), [`AbsoluteResidual`](/pages/api/generated/yohou.metrics.conformity.AbsoluteResidual/), [`GammaResidual`](/pages/api/generated/yohou.metrics.conformity.GammaResidual/))
    - Evaluating interval quality with [`EmpiricalCoverage`](/pages/api/generated/yohou.metrics.interval.EmpiricalCoverage/) and [`IntervalScore`](/pages/api/generated/yohou.metrics.interval.IntervalScore/)
    - Visualizing prediction intervals with [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/)

    ## Prerequisites

    Basic understanding of prediction intervals and [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/).
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

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
    from yohou.model_selection import train_test_split
    from yohou.plotting import (
        plot_forecast,
        plot_score_per_vintage,
        plot_score_time_series,
    )
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
        deepcopy,
        fetch_tourism_monthly,
        pl,
        plot_forecast,
        plot_score_per_vintage,
        plot_score_time_series,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We load the Monthly Tourism dataset and split it into training and test sets for calibrating and evaluating conformal intervals.
    """)


@app.cell
def _(fetch_tourism_monthly, train_test_split):
    y = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})

    # Need enough calibration data: use 80/20 split
    y_train, y_test = train_test_split(y, test_size=0.2)
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
    _y_point = conformal.predict(forecasting_horizon=forecasting_horizon)
    y_pred_int = y_pred_int.hstack(_y_point.drop("time", "vintage_time"))

    print(f"Prediction columns: {y_pred_int.columns}")
    y_pred_int.head()
    return coverage_rates, y_pred_int


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) renders the prediction intervals as shaded bands around
    the point forecast, with separate bands for each coverage rate.
    """)


@app.cell
def _(coverage_rates, plot_forecast, y_pred_int, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_int,
        y_train=y_train,
        coverage_rates=coverage_rates,
        title="Split Conformal Prediction Intervals",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Evaluating Interval Quality

    - [`EmpiricalCoverage`](/pages/api/generated/yohou.metrics.interval.EmpiricalCoverage/): Checks if actual coverage matches nominal (e.g., 95%)
    - [`IntervalScore`](/pages/api/generated/yohou.metrics.interval.IntervalScore/): Penalizes wide intervals and miscoverage
    - [`MeanIntervalWidth`](/pages/api/generated/yohou.metrics.interval.MeanIntervalWidth/): Average interval width (narrower = better, given coverage)
    """)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Conformity Scorers

    The conformity scorer determines how prediction errors are measured:

    - `Residual()`: Raw residuals ($y - \hat{y}$): symmetric intervals
    - `AbsoluteResidual()`: $|y - \hat{y}|$: default, symmetric
    - `GammaResidual()`: $|y - \hat{y}| / \hat{y}$: scale-adaptive, wider intervals where predictions are larger
    """)


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
        print(f"{_name:>20s}  coverage_rates={_coverage}  width={_avg_width}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Using SeasonalNaive as Base

    Conformal intervals work with **any** point forecaster, including [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/).
    """)


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
    _y_point = conformal_naive.predict(forecasting_horizon=forecasting_horizon)
    y_pred_naive_int = y_pred_naive_int.hstack(_y_point.drop("time", "vintage_time"))
    y_pred_naive_int.head()
    return (y_pred_naive_int,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) shows how [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/)-based conformal intervals
    compare visually to the Ridge-based intervals above.
    """)


@app.cell
def _(plot_forecast, y_pred_naive_int, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_naive_int,
        y_train=y_train,
        coverage_rates=[0.9],
        title="Conformal Intervals on SeasonalNaive",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/) adds prediction intervals to any point forecaster
    - Coverage guarantees are distribution-free (no normality assumed)
    - `calibration_size` trades off calibration accuracy vs. training data
    - Conformity scorers control interval shape: symmetric vs. scale-adaptive
    - Evaluate with [`EmpiricalCoverage`](/pages/api/generated/yohou.metrics.interval.EmpiricalCoverage/), [`IntervalScore`](/pages/api/generated/yohou.metrics.interval.IntervalScore/), and [`MeanIntervalWidth`](/pages/api/generated/yohou.metrics.interval.MeanIntervalWidth/)
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-vintage Scoring

    The `observe_predict_interval` method with `stride=1` produces one
    interval forecast per observation point, creating multiple *vintages*.
    Each vintage represents a different forecast origin, so you can analyse
    how interval quality evolves as the model absorbs more data.
    """)


@app.cell
def _(conformal, coverage_rates, deepcopy, forecasting_horizon, y_test):
    _vintage_model = deepcopy(conformal)
    y_pred_vintages = _vintage_model.observe_predict_interval(
        y=y_test,
        stride=1,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage_rates,
    )
    print(f"Vintages: {y_pred_vintages['vintage_time'].n_unique()}")
    y_pred_vintages.head(10)
    return (y_pred_vintages,)


@app.cell
def _(IntervalScore, y_train):
    vintage_scorer = IntervalScore()
    vintage_scorer.fit(y_train)
    return (vintage_scorer,)


@app.cell
def _(vintage_scorer, plot_score_time_series, y_pred_vintages, y_test):
    plot_score_time_series(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="Interval Score over Time",
        y_label="Interval Score",
        height=380,
    )


@app.cell
def _(vintage_scorer, plot_score_per_vintage, y_pred_vintages, y_test):
    plot_score_per_vintage(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="Interval Score per Vintage",
        y_label="Interval Score",
        height=380,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Interval reduction**: See [`interval_reduction.py`](/examples/interval/interval_reduction/) for quantile regression intervals
    - **Scoring**: See [Metrics](/examples/#metrics) for comprehensive interval evaluation
    - **Calibration plots**: See [Plotting](/examples/#plotting) for [`plot_calibration`](/pages/api/generated/yohou.plotting.evaluation.plot_calibration/)
    """)


if __name__ == "__main__":
    app.run()
