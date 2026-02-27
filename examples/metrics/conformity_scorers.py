# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Conformity Scorers",
    "description": "Compare Residual, AbsoluteResidual, GammaResidual, and AbsoluteGammaResidual conformity scorers for calibrating conformal prediction intervals.",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Conformity Scorers for Conformal Prediction

    Conformal prediction builds prediction intervals by measuring how
    "non-conforming" each calibration point is. The **conformity scorer**
    determines that measurement. Different scorers produce intervals with
    different shapes and properties:

    | Scorer | Formula | Interval | Best For |
    |--------|---------|----------|----------|
    | [`Residual`](/pages/api/generated/yohou.metrics.conformity.Residual/) | $y - \hat{y}$ | Asymmetric | General use |
    | [`AbsoluteResidual`](/pages/api/generated/yohou.metrics.conformity.AbsoluteResidual/) | $\|y - \hat{y}\|$ | Symmetric | Balanced errors |
    | [`GammaResidual`](/pages/api/generated/yohou.metrics.conformity.GammaResidual/) | $(y - \hat{y}) / (\hat{y} + \varepsilon)$ | Asymmetric, adaptive | Proportional errors |
    | [`AbsoluteGammaResidual`](/pages/api/generated/yohou.metrics.conformity.AbsoluteGammaResidual/) | $\|(y - \hat{y}) / (\hat{y} + \varepsilon)\|$ | Symmetric, adaptive | Proportional + balanced |

    ## What You'll Learn

    - How each conformity scorer measures non-conformity
    - Symmetric vs. asymmetric intervals
    - Adaptive (gamma) vs. fixed-width intervals
    - Comparing interval quality with calibration and width metrics
    - Choosing the right scorer for your data

    ## Prerequisites

    Familiarity with [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/).
    """)


@app.cell(hide_code=True)
def _():
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    from yohou.datasets import fetch_tourism_monthly
    from yohou.interval import SplitConformalForecaster
    from yohou.metrics import IntervalScore
    from yohou.metrics.conformity import (
        AbsoluteGammaResidual,
        AbsoluteResidual,
        GammaResidual,
        Residual,
    )
    from yohou.plotting import plot_forecast, plot_score_time_series, plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        AbsoluteGammaResidual,
        AbsoluteResidual,
        GammaResidual,
        IntervalScore,
        LagTransformer,
        PointReductionForecaster,
        Residual,
        Ridge,
        SplitConformalForecaster,
        fetch_tourism_monthly,
        plot_forecast,
        plot_score_time_series,
        plot_time_series,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We load the Tourism Monthly dataset and split it into training, calibration,
    and test sets. All four conformity scorers will share the same underlying
    [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) so that differences in interval shape come
    solely from the choice of scorer.
    """)


@app.cell
def _(fetch_tourism_monthly, train_test_split):
    df = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    y_train, _y_rest = train_test_split(df, test_size=0.15, shuffle=False)
    # Cap test size at 24 so it fits within calibration_size
    y_test = _y_rest.head(24)
    return df, y_test, y_train


@app.cell
def _(df, plot_time_series):
    plot_time_series(df, title="Tourism Monthly")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Build a Shared Point Forecaster

    All conformal forecasters share the same underlying point forecaster.
    This isolates the effect of the conformity scorer on interval quality.
    """)


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge):
    base_forecaster = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 6, 12]),
    )
    return (base_forecaster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Residual: `y - y_hat`

    The simplest scorer. Computes raw signed residuals, producing
    **asymmetric** intervals (lower and upper bounds can differ in distance
    from the point prediction).
    """)


@app.cell
def _(
    Residual,
    SplitConformalForecaster,
    base_forecaster,
    plot_forecast,
    y_test,
    y_train,
):
    fc_residual = SplitConformalForecaster(
        point_forecaster=base_forecaster,
        calibration_size=24,
        conformity_scorer=Residual(),
    )
    _horizon = len(y_test)
    fc_residual.fit(y_train, forecasting_horizon=_horizon)
    y_pred_residual = fc_residual.predict_interval(forecasting_horizon=_horizon, coverage_rates=[0.9])
    _y_point = fc_residual.predict(forecasting_horizon=_horizon)
    y_pred_residual = y_pred_residual.hstack(_y_point.drop("time", "observed_time"))

    plot_forecast(
        y_test,
        y_pred_residual,
        y_train=y_train,
        n_history=36,
        coverage_rates=[0.9],
        title="Residual: Asymmetric Intervals",
    )
    return (y_pred_residual,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. AbsoluteResidual: `|y - y_hat|`

    Takes the absolute value, producing **symmetric** intervals centred on the
    point prediction. This is the most common choice when you have no reason
    to expect directional bias.
    """)


@app.cell
def _(
    AbsoluteResidual,
    SplitConformalForecaster,
    base_forecaster,
    plot_forecast,
    y_test,
    y_train,
):
    fc_abs = SplitConformalForecaster(
        point_forecaster=base_forecaster,
        calibration_size=24,
        conformity_scorer=AbsoluteResidual(),
    )
    _horizon = len(y_test)
    fc_abs.fit(y_train, forecasting_horizon=_horizon)
    y_pred_abs = fc_abs.predict_interval(forecasting_horizon=_horizon, coverage_rates=[0.9])
    _y_point = fc_abs.predict(forecasting_horizon=_horizon)
    y_pred_abs = y_pred_abs.hstack(_y_point.drop("time", "observed_time"))

    plot_forecast(
        y_test,
        y_pred_abs,
        y_train=y_train,
        n_history=36,
        coverage_rates=[0.9],
        title="AbsoluteResidual: Symmetric Intervals",
    )
    return (y_pred_abs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. GammaResidual: `(y - y_hat) / (y_hat + epsilon)`

    Normalises residuals by the prediction magnitude, making intervals
    **adaptive** (wider when predictions are large, narrower when small).
    This is especially useful for data with **multiplicative** noise patterns
    like the Tourism Monthly series.
    """)


@app.cell
def _(
    GammaResidual,
    SplitConformalForecaster,
    base_forecaster,
    plot_forecast,
    y_test,
    y_train,
):
    fc_gamma = SplitConformalForecaster(
        point_forecaster=base_forecaster,
        calibration_size=24,
        conformity_scorer=GammaResidual(epsilon=1e-8),
    )
    _horizon = len(y_test)
    fc_gamma.fit(y_train, forecasting_horizon=_horizon)
    y_pred_gamma = fc_gamma.predict_interval(forecasting_horizon=_horizon, coverage_rates=[0.9])
    _y_point = fc_gamma.predict(forecasting_horizon=_horizon)
    y_pred_gamma = y_pred_gamma.hstack(_y_point.drop("time", "observed_time"))

    plot_forecast(
        y_test,
        y_pred_gamma,
        y_train=y_train,
        n_history=36,
        coverage_rates=[0.9],
        title="GammaResidual: Adaptive Asymmetric Intervals",
    )
    return (y_pred_gamma,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. AbsoluteGammaResidual: `|(y - y_hat) / (y_hat + epsilon)|`

    Combines magnitude-adaptive scaling with symmetric intervals.
    """)


@app.cell
def _(
    AbsoluteGammaResidual,
    SplitConformalForecaster,
    base_forecaster,
    plot_forecast,
    y_test,
    y_train,
):
    fc_abs_gamma = SplitConformalForecaster(
        point_forecaster=base_forecaster,
        calibration_size=24,
        conformity_scorer=AbsoluteGammaResidual(epsilon=1e-8),
    )
    _horizon = len(y_test)
    fc_abs_gamma.fit(y_train, forecasting_horizon=_horizon)
    y_pred_abs_gamma = fc_abs_gamma.predict_interval(forecasting_horizon=_horizon, coverage_rates=[0.9])
    _y_point = fc_abs_gamma.predict(forecasting_horizon=_horizon)
    y_pred_abs_gamma = y_pred_abs_gamma.hstack(_y_point.drop("time", "observed_time"))

    plot_forecast(
        y_test,
        y_pred_abs_gamma,
        y_train=y_train,
        n_history=36,
        coverage_rates=[0.9],
        title="AbsoluteGammaResidual: Adaptive Symmetric Intervals",
    )
    return (y_pred_abs_gamma,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Quantitative Comparison

    Compare all four scorers visually with [`plot_score_time_series`](/pages/api/generated/yohou.plotting.evaluation.plot_score_time_series/).
    Each line shows the per-timestep interval score for one conformity
    scorer, making it easy to spot where specific scorers struggle.
    """)


@app.cell
def _(
    IntervalScore,
    mo,
    plot_score_time_series,
    y_pred_abs,
    y_pred_abs_gamma,
    y_pred_gamma,
    y_pred_residual,
    y_test,
):
    _preds = {
        "Residual": y_pred_residual,
        "AbsoluteResidual": y_pred_abs,
        "GammaResidual": y_pred_gamma,
        "AbsoluteGammaResidual": y_pred_abs_gamma,
    }

    mo.vstack([
        plot_score_time_series(
            IntervalScore(coverage_rates=[0.9]),
            y_test,
            _preds,
            title="Interval Score per Timestep",
        ),
    ])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. When to Use Which Scorer

    - **Residual**: Choose when errors may be directionally biased (under- vs. over-prediction) and you want intervals that reflect that asymmetry
    - **AbsoluteResidual**: Default choice for symmetric errors; simplest and most interpretable
    - **GammaResidual**: Choose for multiplicative data (e.g., sales, passenger counts) where absolute error scales with the level
    - **AbsoluteGammaResidual**: Combines adaptiveness with symmetry; good default for multiplicative data with balanced errors

    For this Tourism Monthly series (strong multiplicative trend), the
    **Gamma** variants should produce better-calibrated intervals because
    interval width adapts to the prediction magnitude.
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Four concrete conformity scorers** control interval shape in [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/)
    - **Symmetric** scorers (AbsoluteResidual, AbsoluteGammaResidual) centre intervals on the point prediction
    - **Asymmetric** scorers (Residual, GammaResidual) allow unequal lower/upper widths
    - **Gamma** variants normalise by the prediction magnitude, producing **adaptive** interval widths
    - The `epsilon` parameter in Gamma scorers prevents division by zero
    - Always evaluate with [`EmpiricalCoverage`](/pages/api/generated/yohou.metrics.interval.EmpiricalCoverage/), [`IntervalScore`](/pages/api/generated/yohou.metrics.interval.IntervalScore/), and [`MeanIntervalWidth`](/pages/api/generated/yohou.metrics.interval.MeanIntervalWidth/)
    - For multiplicative data, Gamma scorers typically produce better-calibrated intervals

    ## Next Steps

    - **Distance-based weighting**: See [`examples/interval/distance_similarity.py`](/examples/interval/distance_similarity/) for adaptive conformal intervals using [`DistanceSimilarity`](/pages/api/generated/yohou.interval.similarity.DistanceSimilarity/)
    - **Interval metrics**: See [`examples/metrics/interval_metrics.py`](/examples/metrics/interval_metrics/) for deep-dive into interval evaluation
    """)


if __name__ == "__main__":
    app.run()
