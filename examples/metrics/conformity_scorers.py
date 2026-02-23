"""Conformity Scorers for Conformal Prediction.

Demonstrates the four concrete conformity scorers (Residual, AbsoluteResidual,
GammaResidual, AbsoluteGammaResidual) and how they shape prediction intervals
in SplitConformalForecaster.
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
    # Conformity Scorers for Conformal Prediction

    Conformal prediction builds prediction intervals by measuring how
    "non-conforming" each calibration point is. The **conformity scorer**
    determines that measurement. Different scorers produce intervals with
    different shapes and properties:

    | Scorer | Formula | Interval | Best For |
    |--------|---------|----------|----------|
    | `Residual` | $y - \hat{y}$ | Asymmetric | General use |
    | `AbsoluteResidual` | $\|y - \hat{y}\|$ | Symmetric | Balanced errors |
    | `GammaResidual` | $(y - \hat{y}) / (\hat{y} + \varepsilon)$ | Asymmetric, adaptive | Proportional errors |
    | `AbsoluteGammaResidual` | $\|(y - \hat{y}) / (\hat{y} + \varepsilon)\|$ | Symmetric, adaptive | Proportional + balanced |

    ## What You'll Learn

    - How each conformity scorer measures non-conformity
    - Symmetric vs. asymmetric intervals
    - Adaptive (gamma) vs. fixed-width intervals
    - Comparing interval quality with calibration and width metrics
    - Choosing the right scorer for your data

    ## Prerequisites

    Familiarity with `SplitConformalForecaster`
    (see `examples/interval/conformal_forecasting.py`).
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_tourism_monthly
    from yohou.interval import SplitConformalForecaster
    from yohou.metrics import EmpiricalCoverage, IntervalScore, MeanIntervalWidth
    from yohou.metrics.conformity import (
        AbsoluteGammaResidual,
        AbsoluteResidual,
        GammaResidual,
        Residual,
    )
    from yohou.plotting import plot_forecast
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        AbsoluteGammaResidual,
        AbsoluteResidual,
        EmpiricalCoverage,
        GammaResidual,
        IntervalScore,
        LagTransformer,
        MeanIntervalWidth,
        PointReductionForecaster,
        Residual,
        Ridge,
        SplitConformalForecaster,
        fetch_tourism_monthly,
        pl,
        plot_forecast,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data
    """)
    return


@app.cell
def _(fetch_tourism_monthly):
    df = fetch_tourism_monthly().frame.select("time", "T1__tourists").rename({"T1__tourists": "passengers"})
    split_idx = int(len(df) * 0.85)
    y_train = df.head(split_idx)
    y_test = df.tail(len(df) - split_idx)
    return df, split_idx, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Build a Shared Point Forecaster

    All conformal forecasters share the same underlying point forecaster.
    This isolates the effect of the conformity scorer on interval quality.
    """)
    return


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
    return


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
    y_pred_residual = fc_residual.predict_interval(
        forecasting_horizon=_horizon, coverage_rates=[0.9]
    )

    plot_forecast(
        y_test,
        y_pred_residual,
        y_train=y_train,
        n_history=36,
        coverage_rates=[0.9],
        title="Residual: Asymmetric Intervals",
    )
    return fc_residual, y_pred_residual


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. AbsoluteResidual: `|y - y_hat|`

    Takes the absolute value, producing **symmetric** intervals centred on the
    point prediction. This is the most common choice when you have no reason
    to expect directional bias.
    """)
    return


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
    y_pred_abs = fc_abs.predict_interval(
        forecasting_horizon=_horizon, coverage_rates=[0.9]
    )

    plot_forecast(
        y_test,
        y_pred_abs,
        y_train=y_train,
        n_history=36,
        coverage_rates=[0.9],
        title="AbsoluteResidual: Symmetric Intervals",
    )
    return fc_abs, y_pred_abs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. GammaResidual: `(y - y_hat) / (y_hat + epsilon)`

    Normalises residuals by the prediction magnitude, making intervals
    **adaptive** (wider when predictions are large, narrower when small).
    This is especially useful for data with **multiplicative** noise patterns
    like the Air Passengers series.
    """)
    return


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
    y_pred_gamma = fc_gamma.predict_interval(
        forecasting_horizon=_horizon, coverage_rates=[0.9]
    )

    plot_forecast(
        y_test,
        y_pred_gamma,
        y_train=y_train,
        n_history=36,
        coverage_rates=[0.9],
        title="GammaResidual: Adaptive Asymmetric Intervals",
    )
    return fc_gamma, y_pred_gamma


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. AbsoluteGammaResidual: `|(y - y_hat) / (y_hat + epsilon)|`

    Combines magnitude-adaptive scaling with symmetric intervals.
    """)
    return


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
    y_pred_abs_gamma = fc_abs_gamma.predict_interval(
        forecasting_horizon=_horizon, coverage_rates=[0.9]
    )

    plot_forecast(
        y_test,
        y_pred_abs_gamma,
        y_train=y_train,
        n_history=36,
        coverage_rates=[0.9],
        title="AbsoluteGammaResidual: Adaptive Symmetric Intervals",
    )
    return fc_abs_gamma, y_pred_abs_gamma


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Quantitative Comparison

    Compare all four scorers on three key metrics:
    - **Empirical Coverage**: Fraction of test points inside the interval (target: 0.90)
    - **Interval Score**: Proper scoring rule penalising width + miscoverage
    - **Mean Interval Width**: Average width (narrower is better, given adequate coverage)
    """)
    return


@app.cell
def _(
    EmpiricalCoverage,
    IntervalScore,
    MeanIntervalWidth,
    mo,
    pl,
    y_pred_abs,
    y_pred_abs_gamma,
    y_pred_gamma,
    y_pred_residual,
    y_test,
    y_train,
):
    _cov = EmpiricalCoverage()
    _is = IntervalScore()
    _miw = MeanIntervalWidth()
    _coverage_rates = [0.9]

    _cov.fit(y_train)
    _is.fit(y_train)
    _miw.fit(y_train)

    _results = []
    for _name, _pred in [
        ("Residual", y_pred_residual),
        ("AbsoluteResidual", y_pred_abs),
        ("GammaResidual", y_pred_gamma),
        ("AbsoluteGammaResidual", y_pred_abs_gamma),
    ]:
        _c = _cov.score(y_test, _pred, coverage_rates=_coverage_rates)
        _s = _is.score(y_test, _pred, coverage_rates=_coverage_rates)
        _w = _miw.score(y_test, _pred, coverage_rates=_coverage_rates)
        _results.append({
            "Scorer": _name,
            "Coverage": round(float(_c), 3),
            "Interval Score": round(float(_s), 1),
            "Mean Width": round(float(_w), 1),
        })

    comparison_df = pl.DataFrame(_results)
    mo.ui.table(comparison_df)
    return (comparison_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. When to Use Which Scorer

    - **Residual**: Choose when errors may be directionally biased (under- vs. over-prediction) and you want intervals that reflect that asymmetry
    - **AbsoluteResidual**: Default choice for symmetric errors; simplest and most interpretable
    - **GammaResidual**: Choose for multiplicative data (e.g., sales, passenger counts) where absolute error scales with the level
    - **AbsoluteGammaResidual**: Combines adaptiveness with symmetry; good default for multiplicative data with balanced errors

    For this Air Passengers series (strong multiplicative trend), the
    **Gamma** variants should produce better-calibrated intervals because
    interval width adapts to the prediction magnitude.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Four concrete conformity scorers** control interval shape in `SplitConformalForecaster`
    - **Symmetric** scorers (AbsoluteResidual, AbsoluteGammaResidual) centre intervals on the point prediction
    - **Asymmetric** scorers (Residual, GammaResidual) allow unequal lower/upper widths
    - **Gamma** variants normalise by the prediction magnitude, producing **adaptive** interval widths
    - The `epsilon` parameter in Gamma scorers prevents division by zero
    - Always evaluate with `EmpiricalCoverage`, `IntervalScore`, and `MeanIntervalWidth`
    - For multiplicative data, Gamma scorers typically produce better-calibrated intervals

    ## Next Steps

    - **Distance-based weighting**: See `examples/interval/distance_similarity.py` for adaptive conformal intervals using `DistanceSimilarity`
    - **Conformal prediction basics**: See `examples/interval/conformal_forecasting.py` for the full conformal workflow
    - **Interval metrics**: See `examples/metrics/interval_metrics.py` for deep-dive into interval evaluation
    """)
    return


if __name__ == "__main__":
    app.run()
