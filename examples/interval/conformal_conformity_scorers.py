"""Conformal Prediction with Different Conformity Scorers.

Demonstrates how different conformity scorers affect prediction interval
calibration in SplitConformalForecaster.
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
    # Conformal Prediction with Conformity Scorers

    `SplitConformalForecaster` uses a conformity scorer to compute
    calibration residuals. Different scorers produce different interval
    shapes and widths.

    ## What You'll Learn

    - `Residual` vs `AbsoluteResidual` vs `GammaResidual` vs `AbsoluteGammaResidual`
    - How conformity scorer choice affects interval symmetry and width
    - Coverage and width comparison across scorers
    - Using `DistanceSimilarity` with different conformity scorers
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_tourism_monthly
    from yohou.interval import DistanceSimilarity, SplitConformalForecaster
    from yohou.metrics import AbsoluteResidual, EmpiricalCoverage, GammaResidual, MeanIntervalWidth, Residual
    from yohou.plotting import plot_forecast
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        AbsoluteResidual,
        DistanceSimilarity,
        EmpiricalCoverage,
        GammaResidual,
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
def _(fetch_tourism_monthly, mo):
    air = fetch_tourism_monthly().frame.select("time", "T1__tourists").rename({"T1__tourists": "Passengers"})
    _split = int(len(air) * 0.8)
    y_train = air.head(_split)
    y_test = air.tail(len(air) - _split)
    horizon = len(y_test)
    coverage_rates = [0.9]
    mo.md(f"**Monthly Tourism**: Train={len(y_train)}, Test={len(y_test)}")
    return air, coverage_rates, horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Residual Scorer (Default)

    `Residual()` uses raw residuals: `y_true - y_pred`.
    Produces asymmetric intervals reflecting the sign of typical errors.
    """)
    return


@app.cell
def _(
    LagTransformer,
    PointReductionForecaster,
    Residual,
    Ridge,
    SplitConformalForecaster,
    coverage_rates,
    horizon,
    plot_forecast,
    y_test,
    y_train,
):
    _base = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 12]),
    )
    fc_residual = SplitConformalForecaster(
        point_forecaster=_base,
        calibration_size=horizon + 5,
        conformity_scorer=Residual(),
    )
    fc_residual.fit(y_train, forecasting_horizon=horizon, coverage_rates=coverage_rates)
    y_pred_resid = fc_residual.predict_interval(
        forecasting_horizon=horizon, coverage_rates=coverage_rates
    )
    plot_forecast(
        y_test, y_pred_resid, y_train=y_train, n_history=24,
        coverage_rates=coverage_rates, title="Residual Scorer",
    )
    return fc_residual, y_pred_resid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. AbsoluteResidual Scorer

    `AbsoluteResidual()` uses `|y_true - y_pred|`.
    Produces symmetric intervals centred around the point prediction.
    """)
    return


@app.cell
def _(
    AbsoluteResidual,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SplitConformalForecaster,
    coverage_rates,
    horizon,
    plot_forecast,
    y_test,
    y_train,
):
    _base_abs = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 12]),
    )
    fc_abs = SplitConformalForecaster(
        point_forecaster=_base_abs,
        calibration_size=horizon + 5,
        conformity_scorer=AbsoluteResidual(),
    )
    fc_abs.fit(y_train, forecasting_horizon=horizon, coverage_rates=coverage_rates)
    y_pred_abs = fc_abs.predict_interval(
        forecasting_horizon=horizon, coverage_rates=coverage_rates
    )
    plot_forecast(
        y_test, y_pred_abs, y_train=y_train, n_history=24,
        coverage_rates=coverage_rates, title="AbsoluteResidual Scorer",
    )
    return fc_abs, y_pred_abs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. GammaResidual Scorer

    `GammaResidual()` normalises residuals by the prediction magnitude:
    `(y_true - y_pred) / y_pred`. Produces prediction-proportional
    intervals (wider when predictions are larger).
    """)
    return


@app.cell
def _(
    GammaResidual,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SplitConformalForecaster,
    coverage_rates,
    horizon,
    plot_forecast,
    y_test,
    y_train,
):
    _base_gamma = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 12]),
    )
    fc_gamma = SplitConformalForecaster(
        point_forecaster=_base_gamma,
        calibration_size=horizon + 5,
        conformity_scorer=GammaResidual(),
    )
    fc_gamma.fit(y_train, forecasting_horizon=horizon, coverage_rates=coverage_rates)
    y_pred_gamma = fc_gamma.predict_interval(
        forecasting_horizon=horizon, coverage_rates=coverage_rates
    )
    plot_forecast(
        y_test, y_pred_gamma, y_train=y_train, n_history=24,
        coverage_rates=coverage_rates, title="GammaResidual Scorer",
    )
    return fc_gamma, y_pred_gamma


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Compare Coverage and Width
    """)
    return


@app.cell
def _(EmpiricalCoverage, MeanIntervalWidth, mo, pl, y_pred_abs, y_pred_gamma, y_pred_resid, y_test, y_train):
    _cov = EmpiricalCoverage()
    _width = MeanIntervalWidth()

    _cov.fit(y_train)
    _width.fit(y_train)

    _scorers_map = {
        "Residual": y_pred_resid,
        "AbsoluteResidual": y_pred_abs,
        "GammaResidual": y_pred_gamma,
    }

    _rows = []
    for _name, _pred in _scorers_map.items():
        _c = float(_cov.score(y_test, _pred))
        _w = float(_width.score(y_test, _pred))
        _rows.append({
            "Scorer": _name,
            "Empirical Coverage": round(_c, 3),
            "Mean Interval Width": round(_w, 1),
        })

    mo.ui.table(pl.DataFrame(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. With DistanceSimilarity

    `DistanceSimilarity` reweights calibration residuals based on
    similarity to the current observation. Combined with different
    conformity scorers, this produces adaptive intervals.
    """)
    return


@app.cell
def _(
    AbsoluteResidual,
    DistanceSimilarity,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SplitConformalForecaster,
    coverage_rates,
    horizon,
    mo,
    plot_forecast,
    y_test,
    y_train,
):
    _base_sim = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 12]),
    )
    fc_sim = SplitConformalForecaster(
        point_forecaster=_base_sim,
        calibration_size=horizon + 5,
        conformity_scorer=AbsoluteResidual(),
        similarity=DistanceSimilarity(metric="euclidean"),
    )
    fc_sim.fit(y_train, forecasting_horizon=horizon, coverage_rates=coverage_rates)
    _y_pred_sim = fc_sim.predict_interval(
        forecasting_horizon=horizon, coverage_rates=coverage_rates
    )

    plot_forecast(
        y_test, _y_pred_sim, y_train=y_train, n_history=24,
        coverage_rates=coverage_rates,
        title="AbsoluteResidual + DistanceSimilarity",
    )
    return (fc_sim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    | Scorer | Residual | Interval Shape |
    |--------|----------|---------------|
    | `Residual` | `y - y_pred` | Asymmetric (reflects bias direction) |
    | `AbsoluteResidual` | `|y - y_pred|` | Symmetric around prediction |
    | `GammaResidual` | `(y - y_pred) / y_pred` | Proportional to magnitude |
    | `AbsoluteGammaResidual` | `|y - y_pred| / y_pred` | Symmetric + proportional |

    - **All achieve valid coverage** when calibration set is large enough
    - **GammaResidual** excels when variance scales with level (e.g., growing series)
    - **DistanceSimilarity** makes intervals adaptive to current conditions
    - Choose scorer based on data characteristics and interval shape requirements

    ## Next Steps

    - **Conformity scorers (metrics)**: See `examples/metrics/conformity_scorers.py`
    - **Conformal forecasting**: See `examples/interval/conformal_forecasting.py`
    - **Distance similarity**: See `examples/interval/distance_similarity.py`
    """)
    return


if __name__ == "__main__":
    app.run()
