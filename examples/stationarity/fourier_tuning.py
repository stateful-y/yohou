# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou",
# ]
# ///
"""Fourier Seasonality Tuning.

Demonstrates FourierSeasonalityForecaster with harmonic selection,
estimator choices, and comparison against PatternSeasonalityForecaster.
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
    # Fourier Seasonality Tuning

    `FourierSeasonalityForecaster` models seasonality using Fourier
    terms (sine/cosine pairs). The number of harmonics controls the
    smoothness of the seasonal pattern.

    ## What You'll Learn

    - Harmonics selection: too few = underfitting, too many = overfitting
    - Comparing Fourier vs Pattern seasonality
    - Different estimators (ElasticNet, Ridge) for Fourier regression
    - Using Fourier inside a DecompositionPipeline
    """)

@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import ElasticNet, Ridge

    from yohou.compose import DecompositionPipeline
    from yohou.datasets import fetch_electricity_demand
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast, plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer
    from yohou.stationarity import (
        FourierSeasonalityForecaster,
        PatternSeasonalityForecaster,
        PolynomialTrendForecaster,
    )

    return (
        DecompositionPipeline,
        ElasticNet,
        FourierSeasonalityForecaster,
        LagTransformer,
        MeanAbsoluteError,
        PatternSeasonalityForecaster,
        PointReductionForecaster,
        PolynomialTrendForecaster,
        Ridge,
        fetch_electricity_demand,
        pl,
        plot_forecast,
        plot_time_series,
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load Data

    Use electricity demand data with daily aggregation for clear
    weekly seasonality.
    """)

@app.cell
def _(fetch_electricity_demand, mo, pl):
    _elec = fetch_electricity_demand().frame
    # Resample to daily for clearer weekly patterns (drop trailing all-null days)
    elec_daily = (
        _elec.group_by_dynamic("time", every="1d")
        .agg(pl.col("vic__demand").mean().alias("demand"))
        .drop_nulls()
    )
    _split = int(len(elec_daily) * 0.85)
    y_train = elec_daily.head(_split).select("time", "demand")
    y_test = elec_daily.tail(len(elec_daily) - _split).select("time", "demand")
    horizon = len(y_test)
    mo.md(
        f"**Daily electricity demand**: {len(elec_daily)} days\n\n"
        f"**Train**: {len(y_train)} days, **Test**: {len(y_test)} days\n\n"
        f"**Weekly seasonality** = 7 days"
    )
    return elec_daily, horizon, y_test, y_train

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Harmonics Comparison

    Vary the number of Fourier harmonics for weekly seasonality.
    With `seasonality=7`, max harmonics = 3 (floor(7/2)).
    """)

@app.cell
def _(FourierSeasonalityForecaster, MeanAbsoluteError, horizon, mo, pl, y_test, y_train):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _rows = []

    for _nh in [[1], [1, 2], [1, 2, 3]]:
        _fc = FourierSeasonalityForecaster(seasonality=7, harmonics=_nh)
        _fc.fit(y_train, forecasting_horizon=horizon)
        _y_pred = _fc.predict(forecasting_horizon=horizon)
        _mae = float(_scorer.score(y_test, _y_pred))
        _rows.append({"Harmonics": str(_nh), "MAE": round(_mae, 2)})

    mo.ui.table(pl.DataFrame(_rows))

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Fourier vs Pattern Seasonality
    """)

@app.cell
def _(
    FourierSeasonalityForecaster,
    MeanAbsoluteError,
    PatternSeasonalityForecaster,
    horizon,
    mo,
    plot_forecast,
    y_test,
    y_train,
):
    _fc_fourier = FourierSeasonalityForecaster(seasonality=7, harmonics=[1, 2, 3])
    _fc_fourier.fit(y_train, forecasting_horizon=horizon)
    _y_pred_fourier = _fc_fourier.predict(forecasting_horizon=horizon)

    _fc_pattern = PatternSeasonalityForecaster(seasonality=7, method="average")
    _fc_pattern.fit(y_train, forecasting_horizon=horizon)
    _y_pred_pattern = _fc_pattern.predict(forecasting_horizon=horizon)

    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _mae_f = float(_scorer.score(y_test, _y_pred_fourier))
    _mae_p = float(_scorer.score(y_test, _y_pred_pattern))

    mo.md(
        f"**Fourier (3 harmonics) MAE**: {_mae_f:.2f}\n\n"
        f"**Pattern (average) MAE**: {_mae_p:.2f}\n\n"
        "Fourier produces smoother seasonal patterns; Pattern is more flexible."
    )

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Different Estimators

    The Fourier features are just sine/cosine columns. Different
    regression estimators can affect regularisation.
    """)

@app.cell
def _(
    ElasticNet,
    FourierSeasonalityForecaster,
    MeanAbsoluteError,
    Ridge,
    horizon,
    mo,
    pl,
    y_test,
    y_train,
):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _rows = []
    for _name, _est in [
        ("ElasticNet (default)", ElasticNet()),
        ("Ridge(alpha=0.1)", Ridge(alpha=0.1)),
        ("Ridge(alpha=10)", Ridge(alpha=10.0)),
    ]:
        _fc = FourierSeasonalityForecaster(
            seasonality=7, harmonics=[1, 2, 3], estimator=_est,
        )
        _fc.fit(y_train, forecasting_horizon=horizon)
        _y_pred = _fc.predict(forecasting_horizon=horizon)
        _mae = float(_scorer.score(y_test, _y_pred))
        _rows.append({"Estimator": _name, "MAE": round(_mae, 2)})

    mo.ui.table(pl.DataFrame(_rows))

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Fourier in a DecompositionPipeline

    Combine trend removal with Fourier seasonality for a structured
    forecasting approach.
    """)

@app.cell
def _(
    DecompositionPipeline,
    FourierSeasonalityForecaster,
    LagTransformer,
    PointReductionForecaster,
    PolynomialTrendForecaster,
    Ridge,
    horizon,
    plot_forecast,
    y_test,
    y_train,
):
    fc_decomp_fourier = DecompositionPipeline(
        forecasters=[
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("season", FourierSeasonalityForecaster(seasonality=7, harmonics=[1, 2, 3])),
            (
                "residual",
                PointReductionForecaster(
                    estimator=Ridge(alpha=1.0),
                    feature_transformer=LagTransformer(lag=[1, 7]),
                ),
            ),
        ],
    )
    fc_decomp_fourier.fit(y_train, forecasting_horizon=horizon)
    _y_pred_decomp = fc_decomp_fourier.predict(forecasting_horizon=horizon)

    plot_forecast(
        y_test,
        _y_pred_decomp,
        y_train=y_train,
        n_history=30,
        title="Trend + Fourier Season + Residual",
    )
    return (fc_decomp_fourier,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Harmonics**: [1] = simple wave, [1,2,3] = more complex patterns (max = floor(seasonality/2))
    - **`None` harmonics**: Defaults to [1, 2, 3]
    - **Fourier vs Pattern**: Fourier is smoother; Pattern captures irregular patterns better
    - **Estimator choice**: Ridge/ElasticNet regularisation affects Fourier coefficient magnitudes
    - **Best use**: Inside `DecompositionPipeline` for structured trend + season + residual

    ## Next Steps

    - **Decomposition pipelines**: See `examples/compose/decomposition_variations.py`
    - **Stationarity transforms**: See `examples/stationarity/stationarity_transforms.py`
    - **Decomposition**: See `examples/stationarity/decomposition.py`
    """)

if __name__ == "__main__":
    app.run()
