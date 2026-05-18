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
    "title": "How to Choose a Decomposition Strategy",
    "description": "Build 2- and 3-component DecompositionPipeline forecasters chaining trend, seasonality, and residual models with target pre-transformation.",
    "category": "how-to",
    "section": "forecasting-models",
    "companion": "/pages/how-to/apply-stationarity-transforms/",
    "api_references": ["DecompositionPipeline", "PolynomialTrendForecaster", "PointReductionForecaster", "fetch_sunspot", "LagTransformer", "plot_time_series", "plot_forecast", "MeanAbsoluteError", "LogTransformer", "fetch_tourism_quarterly"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Decomposition Variations

    [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) fits sequential forecasters where each
    subsequent forecaster models the residual of the previous one.
    Final predictions are **summed** across all components.

    ## 1. Prepare Univariate Data

    We load the Sunspot Numbers dataset and resample to monthly frequency.
    This long series (200+ years) provides a good test case for decomposition
    into trend, seasonality, and residual components.
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import DecompositionPipeline
    from yohou.datasets import fetch_sunspot, fetch_tourism_quarterly
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import train_test_split
    from yohou.plotting import (
        plot_decomposition,
        plot_forecast,
        plot_score_per_step,
        plot_score_summary,
        plot_time_series,
    )
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer
    from yohou.stationarity import LogTransformer
    from yohou.stationarity.seasonality import PatternSeasonalityForecaster
    from yohou.stationarity.trend import PolynomialTrendForecaster

    return (
        DecompositionPipeline,
        LagTransformer,
        LogTransformer,
        MeanAbsoluteError,
        PatternSeasonalityForecaster,
        PointReductionForecaster,
        PolynomialTrendForecaster,
        Ridge,
        deepcopy,
        fetch_sunspot,
        fetch_tourism_quarterly,
        pl,
        plot_decomposition,
        plot_forecast,
        plot_score_per_step,
        plot_score_summary,
        plot_time_series,
        train_test_split,
    )


@app.cell
def _(fetch_sunspot, mo, pl, train_test_split):
    _raw = fetch_sunspot().frame
    sunspots = _raw.group_by_dynamic("time", every="1mo").agg(pl.col("sunspot_number").mean())
    y_train, y_test = train_test_split(sunspots, test_size=0.15)
    horizon = len(y_test)
    mo.md(f"**Sunspots**: Train={len(y_train)}, Test={len(y_test)}")
    return horizon, sunspots, y_test, y_train


@app.cell
def _(plot_time_series, sunspots):
    plot_time_series(sunspots, title="Monthly Sunspot Numbers")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Two-Component: Trend + Residual

    [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) fits a [`PolynomialTrendForecaster`](/pages/api/generated/yohou.stationarity.trend.PolynomialTrendForecaster/) first, then
    a [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) on the residual. Final predictions are the
    sum of both components.
    """)


@app.cell
def _(
    DecompositionPipeline,
    LagTransformer,
    PointReductionForecaster,
    PolynomialTrendForecaster,
    Ridge,
    horizon,
    plot_forecast,
    y_test,
    y_train,
):
    fc_two = DecompositionPipeline(
        forecasters=[
            ("trend", PolynomialTrendForecaster(degree=2)),
            (
                "residual",
                PointReductionForecaster(
                    estimator=Ridge(alpha=1.0),
                    feature_transformer=LagTransformer(lag=[1, 12]),
                ),
            ),
        ],
        store_residuals=True,
    )
    fc_two.fit(y_train, forecasting_horizon=horizon)
    y_pred_two = fc_two.predict(forecasting_horizon=horizon)

    plot_forecast(
        y_test,
        y_pred_two,
        y_train=y_train,
        n_history=600,
        title="Two-Component: Quadratic Trend + Residual",
    )
    return fc_two, y_pred_two


@app.cell
def _(fc_two, horizon, plot_decomposition, y_test):
    _components = {}
    for _name, _forecaster in fc_two.forecasters_:
        _pred = _forecaster.predict(forecasting_horizon=horizon)
        _components[_name] = _pred.drop("vintage_time")
    plot_decomposition(y_test, _components, title="Two-Component Decomposition")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Three-Component: Trend + Seasonality + Residual

    Adding a [`PatternSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.PatternSeasonalityForecaster/) between trend and residual
    captures the ~11-year solar cycle explicitly. The residual model then
    only needs to handle irregular fluctuations.
    """)


@app.cell
def _(
    DecompositionPipeline,
    LagTransformer,
    PatternSeasonalityForecaster,
    PointReductionForecaster,
    PolynomialTrendForecaster,
    Ridge,
    horizon,
    plot_forecast,
    y_test,
    y_train,
):
    fc_three = DecompositionPipeline(
        forecasters=[
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("season", PatternSeasonalityForecaster(seasonality=132)),
            (
                "residual",
                PointReductionForecaster(
                    estimator=Ridge(alpha=1.0),
                    feature_transformer=LagTransformer(lag=[1, 12]),
                ),
            ),
        ],
    )
    fc_three.fit(y_train, forecasting_horizon=horizon)
    y_pred_three = fc_three.predict(forecasting_horizon=horizon)

    plot_forecast(
        y_test,
        y_pred_three,
        y_train=y_train,
        n_history=600,
        title="Three-Component: Trend + Season + Residual",
    )
    return fc_three, y_pred_three


@app.cell
def _(fc_three, horizon, plot_decomposition, y_test):
    _components = {}
    for _name, _forecaster in fc_three.forecasters_:
        _pred = _forecaster.predict(forecasting_horizon=horizon)
        _components[_name] = _pred.drop("vintage_time")
    plot_decomposition(y_test, _components, title="Three-Component Decomposition")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. With target_transformer (Log Scale)

    Apply a log transformation before decomposition, then invert after
    prediction. Useful for data with multiplicative trends.
    """)


@app.cell
def _(
    DecompositionPipeline,
    LagTransformer,
    LogTransformer,
    PointReductionForecaster,
    PolynomialTrendForecaster,
    Ridge,
    horizon,
    plot_forecast,
    y_test,
    y_train,
):
    fc_log = DecompositionPipeline(
        forecasters=[
            ("trend", PolynomialTrendForecaster(degree=1)),
            (
                "residual",
                PointReductionForecaster(
                    estimator=Ridge(alpha=1.0),
                    feature_transformer=LagTransformer(lag=[1, 12]),
                ),
            ),
        ],
        target_transformer=LogTransformer(offset=1.0),
    )
    fc_log.fit(y_train, forecasting_horizon=horizon)
    y_pred_log = fc_log.predict(forecasting_horizon=horizon)

    plot_forecast(
        y_test,
        y_pred_log,
        y_train=y_train,
        n_history=600,
        title="Decomposition with Log Target Transformer",
    )
    return (y_pred_log,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Compare All Variations

    [`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.point.MeanAbsoluteError/) scores each decomposition configuration on the
    held-out test data, showing which combination best captures the
    sunspot dynamics.
    """)


@app.cell
def _(
    MeanAbsoluteError,
    mo,
    y_pred_log,
    y_pred_three,
    y_pred_two,
    y_test,
    y_train,
):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _mae_two = float(_scorer.score(y_test, y_pred_two))
    _mae_three = float(_scorer.score(y_test, y_pred_three))
    _mae_log = float(_scorer.score(y_test, y_pred_log))

    mo.md(
        f"| Variation | MAE |\n"
        f"|-----------|-----|\n"
        f"| Trend + Residual | {_mae_two:.2f} |\n"
        f"| Trend + Season + Residual | {_mae_three:.2f} |\n"
        f"| Log + Trend + Residual | {_mae_log:.2f} |"
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Panel Data Decomposition

    Apply the same pipeline to panel data: each group gets its own
    trend and residual model.
    """)


@app.cell
def _(
    DecompositionPipeline,
    LagTransformer,
    PointReductionForecaster,
    PolynomialTrendForecaster,
    Ridge,
    fetch_tourism_quarterly,
    plot_forecast,
    train_test_split,
):
    _tourism = fetch_tourism_quarterly().frame
    # Select 8 series with uniform length for a manageable panel demo
    _tourist_cols = [f"T{i}__tourists" for i in range(3, 11)]
    _tourism = _tourism.select("time", *_tourist_cols).drop_nulls()
    _y_train_p, _y_test_p = train_test_split(_tourism, test_size=0.2)
    _horizon_p = len(_y_test_p)

    _fc_panel = DecompositionPipeline(
        forecasters=[
            ("trend", PolynomialTrendForecaster(degree=1)),
            (
                "residual",
                PointReductionForecaster(
                    estimator=Ridge(alpha=1.0),
                    feature_transformer=LagTransformer(lag=[1, 4]),
                ),
            ),
        ],
    )
    _fc_panel.fit(_y_train_p, forecasting_horizon=_horizon_p)
    _y_pred_panel = _fc_panel.predict(forecasting_horizon=_horizon_p)

    plot_forecast(
        _y_test_p,
        _y_pred_panel,
        y_train=_y_train_p,
        n_history=8,
        groups=["T3", "T4", "T5"],
        title="Panel Decomposition: Trend + Residual",
    )


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
def _(LagTransformer, PointReductionForecaster, Ridge, deepcopy, horizon, y_test, y_train):
    _fc_v = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 12]),
    )
    _fc_v.fit(y_train, forecasting_horizon=horizon)
    _vintage_model = deepcopy(_fc_v)
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
def _(vintage_scorer, plot_score_per_step, y_pred_vintages, y_test):
    plot_score_per_step(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="MAE per Forecast Step",
        y_label="MAE",
        height=380,
    )


@app.cell
def _(vintage_scorer, plot_score_summary, y_pred_vintages, y_test):
    plot_score_summary(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="Model Score Summary",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Pipeline composition**: See [`examples/compose/pipeline_composition.py`](/examples/data-features/pipeline_composition/)
    - **Stationarity transforms**: See [`examples/stationarity/stationarity_transforms.py`](/examples/data-features/stationarity_transforms/)
    - **Panel stationarity**: See [`examples/stationarity/panel_stationarity.py`](/examples/panel-data/panel_stationarity/)
    """)


if __name__ == "__main__":
    app.run()
