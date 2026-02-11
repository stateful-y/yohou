"""Pipeline Composition.

Demonstrates nested pipeline patterns: FeaturePipeline, FeatureUnion
inside forecasters, DecompositionPipeline with feature engineering,
and multi-level nesting.
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
    # Pipeline Composition

    Yohou's composition classes can be nested arbitrarily:
    `FeaturePipeline` steps execute sequentially, `FeatureUnion`
    branches execute in parallel, and `DecompositionPipeline` cascades
    residuals.

    ## What You'll Learn

    - `FeaturePipeline`: sequential transformers (observation_horizon = SUM)
    - Nesting `FeatureUnion` inside `FeaturePipeline`
    - `DecompositionPipeline` with feature-engineered residual models
    - Multi-level nesting: Decomposition → Feature → Union → Forecaster
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import DecompositionPipeline, FeaturePipeline, FeatureUnion
    from yohou.datasets import load_vic_electricity
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import (
        LagTransformer,
        RollingStatisticsTransformer,
        StandardScaler,
    )
    from yohou.stationarity import PolynomialTrendForecaster

    return (
        DecompositionPipeline,
        FeaturePipeline,
        FeatureUnion,
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        PolynomialTrendForecaster,
        Ridge,
        RollingStatisticsTransformer,
        SeasonalNaive,
        StandardScaler,
        load_vic_electricity,
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
def _(load_vic_electricity, mo, pl):
    _elec = load_vic_electricity()
    elec = (
        _elec.group_by_dynamic("time", every="1d")
        .agg(pl.col("Demand").mean())
    )
    _split = int(len(elec) * 0.85)
    y_train = elec.head(_split)
    y_test = elec.tail(len(elec) - _split)
    horizon = len(y_test)
    mo.md(f"**Daily electricity demand**: Train={len(y_train)}, Test={len(y_test)}")
    return elec, horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. FeaturePipeline: Sequential Transformers

    `FeaturePipeline` applies steps sequentially. The combined
    `observation_horizon` is the **sum** of all steps.
    """)
    return


@app.cell
def _(FeaturePipeline, LagTransformer, StandardScaler, mo, y_train):
    fp = FeaturePipeline(
        steps=[
            ("lags", LagTransformer(lag=[1, 7])),
            ("scale", StandardScaler()),
        ],
    )
    fp.fit(y_train)
    _out = fp.transform(y_train)

    mo.md(
        f"**FeaturePipeline output**: {_out.shape}\n\n"
        f"**Columns**: {_out.columns}\n\n"
        f"**observation_horizon**: {fp.observation_horizon} "
        f"(sum: LagTransformer={7} + StandardScaler={0})"
    )
    return (fp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. FeatureUnion Inside FeaturePipeline

    Branch in parallel (FeatureUnion), then apply a sequential step.
    """)
    return


@app.cell
def _(
    FeaturePipeline,
    FeatureUnion,
    LagTransformer,
    RollingStatisticsTransformer,
    StandardScaler,
    mo,
    y_train,
):
    fp_nested = FeaturePipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    transformer_list=[
                        ("lags", LagTransformer(lag=[1, 7])),
                        ("rolling", RollingStatisticsTransformer(window_size=7, statistics="mean")),
                    ],
                ),
            ),
            ("scale", StandardScaler()),
        ],
    )
    fp_nested.fit(y_train)
    _out_nested = fp_nested.transform(y_train)

    mo.md(
        f"**FeaturePipeline(FeatureUnion + StandardScaler)**\n\n"
        f"Output columns: {_out_nested.columns}\n\n"
        f"observation_horizon: {fp_nested.observation_horizon}"
    )
    return (fp_nested,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. FeaturePipeline as feature_transformer

    Use the nested pipeline as `feature_transformer` in a forecaster.
    """)
    return


@app.cell
def _(
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    SeasonalNaive,
    fp_nested,
    horizon,
    mo,
    y_test,
    y_train,
):
    fc_fp = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=fp_nested,
    )
    fc_fp.fit(y_train, forecasting_horizon=horizon)
    y_pred_fp = fc_fp.predict(forecasting_horizon=horizon)

    _naive = SeasonalNaive(seasonality=7)
    _naive.fit(y_train, forecasting_horizon=horizon)
    _y_pred_naive = _naive.predict(forecasting_horizon=horizon)

    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _mae_fp = float(_scorer.score(y_test, y_pred_fp))
    _mae_naive = float(_scorer.score(y_test, _y_pred_naive))

    mo.md(
        f"**FeaturePipeline + Ridge MAE**: {_mae_fp:.2f}\n\n"
        f"**SeasonalNaive MAE**: {_mae_naive:.2f}"
    )
    return fc_fp, y_pred_fp


@app.cell
def _(horizon, plot_forecast, y_pred_fp, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_fp,
        y_train=y_train,
        n_history=30,
        title="FeaturePipeline(Union + Scale) + Ridge",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. DecompositionPipeline with Feature Engineering

    Multi-level nesting: Decomposition removes trend, then the residual
    model uses FeatureUnion.
    """)
    return


@app.cell
def _(
    DecompositionPipeline,
    FeatureUnion,
    LagTransformer,
    PointReductionForecaster,
    PolynomialTrendForecaster,
    Ridge,
    RollingStatisticsTransformer,
    horizon,
    plot_forecast,
    y_test,
    y_train,
):
    _union_deep = FeatureUnion(
        transformer_list=[
            ("lags", LagTransformer(lag=[1, 7])),
            ("rolling", RollingStatisticsTransformer(window_size=7, statistics=["mean", "std"])),
        ],
    )

    fc_deep = DecompositionPipeline(
        forecasters=[
            ("trend", PolynomialTrendForecaster(degree=1)),
            (
                "residual",
                PointReductionForecaster(
                    estimator=Ridge(alpha=1.0),
                    feature_transformer=_union_deep,
                ),
            ),
        ],
    )
    fc_deep.fit(y_train, forecasting_horizon=horizon)
    _y_pred_deep = fc_deep.predict(forecasting_horizon=horizon)

    plot_forecast(
        y_test,
        _y_pred_deep,
        y_train=y_train,
        n_history=30,
        title="Decomposition + FeatureUnion Residual",
    )
    return (fc_deep,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Compare All Pipelines
    """)
    return


@app.cell
def _(MeanAbsoluteError, fc_deep, fc_fp, horizon, mo, pl, y_test, y_train):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _models = {
        "FeaturePipeline + Ridge": fc_fp,
        "Decomposition + Union": fc_deep,
    }
    _rows = []
    for _name, _fc in _models.items():
        _pred = _fc.predict(forecasting_horizon=horizon)
        _mae = float(_scorer.score(y_test, _pred))
        _rows.append({"Pipeline": _name, "MAE": round(_mae, 2)})

    mo.ui.table(pl.DataFrame(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **FeaturePipeline**: Sequential steps, `observation_horizon` = SUM
    - **FeatureUnion**: Parallel branches, `observation_horizon` = MAX
    - **Nesting**: Union inside Pipeline, Pipeline as feature_transformer, etc.
    - **DecompositionPipeline**: Cascade residuals with feature-rich residual models
    - **Best practice**: Start simple, add complexity only if it improves scores

    ## Next Steps

    - **Forecasted feature**: See `examples/compose/forecasted_feature_advanced.py`
    - **Feature union details**: See `examples/compose/feature_union.py`
    - **Panel pipelines**: See `examples/compose/panel_pipelines.py`
    """)
    return


if __name__ == "__main__":
    app.run()
