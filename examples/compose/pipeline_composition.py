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
    "title": "Pipeline Composition",
    "description": "Nest FeaturePipeline, FeatureUnion, and DecompositionPipeline for multi-level feature engineering with trend-season-residual decomposition.",
    "category": "how-to",
    "companion": "/pages/explanation/preprocessing/#composing-transformers",
    "section": "data-features",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pipeline Composition

    Yohou's composition classes can be nested arbitrarily:
    [`FeaturePipeline`](/pages/api/generated/yohou.compose.feature_pipeline.FeaturePipeline/) steps execute sequentially, [`FeatureUnion`](/pages/api/generated/yohou.compose.feature_union.FeatureUnion/)
    branches execute in parallel, and [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) cascades
    residuals.

    This notebook shows how to nest FeaturePipeline, FeatureUnion, and DecompositionPipeline for multi-level feature engineering with trend-season-residual decomposition.
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.compose import DecompositionPipeline, FeaturePipeline, FeatureUnion
    from yohou.datasets import fetch_electricity_demand
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import train_test_split
    from yohou.plotting import plot_forecast, plot_time_series
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
        fetch_electricity_demand,
        pl,
        plot_forecast,
        plot_time_series,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We load the Electricity Demand dataset and aggregate Victorian demand
    to daily frequency. This provides a univariate series with clear daily
    and weekly patterns for demonstrating pipeline composition.
    """)


@app.cell
def _(fetch_electricity_demand, mo, pl, train_test_split):
    _elec = fetch_electricity_demand().frame
    elec = _elec.group_by_dynamic("time", every="1d").agg(pl.col("vic__demand").mean().alias("demand")).drop_nulls()
    y_train, y_test = train_test_split(elec, test_size=0.15)
    horizon = len(y_test)
    mo.md(f"**Daily electricity demand**: Train={len(y_train)}, Test={len(y_test)}")
    return elec, horizon, y_test, y_train


@app.cell
def _(elec, plot_time_series):
    plot_time_series(elec, title="Daily Victorian Electricity Demand")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. FeaturePipeline: Sequential Transformers

    [`FeaturePipeline`](/pages/api/generated/yohou.compose.feature_pipeline.FeaturePipeline/) applies steps sequentially. The combined
    `observation_horizon` is the **sum** of all steps.
    """)


@app.cell
def _(FeaturePipeline, LagTransformer, StandardScaler, mo, y_train):
    fp = FeaturePipeline(
        steps=[
            ("lags", LagTransformer(lag=[1, 7])),
            ("scale", StandardScaler()),
        ],
    )
    fp.fit(y_train)
    fp_out = fp.transform(y_train)

    mo.md(
        f"**FeaturePipeline output**: {fp_out.shape}\n\n"
        f"**Columns**: {fp_out.columns}\n\n"
        f"**observation_horizon**: {fp.observation_horizon} "
        f"(sum: LagTransformer={7} + StandardScaler={0})"
    )
    return (fp_out,)


@app.cell
def _(fp_out, plot_time_series):
    plot_time_series(fp_out, title="FeaturePipeline Output: Lagged + Scaled")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. FeatureUnion Inside FeaturePipeline

    Branch in parallel (FeatureUnion), then apply a sequential step.
    """)


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
    fp_nested_out = fp_nested.transform(y_train)

    mo.md(
        f"**FeaturePipeline(FeatureUnion + StandardScaler)**\n\n"
        f"Output columns: {fp_nested_out.columns}\n\n"
        f"observation_horizon: {fp_nested.observation_horizon}"
    )
    return fp_nested, fp_nested_out


@app.cell
def _(fp_nested_out, plot_time_series):
    plot_time_series(fp_nested_out, title="Nested Pipeline Output: Union + Scale")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. FeaturePipeline as feature_transformer

    Use the nested pipeline as `feature_transformer` in a forecaster.
    """)


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

    mo.md(f"**FeaturePipeline + Ridge MAE**: {_mae_fp:.2f}\n\n**SeasonalNaive MAE**: {_mae_naive:.2f}")
    return fc_fp, y_pred_fp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) displays the nested [`FeaturePipeline`](/pages/api/generated/yohou.compose.feature_pipeline.FeaturePipeline/) predictions
    against the held-out test data.
    """)


@app.cell
def _(plot_forecast, y_pred_fp, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_fp,
        y_train=y_train,
        n_history=30,
        title="FeaturePipeline(Union + Scale) + Ridge",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. DecompositionPipeline with Feature Engineering

    Multi-level nesting: Decomposition removes trend, then the residual
    model uses FeatureUnion.
    """)


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

    [`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.point.MeanAbsoluteError/) scores each pipeline configuration on the test
    data, showing whether the added complexity improves accuracy.
    """)


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



if __name__ == "__main__":
    app.run()
