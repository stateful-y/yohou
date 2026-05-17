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
    "title": "How to Apply Stationarity to Panel Data",
    "description": "Apply per-group stationarity transforms on panel data with SeasonalDifferencing, DecompositionPipeline (polynomial trend + pattern seasonality), and residuals.",
    "category": "how-to",
    "section": "panel-data",
    "companion": "/pages/how-to/apply-stationarity-transforms/",
    "api_references": ["fetch_tourism_quarterly", "inspect_panel", "train_test_split", "plot_time_series", "SeasonalDifferencing", "DecompositionPipeline", "PolynomialTrendForecaster", "PointReductionForecaster", "LagTransformer"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Stationarity of Panel Data

    Stationarity transformations and decomposition work per-group on
    panel data. Each group gets its own trend, seasonality, and residual
    extraction.

    ## 1. Load and Explore Panel Data
    """)


@app.cell
def _(fetch_tourism_quarterly, inspect_panel, mo, train_test_split):
    _bunch = fetch_tourism_quarterly()
    # Select 8 series with uniform length for a manageable panel demo
    _selected = [f"T{i}__tourists" for i in range(3, 11)]
    tourism = _bunch.frame.select("time", *_selected).drop_nulls()
    _globals, groups = inspect_panel(tourism)
    y_train, y_test = train_test_split(tourism, test_size=0.2)
    horizon = len(y_test)

    mo.md(
        f"**Groups** ({len(groups)} panel groups): {list(groups.keys())}\n\n"
        f"**Train**: {len(y_train)} quarters, **Test**: {len(y_test)} quarters"
    )
    return horizon, tourism, y_test, y_train


@app.cell
def _(plot_time_series, tourism):
    plot_time_series(tourism, title="Tourism Quarterly: All Panel Groups")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Seasonal Differencing on Panel Data

    `SeasonalDifferencing(seasonality=4)` removes quarterly seasonality
    from each state independently.
    """)


@app.cell
def _(SeasonalDifferencing, plot_time_series, tourism):
    sd = SeasonalDifferencing(seasonality=4)
    sd.fit(tourism)
    tourism_diff = sd.transform(tourism)
    plot_time_series(tourism_diff, title="Seasonal Differencing (lag=4): All Groups")
    return sd, tourism_diff


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The seasonal differencing transformer is perfectly invertible (within floating-point precision).
    """)


@app.cell
def _(sd, tourism, tourism_diff):
    _inv = sd.inverse_transform(tourism_diff, tourism.head(sd.seasonality))
    # inverse_transform reconstructs the original, except for the first 4 rows
    _overlap = tourism.tail(len(_inv))
    _err = _inv.drop("time").to_numpy() - _overlap.drop("time").to_numpy()
    _max_err = abs(_err).max()
    print(f"Inverse transform check: max error = {_max_err:.2e}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. DecompositionPipeline on Panel Data

    Each group gets its own linear trend + residual model.
    """)


@app.cell
def _(
    DecompositionPipeline,
    LagTransformer,
    PointReductionForecaster,
    PolynomialTrendForecaster,
    horizon,
    y_train,
):
    from sklearn.linear_model import Ridge

    fc_decomp = DecompositionPipeline(
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
        store_residuals=True,
    )
    fc_decomp.fit(y_train, forecasting_horizon=horizon)
    y_pred_decomp = fc_decomp.predict(forecasting_horizon=horizon)
    return Ridge, y_pred_decomp


@app.cell
def _(plot_forecast, y_pred_decomp, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_decomp,
        y_train=y_train,
        groups=["T3", "T4", "T5"],
        title="DecompositionPipeline: Trend + Residual (Panel)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Trend + Seasonality + Residual

    Add a [`PatternSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.PatternSeasonalityForecaster/) between trend and residual
    for a three-component decomposition.
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
            ("season", PatternSeasonalityForecaster(seasonality=12)),
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
    _y_pred_three = fc_three.predict(forecasting_horizon=horizon)

    plot_forecast(
        y_test,
        _y_pred_three,
        y_train=y_train,
        groups=["T3", "T4", "T5"],
        title="Trend + Seasonality + Residual (Panel)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Compare Approaches

    Score each approach per group to see decomposition benefits.
    """)


@app.cell
def _(
    MeanAbsoluteError,
    SeasonalNaive,
    horizon,
    plot_score_time_series,
    y_pred_decomp,
    y_test,
    y_train,
):
    _naive = SeasonalNaive(seasonality=4)
    _naive.fit(y_train, forecasting_horizon=horizon)
    _y_pred_naive = _naive.predict(forecasting_horizon=horizon)

    _scorer = MeanAbsoluteError()

    plot_score_time_series(
        _scorer,
        y_test,
        {"SeasonalNaive": _y_pred_naive, "Decomposition": y_pred_decomp},
        groups=["T3", "T4", "T5"],
        title="MAE over Time per Panel Group",
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
def _(SeasonalNaive, deepcopy, horizon, y_test, y_train):
    _naive_v = SeasonalNaive(seasonality=4)
    _naive_v.fit(y_train, forecasting_horizon=horizon)
    _vintage_model = deepcopy(_naive_v)
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
def _(vintage_scorer, plot_score_per_vintage, y_pred_vintages, y_test):
    plot_score_per_vintage(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="MAE per Forecast Vintage",
        y_label="MAE",
        height=380,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Panel pipelines**: See [`examples/compose/panel_pipelines.py`](/examples/panel-data/panel_pipelines/)
    - **Stationarity transforms**: See [`examples/stationarity/stationarity_transforms.py`](/examples/data-features/stationarity_transforms/)
    - **Decomposition details**: See [`examples/stationarity/decomposition.py`](/examples/getting-started/decomposition/)
    """)


if __name__ == "__main__":
    app.run()
