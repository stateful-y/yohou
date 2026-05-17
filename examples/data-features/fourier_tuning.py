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
    "title": "How to Tune Fourier Seasonality Terms",
    "description": "Explore how Fourier harmonic count affects seasonal fit quality, compare Fourier vs Pattern seasonality, and tune harmonics jointly with GridSearchCV.",
    "category": "how-to",
    "section": "data-features",
    "companion": "/pages/how-to/handle-complex-seasonality/",
    "api_references": ["FourierSeasonalityForecaster", "PatternSeasonalityForecaster", "DecompositionPipeline", "PointReductionForecaster", "LagTransformer", "GridSearchCV", "ExpandingWindowSplitter", "MeanAbsoluteError", "plot_forecast", "train_test_split", "fetch_electricity_demand", "plot_score_per_vintage"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fourier Seasonality Tuning

    [`FourierSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.FourierSeasonalityForecaster/) models seasonality using Fourier
    terms (sine/cosine pairs). Each harmonic adds a sine/cosine pair
    at a specific frequency, and the number of harmonics controls how
    closely the model can approximate complex seasonal shapes.

    ## 1. Load Data

    Use electricity demand data with daily aggregation for clear
    weekly seasonality.
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

    import polars as pl
    from sklearn.base import clone
    from sklearn.linear_model import ElasticNet, Ridge

    from yohou.compose import DecompositionPipeline
    from yohou.datasets import fetch_electricity_demand
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import (
        ExpandingWindowSplitter,
        GridSearchCV,
        train_test_split,
    )
    from yohou.plotting import plot_forecast, plot_score_per_vintage
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer
    from yohou.stationarity.seasonality import (
        FourierSeasonalityForecaster,
        PatternSeasonalityForecaster,
    )

    return (
        DecompositionPipeline,
        ElasticNet,
        ExpandingWindowSplitter,
        FourierSeasonalityForecaster,
        GridSearchCV,
        LagTransformer,
        MeanAbsoluteError,
        PatternSeasonalityForecaster,
        PointReductionForecaster,
        Ridge,
        clone,
        deepcopy,
        fetch_electricity_demand,
        pl,
        plot_forecast,
        plot_score_per_vintage,
        train_test_split,
    )


@app.cell
def _(fetch_electricity_demand, mo, pl, train_test_split):
    _elec = fetch_electricity_demand().frame
    # Resample to daily for clearer weekly patterns (drop trailing all-null days)
    elec_daily = (
        _elec.group_by_dynamic("time", every="1d").agg(pl.col("vic__demand").mean().alias("demand")).drop_nulls()
    )
    y_train, y_test = train_test_split(elec_daily, test_size=0.15)
    horizon = len(y_test)
    mo.md(
        f"**Daily electricity demand**: {len(elec_daily)} days\n\n"
        f"**Train**: {len(y_train)} days, **Test**: {len(y_test)} days\n\n"
        f"**Weekly seasonality** = 7 days"
    )
    return horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Harmonics Comparison

    Each harmonic captures a different frequency of the seasonal
    cycle. `harmonics=[1]` fits only the fundamental period,
    producing a single smooth sine wave. Adding higher harmonics
    (e.g. `[1, 2]` or `[1, 2, 52]`) lets the model represent
    sharper, more complex patterns but risks overfitting if the
    data doesn't support that complexity.

    The maximum useful harmonic is `floor(seasonality / 2)`.
    """)


@app.cell
def _(
    FourierSeasonalityForecaster,
    MeanAbsoluteError,
    Ridge,
    horizon,
    mo,
    pl,
    plot_forecast,
    y_test,
    y_train,
):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _rows = []
    _preds = {}

    for _nh in [[1], [1, 2], [1, 2, 52]]:
        _fc = FourierSeasonalityForecaster(estimator=Ridge(alpha=1e-5), seasonality=365, harmonics=_nh)
        _fc.fit(y_train, forecasting_horizon=horizon)
        _y_pred = _fc.predict(forecasting_horizon=horizon)
        _mae = float(_scorer.score(y_test, _y_pred))
        _rows.append({"Harmonics": str(_nh), "MAE": round(_mae, 2)})
        _preds[f"harmonics={_nh}"] = _y_pred

    mo.vstack([
        mo.ui.table(pl.DataFrame(_rows)),
        plot_forecast(
            y_test,
            _preds,
            y_train=y_train,
            n_history=1000,
            title="Harmonics Comparison",
        ),
    ])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Fourier vs Pattern Seasonality

    [`FourierSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.FourierSeasonalityForecaster/) represents seasonality as a
    weighted sum of sine/cosine terms creating a smooth, parametric model.
    [`PatternSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.PatternSeasonalityForecaster/) directly averages (or mediates)
    historical values at each seasonal position as a non-parametric
    approach that can capture irregular shapes.

    When the true seasonal shape is close to sinusoidal, Fourier
    generalises better. When it has sharp spikes or asymmetric
    patterns, Pattern may win.
    """)


@app.cell
def _(
    FourierSeasonalityForecaster,
    MeanAbsoluteError,
    PatternSeasonalityForecaster,
    Ridge,
    horizon,
    mo,
    plot_forecast,
    y_test,
    y_train,
):
    _fc_fourier = FourierSeasonalityForecaster(estimator=Ridge(alpha=1e-5), seasonality=365, harmonics=[1, 2, 52])
    _fc_fourier.fit(y_train, forecasting_horizon=horizon)
    _y_pred_fourier = _fc_fourier.predict(forecasting_horizon=horizon)

    _fc_pattern = PatternSeasonalityForecaster(seasonality=365, method="average")
    _fc_pattern.fit(y_train, forecasting_horizon=horizon)
    _y_pred_pattern = _fc_pattern.predict(forecasting_horizon=horizon)

    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _mae_f = float(_scorer.score(y_test, _y_pred_fourier))
    _mae_p = float(_scorer.score(y_test, _y_pred_pattern))

    mo.vstack([
        mo.md(
            f"**Fourier (3 harmonics) MAE**: {_mae_f:.2f}\n\n"
            f"**Pattern (average) MAE**: {_mae_p:.2f}\n\n"
            "Fourier produces smoother seasonal patterns; Pattern is more flexible."
        ),
        plot_forecast(
            y_test,
            {"Fourier (3 harmonics)": _y_pred_fourier, "Pattern (average)": _y_pred_pattern},
            y_train=y_train,
            n_history=1000,
            title="Fourier vs Pattern Seasonality",
        ),
    ])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Different Estimators

    Internally, [`FourierSeasonalityForecaster`](/pages/api/generated/yohou.stationarity.seasonality.FourierSeasonalityForecaster/) builds a design matrix
    of sine/cosine columns and fits a linear regression. The choice
    of estimator controls how the Fourier coefficients are
    regularised: ElasticNet applies both L1 and L2 penalties
    (potentially zeroing out harmonics), while Ridge applies only L2
    (shrinking coefficients towards zero without eliminating them).

    Higher `alpha` values increase shrinkage sharply. Try different
    orders of magnitude to see the effect on forecast accuracy.
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
    plot_forecast,
    y_test,
    y_train,
):
    _scorer = MeanAbsoluteError()
    _scorer.fit(y_train)
    _rows = []
    _preds = {}
    for _name, _est in [
        ("ElasticNet (default)", ElasticNet()),
        ("Ridge(alpha=0.1)", Ridge(alpha=0.1)),
        ("Ridge(alpha=10)", Ridge(alpha=10.0)),
    ]:
        _fc = FourierSeasonalityForecaster(
            seasonality=365,
            harmonics=[1, 2, 52],
            estimator=_est,
        )
        _fc.fit(y_train, forecasting_horizon=horizon)
        _y_pred = _fc.predict(forecasting_horizon=horizon)
        _mae = float(_scorer.score(y_test, _y_pred))
        _rows.append({"Estimator": _name, "MAE": round(_mae, 2)})
        _preds[_name] = _y_pred

    mo.vstack([
        mo.ui.table(pl.DataFrame(_rows)),
        plot_forecast(
            y_test,
            _preds,
            y_train=y_train,
            n_history=60,
            title="Estimator Comparison",
        ),
    ])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Fourier in a DecompositionPipeline

    A [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/) lets you stack forecasters
    sequentially: each one fits and removes its component, passing
    the residual to the next. Here we use Fourier seasonality to
    capture the yearly pattern, then a [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/)
    with lag features to model whatever structure remains in the
    residual.
    """)


@app.cell
def _(
    DecompositionPipeline,
    FourierSeasonalityForecaster,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    horizon,
    plot_forecast,
    y_test,
    y_train,
):
    fc_decomp_fourier = DecompositionPipeline(
        forecasters=[
            ("season", FourierSeasonalityForecaster(estimator=Ridge(1e-5), seasonality=365, harmonics=[1, 2, 52])),
            (
                "residual",
                PointReductionForecaster(
                    estimator=Ridge(alpha=1.0),
                    feature_transformer=LagTransformer(lag=[1, 7, 365]),
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
        n_history=1000,
        title="Trend + Fourier Season + Residual",
    )
    return (fc_decomp_fourier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Grid Search: Tuning Harmonics & Ridge Alphas

    Rather than picking harmonics and Ridge alphas by hand, we can
    search over them jointly with [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/). The search
    evaluates every combination using time-series cross-validation
    ([`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.split.ExpandingWindowSplitter/)) and selects the configuration with
    the lowest MAE.

    The grid covers the Fourier harmonic sets, the Fourier
    estimator's regularisation, and the residual estimator's
    regularisation as those three axes interact with each other.
    """)


@app.cell
def _(
    ExpandingWindowSplitter,
    GridSearchCV,
    MeanAbsoluteError,
    clone,
    fc_decomp_fourier,
    horizon,
    mo,
    plot_forecast,
    y_test,
    y_train,
):
    _param_grid = {
        "season__harmonics": [[1], [1, 2], [1, 2, 52], [1, 2, 52, 104]],
        "season__estimator__alpha": [1e-5, 1e-3, 1.0],
        "residual__estimator__alpha": [0.1, 1.0, 10.0],
    }

    _search = GridSearchCV(
        forecaster=clone(fc_decomp_fourier),
        param_grid=_param_grid,
        scoring=MeanAbsoluteError(),
        cv=ExpandingWindowSplitter(n_splits=2, test_size=horizon),
    )
    _search.fit(y_train, forecasting_horizon=horizon)

    _y_pred_best = _search.predict(forecasting_horizon=horizon)

    mo.vstack([
        mo.md(f"**Best params**: `{_search.best_params_}`\n\n**Best CV score**: {_search.best_score_:.2f}"),
        plot_forecast(
            y_test,
            _y_pred_best,
            y_train=y_train,
            n_history=60,
            title="Grid Search Best Pipeline",
        ),
    ])


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
def _(FourierSeasonalityForecaster, Ridge, deepcopy, horizon, y_test, y_train):
    _fc_v = FourierSeasonalityForecaster(estimator=Ridge(alpha=1e-5), seasonality=365, harmonics=[1, 2, 52])
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

    - **Decomposition pipelines**: See [`examples/compose/decomposition_variations.py`](/examples/forecasting-models/decomposition_variations/)
    - **Stationarity transforms**: See [`examples/stationarity/stationarity_transforms.py`](/examples/data-features/stationarity_transforms/)
    - **Decomposition**: See [`examples/stationarity/decomposition.py`](/examples/getting-started/decomposition/)
    - **Hyperparameter search**: See [`examples/model_selection/hyperparameter_search.py`](/examples/evaluation-search/hyperparameter_search/)
    """)


if __name__ == "__main__":
    app.run()
