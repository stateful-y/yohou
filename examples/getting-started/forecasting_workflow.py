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
    "title": "Forecasting Workflow",
    "description": "Evaluate forecasters with cross-validation, search hyperparameters with GridSearchCV, and inspect residuals to diagnose model weaknesses.",
    "category": "tutorial",
    "companion": "/pages/tutorials/forecasting-workflow/",
    "section": "getting-started",
    "api_references": ["DecompositionPipeline", "FeaturePipeline", "GridSearchCV", "MeanAbsoluteScaledError", "PointReductionForecaster", "SeasonalDifferencing", "SeasonalNaive", "plot_residuals", "plot_score_summary"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Forecasting Workflow

        In this notebook, we will evaluate two continuous-valued forecasters using
        temporal cross-validation, search for the best hyperparameters with
        [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/),
        and inspect residuals to diagnose model weaknesses.

        **Prerequisites:** Completed [Getting Started](/pages/tutorials/getting-started/).
        """
    )


@app.cell(hide_code=True)
def _():
    from sklearn.linear_model import Ridge

    from yohou.compose import FeaturePipeline
    from yohou.datasets import fetch_tourism_monthly
    from yohou.metrics import MeanAbsoluteError, MeanAbsoluteScaledError
    from yohou.model_selection import ExpandingWindowSplitter, GridSearchCV
    from yohou.plotting import plot_forecast, plot_residuals, plot_score_summary
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer
    from yohou.stationarity import SeasonalDifferencing

    return (
        ExpandingWindowSplitter,
        FeaturePipeline,
        GridSearchCV,
        LagTransformer,
        MeanAbsoluteError,
        MeanAbsoluteScaledError,
        PointReductionForecaster,
        Ridge,
        SeasonalDifferencing,
        SeasonalNaive,
        fetch_tourism_monthly,
        plot_forecast,
        plot_residuals,
        plot_score_summary,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 1. Load Data and Fit Baselines")


@app.cell
def _(
    FeaturePipeline,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SeasonalDifferencing,
    SeasonalNaive,
    fetch_tourism_monthly,
):
    from yohou.model_selection import train_test_split

    bunch = fetch_tourism_monthly()
    y = (
        bunch.frame
        .select("time", "T1__tourists")
        .drop_nulls()
        .rename({"T1__tourists": "tourists"})
    )
    forecasting_horizon = 12
    y_train, y_test = train_test_split(y, test_size=forecasting_horizon)

    baseline = SeasonalNaive(seasonality=12)
    baseline.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred_baseline = baseline.predict(forecasting_horizon=forecasting_horizon)

    forecaster = PointReductionForecaster(
        estimator=Ridge(),
        target_transformer=SeasonalDifferencing(seasonality=12),
        feature_transformer=FeaturePipeline([
            ("lags", LagTransformer(lag=[1, 2, 3, 12])),
        ]),
    )
    forecaster.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred_ridge = forecaster.predict(forecasting_horizon=forecasting_horizon)

    print(f"Train: {len(y_train)} months, Test: {forecasting_horizon} months")
    y.tail()
    return (
        baseline,
        bunch,
        forecaster,
        forecasting_horizon,
        y,
        y_pred_baseline,
        y_pred_ridge,
        y_test,
        y_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 2. Cross-Validation Settings")


@app.cell(hide_code=True)
def _(mo):
    n_splits_slider = mo.ui.slider(
        start=2,
        stop=6,
        value=3,
        label="n_splits (CV folds)",
    )
    n_splits_slider


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 3. Hyperparameter Search with GridSearchCV")


@app.cell
def _(
    ExpandingWindowSplitter,
    GridSearchCV,
    MeanAbsoluteScaledError,
    forecaster,
    forecasting_horizon,
    n_splits_slider,
    y,
):
    n_splits = n_splits_slider.value
    cv = ExpandingWindowSplitter(n_splits=n_splits, test_size=forecasting_horizon)
    search = GridSearchCV(
        forecaster=forecaster,
        param_grid={"estimator__alpha": [0.1, 1.0, 10.0, 100.0]},
        scoring=MeanAbsoluteScaledError(seasonality=12),
        cv=cv,
    )
    search.fit(y, forecasting_horizon=forecasting_horizon)
    print(f"Best params: {search.best_params_}")
    print(f"CV MASE:     {-search.best_score_:.2f}")
    return cv, n_splits, search


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 4. Compare Models")


@app.cell
def _(
    MeanAbsoluteError,
    MeanAbsoluteScaledError,
    forecasting_horizon,
    plot_score_summary,
    y_pred_baseline,
    y_pred_ridge,
    y_test,
    y_train,
):
    mae = MeanAbsoluteError()
    mae.fit(y_train)
    mase = MeanAbsoluteScaledError(seasonality=12)
    mase.fit(y_train)

    for name, y_pred in [("SeasonalNaive", y_pred_baseline), ("Ridge", y_pred_ridge)]:
        print(f"{name:15s}  MAE={mae.score(y_test, y_pred):.2f}  MASE={mase.score(y_test, y_pred):.2f}")

    fig_summary = plot_score_summary(
        mase,
        y_test,
        {"SeasonalNaive": y_pred_baseline, "Ridge": y_pred_ridge},
    )
    fig_summary
    return fig_summary, mae, mase, name, y_pred


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## 5. Residual Diagnostics")


@app.cell
def _(plot_residuals, y_pred_ridge, y_test):
    fig_resid = plot_residuals(y_pred_ridge, y_test)
    fig_resid


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [Forecasting Workflow](/pages/tutorials/forecasting-workflow/) for the full guide
        - [Evaluate Forecast Accuracy](/pages/how-to/evaluate-forecast-accuracy/) for related techniques
        """
    )
    return


if __name__ == "__main__":
    app.run()
