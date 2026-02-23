"""Reduction Forecasting with sklearn.

Demonstrates PointReductionForecaster for tabular ML-based time series forecasting,
with GridSearchCV for hyperparameter tuning.
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
async def _():
    import sys

    if "pyodide" in sys.modules:
        import micropip

        await micropip.install(["plotly", "scikit-learn", "scipy", "yohou"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reduction Forecasting with sklearn

    This notebook demonstrates **reduction forecasting** - the approach of converting
    time series forecasting into a supervised learning problem that sklearn regressors
    can solve.

    ## What You'll Learn

    - How `PointReductionForecaster` tabularizes time series data using lag features
    - The difference between `target_transformer` (invertible) and `feature_transformer` (features)
    - Tuning hyperparameters with `GridSearchCV`
    - Visualizing forecasts and cross-validation results

    ## Prerequisites

    Basic familiarity with sklearn's fit/predict API and time series concepts (trend, seasonality).
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    from yohou.datasets import fetch_tourism_monthly
    from yohou.metrics import MeanAbsoluteError
    from yohou.model_selection import ExpandingWindowSplitter, GridSearchCV
    from yohou.plotting import (
        plot_cv_results_scatter,
        plot_forecast,
        plot_seasonality,
        plot_time_series,
    )
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer
    from yohou.stationarity import LogTransformer

    return (
        ExpandingWindowSplitter,
        GridSearchCV,
        LagTransformer,
        LogTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        fetch_tourism_monthly,
        pl,
        plot_cv_results_scatter,
        plot_forecast,
        plot_seasonality,
        plot_time_series,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load and Explore the Data

    We use the Monthly Tourism dataset - monthly tourist counts.
    It exhibits strong trend and seasonality, making it ideal for
    demonstrating preprocessing techniques.
    """)


@app.cell
def _(fetch_tourism_monthly):
    y = (
        fetch_tourism_monthly()
        .frame.select("time", "T1__tourists")
        .rename({"T1__tourists": "passengers"})
    )

    print(f"Dataset: {len(y)} observations from {y['time'].min()} to {y['time'].max()}")
    y.head()
    return (y,)


@app.cell
def _(plot_time_series, y):
    plot_time_series(y, title="Monthly Tourism")


@app.cell
def _(plot_seasonality, y):
    plot_seasonality(y, period="month", title="Monthly Seasonality Pattern")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Train/Test Split

    For time series, we must preserve temporal order - no shuffling allowed.
    We hold out the last ~20% (29 months) for testing.
    """)


@app.cell
def _(train_test_split, y):
    y_train, y_test = train_test_split(y, test_size=0.2, shuffle=False)
    forecasting_horizon = 12

    print(f"Training: {len(y_train)} obs ({y_train['time'].min()} to {y_train['time'].max()})")
    print(f"Test: {len(y_test)} obs ({y_test['time'].min()} to {y_test['time'].max()})")
    return forecasting_horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Basic Reduction Forecaster

    `PointReductionForecaster` converts time series forecasting into tabular regression:

    1. **Feature generation**: `feature_transformer` creates lag features from past y values
    2. **Fit**: Trains an sklearn regressor on the (lags, y) tabular data
    3. **Predict**: Recursively forecasts by feeding predictions back as features

    Key distinction:
    - **`feature_transformer`**: Generates input features from y (e.g., `LagTransformer` for lags) - not invertible
    - **`target_transformer`**: Applied to y before fitting, inverted after prediction (e.g., `LogTransformer`) - must be invertible

    We start with a simple Ridge regressor and 12 lag features.
    """)


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, forecasting_horizon, y_train):
    forecaster = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=list(range(1, 13))),
    )

    forecaster.fit(y_train, forecasting_horizon=forecasting_horizon)
    print("Forecaster fitted successfully")
    return (forecaster,)


@app.cell
def _(forecaster, forecasting_horizon):
    y_pred = forecaster.predict(forecasting_horizon=forecasting_horizon)
    y_pred.head()
    return (y_pred,)


@app.cell
def _(MeanAbsoluteError, plot_forecast, y_pred, y_test, y_train):
    fig_basic = plot_forecast(
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        title="Basic Reduction Forecast",
    )

    mae = MeanAbsoluteError()
    y_test_trimmed = y_test.head(len(y_pred))
    mae.fit(y_train)
    score = mae.score(y_test_trimmed, y_pred)
    print(f"MAE: {score:.2f}")
    fig_basic
    return fig_basic, mae, score, y_test_trimmed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Adding Target Transformation

    The Monthly Tourism data has multiplicative seasonality (variance grows with level).
    A `LogTransformer` via `target_transformer` stabilizes variance. It is applied to y
    before fitting and automatically inverted after prediction.
    """)


@app.cell
def _(
    LagTransformer,
    LogTransformer,
    PointReductionForecaster,
    Ridge,
    forecasting_horizon,
    y_train,
):
    forecaster_log = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        target_transformer=LogTransformer(offset=1.0),
        feature_transformer=LagTransformer(lag=list(range(1, 13))),
    )

    forecaster_log.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred_log = forecaster_log.predict(forecasting_horizon=forecasting_horizon)
    return forecaster_log, y_pred_log


@app.cell
def _(MeanAbsoluteError, plot_forecast, y_pred_log, y_test, y_train):
    fig_log = plot_forecast(
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred_log,
        title="Reduction Forecast with Log Transform",
    )

    mae_log = MeanAbsoluteError()
    y_test_log = y_test.head(len(y_pred_log))
    mae_log.fit(y_train)
    score_log = mae_log.score(y_test_log, y_pred_log)
    print(f"MAE with log transform: {score_log:.2f}")
    fig_log
    return fig_log, mae_log, score_log, y_test_log


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Hyperparameter Tuning with GridSearchCV

    We tune Ridge regularization (`estimator__alpha`) and lag count
    (`feature_transformer__lag`) using `GridSearchCV` with time series
    cross-validation via `ExpandingWindowSplitter`.
    """)


@app.cell
def _(
    ExpandingWindowSplitter,
    GridSearchCV,
    LagTransformer,
    LogTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    forecasting_horizon,
    y_train,
):
    forecaster_to_tune = PointReductionForecaster(
        estimator=Ridge(),
        target_transformer=LogTransformer(offset=1.0),
        feature_transformer=LagTransformer(lag=list(range(1, 13))),
    )

    param_grid = {
        "estimator__alpha": [0.1, 1.0, 10.0],
        "feature_transformer__lag": [
            list(range(1, 7)),
            list(range(1, 13)),
        ],
    }

    cv_splitter = ExpandingWindowSplitter(n_splits=2, test_size=24)

    grid_search = GridSearchCV(
        forecaster=forecaster_to_tune,
        cv=cv_splitter,
        param_grid=param_grid,
        scoring=MeanAbsoluteError(),
    )

    grid_search.fit(y_train, forecasting_horizon=forecasting_horizon)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score (MAE): {-grid_search.best_score_:.2f}")
    return cv_splitter, forecaster_to_tune, grid_search, param_grid


@app.cell
def _(grid_search, plot_cv_results_scatter):
    plot_cv_results_scatter(
        grid_search.cv_results_,
        param_name="estimator__alpha",
        title="Grid Search Results: Alpha vs CV Score",
    )


@app.cell
def _(forecasting_horizon, grid_search, plot_forecast, y_test, y_train):
    y_pred_tuned = grid_search.predict(forecasting_horizon=forecasting_horizon)

    plot_forecast(
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred_tuned,
        title="Tuned Reduction Forecast (GridSearchCV)",
    )
    return (y_pred_tuned,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Reduction forecasting** converts time series into tabular regression via lag features
    - `PointReductionForecaster` wraps any sklearn regressor for forecasting
    - **`target_transformer`**: Invertible transforms on y (e.g., `LogTransformer` for variance stabilization)
    - **`feature_transformer`**: Feature generation from y (e.g., `LagTransformer` for lags)
    - `GridSearchCV` with `ExpandingWindowSplitter` provides proper time series CV
    - Log transforms help with multiplicative seasonality (variance scaling with level)
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Naive baselines**: See `naive_forecasters.py` to compare with simple benchmarks
    - **Multi-column forecasting**: See `multi_column_forecasting.py` for multivariate data
    - **Interval prediction**: See `interval/` examples for uncertainty quantification
    - **Decomposition**: See `stationarity/` for trend/seasonality extraction before forecasting
    """)


if __name__ == "__main__":
    app.run()
