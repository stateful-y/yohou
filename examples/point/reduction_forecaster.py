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
    "title": "Reduction Forecaster",
    "description": "Learn how to turn any scikit-learn regressor into a time series forecaster using the reduction pattern.",
    "category": "tutorial",
    "companion": "/pages/explanation/forecasting/",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reduction Forecasting with sklearn

    This tutorial walks through **reduction forecasting**, the approach of
    converting time series forecasting into a supervised learning problem
    that sklearn regressors can solve.

    We will build a basic lag-based forecaster, explore how
    `target_transformer` and `feature_transformer` differ, control feature
    construction with `target_as_feature`, and tune hyperparameters
    (including `reduction_strategy`) with `GridSearchCV`.

    **Prerequisites**: Basic familiarity with sklearn's fit/predict API and
    time series concepts (trend, seasonality).
    """)
    return


@app.cell(hide_code=True)
def _():
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

    We use the Monthly Tourism dataset, which contains monthly tourist counts.
    It exhibits strong trend and seasonality, making it ideal for
    demonstrating the preprocessing techniques we will apply.
    """)
    return


@app.cell
def _(fetch_tourism_monthly):
    y = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})

    print(f"Dataset: {len(y)} observations from {y['time'].min()} to {y['time'].max()}")
    y.head()
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_time_series`](/pages/api/generated/yohou.plotting.exploration.plot_time_series/) shows the raw data, and [`plot_seasonality`](/pages/api/generated/yohou.plotting.diagnostics.plot_seasonality/) overlays each
    year's monthly values on the same seasonal axis (FPP3 gg_season style) to
    reveal the repeating yearly pattern.
    """)
    return


@app.cell
def _(plot_time_series, y):
    plot_time_series(y, title="Monthly Tourism")
    return


@app.cell
def _(plot_seasonality, y):
    plot_seasonality(y, seasonality="month", title="Monthly Seasonality Pattern")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Train/Test Split

    For time series, we must preserve temporal order. No shuffling allowed.
    We hold out the last ~20% (29 months) for testing.
    """)
    return


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

    [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) converts time series forecasting into tabular
    regression:

    1. **Feature generation**: `feature_transformer` creates lag features from past y values
    2. **Fit**: Trains an sklearn regressor on the (lags, y) tabular data
    3. **Predict**: Recursively forecasts by feeding predictions back as features

    Key distinction:
    - **`feature_transformer`**: Generates input features from y (e.g., [`LagTransformer`](/pages/api/generated/yohou.preprocessing.window.LagTransformer/) for lags). Not invertible.
    - **`target_transformer`**: Applied to y before fitting, inverted after prediction (e.g., [`LogTransformer`](/pages/api/generated/yohou.stationarity.transformers.LogTransformer/)). Must be invertible.

    We start with a simple Ridge regressor and 12 lag features.
    """)
    return


@app.cell
def _(
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    forecasting_horizon,
    y_train,
):
    forecaster = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=list(range(1, 13))),
    )

    forecaster.fit(y_train, forecasting_horizon=forecasting_horizon)
    print("Forecaster fitted successfully")
    return (forecaster,)


@app.cell
def _(forecaster, y_test):
    y_pred = forecaster.predict(forecasting_horizon=len(y_test))
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Feature Construction with `target_as_feature`

    The `target_as_feature` parameter controls what enters the feature matrix.
    With no `target_transformer`, `"transformed"` and `"raw"` produce
    identical results. The difference matters when a target_transformer
    (like log or differencing) changes the scale:

    - `"transformed"` (default): lag features from the **transformed** target
    - `"raw"`: lag features from the **original** target
    - `None`: target is **excluded** entirely, so only exogenous X is used
    """)
    return


@app.cell
def _(
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    forecasting_horizon,
    mo,
    y_test,
    y_train,
):
    taf_scores = {}
    for _taf in ["transformed", "raw"]:
        _fc = PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            target_as_feature=_taf,
            feature_transformer=LagTransformer(lag=list(range(1, 13))),
        )
        _fc.fit(y_train, forecasting_horizon=forecasting_horizon)
        _pred = _fc.predict(forecasting_horizon=len(y_test))
        _mae = MeanAbsoluteError()
        _mae.fit(y_train)
        taf_scores[_taf] = _mae.score(y_test.head(len(_pred)), _pred)

    mo.ui.table(
        [{"target_as_feature": k, "MAE": f"{v:.2f}"} for k, v in taf_scores.items()],
        label="MAE by target_as_feature (no target_transformer)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Adding Target Transformation

    Our tourism data has multiplicative seasonality (variance grows with
    level). A [`LogTransformer`](/pages/api/generated/yohou.stationarity.transformers.LogTransformer/) via `target_transformer` stabilizes this.
    It is applied to y before fitting and automatically inverted after
    prediction.
    """)
    return


@app.cell
def _(
    LagTransformer,
    LogTransformer,
    PointReductionForecaster,
    Ridge,
    forecasting_horizon,
    y_test,
    y_train,
):
    forecaster_log = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        target_transformer=LogTransformer(offset=1.0),
        feature_transformer=LagTransformer(lag=list(range(1, 13))),
    )

    forecaster_log.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred_log = forecaster_log.predict(forecasting_horizon=len(y_test))
    return (y_pred_log,)


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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Hyperparameter Tuning with GridSearchCV

    We now tune Ridge regularization (`estimator__alpha`), lag count
    (`feature_transformer__lag`), and **`reduction_strategy`** using
    [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/) with temporal cross-validation via
    [`ExpandingWindowSplitter`](/pages/api/generated/yohou.model_selection.split.ExpandingWindowSplitter/).

    Including `reduction_strategy` in the grid lets CV select the best
    strategy automatically alongside other hyperparameters.
    """)
    return


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
        "reduction_strategy": ["multi-output", "direct"],
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
    return (grid_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_cv_results_scatter`](/pages/api/generated/yohou.plotting.model_selection.plot_cv_results_scatter/) shows how the cross-validation score varies
    with the `alpha` hyperparameter. Error bars represent fold-level variation.
    """)
    return


@app.cell
def _(grid_search, plot_cv_results_scatter):
    plot_cv_results_scatter(
        grid_search.cv_results_,
        param_name="estimator__alpha",
        title="Grid Search Results: Alpha vs CV Score",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) shows the best model's predictions against the test data.
    The best hyperparameters were selected automatically by [`GridSearchCV`](/pages/api/generated/yohou.model_selection.search.GridSearchCV/).
    """)
    return


@app.cell
def _(grid_search, plot_forecast, y_test, y_train):
    y_pred_tuned = grid_search.predict(forecasting_horizon=len(y_test))

    plot_forecast(
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred_tuned,
        title="Tuned Reduction Forecast (GridSearchCV)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What We Built

    We went from a basic 12-lag Ridge forecaster to a tuned model with
    log-transformed target and grid-searched hyperparameters. Along the way
    we explored `target_as_feature` modes, the distinction between
    `target_transformer` and `feature_transformer`, and how to include
    `reduction_strategy` in a cross-validated search.
    """)
    return
