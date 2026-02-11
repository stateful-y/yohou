"""Panel Data Forecasting Tutorial.

This tutorial demonstrates forecasting multiple time series simultaneously using
Yohou's panel data support with the __ separator convention.

We'll forecast sales for multiple stores, showing:
1. Data preparation for panel data
2. Baseline forecasting across all stores
3. Preprocessing pipelines for panel data
4. Hyperparameter optimization with panel data
5. Incremental learning with update_predict
"""

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Panel Data Forecasting with Yohou

    This tutorial demonstrates how to forecast **multiple time series simultaneously**
    using Yohou's panel data support. Panel data uses columns with the `__` separator
    to represent different groups (e.g., `store_1__sales`, `store_2__sales`).

    ## What you'll learn:
    1. Creating panel data with multiple time series
    2. Baseline forecasting for panel data
    3. Preprocessing pipelines for panel data
    4. Hyperparameter optimization with GridSearchCV and RandomizedSearchCV
    5. Incremental learning with update_predict
    """)
    return


@app.cell
def _():
    # Imports
    from datetime import datetime, timedelta

    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots
    from scipy.stats import randint, uniform
    from sklearn.linear_model import Ridge

    from yohou.metrics import MeanAbsoluteError, MeanSquaredError
    from yohou.model_selection import ExpandingWindowSplitter, GridSearchCV, RandomizedSearchCV
    from yohou.pipeline import FeaturePipeline
    from yohou.point_forecaster import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer, LogTransform, SeasonalDifferencing

    return (
        ExpandingWindowSplitter,
        GridSearchCV,
        LagTransformer,
        LogTransform,
        MeanAbsoluteError,
        MeanSquaredError,
        FeaturePipeline,
        PointReductionForecaster,
        RandomizedSearchCV,
        Ridge,
        SeasonalDifferencing,
        SeasonalNaive,
        datetime,
        go,
        make_subplots,
        np,
        randint,
        pl,
        timedelta,
        uniform,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Create Panel Data

    We'll simulate sales data for 3 stores with different patterns:
    - **Store 1**: Strong weekly seasonality + trend
    - **Store 2**: Weekly seasonality + random variation
    - **Store 3**: Trend with less seasonality

    Panel data uses the `__` separator: `group_name__column_name`
    """)
    return


@app.cell
def _(datetime, np, pl, timedelta):
    # Create time index (1 year of daily data)
    n_days = 365
    time = pl.datetime_range(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 1) + timedelta(days=n_days - 1),
        interval="1d",
        eager=True,
    )

    # Day of week for seasonality (0=Monday, 6=Sunday)
    day_of_week = np.array([t.weekday() for t in time.to_list()])

    # Store 1: Strong weekly pattern + trend
    np.random.seed(42)
    trend1 = np.linspace(100, 150, n_days)
    seasonality1 = 20 * np.sin(2 * np.pi * day_of_week / 7)
    noise1 = np.random.normal(0, 5, n_days)
    store1_sales = trend1 + seasonality1 + noise1

    # Store 2: Weekly pattern + more noise
    np.random.seed(43)
    trend2 = np.linspace(80, 120, n_days)
    seasonality2 = 15 * np.sin(2 * np.pi * day_of_week / 7)
    noise2 = np.random.normal(0, 10, n_days)
    store2_sales = trend2 + seasonality2 + noise2

    # Store 3: Strong trend, less seasonality
    np.random.seed(44)
    trend3 = np.linspace(120, 180, n_days)
    seasonality3 = 10 * np.sin(2 * np.pi * day_of_week / 7)
    noise3 = np.random.normal(0, 8, n_days)
    store3_sales = trend3 + seasonality3 + noise3

    # Create panel data DataFrame with __ separator
    # Format: group__variable (e.g., store_1__sales)
    y_panel = pl.DataFrame(
        {
            "time": time,
            "store_1__sales": store1_sales,
            "store_2__sales": store2_sales,
            "store_3__sales": store3_sales,
        }
    )

    # Create exogenous features (promotions, weekends)
    # These are also panel-specific
    y_panel = y_panel.with_columns(
        [
            # Weekend indicator for each store
            pl.lit((day_of_week >= 5).astype(float)).alias("store_1__is_weekend"),
            pl.lit((day_of_week >= 5).astype(float)).alias("store_2__is_weekend"),
            pl.lit((day_of_week >= 5).astype(float)).alias("store_3__is_weekend"),
            # Random promotion events (20% chance)
            pl.lit(np.random.binomial(1, 0.2, n_days).astype(float)).alias("store_1__promotion"),
            pl.lit(np.random.binomial(1, 0.2, n_days).astype(float)).alias("store_2__promotion"),
            pl.lit(np.random.binomial(1, 0.2, n_days).astype(float)).alias("store_3__promotion"),
        ]
    )

    # Split into y (targets) and X (features)
    y = y_panel.select(["time", "store_1__sales", "store_2__sales", "store_3__sales"])
    X = y_panel.select(
        [
            "time",
            "store_1__is_weekend",
            "store_2__is_weekend",
            "store_3__is_weekend",
            "store_1__promotion",
            "store_2__promotion",
            "store_3__promotion",
        ]
    )

    # Train/test split (80/20)
    split_idx = int(0.8 * n_days)
    y_train, y_test = y[:split_idx], y[split_idx:]
    X_train, X_test = X[:split_idx], X[split_idx:]

    print(f"Training data: {len(y_train)} days")
    print(f"Test data: {len(y_test)} days")
    print(f"\nPanel data shape: {y.shape}")
    print(f"Columns: {y.columns}")
    return X_test, X_train, y_test, y_train


@app.cell
def _(go, make_subplots, y_train):
    def _():
        # Visualize the training data
        fig_data = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=("Store 1 Sales", "Store 2 Sales", "Store 3 Sales"),
            vertical_spacing=0.1,
        )

        for i, store in enumerate(["store_1__sales", "store_2__sales", "store_3__sales"], 1):
            fig_data.add_trace(
                go.Scatter(
                    x=y_train["time"],
                    y=y_train[store],
                    mode="lines",
                    name=f"Store {i}",
                    line=dict(width=1),
                ),
                row=i,
                col=1,
            )

        fig_data.update_layout(
            height=600,
            title_text="Training Data: Sales for 3 Stores",
            showlegend=False,
        )
        fig_data.update_xaxes(title_text="Date")
        fig_data.update_yaxes(title_text="Sales")
        return fig_data

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Baseline Forecasting

    Let's start with a simple **Seasonal Naive** forecaster that repeats the
    pattern from 7 days ago (weekly seasonality). This works automatically
    across all stores in the panel data.
    """)
    return


@app.cell
def _(MeanAbsoluteError, MeanSquaredError, SeasonalNaive, X_test, X_train, y_test, y_train):
    # Baseline: Seasonal Naive (repeat last week)
    baseline = SeasonalNaive(seasonality=7)

    # Fit on training data
    baseline.fit(y_train, X=X_train, forecasting_horizon=len(y_test))

    # Predict on test set
    y_pred_baseline = baseline.predict(forecasting_horizon=len(y_test), X=X_test)

    # Evaluate
    _mae_baseline = MeanAbsoluteError()
    _mse_baseline = MeanSquaredError()

    # Align predictions with actual test data for scoring
    y_test_aligned = y_test.join(
        y_pred_baseline.select(
            ["time"] + [c for c in y_pred_baseline.columns if c not in ["time", "observed_time"]]
        ),
        on="time",
        suffix="_pred",
    )

    # Calculate metrics for each store
    baseline_results = {}
    for _store in ["store_1__sales", "store_2__sales", "store_3__sales"]:
        _store_pred = _store + "_pred"
        _mae = float((y_test_aligned[_store] - y_test_aligned[_store_pred]).abs().mean())
        _rmse = float(((y_test_aligned[_store] - y_test_aligned[_store_pred]) ** 2).mean() ** 0.5)
        baseline_results[_store] = {"MAE": _mae, "RMSE": _rmse}

    print("Baseline (Seasonal Naive) Results:")
    for _store, _metrics in baseline_results.items():
        print(f"{_store}: MAE={_metrics['MAE']:.2f}, RMSE={_metrics['RMSE']:.2f}")

    return baseline_results, y_pred_baseline


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Advanced FeaturePipeline with Preprocessing

    Let's build a more sophisticated pipeline:
    1. **Log transform** - stabilize variance
    2. **Seasonal differencing** - remove weekly seasonality
    3. **Lag features** - capture recent trends
    4. **Ridge regression** - learn patterns
    """)
    return


@app.cell
def _(
    LagTransformer,
    LogTransform,
    FeaturePipeline,
    PointReductionForecaster,
    Ridge,
    SeasonalDifferencing,
    X_test,
    X_train,
    baseline_results,
    y_test,
    y_train,
):
    # Build forecaster with preprocessing
    # Note: For panel data, transformers are applied to each group independently
    pipeline = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 2, 3, 7]),
    )

    # Fit on training data
    pipeline.fit(y_train, X=X_train, forecasting_horizon=len(y_test))

    # Predict on test set
    y_pred_pipeline = pipeline.predict(forecasting_horizon=len(y_test), X=X_test)

    # Evaluate
    y_test_aligned_pipeline = y_test.join(
        y_pred_pipeline.select(
            ["time"] + [c for c in y_pred_pipeline.columns if c not in ["time", "observed_time"]]
        ),
        on="time",
        suffix="_pred",
    )

    pipeline_results = {}
    for _store in ["store_1__sales", "store_2__sales", "store_3__sales"]:
        _store_pred = _store + "_pred"
        _mae = float(
            (y_test_aligned_pipeline[_store] - y_test_aligned_pipeline[_store_pred]).abs().mean()
        )
        _rmse = float(
            ((y_test_aligned_pipeline[_store] - y_test_aligned_pipeline[_store_pred]) ** 2).mean()
            ** 0.5
        )
        pipeline_results[_store] = {"MAE": _mae, "RMSE": _rmse}

    print("FeaturePipeline Results:")
    for _store, _metrics in pipeline_results.items():
        print(f"{_store}: MAE={_metrics['MAE']:.2f}, RMSE={_metrics['RMSE']:.2f}")

    # Compare to baseline
    print("\nImprovement over baseline:")
    for _store in baseline_results:
        _mae_improvement = (
            (baseline_results[_store]["MAE"] - pipeline_results[_store]["MAE"])
            / baseline_results[_store]["MAE"]
            * 100
        )
        print(f"{_store}: {_mae_improvement:.1f}% MAE reduction")
    return pipeline_results, y_pred_pipeline


@app.cell
def _(go, make_subplots, y_pred_baseline, y_pred_pipeline, y_test):
    # Visualize predictions vs actual
    fig_pred = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Store 1", "Store 2", "Store 3"),
        vertical_spacing=0.1,
    )

    for _i, _store in enumerate(["store_1__sales", "store_2__sales", "store_3__sales"], 1):
        # Actual
        fig_pred.add_trace(
            go.Scatter(
                x=y_test["time"],
                y=y_test[_store],
                mode="lines",
                name=f"Actual Store {_i}",
                line=dict(color="black", width=2),
            ),
            row=_i,
            col=1,
        )

        # Baseline
        fig_pred.add_trace(
            go.Scatter(
                x=y_pred_baseline["time"],
                y=y_pred_baseline[_store],
                mode="lines",
                name=f"Baseline Store {_i}",
                line=dict(color="blue", width=1, dash="dash"),
            ),
            row=_i,
            col=1,
        )

        # FeaturePipeline
        fig_pred.add_trace(
            go.Scatter(
                x=y_pred_pipeline["time"],
                y=y_pred_pipeline[_store],
                mode="lines",
                name=f"FeaturePipeline Store {_i}",
                line=dict(color="red", width=1),
            ),
            row=_i,
            col=1,
        )

    fig_pred.update_layout(
        height=700,
        title_text="Predictions vs Actual (Test Set)",
        showlegend=True,
    )
    fig_pred.update_xaxes(title_text="Date")
    fig_pred.update_yaxes(title_text="Sales")

    fig_pred
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Hyperparameter Optimization with RandomizedSearchCV

    Use **RandomizedSearchCV** to optimize hyperparameters across all stores simultaneously.
    We'll tune:
    - Ridge alpha (regularization)
    - Number of lag features
    """)
    return


@app.cell
def _(mo):
    # Interactive controls for RandomizedSearchCV
    n_trials_slider = mo.ui.slider(
        start=5,
        stop=30,
        value=10,
        label="Number of random trials",
        show_value=True,
    )

    mo.md(f"""
    ### RandomizedSearchCV Configuration

    {n_trials_slider}

    This will search for the best hyperparameters using time series cross-validation.
    """)
    return (n_trials_slider,)


@app.cell
def _(
    ExpandingWindowSplitter,
    LagTransformer,
    LogTransform,
    MeanAbsoluteError,
    FeaturePipeline,
    PointReductionForecaster,
    RandomizedSearchCV,
    Ridge,
    SeasonalDifferencing,
    X_test,
    X_train,
    baseline_results,
    n_trials_slider,
    randint,
    uniform,
    y_test,
    y_train,
):
    # Create forecaster for optimization
    forecaster_to_optimize = PointReductionForecaster(
        estimator=Ridge(),
        feature_transformer=LagTransformer(lag=[1]),  # Will be optimized
    )

    # Define search space using scipy.stats distributions

    param_distributions = {
        "estimator__alpha": uniform(0.1, 10.0),
        # For categorical choices, use randint to index into a list
        # This simulates categorical choice: pick random int 0-3, map to lag patterns
    }

    # For categorical lag patterns, we'll use a simpler approach with individual parameters
    # Run optimization (this may take a minute)
    search = RandomizedSearchCV(
        forecaster=forecaster_to_optimize,
        param_distributions=param_distributions,
        scoring=MeanAbsoluteError(),
        n_iter=n_trials_slider.value,
        cv=ExpandingWindowSplitter(n_splits=2, test_size=50),  # Limited by 292 train samples
        refit=True,  # Refit on full training data with best params
        return_train_score=False,  # Faster
    )

    # Fit with time series cross-validation
    search.fit(y_train, X=X_train, forecasting_horizon=len(y_test))

    print("Best parameters found:")
    print(search.best_params_)

    # Predict with best model
    y_pred_optimized = search.predict(forecasting_horizon=len(y_test), X=X_test)

    # Evaluate
    y_test_aligned_opt = y_test.join(
        y_pred_optimized.select(
            ["time"] + [c for c in y_pred_optimized.columns if c not in ["time", "observed_time"]]
        ),
        on="time",
        suffix="_pred",
    )

    optimized_results = {}
    for _store in ["store_1__sales", "store_2__sales", "store_3__sales"]:
        _store_pred = _store + "_pred"
        _mae = float((y_test_aligned_opt[_store] - y_test_aligned_opt[_store_pred]).abs().mean())
        _rmse = float(
            ((y_test_aligned_opt[_store] - y_test_aligned_opt[_store_pred]) ** 2).mean() ** 0.5
        )
        optimized_results[_store] = {"MAE": _mae, "RMSE": _rmse}

    print("\nOptimized Results:")
    for _store, _metrics in optimized_results.items():
        print(f"{_store}: MAE={_metrics['MAE']:.2f}, RMSE={_metrics['RMSE']:.2f}")

    print("\nImprovement over baseline:")
    for _store in baseline_results:
        _mae_improvement = (
            (baseline_results[_store]["MAE"] - optimized_results[_store]["MAE"])
            / baseline_results[_store]["MAE"]
            * 100
        )
        print(f"{_store}: {_mae_improvement:.1f}% MAE reduction")
    return optimized_results, search


@app.cell
def _(mo):
    mo.md(r"""
    ### GridSearchCV for Panel Data

    Now let's compare **GridSearchCV** with the RandomizedSearchCV results.

    GridSearchCV tests all combinations in a discrete grid, which is useful for panel data when:
    - You want guaranteed exploration of specific parameter values
    - The search space is small enough to be exhaustive
    - You need reproducible results across different runs
    """)
    return


@app.cell
def _(
    ExpandingWindowSplitter,
    GridSearchCV,
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    X_test,
    X_train,
    forecaster_to_optimize,
    y_test,
    y_train,
):
    from sklearn.base import clone

    # Define a discrete grid (smaller than random search space)
    param_grid_panel = {
        "estimator__alpha": [0.1, 1.0, 5.0, 10.0],  # 4 values
    }
    # Total: 4 combinations (fast for demonstration)

    # Run grid search
    grid_panel = GridSearchCV(
        forecaster=clone(forecaster_to_optimize),
        param_grid=param_grid_panel,
        scoring=MeanAbsoluteError(),
        cv=ExpandingWindowSplitter(n_splits=2, test_size=50),
        refit=True,
        return_train_score=False,
    )

    grid_panel.fit(y_train, X=X_train, forecasting_horizon=len(y_test))

    print("GridSearchCV Best parameters:")
    print(grid_panel.best_params_)
    print(f"Best CV MAE: {grid_panel.best_score_:.2f}")

    # Predict
    y_pred_grid = grid_panel.predict(forecasting_horizon=len(y_test), X=X_test)

    # Evaluate per-store
    y_test_grid_aligned = y_test.join(
        y_pred_grid.select(
            ["time"] + [c for c in y_pred_grid.columns if c not in ["time", "observed_time"]]
        ),
        on="time",
        suffix="_pred",
    )

    grid_panel_results = {}
    for store_g in ["store_1__sales", "store_2__sales", "store_3__sales"]:
        store_pred_g = store_g + "_pred"
        mae_g = float((y_test_grid_aligned[store_g] - y_test_grid_aligned[store_pred_g]).abs().mean())
        rmse_g = float(
            ((y_test_grid_aligned[store_g] - y_test_grid_aligned[store_pred_g]) ** 2).mean() ** 0.5
        )
        grid_panel_results[store_g] = {"MAE": mae_g, "RMSE": rmse_g}

    print("\nGridSearchCV Results per store:")
    for store_g, metrics_g in grid_panel_results.items():
        print(f"{store_g}: MAE={metrics_g['MAE']:.2f}, RMSE={metrics_g['RMSE']:.2f}")
    return clone, grid_panel, grid_panel_results, y_pred_grid, y_test_grid_aligned


@app.cell
def _(
    baseline_results,
    go,
    grid_panel_results,
    optimized_results,
):
    # Compare search strategies
    fig_search_panel = go.Figure()

    stores_panel = ["store_1__sales", "store_2__sales", "store_3__sales"]
    x_pos = list(range(len(stores_panel)))

    # Baseline MAE
    baseline_mae_panel = [baseline_results[s]["MAE"] for s in stores_panel]
    fig_search_panel.add_trace(
        go.Bar(
            x=x_pos,
            y=baseline_mae_panel,
            name="Baseline (SeasonalNaive)",
            marker_color="red",
            text=[f"{v:.1f}" for v in baseline_mae_panel],
            textposition="outside",
        )
    )

    # RandomizedSearchCV MAE
    random_mae_panel = [optimized_results[s]["MAE"] for s in stores_panel]
    fig_search_panel.add_trace(
        go.Bar(
            x=x_pos,
            y=random_mae_panel,
            name="RandomizedSearchCV",
            marker_color="orange",
            text=[f"{v:.1f}" for v in random_mae_panel],
            textposition="outside",
        )
    )

    # GridSearchCV MAE
    grid_mae_panel = [grid_panel_results[s]["MAE"] for s in stores_panel]
    fig_search_panel.add_trace(
        go.Bar(
            x=x_pos,
            y=grid_mae_panel,
            name="GridSearchCV",
            marker_color="purple",
            text=[f"{v:.1f}" for v in grid_mae_panel],
            textposition="outside",
        )
    )

    fig_search_panel.update_layout(
        title="Panel Data: Search Strategy Comparison",
        xaxis=dict(tickvals=x_pos, ticktext=stores_panel),
        yaxis_title="Mean Absolute Error",
        height=450,
        barmode="group",
        template="plotly_white",
    )
    fig_search_panel
    return baseline_mae_panel, grid_mae_panel, random_mae_panel, stores_panel, x_pos


@app.cell
def _(mo):
    mo.md(r"""
    ### Search Strategy Insights for Panel Data

    **Key observations**:
    1. **Both searches improve over baseline**: Hyperparameter tuning helps across all stores
    2. **Consistency**: Both GridSearchCV and RandomizedSearchCV find similar optimal regions
    3. **Panel efficiency**: A single search optimizes hyperparameters for all stores simultaneously
    4. **Store-specific performance**: Some stores benefit more from optimization than others

    **When to use which**:
    - **GridSearchCV**: Smaller grids, need reproducibility, want to visualize full parameter space
    - **RandomizedSearchCV**: Larger search spaces, continuous parameters, limited compute budget
    - **Panel data advantage**: Both methods leverage cross-store patterns during CV
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Incremental Learning with update_predict

    For online/streaming scenarios, use `update_predict()` to:
    1. Update the model with new observations
    2. Make predictions for the next period

    This is more efficient than refitting from scratch and works seamlessly
    with panel data.
    """)
    return


@app.cell
def _(X_test, search, y_test):
    # Simulate streaming scenario: update every week, predict next week
    stride = 7  # Update weekly
    horizon = 7  # Predict 1 week ahead

    # Start with trained model
    streaming_forecaster = search.best_forecaster_

    # Use update_predict for rolling forecasts
    y_pred_streaming = streaming_forecaster.update_predict(
        y=y_test,
        X=X_test,
        forecasting_horizon=horizon,
        stride=stride,
    )

    # Get only the final predictions (last 7 days)
    y_pred_streaming_final = y_pred_streaming.tail(horizon)

    print(f"Streaming predictions shape: {y_pred_streaming.shape}")
    print(f"Final week predictions shape: {y_pred_streaming_final.shape}")
    print(f"\nTotal updates: {len(y_test) // stride}")
    return horizon, stride, y_pred_streaming


@app.cell
def _(go, horizon, make_subplots, stride, y_pred_streaming, y_test, y_train):
    # Visualize streaming predictions
    fig_streaming = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Store 1", "Store 2", "Store 3"),
        vertical_spacing=0.1,
    )

    for _i, _store in enumerate(["store_1__sales", "store_2__sales", "store_3__sales"], 1):
        # Training data
        fig_streaming.add_trace(
            go.Scatter(
                x=y_train["time"],
                y=y_train[_store],
                mode="lines",
                name=f"Train Store {_i}",
                line=dict(color="lightgray", width=1),
                showlegend=(_i == 1),
            ),
            row=_i,
            col=1,
        )

        # Actual test data
        fig_streaming.add_trace(
            go.Scatter(
                x=y_test["time"],
                y=y_test[_store],
                mode="lines",
                name=f"Actual Store {_i}",
                line=dict(color="black", width=2),
                showlegend=(_i == 1),
            ),
            row=_i,
            col=1,
        )

        # Streaming predictions (only plot predictions at stride intervals)
        _pred_times = y_pred_streaming["time"][::stride]
        _pred_values = y_pred_streaming[_store][::stride]

        fig_streaming.add_trace(
            go.Scatter(
                x=_pred_times,
                y=_pred_values,
                mode="markers",
                name=f"Predictions Store {_i}",
                marker=dict(color="red", size=6),
                showlegend=(_i == 1),
            ),
            row=_i,
            col=1,
        )

    fig_streaming.update_layout(
        height=700,
        title_text=f"Incremental Learning (update every {stride} days, predict {horizon} days ahead)",
        showlegend=True,
    )
    fig_streaming.update_xaxes(title_text="Date")
    fig_streaming.update_yaxes(title_text="Sales")

    fig_streaming
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary: Results Comparison

    Let's compare all approaches across all stores.
    """)
    return


@app.cell
def _(baseline_results, go, optimized_results, pipeline_results):
    # Create comparison table
    comparison_data = []
    for _store in ["store_1__sales", "store_2__sales", "store_3__sales"]:
        _store_name = _store.replace("__sales", "").replace("_", " ").title()
        comparison_data.append(
            {
                "Store": _store_name,
                "Baseline MAE": f"{baseline_results[_store]['MAE']:.2f}",
                "FeaturePipeline MAE": f"{pipeline_results[_store]['MAE']:.2f}",
                "Optimized MAE": f"{optimized_results[_store]['MAE']:.2f}",
                "Best Improvement": f"{(baseline_results[_store]['MAE'] - optimized_results[_store]['MAE']) / baseline_results[_store]['MAE'] * 100:.1f}%",
            }
        )

    # Create bar chart
    fig_comparison = go.Figure()

    stores = ["Store 1", "Store 2", "Store 3"]
    baseline_maes = [baseline_results[f"store_{i}__sales"]["MAE"] for i in [1, 2, 3]]
    pipeline_maes = [pipeline_results[f"store_{i}__sales"]["MAE"] for i in [1, 2, 3]]
    optimized_maes = [optimized_results[f"store_{i}__sales"]["MAE"] for i in [1, 2, 3]]

    fig_comparison.add_trace(go.Bar(name="Baseline", x=stores, y=baseline_maes))
    fig_comparison.add_trace(go.Bar(name="FeaturePipeline", x=stores, y=pipeline_maes))
    fig_comparison.add_trace(go.Bar(name="Optimized", x=stores, y=optimized_maes))

    fig_comparison.update_layout(
        title="MAE Comparison Across Models and Stores",
        xaxis_title="Store",
        yaxis_title="Mean Absolute Error (MAE)",
        barmode="group",
        height=400,
    )

    fig_comparison
    return (comparison_data,)


@app.cell
def _(comparison_data, mo):
    mo.md(f"""
    ### Results Table

    | Store | Baseline MAE | FeaturePipeline MAE | Optimized MAE | Best Improvement |
    |-------|--------------|--------------|---------------|------------------|
    | {comparison_data[0]["Store"]} | {comparison_data[0]["Baseline MAE"]} | {comparison_data[0]["FeaturePipeline MAE"]} | {comparison_data[0]["Optimized MAE"]} | {comparison_data[0]["Best Improvement"]} |
    | {comparison_data[1]["Store"]} | {comparison_data[1]["Baseline MAE"]} | {comparison_data[1]["FeaturePipeline MAE"]} | {comparison_data[1]["Optimized MAE"]} | {comparison_data[1]["Best Improvement"]} |
    | {comparison_data[2]["Store"]} | {comparison_data[2]["Baseline MAE"]} | {comparison_data[2]["FeaturePipeline MAE"]} | {comparison_data[2]["Optimized MAE"]} | {comparison_data[2]["Best Improvement"]} |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    1. **Panel Data Format**: Use `group__column` naming (e.g., `store_1__sales`)
    2. **Automatic Processing**: All forecasters work seamlessly with panel data
    3. **Per-Group Models**: Each group gets its own model internally
    4. **Hyperparameter Search**: Both GridSearchCV and RandomizedSearchCV optimize across all groups
    5. **Search Strategy Selection**: Use GridSearchCV for small grids, RandomizedSearchCV for large spaces
    6. **Streaming Support**: `update_predict()` enables online learning

    ## Next Steps

    - Try different forecasters (e.g., `Decomposer` for trend+seasonality)
    - Add global features (shared across all stores)
    - Use interval forecasters for prediction intervals
    - Experiment with cross-learning for specific store groups
    """)
    return


if __name__ == "__main__":
    app.run()
