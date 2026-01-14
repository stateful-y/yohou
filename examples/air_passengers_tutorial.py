import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 📊 Air Passenger Forecasting with Yohou

    Welcome to this tutorial on time series forecasting using **yohou**, a scikit-learn-compatible forecasting framework built on polars.

    ## What You'll Learn

    In this tutorial, we'll use the classic **Air Passengers dataset** (monthly totals from 1949-1960) to learn:

    - **Basic forecasting workflow**: Load data, create train/test splits, and make predictions
    - **Baseline models**: Start with simple approaches like seasonal naive forecasting
    - **Preprocessing pipelines**: Transform data to improve forecast accuracy
    - **Hyperparameter tuning**: Use cross-validation and Optuna to optimize models
    - **Incremental learning**: Update models with new data without full retraining

    ## About the Dataset

    The Air Passengers dataset shows monthly totals of international airline passengers from 1949 to 1960. It's a classic time series that exhibits:
    - Strong upward trend (growth in commercial aviation)
    - Clear yearly seasonality (summer travel peaks)
    - Increasing variance over time (multiplicative seasonality)

    Let's get started! 🚀
    """)
    return


@app.cell
def _():
    # Import required libraries

    import optuna
    import plotly.graph_objects as go
    import polars as pl
    from sklearn.base import clone
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    from yohou.metrics import MAE
    from yohou.model_selection import SearchCV, Splitter
    from yohou.pipeline import FeaturePipeline

    # Yohou imports
    from yohou.point_forecaster import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer, LogTransform, SeasonalDifferencing

    return (
        FeaturePipeline,
        LagTransformer,
        LogTransform,
        MAE,
        PointReductionForecaster,
        Ridge,
        SearchCV,
        SeasonalDifferencing,
        SeasonalNaive,
        Splitter,
        clone,
        go,
        optuna,
        pl,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading the Data

    We'll load the Air Passengers dataset from a public CSV source. The dataset contains:
    - `Month`: Date string in format "YYYY-MM"
    - `Passengers`: Number of passengers (in thousands)

    We need to convert this into a polars DataFrame with:
    - A `"time"` column (datetime type) - required by yohou
    - A `"passengers"` column (float type) - our target variable
    """)
    return


@app.cell
def _(pl):
    # Load air passengers data
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"

    # Read CSV and prepare data
    df_raw = pl.read_csv(url)

    # Convert Month string to datetime and rename columns
    y = df_raw.with_columns(
        [
            pl.col("Month").str.strptime(pl.Datetime, format="%Y-%m").alias("time"),
            pl.col("Passengers").cast(pl.Float64).alias("passengers"),
        ]
    ).select(["time", "passengers"])

    # Display first and last rows
    print("First 5 rows:")
    print(y.head())
    print("\nLast 5 rows:")
    print(y.tail())
    print(f"\nDataset shape: {y.shape[0]} rows, {y.shape[1]} columns")
    print(f"\nDate range: {y['time'].min()} to {y['time'].max()}")
    return (y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualizing the Time Series

    Before building models, let's visualize the data to understand its characteristics. Look for:
    - **Trend**: Is there an overall upward or downward movement?
    - **Seasonality**: Do patterns repeat at regular intervals?
    - **Variance**: Does the spread of values change over time?
    """)
    return


@app.cell(hide_code=True)
def _(go, y):
    # Create time series plot
    fig_raw = go.Figure()
    fig_raw.add_trace(
        go.Scatter(
            x=y["time"].to_list(),
            y=y["passengers"].to_list(),
            mode="lines",
            name="Passengers",
            line=dict(color="royalblue", width=2),
        )
    )
    fig_raw.update_layout(
        title="Air Passengers Time Series (1949-1960)",
        xaxis_title="Time",
        yaxis_title="Passengers (thousands)",
        height=400,
        hovermode="x unified",
    )
    fig_raw
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What We Observe

    Looking at the plot, we can clearly see:

    1. **Strong upward trend**: Passenger numbers grow consistently from ~100k to ~600k
    2. **Yearly seasonality**: Regular peaks during summer months (June-August)
    3. **Increasing variance**: The seasonal fluctuations get larger as the trend increases (multiplicative seasonality)

    These patterns suggest we'll need preprocessing (stationarization) to build accurate models.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train/Test Split

    For time series forecasting, we must preserve temporal order when splitting data:
    - **No shuffling**: Test data must come after training data
    - **Test size**: We'll use ~20% (about 2 years) to simulate future forecasting
    - **Forecasting horizon**: We'll predict 12 months ahead to capture a full seasonal cycle
    """)
    return


@app.cell
def _(train_test_split, y):
    # Split data temporally (no shuffle!)
    y_train, y_test = train_test_split(y, test_size=0.2, shuffle=False)

    # Define forecasting horizon
    forecasting_horizon = 12  # 12 months ahead

    print(
        f"Training set: {len(y_train)} observations ({y_train['time'].min()} to {y_train['time'].max()})"
    )
    print(
        f"Test set: {len(y_test)} observations ({y_test['time'].min()} to {y_test['time'].max()})"
    )
    print(f"\nForecasting horizon: {forecasting_horizon} months")
    return forecasting_horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline Forecasting with Seasonal Naive

    Let's start with the simplest possible approach: **Seasonal Naive Forecasting**.

    **How it works**: Repeat the values from the same season one year ago (seasonality=12 for monthly data).

    **When to use it**:
    - Quick baseline to beat with more complex models
    - Strong seasonality dominates other patterns
    - Production fallback when complex models fail

    **Limitations**:
    - Ignores trend continuation
    - Can't adapt to pattern changes
    - Only uses last seasonal cycle
    """)
    return


@app.cell
def _(MAE, SeasonalNaive, forecasting_horizon, y_test, y_train):
    # Create and fit baseline model
    baseline = SeasonalNaive(seasonality=12)
    baseline.fit(y_train, X=None, forecasting_horizon=forecasting_horizon)

    # Make predictions
    y_pred_baseline = baseline.predict(forecasting_horizon=forecasting_horizon)

    # Evaluate
    mae_scorer = MAE()
    mae_baseline = mae_scorer.score(y_test, y_pred_baseline)

    print(f"Baseline (Seasonal Naive) MAE: {mae_baseline:.2f} thousand passengers")
    return mae_baseline, mae_scorer, y_pred_baseline


@app.cell(hide_code=True)
def _(go, y_pred_baseline, y_test, y_train):
    # Visualize baseline predictions
    fig_baseline = go.Figure()

    # Training data
    fig_baseline.add_trace(
        go.Scatter(
            x=y_train["time"].to_list(),
            y=y_train["passengers"].to_list(),
            mode="lines",
            name="Training data",
            line=dict(color="royalblue", width=2),
        )
    )

    # Test data (actual)
    fig_baseline.add_trace(
        go.Scatter(
            x=y_test["time"].to_list(),
            y=y_test["passengers"].to_list(),
            mode="lines",
            name="Actual (test)",
            line=dict(color="green", width=2),
        )
    )

    # Predictions
    fig_baseline.add_trace(
        go.Scatter(
            x=y_pred_baseline["time"].to_list(),
            y=y_pred_baseline["passengers"].to_list(),
            mode="lines+markers",
            name="Baseline predictions",
            line=dict(color="red", width=2, dash="dash"),
            marker=dict(size=6),
        )
    )

    # Add vertical line at train/test split
    fig_baseline.add_vline(
        x=y_test["time"].min().timestamp() * 1000,
        line_dash="dot",
        line_color="gray",
        annotation_text="Train/Test Split",
    )

    fig_baseline.update_layout(
        title="Baseline Forecasting Results",
        xaxis_title="Time",
        yaxis_title="Passengers (thousands)",
        height=400,
        hovermode="x unified",
    )
    fig_baseline
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Baseline Results Interpretation

    The baseline model provides a reasonable starting point:
    - ✅ Captures the seasonal pattern (summer peaks)
    - ❌ Misses the continued upward trend (predictions are too low)
    - ❌ Simply repeats last year's values without adaptation

    **Our goal**: Build more sophisticated models that beat this baseline MAE while remaining interpretable and robust.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 🚀 Advanced Forecasting Techniques

    Now that we have a baseline, let's explore more sophisticated approaches:

    1. **Preprocessing pipelines** with stationarization transformations
    2. **Interactive parameter exploration** using marimo's reactive UI
    3. **Hyperparameter optimization** with cross-validation and Optuna
    4. **Incremental learning** for production scenarios

    These techniques will help us build more accurate and robust forecasting models.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive Parameter Controls

    Marimo notebooks are **reactive**: when you change a slider, all dependent cells automatically re-run!

    Use these controls to explore how different parameters affect model performance:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Interactive parameter controls
    forecast_horizon_slider = mo.ui.slider(
        start=1, stop=24, value=12, label="Forecast Horizon (months)", show_value=True
    )

    max_lag_slider = mo.ui.slider(
        start=1, stop=24, value=3, label="Maximum Lag Window", show_value=True
    )

    params = mo.ui.dictionary(
        {"forecast_horizon": forecast_horizon_slider, "max_lag": max_lag_slider}
    )

    params
    return (params,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preprocessing FeaturePipeline with Stationarization

    **The Problem**: Our time series is non-stationary (trend + multiplicative seasonality), making it hard for linear models to learn patterns.

    **The Solution**: Build a preprocessing pipeline that transforms the data into a stationary form:

    1. **LogTransform**: Stabilizes variance (converts multiplicative → additive seasonality)
    2. **SeasonalDifferencing**: Removes trend and seasonality by subtracting seasonal lags
    3. **LagTransformer**: Creates features from historical values (lag-1, lag-2, lag-3, lag-12)
    4. **PointReductionForecaster**: Converts time series forecasting to supervised learning with Ridge regression

    **Key advantage**: All transformations are **invertible**, so predictions are automatically transformed back to the original scale!
    """)
    return


@app.cell
def _(
    FeaturePipeline,
    LagTransformer,
    LogTransform,
    PointReductionForecaster,
    Ridge,
    SeasonalDifferencing,
    mae_baseline,
    mae_scorer,
    params,
    y_test,
    y_train,
):
    # Build preprocessing pipeline (reactive to slider values)
    pipeline_target = FeaturePipeline(
        [
            ("log", LogTransform(offset=1.0)),  # Add offset to handle zero values
            ("diff", SeasonalDifferencing(seasonality=12)),  # Remove yearly seasonality
        ]
    )

    pipeline_feature = FeaturePipeline(
        [
            ("log", LogTransform(offset=1.0)),  # Add offset to handle zero values
            ("diff", SeasonalDifferencing(seasonality=12)),  # Remove yearly seasonality
            (
                "lag",
                LagTransformer(lag=list(range(1, params.value["max_lag"] + 1))),
            ),  # Create lag features
        ]
    )

    reduction_forecaster = PointReductionForecaster(
        estimator=Ridge(alpha=10),
        target_transformer=pipeline_target,
        feature_transformer=pipeline_feature,
    )

    # Fit the forecaster
    reduction_forecaster.fit(
        y=y_train,
        X=None,
        forecasting_horizon=params.value["forecast_horizon"],
    )

    # Make predictions
    y_pred_reduction_forecaster = reduction_forecaster.predict(
        forecasting_horizon=params.value["forecast_horizon"]
    )

    # Evaluate
    mae_reduction_forecaster = mae_scorer.score(y_test, y_pred_reduction_forecaster)

    # Calculate improvement
    improvement = ((mae_baseline - mae_reduction_forecaster) / mae_baseline) * 100

    print(f"Reducer MAE: {mae_reduction_forecaster:.2f} thousand passengers")
    print(f"Baseline MAE: {mae_baseline:.2f} thousand passengers")
    print(f"Improvement: {improvement:.1f}%")
    return (
        improvement,
        mae_reduction_forecaster,
        reduction_forecaster,
        y_pred_reduction_forecaster,
    )


@app.cell(hide_code=True)
def _(
    go,
    improvement,
    mae_baseline,
    mae_reduction_forecaster,
    y_pred_baseline,
    y_pred_reduction_forecaster,
    y_test,
    y_train,
):
    # Visualize pipeline predictions vs baseline
    fig_reduction_forecaster = go.Figure()

    # Training data
    fig_reduction_forecaster.add_trace(
        go.Scatter(
            x=y_train["time"].to_list(),
            y=y_train["passengers"].to_list(),
            mode="lines",
            name="Training data",
            line=dict(color="royalblue", width=2),
        )
    )

    # Test data (actual)
    fig_reduction_forecaster.add_trace(
        go.Scatter(
            x=y_test["time"].to_list(),
            y=y_test["passengers"].to_list(),
            mode="lines",
            name="Actual (test)",
            line=dict(color="green", width=2),
        )
    )

    # Baseline predictions
    fig_reduction_forecaster.add_trace(
        go.Scatter(
            x=y_pred_baseline["time"].to_list(),
            y=y_pred_baseline["passengers"].to_list(),
            mode="lines+markers",
            name=f"Baseline (MAE: {mae_baseline:.2f})",
            line=dict(color="red", width=2, dash="dash"),
            marker=dict(size=6),
        )
    )

    # FeaturePipeline predictions
    fig_reduction_forecaster.add_trace(
        go.Scatter(
            x=y_pred_reduction_forecaster["time"].to_list(),
            y=y_pred_reduction_forecaster["passengers"].to_list(),
            mode="lines+markers",
            name=f"Reducer (MAE: {mae_reduction_forecaster:.2f}, {improvement:.1f}%)",
            line=dict(color="purple", width=2, dash="dash"),
            marker=dict(size=6, symbol="diamond"),
        )
    )

    # Add vertical line at train/test split
    fig_reduction_forecaster.add_vline(
        x=y_test["time"].min().timestamp() * 1000,
        line_dash="dot",
        line_color="gray",
        annotation_text="Train/Test Split",
    )

    fig_reduction_forecaster.update_layout(
        title="Reducer vs Baseline Forecasting",
        xaxis_title="Time",
        yaxis_title="Passengers (thousands)",
        height=400,
        hovermode="x unified",
    )
    fig_reduction_forecaster
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### FeaturePipeline Results Interpretation

    The preprocessing pipeline typically improves forecast accuracy by:
    - ✅ **Capturing trend continuation**: Differencing helps model learn growth rate
    - ✅ **Stabilizing variance**: Log transform makes seasonality additive
    - ✅ **Learning complex patterns**: Lag features enable the model to use recent history

    **When preprocessing helps most**:
    - Non-stationary data (trend, seasonality)
    - Linear models (Ridge, Lasso, ElasticNet)
    - Interpretability requirements (transformations are invertible)

    **When it's optional**:
    - Tree-based models (can handle non-stationarity naturally)
    - Deep learning approaches (learn transformations automatically)

    **Try adjusting the sliders above** to see how forecast horizon and lag window affect performance!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cross-Validation and Hyperparameter Tuning

    So far, we've manually chosen hyperparameters (lag windows, Ridge alpha, differencing period). Let's use **cross-validation** to find optimal values systematically.

    ### Time Series Cross-Validation

    Unlike standard CV, time series CV uses an **expanding window**:
    - Training set grows with each fold
    - Test set always comes after training set
    - No data leakage from future to past

    ### Hyperparameter Search with Optuna

    We'll use `SearchCV` to optimize:
    - **Ridge alpha**: Regularization strength (prevents overfitting)
    - **Lag window**: Number of past months to use as features

    **⚠️ Important CV Constraints**:
    - With 115 training samples and forecasting_horizon=12, CV folds must be carefully sized
    - Each fold needs enough data for: `lag + forecasting_horizon + buffer`
    - We use 2 splits to ensure adequate training data in each fold

    ⚠️ **Note**: This will take ~30 seconds (10 trials × 2 CV folds × 4 parallel jobs).
    """)
    return


@app.cell(hide_code=True)
def _(
    MAE,
    SearchCV,
    Splitter,
    clone,
    forecasting_horizon,
    mae_baseline,
    optuna,
    reduction_forecaster,
    y_train,
):
    # Set up cross-validation with carefully chosen parameters
    # With 115 samples: 2 splits gives folds of ~57 and ~77 training samples
    # Each fold must support: lag (max 6) + horizon (12) + reset buffer
    cv_splitter = Splitter(n_splits=2, test_size=forecasting_horizon)

    # Define search space (constrained to work with available data)
    param_distributions = {
        "estimator__alpha": optuna.distributions.FloatDistribution(0.01, 10.0, log=True),
        # Keep lag small (1-6) to work with CV fold sizes
        "feature_transformer__lag__lag": optuna.distributions.IntDistribution(1, 6),
    }

    # Set up hyperparameter search
    search = SearchCV(
        forecaster=clone(reduction_forecaster),
        param_distributions=param_distributions,
        scoring=MAE(),
        cv=cv_splitter,
        n_trials=10,  # Reduced for faster execution
        refit=True,
        return_train_score=False,  # Disable to avoid extra data requirements
        n_jobs=4,
    )

    # Run search
    print("Running hyperparameter search...")
    search.fit(y_train, X=None, forecasting_horizon=forecasting_horizon)

    print("\n✅ Search complete!")
    print("\nBest parameters found:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest cross-validation MAE: {search.best_score_:.2f}")
    print(f"Baseline MAE (for comparison): {mae_baseline:.2f}")
    return (search,)


@app.cell
def _(search):
    search.cv_results_
    return


@app.cell
def _(forecasting_horizon, mae_scorer, search, y_test):
    # Make predictions with the best forecaster
    y_pred_best = search.predict(forecasting_horizon=forecasting_horizon)

    # Evaluate
    mae_best = mae_scorer.score(y_test, y_pred_best)

    print(f"Best forecaster MAE: {mae_best:.2f} thousand passengers")
    return mae_best, y_pred_best


@app.cell(hide_code=True)
def _(
    go,
    mae_baseline,
    mae_best,
    mae_reduction_forecaster,
    y_pred_baseline,
    y_pred_best,
    y_pred_reduction_forecaster,
    y_test,
    y_train,
):
    # Visualize best forecaster vs untuned vs baseline
    fig_comparison = go.Figure()

    # Training data
    fig_comparison.add_trace(
        go.Scatter(
            x=y_train["time"].to_list(),
            y=y_train["passengers"].to_list(),
            mode="lines",
            name="Training data",
            line=dict(color="royalblue", width=2),
        )
    )

    # Test data (actual)
    fig_comparison.add_trace(
        go.Scatter(
            x=y_test["time"].to_list(),
            y=y_test["passengers"].to_list(),
            mode="lines",
            name="Actual (test)",
            line=dict(color="green", width=3),
        )
    )

    # Baseline predictions
    fig_comparison.add_trace(
        go.Scatter(
            x=y_pred_baseline["time"].to_list(),
            y=y_pred_baseline["passengers"].to_list(),
            mode="lines+markers",
            name=f"Baseline (MAE: {mae_baseline:.2f})",
            line=dict(color="red", width=2, dash="dash"),
            marker=dict(size=6),
        )
    )

    # Untuned pipeline predictions
    fig_comparison.add_trace(
        go.Scatter(
            x=y_pred_reduction_forecaster["time"].to_list(),
            y=y_pred_reduction_forecaster["passengers"].to_list(),
            mode="lines+markers",
            name=f"Untuned Reducer (MAE: {mae_reduction_forecaster:.2f})",
            line=dict(color="purple", width=2, dash="dash"),
            marker=dict(size=6, symbol="diamond"),
        )
    )

    # Best forecaster predictions
    fig_comparison.add_trace(
        go.Scatter(
            x=y_pred_best["time"].to_list(),
            y=y_pred_best["passengers"].to_list(),
            mode="lines+markers",
            name=f"Optimized Reducer (MAE: {mae_best:.2f})",
            line=dict(color="orange", width=2, dash="dot"),
            marker=dict(size=7, symbol="star"),
        )
    )

    # Add vertical line at train/test split
    fig_comparison.add_vline(
        x=y_test["time"].min().timestamp() * 1000,
        line_dash="dot",
        line_color="gray",
        annotation_text="Train/Test Split",
    )

    fig_comparison.update_layout(
        title="Model Comparison: Baseline vs. Untuned vs. Optimized",
        xaxis_title="Time",
        yaxis_title="Passengers (thousands)",
        height=500,
        hovermode="x unified",
    )
    fig_comparison
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hyperparameter Tuning Results

    The comparison shows the impact of hyperparameter optimization:

    **Performance progression**:
    1. **Baseline (Seasonal Naive)**: Simple but misses trend
    2. **Untuned Reducer**: Better with default parameters
    3. **Optimized Reducer**: Best performance with tuned hyperparameters

    **Key insights**:
    - The search efficiently explored lag windows (1-6 months) and regularization strengths
    - Optuna's TPE sampler focuses on promising regions of the parameter space
    - 2-fold CV provides reliable performance estimates while working within data constraints
    - Best parameters balance model complexity with predictive accuracy

    **Data constraints matter**:
    - Small datasets require careful CV configuration (we have 115 training samples)
    - Lag values must fit within CV fold sizes: `first_fold_train_size - test_size ≥ lag + horizon`
    - With our setup: ~57 - 12 = 45 > 6 + 12 ✓

    **In practice**:
    - Start with conservative CV parameters for small datasets
    - Increase n_trials and n_splits as more data becomes available
    - Monitor trial failures - they indicate parameter/data mismatches
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Incremental Learning for Production

    In production forecasting scenarios, you often need to update models with new observations without full retraining.

    ### When to Use Incremental Learning

    - **Streaming data**: New observations arrive continuously
    - **Concept drift**: Patterns gradually change over time
    - **Computational constraints**: Full retraining is too expensive

    ### The `update()` Method

    Yohou's `update()` method:
    - Extends internal memory buffers with new observations
    - Does NOT refit the model (no new training)
    - Updates observation horizon for feature generation
    - Much faster than full retraining

    ### Trade-offs

    - ✅ **Fast**: No model refitting required
    - ✅ **Real-time friendly**: Can handle streaming data
    - ❌ **No adaptation**: Model parameters stay fixed
    - ❌ **Memory limited**: Only observation_horizon rows stored

    Let's simulate a rolling forecast scenario to see incremental learning in action!
    """)
    return


@app.cell
def _(mae_scorer, reduction_forecaster, y_test):
    # Simulate rolling forecast with and without updates
    import copy

    # Create two copies of the fitted pipeline
    pipeline_static = copy.deepcopy(reduction_forecaster)  # Will NOT update
    pipeline_incremental = copy.deepcopy(reduction_forecaster)  # Will update

    # Storage for predictions and errors
    mae_static_list = []
    mae_incremental_list = []

    import sys

    # Rolling forecast simulation
    print("Simulating rolling forecast...")
    for i in range(1, len(y_test)):
        # Get current test observation
        y_current = y_test[i : i + 1]

        # Static forecast (no updates)
        # Predict up to the current step (i+1 relative to training end)
        y_pred_static_all = pipeline_static.predict(forecasting_horizon=i + 1)
        # Take only the last prediction which corresponds to y_current
        y_pred_static = y_pred_static_all.head(i + 1).tail(1)

        # Debug print to file
        if i < 3:
            with open("debug_log.txt", "a") as f:
                f.write(f"Step {i}:\n")
                f.write(f"y_current:\n{y_current}\n")
                f.write(f"y_pred_static (sliced):\n{y_pred_static}\n")
                f.write(f"y_pred_static columns: {y_pred_static.columns}\n")
                f.write(f"y_pred_static_all head:\n{y_pred_static_all.head(i + 2)}\n")
                f.write("-" * 20 + "\n")
            print(f"y_pred_static_all tail:\n{y_pred_static_all.tail()}", file=sys.stderr)

        mae_static = mae_scorer.score(y_current, y_pred_static)
        mae_static_list.append(mae_static)

        # Incremental forecast (with updates)
        y_pred_incremental = pipeline_incremental.update_predict(
            y=y_test[i - 1 : i],  # Previous observation
            X=None,
            forecasting_horizon=1,
        )
        mae_incremental = mae_scorer.score(y_current, y_pred_incremental)
        mae_incremental_list.append(mae_incremental)

    # Calculate cumulative statistics
    avg_mae_static = sum(mae_static_list) / len(mae_static_list)
    avg_mae_incremental = sum(mae_incremental_list) / len(mae_incremental_list)

    print(f"\nRolling forecast results ({len(mae_static_list)} steps):")
    print(f"  Static model (no updates): {avg_mae_static:.2f} MAE")
    print(f"  Incremental model (with updates): {avg_mae_incremental:.2f} MAE")
    print(f"  Difference: {avg_mae_static - avg_mae_incremental:.2f}")
    return (
        avg_mae_incremental,
        avg_mae_static,
        mae_incremental_list,
        mae_static_list,
    )


@app.cell
def _(
    avg_mae_incremental,
    avg_mae_static,
    go,
    mae_incremental_list,
    mae_static_list,
):
    # Visualize rolling forecast MAE over time
    fig_incremental = go.Figure()

    steps = list(range(1, len(mae_static_list) + 1))

    # Static model MAE
    fig_incremental.add_trace(
        go.Scatter(
            x=steps,
            y=mae_static_list,
            mode="lines+markers",
            name=f"Static (avg: {avg_mae_static:.2f})",
            line=dict(color="red", width=2),
            marker=dict(size=4),
        )
    )

    # Incremental model MAE
    fig_incremental.add_trace(
        go.Scatter(
            x=steps,
            y=mae_incremental_list,
            mode="lines+markers",
            name=f"Incremental (avg: {avg_mae_incremental:.2f})",
            line=dict(color="green", width=2),
            marker=dict(size=4),
        )
    )

    fig_incremental.update_layout(
        title="Rolling Forecast Performance: Static vs. Incremental Learning",
        xaxis_title="Forecast Step",
        yaxis_title="MAE (thousands)",
        height=400,
        hovermode="x unified",
    )
    fig_incremental
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Incremental Learning Insights

    The comparison shows:

    - **Static model**: Performance may degrade as predictions drift from training distribution
    - **Incremental model**: Maintains context by updating observation window with recent data
    - **Key factor**: Performance difference depends on how much patterns change during test period

    ### When to Update vs. Retrain

    | Scenario | Recommendation |
    |----------|---------------|
    | Streaming data, stable patterns | Use `update()` for efficiency |
    | Detected concept drift | Full `fit()` to learn new patterns |
    | Scheduled maintenance window | Periodic `fit()` with expanded training data |
    | Critical predictions | Full `fit()` for maximum accuracy |

    ### Production Considerations

    1. **Monitor performance**: Track rolling metrics to detect when retraining is needed
    2. **Observation horizon**: Ensure enough memory (higher observation_horizon)
    3. **Update frequency**: Balance computation vs. freshness (every observation vs. batched)
    4. **Fallback strategy**: Keep baseline model for robust predictions when main model fails
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 🎯 Conclusion and Next Steps

    Congratulations! You've learned the fundamentals of time series forecasting with yohou:

    ### What We Covered

    ✅ **Yohou's scikit-learn-compatible API**
    - `fit(y, X, forecasting_horizon)` for training
    - `predict(forecasting_horizon)` for forecasting
    - `update(y)` for incremental learning

    ✅ **Preprocessing pipelines** with invertible transformations
    - `LogTransform` for variance stabilization
    - `SeasonalDifferencing` for stationarization
    - `LagTransformer` for feature engineering

    ✅ **Time series cross-validation** with `Splitter`
    - Expanding window maintains temporal order
    - No data leakage from future to past

    ✅ **Hyperparameter optimization** with `SearchCV` and Optuna
    - Efficient search with TPE sampler
    - Robust evaluation with CV folds

    ✅ **Incremental learning** for production scenarios
    - `update()` extends observation window
    - Trade-offs between speed and adaptation

    ### Explore Further

    Ready to dive deeper? Try these extensions:

    1. **Interval forecasting**: Use `SplitConformalForecaster` for prediction intervals with calibrated coverage
    2. **Panel data**: Forecast multiple time series simultaneously
    3. **Custom transformers**: Build domain-specific preprocessing (holiday effects, promotions)
    4. **Model ensembles**: Combine multiple forecasters for robust predictions
    5. **Feature engineering**: Add exogenous variables

    ### Resources

    - 📚 [Yohou Documentation](https://github.com/gtauzin/yohou)
    - 🔬 [API Reference](https://gtauzin.github.io/yohou/)
    - 💻 [GitHub Repository](https://github.com/gtauzin/yohou)
    - 📝 [Contributing Guide](https://github.com/gtauzin/yohou/blob/main/CONTRIBUTING.md)

    **Happy forecasting!** 🚀📈
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
