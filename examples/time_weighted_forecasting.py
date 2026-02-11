"""Time-Weighted Forecasting Tutorial.

This notebook demonstrates time-based weighting for both training and evaluation
in time series forecasting with Yohou.
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Time-Weighted Forecasting with Yohou

    Time-based weighting allows you to give more importance to certain observations
    during training or evaluation. This is useful for:

    - **Recent data emphasis**: Give more weight to recent observations (e.g., concept drift)
    - **Seasonal peaks**: Emphasize critical periods (e.g., holiday sales)
    - **Quality variation**: Down-weight periods with data quality issues
    - **Evaluation focus**: Weight errors differently based on forecast timing

    This tutorial covers:
    1. Weight functions (exponential, linear, seasonal)
    2. Alignment strategies for tabularized data
    3. Training with time-weighted samples
    4. Evaluation with time-weighted scoring
    5. Visualizations and comparisons
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.linear_model import Ridge

    from yohou.metrics import MeanAbsoluteError
    from yohou.point_forecaster import PointReductionForecaster
    from yohou.utils.weighting import (
        compose_weights,
        exponential_decay_weight,
        linear_decay_weight,
        seasonal_emphasis_weight,
    )


    return (
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        compose_weights,
        exponential_decay_weight,
        go,
        linear_decay_weight,
        make_subplots,
        mo,
        np,
        pl,
        seasonal_emphasis_weight,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Generate Synthetic Time Series
    """)
    return


@app.cell
def _(mo, np, pl):
    def create_synthetic_data(length=120, seed=42):
        """Create time series with trend, seasonality, and noise."""
        np.random.seed(seed)

        time = pl.datetime_range(
            pl.datetime(2020, 1, 1),
            pl.datetime(2020, 1, 1) + pl.duration(days=length - 1),
            interval="1d",
            eager=True
        )

        t = np.arange(length)
        trend = 0.15 * t  # Upward trend
        seasonal = 8 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
        noise = np.random.normal(0, 1.5, length)

        # Add a regime change at 80% (simulating concept drift)
        regime_shift = np.where(t > 0.8 * length, 10, 0)

        values = trend + seasonal + noise + regime_shift

        return pl.DataFrame({"time": time, "value": values})

    data = create_synthetic_data(length=120)
    train_data = data[:90]
    test_data = data[90:]

    mo.ui.table(data.head(10), label="Synthetic Time Series (first 10 rows)")
    return test_data, train_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Visualize Weight Functions
    """)
    return


@app.cell
def _(
    exponential_decay_weight,
    go,
    linear_decay_weight,
    make_subplots,
    seasonal_emphasis_weight,
    train_data,
):
    # Create weight functions
    exp_weight_fn = exponential_decay_weight(half_life=15)
    linear_weight_fn = linear_decay_weight(max_steps=None)  # Linear over entire range
    seasonal_weight_fn = seasonal_emphasis_weight(seasonality=7, emphasis=3.0)

    # Compute weights for visualization
    time_series = train_data["time"]
    exp_weights = exp_weight_fn(time_series)
    linear_weights = linear_weight_fn(time_series)
    seasonal_weights = seasonal_weight_fn(time_series)

    # Create subplots
    fig_weights = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Exponential Decay (half_life=15)",
                       "Linear Decay (max_steps=None - linear over full range)",
                       "Seasonal Emphasis (period=7, emphasis=3.0)"),
        vertical_spacing=0.12
    )

    # Add traces
    fig_weights.add_trace(
        go.Scatter(x=list(range(len(exp_weights))), y=exp_weights.to_list(),
                  mode='lines', name='Exponential', line=dict(color='#1f77b4')),
        row=1, col=1
    )

    fig_weights.add_trace(
        go.Scatter(x=list(range(len(linear_weights))), y=linear_weights.to_list(),
                  mode='lines', name='Linear', line=dict(color='#ff7f0e')),
        row=2, col=1
    )

    fig_weights.add_trace(
        go.Scatter(x=list(range(len(seasonal_weights))), y=seasonal_weights.to_list(),
                  mode='lines', name='Seasonal', line=dict(color='#2ca02c')),
        row=3, col=1
    )

    # Update layout
    fig_weights.update_xaxes(title_text="Time Index", row=3, col=1)
    fig_weights.update_yaxes(title_text="Weight", row=2, col=1)
    fig_weights.update_layout(
        height=700,
        showlegend=False,
        title_text="Time Weight Functions"
    )

    fig_weights
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Compare Alignment Strategies

    When training on tabularized data, each sample predicts multiple future steps.
    The alignment strategy determines how weights are aggregated across the forecast horizon.
    """)
    return


@app.cell
def _(
    PointReductionForecaster,
    Ridge,
    compose_weights,
    exponential_decay_weight,
    go,
    seasonal_emphasis_weight,
    train_data,
):
    alignment_strategies = ["first_step", "mean_step", "weighted_mean_step",
                           "max_weight_step", "min_weight_step"]

    # Use composed weights (exponential × seasonal) to create more variation
    # This makes alignment strategy differences more visible
    time_weight_composed = compose_weights(
        exponential_decay_weight(half_life=20),
        seasonal_emphasis_weight(seasonality=7, emphasis=5.0)
    )
    forecasting_horizon = 7

    alignment_results = {}
    for strategy in alignment_strategies:
        forecaster = PointReductionForecaster(estimator=Ridge(alpha=0.1))
        forecaster.fit(
            train_data,
            forecasting_horizon=forecasting_horizon,
            time_weight=time_weight_composed,  # Composed weights vary more
            sample_weight_alignment=strategy
        )
        pred = forecaster.predict(forecasting_horizon=forecasting_horizon)
        alignment_results[strategy] = pred["value"].to_list()

    # Visualize predictions
    fig_alignment = go.Figure()

    for strategy, predictions in alignment_results.items():
        fig_alignment.add_trace(
            go.Scatter(
                x=list(range(1, forecasting_horizon + 1)),
                y=predictions,
                mode='lines+markers',
                name=strategy
            )
        )

    fig_alignment.update_layout(
        title="Predictions by Alignment Strategy (using exponential × seasonal weights)",
        xaxis_title="Forecast Step",
        yaxis_title="Predicted Value",
        height=400,
        hovermode='x unified',
    )

    fig_alignment
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Training Comparison: Weighted vs Unweighted

    Compare forecasters trained with and without time weighting to see
    how recent data emphasis affects predictions.
    """)
    return


@app.cell
def _(
    PointReductionForecaster,
    Ridge,
    exponential_decay_weight,
    go,
    make_subplots,
    test_data,
    train_data,
):
    # Train without weighting
    forecaster_baseline = PointReductionForecaster(estimator=Ridge(alpha=1.0))
    forecaster_baseline.fit(train_data, forecasting_horizon=10)
    pred_baseline = forecaster_baseline.predict(forecasting_horizon=10)

    # Train with exponential weighting (emphasize recent data)
    forecaster_weighted = PointReductionForecaster(estimator=Ridge(alpha=1.0))
    time_weight_train = exponential_decay_weight(half_life=20)
    forecaster_weighted.fit(
        train_data,
        forecasting_horizon=10,
        time_weight=time_weight_train,
        sample_weight_alignment="first_step"
    )
    pred_weighted = forecaster_weighted.predict(forecasting_horizon=10)

    # Create comparison plot
    fig_comparison = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Training Data with Weights", "Predictions Comparison"),
        vertical_spacing=0.15,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # Plot 1: Training data with weight overlay
    train_values = train_data["value"].to_list()
    train_indices = list(range(len(train_values)))
    weights_normalized = time_weight_train(train_data["time"]).to_list()
    # Normalize for visualization
    weights_viz = [w / max(weights_normalized) * max(train_values) for w in weights_normalized]

    fig_comparison.add_trace(
        go.Scatter(x=train_indices, y=train_values, mode='lines',
                  name='Training Data', line=dict(color='gray')),
        row=1, col=1, secondary_y=False
    )

    fig_comparison.add_trace(
        go.Scatter(x=train_indices, y=weights_viz, mode='lines',
                  name='Weights (scaled)', line=dict(color='red', dash='dash'),
                  fill='tonexty', opacity=0.3),
        row=1, col=1, secondary_y=True
    )

    # Plot 2: Predictions comparison
    actual_values = test_data["value"].to_list()[:10]
    forecast_indices = list(range(len(train_values), len(train_values) + 10))

    fig_comparison.add_trace(
        go.Scatter(x=forecast_indices, y=actual_values, mode='lines+markers',
                  name='Actual', line=dict(color='black', width=2)),
        row=2, col=1
    )

    fig_comparison.add_trace(
        go.Scatter(x=forecast_indices, y=pred_baseline["value"].to_list(),
                  mode='lines+markers', name='Baseline (no weights)',
                  line=dict(color='blue', dash='dash')),
        row=2, col=1
    )

    fig_comparison.add_trace(
        go.Scatter(x=forecast_indices, y=pred_weighted["value"].to_list(),
                  mode='lines+markers', name='Weighted (recent emphasis)',
                  line=dict(color='red')),
        row=2, col=1
    )

    fig_comparison.update_xaxes(title_text="Time Index", row=2, col=1)
    fig_comparison.update_yaxes(title_text="Value", row=1, col=1, secondary_y=False)
    fig_comparison.update_yaxes(title_text="Weight", row=1, col=1, secondary_y=True)
    fig_comparison.update_yaxes(title_text="Value", row=2, col=1)

    fig_comparison.update_layout(height=700, hovermode='x unified')

    fig_comparison
    return (pred_baseline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Time-Weighted Evaluation

    Use time weights during scoring to emphasize errors at specific time points.
    """)
    return


@app.cell
def _(
    MeanAbsoluteError,
    exponential_decay_weight,
    linear_decay_weight,
    pl,
    pred_baseline,
    test_data,
):
    # Prepare evaluation data (align predictions with actuals)
    y_true_eval = test_data[:10]
    y_pred_eval = pred_baseline.select(["time", "value"])

    # Create scorer
    scorer = MeanAbsoluteError()
    scorer.fit(y_true_eval)

    # Evaluate with different weights
    mae_unweighted = scorer.score(y_true_eval, y_pred_eval)

    # Emphasize recent predictions
    eval_weight_exp = exponential_decay_weight(half_life=3)
    mae_exp_weighted = scorer.score(y_true_eval, y_pred_eval, time_weight=eval_weight_exp)

    # Linear decay
    eval_weight_linear = linear_decay_weight(max_steps=7)
    mae_linear_weighted = scorer.score(y_true_eval, y_pred_eval, time_weight=eval_weight_linear)

    scoring_results = pl.DataFrame({
        "Method": ["Unweighted", "Exponential (recent)", "Linear decay"],
        "MAE": [float(mae_unweighted), float(mae_exp_weighted), float(mae_linear_weighted)]
    })

    scoring_results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Seasonal Emphasis

    Emphasize specific seasonal patterns (e.g., weekends, holidays).
    """)
    return


@app.cell
def _(
    PointReductionForecaster,
    Ridge,
    go,
    seasonal_emphasis_weight,
    train_data,
):
    # Fit with weekly seasonal emphasis
    forecaster_seasonal = PointReductionForecaster(estimator=Ridge(alpha=1.0))
    seasonal_weight = seasonal_emphasis_weight(seasonality=7, emphasis=4.0)

    forecaster_seasonal.fit(
        train_data,
        forecasting_horizon=14,
        time_weight=seasonal_weight,
        sample_weight_alignment="mean_step"
    )

    pred_seasonal = forecaster_seasonal.predict(forecasting_horizon=14)

    # Visualize weekly pattern emphasis
    seasonal_weights_viz = seasonal_weight(train_data["time"]).to_list()

    fig_seasonal = go.Figure()

    fig_seasonal.add_trace(
        go.Scatter(
            x=list(range(len(seasonal_weights_viz))),
            y=seasonal_weights_viz,
            mode='lines+markers',
            name='Seasonal Weights',
            marker=dict(size=4)
        )
    )

    # Highlight one complete cycle
    cycle_indices = list(range(0, 14))
    cycle_weights = seasonal_weights_viz[:14]

    fig_seasonal.add_trace(
        go.Scatter(
            x=cycle_indices,
            y=cycle_weights,
            mode='markers',
            name='First 2 Weeks',
            marker=dict(size=10, color='red')
        )
    )

    fig_seasonal.update_layout(
        title="Seasonal Emphasis: Weekly Pattern (period=7, emphasis=4.0)",
        xaxis_title="Time Index",
        yaxis_title="Weight",
        height=400,
        hovermode='x unified'
    )

    fig_seasonal
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Composed Weights

    Combine multiple weight functions to capture complex patterns
    (e.g., recent emphasis + seasonal peaks).
    """)
    return


@app.cell
def _(
    PointReductionForecaster,
    Ridge,
    compose_weights,
    exponential_decay_weight,
    go,
    seasonal_emphasis_weight,
    train_data,
):
    # Compose exponential decay with seasonal emphasis
    composed_weight = compose_weights(
        exponential_decay_weight(half_life=20),
        seasonal_emphasis_weight(seasonality=7, emphasis=2.5)
    )

    # Visualize composed weights
    composed_weights_viz = composed_weight(train_data["time"]).to_list()

    fig_composed = go.Figure()

    fig_composed.add_trace(
        go.Scatter(
            x=list(range(len(composed_weights_viz))),
            y=composed_weights_viz,
            mode='lines',
            name='Composed Weights',
            line=dict(color='purple', width=2)
        )
    )

    fig_composed.update_layout(
        title="Composed Weights: Exponential × Seasonal<br><sub>Recent data + weekly peaks both emphasized</sub>",
        xaxis_title="Time Index",
        yaxis_title="Weight",
        height=400
    )

    # Train with composed weights
    forecaster_composed = PointReductionForecaster(estimator=Ridge(alpha=1.0))
    forecaster_composed.fit(
        train_data,
        forecasting_horizon=7,
        time_weight=composed_weight,
        sample_weight_alignment="weighted_mean_step"
    )

    fig_composed
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. DataFrame-Based Weights

    Specify weights directly via DataFrame for custom patterns.
    """)
    return


@app.cell
def _(PointReductionForecaster, Ridge, go, pl, train_data):
    # Create custom weight DataFrame
    # Emphasize first month and last two weeks
    time_values = train_data["time"].to_list()
    custom_weights = []
    for i in range(len(time_values)):
        if i < 30:
            custom_weights.append(3.0)  # First month: high weight
        elif i > 75:
            custom_weights.append(2.5)  # Last 2 weeks: medium-high weight
        else:
            custom_weights.append(1.0)  # Middle period: normal weight

    weight_df = pl.DataFrame({
        "time": time_values,
        "weight": custom_weights
    })

    # Visualize custom weights
    fig_custom = go.Figure()

    fig_custom.add_trace(
        go.Scatter(
            x=list(range(len(custom_weights))),
            y=custom_weights,
            mode='lines',
            name='Custom Weights',
            line=dict(color='green', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,128,0,0.2)'
        )
    )

    # Add annotations
    fig_custom.add_annotation(
        x=15, y=3.0,
        text="First Month<br>(High Weight)",
        showarrow=True,
        arrowhead=2
    )

    fig_custom.add_annotation(
        x=82, y=2.5,
        text="Last 2 Weeks<br>(Medium-High)",
        showarrow=True,
        arrowhead=2
    )

    fig_custom.update_layout(
        title="Custom DataFrame Weights",
        xaxis_title="Time Index",
        yaxis_title="Weight",
        height=400
    )

    # Train with custom weights
    forecaster_custom = PointReductionForecaster(estimator=Ridge(alpha=1.0))
    forecaster_custom.fit(
        train_data,
        forecasting_horizon=7,
        time_weight=weight_df,
        sample_weight_alignment="first_step"
    )

    fig_custom
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    **Weight Functions**:
    - `exponential_decay_weight`: Recent observations get exponentially more weight
    - `linear_decay_weight`: Linear decay with optional truncation
    - `seasonal_emphasis_weight`: Emphasize seasonal patterns (single or multiple periods)
    - `compose_weights`: Multiply multiple weight functions

    **Alignment Strategies**:
    - `first_step`: Use weight at first forecast step (immediate forecast)
    - `mean_step`: Average across horizon (robust to noise)
    - `weighted_mean_step`: Exponentially weighted mean (gradual decay)
    - `max_weight_step`: Maximum weight (catch peaks)
    - `min_weight_step`: Minimum weight (conservative)

    **Use Cases**:
    - Concept drift → exponential decay
    - Seasonal business → seasonal emphasis
    - Data quality issues → custom DataFrame weights
    - Evaluation focus → time-weighted scoring
    """)
    return


if __name__ == "__main__":
    app.run()
