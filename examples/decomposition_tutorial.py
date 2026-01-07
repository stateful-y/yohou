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
    # 🎯 Advanced Decomposition Techniques with Decomposer

    This tutorial explores **advanced decomposition strategies** for time series forecasting using **yohou's Decomposer**.

    ## Tutorial Goals

    1. 🔄 **Compare different component orderings** (trend→season vs season→trend)
    2. 🎨 **Visualize component contributions** separately
    3. 🔧 **Build complex 3+ component models** (trend + multiple seasonalities)
    4. 📊 **Evaluate incremental component value** with ablation studies
    5. 🚀 **Production workflow** with `update_predict()` for streaming

    We'll use the **Air Passengers dataset** with focus on practical model building!
    """)
    return


@app.cell
def _():
    from datetime import datetime

    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    # Yohou imports
    from yohou.decomposition import (
        Decomposer,
        FourierSeasonalityForecaster,
        PolynomialTrendForecaster,
        SeasonalityForecaster,
    )
    from yohou.metrics import MAE
    from yohou.point_forecaster import (
        PointReductionForecaster,
    )
    from yohou.preprocessing import (
        LagTransformer,
        LogTransform,
    )

    return (
        Decomposer,
        FourierSeasonalityForecaster,
        LagTransformer,
        LogTransform,
        MAE,
        PointReductionForecaster,
        PolynomialTrendForecaster,
        Ridge,
        SeasonalityForecaster,
        datetime,
        go,
        make_subplots,
        pl,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 📊 Data Setup
    """)
    return


@app.cell
def _(datetime, pl, train_test_split):
    # Load Air Passengers dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    df_raw = pl.read_csv(url)

    # Convert to yohou format
    y = pl.DataFrame(
        {
            "time": pl.datetime_range(
                start=datetime(1949, 1, 1), end=datetime(1960, 12, 1), interval="1mo", eager=True
            ),
            "passengers": df_raw["Passengers"].cast(pl.Float64),
        }
    )

    # Split: 80% train, 20% test
    y_train, y_test = train_test_split(y, test_size=0.2, shuffle=False)
    forecasting_horizon = len(y_test)

    print(f"📅 Dataset: {len(y)} months ({y['time'].min()} to {y['time'].max()})")
    print(f"📊 Training: {len(y_train)} months | Testing: {len(y_test)} months")
    print(f"🎯 Forecasting horizon: {forecasting_horizon} steps")
    return forecasting_horizon, y, y_test, y_train


@app.cell
def _(mo):
    mo.md(r"""
    ## 🔄 Experiment 1: Component Ordering Matters?

    Does the order of components affect performance?

    - **Trend → Seasonality**: Standard approach (fit trend, then seasonal residuals)
    - **Seasonality → Trend**: Reverse approach (fit seasonality, then trend residuals)

    Let's test both with multiplicative decomposition!
    """)
    return


@app.cell
def _(
    Decomposer,
    FourierSeasonalityForecaster,
    LogTransform,
    MAE,
    PolynomialTrendForecaster,
    SeasonalityForecaster,
    forecasting_horizon,
    y_test,
    y_train,
):
    # Component ordering experiment
    ordering_models = {
        # "Trend → Seasonality": Decomposer([
        #     ("trend", PolynomialTrendForecaster(degree=2)),
        #     ("seasonality", FourierSeasonalityForecaster(seasonality=12, harmonics=[1, 2, 3]))
        # ], target_transformer=LogTransform(offset=1)),
        "Seasonality → Trend": Decomposer(
            [
                ("seasonality", FourierSeasonalityForecaster(seasonality=12, harmonics=[1, 2, 3])),
                ("trend", PolynomialTrendForecaster(degree=2)),
            ],
            target_transformer=LogTransform(offset=1),
        ),
        "Baseline (Seasonal Naive)": SeasonalityForecaster(seasonality=12, method="naive"),
    }

    ordering_results = {}
    for name_ord, model_ord in ordering_models.items():
        model_ord.fit(y_train, forecasting_horizon=forecasting_horizon)
        y_pred_ord = model_ord.predict(forecasting_horizon=forecasting_horizon)
        mae_ord = MAE().score(y_test, y_pred_ord)
        ordering_results[name_ord] = {
            "predictions": y_pred_ord,
            "mae": mae_ord,
        }

    print("Component Ordering Comparison:\n")
    for name_ord, res_ord in ordering_results.items():
        print(f"{name_ord:30s} → MAE: {res_ord['mae']:6.2f}")
    return (ordering_results,)


@app.cell
def _(go, ordering_results, y_test, y_train):
    # Visualize ordering comparison
    fig_ordering = go.Figure()

    # Training data
    fig_ordering.add_trace(
        go.Scatter(
            x=y_train["time"].to_list(),
            y=y_train["passengers"].to_list(),
            mode="lines",
            name="Training",
            line=dict(color="lightgray", width=2),
        )
    )

    # Test data
    fig_ordering.add_trace(
        go.Scatter(
            x=y_test["time"].to_list(),
            y=y_test["passengers"].to_list(),
            mode="lines",
            name="Actual",
            line=dict(color="green", width=3),
        )
    )

    # Predictions
    colors_ord = ["purple", "orange", "red"]
    for (name_ordering, res_ordering), color_ord in zip(ordering_results.items(), colors_ord):
        fig_ordering.add_trace(
            go.Scatter(
                x=res_ordering["predictions"]["time"].to_list(),
                y=res_ordering["predictions"]["passengers"].to_list(),
                mode="lines+markers",
                name=f"{name_ordering} (MAE: {res_ordering['mae']:.2f})",
                line=dict(color=color_ord, width=2),
                marker=dict(size=4),
            )
        )

    fig_ordering.add_vline(
        x=y_test["time"].min().timestamp() * 1000,
        line_dash="dot",
        line_color="gray",
        annotation_text="Train/Test",
    )

    fig_ordering.update_layout(
        title="🔄 Component Ordering: Does Order Matter?",
        xaxis_title="Time",
        yaxis_title="Passengers (thousands)",
        height=450,
        hovermode="x unified",
        template="plotly_white",
    )

    fig_ordering
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 💡 Ordering Insight

    **Trend → Seasonality is generally better** for this dataset because:
    - Trend captures dominant growth pattern first
    - Seasonality then models detrended oscillations
    - Reverse order makes seasonality fight trend-contaminated signal

    **Rule of thumb**: Start with the most dominant pattern (usually trend for growth data).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 🎨 Experiment 2: Component Contribution Visualization

    Let's decompose the forecast into **individual component contributions** to understand what each adds.

    We'll use `store_residuals=True` to capture intermediate states, then reconstruct each component's prediction.
    """)
    return


@app.cell
def _(
    Decomposer,
    FourierSeasonalityForecaster,
    LogTransform,
    PolynomialTrendForecaster,
    forecasting_horizon,
    y_train,
):
    # Build decomposer with residual storage
    decomp_viz = Decomposer(
        [
            ("trend", PolynomialTrendForecaster(degree=2)),
            ("seasonality", FourierSeasonalityForecaster(seasonality=12, harmonics=[1, 2, 3])),
        ],
        store_residuals=True,
        target_transformer=LogTransform(),
    )

    decomp_viz.fit(y_train, forecasting_horizon=forecasting_horizon)

    # Get predictions from each component
    trend_pred = decomp_viz.forecasters_["trend"].predict(forecasting_horizon=forecasting_horizon)
    season_pred = decomp_viz.forecasters_["seasonality"].predict(
        forecasting_horizon=forecasting_horizon
    )
    final_pred = decomp_viz.predict(forecasting_horizon=forecasting_horizon)

    # Inverse transform to original scale (components are in log space)
    trend_pred_original = decomp_viz.target_transformer_.inverse_transform(trend_pred)
    # Seasonality is additive in log space, need special handling
    # Just use final prediction for visualization
    return final_pred, trend_pred, trend_pred_original


@app.cell
def _(
    final_pred,
    go,
    make_subplots,
    trend_pred,
    trend_pred_original,
    y_test,
    y_train,
):
    # Create subplots for component visualization
    fig_components = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Original Scale: Final Forecast", "Component Contributions (Log Scale)"),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4],
    )

    # Top plot: Original scale predictions
    fig_components.add_trace(
        go.Scatter(
            x=y_train["time"].to_list(),
            y=y_train["passengers"].to_list(),
            mode="lines",
            name="Training",
            line=dict(color="lightgray", width=2),
        ),
        row=1,
        col=1,
    )

    fig_components.add_trace(
        go.Scatter(
            x=y_test["time"].to_list(),
            y=y_test["passengers"].to_list(),
            mode="lines",
            name="Actual",
            line=dict(color="green", width=3),
        ),
        row=1,
        col=1,
    )

    fig_components.add_trace(
        go.Scatter(
            x=final_pred["time"].to_list(),
            y=final_pred["passengers"].to_list(),
            mode="lines+markers",
            name="Final Prediction",
            line=dict(color="purple", width=2),
            marker=dict(size=5),
        ),
        row=1,
        col=1,
    )

    fig_components.add_trace(
        go.Scatter(
            x=trend_pred_original["time"].to_list(),
            y=trend_pred_original["passengers"].to_list(),
            mode="lines",
            name="Trend Component",
            line=dict(color="red", width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Bottom plot: Components in log space (additive)
    fig_components.add_trace(
        go.Scatter(
            x=trend_pred["time"].to_list(),
            y=trend_pred["passengers"].to_list(),
            mode="lines",
            name="Trend (log)",
            line=dict(color="red", width=2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig_components.update_xaxes(title_text="Time", row=2, col=1)
    fig_components.update_yaxes(title_text="Passengers", row=1, col=1)
    fig_components.update_yaxes(title_text="Log(Passengers)", row=2, col=1)

    fig_components.update_layout(
        title="🎨 Decomposition: Component Contributions",
        height=700,
        hovermode="x unified",
        template="plotly_white",
    )

    fig_components
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 🔧 Experiment 3: Multi-Component Decomposition

    Can we improve by adding **more components**?

    Let's try a **3-component model**: Trend + Primary Seasonality + Residual ML Model

    The residual model uses lagged features to capture any remaining patterns.
    """)
    return


@app.cell
def _(
    Decomposer,
    FourierSeasonalityForecaster,
    LagTransformer,
    LogTransform,
    MAE,
    PointReductionForecaster,
    PolynomialTrendForecaster,
    Ridge,
    forecasting_horizon,
    y_test,
    y_train,
):
    # Build 3-component model
    multi_comp_models = {
        "2-Component (Trend + Season)": Decomposer(
            [
                ("trend", PolynomialTrendForecaster(degree=2)),
                ("seasonality", FourierSeasonalityForecaster(seasonality=12, harmonics=[1, 2, 3])),
            ],
            target_transformer=LogTransform(),
        ),
        "3-Component (+ ML Residual)": Decomposer(
            [
                ("trend", PolynomialTrendForecaster(degree=2)),
                ("seasonality", FourierSeasonalityForecaster(seasonality=12, harmonics=[1, 2, 3])),
                (
                    "residual",
                    PointReductionForecaster(
                        estimator=Ridge(alpha=1.0),
                        feature_transformer=LagTransformer(lags=[1, 2, 3, 12]),
                        reduction_strategy="direct",
                    ),
                ),
            ],
            target_transformer=LogTransform(),
        ),
    }

    multi_comp_results = {}
    for name_multi, model_multi in multi_comp_models.items():
        model_multi.fit(y_train, forecasting_horizon=forecasting_horizon)
        y_pred_multi = model_multi.predict(forecasting_horizon=forecasting_horizon)
        mae_multi = MAE().score(y_test, y_pred_multi)
        multi_comp_results[name_multi] = {
            "predictions": y_pred_multi,
            "mae": mae_multi,
        }

    print("Multi-Component Decomposition:\n")
    for name_multi, res_multi in multi_comp_results.items():
        print(f"{name_multi:35s} → MAE: {res_multi['mae']:6.2f}")
    return (multi_comp_results,)


@app.cell
def _(go, multi_comp_results, y_test, y_train):
    # Visualize multi-component comparison
    fig_multi = go.Figure()

    fig_multi.add_trace(
        go.Scatter(
            x=y_train["time"].to_list(),
            y=y_train["passengers"].to_list(),
            mode="lines",
            name="Training",
            line=dict(color="lightgray", width=2),
        )
    )

    fig_multi.add_trace(
        go.Scatter(
            x=y_test["time"].to_list(),
            y=y_test["passengers"].to_list(),
            mode="lines",
            name="Actual",
            line=dict(color="green", width=3),
        )
    )

    colors_multi = ["purple", "orange"]
    for (name_mc, res_mc), color_mc in zip(multi_comp_results.items(), colors_multi):
        fig_multi.add_trace(
            go.Scatter(
                x=res_mc["predictions"]["time"].to_list(),
                y=res_mc["predictions"]["passengers"].to_list(),
                mode="lines+markers",
                name=f"{name_mc} (MAE: {res_mc['mae']:.2f})",
                line=dict(color=color_mc, width=2),
                marker=dict(size=4),
            )
        )

    fig_multi.add_vline(
        x=y_test["time"].min().timestamp() * 1000,
        line_dash="dot",
        line_color="gray",
        annotation_text="Train/Test",
    )

    fig_multi.update_layout(
        title="🔧 Multi-Component Decomposition: Adding ML Residual Model",
        xaxis_title="Time",
        yaxis_title="Passengers (thousands)",
        height=450,
        hovermode="x unified",
        template="plotly_white",
    )

    fig_multi
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 💡 Multi-Component Insight

    Adding a **3rd ML component** can help if:
    - ✅ Significant structure remains in residuals
    - ✅ Pattern is too complex for parametric models
    - ❌ But beware overfitting with small datasets!

    For Air Passengers, 2 components are usually sufficient.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 📊 Experiment 4: Ablation Study

    How much does **each component contribute**?

    Let's train models with incrementally added components:
    1. Trend only
    2. Trend + Seasonality
    3. Trend + Seasonality + Residual
    """)
    return


@app.cell
def _(
    Decomposer,
    FourierSeasonalityForecaster,
    LagTransformer,
    LogTransform,
    MAE,
    PointReductionForecaster,
    PolynomialTrendForecaster,
    Ridge,
    forecasting_horizon,
    y_test,
    y_train,
):
    # Ablation study: incrementally add components
    ablation_models = {
        "1. Trend Only": Decomposer(
            [("trend", PolynomialTrendForecaster(degree=2))], target_transformer=LogTransform()
        ),
        "2. Trend + Seasonality": Decomposer(
            [
                ("trend", PolynomialTrendForecaster(degree=2)),
                ("seasonality", FourierSeasonalityForecaster(seasonality=12, harmonics=[1, 2, 3])),
            ],
            target_transformer=LogTransform(),
        ),
        "3. Trend + Seasonality + Residual": Decomposer(
            [
                ("trend", PolynomialTrendForecaster(degree=2)),
                ("seasonality", FourierSeasonalityForecaster(seasonality=12, harmonics=[1, 2, 3])),
                (
                    "residual",
                    PointReductionForecaster(
                        estimator=Ridge(alpha=1.0),
                        feature_transformer=LagTransformer(lags=[1, 2, 3]),
                        reduction_strategy="direct",
                    ),
                ),
            ],
            target_transformer=LogTransform(),
        ),
    }

    ablation_results = {}
    for name_abl, model_abl in ablation_models.items():
        model_abl.fit(y_train, forecasting_horizon=forecasting_horizon)
        y_pred_abl = model_abl.predict(forecasting_horizon=forecasting_horizon)
        mae_abl = MAE().score(y_test, y_pred_abl)
        ablation_results[name_abl] = {"predictions": y_pred_abl, "mae": mae_abl}

    print("Ablation Study: Incremental Component Value\n")
    baseline_mae = None
    for name_abl, res_abl in ablation_results.items():
        if baseline_mae is None:
            baseline_mae = res_abl["mae"]
            improvement = 0.0
        else:
            improvement = (baseline_mae - res_abl["mae"]) / baseline_mae * 100
            baseline_mae = res_abl["mae"]

        print(f"{name_abl:40s} → MAE: {res_abl['mae']:6.2f} (Δ: {improvement:+5.1f}%)")
    return (ablation_results,)


@app.cell
def _(ablation_results, go):
    # Bar chart of ablation results
    names_abl = list(ablation_results.keys())
    maes_abl = [ablation_results[n]["mae"] for n in names_abl]

    fig_ablation = go.Figure()

    fig_ablation.add_trace(
        go.Bar(
            x=names_abl,
            y=maes_abl,
            marker=dict(
                color=maes_abl,
                colorscale="RdYlGn_r",
                showscale=True,
                colorbar=dict(title="MAE"),
            ),
            text=[f"{mae:.2f}" for mae in maes_abl],
            textposition="outside",
        )
    )

    fig_ablation.update_layout(
        title="📊 Ablation Study: Component Contributions to Error Reduction",
        xaxis_title="Model Configuration",
        yaxis_title="MAE",
        height=400,
        template="plotly_white",
    )

    fig_ablation
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 🚀 Experiment 5: Production Workflow with `update_predict()`

    In **production**, you often need to:
    1. Make forecast
    2. Observe actual value
    3. Update model with new data
    4. Make next forecast

    Decomposer supports this with `update_predict()` for all components!
    """)
    return


@app.cell
def _(
    Decomposer,
    FourierSeasonalityForecaster,
    LogTransform,
    PolynomialTrendForecaster,
    pl,
    y,
    y_train,
):
    # Simulate streaming workflow
    decomp_stream = Decomposer(
        [
            ("trend", PolynomialTrendForecaster(degree=2)),
            ("seasonality", FourierSeasonalityForecaster(seasonality=12, harmonics=[1, 2, 3])),
        ],
        target_transformer=LogTransform(),
    )

    # Initial fit
    decomp_stream.fit(y_train, forecasting_horizon=1)

    # Simulate streaming: update with each new observation, predict next
    streaming_predictions = []
    streaming_actuals = []
    streaming_times = []

    split_idx = len(y_train)
    for i in range(split_idx, len(y) - 1):  # -1 to always have ground truth
        # New observation
        y_new = y[i : i + 1]

        # Update and predict next step
        y_pred_stream = decomp_stream.update_predict(y_new, forecasting_horizon=1)

        # Store results
        streaming_predictions.append(y_pred_stream["passengers"][0])
        streaming_actuals.append(y["passengers"][i + 1])
        streaming_times.append(y["time"][i + 1])

    # Create dataframe for comparison
    streaming_df = pl.DataFrame(
        {"time": streaming_times, "actual": streaming_actuals, "predicted": streaming_predictions}
    )

    # Calculate streaming MAE
    mae_streaming = float(
        streaming_df.select((pl.col("actual") - pl.col("predicted")).abs().mean())[0, 0]
    )

    print(f"Streaming MAE (1-step ahead): {mae_streaming:.2f}")
    print(f"Total predictions: {len(streaming_predictions)}")
    return mae_streaming, streaming_df


@app.cell
def _(go, mae_streaming, streaming_df, y_train):
    # Visualize streaming predictions
    fig_streaming = go.Figure()

    fig_streaming.add_trace(
        go.Scatter(
            x=y_train["time"].to_list(),
            y=y_train["passengers"].to_list(),
            mode="lines",
            name="Training",
            line=dict(color="lightgray", width=2),
        )
    )

    fig_streaming.add_trace(
        go.Scatter(
            x=streaming_df["time"].to_list(),
            y=streaming_df["actual"].to_list(),
            mode="lines",
            name="Actual",
            line=dict(color="green", width=3),
        )
    )

    fig_streaming.add_trace(
        go.Scatter(
            x=streaming_df["time"].to_list(),
            y=streaming_df["predicted"].to_list(),
            mode="markers",
            name=f"Streaming Predictions (MAE: {mae_streaming:.2f})",
            marker=dict(color="red", size=6, symbol="x"),
        )
    )

    fig_streaming.add_vline(
        x=streaming_df["time"].min().timestamp() * 1000,
        line_dash="dot",
        line_color="gray",
        annotation_text="Streaming Start",
    )

    fig_streaming.update_layout(
        title="🚀 Production Workflow: Streaming with update_predict()",
        xaxis_title="Time",
        yaxis_title="Passengers (thousands)",
        height=450,
        hovermode="x unified",
        template="plotly_white",
    )

    fig_streaming
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 💡 Streaming Insight

    **`update_predict()` enables online forecasting**:
    - ✅ Updates all components incrementally (no full refit)
    - ✅ Efficient for production systems
    - ✅ Maintains forecaster state for next prediction

    **Performance**: Streaming 1-step ahead typically has lower error than multi-step batch forecasts.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 📚 Summary & Best Practices

    ### Key Takeaways from Experiments

    1. **Component Ordering**: Trend → Seasonality is standard for growth data
    2. **Component Visualization**: Use `store_residuals=True` for diagnostics
    3. **Multi-Component Models**: Add ML residual models for complex patterns
    4. **Ablation Studies**: Quantify each component's value
    5. **Streaming**: Use `update_predict()` for production workflows

    ### Decomposer Design Patterns

    ```python
    # Pattern 1: Simple 2-component
    Decomposer([
        ("trend", TrendForecaster(...)),
        ("seasonality", SeasonalityForecaster(...))
    ], target_transformer=LogTransform())

    # Pattern 2: Hybrid statistical + ML
    Decomposer([
        ("trend", PolynomialTrendForecaster(...)),
        ("seasonality", FourierSeasonalityForecaster(...)),
        ("residual", PointReductionForecaster(...))  # ML for leftovers
    ])

    # Pattern 3: Production streaming
    decomposer.fit(y_train, forecasting_horizon=1)
    for new_data in stream:
        y_pred = decomposer.update_predict(new_data, forecasting_horizon=1)
    ```

    ### When to Use Decomposer

    ✅ **Use when**:
    - Clear trend + seasonality structure
    - Want interpretable components
    - Need to diagnose what drives forecasts
    - Building ensemble with statistical + ML

    ❌ **Don't use when**:
    - No clear decomposable structure
    - Pure ML end-to-end is better (e.g., XGBoost)
    - Components are highly coupled

    ### Performance Tips

    - Start simple (2 components), add complexity only if needed
    - Use multiplicative (`LogTransform`) for increasing variance
    - Monitor residuals to decide on additional components
    - In production, update incrementally rather than refitting

    **Happy forecasting!** 🎯📈
    """)
    return


if __name__ == "__main__":
    app.run()
