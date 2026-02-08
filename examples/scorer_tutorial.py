"""Scorer Tutorial: Comprehensive Guide to Metrics in Yohou.

This interactive tutorial demonstrates all scorer types with their pros, cons,
and use cases. Covers point scorers, interval scorers, conformity scorers,
and aggregation strategies.
"""

import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Scorer Tutorial: Comprehensive Metrics Guide

    This tutorial demonstrates all available scorers in yohou, their characteristics,
    and when to use each one.

    ## Contents
    1. **Synthetic Data Generation** - Create datasets with different characteristics
    2. **Point Scorers** - Scale-dependent, scale-independent, and robust metrics
    3. **Interval Scorers** - Coverage, sharpness, and proper scoring rules
    4. **Conformity Scorers** - Conformal prediction conformity scores
    5. **Aggregation Strategies** - Timewise, componentwise, groupwise options
    6. **Decision Guide** - How to choose the right scorer
    """)
    return


@app.cell
def _():
    import polars as pl
    import numpy as np
    from datetime import datetime, timedelta
    return datetime, np, pl, timedelta


@app.cell
def _(mo):
    mo.md("""
    ## 1. Synthetic Data Generation

    We'll create three time series with different characteristics:
    - **Series A**: Clean trend with small errors
    - **Series B**: Noisy with outliers
    - **Series C**: Heteroscedastic (increasing variance)
    """)
    return


@app.cell
def _(mo):
    # Interactive controls for data generation
    n_points_slider = mo.ui.slider(
        start=50, stop=200, step=10, value=100, label="Number of points"
    )
    noise_level_slider = mo.ui.slider(
        start=0.1, stop=2.0, step=0.1, value=0.5, label="Noise level"
    )
    seed_slider = mo.ui.slider(
        start=0, stop=100, step=1, value=42, label="Random seed"
    )

    mo.vstack([n_points_slider, noise_level_slider, seed_slider])
    return n_points_slider, noise_level_slider, seed_slider


@app.cell
def _(
    datetime,
    n_points_slider,
    noise_level_slider,
    np,
    pl,
    seed_slider,
    timedelta,
):
    # Generate synthetic data
    np.random.seed(seed_slider.value)
    n = n_points_slider.value
    noise = noise_level_slider.value

    # Time index
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n)]

    # Series A: Clean trend
    trend_a = 100 + 0.5 * np.arange(n)
    series_a = trend_a + np.random.normal(0, 2 * noise, n)

    # Series B: Noisy with outliers
    trend_b = 200 + 0.3 * np.arange(n)
    series_b = trend_b + np.random.normal(0, 5 * noise, n)
    # Add outliers
    outlier_idx = np.random.choice(n, size=max(1, n // 20), replace=False)
    series_b[outlier_idx] += np.random.choice([-1, 1], size=len(outlier_idx)) * 20 * noise

    # Series C: Heteroscedastic (variance increases with level)
    trend_c = 50 + np.arange(n) ** 1.1 / 10
    variance_c = 0.1 * trend_c * noise
    series_c = trend_c + np.random.normal(0, 1, n) * variance_c

    # Create DataFrame
    y_true = pl.DataFrame({
        "time": dates,
        "series_a": series_a,
        "series_b": series_b,
        "series_c": series_c,
    })

    # Create a dummy exogenous feature (constant - not used but needed for SplitConformal)
    X_true = pl.DataFrame({
        "time": dates,
        "const": [1.0] * n,
    })

    # Split into train/test
    train_size = int(0.7 * n)
    y_train = y_true[:train_size]
    y_test = y_true[train_size:]
    X_train = X_true[:train_size]
    X_test = X_true[train_size:]
    return X_train, train_size, y_test, y_train, y_true


@app.cell
def _(mo, y_true):
    # Visualize the data
    import plotly.graph_objects as go

    fig_data = go.Figure()

    for _col in ["series_a", "series_b", "series_c"]:
        fig_data.add_trace(go.Scatter(
            x=y_true["time"],
            y=y_true[_col],
            mode="lines+markers",
            name=_col.replace("_", " ").title(),
            line=dict(width=2),
            marker=dict(size=4)
        ))

    fig_data.update_layout(
        title="Synthetic Time Series Data",
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="x unified",
        height=400,
    )

    mo.ui.plotly(fig_data)
    return (go,)


@app.cell
def _(mo):
    mo.md("""
    ## 2. Generate Forecasts

    Create predictions using different forecasters for comparison:
    - **Naive Forecaster**: Simple baseline
    - **Trend Forecaster**: Linear trend extrapolation
    - **Noisy Forecaster**: Deliberately imperfect predictions
    """)
    return


@app.cell
def _(X_train, datetime, np, pl, timedelta, train_size, y_test, y_train):
    from yohou.point_forecaster import SeasonalNaive
    from sklearn.linear_model import Ridge
    from yohou.point_forecaster import PointReductionForecaster

    # Naive baseline (seasonality=1 for simple naive)
    naive = SeasonalNaive(seasonality=1)
    naive.fit(y_train, X_train, forecasting_horizon=len(y_test))
    y_pred_naive = naive.predict(forecasting_horizon=len(y_test))

    # Reduction with trend
    reduction = PointReductionForecaster(estimator=Ridge(alpha=1.0))
    reduction.fit(y_train, X_train, forecasting_horizon=len(y_test))
    y_pred_reduction = reduction.predict(forecasting_horizon=len(y_test))

    # Create deliberately noisy predictions for comparison
    np.random.seed(42)
    y_pred_noisy = pl.DataFrame({
        "observed_time": [datetime(2020, 1, 1) + timedelta(days=train_size - 1)] * len(y_test),
        "time": y_test["time"],
        "series_a": y_test["series_a"] + np.random.normal(0, 5, len(y_test)),
        "series_b": y_test["series_b"] + np.random.normal(0, 10, len(y_test)),
        "series_c": y_test["series_c"] + np.random.normal(0, 8, len(y_test)),
    })

    y_pred_naive, y_pred_reduction, y_pred_noisy
    return (
        PointReductionForecaster,
        Ridge,
        y_pred_naive,
        y_pred_noisy,
        y_pred_reduction,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 3. Point Scorers

    ### 3.1 Scale-Dependent Metrics

    These metrics are in the same units as the target variable but cannot
    compare across series with different scales.
    """)
    return


@app.cell
def _(mo, pl, y_pred_naive, y_pred_noisy, y_pred_reduction, y_test):
    from yohou.metrics import (
        MeanAbsoluteError,
        RootMeanSquaredError,
        MedianAbsoluteError,
    )

    # Compute scale-dependent metrics
    mae = MeanAbsoluteError(aggregation_method="timewise")
    rmse = RootMeanSquaredError(aggregation_method="timewise")
    medae = MedianAbsoluteError(aggregation_method="timewise")

    results_scale_dep = {}
    for _name, _y_pred in [("Naive", y_pred_naive), ("Reduction", y_pred_reduction), ("Noisy", y_pred_noisy)]:
        results_scale_dep[_name] = {
            "MAE": mae.score(y_test, _y_pred).to_dict(as_series=False),
            "RMSE": rmse.score(y_test, _y_pred).to_dict(as_series=False),
            "MedianAE": medae.score(y_test, _y_pred).to_dict(as_series=False),
        }

    # Create comparison table
    rows = []
    for forecaster in results_scale_dep:
        for metric in results_scale_dep[forecaster]:
            for series in results_scale_dep[forecaster][metric]:
                rows.append({
                    "Forecaster": forecaster,
                    "Metric": metric,
                    "Series": series,
                    "Score": results_scale_dep[forecaster][metric][series][0],
                })

    scale_dep_df = pl.DataFrame(rows)

    mo.ui.table(scale_dep_df, page_size=15)
    return MeanAbsoluteError, scale_dep_df


@app.cell
def _(go, mo, pl, scale_dep_df):
    # Visualize scale-dependent metrics
    fig_scale_dep = go.Figure()

    for forecaster_name in scale_dep_df["Forecaster"].unique():
        for metric_name in scale_dep_df["Metric"].unique():
            subset = scale_dep_df.filter(
                (pl.col("Forecaster") == forecaster_name) & 
                (pl.col("Metric") == metric_name)
            )

            fig_scale_dep.add_trace(go.Bar(
                x=subset["Series"],
                y=subset["Score"],
                name=f"{forecaster_name} - {metric_name}",
                text=[f"{v:.2f}" for v in subset["Score"]],
                textposition="outside",
            ))

    fig_scale_dep.update_layout(
        title="Scale-Dependent Metrics Comparison",
        xaxis_title="Series",
        yaxis_title="Score",
        barmode="group",
        height=500,
    )

    mo.ui.plotly(fig_scale_dep)
    return


@app.cell
def _(mo):
    mo.md("""
    **Key Observations:**

    - **MAE**: Robust to outliers, interpretable (same units as target)
    - **RMSE**: Penalizes large errors more heavily than MAE
    - **MedianAE**: Most robust to outliers (useful for Series B)

    **Pros**: Easy to interpret, widely used

    **Cons**: Cannot compare across series with different scales
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 3.2 Scale-Independent Metrics

    These metrics enable cross-series comparison through normalization.
    """)
    return


@app.cell
def _(mo):
    # Interactive controls for scale-independent metrics
    epsilon_slider = mo.ui.slider(
        start=1e-10, stop=1e-6, step=1e-10, value=1e-8, 
        label="Epsilon (MAPE/sMAPE)", show_value=True
    )

    epsilon_slider
    return (epsilon_slider,)


@app.cell
def _(
    epsilon_slider,
    mo,
    pl,
    y_pred_naive,
    y_pred_noisy,
    y_pred_reduction,
    y_test,
):
    from yohou.metrics import (
        MeanAbsolutePercentageError,
        SymmetricMeanAbsolutePercentageError,
    )

    # Compute scale-independent metrics
    mape = MeanAbsolutePercentageError(
        epsilon=epsilon_slider.value,
        aggregation_method="timewise"
    )
    smape = SymmetricMeanAbsolutePercentageError(
        epsilon=epsilon_slider.value,
        aggregation_method="timewise"
    )

    results_scale_indep = {}
    for _name, _y_pred in [("Naive", y_pred_naive), ("Reduction", y_pred_reduction), ("Noisy", y_pred_noisy)]:
        results_scale_indep[_name] = {
            "MAPE": mape.score(y_test, _y_pred).to_dict(as_series=False),
            "sMAPE": smape.score(y_test, _y_pred).to_dict(as_series=False),
        }

    # Create comparison table
    rows_indep = []
    for forecaster_si in results_scale_indep:
        for metric_si in results_scale_indep[forecaster_si]:
            for series_si in results_scale_indep[forecaster_si][metric_si]:
                rows_indep.append({
                    "Forecaster": forecaster_si,
                    "Metric": metric_si,
                    "Series": series_si,
                    "Score (%)": results_scale_indep[forecaster_si][metric_si][series_si][0],
                })

    scale_indep_df = pl.DataFrame(rows_indep)

    mo.ui.table(scale_indep_df, page_size=12)
    return


@app.cell
def _(mo):
    mo.md("""
    **Key Observations:**

    - **MAPE**: Scale-independent (percentage errors), but asymmetric
      - Penalizes over-predictions more than under-predictions
      - Can be undefined/very large when actual values are near zero

    - **sMAPE**: Symmetric version of MAPE, bounded [0, 200]
      - Treats over/under-predictions equally
      - More robust to small actual values

    **Pros**: Enable cross-series comparison, intuitive percentage interpretation

    **Cons**: Sensitive to near-zero values (even with epsilon), unbounded (MAPE)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 3.3 Calibrated Metrics (Require Training Data)

    These metrics normalize by the in-sample naive forecast error.
    """)
    return


@app.cell
def _(mo):
    # Interactive controls for calibrated metrics
    seasonality_slider = mo.ui.slider(
        start=1, stop=20, step=1, value=7,
        label="Seasonality period"
    )

    seasonality_slider
    return (seasonality_slider,)


@app.cell
def _(
    mo,
    pl,
    seasonality_slider,
    y_pred_naive,
    y_pred_noisy,
    y_pred_reduction,
    y_test,
    y_train,
):
    from yohou.metrics import (
        MeanAbsoluteScaledError,
        RootMeanSquaredScaledError,
    )

    # These require fit() call with training data
    mase = MeanAbsoluteScaledError(
        seasonality=seasonality_slider.value,
        aggregation_method="timewise"
    )
    rmsse = RootMeanSquaredScaledError(
        seasonality=seasonality_slider.value,
        aggregation_method="timewise"
    )

    # Fit on training data
    mase.fit(y_train)
    rmsse.fit(y_train)

    results_calibrated = {}
    for _name, _y_pred in [("Naive", y_pred_naive), ("Reduction", y_pred_reduction), ("Noisy", y_pred_noisy)]:
        results_calibrated[_name] = {
            "MASE": mase.score(y_test, _y_pred).to_dict(as_series=False),
            "RMSSE": rmsse.score(y_test, _y_pred).to_dict(as_series=False),
        }

    # Create comparison table
    rows_calib = []
    for forecaster_c in results_calibrated:
        for metric_c in results_calibrated[forecaster_c]:
            for _series_c in results_calibrated[forecaster_c][metric_c]:
                rows_calib.append({
                    "Forecaster": forecaster_c,
                    "Metric": metric_c,
                    "Series": _series_c,
                    "Score": results_calibrated[forecaster_c][metric_c][_series_c][0],
                })

    calibrated_df = pl.DataFrame(rows_calib)

    mo.ui.table(calibrated_df, page_size=12)
    return


@app.cell
def _(mo):
    mo.md("""
    **Key Observations:**

    - **MASE / RMSSE**: Values < 1 indicate better than naive seasonal forecast
    - **Requires `fit(y_train)`**: Computes scaling factors from training data
    - **Scale-independent**: Enables cross-series comparison

    **Pros**:
    - Scale-independent
    - Interpretable baseline (naive forecast)
    - MASE is robust to outliers

    **Cons**:
    - Requires training data
    - Training data must be longer than seasonality period
    - Sensitive to training data quality
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Interval Scorers

    ### 4.1 Generate Prediction Intervals

    We'll use split conformal prediction to create intervals with coverage guarantees.
    """)
    return


@app.cell
def _(mo):
    # Interactive controls for intervals
    coverage_rate_slider = mo.ui.slider(
        start=0.5, stop=0.99, step=0.05, value=0.9,
        label="Coverage rate", show_value=True
    )

    coverage_rate_slider
    return (coverage_rate_slider,)


@app.cell
def _(
    PointReductionForecaster,
    Ridge,
    X_train,
    coverage_rate_slider,
    train_size,
    y_train,
):
    from yohou.interval_forecaster import SplitConformalForecaster
    from yohou.metrics.conformity import AbsoluteResidual

    # Create interval forecaster with calibration_size parameter
    base_forecaster = PointReductionForecaster(estimator=Ridge(alpha=1.0))
    conformity_scorer = AbsoluteResidual()

    interval_forecaster = SplitConformalForecaster(
        point_forecaster=base_forecaster,
        conformity_scorer=conformity_scorer,
        calibration_size=int(0.4 * train_size),
    )

    # Fit on full training data (forecaster splits internally)
    interval_forecaster.fit(
        y_train,
        X_train,
        forecasting_horizon=10
    )

    # Generate intervals
    y_pred_interval = interval_forecaster.predict_interval(
        forecasting_horizon=10,
        coverage_rates=[coverage_rate_slider.value]
    )
    return (
        AbsoluteResidual,
        SplitConformalForecaster,
        interval_forecaster,
        y_pred_interval,
    )


@app.cell
def _(interval_forecaster):
    y_pred_point = interval_forecaster.predict(forecasting_horizon=10)
    y_pred_point
    return (y_pred_point,)


@app.cell
def _(coverage_rate_slider, go, mo, y_pred_interval, y_pred_point, y_test):
    # Visualize intervals for one series
    series_to_plot = "series_a"
    coverage = coverage_rate_slider.value

    lower_col = f"{series_to_plot}_lower_{coverage}"
    upper_col = f"{series_to_plot}_upper_{coverage}"

    fig_intervals = go.Figure()

    # Add confidence band
    fig_intervals.add_trace(go.Scatter(
        x=y_pred_interval["time"],
        y=y_pred_interval[upper_col],
        mode="lines",
        name=f"{int(coverage*100)}% Upper",
        line=dict(width=0),
        showlegend=False,
    ))

    fig_intervals.add_trace(go.Scatter(
        x=y_pred_interval["time"],
        y=y_pred_interval[lower_col],
        mode="lines",
        name=f"{int(coverage*100)}% Interval",
        fill="tonexty",
        fillcolor="rgba(0, 100, 250, 0.2)",
        line=dict(width=0),
    ))

    # Add point predictions
    fig_intervals.add_trace(go.Scatter(
        x=y_pred_point["time"],
        y=y_pred_point[series_to_plot],
        mode="lines+markers",
        name="Point",
        line=dict(color="green", width=2),
        marker=dict(size=6),
    ))

    # Add actual values
    fig_intervals.add_trace(go.Scatter(
        x=y_test["time"],
        y=y_test[series_to_plot],
        mode="lines+markers",
        name="Actual",
        line=dict(color="red", width=2),
        marker=dict(size=6),
    ))

    fig_intervals.update_layout(
        title=f"Prediction Intervals - {series_to_plot}",
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="x unified",
        height=400,
    )

    mo.ui.plotly(fig_intervals)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 4.2 Coverage Metrics

    These metrics evaluate calibration quality (how often intervals contain actuals).
    """)
    return


@app.cell
def _(coverage_rate_slider, mo, pl, y_pred_interval, y_test):
    from yohou.metrics import EmpiricalCoverage, CalibrationError

    # Compute coverage metrics
    emp_coverage = EmpiricalCoverage(aggregation_method="timewise")
    calib_error = CalibrationError(aggregation_method="all")  # Requires multiple rates

    coverage_result = emp_coverage.score(y_test, y_pred_interval)

    # Format results
    coverage_rows = []
    for col in coverage_result.columns:
        coverage_rows.append({
            "Series": col.replace(f"_coverage_{coverage_rate_slider.value}", ""),
            "Empirical Coverage": coverage_result[col][0],
            "Nominal Coverage": coverage_rate_slider.value,
            "Deviation": abs(coverage_result[col][0] - coverage_rate_slider.value),
        })

    coverage_metrics_df = pl.DataFrame(coverage_rows)

    mo.ui.table(coverage_metrics_df)
    return


@app.cell
def _(mo):
    mo.md("""
    **Key Observations:**

    - **EmpiricalCoverage**: Fraction of actuals falling within intervals
      - Perfect calibration: empirical = nominal coverage
      - Values close to nominal rate indicate good calibration

    - **CalibrationError**: Average deviation across multiple coverage rates
      - Only works with multiple coverage rates
      - Lower is better (perfect = 0)

    **Pros**: Simple, direct measure of calibration

    **Cons**: Doesn't account for interval width (overly conservative intervals score well)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 4.3 Sharpness and Proper Scoring Rules

    Evaluate interval width and combined coverage+sharpness metrics.
    """)
    return


@app.cell
def _(mo, pl, y_pred_interval, y_test):
    from yohou.metrics import MeanIntervalWidth, IntervalScore

    # Compute sharpness and proper scoring
    mean_width = MeanIntervalWidth(aggregation_method="timewise")
    interval_score = IntervalScore(aggregation_method="timewise")

    width_result = mean_width.score(y_test, y_pred_interval)
    is_result = interval_score.score(y_test, y_pred_interval)

    # Format results
    interval_metrics_rows = []
    for col_iw in width_result.columns:
        series_name = col_iw.replace("_mean_interval_width", "")
        interval_metrics_rows.append({
            "Series": series_name,
            "Mean Width": width_result[col_iw][0],
            "Interval Score": is_result.get_column([c for c in is_result.columns if series_name in c][0])[0],
        })

    interval_metrics_df = pl.DataFrame(interval_metrics_rows)

    mo.ui.table(interval_metrics_df)
    return


@app.cell
def _(mo):
    mo.md("""
    **Key Observations:**

    - **MeanIntervalWidth**: Average width of prediction intervals
      - Narrower is better (more informative)
      - Should only compare when coverage is approximately equal

    - **IntervalScore (Winkler Score)**: Proper scoring rule
      - Balances coverage and sharpness
      - Penalizes both wide intervals and coverage violations
      - Lower is better

    **Pros**:
    - Interval Score is a proper scoring rule (incentivizes truthful predictions)
    - Width provides interpretable sharpness measure

    **Cons**:
    - Both are scale-dependent
    - Width alone doesn't consider calibration
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Conformity Scorers

    ### 5.1 Basic Conformity Scorers

    Compare residual-based conformity scorers for interval construction.
    """)
    return


@app.cell
def _(
    AbsoluteResidual,
    PointReductionForecaster,
    Ridge,
    SplitConformalForecaster,
    X_train,
    train_size,
    y_train,
):
    from yohou.metrics.conformity import Residual

    # Create interval forecasters with different conformity scorers
    forecaster_residual = SplitConformalForecaster(
        point_forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        conformity_scorer=Residual(),
        calibration_size=int(0.3 * train_size),
    )

    forecaster_abs_residual = SplitConformalForecaster(
        point_forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        conformity_scorer=AbsoluteResidual(),
        calibration_size=int(0.3 * train_size),
    )

    # Fit both
    forecaster_residual.fit(y_train, X_train, forecasting_horizon=10)
    forecaster_abs_residual.fit(y_train, X_train, forecasting_horizon=10)

    # Generate intervals
    y_pred_residual = forecaster_residual.predict_interval(
        forecasting_horizon=10,
        coverage_rates=[0.9]
    )

    y_pred_abs_residual = forecaster_abs_residual.predict_interval(
        forecasting_horizon=10,
        coverage_rates=[0.9]
    )
    return y_pred_abs_residual, y_pred_residual


@app.cell
def _(go, mo, y_pred_abs_residual, y_pred_residual, y_test):
    # Compare intervals visually
    series_conf = "series_a"

    fig_conformity = go.Figure()

    # Residual (asymmetric) intervals
    fig_conformity.add_trace(go.Scatter(
        x=y_pred_residual["time"],
        y=y_pred_residual[f"{series_conf}_upper_0.9"],
        mode="lines",
        name="Residual Upper",
        line=dict(color="blue", dash="dash"),
    ))
    fig_conformity.add_trace(go.Scatter(
        x=y_pred_residual["time"],
        y=y_pred_residual[f"{series_conf}_lower_0.9"],
        mode="lines",
        name="Residual Lower",
        line=dict(color="blue", dash="dash"),
        fill="tonexty",
        fillcolor="rgba(0, 0, 255, 0.1)",
    ))

    # AbsoluteResidual (symmetric) intervals
    fig_conformity.add_trace(go.Scatter(
        x=y_pred_abs_residual["time"],
        y=y_pred_abs_residual[f"{series_conf}_upper_0.9"],
        mode="lines",
        name="AbsoluteResidual Upper",
        line=dict(color="green", dash="dot"),
    ))
    fig_conformity.add_trace(go.Scatter(
        x=y_pred_abs_residual["time"],
        y=y_pred_abs_residual[f"{series_conf}_lower_0.9"],
        mode="lines",
        name="AbsoluteResidual Lower",
        line=dict(color="green", dash="dot"),
        fill="tonexty",
        fillcolor="rgba(0, 255, 0, 0.1)",
    ))

    # Actual values
    fig_conformity.add_trace(go.Scatter(
        x=y_test["time"],
        y=y_test[series_conf],
        mode="lines+markers",
        name="Actual",
        line=dict(color="red", width=2),
        marker=dict(size=6),
    ))

    fig_conformity.update_layout(
        title=f"Conformity Scorer Comparison - {series_conf}",
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="x unified",
        height=400,
    )

    mo.ui.plotly(fig_conformity)
    return


@app.cell
def _(mo):
    mo.md("""
    **Key Observations:**

    - **Residual**: Uses signed errors (y - ŷ)
      - Creates asymmetric intervals
      - Adapts to error distribution

    - **AbsoluteResidual**: Uses unsigned errors |y - ŷ|
      - Creates symmetric intervals
      - More robust to asymmetric error distributions

    **Use Cases**:
    - Use Residual when errors are expected to be asymmetric
    - Use AbsoluteResidual for symmetric, homoscedastic errors
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### 5.2 Scale-Adaptive Conformity Scorers

    GammaResidual scorers create intervals that scale with prediction magnitude.
    """)
    return


@app.cell
def _(mo):
    # Interactive controls for gamma conformity
    gamma_epsilon_slider = mo.ui.slider(
        start=1e-10, stop=1e-6, step=1e-10, value=1e-8,
        label="Gamma epsilon", show_value=True
    )

    gamma_epsilon_slider
    return (gamma_epsilon_slider,)


@app.cell
def _(
    PointReductionForecaster,
    Ridge,
    SplitConformalForecaster,
    X_train,
    gamma_epsilon_slider,
    train_size,
    y_train,
):
    from yohou.metrics.conformity import GammaResidual, AbsoluteGammaResidual

    # Create forecasters with gamma conformity scorers
    forecaster_gamma = SplitConformalForecaster(
        point_forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        conformity_scorer=GammaResidual(epsilon=gamma_epsilon_slider.value),
        calibration_size=int(0.3 * train_size),
    )

    forecaster_abs_gamma = SplitConformalForecaster(
        point_forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        conformity_scorer=AbsoluteGammaResidual(epsilon=gamma_epsilon_slider.value),
        calibration_size=int(0.3 * train_size),
    )

    # Fit
    forecaster_gamma.fit(y_train, X_train, forecasting_horizon=30)
    forecaster_abs_gamma.fit(y_train, X_train, forecasting_horizon=30)

    # Generate intervals
    y_pred_gamma = forecaster_gamma.predict_interval(
        forecasting_horizon=30,
        coverage_rates=[0.9]
    )

    y_pred_abs_gamma = forecaster_abs_gamma.predict_interval(
        forecasting_horizon=30,
        coverage_rates=[0.9]
    )
    return (y_pred_abs_gamma,)


@app.cell
def _(go, mo, y_pred_abs_gamma, y_test):
    # Visualize heteroscedastic series with gamma intervals
    series_hetero = "series_c"  # This has increasing variance

    fig_gamma = go.Figure()

    # Gamma intervals
    fig_gamma.add_trace(go.Scatter(
        x=y_pred_abs_gamma["time"],
        y=y_pred_abs_gamma[f"{series_hetero}_upper_0.9"],
        mode="lines",
        name="GammaResidual Upper",
        line=dict(color="purple", width=2),
    ))
    fig_gamma.add_trace(go.Scatter(
        x=y_pred_abs_gamma["time"],
        y=y_pred_abs_gamma[f"{series_hetero}_lower_0.9"],
        mode="lines",
        name="GammaResidual Lower",
        line=dict(color="purple", width=2),
        fill="tonexty",
        fillcolor="rgba(128, 0, 128, 0.2)",
    ))

    # Actual values
    fig_gamma.add_trace(go.Scatter(
        x=y_test["time"],
        y=y_test[series_hetero],
        mode="lines+markers",
        name="Actual",
        line=dict(color="red", width=2),
        marker=dict(size=6),
    ))

    fig_gamma.update_layout(
        title=f"Scale-Adaptive Intervals - {series_hetero} (Heteroscedastic)",
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="x unified",
        height=400,
    )

    mo.ui.plotly(fig_gamma)
    return


@app.cell
def _(mo):
    mo.md("""
    **Key Observations:**

    - **GammaResidual**: Relative errors (y - ŷ) / (ŷ + ε)
      - Intervals scale with prediction magnitude
      - Useful for heteroscedastic data (variance increases with level)

    - **AbsoluteGammaResidual**: Symmetric version
      - Adapts to prediction scale
      - More robust for heteroscedastic data

    **Use Cases**:
    - Use when error variance is proportional to prediction magnitude
    - Common in financial, economic, and demand forecasting

    **Pros**: Adapts to heteroscedasticity, scale-independent

    **Cons**: Sensitive to near-zero predictions, requires epsilon tuning
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Aggregation Strategies

    ### 6.1 Aggregation Method Options

    All scorers support flexible aggregation across dimensions.
    """)
    return


@app.cell
def _(mo):
    # Interactive aggregation selector
    agg_method_select = mo.ui.dropdown(
        options=["all", "timewise", "componentwise", ["timewise", "componentwise"]],
        value="all",
        label="Aggregation method"
    )

    agg_method_select
    return (agg_method_select,)


@app.cell
def _(MeanAbsoluteError, agg_method_select, mo, y_pred_reduction, y_test):
    # Demonstrate different aggregation modes
    mae_agg = MeanAbsoluteError(aggregation_method=agg_method_select.value)
    result_agg = mae_agg.score(y_test, y_pred_reduction)

    mo.md(f"""
    **Result type**: `{type(result_agg).__name__}`

    **Result**:
    """)
    return (result_agg,)


@app.cell
def _(mo, result_agg):
    mo.ui.table(result_agg) if hasattr(result_agg, "columns") else mo.md(f"Scalar score: **{result_agg:.4f}**")
    return


@app.cell
def _(mo):
    mo.md("""
    **Aggregation Options Explained:**

    - **`"all"`**: Fully aggregated → scalar float
      - Equivalent to `["timewise", "componentwise", "groupwise"]`

    - **`"timewise"`**: Aggregate across time → per-component DataFrame
      - Useful for comparing component performance

    - **`"componentwise"`**: Aggregate across components → per-timestep DataFrame
      - Useful for identifying difficult time periods

    - **`["timewise", "componentwise"]`**: Combine both → scalar or per-group DataFrame
      - For panel data: returns per-group scores

    **Interval scorers** add:
    - **`"coveragewise"`**: Aggregate across coverage rates
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Decision Guide

    ### How to Choose the Right Scorer?
    """)
    return


@app.cell
def _(mo):
    decision_tree = mo.md(
        """
        ```mermaid
        flowchart TD
            A[What are you evaluating?] -->|Point forecasts| B{Compare across series?}
            A -->|Prediction intervals| C{What aspect?}

            B -->|Yes, need scale-independence| D{Prefer relative or naive baseline?}
            B -->|No, same scale| E{Error sensitivity?}

            D -->|Relative errors| F[MAPE or sMAPE]
            D -->|Naive baseline| G{Have training data?}

            G -->|Yes| H[MASE or RMSSE]
            G -->|No| F

            E -->|Robust to outliers| I[MAE or MedianAE]
            E -->|Penalize large errors| J[RMSE]

            C -->|Calibration only| L[EmpiricalCoverage]
            C -->|Sharpness only| M[MeanIntervalWidth]
            C -->|Combined| N[IntervalScore - Winkler]

            style F fill:#90EE90
            style H fill:#90EE90
            style I fill:#90EE90
            style J fill:#FFD700
            style K fill:#FFD700
            style L fill:#87CEEB
            style M fill:#87CEEB
            style N fill:#87CEEB
        ```

        ### Quick Reference Table

        | Use Case | Recommended Scorer | Notes |
        |----------|-------------------|-------|
        | **General forecasting** | MAE | Robust, interpretable |
        | **Cross-series comparison** | MASE, sMAPE | Scale-independent |
        | **Large errors are costly** | RMSE | Quadratic penalty |
        | **Outliers present** | MedianAE | Most robust |
        | **Interval calibration** | EmpiricalCoverage | Direct calibration check |
        | **Interval sharpness + coverage** | IntervalScore | Proper scoring rule |
        | **Conformal prediction** | AbsoluteResidual | Standard choice |
        | **Heteroscedastic data** | AbsoluteGammaResidual | Scale-adaptive intervals |

        ### Common Pitfalls

        1. **Using scale-dependent metrics for cross-series comparison**
           - ❌ Comparing MAE across series with different scales
           - ✅ Use MASE, sMAPE, or RMSSE instead

        2. **Evaluating intervals with width alone**
           - ❌ MeanIntervalWidth without checking coverage
           - ✅ Use IntervalScore or check coverage + width together

        3. **Forgetting to fit calibrated metrics**
           - ❌ Using MASE/RMSSE without calling `fit(y_train)`
           - ✅ Always fit on training data before scoring

        4. **Using MAPE with near-zero values**
           - ❌ MAPE when actuals can be near zero
           - ✅ Use sMAPE or increase epsilon parameter

        5. **Ignoring outliers in metric choice**
           - ❌ Using RMSE when outliers are present
           - ✅ Use MAE or MedianAE for robustness
        """
    )

    decision_tree
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary

    This tutorial covered:

    ✅ **Point Scorers**: 8 metrics with different properties
    - Scale-dependent: MAE, RMSE, MedianAE

    ✅ **Conformity Scorers**: 4 scorers for conformal prediction
    - Basic: Residual, AbsoluteResidual
    - Scale-adaptive: GammaResidual, AbsoluteGammaResidual

    ✅ **Aggregation**: Flexible dimensions (time, component, group, coverage)

    ✅ **Decision Guide**: How to choose the right scorer for your use case

    **Next Steps**:
    - Try different scorers on your own data
    - Experiment with aggregation methods for panel data
    - Tune epsilon/seasonality parameters interactively
    - Combine multiple metrics for comprehensive evaluation
    """)
    return


if __name__ == "__main__":
    app.run()
