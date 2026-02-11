"""Visualization functions for forecast analysis using Plotly."""

import numpy as np
import plotly.graph_objects as go
import polars as pl
from pydantic import StrictFloat


def plot_prediction_intervals(
    y_pred: pl.DataFrame,
    y_pred_int: pl.DataFrame,
    coverage_rates: list[StrictFloat],
    y_test: pl.DataFrame | None = None,
) -> go.Figure:
    """Plot the prediction intervals.

    Parameters
    ----------
    y_pred : pl.DataFrame
        Predicted time series.

    y_pred_int : pl.DataFrame
        Prediction intervals time series.

    coverage_rates: list of floats, default=[0.05]
        List of miscoverage levels to check calibration for.

    y_test : pl.DataFrame or None, default=None
        Target true time series. If None, it is not plotted.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure.
    """
    fig = go.Figure()

    y_column = y_pred.columns[0]

    for coverage_rate in coverage_rates:
        coverage = 100 - round(100 * coverage_rate)

        y_pred_upper = y_pred_int[f"{y_column}_upper_{coverage_rate}"]
        y_pred_lower = y_pred_int[f"{y_column}_lower_{coverage_rate}"]

        y_int = pl.concat([y_pred_lower, y_pred_upper[::-1]])

        # Get time column for x-axis
        time_col = y_pred_int.select("time").to_series().to_list()

        fig.add_trace(
            go.Scatter(
                x=time_col + time_col[::-1],
                y=y_int,
                fill="toself",
                opacity=0.3,
                name=f"{y_column}_{coverage}%_coverage",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=y_pred.select("time").to_series().to_list(),
            y=y_pred[y_column],
            name=f"{y_column}_prediction",
        )
    )

    if y_test is not None:
        fig.add_trace(
            go.Scatter(
                x=y_test.select("time").to_series().to_list(),
                y=y_test[y_column],
                name=f"{y_column}_truth",
            )
        )

    fig.update_layout(title_text="Prediction with intervals")
    fig["layout"]["yaxis"]["title"] = y_column
    fig["layout"]["xaxis"]["title"] = "Time"

    return fig


def plot_calibration(
    y_pred_int: pl.DataFrame,
    y_test: pl.DataFrame,
    coverage_rates: list[StrictFloat],
) -> go.Figure:
    """Plot prediction interval calibration.

    Parameters
    ----------
    y_pred_int : pl.DataFrame
        Prediction intervals time series.

    y_test : pl.DataFrame
        Target time series.

    coverage_rates: list of floats, default=[0.05]
        List of miscoverage levels to check calibration for.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure.

    """
    import plotly.graph_objects as go

    fig = go.Figure()

    y_column = y_test.columns[0]

    errors = []
    for coverage_rate in coverage_rates:
        y_pred_upper = y_pred_int[f"{y_column}_upper_{coverage_rate}"]
        y_pred_lower = y_pred_int[f"{y_column}_lower_{coverage_rate}"]

        # Convert to numpy arrays for comparison
        y_test_vals = y_test[y_column].to_numpy()
        y_pred_lower_vals = y_pred_lower.to_numpy()
        y_pred_upper_vals = y_pred_upper.to_numpy()

        errors_level = np.logical_or(
            np.less(y_test_vals.flatten(), y_pred_lower_vals.flatten()),
            np.greater(y_test_vals.flatten(), y_pred_upper_vals.flatten()),
        )

        errors.append(np.mean(errors_level))

    fig.add_trace(
        go.Scatter(
            x=coverage_rates,
            y=errors,
            mode="lines",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=coverage_rates,
            y=coverage_rates,
            mode="lines",
            line=dict(color="black", width=3, dash="dash"),
        )
    )

    fig.update_layout(title_text="Calibration plot")
    fig["layout"]["yaxis"]["title"] = "Error rate"
    fig["layout"]["xaxis"]["title"] = "Miscoverage level"

    return fig
