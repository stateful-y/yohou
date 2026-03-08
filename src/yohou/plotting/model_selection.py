"""Cross-validation and hyperparameter search visualization functions."""

import polars as pl
from plotly import graph_objects as go

from yohou.model_selection import BaseSplitter
from yohou.plotting._utils import apply_default_layout, resolve_color_palette
from yohou.utils import validate_plotting_data

__all__ = ["plot_cv_results_scatter", "plot_splits"]


def plot_splits(
    y: pl.DataFrame,
    splitter: BaseSplitter,
    *,
    X: pl.DataFrame | None = None,
    train_color: str | None = None,
    test_color: str | None = None,
    gap_color: str = "#9ca3af",
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """
    Plot cross-validation splits as a timeline visualization.

    Creates a horizontal bar chart showing train/test splits for each fold,
    useful for understanding temporal CV strategies like expanding or sliding windows.

    Parameters
    ----------
    y : pl.DataFrame
        Target time series with mandatory "time" column.
    splitter : BaseSplitter
        A yohou splitter instance (e.g., ExpandingWindowSplitter, SlidingWindowSplitter).
    X : pl.DataFrame | None, default=None
        Optional exogenous features (passed to splitter.split()).
    train_color : str | None, default=None
        Color for train segments. If None, uses first color from yohou palette.
    test_color : str | None, default=None
        Color for test segments. If None, uses second color from yohou palette.
    gap_color : str, default="#9ca3af"
        Color for gap segments (if splitter has gap > 0).
    title : str | None, default=None
        Plot title. Defaults to "Cross-Validation Splits".
    x_label : str | None, default=None
        X-axis label. Defaults to "Time".
    y_label : str | None, default=None
        Y-axis label. Defaults to "Fold".
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels. Defaults to 300 + n_splits * 30.
    **kwargs : dict
        Additional styling parameters:
        - line_width : float, default=10.0
        - gap_opacity : float, default=0.5

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    TypeError
        If y is not a Polars DataFrame or splitter is not a BaseSplitter.
    ValueError
        If DataFrame is empty or missing 'time' column.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_splits
    >>> from yohou.model_selection import ExpandingWindowSplitter

    >>> # Create sample data
    >>> y = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
    ...     "value": list(range(366)),
    ... })

    >>> # Create splitter and plot
    >>> splitter = ExpandingWindowSplitter(n_splits=3, test_size=30)
    >>> fig = plot_splits(y, splitter)
    >>> len(fig.data) > 0
    True

    See Also
    --------
    `plot_cv_results_scatter` : Plot hyperparameter search results.
    `ExpandingWindowSplitter` : Expanding window cross-validation.
    `SlidingWindowSplitter` : Sliding window cross-validation.
    """
    # Validate inputs
    validate_plotting_data(y)

    if not isinstance(splitter, BaseSplitter):
        msg = f"Expected BaseSplitter, got {type(splitter).__name__}"
        raise TypeError(msg)

    # Get colors
    colors = resolve_color_palette(None, 2)
    train_color = train_color or colors[0]
    test_color = test_color or colors[1]

    # Get styling params
    line_width = kwargs.get("line_width", 10.0)
    gap_opacity = kwargs.get("gap_opacity", 0.5)

    # Get splits
    splits = list(splitter.split(y, X))
    n_splits = len(splits)

    # Create figure
    fig = go.Figure()

    # Get time column
    times = y["time"]

    # Check if splitter has gap
    gap = getattr(splitter, "gap", None) or 0

    # Plot each split
    for i, (train_idx, test_idx) in enumerate(splits):
        fold_label = f"Fold {i + 1}"

        # Get train time range
        t_train_start = times[int(train_idx[0])]
        t_train_end = times[int(train_idx[-1])]

        # Get test time range
        t_test_start = times[int(test_idx[0])]
        t_test_end = times[int(test_idx[-1])]

        # Plot train segment
        fig.add_trace(
            go.Scatter(
                x=[t_train_start, t_train_end],
                y=[fold_label, fold_label],
                mode="lines",
                line={"color": train_color, "width": line_width},
                name="Train" if i == 0 else None,
                showlegend=(i == 0),
                legendgroup="train",
                hovertemplate=f"Train<br>Start: %{{x}}<br>Fold: {fold_label}<extra></extra>",
            )
        )

        # Plot gap if present
        if gap > 0:
            fig.add_trace(
                go.Scatter(
                    x=[t_train_end, t_test_start],
                    y=[fold_label, fold_label],
                    mode="lines",
                    line={"color": gap_color, "width": line_width},
                    name="Gap" if i == 0 else None,
                    showlegend=(i == 0),
                    legendgroup="gap",
                    opacity=gap_opacity,
                    hovertemplate=f"Gap<br>Fold: {fold_label}<extra></extra>",
                )
            )

        # Plot test segment
        fig.add_trace(
            go.Scatter(
                x=[t_test_start, t_test_end],
                y=[fold_label, fold_label],
                mode="lines",
                line={"color": test_color, "width": line_width},
                name="Test" if i == 0 else None,
                showlegend=(i == 0),
                legendgroup="test",
                hovertemplate=f"Test<br>Start: %{{x}}<br>Fold: {fold_label}<extra></extra>",
            )
        )

    # Set default labels
    title_default = title or "Cross-Validation Splits"
    x_label_default = x_label or "Time"
    y_label_default = y_label or "Fold"

    # Calculate height based on number of splits
    height_default = height or (300 + n_splits * 30)

    fig = apply_default_layout(
        fig,
        title=title_default,
        x_label=x_label_default,
        y_label=y_label_default,
        width=width,
        height=height_default,
    )

    return fig


def plot_cv_results_scatter(
    cv_results: dict,
    param_name: str,
    scorer_name: str | None = None,
    *,
    higher_is_better: bool = True,
    highlight_best: bool = True,
    color_palette: list[str] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    **kwargs,
) -> go.Figure:
    """
    Plot hyperparameter search results as a scatter plot.

    Creates a scatter plot showing the relationship between a hyperparameter
    and the cross-validation score, with optional highlighting of the best result.

    Parameters
    ----------
    cv_results : dict
        The cv_results_ dictionary from GridSearchCV or RandomizedSearchCV.
        Must contain keys like `param_{param_name}` and `mean_test_{scorer_name}`.
    param_name : str
        Name of the hyperparameter to plot on x-axis (without `param_` prefix).
    scorer_name : str | None, default=None
        Name of the scorer (without `mean_test_` prefix). If None, auto-detects
        from available keys (uses first scorer found).
    higher_is_better : bool, default=True
        Whether higher score values are better. When False, scores are negated
        for display so that metrics like ``neg_mean_squared_error`` appear as
        positive values. The best-point detection still operates on the
        original (un-negated) scores.
    highlight_best : bool, default=True
        Whether to highlight the best parameter value.
    color_palette : list[str] | None, default=None
        Custom color palette. If None, uses yohou palette.
    title : str | None, default=None
        Plot title. Defaults to "CV Results: {param_name}".
    x_label : str | None, default=None
        X-axis label. Defaults to the parameter name.
    y_label : str | None, default=None
        Y-axis label. Defaults to "Mean Test Score".
    width : int | None, default=None
        Plot width in pixels.
    height : int | None, default=None
        Plot height in pixels.
    **kwargs : dict
        Additional styling parameters:
        - marker_size : float, default=10.0
        - best_marker_size : float, default=16.0
        - best_marker_color : str, default="#dc2626"
        - show_std : bool, default=True (shows error bars if std_test_{scorer} exists)

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    ValueError
        If required keys are not found in cv_results.

    Examples
    --------
    >>> from yohou.plotting import plot_cv_results_scatter

    >>> # Example cv_results_ structure from GridSearchCV
    >>> cv_results = {
    ...     "param_alpha": [0.01, 0.1, 1.0, 10.0],
    ...     "mean_test_score": [-0.5, -0.3, -0.2, -0.4],
    ...     "std_test_score": [0.05, 0.03, 0.02, 0.06],
    ...     "rank_test_score": [3, 2, 1, 4],
    ... }

    >>> fig = plot_cv_results_scatter(cv_results, param_name="alpha")
    >>> len(fig.data) > 0
    True

    See Also
    --------
    `plot_splits` : Plot cross-validation splits.
    `GridSearchCV` : Grid search with cross-validation.
    `RandomizedSearchCV` : Randomized search with cross-validation.
    """
    # Construct key names
    param_key = f"param_{param_name}"

    # Auto-detect scorer name if not provided
    if scorer_name is None:
        # Look for mean_test_* keys
        mean_test_keys = [k for k in cv_results if k.startswith("mean_test_")]
        scorer_name = mean_test_keys[0].replace("mean_test_", "") if mean_test_keys else "score"

    mean_key = f"mean_test_{scorer_name}"
    std_key = f"std_test_{scorer_name}"
    rank_key = f"rank_test_{scorer_name}"

    # Validate required keys
    if param_key not in cv_results:
        msg = f"Parameter key '{param_key}' not found in cv_results. Available keys: {list(cv_results.keys())}"
        raise ValueError(msg)

    if mean_key not in cv_results:
        msg = f"Mean score key '{mean_key}' not found in cv_results. Available keys: {list(cv_results.keys())}"
        raise ValueError(msg)

    # Get data
    param_values = cv_results[param_key]
    mean_scores_raw = cv_results[mean_key]
    std_scores = cv_results.get(std_key)
    ranks = cv_results.get(rank_key)

    # Negate scores for display when higher_is_better=False
    sign = 1 if higher_is_better else -1
    mean_scores = [s * sign for s in mean_scores_raw]

    # Get styling params
    marker_size = kwargs.get("marker_size", 10.0)
    best_marker_size = kwargs.get("best_marker_size", 16.0)
    best_marker_color = kwargs.get("best_marker_color", "#dc2626")
    show_std = kwargs.get("show_std", True) and std_scores is not None

    # Get colors
    if color_palette is None:
        color_palette = resolve_color_palette(None, 2)

    # Create figure
    fig = go.Figure()

    # Find best index (use raw scores for ranking)
    best_idx = None
    if highlight_best and ranks is not None:
        best_idx = list(ranks).index(1)
    elif highlight_best:
        # Fall back to finding max raw score (before negation)
        best_idx = list(mean_scores_raw).index(max(mean_scores_raw))

    # Add scatter trace with optional error bars
    error_y = None
    if show_std:
        error_y = {
            "type": "data",
            "array": std_scores,
            "visible": True,
            "color": color_palette[0],
        }

    fig.add_trace(
        go.Scatter(
            x=param_values,
            y=mean_scores,
            mode="markers",
            marker={
                "size": marker_size,
                "color": color_palette[0],
            },
            error_y=error_y,
            name="CV Score",
            hovertemplate=f"{param_name}: %{{x}}<br>Score: %{{y:.4f}}<extra></extra>",
        )
    )

    # Highlight best point
    if highlight_best and best_idx is not None:
        fig.add_trace(
            go.Scatter(
                x=[param_values[best_idx]],
                y=[mean_scores[best_idx]],
                mode="markers",
                marker={
                    "size": best_marker_size,
                    "color": best_marker_color,
                    "symbol": "star",
                    "line": {"width": 2, "color": "#ffffff"},
                },
                name="Best",
                hovertemplate=f"Best<br>{param_name}: %{{x}}<br>Score: %{{y:.4f}}<extra></extra>",
            )
        )

    # Set default labels
    title_default = title or f"CV Results: {param_name}"
    x_label_default = x_label or param_name
    y_label_default = y_label or "Mean Test Score"

    fig = apply_default_layout(
        fig,
        title=title_default,
        x_label=x_label_default,
        y_label=y_label_default,
        width=width,
        height=height,
    )

    return fig
