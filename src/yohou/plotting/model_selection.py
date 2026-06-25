"""Cross-validation and hyperparameter search visualization functions."""

import math
from typing import Literal

import numpy as np
import polars as pl

try:
    from plotly import graph_objects as go
except ImportError as e:
    msg = "plotly is required for yohou plotting. Install: pip install yohou[plotting]"
    raise ImportError(msg) from e

from yohou.model_selection import BaseSplitter
from yohou.plotting._utils import (
    LegendTracker,
    _create_figure,
    _create_subplots,
    _subplot_spacing,
    apply_default_layout,
    resolve_color_palette,
)
from yohou.utils import validate_plotting_data, validate_plotting_params

__all__ = ["plot_cv_results_scatter", "plot_nested_splits", "plot_splits"]


def _add_segment(
    fig: go.Figure,
    times: pl.Series,
    idx: np.ndarray,
    row_label: str,
    *,
    color: str,
    name: str,
    legendgroup: str,
    show_legend: bool,
    line_width: float,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Draw one horizontal segment (``idx[0]``..``idx[-1]``) on the ``row_label`` row.

    Shared primitive for :func:`plot_splits` and :func:`plot_nested_splits`. When
    ``show_legend`` is ``False`` the trace name is dropped (matching the historical
    ``plot_splits`` behavior for non-first folds) while the ``legendgroup`` is kept so
    grouped traces still toggle together.
    """
    add_kwargs = {} if row is None else {"row": row, "col": col}
    t_start = times[int(idx[0])]
    t_end = times[int(idx[-1])]
    fig.add_trace(
        go.Scatter(
            x=[t_start, t_end],
            y=[row_label, row_label],
            mode="lines",
            line={"color": color, "width": line_width},
            name=name if show_legend else None,
            showlegend=show_legend,
            legendgroup=legendgroup,
            customdata=[[t_start, t_end], [t_start, t_end]],
            hovertemplate=(
                f"{name}<br>Start: %{{customdata[0]}}<br>End: %{{customdata[1]}}<br>Fold: {row_label}<extra></extra>"
            ),
        ),
        **add_kwargs,
    )


def _add_split_row(
    fig: go.Figure,
    times: pl.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    row_label: str,
    *,
    train_color: str,
    test_color: str,
    train_name: str,
    test_name: str,
    line_width: float,
    show_train_legend: bool,
    show_test_legend: bool,
    train_group: str = "train",
    test_group: str = "test",
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Draw a train segment and a test segment on a single ``row_label`` row.

    Reused by :func:`plot_splits` (one row per outer fold) and
    :func:`plot_nested_splits` (one row per inner fold, plus the outer evaluation
    row). ``train_group`` / ``test_group`` set the ``legendgroup`` so that, for
    example, the inner and outer rows toggle independently in the legend.
    """
    _add_segment(
        fig,
        times,
        train_idx,
        row_label,
        color=train_color,
        name=train_name,
        legendgroup=train_group,
        show_legend=show_train_legend,
        line_width=line_width,
        row=row,
        col=col,
    )
    _add_segment(
        fig,
        times,
        test_idx,
        row_label,
        color=test_color,
        name=test_name,
        legendgroup=test_group,
        show_legend=show_test_legend,
        line_width=line_width,
        row=row,
        col=col,
    )


def plot_splits(
    y: pl.DataFrame,
    splitter: BaseSplitter,
    *,
    X_actual: pl.DataFrame | None = None,
    train_color: str | None = None,
    test_color: str | None = None,
    train_name: str = "Train",
    test_name: str = "Test",
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    resampler: bool | Literal["widget"] | None = None,
    line_width: float = 10.0,
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
    X_actual : pl.DataFrame or None, default=None
        Actual features passed to ``splitter.split()``. Not used for
        splitting but accepted for API consistency.
    train_color : str | None, default=None
        Color for train segments. If None, uses first color from yohou palette.
    test_color : str | None, default=None
        Color for test segments. If None, uses second color from yohou palette.
    train_name : str, default="Train"
        Legend label for the train segments.
    test_name : str, default="Test"
        Legend label for the test segments.
    show_legend : bool, default=True
        Whether to show the legend.
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
    resampler : bool | Literal["widget"] | None, default=None
        Enable plotly-resampler for large datasets.  ``True`` or
        ``"widget"`` creates a ``FigureWidgetResampler``; ``False`` or
        ``None`` uses a plain ``go.Figure``.
    line_width : float, default=10.0
        Width of the train/test bars.

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
    [`plot_cv_results_scatter`][yohou.plotting.plot_cv_results_scatter] : Plot hyperparameter search results.
    `ExpandingWindowSplitter` : Expanding window cross-validation.
    `SlidingWindowSplitter` : Sliding window cross-validation.
    """
    # Validate inputs
    validate_plotting_data(y)
    validate_plotting_params(width=width, height=height)

    if not isinstance(splitter, BaseSplitter):
        msg = f"Expected BaseSplitter, got {type(splitter).__name__}"
        raise TypeError(msg)

    # Get colors
    colors = resolve_color_palette(None, 2)
    train_color = train_color or colors[0]
    test_color = test_color or colors[1]

    # Get splits
    splits = list(splitter.split(y, X_actual))
    n_splits = len(splits)
    if n_splits == 0:
        msg = (
            "Splitter produced no splits. Check that 'y' has enough rows for the configured 'n_splits' and 'test_size'."
        )
        raise ValueError(msg)

    # Create figure
    fig = _create_figure(resampler)

    # Get time column
    times = y["time"]

    # Plot each split (one row per fold: train segment + test segment).
    for i, (train_idx, test_idx) in enumerate(splits):
        _add_split_row(
            fig,
            times,
            train_idx,
            test_idx,
            f"Fold {i + 1}",
            train_color=train_color,
            test_color=test_color,
            train_name=train_name,
            test_name=test_name,
            line_width=line_width,
            show_train_legend=(i == 0),
            show_test_legend=(i == 0),
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
    fig.update_layout(showlegend=show_legend)

    return fig


def _plot_nested_all(
    y: pl.DataFrame,
    times: pl.Series,
    outer_splits: list,
    inner_splitter: BaseSplitter,
    X_actual: pl.DataFrame | None,
    *,
    train_color: str,
    test_color: str,
    outer_train_color: str,
    outer_test_color: str,
    train_name: str,
    test_name: str,
    outer_train_name: str,
    outer_test_name: str,
    show_legend: bool,
    title: str,
    x_label: str,
    y_label: str,
    width: int | None,
    height: int | None,
    resampler: bool | Literal["widget"] | None,
    line_width: float,
    facet_n_cols: int,
) -> go.Figure:
    """Render one subplot per outer fold (faceted nested-CV view).

    Used by :func:`plot_nested_splits` when ``outer_fold="all"``. A
    :class:`~yohou.plotting._utils.LegendTracker` ensures each legend entry appears
    once across all subplots.
    """
    n_outer = len(outer_splits)
    n_cols = min(n_outer, max(1, facet_n_cols))
    n_rows = math.ceil(n_outer / n_cols)
    n_cells = n_rows * n_cols
    subplot_titles = [f"Outer fold {k + 1}" if k < n_outer else "" for k in range(n_cells)]

    fig = _create_subplots(
        resampler,
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=_subplot_spacing(n_rows),
    )
    tracker = LegendTracker(show_legend)

    for k, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        row = k // n_cols + 1
        col = k % n_cols + 1
        y_inner = y[outer_train_idx]
        x_inner = X_actual[outer_train_idx] if X_actual is not None else None
        inner_times = y_inner["time"]
        try:
            inner_splits = list(inner_splitter.split(y_inner, x_inner))
        except Exception as exc:
            msg = f"inner split failed for outer fold {k + 1} (training slice of {len(outer_train_idx)} rows): {exc}"
            raise ValueError(msg) from exc

        for j, (inner_train_idx, inner_test_idx) in enumerate(inner_splits):
            _add_split_row(
                fig,
                inner_times,
                inner_train_idx,
                inner_test_idx,
                f"Inner fold {j + 1}",
                train_color=train_color,
                test_color=test_color,
                train_name=train_name,
                test_name=test_name,
                line_width=line_width,
                show_train_legend=tracker.should_show(train_name),
                show_test_legend=tracker.should_show(test_name),
                train_group="inner_train",
                test_group="inner_val",
                row=row,
                col=col,
            )
        # Outer evaluation row: refit window + held-out scored test window.
        _add_split_row(
            fig,
            times,
            outer_train_idx,
            outer_test_idx,
            "Outer fold",
            train_color=outer_train_color,
            test_color=outer_test_color,
            train_name=outer_train_name,
            test_name=outer_test_name,
            line_width=line_width,
            show_train_legend=tracker.should_show(outer_train_name),
            show_test_legend=tracker.should_show(outer_test_name),
            train_group="outer_train",
            test_group="outer_test",
            row=row,
            col=col,
        )

    height_default = height or max(400, n_rows * 320)
    fig = apply_default_layout(fig, title=title, width=width, height=height_default)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    fig.update_layout(showlegend=show_legend)
    return fig


def plot_nested_splits(
    y: pl.DataFrame,
    outer_splitter: BaseSplitter,
    inner_splitter: BaseSplitter,
    *,
    outer_fold: int | Literal["all"] = -1,
    X_actual: pl.DataFrame | None = None,
    train_color: str | None = None,
    test_color: str | None = None,
    outer_train_color: str | None = None,
    outer_test_color: str | None = None,
    train_name: str = "Inner train",
    test_name: str = "Inner validation",
    outer_train_name: str = "Outer train (refit, best params)",
    outer_test_name: str = "Outer test (scored)",
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    resampler: bool | Literal["widget"] | None = None,
    line_width: float = 10.0,
    facet_n_cols: int = 2,
) -> go.Figure:
    """
    Plot a nested cross-validation as a timeline visualization.

    Shows both levels of a nested cross-validation. The **inner** rows are the
    cross-validation carved from one outer fold's training window, which selects the
    hyperparameters. The **outer** row above them shows the full outer training
    window the model is then refit on with those hyperparameters, plus the outer test
    window it is scored on (held out from the inner tuning). The figure is generated
    from the splitters themselves, so it is faithful to actual splitting behavior
    rather than hand-drawn.

    Parameters
    ----------
    y : pl.DataFrame
        Target time series with mandatory "time" column.
    outer_splitter : BaseSplitter
        Splitter for the outer (evaluation) loop.
    inner_splitter : BaseSplitter
        Splitter for the inner (tuning) loop, applied to an outer fold's training slice.
    outer_fold : int or "all", default=-1
        Which outer fold to render. An integer selects a single outer fold (default
        ``-1``, the last fold, whose training window is largest). ``"all"`` renders one
        subplot per outer fold.
    X_actual : pl.DataFrame or None, default=None
        Actual features passed to ``split()``. Sliced to the outer training window for
        the inner split.
    train_color, test_color, outer_train_color, outer_test_color : str or None, default=None
        Colors for the inner-train, inner-validation, outer-refit-train, and
        outer-test segments. If None, the first four colors from the yohou palette
        are used.
    train_name : str, default="Inner train"
        Legend label for the inner training segments.
    test_name : str, default="Inner validation"
        Legend label for the inner validation segments.
    outer_train_name : str, default="Outer train (refit, best params)"
        Legend label for the outer training window the model is refit on.
    outer_test_name : str, default="Outer test (scored)"
        Legend label for the outer test window the refit model is scored on.
    show_legend : bool, default=True
        Whether to show the legend.
    title : str or None, default=None
        Plot title. Defaults to "Nested Cross-Validation Splits".
    x_label : str or None, default=None
        X-axis label. Defaults to "Time".
    y_label : str or None, default=None
        Y-axis label. Defaults to "Fold".
    width : int or None, default=None
        Plot width in pixels.
    height : int or None, default=None
        Plot height in pixels.
    resampler : bool or "widget" or None, default=None
        Enable plotly-resampler for large datasets.
    line_width : float, default=10.0
        Width of the segment bars.
    facet_n_cols : int, default=2
        Number of subplot columns when ``outer_fold="all"``.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Raises
    ------
    TypeError
        If a splitter is not a ``BaseSplitter``.
    ValueError
        If ``y`` is empty or missing 'time', ``outer_fold`` is out of range, or an
        outer fold's training slice is too small for the inner splitter.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.plotting import plot_nested_splits
    >>> from yohou.model_selection import ExpandingWindowSplitter

    >>> y = pl.DataFrame({
    ...     "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
    ...     "value": list(range(366)),
    ... })

    >>> fig = plot_nested_splits(
    ...     y,
    ...     ExpandingWindowSplitter(n_splits=3, test_size=30),
    ...     ExpandingWindowSplitter(n_splits=3, test_size=20),
    ... )
    >>> len(fig.data) > 0
    True

    See Also
    --------
    [`plot_splits`][yohou.plotting.plot_splits] : Plot a flat (single-level) CV.
    `ExpandingWindowSplitter` : Expanding window cross-validation.
    """
    validate_plotting_data(y)
    validate_plotting_params(width=width, height=height)
    for label, splitter in (("outer_splitter", outer_splitter), ("inner_splitter", inner_splitter)):
        if not isinstance(splitter, BaseSplitter):
            msg = f"Expected BaseSplitter for {label}, got {type(splitter).__name__}"
            raise TypeError(msg)

    colors = resolve_color_palette(None, 4)
    train_color = train_color or colors[0]
    test_color = test_color or colors[1]
    outer_train_color = outer_train_color or colors[2]
    outer_test_color = outer_test_color or colors[3]

    times = y["time"]
    outer_splits = list(outer_splitter.split(y, X_actual))
    n_outer = len(outer_splits)

    title_default = title or "Nested Cross-Validation Splits"
    x_label_default = x_label or "Time"
    y_label_default = y_label or "Fold"

    if outer_fold == "all":
        return _plot_nested_all(
            y,
            times,
            outer_splits,
            inner_splitter,
            X_actual,
            train_color=train_color,
            test_color=test_color,
            outer_train_color=outer_train_color,
            outer_test_color=outer_test_color,
            train_name=train_name,
            test_name=test_name,
            outer_train_name=outer_train_name,
            outer_test_name=outer_test_name,
            show_legend=show_legend,
            title=title_default,
            x_label=x_label_default,
            y_label=y_label_default,
            width=width,
            height=height,
            resampler=resampler,
            line_width=line_width,
            facet_n_cols=facet_n_cols,
        )

    if not isinstance(outer_fold, int):
        msg = f"outer_fold must be an int or 'all', got {outer_fold!r}"
        raise ValueError(msg)
    if not -n_outer <= outer_fold < n_outer:
        msg = f"outer_fold={outer_fold} is out of range for {n_outer} outer folds"
        raise ValueError(msg)

    outer_train_idx, outer_test_idx = outer_splits[outer_fold]
    y_inner = y[outer_train_idx]
    x_inner = X_actual[outer_train_idx] if X_actual is not None else None
    inner_times = y_inner["time"]
    inner_splits = list(inner_splitter.split(y_inner, x_inner))

    fig = _create_figure(resampler)
    for j, (inner_train_idx, inner_test_idx) in enumerate(inner_splits):
        _add_split_row(
            fig,
            inner_times,
            inner_train_idx,
            inner_test_idx,
            f"Inner fold {j + 1}",
            train_color=train_color,
            test_color=test_color,
            train_name=train_name,
            test_name=test_name,
            line_width=line_width,
            show_train_legend=(j == 0),
            show_test_legend=(j == 0),
            train_group="inner_train",
            test_group="inner_val",
        )
    # Outer evaluation row: the full outer training window the model is refit on with
    # the best hyperparameters, plus the held-out outer test window it is scored on.
    _add_split_row(
        fig,
        times,
        outer_train_idx,
        outer_test_idx,
        "Outer fold",
        train_color=outer_train_color,
        test_color=outer_test_color,
        train_name=outer_train_name,
        test_name=outer_test_name,
        line_width=line_width,
        show_train_legend=show_legend,
        show_test_legend=show_legend,
        train_group="outer_train",
        test_group="outer_test",
    )

    n_plot_rows = len(inner_splits) + 1
    height_default = height or (300 + n_plot_rows * 30)
    fig = apply_default_layout(
        fig,
        title=title_default,
        x_label=x_label_default,
        y_label=y_label_default,
        width=width,
        height=height_default,
    )
    fig.update_layout(showlegend=show_legend)
    return fig


def plot_cv_results_scatter(
    cv_results: dict,
    param_name: str,
    scorer_name: str | None = None,
    *,
    higher_is_better: bool = True,
    highlight_best: bool = True,
    color_palette: list[str] | None = None,
    show_legend: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    marker_size: float = 10.0,
    marker_opacity: float = 0.8,
    best_marker_size: float = 16.0,
    best_marker_color: str = "#dc2626",
    show_std: bool = True,
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
        original (un-negated) scores, selecting the maximum when
        ``higher_is_better`` is True and the minimum otherwise.
    highlight_best : bool, default=True
        Whether to highlight the best parameter value.
    color_palette : list[str] | None, default=None
        Custom color palette. If None, uses yohou palette.
    show_legend : bool, default=True
        Whether to show the legend.
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
    marker_size : float, default=10.0
        Size of the scatter markers.
    marker_opacity : float, default=0.8
        Opacity of scatter markers.
    best_marker_size : float, default=16.0
        Size of the best-result star marker.
    best_marker_color : str, default="#dc2626"
        Color of the best-result star marker.
    show_std : bool, default=True
        Whether to show error bars (if std_test_{scorer} exists in cv_results).

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
    [`plot_splits`][yohou.plotting.plot_splits] : Plot cross-validation splits.
    `GridSearchCV` : Grid search with cross-validation.
    `RandomizedSearchCV` : Randomized search with cross-validation.
    """
    # Construct key names
    param_key = f"param_{param_name}"
    validate_plotting_params(width=width, height=height)

    # Auto-detect scorer name if not provided
    if scorer_name is None:
        # Look for mean_test_* keys
        mean_test_keys = [k for k in cv_results if k.startswith("mean_test_")]
        if not mean_test_keys:
            msg = (
                "No 'mean_test_*' keys found in cv_results. Available keys: "
                f"{list(cv_results.keys())}. Pass scorer_name= explicitly."
            )
            raise ValueError(msg)
        scorer_name = mean_test_keys[0].replace("mean_test_", "")

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
    show_std_effective = show_std and std_scores is not None

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
        # Fall back to the optimal raw score (before negation); the direction
        # depends on whether higher or lower raw scores are better.
        best_raw = max(mean_scores_raw) if higher_is_better else min(mean_scores_raw)
        best_idx = list(mean_scores_raw).index(best_raw)

    # Add scatter trace with optional error bars
    error_y = None
    if show_std_effective:
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
                "opacity": marker_opacity,
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
    fig.update_layout(showlegend=show_legend)

    return fig
