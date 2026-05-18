"""Data validation functions for different estimator types."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, overload

import polars as pl
import polars.selectors as cs

from yohou.utils.panel import inspect_panel
from yohou.utils.polars import get_numeric_columns, is_categorical_dtype
from yohou.utils.validation import (
    check_continuity,
    check_groups,
    check_inputs,
    check_interval_consistency,
    check_panel_groups_match,
    check_panel_internal_consistency,
    check_schema,
    check_scorer_column_selection,
    check_sufficient_rows,
    check_time_column,
    interval_to_timedelta,
    parse_interval,
)

__all__ = [
    "validate_forecaster_data",
    "validate_plotting_data",
    "validate_plotting_params",
    "validate_scorer_data",
    "validate_splitter_data",
    "validate_time_weight",
    "validate_transformer_data",
]


def validate_plotting_data(
    df: pl.DataFrame,
    *,
    columns: str | list[str] | None = None,
    groups: list[str] | None = None,
    min_rows: int = 1,
    exclude: list[str] | None = None,
    include_categorical: bool = False,
) -> list[str]:
    """Validate a DataFrame for plotting and resolve columns.

    Combines type checking, structural validation, and column resolution
    into a single entry point for all plotting functions.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame to validate.
    columns : str, list of str, or None, default=None
        Column specification.  When ``groups`` is ``None`` this
        selects standard columns (``None`` means all numeric).  For panel
        data this selects *member postfixes* within the requested groups.
    groups : list of str or None, default=None
        When not ``None``, resolve panel columns
        (``group__member`` pattern) instead of plain columns.
    min_rows : int, default=1
        Minimum number of rows required.
    exclude : list of str or None, default=None
        Column names to exclude when ``columns=None`` and
        ``groups`` is ``None``.
    include_categorical : bool, default=False
        When ``True`` and ``columns=None``, also include
        ``pl.String``, ``pl.Categorical``, and ``pl.Enum`` columns
        alongside numeric columns.

    Returns
    -------
    list[str]
        Resolved column names to plot.

    Raises
    ------
    TypeError
        If *df* is not a ``pl.DataFrame``.
    ValueError
        If the DataFrame is empty, missing a ``"time"`` column, or the
        requested columns do not exist.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.utils.validate_data import validate_plotting_data
    >>> df = pl.DataFrame({"time": [1, 2, 3], "y": [10.0, 20.0, 30.0]})
    >>> validate_plotting_data(df, exclude=["time"])
    ['y']

    >>> df_panel = pl.DataFrame({
    ...     "time": [1, 2],
    ...     "sales__a": [10, 20],
    ...     "sales__b": [30, 40],
    ... })
    >>> validate_plotting_data(df_panel, groups=["sales"], columns="a")
    ['sales__a']

    See Also
    --------
    - [`validate_plotting_params`][yohou.utils.validate_data.validate_plotting_params] : Validate common plotting parameters.
    - [`get_numeric_columns`][yohou.utils.polars.get_numeric_columns] : Resolve numeric columns from a DataFrame.
    - [`inspect_panel`][yohou.utils.panel.inspect_panel] : Detect panel group structure.

    """
    if not isinstance(df, pl.DataFrame):
        msg = f"Expected pl.DataFrame, got {type(df).__name__}"
        raise TypeError(msg)

    check_sufficient_rows(df, min_rows=min_rows, context="for plotting")

    # Non-datetime time columns (e.g. integer indices) only need existence check
    if "time" in df.columns and not isinstance(df["time"].dtype, pl.Datetime | pl.Date):
        pass  # integer time is acceptable for plotting
    elif "vintage_time" in df.columns:
        _check_multi_vintage_time(df)
    else:
        check_time_column(df)

    # Panel column resolution
    if groups is not None:
        if isinstance(columns, str):
            columns = [columns]

        _, panels = inspect_panel(df)
        cols: list[str] = []
        for prefix, members in panels.items():
            if prefix not in groups:
                continue
            if columns is not None:
                for member in members:
                    _, _, postfix = member.partition("__")
                    if postfix in columns:
                        cols.append(member)
            else:
                cols.extend(members)

        if not cols:
            if columns is not None:
                msg = f"No panel columns found for groups={groups} with members={columns}"
            else:
                msg = f"No panel columns found for groups: {groups}"
            raise ValueError(msg)
        return cols

    # Standard column resolution
    if columns is None:
        result = get_numeric_columns(df, exclude=exclude)
        if include_categorical:
            _excl = set(exclude or [])
            result += [
                c for c in df.columns if is_categorical_dtype(df[c].dtype) and c not in _excl and c not in result
            ]
        return result

    if isinstance(columns, str):
        columns = [columns]

    missing = [col for col in columns if col not in df.columns]
    if missing:
        msg = f"Columns not found in DataFrame: {missing}"
        raise ValueError(msg)

    return columns


def validate_plotting_params(
    *,
    kind: str | None = None,
    valid_kinds: set[str] | None = None,
    facet_n_cols: int | None = None,
    n_bins: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """Validate common plotting function parameters.

    Parameters
    ----------
    kind : str or None
        Plot sub-type to validate.
    valid_kinds : set of str or None
        Allowed values for *kind*.  Both must be provided together.
    facet_n_cols : int or None
        Number of facet columns (must be >= 1).
    n_bins : int or None
        Number of histogram bins (must be >= 1).
    width : int or None
        Figure width in pixels (must be >= 1).
    height : int or None
        Figure height in pixels (must be >= 1).

    Raises
    ------
    ValueError
        If any parameter is invalid.

    Examples
    --------
    >>> from yohou.utils.validate_data import validate_plotting_params
    >>> validate_plotting_params(kind="line", valid_kinds={"line", "bar"})
    >>> validate_plotting_params(facet_n_cols=2, n_bins=10)

    See Also
    --------
    - [`validate_plotting_data`][yohou.utils.validate_data.validate_plotting_data] : Validate DataFrame structure for plotting.

    """
    if kind is not None and valid_kinds is not None and kind not in valid_kinds:
        msg = f"kind must be one of {sorted(valid_kinds)!r}, got {kind!r}"
        raise ValueError(msg)
    if facet_n_cols is not None and facet_n_cols < 1:
        msg = f"facet_n_cols must be >= 1, got {facet_n_cols}"
        raise ValueError(msg)
    if n_bins is not None and n_bins < 1:
        msg = f"n_bins must be >= 1, got {n_bins}"
        raise ValueError(msg)
    if width is not None and width < 1:
        msg = f"width must be >= 1, got {width}"
        raise ValueError(msg)
    if height is not None and height < 1:
        msg = f"height must be >= 1, got {height}"
        raise ValueError(msg)


if TYPE_CHECKING:
    from yohou.base import BaseForecaster, BaseTransformer
    from yohou.metrics._context import ScoringContext
    from yohou.metrics.base import BaseScorer
    from yohou.model_selection import BaseSplitter


def validate_time_weight(
    time_weight: Callable | pl.DataFrame | None,
    y: pl.DataFrame,
    groups: list[str] | None = None,
) -> None:
    """Validate time_weight parameter for forecasters and scorers.

    Parameters
    ----------
    time_weight : callable, pl.DataFrame, or None
        Time weighting specification to validate.
    y : pl.DataFrame
        Target time series with "time" column.
    groups : list of str or None
        Panel group names if panel data.

    Raises
    ------
    ValueError
        If time_weight validation fails.

    See Also
    --------
    - [`validate_forecaster_data`][yohou.utils.validate_data.validate_forecaster_data] : Validate forecaster input data.
    - [`validate_scorer_data`][yohou.utils.validate_data.validate_scorer_data] : Validate scorer input data.

    """
    if time_weight is None:
        return

    if callable(time_weight):
        # Callable validation is done via validate_callable_signature
        # in the actual processing methods
        return

    # DataFrame validation
    if not isinstance(time_weight, pl.DataFrame):
        raise ValueError(f"time_weight must be callable, pl.DataFrame, or None, got {type(time_weight).__name__}")

    # Must have time column
    if "time" not in time_weight.columns:
        raise ValueError("time_weight DataFrame must have 'time' column")

    # Check for weight columns
    weight_cols = [c for c in time_weight.columns if c != "time"]
    if not weight_cols:
        raise ValueError(
            "time_weight DataFrame must have at least one weight column "
            "('weight' for global data or '{group}_weight' for panel data)"
        )

    # Validate weight column naming
    if groups is None:
        # Global data: must have "weight" column
        if "weight" not in time_weight.columns:
            raise ValueError("time_weight DataFrame for global data must have 'weight' column")
        weight_cols_to_check = ["weight"]
    else:
        # Panel data: check for group-specific or global weight columns
        expected_group_cols = {f"{group}_weight" for group in groups}
        has_group_specific = any(col in time_weight.columns for col in expected_group_cols)
        has_global = "weight" in time_weight.columns

        if not has_group_specific and not has_global:
            raise ValueError(
                f"time_weight DataFrame for panel data must have either "
                f"group-specific columns {sorted(expected_group_cols)} "
                f"or global 'weight' column"
            )

        # Collect all weight columns to validate
        weight_cols_to_check = [c for c in weight_cols if c.endswith("_weight") or c == "weight"]

    # Validate weight values (non-negative, finite, non-zero sum)
    for col in weight_cols_to_check:
        if col not in time_weight.columns:
            continue

        weights = time_weight[col]

        # Check for NaN
        if weights.is_null().any():
            raise ValueError(f"Weight column '{col}' contains NaN values")

        # Check for negative values
        if (weights < 0).any():
            raise ValueError(f"Weight column '{col}' contains negative values")

        # Check for infinite values
        if weights.is_infinite().any():
            raise ValueError(f"Weight column '{col}' contains infinite values")

        # Check for all-zero weights
        if weights.sum() == 0:
            raise ValueError(f"Weight column '{col}' sums to zero")


def _compute_forecasting_step(
    time: pl.Series,
    vintage_time: pl.Series,
    interval: str,
) -> pl.Series:
    """Compute integer forecasting step from time, vintage_time, and interval.

    For fixed-length intervals (daily, hourly, etc.), divides the duration
    between ``time`` and ``vintage_time`` by the interval timedelta.

    For variable-length intervals (monthly, quarterly, yearly), uses
    calendar-unit arithmetic instead.

    Parameters
    ----------
    time : pl.Series
        Prediction time points.
    vintage_time : pl.Series
        Observation origin time points.
    interval : str
        Interval string (e.g. ``"1d"``, ``"1mo"``, ``"1q"``, ``"1y"``).

    Returns
    -------
    pl.Series
        Integer forecasting steps (1-indexed).

    """
    td = interval_to_timedelta(interval)

    if td is not None:
        # Fixed-length interval: fast duration division
        step_durations = time - vintage_time
        return (step_durations / td).round(0).cast(pl.Int64)

    # Variable-length interval: calendar-unit arithmetic
    multiplier, unit = parse_interval(interval)

    time_expr = time.to_frame("t")
    obs_expr = vintage_time.to_frame("o")
    combined = pl.concat([time_expr, obs_expr], how="horizontal")

    if unit in ("mo", "q"):
        months_per_unit = multiplier if unit == "mo" else multiplier * 3
        month_diff = (combined["t"].dt.year() - combined["o"].dt.year()) * 12 + (
            combined["t"].dt.month() - combined["o"].dt.month()
        )
        return (month_diff // months_per_unit).cast(pl.Int64)

    if unit == "y":
        year_diff = combined["t"].dt.year() - combined["o"].dt.year()
        return (year_diff // multiplier).cast(pl.Int64)

    msg = f"Unsupported variable-length interval unit: {unit!r}"
    raise ValueError(msg)


def _check_multi_vintage_time(y_pred: pl.DataFrame) -> None:
    """Validate time column for multi-vintage predictions.

    Multi-vintage predictions have an ``vintage_time`` column that groups
    rows by forecast origin.  The global ``time`` column is not
    monotonically sorted, but time *within each vintage* must be.

    """
    if "time" not in y_pred.columns:
        raise ValueError("y_pred must contain a 'time' column.")

    time_col = y_pred["time"]
    if not isinstance(time_col.dtype, pl.Datetime | pl.Date):
        raise ValueError(f"'time' column in y_pred must have dtype pl.Datetime or pl.Date, but got {time_col.dtype}")
    if time_col.null_count() > 0:
        raise ValueError(
            f"'time' column in y_pred contains {time_col.null_count()} null values. "
            "'time' column must not have missing values."
        )

    # Check sorting within each vintage
    for ot in y_pred["vintage_time"].unique().sort():
        vintage = y_pred.filter(pl.col("vintage_time") == ot)
        if not vintage["time"].is_sorted():
            raise ValueError(f"'time' column within vintage_time={ot} is not sorted in ascending order.")


def _truncate_partial_vintage(
    y_true: pl.DataFrame,
    y_pred: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Remove the last vintage if it came from a partial observation window.

    When ``observe_predict`` runs with ``len(y) % stride != 0``, the last
    observe call consumes fewer rows than ``stride``, producing a partial
    vintage.  This function detects that case by comparing consecutive
    ``vintage_time`` gaps and drops the partial vintage.

    Requires at least 3 unique ``vintage_time`` values to infer the
    regular stride; otherwise returns data unchanged.
    """
    unique_times = y_pred["vintage_time"].unique().sort()
    if len(unique_times) < 3:
        return y_true, y_pred

    diffs = unique_times.diff().drop_nulls()
    regular_diff = diffs.mode()[0]
    last_diff = diffs[-1]

    if last_diff < regular_diff:
        partial_time = unique_times[-1]
        mask = y_pred["vintage_time"] != partial_time
        y_pred = y_pred.filter(mask)
        y_true = y_true.filter(mask)

    return y_true, y_pred


@overload
def validate_scorer_data(
    scorer: BaseScorer,
    y_true: pl.DataFrame,
    y_pred: None = None,
    *,
    scores: None = None,
    reset: bool = True,
    inverse: bool = False,
) -> tuple[pl.DataFrame, None, None]: ...


@overload
def validate_scorer_data(
    scorer: BaseScorer,
    y_true: None = None,
    y_pred: pl.DataFrame = ...,
    *,
    scores: pl.DataFrame = ...,
    reset: bool = False,
    inverse: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, ScoringContext]: ...


@overload
def validate_scorer_data(
    scorer: BaseScorer,
    y_true: pl.DataFrame = ...,
    y_pred: pl.DataFrame = ...,
    *,
    scores: None = None,
    reset: bool = False,
    inverse: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame, ScoringContext]: ...


def validate_scorer_data(
    scorer: BaseScorer,
    y_true: pl.DataFrame | None = None,
    y_pred: pl.DataFrame | None = None,
    *,
    scores: pl.DataFrame | None = None,
    reset: bool = False,
    inverse: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame | None, ScoringContext | None]:
    """Validate and prepare scorer input data.

    Parameters
    ----------
    scorer : BaseScorer
        The scorer instance calling this function.
    y_true : pl.DataFrame, default=None
        True values with "time" column.
        - In fit context (reset=True): This is y_train. Always required.
        - In inverse context: Can be None (use scores parameter instead).
        - In normal score context: Always required.
    y_pred : pl.DataFrame, default=None
        Predicted values with "time" column. Required in normal score context.
    scores : pl.DataFrame, default=None
        Conformity scores with "time" column. Required when inverse=True.
    reset : bool, default=False
        If True, validate in fit context (skips prediction structure checks).
        Implies align_by_time=False and drop_time_columns=True.
    inverse : bool, default=False
        If True, validate in inverse_score context. Requires scores parameter.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame | None, ScoringContext | None]
        Validated and prepared DataFrames and scoring context:
        - Normal context: (y_true, y_pred, ScoringContext)
        - Inverse context: (y_pred, scores, ScoringContext)
        - Fit context (reset=True): (y_train, None, None)

    Notes
    -----
    - When drop_time_columns=False, time column is ALWAYS first in output
    - Performs basic validation: None checks, time column existence, panel consistency
    - Time alignment preserves common time points only (inner join)
    - Time values are extracted before validation for point/interval scorers

    See Also
    --------
    - [`BaseScorer`][yohou.metrics.base.BaseScorer] : Base class for all scorers.
    - [`validate_time_weight`][yohou.utils.validate_data.validate_time_weight] : Validate time weighting parameters.

    """
    # Runtime validation: enforce parameter requirements for each context
    if not inverse and not reset and y_pred is None:
        # Normal score context: y_pred required
        raise ValueError("`y_pred` cannot be None for scoring. Set reset=True for fit/calibration.")

    if inverse:
        # Type narrowing: inverse scoring requires y_pred and scores
        if y_pred is None:
            raise ValueError("`y_pred` is required for inverse scoring. Cannot be None.")
        if scores is None:
            raise ValueError("`scores` is required for inverse scoring. Cannot be None.")

        # Validate time columns (required)
        check_time_column(y_pred)
        check_time_column(scores)

        # Check column schema compatibility (exclude time/vintage_time columns)
        exclude_cols_pred = ["time"]
        exclude_cols_scores = ["time"]

        if "vintage_time" in y_pred.columns:
            exclude_cols_pred.append("vintage_time")

        y_pred_cols = set(y_pred.select(~cs.by_name(*exclude_cols_pred)).columns)
        score_cols = set(scores.select(~cs.by_name(*exclude_cols_scores)).columns)

        if y_pred_cols != score_cols:
            raise ValueError(
                f"Column mismatch between y_pred and conformity_scores. "
                f"y_pred has {sorted(y_pred_cols)}, conformity_scores has {sorted(score_cols)}."
            )

        # Extract vintage_time before dropping (used by multi-vintage aggregation)
        vintage_time = None
        forecasting_step = None
        if "vintage_time" in y_pred.columns:
            vintage_time = y_pred["vintage_time"]
            # Compute forecasting step if scorer has interval_ attribute
            if hasattr(scorer, "interval_") and scorer.interval_ is not None:
                forecasting_step = _compute_forecasting_step(y_pred["time"], y_pred["vintage_time"], scorer.interval_)
            y_pred = y_pred.drop("vintage_time")

        # Extract time values before dropping (always present after validation)
        time_values = y_pred["time"].to_list()

        # Drop time columns for consistency with normal path
        y_pred = y_pred.drop("time")
        scores = scores.drop("time")

        from yohou.metrics._context import ScoringContext as _ScoringContext  # noqa: PLC0415

        return (
            y_pred,
            scores,
            _ScoringContext(
                time_values=time_values,
                vintage_time=vintage_time,
                forecasting_step=forecasting_step,
            ),
        )

    if reset:
        # At fit time, y_true is y_train (always required), y_pred is always None
        if y_true is None:
            raise ValueError("`y_train` is required for scorer.fit(). Cannot be None.")

        check_time_column(y_true)

        # Validate seasonality for scorers with seasonality parameter
        # y_true still has time column, so subtract 1 for data rows
        if hasattr(scorer, "seasonality"):
            seasonality_val = scorer.seasonality
            if isinstance(seasonality_val, int) and len(y_true) <= seasonality_val:
                raise ValueError(
                    f"Training data length ({len(y_true) - 1}) must be greater than "
                    f"seasonality ({seasonality_val}). Cannot compute seasonal naive forecast errors."
                )

        # At fit time: drop time from y_train
        y_true = y_true.drop("time")

        return y_true, None, None

    # At score time, y_true is always required
    if y_true is None:
        raise ValueError("`y_true` cannot be None for scorer.")

    if y_pred is None:
        raise ValueError("`y_pred` cannot be None for scorer.")

    # Validate time columns
    check_time_column(y_true)
    # Multi-vintage predictions (from observe_predict with stride) have
    # repeating time values across vintages, so the global time column is
    # not monotonically sorted.  Validate per-vintage sorting instead.
    if "vintage_time" in y_pred.columns:
        _check_multi_vintage_time(y_pred)
    else:
        check_time_column(y_pred)

    tags = scorer.__sklearn_tags__()
    scorer_tags = getattr(tags, "scorer_tags", None)
    pred_type = getattr(scorer_tags, "prediction_type", None) if scorer_tags is not None else None

    if pred_type is None:
        raise ValueError("Scorer tags must have prediction_type attribute")

    # Panel consistency check
    _, y_groups = inspect_panel(y_true)
    _, X_groups = inspect_panel(y_pred)
    if set(y_groups.keys()) != set(X_groups.keys()):
        raise ValueError(
            f"Panel groups mismatch. `y_true` has {sorted(y_groups.keys())}. `y_pred` has {sorted(X_groups.keys())}."
        )

    # Validate column presence and types
    for col in y_true.columns:
        if col == "time":
            continue

        if pred_type == "point":
            if col not in y_pred.columns:
                raise ValueError(f"'{col}' is present in `y_true` but missing in `y_pred`.")
            # Relaxed check: do not enforce exact dtype match (e.g. Int64 vs Float64 is fine)
            # But ensure both are numeric to avoid invalid operations
            if not (y_true.schema[col].is_numeric() and y_pred.schema[col].is_numeric()):
                raise ValueError(
                    f"Column '{col}' type mismatch. `y_true`: {y_true.schema[col]}, "
                    f"`y_pred`: {y_pred.schema[col]}. Both must be numeric."
                )
        elif pred_type == "interval":
            related_cols = [c for c in y_pred.columns if c.startswith(f"{col}_lower_") or c.startswith(f"{col}_upper_")]
            lower_found = any(c.startswith(f"{col}_lower_") for c in related_cols)
            upper_found = any(c.startswith(f"{col}_upper_") for c in related_cols)
            if not lower_found or not upper_found:
                raise ValueError(f"Interval columns for `y_true` '{col}' missing in `y_pred`.")

            for rc in related_cols:
                if not (y_true.schema[col].is_numeric() and y_pred.schema[rc].is_numeric()):
                    raise ValueError(
                        f"Column '{rc}' type mismatch. `y_true` '{col}': {y_true.schema[col]}, "
                        f"`y_pred`: {y_pred.schema[rc]}. Both must be numeric."
                    )
        elif pred_type == "class_proba":
            proba_cols = [c for c in y_pred.columns if c.startswith(f"{col}_proba_")]
            if not proba_cols:
                raise ValueError(
                    f"No probability columns found for target '{col}' in `y_pred`. "
                    f"Expected columns matching '{col}_proba_<class_label>'."
                )
            for pc in proba_cols:
                if not y_pred.schema[pc].is_numeric():
                    raise ValueError(f"Probability column '{pc}' must be numeric, got {y_pred.schema[pc]}.")

    # Align by time.
    # Use semi-joins so that duplicate times in y_pred (multi-vintage data
    # from observe_predict) are preserved rather than cross-producted.
    unique_truth_times = y_true.select("time").unique()
    unique_pred_times = y_pred.select("time").unique()

    y_pred = y_pred.join(unique_truth_times, on="time", how="semi")
    y_true_filtered = y_true.join(unique_pred_times, on="time", how="semi")

    # Replicate y_true rows to match y_pred (1:1 for single-vintage,
    # 1:N when y_pred has multiple vintages per time point).
    y_true = y_pred.select("time").join(y_true_filtered, on="time", how="left")

    # Subselect columns based on scorer configuration
    coverage_rates: list[float] | None = getattr(scorer, "coverage_rates", None)
    # Extract filter keys from polymorphic param (list or dict)
    if isinstance(coverage_rates, dict):
        coverage_rates = [float(k) for k in coverage_rates]
    interval_pattern = re.compile(r"^(.+)_(lower|upper)_([\d.]+)$")

    y_true, y_pred = check_scorer_column_selection(
        scorer=scorer,
        y_true=y_true,
        y_pred=y_pred,
        pred_type=pred_type,
        coverage_rates=coverage_rates,
        interval_pattern=interval_pattern,
    )

    # Truncate partial vintage: if the last observe_predict window was
    # shorter than the regular stride, drop that vintage before scoring.
    if "vintage_time" in y_pred.columns:
        y_true, y_pred = _truncate_partial_vintage(y_true, y_pred)

    # Extract time values before dropping (all scorers get time-less DataFrames)
    time_values = y_true["time"].to_list() if "time" in y_true.columns else None

    # Extract vintage_time and compute forecasting_step before dropping
    vintage_time: pl.Series | None = None
    forecasting_step: pl.Series | None = None

    if "vintage_time" in y_pred.columns:
        vintage_time = y_pred["vintage_time"]

        # Compute forecasting_step if scorer has interval_ from fit()
        interval_ = getattr(scorer, "interval_", None)
        if interval_ is not None and time_values is not None:
            forecasting_step = _compute_forecasting_step(y_pred["time"], vintage_time, interval_)

        y_pred = y_pred.drop("vintage_time")

    # Drop time columns for all scorers (conformity scorers can reconstruct from time_values)
    y_true = y_true.drop("time")

    if "time" in y_pred.columns:
        y_pred = y_pred.drop("time")

    # For point scorers, strip extra columns (e.g. interval _lower_/_upper_ columns
    # that may be present in mixed multimetric scenarios) to prevent schema mismatches.
    if pred_type == "point":
        extra_cols = [c for c in y_pred.columns if c not in y_true.columns]
        if extra_cols:
            y_pred = y_pred.drop(extra_cols)

    from yohou.metrics._context import ScoringContext as _ScoringContext  # noqa: PLC0415

    context = _ScoringContext(
        time_values=time_values,  # type: ignore
        vintage_time=vintage_time,
        forecasting_step=forecasting_step,
    )

    return y_true, y_pred, context


@overload
def validate_splitter_data(
    splitter: BaseSplitter, y: pl.DataFrame, X_actual: pl.DataFrame | None
) -> tuple[pl.DataFrame, pl.DataFrame | None]: ...


@overload
def validate_splitter_data(
    splitter: BaseSplitter, y: None, X_actual: pl.DataFrame | None
) -> tuple[None, pl.DataFrame | None]: ...


def validate_splitter_data(
    splitter: BaseSplitter, y: pl.DataFrame | None, X_actual: pl.DataFrame | None
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Validate and prepare input data for time series splitters.

    Checks that ``y`` and ``X_actual`` have valid ``"time"`` columns, consistent
    panel structure, and matching panel groups.  When ``y`` is provided the
    time interval is inferred and stored on the splitter as ``interval_``.

    Parameters
    ----------
    splitter : BaseSplitter
        The splitter instance.  ``interval_`` will be set on it when
        ``y`` is not ``None``.
    y : pl.DataFrame or None
        Target time series.  When ``None``, only ``X_actual`` is validated.
    X_actual : pl.DataFrame or None
        Exogenous features.  When ``None``, only ``y`` is validated.

    Returns
    -------
    tuple of (pl.DataFrame or None, pl.DataFrame or None)
        Validated ``(y, X_actual)`` pair.

    Raises
    ------
    ValueError
        If time columns are missing, panel groups are inconsistent, or
        input validation fails.

    See Also
    --------
    - [`BaseSplitter`][yohou.model_selection.split.BaseSplitter] : Base class for time series CV splitters.
    - [`check_inputs`][yohou.utils.validation.check_inputs] : Low-level input validation helper.

    """
    if y is not None:
        check_time_column(y)
        check_panel_internal_consistency(y, "y")

    if X_actual is not None:
        check_time_column(X_actual)
        check_panel_internal_consistency(X_actual, "X_actual")

        if y is not None:
            check_panel_groups_match(y, X_actual)

    # Type narrowing: check_inputs requires non-None y
    if y is not None:
        interval = check_inputs(y, X_actual)
        splitter.interval_ = interval

    return y, X_actual


def _validate_X_future_structure(X_future: pl.DataFrame) -> None:
    """Validate X_future has a time column of Date/Datetime type."""
    if "time" not in X_future.columns:
        raise ValueError(f"X_future must contain a 'time' column. Found columns: {list(X_future.columns)}")
    if not isinstance(X_future["time"].dtype, pl.Datetime | pl.Date):
        raise ValueError(
            f"'time' column in X_future must have dtype pl.Datetime or pl.Date, but got {X_future['time'].dtype}"
        )


def _validate_X_forecast_structure(X_forecast: pl.DataFrame) -> None:
    """Validate X_forecast has vintage_time and time columns of Date/Datetime type."""
    for col_name in ("vintage_time", "time"):
        if col_name not in X_forecast.columns:
            raise ValueError(
                f"X_forecast must contain a '{col_name}' column. Found columns: {list(X_forecast.columns)}"
            )
        if not isinstance(X_forecast[col_name].dtype, pl.Datetime | pl.Date):
            raise ValueError(
                f"'{col_name}' column in X_forecast must have dtype pl.Datetime or pl.Date, "
                f"but got {X_forecast[col_name].dtype}"
            )


def _validate_step_source_schema(
    df: pl.DataFrame,
    expected_schema: dict[str, pl.DataType],
    name: str,
    exclude_cols: set[str],
) -> None:
    """Validate that a step source DataFrame matches the fitted schema."""
    actual_schema = {col: dtype for col, dtype in df.schema.items() if col not in exclude_cols}
    if actual_schema != expected_schema:
        raise ValueError(f"{name} schema mismatch. Expected: {expected_schema}, got: {actual_schema}")


@overload
def validate_forecaster_data(
    forecaster: BaseForecaster,
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None = None,
    *,
    reset: Literal[True] = True,
    groups: list[str] | None = None,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame | None, None]: ...


@overload
def validate_forecaster_data(
    forecaster: BaseForecaster,
    y: None,
    X_actual: pl.DataFrame | None = None,
    *,
    reset: Literal[True] = True,
    groups: list[str] | None = None,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
) -> tuple[None, pl.DataFrame | None, None]: ...


@overload
def validate_forecaster_data(
    forecaster: BaseForecaster,
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None = None,
    *,
    reset: Literal[False],
    groups: list[str] | None = None,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame | None, list[str] | None]: ...


@overload
def validate_forecaster_data(
    forecaster: BaseForecaster,
    y: None,
    X_actual: pl.DataFrame | None = None,
    *,
    reset: Literal[False],
    groups: list[str] | None = None,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
) -> tuple[None, pl.DataFrame | None, list[str] | None]: ...


def validate_forecaster_data(
    forecaster: BaseForecaster,
    y: pl.DataFrame | None = None,
    X_actual: pl.DataFrame | None = None,
    *,
    reset: bool = True,
    groups: list[str] | None = None,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None, list[str] | None]:
    """Validate and prepare input data for forecasters.

    Handles two contexts: **fit** (``reset=True``) where time interval is
    inferred and stored on the forecaster, and **predict/update**
    (``reset=False``) where schemas and panel groups are validated
    against the fitted state.

    Parameters
    ----------
    forecaster : BaseForecaster
        The forecaster instance.  ``interval_`` and schema attributes are
        set during fit context.
    y : pl.DataFrame or None
        Target time series with ``"time"`` column.
    X_actual : pl.DataFrame or None, default=None
        Exogenous features with ``"time"`` column.
    reset : bool, default=True
        If ``True``, validate in fit context (infer interval, set schemas).
        If ``False``, validate in predict/update context (check schemas).
    groups : list of str or None, default=None
        Panel groups to validate. Normalized against the fitted groups
        when ``reset=False``.
    X_future : pl.DataFrame or None, default=None
        Known future features with a ``"time"`` column. Validated for
        structure and, when fitted, schema consistency.
    X_forecast : pl.DataFrame or None, default=None
        External forecasts with ``"vintage_time"`` and ``"time"`` columns.
        Validated for structure and, when fitted, schema consistency.

    Returns
    -------
    tuple of (pl.DataFrame or None, pl.DataFrame or None, list of str or None)
        Validated ``(y, X_actual, groups)``.  ``groups`` is
        ``None`` in fit context.

    Raises
    ------
    ValueError
        If time columns are missing, schema does not match fitted state,
        or panel groups are inconsistent.

    See Also
    --------
    - [`BaseForecaster`][yohou.base.forecaster.BaseForecaster] : Base class for all forecasters.
    - [`validate_time_weight`][yohou.utils.validate_data.validate_time_weight] : Validate time weighting parameters.
    - [`check_inputs`][yohou.utils.validation.check_inputs] : Low-level input validation helper.

    """
    # Validate X_future structure
    if X_future is not None:
        _validate_X_future_structure(X_future)

    # Validate X_forecast structure
    if X_forecast is not None:
        _validate_X_forecast_structure(X_forecast)

    if reset:
        # Fit context: validate and set interval
        # Type narrowing: check_inputs requires non-None y
        if y is not None:
            interval = check_inputs(y, X_actual)
            forecaster.interval_ = interval
        return y, X_actual, None

    # Predict/Update context (reset=False)

    # Validate time columns
    if y is not None:
        check_time_column(y)
    if X_actual is not None:
        check_time_column(X_actual)

    # Validate and normalize groups parameter
    groups = check_groups(
        fitted_panel_groups=forecaster.groups_,
        requested_panel_groups=groups,
    )

    # Validate schema and enforce column order
    if y is not None:
        y = check_schema(
            y,
            forecaster.local_y_schema_,
            groups=groups,
        )

    if X_actual is not None:
        # Handle panel data X_actual (local + global schemas)
        if forecaster.groups_ is not None:
            # Validate local X_actual columns (with panel prefixes)
            if hasattr(forecaster, "local_X_actual_schema_") and forecaster.local_X_actual_schema_:
                X_local = check_schema(
                    X_actual,
                    forecaster.local_X_actual_schema_,
                    groups=forecaster.groups_,
                )

            # Validate shared X_actual columns (no prefixes)
            X_shared = None
            if hasattr(forecaster, "shared_X_actual_schema_") and forecaster.shared_X_actual_schema_:
                X_shared = check_schema(X_actual, forecaster.shared_X_actual_schema_)

            # Reconstruct X_actual with both local and shared columns
            if (
                hasattr(forecaster, "local_X_actual_schema_")
                and forecaster.local_X_actual_schema_
                and hasattr(forecaster, "shared_X_actual_schema_")
                and forecaster.shared_X_actual_schema_
            ):
                assert X_shared is not None
                X_actual = pl.concat(
                    [X_local, X_shared.select(~cs.by_name("time"))],
                    how="horizontal",
                )
            elif hasattr(forecaster, "local_X_actual_schema_") and forecaster.local_X_actual_schema_:
                X_actual = X_local
            elif hasattr(forecaster, "shared_X_actual_schema_") and forecaster.shared_X_actual_schema_:
                X_actual = X_shared
        # Non-panel data: simple schema check (if schema exists)
        elif (
            X_actual is not None and hasattr(forecaster, "local_X_actual_schema_") and forecaster.local_X_actual_schema_
        ):
            X_actual = check_schema(X_actual, forecaster.local_X_actual_schema_)

    # Validate X_future schema against fitted state
    if X_future is not None:
        fitted_schema = getattr(forecaster, "_X_future_schema_", None)
        if fitted_schema is not None:
            _validate_step_source_schema(X_future, fitted_schema, "X_future", exclude_cols={"time"})

    # Validate X_forecast schema against fitted state
    if X_forecast is not None:
        fitted_schema = getattr(forecaster, "_X_forecast_schema_", None)
        if fitted_schema is not None:
            _validate_step_source_schema(X_forecast, fitted_schema, "X_forecast", exclude_cols={"time", "vintage_time"})

    return y, X_actual, groups


@overload
def validate_transformer_data(
    transformer: BaseTransformer,
    X: pl.DataFrame | None = None,
    *,
    reset: Literal[True],
    inverse: bool = False,
    X_t: pl.DataFrame | None = None,
    X_p: pl.DataFrame | None = None,
    observation_horizon: int | None = None,
    stateful: bool = False,
    **check_params,
) -> pl.DataFrame: ...


@overload
def validate_transformer_data(
    transformer: BaseTransformer,
    X: pl.DataFrame | None = None,
    *,
    reset: Literal[False],
    inverse: Literal[True],
    X_t: pl.DataFrame | None = None,
    X_p: pl.DataFrame | None = None,
    observation_horizon: int | None = None,
    stateful: Literal[True],
    **check_params,
) -> tuple[pl.DataFrame, pl.DataFrame]: ...


@overload
def validate_transformer_data(
    transformer: BaseTransformer,
    X: pl.DataFrame | None = None,
    *,
    reset: Literal[False],
    inverse: Literal[True],
    X_t: pl.DataFrame | None = None,
    X_p: pl.DataFrame | None = None,
    observation_horizon: int | None = None,
    stateful: Literal[False] = ...,
    **check_params,
) -> tuple[pl.DataFrame, None]: ...


@overload
def validate_transformer_data(
    transformer: BaseTransformer,
    X: pl.DataFrame | None = None,
    *,
    reset: Literal[False],
    inverse: Literal[False] = ...,
    X_t: pl.DataFrame | None = None,
    X_p: pl.DataFrame | None = None,
    observation_horizon: int | None = None,
    stateful: bool = False,
    **check_params,
) -> pl.DataFrame: ...


def validate_transformer_data(
    transformer: BaseTransformer,
    X: pl.DataFrame | None = None,
    *,
    reset: bool = True,
    inverse: bool = False,
    X_t: pl.DataFrame | None = None,
    X_p: pl.DataFrame | None = None,
    observation_horizon: int | None = None,
    stateful: bool = False,
    **check_params,
) -> pl.DataFrame | tuple[pl.DataFrame, pl.DataFrame | None] | tuple[pl.DataFrame, pl.DataFrame]:
    """Validate data for transformers.

    Parameters
    ----------
    transformer : BaseTransformer
        The transformer instance.
    X : pl.DataFrame or None, default=None
        Input data.
    reset : bool, default=True
        Whether this is a fit context.
    inverse : bool, default=False
        Whether this is an inverse transform context.
    X_t : pl.DataFrame or None, default=None
        Transformed data for inverse transform.
    X_p : pl.DataFrame or None, default=None
        Previous untransformed data for stateful inverse transform.
    observation_horizon : int or None, default=None
        Required observation horizon for inverse transform.
    stateful : bool, default=False
        If True (and inverse=True), X_p is required and guaranteed non-None in return.
        Use Literal[True] at call site for type narrowing.
    **check_params : dict
        Additional validation parameters.

    Returns
    -------
    pl.DataFrame or tuple
        Validated data.

    See Also
    --------
    - [`BaseTransformer`][yohou.base.transformer.BaseTransformer] : Base class for all transformers.
    - [`check_inputs`][yohou.utils.validation.check_inputs] : Low-level input validation helper.

    """
    if reset:
        # Fit context
        if X is None:
            raise ValueError("`X` cannot be None in fit context.")
        interval = check_inputs(X, None)
        transformer.interval_ = interval
        transformer.feature_names_in_ = X.select(~cs.by_name("time")).columns
        transformer.n_features_in_ = len(transformer.feature_names_in_)
        transformer.X_schema_ = dict(X.select(~cs.by_name("time")).schema)
        return X

    # Transform/Inverse context (reset=False)
    if inverse:
        # Use X_t if provided, otherwise treat X as X_t (transformed data)
        if X_t is None:
            if X is None:
                raise ValueError("Either `X_t` or `X` must be provided for inverse transform.")
            X_t = X

        # Validate time columns
        check_time_column(X_t)
        if X_p is not None:
            check_time_column(X_p)

        if stateful and X_p is None:
            raise ValueError(
                "X_p cannot be None for stateful inverse transform. Provide the necessary previous untransformed data."
            )

        if observation_horizon is not None and observation_horizon > 0 and X_p is None:
            raise ValueError(
                "X_p cannot be None to invert a transform that has observation_horizon > 0. "
                "Provide the necessary previous untransformed data."
            )

        X_t_interval = None
        if len(X_t) >= 2:
            X_t_interval = check_interval_consistency(X_t)

        if X_p is not None and len(X_p) > 0 and observation_horizon is not None:
            if len(X_p) < observation_horizon:
                raise ValueError(
                    f"X_p must have at least {observation_horizon} rows (observation_horizon), "
                    f"but has only {len(X_p)} rows."
                )

            if len(X_p) > 1:
                X_p_interval = check_interval_consistency(X_p)
                if X_t_interval is not None and X_p_interval != X_t_interval:
                    raise ValueError(
                        f"Time intervals do not match: X_p has interval {X_p_interval}, "
                        f"but X_t has interval {X_t_interval}."
                    )

        return X_t, X_p

    # transform context
    if X is None:
        raise ValueError("`X` cannot be None for transform (when inverse=False).")
    check_time_column(X)
    X = check_schema(X, transformer.X_schema_)

    if check_params.get("check_intervals", True) and len(X) >= 2:
        check_interval_consistency(X)

    if (
        check_params.get("check_continuity", True)
        and hasattr(transformer, "_X_observed")
        and len(transformer._X_observed) > 0
    ):
        interval = None
        if len(X) >= 2:
            interval = check_interval_consistency(X)
        check_continuity(
            transformer._X_observed,
            X,
            expected_interval=interval,
            check_intervals=(interval is not None),
        )

    return X
