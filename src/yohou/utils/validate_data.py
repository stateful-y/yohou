"""Data validation functions for different estimator types."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs

from yohou.utils.panel import inspect_locality
from yohou.utils.validation import (
    check_continuity,
    check_inputs,
    check_interval_consistency,
    check_panel_group_names,
    check_panel_groups_match,
    check_panel_internal_consistency,
    check_schema,
    check_scorer_column_selection,
    check_time_column,
)

if TYPE_CHECKING:
    from yohou.base import BaseForecaster, BaseTransformer
    from yohou.metrics.base import BaseScorer
    from yohou.model_selection import BaseSplitter


def validate_time_weight(
    time_weight: Callable | pl.DataFrame | None,
    y: pl.DataFrame,
    panel_group_names: list[str] | None = None,
) -> None:
    """Validate time_weight parameter for forecasters and scorers.

    Parameters
    ----------
    time_weight : callable, pl.DataFrame, or None
        Time weighting specification to validate.
    y : pl.DataFrame
        Target time series with "time" column.
    panel_group_names : list of str or None
        Panel group names if panel data.

    Raises
    ------
    ValueError
        If time_weight validation fails.

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
    if panel_group_names is None:
        # Global data: must have "weight" column
        if "weight" not in time_weight.columns:
            raise ValueError("time_weight DataFrame for global data must have 'weight' column")
        weight_cols_to_check = ["weight"]
    else:
        # Panel data: check for group-specific or global weight columns
        expected_group_cols = {f"{group}_weight" for group in panel_group_names}
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


def validate_scorer_data(
    scorer: BaseScorer,
    y_true: pl.DataFrame | None = None,
    y_pred: pl.DataFrame | None = None,
    *,
    scores: pl.DataFrame | None = None,
    reset: bool = False,
    inverse: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame, list | None]:
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
    tuple[pl.DataFrame, pl.DataFrame, list | None]
        Validated and prepared DataFrames and time values:
        - Normal context: (y_true, y_pred, time_values)
        - Inverse context: (y_pred, scores, time_values)
        - Fit context (reset=True): (y_train, None, None)

    Notes
    -----
    - When drop_time_columns=False, time column is ALWAYS first in output
    - Performs basic validation: None checks, time column existence, panel consistency
    - Time alignment preserves common time points only (inner join)
    - Time values are extracted before validation for point/interval scorers

    """
    if inverse:
        # For inverse_score: validate y_pred (point predictions) and scores (conformity scores)
        if y_pred is None:
            raise ValueError("`y_pred` is required for inverse_score context.")
        if scores is None:
            raise ValueError("`scores` is required for inverse_score context.")

        # Validate time columns (required)
        check_time_column(y_pred)
        check_time_column(scores)

        # Check column schema compatibility (exclude time/observed_time columns)
        exclude_cols_pred = ["time"]
        exclude_cols_scores = ["time"]

        if "observed_time" in y_pred.columns:
            exclude_cols_pred.append("observed_time")

        y_pred_cols = set(y_pred.select(~cs.by_name(*exclude_cols_pred)).columns)
        score_cols = set(scores.select(~cs.by_name(*exclude_cols_scores)).columns)

        if y_pred_cols != score_cols:
            raise ValueError(
                f"Column mismatch between y_pred and conformity_scores. "
                f"y_pred has {sorted(y_pred_cols)}, conformity_scores has {sorted(score_cols)}."
            )

        # Drop observed_time if present in y_pred
        if "observed_time" in y_pred.columns:
            y_pred = y_pred.drop("observed_time")

        # Extract time values before dropping (always present after validation)
        time_values = y_pred["time"].to_list()

        # Drop time columns for consistency with normal path
        y_pred = y_pred.drop("time")
        scores = scores.drop("time")

        return y_pred, scores, time_values

    if reset:
        # At fit time, y_true is y_train (always required), y_pred is always None
        if y_true is None:
            raise ValueError("`y_train` is required for scorer.fit(). Cannot be None.")

        check_time_column(y_true)

        # Validate seasonality for scorers with seasonality parameter
        if hasattr(scorer, "seasonality"):
            # y_true still has time column, so subtract 1 for data rows
            if len(y_true) <= scorer.seasonality:
                scorer_name = scorer.__class__.__name__
                raise ValueError(
                    f"Training data length ({len(y_true) - 1}) must be greater than "
                    f"seasonality ({scorer.seasonality}). Cannot compute seasonal naive forecast errors."
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
    check_time_column(y_pred)

    # Panel consistency check
    _, y_groups = inspect_locality(y_true)
    _, X_groups = inspect_locality(y_pred)
    if set(y_groups.keys()) != set(X_groups.keys()):
        raise ValueError(
            f"Panel groups mismatch. `y_true` has {sorted(y_groups.keys())}. `y_pred` has {sorted(X_groups.keys())}."
        )

    tags = scorer.__sklearn_tags__()
    pred_type = tags.scorer_tags.prediction_type

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
                if y_pred.schema[rc] != y_true.schema[col]:
                    raise ValueError(
                        f"Column '{rc}' type mismatch. `y_true` '{col}': {y_true.schema[col]}, "
                        f"`y_pred`: {y_pred.schema[rc]}"
                    )

    # Align by time (inner join on time column)
    time_truth = y_true.select("time")
    time_pred = y_pred.select("time")
    common_times = time_truth.join(time_pred, on="time", how="inner")

    y_true = y_true.join(common_times.select("time"), on="time", how="inner")
    y_pred = y_pred.join(common_times.select("time"), on="time", how="inner")

    # Subselect columns based on scorer configuration
    coverage_rates = getattr(scorer, "coverage_rates", None)
    interval_pattern = re.compile(r"^(.+)_(lower|upper)_([\d.]+)$")

    y_true, y_pred = check_scorer_column_selection(
        scorer=scorer,
        y_true=y_true,
        y_pred=y_pred,
        pred_type=pred_type,
        coverage_rates=coverage_rates,
        interval_pattern=interval_pattern,
    )

    # Extract time values before dropping (all scorers get time-less DataFrames)
    time_values = y_true["time"].to_list() if "time" in y_true.columns else None

    # Drop time columns for all scorers (conformity scorers can reconstruct from time_values)
    y_true = y_true.drop("time")

    if "observed_time" in y_pred.columns:
        y_pred = y_pred.drop("observed_time")
    if "time" in y_pred.columns:
        y_pred = y_pred.drop("time")

    return y_true, y_pred, time_values


def validate_splitter_data(
    splitter: BaseSplitter, y: pl.DataFrame | None, X: pl.DataFrame | None
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Validate data for splitters."""
    if y is not None:
        check_time_column(y)
        check_panel_internal_consistency(y, "y")

    if X is not None:
        check_time_column(X)
        check_panel_internal_consistency(X, "X")

        if y is not None:
            check_panel_groups_match(y, X)

    interval = check_inputs(y, X)
    splitter.interval_ = interval

    return y, X


def validate_forecaster_data(
    forecaster: BaseForecaster,
    y: pl.DataFrame | None = None,
    X: pl.DataFrame | None = None,
    *,
    reset: bool = True,
    panel_group_names: list[str] | None = None,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None, list[str] | None]:
    """Validate data for forecasters."""
    if reset:
        # Fit context: validate and set interval
        interval = check_inputs(y, X)
        forecaster.interval_ = interval
        return y, X, None

    # Predict/Update context (reset=False)

    # Validate time columns
    if y is not None:
        check_time_column(y)
    if X is not None:
        check_time_column(X)

    # Validate and normalize panel_group_names parameter
    panel_group_names = check_panel_group_names(
        fitted_panel_groups=forecaster.panel_group_names_,
        requested_panel_groups=panel_group_names,
    )

    # Validate schema and enforce column order
    if y is not None:
        y = check_schema(
            y,
            forecaster.local_y_schema_,
            panel_group_names=panel_group_names,
        )

    if X is not None:
        # Handle panel data X (local + global schemas)
        if forecaster.panel_group_names_ is not None:
            # Validate local X columns (with panel prefixes)
            if hasattr(forecaster, "local_X_schema_") and forecaster.local_X_schema_:
                X_local = check_schema(
                    X,
                    forecaster.local_X_schema_,
                    panel_group_names=forecaster.panel_group_names_,
                )

            # Validate global X columns (no prefixes)
            X_global = None
            if hasattr(forecaster, "global_X_schema_") and forecaster.global_X_schema_:
                X_global = check_schema(X, forecaster.global_X_schema_)

            # Reconstruct X with both local and global columns
            if (
                hasattr(forecaster, "local_X_schema_")
                and forecaster.local_X_schema_
                and hasattr(forecaster, "global_X_schema_")
                and forecaster.global_X_schema_
            ):
                X = pl.concat(
                    [X_local, X_global.select(~cs.by_name("time"))],
                    how="horizontal",
                )
            elif hasattr(forecaster, "local_X_schema_") and forecaster.local_X_schema_:
                X = X_local
            elif hasattr(forecaster, "global_X_schema_") and forecaster.global_X_schema_:
                X = X_global
        # Non-panel data: simple schema check (if schema exists)
        elif X is not None and hasattr(forecaster, "local_X_schema_") and forecaster.local_X_schema_:
            X = check_schema(X, forecaster.local_X_schema_)

    return y, X, panel_group_names


def validate_transformer_data(
    transformer: BaseTransformer,
    X: pl.DataFrame | None = None,
    *,
    reset: bool = True,
    inverse: bool = False,
    X_t: pl.DataFrame | None = None,
    X_p: pl.DataFrame | None = None,
    observation_horizon: int | None = None,
    **check_params,
) -> pl.DataFrame:
    """Validate data for transformers."""
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

        return X_t

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
