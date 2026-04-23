"""Base classes for forecasting metrics and scoring functions."""

from __future__ import annotations

import abc
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from yohou.utils import Tags, inspect_panel, validate_scorer_data
from yohou.utils._compat import StrOptions, _fit_context
from yohou.utils.validation import check_interval_consistency
from yohou.utils.weighting import (
    combine_weight_vectors,
    normalize_weights,
    resolve_weight_to_array,
    validate_callable_signature,
)

if TYPE_CHECKING:
    from datetime import datetime

    from yohou.metrics._context import ScoringContext

__all__ = ["BaseClassProbaScorer", "BaseIntervalScorer", "BasePointScorer", "BaseScorer"]


class BaseScorer(BaseEstimator, metaclass=abc.ABCMeta):
    """Base class for all forecasting metrics.

    Defines the interface for scoring forecast quality. All scorers must implement
    the `score` method and can optionally override `fit` for metrics
    that require training data statistics.

    Parameters
    ----------
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict). If None,
        all panel groups are included with equal weight.
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict). If None,
        all components are included with equal weight.

    Notes
    -----
    The ``aggregation_method`` parameter (on subclasses) controls which
    dimensions are collapsed when computing scores. Orthogonal modes:
    ``"stepwise"``, ``"vintagewise"``, ``"componentwise"``,
    ``"groupwise"``, ``"coveragewise"`` (interval only), or ``"all"``.

    See Also
    --------
    `BasePointScorer` : Base class for point-prediction metrics.
    `BaseIntervalScorer` : Base class for interval-prediction metrics.
    `BaseConformityScorer` : Base class for conformity scorers.

    """

    _lower_is_better: bool = True

    _parameter_constraints: dict = {
        "groups": [list, dict, None],
        "components": [list, dict, None],
    }

    def __init__(
        self,
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ):
        self.groups = groups
        self.components = components

    @staticmethod
    def _filter_keys(param: list | dict | None) -> list | None:
        """Extract filter list from a polymorphic param."""
        if isinstance(param, dict):
            return list(param.keys())
        return param

    @staticmethod
    def _weight_dict(param: list | dict | None) -> dict | None:
        """Extract weight dict from a polymorphic param."""
        if isinstance(param, dict):
            return param
        return None

    @property
    def lower_is_better(self) -> bool:
        """Whether lower scores indicate better performance."""
        return self._lower_is_better

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with scorer-specific attributes.

        """
        tags = Tags(estimator_type="scorer", requires_fit=False)

        # Subclasses set prediction_type in their __sklearn_tags__() method
        # Most scorers don't require calibration (fit is optional)
        assert tags.scorer_tags is not None
        tags.scorer_tags.requires_calibration = False
        tags.scorer_tags.lower_is_better = self._lower_is_better

        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y_train: pl.DataFrame, *, forecaster=None, **params) -> BaseScorer:
        """Fit the scorer on training data.

        Validates ``groups`` and ``component_names`` against
        training data.  Stores training data statistics for scaled metrics
        (e.g., MASE).  Subclasses should override to add type-specific
        parameter validation.

        Parameters
        ----------
        y_train : pl.DataFrame
            Training target time series with a ``"time"`` column and one or
            more numeric value columns.
        forecaster : BaseForecaster or None, default=None
            If provided, metadata is extracted directly from the fitted
            forecaster (``interval_``, ``groups_``, ``forecaster_horizon_``)
            instead of being re-inferred from ``y_train``.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted scorer instance.

        Raises
        ------
        ValueError
            If ``groups`` or ``component_names`` contain names not
            present in ``y_train``.

        """
        # Validate base parameters (groups, component_names)
        self._validate_parameters(y_train=y_train)

        # Validate input structure without aligning (single dataframe)
        validate_scorer_data(self, y_true=y_train, y_pred=None, reset=True)

        # Infer groups_ from y_train panel structure
        _, panel_groups = inspect_panel(y_train)
        self.groups_ = list(panel_groups.keys()) if panel_groups else None

        if forecaster is not None:
            # Extract metadata from fitted forecaster
            check_is_fitted(forecaster)
            self.interval_ = forecaster.interval_
            self.forecaster_horizon_ = forecaster.fit_forecasting_horizon_
        # Infer interval from training data time column
        elif "time" in y_train.columns and len(y_train) >= 2:
            self.interval_ = check_interval_consistency(y_train)
        else:
            self.interval_ = None

        # Mark as fitted
        self._is_fitted = True

        return self

    def _collapse_groups(self, df: pl.DataFrame) -> pl.DataFrame:
        """Collapse panel groups via weighted average.

        For each component (suffix after ``__``), computes a weighted average
        across all panel groups containing that component. Non-panel data
        is returned unchanged.

        """
        # Identify metadata columns (coverage_rate, context dims) to preserve
        meta_names = {"coverage_rate", "forecasting_step", "vintage_time", "time"}
        meta_cols = [c for c in df.columns if c in meta_names]
        value_df = df.select([c for c in df.columns if c not in meta_names])

        _, panel_groups = inspect_panel(value_df)
        if len(panel_groups) == 0:
            return df

        # component -> [(group_name, column_name)]
        components: dict[str, list[tuple[str, str]]] = {}
        for group_name, group_cols in panel_groups.items():
            for col in group_cols:
                component = col.split("__", 1)[1]
                if component not in components:
                    components[component] = []
                components[component].append((group_name, col))

        weights: dict[str, float] = {}
        for group_name in panel_groups:
            gw = self._weight_dict(self.groups)
            weights[group_name] = gw.get(group_name, 1.0) if gw else 1.0

        exprs: list[pl.Expr] = [pl.col(c) for c in meta_cols]
        for component, group_cols in components.items():
            total_weight = sum(weights[gn] for gn, _ in group_cols)
            if total_weight == 0:
                raise ValueError("Total panel group weight is zero")
            weighted_terms = [pl.col(col_name) * (weights[gn] / total_weight) for gn, col_name in group_cols]
            exprs.append(pl.sum_horizontal(weighted_terms).alias(component))

        return df.select(exprs)

    def _collapse_coverage_rates(self, df: pl.DataFrame) -> pl.DataFrame:
        """Collapse coverage_rate dimension via weighted average.

        Rows within each coverage rate block are assumed to be in the same
        order (one row per timestep). Collapsing averages across rates for
        each timestep, preserving row dimension.
        """
        if "coverage_rate" not in df.columns:
            return df

        if len(df) == 0:
            return df.drop("coverage_rate")

        meta_names = {"forecasting_step", "vintage_time", "time"}
        meta_cols = [c for c in df.columns if c in meta_names]
        val_cols = [c for c in df.columns if c not in meta_names and c != "coverage_rate"]

        n_rates = df["coverage_rate"].n_unique()
        n_rows_per_rate = len(df) // n_rates

        # Add row index to identify timesteps across rate blocks
        row_idx = list(range(n_rows_per_rate)) * n_rates
        df = df.with_columns(pl.Series("_row_idx", row_idx))
        group_cols = ["_row_idx"] + meta_cols

        cw = self._weight_dict(getattr(self, "coverage_rates", None))

        if cw is not None:
            df = df.with_columns(
                pl
                .col("coverage_rate")
                .replace_strict(
                    {r: cw.get(r, 1.0) for r in df["coverage_rate"].unique().to_list()},
                    default=1.0,
                )
                .alias("_cw")
            )
            result = df.group_by(group_cols, maintain_order=True).agg([
                (pl.col(c) * pl.col("_cw")).sum() / pl.col("_cw").sum() for c in val_cols
            ])
        else:
            result = df.group_by(group_cols, maintain_order=True).agg([pl.col(c).mean() for c in val_cols])

        return result.sort("_row_idx").drop("_row_idx")

    def _collapse_rows(
        self,
        df: pl.DataFrame,
        context: ScoringContext | None,
        dims: set[str],
    ) -> pl.DataFrame:
        """Collapse row dimensions (stepwise and/or vintagewise).

        When both stepwise and vintagewise are in *dims*, all rows are
        averaged (within each coverage_rate group if present). For partial
        collapse, rows are grouped by the non-collapsed dimension.
        """
        collapse_steps = "stepwise" in dims
        collapse_vintages = "vintagewise" in dims

        if not collapse_steps and not collapse_vintages:
            return df

        meta_names = {"coverage_rate"}
        meta_cols = [c for c in df.columns if c in meta_names]
        val_cols = [c for c in df.columns if c not in meta_names]

        if collapse_steps and collapse_vintages:
            if meta_cols:
                return df.group_by(meta_cols, maintain_order=True).agg([pl.col(c).mean() for c in val_cols])
            return df.select(pl.all().mean())

        # Partial collapse: keep one dimension, collapse the other
        keep_dim = "forecasting_step" if collapse_vintages else "vintage_time"
        dim_values = getattr(context, keep_dim, None) if context is not None else None

        if dim_values is None:
            if meta_cols:
                return df.group_by(meta_cols, maintain_order=True).agg([pl.col(c).mean() for c in val_cols])
            return df.select(pl.all().mean())

        # Tile context values for interval data (coverage_rate repeats rows)
        dim_list = dim_values.to_list()
        if "coverage_rate" in df.columns:
            n_rates = df["coverage_rate"].n_unique()
            dim_list = dim_list * n_rates

        group_cols = [keep_dim] + meta_cols

        return (
            df
            .with_columns(pl.Series(keep_dim, dim_list))
            .group_by(group_cols, maintain_order=True)
            .agg([pl.col(c).mean() for c in val_cols])
            .sort(keep_dim)
        )

    def _collapse_components(self, df: pl.DataFrame) -> pl.DataFrame:
        """Collapse component columns into a single score via weighted average.

        Metadata columns (coverage_rate, forecasting_step, etc.) are preserved.
        For panel data (``__`` prefixed columns), each group is collapsed
        separately into ``{group}__score``.
        """
        meta_names = {"coverage_rate", "forecasting_step", "vintage_time", "time"}
        meta_cols = [c for c in df.columns if c in meta_names]
        value_df = df.select([c for c in df.columns if c not in meta_names])

        if len(value_df.columns) == 0:
            return df

        _, panel_groups = inspect_panel(value_df)
        cw = self._weight_dict(self.components)

        keep = [pl.col(c) for c in meta_cols]

        if len(panel_groups) > 0:
            new_cols = []
            for group_name, group_cols in panel_groups.items():
                if cw is not None:
                    unprefixed = [c.split("__", 1)[1] for c in group_cols]
                    weights = [cw.get(n, 1.0) for n in unprefixed]
                    total = sum(weights)
                    weighted = pl.sum_horizontal([
                        pl.col(c) * (w / total) for c, w in zip(group_cols, weights, strict=True)
                    ])
                else:
                    weighted = pl.sum_horizontal([pl.col(c) for c in group_cols]) / len(group_cols)
                new_cols.append(weighted.alias(f"{group_name}__score"))
            return df.select(keep + new_cols)

        val_cols = value_df.columns
        if cw is not None:
            weights = [cw.get(c, 1.0) for c in val_cols]
            total = sum(weights)
            score = pl.sum_horizontal([pl.col(c) * (w / total) for c, w in zip(val_cols, weights, strict=True)])
        else:
            score = pl.sum_horizontal([pl.col(c) for c in val_cols]) / len(val_cols)

        return df.select(keep + [score.alias("score")])

    def _finalize(
        self,
        result: pl.DataFrame,
        context: ScoringContext | None,
        dims: set[str],
    ) -> float | pl.DataFrame:
        """Attach remaining row labels and convert 1x1 results to scalar.

        After all collapse steps, determines whether to return a scalar
        (when all dimensions are collapsed) or a labelled DataFrame.
        """
        meta_names = {"coverage_rate", "forecasting_step", "vintage_time", "time"}
        existing_labels = [c for c in result.columns if c in meta_names]
        value_cols = [c for c in result.columns if c not in meta_names]

        collapse_steps = "stepwise" in dims
        collapse_vintages = "vintagewise" in dims
        rows_collapsed = collapse_steps and collapse_vintages

        # Add time labels if rows are NOT fully collapsed and no row label exists yet
        has_row_label = any(c in existing_labels for c in ("forecasting_step", "vintage_time", "time"))
        if not rows_collapsed and not has_row_label and context is not None and context.time_values is not None:
            time_values = context.time_values
            if "coverage_rate" in result.columns:
                # Tile time values for each coverage rate
                n_rates = result["coverage_rate"].n_unique()
                if len(time_values) * n_rates == len(result):
                    tiled_times = time_values * n_rates
                    result = result.with_columns(pl.Series("time", tiled_times).cast(pl.Datetime))
                    # Reorder: time first, then all others
                    cols = ["time"] + [c for c in result.columns if c != "time"]
                    result = result.select(cols)
            elif len(time_values) == len(result):
                result = pl.concat(
                    [
                        pl.DataFrame({"time": time_values}).cast({"time": pl.Datetime}),
                        result,
                    ],
                    how="horizontal",
                )

        # Attach vintage_time if available and rows are not collapsed
        if (
            not rows_collapsed
            and "vintage_time" not in result.columns
            and context is not None
            and context.vintage_time is not None
        ):
            ot_values = context.vintage_time.to_list()
            if "coverage_rate" in result.columns:
                n_rates = result["coverage_rate"].n_unique()
                # Each vintage_time repeats n_rates times (once per coverage_rate)
                if len(ot_values) * n_rates == len(result):
                    ot_values = ot_values * n_rates
            if len(ot_values) == len(result):
                # Insert after time if present, otherwise at position 0
                insert_pos = result.columns.index("time") + 1 if "time" in result.columns else 0
                ot_series = pl.Series("vintage_time", ot_values).cast(pl.Datetime)
                cols = list(result.columns)
                cols.insert(insert_pos, "vintage_time")
                result = result.with_columns(ot_series).select(cols)

        # Re-check for labels after potential time/vintage_time label addition
        existing_labels = [c for c in result.columns if c in meta_names]
        value_cols = [c for c in result.columns if c not in meta_names]

        # Scalar: only when ALL spatial dimensions are collapsed
        # (i.e., stepwise+vintagewise+componentwise+groupwise all in dims, and no remaining labels)
        all_spatial_collapsed = "stepwise" in dims and "vintagewise" in dims and "componentwise" in dims
        # A single-value coverage_rate is trivial metadata when all other dims are collapsed
        if all_spatial_collapsed and existing_labels == ["coverage_rate"] and len(result) == 1:
            existing_labels = []
            result = result.drop("coverage_rate")
            value_cols = [c for c in result.columns if c not in meta_names]
        if len(result) == 1 and len(value_cols) <= 1 and not existing_labels and all_spatial_collapsed:
            if len(value_cols) == 0:
                return float("nan")
            return float(result[value_cols[0]][0])

        return result

    def _pre_filter_zero_weights(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
        context: ScoringContext,
        time_weight: Callable | pl.DataFrame | dict | None = None,
        step_weight: Callable | pl.DataFrame | dict | None = None,
        vintage_weight: Callable | pl.DataFrame | dict | None = None,
    ) -> tuple[
        pl.DataFrame,
        pl.DataFrame,
        ScoringContext,
        np.ndarray | dict[str, np.ndarray] | None,
        np.ndarray | dict[str, np.ndarray] | None,
        np.ndarray | dict[str, np.ndarray] | None,
    ]:
        """Resolve weights and pre-filter rows with zero weight.

        For group-uniform sources (dict, 1-param callable, DataFrame
        without group columns), zeros are combined into a mask and matching
        rows are removed from ``y_truth``, ``y_pred``, and ``context``.
        For panel-aware sources, weights are resolved per-group into a
        ``dict[str, np.ndarray]`` but NOT used for row pre-filtering.

        Returns the filtered data plus pre-resolved weights to avoid
        resolving twice.

        """
        _, panel_groups = inspect_panel(y_truth)
        has_panel = len(panel_groups) > 0

        zero_mask = np.zeros(len(y_truth), dtype=bool)

        def _resolve_one(
            w: Callable | pl.DataFrame | dict | None,
            key_series: pl.Series,
            join_column: str,
            name: str,
        ) -> np.ndarray | dict[str, np.ndarray] | None:
            """Resolve a single weight argument into aligned arrays."""
            if w is None:  # pragma: no cover
                return None

            # Detect panel-awareness
            is_panel_aware = False
            if has_panel and callable(w) and not isinstance(w, dict):
                n_params = validate_callable_signature(w)
                is_panel_aware = n_params == 2
            elif has_panel and isinstance(w, pl.DataFrame):
                # Check if it has group-specific weight columns
                for g in panel_groups:
                    if f"{g}_weight" in w.columns:
                        is_panel_aware = True
                        break

            if is_panel_aware and has_panel:
                # Resolve per-group, no row pre-filtering
                result_dict: dict[str, np.ndarray] = {}
                for group_name in panel_groups:
                    arr = resolve_weight_to_array(w, key_series, join_column, group_name)
                    result_dict[group_name] = arr
                return result_dict

            # Group-uniform: resolve once, track zeros for filtering
            group_name = next(iter(panel_groups)) if has_panel else None
            arr = resolve_weight_to_array(w, key_series, join_column, group_name)
            return arr

        # Build key series for each weight type
        time_series = pl.Series("time", context.time_values) if context.time_values is not None else None
        step_series = context.forecasting_step
        vintage_series = context.vintage_time

        # Resolve time_weight
        tw_resolved = None
        if time_weight is not None:
            if time_series is None:  # pragma: no cover
                raise ValueError("time_values unavailable in context but time_weight was provided")
            tw_resolved = _resolve_one(time_weight, time_series, "time", "time_weight")
            if isinstance(tw_resolved, np.ndarray):
                zero_mask |= tw_resolved == 0.0

        # Resolve step_weight (silently ignored when forecasting_step unavailable)
        sw_resolved = None
        if step_weight is not None and step_series is not None:
            sw_resolved = _resolve_one(step_weight, step_series, "forecasting_step", "step_weight")
            if isinstance(sw_resolved, np.ndarray):
                zero_mask |= sw_resolved == 0.0

        # Resolve vintage_weight (silently ignored when vintage_time unavailable)
        vw_resolved = None
        if vintage_weight is not None and vintage_series is not None:
            vw_resolved = _resolve_one(vintage_weight, vintage_series, "vintage_time", "vintage_weight")
            if isinstance(vw_resolved, np.ndarray):
                zero_mask |= vw_resolved == 0.0

        # Apply zero-mask filter
        if np.any(zero_mask):
            keep = ~zero_mask
            if not np.any(keep):
                raise ValueError(
                    "All rows have zero weight after pre-filtering. "
                    "Check that weight dicts/callables assign non-zero weights to at least some data."
                )

            from yohou.metrics._context import ScoringContext as _ScoringContext  # noqa: PLC0415

            y_truth = y_truth.filter(keep)
            y_pred = y_pred.filter(keep)
            context = _ScoringContext(
                time_values=[t for t, m in zip(context.time_values, keep.tolist(), strict=True) if m],
                vintage_time=(context.vintage_time.filter(keep) if context.vintage_time is not None else None),
                forecasting_step=(
                    context.forecasting_step.filter(keep) if context.forecasting_step is not None else None
                ),
            )

            # Slice resolved weight arrays to match
            if isinstance(tw_resolved, np.ndarray):
                tw_resolved = tw_resolved[keep]  # ty: ignore[invalid-argument-type]
            elif isinstance(tw_resolved, dict):
                tw_resolved = {g: a[keep] for g, a in tw_resolved.items()}

            if isinstance(sw_resolved, np.ndarray):
                sw_resolved = sw_resolved[keep]  # ty: ignore[invalid-argument-type]
            elif isinstance(sw_resolved, dict):
                sw_resolved = {g: a[keep] for g, a in sw_resolved.items()}

            if isinstance(vw_resolved, np.ndarray):
                vw_resolved = vw_resolved[keep]  # ty: ignore[invalid-argument-type]
            elif isinstance(vw_resolved, dict):
                vw_resolved = {g: a[keep] for g, a in vw_resolved.items()}

        return y_truth, y_pred, context, tw_resolved, sw_resolved, vw_resolved

    def _apply_weights(
        self,
        scores: pl.DataFrame,
        time_weight_resolved: np.ndarray | dict[str, np.ndarray] | None,
        step_weight_resolved: np.ndarray | dict[str, np.ndarray] | None,
        vintage_weight_resolved: np.ndarray | dict[str, np.ndarray] | None,
        n_rates: int = 1,
    ) -> pl.DataFrame:
        """Apply pre-resolved weights to score DataFrame.

        Two-stage normalization: (1) normalize and apply time_weight,
        (2) combine step_weight + vintage_weight, normalize, and apply.

        """
        _, panel_groups = inspect_panel(scores)

        def _apply_array(df: pl.DataFrame, w: np.ndarray, cols: list[str]) -> pl.DataFrame:
            """Multiply columns by weight array."""
            return df.with_columns([(pl.col(c) * w).alias(c) for c in cols])

        def _apply_one_weight(
            df: pl.DataFrame,
            w_resolved: np.ndarray | dict[str, np.ndarray],
            value_cols: list[str],
        ) -> pl.DataFrame:
            """Apply a single normalized weight (array or dict) to scores."""
            if isinstance(w_resolved, dict):
                for group_name, group_arr in w_resolved.items():
                    normed = normalize_weights(group_arr)  # ty: ignore[invalid-argument-type]
                    tiled = np.tile(normed, n_rates) if n_rates > 1 else normed
                    group_cols = [c for c in panel_groups.get(group_name, []) if c in value_cols]  # ty: ignore[no-matching-overload]
                    if group_cols:
                        df = _apply_array(df, tiled, group_cols)
            else:
                normed = normalize_weights(w_resolved)
                tiled = np.tile(normed, n_rates) if n_rates > 1 else normed
                df = _apply_array(df, tiled, value_cols)
            return df

        value_cols = [c for c in scores.columns if c != "coverage_rate"]

        # Stage 1: time_weight (normalized independently)
        if time_weight_resolved is not None:
            scores = _apply_one_weight(scores, time_weight_resolved, value_cols)

        # Stage 2: step_weight + vintage_weight (combined, then normalized)
        if step_weight_resolved is not None or vintage_weight_resolved is not None:
            # For dict weights, combine per-group
            if isinstance(step_weight_resolved, dict) or isinstance(vintage_weight_resolved, dict):
                all_groups = set()
                if isinstance(step_weight_resolved, dict):
                    all_groups |= step_weight_resolved.keys()
                if isinstance(vintage_weight_resolved, dict):
                    all_groups |= vintage_weight_resolved.keys()

                for group_name in all_groups:
                    sw_g = (
                        step_weight_resolved.get(group_name)
                        if isinstance(step_weight_resolved, dict)
                        else step_weight_resolved
                    )
                    vw_g = (
                        vintage_weight_resolved.get(group_name)
                        if isinstance(vintage_weight_resolved, dict)
                        else vintage_weight_resolved
                    )
                    n = len(sw_g) if sw_g is not None else len(vw_g)  # ty: ignore[invalid-argument-type]
                    combined = combine_weight_vectors(sw_g, vw_g, n=n)  # ty: ignore[invalid-argument-type]
                    if combined is not None:
                        tiled = np.tile(combined, n_rates) if n_rates > 1 else combined
                        group_cols = [c for c in panel_groups.get(group_name, []) if c in value_cols]
                        if group_cols:
                            scores = _apply_array(scores, tiled, group_cols)
            else:
                # Both are plain arrays or None
                n = len(step_weight_resolved) if step_weight_resolved is not None else len(vintage_weight_resolved)  # ty: ignore[invalid-argument-type]
                combined = combine_weight_vectors(step_weight_resolved, vintage_weight_resolved, n=n)
                if combined is not None:
                    tiled = np.tile(combined, n_rates) if n_rates > 1 else combined
                    scores = _apply_array(scores, tiled, value_cols)

        return scores

    def _validate_parameters(
        self,
        y_train: pl.DataFrame | None = None,
        aggregation_method: list[str] | str | None = None,
        valid_aggregation_methods: set[str] | None = None,
    ) -> None:
        """Validate scorer parameters.

        Parameters
        ----------
        y_train : pl.DataFrame or None
            Training data to validate against. If None, only type validation is performed.
        aggregation_method : list of str or str or None
            Aggregation method to validate. If None, aggregation validation is skipped.
        valid_aggregation_methods : set of str or None
            Set of valid aggregation method strings. Required if aggregation_method is provided.

        Raises
        ------
        ValueError
            If validation fails.

        """
        # Validate aggregation_method if provided
        if aggregation_method is not None:
            if valid_aggregation_methods is None:
                raise ValueError("valid_aggregation_methods must be provided when validating aggregation_method")

            # Handle single string
            if isinstance(aggregation_method, str):
                # "all" is a special value that means aggregate across all dimensions
                if aggregation_method != "all" and aggregation_method not in valid_aggregation_methods:
                    raise ValueError(
                        f"Invalid aggregation_method '{aggregation_method}'. "
                        f"Valid options are: 'all' or {sorted(valid_aggregation_methods)}"
                    )
            # Handle list
            elif isinstance(aggregation_method, list):
                # Check all elements are strings
                if not all(isinstance(method, str) for method in aggregation_method):
                    raise ValueError(f"All elements in aggregation_method must be strings, got: {aggregation_method}")
                if len(aggregation_method) == 0:
                    raise ValueError(
                        f"aggregation_method list cannot be empty. "
                        f"Use 'all' or provide at least one method: {sorted(valid_aggregation_methods)}"
                    )
                for method in aggregation_method:
                    if method not in valid_aggregation_methods:
                        raise ValueError(
                            f"Invalid aggregation_method '{method}' in list. "
                            f"Valid list elements are: {sorted(valid_aggregation_methods)}"
                        )
            else:
                raise ValueError(
                    f"aggregation_method must be a string or list of strings, got {type(aggregation_method)}"
                )

        # Validate groups type (list or dict)
        group_filter = self._filter_keys(self.groups)
        if group_filter is not None:
            if not all(isinstance(name, str) for name in group_filter):
                raise ValueError("All group names must be strings")
            if len(group_filter) == 0:
                raise ValueError("groups cannot be empty")

        # Validate components type (list or dict)
        comp_filter = self._filter_keys(self.components)
        if comp_filter is not None:
            if not all(isinstance(name, str) for name in comp_filter):
                raise ValueError("All component names must be strings")
            if len(comp_filter) == 0:
                raise ValueError("components cannot be empty")

        # If y_train is provided, validate against actual data
        if y_train is not None:
            _, panel_groups = inspect_panel(y_train)
            available_groups = set(panel_groups.keys())

            # Validate groups exist in data
            if group_filter is not None:
                if len(available_groups) == 0:
                    # No panel data, but user specified groups
                    raise ValueError(
                        f"groups specified but data contains no panel groups. "
                        f"Data has only global columns: {sorted(set(y_train.columns) - {'time'})}"
                    )
                requested_groups = set(group_filter)
                missing_groups = requested_groups - available_groups
                if missing_groups:
                    raise ValueError(
                        f"Requested groups {sorted(missing_groups)} not found in data. "
                        f"Available groups: {sorted(available_groups)}"
                    )

            # Validate components exist in data
            if comp_filter is not None:
                if len(panel_groups) > 0:
                    # Panel data: check unprefixed column names
                    available_components = set()
                    for group_cols in panel_groups.values():
                        for col in group_cols:
                            # Extract unprefixed column name
                            available_components.add(col.split("__", 1)[1])
                else:
                    # Global data: check column names directly
                    available_components = set(y_train.columns) - {"time"}

                requested_components = set(comp_filter)
                missing_components = requested_components - available_components
                if missing_components:
                    raise ValueError(
                        f"Requested components {sorted(missing_components)} "
                        f"not found in data. Available components: {sorted(available_components)}"
                    )

    @staticmethod
    def _normalize_agg_methods(
        aggregation_method: list[str] | str,
        include_coveragewise: bool = False,
    ) -> set[str]:
        """Normalize aggregation_method to a set of orthogonal mode names.

        Expands ``"all"`` to the full set of modes.

        Parameters
        ----------
        aggregation_method : list of str or str
            Raw aggregation_method value from the scorer.
        include_coveragewise : bool, default=False
            Whether to include ``"coveragewise"`` in the ``"all"`` expansion
            (interval scorers only).

        Returns
        -------
        set of str
            Normalized set of aggregation mode names.

        """
        if aggregation_method == "all":
            modes = {"stepwise", "vintagewise", "componentwise", "groupwise"}
            if include_coveragewise:
                modes.add("coveragewise")
            return modes

        modes = {aggregation_method} if isinstance(aggregation_method, str) else set(aggregation_method)

        return modes

    def _aggregate_scores(
        self, raw_scores: pl.DataFrame, context: ScoringContext | None = None
    ) -> float | pl.DataFrame:
        """Apply sequential aggregation pipeline to raw scores.

        Collapses dimensions one at a time in order: groups → coverage →
        rows (steps/vintages) → components → finalize.

        Parameters
        ----------
        raw_scores : pl.DataFrame
            DataFrame with per-timestep (rows) per-component (columns) scores.
            May contain a ``"coverage_rate"`` column for interval scorers.
        context : ScoringContext or None, default=None
            Scoring context with time values and metadata.

        Returns
        -------
        float or pl.DataFrame
            Aggregated scores based on aggregation_method.

        """
        has_coverage_rate = "coverage_rate" in raw_scores.columns
        dims = self._normalize_agg_methods(
            self.aggregation_method,  # ty: ignore[unresolved-attribute]
            include_coveragewise=has_coverage_rate,
        )

        result = raw_scores

        # 1. Collapse coverage rates (interval only)
        if "coveragewise" in dims:
            result = self._collapse_coverage_rates(result)

        # 2. Collapse row dimensions (steps and/or vintages)
        result = self._collapse_rows(result, context, dims)

        # 3. Collapse components
        if "componentwise" in dims:
            result = self._collapse_components(result)

        # 4. Collapse panel groups
        if "groupwise" in dims:
            result = self._collapse_groups(result)

        # 5. Finalize: attach labels, convert to scalar
        return self._finalize(result, context, dims)

    @abc.abstractmethod
    def score(
        self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, /, **params
    ) -> pl.DataFrame | float | dict[str | float, float | pl.DataFrame]:
        """Compute the metric score.

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth time series to score against.  Must have a
            ``"time"`` column and one or more numeric value columns.
        y_pred : pl.DataFrame
            Predicted time series to evaluate.  Must have ``"vintage_time"``
            and ``"time"`` columns and columns matching ``y_truth``.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame or float or dict
            Aggregated score(s).  A ``float`` when
            ``aggregation_method="all"``, a ``pl.DataFrame`` for partial
            aggregations, or a ``dict`` mapping coverage rates to scores
            for interval scorers.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the scorer has not been fitted yet (when calibration is
            required).
        ValueError
            If ``y_truth`` and ``y_pred`` have mismatched columns or
            incompatible shapes.

        """

    def _rename_metric_columns(self, result: pl.DataFrame) -> pl.DataFrame:
        """Rename aggregation output columns to use the metric name.

        Replaces ``"score"`` with ``_metric_name`` and ``"__score"``
        suffixes with ``"__<_metric_name>"``.

        Parameters
        ----------
        result : pl.DataFrame
            DataFrame from ``_aggregate_scores`` that may contain
            ``"score"`` or ``"*__score"`` columns.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns renamed to use the metric name.

        """
        metric_name: str = getattr(self, "_metric_name", "score")
        rename_map: dict[str, str] = {}
        if "score" in result.columns:
            rename_map["score"] = metric_name
        for col in result.columns:
            if col.endswith("__score"):
                rename_map[col] = col.replace("__score", f"__{metric_name}")
        if rename_map:
            result = result.rename(rename_map)
        return result

    def __call__(
        self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, **params
    ) -> pl.DataFrame | float | dict[str | float, float | pl.DataFrame]:
        """Compute score using callable interface.

        Enables using scorers as functions: scorer(y_truth, y_pred).

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth values.

        y_pred : pl.DataFrame
            Predicted values.

        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame or float or dict
            Metric score.

        """
        return self.score(y_truth, y_pred, **params)


class BasePointScorer(BaseScorer, metaclass=abc.ABCMeta):
    """Base class for point forecast metrics.

    Point forecasters produce single-value predictions. Metrics derived from this
    class evaluate prediction accuracy (e.g., MeanAbsoluteError, RootMeanSquaredError, MAPE).

    .. note:: The ``_response_method`` attribute indicates which forecaster
       method produces the predictions that this scorer expects.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. Options:
        - "stepwise": Aggregate across forecasting steps.
        - "vintagewise": Aggregate across vintages (observed times).
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["stepwise", "vintagewise", "componentwise", "groupwise"].
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict). If None,
        all panel groups are included with equal weight.
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict). If None,
        all components are included with equal weight.

    See Also
    --------
    `MeanAbsoluteError` : Concrete point scorer implementation.
    `MeanSquaredError` : Concrete point scorer implementation.
    `BasePointForecaster` : Produces point forecasts.

    """

    _response_method: str = "predict"

    _parameter_constraints: dict = {
        **BaseScorer._parameter_constraints,
        "aggregation_method": [
            list,
            StrOptions({"all", "stepwise", "vintagewise", "componentwise", "groupwise"}),
        ],
    }

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ):
        super().__init__(
            groups=groups,
            components=components,
        )
        self.aggregation_method = aggregation_method

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y_train: pl.DataFrame, *, forecaster=None, **params) -> BasePointScorer:
        """Fit the scorer on training data.

        Validates ``aggregation_method``, ``groups``, and
        ``component_names``.

        Parameters
        ----------
        y_train : pl.DataFrame
            Training target time series with a ``"time"`` column and one or
            more numeric value columns.
        forecaster : BaseForecaster or None, default=None
            If provided, metadata is extracted directly from the fitted
            forecaster instead of being re-inferred from ``y_train``.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted scorer instance.

        Raises
        ------
        ValueError
            If ``aggregation_method`` contains invalid values, or if
            ``groups`` / ``component_names`` are not found in
            ``y_train``.

        """
        # Validate point-specific parameters (aggregation_method)
        valid_methods = {"stepwise", "vintagewise", "componentwise", "groupwise"}
        self._validate_parameters(
            y_train=y_train,
            aggregation_method=self.aggregation_method,
            valid_aggregation_methods=valid_methods,
        )

        return super().fit(y_train, forecaster=forecaster, **params)

    @abc.abstractmethod
    def _compute_raw_errors(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Compute per-timestep per-component raw errors.

        Subclasses implement only this method.  Access fitted attributes
        (e.g. ``self.scales_``, ``self.naive_errors_``) via ``self``.

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth values (time column already removed).
        y_pred : pl.DataFrame
            Predicted values (time column already removed).

        Returns
        -------
        pl.DataFrame
            Raw error values, same shape as inputs.

        """

    def _post_aggregate(self, result: float | pl.DataFrame) -> float | pl.DataFrame:
        """Apply an optional post-aggregation transform (e.g. sqrt for RMSE).

        Override in subclasses that need a transform after aggregation.
        Default implementation is identity.

        Parameters
        ----------
        result : float or pl.DataFrame
            Aggregated scores.

        Returns
        -------
        float or pl.DataFrame
            Transformed scores.

        """
        return result

    def score(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
        /,
        time_weight: Callable | pl.DataFrame | dict[datetime | str, float] | None = None,
        step_weight: Callable | pl.DataFrame | dict[int | str, float] | None = None,
        vintage_weight: Callable | pl.DataFrame | dict[datetime | str, float] | None = None,
        **params,
    ) -> float | pl.DataFrame:
        """Compute the point metric score.

        Template method: validate -> pre-filter zeros -> compute raw errors
        -> apply weights -> aggregate -> post-aggregate transform -> rename.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True values with ``"time"`` column.
        y_pred : pl.DataFrame
            Predicted values with ``"time"`` column.
        time_weight : callable, pl.DataFrame, dict, or None, default=None
            Time-based evaluation weights. Accepts a callable
            ``f(time_series) -> pl.Series``, a panel-aware callable
            ``f(time_series, group_name) -> pl.Series``, a DataFrame
            with ``"time"`` and ``"weight"`` columns, or a
            ``{datetime_or_str: float}`` dict (``"*"`` key sets default).
        step_weight : callable, pl.DataFrame, dict, or None, default=None
            Per-step weights. Same formats as ``time_weight`` but keyed on
            ``"forecasting_step"``. Use ``{"*": 0.0, 1: 1.0}`` to score
            only step 1.
        vintage_weight : callable, pl.DataFrame, dict, or None, default=None
            Per-vintage weights. Same formats as ``time_weight`` but keyed
            on ``"vintage_time"``.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        float or pl.DataFrame
            Aggregated metric score.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, context = validate_scorer_data(
            self,
            y_truth,
            y_pred,
        )

        # 0. Resolve weights and pre-filter zero-weight rows
        y_truth, y_pred, context, tw, sw, vw = self._pre_filter_zero_weights(
            y_truth,
            y_pred,
            context,
            time_weight,
            step_weight,
            vintage_weight,
        )

        # 1. Compute raw per-timestep per-component errors
        scores = self._compute_raw_errors(y_truth, y_pred)

        # 2. Apply weights (two-stage: time first, then step+vintage)
        scores = self._apply_weights(scores, tw, sw, vw)

        # 3. Aggregate
        result = self._aggregate_scores(scores, context=context)

        # 4. Post-aggregation transform (e.g. sqrt for RMSE/RMSSE)
        result = self._post_aggregate(result)

        # 5. Rename columns
        if isinstance(result, pl.DataFrame):
            result = self._rename_metric_columns(result)

        return result

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with scorer-specific attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.scorer_tags is not None
        tags.scorer_tags.prediction_type = "point"
        return tags


class BaseIntervalScorer(BaseScorer, metaclass=abc.ABCMeta):
    """Base class for interval forecast metrics.

    Interval forecasters produce prediction intervals. Metrics derived from this
    class evaluate coverage and width trade-offs.

    .. note:: The ``_response_method`` attribute indicates which forecaster
       method produces the predictions that this scorer expects.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to collapse when aggregating scores. Orthogonal modes:

        - "stepwise": Collapse the forecasting-step dimension (average across steps).
        - "vintagewise": Collapse the vintage/observed-time dimension.
        - "componentwise": Collapse components, return per-timestep scores.
        - "groupwise": Collapse panel groups (panel data only).
        - "coveragewise": Collapse coverage rates (return average across rates).
        - "all": Collapse all dimensions (returns scalar). Same as
          ``["stepwise", "vintagewise", "componentwise", "groupwise", "coveragewise"]``.
    coverage_rates : list of float, dict of float to float, or None, default=None
        Coverage rate filter (list) or filter with weights (dict). If None,
        all coverage rates are included with equal weight.
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict). If None,
        all panel groups are included with equal weight.
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict). If None,
        all components are included with equal weight.

    See Also
    --------
    `IntervalScore` : Concrete interval scorer implementation.
    `CoverageScore` : Concrete interval scorer implementation.
    `BaseIntervalForecaster` : Produces interval forecasts.

    """

    _response_method: str = "predict_interval"

    _parameter_constraints: dict = {
        **BaseScorer._parameter_constraints,
        "aggregation_method": [
            list,
            StrOptions({"all", "stepwise", "vintagewise", "componentwise", "groupwise", "coveragewise"}),
        ],
        "coverage_rates": [list, dict, None],
    }

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        coverage_rates: list[float] | dict[float, float] | None = None,
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ):
        super().__init__(
            groups=groups,
            components=components,
        )
        self.aggregation_method = aggregation_method
        self.coverage_rates = coverage_rates

    def _validate_coverage_rates(self) -> None:
        """Validate coverage parameter.

        Raises
        ------
        ValueError
            If coverage validation fails.
        TypeError
            If coverage contains non-hashable types.

        """
        coverage_filter = self._filter_keys(self.coverage_rates)
        if coverage_filter is not None:
            if len(coverage_filter) == 0:
                raise ValueError("coverage_rates cannot be empty")

            # Check for hashable types (catch lists, dicts, etc.)
            for i, rate in enumerate(coverage_filter):
                try:
                    hash(rate)
                except TypeError:
                    raise TypeError(
                        f"coverage_rates[{i}] is not hashable (got {type(rate).__name__}). "
                        f"All elements must be numeric (int or float)."
                    ) from None

            # Check all elements are numeric
            if not all(isinstance(rate, int | float) for rate in coverage_filter):
                raise ValueError(
                    f"All elements in coverage_rates must be numeric (int or float), "
                    f"got types: {[type(r).__name__ for r in coverage_filter]}"
                )

            # Check range
            for rate in coverage_filter:
                if not 0 <= rate <= 1:
                    raise ValueError(f"All coverage rates must be between 0 and 1 (inclusive), got {rate}")

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y_train: pl.DataFrame, *, forecaster=None, **params) -> BaseIntervalScorer:
        """Fit the scorer on training data.

        Validates ``coverage_rates``, ``aggregation_method``,
        ``groups``, and ``component_names``.

        Parameters
        ----------
        y_train : pl.DataFrame
            Training target time series with a ``"time"`` column and one or
            more numeric value columns.
        forecaster : BaseForecaster or None, default=None
            If provided, metadata is extracted directly from the fitted
            forecaster instead of being re-inferred from ``y_train``.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted scorer instance.

        Raises
        ------
        ValueError
            If ``coverage_rates`` are invalid, ``aggregation_method`` contains
            invalid values, or if ``groups`` / ``component_names``
            are not found in ``y_train``.

        """
        # Validate coverage_rates
        self._validate_coverage_rates()

        # Validate interval-specific parameters (aggregation_method with coveragewise)
        valid_methods = {"stepwise", "vintagewise", "componentwise", "groupwise", "coveragewise"}
        self._validate_parameters(
            y_train=y_train,
            aggregation_method=self.aggregation_method,
            valid_aggregation_methods=valid_methods,
        )

        return super().fit(y_train, forecaster=forecaster, **params)

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with scorer-specific attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.scorer_tags is not None
        tags.scorer_tags.prediction_type = "interval"
        return tags

    @abc.abstractmethod
    def _compute_raw_scores(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
        coverage_rates: list[float],
        target_columns: list[str],
    ) -> pl.DataFrame:
        """Compute per-timestep per-component raw scores for each coverage rate.

        Subclasses implement only this method.

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth values (time column already removed).
        y_pred : pl.DataFrame
            Predicted intervals (time column already removed).
        coverage_rates : list of float
            Coverage rates extracted from prediction columns.
        target_columns : list of str
            Target column base names from ground truth.

        Returns
        -------
        pl.DataFrame
            Flat DataFrame with component columns plus a ``coverage_rate`` column.
            Rows = n_timesteps * n_rates.

        """

    def score(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
        /,
        time_weight: Callable | pl.DataFrame | dict[datetime | str, float] | None = None,
        step_weight: Callable | pl.DataFrame | dict[int | str, float] | None = None,
        vintage_weight: Callable | pl.DataFrame | dict[datetime | str, float] | None = None,
        **params,
    ) -> float | pl.DataFrame:
        """Compute the interval metric score.

        Template method: validate -> pre-filter zeros -> extract rates/columns
        -> compute raw scores -> apply weights -> aggregate -> rename.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True values with ``"time"`` column.
        y_pred : pl.DataFrame
            Predicted intervals with ``"time"`` column.
        time_weight : callable, pl.DataFrame, dict, or None, default=None
            Time-based evaluation weights. Accepts a callable
            ``f(time_series) -> pl.Series``, a panel-aware callable
            ``f(time_series, group_name) -> pl.Series``, a DataFrame
            with ``"time"`` and ``"weight"`` columns, or a
            ``{datetime_or_str: float}`` dict (``"*"`` key sets default).
        step_weight : callable, pl.DataFrame, dict, or None, default=None
            Per-step weights. Same formats as ``time_weight`` but keyed on
            ``"forecasting_step"``.
        vintage_weight : callable, pl.DataFrame, dict, or None, default=None
            Per-vintage weights. Same formats as ``time_weight`` but keyed
            on ``"vintage_time"``.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        float or pl.DataFrame
            Aggregated metric score.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, context = validate_scorer_data(
            self,
            y_truth,
            y_pred,
        )

        # 0. Resolve weights and pre-filter zero-weight rows
        y_truth, y_pred, context, tw, sw, vw = self._pre_filter_zero_weights(
            y_truth,
            y_pred,
            context,
            time_weight,
            step_weight,
            vintage_weight,
        )

        coverage_rates = self._extract_coverage_rates(y_pred)
        target_columns = self._extract_target_columns(y_truth)

        # 1. Compute raw per-timestep per-component per-rate scores (flat DataFrame)
        raw_scores = self._compute_raw_scores(y_truth, y_pred, coverage_rates, target_columns)

        # 2. Apply weights (two-stage, tiled across coverage rates)
        n_rates = raw_scores["coverage_rate"].n_unique() if "coverage_rate" in raw_scores.columns else 1
        raw_scores = self._apply_weights(raw_scores, tw, sw, vw, n_rates=n_rates)

        # 3. Aggregate
        result = self._aggregate_scores(raw_scores, context=context)

        # 4. Rename columns
        if isinstance(result, pl.DataFrame):
            result = self._rename_metric_columns(result)

        return result

    def _extract_coverage_rates(self, y_pred: pl.DataFrame) -> list[float]:
        """Extract unique coverage rates from interval prediction columns.

        Parses column names like "value_lower_0.95", "sales__store_1_upper_0.5"
        to extract all unique coverage rates present in the DataFrame.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Interval predictions with columns following pattern
            "{col}_lower_{rate}" or "{col}_upper_{rate}".

        Returns
        -------
        list of float
            Sorted list of unique coverage rates.

        """
        rates = set()
        # Match both global (value_lower_0.95) and panel (sales__store_1_lower_0.95) patterns
        pattern = re.compile(r"^(.+)_(lower|upper)_(\d+\.?\d*)$")

        for col in y_pred.columns:
            match = pattern.match(col)
            if match:
                rate_str = match.group(3)
                rates.add(float(rate_str))

        return sorted(rates)

    def _extract_target_columns(self, y_truth: pl.DataFrame) -> list[str]:
        """Extract target column base names from ground truth.

        Returns non-time column names from the ground truth DataFrame.
        For global data: ["value", "sales"]
        For panel data: ["sales__store_1", "sales__store_2"]

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth with target columns (time columns already removed).

        Returns
        -------
        list of str
            Target column names.

        """
        # After _validate_inputs, time columns are already removed
        return y_truth.columns


class BaseClassProbaScorer(BaseScorer, metaclass=abc.ABCMeta):
    """Base class for class-probability forecast metrics.

    Class-probability forecasters produce per-class probability distributions.
    Metrics derived from this class evaluate the quality of predicted
    probability distributions against true class labels.

    .. note:: The ``_response_method`` attribute indicates which forecaster
       method produces the predictions that this scorer expects.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. Options:
        - "stepwise": Aggregate across forecasting steps.
        - "vintagewise": Aggregate across vintages (observed times).
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["stepwise", "vintagewise", "componentwise", "groupwise"].
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter (list) or filter with weights (dict). If None,
        all panel groups are included with equal weight.
    components : list of str, dict of str to float, or None, default=None
        Component filter (list) or filter with weights (dict). If None,
        all components are included with equal weight.

    See Also
    --------
    `LogLoss` : Logarithmic loss scorer.
    `BrierScore` : Brier score for multi-class probabilities.
    `Accuracy` : Accuracy from argmax of predicted probabilities.
    `BaseClassProbaForecaster` : Produces class-probability forecasts.

    """

    _response_method: str = "predict_class_proba"

    _parameter_constraints: dict = {
        **BaseScorer._parameter_constraints,
        "aggregation_method": [
            list,
            StrOptions({"all", "stepwise", "vintagewise", "componentwise", "groupwise"}),
        ],
    }

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ):
        super().__init__(
            groups=groups,
            components=components,
        )
        self.aggregation_method = aggregation_method

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y_train: pl.DataFrame, *, forecaster=None, **params) -> BaseClassProbaScorer:
        """Fit the scorer on training data.

        Validates ``aggregation_method``, ``groups``, and
        ``component_names``.

        Parameters
        ----------
        y_train : pl.DataFrame
            Training target time series with a ``"time"`` column and one or
            more categorical value columns.
        forecaster : BaseForecaster or None, default=None
            If provided, metadata is extracted directly from the fitted
            forecaster instead of being re-inferred from ``y_train``.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted scorer instance.

        Raises
        ------
        ValueError
            If ``aggregation_method`` contains invalid values, or if
            ``groups`` / ``component_names`` are not found in
            ``y_train``.

        """
        valid_methods = {"stepwise", "vintagewise", "componentwise", "groupwise"}
        self._validate_parameters(
            y_train=y_train,
            aggregation_method=self.aggregation_method,
            valid_aggregation_methods=valid_methods,
        )
        return super().fit(y_train, forecaster=forecaster, **params)

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with scorer-specific attributes.

        """
        tags = super().__sklearn_tags__()
        assert tags.scorer_tags is not None
        tags.scorer_tags.prediction_type = "class_proba"
        return tags

    @staticmethod
    def _extract_class_proba_columns(y_pred: pl.DataFrame, target_col: str) -> tuple[list[str], list[str]]:
        """Extract probability columns and class labels for a target.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Probability predictions with columns ``{target}_proba_{class}``.
        target_col : str
            Target column name.

        Returns
        -------
        tuple of (list of str, list of str)
            Probability column names and corresponding class labels.

        """
        proba_cols = [c for c in y_pred.columns if c.startswith(f"{target_col}_proba_")]
        class_labels = [c.split("_proba_", 1)[1] for c in proba_cols]
        return proba_cols, class_labels

    @staticmethod
    def _extract_target_columns(y_truth: pl.DataFrame) -> list[str]:
        """Extract target column names from truth DataFrame.

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth (time columns already removed).

        Returns
        -------
        list of str
            Target column names.

        """
        return y_truth.columns

    def _validate_probabilities(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
    ) -> None:
        """Validate that probability columns contain valid values.

        Checks that all probability columns are finite and in [0, 1].

        Raises
        ------
        ValueError
            If any probability column contains NaN, infinite, or
            out-of-range values.

        """
        target_cols = self._extract_target_columns(y_truth)
        for target_col in target_cols:
            proba_cols, _ = self._extract_class_proba_columns(y_pred, target_col)
            if not proba_cols:
                continue
            proba_data = y_pred.select(proba_cols)
            arr = proba_data.to_numpy()
            if not np.all(np.isfinite(arr)):
                bad_cols = [c for c in proba_cols if not np.all(np.isfinite(y_pred[c].to_numpy()))]
                raise ValueError(
                    f"Probability columns contain NaN or infinite values: {bad_cols}. All probabilities must be finite."
                )
            if np.any(arr < 0) or np.any(arr > 1):
                bad_cols = [
                    c for c in proba_cols if np.any(y_pred[c].to_numpy() < 0) or np.any(y_pred[c].to_numpy() > 1)
                ]
                raise ValueError(
                    f"Probability columns contain values outside [0, 1]: {bad_cols}. "
                    "All probabilities must be between 0 and 1."
                )

    @abc.abstractmethod
    def _compute_raw_errors(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
    ) -> pl.DataFrame:
        """Compute per-timestep per-component raw scores.

        Subclasses implement only this method.  Access fitted attributes
        and helper methods (e.g. ``_extract_class_proba_columns``,
        ``_extract_target_columns``) via ``self``.

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth values (time column already removed).
        y_pred : pl.DataFrame
            Predicted probabilities (time column already removed).

        Returns
        -------
        pl.DataFrame
            Raw error values with one column per target component.

        """

    def score(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
        /,
        time_weight: Callable | pl.DataFrame | dict[datetime | str, float] | None = None,
        step_weight: Callable | pl.DataFrame | dict[int | str, float] | None = None,
        vintage_weight: Callable | pl.DataFrame | dict[datetime | str, float] | None = None,
        **params,
    ) -> float | pl.DataFrame:
        """Compute the class-probability metric score.

        Template method: validate -> pre-filter zeros -> compute raw errors
        -> apply weights -> aggregate -> rename.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True class labels with ``"time"`` column.
        y_pred : pl.DataFrame
            Predicted probabilities with ``"time"`` column.
        time_weight : callable, pl.DataFrame, dict, or None, default=None
            Time-based evaluation weights. Accepts a callable
            ``f(time_series) -> pl.Series``, a panel-aware callable
            ``f(time_series, group_name) -> pl.Series``, a DataFrame
            with ``"time"`` and ``"weight"`` columns, or a
            ``{datetime_or_str: float}`` dict (``"*"`` key sets default).
        step_weight : callable, pl.DataFrame, dict, or None, default=None
            Per-step weights. Same formats as ``time_weight`` but keyed on
            ``"forecasting_step"``.
        vintage_weight : callable, pl.DataFrame, dict, or None, default=None
            Per-vintage weights. Same formats as ``time_weight`` but keyed
            on ``"vintage_time"``.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        float or pl.DataFrame
            Aggregated metric score.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, context = validate_scorer_data(self, y_truth, y_pred)

        # 0. Resolve weights and pre-filter zero-weight rows
        y_truth, y_pred, context, tw, sw, vw = self._pre_filter_zero_weights(
            y_truth,
            y_pred,
            context,
            time_weight,
            step_weight,
            vintage_weight,
        )

        # 0b. Validate probability columns are finite and in [0, 1]
        self._validate_probabilities(y_truth, y_pred)

        # 1. Compute raw per-timestep per-component errors
        scores = self._compute_raw_errors(y_truth, y_pred)

        # 2. Apply weights (two-stage: time first, then step+vintage)
        scores = self._apply_weights(scores, tw, sw, vw)

        # 3. Aggregate
        result = self._aggregate_scores(scores, context=context)

        # 4. Rename columns
        if isinstance(result, pl.DataFrame):
            result = self._rename_metric_columns(result)

        return result
