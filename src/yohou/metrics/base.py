"""Base classes for forecasting metrics and scoring functions."""

from __future__ import annotations

import abc
import re
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from yohou.metrics._context import ScoringContext
from yohou.utils import Tags, inspect_panel, validate_scorer_data
from yohou.utils._compat import StrOptions, _fit_context
from yohou.utils.validation import check_interval_consistency
from yohou.utils.weighting import (
    normalize_weights,
    resolve_weight_to_array,
    validate_callable_signature,
)

if TYPE_CHECKING:
    from datetime import datetime

__all__ = [
    "BaseClassProbaScorer",
    "BaseHardLabelScorer",
    "BaseIntervalScorer",
    "BasePointScorer",
    "BaseRankingScorer",
    "BaseScorer",
]


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
    - [`BasePointScorer`][yohou.metrics.base.BasePointScorer] : Base class for point-prediction metrics.
    - [`BaseIntervalScorer`][yohou.metrics.base.BaseIntervalScorer] : Base class for interval-prediction metrics.
    - [`BaseConformityScorer`][yohou.metrics.conformity_base.BaseConformityScorer] : Base class for conformity scorers.

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
        """Collapse row dimensions (stepwise and/or vintagewise) using mean."""
        return self._collapse_rows_with(df, context, dims, agg_fn="mean")

    def _collapse_rows_with(
        self,
        df: pl.DataFrame,
        context: ScoringContext | None,
        dims: set[str],
        agg_fn: str = "mean",
    ) -> pl.DataFrame:
        """Collapse row dimensions using the specified aggregation function.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with per-row scores.
        context : ScoringContext or None
            Scoring context with time values and metadata.
        dims : set of str
            Aggregation dimensions.
        agg_fn : str, default="mean"
            Polars aggregation function name ("mean", "sum", "max").

        """
        collapse_steps = "stepwise" in dims
        collapse_vintages = "vintagewise" in dims

        if not collapse_steps and not collapse_vintages:
            return df

        meta_names = {"coverage_rate"}
        meta_cols = [c for c in df.columns if c in meta_names]
        val_cols = [c for c in df.columns if c not in meta_names]

        def _agg_exprs(cols: list[str]) -> list[pl.Expr]:
            """Build aggregation expressions for the given columns."""
            return [getattr(pl.col(c), agg_fn)() for c in cols]

        if collapse_steps and collapse_vintages:
            # Per-vintage-first: group by vintage_time within each vintage,
            # then return per-vintage rows (vintage collapse happens later).
            vintage_time = getattr(context, "vintage_time", None) if context is not None else None
            if vintage_time is not None and vintage_time.n_unique() > 1:
                vt_list = vintage_time.to_list()
                if "coverage_rate" in df.columns:
                    n_rates = df["coverage_rate"].n_unique()
                    vt_list = vt_list * n_rates
                group_cols = ["vintage_time"] + meta_cols
                return (
                    df
                    .with_columns(pl.Series("vintage_time", vt_list))
                    .group_by(group_cols, maintain_order=True)
                    .agg(_agg_exprs(val_cols))
                    .sort("vintage_time")
                )

            if meta_cols:
                return df.group_by(meta_cols, maintain_order=True).agg(_agg_exprs(val_cols))
            if agg_fn == "mean":
                return df.select(pl.all().mean())
            return df.select(_agg_exprs(val_cols))

        # Partial collapse: keep one dimension, collapse the other
        keep_dim = "forecasting_step" if collapse_vintages else "vintage_time"
        dim_values = getattr(context, keep_dim, None) if context is not None else None

        if dim_values is None:
            if meta_cols:
                return df.group_by(meta_cols, maintain_order=True).agg(_agg_exprs(val_cols))
            if agg_fn == "mean":
                return df.select(pl.all().mean())
            return df.select(_agg_exprs(val_cols))

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
            .agg(_agg_exprs(val_cols))
            .sort(keep_dim)
        )

    def _collapse_vintage_dimension(
        self,
        df: pl.DataFrame,
        context: ScoringContext | None,
        dims: set[str],
    ) -> pl.DataFrame:
        """Collapse vintage_time rows via (weighted) mean.

        Internal method, not an override point. Fires after _transform_scores
        to produce cross-vintage aggregation. No-op when vintage_time column
        is absent or vintagewise not in dims.
        """
        if "vintage_time" not in df.columns or "vintagewise" not in dims:
            return df

        meta_names = {"coverage_rate", "forecasting_step", "time"}
        meta_cols = [c for c in df.columns if c in meta_names]
        val_cols = [c for c in df.columns if c not in meta_names and c != "vintage_time"]

        vintage_weight = context.vintage_weight if context is not None else None

        if vintage_weight is not None:
            # Weighted mean across vintages
            # Attach weight per vintage row
            unique_vintages = df["vintage_time"].unique(maintain_order=True).to_list()
            if len(vintage_weight) != len(unique_vintages):
                raise ValueError(
                    f"vintage_weight length ({len(vintage_weight)}) does not match "
                    f"number of unique vintages ({len(unique_vintages)})"
                )
            weight_map = dict(zip(unique_vintages, vintage_weight.tolist(), strict=True))
            df = df.with_columns(pl.col("vintage_time").replace_strict(weight_map, default=1.0).alias("_vw"))
            if meta_cols:
                result = df.group_by(meta_cols, maintain_order=True).agg([
                    (pl.col(c) * pl.col("_vw")).sum() / pl.col("_vw").sum() for c in val_cols
                ])
            else:
                total = df["_vw"].sum()
                result = df.select([(pl.col(c) * pl.col("_vw")).sum() / total for c in val_cols])
        # Unweighted mean across vintages
        elif meta_cols:
            result = df.group_by(meta_cols, maintain_order=True).agg([pl.col(c).mean() for c in val_cols])
        else:
            result = df.select([pl.col(c).mean() for c in val_cols])

        return result

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

        # Derive per-unique-vintage weights and store in context
        if isinstance(vw_resolved, dict) and context.vintage_time is not None:
            # Panel-aware vintage_weight: all groups share the same vintage_time
            # axis, so use the first group's weights for cross-vintage weighting.
            first_group_weights = next(iter(vw_resolved.values()))  # ty: ignore[invalid-assignment]
            context = self._set_vintage_weight_on_context(context, first_group_weights)  # ty: ignore[invalid-argument-type]
            vw_resolved = None
        elif isinstance(vw_resolved, np.ndarray) and context.vintage_time is not None:
            context = self._set_vintage_weight_on_context(context, vw_resolved)
            # vintage_weight handled at cross-vintage level; don't pass to _apply_weights
            vw_resolved = None

        return y_truth, y_pred, context, tw_resolved, sw_resolved, vw_resolved

    @staticmethod
    def _set_vintage_weight_on_context(
        context: ScoringContext,
        vw_array: np.ndarray,
    ) -> ScoringContext:
        """Attach per-unique-vintage weights to a scoring context."""
        vt_df = pl.DataFrame({
            "vintage_time": context.vintage_time,
            "_vw": vw_array,
        })
        per_vintage = vt_df.group_by("vintage_time", maintain_order=True).agg(pl.col("_vw").first())
        return ScoringContext(
            time_values=context.time_values,
            vintage_time=context.vintage_time,
            forecasting_step=context.forecasting_step,
            vintage_weight=per_vintage["_vw"].to_numpy(),
        )

    def _resolve_vintage_weight_to_context(
        self,
        context: ScoringContext,
        vintage_weight: Callable | pl.DataFrame | dict | None,
    ) -> ScoringContext:
        """Resolve a vintage_weight argument into context.vintage_weight.

        Lightweight alternative to ``_pre_filter_zero_weights`` for scorers
        that only need vintage_weight resolution (no time/step weights).
        """
        if vintage_weight is None:
            return context
        if context.vintage_time is None:
            return context
        vw_resolved = resolve_weight_to_array(
            vintage_weight,
            context.vintage_time,
            "vintage_time",
        )
        if isinstance(vw_resolved, dict):
            raise TypeError(
                "Panel-aware (dict) vintage_weight is not supported. "
                "Use a flat callable, DataFrame, or dict keyed on vintage_time values."
            )
        return self._set_vintage_weight_on_context(context, vw_resolved)

    def _apply_weights(
        self,
        scores: pl.DataFrame,
        time_weight_resolved: np.ndarray | dict[str, np.ndarray] | None,
        step_weight_resolved: np.ndarray | dict[str, np.ndarray] | None,
        n_rates: int = 1,
    ) -> pl.DataFrame:
        """Apply pre-resolved weights to score DataFrame.

        Two-stage normalization: (1) normalize and apply time_weight,
        (2) apply step_weight (normalized).

        Vintage weights are not applied at row level; they are handled
        during cross-vintage aggregation in ``_collapse_vintage_dimension``.

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

        # Stage 2: step_weight (normalized independently)
        if step_weight_resolved is not None:
            scores = _apply_one_weight(scores, step_weight_resolved, value_cols)

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

    def _transform_scores(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply element-wise transform to aggregated value columns.

        Override in subclasses needing post-aggregation transforms
        (e.g. sqrt for RMSE). Called inside ``_aggregate_per_vintage_scores``
        after component/group collapse, before vintage collapse.
        Receives only value columns (no meta columns).

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with value columns only.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame.

        """
        return df

    @staticmethod
    def _reject_weights(**params: object) -> None:
        """Raise if any weight kwargs are passed to a scorer that doesn't support them."""
        weight_keys = [
            k for k in params if k in {"time_weight", "step_weight", "vintage_weight"} and params[k] is not None
        ]
        if weight_keys:
            raise TypeError(
                f"This scorer does not support sample weights, "
                f"but received: {', '.join(sorted(weight_keys))}. "
                f"Remove the weight arguments or use a scorer that supports weighting."
            )

    def _aggregate_per_vintage_scores(
        self,
        result: pl.DataFrame,
        context: ScoringContext | None,
    ) -> float | pl.DataFrame:
        """Shared pipeline tail for all scorer families.

        Input: a per-vintage DataFrame (one row per vintage with
        ``vintage_time`` column for multi-vintage, or single-row without
        it for single-vintage). Pipeline:
        components → groups → strip meta → _transform_scores → reattach meta
        → _collapse_vintage_dimension → finalize → rename.
        """
        dims = self._normalize_agg_methods(self.aggregation_method)  # ty: ignore[unresolved-attribute]

        # Input validation: multi-vintage context but no vintage_time column.
        # Only fires when both stepwise and vintagewise are requested, because
        # in that case _collapse_rows preserves vintage_time for per-vintage-first
        # aggregation and _collapse_vintage_dimension handles the final collapse.
        # When only vintagewise is requested, _collapse_rows already collapses
        # vintages by grouping by forecasting_step, so vintage_time is gone.
        if (
            "stepwise" in dims
            and "vintagewise" in dims
            and context is not None
            and context.vintage_time is not None
            and context.vintage_time.n_unique() > 1
            and "vintage_time" not in result.columns
        ):
            raise ValueError(
                "Context has multiple vintages but input DataFrame is missing "
                "'vintage_time' column. Attach vintage_time before calling "
                "_aggregate_per_vintage_scores."
            )

        if "componentwise" in dims:
            result = self._collapse_components(result)

        if "groupwise" in dims:
            result = self._collapse_groups(result)

        # Strip meta → _transform_scores → reattach meta
        meta_names = {"coverage_rate", "forecasting_step", "vintage_time", "time"}
        meta_cols = [c for c in result.columns if c in meta_names]
        val_cols = [c for c in result.columns if c not in meta_names]

        if val_cols:
            transformed = self._transform_scores(result.select(val_cols))
            result = pl.concat([result.select(meta_cols), transformed], how="horizontal") if meta_cols else transformed

        # Collapse vintage dimension (weighted/unweighted mean across vintages)
        result = self._collapse_vintage_dimension(result, context, dims)

        # Finalize: attach labels, convert to scalar
        finalized = self._finalize(result, context, dims)

        if isinstance(finalized, pl.DataFrame):
            finalized = self._rename_metric_columns(finalized)

        return finalized

    def _map_per_vintage(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
        context: ScoringContext | None,
        compute_fn: Callable[[pl.DataFrame, pl.DataFrame], pl.DataFrame | None],
    ) -> pl.DataFrame:
        """Group data by vintage and apply compute_fn per group.

        Utility for Pattern 2 scorers (whole-column computation).
        Returns a per-vintage DataFrame with ``vintage_time`` column
        for multi-vintage, or a single-row DataFrame (no vintage_time)
        for single-vintage.

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth (time column removed).
        y_pred : pl.DataFrame
            Predictions (time column removed).
        context : ScoringContext or None
            Scoring context.
        compute_fn : callable
            ``(y_truth_slice, y_pred_slice) -> pl.DataFrame | None``.
            Returns single-row DataFrame per vintage or None to skip.

        Returns
        -------
        pl.DataFrame
            Per-vintage results concatenated.

        Raises
        ------
        ValueError
            If all vintages are skipped.

        """
        vintage_time = context.vintage_time if context is not None else None

        # Single-vintage fallthrough
        if vintage_time is None or vintage_time.n_unique() <= 1:
            result = compute_fn(y_truth, y_pred)
            if result is None:
                raise ValueError("All vintage groups were skipped. No valid data to compute the metric.")
            return result

        # Multi-vintage: group by vintage
        vt_values = vintage_time.to_list()
        unique_vintages = vintage_time.unique(maintain_order=True).to_list()

        results: list[pl.DataFrame] = []
        for vt in unique_vintages:
            mask = [v == vt for v in vt_values]
            yt_slice = y_truth.filter(mask)
            yp_slice = y_pred.filter(mask)
            row = compute_fn(yt_slice, yp_slice)
            if row is None:
                continue
            row = row.with_columns(pl.lit(vt).alias("vintage_time").cast(pl.Datetime))
            results.append(row)

        if not results:
            raise ValueError("All vintage groups were skipped. No valid data to compute the metric.")

        return pl.concat(results, how="diagonal_relaxed")

    def _aggregate_scores(
        self, raw_scores: pl.DataFrame, context: ScoringContext | None = None
    ) -> float | pl.DataFrame:
        """Apply sequential aggregation pipeline to raw scores.

        Pipeline: coverage → rows (per-vintage) → _aggregate_per_vintage_scores.

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

        # 3. Delegate tail to shared pipeline
        return self._aggregate_per_vintage_scores(result, context)

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
    - [`MeanAbsoluteError`][yohou.metrics.point.MeanAbsoluteError] : Concrete point scorer implementation.
    - [`MeanSquaredError`][yohou.metrics.point.MeanSquaredError] : Concrete point scorer implementation.
    - [`BasePointForecaster`][yohou.point.base.BasePointForecaster] : Produces point forecasts.

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
        y_truth, y_pred, context, tw, sw, _ = self._pre_filter_zero_weights(
            y_truth,
            y_pred,
            context,
            time_weight,
            step_weight,
            vintage_weight,
        )

        # 1. Compute raw per-timestep per-component errors
        scores = self._compute_raw_errors(y_truth, y_pred)

        # 2. Apply weights (time first, then step)
        scores = self._apply_weights(scores, tw, sw)

        # 3. Aggregate (includes transform + rename via _aggregate_per_vintage_scores)
        return self._aggregate_scores(scores, context=context)

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
    - [`IntervalScore`][yohou.metrics.interval.IntervalScore] : Concrete interval scorer implementation.
    - [`EmpiricalCoverage`][yohou.metrics.interval.EmpiricalCoverage] : Concrete interval scorer implementation.
    - [`BaseIntervalForecaster`][yohou.interval.base.BaseIntervalForecaster] : Produces interval forecasts.

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
        y_truth, y_pred, context, tw, sw, _ = self._pre_filter_zero_weights(
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
        raw_scores = self._apply_weights(raw_scores, tw, sw, n_rates=n_rates)

        # 3. Aggregate (includes transform + rename via _aggregate_per_vintage_scores)
        return self._aggregate_scores(raw_scores, context=context)

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
    - [`LogLoss`][yohou.metrics.class_proba.LogLoss] : Logarithmic loss scorer.
    - [`BrierScore`][yohou.metrics.class_proba.BrierScore] : Brier score for multi-class probabilities.
    - [`Accuracy`][yohou.metrics.classification.Accuracy] : Accuracy from argmax of predicted probabilities.
    - [`BaseClassProbaForecaster`][yohou.class_proba.base.BaseClassProbaForecaster] : Produces class-probability forecasts.

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
        y_truth, y_pred, context, tw, sw, _ = self._pre_filter_zero_weights(
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

        # 2. Apply weights (time first, then step)
        scores = self._apply_weights(scores, tw, sw)

        # 3. Aggregate (includes transform + rename via _aggregate_per_vintage_scores)
        return self._aggregate_scores(scores, context=context)


class BaseHardLabelScorer(BaseClassProbaScorer, metaclass=abc.ABCMeta):
    """Base class for confusion-matrix classification metrics.

    Extends :class:`BaseClassProbaScorer` for metrics that argmax predicted
    probabilities into hard labels and compute scores from the resulting
    confusion matrix (TP, FP, FN counts).

    The ``score()`` method is inherited from :class:`BaseClassProbaScorer`.
    Subclasses override :meth:`_compute_raw_errors` (produces per-row
    confusion indicators) and :meth:`_aggregate_scores` (sums indicators,
    computes metrics, applies class averaging).

    Parameters
    ----------
    average : str, default="macro"
        Class averaging strategy: ``"macro"`` (unweighted mean across
        classes), ``"micro"`` (aggregate counts across classes first),
        or ``"weighted"`` (support-weighted mean).
    zero_division : float, default=0.0
        Value to return when a metric denominator is zero.
    aggregation_method : list of str or str, default="all"
        Which dimensions to aggregate.
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter or filter with weights.
    components : list of str, dict of str to float, or None, default=None
        Component filter or filter with weights.

    """

    _parameter_constraints: dict = {
        **BaseClassProbaScorer._parameter_constraints,
        "average": [StrOptions({"macro", "micro", "weighted"})],
        "zero_division": "no_validation",
    }

    def __init__(
        self,
        average: str = "macro",
        zero_division: float = 0.0,
        aggregation_method: list[str] | str = "all",
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ):
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
        )
        self.average = average
        self.zero_division = zero_division

    def _compute_raw_errors(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Compute per-row confusion indicators for each class and target.

        Argmaxes predicted probabilities into hard labels and computes
        binary TP/FP/FN indicators per row per class.

        Returns a wide DataFrame with columns like
        ``{target}_tp_{class}``, ``{target}_fp_{class}``,
        ``{target}_fn_{class}`` (target includes panel group prefix
        when present, e.g. ``grpA__weather_tp_sunny``).

        """
        target_cols = self._extract_target_columns(y_truth)
        all_indicator_cols: list[pl.Series] = []

        for target_col in target_cols:
            proba_cols, class_labels = self._extract_class_proba_columns(y_pred, target_col)
            if not proba_cols:
                continue
            self._build_confusion_indicators(
                y_truth[target_col],
                y_pred.select(proba_cols),
                class_labels,
                target_col,
                all_indicator_cols,
            )

        return pl.DataFrame(all_indicator_cols)

    @staticmethod
    def _build_confusion_indicators(
        truth_series: pl.Series,
        proba_df: pl.DataFrame,
        class_labels: list[str],
        prefix: str,
        out: list[pl.Series],
    ) -> None:
        """Build TP/FP/FN indicator columns for one target.

        Parameters
        ----------
        truth_series : pl.Series
            True class labels.
        proba_df : pl.DataFrame
            Probability columns for this target.
        class_labels : list of str
            Class label names (one per proba column).
        prefix : str
            Column name prefix (target or group__target).
        out : list of pl.Series
            Output list; indicator Series are appended in place.

        """
        truth_arr = truth_series.to_numpy().astype(str)
        proba_arr = proba_df.to_numpy()
        pred_indices = np.argmax(proba_arr, axis=1)
        pred_labels = np.array(class_labels)[pred_indices]

        for cls in class_labels:
            is_true = (truth_arr == cls).astype(np.float64)
            is_pred = (pred_labels == cls).astype(np.float64)
            tp = is_true * is_pred
            fp = (1.0 - is_true) * is_pred
            fn = is_true * (1.0 - is_pred)
            out.append(pl.Series(f"{prefix}_tp_{cls}", tp))
            out.append(pl.Series(f"{prefix}_fp_{cls}", fp))
            out.append(pl.Series(f"{prefix}_fn_{cls}", fn))

    def _aggregate_scores(
        self, raw_scores: pl.DataFrame, context: ScoringContext | None = None
    ) -> float | pl.DataFrame:
        """Aggregate confusion indicators into final metric scores.

        Pipeline: sum rows (per-vintage) → compute metric from counts →
        delegate tail to _aggregate_per_vintage_scores.

        """
        dims = self._normalize_agg_methods(self.aggregation_method)

        # 1. Collapse rows via SUM (not mean) — produces per-vintage sums
        result = self._collapse_rows_sum(raw_scores, context, dims)

        # 2 & 3. Compute metric from counts and apply class averaging
        result = self._compute_and_average(result)

        # 4. Delegate tail (components → groups → transform → vintage collapse → finalize)
        return self._aggregate_per_vintage_scores(result, context)

    def _collapse_rows_sum(
        self,
        df: pl.DataFrame,
        context: ScoringContext | None,
        dims: set[str],
    ) -> pl.DataFrame:
        """Collapse row dimensions using sum (for confusion counts)."""
        return self._collapse_rows_with(df, context, dims, agg_fn="sum")

    def _compute_and_average(self, df: pl.DataFrame) -> pl.DataFrame:
        """Compute metric from counts and apply class averaging.

        For micro averaging, sums TP/FP/FN across all classes before
        computing the metric. For macro/weighted, computes per-class
        metrics first, then averages.

        """
        meta_names = {"coverage_rate", "forecasting_step", "vintage_time", "time"}
        meta_cols = [c for c in df.columns if c in meta_names]
        val_cols = [c for c in df.columns if c not in meta_names]

        # Parse column structure: group targets and their classes
        targets = self._parse_indicator_columns(val_cols)

        result_cols: list[pl.Series] = []

        for col_prefix, class_labels in targets.items():
            if self.average == "micro":
                score_series = self._micro_average(df, col_prefix, class_labels)
            elif self.average == "weighted":
                score_series = self._weighted_average(df, col_prefix, class_labels)
            else:
                score_series = self._macro_average(df, col_prefix, class_labels)
            result_cols.append(score_series)

        meta_df = df.select(meta_cols) if meta_cols else pl.DataFrame()
        score_df = pl.DataFrame(result_cols)
        if len(meta_df) > 0:
            return pl.concat([meta_df, score_df], how="horizontal")
        return score_df

    @staticmethod
    def _parse_indicator_columns(val_cols: list[str]) -> dict[str, list[str]]:
        """Parse indicator columns into {target_prefix: [class_labels]}.

        Columns follow the pattern ``{prefix}_tp_{class}``,
        ``{prefix}_fp_{class}``, ``{prefix}_fn_{class}``.
        """
        targets: dict[str, list[str]] = {}
        for col in val_cols:
            for indicator in ("_tp_", "_fp_", "_fn_"):
                if indicator in col:
                    # Use rindex to match the last occurrence, so class
                    # labels containing "_tp_" etc. are handled correctly.
                    idx = col.rindex(indicator)
                    prefix = col[:idx]
                    cls = col[idx + len(indicator) :]
                    if prefix not in targets:
                        targets[prefix] = []
                    if cls not in targets[prefix]:
                        targets[prefix].append(cls)
                    break
        return targets

    def _micro_average(self, df: pl.DataFrame, col_prefix: str, class_labels: list[str]) -> pl.Series:
        """Micro averaging: sum counts across classes, then compute metric."""
        tp_cols = [f"{col_prefix}_tp_{cls}" for cls in class_labels]
        fp_cols = [f"{col_prefix}_fp_{cls}" for cls in class_labels]
        fn_cols = [f"{col_prefix}_fn_{cls}" for cls in class_labels]

        tp_total = df.select(pl.sum_horizontal(tp_cols)).to_series()
        fp_total = df.select(pl.sum_horizontal(fp_cols)).to_series()
        fn_total = df.select(pl.sum_horizontal(fn_cols)).to_series()

        metric_df = self._compute_metric_from_counts(pl.DataFrame({"tp": tp_total, "fp": fp_total, "fn": fn_total}))
        return metric_df.to_series().alias(col_prefix)

    def _macro_average(self, df: pl.DataFrame, col_prefix: str, class_labels: list[str]) -> pl.Series:
        """Macro averaging: compute per-class metric, then unweighted mean."""
        per_class: list[pl.Series] = []
        for i, cls in enumerate(class_labels):
            counts = pl.DataFrame({
                "tp": df[f"{col_prefix}_tp_{cls}"],
                "fp": df[f"{col_prefix}_fp_{cls}"],
                "fn": df[f"{col_prefix}_fn_{cls}"],
            })
            metric_df = self._compute_metric_from_counts(counts)
            per_class.append(metric_df.to_series().alias(f"_cls_{i}"))

        stacked = pl.DataFrame(per_class)
        return stacked.select(pl.mean_horizontal(pl.all())).to_series().alias(col_prefix)

    def _weighted_average(self, df: pl.DataFrame, col_prefix: str, class_labels: list[str]) -> pl.Series:
        """Weighted averaging: per-class metric weighted by class support."""
        per_class_metrics: list[pl.Series] = []
        supports: list[pl.Series] = []
        for i, cls in enumerate(class_labels):
            counts = pl.DataFrame({
                "tp": df[f"{col_prefix}_tp_{cls}"],
                "fp": df[f"{col_prefix}_fp_{cls}"],
                "fn": df[f"{col_prefix}_fn_{cls}"],
            })
            metric_df = self._compute_metric_from_counts(counts)
            per_class_metrics.append(metric_df.to_series().alias(f"_cls_{i}"))
            # Support = TP + FN (number of true instances of this class)
            supports.append((df[f"{col_prefix}_tp_{cls}"] + df[f"{col_prefix}_fn_{cls}"]).alias(f"_sup_{i}"))

        # Build a temporary DataFrame for clean computation
        tmp_data: dict[str, pl.Series] = {}
        for i, (m, s) in enumerate(zip(per_class_metrics, supports, strict=True)):
            tmp_data[f"_m_{i}"] = m
            tmp_data[f"_s_{i}"] = s
        tmp = pl.DataFrame(tmp_data)

        n_classes = len(class_labels)
        metric_cols = [f"_m_{i}" for i in range(n_classes)]
        support_cols = [f"_s_{i}" for i in range(n_classes)]

        result = tmp.select(
            (
                pl.sum_horizontal([pl.col(mc) * pl.col(sc) for mc, sc in zip(metric_cols, support_cols, strict=True)])
                / pl.max_horizontal(pl.sum_horizontal([pl.col(sc) for sc in support_cols]), pl.lit(1.0))
            ).alias(col_prefix)
        )
        return result.to_series()

    @abc.abstractmethod
    def _compute_metric_from_counts(self, counts: pl.DataFrame) -> pl.DataFrame:
        """Compute metric value from confusion counts.

        Parameters
        ----------
        counts : pl.DataFrame
            DataFrame with ``"tp"``, ``"fp"``, ``"fn"`` columns.
            Each row corresponds to a temporal group (or a single row
            when all rows are collapsed).

        Returns
        -------
        pl.DataFrame
            Single-column DataFrame with computed metric values.

        """


class BaseRankingScorer(BaseClassProbaScorer, metaclass=abc.ABCMeta):
    """Base class for ranking classification metrics.

    Extends :class:`BaseClassProbaScorer` for metrics that require the
    full probability vector and cannot produce meaningful per-row values
    (e.g., ROC AUC, PR AUC). Overrides ``score()`` entirely.

    Parameters
    ----------
    average : str, default="macro"
        Class averaging strategy: ``"macro"`` (unweighted mean across
        classes) or ``"weighted"`` (support-weighted mean).
    aggregation_method : list of str or str, default="all"
        Which dimensions to aggregate.
    groups : list of str, dict of str to float, or None, default=None
        Panel group filter or filter with weights.
    components : list of str, dict of str to float, or None, default=None
        Component filter or filter with weights.

    """

    _parameter_constraints: dict = {
        **BaseClassProbaScorer._parameter_constraints,
        "average": [StrOptions({"macro", "weighted"})],
    }

    def __init__(
        self,
        average: str = "macro",
        aggregation_method: list[str] | str = "all",
        groups: list[str] | dict[str, float] | None = None,
        components: list[str] | dict[str, float] | None = None,
    ):
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
        )
        self.average = average

    def _compute_raw_errors(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Not used by ranking scorers (score() is overridden)."""
        raise NotImplementedError("Ranking scorers override score() directly")

    @abc.abstractmethod
    def _compute_ranking_metric(
        self,
        y_true_binary: np.ndarray,
        y_proba: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """Compute ranking metric for a single class (one-vs-rest).

        Parameters
        ----------
        y_true_binary : np.ndarray
            Binary array (1 for positive class, 0 otherwise).
        y_proba : np.ndarray
            Predicted probabilities for the positive class.
        sample_weight : np.ndarray or None
            Per-sample weights.

        Returns
        -------
        float
            Metric value for this class.

        """

    def score(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
        /,
        time_weight: Callable | pl.DataFrame | dict | None = None,
        step_weight: Callable | pl.DataFrame | dict | None = None,
        vintage_weight: Callable | pl.DataFrame | dict | None = None,
        **params,
    ) -> float | pl.DataFrame:
        """Compute the ranking metric score.

        Overrides the full scoring pipeline because ranking metrics
        require the complete probability vector per group and cannot
        produce meaningful per-row error values.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True class labels with ``"time"`` column.
        y_pred : pl.DataFrame
            Predicted probabilities with ``"time"`` column.
        time_weight : callable, pl.DataFrame, dict, or None, default=None
            Time-based evaluation weights.
        step_weight : callable, pl.DataFrame, dict, or None, default=None
            Per-step weights.
        vintage_weight : callable, pl.DataFrame, dict, or None, default=None
            Per-vintage weights.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        float or pl.DataFrame
            Aggregated metric score.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, context = validate_scorer_data(self, y_truth, y_pred)

        y_truth, y_pred, context, tw, sw, _ = self._pre_filter_zero_weights(
            y_truth,
            y_pred,
            context,
            time_weight,
            step_weight,
            vintage_weight,
        )

        self._validate_probabilities(y_truth, y_pred)

        # Resolve sample weights (tw + sw only; vintage weight handled via context)
        sample_weight = self._resolve_combined_weights(tw, sw, len(y_truth))

        target_cols = self._extract_target_columns(y_truth)
        vintage_time = context.vintage_time if context is not None else None

        # Per-vintage computation
        if vintage_time is not None and vintage_time.n_unique() > 1:
            vt_values = vintage_time.to_list()
            unique_vintages = vintage_time.unique(maintain_order=True).to_list()

            vintage_results: list[pl.DataFrame] = []
            for vt in unique_vintages:
                mask = [v == vt for v in vt_values]
                yt_slice = y_truth.filter(mask)
                yp_slice = y_pred.filter(mask)
                sw_slice = sample_weight[mask] if sample_weight is not None else None

                row_data: dict[str, list[float]] = {}
                for target_col in target_cols:
                    proba_cols, class_labels = self._extract_class_proba_columns(yp_slice, target_col)
                    if not proba_cols:
                        continue
                    score_val = self._compute_ovr_metric(
                        yt_slice[target_col],
                        yp_slice.select(proba_cols),
                        class_labels,
                        sw_slice,
                    )
                    row_data[target_col] = [score_val]

                row = pl.DataFrame(row_data)
                row = row.with_columns(pl.lit(vt).alias("vintage_time").cast(pl.Datetime))
                vintage_results.append(row)

            result = pl.concat(vintage_results, how="diagonal_relaxed")
        else:
            # Single-vintage: compute across all rows
            result_data: dict[str, list[float]] = {}
            for target_col in target_cols:
                proba_cols, class_labels = self._extract_class_proba_columns(y_pred, target_col)
                if not proba_cols:
                    continue
                score_val = self._compute_ovr_metric(
                    y_truth[target_col],
                    y_pred.select(proba_cols),
                    class_labels,
                    sample_weight,
                )
                result_data[target_col] = [score_val]
            result = pl.DataFrame(result_data)

        # Delegate tail
        return self._aggregate_per_vintage_scores(result, context)

    def _compute_ovr_metric(
        self,
        truth_series: pl.Series,
        proba_df: pl.DataFrame,
        class_labels: list[str],
        sample_weight: np.ndarray | None,
    ) -> float:
        """Compute one-vs-rest ranking metric with class averaging.

        Parameters
        ----------
        truth_series : pl.Series
            True class labels.
        proba_df : pl.DataFrame
            Probability columns.
        class_labels : list of str
            Class label names.
        sample_weight : np.ndarray or None
            Combined sample weights.

        Returns
        -------
        float
            Averaged metric across classes.

        """
        truth_arr = truth_series.to_numpy().astype(str)
        proba_arr = proba_df.to_numpy()

        per_class_scores: list[float] = []
        supports: list[int] = []

        for i, cls in enumerate(class_labels):
            y_binary = (truth_arr == cls).astype(np.float64)
            support = int(y_binary.sum())
            # Skip classes with no positive or no negative samples
            if support == 0 or support == len(y_binary):
                continue
            score_val = self._compute_ranking_metric(
                y_binary,
                proba_arr[:, i],
                sample_weight,
            )
            per_class_scores.append(score_val)
            supports.append(support)

        if not per_class_scores:
            return 0.0

        if self.average == "weighted":
            total_support = sum(supports)
            return sum(s * v / total_support for s, v in zip(supports, per_class_scores, strict=True))
        # macro
        return sum(per_class_scores) / len(per_class_scores)

    @staticmethod
    def _resolve_combined_weights(
        tw: np.ndarray | dict[str, np.ndarray] | None,
        sw: np.ndarray | dict[str, np.ndarray] | None,
        n: int,
    ) -> np.ndarray | None:
        """Combine resolved weight arrays into a single sample weight vector.

        For simplicity, ranking scorers use a single combined weight array.
        Dict (panel-aware) weights are not supported and are skipped
        with a warning.
        """
        arrays = []
        for w in (tw, sw):
            if w is None:
                continue
            if isinstance(w, dict):
                warnings.warn(
                    "Panel-aware (dict) weights are not supported by ranking scorers and will be ignored. "
                    "Use a flat callable or DataFrame weight instead.",
                    UserWarning,
                    stacklevel=3,
                )
                continue
            arrays.append(w)

        if not arrays:
            return None

        combined = np.ones(n, dtype=np.float64)
        for arr in arrays:
            combined *= arr
        return combined
