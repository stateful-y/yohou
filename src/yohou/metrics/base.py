"""Base classes for forecasting metrics and scoring functions."""

from __future__ import annotations

import abc
import inspect
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from yohou.utils import Tags, inspect_panel, validate_callable_signature, validate_scorer_data
from yohou.utils._compat import StrOptions, _fit_context
from yohou.utils.validation import check_interval_consistency

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
    groups : list of str or None, default=None
        List of panel group names to include in scoring. If None, all panel groups
        are included. Only applicable for panel data.
    component_names : list of str or None, default=None
        List of component (target column) names to include in scoring. If None, all
        components are included. For panel data, these are unprefixed column names.
    group_weight : dict or None, default=None
        Dictionary mapping panel group names to weights for weighted aggregation.
        If None, all panel groups weighted equally. Only applicable for panel data.

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
        "groups": [list, None],
        "component_names": [list, None],
        "group_weight": [dict, None],
        "forecasting_steps": [list, None],
        "component_weight": [dict, None],
        "step_weight": [dict, None],
        "vintage_weight": [dict, None],
    }

    def __init__(
        self,
        groups: list[str] | None = None,
        component_names: list[str] | None = None,
        group_weight: dict[str, float] | None = None,
        forecasting_steps: list[int] | None = None,
        component_weight: dict[str, float] | None = None,
        step_weight: dict[int, float] | None = None,
        vintage_weight: dict[datetime, float] | None = None,
    ):
        self.groups = groups
        self.component_names = component_names
        self.group_weight = group_weight
        self.forecasting_steps = forecasting_steps
        self.component_weight = component_weight
        self.step_weight = step_weight
        self.vintage_weight = vintage_weight

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

    def _apply_panel_weights(self, scores: dict[str, float], groups: list[str]) -> float:
        """Apply panel group weights to aggregate scores.

        Parameters
        ----------
        scores : dict[str, float]
            Mapping of panel group names to their scores.

        groups : list[str]
            List of panel group names present in data.

        Returns
        -------
        float
            Weighted average score.

        """
        if self.group_weight is None:
            # Equal weighting
            return float(np.mean(list(scores.values())))

        # Apply custom weights
        weighted_sum = 0.0
        total_weight = 0.0

        for group in groups:
            if group in scores:
                weight = self.group_weight.get(group, 1.0)
                weighted_sum += scores[group] * weight
                total_weight += weight

        if total_weight == 0:
            raise ValueError("Total panel group weight is zero")

        return weighted_sum / total_weight

    def _process_time_weights(
        self,
        raw_scores: pl.DataFrame,
        time_weight: Callable | pl.DataFrame | None,
        time_values: list | None,
        group_name: str | None = None,
    ) -> pl.DataFrame:
        """Apply time-based weights to raw per-timestep scores.

        Parameters
        ----------
        raw_scores : pl.DataFrame
            Per-timestep per-component scores without "time" column.
        time_weight : callable, pl.DataFrame, or None
            Time weighting specification.
        time_values : list or None
            Time values corresponding to raw_scores rows.
        group_name : str or None
            Panel group name for 2-parameter callables. None for global data.

        Returns
        -------
        pl.DataFrame
            Weighted scores with same shape as raw_scores.

        """
        if time_weight is None:
            return raw_scores

        if callable(time_weight):
            # Validate signature (1 or 2 parameters)
            validate_callable_signature(time_weight)

            if time_values is None:
                raise ValueError("time_values cannot be None when time_weight is callable")

            # Reconstruct time Series for callable
            time_series = pl.Series("time", time_values)

            # Call weight function
            tw = cast("Callable[..., pl.Series]", time_weight)
            sig = inspect.signature(time_weight)
            if len(sig.parameters) == 1:
                # Global weight function
                weights_series = tw(time_series)
            elif len(sig.parameters) == 2:
                # Panel-aware weight function
                weights_series = tw(time_series, group_name)
            else:
                raise ValueError(f"time_weight callable must have 1 or 2 parameters, got {len(sig.parameters)}")

            # Validate weights
            if not isinstance(weights_series, pl.Series):
                raise ValueError(f"time_weight callable must return pl.Series, got {type(weights_series).__name__}")

            if len(weights_series) != len(raw_scores):
                raise ValueError(
                    f"time_weight callable returned {len(weights_series)} weights, "
                    f"but raw_scores has {len(raw_scores)} rows"
                )

            # Convert to numpy array for multiplication
            weights_np = weights_series.to_numpy()

        elif isinstance(time_weight, pl.DataFrame):
            # DataFrame: join on time and extract weight column
            if time_values is None:
                raise ValueError("time_values cannot be None when time_weight is DataFrame")

            # Create DataFrame with time column for joining
            time_df = pl.DataFrame({"time": time_values})

            # Join on time
            joined = time_df.join(time_weight, on="time", how="left")

            # Determine which weight column to use
            if group_name is not None:
                # Panel data: try group-specific column first, fallback to global
                group_col = f"{group_name}_weight"
                if group_col in joined.columns:
                    weights_series = joined[group_col]
                elif "weight" in joined.columns:
                    weights_series = joined["weight"]
                else:
                    raise ValueError(f"time_weight DataFrame missing both '{group_col}' and 'weight' columns")
            else:
                # Global data: use "weight" column
                if "weight" not in joined.columns:
                    raise ValueError("time_weight DataFrame must have 'weight' column for global data")
                weights_series = joined["weight"]

            # Convert to numpy array
            weights_np = weights_series.to_numpy()

        else:
            raise ValueError(f"time_weight must be callable, pl.DataFrame, or None, got {type(time_weight).__name__}")

        # Validate weights
        nan_mask = np.isnan(weights_np)
        if np.any(nan_mask):
            nan_indices = np.where(nan_mask)[0].tolist()
            if time_values is not None:
                missing_times = [time_values[i] for i in nan_indices]
                raise ValueError(
                    f"Time weights contain NaN at {len(nan_indices)} position(s). "
                    f"The time_weight source has no values for times: {missing_times[:5]}"
                    f"{'...' if len(missing_times) > 5 else ''}. "
                    "Check that time_weight covers all scored time points."
                )
            raise ValueError(
                f"Time weights contain NaN at indices {nan_indices[:10]}. "
                "Check that time_weight covers all scored time points."
            )
        if np.any(weights_np < 0):
            raise ValueError("Time weights contain negative values")
        if np.any(np.isinf(weights_np)):
            raise ValueError("Time weights contain infinite values")
        if np.sum(weights_np) == 0:
            raise ValueError(
                "Time weights sum to zero. All weights are zero, so no time points contribute to the score."
            )

        # Normalize weights to sum to number of samples (preserves scale)
        weights_np = weights_np * (len(weights_np) / np.sum(weights_np))

        # Apply weights to each column (multiply each row by its weight)
        weighted_scores = raw_scores.with_columns([(pl.col(col) * weights_np).alias(col) for col in raw_scores.columns])

        return weighted_scores

    def _aggregate_groupwise(self, raw_scores: pl.DataFrame) -> pl.DataFrame:
        """Aggregate across panel groups, returning per-component columns.

        For each component (suffix after ``__``), computes a weighted average
        of the per-timestep scores across all panel groups that contain that
        component.  When ``group_weight`` is ``None``, all groups are
        weighted equally.

        For non-panel data the DataFrame is returned unchanged.

        Parameters
        ----------
        raw_scores : pl.DataFrame
            Per-timestep per-component scores (no ``"time"`` column).

        Returns
        -------
        pl.DataFrame
            DataFrame with panel prefixes removed and groups collapsed.

        """
        _, panel_groups = inspect_panel(raw_scores)
        if len(panel_groups) == 0:
            return raw_scores

        # component -> [(group_name, column_name)]
        components: dict[str, list[tuple[str, str]]] = {}
        for group_name, group_cols in panel_groups.items():
            for col in group_cols:
                component = col.split("__", 1)[1]
                if component not in components:
                    components[component] = []
                components[component].append((group_name, col))

        # Get weight per group
        weights: dict[str, float] = {}
        for group_name in panel_groups:
            w = 1.0
            if self.group_weight is not None:
                w = self.group_weight.get(group_name, 1.0)
            weights[group_name] = w

        # Weighted average across groups for each component
        exprs: list[pl.Expr] = []
        for component, group_cols in components.items():
            total_weight = sum(weights[gn] for gn, _ in group_cols)
            if total_weight == 0:
                msg = "Total panel group weight is zero"
                raise ValueError(msg)
            weighted_terms = [pl.col(col_name) * (weights[gn] / total_weight) for gn, col_name in group_cols]
            exprs.append(pl.sum_horizontal(weighted_terms).alias(component))

        return raw_scores.select(exprs)

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

        # Validate groups type
        if self.groups is not None:
            if not isinstance(self.groups, list):
                raise ValueError(f"groups must be a list or None, got {type(self.groups)}")
            if not all(isinstance(name, str) for name in self.groups):
                raise ValueError("All elements in groups must be strings")
            if len(self.groups) == 0:
                raise ValueError("groups cannot be an empty list")

        # Validate component_names type
        if self.component_names is not None:
            if not isinstance(self.component_names, list):
                raise ValueError(f"component_names must be a list or None, got {type(self.component_names)}")
            if not all(isinstance(name, str) for name in self.component_names):
                raise ValueError("All elements in component_names must be strings")
            if len(self.component_names) == 0:
                raise ValueError("component_names cannot be an empty list")

        # If y_train is provided, validate against actual data
        if y_train is not None:
            _, panel_groups = inspect_panel(y_train)
            available_groups = set(panel_groups.keys())

            # Validate groups exist in data
            if self.groups is not None:
                if len(available_groups) == 0:
                    # No panel data, but user specified groups
                    raise ValueError(
                        f"groups specified but data contains no panel groups. "
                        f"Data has only global columns: {sorted(set(y_train.columns) - {'time'})}"
                    )
                requested_groups = set(self.groups)
                missing_groups = requested_groups - available_groups
                if missing_groups:
                    raise ValueError(
                        f"Requested groups {sorted(missing_groups)} not found in data. "
                        f"Available groups: {sorted(available_groups)}"
                    )

            # Validate component_names exist in data
            if self.component_names is not None:
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

                requested_components = set(self.component_names)
                missing_components = requested_components - available_components
                if missing_components:
                    raise ValueError(
                        f"Requested component_names {sorted(missing_components)} "
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

    def _apply_step_filter(
        self,
        y_truth: pl.DataFrame,
        y_pred: pl.DataFrame,
        context: ScoringContext,
    ) -> tuple[pl.DataFrame, pl.DataFrame, ScoringContext]:
        """Filter rows to the requested ``forecasting_steps``.

        When ``self.forecasting_steps`` is ``None`` or
        ``context.forecasting_step`` is unavailable, returns the inputs
        unchanged.

        """
        if self.forecasting_steps is None or context.forecasting_step is None:
            return y_truth, y_pred, context

        mask = context.forecasting_step.is_in(self.forecasting_steps)

        from yohou.metrics._context import ScoringContext as _ScoringContext  # noqa: PLC0415

        return (
            y_truth.filter(mask),
            y_pred.filter(mask),
            _ScoringContext(
                time_values=[t for t, m in zip(context.time_values, mask.to_list(), strict=True) if m],
                observed_time=(context.observed_time.filter(mask) if context.observed_time is not None else None),
                forecasting_step=context.forecasting_step.filter(mask),
            ),
        )

    def _group_by_context_dim(
        self,
        raw_scores: pl.DataFrame,
        context: ScoringContext | None,
        dim_name: str,
    ) -> tuple[pl.DataFrame, str | None]:
        """Group rows by a ScoringContext dimension and average within groups.

        When collapsing the *grouped* dimension into a scalar, applies
        ``self.step_weight`` (for ``forecasting_step``) as a weighted mean.

        Parameters
        ----------
        raw_scores : pl.DataFrame
            Per-row per-component scores.
        context : ScoringContext or None
            Scoring context carrying dimension values.
        dim_name : str
            Attribute name on context (``"forecasting_step"`` or ``"observed_time"``).

        Returns
        -------
        tuple of (pl.DataFrame, str or None)
            Grouped DataFrame (with *dim_name* column) and the label column
            name, or ``(collapsed, None)`` when the dimension is unavailable
            (fallback: collapse all rows).

        """
        if context is None:
            return raw_scores.select(pl.all().mean()), None

        dim_values = getattr(context, dim_name, None)
        if dim_values is None:
            # Dimension unavailable -> collapse all rows
            return raw_scores.select(pl.all().mean()), None

        grouped = (
            raw_scores
            .with_columns(pl.Series(dim_name, dim_values))
            .group_by(dim_name, maintain_order=True)
            .agg(pl.all().mean())
            .sort(dim_name)
        )
        return grouped, dim_name

    @staticmethod
    def _weighted_col_mean(df: pl.DataFrame, weights: dict[str, float]) -> float:
        """Weighted mean across columns of *df*.

        Parameters
        ----------
        df : pl.DataFrame
            Single-row or multi-row DataFrame whose columns are components.
        weights : dict[str, float]
            Maps column names to weights.  Missing columns get weight 1.0.

        Returns
        -------
        float

        """
        cols = df.columns
        w = np.array([weights.get(c, 1.0) for c in cols])
        total = w.sum()
        arr = df.to_numpy()  # shape (n_rows, n_cols)
        return float(np.nanmean(np.average(arr, axis=1, weights=w / total)))

    def _row_weights_from_steps(
        self,
        context: ScoringContext | None,
        n_rows: int,
    ) -> np.ndarray | None:
        """Build per-row weight array from ``step_weight`` and context.

        Returns ``None`` when ``step_weight`` is ``None`` or when
        ``context.forecasting_step`` is unavailable, meaning equal weights.
        """
        sw = self.step_weight
        if sw is None or context is None:
            return None
        steps = getattr(context, "forecasting_step", None)
        if steps is None:
            return None
        rw = np.array([sw.get(int(s), 1.0) for s in steps.to_list()])
        if rw.sum() == 0:
            return None
        return rw

    def _row_weights_from_vintages(
        self,
        context: ScoringContext | None,
        n_rows: int,
    ) -> np.ndarray | None:
        """Build per-row weight array from ``vintage_weight`` and context.

        Returns ``None`` when ``vintage_weight`` is ``None`` or when
        ``context.observed_time`` is unavailable, meaning equal weights.
        """
        vw = self.vintage_weight
        if vw is None or context is None:
            return None
        obs = getattr(context, "observed_time", None)
        if obs is None:
            return None
        rw = np.array([vw.get(v, 1.0) for v in obs.to_list()])
        if rw.sum() == 0:
            return None
        return rw

    def _combine_row_weights(
        self,
        context: ScoringContext | None,
        n_rows: int,
    ) -> np.ndarray | None:
        """Combine step_weight and vintage_weight into a single per-row array.

        Multiplicative combination: if both are set, element-wise product.
        """
        sw = self._row_weights_from_steps(context, n_rows)
        vw = self._row_weights_from_vintages(context, n_rows)
        if sw is None and vw is None:
            return None
        if sw is None:
            return vw
        if vw is None:
            return sw
        combined = sw * vw
        if combined.sum() == 0:
            return None
        return combined

    def _scalar_from_df(
        self,
        df: pl.DataFrame,
        cw: dict[str, float] | None,
        rw: np.ndarray | None,
    ) -> float:
        """Compute a single scalar from a DataFrame using optional weights."""
        arr = df.to_numpy()
        if cw is not None:
            col_w = np.array([cw.get(c, 1.0) for c in df.columns])
            row_means = np.average(arr, axis=1, weights=col_w)
        else:
            row_means = np.nanmean(arr, axis=1)
        if rw is not None:
            return float(np.average(row_means, weights=rw))
        return float(np.nanmean(row_means))

    def _componentwise_reduce(
        self,
        raw_scores: pl.DataFrame,
        row_label: str | None = None,
        time_values: list | None = None,
    ) -> float | pl.DataFrame:
        """Collapse component columns, keeping rows indexed by *row_label*.

        Uses ``self.component_weight`` (if set) for a weighted mean across
        components instead of equal weighting.

        Parameters
        ----------
        raw_scores : pl.DataFrame
            Score DataFrame whose columns are components (plus optionally a
            label column such as ``"forecasting_step"`` or ``"observed_time"``).
        row_label : str or None
            Name of the row-label column already present in *raw_scores*,
            or ``None`` when all rows were already collapsed.
        time_values : list or None
            Original time values to use as row label when *row_label* is
            ``None`` and the data still has multiple rows.

        Returns
        -------
        float or pl.DataFrame

        """
        cw = self.component_weight

        # Separate label from value columns
        if row_label is not None and row_label in raw_scores.columns:
            labels = raw_scores[row_label].to_list()
            value_df = raw_scores.drop(row_label)
        elif time_values is not None:
            labels = time_values
            row_label = "time"
            value_df = raw_scores
        else:
            # No label -> scalar
            if cw is not None:
                return self._weighted_col_mean(raw_scores, cw)
            return float(np.nanmean(raw_scores.to_numpy()))

        _, panel_groups = inspect_panel(value_df)

        if len(panel_groups) > 0:
            step_data: dict[str, list] = {row_label: labels}
            for group_name, group_cols in panel_groups.items():
                group_scores = []
                for i in range(len(value_df)):
                    step_errors = [float(value_df[col][i]) for col in group_cols]
                    if cw is not None:
                        unprefixed = [col.split("__", 1)[1] for col in group_cols]
                        weights = [cw.get(n, 1.0) for n in unprefixed]
                        total = sum(weights)
                        group_scores.append(sum(e * w for e, w in zip(step_errors, weights, strict=True)) / total)
                    else:
                        group_scores.append(float(np.mean(step_errors)))
                step_data[f"{group_name}__score"] = group_scores
            return pl.DataFrame(step_data)
        else:
            step_scores = []
            cols = value_df.columns
            if cw is not None:
                weights = [cw.get(c, 1.0) for c in cols]
                total = sum(weights)
                for i in range(len(value_df)):
                    vals = [float(value_df[c][i]) for c in cols]
                    step_scores.append(sum(v * w for v, w in zip(vals, weights, strict=True)) / total)
            else:
                for i in range(len(value_df)):
                    step_errors = [float(value_df[col][i]) for col in cols]
                    step_scores.append(float(np.mean(step_errors)))
            return pl.DataFrame({row_label: labels, "score": step_scores})

    def _aggregate_scores(
        self, raw_scores: pl.DataFrame, context: ScoringContext | None = None
    ) -> float | pl.DataFrame:
        """Apply aggregation strategy to raw per-timestep per-component scores.

        Handles both spatial dimensions (steps, vintages, components, groups)
        and the interval-specific coverage-rate dimension.

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
        agg_methods = self._normalize_agg_methods(
            self.aggregation_method,  # ty: ignore[unresolved-attribute]
            include_coveragewise=has_coverage_rate,
        )

        if has_coverage_rate:
            return self._aggregate_with_rates(raw_scores, context, agg_methods)

        return self._aggregate_spatial(raw_scores, context, agg_methods)

    def _aggregate_spatial(
        self,
        raw_scores: pl.DataFrame,
        context: ScoringContext | None,
        agg_methods: set[str],
    ) -> float | pl.DataFrame:
        """Apply spatial aggregation modes (all except coveragewise).

        Parameters
        ----------
        raw_scores : pl.DataFrame
            Per-timestep per-component score DataFrame.
        context : ScoringContext or None
            Scoring context with time values and metadata.
        agg_methods : set of str
            Normalized set of aggregation modes to apply.

        Returns
        -------
        float or pl.DataFrame

        """
        time_values = context.time_values if context is not None else None

        collapse_steps = "stepwise" in agg_methods
        collapse_vintages = "vintagewise" in agg_methods
        collapse_cols = "componentwise" in agg_methods
        collapse_groups = "groupwise" in agg_methods

        # Both time-dimensions collapsed -> all rows merge.
        collapse_all_rows = collapse_steps and collapse_vintages

        # Full scalar: all rows + all cols (+ groups)
        if collapse_all_rows and collapse_cols:
            if collapse_groups:
                _, panel_groups = inspect_panel(raw_scores)

                if len(panel_groups) > 0:
                    col_means = raw_scores.mean()
                    cw = self.component_weight
                    group_scores = {}
                    for group_name, group_cols in panel_groups.items():
                        vals = [float(col_means[col][0]) for col in group_cols]
                        if cw is not None:
                            unprefixed = [c.split("__", 1)[1] for c in group_cols]
                            weights = [cw.get(n, 1.0) for n in unprefixed]
                            total = sum(weights)
                            group_scores[group_name] = sum(v * w for v, w in zip(vals, weights, strict=True)) / total
                        else:
                            group_scores[group_name] = float(np.mean(vals))
                    return self._apply_panel_weights(group_scores, list(panel_groups.keys()))

            cw = self.component_weight
            rw = self._combine_row_weights(context, len(raw_scores))
            return self._scalar_from_df(raw_scores, cw, rw)

        # Groupwise pre-processing
        if collapse_groups:
            raw_scores = self._aggregate_groupwise(raw_scores)

        # Row reduction
        if collapse_all_rows:
            # All rows -> single mean row (per-component)
            rw = self._combine_row_weights(context, len(raw_scores))
            if rw is not None:
                arr = raw_scores.to_numpy()
                weighted = np.average(arr, axis=0, weights=rw)
                return pl.DataFrame([weighted.tolist()], schema=raw_scores.schema, orient="row")
            return raw_scores.select(pl.all().mean())

        # Partial row reduction: collapse ONE time dimension, keep the other.
        row_label: str | None = None
        if collapse_vintages and not collapse_steps:
            # Collapse vintages -> per-step output
            raw_scores, row_label = self._group_by_context_dim(raw_scores, context, "forecasting_step")
        elif collapse_steps and not collapse_vintages:
            # Collapse steps -> per-vintage output
            raw_scores, row_label = self._group_by_context_dim(raw_scores, context, "observed_time")

        # Column reduction (componentwise)
        if collapse_cols:
            return self._componentwise_reduce(raw_scores, row_label=row_label, time_values=time_values)

        # Groups only (no row/col reduction)
        if collapse_groups and row_label is None:
            result_data: dict[str, list] = {"time": time_values} if time_values is not None else {}
            for col in raw_scores.columns:
                result_data[col] = raw_scores[col].to_list()
            return pl.DataFrame(result_data)

        # Partial row reduction only (no componentwise)
        if row_label is not None:
            return raw_scores

        # No aggregation
        return raw_scores

    def _aggregate_with_rates(
        self,
        raw_scores: pl.DataFrame,
        context: ScoringContext | None,
        agg_methods: set[str],
    ) -> float | pl.DataFrame:
        """Handle coverage_rate dimension then delegate spatial aggregation.

        Splits the DataFrame by ``coverage_rate``, applies spatial aggregation
        to each rate slice, then optionally collapses the rate dimension.

        Parameters
        ----------
        raw_scores : pl.DataFrame
            Flat DataFrame with component columns plus ``coverage_rate``.
        context : ScoringContext or None
            Scoring context with time values and metadata.
        agg_methods : set of str
            Normalized set of aggregation modes including possibly ``"coveragewise"``.

        Returns
        -------
        float or pl.DataFrame

        """
        aggregate_rates = "coveragewise" in agg_methods
        spatial_modes = agg_methods - {"coveragewise"}

        coverage_rates = raw_scores["coverage_rate"].unique().sort().to_list()
        value_cols = [c for c in raw_scores.columns if c != "coverage_rate"]

        if len(coverage_rates) == 0:
            return float("nan")

        # Apply spatial aggregation per rate
        per_rate_results: dict[float, float | pl.DataFrame] = {}
        effective_spatial = (
            spatial_modes if spatial_modes else {"stepwise", "vintagewise", "componentwise", "groupwise"}
        )
        for rate in coverage_rates:
            rate_df = raw_scores.filter(pl.col("coverage_rate") == rate).select(value_cols)
            per_rate_results[rate] = self._aggregate_spatial(rate_df, context, effective_spatial)

        # All rates produced scalars -> collapse or return per-rate
        first_result = next(iter(per_rate_results.values()))
        if isinstance(first_result, float):
            if aggregate_rates:
                cw = getattr(self, "coverage_weight", None)
                values = list(per_rate_results.values())
                if cw is not None:
                    weights = [cw.get(r, 1.0) for r in per_rate_results]
                    return float(np.average(values, weights=weights))  # type: ignore
                return float(np.mean(values))  # type: ignore
            # Single rate: just return the scalar
            if len(per_rate_results) == 1:
                return next(iter(per_rate_results.values()))
            # Multiple rates without coveragewise: return DataFrame with per-rate rows
            return pl.DataFrame({
                "coverage_rate": list(per_rate_results.keys()),
                "score": list(per_rate_results.values()),
            })

        # Results are DataFrames - stack with coverage_rate column
        frames = []
        for rate, result_df in per_rate_results.items():
            frames.append(result_df.with_columns(pl.lit(rate).alias("coverage_rate")))  # type: ignore
        combined = pl.concat(frames)

        if aggregate_rates:
            # Collapse coverage_rate dimension: weighted average across rates
            rate_col = "coverage_rate"
            other_cols = [c for c in combined.columns if c != rate_col]
            # Find dimension columns (non-numeric) vs value columns
            dim_cols = [c for c in other_cols if combined[c].dtype not in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
            val_cols = [c for c in other_cols if c not in dim_cols]

            cw = getattr(self, "coverage_weight", None)
            if cw is not None:
                # Add weight column based on coverage_weight dict
                combined = combined.with_columns(
                    pl
                    .col("coverage_rate")
                    .replace_strict(
                        {r: cw.get(r, 1.0) for r in combined["coverage_rate"].unique().to_list()},
                        default=1.0,
                    )
                    .alias("_cw")
                )
                # Weighted mean per group
                if dim_cols:
                    combined = combined.group_by(dim_cols, maintain_order=True).agg([
                        (pl.col(c) * pl.col("_cw")).sum() / pl.col("_cw").sum() for c in val_cols
                    ])
                else:
                    total_w = combined["_cw"].sum()
                    combined = combined.select([(pl.col(c) * pl.col("_cw")).sum() / total_w for c in val_cols])
            elif dim_cols:
                combined = combined.group_by(dim_cols, maintain_order=True).agg([pl.col(c).mean() for c in val_cols])
            else:
                combined = combined.select([pl.col(c).mean() for c in val_cols])
            return combined

        return combined

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
            Predicted time series to evaluate.  Must have ``"observed_time"``
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
        Example outputs:
        - ["stepwise", "vintagewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["stepwise", "vintagewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    groups : list of str or None, default=None
        List of panel group names to include in scoring. If None, all panel groups
        are included. Only applicable for panel data. Validated at fit time.
    component_names : list of str or None, default=None
        List of component (target column) names to include in scoring. If None, all
        components are included. For panel data, these are unprefixed column names.
        Validated at fit time.
    group_weight : dict or None, default=None
        Dictionary mapping panel group names to weights for weighted aggregation.
        If None, all panel groups weighted equally. Only applicable for panel data.

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
        groups: list[str] | None = None,
        component_names: list[str] | None = None,
        group_weight: dict[str, float] | None = None,
        forecasting_steps: list[int] | None = None,
        component_weight: dict[str, float] | None = None,
        step_weight: dict[int, float] | None = None,
        vintage_weight: dict[datetime, float] | None = None,
    ):
        super().__init__(
            groups=groups,
            component_names=component_names,
            group_weight=group_weight,
            forecasting_steps=forecasting_steps,
            component_weight=component_weight,
            step_weight=step_weight,
            vintage_weight=vintage_weight,
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
        time_weight: Callable | pl.DataFrame | None = None,
        **params,
    ) -> float | pl.DataFrame:
        """Compute the point metric score.

        Template method: validate -> compute raw errors -> apply time
        weights -> aggregate -> post-aggregate transform -> rename columns.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True values with ``"time"`` column.
        y_pred : pl.DataFrame
            Predicted values with ``"time"`` column.
        time_weight : callable, pl.DataFrame, or None, default=None
            Time-based evaluation weights.
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

        # 0. Apply forecasting_steps filter
        y_truth, y_pred, context = self._apply_step_filter(y_truth, y_pred, context)

        # 1. Compute raw per-timestep per-component errors
        scores = self._compute_raw_errors(y_truth, y_pred)

        # 2. Apply time weights if provided
        if time_weight is not None:
            _, panel_groups = inspect_panel(scores)
            time_values = context.time_values

            if len(panel_groups) > 0:
                weighted_parts = []
                for group_name, group_cols in panel_groups.items():
                    group_scores = scores.select(group_cols)
                    weighted_group = self._process_time_weights(group_scores, time_weight, time_values, group_name)
                    weighted_parts.append(weighted_group)
                scores = pl.concat(weighted_parts, how="horizontal")
            else:
                scores = self._process_time_weights(scores, time_weight, time_values, group_name=None)

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

        Example outputs:

        - ``["stepwise", "vintagewise"]``: Per-component (and per-group) DataFrame.
        - ``"componentwise"`` or ``["componentwise"]``: Per-timestep (and per-group) DataFrame.
        - ``"groupwise"`` or ``["groupwise"]``: Per-component per-timestep DataFrame (panel aggregated).
        - ``["stepwise", "vintagewise", "componentwise"]``: Scalar (global) or per-group DataFrame (panel).
        - ``"all"``: Scalar float (hierarchically aggregated for panel data).
    groups : list of str or None, default=None
        List of panel group names to include in scoring. If None, all panel groups
        are included. Only applicable for panel data. Validated at fit time.
    component_names : list of str or None, default=None
        List of component (target column) names to include in scoring. If None, all
        components are included. For panel data, these are unprefixed column names.
        Validated at fit time.
    coverage_rates : list of float or None, default=None
        List of coverage rates to include in scoring. If None, all coverage rates
        are included. Rates are validated against actual prediction columns during scoring.
    group_weight : dict or None, default=None
        Weights for panel groups. See BaseScorer for details.
    coverage_weight : dict or None, default=None
        Dictionary mapping coverage rates (float) to weights for weighted
        aggregation when collapsing the coverage dimension. If None, all
        rates are weighted equally. Missing keys default to weight 1.0.

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
        "coverage_rates": [list, None],
        "coverage_weight": [dict, None],
    }

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        coverage_rates: list[float] | None = None,
        groups: list[str] | None = None,
        component_names: list[str] | None = None,
        group_weight: dict[str, float] | None = None,
        coverage_weight: dict[float, float] | None = None,
        forecasting_steps: list[int] | None = None,
        component_weight: dict[str, float] | None = None,
        step_weight: dict[int, float] | None = None,
        vintage_weight: dict[datetime, float] | None = None,
    ):
        super().__init__(
            groups=groups,
            component_names=component_names,
            group_weight=group_weight,
            forecasting_steps=forecasting_steps,
            component_weight=component_weight,
            step_weight=step_weight,
            vintage_weight=vintage_weight,
        )
        self.aggregation_method = aggregation_method
        self.coverage_rates = coverage_rates
        self.coverage_weight = coverage_weight

    def _validate_coverage_rates(self) -> None:
        """Validate coverage_rates parameter.

        Raises
        ------
        ValueError
            If coverage_rates validation fails.
        TypeError
            If coverage_rates contains non-hashable types.

        """
        if self.coverage_rates is not None:
            if not isinstance(self.coverage_rates, list):
                raise ValueError(f"coverage_rates must be a list or None, got {type(self.coverage_rates)}")
            if len(self.coverage_rates) == 0:
                raise ValueError("coverage_rates cannot be an empty list")

            # Check for hashable types (catch lists, dicts, etc.)
            for i, rate in enumerate(self.coverage_rates):
                try:
                    hash(rate)
                except TypeError:
                    raise TypeError(
                        f"coverage_rates[{i}] is not hashable (got {type(rate).__name__}). "
                        f"All elements must be numeric (int or float)."
                    ) from None

            # Check all elements are numeric
            if not all(isinstance(rate, int | float) for rate in self.coverage_rates):
                raise ValueError(
                    f"All elements in coverage_rates must be numeric (int or float), "
                    f"got types: {[type(r).__name__ for r in self.coverage_rates]}"
                )

            # Check range
            for rate in self.coverage_rates:
                if not 0 < rate < 1:
                    raise ValueError(f"All coverage_rates must be between 0 and 1 (exclusive), got {rate}")

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
        **params,
    ) -> float | pl.DataFrame:
        """Compute the interval metric score.

        Template method: validate -> extract rates/columns -> compute raw
        scores -> aggregate -> rename columns.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True values with ``"time"`` column.
        y_pred : pl.DataFrame
            Predicted intervals with ``"time"`` column.
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

        # 0. Apply forecasting_steps filter
        y_truth, y_pred, context = self._apply_step_filter(y_truth, y_pred, context)

        coverage_rates = self._extract_coverage_rates(y_pred)
        target_columns = self._extract_target_columns(y_truth)

        # 1. Compute raw per-timestep per-component per-rate scores (flat DataFrame)
        raw_scores = self._compute_raw_scores(y_truth, y_pred, coverage_rates, target_columns)

        # 2. Aggregate
        result = self._aggregate_scores(raw_scores, context=context)

        # 3. Rename columns
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
        Example outputs:
        - ["stepwise", "vintagewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["stepwise", "vintagewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    groups : list of str or None, default=None
        List of panel group names to include in scoring. If None, all panel groups
        are included. Only applicable for panel data. Validated at fit time.
    component_names : list of str or None, default=None
        List of component (target column) names to include in scoring. If None, all
        components are included. For panel data, these are unprefixed column names.
        Validated at fit time.
    group_weight : dict or None, default=None
        Weights for panel groups. See `BaseScorer` for details.

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
        groups: list[str] | None = None,
        component_names: list[str] | None = None,
        group_weight: dict[str, float] | None = None,
        forecasting_steps: list[int] | None = None,
        component_weight: dict[str, float] | None = None,
        step_weight: dict[int, float] | None = None,
        vintage_weight: dict[datetime, float] | None = None,
    ):
        super().__init__(
            groups=groups,
            component_names=component_names,
            group_weight=group_weight,
            forecasting_steps=forecasting_steps,
            component_weight=component_weight,
            step_weight=step_weight,
            vintage_weight=vintage_weight,
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
        time_weight: Callable | pl.DataFrame | None = None,
        **params,
    ) -> float | pl.DataFrame:
        """Compute the class-probability metric score.

        Template method: validate -> compute raw errors -> apply time
        weights -> aggregate -> rename columns.

        Parameters
        ----------
        y_truth : pl.DataFrame
            True class labels with ``"time"`` column.
        y_pred : pl.DataFrame
            Predicted probabilities with ``"time"`` column.
        time_weight : callable, pl.DataFrame, or None, default=None
            Time-based evaluation weights.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        float or pl.DataFrame
            Aggregated metric score.

        """
        check_is_fitted(self, ["_is_fitted"])

        y_truth, y_pred, context = validate_scorer_data(self, y_truth, y_pred)

        # 0. Apply forecasting_steps filter
        y_truth, y_pred, context = self._apply_step_filter(y_truth, y_pred, context)

        # 0b. Validate probability columns are finite and in [0, 1]
        self._validate_probabilities(y_truth, y_pred)

        # 1. Compute raw per-timestep per-component errors
        scores = self._compute_raw_errors(y_truth, y_pred)

        # 2. Apply time weights if provided
        if time_weight is not None:
            _, panel_groups = inspect_panel(scores)
            time_values = context.time_values

            if len(panel_groups) > 0:
                weighted_parts = []
                for group_name, group_cols in panel_groups.items():
                    group_scores = scores.select(group_cols)
                    weighted_group = self._process_time_weights(group_scores, time_weight, time_values, group_name)
                    weighted_parts.append(weighted_group)
                scores = pl.concat(weighted_parts, how="horizontal")
            else:
                scores = self._process_time_weights(scores, time_weight, time_values, group_name=None)

        # 3. Aggregate
        result = self._aggregate_scores(scores, context=context)

        # 4. Rename columns
        if isinstance(result, pl.DataFrame):
            result = self._rename_metric_columns(result)

        return result
