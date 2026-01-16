"""Base classes for forecasting metrics and scoring functions."""

import abc

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, _fit_context
from sklearn.utils._param_validation import StrOptions

from yohou.utils import Tags


class BaseScorer(BaseEstimator, metaclass=abc.ABCMeta):
    """Base class for all forecasting metrics.

    Defines the interface for scoring forecast quality. All scorers must implement
    the :meth:`score` method and can optionally override :meth:`fit` for metrics
    that require training data statistics.

    Parameters
    ----------
    panel_group_weights : dict or None, default=None
        Dictionary mapping panel group names to weights for weighted aggregation.
        If None, all panel groups weighted equally. Only applicable for panel data.

    """

    _parameter_constraints: dict = {
        "panel_group_weights": [dict, None],
    }

    def __init__(
        self,
        panel_group_weights: dict[str, float] | None = None,
    ):
        self.panel_group_weights = panel_group_weights

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
        tags.scorer_tags.requires_calibration = False

        return tags

    def _validate_inputs(
        self, y_truth: pl.DataFrame, y_pred: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Align ground truth and predictions by matching time indices.

        Ensures that predictions and actuals are properly aligned. Subclasses
        override this to add prediction-type-specific validation.

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth values with "time" column.

        y_pred : pl.DataFrame
            Predicted values with "observed_time" and "time" columns.

        Returns
        -------
        y_truth : pl.DataFrame
            Aligned ground truth with time column removed.

        y_pred : pl.DataFrame
            Aligned predictions with time columns removed.

        Raises
        ------
        ValueError
            If validation fails.

        """
        # Align by time
        y_truth = y_truth.join(y_pred[["time"]], on="time")
        y_pred = y_pred.filter(pl.col("time").is_in(y_truth["time"].implode()))

        # Store aligned time values as instance attribute for later use in aggregate modes
        self._time_values_ = y_pred["time"].to_list()

        y_truth = y_truth.drop("time")
        y_pred = y_pred.drop("observed_time", "time")

        return y_truth, y_pred

    @abc.abstractmethod
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y_train: pl.DataFrame | None, **params) -> "BaseScorer":
        """Fit the scorer on training data if needed."""
        ...

    def _apply_panel_weights(self, scores: dict[str, float], panel_group_names: list[str]) -> float:
        """Apply panel group weights to aggregate scores.

        Parameters
        ----------
        scores : dict[str, float]
            Mapping of panel group names to their scores.

        panel_group_names : list[str]
            List of panel group names present in data.

        Returns
        -------
        float
            Weighted average score.

        """
        if self.panel_group_weights is None:
            # Equal weighting
            return float(np.mean(list(scores.values())))

        # Apply custom weights
        weighted_sum = 0.0
        total_weight = 0.0

        for group in panel_group_names:
            if group in scores:
                weight = self.panel_group_weights.get(group, 1.0)
                weighted_sum += scores[group] * weight
                total_weight += weight

        if total_weight == 0:
            raise ValueError("Total panel group weight is zero")

        return weighted_sum / total_weight

    @abc.abstractmethod
    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, **params) -> pl.DataFrame:
        """Compute the metric score.

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth values with "time" column.

        y_pred : pl.DataFrame
            Predicted values with "observed_time" and "time" columns.

        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Metric value dataframe. Lower is better for error metrics.

        """

    def __call__(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, **params) -> pl.DataFrame:
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
        pl.DataFrame
            Metric score dataframe.

        """
        return self.score(y_truth, y_pred, **params)


class BasePointScorer(BaseScorer, metaclass=abc.ABCMeta):
    """Base class for point forecast metrics.

    Point forecasters produce single-value predictions. Metrics derived from this
    class evaluate prediction accuracy (e.g., MeanAbsoluteError, RootMeanSquaredError, MAPE).

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. Options:
        - "timewise": Aggregate across time, return per-component DataFrame
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["timewise", "componentwise", "groupwise"].
        Example outputs:
        - "timewise" or ["timewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["timewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    panel_group_weights : dict or None, default=None
        Weights for panel groups. See BaseScorer for details.

    See Also
    --------
    :mod:`yohou.metrics.point` : Concrete implementations (MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError, MAPE)
    :class:`yohou.point_forecaster.base.BasePointForecaster` : Produces point forecasts

    """

    _parameter_constraints: dict = {
        **BaseScorer._parameter_constraints,
        "aggregation_method": [list, StrOptions({"all", "timewise", "componentwise", "groupwise"})],
    }

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        panel_group_weights: dict[str, float] | None = None,
    ):
        super().__init__(panel_group_weights=panel_group_weights)
        self.aggregation_method = aggregation_method

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y_train: pl.DataFrame | None, **params) -> "BasePointScorer":
        """Fit the scorer on training data if needed."""
        # Validate list elements
        if isinstance(self.aggregation_method, list):
            valid_methods = {"timewise", "componentwise", "groupwise"}
            for method in self.aggregation_method:
                if method not in valid_methods:
                    raise ValueError(
                        f"Invalid aggregation_method '{method}' in list. "
                        f"Valid list elements are: {valid_methods}"
                    )
            if len(self.aggregation_method) == 0:
                raise ValueError(
                    "aggregation_method list cannot be empty. "
                    "Use 'all' or provide at least one method: "
                    "{'timewise', 'componentwise', 'groupwise'}"
                )
        return super().fit(y_train, **params)

    def _validate_inputs(
        self, y_truth: pl.DataFrame, y_pred: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Validate point prediction inputs.

        Validates panel data consistency for point predictions where column names
        match directly between y_truth and y_pred.

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth values with "time" column.

        y_pred : pl.DataFrame
            Predicted values with "observed_time" and "time" columns.

        Returns
        -------
        y_truth : pl.DataFrame
            Aligned ground truth with time column removed.

        y_pred : pl.DataFrame
            Aligned predictions with time columns removed.

        Raises
        ------
        ValueError
            If panel data validation fails.

        """
        from yohou.utils.panel import inspect_locality
        from yohou.utils.validation import check_schema

        # Check if data is panel (has prefixed columns)
        _, y_truth_groups = inspect_locality(y_truth)
        _, y_pred_groups = inspect_locality(y_pred)

        is_panel = len(y_truth_groups) > 0 or len(y_pred_groups) > 0

        if is_panel:
            # Validate panel data consistency
            y_truth_group_names = set(y_truth_groups.keys())
            y_pred_group_names = set(y_pred_groups.keys())

            # Check that all y_pred groups exist in y_truth
            missing_groups = y_pred_group_names - y_truth_group_names
            if missing_groups:
                raise ValueError(
                    f"Prediction contains panel groups not in ground truth: {missing_groups}. "
                    f"y_truth groups: {y_truth_group_names}, y_pred groups: {y_pred_group_names}"
                )

            # For point predictions, column names match directly
            for group_name in y_pred_group_names:
                # Extract unprefixed column names (suffixes after __)
                y_truth_cols = sorted([col.split("__", 1)[1] for col in y_truth_groups[group_name]])
                y_pred_cols = sorted([col.split("__", 1)[1] for col in y_pred_groups[group_name]])

                if y_truth_cols != y_pred_cols:
                    raise ValueError(
                        f"Column mismatch for panel group '{group_name}'. "
                        f"y_truth has: {y_truth_cols}, y_pred has: {y_pred_cols}"
                    )

            # Validate schemas match using check_schema utility
            # Build expected schema from y_truth structure
            for group_name in y_pred_group_names:
                # Extract schema for this panel group from y_truth (unprefixed column names)
                group_cols = {
                    col.split("__", 1)[1]: y_truth[col].dtype for col in y_truth_groups[group_name]
                }

                # Validate y_pred has matching schema for this group
                try:
                    check_schema(y_pred, group_cols, panel_group_names=[group_name])
                except ValueError as e:
                    raise ValueError(f"Schema mismatch for panel group '{group_name}': {e}") from e

        # Call parent for common validation (time alignment, etc.)
        return super()._validate_inputs(y_truth, y_pred)

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with scorer-specific attributes.

        """
        tags = super().__sklearn_tags__()
        tags.scorer_tags.prediction_type = "point"
        return tags

    def _aggregate_scores(self, raw_scores: pl.DataFrame) -> float | pl.DataFrame:
        """Apply aggregation strategy to raw per-timestep per-component scores.

        Parameters
        ----------
        raw_scores : pl.DataFrame
            DataFrame with shape (n_timesteps, n_components) containing raw scores.

        Returns
        -------
        float or pl.DataFrame
            Aggregated scores based on aggregation_method.

        """
        # Normalize aggregation_method to list
        if self.aggregation_method == "all":
            agg_methods = ["timewise", "componentwise", "groupwise"]
        elif isinstance(self.aggregation_method, str):
            agg_methods = [self.aggregation_method]
        else:
            agg_methods = self.aggregation_method

        # Apply aggregations in order
        if "timewise" in agg_methods and "componentwise" in agg_methods:
            # Check for groupwise aggregation on panel data
            if "groupwise" in agg_methods:
                from yohou.utils.panel import inspect_locality

                _, panel_groups = inspect_locality(raw_scores)

                if len(panel_groups) > 0:
                    # 1. Aggregate time (mean per column)
                    col_means = raw_scores.mean()

                    # 2. Aggregate components within groups
                    group_scores = {}
                    for group_name, group_cols in panel_groups.items():
                        # Extract float values for this group's columns
                        vals = [float(col_means[col][0]) for col in group_cols]
                        group_scores[group_name] = float(np.mean(vals))

                    # 3. Aggregate groups (weighted)
                    return self._apply_panel_weights(group_scores, list(panel_groups.keys()))

            # Aggregate both dimensions to scalar (flat aggregation)
            return float(np.nanmean(raw_scores.to_numpy()))
        elif "timewise" in agg_methods:
            # Aggregate across time, return per-component scores (1 row)
            return raw_scores.select(pl.all().mean())
        elif "componentwise" in agg_methods:
            # Aggregate across components, return per-timestep scores
            from yohou.utils.panel import inspect_locality

            time_values = self._time_values_
            _, panel_groups = inspect_locality(raw_scores)

            if len(panel_groups) > 0:
                # Panel data: Aggregate within each group separately
                step_data = {"time": time_values}

                for group_name, group_cols in panel_groups.items():
                    group_scores = []
                    for step in range(len(raw_scores)):
                        step_errors = [float(raw_scores[col][step]) for col in group_cols]
                        group_scores.append(float(np.mean(step_errors)))
                    step_data[f"{group_name}__score"] = group_scores

                return pl.DataFrame(step_data)
            else:
                # Global data: Aggregate across all components
                time_values = self._time_values_
                step_scores = []

                for step in range(len(raw_scores)):
                    step_errors = [float(raw_scores[col][step]) for col in raw_scores.columns]
                    step_scores.append(float(np.mean(step_errors)))

                return pl.DataFrame({"time": time_values, "score": step_scores})
        else:
            # No aggregation specified, return raw scores
            return raw_scores


class BaseIntervalScorer(BaseScorer, metaclass=abc.ABCMeta):
    """Base class for interval forecast metrics.

    Interval forecasters produce prediction intervals. Metrics derived from this
    class evaluate coverage and width trade-offs.

    Parameters
    ----------
    aggregation_method : list of str or str, default="all"
        Dimensions to aggregate over. Options:
        - "timewise": Aggregate across time, return per-component DataFrame
        - "componentwise": Aggregate across components, return per-timestep DataFrame
        - "groupwise": Aggregate across panel groups (panel data only)
        - "coveragewise": Aggregate across coverage rates (return average coverage)
        - "all": Aggregate across all dimensions (returns scalar). Same as
          ["timewise", "componentwise", "groupwise", "coveragewise"].
        Example outputs:
        - "timewise" or ["timewise"]: Per-component (and per-group) DataFrame.
        - "componentwise" or ["componentwise"]: Per-timestep (and per-group) DataFrame.
        - "groupwise" or ["groupwise"]: Per-component per-timestep DataFrame (panel aggregated).
        - ["timewise", "componentwise"]: Scalar (global) or per-group DataFrame (panel).
        - "all": Scalar float (hierarchically aggregated for panel data).
    panel_group_weights : dict or None, default=None
        Weights for panel groups. See BaseScorer for details.

    See Also
    --------
    :mod:`yohou.metrics.interval` : Concrete implementations
    :class:`yohou.interval_forecaster.base.BaseIntervalForecaster` : Produces intervals

    """

    _parameter_constraints: dict = {
        **BaseScorer._parameter_constraints,
        "aggregation_method": [
            list,
            StrOptions({"all", "timewise", "componentwise", "groupwise", "coveragewise"}),
        ],
    }

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        panel_group_weights: dict[str, float] | None = None,
    ):
        super().__init__(panel_group_weights=panel_group_weights)
        self.aggregation_method = aggregation_method

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y_train: pl.DataFrame | None, **params) -> "BaseIntervalScorer":
        """Fit the scorer on training data if needed."""
        # Validate list elements
        if isinstance(self.aggregation_method, list):
            valid_methods = {"timewise", "componentwise", "groupwise", "coveragewise"}
            for method in self.aggregation_method:
                if method not in valid_methods:
                    raise ValueError(
                        f"Invalid aggregation_method '{method}' in list. "
                        f"Valid list elements are: {valid_methods}"
                    )
            if len(self.aggregation_method) == 0:
                raise ValueError(
                    "aggregation_method list cannot be empty. "
                    "Use 'all' or provide at least one method: "
                    "{'timewise', 'componentwise', 'groupwise', 'coveragewise'}"
                )
        return super().fit(y_train, **params)

    def _validate_inputs(
        self, y_truth: pl.DataFrame, y_pred: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Validate interval prediction inputs.

        Validates panel data consistency for interval predictions where y_pred
        column names have _lower_* and _upper_* suffixes that need to be stripped
        before comparison with y_truth.

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth values with "time" column.
        y_pred : pl.DataFrame
            Predicted intervals with "observed_time" and "time" columns.

        Returns
        -------
        y_truth : pl.DataFrame
            Aligned ground truth with time column removed.
        y_pred : pl.DataFrame
            Aligned predictions with time columns removed.

        Raises
        ------
        ValueError
            If panel data validation fails.

        """
        import re

        from yohou.utils.panel import inspect_locality

        # Check if data is panel (has prefixed columns)
        _, y_truth_groups = inspect_locality(y_truth)
        _, y_pred_groups = inspect_locality(y_pred)

        is_panel = len(y_truth_groups) > 0 or len(y_pred_groups) > 0

        if is_panel:
            # Validate panel data consistency
            y_truth_group_names = set(y_truth_groups.keys())
            y_pred_group_names = set(y_pred_groups.keys())

            # Check that all y_pred groups exist in y_truth
            missing_groups = y_pred_group_names - y_truth_group_names
            if missing_groups:
                raise ValueError(
                    f"Prediction contains panel groups not in ground truth: {missing_groups}. "
                    f"y_truth groups: {y_truth_group_names}, y_pred groups: {y_pred_group_names}"
                )

            # For interval predictions, strip _lower_* and _upper_* suffixes
            for group_name in y_pred_group_names:
                # Extract unprefixed column names (suffixes after __)
                y_truth_cols = sorted([col.split("__", 1)[1] for col in y_truth_groups[group_name]])
                y_pred_cols_raw = sorted(
                    [col.split("__", 1)[1] for col in y_pred_groups[group_name]]
                )

                # Extract base column names from interval predictions
                # Pattern: col_lower_0.9 or col_upper_0.95 -> col
                y_pred_cols_base = set()
                for col in y_pred_cols_raw:
                    # Remove _lower_XXX or _upper_XXX suffixes
                    base_col = re.sub(r"_(lower|upper)_[\d.]+$", "", col)
                    y_pred_cols_base.add(base_col)

                y_pred_cols = sorted(y_pred_cols_base)

                if y_truth_cols != y_pred_cols:
                    raise ValueError(
                        f"Column mismatch for panel group '{group_name}'. "
                        f"y_truth has: {y_truth_cols}, y_pred has: {y_pred_cols}"
                    )

                # Validate schemas match
                # For interval predictions, manually validate dtypes since y_pred has _lower_*/_upper_* suffixes
                for truth_col_full in y_truth_groups[group_name]:
                    # Extract unprefixed column name
                    col = truth_col_full.split("__", 1)[1]
                    truth_dtype = y_truth[truth_col_full].dtype

                    # Check that all interval bounds have compatible types
                    for pred_col_full in y_pred_groups[group_name]:
                        # Extract unprefixed column name
                        pred_col = pred_col_full.split("__", 1)[1]
                        # Check if this is a bound for our target column
                        base_col = re.sub(r"_(lower|upper)_[\d.]+$", "", pred_col)
                        if base_col == col:
                            pred_dtype = y_pred[pred_col_full].dtype
                            if truth_dtype != pred_dtype:
                                raise ValueError(
                                    f"Schema mismatch for column '{pred_col_full}': "
                                    f"expected dtype {truth_dtype} (from y_truth), got {pred_dtype}"
                                )

        # Call parent for common validation (time alignment, etc.)
        return super()._validate_inputs(y_truth, y_pred)

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with scorer-specific attributes.

        """
        tags = super().__sklearn_tags__()
        tags.scorer_tags.prediction_type = "interval"
        return tags

    def _aggregate_scores(
        self,
        raw_scores: dict[float, pl.DataFrame],
    ) -> float | dict[float, float] | pl.DataFrame:
        """Apply aggregation strategy to raw per-timestep per-component per-rate scores.

        Parameters
        ----------
        raw_scores : dict[float, pl.DataFrame]
            Dict mapping coverage_rate -> DataFrame(n_timesteps, n_components) with raw scores.

        Returns
        -------
        float or dict or pl.DataFrame
            Aggregated scores based on aggregation_method.

        """
        # Normalize aggregation_method to set
        if self.aggregation_method == "all":
            # "all" implies aggregating over everything, including rates
            agg_methods = {"timewise", "componentwise", "groupwise", "coveragewise"}
        else:
            if isinstance(self.aggregation_method, str):
                agg_methods = {self.aggregation_method}
            else:
                agg_methods = set(self.aggregation_method)

        aggregate_rates = "coveragewise" in agg_methods

        if "timewise" in agg_methods and "componentwise" in agg_methods:
            # Aggregate across BOTH time and components to get scalar or dict

            # Check for groupwise aggregation on panel data
            use_groupwise = False
            if "groupwise" in agg_methods:
                from yohou.utils.panel import inspect_locality

                # Check locality on first rate's dataframe
                first_df = next(iter(raw_scores.values()))
                _, panel_groups = inspect_locality(first_df)
                if len(panel_groups) > 0:
                    use_groupwise = True

            if not aggregate_rates:
                # Return score per rate
                score_by_rate = {}
                for rate, rate_scores in raw_scores.items():
                    if use_groupwise:
                        # Hierarchical aggregation per rate
                        col_means = rate_scores.mean()
                        group_scores = {}
                        for group_name, group_cols in panel_groups.items():
                            vals = [float(col_means[col][0]) for col in group_cols]
                            group_scores[group_name] = float(np.mean(vals))
                        score_by_rate[rate] = self._apply_panel_weights(
                            group_scores, list(panel_groups.keys())
                        )
                    else:
                        # Flat aggregation per rate
                        score_by_rate[rate] = float(np.nanmean(rate_scores.to_numpy()))
                return score_by_rate
            else:
                # Return single scalar (average across all rates and dimensions)
                all_scores = []
                for rate_scores in raw_scores.values():
                    if use_groupwise:
                        # Hierarchical aggregation
                        col_means = rate_scores.mean()
                        group_scores = {}
                        for group_name, group_cols in panel_groups.items():
                            vals = [float(col_means[col][0]) for col in group_cols]
                            group_scores[group_name] = float(np.mean(vals))
                        all_scores.append(
                            self._apply_panel_weights(group_scores, list(panel_groups.keys()))
                        )
                    else:
                        # Flat aggregation
                        all_scores.append(float(np.nanmean(rate_scores.to_numpy())))
                return float(np.mean(all_scores))

        elif "componentwise" in agg_methods:
            # Aggregate across components, return per-timestep scores
            from yohou.utils.panel import inspect_locality

            time_values = self._time_values_
            step_data = {"time": time_values}

            n_steps = len(time_values)

            # Detect panel groups from first rate's data
            first_rate_df = next(iter(raw_scores.values()))
            _, panel_groups = inspect_locality(first_rate_df)
            is_panel = len(panel_groups) > 0

            if is_panel:
                if not aggregate_rates:
                    # Return separate columns for each group and rate
                    for rate, rate_scores in raw_scores.items():
                        for group_name, group_cols in panel_groups.items():
                            col_name = f"{group_name}__rate_{rate}"
                            step_scores = []
                            for step in range(n_steps):
                                step_values = [float(rate_scores[col][step]) for col in group_cols]
                                step_scores.append(float(np.mean(step_values)))
                            step_data[col_name] = step_scores
                else:
                    # Aggregate across rates, return separate columns for each group
                    for group_name, group_cols in panel_groups.items():
                        step_scores = []
                        for step in range(n_steps):
                            step_values = []
                            for rate_scores in raw_scores.values():
                                for col in group_cols:
                                    step_values.append(float(rate_scores[col][step]))
                            step_scores.append(float(np.mean(step_values)))
                        step_data[f"{group_name}__score"] = step_scores
            else:
                if not aggregate_rates:
                    # Add column for each rate
                    for rate, rate_scores in raw_scores.items():
                        rate_col_name = f"rate_{rate}"
                        step_scores = []

                        for step in range(n_steps):
                            step_values = [
                                float(rate_scores[col][step]) for col in rate_scores.columns
                            ]
                            step_scores.append(float(np.mean(step_values)))

                        step_data[rate_col_name] = step_scores
                else:
                    # Single aggregated score column
                    step_scores = []
                    for step in range(n_steps):
                        step_values = []
                        for rate_scores in raw_scores.values():
                            for col in rate_scores.columns:
                                step_values.append(float(rate_scores[col][step]))
                        step_scores.append(float(np.mean(step_values)))

                    step_data["score"] = step_scores

            return pl.DataFrame(step_data)

        elif "timewise" in agg_methods:
            # Aggregate across time, return per-component scores
            result_data = {}

            if not aggregate_rates:
                # One column per component per rate
                for rate, rate_scores in raw_scores.items():
                    rate_means = rate_scores.select(pl.all().mean())
                    for col in rate_means.columns:
                        result_data[f"{col}_rate_{rate}"] = [float(rate_means[col][0])]
            else:
                # Single row per component (aggregate across rates)
                all_components = set()
                for rate_scores in raw_scores.values():
                    all_components.update(rate_scores.columns)

                for col in sorted(all_components):
                    col_scores = []
                    for rate_scores in raw_scores.values():
                        if col in rate_scores.columns:
                            col_scores.append(float(rate_scores[col].mean()))
                    result_data[col] = [float(np.mean(col_scores))]

            return pl.DataFrame(result_data)

        else:
            # No aggregation or unhandled combination, aggregate both time and components to scalar/dict
            if not aggregate_rates:
                # Return score per rate
                score_by_rate = {}
                for rate, rate_scores in raw_scores.items():
                    score_by_rate[rate] = float(np.nanmean(rate_scores.to_numpy()))

                return score_by_rate

            else:
                # Return average score across all rates and columns
                all_scores = []
                for rate_scores in raw_scores.values():
                    all_scores.append(float(np.nanmean(rate_scores.to_numpy())))

                return float(np.mean(all_scores))

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
        import re

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


class BaseConformityScorer(BaseScorer, metaclass=abc.ABCMeta):
    """Base class for conformal prediction conformity scorers.

    Conformity scorers quantify how "unusual" a prediction is compared to the
    calibration set. Used in conformal prediction to construct valid prediction
    intervals with coverage guarantees.

    See Also
    --------
    :mod:`yohou.metrics.conformity` : Concrete conformity scorers
    :class:`yohou.interval_forecaster.split_conformal.SplitConformalForecaster` : Uses conformity
    scores

    """

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y_train: pl.DataFrame | None, **params) -> "BaseConformityScorer":
        """Fit the scorer on training data if needed."""
        # Conformity scorers typically don't aggregate results in the same way,
        # so they don't use aggregation_method, but they must implement fit.
        return super().fit(y_train, **params)

    @staticmethod
    def _compute_assymetric_quantiles(
        conformity_scores: pl.DataFrame, coverage_rate: float
    ) -> tuple[float, float]:
        """Compute lower and upper quantiles for asymmetric intervals.

        Parameters
        ----------
        conformity_scores : pl.DataFrame
            Conformity scores from calibration.

        coverage_rate : float
            Target coverage rate.

        Returns
        -------
        lower_quantile : float
            Lower quantile value.

        upper_quantile : float
            Upper quantile value.

        """
        lower_quantile: float = np.quantile(conformity_scores, coverage_rate / 2.0, method="lower")  # type: ignore[call-overload]

        upper_quantile: float = np.quantile(
            conformity_scores, 1 - coverage_rate / 2.0, method="upper"
        )  # type: ignore[call-overload]

        return lower_quantile, upper_quantile

    @staticmethod
    def _compute_symetric_quantiles(conformity_scores: pl.DataFrame, coverage_rate: float) -> float:
        """Compute quantile for symmetric intervals.

        Parameters
        ----------
        conformity_scores : pl.DataFrame
            Conformity scores from calibration.

        coverage_rate : float
            Target coverage rate.

        Returns
        -------
        float
            Quantile value for symmetric intervals.

        """
        quantile: float = np.quantile(conformity_scores, 1 - coverage_rate, method="lower")  # type: ignore[call-overload]

        return quantile

    @staticmethod
    def _format_y_pred_interval(
        lower_bound: pl.DataFrame, upper_bound: pl.DataFrame, coverage_rate: float
    ) -> pl.DataFrame:
        """Format lower and upper bounds into interval DataFrame.

        Parameters
        ----------
        lower_bound : pl.DataFrame
            Lower bound predictions.

        upper_bound : pl.DataFrame
            Upper bound predictions.

        coverage_rate : float
            Coverage rate for labeling columns.

        Returns
        -------
        pl.DataFrame
            Formatted prediction intervals.

        """
        lower_bound.columns = [f"{col}_lower_{coverage_rate}" for col in lower_bound.columns]
        upper_bound.columns = [f"{col}_upper_{coverage_rate}" for col in upper_bound.columns]

        y_pred_interval = pl.concat([lower_bound, upper_bound], how="horizontal")

        return y_pred_interval

    @abc.abstractmethod
    def inverse_score(
        self, y_pred: pl.DataFrame, conformity_scores: pl.DataFrame, coverage_rate: float
    ) -> pl.DataFrame:
        """Transform conformity scores into prediction intervals.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Point predictions.

        conformity_scores : pl.DataFrame
            Conformity scores from calibration.

        coverage_rate : float
            Target coverage probability.

        Returns
        -------
        pl.DataFrame
            Prediction intervals.

        """
