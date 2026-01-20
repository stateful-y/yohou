"""Base classes for forecasting metrics and scoring functions."""

import abc

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, _fit_context
from sklearn.utils._param_validation import StrOptions

from yohou.utils import Tags, validate_scorer_data


class BaseScorer(BaseEstimator, metaclass=abc.ABCMeta):
    """Base class for all forecasting metrics.

    Defines the interface for scoring forecast quality. All scorers must implement
    the :meth:`score` method and can optionally override :meth:`fit` for metrics
    that require training data statistics.

    Parameters
    ----------
    panel_group_names : list of str or None, default=None
        List of panel group names to include in scoring. If None, all panel groups
        are included. Only applicable for panel data.
    component_names : list of str or None, default=None
        List of component (target column) names to include in scoring. If None, all
        components are included. For panel data, these are unprefixed column names.
    panel_group_weight : dict or None, default=None
        Dictionary mapping panel group names to weights for weighted aggregation.
        If None, all panel groups weighted equally. Only applicable for panel data.

    """

    _parameter_constraints: dict = {
        "panel_group_names": [list, None],
        "component_names": [list, None],
        "panel_group_weight": [dict, None],
    }

    def __init__(
        self,
        panel_group_names: list[str] | None = None,
        component_names: list[str] | None = None,
        panel_group_weight: dict[str, float] | None = None,
    ):
        self.panel_group_names = panel_group_names
        self.component_names = component_names
        self.panel_group_weight = panel_group_weight

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

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y_train: pl.DataFrame, **params) -> "BaseScorer":
        """Fit the scorer on training data if needed.

        Validates panel_group_names and component_names against training data.
        Subclasses should override to add type-specific parameter validation.
        """
        # Validate base parameters (panel_group_names, component_names)
        self._validate_parameters(y_train=y_train)

        # Validate input structure without aligning (single dataframe)
        validate_scorer_data(self, y_true=y_train, y_pred=None, reset=True)

        # Mark as fitted
        self._is_fitted = True

        return self

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
        if self.panel_group_weight is None:
            # Equal weighting
            return float(np.mean(list(scores.values())))

        # Apply custom weights
        weighted_sum = 0.0
        total_weight = 0.0

        for group in panel_group_names:
            if group in scores:
                weight = self.panel_group_weight.get(group, 1.0)
                weighted_sum += scores[group] * weight
                total_weight += weight

        if total_weight == 0:
            raise ValueError("Total panel group weight is zero")

        return weighted_sum / total_weight

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
                raise ValueError(
                    "valid_aggregation_methods must be provided when validating aggregation_method"
                )

            # Handle single string
            if isinstance(aggregation_method, str):
                # "all" is a special value that means aggregate across all dimensions
                if (
                    aggregation_method != "all"
                    and aggregation_method not in valid_aggregation_methods
                ):
                    raise ValueError(
                        f"Invalid aggregation_method '{aggregation_method}'. "
                        f"Valid options are: 'all' or {sorted(valid_aggregation_methods)}"
                    )
            # Handle list
            elif isinstance(aggregation_method, list):
                # Check all elements are strings
                if not all(isinstance(method, str) for method in aggregation_method):
                    raise ValueError(
                        f"All elements in aggregation_method must be strings, got: {aggregation_method}"
                    )
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

        # Validate panel_group_names type
        if self.panel_group_names is not None:
            if not isinstance(self.panel_group_names, list):
                raise ValueError(
                    f"panel_group_names must be a list or None, got {type(self.panel_group_names)}"
                )
            if not all(isinstance(name, str) for name in self.panel_group_names):
                raise ValueError("All elements in panel_group_names must be strings")
            if len(self.panel_group_names) == 0:
                raise ValueError("panel_group_names cannot be an empty list")

        # Validate component_names type
        if self.component_names is not None:
            if not isinstance(self.component_names, list):
                raise ValueError(
                    f"component_names must be a list or None, got {type(self.component_names)}"
                )
            if not all(isinstance(name, str) for name in self.component_names):
                raise ValueError("All elements in component_names must be strings")
            if len(self.component_names) == 0:
                raise ValueError("component_names cannot be an empty list")

        # If y_train is provided, validate against actual data
        if y_train is not None:
            from yohou.utils.panel import inspect_locality

            _, panel_groups = inspect_locality(y_train)
            available_groups = set(panel_groups.keys())

            # Validate panel_group_names exist in data
            if self.panel_group_names is not None:
                if len(available_groups) == 0:
                    # No panel data, but user specified panel_group_names
                    raise ValueError(
                        f"panel_group_names specified but data contains no panel groups. "
                        f"Data has only global columns: {sorted(set(y_train.columns) - {'time'})}"
                    )
                requested_groups = set(self.panel_group_names)
                missing_groups = requested_groups - available_groups
                if missing_groups:
                    raise ValueError(
                        f"Requested panel_group_names {sorted(missing_groups)} not found in data. "
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

    @abc.abstractmethod
    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame, **params) -> pl.DataFrame:
        """Compute the metric score.

        Implementations should validate parameters before validating inputs.

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
    panel_group_names : list of str or None, default=None
        List of panel group names to include in scoring. If None, all panel groups
        are included. Only applicable for panel data. Validated at fit time.
    component_names : list of str or None, default=None
        List of component (target column) names to include in scoring. If None, all
        components are included. For panel data, these are unprefixed column names.
        Validated at fit time.
    panel_group_weight : dict or None, default=None
        Dictionary mapping panel group names to weights for weighted aggregation.
        If None, all panel groups weighted equally. Only applicable for panel data.

    See Also
    --------
    :mod:`yohou.metrics.point` : Concrete implementations (MeanAbsoluteError,
        MeanSquaredError, RootMeanSquaredError, MeanAbsolutePercentageError, etc.)
    :class:`yohou.point_forecaster.base.BasePointForecaster` : Produces point forecasts

    """

    _parameter_constraints: dict = {
        **BaseScorer._parameter_constraints,
        "aggregation_method": [list, StrOptions({"all", "timewise", "componentwise", "groupwise"})],
    }

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        panel_group_names: list[str] | None = None,
        component_names: list[str] | None = None,
        panel_group_weight: dict[str, float] | None = None,
    ):
        super().__init__(
            panel_group_names=panel_group_names,
            component_names=component_names,
            panel_group_weight=panel_group_weight,
        )
        self.aggregation_method = aggregation_method

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y_train: pl.DataFrame, **params) -> "BasePointScorer":
        """Fit the scorer on training data if needed.

        Validates aggregation_method, panel_group_names, and component_names.
        """
        # Validate point-specific parameters (aggregation_method)
        valid_methods = {"timewise", "componentwise", "groupwise"}
        self._validate_parameters(
            y_train=y_train,
            aggregation_method=self.aggregation_method,
            valid_aggregation_methods=valid_methods,
        )

        return super().fit(y_train, **params)

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

    def _aggregate_scores(
        self, raw_scores: pl.DataFrame, time_values: list | None = None
    ) -> float | pl.DataFrame:
        """Apply aggregation strategy to raw per-timestep per-component scores.

        Parameters
        ----------
        raw_scores : pl.DataFrame
            DataFrame with per-timestep (rows) per-component (columns) scores.
        time_values : list or None, default=None
            Time values for componentwise aggregation. Required if aggregation_method
            includes "componentwise".

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
    panel_group_names : list of str or None, default=None
        List of panel group names to include in scoring. If None, all panel groups
        are included. Only applicable for panel data. Validated at fit time.
    component_names : list of str or None, default=None
        List of component (target column) names to include in scoring. If None, all
        components are included. For panel data, these are unprefixed column names.
        Validated at fit time.
    coverage_rates : list of float or None, default=None
        List of coverage rates to include in scoring. If None, all coverage rates
        are included. Rates are validated against actual prediction columns during scoring.
    panel_group_weight : dict or None, default=None
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
        "coverage_rates": [list, None],
    }

    def __init__(
        self,
        aggregation_method: list[str] | str = "all",
        coverage_rates: list[float] | None = None,
        panel_group_names: list[str] | None = None,
        component_names: list[str] | None = None,
        panel_group_weight: dict[str, float] | None = None,
    ):
        super().__init__(
            panel_group_names=panel_group_names,
            component_names=component_names,
            panel_group_weight=panel_group_weight,
        )
        self.aggregation_method = aggregation_method
        self.coverage_rates = coverage_rates

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
                raise ValueError(
                    f"coverage_rates must be a list or None, got {type(self.coverage_rates)}"
                )
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
                    )

            # Check all elements are numeric
            if not all(isinstance(rate, (int, float)) for rate in self.coverage_rates):
                raise ValueError(
                    f"All elements in coverage_rates must be numeric (int or float), "
                    f"got types: {[type(r).__name__ for r in self.coverage_rates]}"
                )

            # Check range
            for rate in self.coverage_rates:
                if not 0 < rate < 1:
                    raise ValueError(
                        f"All coverage_rates must be between 0 and 1 (exclusive), got {rate}"
                    )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y_train: pl.DataFrame, **params) -> "BaseIntervalScorer":
        """Fit the scorer on training data if needed.

        Validates coverage_rates, aggregation_method, panel_group_names, and component_names.
        """
        # Validate coverage_rates
        self._validate_coverage_rates()

        # Validate interval-specific parameters (aggregation_method with coveragewise)
        valid_methods = {"timewise", "componentwise", "groupwise", "coveragewise"}
        self._validate_parameters(
            y_train=y_train,
            aggregation_method=self.aggregation_method,
            valid_aggregation_methods=valid_methods,
        )

        return super().fit(y_train, **params)

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
        time_values: list | None = None,
    ) -> float | dict[float, float] | pl.DataFrame:
        """Apply aggregation strategy to raw per-timestep per-component per-rate scores.

        Parameters
        ----------
        raw_scores : dict[float, pl.DataFrame]
            Dict mapping coverage_rate -> DataFrame(n_timesteps, n_components) with raw scores.
        time_values : list or None, default=None
            Time values for componentwise aggregation. Required if aggregation_method
            includes "componentwise".

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
            # No aggregation or unhandled combination,
            # aggregate both time and components to scalar/dict
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

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with conformity scorer attributes.

        """
        tags = super().__sklearn_tags__()
        tags.scorer_tags.prediction_type = "conformity"
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, y_train: pl.DataFrame, **params) -> "BaseConformityScorer":
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
            conformity_scores, 1 - coverage_rate / 2.0, method="higher"
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

        Raises
        ------
        ValueError
            If conformity_scores is empty.

        """
        # Convert to numpy array for quantile computation
        conformity_array = conformity_scores.to_numpy()

        # Check if array is empty
        if conformity_array.size == 0:
            raise ValueError(
                "Cannot compute quantile: conformity_scores is empty. "
                "This typically happens when the calibration set is too small. "
                "Increase calibration_size or reduce forecasting_horizon."
            )

        quantile: float = np.quantile(conformity_array, 1 - coverage_rate, method="lower")  # type: ignore[call-overload]

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
