"""Input validation utilities for time series data."""

import calendar
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs

from yohou.utils.panel import inspect_panel

__all__ = [
    "add_interval",
    "check_continuity",
    "check_X_actual_required",
    "check_forecasting_horizon_positive",
    "check_inputs",
    "check_interval_consistency",
    "check_groups",
    "check_groups_exist",
    "check_panel_groups_match",
    "check_panel_internal_consistency",
    "check_schema",
    "check_scorer_column_selection",
    "check_sufficient_rows",
    "check_time_column",
    "interval_to_timedelta",
    "parse_interval",
    "validate_column_names",
    "validate_search_data",
]

if TYPE_CHECKING:
    from yohou.metrics.base import BaseScorer


def check_time_column(df: pl.DataFrame, df_name: str = "DataFrame") -> None:
    """Validate that time column exists, has proper dtype, no nulls, and is sorted.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to validate.
    df_name : str, default="DataFrame"
        Name of DataFrame in error message.

    Raises
    ------
    ValueError
        If time column is missing, has wrong dtype, contains nulls, or is not sorted.

    See Also
    --------
    `check_interval_consistency` : Validate uniform time spacing.
    `check_continuity` : Validate temporal continuity between DataFrames.
    `check_inputs` : Validate consistent intervals across y and X_actual.

    """
    if "time" not in df.columns:
        raise ValueError(f"{df_name} must contain a 'time' column. Found columns: {list(df.columns)}")

    time_col = df["time"]
    # Check dtype
    if not isinstance(time_col.dtype, pl.Datetime | pl.Date):
        raise ValueError(f"'time' column in {df_name} must have dtype pl.Datetime or pl.Date, but got {time_col.dtype}")

    # Check for nulls
    if time_col.null_count() > 0:
        raise ValueError(
            f"'time' column in {df_name} contains {time_col.null_count()} null values. "
            "'time' column must not have missing values."
        )

    # Check sorting (ascending)
    if not time_col.is_sorted():
        raise ValueError(f"'time' column in {df_name} must be sorted in ascending order. Call df.sort('time') to fix.")


def check_scorer_column_selection(
    scorer: "BaseScorer",
    y_true: pl.DataFrame,
    y_pred: pl.DataFrame,
    pred_type: str,
    coverage_rates: list[float] | None = None,
    interval_pattern: re.Pattern | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Subselect columns based on scorer configuration.

    Parameters
    ----------
    scorer : BaseScorer
        Scorer instance with groups and components attributes.
    y_true : pl.DataFrame
        True values DataFrame.
    y_pred : pl.DataFrame
        Predicted values DataFrame.
    pred_type : str
        Prediction type ('point', 'interval', 'conformity', 'class_proba').
    coverage_rates : list[float], optional
        Coverage rates for interval forecasts.
    interval_pattern : re.Pattern, optional
        Regex pattern for matching interval column names.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        Filtered (y_true, y_pred) DataFrames.

    Raises
    ------
    ValueError
        If groups or components are invalid.

    See Also
    --------
    `inspect_panel` : Detect panel groups in a DataFrame.
    `check_groups` : Validate panel group names for forecaster operations.

    """
    has_panel_specs = hasattr(scorer, "groups") and scorer.groups is not None
    has_component_specs = hasattr(scorer, "components") and scorer.components is not None

    if not (has_panel_specs or has_component_specs or coverage_rates is not None):
        return y_true, y_pred

    # Validate coverage_rates if present (interval scorers)
    if coverage_rates is not None and pred_type == "interval" and interval_pattern is not None:
        available_rates = set()
        for col in y_pred.columns:
            if col in ("time", "vintage_time"):
                continue
            match = interval_pattern.match(col)
            if match:
                available_rates.add(float(match.group(3)))

        missing_rates = set(coverage_rates) - available_rates
        if missing_rates:
            raise ValueError(
                f"Requested coverage_rates {sorted(missing_rates)} not found in predictions. "
                f"Available rates: {sorted(available_rates)}"
            )

    _, y_groups = inspect_panel(y_true)

    # Validate panel groups if specified (must exist in data)
    if has_panel_specs:
        assert scorer.groups is not None
        missing_groups = set(scorer.groups) - set(y_groups.keys())
        if missing_groups:
            raise ValueError(
                f"Invalid groups: {sorted(missing_groups)} not found in data. "
                f"Available groups: {sorted(y_groups.keys())}"
            )

    is_panel = len(y_groups) > 0

    if is_panel:
        # Panel data: filter by groups and/or components
        selected_cols = []

        # Determine which groups to include
        groups_to_include = scorer.groups if has_panel_specs else list(y_groups.keys())

        # Type narrowing for iteration
        assert groups_to_include is not None
        for group_name in groups_to_include:
            if group_name in y_groups:
                group_cols = y_groups[group_name]

                # Filter by components if specified
                if has_component_specs:
                    assert scorer.components is not None
                    # Extract unprefixed column names and check against components
                    filtered_cols = [col for col in group_cols if col.split("__", 1)[1] in scorer.components]
                    selected_cols.extend(filtered_cols)
                else:
                    selected_cols.extend(group_cols)

        # Filter DataFrames
        if selected_cols:
            # Always preserve time column during subselection
            if "time" not in selected_cols:
                selected_cols = ["time"] + selected_cols

            # Check for interval columns in y_pred if prediction_type is interval
            if pred_type == "interval":
                # For interval, y_pred has _lower_ and _upper_ columns corresponding to y_true columns
                y_pred_selected_cols = ["time"]

                for col in selected_cols:
                    if col == "time":
                        continue
                    # Find matching columns in y_pred
                    matches = [
                        c for c in y_pred.columns if c.startswith(f"{col}_lower_") or c.startswith(f"{col}_upper_")
                    ]

                    # Filter by coverage rates
                    if coverage_rates is not None and interval_pattern is not None:
                        rate_filtered = []
                        for m in matches:
                            match = interval_pattern.match(m)
                            if match and float(match.group(3)) in coverage_rates:
                                rate_filtered.append(m)
                        matches = rate_filtered

                    y_pred_selected_cols.extend(matches)

                y_true = y_true.select(selected_cols)
                y_pred = y_pred.select(y_pred_selected_cols)
            elif pred_type == "class_proba":
                # Class proba: y_pred has {target}_proba_{class} columns
                y_pred_selected_cols = ["time"] if "time" in y_pred.columns else []
                for col in selected_cols:
                    if col == "time":
                        continue
                    y_pred_selected_cols.extend(c for c in y_pred.columns if c.startswith(f"{col}_proba_"))
                y_true = y_true.select(selected_cols)
                y_pred = y_pred.select(y_pred_selected_cols)
            else:
                # Point forecast: columns should match directly
                y_pred_cols = set(y_pred.columns)
                valid_y_pred_cols = [c for c in selected_cols if c in y_pred_cols]

                y_true = y_true.select(selected_cols)
                y_pred = y_pred.select(valid_y_pred_cols)

    # Global data: filter by components only
    elif has_component_specs:
        assert scorer.components is not None
        # Validate component names exist in data
        available_components = [col for col in y_true.columns if col != "time"]
        missing_components = set(scorer.components) - set(available_components)
        if missing_components:
            raise ValueError(
                f"Invalid components: {sorted(missing_components)} not found in data. "
                f"Available components: {sorted(available_components)}"
            )

        selected_cols = [col for col in y_true.columns if col in scorer.components]
        if selected_cols:
            # Always preserve time column during subselection
            if "time" not in selected_cols:
                selected_cols = ["time"] + selected_cols

            # For global data, logic is simpler
            if pred_type == "interval":
                y_pred_selected_cols = ["time"]
                for col in selected_cols:
                    if col == "time":
                        continue
                    matches = [
                        c for c in y_pred.columns if c.startswith(f"{col}_lower_") or c.startswith(f"{col}_upper_")
                    ]

                    # Filter by coverage rates
                    if coverage_rates is not None and interval_pattern is not None:
                        rate_filtered = []
                        for m in matches:
                            match = interval_pattern.match(m)
                            if match and float(match.group(3)) in coverage_rates:
                                rate_filtered.append(m)
                        matches = rate_filtered

                    y_pred_selected_cols.extend(matches)
                y_true = y_true.select(selected_cols)
                y_pred = y_pred.select(y_pred_selected_cols)
            elif pred_type == "class_proba":
                y_pred_selected_cols = ["time"] if "time" in y_pred.columns else []
                for col in selected_cols:
                    if col == "time":
                        continue
                    y_pred_selected_cols.extend(c for c in y_pred.columns if c.startswith(f"{col}_proba_"))
                y_true = y_true.select(selected_cols)
                y_pred = y_pred.select(y_pred_selected_cols)
            else:
                y_pred_cols = set(y_pred.columns)
                valid_y_pred_cols = [c for c in selected_cols if c in y_pred_cols]

                y_true = y_true.select(selected_cols)
                y_pred = y_pred.select(valid_y_pred_cols)
    elif coverage_rates is not None and pred_type == "interval" and interval_pattern is not None:
        # No component filter, but coverage rate filter
        y_pred_selected_cols = ["time"]

        # Filter y_pred columns to only those matching requested rates
        for col in y_pred.columns:
            if col in ("time", "vintage_time"):
                continue
            match = interval_pattern.match(col)
            if match and float(match.group(3)) in coverage_rates:
                y_pred_selected_cols.append(col)

        y_pred = y_pred.select(y_pred_selected_cols)

    return y_true, y_pred


def check_sufficient_rows(
    df: pl.DataFrame,
    min_rows: int,
    context: str,
    df_name: str = "DataFrame",
) -> None:
    """Validate DataFrame has sufficient rows for operation.

    Generic validation consolidating observation horizon, seasonality cycle,
    and interval inference checks.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to validate.
    min_rows : int
        Minimum number of rows required.
    context : str
        Description of why rows are needed (for error message).
        Examples: "for memory buffer", "for seasonal decomposition",
        "to compute time intervals"
    df_name : str, default="DataFrame"
        Name of DataFrame in error message.

    Raises
    ------
    ValueError
        If DataFrame has fewer rows than required.

    See Also
    --------
    `check_forecasting_horizon_positive` : Validate forecasting horizon is positive.
    `check_X_actual_required` : Validate X_actual is provided for recursive prediction.

    """
    actual_rows = len(df)
    if actual_rows < min_rows:
        raise ValueError(f"{df_name} has {actual_rows} rows but requires at least {min_rows} rows {context}.")


def check_groups(
    fitted_panel_groups: list[str] | None,
    requested_panel_groups: list[str] | None,
) -> list[str] | None:
    """Validate and normalize panel group names for forecaster operations.

    Validates that requested panel groups exist in the fitted forecaster and
    returns the normalized list of groups to use.

    Parameters
    ----------
    fitted_panel_groups : list of str or None
        Panel group names from fitted forecaster (groups_).
        None indicates the forecaster was fitted on global (non-panel) data.

    requested_panel_groups : list of str or None
        Panel group names requested for operation.
        If None, all fitted panel groups will be used.

    Returns
    -------
    list of str or None
        Validated panel group names to use for the operation.
        None for global (non-panel) data.

    Raises
    ------
    ValueError
        If requested_panel_groups is provided but forecaster was fitted on global data,
        or if any requested panel group was not present during fit.

    Examples
    --------
    >>> # Global data: no panel groups
    >>> result = check_groups(fitted_panel_groups=None, requested_panel_groups=None)
    >>> result is None
    True

    >>> # Panel data: use all fitted groups
    >>> check_groups(fitted_panel_groups=["sales", "inventory"], requested_panel_groups=None)
    ['sales', 'inventory']

    >>> # Panel data: validate specific groups
    >>> check_groups(fitted_panel_groups=["sales", "inventory"], requested_panel_groups=["sales"])
    ['sales']

    See Also
    --------
    `check_groups_exist` : Validate requested panel groups exist (deprecated).
    `check_panel_groups_match` : Validate y and X_actual have matching panel groups.
    `inspect_panel` : Detect panel groups in a DataFrame.

    """
    # If no groups requested, use all fitted groups
    if requested_panel_groups is None:
        return fitted_panel_groups

    # Validate that requested groups are compatible with fitted state
    if fitted_panel_groups is None:
        raise ValueError("The forecaster was fitted on global data, but `groups` were provided.")

    # Check that all requested groups exist in fitted groups
    missing_groups = set(requested_panel_groups) - set(fitted_panel_groups)
    if missing_groups:
        raise ValueError(
            f"Panel group(s) {sorted(missing_groups)} not found in fitted forecaster. "
            f"Available groups: {sorted(fitted_panel_groups)}."
        )

    return requested_panel_groups


def check_groups_exist(
    fitted_panel_groups: list[str],
    requested_panel_groups: list[str] | None,
    context: str,
) -> None:
    """Validate all requested panel groups exist in fitted forecaster.

    .. deprecated::
        Use `check_groups` instead.

    Consolidates duplicated validation in predict, observe, rewind methods.

    Parameters
    ----------
    fitted_panel_groups : list of str
        Panel group names from fitted forecaster (groups_).
    requested_panel_groups : list of str or None
        Panel group names requested for operation.
    context : str
        Method name for error message (e.g., "predict", "observe", "rewind").

    Raises
    ------
    ValueError
        If any requested panel group was not present during fit.

    See Also
    --------
    `check_groups` : Preferred replacement for this function.
    `check_panel_groups_match` : Validate y and X_actual have matching panel groups.

    """
    if requested_panel_groups is None:
        return

    missing_groups = set(requested_panel_groups) - set(fitted_panel_groups)
    if missing_groups:
        raise ValueError(
            f"Panel groups {sorted(missing_groups)} not found in fitted forecaster. "
            f"Available groups: {sorted(fitted_panel_groups)}. "
            f"Cannot {context} for groups that were not present during fit."
        )


def check_panel_internal_consistency(df: pl.DataFrame, df_name: str = "DataFrame") -> None:
    """Validate that all panel groups in a DataFrame have the same local column structure.

    For panel data with multiple groups (e.g., sales__store_1, sales__store_2),
    this checks that all groups within the same prefix have identical local column names.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to validate. Must have "time" column.
    df_name : str, default="DataFrame"
        Name of DataFrame in error message (e.g., "y", "X_actual", "y_pred").

    Raises
    ------
    ValueError
        If panel groups have mismatched local column structures.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> # Valid panel data - both groups have same local columns
    >>> df = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
    ...     "sales__store_1": [10, 20],
    ...     "sales__store_2": [30, 40],
    ... })
    >>> check_panel_internal_consistency(df, "y")  # No error

    >>> # Invalid - store_2 missing in second group
    >>> df_bad = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1)],
    ...     "sales__store_1": [10],
    ...     "revenue__store_1": [100],
    ... })
    >>> check_panel_internal_consistency(df_bad, "y")  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: Panel structure mismatch in y...

    See Also
    --------
    `check_panel_groups_match` : Validate y and X_actual have matching panel groups.
    `check_groups` : Validate panel group names for forecaster operations.
    `inspect_panel` : Detect panel groups in a DataFrame.

    """
    _, groups = inspect_panel(df)
    if not groups:
        return  # No panel data, nothing to check

    # Get reference group (first one)
    ref_grp = next(iter(groups))
    ref_cols = sorted(c.split("__", 1)[1] for c in groups[ref_grp])

    # Check all other groups match reference
    for grp, cols in groups.items():
        curr_cols = sorted(c.split("__", 1)[1] for c in cols)
        if curr_cols != ref_cols:
            raise ValueError(
                f"Panel structure mismatch in `{df_name}`. Group '{ref_grp}' has local "
                f"columns {ref_cols}, but group '{grp}' has {curr_cols}."
            )


def check_panel_groups_match(
    y: pl.DataFrame | None,
    X_actual: pl.DataFrame | None,
) -> None:
    """Validate that y and X_actual have compatible panel group structures.

    When both DataFrames contain panel columns (using the ``__`` separator),
    they must share the same entity prefixes. For example, if y has
    ``"store_1__sales"`` and ``"store_2__sales"``, then X_actual must also have
    panel columns with ``"store_1__"`` and ``"store_2__"`` prefixes.

    Global-only X_actual (no ``__`` columns) is always valid regardless of y's
    structure. Global features are broadcast to every panel group.

    Parameters
    ----------
    y : pl.DataFrame or None
        Target DataFrame. Can be None.
    X_actual : pl.DataFrame or None
        Feature DataFrame. Can be None.

    Raises
    ------
    ValueError
        If both y and X_actual have panel columns but with different group prefixes,
        or if y is global but X_actual has panel columns.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> # Valid: both have same entity prefixes
    >>> y = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1)],
    ...     "store_1__sales": [10],
    ...     "store_2__sales": [20],
    ... })
    >>> X_actual = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1)],
    ...     "store_1__temp": [100],
    ...     "store_2__temp": [200],
    ... })
    >>> check_panel_groups_match(y, X_actual)  # No error

    >>> # Valid: panel y with global-only X_actual (broadcast to all groups)
    >>> X_global = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1)],
    ...     "weather": [25.0],
    ... })
    >>> check_panel_groups_match(y, X_global)  # No error

    >>> # Invalid: different entity prefixes
    >>> X_bad = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1)],
    ...     "sensor_1__temp": [25.0],
    ... })
    >>> check_panel_groups_match(y, X_bad)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: Panel groups mismatch between y and X_actual...

    See Also
    --------
    `check_panel_internal_consistency` : Validate panel groups have consistent structure.
    `check_groups` : Validate panel group names for forecaster operations.
    `inspect_panel` : Detect panel groups in a DataFrame.

    """
    if y is None or X_actual is None:
        return  # Can't check if one is missing

    _, y_groups = inspect_panel(y)
    _, X_groups = inspect_panel(X_actual)

    # Global-only X_actual (no panel columns) is valid with any y structure.
    # Global features are broadcast to every panel group.
    if X_groups and set(y_groups.keys()) != set(X_groups.keys()):
        raise ValueError(
            f"Panel groups mismatch between `y` and `X_actual`. "
            f"`y` groups: {sorted(y_groups.keys())}, "
            f"`X_actual` groups: {sorted(X_groups.keys())}."
        )


def check_forecasting_horizon_positive(
    horizon: int | None,
    allow_none: bool = False,
) -> None:
    """Validate forecasting horizon is positive.

    Parameters
    ----------
    horizon : int or None
        Forecasting horizon value.
    allow_none : bool, default=False
        Whether None is acceptable (for predict with optional horizon override).

    Raises
    ------
    ValueError
        If horizon is not positive or is None when not allowed.

    See Also
    --------
    `check_X_actual_required` : Validate X_actual is provided for recursive prediction.
    `check_sufficient_rows` : Validate DataFrame has enough rows for an operation.

    """
    if horizon is None:
        if not allow_none:
            raise ValueError("forecasting_horizon cannot be None")
        return

    if horizon < 1:
        raise ValueError(f"forecasting_horizon must be >= 1, got {horizon}")


def check_X_actual_required(
    X_actual: pl.DataFrame | None,
    observation_horizon: int,
    context: str,
) -> None:
    """Validate X_actual is provided when required for recursive prediction.

    Consolidates duplicated validation in point and interval forecasters.

    Parameters
    ----------
    X_actual : pl.DataFrame or None
        Actual exogenous features (``X_actual``).
    observation_horizon : int
        Observation horizon value.
    context : str
        Context for error message (e.g., "predict", "predict_interval").

    Raises
    ------
    ValueError
        If X_actual is None but observation_horizon > 0 (recursive prediction needs
        X_actual).

    See Also
    --------
    `check_forecasting_horizon_positive` : Validate forecasting horizon is positive.
    `check_sufficient_rows` : Validate DataFrame has enough rows for an operation.

    """
    if observation_horizon > 0 and X_actual is None:
        raise ValueError(
            f"For recursive predictions with observation_horizon > 0, "
            f"X_actual must be provided for {context}. "
            f"Got observation_horizon={observation_horizon} but X_actual=None."
        )


def check_interval_consistency(df: pl.DataFrame) -> str:
    """Validate that a time series has uniform time spacing.

    Checks that all consecutive time steps in the DataFrame have the same interval.
    Supports both fixed intervals (daily, hourly) and variable-length intervals
    (monthly, quarterly, yearly).

    Parameters
    ----------
    df : pl.DataFrame
        Time series DataFrame with a "time" column containing datetime values.

    Returns
    -------
    str
        String representation of the interval.
        Examples: "1d", "1h", "1w", "1mo", "3mo", "1q", "1y"

    Raises
    ------
    ValueError
        If the time intervals are not consistent throughout the DataFrame.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> # Valid: uniform 1-day intervals
    >>> df = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1d", eager=True
    ...     ),
    ...     "value": [10, 20, 30, 40, 50],
    ... })
    >>> interval = check_interval_consistency(df)
    >>> interval
    '1d'

    >>> # Invalid: inconsistent intervals
    >>> df_bad = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 4)],
    ...     "value": [10, 20, 30],
    ... })
    >>> check_interval_consistency(df_bad)  # doctest: +SKIP
    ValueError

    See Also
    --------
    `check_inputs` : Validates multiple DataFrames have matching intervals
    `check_continuity` : Validates temporal continuity between DataFrames
    `add_interval` : Add intervals to datetime values

    """
    if df is None:
        raise ValueError("DataFrame cannot be None")

    time_series = df["time"].to_list()

    if len(time_series) < 2:
        raise ValueError("Need at least 2 time points to infer interval")

    # Calculate deltas
    deltas = [time_series[i + 1] - time_series[i] for i in range(len(time_series) - 1)]
    unique_deltas = sorted(set(deltas))

    # Check if deltas are all similar (within small tolerance for rounding)
    delta_days = [d.days for d in unique_deltas]
    max_delta = max(delta_days)

    # Sub-day intervals with small variation (e.g., hourly with DST)
    if max_delta == 0:
        # All deltas are sub-day
        delta_seconds = [d.total_seconds() for d in unique_deltas]
        if max(delta_seconds) - min(delta_seconds) <= 3600:  # ±1 hour tolerance
            median_seconds = sorted(delta_seconds)[len(delta_seconds) // 2]
            return _timedelta_to_string(timedelta(seconds=median_seconds))

    # Infer based on delta distribution
    freq = _infer_freq_from_deltas(time_series, unique_deltas)
    if freq is not None:
        return freq

    # Could not infer - raise detailed error
    raise ValueError(
        f"Time series has inconsistent intervals. "
        f"Found {len(unique_deltas)} different intervals: {unique_deltas}. "
        f"Cannot infer a regular frequency pattern."
    )


def check_inputs(y: pl.DataFrame, X_actual: pl.DataFrame | None) -> str:
    """Validate that target and feature DataFrames have consistent time intervals.

    Ensures all input DataFrames (target y and exogenous features X_actual) have the same
    uniform time interval. This is required for proper alignment in forecasting
    operations.

    Parameters
    ----------
    y : pl.DataFrame
        Target time series with "time" column.

    X_actual : pl.DataFrame or None
        Exogenous feature time series with "time" column, or None.

    Returns
    -------
    str
        The common time interval shared by all provided DataFrames (e.g., "1d", "1mo").

    Raises
    ------
    ValueError
        If any DataFrame has inconsistent intervals internally, or if the intervals
        don't match across DataFrames.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> time_index = pl.datetime_range(
    ...     start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1d", eager=True
    ... )
    >>> y = pl.DataFrame({"time": time_index, "sales": [100, 110, 120, 130, 140]})
    >>> X_actual = pl.DataFrame({"time": time_index, "holiday": [0, 0, 1, 0, 0]})
    >>> interval = check_inputs(y, X_actual)
    >>> interval
    '1d'

    See Also
    --------
    `check_interval_consistency` : Validates single DataFrame intervals
    `validate_column_names` : Validates column names don't misuse __ separator

    """
    # Validate column names first
    validate_column_names(y)
    if X_actual is not None:
        validate_column_names(X_actual)

    y_interval = check_interval_consistency(y)
    if X_actual is not None:
        X_interval = check_interval_consistency(X_actual)

        if X_interval != y_interval:
            raise ValueError(
                f"Time interval mismatch: y has interval {y_interval}, but X_actual has interval "
                f"{X_interval}. All inputs must have the same time interval."
            )

    return y_interval


def validate_search_data(y: pl.DataFrame, X_actual: pl.DataFrame | None) -> str:
    """Validate input data for hyperparameter search (GridSearchCV, RandomizedSearchCV).

    Performs comprehensive validation of time series data for cross-validation:
    - Checks that y is not None
    - Validates time column presence, dtype, nulls, and sorting
    - Validates panel data internal consistency
    - Validates panel data group matching between y and X_actual
    - Validates consistent time intervals across DataFrames

    This function is designed for SearchCV contexts where we validate data
    without modifying forecaster state (unlike validate_forecaster_data).

    Parameters
    ----------
    y : pl.DataFrame
        Target time series with "time" column.

    X_actual : pl.DataFrame or None
        Exogenous feature time series with "time" column, or None.

    Returns
    -------
    str
        The common time interval shared by all provided DataFrames (e.g., "1d", "1mo").

    Raises
    ------
    ValueError
        If y is None, time columns are invalid, panel data is inconsistent,
        or intervals don't match across DataFrames.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> time_index = pl.datetime_range(
    ...     start=datetime(2020, 1, 1), end=datetime(2020, 1, 5), interval="1d", eager=True
    ... )
    >>> y = pl.DataFrame({"time": time_index, "sales": [100, 110, 120, 130, 140]})
    >>> X_actual = pl.DataFrame({"time": time_index, "holiday": [0, 0, 1, 0, 0]})
    >>> interval = validate_search_data(y, X_actual)
    >>> interval
    '1d'

    See Also
    --------
    `validate_forecaster_data` : Data validation with forecaster state management
    `check_inputs` : Validates consistent time intervals
    `check_time_column` : Validates time column properties

    """
    if y is None:
        raise ValueError("`y` cannot be None")

    # Validate time columns
    check_time_column(y, "y")
    if X_actual is not None:
        check_time_column(X_actual, "X_actual")

    # Validate panel data internal consistency
    check_panel_internal_consistency(y, "y")
    if X_actual is not None:
        check_panel_internal_consistency(X_actual, "X_actual")
        # Validate panel data groups match
        check_panel_groups_match(y, X_actual)

    # Validate consistent time intervals and return the interval
    return check_inputs(y, X_actual)


def validate_column_names(df: pl.DataFrame) -> None:
    """Validate that __ separator is used only for panel data group names.

    The __ separator is reserved for panel data groups following the pattern
    <GROUP>__<SERIES> (e.g., "sales__store_1"). This function ensures column
    names either:
    - Don't contain __ at all (global columns), OR
    - Follow the exact pattern ^[^_]+__[^_]+.*$ (group columns)

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to validate.

    Raises
    ------
    ValueError
        If any column name contains __ but doesn't match the group pattern,
        or if __ appears multiple times in inconsistent way.

    Examples
    --------
    >>> import polars as pl
    >>> # Valid: no __ separator
    >>> df = pl.DataFrame({"time": [1, 2], "value": [10, 20]})
    >>> validate_column_names(df)  # No error

    >>> # Valid: proper group pattern
    >>> df = pl.DataFrame({"time": [1, 2], "sales__store_1": [100, 110]})
    >>> validate_column_names(df)  # No error

    >>> # Invalid: __ without proper pattern
    >>> df = pl.DataFrame({"time": [1, 2], "my__bad__col": [10, 20]})
    >>> validate_column_names(df)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: Column 'my__bad__col' contains multiple __ separators...

    See Also
    --------
    `check_inputs` : Validates time intervals and calls this function

    """
    # Pattern: allows underscores in group/series names, but not adjacent to __
    # Valid: store_1__sales, my_store__my_sales
    # Invalid: store___sales (underscore adjacent to __), _store__sales, store__sales_
    # Strategy: split on __, check parts don't start/end with _ and are non-empty

    # Handle None case
    if df is None:
        return

    for col_name in df.columns:
        if col_name == "time":
            continue

        if "__" not in col_name:
            # No __ separator - valid global column
            continue

        # Column contains __ - validate it follows the pattern
        parts = col_name.split("__")

        # Check for common issues to provide helpful error messages
        if len(parts) != 2:
            raise ValueError(
                f"Column '{col_name}' contains multiple __ separators. "
                f"The __ separator is reserved for panel data groups and must appear "
                f"exactly once, following the pattern '<GROUP>__<SERIES>' "
                f"(e.g., 'sales__store_1'). If this column was produced by a "
                f"meta-transformer (FeatureUnion, ColumnTransformer), ensure it "
                f"uses panel-safe prefixing with single underscore separators. "
                f"Please rename columns to avoid using __ "
                f"or use it only for panel data groups."
            )

        group, series = parts

        # Check for empty parts
        if not group or not series:
            raise ValueError(
                f"Column '{col_name}' has __ at the beginning or end. "
                f"The __ separator must separate a non-empty group prefix from a "
                f"non-empty series suffix (e.g., 'sales__store_1')."
            )

        # Check for underscores adjacent to __
        if group.endswith("_") or series.startswith("_"):
            raise ValueError(
                f"Column '{col_name}' has underscores adjacent to the __ separator. "
                f"The pattern '<GROUP>__<SERIES>' requires that the group part doesn't "
                f"end with _ and the series part doesn't start with _ "
                f"(e.g., 'store_1__sales' is valid, but 'store_1___sales' or 'store1_"
                "__sales' are not)."
            )


def check_schema(
    df: pl.DataFrame,
    expected_schema: dict[str, pl.DataType],
    groups: list[str] | None = None,
) -> pl.DataFrame:
    """Validate DataFrame schema and return with proper column ordering.

    Ensures that data has the same column names and dtypes as expected,
    and returns the DataFrame with columns in the correct order (time column first,
    followed by schema columns in order).

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to validate (should include "time" column).
    expected_schema : dict[str, pl.DataType]
        Expected schema for non-time columns.
        For panel data, this should contain unprefixed column names.
    groups : list[str] or None, default=None
        Group prefixes for panel data. If provided, constructs expected
        schema with prefixes (e.g., "panel__series_0"). None for global data.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns in proper order: ["time"] + schema columns.

    Raises
    ------
    ValueError
        If incoming schema doesn't match expected schema.

    Examples
    --------
    >>> import polars as pl
    >>> # Non-panel data validation
    >>> df = pl.DataFrame({"value": [10, 20], "time": [1, 2]})
    >>> expected_schema = {"value": pl.Int64}
    >>> result = check_schema(df, expected_schema)
    >>> list(result.columns)
    ['time', 'value']

    >>> # Schema mismatch raises error
    >>> df_wrong = pl.DataFrame({"time": [1, 2], "value": [10.0, 20.0]})  # Float64
    >>> check_schema(df_wrong, expected_schema)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: Schema mismatch. Expected: {'value': Int64}, got: {'value': Float64}

    >>> # Panel data validation (constructs prefixed schema automatically)
    >>> df_panel = pl.DataFrame({"panel__s1": [15, 25], "time": [1, 2], "panel__s0": [10, 20]})
    >>> expected_schema = {"s0": pl.Int64, "s1": pl.Int64}
    >>> result = check_schema(df_panel, expected_schema, groups=["panel"])
    >>> list(result.columns)
    ['time', 'panel__s0', 'panel__s1']

    See Also
    --------
    `check_inputs` : Validates time intervals
    `BaseForecaster` : Uses this function to validate incoming data

    Notes
    -----
    For panel data, this function automatically constructs the expected schema
    with prefixes (e.g., "sales__store_1") from the unprefixed expected_schema.
    The returned DataFrame has columns ordered consistently with the schema.

    """
    # Construct expected column list based on groups
    if groups is None:
        # Non-panel data: use schema as-is
        expected_columns = ["time"] + list(expected_schema.keys())
        expected_full_schema = expected_schema
    else:
        # Panel data: construct prefixed schema
        expected_columns = ["time"]
        expected_full_schema = {}
        for group_name in groups:
            for col, dtype in expected_schema.items():
                prefixed_col = f"{group_name}__{col}"
                expected_columns.append(prefixed_col)
                expected_full_schema[prefixed_col] = dtype

    # Select columns in proper order (also validates presence)
    df = df.select(expected_columns)

    # Extract actual schema (excluding time column) for validation
    incoming_schema = dict(df.select(~cs.by_name("time")).schema)

    # Validate dtypes
    if incoming_schema != expected_full_schema:
        raise ValueError(f"Schema mismatch. Expected: {expected_full_schema}, got: {incoming_schema}")

    return df


def _normalize_interval(interval: str) -> str:
    """Normalize interval string to canonical form.

    Converts day-based intervals that represent monthly patterns to their
    canonical monthly representation (e.g., "28d", "29d", "30d", "31d" -> "1mo").

    Parameters
    ----------
    interval : str
        Interval string like "1d", "31d", "1mo", etc.

    Returns
    -------
    str
        Normalized interval string.

    Examples
    --------
    >>> _normalize_interval("31d")
    '1mo'
    >>> _normalize_interval("28d")
    '1mo'
    >>> _normalize_interval("1d")
    '1d'
    >>> _normalize_interval("1mo")
    '1mo'

    """
    # Parse the interval
    multiplier, unit = parse_interval(interval)

    # If already in monthly/quarterly/yearly format, return as-is
    if unit in {"mo", "q", "y"}:
        return interval

    # Convert day intervals to canonical form if they match monthly patterns
    if unit == "d":
        total_days = multiplier
        # 1 month: 28-31 days
        if 28 <= total_days <= 31:
            return "1mo"
        # 2 months: 59-62 days
        if 59 <= total_days <= 62:
            return "2mo"
        # 3 months (quarterly): 89-92 days
        if 89 <= total_days <= 92:
            return "3mo"
        # 6 months: 181-184 days
        if 181 <= total_days <= 184:
            return "6mo"
        # 1 year: 365-366 days
        if 365 <= total_days <= 366:
            return "1y"

    # Return as-is if no normalization applies
    return interval


def check_continuity(
    df_p: pl.DataFrame,
    df_n: pl.DataFrame,
    expected_interval: str | None,
    check_intervals: bool = True,
) -> None:
    """Validate temporal continuity between consecutive DataFrames.

    Ensures that two DataFrames representing consecutive time periods have no gaps
    or overlaps in their time indices. Used when appending new data to existing
    time series.

    Parameters
    ----------
    df_p : pl.DataFrame
        Previous (earlier) time series DataFrame with "time" column.

    df_n : pl.DataFrame
        Next (later) time series DataFrame with "time" column.

    expected_interval : str | None
        Expected time interval between consecutive observations.
        Examples: "1d", "1h", "1mo", "3mo", "1y"
        If None, skip interval validation (used for single-step predictions).

    check_intervals : bool, default=True
        If True, validates that both DataFrames have consistent internal intervals.

    Raises
    ------
    ValueError
        If there is a gap or overlap between df_p and df_n, or if internal
        intervals don't match expected_interval.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> # Continuous time series
    >>> df1 = pl.DataFrame({
    ...     "time": pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 3), "1d", eager=True),
    ...     "value": [10, 20, 30],
    ... })
    >>> df2 = pl.DataFrame({
    ...     "time": pl.datetime_range(datetime(2020, 1, 4), datetime(2020, 1, 6), "1d", eager=True),
    ...     "value": [40, 50, 60],
    ... })
    >>> check_continuity(df1, df2, "1d")

    See Also
    --------
    `check_interval_consistency` : Validates uniform time spacing

    """
    # Skip validation if expected_interval is None (e.g., single-step prediction)
    if expected_interval is None:
        return

    if check_intervals:
        if len(df_p) > 1:
            interval_p = check_interval_consistency(df_p)
            # Normalize intervals for comparison (e.g., "31d" -> "1mo")
            interval_p_norm = _normalize_interval(interval_p)
            expected_interval_norm = _normalize_interval(expected_interval)

            if interval_p_norm != expected_interval_norm:
                raise ValueError(
                    f"Previous DataFrame has interval {interval_p}, but expected interval is {expected_interval}."
                )

        if len(df_n) > 1:
            interval_n = check_interval_consistency(df_n)
            interval_n_norm = _normalize_interval(interval_n)

            if len(df_p) > 1:
                interval_p_norm = _normalize_interval(interval_p)
                if interval_p_norm != interval_n_norm:
                    raise ValueError(
                        "Interval mismatch between DataFrames: previous DataFrame has interval "
                        f"{interval_p}, but next DataFrame has interval {interval_n}."
                    )

            expected_interval_norm = _normalize_interval(expected_interval)
            if interval_n_norm != expected_interval_norm:
                raise ValueError(
                    f"Next DataFrame has interval {interval_n}, but expected interval is {expected_interval}."
                )

    time_p = df_p.select(cs.by_name("time")).tail(1)
    time_n = df_n.select(cs.by_name("time")).head(1)

    time = pl.concat([time_p, time_n])

    time_change = time.select(pl.col("time").diff()).fill_null(strategy="backward")
    interval_td = time_change[0, 0]

    # Convert expected_interval string to timedelta for comparison
    expected_interval_td = interval_to_timedelta(expected_interval)

    # For variable-length intervals (monthly, quarterly, yearly), we need to compute
    # the expected timedelta using calendar arithmetic
    if expected_interval_td is None:
        # Use add_interval to compute what the expected next time should be
        last_time = time_p["time"].item()  # Extract scalar from single-row DataFrame
        expected_next_time = add_interval(last_time, expected_interval, 1)
        first_time = time_n["time"].item()  # Extract scalar from single-row DataFrame

        if first_time != expected_next_time:
            if first_time > expected_next_time:
                raise ValueError(
                    f"Gap detected between DataFrames: previous DataFrame ends at {last_time}, "
                    f"next DataFrame starts at {first_time}, expected {expected_next_time} "
                    f"(interval {expected_interval})."
                )
            else:
                raise ValueError(
                    f"Overlap detected between DataFrames: previous DataFrame ends at {last_time}, "
                    f"next DataFrame starts at {first_time}, expected {expected_next_time} "
                    f"(interval {expected_interval})."
                )
    # Fixed interval - can use timedelta comparison
    elif interval_td != expected_interval_td:
        last_time_p = time_p[0, 0]
        first_time_n = time_n[0, 0]
        if interval_td > expected_interval_td:
            raise ValueError(
                f"Gap detected between DataFrames: previous DataFrame ends at {last_time_p}, "
                f"next DataFrame starts at {first_time_n}, creating a gap of {interval_td} "
                f"(expected {expected_interval})."
            )
        else:
            raise ValueError(
                f"Overlap detected between DataFrames: previous DataFrame ends at "
                f"{last_time_p}, next DataFrame starts at {first_time_n}, with interval "
                f"{interval_td} (expected {expected_interval})."
            )


def _timedelta_to_string(td: timedelta) -> str:
    r"""Convert a timedelta to string interval format.

    Parameters
    ----------
    td : timedelta
        Timedelta to convert.

    Returns
    -------
    str
        String representation like "1d", "1h", "1w", "2w", "1ms", "1us", etc.

    Examples
    --------
    >>> _timedelta_to_string(timedelta(days=1))
    '1d'
    >>> _timedelta_to_string(timedelta(hours=1))
    '1h'
    >>> _timedelta_to_string(timedelta(days=7))
    '7d'
    >>> _timedelta_to_string(timedelta(milliseconds=1))
    '1ms'
    >>> _timedelta_to_string(timedelta(microseconds=100))
    '100us'

    """
    total_seconds = td.total_seconds()

    # Try common patterns from largest to smallest
    # Prefer day-based representation for consistency
    if total_seconds % 86400 == 0:
        days = int(total_seconds // 86400)
        return f"{days}d"
    elif total_seconds % 3600 == 0:
        hours = int(total_seconds // 3600)
        return f"{hours}h"
    elif total_seconds % 60 == 0:
        minutes = int(total_seconds // 60)
        return f"{minutes}m"
    elif total_seconds >= 1 and total_seconds == int(total_seconds):
        seconds = int(total_seconds)
        return f"{seconds}s"
    else:
        # Subsecond intervals: try milliseconds first, then microseconds
        total_microseconds = td.total_seconds() * 1_000_000
        if total_microseconds >= 1000 and total_microseconds % 1000 == 0:
            milliseconds = int(total_microseconds // 1000)
            return f"{milliseconds}ms"
        else:
            microseconds = int(total_microseconds)
            return f"{microseconds}us"


def _infer_freq_from_deltas(time_series: list[datetime], unique_deltas: list[timedelta]) -> str | None:
    """Infer frequency pattern from delta distribution.

    Parameters
    ----------
    time_series : list[datetime]
        List of datetime values.

    unique_deltas : list[timedelta]
        Unique time deltas between consecutive dates.

    Returns
    -------
    str or None
        Inferred frequency string or None if cannot infer.

    """
    delta_days = [d.days for d in unique_deltas]
    min_delta, max_delta = min(delta_days), max(delta_days)

    # Try monthly inference first (handles 1mo, 2mo, 3mo, 6mo, etc.)
    # Monthly patterns: 28-31 days (1mo), 59-62 days (2mo), 89-92 days (3mo), 181-184 days (6mo)
    if 28 <= min_delta <= 31 or 59 <= min_delta <= 62 or 89 <= min_delta <= 92 or 181 <= min_delta <= 184:
        freq = _infer_monthly_freq(time_series)
        if freq is not None:
            return freq

    # Yearly patterns: 365-366 days
    if 365 <= min_delta <= 366 and 365 <= max_delta <= 366:
        return "1y"

    # Daily pattern: uniform day intervals
    if len(unique_deltas) == 1:
        return _timedelta_to_string(unique_deltas[0])

    return None


def _infer_monthly_freq(time_series: list[datetime]) -> str | None:
    r"""Infer monthly frequency (1mo, 2mo, 3mo, etc.) by checking month differences.

    Parameters
    ----------
    time_series : list[datetime]
        List of datetime values.

    Returns
    -------
    str or None
        Frequency string like "1mo", "2mo", "3mo", etc., or None if not monthly.

    """
    if len(time_series) < 2:
        return None

    month_diffs = []
    for i in range(len(time_series) - 1):
        d1, d2 = time_series[i], time_series[i + 1]
        month_diff = (d2.year - d1.year) * 12 + (d2.month - d1.month)
        month_diffs.append(month_diff)

    unique_month_diffs = set(month_diffs)
    if len(unique_month_diffs) == 1:
        n_months = month_diffs[0]
        if _check_day_of_month_consistency(time_series):
            return f"{n_months}mo" if n_months > 1 else "1mo"


def _check_day_of_month_consistency(time_series: list[datetime]) -> bool:
    """Check if day-of-month is consistent (allowing for month-end edge cases).

    Parameters
    ----------
    time_series : list[datetime]
        List of datetime values.

    Returns
    -------
    bool
        True if day-of-month is consistent.

    """
    if not time_series:
        return False

    # The target day is the day from the first date in the series
    target_day = time_series[0].day

    for dt in time_series:
        days_in_month = calendar.monthrange(dt.year, dt.month)[1]
        actual_day = dt.day

        # If target day exceeds days in month, should be capped at month end
        if target_day > days_in_month:
            if actual_day != days_in_month:
                return False
        elif target_day != actual_day:
            return False

    return True


def parse_interval(interval: str) -> tuple[int, str]:
    r"""Parse interval string into (multiplier, unit).

    Parameters
    ----------
    interval : str
        Interval string like "1d", "3mo", "2w".

    Returns
    -------
    tuple[int, str]
        Tuple of (multiplier, unit).

    Examples
    --------
    >>> parse_interval("1d")
    (1, 'd')
    >>> parse_interval("3mo")
    (3, 'mo')
    >>> parse_interval("2w")
    (2, 'w')
    >>> parse_interval("100ms")
    (100, 'ms')
    >>> parse_interval("50us")
    (50, 'us')

    See Also
    --------
    `interval_to_timedelta` : Convert interval string to timedelta.
    `add_interval` : Add intervals to a datetime value.
    `check_interval_consistency` : Validate uniform time spacing.

    """
    match = re.match(r"(\d+)(mo|q|y|w|d|h|min|ms|us|m|s)", interval)
    if not match:
        raise ValueError(f"Invalid interval format: {interval}")
    unit = match.group(2)
    # Normalize 'min' to 'm' (polars convention)
    if unit == "min":
        unit = "m"
    return int(match.group(1)), unit


def interval_to_timedelta(interval: str) -> timedelta | None:
    r"""Convert fixed interval to timedelta, or None for variable intervals.

    Parameters
    ----------
    interval : str
        Interval string.

    Returns
    -------
    timedelta or None
        Timedelta for fixed intervals, None for variable intervals.

    Examples
    --------
    >>> interval_to_timedelta("1d")
    datetime.timedelta(days=1)
    >>> interval_to_timedelta("2h")
    datetime.timedelta(seconds=7200)
    >>> interval_to_timedelta("1mo") is None
    True
    >>> interval_to_timedelta("100ms")
    datetime.timedelta(microseconds=100000)
    >>> interval_to_timedelta("50us")
    datetime.timedelta(microseconds=50)

    See Also
    --------
    `parse_interval` : Parse interval string into multiplier and unit.
    `add_interval` : Add intervals to a datetime value.
    `check_interval_consistency` : Validate uniform time spacing.

    """
    multiplier, unit = parse_interval(interval)

    if unit == "d":
        return timedelta(days=multiplier)
    elif unit == "h":
        return timedelta(hours=multiplier)
    elif unit == "m":
        return timedelta(minutes=multiplier)
    elif unit == "s":
        return timedelta(seconds=multiplier)
    elif unit == "w":
        return timedelta(weeks=multiplier)
    elif unit == "ms":
        return timedelta(milliseconds=multiplier)
    elif unit == "us":
        return timedelta(microseconds=multiplier)
    else:
        # Variable-length interval, cannot convert to timedelta
        return None


def add_interval(start: datetime, interval: str | timedelta, n: int = 1) -> datetime:
    r"""Add n intervals to a datetime (handles variable-length intervals).

    Supports multi-period intervals like "2mo", "3mo", "6mo", etc.

    Parameters
    ----------
    start : datetime
        Starting datetime.

    interval : str or timedelta
        Interval string like "1d", "1mo", "3mo", "1q", "1y" or a timedelta.

    n : int, default=1
        Number of intervals to add.

    Returns
    -------
    datetime
        Result datetime.

    Examples
    --------
    >>> from datetime import datetime
    >>> add_interval(datetime(2020, 1, 15), "1d", 5)
    datetime.datetime(2020, 1, 20, 0, 0)
    >>> add_interval(datetime(2020, 1, 31), "1mo", 1)
    datetime.datetime(2020, 2, 29, 0, 0)
    >>> add_interval(datetime(2020, 1, 31), "2mo", 2)
    datetime.datetime(2020, 5, 31, 0, 0)

    See Also
    --------
    `parse_interval` : Parse interval string into multiplier and unit.
    `interval_to_timedelta` : Convert interval string to timedelta.
    `check_interval_consistency` : Validate uniform time spacing.

    """
    if isinstance(interval, timedelta):
        return start + (interval * n)

    multiplier, unit = parse_interval(interval)
    total_units = multiplier * n

    if unit == "d":
        return start + timedelta(days=total_units)
    elif unit == "h":
        return start + timedelta(hours=total_units)
    elif unit == "m":
        return start + timedelta(minutes=total_units)
    elif unit == "s":
        return start + timedelta(seconds=total_units)
    elif unit == "w":
        return start + timedelta(weeks=total_units)
    elif unit == "mo":
        # Add months with day-of-month preservation
        month = start.month - 1 + total_units
        year = start.year + month // 12
        month = month % 12 + 1
        day = min(start.day, calendar.monthrange(year, month)[1])
        return start.replace(year=year, month=month, day=day)
    elif unit == "q":
        # Quarters are 3 months
        return add_interval(start, "3mo", n)
    elif unit == "y":
        # Add years (handles leap years)
        return start.replace(year=start.year + total_units)
    else:
        raise ValueError(f"Unsupported interval unit: {unit}")
