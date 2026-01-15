"""Input validation utilities for time series data."""

import calendar
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from yohou.base import BaseEstimator


def validate_data(
    estimator: "BaseEstimator",
    y: pl.DataFrame | None = None,
    X: pl.DataFrame | None = None,
    *,
    X_t: pl.DataFrame | None = None,
    X_p: pl.DataFrame | None = None,
    reset: bool = True,
    panel_group_names: list[str] | None = None,
    observation_horizon: int | None = None,
    **check_params,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None, list[str] | None]:
    """Validate and prepare polars DataFrames for estimator methods.

    This is the central validation orchestrator following sklearn's validate_data() pattern.
    It handles schema validation, fitted attribute management, and delegates to atomic
    validation functions based on estimator tags and context.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator instance being validated.
    y : pl.DataFrame or None, default=None
        Target time series DataFrame.
    X : pl.DataFrame or None, default=None
        Feature time series DataFrame (untransformed).
    X_t : pl.DataFrame or None, default=None
        Transformed time series for inverse_transform validation.
        Mutually exclusive with X (use X_t for inverse_transform, X otherwise).
    X_p : pl.DataFrame or None, default=None
        Previous untransformed time series for inverse_transform validation.
        When provided with observation_horizon, validates inverse_transform requirements.
    reset : bool, default=True
        Whether to set fitted attributes (True for fit context) or validate against
        them (False for transform/predict context). When True, sets interval_,
        local_y_schema_, local_X_schema_, panel_group_names_, n_features_in_.
    panel_group_names : list of str or None, default=None
        Panel group names to validate/predict for. Only used for forecasters.
        If None, uses all fitted panel groups.
    observation_horizon : int or None, default=None
        Number of previous observations required for inverse transformation.
        When provided with X_p, performs inverse_transform validation.
    **check_params : dict
        Additional validation parameters passed to atomic check functions.

    Returns
    -------
    y_validated : pl.DataFrame or None
        Validated target DataFrame with columns ordered according to schema.
    X_validated : pl.DataFrame or None
        Validated feature DataFrame with columns ordered according to schema.
    panel_group_names : list of str or None
        Validated panel group names (either from parameter or from estimator's fitted state).

    Notes
    -----
    This function implements sklearn's validation philosophy adapted for polars DataFrames
    and time series forecasting. The reset parameter mirrors sklearn's usage:
    - reset=True in fit(): establish the core fitted attributes
    - reset=False in predict/transform(): validate consistency against cache

    For inverse_transform validation, pass both X_p and observation_horizon to validate:
    - X_p is provided when observation_horizon > 0
    - X_p has sufficient length
    - X_p and X are temporally continuous

    Examples
    --------
    In fit method (reset=True)::

        y_val, X_val, _ = validate_data(self, y, X, reset=True)  # Sets fitted attributes

    In predict method (reset=False)::

        y_val, X_val, _ = validate_data(self, y, X, reset=False)  # Checks consistency

    In inverse_transform method::

        _, X_t_val, _ = validate_data(
            self, X_t=X_t, reset=False,
            X_p=X_p, observation_horizon=self.observation_horizon
        )  # Validates inverse_transform requirements

    """
    # Check if this is a transformer based on estimator_type tag
    estimator_type = estimator.__sklearn_tags__().estimator_type

    # Validate X and X_t are mutually exclusive
    if X is not None and X_t is not None:
        raise ValueError(
            "X and X_t are mutually exclusive. Use X for normal transform/predict "
            "and X_t for inverse_transform."
        )

    if reset:
        # Fit context: validate and set interval
        # For transformers, we set fitted attributes here (feature_names_in_, n_features_in_, X_schema_)
        # Note: For forecasters, schema and panel group attributes are set by _set_input_attributes()
        if y is not None:
            interval = check_inputs(y, X)
            estimator.interval_ = interval
        elif X is not None:
            # Transformer-only scenario: validate X and set fitted attributes
            interval = check_inputs(X, None)
            estimator.interval_ = interval

            # Set transformer fitted attributes
            estimator.feature_names_in_ = X.select(~cs.by_name("time")).columns
            estimator.n_features_in_ = len(estimator.feature_names_in_)

            if estimator_type == "transformer":
                estimator.X_schema_ = dict(X.select(~cs.by_name("time")).schema)

        return y, X, None

    if estimator_type == "transformer":
        # Transformer validation: check against X_schema_
        check_is_fitted(estimator, ["X_schema_", "feature_names_in_", "n_features_in_"])

        # Handle inverse_transform validation if X_t is provided
        if X_t is not None:
            # Validate observation_horizon requirement
            if observation_horizon is not None and observation_horizon > 0 and X_p is None:
                raise ValueError(
                    "X_p cannot be None to invert a transform that has observation_horizon > 0. "
                    "Provide the necessary previous untransformed data."
                )

            # Check interval consistency for X_t (transformed data)
            if len(X_t) >= 2:
                X_t_interval = check_interval_consistency(X_t)
            else:
                # Single-step prediction: interval cannot be inferred, skip validation
                X_t_interval = None

            # If X_p is provided, validate it and check intervals match
            if X_p is not None and len(X_p) > 0 and observation_horizon is not None:
                # Validate X_p has sufficient length
                if len(X_p) < observation_horizon:
                    raise ValueError(
                        f"X_p must have at least {observation_horizon} rows (observation_horizon), "
                        f"but has only {len(X_p)} rows."
                    )

                # Check interval consistency for X_p (only if it has more than 1 row)
                if len(X_p) > 1:
                    X_p_interval = check_interval_consistency(X_p)

                    # Ensure intervals match (only if X_t_interval was inferred)
                    if X_t_interval is not None and X_p_interval != X_t_interval:
                        raise ValueError(
                            f"Time intervals do not match: X_p has interval {X_p_interval}, "
                            f"but X_t has interval {X_t_interval}."
                        )

                # Note: We do NOT check temporal continuity between X_p and X_t for inverse_transform
                # X_p is context data from before transformation, not necessarily continuous with X_t

        if X is not None:
            # Normal transform/predict context: validate schema
            X = check_schema(X, estimator.X_schema_)

            # Check interval consistency if requested
            if check_params.get("check_intervals", True) and len(X) >= 2:
                check_interval_consistency(X)

            # Check continuity if _X_observed exists, is non-empty, and check_continuity is not False
            if (
                check_params.get("check_continuity", True)
                and hasattr(estimator, "_X_observed")
                and len(estimator._X_observed) > 0
            ):
                interval = None
                if len(X) >= 2:
                    interval = check_interval_consistency(X)
                check_continuity(
                    estimator._X_observed,
                    X,
                    expected_interval=interval,
                    check_intervals=(interval is not None),
                )

        # For inverse_transform, X_t has transformed column names so we skip schema validation
        # Return X_t if provided (inverse_transform), otherwise X
        return y, X_t if X_t is not None else X, None

    # Forecaster validation: check against local schemas
    check_is_fitted(
        estimator,
        ["local_y_schema_", "local_X_schema_", "global_X_schema_", "panel_group_names_"],
    )

    # Validate and normalize panel_group_names parameter
    panel_group_names = check_panel_group_names(
        fitted_panel_groups=estimator.panel_group_names_,
        requested_panel_groups=panel_group_names,
    )

    # Validate schema and enforce column order
    if y is not None:
        y = check_schema(
            y,
            estimator.local_y_schema_,
            panel_group_names=panel_group_names,
        )

    if X is not None:
        # Handle panel data X (local + global schemas)
        if estimator.panel_group_names_ is not None:
            # Validate local X columns (with panel prefixes)
            if hasattr(estimator, "local_X_schema_") and estimator.local_X_schema_:
                X_local = check_schema(
                    X,
                    estimator.local_X_schema_,
                    panel_group_names=estimator.panel_group_names_,
                )

            # Validate global X columns (no prefixes)
            X_global = None
            if hasattr(estimator, "global_X_schema_") and estimator.global_X_schema_:
                X_global = check_schema(X, estimator.global_X_schema_)

            # Reconstruct X with both local and global columns
            if (
                hasattr(estimator, "local_X_schema_")
                and estimator.local_X_schema_
                and hasattr(estimator, "global_X_schema_")
                and estimator.global_X_schema_
            ):
                X = pl.concat(
                    [X_local, X_global.select(~cs.by_name("time"))],
                    how="horizontal",
                )
            elif hasattr(estimator, "local_X_schema_") and estimator.local_X_schema_:
                X = X_local
            elif hasattr(estimator, "global_X_schema_") and estimator.global_X_schema_:
                X = X_global
        else:
            # Non-panel data: simple schema check
            X = check_schema(X, estimator.local_X_schema_)

    return y, X, panel_group_names


def check_time_column(df: pl.DataFrame) -> None:
    """Validate that time column exists, has proper dtype, no nulls, and is sorted.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to validate.

    Raises
    ------
    ValueError
        If time column is missing, has wrong dtype, contains nulls, or is not sorted.

    """
    if "time" not in df.columns:
        raise ValueError(f"DataFrame must contain 'time' column. Found columns: {list(df.columns)}")

    time_col = df["time"]
    # Check dtype
    if not isinstance(time_col.dtype, (pl.Datetime, pl.Date)):
        raise ValueError(
            f"'time' column must have dtype pl.Datetime or pl.Date, but got {time_col.dtype}"
        )

    # Check for nulls
    if time_col.null_count() > 0:
        raise ValueError(
            f"'time' column contains {time_col.null_count()} null values. "
            "'time' column must not have missing values."
        )

    # Check sorting (ascending)
    if not time_col.is_sorted():
        raise ValueError(
            "'time' column must be sorted in ascending order. Call df.sort('time') to fix."
        )


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

    """
    actual_rows = len(df)
    if actual_rows < min_rows:
        raise ValueError(
            f"{df_name} has {actual_rows} rows but requires at least {min_rows} rows {context}."
        )


def check_panel_group_names(
    fitted_panel_groups: list[str] | None,
    requested_panel_groups: list[str] | None,
) -> list[str] | None:
    """Validate and normalize panel group names for forecaster operations.

    Validates that requested panel groups exist in the fitted forecaster and
    returns the normalized list of groups to use.

    Parameters
    ----------
    fitted_panel_groups : list of str or None
        Panel group names from fitted forecaster (panel_group_names_).
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
    >>> result = check_panel_group_names(fitted_panel_groups=None, requested_panel_groups=None)
    >>> result is None
    True

    >>> # Panel data: use all fitted groups
    >>> check_panel_group_names(
    ...     fitted_panel_groups=["sales", "inventory"],
    ...     requested_panel_groups=None
    ... )
    ['sales', 'inventory']

    >>> # Panel data: validate specific groups
    >>> check_panel_group_names(
    ...     fitted_panel_groups=["sales", "inventory"],
    ...     requested_panel_groups=["sales"]
    ... )
    ['sales']

    """
    # If no groups requested, use all fitted groups
    if requested_panel_groups is None:
        return fitted_panel_groups

    # Validate that requested groups are compatible with fitted state
    if fitted_panel_groups is None:
        raise ValueError(
            "The forecaster was fitted on global data, but `panel_group_names` were provided."
        )

    # Check that all requested groups exist in fitted groups
    missing_groups = set(requested_panel_groups) - set(fitted_panel_groups)
    if missing_groups:
        raise ValueError(
            f"Panel group(s) {sorted(missing_groups)} not found in fitted forecaster. "
            f"Available groups: {sorted(fitted_panel_groups)}."
        )

    return requested_panel_groups


def check_panel_group_names_exist(
    fitted_panel_groups: list[str],
    requested_panel_groups: list[str] | None,
    context: str,
) -> None:
    """Validate all requested panel groups exist in fitted forecaster.

    .. deprecated::
        Use :func:`check_panel_group_names` instead.

    Consolidates duplicated validation in predict, update, reset methods.

    Parameters
    ----------
    fitted_panel_groups : list of str
        Panel group names from fitted forecaster (panel_group_names_).
    requested_panel_groups : list of str or None
        Panel group names requested for operation.
    context : str
        Method name for error message (e.g., "predict", "update", "reset").

    Raises
    ------
    ValueError
        If any requested panel group was not present during fit.

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

    """
    if horizon is None:
        if not allow_none:
            raise ValueError("forecasting_horizon cannot be None")
        return

    if horizon < 1:
        raise ValueError(f"forecasting_horizon must be >= 1, got {horizon}")


def check_time_column(df: pl.DataFrame, df_name: str = "DataFrame") -> None:
    """Validate DataFrame has a 'time' column with Datetime type.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to validate.
    df_name : str, default="DataFrame"
        Name of DataFrame in error message.

    Raises
    ------
    ValueError
        If 'time' column is missing or not Datetime type.

    Examples
    --------
    >>> import polars as pl
    >>> from yohou.utils.validation import check_time_column
    >>> df = pl.DataFrame({"time": ["2023-01-01"], "value": [1]})
    >>> check_time_column(df)  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: 'time' column must be Datetime type, got String.

    """
    if "time" not in df.columns:
        raise ValueError(f"{df_name} must contain a 'time' column.")

    if df["time"].dtype != pl.Datetime:
        raise ValueError(f"'time' column must be Datetime type, got {df['time'].dtype}.")


def check_exogenous_required(
    X: pl.DataFrame | None,
    observation_horizon: int,
    context: str,
) -> None:
    """Validate X is provided when required for recursive prediction.

    Consolidates duplicated validation in point and interval forecasters.

    Parameters
    ----------
    X : pl.DataFrame or None
        Exogenous features.
    observation_horizon : int
        Observation horizon value.
    context : str
        Context for error message (e.g., "predict", "predict_interval").

    Raises
    ------
    ValueError
        If X is None but observation_horizon > 0 (recursive prediction needs X).

    """
    if observation_horizon > 0 and X is None:
        raise ValueError(
            f"For recursive predictions with observation_horizon > 0, "
            f"X must be provided for {context}. "
            f"Got observation_horizon={observation_horizon} but X=None."
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
    ...         start=datetime(2020, 1, 1),
    ...         end=datetime(2020, 1, 5),
    ...         interval="1d",
    ...         eager=True
    ...     ),
    ...     "value": [10, 20, 30, 40, 50]
    ... })
    >>> interval = check_interval_consistency(df)
    >>> interval
    '1d'

    >>> # Invalid: inconsistent intervals
    >>> df_bad = pl.DataFrame({
    ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 4)],
    ...     "value": [10, 20, 30]
    ... })
    >>> check_interval_consistency(df_bad)  # doctest: +SKIP
    ValueError

    See Also
    --------
    check_inputs : Validates multiple DataFrames have matching intervals
    check_continuity : Validates temporal continuity between DataFrames
    add_interval : Add intervals to datetime values

    """
    time_series = df["time"].to_list()

    if len(time_series) < 2:
        raise ValueError("Need at least 2 time points to infer interval")

    # Calculate deltas
    deltas = [time_series[i + 1] - time_series[i] for i in range(len(time_series) - 1)]
    unique_deltas = sorted(set(deltas))

    # Fast path: exact timedelta match - convert to string
    if len(unique_deltas) == 1:
        return _timedelta_to_string(unique_deltas[0])

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


def check_inputs(y: pl.DataFrame, X: pl.DataFrame | None) -> str:
    """Validate that target and feature DataFrames have consistent time intervals.

    Ensures all input DataFrames (target y and exogenous features X) have the same
    uniform time interval. This is required for proper alignment in forecasting
    operations.

    Parameters
    ----------
    y : pl.DataFrame
        Target time series with "time" column.

    X : pl.DataFrame or None
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
    ...     start=datetime(2020, 1, 1),
    ...     end=datetime(2020, 1, 5),
    ...     interval="1d",
    ...     eager=True
    ... )
    >>> y = pl.DataFrame({"time": time_index, "sales": [100, 110, 120, 130, 140]})
    >>> X = pl.DataFrame({"time": time_index, "holiday": [0, 0, 1, 0, 0]})
    >>> interval = check_inputs(y, X)
    >>> interval
    '1d'

    See Also
    --------
    check_interval_consistency : Validates single DataFrame intervals
    validate_column_names : Validates column names don't misuse __ separator

    """
    # Validate column names first
    validate_column_names(y)
    if X is not None:
        validate_column_names(X)

    y_interval = check_interval_consistency(y)
    if X is not None:
        X_interval = check_interval_consistency(X)

        if X_interval != y_interval:
            raise ValueError(
                f"Time interval mismatch: y has interval {y_interval}, but X has interval "
                f"{X_interval}. All inputs must have the same time interval."
            )

    return y_interval


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
    check_inputs : Validates time intervals and calls this function

    """

    # Pattern: allows underscores in group/series names, but not adjacent to __
    # Valid: store_1__sales, my_store__my_sales
    # Invalid: store___sales (underscore adjacent to __), _store__sales, store__sales_
    # Strategy: split on __, check parts don't start/end with _ and are non-empty
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
                f"(e.g., 'sales__store_1'). Please rename columns to avoid using __ "
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
    panel_group_names: list[str] | None = None,
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
    panel_group_names : list[str] or None, default=None
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
    >>> df_panel = pl.DataFrame({
    ...     "panel__s1": [15, 25],
    ...     "time": [1, 2],
    ...     "panel__s0": [10, 20]
    ... })
    >>> expected_schema = {"s0": pl.Int64, "s1": pl.Int64}
    >>> result = check_schema(df_panel, expected_schema, panel_group_names=["panel"])
    >>> list(result.columns)
    ['time', 'panel__s0', 'panel__s1']

    See Also
    --------
    check_inputs : Validates time intervals
    BaseForecaster.update : Uses this function to validate incoming data

    Notes
    -----
    For panel data, this function automatically constructs the expected schema
    with prefixes (e.g., "sales__store_1") from the unprefixed expected_schema.
    The returned DataFrame has columns ordered consistently with the schema.

    """
    # Construct expected column list based on panel_group_names
    if panel_group_names is None:
        # Non-panel data: use schema as-is
        expected_columns = ["time"] + list(expected_schema.keys())
        expected_full_schema = expected_schema
    else:
        # Panel data: construct prefixed schema
        expected_columns = ["time"]
        expected_full_schema = {}
        for group_name in panel_group_names:
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
        raise ValueError(
            f"Schema mismatch. Expected: {expected_full_schema}, got: {incoming_schema}"
        )

    return df


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
    ...     "time": pl.datetime_range(
    ...         datetime(2020, 1, 1), datetime(2020, 1, 3), "1d", eager=True
    ...     ),
    ...     "value": [10, 20, 30]
    ... })
    >>> df2 = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         datetime(2020, 1, 4), datetime(2020, 1, 6), "1d", eager=True
    ...     ),
    ...     "value": [40, 50, 60]
    ... })
    >>> check_continuity(df1, df2, "1d")

    See Also
    --------
    check_interval_consistency : Validates uniform time spacing

    """
    # Skip validation if expected_interval is None (e.g., single-step prediction)
    if expected_interval is None:
        return

    if check_intervals:
        if len(df_p) > 1:
            interval_p = check_interval_consistency(df_p)

            if interval_p != expected_interval:
                raise ValueError(
                    f"Previous DataFrame has interval {interval_p}, but expected interval is "
                    f"{expected_interval}."
                )

        if len(df_n) > 1:
            interval_n = check_interval_consistency(df_n)

            if len(df_p) > 1 and interval_p != interval_n:
                raise ValueError(
                    "Interval mismatch between DataFrames: previous DataFrame has interval "
                    f"{interval_p}, but next DataFrame has interval {interval_n}."
                )

            if interval_n != expected_interval:
                raise ValueError(
                    f"Next DataFrame has interval {interval_n}, "
                    f"but expected interval is {expected_interval}."
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
    else:
        # Fixed interval - can use timedelta comparison
        if interval_td != expected_interval_td:
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
    """Convert a timedelta to string interval format.

    Parameters
    ----------
    td : timedelta
        Timedelta to convert.

    Returns
    -------
    str
        String representation like \"1d\", \"1h\", \"1w\", \"2w\", etc.

    Examples
    --------
    >>> _timedelta_to_string(timedelta(days=1))
    '1d'
    >>> _timedelta_to_string(timedelta(hours=1))
    '1h'
    >>> _timedelta_to_string(timedelta(days=7))
    '7d'

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
        return f"{minutes}min"
    else:
        seconds = int(total_seconds)
        return f"{seconds}s"


def _infer_freq_from_deltas(
    time_series: list[datetime], unique_deltas: list[timedelta]
) -> str | None:
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

    # Daily pattern: uniform day intervals
    if len(unique_deltas) == 1:
        return _timedelta_to_string(unique_deltas[0])

    # Try monthly inference first (handles 1mo, 2mo, 3mo, 6mo, etc.)
    # Monthly patterns: 28-31 days (1mo), 59-62 days (2mo), 89-92 days (3mo), 181-184 days (6mo)
    if (
        28 <= min_delta <= 31
        or 59 <= min_delta <= 62
        or 89 <= min_delta <= 92
        or 181 <= min_delta <= 184
    ):
        freq = _infer_monthly_freq(time_series)
        if freq is not None:
            return freq

    # Yearly patterns: 365-366 days
    if 365 <= min_delta <= 366 and 365 <= max_delta <= 366:
        return "1y"

    return None


def _infer_monthly_freq(time_series: list[datetime]) -> str | None:
    """Infer monthly frequency (1mo, 2mo, 3mo, etc.) by checking month differences.

    Parameters
    ----------
    time_series : list[datetime]
        List of datetime values.

    Returns
    -------
    str or None
        Frequency string like \"1mo\", \"2mo\", \"3mo\", etc., or None if not monthly.

    """
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
    return None


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
    """Parse interval string into (multiplier, unit).

    Parameters
    ----------
    interval : str
        Interval string like \"1d\", \"3mo\", \"2w\".

    Returns
    -------
    tuple[int, str]
        Tuple of (multiplier, unit).

    Examples
    --------
    >>> parse_interval(\"1d\")
    (1, 'd')
    >>> parse_interval(\"3mo\")
    (3, 'mo')
    >>> parse_interval(\"2w\")
    (2, 'w')

    """
    match = re.match(r"(\d+)(mo|q|y|w|d|h|min|s)", interval)
    if not match:
        raise ValueError(f"Invalid interval format: {interval}")
    return int(match.group(1)), match.group(2)


def interval_to_timedelta(interval: str) -> timedelta | None:
    """Convert fixed interval to timedelta, or None for variable intervals.

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
    >>> interval_to_timedelta(\"1d\")
    datetime.timedelta(days=1)
    >>> interval_to_timedelta(\"2h\")
    datetime.timedelta(seconds=7200)
    >>> interval_to_timedelta(\"1mo\") is None
    True

    """
    multiplier, unit = parse_interval(interval)

    if unit == "d":
        return timedelta(days=multiplier)
    elif unit == "h":
        return timedelta(hours=multiplier)
    elif unit == "min":
        return timedelta(minutes=multiplier)
    elif unit == "s":
        return timedelta(seconds=multiplier)
    elif unit == "w":
        return timedelta(weeks=multiplier)
    else:
        # Variable-length interval, cannot convert to timedelta
        return None


def add_interval(start: datetime, interval: str, n: int = 1) -> datetime:
    """Add n intervals to a datetime (handles variable-length intervals).

    Supports multi-period intervals like \"2mo\", \"3mo\", \"6mo\", etc.

    Parameters
    ----------
    start : datetime
        Starting datetime.

    interval : str
        Interval string like \"1d\", \"1mo\", \"3mo\", \"1q\", \"1y\".

    n : int, default=1
        Number of intervals to add.

    Returns
    -------
    datetime
        Result datetime.

    Examples
    --------
    >>> from datetime import datetime
    >>> add_interval(datetime(2020, 1, 15), \"1d\", 5)
    datetime.datetime(2020, 1, 20, 0, 0)
    >>> add_interval(datetime(2020, 1, 31), \"1mo\", 1)
    datetime.datetime(2020, 2, 29, 0, 0)
    >>> add_interval(datetime(2020, 1, 31), \"2mo\", 2)
    datetime.datetime(2020, 5, 31, 0, 0)

    """
    multiplier, unit = parse_interval(interval)
    total_units = multiplier * n

    if unit == "d":
        return start + timedelta(days=total_units)
    elif unit == "h":
        return start + timedelta(hours=total_units)
    elif unit == "min":
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
