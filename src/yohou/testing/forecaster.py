"""Check functions for yohou forecasters (common checks).

This module provides validation functions for common forecaster behaviors
that apply to all BaseForecaster implementations (both point and interval).
"""

import warnings

import polars as pl
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from yohou.compose import PerVintageActualTransformer
from yohou.preprocessing import FunctionTransformer

from .contract import _safe_equal, check_clone_preserves_params

__all__ = [
    "check_clone_preserves_forecaster_params",
    "check_fit_predict_with_X_forecast",
    "check_fit_predict_with_X_future",
    "check_fit_predict_without_exogenous",
    "check_fit_sets_forecaster_attributes",
    "check_forecaster_methods_call_check_is_fitted",
    "check_forecaster_not_fitted_error",
    "check_forecaster_tags_accessible_before_fit",
    "check_forecaster_tags_match_capabilities",
    "check_forecaster_tags_static_after_fit",
    "check_forecasting_horizon_validation",
    "check_requires_exogenous_warns_on_X_future_X_forecast",
    "check_observe_auto_rederives_step_columns",
    "check_observe_extends_observations",
    "check_observe_predict_interval_with_step_columns",
    "check_observe_predict_with_step_columns",
    "check_predict_time_columns",
    "check_predict_X_forecast_override",
    "check_prediction_types_property",
    "check_rewind_propagates_to_transformers",
    "check_rewind_replaces_observations",
    "check_step_feature_alignment_filters",
]


def _assert_observed_time_consistent(forecaster, phase: str) -> None:
    """Assert ``observed_time_`` matches the last time in the observation buffers.

    Shared precondition for the observe/rewind buffer checks: before the
    mutating call, the forecaster's ``observed_time_`` must equal the last
    ``"time"`` value held in ``_y_observed`` and ``_X_t_observed``. Handles
    both panel (dict-valued buffers) and non-panel (frame-valued) layouts, and
    tolerates ``None`` group buffers (``observation_horizon == 0``).

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster instance, inspected before observe()/rewind().
    phase : str
        Either ``"observe"`` or ``"rewind"``, used only in assertion messages.

    Raises
    ------
    AssertionError
        If ``observed_time_`` disagrees with a buffer's last time.

    """
    observed_time = forecaster.observed_time_

    if forecaster._y_observed is not None:
        if isinstance(forecaster._y_observed, dict):
            first_group = next(iter(forecaster._y_observed.keys()))
            first_group_y = forecaster._y_observed[first_group]
            if first_group_y is not None:
                assert observed_time[first_group] == first_group_y["time"][-1], (
                    f"observed_time_ should match last time in _y_observed before {phase}()"
                )
        else:
            assert observed_time == forecaster._y_observed["time"][-1], (
                f"observed_time_ should match last time in _y_observed before {phase}()"
            )

    if forecaster._X_t_observed is not None:
        if isinstance(forecaster._X_t_observed, dict):
            first_group = next(iter(forecaster._X_t_observed.keys()))
            if forecaster._X_t_observed[first_group] is not None:
                assert observed_time[first_group] == forecaster._X_t_observed[first_group]["time"][-1], (
                    f"observed_time_ should match last time in _X_t_observed before {phase}()"
                )
        else:
            assert observed_time == forecaster._X_t_observed["time"][-1], (
                f"observed_time_ should match last time in _X_t_observed before {phase}()"
            )


def check_fit_sets_forecaster_attributes(
    forecaster,
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None = None,
    forecasting_horizon: int = 3,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
) -> None:
    """Check fit() sets required forecaster attributes.

    Verifies that fit() fully initializes the stateful lifecycle so that
    observe(), rewind(), and predict() can be called immediately afterward:
    the horizon and schema attributes that downstream calls read, the
    observation buffers that observe()/rewind() mutate, and the fitted
    transformer references when transformers are configured. The specific
    attribute names and the step-column contract live in the assertions below.

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance
    y : pl.DataFrame
        Training target data with "time" column
    X_actual : pl.DataFrame, optional
        Training features with "time" column
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast
    X_future : pl.DataFrame, optional
        Known-future features with "time" column
    X_forecast : pl.DataFrame, optional
        External forecasts in tidy format

    Raises
    ------
    AssertionError
        If required attributes are not set after fit()

    """
    forecaster_clone = clone(forecaster)
    forecaster_clone.fit(y, X_actual, forecasting_horizon=forecasting_horizon, X_future=X_future, X_forecast=X_forecast)

    # Check core fitted attributes
    assert hasattr(forecaster_clone, "fit_forecasting_horizon_"), "fit() must set fit_forecasting_horizon_ attribute"
    assert forecaster_clone.fit_forecasting_horizon_ == forecasting_horizon, (
        f"fit_forecasting_horizon_ should be {forecasting_horizon}, got {forecaster_clone.fit_forecasting_horizon_}"
    )

    assert hasattr(forecaster_clone, "interval_"), "fit() must set interval_ attribute (str)"

    assert hasattr(forecaster_clone, "groups_"), "fit() must set groups_ attribute (None or list)"
    assert hasattr(forecaster_clone, "local_y_schema_"), (
        "fit() must set local_y_schema_ attribute (dict[str, pl.DataType])"
    )
    assert hasattr(forecaster_clone, "local_X_actual_schema_"), (
        "fit() must set local_X_actual_schema_ attribute (dict[str, pl.DataType])"
    )
    assert hasattr(forecaster_clone, "shared_X_actual_schema_"), (
        "fit() must set shared_X_actual_schema_ attribute (None or dict[str, pl.DataType])"
    )
    assert hasattr(forecaster_clone, "local_y_t_schema_"), (
        "fit() must set local_y_t_schema_ attribute (None or dict[str, pl.DataType])"
    )
    assert hasattr(forecaster_clone, "local_X_t_schema_"), (
        "fit() must set local_X_t_schema_ attribute (None or dict[str, pl.DataType])"
    )

    # Check observation buffers
    assert hasattr(forecaster_clone, "_y_observed"), "fit() must set _y_observed buffer"
    assert hasattr(forecaster_clone, "_X_t_observed"), "fit() must set _X_t_observed buffer"

    # Check transformer attributes
    if forecaster_clone.target_transformer is not None:
        assert hasattr(forecaster_clone, "target_transformer_"), (
            "fit() must set target_transformer_ when target_transformer provided"
        )

    if forecaster_clone.actual_transformer is not None:
        assert hasattr(forecaster_clone, "actual_transformer_"), (
            "fit() must set actual_transformer_ when actual_transformer provided"
        )

    # Check step column attributes when X_future/X_forecast provided
    if X_future is not None or X_forecast is not None:
        assert hasattr(forecaster_clone, "_step_column_names_"), "fit() must set _step_column_names_ attribute"
        assert len(forecaster_clone._step_column_names_) > 0, (
            "_step_column_names_ should be non-empty when X_future/X_forecast provided"
        )
        if X_future is not None:
            assert hasattr(forecaster_clone, "_X_future_schema_"), (
                "fit() must set _X_future_schema_ when X_future provided"
            )
            assert hasattr(forecaster_clone, "_X_future_raw_"), "fit() must set _X_future_raw_ when X_future provided"
            assert forecaster_clone._X_future_raw_ is not None, (
                "_X_future_raw_ should not be None when X_future provided"
            )
        if X_forecast is not None:
            assert hasattr(forecaster_clone, "_X_forecast_schema_"), (
                "fit() must set _X_forecast_schema_ when X_forecast provided"
            )
            assert hasattr(forecaster_clone, "_X_forecast_raw_"), (
                "fit() must set _X_forecast_raw_ when X_forecast provided"
            )
            assert forecaster_clone._X_forecast_raw_ is not None, (
                "_X_forecast_raw_ should not be None when X_forecast provided"
            )
    elif hasattr(forecaster_clone, "_step_column_names_"):
        assert len(forecaster_clone._step_column_names_) == 0, (
            "_step_column_names_ should be empty when no X_future/X_forecast"
        )


def check_forecaster_not_fitted_error(forecaster, y: pl.DataFrame, X_actual: pl.DataFrame | None = None) -> None:
    """Check accessing fitted attributes before fit() raises NotFittedError.

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance
    y : pl.DataFrame
        Unused; retained for API uniformity with other check functions.
    X_actual : pl.DataFrame, optional
        Unused; retained for API uniformity with other check functions.

    Raises
    ------
    AssertionError
        If NotFittedError is not raised when accessing fitted attributes

    """
    forecaster_clone = clone(forecaster)

    # Should raise NotFittedError when checking if fitted
    try:
        check_is_fitted(forecaster_clone, "fit_forecasting_horizon_")
        raise AssertionError(
            f"{forecaster_clone.__class__.__name__} should raise NotFittedError "
            f"when accessing fit_forecasting_horizon_ before fit()"
        )
    except NotFittedError:
        # Expected behavior
        pass


def check_predict_time_columns(forecaster, y_test: pl.DataFrame, X_actual_test: pl.DataFrame | None = None) -> None:
    """Check predictions have vintage_time and time columns.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster instance
    y_test : pl.DataFrame
        Test target data
    X_actual_test : pl.DataFrame, optional
        Test features

    Raises
    ------
    AssertionError
        If predictions lack required time columns

    """
    forecasting_horizon = min(3, len(y_test))

    # Check if forecaster is an interval forecaster
    if hasattr(forecaster, "predict_interval"):
        y_pred = forecaster.predict_interval(forecasting_horizon=forecasting_horizon)
    else:
        y_pred = forecaster.predict(forecasting_horizon=forecasting_horizon)

    assert "vintage_time" in y_pred.columns, "Predictions must have 'vintage_time' column"
    assert "time" in y_pred.columns, "Predictions must have 'time' column"

    # Validate shapes. Panel forecasters may stack predictions per group, so the
    # total row count is a multiple of the horizon rather than the horizon
    # itself. Only assert the strict single-group row count for non-panel
    # forecasters (groups_ is None); for panel forecasters, require the per-group
    # row count (rows // n_groups) to equal the horizon.
    groups = getattr(forecaster, "groups_", None)
    if groups is None:
        assert len(y_pred) == forecasting_horizon, (
            f"Predictions should have {forecasting_horizon} rows, got {len(y_pred)}"
        )
    else:
        n_groups = len(groups)
        assert len(y_pred) % forecasting_horizon == 0, (
            f"Panel predictions should have a multiple of {forecasting_horizon} rows, got {len(y_pred)}"
        )
        assert len(y_pred) in (forecasting_horizon, n_groups * forecasting_horizon), (
            f"Panel predictions should have {forecasting_horizon} (wide) or "
            f"{n_groups * forecasting_horizon} (stacked) rows, got {len(y_pred)}"
        )

    # Validate time column types
    assert isinstance(y_pred["vintage_time"].dtype, pl.Datetime | pl.Date), (
        "vintage_time must be Datetime or Date dtype"
    )
    assert isinstance(y_pred["time"].dtype, pl.Datetime | pl.Date), "time must be Datetime or Date dtype"


def check_observe_extends_observations(
    forecaster,
    y_observe: pl.DataFrame,
    X_actual_observe: pl.DataFrame | None = None,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
) -> None:
    """Check observe() extends observation buffers correctly.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster instance
    y_observe : pl.DataFrame
        New data for update
    X_actual_observe : pl.DataFrame, optional
        Features for update
    X_future : pl.DataFrame, optional
        Known-future features forwarded to observe()
    X_forecast : pl.DataFrame, optional
        External forecast features forwarded to observe()

    Raises
    ------
    AssertionError
        If observation buffers are not extended correctly

    """
    # Store original buffer length
    original_observed_time = forecaster.observed_time_

    # Precondition: observed_time_ agrees with the observation buffers.
    _assert_observed_time_consistent(forecaster, "observe")

    # Update with new data
    forecaster.observe(y_observe, X_actual_observe, X_future=X_future, X_forecast=X_forecast)

    # Check buffers were extended
    updated_observed_time = forecaster.observed_time_

    # Handle both panel and non-panel data for comparison
    if isinstance(updated_observed_time, dict):
        # Panel data: check all groups were updated
        for group_name in updated_observed_time:
            assert updated_observed_time[group_name] >= original_observed_time[group_name], (
                f"observed_time_ for group {group_name} should be updated"
            )
    else:
        # Non-panel data
        assert updated_observed_time >= original_observed_time, (
            "observed_time_ should be updated to at least the last time in update data"
        )

    if forecaster._y_observed is not None:
        if isinstance(forecaster._y_observed, dict):
            # Panel data
            for group_name, y_obs in forecaster._y_observed.items():
                # _y_observed[group] can be None when observation_horizon == 0
                if y_obs is not None:
                    updated_y_observed_last_time = y_obs["time"][-1]
                    assert updated_y_observed_last_time == updated_observed_time[group_name], (
                        f"Last time in _y_observed['{group_name}'] should match updated observed_time_"
                    )
        else:
            # Non-panel data
            updated_y_observed_last_time = forecaster._y_observed["time"][-1]
            assert updated_y_observed_last_time == updated_observed_time, (
                "Last time in _y_observed should match updated observed_time_ after observe()"
            )

    if forecaster._X_t_observed is not None:
        if isinstance(forecaster._X_t_observed, dict):
            # Panel data
            for group_name, X_t_obs in forecaster._X_t_observed.items():
                if X_t_obs is not None:
                    updated_X_t_observed_last_time = X_t_obs["time"][-1]
                    assert updated_X_t_observed_last_time == updated_observed_time[group_name], (
                        f"Last time in _X_t_observed['{group_name}'] should match updated observed_time_"
                    )
        else:
            # Non-panel data
            updated_X_t_observed_last_time = forecaster._X_t_observed["time"][-1]
            assert updated_X_t_observed_last_time == updated_observed_time, (
                "Last time in _X_t_observed should match updated observed_time_ after observe()"
            )


def check_rewind_replaces_observations(
    forecaster,
    y_reset: pl.DataFrame,
    X_actual_reset: pl.DataFrame | None = None,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
) -> None:
    """Check rewind() replaces observation buffers correctly.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster instance
    y_reset : pl.DataFrame
        New data for reset
    X_actual_reset : pl.DataFrame, optional
        Features for reset
    X_future : pl.DataFrame, optional
        Known-future features forwarded to rewind()
    X_forecast : pl.DataFrame, optional
        External forecast features forwarded to rewind()

    Raises
    ------
    AssertionError
        If observation buffers are not replaced correctly

    """
    # Precondition: observed_time_ agrees with the observation buffers.
    _assert_observed_time_consistent(forecaster, "rewind")

    # Reset to new data
    forecaster.rewind(y_reset, X_actual_reset, X_future=X_future, X_forecast=X_forecast)

    # Check buffers were replaced
    reset_observed_time = forecaster.observed_time_

    # Handle both panel and non-panel data
    if isinstance(reset_observed_time, dict):
        # Panel data: check each group's observed_time matches
        for group_name in reset_observed_time:
            # All groups share the global 'time' column; the expected reset timestamp is the same for all groups.
            assert reset_observed_time[group_name] == y_reset["time"][-1], (
                f"observed_time_['{group_name}'] should be reset to last time in reset data"
            )
    else:
        # Non-panel data
        assert reset_observed_time == y_reset["time"][-1], "observed_time_ should be reset to last time in reset data"

    if forecaster._y_observed is not None:
        if isinstance(forecaster._y_observed, dict):
            # Panel data
            for group_name, y_obs in forecaster._y_observed.items():
                # _y_observed[group] can be None when observation_horizon == 0
                if y_obs is not None:
                    reset_y_observed_last_time = y_obs["time"][-1]
                    assert reset_y_observed_last_time == reset_observed_time[group_name], (
                        f"Last time in _y_observed['{group_name}'] should match reset observed_time_"
                    )
        else:
            # Non-panel data
            reset_y_observed_last_time = forecaster._y_observed["time"][-1]
            assert reset_y_observed_last_time == reset_observed_time, (
                "Last time in _y_observed should match reset observed_time_ after rewind()"
            )

    if forecaster._X_t_observed is not None:
        if isinstance(forecaster._X_t_observed, dict):
            # Panel data
            for group_name, X_t_obs in forecaster._X_t_observed.items():
                if X_t_obs is not None:
                    reset_X_t_observed_last_time = X_t_obs["time"][-1]
                    assert reset_X_t_observed_last_time == reset_observed_time[group_name], (
                        f"Last time in _X_t_observed['{group_name}'] should match reset observed_time_"
                    )
        else:
            # Non-panel data
            reset_X_t_observed_last_time = forecaster._X_t_observed["time"][-1]
            assert reset_X_t_observed_last_time == reset_observed_time, (
                "Last time in _X_t_observed should match reset observed_time_ after rewind()"
            )


def check_rewind_propagates_to_transformers(
    forecaster,
    y_reset: pl.DataFrame,
    X_actual_reset: pl.DataFrame | None = None,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
) -> None:
    """Check rewind() propagates to transformers in forecaster.

    When a forecaster with transformers calls rewind(), the transformers
    should also have their observation buffers reset accordingly.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster instance with transformers
    y_reset : pl.DataFrame
        New data for reset
    X_actual_reset : pl.DataFrame, optional
        Features for reset
    X_future : pl.DataFrame, optional
        Known-future features forwarded to rewind()
    X_forecast : pl.DataFrame, optional
        External forecast features forwarded to rewind()

    Raises
    ------
    AssertionError
        If transformers are not properly reset

    """
    # Check if forecaster has transformers (target_transformer or actual_transformer)
    if not hasattr(forecaster, "target_transformer_") and not hasattr(
        forecaster, "actual_transformer_"
    ):  # pragma: no branch - the generator only yields this check for transformer-using forecasters
        return  # Nothing to check

    # Rewind the forecaster
    forecaster.rewind(y_reset, X_actual=X_actual_reset, X_future=X_future, X_forecast=X_forecast)

    # Check target transformer is reset
    if hasattr(forecaster, "target_transformer_") and forecaster.target_transformer_ is not None:
        if isinstance(forecaster.target_transformer_, dict):
            # Panel data - check each transformer
            for group_name, transformer in forecaster.target_transformer_.items():
                if (
                    hasattr(transformer, "_X_observed")
                    and transformer._X_observed is not None
                    and getattr(transformer, "observation_horizon", 0) > 0
                ):
                    # Transformer should have observation data matching reset data
                    assert len(transformer._X_observed) > 0, (
                        f"Target transformer for group '{group_name}' should have observations after rewind"
                    )
        # Non-panel data
        elif (
            hasattr(forecaster.target_transformer_, "_X_observed")
            and forecaster.target_transformer_._X_observed is not None
            and getattr(forecaster.target_transformer_, "observation_horizon", 0) > 0
        ):
            assert len(forecaster.target_transformer_._X_observed) > 0, (
                "Target transformer should have observations after rewind"
            )

    # Check actual transformer is reset (if exists)
    if hasattr(forecaster, "actual_transformer_") and forecaster.actual_transformer_ is not None:
        if isinstance(forecaster.actual_transformer_, dict):
            # Panel data - check each transformer
            for group_name, transformer in forecaster.actual_transformer_.items():
                if (
                    hasattr(transformer, "_X_observed")
                    and transformer._X_observed is not None
                    and getattr(transformer, "observation_horizon", 0) > 0
                ):
                    assert len(transformer._X_observed) > 0, (
                        f"Actual transformer for group '{group_name}' should have observations after rewind"
                    )
        # Non-panel data
        elif (
            hasattr(forecaster.actual_transformer_, "_X_observed")
            and forecaster.actual_transformer_._X_observed is not None
            and getattr(forecaster.actual_transformer_, "observation_horizon", 0) > 0
        ):
            assert len(forecaster.actual_transformer_._X_observed) > 0, (
                "Actual transformer should have observations after rewind"
            )


def check_forecasting_horizon_validation(
    forecaster,
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None = None,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
) -> None:
    """Check forecasting_horizon < 1 raises ValueError.

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance
    y : pl.DataFrame
        Training target data
    X_actual : pl.DataFrame, optional
        Training features
    X_future : pl.DataFrame, optional
        Known-future features forwarded to fit()
    X_forecast : pl.DataFrame, optional
        External forecast features forwarded to fit()

    Raises
    ------
    AssertionError
        If invalid horizon doesn't raise ValueError

    """
    forecaster_clone = clone(forecaster)

    # Test horizon = 0
    try:
        forecaster_clone.fit(y, X_actual, forecasting_horizon=0, X_future=X_future, X_forecast=X_forecast)
        raise AssertionError(f"{forecaster_clone.__class__.__name__} should raise ValueError for forecasting_horizon=0")
    except ValueError as e:
        assert "forecasting_horizon" in str(e).lower() or "positive" in str(e).lower(), (
            f"ValueError should mention forecasting_horizon, got: {e}"
        )

    # Test negative horizon
    forecaster_clone = clone(forecaster)
    try:
        forecaster_clone.fit(y, X_actual, forecasting_horizon=-1, X_future=X_future, X_forecast=X_forecast)
        raise AssertionError(
            f"{forecaster_clone.__class__.__name__} should raise ValueError for forecasting_horizon=-1"
        )
    except ValueError as e:
        assert "forecasting_horizon" in str(e).lower() or "positive" in str(e).lower(), (
            f"ValueError should mention forecasting_horizon, got: {e}"
        )


def check_prediction_types_property(forecaster) -> None:
    """Check forecaster_type tag is None or a frozenset of known prediction types.

    Parameters
    ----------
    forecaster : BaseForecaster
        Forecaster instance (fitted or unfitted)

    Raises
    ------
    AssertionError
        If forecaster_type tag is not valid

    """
    tags = forecaster.__sklearn_tags__()
    forecaster_type = tags.forecaster_tags.forecaster_type if tags.forecaster_tags else None

    valid_elements = {"point", "interval", "class_proba"}
    if forecaster_type is not None:
        assert isinstance(forecaster_type, frozenset), (
            f"forecaster_type tag should be a frozenset or None, got {type(forecaster_type).__name__}"
        )
        assert forecaster_type.issubset(valid_elements), (
            f"forecaster_type tag should only contain {valid_elements}, got {forecaster_type}"
        )


def check_clone_preserves_forecaster_params(forecaster) -> None:
    """Check sklearn's clone() preserves init parameters.

    Delegates the shallow-param and nested-estimator contract to the shared,
    family-agnostic ``check_clone_preserves_params`` (which uses ``_safe_equal``
    and so tolerates DataFrame/array-valued params whose ``==`` is not a bool).
    Adds the forecaster-specific check for ``(name, estimator, columns)``
    3-tuples (e.g. ``ColumnForecaster``): the trailing ``columns`` element must
    survive clone unchanged.

    Parameters
    ----------
    forecaster : BaseForecaster
        Forecaster instance

    Raises
    ------
    AssertionError
        If cloned forecaster has different parameters

    """
    check_clone_preserves_params(forecaster)

    forecaster_clone = clone(forecaster)
    original_params = forecaster.get_params(deep=False)
    cloned_params = forecaster_clone.get_params(deep=False)

    for key, orig_val in original_params.items():
        cloned_val = cloned_params[key]
        if not (isinstance(orig_val, list) and orig_val and isinstance(orig_val[0], tuple)):
            continue
        for i, (orig_item, cloned_item) in enumerate(zip(orig_val, cloned_val, strict=True)):
            if not (isinstance(orig_item, tuple) and isinstance(cloned_item, tuple)):
                continue
            if len(orig_item) == 3 and len(cloned_item) == 3:
                orig_cols, cloned_cols = orig_item[-1], cloned_item[-1]
                assert _safe_equal(orig_cols, cloned_cols), (
                    f"Parameter {key}[{i}] columns: {cloned_cols!r} != {orig_cols!r}"
                )

    assert forecaster_clone is not forecaster, "clone() should create new instance"


def check_forecaster_tags_accessible_before_fit(forecaster) -> None:
    """Check __sklearn_tags__() is accessible before fit().

    Tags should be static class capabilities, not fitted state.
    They must be accessible before calling fit().

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance

    Raises
    ------
    AssertionError
        If __sklearn_tags__() raises error or is not callable

    """
    forecaster_clone = clone(forecaster)

    assert hasattr(forecaster_clone, "__sklearn_tags__"), (
        f"{forecaster_clone.__class__.__name__} must implement __sklearn_tags__() method"
    )

    try:
        tags = forecaster_clone.__sklearn_tags__()
    except Exception as e:
        raise AssertionError(
            f"{forecaster_clone.__class__.__name__}.__sklearn_tags__() raised {type(e).__name__}: {e}"
        ) from e

    # Validate tag structure
    assert hasattr(tags, "forecaster_tags"), "Tags must have forecaster_tags attribute"
    assert hasattr(tags, "input_tags"), "Tags must have input_tags attribute"


def check_forecaster_tags_static_after_fit(
    forecaster,
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None = None,
    forecasting_horizon: int = 3,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
) -> None:
    """Check forecaster tags remain static after fit().

    Tags represent capabilities, not fitted state. They should have
    the same values before and after fit().

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance
    y : pl.DataFrame
        Training target data
    X_actual : pl.DataFrame, optional
        Training features
    forecasting_horizon : int, default=3
        Forecasting horizon
    X_future : pl.DataFrame, optional
        Known-future features forwarded to fit().
    X_forecast : pl.DataFrame, optional
        External forecast features forwarded to fit().

    Raises
    ------
    AssertionError
        If tags change after fit()

    """
    forecaster_clone = clone(forecaster)

    # Get tags before fit
    tags_before = forecaster_clone.__sklearn_tags__()
    forecaster_type_before = tags_before.forecaster_tags.forecaster_type if tags_before.forecaster_tags else None
    stateful_before = tags_before.forecaster_tags.stateful if tags_before.forecaster_tags else None
    uses_reduction_before = tags_before.forecaster_tags.uses_reduction if tags_before.forecaster_tags else None
    supports_panel_data_before = (
        tags_before.forecaster_tags.supports_panel_data if tags_before.forecaster_tags else None
    )

    # Fit the forecaster
    forecaster_clone.fit(y, X_actual, forecasting_horizon=forecasting_horizon, X_future=X_future, X_forecast=X_forecast)

    # Get tags after fit
    tags_after = forecaster_clone.__sklearn_tags__()
    forecaster_type_after = tags_after.forecaster_tags.forecaster_type if tags_after.forecaster_tags else None
    stateful_after = tags_after.forecaster_tags.stateful if tags_after.forecaster_tags else None
    uses_reduction_after = tags_after.forecaster_tags.uses_reduction if tags_after.forecaster_tags else None
    supports_panel_data_after = tags_after.forecaster_tags.supports_panel_data if tags_after.forecaster_tags else None

    # Verify tags didn't change
    assert forecaster_type_before == forecaster_type_after, (
        f"forecaster_type tag changed after fit: {forecaster_type_before} -> {forecaster_type_after}"
    )
    assert stateful_before == stateful_after, f"stateful tag changed after fit: {stateful_before} -> {stateful_after}"
    assert uses_reduction_before == uses_reduction_after, (
        f"uses_reduction tag changed after fit: {uses_reduction_before} -> {uses_reduction_after}"
    )
    assert supports_panel_data_before == supports_panel_data_after, (
        f"supports_panel_data tag changed after fit: {supports_panel_data_before} -> {supports_panel_data_after}"
    )


def check_forecaster_tags_match_capabilities(forecaster) -> None:
    """Check forecaster tags accurately reflect capabilities.

    Validates that tag values match actual forecaster behavior:
    - forecaster_type is consistent with the forecaster's prediction_types,
      checked only when the forecaster exposes a prediction_types attribute
      (a no-op for standard forecasters, which do not)
    - uses_reduction tag matches estimator attribute
    - uses_target_transformer matches target_transformer parameter
    - uses_actual_transformer matches actual_transformer parameter
    - uses_forecast_transformer matches forecast_transformer parameter

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster instance

    Raises
    ------
    AssertionError
        If tags don't match actual capabilities

    """
    tags = forecaster.__sklearn_tags__()

    if not tags.forecaster_tags:
        return

    # Check forecaster_type matches prediction_types
    if hasattr(forecaster, "prediction_types"):
        pred_types = forecaster.prediction_types
        forecaster_type = tags.forecaster_tags.forecaster_type

        if forecaster_type is not None:
            if "point" in pred_types:
                assert "point" in forecaster_type, (
                    f"forecaster_type should contain 'point' for forecaster with point predictions, got {forecaster_type}"
                )
            if "interval" in pred_types:
                assert "interval" in forecaster_type, (
                    f"forecaster_type should contain 'interval' for forecaster with interval predictions, got {forecaster_type}"
                )

    # Check uses_reduction matches estimator attribute
    # Note: Some forecasters have estimator for internal use but don't follow reduction pattern
    # (e.g., FourierSeasonalityForecaster, PolynomialTrendForecaster fit sklearn model directly)
    # Only check if uses_reduction=True implies estimator exists
    uses_reduction = tags.forecaster_tags.uses_reduction

    if uses_reduction:
        has_estimator = hasattr(forecaster, "estimator")
        if not has_estimator:
            raise AssertionError(f"{forecaster.__class__.__name__} has uses_reduction=True but no estimator attribute")

    # Check uses_target_transformer matches parameter
    if hasattr(forecaster, "target_transformer"):
        has_target_transformer = forecaster.target_transformer is not None
        uses_target_transformer = tags.forecaster_tags.uses_target_transformer

        if has_target_transformer != uses_target_transformer:
            raise AssertionError(
                f"{forecaster.__class__.__name__} target_transformer parameter ({has_target_transformer}) "
                f"doesn't match uses_target_transformer tag ({uses_target_transformer})"
            )

    # Check uses_actual_transformer matches parameter
    if hasattr(forecaster, "actual_transformer"):  # pragma: no branch - every forecaster declares the slot
        has_actual_transformer = forecaster.actual_transformer is not None
        uses_actual_transformer = tags.forecaster_tags.uses_actual_transformer

        if has_actual_transformer != uses_actual_transformer:
            raise AssertionError(
                f"{forecaster.__class__.__name__} actual_transformer parameter ({has_actual_transformer}) "
                f"doesn't match uses_actual_transformer tag ({uses_actual_transformer})"
            )

    # Check uses_forecast_transformer matches parameter
    if hasattr(forecaster, "forecast_transformer"):  # pragma: no branch - every forecaster declares the slot
        has_forecast_transformer = forecaster.forecast_transformer is not None
        uses_forecast_transformer = tags.forecaster_tags.uses_forecast_transformer

        if has_forecast_transformer != uses_forecast_transformer:
            raise AssertionError(
                f"{forecaster.__class__.__name__} forecast_transformer parameter ({has_forecast_transformer}) "
                f"doesn't match uses_forecast_transformer tag ({uses_forecast_transformer})"
            )

    # Check uses_step_transformer matches parameter
    if hasattr(forecaster, "step_transformer"):  # pragma: no branch - every forecaster declares the slot
        has_step_transformer = forecaster.step_transformer is not None
        uses_step_transformer = tags.forecaster_tags.uses_step_transformer

        if has_step_transformer != uses_step_transformer:
            raise AssertionError(
                f"{forecaster.__class__.__name__} step_transformer parameter ({has_step_transformer}) "
                f"doesn't match uses_step_transformer tag ({uses_step_transformer})"
            )


def check_forecaster_methods_call_check_is_fitted(
    forecaster,
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None = None,
    forecasting_horizon: int = 3,
) -> None:
    """Check all forecaster methods (except fit) raise NotFittedError when unfitted.

    Validates that predict()/predict_interval(), observe(), rewind(), and
    observe_predict()/observe_predict_interval() methods all check fitted state
    and raise NotFittedError before operating on an unfitted forecaster.

    For class-probability forecasters, predict() is the prediction method
    exercised (they also expose predict_class_proba(), which delegates through
    the same fitted-state guard).

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance
    y : pl.DataFrame
        Training/test target data with "time" column
    X_actual : pl.DataFrame, optional
        Training/test features with "time" column
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast

    Raises
    ------
    AssertionError
        If any method fails to raise NotFittedError when called on unfitted forecaster

    """
    forecaster_clone = clone(forecaster)

    # Length-safe slice so observe()/rewind() receive a non-empty frame even on short y.
    stride = max(1, len(y) // 10)
    y_slice = y[:stride]
    X_actual_slice = X_actual[:stride] if X_actual is not None else None

    # Determine if this is an interval forecaster. Only interval forecasters
    # expose predict_interval(); point and class-proba forecasters do not.
    is_interval = hasattr(forecaster_clone, "predict_interval")

    # Test that predict() or predict_interval() raises NotFittedError when unfitted
    try:
        if is_interval:
            forecaster_clone.predict_interval(
                forecasting_horizon=forecasting_horizon,
                coverage_rates=[0.9],
            )
            method_name = "predict_interval"
        else:
            forecaster_clone.predict(forecasting_horizon=forecasting_horizon)
            method_name = "predict"
        raise AssertionError(
            f"{forecaster_clone.__class__.__name__}.{method_name}() must raise NotFittedError when called on unfitted forecaster"
        )
    except NotFittedError:
        pass  # Expected

    # Test that observe() raises NotFittedError when unfitted
    try:
        forecaster_clone.observe(y_slice, X_actual_slice)
        raise AssertionError(
            f"{forecaster_clone.__class__.__name__}.observe() must raise NotFittedError when called on unfitted forecaster"
        )
    except NotFittedError:
        pass  # Expected

    # Test that rewind() raises NotFittedError when unfitted
    try:
        forecaster_clone.rewind(y_slice, X_actual_slice)
        raise AssertionError(
            f"{forecaster_clone.__class__.__name__}.rewind() must raise NotFittedError when called on unfitted forecaster"
        )
    except NotFittedError:
        pass  # Expected

    # Test that observe_predict() or observe_predict_interval() raises NotFittedError when unfitted
    try:
        if is_interval:
            forecaster_clone.observe_predict_interval(y_slice, X_actual_slice, coverage_rates=[0.9])
            method_name = "observe_predict_interval"
        else:
            forecaster_clone.observe_predict(y_slice, X_actual_slice)
            method_name = "observe_predict"
        raise AssertionError(
            f"{forecaster_clone.__class__.__name__}.{method_name}() must raise NotFittedError when called on unfitted forecaster"
        )
    except NotFittedError:
        pass  # Expected


def check_fit_predict_without_exogenous(
    forecaster,
    y: pl.DataFrame,
    requires_exogenous: bool = False,
    target_as_feature: str | None = "transformed",
    forecasting_horizon: int = 3,
) -> None:
    """Check forecaster behavior when X_actual=None at fit time.

    Validates two clear-cut scenarios based on ``requires_exogenous`` tag
    and ``target_as_feature`` parameter:

    * ``requires_exogenous=False``: fit(y, X_actual=None) succeeds and predict()
      returns valid output.
    * ``requires_exogenous=True`` + ``target_as_feature=None``:
      fit(y, X_actual=None) raises ``ValueError``.

    When ``requires_exogenous=True`` and ``target_as_feature`` is not
    ``None``, the check is skipped because behaviour depends on the
    specific forecaster (some compositions always require X_actual).

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance (will be cloned internally).
    y : pl.DataFrame
        Target time series with ``"time"`` column.
    requires_exogenous : bool, default=False
        Value of the ``requires_exogenous`` forecaster tag.
    target_as_feature : str or None, default="transformed"
        Value of the ``target_as_feature`` forecaster parameter.
    forecasting_horizon : int, default=3
        Forecasting horizon to use for fit/predict.

    """
    forecaster_clone = clone(forecaster)
    name = forecaster_clone.__class__.__name__

    if not requires_exogenous:
        # Forecasters that don't require exogenous must succeed with X_actual=None
        forecaster_clone.fit(y, X_actual=None, forecasting_horizon=forecasting_horizon)
        y_pred = forecaster_clone.predict(forecasting_horizon=forecasting_horizon)
        assert isinstance(y_pred, pl.DataFrame), (
            f"{name}.predict() must return pl.DataFrame after fit(y, X_actual=None), got {type(y_pred).__name__}"
        )
        assert "time" in y_pred.columns, (
            f"{name}.predict() output must contain 'time' column after fit(y, X_actual=None)"
        )
    elif target_as_feature is None:
        # target_as_feature=None and requires_exogenous=True → must raise
        try:
            forecaster_clone.fit(y, X_actual=None, forecasting_horizon=forecasting_horizon)
            raise AssertionError(
                f"{name}.fit(y, X_actual=None) must raise ValueError when target_as_feature=None and requires_exogenous=True"
            )
        except ValueError:
            pass  # Expected
    # else: requires_exogenous=True + target_as_feature is not None
    # → skip: behaviour is forecaster-specific


def check_fit_predict_with_X_future(
    forecaster,
    y_train: pl.DataFrame,
    X_actual_train: pl.DataFrame | None,
    X_future: pl.DataFrame,
    forecasting_horizon: int = 3,
) -> None:
    """Check fit + predict works with X_future provided.

    Validates that fitting with X_future sets ``_X_future_schema_``,
    populates ``_step_column_names_``, and that predict returns valid output.

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance.
    y_train : pl.DataFrame
        Training target data.
    X_actual_train : pl.DataFrame or None
        Training features.
    X_future : pl.DataFrame
        Known-future features with ``"time"`` column.
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast.

    """
    forecaster_clone = clone(forecaster)
    forecaster_clone.fit(
        y_train,
        X_actual_train,
        forecasting_horizon=forecasting_horizon,
        X_future=X_future,
    )

    # Schema set
    assert forecaster_clone._X_future_schema_ is not None, "fit() with X_future must set _X_future_schema_"

    # Step columns populated
    assert len(forecaster_clone._step_column_names_) > 0, (
        "_step_column_names_ should be non-empty after fit with X_future"
    )

    # Raw stored
    assert forecaster_clone._X_future_raw_ is not None, "fit() with X_future must store _X_future_raw_"

    # Predict works
    y_pred = forecaster_clone.predict(forecasting_horizon=forecasting_horizon)
    assert isinstance(y_pred, pl.DataFrame), f"predict() must return pl.DataFrame, got {type(y_pred).__name__}"
    assert "time" in y_pred.columns, "predict() output must contain 'time' column"


def check_fit_predict_with_X_forecast(
    forecaster,
    y_train: pl.DataFrame,
    X_actual_train: pl.DataFrame | None,
    X_forecast: pl.DataFrame,
    forecasting_horizon: int = 3,
) -> None:
    """Check fit + predict works with X_forecast provided.

    Validates that fitting with X_forecast sets ``_X_forecast_schema_``,
    populates ``_step_column_names_``, and that predict returns valid output.

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance.
    y_train : pl.DataFrame
        Training target data.
    X_actual_train : pl.DataFrame or None
        Training features.
    X_forecast : pl.DataFrame
        External forecasts in tidy format.
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast.

    """
    forecaster_clone = clone(forecaster)
    forecaster_clone.fit(
        y_train,
        X_actual_train,
        forecasting_horizon=forecasting_horizon,
        X_forecast=X_forecast,
    )

    # Schema set
    assert forecaster_clone._X_forecast_schema_ is not None, "fit() with X_forecast must set _X_forecast_schema_"

    # Step columns populated
    assert len(forecaster_clone._step_column_names_) > 0, (
        "_step_column_names_ should be non-empty after fit with X_forecast"
    )

    # Raw stored (the full frame at fit; observe/rewind retain per channel)
    assert forecaster_clone._X_forecast_raw_ is not None, "fit() with X_forecast must store _X_forecast_raw_"

    # Predict works
    y_pred = forecaster_clone.predict(forecasting_horizon=forecasting_horizon)
    assert isinstance(y_pred, pl.DataFrame), f"predict() must return pl.DataFrame, got {type(y_pred).__name__}"
    assert "time" in y_pred.columns, "predict() output must contain 'time' column"


def check_predict_X_forecast_override(
    forecaster,
    X_forecast: pl.DataFrame,
    forecasting_horizon: int = 3,
) -> None:
    """Check predict with X_forecast override produces different results.

    Validates that passing X_forecast at predict time overrides the stored
    forecasts without mutating forecaster state.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster instance (fitted with X_forecast).
    X_forecast : pl.DataFrame
        External forecasts for override.
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast.

    """
    # Store a snapshot of the original raw so we can detect in-place mutation.
    original_raw = forecaster._X_forecast_raw_
    if original_raw is not None:
        original_raw = original_raw.clone()

    # Predict with override
    y_pred = forecaster.predict(
        forecasting_horizon=forecasting_horizon,
        X_forecast=X_forecast,
    )

    assert isinstance(y_pred, pl.DataFrame), (
        f"predict() with X_forecast override must return pl.DataFrame, got {type(y_pred).__name__}"
    )

    # State unchanged (predict must not mutate stored raw)
    if original_raw is not None:
        assert forecaster._X_forecast_raw_.equals(original_raw), (
            "predict() with X_forecast override must not mutate _X_forecast_raw_"
        )


def check_mixed_cadence_X_forecast_resolves(
    forecaster,
    y_train: pl.DataFrame,
    X_actual_train: pl.DataFrame | None,
    X_forecast: pl.DataFrame,
    forecasting_horizon: int = 3,
) -> None:
    """Check that a channel on a slower schedule survives step-column derivation.

    A single ``X_forecast`` frame may carry channels issued on different
    schedules. Each base column resolves against its own newest applicable
    vintage, densified before derivation, so a slower channel is not collapsed
    to null by the frame-wide as-of. This check drives every ``X_forecast``
    consuming forecaster with a mixed-cadence frame and confirms the slower
    channel reaches the dense cache rather than being dropped.

    The frame is made mixed by issuing the last base column at only its earliest
    vintage while the rest stay at every vintage. That leaves the slow channel
    null for later observations, so the check runs on a clone using a
    null-tolerant estimator (``DummyRegressor`` with ``nan_handling="pass"``
    where those params exist): the resolution path is what is under test, not the
    estimator's response to nulls, which the fully-covered checks already carry.

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance that consumes ``X_forecast``.
    y_train : pl.DataFrame
        Training target data.
    X_actual_train : pl.DataFrame or None
        Training features.
    X_forecast : pl.DataFrame
        External forecasts in tidy format, used as the uniform baseline from
        which a mixed-cadence variant is derived.
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast.

    """
    value_cols = [c for c in X_forecast.columns if c not in ("vintage_time", "time")]
    vintages = X_forecast.select("vintage_time").unique().sort("vintage_time")["vintage_time"].to_list()
    if len(value_cols) < 2 or len(vintages) < 2:
        return  # need at least two channels and two vintages to express mixed cadence

    slow_col = value_cols[-1]
    earliest = vintages[0]
    mixed = X_forecast.with_columns(
        pl.when(pl.col("vintage_time") == earliest).then(pl.col(slow_col)).otherwise(None).alias(slow_col)
    )

    forecaster_clone = clone(forecaster)
    params = forecaster_clone.get_params()
    overrides = {}
    if "nan_handling" in params:
        overrides["nan_handling"] = "pass"
    if "estimator" in params:
        overrides["estimator"] = DummyRegressor()
    if overrides:
        forecaster_clone.set_params(**overrides)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster_clone.fit(
            y_train,
            X_actual_train,
            forecasting_horizon=forecasting_horizon,
            X_forecast=mixed,
        )
        y_pred = forecaster_clone.predict(forecasting_horizon=forecasting_horizon)

    assert isinstance(y_pred, pl.DataFrame), "predict() on a mixed-cadence frame must return pl.DataFrame"

    # Where the dense cache is observable, densification must have carried the
    # slow channel forward onto vintages beyond its single issue vintage. The raw
    # mixed frame carries it at exactly one vintage; a dense frame that still does
    # is one where the frame-wide as-of would collapse the channel. Comparing
    # against the raw count (rather than asserting non-null) keeps this from
    # passing vacuously on the channel's own issue rows.
    raw_slow_vintages = mixed.filter(pl.col(slow_col).is_not_null())["vintage_time"].n_unique()
    dense = getattr(forecaster_clone, "_X_forecast_t_", None)
    if dense is not None and slow_col in dense.columns and raw_slow_vintages >= 1:
        dense_slow_vintages = dense.filter(pl.col(slow_col).is_not_null())["vintage_time"].n_unique()
        assert dense_slow_vintages > raw_slow_vintages, (
            f"the slower '{slow_col}' channel was not resolved per column: it reaches "
            f"{dense_slow_vintages} vintage(s) after densification, no more than the "
            f"{raw_slow_vintages} it was issued at, so the frame-wide as-of would drop it"
        )


def check_forecast_transformer_slot(
    forecaster,
    y_train: pl.DataFrame,
    X_actual_train: pl.DataFrame | None,
    X_forecast: pl.DataFrame,
    forecasting_horizon: int = 3,
) -> None:
    """Check the ``forecast_transformer`` slot fits and reports its usage tag.

    Sets an identity forecast-kind transformer on a slot-bearing forecaster, fits
    with ``X_forecast``, and asserts the fitted attribute and tag follow. Runs for
    every forecaster family that exposes the slot, not only point reduction.

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance exposing ``forecast_transformer``.
    y_train : pl.DataFrame
        Training target data.
    X_actual_train : pl.DataFrame or None
        Training features.
    X_forecast : pl.DataFrame
        External forecasts in tidy format.
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast.

    """
    slot = PerVintageActualTransformer(FunctionTransformer())
    forecaster_clone = clone(forecaster)
    forecaster_clone.set_params(forecast_transformer=slot)

    assert forecaster_clone.__sklearn_tags__().forecaster_tags.uses_forecast_transformer, (
        "uses_forecast_transformer tag must be True when a forecast_transformer is set"
    )

    forecaster_clone.fit(
        y_train,
        X_actual_train,
        forecasting_horizon=forecasting_horizon,
        X_forecast=X_forecast,
    )

    assert hasattr(forecaster_clone, "forecast_transformer_"), (
        "fit() with X_forecast must set forecast_transformer_ when forecast_transformer provided"
    )
    assert forecaster_clone.forecast_transformer_ is not None, (
        "forecast_transformer_ should not be None when the slot is set and X_forecast is provided"
    )


def check_step_transformer_slot(
    forecaster,
    y_train: pl.DataFrame,
    X_actual_train: pl.DataFrame | None,
    X_future: pl.DataFrame,
    forecasting_horizon: int = 3,
) -> None:
    """Check the ``step_transformer`` slot fits, reduces, and reports its usage tag.

    Sets a `StepAggregator` on a slot-bearing forecaster, fits with ``X_future``,
    and asserts the fitted attribute, the tag, and that the tracked step columns
    are the reduced ones rather than the raw ``_step_h`` block. That last
    assertion is what proves the slot is actually applied rather than merely
    accepted: an inert slot would leave the raw block in place.

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance exposing ``step_transformer``.
    y_train : pl.DataFrame
        Training target data.
    X_actual_train : pl.DataFrame or None
        Training features.
    X_future : pl.DataFrame
        Known-future features with a ``"time"`` column.
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast.

    """
    from yohou.preprocessing import StepAggregator  # noqa: PLC0415

    forecaster_clone = clone(forecaster)
    forecaster_clone.set_params(step_transformer=StepAggregator(aggregations=("mean",)))

    assert forecaster_clone.__sklearn_tags__().forecaster_tags.uses_step_transformer, (
        "uses_step_transformer tag must be True when a step_transformer is set"
    )

    forecaster_clone.fit(
        y_train,
        X_actual_train,
        forecasting_horizon=forecasting_horizon,
        X_future=X_future,
    )

    assert hasattr(forecaster_clone, "step_transformer_"), (
        "fit() with X_future must set step_transformer_ when step_transformer provided"
    )
    assert forecaster_clone.step_transformer_ is not None, (
        "step_transformer_ should not be None when the slot is set and X_future is provided"
    )

    tracked = forecaster_clone._step_column_names_
    assert tracked, "_step_column_names_ must be non-empty after fit with X_future"
    assert all(name.endswith("_step_mean") for name in tracked), (
        f"_step_column_names_ must hold the reduced columns, got {sorted(tracked)}; "
        f"a raw _step_h block here means the slot was accepted but never applied"
    )


def _collect_n_features(fitted) -> list[int]:
    """Collect ``n_features_in_`` from a fitted estimator container, in order.

    Reduction forecasters hold their fitted estimators differently per prediction
    type: one estimator, a list of them for the per-step strategies, or a nested
    container keyed by bound and coverage rate. Walking the structure keeps the
    alignment check indifferent to which one it was handed.
    """
    found: list[int] = []
    if hasattr(fitted, "n_features_in_"):
        found.append(int(fitted.n_features_in_))
        return found
    if isinstance(fitted, dict):
        for value in fitted.values():
            found.extend(_collect_n_features(value))
        return found
    if isinstance(fitted, list | tuple):
        for value in fitted:
            found.extend(_collect_n_features(value))
    return found


def check_step_feature_alignment_filters(
    forecaster,
    y_train: pl.DataFrame,
    X_actual_train: pl.DataFrame | None = None,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
    forecasting_horizon: int = 3,
) -> None:
    """Check ``step_feature_alignment="matched"`` actually narrows the feature set.

    The failure this exists to catch is silent. ``_filter_step_features`` can only
    drop the step columns it recognizes, and a name mismatch between the recorded
    step column names and the tabularized ones makes it recognize none, so it
    filters nothing and every per-step estimator trains on every step's columns
    while the configuration says otherwise. Nothing raises, the fit succeeds, and
    the forecaster reports the alignment it was asked for. Only the fitted feature
    counts show the difference, so this check compares them directly: ``matched``
    must leave each per-step estimator with strictly fewer features than ``all``.

    Applies to the ``"direct"`` strategy only, which is the one that filters. The
    check fits the forecaster twice, so it needs a forecaster exposing
    ``reduction_strategy`` and ``step_feature_alignment``, and step columns to
    filter (pass ``X_future`` or ``X_forecast``).

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted reduction forecaster exposing ``reduction_strategy`` and
        ``step_feature_alignment``.
    y_train : pl.DataFrame
        Training target data.
    X_actual_train : pl.DataFrame or None, default=None
        Training features.
    X_future : pl.DataFrame or None, default=None
        Known future features. Provide this or ``X_forecast``.
    X_forecast : pl.DataFrame or None, default=None
        External forecasts. Provide this or ``X_future``.
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast. Must exceed 1 for the comparison to
        have any step columns to drop.

    """
    params = forecaster.get_params()
    assert "step_feature_alignment" in params, (
        "check_step_feature_alignment_filters needs a forecaster exposing step_feature_alignment"
    )
    assert X_future is not None or X_forecast is not None, (
        "check_step_feature_alignment_filters needs X_future or X_forecast to derive step columns"
    )
    assert forecasting_horizon > 1, (
        f"forecasting_horizon must exceed 1 to compare alignments, got {forecasting_horizon}"
    )

    def _fit_and_count(alignment: str) -> tuple[list[int], int]:
        clone_ = clone(forecaster)
        clone_.set_params(reduction_strategy="direct", step_feature_alignment=alignment)
        clone_.fit(
            y_train,
            X_actual_train,
            forecasting_horizon=forecasting_horizon,
            X_future=X_future,
            X_forecast=X_forecast,
        )
        return _collect_n_features(clone_.estimator_), len(clone_._step_column_names_)

    counts_all, n_step_cols = _fit_and_count("all")
    counts_matched, _ = _fit_and_count("matched")

    # Guard the comparison itself: with no step columns derived, both fits would
    # agree trivially and the check would pass while testing nothing.
    assert n_step_cols > 0, (
        "fit derived no step columns, so this check cannot tell filtering from its absence; "
        "pass an X_future or X_forecast that covers the horizon"
    )
    assert counts_all and len(counts_all) == len(counts_matched), (
        f"expected the same fitted estimator layout under both alignments, got "
        f"{len(counts_all)} feature counts for 'all' and {len(counts_matched)} for 'matched'"
    )
    offenders = [(i, a, m) for i, (a, m) in enumerate(zip(counts_all, counts_matched, strict=True)) if m >= a]
    assert not offenders, (
        f"step_feature_alignment='matched' did not narrow the feature set: estimator(s) "
        f"{[i for i, _, m in offenders]} saw {[m for _, _, m in offenders]} features against "
        f"{[a for _, a, _ in offenders]} under 'all', with {n_step_cols} step column(s) derived. "
        f"The filter recognized no step column to drop, which is the silent no-op this check exists "
        f"to catch, not a legitimate configuration."
    )


def check_observe_auto_rederives_step_columns(
    forecaster,
    y_observe: pl.DataFrame,
    X_actual_observe: pl.DataFrame | None,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
) -> None:
    """Check observe() re-derives step columns from stored raws.

    After observe, step columns should be re-derived from stored
    ``_X_future_raw_`` / ``_X_forecast_raw_`` (or from provided overrides).

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster instance (fitted with X_future/X_forecast).
    y_observe : pl.DataFrame
        Update observation data.
    X_actual_observe : pl.DataFrame or None
        Update features.
    X_future : pl.DataFrame or None
        Optional X_future override for observe.
    X_forecast : pl.DataFrame or None
        Optional X_forecast override for observe.

    """
    # Verify step columns exist before observe
    assert len(forecaster._step_column_names_) > 0, "Forecaster must have non-empty _step_column_names_ before observe"

    step_cols_before = forecaster._step_column_names_.copy()

    # Observe
    forecaster.observe(y_observe, X_actual_observe, X_future=X_future, X_forecast=X_forecast)

    # Step column names should still be the same set
    assert forecaster._step_column_names_ == step_cols_before, "_step_column_names_ should be preserved after observe"


def check_observe_predict_with_step_columns(
    forecaster,
    y_train: pl.DataFrame,
    X_actual_train: pl.DataFrame | None,
    y_test: pl.DataFrame,
    X_actual_test: pl.DataFrame | None = None,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
    forecasting_horizon: int = 3,
) -> None:
    """Check observe_predict works with step columns (lightweight).

    Runs observe_predict with stride=len(y_test)//2 (2 iterations) and
    validates output structure.

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance.
    y_train : pl.DataFrame
        Training target data.
    X_actual_train : pl.DataFrame or None
        Training features.
    y_test : pl.DataFrame
        Test target data (at least 10 rows).
    X_actual_test : pl.DataFrame or None
        Test features.
    X_future : pl.DataFrame or None
        Known-future features.
    X_forecast : pl.DataFrame or None
        External forecasts.
    forecasting_horizon : int, default=3
        Number of steps ahead.

    """
    forecaster_clone = clone(forecaster)
    forecaster_clone.fit(
        y_train,
        X_actual_train,
        forecasting_horizon=forecasting_horizon,
        X_future=X_future,
        X_forecast=X_forecast,
    )

    stride = max(1, len(y_test) // 2)
    y_pred = forecaster_clone.observe_predict(
        y_test,
        X_actual=X_actual_test,
        forecasting_horizon=forecasting_horizon,
        stride=stride,
        X_future=X_future,
        X_forecast=X_forecast,
    )

    assert isinstance(y_pred, pl.DataFrame), f"observe_predict() must return pl.DataFrame, got {type(y_pred).__name__}"
    assert "time" in y_pred.columns, "observe_predict() output must contain 'time' column"
    assert "vintage_time" in y_pred.columns, "observe_predict() output must contain 'vintage_time' column"
    assert len(y_pred) > 0, "observe_predict() must return non-empty DataFrame"


def check_observe_predict_interval_with_step_columns(
    forecaster,
    y_train: pl.DataFrame,
    X_actual_train: pl.DataFrame | None,
    y_test: pl.DataFrame,
    X_actual_test: pl.DataFrame | None = None,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
    forecasting_horizon: int = 3,
    coverage_rates: list[float] | None = None,
) -> None:
    """Check observe_predict_interval works with step columns (lightweight).

    Runs observe_predict_interval with stride=len(y_test)//2 (2 iterations)
    and validates output structure and per-vintage time sorting.

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted interval forecaster instance.
    y_train : pl.DataFrame
        Training target data.
    X_actual_train : pl.DataFrame or None
        Training features.
    y_test : pl.DataFrame
        Test target data (at least 10 rows).
    X_actual_test : pl.DataFrame or None
        Test features.
    X_future : pl.DataFrame or None
        Known-future features.
    X_forecast : pl.DataFrame or None
        External forecasts.
    forecasting_horizon : int, default=3
        Number of steps ahead.
    coverage_rates : list of float or None, default=None
        Coverage rates for prediction intervals. Defaults to [0.9].

    """
    if coverage_rates is None:
        coverage_rates = [0.9]

    forecaster_clone = clone(forecaster)
    forecaster_clone.fit(
        y_train,
        X_actual_train,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage_rates,
        X_future=X_future,
        X_forecast=X_forecast,
    )

    stride = max(1, len(y_test) // 2)
    y_pred = forecaster_clone.observe_predict_interval(
        y_test,
        X_actual=X_actual_test,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage_rates,
        stride=stride,
        X_future=X_future,
        X_forecast=X_forecast,
    )

    assert isinstance(y_pred, pl.DataFrame), (
        f"observe_predict_interval() must return pl.DataFrame, got {type(y_pred).__name__}"
    )
    assert "time" in y_pred.columns, "observe_predict_interval() output must contain 'time' column"
    assert "vintage_time" in y_pred.columns, "observe_predict_interval() output must contain 'vintage_time' column"
    assert len(y_pred) > 0, "observe_predict_interval() must return non-empty DataFrame"

    # Validate per-vintage time sorting (catches stale observation state bugs)
    for vt in y_pred["vintage_time"].unique():
        vintage = y_pred.filter(pl.col("vintage_time") == vt)
        assert vintage["time"].is_sorted(), f"'time' column within vintage_time={vt} is not sorted in ascending order"


def check_requires_exogenous_warns_on_X_future_X_forecast(
    forecaster,
    y_train: pl.DataFrame,
    X_future: pl.DataFrame | None = None,
    X_forecast: pl.DataFrame | None = None,
    forecasting_horizon: int = 3,
) -> None:
    """Check that a forecaster with requires_exogenous=False warns when X_future/X_forecast provided.

    Forecasters with ``requires_exogenous=False`` should emit a UserWarning
    when X_future or X_forecast is passed to fit(). The check always calls
    ``fit`` with ``X_actual=None`` (it exercises only the step-feature path),
    and any UserWarning emitted during that call satisfies the contract; the
    warning need not specifically mention X_future or X_forecast.

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster with requires_exogenous=False.
    y_train : pl.DataFrame
        Training target data.
    X_future : pl.DataFrame or None
        Known-future features.
    X_forecast : pl.DataFrame or None
        External forecasts.
    forecasting_horizon : int, default=3
        Number of steps ahead.

    """
    forecaster_clone = clone(forecaster)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forecaster_clone.fit(
            y_train,
            X_actual=None,
            forecasting_horizon=forecasting_horizon,
            X_future=X_future,
            X_forecast=X_forecast,
        )

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) > 0, (
        f"{forecaster_clone.__class__.__name__} with requires_exogenous=False "
        f"should emit UserWarning when X_future/X_forecast provided"
    )
