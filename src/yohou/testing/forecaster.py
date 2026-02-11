"""Check functions for yohou forecasters (common checks).

This module provides validation functions for common forecaster behaviors
that apply to all BaseForecaster implementations (both point and interval).
"""

try:
    import polars as pl
    from polars.testing import assert_frame_equal
except ImportError as e:
    raise ImportError("polars.testing is required for yohou.testing module. Install with: uv sync --group tests") from e

from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


def check_fit_sets_forecaster_attributes(
    forecaster, y: pl.DataFrame, X: pl.DataFrame | None = None, forecasting_horizon: int = 3
) -> None:
    """Check fit() sets required forecaster attributes.

    Validates that fit() creates all required attributes for forecasters including
    fit_forecasting_horizon_, interval_, panel_group_names_, local_y_schema_,
    observation buffers, and transformer references.

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance
    y : pl.DataFrame
        Training target data with "time" column
    X : pl.DataFrame, optional
        Training features with "time" column
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast

    Raises
    ------
    AssertionError
        If required attributes are not set after fit()

    """
    forecaster_clone = clone(forecaster)
    forecaster_clone.fit(y, X, forecasting_horizon=forecasting_horizon)

    # Check core fitted attributes
    assert hasattr(forecaster_clone, "fit_forecasting_horizon_"), "fit() must set fit_forecasting_horizon_ attribute"
    assert forecaster_clone.fit_forecasting_horizon_ == forecasting_horizon, (
        f"fit_forecasting_horizon_ should be {forecasting_horizon}, got {forecaster_clone.fit_forecasting_horizon_}"
    )

    assert hasattr(forecaster_clone, "interval_"), "fit() must set interval_ attribute (timedelta)"

    assert hasattr(forecaster_clone, "panel_group_names_"), "fit() must set panel_group_names_ attribute (None or list)"
    assert hasattr(forecaster_clone, "local_y_schema_"), (
        "fit() must set local_y_schema_ attribute (dict[str, pl.DataType])"
    )
    assert hasattr(forecaster_clone, "local_X_schema_"), (
        "fit() must set local_X_schema_ attribute (dict[str, pl.DataType])"
    )
    assert hasattr(forecaster_clone, "global_X_schema_"), (
        "fit() must set global_X_schema_ attribute (None or dict[str, pl.DataType])"
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

    if forecaster_clone.feature_transformer is not None:
        assert hasattr(forecaster_clone, "feature_transformer_"), (
            "fit() must set feature_transformer_ when feature_transformer provided"
        )


def check_forecaster_not_fitted_error(forecaster, y: pl.DataFrame, X: pl.DataFrame | None = None) -> None:
    """Check accessing fitted attributes before fit() raises NotFittedError.

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance
    y : pl.DataFrame
        Test target data
    X : pl.DataFrame, optional
        Test features

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


def check_predict_time_columns(forecaster, y_test: pl.DataFrame, X_test: pl.DataFrame | None = None) -> None:
    """Check predictions have observed_time and time columns.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster instance
    y_test : pl.DataFrame
        Test target data
    X_test : pl.DataFrame, optional
        Test features

    Raises
    ------
    AssertionError
        If predictions lack required time columns

    """
    forecasting_horizon = min(3, len(y_test))

    # Check if forecaster is an interval forecaster
    if hasattr(forecaster, "predict_interval"):
        y_pred = forecaster.predict_interval(forecasting_horizon=forecasting_horizon, X=X_test)
    else:
        y_pred = forecaster.predict(forecasting_horizon=forecasting_horizon, X=X_test)

    assert "observed_time" in y_pred.columns, "Predictions must have 'observed_time' column"
    assert "time" in y_pred.columns, "Predictions must have 'time' column"

    # Validate shapes
    assert len(y_pred) == forecasting_horizon, f"Predictions should have {forecasting_horizon} rows, got {len(y_pred)}"

    # Validate time column types
    assert y_pred["observed_time"].dtype == pl.Datetime, "observed_time must be Datetime dtype"
    assert y_pred["time"].dtype == pl.Datetime, "time must be Datetime dtype"


def check_update_extends_observations(
    forecaster,
    y_train: pl.DataFrame,
    y_update: pl.DataFrame,
    X_train: pl.DataFrame | None = None,
    X_update: pl.DataFrame | None = None,
) -> None:
    """Check update() extends observation buffers correctly.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster instance
    y_train : pl.DataFrame
        Original training data
    y_update : pl.DataFrame
        New data for update
    X_train : pl.DataFrame, optional
        Features for training
    X_update : pl.DataFrame, optional
        Features for update

    Raises
    ------
    AssertionError
        If observation buffers are not extended correctly

    """
    # Store original buffer length
    original_observed_time = forecaster.observed_time_

    # Handle both panel (dict) and non-panel (DataFrame or scalar) data
    if forecaster._y_observed is not None:
        if isinstance(forecaster._y_observed, dict):
            # Panel data: observed_time_ is a dict
            # Check the first group as a representative
            first_group = next(iter(forecaster._y_observed.keys()))
            original_y_observed_last_time = forecaster._y_observed[first_group]["time"][-1]
            assert original_observed_time[first_group] == original_y_observed_last_time, (
                "observed_time_ should match last time in _y_observed before update()"
            )
        else:
            # Non-panel data: observed_time_ is a scalar
            original_y_observed_last_time = forecaster._y_observed["time"][-1]
            assert original_observed_time == original_y_observed_last_time, (
                "observed_time_ should match last time in _y_observed before update()"
            )

    if forecaster._X_t_observed is not None:
        if isinstance(forecaster._X_t_observed, dict):
            # Panel data
            first_group = next(iter(forecaster._X_t_observed.keys()))
            if forecaster._X_t_observed[first_group] is not None:
                original_X_t_observed_last_time = forecaster._X_t_observed[first_group]["time"][-1]
                assert original_observed_time[first_group] == original_X_t_observed_last_time, (
                    "observed_time_ should match last time in _X_t_observed before update()"
                )
        else:
            # Non-panel data
            original_X_t_observed_last_time = forecaster._X_t_observed["time"][-1]
            assert original_observed_time == original_X_t_observed_last_time, (
                "observed_time_ should match last time in _X_t_observed before update()"
            )

    # Update with new data
    forecaster.update(y_update, X_update)

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
                updated_y_observed_last_time = y_obs["time"][-1]
                assert updated_y_observed_last_time == updated_observed_time[group_name], (
                    f"Last time in _y_observed['{group_name}'] should match updated observed_time_"
                )
        else:
            # Non-panel data
            updated_y_observed_last_time = forecaster._y_observed["time"][-1]
            assert updated_y_observed_last_time == updated_observed_time, (
                "Last time in _y_observed should match updated observed_time_ after update()"
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
                "Last time in _X_t_observed should match updated observed_time_ after update()"
            )


def check_reset_replaces_observations(
    forecaster,
    y_train: pl.DataFrame,
    y_reset: pl.DataFrame,
    X_train: pl.DataFrame | None = None,
    X_reset: pl.DataFrame | None = None,
) -> None:
    """Check reset() replaces observation buffers correctly.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster instance
    y_train : pl.DataFrame
        Original training data
    y_reset : pl.DataFrame
        New data for reset
    X_train : pl.DataFrame, optional
        Features for training
    X_reset : pl.DataFrame, optional
        Features for reset

    Raises
    ------
    AssertionError
        If observation buffers are not replaced correctly

    """
    # Store original buffer length
    original_observed_time = forecaster.observed_time_

    # Handle both panel (dict) and non-panel (DataFrame or scalar) data
    if forecaster._y_observed is not None:
        if isinstance(forecaster._y_observed, dict):
            # Panel data
            first_group = next(iter(forecaster._y_observed.keys()))
            original_y_observed_last_time = forecaster._y_observed[first_group]["time"][-1]
            assert original_observed_time[first_group] == original_y_observed_last_time, (
                "observed_time_ should match last time in _y_observed before update()"
            )
        else:
            # Non-panel data
            original_y_observed_last_time = forecaster._y_observed["time"][-1]
            assert original_observed_time == original_y_observed_last_time, (
                "observed_time_ should match last time in _y_observed before update()"
            )

    if forecaster._X_t_observed is not None:
        if isinstance(forecaster._X_t_observed, dict):
            # Panel data
            first_group = next(iter(forecaster._X_t_observed.keys()))
            if forecaster._X_t_observed[first_group] is not None:
                original_X_t_observed_last_time = forecaster._X_t_observed[first_group]["time"][-1]
                assert original_observed_time[first_group] == original_X_t_observed_last_time, (
                    "observed_time_ should match last time in _X_t_observed before update()"
                )
        else:
            # Non-panel data
            original_X_t_observed_last_time = forecaster._X_t_observed["time"][-1]
            assert original_observed_time == original_X_t_observed_last_time, (
                "observed_time_ should match last time in _X_t_observed before update()"
            )

    # Reset to new data
    forecaster.reset(y_reset, X_reset)

    # Check buffers were replaced
    reset_observed_time = forecaster.observed_time_

    # Handle both panel and non-panel data
    if isinstance(reset_observed_time, dict):
        # Panel data: check each group's observed_time matches
        for group_name in reset_observed_time:
            # Get expected time from y_reset (last row for this group's column)
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
                reset_y_observed_last_time = y_obs["time"][-1]
                assert reset_y_observed_last_time == reset_observed_time[group_name], (
                    f"Last time in _y_observed['{group_name}'] should match reset observed_time_"
                )
        else:
            # Non-panel data
            reset_y_observed_last_time = forecaster._y_observed["time"][-1]
            assert reset_y_observed_last_time == reset_observed_time, (
                "Last time in _y_observed should match reset observed_time_ after reset()"
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
                "Last time in _X_t_observed should match reset observed_time_ after reset()"
            )


def check_reset_propagates_to_transformers(
    forecaster,
    y_train: pl.DataFrame,
    y_reset: pl.DataFrame,
    X_train: pl.DataFrame | None = None,
    X_reset: pl.DataFrame | None = None,
) -> None:
    """Check reset() propagates to transformers in forecaster.

    When a forecaster with transformers calls reset(), the transformers
    should also have their observation buffers reset accordingly.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster instance with transformers
    y_train : pl.DataFrame
        Original training data
    y_reset : pl.DataFrame
        New data for reset
    X_train : pl.DataFrame, optional
        Features for training
    X_reset : pl.DataFrame, optional
        Features for reset

    Raises
    ------
    AssertionError
        If transformers are not properly reset

    """
    # Check if forecaster has transformers (target_transformer or feature_transformer)
    if not hasattr(forecaster, "target_transformer_") and not hasattr(forecaster, "feature_transformer_"):
        return  # Nothing to check

    # Reset the forecaster
    forecaster.reset(y_reset, X=X_reset)

    # Check target transformer is reset
    if hasattr(forecaster, "target_transformer_") and forecaster.target_transformer_ is not None:
        if isinstance(forecaster.target_transformer_, dict):
            # Panel data - check each transformer
            for group_name, transformer in forecaster.target_transformer_.items():
                if hasattr(transformer, "_X_observed") and transformer._X_observed is not None:
                    # Transformer should have observation data matching reset data
                    if getattr(transformer, "observation_horizon", 0) > 0:
                        assert len(transformer._X_observed) > 0, (
                            f"Target transformer for group '{group_name}' should have observations after reset"
                        )
        # Non-panel data
        elif (
            hasattr(forecaster.target_transformer_, "_X_observed")
            and forecaster.target_transformer_._X_observed is not None
            and getattr(forecaster.target_transformer_, "observation_horizon", 0) > 0
        ):
            assert len(forecaster.target_transformer_._X_observed) > 0, (
                "Target transformer should have observations after reset"
            )

    # Check feature transformer is reset (if exists)
    if hasattr(forecaster, "feature_transformer_") and forecaster.feature_transformer_ is not None:
        if isinstance(forecaster.feature_transformer_, dict):
            # Panel data - check each transformer
            for group_name, transformer in forecaster.feature_transformer_.items():
                if hasattr(transformer, "_X_observed") and transformer._X_observed is not None:
                    if getattr(transformer, "observation_horizon", 0) > 0:
                        assert len(transformer._X_observed) > 0, (
                            f"Feature transformer for group '{group_name}' should have observations after reset"
                        )
        # Non-panel data
        elif (
            hasattr(forecaster.feature_transformer_, "_X_observed")
            and forecaster.feature_transformer_._X_observed is not None
            and getattr(forecaster.feature_transformer_, "observation_horizon", 0) > 0
        ):
            assert len(forecaster.feature_transformer_._X_observed) > 0, (
                "Feature transformer should have observations after reset"
            )


def check_forecasting_horizon_validation(forecaster, y: pl.DataFrame, X: pl.DataFrame | None = None) -> None:
    """Check forecasting_horizon < 1 raises ValueError.

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance
    y : pl.DataFrame
        Training target data
    X : pl.DataFrame, optional
        Training features

    Raises
    ------
    AssertionError
        If invalid horizon doesn't raise ValueError

    """
    forecaster_clone = clone(forecaster)

    # Test horizon = 0
    try:
        forecaster_clone.fit(y, X, forecasting_horizon=0)
        raise AssertionError(f"{forecaster_clone.__class__.__name__} should raise ValueError for forecasting_horizon=0")
    except ValueError as e:
        assert "forecasting_horizon" in str(e).lower() or "positive" in str(e).lower(), (
            f"ValueError should mention forecasting_horizon, got: {e}"
        )

    # Test negative horizon
    forecaster_clone = clone(forecaster)
    try:
        forecaster_clone.fit(y, X, forecasting_horizon=-1)
        raise AssertionError(
            f"{forecaster_clone.__class__.__name__} should raise ValueError for forecasting_horizon=-1"
        )
    except ValueError as e:
        assert "forecasting_horizon" in str(e).lower() or "positive" in str(e).lower(), (
            f"ValueError should mention forecasting_horizon, got: {e}"
        )


def check_prediction_types_property(forecaster) -> None:
    """Check forecaster_type tag is set correctly.

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

    valid_types = {"point", "interval", "both", None}
    assert forecaster_type in valid_types, f"forecaster_type tag should be one of {valid_types}, got {forecaster_type}"


def check_clone_preserves_forecaster_params(forecaster) -> None:
    """Check sklearn's clone() preserves init parameters.

    Enhanced version that handles nested estimators and meta-forecasters like
    Decomposer and ColumnForecaster with list of (name, estimator) tuples.

    Parameters
    ----------
    forecaster : BaseForecaster
        Forecaster instance

    Raises
    ------
    AssertionError
        If cloned forecaster has different parameters

    """
    forecaster_clone = clone(forecaster)

    # Get parameters
    original_params = forecaster.get_params(deep=False)
    cloned_params = forecaster_clone.get_params(deep=False)

    # Check same parameter keys
    assert set(original_params.keys()) == set(cloned_params.keys()), (
        f"clone() should have same parameter keys, got {set(cloned_params.keys())} vs {set(original_params.keys())}"
    )

    # Check parameter values (for nested estimators, check type)
    for key in original_params:
        orig_val = original_params[key]
        cloned_val = cloned_params[key]

        # For None values
        if orig_val is None:
            assert cloned_val is None, f"Parameter {key}: expected None, got {cloned_val}"
        # For list of (name, estimator) tuples (meta-estimators like Decomposer, FeaturePipeline)
        elif isinstance(orig_val, list) and len(orig_val) > 0 and isinstance(orig_val[0], tuple):
            assert isinstance(cloned_val, list), f"Parameter {key}: expected list, got {type(cloned_val)}"
            assert len(orig_val) == len(cloned_val), f"Parameter {key}: different lengths"

            for i, (orig_item, cloned_item) in enumerate(zip(orig_val, cloned_val, strict=False)):
                assert isinstance(orig_item, tuple), f"Parameter {key}[{i}]: expected tuple"
                assert isinstance(cloned_item, tuple), f"Parameter {key}[{i}]: expected tuple"
                assert len(orig_item) == 2, f"Parameter {key}[{i}]: expected (name, estimator) tuple"
                assert len(cloned_item) == 2, f"Parameter {key}[{i}]: expected (name, estimator) tuple"

                orig_name, orig_est = orig_item
                cloned_name, cloned_est = cloned_item

                # Names should match exactly
                assert orig_name == cloned_name, f"Parameter {key}[{i}]: different names {cloned_name} != {orig_name}"

                # Estimators should be different instances but same type
                assert type(orig_est) == type(cloned_est), (
                    f"Parameter {key}[{i}] estimator: different types {type(cloned_est)} vs {type(orig_est)}"
                )
                assert orig_est is not cloned_est, (
                    f"Parameter {key}[{i}] estimator: should be cloned, not same instance"
                )

                # Check estimator params match
                if hasattr(orig_est, "get_params"):
                    orig_est_params = orig_est.get_params(deep=True)
                    cloned_est_params = cloned_est.get_params(deep=True)
                    for param_key in orig_est_params.keys():
                        orig_param = orig_est_params.get(param_key)
                        cloned_param = cloned_est_params.get(param_key)
                        if hasattr(orig_param, "get_params"):
                            assert type(orig_param) == type(cloned_param), (
                                f"Parameter {key}[{i}]__{param_key}: different types"
                            )
                        elif orig_param != cloned_param:
                            assert orig_param == cloned_param, (
                                f"Parameter {key}[{i}]__{param_key}: {cloned_param} != {orig_param}"
                            )
        # For estimator instances, check type and params (recursively)
        elif hasattr(orig_val, "get_params"):
            assert type(orig_val) == type(cloned_val), (
                f"Parameter {key}: different types {type(cloned_val)} vs {type(orig_val)}"
            )
            # Use deep=True to get all nested params, compare them
            orig_deep_params = orig_val.get_params(deep=True)
            cloned_deep_params = cloned_val.get_params(deep=True)

            # Compare only primitive values and types (not object instances)
            for param_key in orig_deep_params.keys():
                orig_param = orig_deep_params.get(param_key)
                cloned_param = cloned_deep_params.get(param_key)

                # Skip comparing estimator instances themselves, just check types
                if hasattr(orig_param, "get_params"):
                    assert type(orig_param) == type(cloned_param), f"Parameter {key}__{param_key}: different types"
                elif orig_param != cloned_param:
                    assert orig_param == cloned_param, f"Parameter {key}__{param_key}: {cloned_param} != {orig_param}"
        # For other values, direct comparison
        else:
            assert orig_val == cloned_val, f"Parameter {key}: {cloned_val} != {orig_val}"

    # Check they are different objects
    assert forecaster_clone is not forecaster, "clone() should create new instance"


def check_forecaster_tags_accessible_before_fit(
    forecaster, y: pl.DataFrame | None = None, X: pl.DataFrame | None = None
) -> None:
    """Check __sklearn_tags__() is accessible before fit().

    Tags should be static class capabilities, not fitted state.
    They must be accessible before calling fit().

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance
    y : pl.DataFrame, optional
        Not used, included for signature consistency
    X : pl.DataFrame, optional
        Not used, included for signature consistency

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
    forecaster, y: pl.DataFrame, X: pl.DataFrame | None = None, forecasting_horizon: int = 3
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
    X : pl.DataFrame, optional
        Training features
    forecasting_horizon : int, default=3
        Forecasting horizon

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
    forecaster_clone.fit(y, X, forecasting_horizon=forecasting_horizon)

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


def check_forecaster_tags_match_capabilities(
    forecaster, y: pl.DataFrame | None = None, X: pl.DataFrame | None = None
) -> None:
    """Check forecaster tags accurately reflect capabilities.

    Validates that tag values match actual forecaster behavior:
    - forecaster_type matches prediction_types property
    - stateful tag matches observation horizon or transformer statefulness
    - uses_reduction tag matches estimator attribute
    - uses_target_transformer matches target_transformer parameter
    - uses_feature_transformer matches feature_transformer parameter

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster instance
    y : pl.DataFrame, optional
        Not used, for consistency
    X : pl.DataFrame, optional
        Not used, for consistency

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

        if "point" in pred_types and "interval" in pred_types:
            assert forecaster_type in ["point", "interval"], (
                f"forecaster_type should be 'point' or 'interval' for dual forecaster, got {forecaster_type}"
            )
        elif "point" in pred_types:
            assert forecaster_type == "point", f"forecaster_type should be 'point', got {forecaster_type}"
        elif "interval" in pred_types:
            assert forecaster_type == "interval", f"forecaster_type should be 'interval', got {forecaster_type}"

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

    # Check uses_feature_transformer matches parameter
    if hasattr(forecaster, "feature_transformer"):
        has_feature_transformer = forecaster.feature_transformer is not None
        uses_feature_transformer = tags.forecaster_tags.uses_feature_transformer

        if has_feature_transformer != uses_feature_transformer:
            raise AssertionError(
                f"{forecaster.__class__.__name__} feature_transformer parameter ({has_feature_transformer}) "
                f"doesn't match uses_feature_transformer tag ({uses_feature_transformer})"
            )


def check_forecaster_methods_call_check_is_fitted(
    forecaster, y: pl.DataFrame, X: pl.DataFrame | None = None, forecasting_horizon: int = 3
) -> None:
    """Check all forecaster methods (except fit) raise NotFittedError when unfitted.

    Validates that predict()/predict_interval(), update(), reset(), and
    update_predict()/update_predict_interval() methods all check fitted state
    and raise NotFittedError before operating on an unfitted forecaster.

    Parameters
    ----------
    forecaster : BaseForecaster
        Unfitted forecaster instance
    y : pl.DataFrame
        Training/test target data with "time" column
    X : pl.DataFrame, optional
        Training/test features with "time" column
    forecasting_horizon : int, default=3
        Number of steps ahead to forecast

    Raises
    ------
    AssertionError
        If any method fails to raise NotFittedError when called on unfitted forecaster

    """
    forecaster_clone = clone(forecaster)

    # Determine if this is a point or interval forecaster
    is_interval = hasattr(forecaster_clone, "predict_interval") and not hasattr(forecaster_clone, "predict")

    # Test that predict() or predict_interval() raises NotFittedError when unfitted
    try:
        if is_interval:
            forecaster_clone.predict_interval(
                forecasting_horizon=forecasting_horizon,
                X=X[50:53] if X is not None else None,
                coverage_rates=[0.9],
            )
            method_name = "predict_interval"
        else:
            forecaster_clone.predict(forecasting_horizon=forecasting_horizon, X=X[50:53] if X is not None else None)
            method_name = "predict"
        raise AssertionError(
            f"{forecaster_clone.__class__.__name__}.{method_name}() must raise NotFittedError when called on unfitted forecaster"
        )
    except NotFittedError:
        pass  # Expected

    # Test that update() raises NotFittedError when unfitted
    try:
        forecaster_clone.update(y[50:53], X[50:53] if X is not None else None)
        raise AssertionError(
            f"{forecaster_clone.__class__.__name__}.update() must raise NotFittedError when called on unfitted forecaster"
        )
    except NotFittedError:
        pass  # Expected

    # Test that reset() raises NotFittedError when unfitted
    try:
        forecaster_clone.reset(y[40:50], X[40:50] if X is not None else None)
        raise AssertionError(
            f"{forecaster_clone.__class__.__name__}.reset() must raise NotFittedError when called on unfitted forecaster"
        )
    except NotFittedError:
        pass  # Expected

    # Test that update_predict() or update_predict_interval() raises NotFittedError when unfitted
    try:
        if is_interval:
            forecaster_clone.update_predict_interval(
                y[50:53], X[50:53] if X is not None else None, coverage_rates=[0.9]
            )
            method_name = "update_predict_interval"
        else:
            forecaster_clone.update_predict(y[50:53], X[50:53] if X is not None else None)
            method_name = "update_predict"
        raise AssertionError(
            f"{forecaster_clone.__class__.__name__}.{method_name}() must raise NotFittedError when called on unfitted forecaster"
        )
    except NotFittedError:
        pass  # Expected
