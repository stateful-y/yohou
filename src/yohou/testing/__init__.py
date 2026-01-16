"""Testing utilities for Yohou estimators.

This module provides systematic check functions for validating forecasters
and transformers, organized by type and capability.

Modules
-------
transformer : Transformer validation checks (21 functions)
forecaster : Common forecaster validation checks (12 functions)
point : Point forecaster checks (2 functions)
interval : Interval forecaster checks (5 functions)
reduction : Reduction forecaster checks (2 functions)
panel : Panel data/cross-learning checks (3 functions)
common : Shared checks for metadata routing (2 functions)
generators : Generator functions for systematic testing (2 functions)
metadata_routing : Test utilities for metadata routing validation

Examples
--------
Using the generator functions for systematic testing::

    from yohou.testing import _yield_yohou_forecaster_checks
    from yohou.point_forecaster import SeasonalNaive

    forecaster = SeasonalNaive(seasonality=12)
    forecaster.fit(y_train, X_train, forecasting_horizon=3)

    for check_name, check_func, check_kwargs in _yield_yohou_forecaster_checks(
        forecaster, y_train, X_train, y_test, X_test
    ):
        check_func(forecaster, **check_kwargs)

Using individual check functions::

    from yohou.testing import check_fit_sets_forecaster_attributes

    check_fit_sets_forecaster_attributes(forecaster, y_train, X_train, forecasting_horizon=3)

"""

# Transformer checks (21 functions)
# Common checks (2 functions)
from .common import (
    check_metadata_routing_default_request,
    check_metadata_routing_get_metadata_routing,
)

# Forecaster checks (12 functions)
from .forecaster import (
    check_clone_preserves_forecaster_params,
    check_fit_sets_forecaster_attributes,
    check_forecaster_not_fitted_error,
    check_forecaster_tags_accessible_before_fit,
    check_forecaster_tags_match_capabilities,
    check_forecaster_tags_static_after_fit,
    check_forecasting_horizon_validation,
    check_predict_time_columns,
    check_prediction_types_property,
    check_reset_propagates_to_transformers,
    check_reset_replaces_observations,
    check_update_extends_observations,
)

# Generator functions (2 functions)
from .generators import (
    _yield_yohou_forecaster_checks,
    _yield_yohou_transformer_checks,
)

# Interval forecaster checks (5 functions)
from .interval import (
    check_coverage_rates_parameter,
    check_coverage_rates_validation,
    check_interval_bounds,
    check_interval_prediction_columns,
    check_interval_prediction_types,
)

# Metadata routing utilities
from .metadata_routing import (
    _Registry,
    assert_request_equal,
    assert_request_is_empty,
    check_recorded_metadata,
    record_metadata,
    record_metadata_not_default,
)

# Panel data checks (3 functions)
from .panel import (
    check_panel_data,
    check_panel_invalid_group_raises,
    check_panel_single_group,
)

# Point forecaster checks (2 functions)
from .point import check_point_prediction_structure, check_point_prediction_types

# Reduction forecaster checks (2 functions)
from .reduction import check_estimator_parameter, check_reduction_strategy
from .transformer import (
    check_feature_names_out_match,
    check_fit_idempotent,
    check_fit_sets_attributes,
    check_fit_transform_equivalence,
    check_insufficient_data_raises,
    check_inverse_transform_identity,
    check_inverse_transform_round_trip,
    check_memory_bounded,
    check_observation_horizon_after_fit,
    check_observation_horizon_not_fitted,
    check_panel_data_support,
    check_reset_updates_memory,
    check_tags_accessible_before_fit,
    check_tags_match_capabilities,
    check_tags_static_after_fit,
    check_transform_output_structure,
    check_transformer_preserve_dtypes,
    check_transformers_unfitted_stateless,
    check_update_concatenates_memory,
    check_update_transform_equivalence,
)

__all__ = [
    # Transformer checks (21)
    "check_feature_names_out_match",
    "check_fit_idempotent",
    "check_fit_sets_attributes",
    "check_fit_transform_equivalence",
    "check_insufficient_data_raises",
    "check_inverse_transform_identity",
    "check_inverse_transform_round_trip",
    "check_memory_bounded",
    "check_observation_horizon_after_fit",
    "check_observation_horizon_not_fitted",
    "check_panel_data_support",
    "check_reset_updates_memory",
    "check_tags_accessible_before_fit",
    "check_tags_match_capabilities",
    "check_tags_static_after_fit",
    "check_transform_output_structure",
    "check_transformer_preserve_dtypes",
    "check_transformers_unfitted_stateless",
    "check_update_concatenates_memory",
    "check_update_transform_equivalence",
    # Forecaster checks (12)
    "check_clone_preserves_forecaster_params",
    "check_fit_sets_forecaster_attributes",
    "check_forecaster_not_fitted_error",
    "check_forecaster_tags_accessible_before_fit",
    "check_forecaster_tags_match_capabilities",
    "check_forecaster_tags_static_after_fit",
    "check_forecasting_horizon_validation",
    "check_predict_time_columns",
    "check_prediction_types_property",
    "check_reset_propagates_to_transformers",
    "check_reset_replaces_observations",
    "check_update_extends_observations",
    # Point forecaster checks (2)
    "check_point_prediction_structure",
    "check_point_prediction_types",
    # Interval forecaster checks (5)
    "check_coverage_rates_parameter",
    "check_coverage_rates_validation",
    "check_interval_bounds",
    "check_interval_prediction_columns",
    "check_interval_prediction_types",
    # Reduction forecaster checks (2)
    "check_estimator_parameter",
    "check_reduction_strategy",
    # Panel data checks (3)
    "check_panel_data",
    "check_panel_invalid_group_raises",
    "check_panel_single_group",
    # Common checks (2)
    "check_metadata_routing_default_request",
    "check_metadata_routing_get_metadata_routing",
    # Generator functions (2)
    "_yield_yohou_forecaster_checks",
    "_yield_yohou_transformer_checks",
    # Metadata routing utilities (6)
    "_Registry",
    "assert_request_equal",
    "assert_request_is_empty",
    "check_recorded_metadata",
    "record_metadata",
    "record_metadata_not_default",
]
