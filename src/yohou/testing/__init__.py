"""Testing utilities for Yohou estimators.

This module provides systematic check functions for validating forecasters
and transformers, organized by type and capability.

Modules
-------
transformer : Transformer validation checks (23 functions)
forecaster : Common forecaster validation checks (12 functions)
point : Point forecaster checks (2 functions)
interval : Interval forecaster checks (5 functions)
reduction : Reduction forecaster checks (2 functions)
panel : Panel data/cross-learning checks (3 functions)
splitter : Splitter validation checks (8 functions)
scorer : Scorer validation checks (10 functions)
search : Search CV validation checks (19 functions)
common : Shared checks for metadata routing (2 functions)
generators : Generator functions for systematic testing (5 functions)
metadata_routing : Test utilities for metadata routing validation

Examples
--------
Using the generator functions for systematic testing::

    from yohou.testing import _yield_yohou_forecaster_checks
    from yohou.point import SeasonalNaive

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

# Transformer checks (23 functions)
# Common checks (2 functions)
from .common import (
    check_metadata_routing_default_request,
    check_metadata_routing_get_metadata_routing,
)

# Forecaster checks (12 functions)
from .forecaster import (
    check_clone_preserves_forecaster_params,
    check_fit_sets_forecaster_attributes,
    check_forecaster_methods_call_check_is_fitted,
    check_forecaster_not_fitted_error,
    check_forecaster_tags_accessible_before_fit,
    check_forecaster_tags_match_capabilities,
    check_forecaster_tags_static_after_fit,
    check_forecasting_horizon_validation,
    check_observe_extends_observations,
    check_predict_time_columns,
    check_prediction_types_property,
    check_rewind_propagates_to_transformers,
    check_rewind_replaces_observations,
)

# Generator functions (5 functions)
from .generators import (
    _yield_yohou_forecaster_checks,
    _yield_yohou_scorer_checks,
    _yield_yohou_search_checks,
    _yield_yohou_splitter_checks,
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

# Scorer checks (10 functions)
from .scorer import (
    check_scorer_aggregation_methods,
    check_scorer_component_subselection,
    check_scorer_coverage_rate_subselection,
    check_scorer_lower_is_better,
    check_scorer_methods_call_check_is_fitted,
    check_scorer_panel_subselection,
    check_scorer_parameter_validation,
    check_scorer_prediction_type_compatibility,
    check_scorer_tags_accessible_before_fit,
    check_scorer_tags_match_capabilities,
    check_scorer_tags_static_after_fit,
)

# Search CV checks (19 functions)
from .search import (
    check_grid_search_exhaustive,
    check_grid_search_param_grid_validation,
    check_randomized_search_distributions,
    check_randomized_search_n_iter,
    check_randomized_search_reproducibility,
    check_search_clone_preserves_params,
    check_search_cv_results_structure,
    check_search_error_score_handling,
    check_search_fit_sets_attributes,
    check_search_method_availability,
    check_search_multimetric_scoring,
    check_search_not_fitted_error,
    check_search_observe_delegates,
    check_search_panel_data,
    check_search_predict_delegates,
    check_search_refit_false_no_forecaster,
    check_search_return_train_score,
    check_search_rewind_delegates,
    check_search_score_delegates,
)

# Splitter checks (8 functions)
from .splitter import (
    check_splitter_n_splits_consistency,
    check_splitter_non_overlapping_tests,
    check_splitter_panel_data_support,
    check_splitter_parameter_constraints,
    check_splitter_produces_valid_indices,
    check_splitter_tags_accessible_before_fit,
    check_splitter_tags_match_capabilities,
    check_splitter_tags_static_after_fit,
)
from .transformer import (
    check_feature_names_out_match,
    check_fit_idempotent,
    check_fit_sets_attributes,
    check_fit_transform_equivalence,
    check_insufficient_data_raises,
    check_inverse_observe_transform_identity,
    check_inverse_transform_identity,
    check_inverse_transform_round_trip,
    check_memory_bounded,
    check_observation_horizon_after_fit,
    check_observation_horizon_not_fitted,
    check_observe_concatenates_memory,
    check_observe_transform_equivalence,
    check_observe_transform_sequential_consistency,
    check_panel_data_support,
    check_panel_group_preservation,
    check_rewind_transform_behavior,
    check_rewind_updates_memory,
    check_tags_accessible_before_fit,
    check_tags_match_capabilities,
    check_tags_static_after_fit,
    check_transform_output_structure,
    check_transformer_methods_call_check_is_fitted,
    check_transformer_preserve_dtypes,
    check_transformers_unfitted_stateless,
)

__all__ = [
    # Transformer checks (24)
    "check_feature_names_out_match",
    "check_fit_idempotent",
    "check_fit_sets_attributes",
    "check_fit_transform_equivalence",
    "check_insufficient_data_raises",
    "check_inverse_transform_identity",
    "check_inverse_transform_round_trip",
    "check_inverse_observe_transform_identity",
    "check_memory_bounded",
    "check_observation_horizon_after_fit",
    "check_observation_horizon_not_fitted",
    "check_panel_data_support",
    "check_panel_group_preservation",
    "check_rewind_transform_behavior",
    "check_rewind_updates_memory",
    "check_tags_accessible_before_fit",
    "check_tags_match_capabilities",
    "check_tags_static_after_fit",
    "check_transform_output_structure",
    "check_transformer_methods_call_check_is_fitted",
    "check_transformer_preserve_dtypes",
    "check_transformers_unfitted_stateless",
    "check_observe_concatenates_memory",
    "check_observe_transform_equivalence",
    "check_observe_transform_sequential_consistency",
    # Forecaster checks (12)
    "check_clone_preserves_forecaster_params",
    "check_fit_sets_forecaster_attributes",
    "check_forecaster_methods_call_check_is_fitted",
    "check_forecaster_not_fitted_error",
    "check_forecaster_tags_accessible_before_fit",
    "check_forecaster_tags_match_capabilities",
    "check_forecaster_tags_static_after_fit",
    "check_forecasting_horizon_validation",
    "check_predict_time_columns",
    "check_prediction_types_property",
    "check_rewind_propagates_to_transformers",
    "check_rewind_replaces_observations",
    "check_observe_extends_observations",
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
    # Splitter checks (8)
    "check_splitter_n_splits_consistency",
    "check_splitter_non_overlapping_tests",
    "check_splitter_panel_data_support",
    "check_splitter_parameter_constraints",
    "check_splitter_produces_valid_indices",
    "check_splitter_tags_accessible_before_fit",
    "check_splitter_tags_match_capabilities",
    "check_splitter_tags_static_after_fit",
    # Scorer checks (10)
    "check_scorer_aggregation_methods",
    "check_scorer_component_subselection",
    "check_scorer_coverage_rate_subselection",
    "check_scorer_lower_is_better",
    "check_scorer_methods_call_check_is_fitted",
    "check_scorer_panel_subselection",
    "check_scorer_parameter_validation",
    "check_scorer_prediction_type_compatibility",
    "check_scorer_tags_accessible_before_fit",
    "check_scorer_tags_match_capabilities",
    "check_scorer_tags_static_after_fit",
    # Search CV checks (19)
    "check_grid_search_exhaustive",
    "check_grid_search_param_grid_validation",
    "check_randomized_search_distributions",
    "check_randomized_search_n_iter",
    "check_randomized_search_reproducibility",
    "check_search_clone_preserves_params",
    "check_search_cv_results_structure",
    "check_search_error_score_handling",
    "check_search_fit_sets_attributes",
    "check_search_method_availability",
    "check_search_multimetric_scoring",
    "check_search_not_fitted_error",
    "check_search_panel_data",
    "check_search_predict_delegates",
    "check_search_refit_false_no_forecaster",
    "check_search_rewind_delegates",
    "check_search_return_train_score",
    "check_search_score_delegates",
    "check_search_observe_delegates",
    # Common checks (2)
    "check_metadata_routing_default_request",
    "check_metadata_routing_get_metadata_routing",
    # Generator functions (5)
    "_yield_yohou_forecaster_checks",
    "_yield_yohou_transformer_checks",
    "_yield_yohou_splitter_checks",
    "_yield_yohou_scorer_checks",
    "_yield_yohou_search_checks",
    "_Registry",
    "assert_request_equal",
    "assert_request_is_empty",
    "check_recorded_metadata",
    "record_metadata",
    "record_metadata_not_default",
]
