# yohou.testing

Systematic check functions for testing custom estimators. Use these to validate that your forecasters, transformers, scorers, and splitters conform to Yohou's API contracts.

**User guide**: See the [Advanced Topics](../user-guide/advanced.md) section for further details on creating custom estimators.

## Generator Functions

::: yohou.testing._yield_yohou_forecaster_checks
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing._yield_yohou_transformer_checks
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing._yield_yohou_splitter_checks
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing._yield_yohou_scorer_checks
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing._yield_yohou_search_checks
    options:
      show_root_heading: true
      show_source: false

## Transformer Checks

::: yohou.testing.check_fit_sets_attributes
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_fit_idempotent
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_fit_transform_equivalence
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_transform_output_structure
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_transformer_preserve_dtypes
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_transformer_methods_call_check_is_fitted
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_transformers_unfitted_stateless
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_observation_horizon_after_fit
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_observation_horizon_not_fitted
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_observe_concatenates_memory
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_rewind_updates_memory
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_rewind_transform_behavior
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_memory_bounded
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_observe_transform_equivalence
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_observe_transform_sequential_consistency
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_inverse_transform_identity
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_inverse_transform_round_trip
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_inverse_observe_transform_identity
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_feature_names_out_match
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_insufficient_data_raises
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_panel_data_support
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_tags_accessible_before_fit
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_tags_match_capabilities
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_tags_static_after_fit
    options:
      show_root_heading: true
      show_source: false

## Forecaster Checks

::: yohou.testing.check_clone_preserves_forecaster_params
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_fit_sets_forecaster_attributes
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_forecaster_methods_call_check_is_fitted
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_forecaster_not_fitted_error
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_forecaster_tags_accessible_before_fit
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_forecaster_tags_match_capabilities
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_forecaster_tags_static_after_fit
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_forecasting_horizon_validation
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_predict_time_columns
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_prediction_types_property
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_rewind_propagates_to_transformers
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_rewind_replaces_observations
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_observe_extends_observations
    options:
      show_root_heading: true
      show_source: false

## Point Forecaster Checks

::: yohou.testing.check_point_prediction_structure
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_point_prediction_types
    options:
      show_root_heading: true
      show_source: false

## Interval Forecaster Checks

::: yohou.testing.check_coverage_rates_parameter
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_coverage_rates_validation
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_interval_bounds
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_interval_prediction_columns
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_interval_prediction_types
    options:
      show_root_heading: true
      show_source: false

## Reduction Forecaster Checks

::: yohou.testing.check_estimator_parameter
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_reduction_strategy
    options:
      show_root_heading: true
      show_source: false

## Panel Data Checks

::: yohou.testing.check_panel_data
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_panel_invalid_group_raises
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_panel_single_group
    options:
      show_root_heading: true
      show_source: false

## Splitter Checks

::: yohou.testing.check_splitter_n_splits_consistency
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_splitter_non_overlapping_tests
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_splitter_panel_data_support
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_splitter_parameter_constraints
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_splitter_produces_valid_indices
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_splitter_tags_accessible_before_fit
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_splitter_tags_match_capabilities
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_splitter_tags_static_after_fit
    options:
      show_root_heading: true
      show_source: false

## Scorer Checks

::: yohou.testing.check_scorer_aggregation_methods
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_scorer_component_subselection
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_scorer_coverage_rate_subselection
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_scorer_lower_is_better
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_scorer_methods_call_check_is_fitted
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_scorer_panel_subselection
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_scorer_parameter_validation
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_scorer_prediction_type_compatibility
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_scorer_tags_accessible_before_fit
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_scorer_tags_match_capabilities
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_scorer_tags_static_after_fit
    options:
      show_root_heading: true
      show_source: false

## Search CV Checks

::: yohou.testing.check_search_clone_preserves_params
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_search_fit_sets_attributes
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_search_not_fitted_error
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_search_method_availability
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_search_predict_delegates
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_search_observe_delegates
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_search_rewind_delegates
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_search_score_delegates
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_search_cv_results_structure
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_search_refit_false_no_forecaster
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_search_return_train_score
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_search_multimetric_scoring
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_search_error_score_handling
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_search_panel_data
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_grid_search_exhaustive
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_grid_search_param_grid_validation
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_randomized_search_n_iter
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_randomized_search_distributions
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_randomized_search_reproducibility
    options:
      show_root_heading: true
      show_source: false

## Common Checks

::: yohou.testing.check_metadata_routing_default_request
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_metadata_routing_get_metadata_routing
    options:
      show_root_heading: true
      show_source: false

## Metadata Routing Utilities

::: yohou.testing.assert_request_equal
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.assert_request_is_empty
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.check_recorded_metadata
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.record_metadata
    options:
      show_root_heading: true
      show_source: false

::: yohou.testing.record_metadata_not_default
    options:
      show_root_heading: true
      show_source: false
