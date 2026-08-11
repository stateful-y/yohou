"""Testing utilities for Yohou estimators.

This module provides systematic check functions for validating forecasters
and transformers, organized by type and capability.

Modules
-------
transformer : Transformer validation checks
forecaster : Common forecaster validation checks
point : Point forecaster checks
interval : Interval forecaster checks
reduction : Reduction forecaster checks
panel : Panel data/cross-learning checks
splitter : Splitter validation checks
scorer : Scorer validation checks
search : Search CV validation checks
common : Shared checks for metadata routing
generators : Generator functions for systematic testing
metadata_routing : Test utilities for metadata routing validation

Examples
--------
Using the generator functions for systematic testing::

    from yohou.testing import _yield_yohou_forecaster_checks
    from yohou.point import SeasonalNaive

    forecaster = SeasonalNaive(seasonality=12)
    forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)

    for check_name, check_func, check_kwargs in _yield_yohou_forecaster_checks(
        forecaster, y_train, X_actual_train, y_test, X_actual_test
    ):
        check_func(forecaster, **check_kwargs)

Using individual check functions::

    from yohou.testing import check_fit_sets_forecaster_attributes

    check_fit_sets_forecaster_attributes(forecaster, y_train, X_actual_train, forecasting_horizon=3)

"""

from .class_proba import (
    check_class_proba_classes_attribute,
    check_class_proba_predict_returns_labels,
    check_class_proba_prediction_bounds,
    check_class_proba_prediction_structure,
    check_class_proba_prediction_sums,
    check_class_proba_prediction_types,
)
from .common import (
    check_metadata_routing_default_request,
    check_metadata_routing_get_metadata_routing,
)
from .composite import (
    _yield_composite_reducer_checks as _yield_composite_reducer_checks,
)
from .composite import (
    check_composite_combination_validation,
    check_composite_rejects_bare_list,
    check_composite_weights_length_validation,
)
from .contract import (
    _yield_composition_contract_checks as _yield_composition_contract_checks,
)
from .contract import (
    _yield_estimator_contract_checks as _yield_estimator_contract_checks,
)
from .contract import (
    check_clone_preserves_params,
    check_composition_clone_deep_clones_components,
    check_composition_nested_param_addressable,
    check_get_set_params_round_trip,
    check_init_no_param_mutation,
)
from .forecaster import (
    check_clone_preserves_forecaster_params,
    check_fit_predict_with_X_forecast,
    check_fit_predict_with_X_future,
    check_fit_predict_without_exogenous,
    check_fit_sets_forecaster_attributes,
    check_forecaster_methods_call_check_is_fitted,
    check_forecaster_not_fitted_error,
    check_forecaster_tags_accessible_before_fit,
    check_forecaster_tags_match_capabilities,
    check_forecaster_tags_static_after_fit,
    check_forecasting_horizon_validation,
    check_mixed_cadence_X_forecast_resolves,
    check_observe_auto_rederives_step_columns,
    check_observe_extends_observations,
    check_observe_predict_interval_with_step_columns,
    check_observe_predict_with_step_columns,
    check_predict_time_columns,
    check_predict_X_forecast_override,
    check_prediction_types_property,
    check_requires_exogenous_warns_on_X_future_X_forecast,
    check_rewind_propagates_to_transformers,
    check_rewind_replaces_observations,
    check_step_feature_alignment_filters,
)
from .generators import (
    _yield_yohou_conformal_adapter_checks as _yield_yohou_conformal_adapter_checks,
)
from .generators import (
    _yield_yohou_forecaster_checks as _yield_yohou_forecaster_checks,
)
from .generators import (
    _yield_yohou_scorer_checks as _yield_yohou_scorer_checks,
)
from .generators import (
    _yield_yohou_search_checks as _yield_yohou_search_checks,
)
from .generators import (
    _yield_yohou_similarity_checks as _yield_yohou_similarity_checks,
)
from .generators import (
    _yield_yohou_splitter_checks as _yield_yohou_splitter_checks,
)
from .generators import (
    _yield_yohou_step_transformer_checks as _yield_yohou_step_transformer_checks,
)
from .generators import (
    _yield_yohou_transformer_checks as _yield_yohou_transformer_checks,
)
from .generators import (
    _yield_yohou_weighter_checks as _yield_yohou_weighter_checks,
)
from .interval import (
    check_coverage_rates_parameter,
    check_coverage_rates_validation,
    check_interval_bounds,
    check_interval_prediction_columns,
    check_interval_prediction_types,
)
from .metadata_routing import (
    assert_request_equal,
    assert_request_is_empty,
    check_recorded_metadata,
    record_metadata,
    record_metadata_not_default,
)
from .panel import (
    check_panel_data,
    check_panel_invalid_group_raises,
    check_panel_single_group,
)
from .point import check_point_prediction_structure, check_point_prediction_types
from .reduction import check_estimator_parameter, check_reduction_strategy
from .scorer import (
    check_scorer_aggregation_methods,
    check_scorer_component_subselection,
    check_scorer_coverage_rate_subselection,
    check_scorer_lower_is_better,
    check_scorer_methods_call_check_is_fitted,
    check_scorer_multi_vintage,
    check_scorer_panel_subselection,
    check_scorer_parameter_validation,
    check_scorer_prediction_type_compatibility,
    check_scorer_tags_accessible_before_fit,
    check_scorer_tags_match_capabilities,
    check_scorer_tags_static_after_fit,
)
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
    check_search_interval_predict_delegates,
    check_search_method_availability,
    check_search_multimetric_scoring,
    check_search_not_fitted_error,
    check_search_observe_delegates,
    check_search_panel_data,
    check_search_predict_delegates,
    check_search_refit_false_no_forecaster,
    check_search_return_train_score,
    check_search_rewind_delegates,
)
from .similarity import (
    check_similarity_methods_call_check_is_fitted,
    check_similarity_metric_params_verbatim,
    check_similarity_predict_matrix_shape,
    check_similarity_to_weights_rows_reserve_mass,
)
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
    check_batch_invariance,
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
    check_transform_drops_warmup_rows,
    check_transform_output_structure,
    check_transformer_methods_call_check_is_fitted,
    check_transformer_preserve_dtypes,
    check_transformers_unfitted_stateless,
)
from .weighter import (
    check_weighter_compute_weights_alignment,
    check_weighter_default_constructible,
    check_weighter_fit_noop_returns_self,
    check_weighter_resolved_array_validation,
    check_weighter_tags_accessible_before_fit,
    check_weighter_tags_static_after_fit,
)

__all__ = [
    "check_class_proba_classes_attribute",
    "check_class_proba_predict_returns_labels",
    "check_class_proba_prediction_bounds",
    "check_class_proba_prediction_structure",
    "check_class_proba_prediction_sums",
    "check_class_proba_prediction_types",
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
    "check_transform_drops_warmup_rows",
    "check_transform_output_structure",
    "check_transformer_methods_call_check_is_fitted",
    "check_transformer_preserve_dtypes",
    "check_transformers_unfitted_stateless",
    "check_observe_concatenates_memory",
    "check_observe_transform_equivalence",
    "check_batch_invariance",
    "check_observe_transform_sequential_consistency",
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
    "check_mixed_cadence_X_forecast_resolves",
    "check_predict_X_forecast_override",
    "check_prediction_types_property",
    "check_rewind_propagates_to_transformers",
    "check_rewind_replaces_observations",
    "check_step_feature_alignment_filters",
    "check_point_prediction_structure",
    "check_point_prediction_types",
    "check_coverage_rates_parameter",
    "check_coverage_rates_validation",
    "check_interval_bounds",
    "check_interval_prediction_columns",
    "check_interval_prediction_types",
    "check_estimator_parameter",
    "check_reduction_strategy",
    "check_panel_data",
    "check_panel_invalid_group_raises",
    "check_panel_single_group",
    "check_splitter_n_splits_consistency",
    "check_splitter_non_overlapping_tests",
    "check_splitter_panel_data_support",
    "check_splitter_parameter_constraints",
    "check_splitter_produces_valid_indices",
    "check_splitter_tags_accessible_before_fit",
    "check_splitter_tags_match_capabilities",
    "check_splitter_tags_static_after_fit",
    "check_scorer_aggregation_methods",
    "check_scorer_component_subselection",
    "check_scorer_coverage_rate_subselection",
    "check_scorer_lower_is_better",
    "check_scorer_methods_call_check_is_fitted",
    "check_scorer_multi_vintage",
    "check_scorer_panel_subselection",
    "check_scorer_parameter_validation",
    "check_scorer_prediction_type_compatibility",
    "check_scorer_tags_accessible_before_fit",
    "check_scorer_tags_match_capabilities",
    "check_scorer_tags_static_after_fit",
    "check_grid_search_exhaustive",
    "check_grid_search_param_grid_validation",
    "check_randomized_search_distributions",
    "check_randomized_search_n_iter",
    "check_randomized_search_reproducibility",
    "check_search_clone_preserves_params",
    "check_search_cv_results_structure",
    "check_search_error_score_handling",
    "check_search_fit_sets_attributes",
    "check_search_interval_predict_delegates",
    "check_search_method_availability",
    "check_search_multimetric_scoring",
    "check_search_not_fitted_error",
    "check_search_panel_data",
    "check_search_predict_delegates",
    "check_search_refit_false_no_forecaster",
    "check_search_rewind_delegates",
    "check_search_return_train_score",
    "check_search_observe_delegates",
    "check_metadata_routing_default_request",
    "check_metadata_routing_get_metadata_routing",
    "check_clone_preserves_params",
    "check_get_set_params_round_trip",
    "check_init_no_param_mutation",
    "check_composition_nested_param_addressable",
    "check_composition_clone_deep_clones_components",
    "check_composite_rejects_bare_list",
    "check_composite_combination_validation",
    "check_composite_weights_length_validation",
    "check_weighter_compute_weights_alignment",
    "check_weighter_fit_noop_returns_self",
    "check_weighter_resolved_array_validation",
    "check_weighter_default_constructible",
    "check_weighter_tags_accessible_before_fit",
    "check_weighter_tags_static_after_fit",
    "check_similarity_predict_matrix_shape",
    "check_similarity_to_weights_rows_reserve_mass",
    "check_similarity_metric_params_verbatim",
    "check_similarity_methods_call_check_is_fitted",
    "assert_request_equal",
    "assert_request_is_empty",
    "check_recorded_metadata",
    "record_metadata",
    "record_metadata_not_default",
]
