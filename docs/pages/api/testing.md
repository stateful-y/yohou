---
template: api-submodule.html
---

# yohou.testing

Testing utilities for Yohou estimators.

### Functions

| Name | Description |
|------|-------------|
| [`check_class_proba_classes_attribute`](generated/yohou.testing.check_class_proba_classes_attribute.md) | Check classes_ and n_classes_ attributes are populated correctly after fit. |
| [`check_class_proba_predict_returns_labels`](generated/yohou.testing.check_class_proba_predict_returns_labels.md) | Check predict() returns argmax class labels, not probabilities. |
| [`check_class_proba_prediction_bounds`](generated/yohou.testing.check_class_proba_prediction_bounds.md) | Check all probability values are in [0, 1]. |
| [`check_class_proba_prediction_structure`](generated/yohou.testing.check_class_proba_prediction_structure.md) | Check class-probability predictions have correct column structure. |
| [`check_class_proba_prediction_sums`](generated/yohou.testing.check_class_proba_prediction_sums.md) | Check probabilities sum to 1.0 per row per target (tolerance: 1e-6). |
| [`check_class_proba_prediction_types`](generated/yohou.testing.check_class_proba_prediction_types.md) | Check class-proba forecaster has 'class_proba' in forecaster_type tag. |
| [`check_metadata_routing_default_request`](generated/yohou.testing.check_metadata_routing_default_request.md) | Check that by default metadata routing request is empty. |
| [`check_metadata_routing_get_metadata_routing`](generated/yohou.testing.check_metadata_routing_get_metadata_routing.md) | Check that get_metadata_routing() is implemented correctly. |
| [`check_composite_combination_validation`](generated/yohou.testing.check_composite_combination_validation.md) | Check an unknown ``combination`` raises. |
| [`check_composite_rejects_bare_list`](generated/yohou.testing.check_composite_rejects_bare_list.md) | Check a bare ``[estimator, ...]`` list (no names) is rejected. |
| [`check_composite_weights_length_validation`](generated/yohou.testing.check_composite_weights_length_validation.md) | Check a ``weights`` list of the wrong length raises. |
| [`check_clone_preserves_params`](generated/yohou.testing.check_clone_preserves_params.md) | Check ``clone()`` preserves shallow params and freshens nested estimators. |
| [`check_composition_clone_deep_clones_components`](generated/yohou.testing.check_composition_clone_deep_clones_components.md) | Check ``clone(compositor)`` produces fresh same-type components. |
| [`check_composition_nested_param_addressable`](generated/yohou.testing.check_composition_nested_param_addressable.md) | Check nested ``<name>__<param>`` get/set on a ``_BaseComposition``. |
| [`check_get_set_params_round_trip`](generated/yohou.testing.check_get_set_params_round_trip.md) | Check ``set_params(**get_params(deep=True))`` is a no-op. |
| [`check_init_no_param_mutation`](generated/yohou.testing.check_init_no_param_mutation.md) | Check ``__init__`` stores constructor arguments verbatim. |
| [`check_clone_preserves_forecaster_params`](generated/yohou.testing.check_clone_preserves_forecaster_params.md) | Check sklearn's clone() preserves init parameters. |
| [`check_fit_predict_with_X_forecast`](generated/yohou.testing.check_fit_predict_with_X_forecast.md) | Check fit + predict works with X_forecast provided. |
| [`check_fit_predict_with_X_future`](generated/yohou.testing.check_fit_predict_with_X_future.md) | Check fit + predict works with X_future provided. |
| [`check_fit_predict_without_exogenous`](generated/yohou.testing.check_fit_predict_without_exogenous.md) | Check forecaster behavior when X_actual=None at fit time. |
| [`check_fit_sets_forecaster_attributes`](generated/yohou.testing.check_fit_sets_forecaster_attributes.md) | Check fit() sets required forecaster attributes. |
| [`check_forecaster_methods_call_check_is_fitted`](generated/yohou.testing.check_forecaster_methods_call_check_is_fitted.md) | Check all forecaster methods (except fit) raise NotFittedError when unfitted. |
| [`check_forecaster_not_fitted_error`](generated/yohou.testing.check_forecaster_not_fitted_error.md) | Check accessing fitted attributes before fit() raises NotFittedError. |
| [`check_forecaster_tags_accessible_before_fit`](generated/yohou.testing.check_forecaster_tags_accessible_before_fit.md) | Check __sklearn_tags__() is accessible before fit(). |
| [`check_forecaster_tags_match_capabilities`](generated/yohou.testing.check_forecaster_tags_match_capabilities.md) | Check forecaster tags accurately reflect capabilities. |
| [`check_forecaster_tags_static_after_fit`](generated/yohou.testing.check_forecaster_tags_static_after_fit.md) | Check forecaster tags remain static after fit(). |
| [`check_forecasting_horizon_validation`](generated/yohou.testing.check_forecasting_horizon_validation.md) | Check forecasting_horizon < 1 raises ValueError. |
| [`check_mixed_cadence_X_forecast_resolves`](generated/yohou.testing.check_mixed_cadence_X_forecast_resolves.md) | Check that a channel on a slower schedule survives step-column derivation. |
| [`check_observe_auto_rederives_step_columns`](generated/yohou.testing.check_observe_auto_rederives_step_columns.md) | Check observe() re-derives step columns from stored raws. |
| [`check_observe_extends_observations`](generated/yohou.testing.check_observe_extends_observations.md) | Check observe() extends observation buffers correctly. |
| [`check_observe_predict_interval_with_step_columns`](generated/yohou.testing.check_observe_predict_interval_with_step_columns.md) | Check observe_predict_interval works with step columns (lightweight). |
| [`check_observe_predict_with_step_columns`](generated/yohou.testing.check_observe_predict_with_step_columns.md) | Check observe_predict works with step columns (lightweight). |
| [`check_predict_time_columns`](generated/yohou.testing.check_predict_time_columns.md) | Check predictions have vintage_time and time columns. |
| [`check_predict_X_forecast_override`](generated/yohou.testing.check_predict_X_forecast_override.md) | Check predict with X_forecast override produces different results. |
| [`check_prediction_types_property`](generated/yohou.testing.check_prediction_types_property.md) | Check forecaster_type tag is None or a frozenset of known prediction types. |
| [`check_requires_exogenous_warns_on_X_future_X_forecast`](generated/yohou.testing.check_requires_exogenous_warns_on_X_future_X_forecast.md) | Check that a forecaster with requires_exogenous=False warns when X_future/X_forecast provided. |
| [`check_rewind_propagates_to_transformers`](generated/yohou.testing.check_rewind_propagates_to_transformers.md) | Check rewind() propagates to transformers in forecaster. |
| [`check_rewind_replaces_observations`](generated/yohou.testing.check_rewind_replaces_observations.md) | Check rewind() replaces observation buffers correctly. |
| [`check_step_feature_alignment_filters`](generated/yohou.testing.check_step_feature_alignment_filters.md) | Check ``step_feature_alignment="matched"`` actually narrows the feature set. |
| [`check_coverage_rates_parameter`](generated/yohou.testing.check_coverage_rates_parameter.md) | Check fit_coverage_rates_ fitted attribute is a non-empty list of floats in [0, 1]. |
| [`check_coverage_rates_validation`](generated/yohou.testing.check_coverage_rates_validation.md) | Check invalid coverage_rates raise ValueError during fit and predict. |
| [`check_interval_bounds`](generated/yohou.testing.check_interval_bounds.md) | Check upper >= lower for all coverage rates and time steps. |
| [`check_interval_prediction_columns`](generated/yohou.testing.check_interval_prediction_columns.md) | Check interval predictions have {col}_lower_{rate} and {col}_upper_{rate} format. |
| [`check_interval_prediction_types`](generated/yohou.testing.check_interval_prediction_types.md) | Check interval forecaster has 'interval' in forecaster_type tag. |
| [`assert_request_equal`](generated/yohou.testing.assert_request_equal.md) | Assert metadata request matches expected dictionary. |
| [`assert_request_is_empty`](generated/yohou.testing.assert_request_is_empty.md) | Check if a metadata request dict is empty. |
| [`check_recorded_metadata`](generated/yohou.testing.check_recorded_metadata.md) | Check whether the expected metadata is passed to the object's method. |
| [`record_metadata`](generated/yohou.testing.record_metadata.md) | Store passed metadata to a method of obj. |
| [`check_panel_data`](generated/yohou.testing.check_panel_data.md) | Check cross-learning with panel data predicts all groups by default. |
| [`check_panel_invalid_group_raises`](generated/yohou.testing.check_panel_invalid_group_raises.md) | Check that an invalid group name raises ValueError. |
| [`check_panel_single_group`](generated/yohou.testing.check_panel_single_group.md) | Check cross-learning filters to specified panel group. |
| [`check_point_prediction_structure`](generated/yohou.testing.check_point_prediction_structure.md) | Check point predictions have correct column structure. |
| [`check_point_prediction_types`](generated/yohou.testing.check_point_prediction_types.md) | Check point forecaster has 'point' in forecaster_type tag. |
| [`check_estimator_parameter`](generated/yohou.testing.check_estimator_parameter.md) | Check estimator parameter is sklearn BaseEstimator. |
| [`check_reduction_strategy`](generated/yohou.testing.check_reduction_strategy.md) | Check reduction_strategy is one of 'direct', 'dir-rec', 'multi-output'. |
| [`check_scorer_aggregation_methods`](generated/yohou.testing.check_scorer_aggregation_methods.md) | Check all aggregation_method combinations produce valid output. |
| [`check_scorer_component_subselection`](generated/yohou.testing.check_scorer_component_subselection.md) | Check components filtering works correctly. |
| [`check_scorer_coverage_rate_subselection`](generated/yohou.testing.check_scorer_coverage_rate_subselection.md) | Check coverage parameter filters interval predictions correctly. |
| [`check_scorer_lower_is_better`](generated/yohou.testing.check_scorer_lower_is_better.md) | Check the ``lower_is_better`` tag is a boolean. |
| [`check_scorer_methods_call_check_is_fitted`](generated/yohou.testing.check_scorer_methods_call_check_is_fitted.md) | Check all scorer methods (except fit) raise NotFittedError when unfitted. |
| [`check_scorer_multi_vintage`](generated/yohou.testing.check_scorer_multi_vintage.md) | Check that scorer produces a finite result on multi-vintage input. |
| [`check_scorer_panel_subselection`](generated/yohou.testing.check_scorer_panel_subselection.md) | Check groups filtering works correctly. |
| [`check_scorer_parameter_validation`](generated/yohou.testing.check_scorer_parameter_validation.md) | Check parameter validation raises ValueError for invalid inputs. |
| [`check_scorer_prediction_type_compatibility`](generated/yohou.testing.check_scorer_prediction_type_compatibility.md) | Check scorer works with correct forecaster output type. |
| [`check_scorer_tags_accessible_before_fit`](generated/yohou.testing.check_scorer_tags_accessible_before_fit.md) | Check __sklearn_tags__() is callable on scorer instance. |
| [`check_scorer_tags_match_capabilities`](generated/yohou.testing.check_scorer_tags_match_capabilities.md) | Check tag values match actual scorer behavior. |
| [`check_scorer_tags_static_after_fit`](generated/yohou.testing.check_scorer_tags_static_after_fit.md) | Check tags remain unchanged after fit. |
| [`check_grid_search_exhaustive`](generated/yohou.testing.check_grid_search_exhaustive.md) | Check GridSearchCV evaluates all parameter combinations. |
| [`check_grid_search_param_grid_validation`](generated/yohou.testing.check_grid_search_param_grid_validation.md) | Check param_grid format is validated (dict or list of dicts). |
| [`check_randomized_search_distributions`](generated/yohou.testing.check_randomized_search_distributions.md) | Check scipy.stats distributions work for parameter sampling. |
| [`check_randomized_search_n_iter`](generated/yohou.testing.check_randomized_search_n_iter.md) | Check n_iter controls number of parameter combinations evaluated. |
| [`check_randomized_search_reproducibility`](generated/yohou.testing.check_randomized_search_reproducibility.md) | Check random_state produces same parameter samples. |
| [`check_search_clone_preserves_params`](generated/yohou.testing.check_search_clone_preserves_params.md) | Check sklearn clone() preserves search CV parameters. |
| [`check_search_cv_results_structure`](generated/yohou.testing.check_search_cv_results_structure.md) | Check cv_results_ has required structure. |
| [`check_search_error_score_handling`](generated/yohou.testing.check_search_error_score_handling.md) | Check error_score parameter handles failing fits correctly. |
| [`check_search_fit_sets_attributes`](generated/yohou.testing.check_search_fit_sets_attributes.md) | Check fit() sets required search CV attributes. |
| [`check_search_interval_predict_delegates`](generated/yohou.testing.check_search_interval_predict_delegates.md) | Check predict_interval() works after interval search with refit. |
| [`check_search_method_availability`](generated/yohou.testing.check_search_method_availability.md) | Check @available_if decorator logic with refit=True/False. |
| [`check_search_multimetric_scoring`](generated/yohou.testing.check_search_multimetric_scoring.md) | Check multi-metric scoring with dict scorer works correctly. |
| [`check_search_not_fitted_error`](generated/yohou.testing.check_search_not_fitted_error.md) | Check fitted attributes before fit() raise NotFittedError. |
| [`check_search_observe_delegates`](generated/yohou.testing.check_search_observe_delegates.md) | Check observe() delegates to best_forecaster_.observe() correctly. |
| [`check_search_panel_data`](generated/yohou.testing.check_search_panel_data.md) | Check groups parameter propagates correctly. |
| [`check_search_predict_delegates`](generated/yohou.testing.check_search_predict_delegates.md) | Check predict() delegates to best_forecaster_.predict() correctly. |
| [`check_search_refit_false_no_forecaster`](generated/yohou.testing.check_search_refit_false_no_forecaster.md) | Check refit=False doesn't create best_forecaster_. |
| [`check_search_return_train_score`](generated/yohou.testing.check_search_return_train_score.md) | Check return_train_score=True adds train score keys to cv_results_. |
| [`check_search_rewind_delegates`](generated/yohou.testing.check_search_rewind_delegates.md) | Check rewind() delegates to best_forecaster_.rewind() correctly. |
| [`check_similarity_methods_call_check_is_fitted`](generated/yohou.testing.check_similarity_methods_call_check_is_fitted.md) | Check ``predict``, ``observe``, and ``rewind`` raise ``NotFittedError`` before ``fit``. |
| [`check_similarity_metric_params_verbatim`](generated/yohou.testing.check_similarity_metric_params_verbatim.md) | Check ``metric_params=None`` is stored verbatim (no init-mutation). |
| [`check_similarity_predict_matrix_shape`](generated/yohou.testing.check_similarity_predict_matrix_shape.md) | Check ``predict`` returns an ``(n_pred, n_calib)`` weight matrix. |
| [`check_similarity_to_weights_rows_reserve_mass`](generated/yohou.testing.check_similarity_to_weights_rows_reserve_mass.md) | Check each predicted weight row is non-negative and sums below 1. |
| [`check_splitter_n_splits_consistency`](generated/yohou.testing.check_splitter_n_splits_consistency.md) | Check get_n_splits() matches actual split count. |
| [`check_splitter_non_overlapping_tests`](generated/yohou.testing.check_splitter_non_overlapping_tests.md) | Check test sets don't overlap if produces_non_overlapping_tests=True. |
| [`check_splitter_panel_data_support`](generated/yohou.testing.check_splitter_panel_data_support.md) | Check splitter handles panel data if supports_panel_data=True. |
| [`check_splitter_parameter_constraints`](generated/yohou.testing.check_splitter_parameter_constraints.md) | Check parameter constraints are enforced via sklearn validation. |
| [`check_splitter_produces_valid_indices`](generated/yohou.testing.check_splitter_produces_valid_indices.md) | Check all train/test indices are valid row positions. |
| [`check_splitter_tags_accessible_before_fit`](generated/yohou.testing.check_splitter_tags_accessible_before_fit.md) | Check __sklearn_tags__() is callable on splitter instance. |
| [`check_splitter_tags_match_capabilities`](generated/yohou.testing.check_splitter_tags_match_capabilities.md) | Check tag values match actual splitter behavior. |
| [`check_splitter_tags_static_after_fit`](generated/yohou.testing.check_splitter_tags_static_after_fit.md) | Check tags remain unchanged after fit. |
| [`check_batch_invariance`](generated/yohou.testing.check_batch_invariance.md) | Check that a transformer declaring ``batch_invariant`` earns the claim. |
| [`check_feature_names_out_match`](generated/yohou.testing.check_feature_names_out_match.md) | Check get_feature_names_out() matches transform() output columns. |
| [`check_fit_idempotent`](generated/yohou.testing.check_fit_idempotent.md) | Check that fit(X).fit(X) equals fit(X). |
| [`check_fit_sets_attributes`](generated/yohou.testing.check_fit_sets_attributes.md) | Check fit() sets required attributes. |
| [`check_fit_transform_equivalence`](generated/yohou.testing.check_fit_transform_equivalence.md) | Check fit_transform(X) == fit(X).transform(X). |
| [`check_insufficient_data_raises`](generated/yohou.testing.check_insufficient_data_raises.md) | Check behavior when data length < observation_horizon. |
| [`check_inverse_observe_transform_identity`](generated/yohou.testing.check_inverse_observe_transform_identity.md) | Check inverse_transform(observe_transform(X)) ≈ X. |
| [`check_inverse_transform_identity`](generated/yohou.testing.check_inverse_transform_identity.md) | Check inverse_transform(transform(X)) ≈ X. |
| [`check_inverse_transform_round_trip`](generated/yohou.testing.check_inverse_transform_round_trip.md) | Check inverse_transform(transform(X)) ≈ X with shape validation. |
| [`check_memory_bounded`](generated/yohou.testing.check_memory_bounded.md) | Check memory doesn't grow unbounded with sequential updates. |
| [`check_observation_horizon_after_fit`](generated/yohou.testing.check_observation_horizon_after_fit.md) | Check observation_horizon is valid after fit(). |
| [`check_observation_horizon_not_fitted`](generated/yohou.testing.check_observation_horizon_not_fitted.md) | Check observation_horizon behavior before fit(). |
| [`check_observe_concatenates_memory`](generated/yohou.testing.check_observe_concatenates_memory.md) | Check observe() appends new data and maintains horizon size. |
| [`check_observe_transform_equivalence`](generated/yohou.testing.check_observe_transform_equivalence.md) | Check observe() does not change transform() output for a fitted transformer. |
| [`check_observe_transform_sequential_consistency`](generated/yohou.testing.check_observe_transform_sequential_consistency.md) | Check observe_transform(A) then observe_transform(B) == observe_transform(A+B). |
| [`check_panel_data_support`](generated/yohou.testing.check_panel_data_support.md) | Check transformer handles panel columns (panel data) correctly. |
| [`check_panel_group_preservation`](generated/yohou.testing.check_panel_group_preservation.md) | Check that transformers preserve panel group names after transformation. |
| [`check_rewind_transform_behavior`](generated/yohou.testing.check_rewind_transform_behavior.md) | Check rewind_transform() behavior and contract. |
| [`check_rewind_updates_memory`](generated/yohou.testing.check_rewind_updates_memory.md) | Check rewind(X) sets _X_observed to X.tail(observation_horizon). |
| [`check_tags_accessible_before_fit`](generated/yohou.testing.check_tags_accessible_before_fit.md) | Check __sklearn_tags__() is accessible before fit(). |
| [`check_tags_match_capabilities`](generated/yohou.testing.check_tags_match_capabilities.md) | Check tags accurately reflect transformer capabilities. |
| [`check_tags_static_after_fit`](generated/yohou.testing.check_tags_static_after_fit.md) | Check tags remain static (don't change) after fit(). |
| [`check_transform_drops_warmup_rows`](generated/yohou.testing.check_transform_drops_warmup_rows.md) | Check stateful transformers drop exactly observation_horizon rows. |
| [`check_transform_output_structure`](generated/yohou.testing.check_transform_output_structure.md) | Check transform() output has "time" column and valid structure. |
| [`check_transformer_methods_call_check_is_fitted`](generated/yohou.testing.check_transformer_methods_call_check_is_fitted.md) | Check all transformer methods (except fit) raise NotFittedError when unfitted. |
| [`check_transformer_preserve_dtypes`](generated/yohou.testing.check_transformer_preserve_dtypes.md) | Check transformer preserves input dtypes. |
| [`check_transformers_unfitted_stateless`](generated/yohou.testing.check_transformers_unfitted_stateless.md) | Check stateless transformers transform deterministically across fits. |
| [`check_weighter_compute_weights_alignment`](generated/yohou.testing.check_weighter_compute_weights_alignment.md) | Check ``compute_weights`` returns one weight per key element. |
| [`check_weighter_default_constructible`](generated/yohou.testing.check_weighter_default_constructible.md) | Check the weighter class is constructible with no arguments. |
| [`check_weighter_fit_noop_returns_self`](generated/yohou.testing.check_weighter_fit_noop_returns_self.md) | Check ``fit`` is a no-op returning self that leaves params unchanged. |
| [`check_weighter_resolved_array_validation`](generated/yohou.testing.check_weighter_resolved_array_validation.md) | Check resolved-array validation rejects NaN/negative/inf/all-zero. |
| [`check_weighter_tags_accessible_before_fit`](generated/yohou.testing.check_weighter_tags_accessible_before_fit.md) | Check ``__sklearn_tags__()`` is callable on an unfitted weighter. |
| [`check_weighter_tags_static_after_fit`](generated/yohou.testing.check_weighter_tags_static_after_fit.md) | Check tags are unchanged by the no-op ``fit``. |
