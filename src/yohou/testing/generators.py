"""Generator functions for systematic testing of forecasters and transformers.

This module provides _yield_* generator functions that dynamically generate
applicable check functions based on estimator tags.
"""

from collections.abc import Callable, Generator
from typing import Any

import polars as pl

from yohou.model_selection import GridSearchCV
from yohou.utils import inspect_panel

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
from .forecaster import (
    check_clone_preserves_forecaster_params,
    check_fit_predict_without_exogenous,
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
from .interval import (
    check_coverage_rates_parameter,
    check_coverage_rates_validation,
    check_interval_bounds,
    check_interval_prediction_columns,
    check_interval_prediction_types,
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
    check_scorer_coverage_rate_subselection,
    check_scorer_lower_is_better,
    check_scorer_methods_call_check_is_fitted,
    check_scorer_multi_vintage,
    check_scorer_parameter_validation,
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
    check_transform_drops_warmup_rows,
    check_transform_output_structure,
    check_transformer_methods_call_check_is_fitted,
    check_transformer_preserve_dtypes,
    check_transformers_unfitted_stateless,
)


def _yield_yohou_transformer_checks(
    transformer,
    X_train: pl.DataFrame,
    y_train: pl.DataFrame | None,
    X_test: pl.DataFrame,
    y_test: pl.DataFrame | None = None,
    tags: dict[str, Any] | None = None,
) -> Generator[tuple[str, Callable, dict], None, None]:
    """Generate applicable checks for a transformer based on tags.

    Parameters
    ----------
    transformer : BaseTransformer
        Fitted transformer instance
    X_train : pl.DataFrame
        Training data with "time" column
    y_train : pl.DataFrame, optional
        Training target data (for supervised transformers)
    X_test : pl.DataFrame
        Test data
    y_test : pl.DataFrame, optional
        Test target data
    tags : dict, optional
        Transformer metadata tags (if None, auto-detected from __sklearn_tags__):
        - requires_y: bool
        - stateful: bool
        - observation_horizon: int | None
        - preserves_dtype: bool
        - invertible: bool
        - supports_panel_data: bool

    Yields
    ------
    check_name : str
        Name of the check function
    check_func : callable
        Check function to execute
    check_kwargs : dict
        Keyword arguments for check function (bundled data)

    """
    if tags is None:
        # Get tags from __sklearn_tags__ method
        sklearn_tags = transformer.__sklearn_tags__()
        tags = {
            "requires_y": False,  # Yohou transformers don't have a requires_y concept
            "stateful": sklearn_tags.transformer_tags.stateful if sklearn_tags.transformer_tags else False,
            "observation_horizon": (
                transformer.observation_horizon if hasattr(transformer, "observation_horizon") else None
            ),
            "preserves_dtype": sklearn_tags.transformer_tags.preserves_dtype if sklearn_tags.transformer_tags else True,
            "invertible": sklearn_tags.transformer_tags.invertible if sklearn_tags.transformer_tags else False,
            "supports_panel_data": True,  # All Yohou transformers support panel data (prefixed columns)
        }

    # Core transformer checks (always yield)
    yield (
        "check_fit_sets_attributes",
        check_fit_sets_attributes,
        {"X": X_train, "y": y_train},
    )
    yield (
        "check_observation_horizon_not_fitted",
        check_observation_horizon_not_fitted,
        {"X": X_train},
    )
    yield (
        "check_observation_horizon_after_fit",
        check_observation_horizon_after_fit,
        {"X": X_train, "y": y_train},
    )
    yield (
        "check_transform_output_structure",
        check_transform_output_structure,
        {"X": X_test},
    )
    yield (
        "check_feature_names_out_match",
        check_feature_names_out_match,
        {"X": X_test},
    )
    yield (
        "check_transformers_unfitted_stateless",
        check_transformers_unfitted_stateless,
        {"X": X_train},
    )
    yield (
        "check_transformer_methods_call_check_is_fitted",
        check_transformer_methods_call_check_is_fitted,
        {"X": X_train, "y": y_train},
    )

    # Tag system checks
    yield (
        "check_tags_accessible_before_fit",
        check_tags_accessible_before_fit,
        {"X": X_train},
    )
    yield (
        "check_tags_static_after_fit",
        check_tags_static_after_fit,
        {"X": X_train, "y": y_train},
    )
    yield (
        "check_tags_match_capabilities",
        check_tags_match_capabilities,
        {"X": X_train, "y": y_train},
    )

    # rewind_transform check (works for both stateful and stateless)
    yield (
        "check_rewind_transform_behavior",
        check_rewind_transform_behavior,
        {"X": X_train, "y": y_train},
    )

    # Warmup row contract (works for both stateful and stateless)
    yield (
        "check_transform_drops_warmup_rows",
        check_transform_drops_warmup_rows,
        {"X": X_train, "y": y_train},
    )

    # Stateful transformer checks
    if tags.get("stateful", False):
        yield (
            "check_observe_concatenates_memory",
            check_observe_concatenates_memory,
            {"X": X_train, "y": y_train},
        )
        yield (
            "check_observe_transform_equivalence",
            check_observe_transform_equivalence,
            {"X": X_train, "y": y_train},
        )
        yield (
            "check_rewind_updates_memory",
            check_rewind_updates_memory,
            {"X": X_train, "y": y_train},
        )
        yield (
            "check_memory_bounded",
            check_memory_bounded,
            {"X_train": X_train, "X_test": X_test, "y": y_train},
        )
        yield (
            "check_insufficient_data_raises",
            check_insufficient_data_raises,
            {"X": X_train, "y": y_train},
        )
        yield (
            "check_observe_transform_sequential_consistency",
            check_observe_transform_sequential_consistency,
            {"X": X_train, "y": y_train},
        )

    # Invertible transformer checks
    if tags.get("invertible", False):
        yield (
            "check_inverse_transform_identity",
            check_inverse_transform_identity,
            {"X": X_test, "y": y_train},
        )
        yield (
            "check_inverse_transform_round_trip",
            check_inverse_transform_round_trip,
            {"X": X_test, "y": y_train},
        )
        yield (
            "check_inverse_observe_transform_identity",
            check_inverse_observe_transform_identity,
            {"X": X_train, "y": y_train},
        )

    # Enhanced sklearn checks
    yield (
        "check_transformer_preserve_dtypes",
        check_transformer_preserve_dtypes,
        {"X": X_train, "y": y_train},
    )
    yield (
        "check_fit_idempotent",
        check_fit_idempotent,
        {"X": X_train, "y": y_train},
    )
    yield (
        "check_fit_transform_equivalence",
        check_fit_transform_equivalence,
        {"X": X_train, "y": y_train},
    )

    # Panel data check
    if tags.get("supports_panel_data", False):
        _, X_panel_groups = inspect_panel(X_train)
        if len(X_panel_groups) > 0:
            yield (
                "check_panel_data_support",
                check_panel_data_support,
                {"X_panel": X_train, "y": y_train},
            )
            yield (
                "check_panel_group_preservation",
                check_panel_group_preservation,
                {"X_panel": X_train, "y": y_train},
            )

    # Metadata routing checks (always applicable)
    yield (
        "check_metadata_routing_default_request",
        check_metadata_routing_default_request,
        {},
    )
    yield (
        "check_metadata_routing_get_metadata_routing",
        check_metadata_routing_get_metadata_routing,
        {},
    )


def _yield_yohou_forecaster_checks(
    forecaster,
    y_train: pl.DataFrame,
    X_train: pl.DataFrame | None,
    y_test: pl.DataFrame,
    X_test: pl.DataFrame | None,
    tags: dict[str, Any] | None = None,
) -> Generator[tuple[str, Callable, dict], None, None]:
    """Generate applicable checks for a forecaster based on tags.

    Parameters
    ----------
    forecaster : BaseForecaster
        Fitted forecaster instance
    y_train : pl.DataFrame
        Training target data with "time" column
    X_train : pl.DataFrame, optional
        Training features
    y_test : pl.DataFrame
        Test target data
    X_test : pl.DataFrame, optional
        Test features
    tags : dict, optional
        Forecaster metadata tags (if None, auto-detected from __sklearn_tags__):
        - forecaster_type: frozenset of str (e.g., frozenset({"point"}))
        - uses_reduction: bool
        - supports_panel_data: bool
        - uses_target_transformer: bool
        - uses_feature_transformer: bool
        - supports_scoring: bool

    Yields
    ------
    check_name : str
        Name of the check function
    check_func : callable
        Check function to execute
    check_kwargs : dict
        Keyword arguments for check function (bundled data)

    """
    if tags is None:
        # Get tags from __sklearn_tags__ method
        sklearn_tags = forecaster.__sklearn_tags__()
        tags = {
            "forecaster_type": sklearn_tags.forecaster_tags.forecaster_type if sklearn_tags.forecaster_tags else None,
            "uses_reduction": sklearn_tags.forecaster_tags.uses_reduction if sklearn_tags.forecaster_tags else False,
            "supports_panel_data": sklearn_tags.forecaster_tags.supports_panel_data
            if sklearn_tags.forecaster_tags
            else True,
            "uses_target_transformer": sklearn_tags.forecaster_tags.uses_target_transformer
            if sklearn_tags.forecaster_tags
            else False,
            "uses_feature_transformer": sklearn_tags.forecaster_tags.uses_feature_transformer
            if sklearn_tags.forecaster_tags
            else False,
            "tracks_observations": sklearn_tags.forecaster_tags.tracks_observations
            if sklearn_tags.forecaster_tags
            else True,
            "ignores_exogenous": sklearn_tags.forecaster_tags.ignores_exogenous
            if sklearn_tags.forecaster_tags
            else False,
            "supports_scoring": True,  # Default assumption
        }

    # Bundle data for check functions

    # Common forecaster checks (always yield)
    yield (
        "check_fit_sets_forecaster_attributes",
        check_fit_sets_forecaster_attributes,
        {"y": y_train, "X": X_train, "forecasting_horizon": 3},
    )
    yield (
        "check_forecaster_not_fitted_error",
        check_forecaster_not_fitted_error,
        {"y": y_train, "X": X_train},
    )
    yield (
        "check_predict_time_columns",
        check_predict_time_columns,
        {"y_test": y_test, "X_test": X_test},
    )
    yield (
        "check_forecasting_horizon_validation",
        check_forecasting_horizon_validation,
        {"y": y_train, "X": X_train},
    )
    yield "check_prediction_types_property", check_prediction_types_property, {}
    yield "check_clone_preserves_forecaster_params", check_clone_preserves_forecaster_params, {}
    yield (
        "check_forecaster_methods_call_check_is_fitted",
        check_forecaster_methods_call_check_is_fitted,
        {"y": y_train, "X": X_train, "forecasting_horizon": 3},
    )
    yield (
        "check_fit_predict_without_exogenous",
        check_fit_predict_without_exogenous,
        {
            "y": y_train,
            "ignores_exogenous": tags.get("ignores_exogenous", False),
            "target_as_feature": getattr(forecaster, "target_as_feature", "transformed"),
        },
    )

    # Tag system checks (always run)
    yield (
        "check_forecaster_tags_accessible_before_fit",
        check_forecaster_tags_accessible_before_fit,
        {"y": y_train, "X": X_train},
    )
    yield (
        "check_forecaster_tags_static_after_fit",
        check_forecaster_tags_static_after_fit,
        {"y": y_train, "X": X_train, "forecasting_horizon": 3},
    )
    yield (
        "check_forecaster_tags_match_capabilities",
        check_forecaster_tags_match_capabilities,
        {"y": y_train, "X": X_train},
    )

    # Update/reset checks (if enough data and forecaster tracks observations)
    # Meta-forecasters like DecompositionPipeline set tracks_observations=False because they
    # delegate observation tracking to child forecasters with custom logic
    if len(y_test) >= 10 and tags.get("tracks_observations", True):
        y_update = y_test[:3]
        y_reset = y_test[:10]
        X_update = X_test[:3] if X_test is not None else None
        X_reset = X_test[:10] if X_test is not None else None

        yield (
            "check_observe_extends_observations",
            check_observe_extends_observations,
            {
                "y_train": y_train,
                "y_update": y_update,
                "X_train": X_train,
                "X_update": X_update,
            },
        )
        yield (
            "check_rewind_replaces_observations",
            check_rewind_replaces_observations,
            {
                "y_train": y_train,
                "y_reset": y_reset,
                "X_train": X_train,
                "X_reset": X_reset,
            },
        )

    # Transformer composition checks
    if (tags.get("uses_target_transformer", False) or tags.get("uses_feature_transformer", False)) and len(y_test) >= 5:
        y_reset = y_test[:10] if len(y_test) >= 10 else y_test
        X_reset = X_test[:10] if X_test is not None and len(X_test) >= 10 else X_test
        yield (
            "check_rewind_propagates_to_transformers",
            check_rewind_propagates_to_transformers,
            {
                "y_train": y_train,
                "y_reset": y_reset,
                "X_train": X_train,
                "X_reset": X_reset,
            },
        )

    # Point forecaster checks
    forecaster_type = tags.get("forecaster_type")
    if forecaster_type is not None and "point" in forecaster_type:
        yield (
            "check_point_prediction_structure",
            check_point_prediction_structure,
            {"y_test": y_test, "X_test": X_test},
        )
        yield "check_point_prediction_types", check_point_prediction_types, {}

    # Interval forecaster checks
    if forecaster_type is not None and "interval" in forecaster_type:
        yield (
            "check_interval_prediction_columns",
            check_interval_prediction_columns,
            {"y_test": y_test, "X_test": X_test},
        )
        yield (
            "check_interval_bounds",
            check_interval_bounds,
            {"y_test": y_test, "X_test": X_test},
        )
        yield "check_interval_prediction_types", check_interval_prediction_types, {}
        yield "check_coverage_rates_parameter", check_coverage_rates_parameter, {}
        yield (
            "check_coverage_rates_validation",
            check_coverage_rates_validation,
            {"y": y_train, "X": X_train},
        )

    # Class-probability forecaster checks
    if forecaster_type is not None and "class_proba" in forecaster_type:
        yield (
            "check_class_proba_prediction_structure",
            check_class_proba_prediction_structure,
            {"y_test": y_test, "X_test": X_test},
        )
        yield (
            "check_class_proba_prediction_bounds",
            check_class_proba_prediction_bounds,
            {"y_test": y_test, "X_test": X_test},
        )
        yield (
            "check_class_proba_prediction_sums",
            check_class_proba_prediction_sums,
            {"y_test": y_test, "X_test": X_test},
        )
        yield "check_class_proba_prediction_types", check_class_proba_prediction_types, {}
        yield "check_class_proba_classes_attribute", check_class_proba_classes_attribute, {}
        yield (
            "check_class_proba_predict_returns_labels",
            check_class_proba_predict_returns_labels,
            {"y_test": y_test, "X_test": X_test},
        )

    # Reduction forecaster checks
    if tags.get("uses_reduction", False):
        yield "check_estimator_parameter", check_estimator_parameter, {}
        yield "check_reduction_strategy", check_reduction_strategy, {}

    # Cross-learning checks (for panel data)
    if tags.get("supports_panel_data", False):
        # Need to check if we have panel data available

        _, y_panel_groups = inspect_panel(y_train)
        if len(y_panel_groups) > 0:
            # We have panel data, run cross-learning checks
            yield (
                "check_panel_data",
                check_panel_data,
                {"y_panel": y_test, "X_panel": X_test},
            )
            yield (
                "check_panel_single_group",
                check_panel_single_group,
                {"y_panel": y_test, "X_panel": X_test},
            )
            yield (
                "check_panel_invalid_group_raises",
                check_panel_invalid_group_raises,
                {"y_panel": y_test, "X_panel": X_test},
            )

    # Metadata routing checks (always applicable)
    yield (
        "check_metadata_routing_default_request",
        check_metadata_routing_default_request,
        {},
    )
    yield (
        "check_metadata_routing_get_metadata_routing",
        check_metadata_routing_get_metadata_routing,
        {},
    )


def _yield_yohou_splitter_checks(
    splitter,
    y: pl.DataFrame,
    X: pl.DataFrame | None = None,
    tags: dict[str, Any] | None = None,
) -> Generator[tuple[str, Callable, dict], None, None]:
    """Generate applicable checks for a splitter based on tags.

    Parameters
    ----------
    splitter : BaseSplitter
        Splitter instance
    y : pl.DataFrame
        Target time series with "time" column
    X : pl.DataFrame, optional
        Exogenous features
    tags : dict, optional
        Splitter metadata tags (if None, auto-detected from __sklearn_tags__):
        - splitter_type: str | None
        - supports_panel_data: bool
        - produces_non_overlapping_tests: bool
        - stateful: bool

    Yields
    ------
    check_name : str
        Name of the check function
    check_func : callable
        Check function to execute
    check_kwargs : dict
        Keyword arguments for check function (bundled data)

    """
    if tags is None:
        # Get tags from __sklearn_tags__ method
        sklearn_tags = splitter.__sklearn_tags__()
        tags = {
            "splitter_type": sklearn_tags.splitter_tags.splitter_type if sklearn_tags.splitter_tags else None,
            "supports_panel_data": sklearn_tags.splitter_tags.supports_panel_data
            if sklearn_tags.splitter_tags
            else False,
            "produces_non_overlapping_tests": sklearn_tags.splitter_tags.produces_non_overlapping_tests
            if sklearn_tags.splitter_tags
            else True,
            "stateful": sklearn_tags.splitter_tags.stateful if sklearn_tags.splitter_tags else False,
        }

    # Tag checks (always yield)
    yield (
        "check_splitter_tags_accessible_before_fit",
        check_splitter_tags_accessible_before_fit,
        {"splitter": splitter},
    )
    yield (
        "check_splitter_tags_static_after_fit",
        check_splitter_tags_static_after_fit,
        {"splitter": splitter, "y": y, "X": X},
    )
    expected_tags = {k: v for k, v in tags.items() if k != "stateful"}  # stateful not usually tested
    yield (
        "check_splitter_tags_match_capabilities",
        check_splitter_tags_match_capabilities,
        {"splitter": splitter, "y": y, "X": X, "expected_tags": expected_tags},
    )

    # Functionality checks (always yield)
    yield (
        "check_splitter_produces_valid_indices",
        check_splitter_produces_valid_indices,
        {"y": y, "X": X},
    )
    yield (
        "check_splitter_n_splits_consistency",
        check_splitter_n_splits_consistency,
        {"y": y, "X": X},
    )
    yield (
        "check_splitter_non_overlapping_tests",
        check_splitter_non_overlapping_tests,
        {"y": y, "X": X},
    )

    # Panel data support check (conditional)
    if tags.get("supports_panel_data", False):
        # Generate panel data for testing
        y_panel = y.rename({col: f"{col}__group1" for col in y.columns if col != "time"})
        X_panel = X.rename({col: f"{col}__group1" for col in X.columns if col != "time"}) if X is not None else None
        yield (
            "check_splitter_panel_data_support",
            check_splitter_panel_data_support,
            {"y_panel": y_panel, "X_panel": X_panel},
        )

    # Parameter validation checks (yield parametrized invalid values)
    splitter_class = type(splitter)
    if hasattr(splitter_class, "_parameter_constraints"):
        constraints = splitter_class._parameter_constraints
        param_test_cases = []

        # Generate invalid values based on parameter constraints
        if "n_splits" in constraints:
            param_test_cases.append(("n_splits", [1, 0, -1]))  # Must be >= 2
        if "test_size" in constraints:
            param_test_cases.append(("test_size", [0, -1]))  # Must be >= 1
        if "train_size" in constraints:
            param_test_cases.append(("train_size", [0, -1]))  # Must be >= 1
        if "gap" in constraints:
            param_test_cases.append(("gap", [-1]))  # Must be >= 0

        for param_name, invalid_values in param_test_cases:
            yield (
                f"check_splitter_parameter_constraints[{param_name}]",
                check_splitter_parameter_constraints,
                {
                    "splitter_class": splitter_class,
                    "param_name": param_name,
                    "invalid_values": invalid_values,
                },
            )


def _yield_yohou_scorer_checks(
    scorer,
    y_truth: pl.DataFrame,
    y_pred: pl.DataFrame,
    tags: dict[str, Any] | None = None,
) -> Generator[tuple[str, Callable, dict], None, None]:
    """Generate applicable checks for a scorer based on tags.

    Parameters
    ----------
    scorer : BaseScorer
        Scorer instance
    y_truth : pl.DataFrame
        Ground truth with "time" column
    y_pred : pl.DataFrame
        Predictions with "vintage_time" and "time" columns
    tags : dict, optional
        Scorer metadata tags (if None, auto-detected from __sklearn_tags__):
        - prediction_type: str | None
        - lower_is_better: bool
        - requires_calibration: bool

    Yields
    ------
    check_name : str
        Name of the check function
    check_func : callable
        Check function to execute
    check_kwargs : dict
        Keyword arguments for check function (bundled data)

    """
    if tags is None:
        # Get tags from __sklearn_tags__ method
        sklearn_tags = scorer.__sklearn_tags__()
        tags = {
            "prediction_type": sklearn_tags.scorer_tags.prediction_type if sklearn_tags.scorer_tags else None,
            "lower_is_better": sklearn_tags.scorer_tags.lower_is_better if sklearn_tags.scorer_tags else True,
            "requires_calibration": sklearn_tags.scorer_tags.requires_calibration
            if sklearn_tags.scorer_tags
            else False,
        }

    # Tag checks (always yield)
    yield (
        "check_scorer_tags_accessible_before_fit",
        check_scorer_tags_accessible_before_fit,
        {"scorer": scorer},
    )
    yield (
        "check_scorer_tags_static_after_fit",
        check_scorer_tags_static_after_fit,
        {"scorer": scorer, "y_truth": y_truth, "y_pred": y_pred},
    )
    expected_tags = {k: v for k, v in tags.items() if v is not None}
    yield (
        "check_scorer_tags_match_capabilities",
        check_scorer_tags_match_capabilities,
        {"scorer": scorer, "y_truth": y_truth, "y_pred": y_pred, "expected_tags": expected_tags},
    )

    # Functionality checks (always yield)
    yield (
        "check_scorer_lower_is_better",
        check_scorer_lower_is_better,
        {},
    )
    yield (
        "check_scorer_methods_call_check_is_fitted",
        check_scorer_methods_call_check_is_fitted,
        {"y_train": y_truth, "y_pred": y_pred},
    )

    # Aggregation methods check (if scorer has aggregation_method parameter)
    if hasattr(scorer, "aggregation_method"):
        aggregation_methods = ["stepwise", "vintagewise", "componentwise"]
        yield (
            "check_scorer_aggregation_methods",
            check_scorer_aggregation_methods,
            {"y_truth": y_truth, "y_pred": y_pred, "aggregation_methods": aggregation_methods},
        )

    # Coverage rate subselection (interval scorers only)
    if tags.get("prediction_type") == "interval" and "coverage_rate" in y_pred.columns:
        coverage_rates = y_pred["coverage_rate"].unique().to_list()
        yield (
            "check_scorer_coverage_rate_subselection",
            check_scorer_coverage_rate_subselection,
            {"y_truth": y_truth, "y_pred_interval": y_pred, "coverage_rates": coverage_rates[:1]},
        )

    # Parameter validation checks
    scorer_class = type(scorer)
    validation_test_cases: list[tuple[str, list, str]] = [
        ("groups", ["nonexistent_group"], "groups"),
        ("components", ["nonexistent_component"], "components"),
    ]

    # Aggregation method validation
    if hasattr(scorer, "aggregation_method"):
        validation_test_cases.append(("aggregation_method", ["invalid_method"], "aggregation_method"))

    # Add coverage validation for interval scorers
    if tags.get("prediction_type") == "interval":
        validation_test_cases.extend([
            ("coverage_rates", [1.5], "coverage"),  # Out of range
            ("coverage_rates", [-0.5], "coverage"),  # Negative
            ("coverage_rates", [[]], "coverage"),  # Invalid type (nested list)
        ])

    for param_name, invalid_value, error_match in validation_test_cases:
        yield (
            f"check_scorer_parameter_validation[{param_name}={invalid_value}]",
            check_scorer_parameter_validation,
            {
                "scorer_class": scorer_class,
                "param_name": param_name,
                "invalid_value": invalid_value,
                "error_match": error_match,
            },
        )

    # Multi-vintage smoke test
    yield (
        "check_scorer_multi_vintage",
        check_scorer_multi_vintage,
        {"y_truth": y_truth, "y_pred": y_pred},
    )


def _yield_yohou_search_checks(
    search_cv,
    y_train: pl.DataFrame,
    X_train: pl.DataFrame | None,
    y_test: pl.DataFrame,
    X_test: pl.DataFrame | None,
    tags: dict[str, Any] | None = None,
) -> Generator[tuple[str, Callable, dict], None, None]:
    """Generate applicable checks for a search CV instance based on tags.

    Parameters
    ----------
    search_cv : BaseSearchCV
        Fitted search CV instance (GridSearchCV or RandomizedSearchCV)
    y_train : pl.DataFrame
        Training target data with "time" column
    X_train : pl.DataFrame, optional
        Training features
    y_test : pl.DataFrame
        Test target data
    X_test : pl.DataFrame, optional
        Test features
    tags : dict, optional
        Search CV metadata tags:
        - search_type: "grid" | "randomized"
        - refit: bool (default True)
        - multimetric: bool (default False)
        - supports_panel_data: bool (default True)

    Yields
    ------
    check_name : str
        Name of the check function
    check_func : callable
        Check function to execute
    check_kwargs : dict
        Keyword arguments for check function (bundled data)

    """
    if tags is None:
        # Infer tags from search CV instance

        # Detect interval scoring from scorer type
        from yohou.metrics.base import BaseIntervalScorer  # noqa: PLC0415

        _scorer = getattr(search_cv, "scoring", None)
        _needs_interval = False
        if isinstance(_scorer, dict):
            _needs_interval = any(isinstance(s, BaseIntervalScorer) for s in _scorer.values())
        elif isinstance(_scorer, BaseIntervalScorer):
            _needs_interval = True

        tags = {
            "search_type": "grid" if isinstance(search_cv, GridSearchCV) else "randomized",
            "refit": getattr(search_cv, "refit", True),
            "multimetric": isinstance(getattr(search_cv, "scoring", None), dict),
            "supports_panel_data": True,  # All search CVs support panel data
            "interval_scoring": _needs_interval,
        }

    # Common search CV checks (always yield)
    yield (
        "check_search_fit_sets_attributes",
        check_search_fit_sets_attributes,
        {"y": y_train, "X": X_train, "forecasting_horizon": 3},
    )
    yield (
        "check_search_not_fitted_error",
        check_search_not_fitted_error,
        {"y": y_train, "X": X_train},
    )
    yield (
        "check_search_cv_results_structure",
        check_search_cv_results_structure,
        {"y": y_train, "X": X_train, "forecasting_horizon": 3},
    )
    yield (
        "check_search_clone_preserves_params",
        check_search_clone_preserves_params,
        {},
    )
    yield (
        "check_search_error_score_handling",
        check_search_error_score_handling,
        {"y": y_train, "X": X_train, "forecasting_horizon": 3},
    )

    # refit checks
    if tags.get("refit", True):
        # Delegation checks (only when refit=True)
        yield (
            "check_search_predict_delegates",
            check_search_predict_delegates,
            {"y_train": y_train, "y_test": y_test, "X_train": X_train, "X_test": X_test},
        )

        # Update/reset checks (need enough data)
        if len(y_test) >= 10:
            y_update = y_test[:3]
            X_update = X_test[:3] if X_test is not None else None
            yield (
                "check_search_observe_delegates",
                check_search_observe_delegates,
                {
                    "y_train": y_train,
                    "y_update": y_update,
                    "X_train": X_train,
                    "X_update": X_update,
                },
            )

            # Ensure y_reset has enough rows for the best forecaster's
            # observation horizon (e.g. SeasonalNaive with large seasonality).
            obs_h = getattr(search_cv, "best_forecaster_", None)
            min_reset = getattr(obs_h, "observation_horizon", 0) + 1 if obs_h is not None else 10
            reset_len = max(10, min_reset)
            if len(y_test) >= reset_len:
                y_reset = y_test[:reset_len]
                X_reset = X_test[:reset_len] if X_test is not None else None
                yield (
                    "check_search_rewind_delegates",
                    check_search_rewind_delegates,
                    {"y_train": y_train, "y_reset": y_reset, "X_train": X_train, "X_reset": X_reset},
                )
    else:
        # refit=False checks
        yield (
            "check_search_refit_false_no_forecaster",
            check_search_refit_false_no_forecaster,
            {"y": y_train, "X": X_train, "forecasting_horizon": 3},
        )

    # Method availability checks (always yield)
    yield (
        "check_search_method_availability",
        check_search_method_availability,
        {"y": y_train, "X": X_train, "forecasting_horizon": 3},
    )

    # Multi-metric checks
    if tags.get("multimetric", False):
        yield (
            "check_search_multimetric_scoring",
            check_search_multimetric_scoring,
            {"y": y_train, "X": X_train, "forecasting_horizon": 3},
        )

    # Interval scoring checks (when interval scorers are used)
    if tags.get("interval_scoring", False) and tags.get("refit", True):
        yield (
            "check_search_interval_predict_delegates",
            check_search_interval_predict_delegates,
            {"y_train": y_train, "y_test": y_test, "X_train": X_train, "X_test": X_test},
        )

    # Return train score check (parameterized)
    yield (
        "check_search_return_train_score",
        check_search_return_train_score,
        {"y": y_train, "X": X_train, "forecasting_horizon": 3},
    )

    # GridSearchCV-specific checks
    if tags.get("search_type") == "grid":
        yield (
            "check_grid_search_exhaustive",
            check_grid_search_exhaustive,
            {"y": y_train, "X": X_train, "forecasting_horizon": 3},
        )
        yield (
            "check_grid_search_param_grid_validation",
            check_grid_search_param_grid_validation,
            {},
        )

    # RandomizedSearchCV-specific checks
    if tags.get("search_type") == "randomized":
        yield (
            "check_randomized_search_n_iter",
            check_randomized_search_n_iter,
            {"y": y_train, "X": X_train, "forecasting_horizon": 3},
        )

        # Reproducibility check (only if random_state is set)
        if hasattr(search_cv, "random_state") and search_cv.random_state is not None:
            yield (
                "check_randomized_search_reproducibility",
                check_randomized_search_reproducibility,
                {"y": y_train, "X": X_train, "forecasting_horizon": 3},
            )

        yield (
            "check_randomized_search_distributions",
            check_randomized_search_distributions,
            {"y": y_train, "X": X_train, "forecasting_horizon": 3},
        )

    # Panel data checks (if panel data available)
    if tags.get("supports_panel_data", True):
        _, y_panel_groups = inspect_panel(y_train)
        if len(y_panel_groups) > 0:
            # Extract first group name for testing
            groups = list(y_panel_groups.keys())[:1]
            yield (
                "check_search_panel_data",
                check_search_panel_data,
                {
                    "y_train": y_train,
                    "y_test": y_test,
                    "X_train": X_train,
                    "X_test": X_test,
                    "groups": groups,
                },
            )
