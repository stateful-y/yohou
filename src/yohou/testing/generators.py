"""Generator functions for systematic testing of forecasters and transformers.

This module provides _yield_* generator functions that dynamically generate
applicable check functions based on estimator tags.
"""

from collections.abc import Callable, Generator
from typing import Any

try:
    import polars as pl
except ImportError as e:
    raise ImportError(
        "polars.testing is required for yohou.testing module. Install with: uv sync --group tests"
    ) from e

from .common import (
    check_metadata_routing_default_request,
    check_metadata_routing_get_metadata_routing,
)
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
            "stateful": sklearn_tags.transformer_tags.stateful
            if sklearn_tags.transformer_tags
            else False,
            "observation_horizon": (
                transformer.observation_horizon
                if hasattr(transformer, "observation_horizon")
                else None
            ),
            "preserves_dtype": sklearn_tags.transformer_tags.preserves_dtype
            if sklearn_tags.transformer_tags
            else True,
            "invertible": sklearn_tags.transformer_tags.invertible
            if sklearn_tags.transformer_tags
            else False,
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

    # Stateful transformer checks
    if tags.get("stateful", False):
        yield (
            "check_update_concatenates_memory",
            check_update_concatenates_memory,
            {"X": X_train, "y": y_train},
        )
        yield (
            "check_update_transform_equivalence",
            check_update_transform_equivalence,
            {"X": X_train, "y": y_train},
        )
        yield (
            "check_reset_updates_memory",
            check_reset_updates_memory,
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
        from yohou.utils.panel import inspect_locality

        _, X_panel_groups = inspect_locality(X_train)
        if len(X_panel_groups) > 0:
            yield (
                "check_panel_data_support",
                check_panel_data_support,
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
        - forecaster_type: "point" | "interval" | "both"
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
            "forecaster_type": sklearn_tags.forecaster_tags.forecaster_type
            if sklearn_tags.forecaster_tags
            else None,
            "uses_reduction": sklearn_tags.forecaster_tags.uses_reduction
            if sklearn_tags.forecaster_tags
            else False,
            "supports_panel_data": sklearn_tags.forecaster_tags.supports_panel_data
            if sklearn_tags.forecaster_tags
            else True,
            "uses_target_transformer": sklearn_tags.forecaster_tags.uses_target_transformer
            if sklearn_tags.forecaster_tags
            else False,
            "uses_feature_transformer": sklearn_tags.forecaster_tags.uses_feature_transformer
            if sklearn_tags.forecaster_tags
            else False,
            "supports_scoring": True,  # Default assumption
        }

    # Bundle data for check functions
    check_kwargs = {
        "y_train": y_train,
        "X_train": X_train,
        "y_test": y_test,
        "X_test": X_test,
    }

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

    # Update/reset checks (if enough data)
    if len(y_test) >= 10:
        y_update = y_test[:3]
        y_reset = y_test[:10]
        X_update = X_test[:3] if X_test is not None else None
        X_reset = X_test[:10] if X_test is not None else None

        yield (
            "check_update_extends_observations",
            check_update_extends_observations,
            {
                "y_train": y_train,
                "y_update": y_update,
                "X_train": X_train,
                "X_update": X_update,
            },
        )
        yield (
            "check_reset_replaces_observations",
            check_reset_replaces_observations,
            {
                "y_train": y_train,
                "y_reset": y_reset,
                "X_train": X_train,
                "X_reset": X_reset,
            },
        )

    # Transformer composition checks
    if tags.get("uses_target_transformer", False) or tags.get("uses_feature_transformer", False):
        if len(y_test) >= 5:
            y_reset = y_test[:10] if len(y_test) >= 10 else y_test
            X_reset = X_test[:10] if X_test is not None and len(X_test) >= 10 else X_test
            yield (
                "check_reset_propagates_to_transformers",
                check_reset_propagates_to_transformers,
                {
                    "y_train": y_train,
                    "y_reset": y_reset,
                    "X_train": X_train,
                    "X_reset": X_reset,
                },
            )

    # Point forecaster checks
    if tags.get("forecaster_type") == "point":
        yield (
            "check_point_prediction_structure",
            check_point_prediction_structure,
            {"y_test": y_test, "X_test": X_test},
        )
        yield "check_point_prediction_types", check_point_prediction_types, {}

    # Interval forecaster checks
    if tags.get("forecaster_type") == "interval":
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

    # Reduction forecaster checks
    if tags.get("uses_reduction", False):
        yield "check_estimator_parameter", check_estimator_parameter, {}
        yield "check_reduction_strategy", check_reduction_strategy, {}

    # Cross-learning checks (for panel data)
    if tags.get("supports_panel_data", False):
        # Need to check if we have panel data available
        from yohou.utils.panel import inspect_locality

        _, y_panel_groups = inspect_locality(y_train)
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
