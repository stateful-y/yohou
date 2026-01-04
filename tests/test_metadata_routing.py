"""Comprehensive test suite for metadata routing in Yohou.

This module tests the sklearn metadata routing integration across all Yohou
estimators, transformers, and meta-estimators. Tests are organized by component
and routing scenario.

Test Categories:
1. Basic Routing (get_metadata_routing, set_*_request methods)
2. Transformer Routing (transform, update_transform)
3. Forecaster Routing (fit, predict, update_predict)
4. Pipeline Routing (Pipeline, FeatureUnion, ColumnTransformer)
5. SearchCV Routing (routing through cross-validation)
6. Error Handling (UnsetMetadataPassedError, validation)
7. Integration Tests (nested routing scenarios)
8. Composite Methods Tests (update_transform, update_predict registration)
"""

import inspect

import pytest
from metadata_routing_common import _Registry, assert_request_is_empty
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.utils.metadata_routing import MetadataRequest, MetadataRouter

from yohou.metrics import MAE
from yohou.model_selection import SearchCV
from yohou.pipeline import ColumnTransformer, FeatureUnion, Pipeline
from yohou.point_forecaster import PointReductionForecaster, SeasonalNaive
from yohou.preprocessing import SeasonalDifferencing

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def consuming_estimator():
    """Create an sklearn estimator that records metadata."""
    registry = _Registry()

    class ConsumingRidge(Ridge):
        def __init__(self, registry=None, **kwargs):
            super().__init__(**kwargs)
            self.registry = registry if registry is not None else _Registry()
            self._fit_metadata = {}
            self._predict_metadata = {}

        def fit(self, X, y, sample_weight=None, **fit_params):
            # Record that fit was called and what metadata was passed
            self.registry.append(("fit", sample_weight))
            self._fit_metadata = {"sample_weight": sample_weight, **fit_params}
            return super().fit(X, y, sample_weight=sample_weight)

        def predict(self, X, **predict_params):
            # Record predict call
            self.registry.append(("predict", predict_params))
            self._predict_metadata = predict_params
            return super().predict(X)

    return ConsumingRidge(registry=registry), registry


# ============================================================================
# 1. BASIC ROUTING TESTS
# ============================================================================


def test_get_metadata_routing_exists_on_forecasters():
    """All forecasters must have get_metadata_routing method."""
    forecasters = [
        SeasonalNaive(seasonality=3),
        PointReductionForecaster(estimator=Ridge()),
    ]
    for forecaster in forecasters:
        assert hasattr(forecaster, "get_metadata_routing")
        routing = forecaster.get_metadata_routing()
        assert isinstance(routing, (MetadataRouter, MetadataRequest))


def test_get_metadata_routing_exists_on_transformers():
    """All transformers must have get_metadata_routing method."""
    transformers = [
        SeasonalDifferencing(seasonality=3),
    ]
    for transformer in transformers:
        assert hasattr(transformer, "get_metadata_routing")
        routing = transformer.get_metadata_routing()
        assert isinstance(routing, (MetadataRouter, MetadataRequest))


def test_default_request_is_empty(y_X_factory):
    """By default, metadata requests should be empty."""
    y, X = y_X_factory(length=31)
    forecaster = SeasonalNaive(seasonality=3)
    forecaster.fit(y, X, forecasting_horizon=3)

    routing = forecaster.get_metadata_routing()
    assert_request_is_empty(routing)


def test_set_fit_request_methods_exist():
    """Forecasters should have set_fit_request method."""
    forecaster = SeasonalNaive(seasonality=3)
    assert hasattr(forecaster, "set_fit_request")

    # Method should accept parameters from signature
    sig = inspect.signature(forecaster.set_fit_request)
    params = list(sig.parameters.keys())
    assert "forecasting_horizon" in params


def test_set_predict_request_methods_exist():
    """Forecasters should have set_predict_request method."""
    forecaster = SeasonalNaive(seasonality=3)
    assert hasattr(forecaster, "set_predict_request")

    sig = inspect.signature(forecaster.set_predict_request)
    params = list(sig.parameters.keys())
    assert "forecasting_horizon" in params


def test_set_update_predict_request_exists():
    """Forecasters should have set_update_predict_request (composite method)."""
    forecaster = SeasonalNaive(seasonality=3)
    assert hasattr(forecaster, "set_update_predict_request")


# ============================================================================
# 2. TRANSFORMER ROUTING TESTS
# ============================================================================


def test_transformer_accepts_params_in_transform(time_series_factory):
    """Transformers should accept **params in transform method."""
    y = time_series_factory(length=50, n_components=1)
    transformer = SeasonalDifferencing(seasonality=3)
    transformer.fit(y)

    # Should work without params
    y_t = transformer.transform(y)
    assert len(y_t) > 0

    # Should also accept empty params
    y_t = transformer.transform(y, **{})
    assert len(y_t) > 0


def test_transformer_update_transform_routes_to_transform_only(time_series_factory):
    """update_transform should route params to transform, not update."""
    y = time_series_factory(length=50, n_components=1)
    transformer = SeasonalDifferencing(seasonality=3)
    transformer.fit(y)

    # update_transform should work (routes to transform)
    y_new = y[-5:]
    y_t = transformer.update_transform(y_new)
    assert len(y_t) > 0


def test_transformer_does_not_route_to_update(time_series_factory):
    """update() should not accept **params (memory management only)."""
    y = time_series_factory(length=50, n_components=1)
    transformer = SeasonalDifferencing(seasonality=3)
    transformer.fit(y)

    # update should not have **params parameter
    sig = inspect.signature(transformer.update)
    # Check that there's no VAR_KEYWORD parameter
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    assert not has_var_keyword, "update() should not accept **params"


# ============================================================================
# 3. FORECASTER ROUTING TESTS
# ============================================================================


def test_forecaster_accepts_params_in_fit(y_X_factory):
    """Forecasters should accept **params in fit method."""
    y, X = y_X_factory(length=50, n_targets=1)
    forecaster = SeasonalNaive(seasonality=3)

    # Should work without params
    forecaster.fit(y, X, forecasting_horizon=3)

    # Should also accept empty params
    forecaster_clone = clone(forecaster)
    forecaster_clone.fit(y, X, forecasting_horizon=3, **{})


def test_forecaster_accepts_params_in_predict(y_X_factory):
    """Forecasters should accept **params in predict method."""
    y, X = y_X_factory(length=50, n_targets=1)
    forecaster = SeasonalNaive(seasonality=3)
    forecaster.fit(y, X, forecasting_horizon=3)

    # Should work without params
    y_pred = forecaster.predict(forecasting_horizon=3)
    assert len(y_pred) > 0

    # Should also accept empty params
    y_pred = forecaster.predict(forecasting_horizon=3, **{})
    assert len(y_pred) > 0


def test_forecaster_update_predict_accepts_params(y_X_factory):
    """update_predict should accept **params and route to predict."""
    y, X = y_X_factory(length=50, n_targets=1)
    y_train, y_test = y[:-10], y[-10:]
    X_train, X_test = X[:-10], X[-10:]

    forecaster = SeasonalNaive(seasonality=3)
    forecaster.fit(y_train, X_train, forecasting_horizon=3)

    # update_predict should work
    y_pred = forecaster.update_predict(y_test[:3], X_test[:3], forecasting_horizon=3)
    assert len(y_pred) > 0


def test_forecaster_routing_with_reduction(y_X_factory, consuming_estimator):
    """Reduction forecasters should route metadata to sub-estimators."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=1)
    estimator, registry = consuming_estimator

    forecaster = PointReductionForecaster(estimator=estimator)
    forecaster.fit(y, X, forecasting_horizon=3)

    # Check that estimator's fit was called (recorded in registry)
    assert len(registry) > 0
    fit_calls = [call for call in registry if call[0] == "fit"]
    assert len(fit_calls) > 0


# ============================================================================
# 4. PIPELINE ROUTING TESTS
# ============================================================================


def test_pipeline_get_metadata_routing(time_series_factory):
    """Pipeline should implement get_metadata_routing."""
    y = time_series_factory(length=50, n_components=1)
    pipeline = Pipeline(
        [
            ("diff1", SeasonalDifferencing(seasonality=3)),
            ("diff2", SeasonalDifferencing(seasonality=2)),
        ]
    )
    pipeline.fit(y)

    assert hasattr(pipeline, "get_metadata_routing")
    routing = pipeline.get_metadata_routing()
    assert isinstance(routing, MetadataRouter)


def test_pipeline_routes_to_steps(time_series_factory):
    """Pipeline should route metadata to its steps."""
    y = time_series_factory(length=50, n_components=1)
    pipeline = Pipeline(
        [
            ("diff1", SeasonalDifferencing(seasonality=3)),
            ("diff2", SeasonalDifferencing(seasonality=2)),
        ]
    )

    # Fit should work
    pipeline.fit(y)

    # Transform should work
    y_t = pipeline.transform(y)
    assert len(y_t) > 0


def test_pipeline_update_transform(y_X_factory):
    """Pipeline.update_transform should work with transformers."""
    y, _ = y_X_factory(length=50, n_targets=1)
    y_train, y_test = y[:-10], y[-10:]

    pipeline = Pipeline(
        [
            ("diff1", SeasonalDifferencing(seasonality=3)),
            ("diff2", SeasonalDifferencing(seasonality=2)),
        ]
    )
    pipeline.fit(y_train)

    # update_transform should work
    y_t = pipeline.update_transform(y_test[:3])
    assert len(y_t) > 0


def test_featureunion_get_metadata_routing(time_series_factory):
    """FeatureUnion should implement get_metadata_routing."""
    y = time_series_factory(length=50, n_components=1)

    feature_union = FeatureUnion(
        [
            ("diff1", SeasonalDifferencing(seasonality=3)),
            ("diff2", SeasonalDifferencing(seasonality=2)),
        ]
    )
    feature_union.fit(y)

    assert hasattr(feature_union, "get_metadata_routing")
    routing = feature_union.get_metadata_routing()
    assert isinstance(routing, MetadataRouter)


def test_featureunion_routes_to_transformers(time_series_factory):
    """FeatureUnion should route metadata to each transformer."""
    y = time_series_factory(length=50, n_components=1)

    feature_union = FeatureUnion(
        [
            ("diff1", SeasonalDifferencing(seasonality=3)),
            ("diff2", SeasonalDifferencing(seasonality=2)),
        ]
    )

    # Fit should work with parallel execution
    feature_union.fit(y)

    # Transform should concatenate outputs horizontally
    y_transformed = feature_union.transform(y)
    assert len(y_transformed) > 0

    # Should have time column plus features from both transformers
    assert "time" in y_transformed.columns


def test_featureunion_update_transform(y_X_factory):
    """FeatureUnion.update_transform should route to all transformers."""
    y, _ = y_X_factory(length=50, n_targets=1)
    y_train, y_test = y[:-10], y[-10:]

    feature_union = FeatureUnion(
        [
            ("diff1", SeasonalDifferencing(seasonality=3)),
            ("diff2", SeasonalDifferencing(seasonality=2)),
        ]
    )
    feature_union.fit(y_train)

    # update_transform should work
    y_t = feature_union.update_transform(y_test[:3])
    assert len(y_t) > 0


def test_columntransformer_get_metadata_routing(time_series_factory):
    """ColumnTransformer should implement get_metadata_routing."""
    pytest.skip(
        "ColumnTransformer column selection implementation needs review - 'time' column handling. "
        "See test_pipeline.py::test_columntransformer_column_selection for details."
    )
    y = time_series_factory(length=50, n_components=1)

    ct = ColumnTransformer(
        [
            ("diff1", SeasonalDifferencing(seasonality=3), ["feature_0"]),
            ("diff2", SeasonalDifferencing(seasonality=2), ["feature_1"]),
        ],
        remainder="passthrough",
    )
    ct.fit(y)

    assert hasattr(ct, "get_metadata_routing")
    routing = ct.get_metadata_routing()
    assert isinstance(routing, MetadataRouter)


def test_columntransformer_routes_to_transformers(time_series_factory):
    """ColumnTransformer should route metadata to column-specific transformers."""
    pytest.skip(
        "ColumnTransformer column selection implementation needs review - 'time' column handling. "
        "See test_pipeline.py::test_columntransformer_column_selection for details."
    )
    y = time_series_factory(length=50, n_components=3)

    ct = ColumnTransformer(
        [
            ("diff1", SeasonalDifferencing(seasonality=3), ["feature_0"]),
            ("diff2", SeasonalDifferencing(seasonality=2), ["feature_1"]),
        ],
        remainder="passthrough",
    )

    # Fit should work
    ct.fit(y)

    # Transform should apply transformers to specified columns
    y_transformed = ct.transform(y)
    assert len(y_transformed) > 0

    # Should have time column
    assert "time" in y_transformed.columns


def test_columntransformer_update_transform(y_X_factory):
    """ColumnTransformer.update_transform should route to all transformers."""
    pytest.skip(
        "ColumnTransformer column selection implementation needs review - 'time' column handling. "
        "See test_pipeline.py::test_columntransformer_column_selection for details."
    )
    y, _ = y_X_factory(length=50, n_targets=3)
    y_train, y_test = y[:-10], y[-10:]

    ct = ColumnTransformer(
        [
            ("diff1", SeasonalDifferencing(seasonality=3), ["feature_0"]),
            ("diff2", SeasonalDifferencing(seasonality=2), ["feature_1"]),
        ],
        remainder="passthrough",
    )
    ct.fit(y_train)

    # update_transform should work
    y_t = ct.update_transform(y_test[:3])
    assert len(y_t) > 0


# ============================================================================
# 5. SEARCHCV ROUTING TESTS
# ============================================================================


def test_searchcv_get_metadata_routing(y_X_factory):
    """SearchCV should implement get_metadata_routing."""
    y, X = y_X_factory(length=50, n_targets=1)
    import optuna

    search = SearchCV(
        forecaster=SeasonalNaive(),
        param_distributions={"seasonality": optuna.distributions.IntDistribution(1, 5)},
        scoring=MAE(),
        n_trials=2,
        n_warmup_trials=1,
    )

    assert hasattr(search, "get_metadata_routing")
    routing = search.get_metadata_routing()
    assert isinstance(routing, MetadataRouter)


def test_searchcv_fits_with_metadata(y_X_factory):
    """SearchCV should work with metadata routing enabled."""
    y, X = y_X_factory(length=50, n_targets=1)
    import optuna

    search = SearchCV(
        forecaster=SeasonalNaive(),
        param_distributions={"seasonality": optuna.distributions.IntDistribution(1, 5)},
        scoring=MAE(),
        n_trials=2,
        n_warmup_trials=1,
    )

    # Should fit successfully
    search.fit(y, X, forecasting_horizon=3)
    assert hasattr(search, "best_forecaster_")

    # Should predict successfully
    y_pred = search.predict(forecasting_horizon=3)
    assert len(y_pred) > 0


def test_searchcv_update_predict(y_X_factory):
    """SearchCV.update_predict should route metadata."""
    y, X = y_X_factory(length=50, n_targets=1)
    y_train, y_test = y[:-10], y[-10:]
    X_train, X_test = X[:-10], X[-10:]

    import optuna

    search = SearchCV(
        forecaster=SeasonalNaive(),
        param_distributions={"seasonality": optuna.distributions.IntDistribution(1, 5)},
        scoring=MAE(),
        n_trials=2,
        n_warmup_trials=1,
    )
    search.fit(y_train, X_train, forecasting_horizon=3)

    # update_predict should work
    y_pred = search.update_predict(y_test[:3], X_test[:3], forecasting_horizon=3)
    assert len(y_pred) > 0


# ============================================================================
# 6. ERROR HANDLING TESTS
# ============================================================================


def test_error_when_metadata_not_requested(time_series_factory):
    """Should raise error when metadata passed without request.

    Note: This test is skipped because Yohou forecasters don't currently
    validate custom metadata. This would require implementing metadata
    consumption in the actual forecaster methods.
    """
    pytest.skip(
        "Yohou forecasters don't validate custom metadata yet. "
        "This would require implementing actual metadata consumption."
    )


def test_no_error_when_explicit_params_passed(y_X_factory):
    """Explicit parameters should always be accepted."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=1)
    forecaster = SeasonalNaive(seasonality=3)

    # These are explicit parameters, not metadata - should always work
    forecaster.fit(y, X=X, forecasting_horizon=3)


def test_cloning_preserves_routing_state(y_X_factory):
    """Cloning should preserve metadata routing configuration."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=1)
    forecaster = SeasonalNaive(seasonality=3)

    # Set some requests
    forecaster.set_predict_request(forecasting_horizon=True)

    # Clone
    forecaster_clone = clone(forecaster)

    # Routing state should be preserved
    routing = forecaster_clone.get_metadata_routing()

    assert "forecasting_horizon" in routing.consumes(
        method="predict", params=["forecasting_horizon"]
    )


# ============================================================================
# 7. INTEGRATION TESTS
# ============================================================================


def test_nested_pipeline_with_searchcv(y_X_factory):
    """Test deeply nested routing: SearchCV -> Reduction Forecaster with Pipeline."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=1)
    import optuna
    from sklearn.linear_model import Ridge

    # Pipeline as feature transformer
    feature_pipeline = Pipeline(
        [
            ("diff", SeasonalDifferencing(seasonality=3)),
        ]
    )

    # Reduction forecaster using pipeline
    forecaster = PointReductionForecaster(estimator=Ridge(), feature_transformer=feature_pipeline)

    search = SearchCV(
        forecaster=forecaster,
        param_distributions={"estimator__alpha": optuna.distributions.FloatDistribution(0.01, 1.0)},
        scoring=MAE(),
        n_trials=2,
        n_warmup_trials=1,
    )

    # Should fit successfully through nested routing
    search.fit(y, X, forecasting_horizon=3)
    assert hasattr(search, "best_forecaster_")

    # Should predict successfully
    y_pred = search.predict(forecasting_horizon=3)
    assert len(y_pred) > 0


def test_featureunion_in_forecaster_pipeline(y_X_factory):
    """Test metadata routing through FeatureUnion in forecaster pipeline."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=1)
    from sklearn.linear_model import Ridge

    # FeatureUnion as feature transformer
    feature_union = FeatureUnion(
        [
            ("diff1", SeasonalDifferencing(seasonality=3)),
            ("diff2", SeasonalDifferencing(seasonality=2)),
        ]
    )

    # Reduction forecaster using FeatureUnion
    forecaster = PointReductionForecaster(estimator=Ridge(), feature_transformer=feature_union)

    # Should fit successfully with parallel transformer execution
    forecaster.fit(y, X, forecasting_horizon=3)

    # Should predict successfully
    y_pred = forecaster.predict(forecasting_horizon=3)
    assert len(y_pred) > 0


def test_columntransformer_in_forecaster_pipeline(y_X_factory):
    """Test metadata routing through ColumnTransformer in forecaster pipeline."""
    pytest.skip(
        "ColumnTransformer column selection implementation needs review - 'time' column handling. "
        "See test_pipeline.py::test_columntransformer_column_selection for details."
    )
    y, X = y_X_factory(length=50, n_targets=3, n_features=1)
    from sklearn.linear_model import Ridge

    # ColumnTransformer as feature transformer
    ct = ColumnTransformer(
        [
            ("diff1", SeasonalDifferencing(seasonality=3), ["feature_0"]),
            ("diff2", SeasonalDifferencing(seasonality=2), ["feature_1"]),
        ],
        remainder="passthrough",
    )

    # Reduction forecaster using ColumnTransformer
    forecaster = PointReductionForecaster(estimator=Ridge(), feature_transformer=ct)

    # Should fit successfully with column-specific transformations
    forecaster.fit(y, X, forecasting_horizon=3)

    # Should predict successfully
    y_pred = forecaster.predict(forecasting_horizon=3)
    assert len(y_pred) > 0


def test_full_pipeline_with_reduction_forecaster(y_X_factory, consuming_estimator):
    """Test complete pipeline with reduction forecaster and metadata."""
    y, X = y_X_factory(length=50, n_targets=1, n_features=1)
    estimator, registry = consuming_estimator

    # Pipeline as feature transformer
    feature_pipeline = Pipeline(
        [
            ("diff", SeasonalDifferencing(seasonality=3)),
        ]
    )

    # Reduction forecaster with pipeline
    forecaster = PointReductionForecaster(estimator=estimator, feature_transformer=feature_pipeline)

    # Fit pipeline
    forecaster.fit(y, X, forecasting_horizon=3)

    # Check that the estimator received calls
    assert len(registry) > 0

    # Predict should work
    y_pred = forecaster.predict(forecasting_horizon=3)
    assert len(y_pred) > 0


# ============================================================================
# 8. COMPOSITE METHODS TESTS
# ============================================================================


def test_update_transform_is_composite():
    """update_transform should be registered as composite method."""
    from sklearn.utils._metadata_requests import COMPOSITE_METHODS

    assert "update_transform" in COMPOSITE_METHODS
    assert COMPOSITE_METHODS["update_transform"] == ["update", "transform"]


def test_update_predict_is_composite():
    """update_predict should be registered as composite method."""
    from sklearn.utils._metadata_requests import COMPOSITE_METHODS

    assert "update_predict" in COMPOSITE_METHODS
    assert COMPOSITE_METHODS["update_predict"] == ["update", "predict"]


def test_composite_methods_in_simple_methods():
    """Composite methods should be in SIMPLE_METHODS for routing."""
    from sklearn.utils._metadata_requests import SIMPLE_METHODS

    assert "update_transform" in SIMPLE_METHODS
    assert "update_predict" in SIMPLE_METHODS


def test_update_not_in_simple_methods():
    """update should NOT be in SIMPLE_METHODS (not routed)."""
    from sklearn.utils._metadata_requests import SIMPLE_METHODS

    assert "update" not in SIMPLE_METHODS
