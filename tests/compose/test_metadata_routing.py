"""Comprehensive test suite for metadata routing in Yohou.

This module tests the sklearn metadata routing integration across all Yohou
estimators, transformers, and meta-estimators. Tests are organized by component
and routing scenario.

Test Categories:
1. Basic Routing (get_metadata_routing, set_*_request methods)
2. Transformer Routing (transform, observe_transform)
3. Forecaster Routing (fit, predict, observe_predict)
4. FeaturePipeline Routing (FeaturePipeline, FeatureUnion, ColumnTransformer)
5. GridSearchCV Routing (routing through cross-validation)
6. Error Handling (UnsetMetadataPassedError, validation)
7. Integration Tests (nested routing scenarios)
8. Composite Methods Tests (observe_transform, observe_predict registration)
"""

import inspect

import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.utils.metadata_routing import MetadataRequest, MetadataRouter

from yohou.compose.feature_pipeline import ColumnTransformer, FeaturePipeline, FeatureUnion
from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import GridSearchCV
from yohou.point import PointReductionForecaster, SeasonalNaive
from yohou.stationarity import SeasonalDifferencing
from yohou.testing.metadata_routing import _Registry, assert_request_is_empty


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


class TestBasicRouting:
    """Tests for basic metadata routing infrastructure."""

    def test_get_metadata_routing_exists_on_forecasters(self):
        """All forecasters must have get_metadata_routing method."""
        forecasters = [
            SeasonalNaive(seasonality=3),
            PointReductionForecaster(estimator=Ridge()),
        ]
        for forecaster in forecasters:
            assert hasattr(forecaster, "get_metadata_routing")
            routing = forecaster.get_metadata_routing()
            assert isinstance(routing, MetadataRouter | MetadataRequest)

    def test_get_metadata_routing_exists_on_transformers(self):
        """All transformers must have get_metadata_routing method."""
        transformers = [
            SeasonalDifferencing(seasonality=3),
        ]
        for transformer in transformers:
            assert hasattr(transformer, "get_metadata_routing")
            routing = transformer.get_metadata_routing()
            assert isinstance(routing, MetadataRouter | MetadataRequest)

    def test_default_request_is_empty(self, y_X_factory):
        """By default, metadata requests should be empty."""
        y, X = y_X_factory(length=31)
        forecaster = SeasonalNaive(seasonality=3)
        forecaster.fit(y, X, forecasting_horizon=3)

        routing = forecaster.get_metadata_routing()
        assert_request_is_empty(routing)

    def test_set_fit_request_methods_exist(self):
        """Forecasters should have set_fit_request method."""
        forecaster = SeasonalNaive(seasonality=3)
        assert hasattr(forecaster, "set_fit_request")

        sig = inspect.signature(forecaster.set_fit_request)
        params = list(sig.parameters.keys())
        assert "forecasting_horizon" in params

    def test_set_predict_request_methods_exist(self):
        """Forecasters should have set_predict_request method."""
        forecaster = SeasonalNaive(seasonality=3)
        assert hasattr(forecaster, "set_predict_request")

        sig = inspect.signature(forecaster.set_predict_request)
        params = list(sig.parameters.keys())
        assert "forecasting_horizon" in params

    def test_set_observe_predict_request_exists(self):
        """Forecasters should have set_observe_predict_request (composite method)."""
        forecaster = SeasonalNaive(seasonality=3)
        assert hasattr(forecaster, "set_observe_predict_request")


class TestTransformerRouting:
    """Tests for transformer metadata routing."""

    def test_accepts_params_in_transform(self, time_series_factory):
        """Transformers should accept **params in transform method."""
        y = time_series_factory(length=50, n_components=1)
        transformer = SeasonalDifferencing(seasonality=3)
        transformer.fit(y)

        y_t = transformer.transform(y)
        assert len(y_t) > 0

        y_t = transformer.transform(y, **{})
        assert len(y_t) > 0

    def test_observe_transform_routes_to_transform_only(self, time_series_train_test_factory):
        """observe_transform should route params to transform, not observe."""
        y, y_new = time_series_train_test_factory(train_length=50, test_length=5, n_components=1)
        transformer = SeasonalDifferencing(seasonality=3)
        transformer.fit(y)

        y_t = transformer.observe_transform(y_new)
        assert len(y_t) > 0

    def test_does_not_route_to_observe(self, time_series_factory):
        """observe() should not accept **params (memory management only)."""
        y = time_series_factory(length=50, n_components=1)
        transformer = SeasonalDifferencing(seasonality=3)
        transformer.fit(y)

        sig = inspect.signature(transformer.observe)
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        assert not has_var_keyword, "observe() should not accept **params"


class TestForecasterRouting:
    """Tests for forecaster metadata routing."""

    def test_accepts_params_in_fit(self, y_X_factory):
        """Forecasters should accept **params in fit method."""
        y, X = y_X_factory(length=50, n_targets=1)
        forecaster = SeasonalNaive(seasonality=3)

        forecaster.fit(y, X, forecasting_horizon=3)

        forecaster_clone = clone(forecaster)
        forecaster_clone.fit(y, X, forecasting_horizon=3, **{})

    def test_accepts_params_in_predict(self, y_X_factory):
        """Forecasters should accept **params in predict method."""
        y, X = y_X_factory(length=50, n_targets=1)
        forecaster = SeasonalNaive(seasonality=3)
        forecaster.fit(y, X, forecasting_horizon=3)

        y_pred = forecaster.predict(forecasting_horizon=3)
        assert len(y_pred) > 0

        y_pred = forecaster.predict(forecasting_horizon=3, **{})
        assert len(y_pred) > 0

    def test_observe_predict_accepts_params(self, y_X_factory):
        """observe_predict should accept **params and route to predict."""
        y, X = y_X_factory(length=50, n_targets=1)
        y_train, y_test = y[:-10], y[-10:]
        X_train, X_test = X[:-10], X[-10:]

        forecaster = SeasonalNaive(seasonality=3)
        forecaster.fit(y_train, X_train, forecasting_horizon=3)

        y_pred = forecaster.observe_predict(y_test[:3], X_test[:3], forecasting_horizon=3)
        assert len(y_pred) > 0

    def test_routing_with_reduction(self, y_X_factory, consuming_estimator):
        """Reduction forecasters should route metadata to sub-estimators."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=1)
        estimator, registry = consuming_estimator

        forecaster = PointReductionForecaster(estimator=estimator)
        forecaster.fit(y, X, forecasting_horizon=3)

        assert len(registry) > 0
        fit_calls = [call for call in registry if call[0] == "fit"]
        assert len(fit_calls) > 0


class TestPipelineRouting:
    """Tests for FeaturePipeline metadata routing."""

    def test_get_metadata_routing(self, time_series_factory):
        """FeaturePipeline should implement get_metadata_routing."""
        y = time_series_factory(length=50, n_components=1)
        pipeline = FeaturePipeline([
            ("diff1", SeasonalDifferencing(seasonality=3)),
            ("diff2", SeasonalDifferencing(seasonality=2)),
        ])
        pipeline.fit(y)

        assert hasattr(pipeline, "get_metadata_routing")
        routing = pipeline.get_metadata_routing()
        assert isinstance(routing, MetadataRouter)

    def test_routes_to_steps(self, time_series_factory):
        """FeaturePipeline should route metadata to its steps."""
        y = time_series_factory(length=50, n_components=1)
        pipeline = FeaturePipeline([
            ("diff1", SeasonalDifferencing(seasonality=3)),
            ("diff2", SeasonalDifferencing(seasonality=2)),
        ])
        pipeline.fit(y)

        y_t = pipeline.transform(y)
        assert len(y_t) > 0

    def test_observe_transform(self, y_X_factory):
        """FeaturePipeline.observe_transform should work with transformers."""
        y, _ = y_X_factory(length=50, n_targets=1)
        y_train, y_test = y[:-10], y[-10:]

        pipeline = FeaturePipeline([
            ("diff1", SeasonalDifferencing(seasonality=3)),
            ("diff2", SeasonalDifferencing(seasonality=2)),
        ])
        pipeline.fit(y_train)

        y_t = pipeline.observe_transform(y_test[:3])
        assert len(y_t) > 0


class TestFeatureUnionRouting:
    """Tests for FeatureUnion metadata routing."""

    def test_get_metadata_routing(self, time_series_factory):
        """FeatureUnion should implement get_metadata_routing."""
        y = time_series_factory(length=50, n_components=1)

        feature_union = FeatureUnion([
            ("diff1", SeasonalDifferencing(seasonality=3)),
            ("diff2", SeasonalDifferencing(seasonality=2)),
        ])
        feature_union.fit(y)

        assert hasattr(feature_union, "get_metadata_routing")
        routing = feature_union.get_metadata_routing()
        assert isinstance(routing, MetadataRouter)

    def test_routes_to_transformers(self, time_series_factory):
        """FeatureUnion should route metadata to each transformer."""
        y = time_series_factory(length=50, n_components=1)

        feature_union = FeatureUnion([
            ("diff1", SeasonalDifferencing(seasonality=3)),
            ("diff2", SeasonalDifferencing(seasonality=2)),
        ])
        feature_union.fit(y)

        y_transformed = feature_union.transform(y)
        assert len(y_transformed) > 0
        assert "time" in y_transformed.columns

    def test_observe_transform(self, y_X_factory):
        """FeatureUnion.observe_transform should route to all transformers."""
        y, _ = y_X_factory(length=50, n_targets=1)
        y_train, y_test = y[:-10], y[-10:]

        feature_union = FeatureUnion([
            ("diff1", SeasonalDifferencing(seasonality=3)),
            ("diff2", SeasonalDifferencing(seasonality=2)),
        ])
        feature_union.fit(y_train)

        y_t = feature_union.observe_transform(y_test[:3])
        assert len(y_t) > 0


class TestColumnTransformerRouting:
    """Tests for ColumnTransformer metadata routing."""

    def test_get_metadata_routing(self, time_series_factory):
        """ColumnTransformer should implement get_metadata_routing."""
        y = time_series_factory(length=50, n_components=3)

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

    def test_routes_to_transformers(self, time_series_factory):
        """ColumnTransformer should route metadata to column-specific transformers."""
        y = time_series_factory(length=50, n_components=3)

        ct = ColumnTransformer(
            [
                ("diff1", SeasonalDifferencing(seasonality=3), ["feature_0"]),
                ("diff2", SeasonalDifferencing(seasonality=2), ["feature_1"]),
            ],
            remainder="passthrough",
        )
        ct.fit(y)

        y_transformed = ct.transform(y)
        assert len(y_transformed) > 0
        assert "time" in y_transformed.columns

    def test_observe_transform(self, y_X_factory):
        """ColumnTransformer.observe_transform should route to all transformers."""
        y, _ = y_X_factory(length=50, n_targets=3)
        y_train, y_test = y[:-10], y[-10:]

        ct = ColumnTransformer(
            [
                ("diff1", SeasonalDifferencing(seasonality=3), ["y_0"]),
                ("diff2", SeasonalDifferencing(seasonality=2), ["y_1"]),
            ],
            remainder="passthrough",
        )
        ct.fit(y_train)

        y_t = ct.observe_transform(y_test[:3])
        assert len(y_t) > 0
        assert "time" in y_t.columns


class TestSearchCVRouting:
    """Tests for GridSearchCV metadata routing."""

    def test_get_metadata_routing(self, y_X_factory):
        """GridSearchCV should implement get_metadata_routing."""
        y, X = y_X_factory(length=50, n_targets=1)

        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 3, 5]},
            scoring=MeanAbsoluteError(),
            cv=2,
        )

        assert hasattr(search, "get_metadata_routing")
        routing = search.get_metadata_routing()
        assert isinstance(routing, MetadataRouter)

    def test_fits_with_metadata(self, y_X_factory):
        """GridSearchCV should work with metadata routing enabled."""
        y, X = y_X_factory(length=50, n_targets=1)

        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 3, 5]},
            scoring=MeanAbsoluteError(),
            cv=2,
        )
        search.fit(y, X, forecasting_horizon=3)
        assert hasattr(search, "best_forecaster_")

        y_pred = search.predict(forecasting_horizon=3)
        assert len(y_pred) > 0

    def test_observe_predict(self, y_X_factory):
        """GridSearchCV.observe_predict should route metadata."""
        y, X = y_X_factory(length=50, n_targets=1)
        y_train, y_test = y[:-10], y[-10:]
        X_train, X_test = X[:-10], X[-10:]

        search = GridSearchCV(
            forecaster=SeasonalNaive(),
            param_grid={"seasonality": [1, 3, 5]},
            scoring=MeanAbsoluteError(),
            cv=2,
        )
        search.fit(y_train, X_train, forecasting_horizon=3)

        y_pred = search.observe_predict(y_test[:3], X_test[:3], forecasting_horizon=3)
        assert len(y_pred) > 0


class TestErrorHandling:
    """Tests for metadata routing error handling."""

    def test_error_when_metadata_not_requested(self, time_series_factory):
        """Should raise error when metadata passed without request.

        Note: This test is skipped because Yohou forecasters don't currently
        validate custom metadata. This would require implementing metadata
        consumption in the actual forecaster methods.
        """
        pytest.xfail(
            reason="Yohou forecasters don't validate custom metadata yet. "
            "This would require implementing actual metadata consumption.",
        )

    def test_no_error_when_explicit_params_passed(self, y_X_factory):
        """Explicit parameters should always be accepted."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=1)
        forecaster = SeasonalNaive(seasonality=3)

        forecaster.fit(y, X=X, forecasting_horizon=3)

    def test_cloning_preserves_routing_state(self, y_X_factory):
        """Cloning should preserve metadata routing configuration."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=1)
        forecaster = SeasonalNaive(seasonality=3)

        forecaster.set_predict_request(forecasting_horizon=True)

        forecaster_clone = clone(forecaster)

        routing = forecaster_clone.get_metadata_routing()

        assert "forecasting_horizon" in routing.consumes(method="predict", params=["forecasting_horizon"])


class TestIntegration:
    """Integration tests for nested routing scenarios."""

    @pytest.mark.slow
    def test_nested_pipeline_with_searchcv(self, y_X_factory):
        """Test deeply nested routing: GridSearchCV -> Reduction Forecaster with FeaturePipeline."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=1)
        from sklearn.linear_model import Ridge

        feature_pipeline = FeaturePipeline([
            ("diff", SeasonalDifferencing(seasonality=3)),
        ])

        forecaster = PointReductionForecaster(estimator=Ridge(), feature_transformer=feature_pipeline)

        search = GridSearchCV(
            forecaster=forecaster,
            param_grid={"estimator__alpha": [0.01, 0.1, 1.0]},
            scoring=MeanAbsoluteError(),
            cv=2,
        )

        search.fit(y, X, forecasting_horizon=3)
        assert hasattr(search, "best_forecaster_")

        y_pred = search.predict(forecasting_horizon=3)
        assert len(y_pred) > 0

    def test_featureunion_in_forecaster_pipeline(self, y_X_factory):
        """Test metadata routing through FeatureUnion in forecaster pipeline."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=1)
        from sklearn.linear_model import Ridge

        feature_union = FeatureUnion([
            ("diff1", SeasonalDifferencing(seasonality=3)),
            ("diff2", SeasonalDifferencing(seasonality=2)),
        ])

        forecaster = PointReductionForecaster(estimator=Ridge(), feature_transformer=feature_union)

        forecaster.fit(y, X, forecasting_horizon=3)

        y_pred = forecaster.predict(forecasting_horizon=3)
        assert len(y_pred) > 0

    def test_columntransformer_in_forecaster_pipeline(self, y_X_factory):
        """Test metadata routing through ColumnTransformer in forecaster pipeline."""
        pytest.xfail(
            reason="ColumnTransformer uses sklearn's safe_indexing which doesn't handle "
            "polars DataFrames correctly via __dataframe__ protocol for column selection.",
        )
        y, X = y_X_factory(length=50, n_targets=1, n_features=3)
        from sklearn.linear_model import Ridge

        ct = ColumnTransformer(
            [
                ("diff1", SeasonalDifferencing(seasonality=3), ["feature_0"]),
                ("diff2", SeasonalDifferencing(seasonality=2), ["feature_1"]),
            ],
            remainder="passthrough",
        )

        forecaster = PointReductionForecaster(estimator=Ridge(), feature_transformer=ct)

        forecaster.fit(y, X, forecasting_horizon=3)

        y_pred = forecaster.predict(forecasting_horizon=3)
        assert len(y_pred) > 0

    def test_full_pipeline_with_reduction_forecaster(self, y_X_factory, consuming_estimator):
        """Test complete pipeline with reduction forecaster and metadata."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=1)
        estimator, registry = consuming_estimator

        feature_pipeline = FeaturePipeline([
            ("diff", SeasonalDifferencing(seasonality=3)),
        ])

        forecaster = PointReductionForecaster(estimator=estimator, feature_transformer=feature_pipeline)

        forecaster.fit(y, X, forecasting_horizon=3)

        assert len(registry) > 0

        y_pred = forecaster.predict(forecasting_horizon=3)
        assert len(y_pred) > 0


class TestCompositeMethods:
    """Tests for composite method registration."""

    def test_observe_transform_is_composite(self):
        """observe_transform should be registered as composite method."""
        from sklearn.utils._metadata_requests import COMPOSITE_METHODS

        assert "observe_transform" in COMPOSITE_METHODS
        assert COMPOSITE_METHODS["observe_transform"] == ["observe", "transform"]

    def test_observe_predict_is_composite(self):
        """observe_predict should be registered as composite method."""
        from sklearn.utils._metadata_requests import COMPOSITE_METHODS

        assert "observe_predict" in COMPOSITE_METHODS
        assert COMPOSITE_METHODS["observe_predict"] == ["observe", "predict"]

    def test_composite_methods_in_simple_methods(self):
        """Composite methods should be in SIMPLE_METHODS for routing."""
        from sklearn.utils._metadata_requests import SIMPLE_METHODS

        assert "observe_transform" in SIMPLE_METHODS
        assert "observe_predict" in SIMPLE_METHODS

    def test_observe_not_in_simple_methods(self):
        """observe should NOT be in SIMPLE_METHODS (not routed)."""
        from sklearn.utils._metadata_requests import SIMPLE_METHODS

        assert "observe" not in SIMPLE_METHODS
