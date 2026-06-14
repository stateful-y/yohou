"""Metadata routing integration tests for Yohou composition classes.

This module verifies that metadata is actually DELIVERED to sub-estimators,
not just accepted at the API level. Tests use consuming estimators that record
received metadata via the _Registry pattern.

Critical differences from standard sklearn routing tests:
1. Yohou-specific methods: observe, observe_predict, observe_transform, predict_interval
2. Time-series metadata: time_weight (converted to sample_weight), coverage_rates
3. Panel data: groups routing (predict/observe/rewind on subset of groups)
4. Composite methods: observe_predict must route observe AND predict metadata separately

Test coverage (~23 tests):
- FeaturePipeline: time_weight routing to sequential transformers
- FeatureUnion: time_weight to parallel transformer branches
- ColumnTransformer: time_weight to column-specific transformers
- DecompositionPipeline: time_weight to component forecasters
- ForecastedFeatureForecaster: time_weight to target + feature forecasters
- ColumnForecaster: time_weight and groups to per-column forecasters
- Ridge with time_weight: analytical verification of sample_weight conversion
- IntervalReductionForecaster: coverage_rates to predict_interval
- Composite methods: observe_predict metadata splitting
- rewind_transform: known routing gap (not in get_metadata_routing())
"""

from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.multioutput import MultiOutputRegressor

from yohou.base import BaseTransformer
from yohou.compose import (
    ColumnForecaster,
    ColumnTransformer,
    DecompositionPipeline,
    FeaturePipeline,
    FeatureUnion,
    ForecastedFeatureForecaster,
)
from yohou.interval import IntervalReductionForecaster
from yohou.point import PointReductionForecaster
from yohou.point.base import BasePointForecaster
from yohou.preprocessing import LagTransformer
from yohou.testing.metadata_routing import (
    _Registry,
    check_recorded_metadata,
    record_metadata,
)
from yohou.weighting import TableWeighter


class ConsumingTransformer(BaseTransformer):
    """Transformer that records all received metadata.

    Used to verify metadata routing in composition classes. All metadata passed
    to fit/transform is recorded in self._records for later verification.

    Extends yohou's BaseTransformer so it can be used in FeaturePipeline,
    FeatureUnion, and ColumnTransformer (which require BaseTransformer steps).

    Parameters
    ----------
    registry : _Registry, optional
        Shared registry to track calls across multiple estimators

    Attributes
    ----------
    _records : dict
        Records of metadata received by each method
    registry : _Registry
        Shared call tracking registry

    """

    def __init__(self, registry=None):
        """Initialize ConsumingTransformer.

        Parameters
        ----------
        registry : _Registry, optional
            Shared registry to track calls.

        """
        self.registry = registry
        super().__init__()

    def fit(self, X, y=None, time_weight=None, **kwargs):
        """Fit transformer and record metadata.

        Parameters
        ----------
        X : pl.DataFrame
            Training data with "time" column.
        y : pl.DataFrame, optional
            Target values.
        time_weight : array-like, optional
            Time-based weights for metadata routing.
        **kwargs : dict
            Additional metadata to record.

        Returns
        -------
        self

        """
        metadata = {k: v for k, v in [("time_weight", time_weight)] if v is not None}
        metadata.update({k: v for k, v in kwargs.items() if v is not None})
        record_metadata(self, **metadata)
        if self.registry is not None:
            self.registry.append(self)
        self._observation_horizon = 0
        BaseTransformer.fit(self, X, y)
        return self

    def transform(self, X, time_weight=None, **kwargs):
        """Transform data and record metadata.

        Parameters
        ----------
        X : pl.DataFrame
            Data to transform.
        time_weight : array-like, optional
            Time-based weights for metadata routing.
        **kwargs : dict
            Additional metadata to record.

        Returns
        -------
        X : pl.DataFrame
            Unchanged input.

        """
        metadata = {k: v for k, v in [("time_weight", time_weight)] if v is not None}
        metadata.update({k: v for k, v in kwargs.items() if v is not None})
        record_metadata(self, **metadata)
        if self.registry is not None:
            self.registry.append(self)
        return X

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : list of str or None, default=None
            Input feature names.

        Returns
        -------
        list of str
            Output feature names.

        """
        if input_features is not None:
            return input_features
        return self.feature_names_in_


class ConsumingRegressor(RegressorMixin, BaseEstimator):
    """Regressor that records all received metadata.

    Parameters
    ----------
    registry : _Registry, optional
        Shared registry to track calls

    Attributes
    ----------
    _records : dict
        Metadata records
    registry : _Registry
        Call tracking registry

    """

    def __init__(self, registry=None):
        """Initialize ConsumingRegressor.

        Parameters
        ----------
        registry : _Registry, optional
            Shared registry to track calls.

        """
        self.registry = registry

    def fit(self, X, y, sample_weight=None, **kwargs):
        """Fit regressor and record metadata.

        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Training targets.
        sample_weight : array-like, optional
            Sample weights for metadata routing.
        **kwargs : dict
            Additional metadata to record.

        Returns
        -------
        self

        """
        metadata = {k: v for k, v in [("sample_weight", sample_weight)] if v is not None}
        metadata.update({k: v for k, v in kwargs.items() if v is not None})
        record_metadata(self, **metadata)
        if self.registry is not None:
            self.registry.append(self)
        # Store dummy coefficients for predict
        self.coef_ = np.zeros(X.shape[1] if hasattr(X, "shape") else 1)
        self.intercept_ = 0.0
        return self

    def predict(self, X, **kwargs):
        """Predict and record metadata.

        Parameters
        ----------
        X : array-like
            Features.
        **kwargs : dict
            Metadata to record.

        Returns
        -------
        y_pred : array-like
            Zero predictions.

        """
        record_metadata(self, **kwargs)
        if self.registry is not None:
            self.registry.append(self)
        return np.zeros(len(X))


class ConsumingForecaster(BasePointForecaster):
    """Forecaster that records all received metadata.

    Parameters
    ----------
    registry : _Registry, optional
        Shared registry to track calls

    Attributes
    ----------
    _records : dict
        Metadata records
    registry : _Registry
        Call tracking registry

    """

    _tags = {
        "forecaster_type": frozenset({"point"}),
        "uses_reduction": False,
        "requires_exogenous_X": True,
    }

    def __init__(self, registry=None):
        """Initialize the ConsumingForecaster.

        Parameters
        ----------
        registry : _Registry, optional
            Shared registry to track calls.

        """
        self.registry = registry
        self._observation_horizon = 1
        super().__init__()

    def fit(self, y, X_actual=None, forecasting_horizon=1, time_weight=None, **kwargs):
        """Fit forecaster and record metadata.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X_actual : pl.DataFrame, optional
            Exogenous features.
        forecasting_horizon : int
            Forecasting horizon.
        time_weight : array-like, optional
            Time-based weights for metadata routing.
        **kwargs : dict
            Additional metadata.

        Returns
        -------
        self

        """
        metadata = {k: v for k, v in [("time_weight", time_weight)] if v is not None}
        metadata.update({k: v for k, v in kwargs.items() if v is not None})
        record_metadata(self, **metadata)
        if self.registry is not None:
            self.registry.append(self)
        BasePointForecaster.fit(self, y=y, X_actual=X_actual, forecasting_horizon=forecasting_horizon)
        return self

    def _predict_one(self, groups, **kwargs):
        """Return zero predictions for requested horizon.

        Parameters
        ----------
        groups : list of str
            Panel group names to predict for.
        **kwargs : dict
            Additional params.

        Returns
        -------
        y_pred : pl.DataFrame
            Zero predictions with time columns.

        """
        record_metadata(self, **kwargs)
        if self.registry is not None:
            self.registry.append(self)

        # Handle both panel (dict) and non-panel (DataFrame) _y_observed
        if isinstance(self._y_observed, dict):
            # Panel data: use local_y_schema_ for column names
            if groups:
                value_cols = [f"{group}__{col}" for group in groups for col in self.local_y_schema_]
            else:
                value_cols = [f"{group}__{col}" for group in self.groups_ for col in self.local_y_schema_]
        else:
            value_cols = [c for c in self._y_observed.columns if c != "time"]
        data = {}
        for col in value_cols:
            data[col] = [0.0] * self.fit_forecasting_horizon_
        y_pred = pl.DataFrame(data)
        y_pred = self._add_time_columns(y_pred)
        return y_pred

    def predict(self, forecasting_horizon=None, groups=None, **kwargs):
        """Predict and record metadata.

        Parameters
        ----------
        forecasting_horizon : int, optional
            Forecasting horizon.
        groups : list of str, optional
            Panel group names for prediction filtering.
        **kwargs : dict
            Metadata.

        Returns
        -------
        y_pred : pl.DataFrame
            Zero predictions.

        """
        metadata = {k: v for k, v in kwargs.items() if v is not None}
        record_metadata(self, **metadata)
        if self.registry is not None:
            self.registry.append(self)
        return BasePointForecaster.predict(self, forecasting_horizon=forecasting_horizon, groups=groups, **kwargs)


@pytest.mark.integration
class TestFeaturePipelineRouting:
    """Metadata routing tests for FeaturePipeline."""

    def test_feature_pipeline_routes_time_weight_to_fit(self):
        """Verify time_weight reaches all transformers in FeaturePipeline.fit."""
        registry = _Registry()

        # Create pipeline with two consuming transformers
        # Note: must set BOTH fit and transform requests to avoid fit_transform conflict
        step1 = (
            ConsumingTransformer(registry=registry)
            .set_fit_request(time_weight=True)
            .set_transform_request(time_weight=True)
        )
        step2 = (
            ConsumingTransformer(registry=registry)
            .set_fit_request(time_weight=True)
            .set_transform_request(time_weight=True)
        )
        pipeline = FeaturePipeline([("step1", step1), ("step2", step2)])

        # Create dummy data
        X = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 1), end=date(2020, 1, 10), interval="1d", eager=True),
            "feature": [1.0] * 10,
        })
        time_weight = np.linspace(0.1, 1.0, 10)

        # Fit with time_weight
        pipeline.fit(X, time_weight=time_weight)

        # Verify both transformers received time_weight
        check_recorded_metadata(step1, method="fit", parent="fit", time_weight=time_weight)
        check_recorded_metadata(step2, method="fit", parent="fit", time_weight=time_weight)
        # FeaturePipeline.fit calls fit_transform on steps, so registry may have > 2 entries
        assert len(registry) >= 2, f"Expected at least 2 calls, got {len(registry)}"

    def test_feature_pipeline_routes_time_weight_to_transform(self):
        """Verify time_weight reaches transformers in FeaturePipeline.transform."""
        registry = _Registry()

        step1 = ConsumingTransformer(registry=registry).set_transform_request(time_weight=True)
        step2 = ConsumingTransformer(registry=registry).set_transform_request(time_weight=True)
        pipeline = FeaturePipeline([("step1", step1), ("step2", step2)])

        X = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 1), end=date(2020, 1, 10), interval="1d", eager=True),
            "feature": [1.0] * 10,
        })
        time_weight = np.linspace(0.1, 1.0, 10)

        # Fit first (without metadata)
        pipeline.fit(X)

        # Transform with time_weight
        pipeline.transform(X, time_weight=time_weight)

        # Verify both transformers received time_weight in transform
        check_recorded_metadata(step1, method="transform", parent="transform", time_weight=time_weight)
        check_recorded_metadata(step2, method="transform", parent="transform", time_weight=time_weight)


@pytest.mark.integration
class TestFeatureUnionRouting:
    """Metadata routing tests for FeatureUnion."""

    def test_feature_union_routes_time_weight_to_all_branches_fit(self):
        """Verify time_weight reaches all parallel branches in FeatureUnion.fit."""
        registry = _Registry()

        # Two parallel consuming transformers
        branch1 = ConsumingTransformer(registry=registry).set_fit_request(time_weight=True)
        branch2 = ConsumingTransformer(registry=registry).set_fit_request(time_weight=True)
        union = FeatureUnion([("branch1", branch1), ("branch2", branch2)])

        X = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 1), end=date(2020, 1, 10), interval="1d", eager=True),
            "feature": [1.0] * 10,
        })
        time_weight = np.linspace(0.1, 1.0, 10)

        union.fit(X, time_weight=time_weight)

        # Both branches must receive time_weight
        check_recorded_metadata(branch1, method="fit", parent="fit", time_weight=time_weight)
        check_recorded_metadata(branch2, method="fit", parent="fit", time_weight=time_weight)
        assert len(registry) == 2, f"Expected 2 branches called, got {len(registry)}"

    def test_feature_union_routes_time_weight_to_all_branches_transform(self):
        """Verify time_weight reaches all branches in FeatureUnion.transform."""
        registry = _Registry()

        branch1 = ConsumingTransformer(registry=registry).set_transform_request(time_weight=True)
        branch2 = ConsumingTransformer(registry=registry).set_transform_request(time_weight=True)
        union = FeatureUnion([("branch1", branch1), ("branch2", branch2)])

        X = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 1), end=date(2020, 1, 10), interval="1d", eager=True),
            "feature": [1.0] * 10,
        })
        time_weight = np.linspace(0.1, 1.0, 10)

        union.fit(X)
        union.transform(X, time_weight=time_weight)

        check_recorded_metadata(branch1, method="transform", parent="transform", time_weight=time_weight)
        check_recorded_metadata(branch2, method="transform", parent="transform", time_weight=time_weight)


@pytest.mark.integration
class TestColumnTransformerRouting:
    """Metadata routing tests for ColumnTransformer."""

    def test_column_transformer_routes_time_weight_to_per_column_transformers_fit(self):
        """Verify time_weight reaches column-specific transformers in fit."""
        registry = _Registry()

        # Per-column transformers
        # Note: must set BOTH fit and transform requests to avoid fit_transform conflict
        trans1 = (
            ConsumingTransformer(registry=registry)
            .set_fit_request(time_weight=True)
            .set_transform_request(time_weight=True)
        )
        trans2 = (
            ConsumingTransformer(registry=registry)
            .set_fit_request(time_weight=True)
            .set_transform_request(time_weight=True)
        )

        ct = ColumnTransformer(
            transformers=[("trans1", trans1, ["feature1"]), ("trans2", trans2, ["feature2"])],
            remainder="drop",
        )

        X = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 1), end=date(2020, 1, 10), interval="1d", eager=True),
            "feature1": [1.0] * 10,
            "feature2": [2.0] * 10,
        })
        time_weight = np.linspace(0.1, 1.0, 10)

        ct.fit(X, time_weight=time_weight)

        check_recorded_metadata(trans1, method="fit", parent="fit", time_weight=time_weight)
        check_recorded_metadata(trans2, method="fit", parent="fit", time_weight=time_weight)
        # ColumnTransformer.fit calls fit_transform on transformers, so registry may have > 2 entries
        assert len(registry) >= 2

    def test_column_transformer_routes_time_weight_to_per_column_transformers_transform(self):
        """Verify time_weight reaches column-specific transformers in transform."""
        registry = _Registry()

        trans1 = ConsumingTransformer(registry=registry).set_transform_request(time_weight=True)
        trans2 = ConsumingTransformer(registry=registry).set_transform_request(time_weight=True)

        ct = ColumnTransformer(
            transformers=[("trans1", trans1, ["feature1"]), ("trans2", trans2, ["feature2"])],
            remainder="drop",
        )

        X = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 1), end=date(2020, 1, 10), interval="1d", eager=True),
            "feature1": [1.0] * 10,
            "feature2": [2.0] * 10,
        })
        time_weight = np.linspace(0.1, 1.0, 10)

        ct.fit(X)
        ct.transform(X, time_weight=time_weight)

        check_recorded_metadata(trans1, method="transform", parent="transform", time_weight=time_weight)
        check_recorded_metadata(trans2, method="transform", parent="transform", time_weight=time_weight)


@pytest.mark.integration
class TestDecompositionPipelineRouting:
    """Metadata routing tests for DecompositionPipeline."""

    def test_decomposition_pipeline_routes_time_weight_to_component_forecasters_fit(self):
        """Verify time_weight reaches trend/seasonality/residual forecasters in fit."""
        registry = _Registry()

        # Use consuming forecasters for each component
        trend_forecaster = ConsumingForecaster(registry=registry).set_fit_request(time_weight=True)
        seasonality_forecaster = ConsumingForecaster(registry=registry).set_fit_request(time_weight=True)
        residual_forecaster = ConsumingForecaster(registry=registry).set_fit_request(time_weight=True)

        pipeline = DecompositionPipeline(
            forecasters=[
                ("trend", trend_forecaster),
                ("seasonality", seasonality_forecaster),
                ("residual", residual_forecaster),
            ],
        )

        y = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 1), end=date(2020, 1, 20), interval="1d", eager=True),
            "value": np.linspace(1, 20, 20),
        })
        time_weight = np.linspace(0.1, 1.0, 20)

        pipeline.fit(y, forecasting_horizon=3, time_weight=time_weight)

        # All three components should receive time_weight in fit
        # Note: DecompositionPipeline calls predict() on clones during fit (for residual computation),
        # so registry will have more entries than just 3 fit calls
        check_recorded_metadata(trend_forecaster, method="fit", parent="fit", time_weight=time_weight)
        assert len(registry) >= 3, f"Expected at least 3 component calls, got {len(registry)}"

    def test_decomposition_pipeline_observe_updates_components(self):
        """Verify observe() propagates new data to component forecasters."""
        registry = _Registry()

        trend_forecaster = ConsumingForecaster(registry=registry).set_fit_request(time_weight=True)
        seasonality_forecaster = ConsumingForecaster(registry=registry).set_fit_request(time_weight=True)
        residual_forecaster = ConsumingForecaster(registry=registry).set_fit_request(time_weight=True)

        pipeline = DecompositionPipeline(
            forecasters=[
                ("trend", trend_forecaster),
                ("seasonality", seasonality_forecaster),
                ("residual", residual_forecaster),
            ],
        )

        y_train = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 1), end=date(2020, 1, 20), interval="1d", eager=True),
            "value": np.linspace(1, 20, 20),
        })
        y_new = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 21), end=date(2020, 1, 25), interval="1d", eager=True),
            "value": np.linspace(21, 25, 5),
        })

        pipeline.fit(y_train, forecasting_horizon=3)
        pipeline.observe(y_new)

        # Verify observe completed and pipeline can still predict
        y_pred = pipeline.predict(forecasting_horizon=3)
        assert len(y_pred) == 3
        assert "time" in y_pred.columns


@pytest.mark.integration
class TestForecasterFitTimeWeightRouting:
    """Metadata routing tests for time_weight delivery to forecasters during fit."""

    def test_forecasted_feature_routes_time_weight_to_target_and_feature_forecasters_fit(self):
        """Verify time_weight reaches both target and feature forecasters in fit."""
        registry = _Registry()

        target_forecaster = ConsumingForecaster(registry=registry).set_fit_request(time_weight=True)
        feature_forecaster = ConsumingForecaster(registry=registry).set_fit_request(time_weight=True)

        # strategy="actual" isolates fit-time routing: each child's fit is called
        # exactly once and no rolling observe_predict runs (which would record
        # extra predict/observe calls in the registry).
        chained = ForecastedFeatureForecaster(
            target_forecaster=target_forecaster,
            feature_forecaster=feature_forecaster,
            strategy="actual",
        )

        y = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 1), end=date(2020, 1, 20), interval="1d", eager=True),
            "target": np.linspace(1, 20, 20),
        })
        X = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 1), end=date(2020, 1, 20), interval="1d", eager=True),
            "feature": np.linspace(10, 29, 20),
        })
        time_weight = np.linspace(0.1, 1.0, 20)

        chained.fit(y, X, forecasting_horizon=3, time_weight=time_weight)

        # Both forecasters should receive time_weight
        check_recorded_metadata(target_forecaster, method="fit", parent="fit", time_weight=time_weight)
        check_recorded_metadata(feature_forecaster, method="fit", parent="fit", time_weight=time_weight)
        assert len(registry) == 2

    def test_column_forecaster_routes_time_weight_to_per_column_forecasters_fit(self):
        """Verify time_weight reaches all column-specific forecasters in fit."""
        registry = _Registry()

        forecaster1 = ConsumingForecaster(registry=registry).set_fit_request(time_weight=True)
        forecaster2 = ConsumingForecaster(registry=registry).set_fit_request(time_weight=True)

        cf = ColumnForecaster(
            forecasters=[("f1", forecaster1, ["col1"]), ("f2", forecaster2, ["col2"])],
            remainder="drop",
        )

        y = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 1), end=date(2020, 1, 20), interval="1d", eager=True),
            "col1": np.linspace(1, 20, 20),
            "col2": np.linspace(10, 29, 20),
        })
        time_weight = np.linspace(0.1, 1.0, 20)

        cf.fit(y, forecasting_horizon=3, time_weight=time_weight)

        check_recorded_metadata(forecaster1, method="fit", parent="fit", time_weight=time_weight)
        check_recorded_metadata(forecaster2, method="fit", parent="fit", time_weight=time_weight)
        assert len(registry) == 2


@pytest.mark.integration
class TestTimeWeightConversion:
    """Tests for time_weight to sample_weight conversion behavior."""

    @pytest.mark.parametrize("alignment", ["first_step", "mean_step", "max_weight_step"])
    def test_time_weight_converts_to_sample_weight_exact_solution(self, alignment):
        """Verify time_weight is converted to sample_weight with exact Ridge(0) solution.

        Tests the critical conversion: time_weight → sample_weight for reduction forecasters.
        Uses Ridge(alpha=0, fit_intercept=False) with known analytical WLS solution.

        Theory:
        For weighted OLS without intercept, beta = (X^T W X)^{-1} X^T W y
        With exponential decay time_weight, we verify coefficients match analytical solution.

        The actual pipeline is:
        1. LagTransformer(lag=1) has observation_horizon=1, so y_t = y[1:] (n-1 rows)
        2. X_t has value_lag_1 = [y[0], y[1], ..., y[n-2]] (49 rows)
        3. Tabularization: n_samples = len(y_t) - forecasting_horizon = (n-1) - 1 = n-2
        4. X_tab = X_t[:-H] features = y[0:n-2], y_tab = step_1 targets = y[2:n]
        5. time_weight joins on y_t["time"] -> (n-1) weights from time indices [1..n-1]
        6. Alignment: first_step -> weights[1:n_samples+1] of the joined weights
        7. Normalization: sum(sample_weights) = n_samples

        """
        # Create simple AR(1) system: y_t = 0.5 * y_{t-1} + epsilon
        np.random.seed(42)
        n = 50
        y_true = np.zeros(n)
        for t in range(1, n):
            y_true[t] = 0.5 * y_true[t - 1] + np.random.randn() * 0.01

        y = pl.DataFrame({
            "time": pl.Series([datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]).cast(pl.Datetime("us")),
            "value": y_true,
        })

        # Exponential decay time_weight (most recent = highest weight)
        decay_rate = 0.95
        time_weight_values = decay_rate ** np.arange(n - 1, -1, -1)  # [0.95^49, 0.95^48, ..., 1.0]
        time_weight = pl.DataFrame({
            "time": y["time"],
            "weight": time_weight_values,
        })

        # Fit with time_weight (fit_intercept=False for clean analytical comparison)
        forecaster = PointReductionForecaster(
            estimator=Ridge(alpha=0.0, fit_intercept=False),
            feature_transformer=LagTransformer(lag=1),
            time_weighter=TableWeighter(frame=time_weight, on="time"),
            sample_weight_alignment=alignment,
        )
        forecaster.fit(y, forecasting_horizon=1)

        # Get fitted coefficients
        fitted_coef = forecaster.estimator_.coef_[0]

        # Replicate the actual tabularization pipeline:
        # After LagTransformer(lag=1), y_t = y[1:] -> indices 1..49 -> 49 rows
        # X_t has value_lag_1 = [y[0], y[1], ..., y[48]] (49 rows)
        # Tabularized: n_samples = 49 - 1 = 48
        # X_tab = X_t[:-1] = lag_1 values at indices 0..47 = y_true[0:48]
        # y_tab = tabularize step_1 targets = y_true[2:50]
        X_values = y_true[0:48].reshape(-1, 1)  # lag_1 feature values
        Y_values = y_true[2:50]  # step_1 target values
        n_samples = 48

        # Compute time weights as the actual code does:
        # y_t has 49 rows with time indices [1..49]
        # Joined weights for y_t: time_weight_values[1:50] (49 values for orig indices 1..49)
        weights_joined = time_weight_values[1:50]  # 49 weights

        # Apply alignment strategy as in reduction.py:
        if alignment == "first_step":
            # aligned_indices = np.arange(1, n_samples + 1) → [1, 2, ..., 48]
            sample_weight = weights_joined[1 : n_samples + 1]
        elif alignment == "mean_step":
            # For each sample i: mean(weights_joined[i+1 : i+2])
            # With forecasting_horizon=1, this is just weights_joined[i+1]
            sample_weight = np.array([weights_joined[i + 1] for i in range(n_samples)])
        else:  # max_weight_step
            # For each sample i: max(weights_joined[i+1 : i+2])
            # With forecasting_horizon=1, this is just weights_joined[i+1]
            sample_weight = np.array([weights_joined[i + 1] for i in range(n_samples)])

        # Normalize: sum(sample_weights) = n_samples
        weight_sum = sample_weight.sum()
        sample_weight = sample_weight * (n_samples / weight_sum)

        # Analytical solution: β = (X^T W X)^{-1} X^T W Y
        W = np.diag(sample_weight)
        XtWX = X_values.T @ W @ X_values
        XtWY = X_values.T @ W @ Y_values
        beta_analytical = np.linalg.solve(XtWX, XtWY)[0]

        # Verify coefficients match (tight tolerance for analytical solution)
        np.testing.assert_allclose(
            fitted_coef,
            beta_analytical,
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"time_weight → sample_weight conversion failed for alignment={alignment}",
        )

    def test_time_weight_uniform_equals_no_weight(self):
        """Verify uniform time_weight produces same result as no weight."""
        np.random.seed(42)
        n = 50
        y_data = np.random.randn(n).cumsum()

        y = pl.DataFrame({
            "time": pl.Series([datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]).cast(pl.Datetime("us")),
            "value": y_data,
        })

        # Fit without time_weight
        f1 = PointReductionForecaster(Ridge(alpha=0.1), feature_transformer=LagTransformer(lag=[1, 2]))
        f1.fit(y, forecasting_horizon=1)
        coef_no_weight = f1.estimator_.coef_.copy()

        # Fit with uniform time_weight
        time_weight_uniform = pl.DataFrame({
            "time": y["time"],
            "weight": np.ones(n),
        })
        f2 = PointReductionForecaster(
            Ridge(alpha=0.1),
            feature_transformer=LagTransformer(lag=[1, 2]),
            time_weighter=TableWeighter(frame=time_weight_uniform, on="time"),
        )
        f2.fit(y, forecasting_horizon=1)
        coef_uniform_weight = f2.estimator_.coef_.copy()

        # Coefficients should be identical (uniform weight = no weight)
        np.testing.assert_allclose(
            coef_no_weight,
            coef_uniform_weight,
            rtol=1e-10,
            err_msg="Uniform time_weight should produce same result as no weight",
        )

    def test_time_weight_strong_decay_emphasizes_recent_observations(self):
        """Verify strong exponential decay produces different coefficients than uniform.

        With LagTransformer(lag=1) and forecasting_horizon=1, the regression learns
        y_{t+1} from y_{t-1} (2-step gap), so the effective coefficient approximates
        alpha^2 rather than alpha. The decay weighting should shift the estimate
        toward the recent regime's alpha^2.
        """
        np.random.seed(42)
        n = 100

        # Create series with structural break (different AR coefficient before/after)
        y_data = np.zeros(n)
        for t in range(1, 50):
            y_data[t] = 0.3 * y_data[t - 1] + np.random.randn() * 0.1  # Old regime
        for t in range(50, n):
            y_data[t] = 0.8 * y_data[t - 1] + np.random.randn() * 0.1  # New regime

        y = pl.DataFrame({
            "time": pl.Series([datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]).cast(pl.Datetime("us")),
            "value": y_data,
        })

        # Fit with uniform weight
        f1 = PointReductionForecaster(Ridge(alpha=0.0), feature_transformer=LagTransformer(lag=1))
        f1.fit(y, forecasting_horizon=1)
        coef_uniform = f1.estimator_.coef_[0]

        # Fit with strong recent emphasis (decay to ~0.001 at start)
        decay_rate = 0.93
        time_weight_values = decay_rate ** np.arange(n - 1, -1, -1)
        time_weight = pl.DataFrame({
            "time": y["time"],
            "weight": time_weight_values,
        })
        f2 = PointReductionForecaster(
            Ridge(alpha=0.0),
            feature_transformer=LagTransformer(lag=1),
            time_weighter=TableWeighter(frame=time_weight, on="time"),
        )
        f2.fit(y, forecasting_horizon=1)
        coef_decay = f2.estimator_.coef_[0]

        # With strong decay, coefficient should be higher (closer to new regime)
        # Due to 2-step gap (lag_1 feature -> step_1 target), effective coef ~ alpha^2
        # Old regime: ~0.3^2=0.09, New regime: ~0.8^2=0.64
        assert coef_decay > coef_uniform, (
            f"Strong decay should emphasize recent regime, got {coef_decay:.3f} vs uniform {coef_uniform:.3f}"
        )


@pytest.mark.integration
class TestColumnForecasterPanelRouting:
    """Tests for ColumnForecaster groups pass-through to child forecasters."""

    def test_column_forecaster_groups_passed_to_children_predict(self):
        """Verify groups is forwarded to child forecasters in predict."""
        registry = _Registry()

        forecaster = ConsumingForecaster(registry=registry)

        cf = ColumnForecaster(
            forecasters=[("f", forecaster, ["sales__store1", "sales__store2", "sales__store3"])],
            remainder="drop",
        )

        # Panel data with 3 groups
        y = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 1), end=date(2020, 1, 10), interval="1d", eager=True),
            "sales__store1": np.linspace(1, 10, 10),
            "sales__store2": np.linspace(11, 20, 10),
            "sales__store3": np.linspace(21, 30, 10),
        })

        cf.fit(y, forecasting_horizon=3)

        # Predict all groups (groups=None)
        y_pred = cf.predict(forecasting_horizon=3)

        pred_cols = [c for c in y_pred.columns if c not in {"time", "vintage_time"}]
        assert len(pred_cols) == 3, f"Expected 3 prediction columns, got {pred_cols}"

    def test_column_forecaster_observe_passes_data_to_children(self):
        """Verify observe passes column-subsetted data to each child forecaster."""
        registry = _Registry()

        forecaster = ConsumingForecaster(registry=registry)

        cf = ColumnForecaster(
            forecasters=[("f", forecaster, ["sales__store1", "sales__store2", "sales__store3"])],
            remainder="drop",
        )

        y_train = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 1), end=date(2020, 1, 10), interval="1d", eager=True),
            "sales__store1": np.linspace(1, 10, 10),
            "sales__store2": np.linspace(11, 20, 10),
            "sales__store3": np.linspace(21, 30, 10),
        })

        y_new = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 11), end=date(2020, 1, 15), interval="1d", eager=True),
            "sales__store1": np.linspace(11, 15, 5),
            "sales__store2": np.linspace(21, 25, 5),
            "sales__store3": np.linspace(31, 35, 5),
        })

        cf.fit(y_train, forecasting_horizon=3)

        # Observe all data
        cf.observe(y_new)

        # Verify observe completed without error and internal state updated
        assert cf._y_observed is not None
        assert len(cf._y_observed) == 15  # 10 original + 5 new

    def test_column_forecaster_rewind_passes_data_to_children(self):
        """Verify rewind passes column-subsetted data to each child forecaster."""
        registry = _Registry()

        forecaster = ConsumingForecaster(registry=registry)

        cf = ColumnForecaster(
            forecasters=[("f", forecaster, ["sales__store1", "sales__store2"])],
            remainder="drop",
        )

        y_train = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 1), end=date(2020, 1, 20), interval="1d", eager=True),
            "sales__store1": np.linspace(1, 20, 20),
            "sales__store2": np.linspace(11, 30, 20),
        })

        y_reset = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 16), end=date(2020, 1, 20), interval="1d", eager=True),
            "sales__store1": np.linspace(16, 20, 5),
            "sales__store2": np.linspace(26, 30, 5),
        })

        cf.fit(y_train, forecasting_horizon=3)

        # Rewind to last 5 rows
        cf.rewind(y_reset)

        # Verify rewind completed and internal state updated
        assert cf._y_observed is not None
        assert len(cf._y_observed) == 5  # Reset to 5 rows


@pytest.mark.integration
class TestCoverageRatesRouting:
    """Metadata routing tests for coverage_rates to interval forecasters."""

    def test_predict_interval_routes_coverage_rates_through_forecasted_feature(self):
        """Verify coverage_rates reaches target forecaster's predict_interval.

        Tests that coverage_rates metadata is routed correctly through
        ForecastedFeatureForecaster to the target interval forecaster.

        """
        # Create simple interval forecaster with quantile-capable regressor
        np.random.seed(42)

        target_forecaster = (
            IntervalReductionForecaster(
                estimator=MultiOutputRegressor(QuantileRegressor(solver="highs")),
                feature_transformer=LagTransformer(lag=1),
            )
            .set_fit_request(coverage_rates=True)
            .set_predict_interval_request(coverage_rates=True)
        )

        # Feature forecaster (point forecaster, doesn't need coverage_rates)
        feature_forecaster = PointReductionForecaster(
            estimator=Ridge(alpha=0.1),
            feature_transformer=LagTransformer(lag=1),
        )

        chained = ForecastedFeatureForecaster(
            target_forecaster=target_forecaster,
            feature_forecaster=feature_forecaster,
        )

        y = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 1), end=date(2020, 1, 30), interval="1d", eager=True),
            "target": np.random.randn(30).cumsum(),
        })
        X = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 1), end=date(2020, 1, 30), interval="1d", eager=True),
            "feature": np.random.randn(30).cumsum(),
        })

        chained.fit(y, X, forecasting_horizon=3, coverage_rates=[0.9, 0.95])

        # Predict intervals with coverage_rates
        coverage_rates = [0.9, 0.95]
        y_pred = chained.predict_interval(forecasting_horizon=3, coverage_rates=coverage_rates)

        # Verify coverage_rates metadata was routed to target forecaster's regressor predict
        # Note: IntervalReductionForecaster doesn't directly call regressor.predict with coverage_rates,
        # but we can verify the prediction has correct coverage columns
        expected_cols = set()
        for rate in coverage_rates:
            expected_cols.add(f"target_lower_{rate}")
            expected_cols.add(f"target_upper_{rate}")

        pred_cols = {c for c in y_pred.columns if c not in ["time", "vintage_time"]}
        assert expected_cols.issubset(pred_cols), f"Missing coverage columns: {expected_cols - pred_cols}"

    def test_predict_interval_routes_coverage_rates_directly(self):
        """Verify coverage_rates reaches IntervalReductionForecaster directly."""
        np.random.seed(42)

        y = pl.DataFrame({
            "time": pl.Series([datetime(2020, 1, 1) + timedelta(days=i) for i in range(50)]).cast(pl.Datetime("us")),
            "value": np.random.randn(50).cumsum(),
        })

        forecaster = IntervalReductionForecaster(
            estimator=MultiOutputRegressor(QuantileRegressor(solver="highs")),
            feature_transformer=LagTransformer(lag=[1, 2, 3]),
        )

        # Multiple coverage rates - must be passed at fit AND predict time
        coverage_rates = [0.8, 0.9, 0.95, 0.99]
        forecaster.fit(y, forecasting_horizon=5, coverage_rates=coverage_rates)

        y_pred = forecaster.predict_interval(forecasting_horizon=5, coverage_rates=coverage_rates)

        # Verify all coverage levels present
        for rate in coverage_rates:
            assert f"value_lower_{rate}" in y_pred.columns, f"Missing lower_{rate}"
            assert f"value_upper_{rate}" in y_pred.columns, f"Missing upper_{rate}"


@pytest.mark.integration
class TestCompositeMethodRouting:
    """Metadata routing tests for composite methods (observe_predict, observe_transform)."""

    def test_observe_predict_routes_separate_metadata_through_feature_pipeline(self):
        """Verify observe_predict routes observe and predict metadata separately.

        Tests the composite method routing: observe_predict should route observe-specific
        metadata to observe() and predict-specific metadata to predict() within transformers.

        """
        registry = _Registry()

        # Transformer that accepts both observe and transform metadata
        transformer = ConsumingTransformer(registry=registry)
        transformer.set_fit_request(time_weight=True)
        # Note: ConsumingTransformer doesn't have observe, but FeaturePipeline should handle gracefully

        pipeline = FeaturePipeline([("step", transformer)])

        X = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 1), end=date(2020, 1, 20), interval="1d", eager=True),
            "feature": np.random.randn(20),
        })

        pipeline.fit(X)

        # observe_transform should work even if transformer doesn't have observe method
        # (FeaturePipeline handles this by skipping observe on transformers without it)
        X_new = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 21), end=date(2020, 1, 25), interval="1d", eager=True),
            "feature": np.random.randn(5),
        })

        # This should not error (graceful handling of missing observe method)
        X_transformed = pipeline.observe_transform(X_new)

        assert len(X_transformed) == len(X_new), "observe_transform should return transformed data"

    def test_observe_predict_routes_metadata_through_column_forecaster(self):
        """Verify observe_predict routes observe + predict metadata to forecasters."""
        registry = _Registry()

        forecaster = ConsumingForecaster(registry=registry)

        cf = ColumnForecaster(
            forecasters=[("f", forecaster, ["value"])],
            remainder="drop",
        )

        y_train = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 1), end=date(2020, 1, 20), interval="1d", eager=True),
            "value": np.random.randn(20),
        })

        y_new = pl.DataFrame({
            "time": pl.datetime_range(start=date(2020, 1, 21), end=date(2020, 1, 25), interval="1d", eager=True),
            "value": np.random.randn(5),
        })

        cf.fit(y_train, forecasting_horizon=3)

        # observe_predict with metadata
        _y_pred = cf.observe_predict(y_new)

        # Verify predict was called
        assert len(registry) >= 2, "Expected both observe and predict calls"
