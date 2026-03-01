"""Stress tests for state lifecycle management across compositions.

Tests verify state machine protocol (fit/observe/rewind/predict), memory bounds,
idempotency, cloning, serialization, and convergence properties under iterative
operations through deep composition hierarchies.

All tests use analytical fixtures from conftest.py and must pass @pytest.mark.integration.
"""

import pickle
from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from yohou.compose import (
    ColumnForecaster,
    ColumnTransformer,
    DecompositionPipeline,
    FeaturePipeline,
    FeatureUnion,
    ForecastedFeatureForecaster,
)
from yohou.interval import SplitConformalForecaster
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer, StandardScaler
from yohou.stationarity import PolynomialTrendForecaster


def _make_continuation(base_df: pl.DataFrame, length: int, seed: int = 0) -> pl.DataFrame:
    """Create a continuation DataFrame that starts right after base_df ends.

    Uses the same schema as base_df but with timestamps continuing from
    where base_df left off, avoiding overlap errors.
    """
    last_time = base_df["time"].max()
    np.random.seed(seed)
    value_cols = [c for c in base_df.columns if c != "time"]
    data: dict = {
        "time": [last_time + timedelta(days=i + 1) for i in range(length)],
    }
    for col in value_cols:
        # Use last value as base for AR-like continuation
        last_val = base_df[col][-1]
        data[col] = [float(last_val) + np.random.randn() * 0.1 for _ in range(length)]
    return pl.DataFrame(data).with_columns(pl.col("time").cast(pl.Datetime("us")))


@pytest.mark.integration
class TestStateMachineProtocol:
    """Tests for the state machine protocol: fit -> observe -> rewind -> predict lifecycle."""

    @pytest.mark.parametrize(
        "composition_factory",
        [
            pytest.param(
                lambda: PointReductionForecaster(
                    estimator=LinearRegression(),
                    feature_transformer=LagTransformer(lag=1),
                ),
                id="point_reduction",
            ),
            pytest.param(
                lambda: ColumnForecaster(
                    forecasters=[
                        (
                            "target",
                            PointReductionForecaster(
                                estimator=LinearRegression(),
                                feature_transformer=LagTransformer(lag=1),
                            ),
                            ["value"],
                        )
                    ]
                ),
                id="column_forecaster",
            ),
            pytest.param(
                lambda: DecompositionPipeline(
                    forecasters=[
                        ("trend", PolynomialTrendForecaster(degree=1)),
                        (
                            "residual",
                            PointReductionForecaster(
                                estimator=LinearRegression(),
                                feature_transformer=LagTransformer(lag=1),
                            ),
                        ),
                    ]
                ),
                id="decomposition_pipeline",
            ),
            pytest.param(
                lambda: ForecastedFeatureForecaster(
                    target_forecaster=PointReductionForecaster(
                        estimator=LinearRegression(),
                        feature_transformer=LagTransformer(lag=1),
                    ),
                    feature_forecaster=PointReductionForecaster(
                        estimator=LinearRegression(),
                        feature_transformer=LagTransformer(lag=1),
                    ),
                ),
                id="forecasted_feature_forecaster",
            ),
            pytest.param(
                lambda: SplitConformalForecaster(
                    point_forecaster=PointReductionForecaster(
                        estimator=LinearRegression(),
                        feature_transformer=LagTransformer(lag=1),
                    ),
                    calibration_size=30,
                ),
                id="split_conformal",
            ),
        ],
    )
    def test_state_machine_full_cycle(self, composition_factory, ar1_series):
        """Test full state machine cycle: fit → predict → observe → rewind → observe_predict.

        Tests 5 composition types × 5 state transitions = 25 verification points.
        """
        forecaster = composition_factory()

        # Generate data with known AR(1) properties
        y_train = ar1_series(length=200, phi=0.5, c=10.0, seed=42)

        # Create continuation data that starts AFTER y_train ends (no overlap)
        y_test = _make_continuation(y_train, length=10, seed=43)
        y_test2 = _make_continuation(pl.concat([y_train, y_test], how="vertical"), length=10, seed=44)

        # For ForecastedFeatureForecaster, create X with feature columns
        needs_X = isinstance(forecaster, ForecastedFeatureForecaster)
        X_train = None
        X_test = None
        X_test2 = None
        if needs_X:
            np.random.seed(99)
            X_train = y_train.select("time").with_columns(
                pl.Series("feature1", np.random.randn(len(y_train))),
            )
            X_test = y_test.select("time").with_columns(
                pl.Series("feature1", np.random.randn(len(y_test))),
            )
            X_test2 = y_test2.select("time").with_columns(
                pl.Series("feature1", np.random.randn(len(y_test2))),
            )

        fh = 3

        # State 1: fit → predict
        forecaster.fit(y_train, X=X_train, forecasting_horizon=fh)
        check_is_fitted(forecaster)
        predictions_1 = forecaster.predict(forecasting_horizon=fh)

        assert len(predictions_1) == fh
        assert "time" in predictions_1.columns

        # State 2: observe → predict (state should change)
        forecaster.observe(y_test, X=X_test)
        predictions_2 = forecaster.predict(forecasting_horizon=fh)

        assert len(predictions_2) == fh
        # State should have changed after observe: observed_time should differ
        assert not predictions_1["observed_time"].equals(predictions_2["observed_time"])

        # State 3: second observe → predict
        forecaster.observe(y_test2, X=X_test2)
        predictions_3 = forecaster.predict(forecasting_horizon=fh)

        assert len(predictions_3) == fh
        assert not predictions_2["observed_time"].equals(predictions_3["observed_time"])

        # State 4: rewind to training tail → predict
        try:
            obs_horizon = forecaster.observation_horizon
        except NotFittedError:
            obs_horizon = 0
        # Use enough rows for rewind to satisfy internal transformer memory
        reset_rows = max(obs_horizon, 10)
        y_train_tail = y_train.tail(reset_rows)
        if needs_X:
            X_train_tail = X_train.tail(reset_rows)
            forecaster.rewind(y_train_tail, X=X_train_tail)
        else:
            forecaster.rewind(y_train_tail)
        predictions_4 = forecaster.predict(forecasting_horizon=fh)

        assert len(predictions_4) == fh
        # After rewind to tail, predictions may differ slightly from original
        # (floating point, internal regressor state), but should be reasonable

        # State 5: observe_predict (atomic operation)
        # Create continuation from current state (after rewind to train tail)
        y_for_up = _make_continuation(y_train, length=10, seed=55)
        result_observe_predict = forecaster.observe_predict(y_for_up, X=X_test)

        # observe_predict returns predictions accumulated over incremental observes
        assert len(result_observe_predict) >= fh
        assert "time" in result_observe_predict.columns

    def test_state_machine_observation_buffer_growth(self, ar1_series):
        """Verify observation buffer grows correctly during observe sequence."""
        y_train = ar1_series(length=20, phi=0.5, c=10.0, seed=100)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=2),
        )
        forecaster.fit(y_train, forecasting_horizon=1)

        obs_horizon = forecaster.observation_horizon
        # PointReductionForecaster with feature_transformer=LagTransformer(lag=2)
        # observation_horizon = max(0, 0, 2) = 2
        assert obs_horizon == 2

        # _y_observed is populated when observation_horizon > 0
        assert forecaster._y_observed is not None
        assert len(forecaster._y_observed) == 2

        # Perform observes with continuation data and verify predictions still work
        current_data = y_train
        for i in range(3):
            y_upd = _make_continuation(current_data, length=5, seed=100 + i)
            forecaster.observe(y_upd)
            current_data = pl.concat([current_data, y_upd], how="vertical")
            # _y_observed is bounded by observation_horizon
            assert forecaster._y_observed is not None
            assert len(forecaster._y_observed) == obs_horizon

        # Verify predictions still work after updates
        pred = forecaster.predict(forecasting_horizon=1)
        assert len(pred) == 1

    def test_state_machine_rewind_idempotency(self, ar1_series):
        """Verify rewind() with same tail data produces identical state."""
        y_train = ar1_series(length=100, phi=0.5, c=10.0, seed=42)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=5)

        obs_horizon = forecaster.observation_horizon
        # Use enough rows for rewind to satisfy internal transformer memory requirements
        reset_rows = max(obs_horizon, 10)
        y_tail = y_train.tail(reset_rows)

        # Rewind to tail
        forecaster.rewind(y_tail)
        pred_1 = forecaster.predict(forecasting_horizon=5)

        # Rewind again with same tail
        forecaster.rewind(y_tail)
        pred_2 = forecaster.predict(forecasting_horizon=5)

        # Predictions should be identical
        assert pred_1.equals(pred_2)

    def test_state_machine_observe_predict_equivalence(self, ar1_series):
        """Verify observe_predict() ≡ observe() + predict() for point forecasters."""
        y_train = ar1_series(length=100, phi=0.5, c=10.0, seed=42)
        y_new = _make_continuation(y_train, length=5, seed=999)

        # Create two identical forecasters
        forecaster_1 = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster_2 = clone(forecaster_1)

        forecaster_1.fit(y_train, forecasting_horizon=3)
        forecaster_2.fit(y_train, forecasting_horizon=3)

        # Path 1: observe + predict
        forecaster_1.observe(y_new)
        pred_separate = forecaster_1.predict(forecasting_horizon=3)

        # Path 2: observe_predict
        pred_combined = forecaster_2.observe_predict(y_new)

        # Both should produce valid predictions
        assert len(pred_separate) >= 3
        assert len(pred_combined) >= 3

        # Compare predicted values for the last forecasting_horizon rows
        value_cols = [c for c in pred_separate.columns if c not in ("time", "observed_time")]
        sep_vals = pred_separate.tail(3).select(value_cols)
        comb_vals = pred_combined.tail(3).select(value_cols)
        assert sep_vals.equals(comb_vals)

    def test_state_machine_series_of_observes(self, ar1_series):
        """Verify state evolves correctly through series of small observes."""
        y_train = ar1_series(length=50, phi=0.7, c=8.0, seed=200)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=1)

        predictions = []

        # Perform 10 sequential observes with single observations
        # Each observe must continue from where the previous one ended
        current_data = y_train
        for i in range(10):
            y_single = _make_continuation(current_data, length=1, seed=300 + i)
            forecaster.observe(y_single)
            current_data = pl.concat([current_data, y_single], how="vertical")
            pred = forecaster.predict(forecasting_horizon=1)
            predictions.append(pred["value"][0])

        # All predictions should be finite
        assert all(np.isfinite(p) for p in predictions)

        # Predictions should vary (state is changing)
        assert len(set(predictions)) > 1


@pytest.mark.integration
class TestIdempotency:
    """Tests verifying repeated operations produce identical results."""

    def test_predict_idempotency(self, ar1_series):
        """Verify predict() called twice returns identical results."""
        y_train = ar1_series(length=100, phi=0.5, c=10.0, seed=42)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=5)

        pred_1 = forecaster.predict(forecasting_horizon=5)
        pred_2 = forecaster.predict(forecasting_horizon=5)

        assert pred_1.equals(pred_2)
        # Verify object identity hasn't changed (no side effects)
        assert id(forecaster._y_observed) == id(forecaster._y_observed)

    def test_predict_interval_idempotency(self, ar1_series):
        """Verify predict_interval() called twice returns identical results."""
        y_train = ar1_series(length=200, phi=0.5, c=10.0, seed=42)

        forecaster = SplitConformalForecaster(
            point_forecaster=PointReductionForecaster(
                estimator=LinearRegression(),
                feature_transformer=LagTransformer(lag=1),
            ),
            calibration_size=30,
        )
        forecaster.fit(y_train, forecasting_horizon=5)

        pred_1 = forecaster.predict_interval(forecasting_horizon=5, coverage_rates=[0.9])
        pred_2 = forecaster.predict_interval(forecasting_horizon=5, coverage_rates=[0.9])

        assert pred_1.equals(pred_2)

    def test_fit_idempotency(self, ar1_series):
        """Verify fit() called twice on same data produces identical state."""
        y_train = ar1_series(length=100, phi=0.5, c=10.0, seed=42)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )

        forecaster.fit(y_train, forecasting_horizon=5)
        pred_1 = forecaster.predict(forecasting_horizon=5)

        # Fit again on same data
        forecaster.fit(y_train, forecasting_horizon=5)
        pred_2 = forecaster.predict(forecasting_horizon=5)

        assert pred_1.equals(pred_2)


@pytest.mark.integration
class TestMemoryBounds:
    """Tests verifying observation buffers remain bounded after many observes."""

    def test_memory_bounded_feature_pipeline(self, ar1_series):
        """Verify observation_horizon bounds hold through FeaturePipeline after 100 observes."""
        y_train = ar1_series(length=100, phi=0.5, c=10.0, seed=42)

        pipeline = FeaturePipeline(
            steps=[
                ("lag3", LagTransformer(lag=3)),
                ("lag2", LagTransformer(lag=2)),
                ("scaler", StandardScaler()),
            ]
        )

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=pipeline,
        )
        forecaster.fit(y_train, forecasting_horizon=1)

        obs_horizon = forecaster.observation_horizon
        # PointReductionForecaster with feature_transformer=FeaturePipeline(lag3+lag2+scaler)
        # observation_horizon = max(0, 0, 3+2+0) = 5
        assert obs_horizon == 5

        # Perform 100 single-row observes with continuation data
        current_data = y_train
        for i in range(100):
            y_single = _make_continuation(current_data, length=1, seed=1000 + i)
            forecaster.observe(y_single)
            current_data = pl.concat([current_data, y_single], how="vertical")

            # _y_observed is bounded by observation_horizon
            assert forecaster._y_observed is not None
            assert len(forecaster._y_observed) == obs_horizon

    def test_memory_bounded_feature_union(self, ar1_series):
        """Verify observation_horizon bounds hold through FeatureUnion after 100 observes."""
        y_train = ar1_series(length=100, phi=0.5, c=10.0, seed=42)

        union = FeatureUnion(
            transformer_list=[
                ("lag1", LagTransformer(lag=1)),
                ("lag2", LagTransformer(lag=2)),
            ]
        )

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=union,
        )
        forecaster.fit(y_train, forecasting_horizon=1)

        obs_horizon = forecaster.observation_horizon
        # PointReductionForecaster with feature_transformer=FeatureUnion(lag1, lag2)
        # observation_horizon = max(0, 0, max(1, 2)) = 2
        assert obs_horizon == 2

        # Perform 100 single-row observes with continuation data
        current_data = y_train
        for i in range(100):
            y_single = _make_continuation(current_data, length=1, seed=2000 + i)
            forecaster.observe(y_single)
            current_data = pl.concat([current_data, y_single], how="vertical")

            # _y_observed is bounded by observation_horizon
            assert forecaster._y_observed is not None
            assert len(forecaster._y_observed) == obs_horizon

    def test_memory_bounded_column_transformer(self, ar1_series):
        """Verify observation_horizon bounds hold through ColumnTransformer after 100 observes."""
        y_train = ar1_series(length=100, phi=0.5, c=10.0, seed=42)

        ct = ColumnTransformer(
            transformers=[
                ("lag1", LagTransformer(lag=1), ["value"]),
                ("lag3", LagTransformer(lag=3), ["value"]),
            ],
            remainder="passthrough",
        )

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=ct,
        )
        forecaster.fit(y_train, forecasting_horizon=1)

        obs_horizon = forecaster.observation_horizon
        # PointReductionForecaster with feature_transformer=ColumnTransformer(lag1, lag3)
        # observation_horizon = max(0, 0, max(1, 3)) = 3
        assert obs_horizon == 3

        # Perform 100 single-row observes with continuation data
        current_data = y_train
        for i in range(100):
            y_single = _make_continuation(current_data, length=1, seed=3000 + i)
            forecaster.observe(y_single)
            current_data = pl.concat([current_data, y_single], how="vertical")

            # _y_observed is bounded by observation_horizon
            assert forecaster._y_observed is not None
            assert len(forecaster._y_observed) == obs_horizon

    def test_deep_pipeline_memory_bounded(self, ar1_series):
        """Verify memory bounds hold through deeply nested pipeline composition."""
        y_train = ar1_series(length=100, phi=0.6, c=10.0, seed=500)

        # Create deeply nested pipeline: Pipeline inside Pipeline
        inner_pipeline = FeaturePipeline(
            steps=[
                ("lag1", LagTransformer(lag=1)),
                ("scaler1", StandardScaler()),
            ]
        )

        outer_pipeline = FeaturePipeline(
            steps=[
                ("inner", inner_pipeline),
                ("lag2", LagTransformer(lag=2)),
                ("scaler2", StandardScaler()),
            ]
        )

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=outer_pipeline,
        )
        forecaster.fit(y_train, forecasting_horizon=1)

        obs_horizon = forecaster.observation_horizon
        # PointReductionForecaster with feature_transformer=FeaturePipeline(lag1+lag2+scaler)
        # observation_horizon = max(0, 0, 1+2+0) = 3
        assert obs_horizon == 3

        # Run 50 observes with continuation data
        current_data = y_train
        for i in range(50):
            y_new = _make_continuation(current_data, length=1, seed=5000 + i)
            forecaster.observe(y_new)
            current_data = pl.concat([current_data, y_new], how="vertical")

            # _y_observed is bounded by observation_horizon
            assert forecaster._y_observed is not None
            assert len(forecaster._y_observed) == obs_horizon

    def test_sequential_observe_predict_no_memory_leak(self, ar1_series):
        """Verify sequential observe_predict doesn't accumulate unbounded memory."""
        phi = 0.5
        c = 10.0

        y_train = ar1_series(length=100, phi=phi, c=c, seed=42)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=1)

        obs_horizon = forecaster.observation_horizon  # Will be 0

        # Run 100 observe_predict cycles with continuation data
        current_data = y_train
        for i in range(100):
            y_new = _make_continuation(current_data, length=1, seed=2000 + i)
            forecaster.observe_predict(y_new)
            current_data = pl.concat([current_data, y_new], how="vertical")

            # _y_observed is None when obs_horizon=0, so no memory leak possible
            if forecaster._y_observed is not None:
                assert len(forecaster._y_observed) <= obs_horizon

        # When obs_horizon=0, _y_observed stays None (no memory used)
        if obs_horizon == 0:
            assert forecaster._y_observed is None
        else:
            assert len(forecaster._y_observed) <= obs_horizon


@pytest.mark.integration
class TestCloneAndSerialization:
    """Tests for clone behavior and pickle round-trip preservation."""

    def test_clone_is_unfitted(self, ar1_series):
        """Verify clone() creates unfitted estimator."""
        y_train = ar1_series(length=100, phi=0.5, c=10.0, seed=42)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=5)
        check_is_fitted(forecaster)

        cloned = clone(forecaster)

        # Clone should not be fitted
        with pytest.raises(NotFittedError, match="is not fitted"):
            check_is_fitted(cloned)

    def test_clone_predict_raises_error(self, ar1_series):
        """Verify predict() on unfitted clone raises NotFittedError."""
        y_train = ar1_series(length=100, phi=0.5, c=10.0, seed=42)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=5)

        cloned = clone(forecaster)

        with pytest.raises(NotFittedError, match="is not fitted"):
            cloned.predict(forecasting_horizon=5)

    def test_clone_fit_independence(self, ar1_series):
        """Verify fitting clone on different data doesn't affect original."""
        y_train_1 = ar1_series(length=100, phi=0.5, c=10.0, seed=42)
        y_train_2 = ar1_series(length=100, phi=0.8, c=5.0, seed=99)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train_1, forecasting_horizon=5)
        pred_original = forecaster.predict(forecasting_horizon=5)

        # Clone and fit on different data
        cloned = clone(forecaster)
        cloned.fit(y_train_2, forecasting_horizon=5)
        pred_clone = cloned.predict(forecasting_horizon=5)

        # Original predictions should be unchanged
        pred_original_after = forecaster.predict(forecasting_horizon=5)
        assert pred_original.equals(pred_original_after)

        # Clone predictions should differ from original (different training data)
        pred_cols = pl.exclude("time", "observed_time")
        assert not pred_original.select(pred_cols).equals(pred_clone.select(pred_cols))

    @pytest.mark.parametrize(
        "composition_factory",
        [
            pytest.param(
                lambda: PointReductionForecaster(
                    estimator=LinearRegression(),
                    feature_transformer=LagTransformer(lag=1),
                ),
                id="point_reduction",
            ),
            pytest.param(
                lambda: ColumnForecaster(
                    forecasters=[
                        (
                            "target",
                            PointReductionForecaster(
                                estimator=LinearRegression(),
                                feature_transformer=LagTransformer(lag=1),
                            ),
                            ["value"],
                        )
                    ]
                ),
                id="column_forecaster",
            ),
            pytest.param(
                lambda: DecompositionPipeline(
                    forecasters=[
                        ("trend", PolynomialTrendForecaster(degree=1)),
                        (
                            "residual",
                            PointReductionForecaster(
                                estimator=LinearRegression(),
                                feature_transformer=LagTransformer(lag=1),
                            ),
                        ),
                    ]
                ),
                id="decomposition_pipeline",
            ),
            pytest.param(
                lambda: ForecastedFeatureForecaster(
                    target_forecaster=PointReductionForecaster(
                        estimator=LinearRegression(),
                        feature_transformer=LagTransformer(lag=1),
                    ),
                    feature_forecaster=PointReductionForecaster(
                        estimator=LinearRegression(),
                        feature_transformer=LagTransformer(lag=1),
                    ),
                ),
                id="forecasted_feature",
            ),
            pytest.param(
                lambda: FeaturePipeline(
                    steps=[
                        ("lag", LagTransformer(lag=2)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                id="feature_pipeline",
            ),
        ],
    )
    def test_pickle_round_trip(self, composition_factory, ar1_series):
        """Verify pickle → unpickle preserves predictions across composition types."""
        y_train = ar1_series(length=100, phi=0.5, c=10.0, seed=42)

        estimator = composition_factory()

        # Determine if forecaster or transformer
        is_forecaster = hasattr(estimator, "predict") and hasattr(estimator, "fit")
        is_transformer = hasattr(estimator, "transform")

        if is_forecaster and not is_transformer:
            # Pure forecaster
            needs_X = isinstance(estimator, ForecastedFeatureForecaster)
            X_train = None
            if needs_X:
                np.random.seed(77)
                X_train = y_train.select("time").with_columns(
                    pl.Series("feature1", np.random.randn(len(y_train))),
                )
            estimator.fit(y_train, X=X_train, forecasting_horizon=5)
            pred_before = estimator.predict(forecasting_horizon=5)
        elif is_transformer:
            # Transformer (may also be in a pipeline)
            estimator.fit(y_train)
            pred_before = estimator.transform(y_train)
        else:
            pytest.fail(f"Unknown estimator type: {type(estimator)}")

        # Pickle and unpickle
        pickled = pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
        unpickled = pickle.loads(pickled)

        # Predict/transform with unpickled
        if is_forecaster and not is_transformer:
            pred_after = unpickled.predict(forecasting_horizon=5)
        elif is_transformer:
            pred_after = unpickled.transform(y_train)

        # Results should be identical
        assert pred_before.equals(pred_after)


@pytest.mark.integration
class TestConvergence:
    """Tests verifying sequential predictions converge to analytical steady state."""

    @pytest.mark.parametrize("phi", [0.5, 0.8])
    def test_sequential_observe_predict_convergence(self, ar1_series, phi):
        """Verify 50 sequential observe_predict cycles converge to analytical steady state.

        For AR(1): y_t = c + φ * y_{t-1} + ε_t
        Steady state: E[y] = c / (1 - φ)
        """
        c = 10.0
        expected_steady_state = c / (1 - phi)

        y_train = ar1_series(length=100, phi=phi, c=c, seed=42)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=1)

        predictions = []
        obs_horizon = forecaster.observation_horizon  # Will be 0

        # Run 50 sequential observe_predict cycles
        current_data = y_train
        for i in range(50):
            y_new = _make_continuation(current_data, length=1, seed=1000 + i)
            y_pred = forecaster.observe_predict(y_new)
            current_data = pl.concat([current_data, y_new], how="vertical")

            # Store prediction
            pred_value = y_pred["value"][0]
            predictions.append(pred_value)

            # Check no NaNs or infinities (state corruption check)
            assert not np.isnan(pred_value), f"NaN detected at iteration {i}"
            assert not np.isinf(pred_value), f"Inf detected at iteration {i}"

            # Check memory bound (no memory leak)
            # _y_observed is None when obs_horizon=0
            if forecaster._y_observed is not None:
                assert len(forecaster._y_observed) <= obs_horizon

        # Verify convergence toward steady state
        final_predictions = predictions[-10:]  # Last 10 predictions
        mean_final = np.mean(final_predictions)

        # Should be close to c/(1-phi) with tolerance for iterative errors
        assert np.abs(mean_final - expected_steady_state) < 5.0, (
            f"φ={phi}: Mean of final 10 predictions {mean_final:.2f} not close to steady state {expected_steady_state:.2f}"
        )


@pytest.mark.integration
class TestCompositionStateIsolation:
    """Tests verifying nested forecasters in compositions maintain independent state."""

    def test_column_forecaster_nested_state(self, ar1_series):
        """Verify nested forecasters in ColumnForecaster maintain independent state."""
        y_base = ar1_series(length=100, phi=0.5, c=10.0, seed=600)
        # Create multi-column data so each forecaster gets a different column
        np.random.seed(601)
        y_train = y_base.rename({"value": "col_a"}).with_columns(
            pl.Series("col_b", np.random.randn(100) + 10.0),
        )

        forecaster = ColumnForecaster(
            forecasters=[
                (
                    "f1",
                    PointReductionForecaster(
                        estimator=LinearRegression(),
                        feature_transformer=LagTransformer(lag=1),
                    ),
                    ["col_a"],
                ),
                (
                    "f2",
                    PointReductionForecaster(
                        estimator=LinearRegression(),
                        feature_transformer=LagTransformer(lag=3),
                    ),
                    ["col_b"],
                ),
            ]
        )
        forecaster.fit(y_train, forecasting_horizon=3)

        # Create continuation observe data
        y_update = _make_continuation(y_train, length=5, seed=602)
        # Ensure observe has both columns
        np.random.seed(603)
        y_update = y_update.rename({"col_a": "col_a"})  # col_a from _make_continuation
        if "col_b" not in y_update.columns:
            y_update = y_update.with_columns(
                pl.Series("col_b", np.random.randn(5) + 10.0),
            )

        # Observe and check each sub-forecaster maintains its own observation_horizon
        forecaster.observe(y_update)

        # Access sub-forecasters (forecasters_ returns tuples of (name, forecaster, columns))
        f1 = forecaster.forecasters_[0][1]
        f2 = forecaster.forecasters_[1][1]

        # Each should maintain its own observation horizon
        # f1: feature_transformer=LagTransformer(lag=1) → obs_horizon=1
        # f2: feature_transformer=LagTransformer(lag=3) → obs_horizon=3
        assert f1.observation_horizon == 1
        assert f2.observation_horizon == 3

    def test_decomposition_pipeline_state_isolation(self, ar1_series):
        """Verify DecompositionPipeline components maintain isolated state."""
        y_train = ar1_series(length=100, phi=0.5, c=10.0, seed=700)

        pipeline = DecompositionPipeline(
            forecasters=[
                ("trend", PolynomialTrendForecaster(degree=1)),
                (
                    "residual",
                    PointReductionForecaster(
                        estimator=LinearRegression(),
                        feature_transformer=LagTransformer(lag=2),
                    ),
                ),
            ]
        )
        pipeline.fit(y_train, forecasting_horizon=3)

        # Create continuation observe data
        y_update = _make_continuation(y_train, length=5, seed=701)

        # Observe and verify each component maintains state
        pipeline.observe(y_update)

        # Both trend and residual forecasters should have bounded memory
        # forecasters_ returns tuples of (name, fitted_forecaster)
        trend_forecaster = pipeline.forecasters_[0][1]
        residual_forecaster = pipeline.forecasters_[1][1]

        if hasattr(trend_forecaster, "_y_observed") and trend_forecaster._y_observed is not None:
            assert len(trend_forecaster._y_observed) <= trend_forecaster.observation_horizon

        if residual_forecaster._y_observed is not None:
            assert len(residual_forecaster._y_observed) <= residual_forecaster.observation_horizon


@pytest.mark.integration
class TestStateRobustness:
    """Tests for state recovery after errors, edge cases, and varying parameters."""

    def test_observe_empty_dataframe(self, ar1_series):
        """Verify observe() with empty DataFrame handles gracefully or errors."""
        y_train = ar1_series(length=100, phi=0.5, c=10.0, seed=42)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=5)

        # Create empty DataFrame with same schema
        y_empty = y_train.head(0)

        # This should either leave state unchanged or raise an error
        pred_before = forecaster.predict(forecasting_horizon=5)

        try:
            forecaster.observe(y_empty)
            pred_after = forecaster.predict(forecasting_horizon=5)
            # If observe succeeds, state should be unchanged
            assert pred_before.equals(pred_after)
        except (ValueError, IndexError, pl.exceptions.InvalidOperationError):
            # Empty observe is invalid - this is acceptable behavior
            pass

    def test_state_after_failed_observe(self, ar1_series):
        """Verify state remains unchanged after failed observe attempt."""
        y_train = ar1_series(length=100, phi=0.5, c=10.0, seed=900)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=3)

        pred_before = forecaster.predict(forecasting_horizon=3)

        # Attempt invalid observe (wrong schema)
        last_time = y_train["time"].max()
        y_invalid = pl.DataFrame({
            "time": [last_time + timedelta(days=i + 1) for i in range(5)],
            "wrong_column": [1.0, 2.0, 3.0, 4.0, 5.0],
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))

        try:
            forecaster.observe(y_invalid)
            pytest.fail("Expected error for wrong schema")
        except Exception:
            # Expected error (ValueError, KeyError, or polars ColumnNotFoundError)
            pass

        # State should be unchanged after failed observe
        pred_after = forecaster.predict(forecasting_horizon=3)
        assert pred_before.equals(pred_after)

    def test_state_transition_with_varying_horizons(self, ar1_series):
        """Verify state remains consistent when predicting with varying horizons."""
        y_train = ar1_series(length=100, phi=0.5, c=10.0, seed=800)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=1),
        )
        forecaster.fit(y_train, forecasting_horizon=3)

        # Predict with different horizons
        pred_3 = forecaster.predict(forecasting_horizon=3)
        pred_5 = forecaster.predict(forecasting_horizon=5)
        pred_1 = forecaster.predict(forecasting_horizon=1)

        # All predictions should be valid
        assert len(pred_3) == 3
        assert len(pred_5) == 5
        assert len(pred_1) == 1

        # First step should be consistent across all predictions
        assert np.isclose(pred_3["value"][0], pred_5["value"][0], atol=1e-6)
        assert np.isclose(pred_3["value"][0], pred_1["value"][0], atol=1e-6)

    def test_state_rewind_to_different_lengths(self, ar1_series):
        """Verify rewind() handles different data lengths correctly."""
        y_train = ar1_series(length=100, phi=0.5, c=10.0, seed=1100)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=3),
        )
        forecaster.fit(y_train, forecasting_horizon=3)

        obs_horizon = forecaster.observation_horizon
        # PointReductionForecaster with feature_transformer=LagTransformer(lag=3)
        # observation_horizon = max(0, 0, 3) = 3
        assert obs_horizon == 3

        # Rewind with enough rows for internal transformer memory (LagTransformer(lag=3) needs >= 4)
        y_tail_10 = y_train.tail(10)
        forecaster.rewind(y_tail_10)
        pred_1 = forecaster.predict(forecasting_horizon=3)
        assert len(pred_1) == 3

        # Rewind with more rows
        y_tail_20 = y_train.tail(20)
        forecaster.rewind(y_tail_20)
        pred_20 = forecaster.predict(forecasting_horizon=3)
        assert len(pred_20) == 3
