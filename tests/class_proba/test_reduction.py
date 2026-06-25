"""Tests for class-probability reduction forecaster."""

from datetime import datetime, timedelta

import polars as pl
import pytest
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier

from conftest import run_checks
from yohou.class_proba import ClassProbaReductionForecaster
from yohou.testing import _yield_yohou_forecaster_checks


@pytest.fixture(scope="module")
def class_proba_data():
    """Module-scoped deterministic categorical data for reduction tests."""
    length = 100
    classes = ["sunny", "rainy", "cloudy"]
    # Deterministic repeating pattern
    values = [classes[i % 3] for i in range(length)]

    start = datetime(2021, 12, 16)
    y = pl.DataFrame({
        "time": pl.datetime_range(
            start=start,
            end=start + timedelta(seconds=length - 1),
            interval="1s",
            eager=True,
        ),
        "weather": values,
    })

    X_actual = pl.DataFrame({
        "time": pl.datetime_range(
            start=start,
            end=start + timedelta(seconds=length - 1),
            interval="1s",
            eager=True,
        ),
        "temp": [20.0 + (i % 10) for i in range(length)],
        "humidity": [50.0 + (i % 5) for i in range(length)],
    })

    y_train, y_test = y[:80], y[80:]
    X_actual_train, X_actual_test = X_actual[:80], X_actual[80:]
    return y_train, y_test, X_actual_train, X_actual_test


class TestClassProbaReductionSystematic:
    """Systematic checks for ClassProbaReductionForecaster."""

    @pytest.mark.parametrize(
        "forecaster",
        [
            ClassProbaReductionForecaster(
                estimator=DecisionTreeClassifier(random_state=42),
                reduction_strategy="multi-output",
            ),
            ClassProbaReductionForecaster(
                estimator=DecisionTreeClassifier(random_state=42),
                reduction_strategy="direct",
            ),
        ],
    )
    def test_systematic_checks(self, forecaster, class_proba_y_X_factory):
        """Run all systematic checks on ClassProbaReductionForecaster."""
        y, X_actual = class_proba_y_X_factory(length=200, n_targets=1, n_features=2, n_classes=3, seed=42)
        y_train, y_test = y[:160], y[160:]
        X_actual_train, X_actual_test = X_actual[:160], X_actual[160:]

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_actual_train, forecasting_horizon=3)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(forecaster_fitted, y_train, X_actual_train, y_test, X_actual_test),
        )

    def test_systematic_checks_panel(self, class_proba_y_X_factory):
        """Run systematic checks with panel data."""
        y, X_actual = class_proba_y_X_factory(
            length=200,
            n_targets=1,
            n_features=2,
            n_classes=3,
            seed=42,
            panel=True,
            n_groups=2,
        )
        y_train, y_test = y[:160], y[160:]
        X_actual_train, X_actual_test = (X_actual[:160], X_actual[160:]) if X_actual is not None else (None, None)

        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)

        run_checks(
            forecaster,
            _yield_yohou_forecaster_checks(forecaster, y_train, X_actual_train, y_test, X_actual_test),
        )


class TestClassProbaReductionFitPredict:
    """Tests for fit/predict lifecycle."""

    def test_predict_class_proba_returns_probabilities(self, class_proba_data):
        """predict_class_proba returns probability columns."""
        y_train, y_test, X_actual_train, X_actual_test = class_proba_data
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)
        y_pred = forecaster.predict_class_proba(forecasting_horizon=3)

        assert "vintage_time" in y_pred.columns
        assert "time" in y_pred.columns
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) == 3  # 3 classes

    def test_predict_returns_string_labels(self, class_proba_data):
        """predict returns argmax class labels."""
        y_train, y_test, X_actual_train, X_actual_test = class_proba_data
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3)

        assert "weather" in y_pred.columns
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) == 0

        valid_classes = set(forecaster.classes_["weather"])
        for val in y_pred["weather"].cast(pl.String).to_list():
            assert val in valid_classes

    def test_classes_discovered_at_fit(self, class_proba_data):
        """classes_ and label_to_code_ populated correctly after fit."""
        y_train, _, X_actual_train, _ = class_proba_data
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=1)

        assert "weather" in forecaster.classes_
        assert sorted(forecaster.classes_["weather"]) == forecaster.classes_["weather"]
        assert set(forecaster.classes_["weather"]) == {"sunny", "rainy", "cloudy"}

        assert "weather" in forecaster.label_to_code_
        assert len(forecaster.label_to_code_["weather"]) == 3

    def test_probabilities_sum_to_one(self, class_proba_data):
        """Per-row probabilities sum to approximately 1.0."""
        y_train, _, X_actual_train, _ = class_proba_data
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=1)
        y_pred = forecaster.predict_class_proba(forecasting_horizon=1)

        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        row_sums = y_pred.select(proba_cols).sum_horizontal()
        for s in row_sums:
            assert abs(s - 1.0) < 1e-6

    def test_probabilities_in_bounds(self, class_proba_data):
        """All probability values in [0, 1]."""
        y_train, _, X_actual_train, _ = class_proba_data
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=1)
        y_pred = forecaster.predict_class_proba(forecasting_horizon=1)

        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        for col in proba_cols:
            assert y_pred[col].min() >= 0.0
            assert y_pred[col].max() <= 1.0

    def test_direct_strategy(self, class_proba_data):
        """Direct strategy produces correct output."""
        y_train, _, X_actual_train, _ = class_proba_data
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
            reduction_strategy="direct",
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)
        y_pred = forecaster.predict_class_proba(forecasting_horizon=3)

        assert len(y_pred) == 3
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) == 3
        row_sums = y_pred.select(proba_cols).sum_horizontal()
        for s in row_sums:
            assert abs(s - 1.0) < 1e-6


class TestObservePredictClassProba:
    """Tests for observe_predict_class_proba lifecycle."""

    def test_observe_predict_class_proba(self, class_proba_data):
        """observe_predict_class_proba returns probability predictions."""
        y_train, y_test, X_actual_train, X_actual_test = class_proba_data
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)

        y_pred = forecaster.observe_predict_class_proba(y=y_test[:3], X_actual=X_actual_test[:3], forecasting_horizon=3)
        assert "vintage_time" in y_pred.columns
        assert "time" in y_pred.columns
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) == 3

    def test_observe_predict_returns_labels(self, class_proba_data):
        """observe_predict returns argmax labels."""
        y_train, y_test, X_actual_train, X_actual_test = class_proba_data
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)

        y_pred = forecaster.observe_predict(y=y_test[:3], X_actual=X_actual_test[:3], forecasting_horizon=3)
        assert "weather" in y_pred.columns
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) == 0


class TestTags:
    """Tests for tag reporting."""

    def test_forecaster_type_tag(self):
        """forecaster_type should be 'class_proba'."""
        forecaster = ClassProbaReductionForecaster()
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.forecaster_type == frozenset({"class_proba"})

    def test_uses_reduction_tag(self):
        """uses_reduction should be True."""
        forecaster = ClassProbaReductionForecaster()
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.uses_reduction is True


class TestRecursivePredict:
    """Tests for recursive prediction with horizon > fit horizon."""

    def test_predict_longer_horizon_than_fit(self, class_proba_data):
        """predict_class_proba with horizon > fit horizon uses recursive prediction (no X_actual)."""
        y_train, _, X_actual_train, X_actual_test = class_proba_data
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y_train, forecasting_horizon=3)

        # Predict with horizon=6, double the fit horizon
        y_pred = forecaster.predict_class_proba(forecasting_horizon=6)
        assert len(y_pred) == 6
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) == 3

    def test_recursive_predict_panel(self, class_proba_y_X_factory):
        """Recursive prediction on panel data re-encodes prefixed columns.

        Regression for the 2026-06-15 QA finding: ``_encode_observation``
        looked up ``label_to_code_`` by the raw column name, so panel columns
        like ``group_0__weather`` raised ``KeyError`` (the dict is keyed by the
        base target name ``weather``). Recursive prediction (horizon > fit
        horizon, no X_actual) is the path that exercises it.
        """
        y, _ = class_proba_y_X_factory(
            length=120,
            n_targets=1,
            n_features=0,
            n_classes=3,
            seed=42,
            panel=True,
            n_groups=2,
        )
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y, forecasting_horizon=3)

        y_pred = forecaster.predict_class_proba(forecasting_horizon=6)
        assert len(y_pred) == 6

    def test_predict_longer_horizon_without_X(self, class_proba_data):
        """Recursive prediction works without exogenous features."""
        y_train, _, X_actual_train, _ = class_proba_data
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y_train, forecasting_horizon=3)

        y_pred = forecaster.predict_class_proba(forecasting_horizon=6)
        assert len(y_pred) == 6

    def test_predict_argmax_longer_horizon(self, class_proba_data):
        """predict (argmax) works with recursive prediction (no X_actual)."""
        y_train, _, X_actual_train, X_actual_test = class_proba_data
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y_train, forecasting_horizon=3)

        y_pred = forecaster.predict(forecasting_horizon=6)
        assert len(y_pred) == 6
        assert "weather" in y_pred.columns
        valid_classes = set(forecaster.classes_["weather"])
        for val in y_pred["weather"].cast(pl.String).to_list():
            assert val in valid_classes


class TestDirectStrategyPanelData:
    """Tests for direct strategy with panel data."""

    def test_direct_strategy_panel(self, class_proba_y_X_factory):
        """Direct strategy works with panel data."""
        y, X_actual = class_proba_y_X_factory(
            length=100,
            n_targets=1,
            n_features=2,
            n_classes=3,
            seed=42,
            panel=True,
            n_groups=2,
        )
        y_train = y[:80]
        X_actual_train = X_actual[:80] if X_actual is not None else None

        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
            reduction_strategy="direct",
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)
        y_pred = forecaster.predict_class_proba(forecasting_horizon=3)

        assert len(y_pred) == 3
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) > 0

    def test_multi_output_strategy_panel(self, class_proba_y_X_factory):
        """Multi-output strategy works with panel data."""
        y, X_actual = class_proba_y_X_factory(
            length=100,
            n_targets=1,
            n_features=2,
            n_classes=3,
            seed=42,
            panel=True,
            n_groups=2,
        )
        y_train = y[:80]
        X_actual_train = X_actual[:80] if X_actual is not None else None

        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
            reduction_strategy="multi-output",
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)
        y_pred = forecaster.predict_class_proba(forecasting_horizon=3)

        assert len(y_pred) == 3
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) > 0


class TestMultiTargetReduction:
    """Tests for multi-target class-probability forecasting."""

    def test_multi_target_multi_output(self, class_proba_y_X_factory):
        """Multi-target with multi-output strategy predicts all targets."""
        y, X_actual = class_proba_y_X_factory(
            length=100,
            n_targets=2,
            n_features=2,
            n_classes=3,
            seed=42,
        )
        y_train = y[:80]
        X_actual_train = X_actual[:80] if X_actual is not None else None

        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
            reduction_strategy="multi-output",
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=1)
        y_pred = forecaster.predict_class_proba(forecasting_horizon=1)

        assert len(y_pred) == 1
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        # 2 targets x 3 classes = 6 proba columns
        assert len(proba_cols) == 6

    def test_multi_target_direct(self, class_proba_y_X_factory):
        """Multi-target with direct strategy predicts all targets."""
        y, X_actual = class_proba_y_X_factory(
            length=100,
            n_targets=2,
            n_features=2,
            n_classes=3,
            seed=42,
        )
        y_train = y[:80]
        X_actual_train = X_actual[:80] if X_actual is not None else None

        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
            reduction_strategy="direct",
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)
        y_pred = forecaster.predict_class_proba(forecasting_horizon=3)

        assert len(y_pred) == 3
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) == 6

    def test_multi_target_predict_returns_labels(self, class_proba_y_X_factory):
        """Multi-target predict returns argmax labels for each target."""
        y, X_actual = class_proba_y_X_factory(
            length=100,
            n_targets=2,
            n_features=2,
            n_classes=3,
            seed=42,
        )
        y_train = y[:80]
        X_actual_train = X_actual[:80] if X_actual is not None else None

        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=1)
        y_pred = forecaster.predict(forecasting_horizon=1)

        assert "y_0" in y_pred.columns
        assert "y_1" in y_pred.columns
