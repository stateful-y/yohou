"""Tests for class-probability reduction forecaster."""

from datetime import datetime, timedelta

import polars as pl
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
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

    X = pl.DataFrame({
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
    X_train, X_test = X[:80], X[80:]
    return y_train, y_test, X_train, X_test


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
        y, X = class_proba_y_X_factory(length=200, n_targets=1, n_features=2, n_classes=3, seed=42)
        y_train, y_test = y[:160], y[160:]
        X_train, X_test = X[:160], X[160:]

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_train, forecasting_horizon=3)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(forecaster_fitted, y_train, X_train, y_test, X_test),
        )


class TestClassProbaReductionFitPredict:
    """Tests for fit/predict lifecycle."""

    def test_predict_class_proba_returns_probabilities(self, class_proba_data):
        """predict_class_proba returns probability columns."""
        y_train, y_test, X_train, X_test = class_proba_data
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y_train, X_train, forecasting_horizon=3)
        y_pred = forecaster.predict_class_proba(forecasting_horizon=3, X=X_test[:3])

        assert "observed_time" in y_pred.columns
        assert "time" in y_pred.columns
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) == 3  # 3 classes

    def test_predict_returns_string_labels(self, class_proba_data):
        """predict returns argmax class labels."""
        y_train, y_test, X_train, X_test = class_proba_data
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y_train, X_train, forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3, X=X_test[:3])

        assert "weather" in y_pred.columns
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) == 0

        valid_classes = set(forecaster.classes_["weather"])
        for val in y_pred["weather"].cast(pl.String).to_list():
            assert val in valid_classes

    def test_classes_discovered_at_fit(self, class_proba_data):
        """classes_ and label_to_code_ populated correctly after fit."""
        y_train, _, X_train, _ = class_proba_data
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y_train, X_train, forecasting_horizon=1)

        assert "weather" in forecaster.classes_
        assert sorted(forecaster.classes_["weather"]) == forecaster.classes_["weather"]
        assert set(forecaster.classes_["weather"]) == {"sunny", "rainy", "cloudy"}

        assert "weather" in forecaster.label_to_code_
        assert len(forecaster.label_to_code_["weather"]) == 3

    def test_probabilities_sum_to_one(self, class_proba_data):
        """Per-row probabilities sum to approximately 1.0."""
        y_train, _, X_train, _ = class_proba_data
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y_train, X_train, forecasting_horizon=1)
        y_pred = forecaster.predict_class_proba(forecasting_horizon=1)

        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        row_sums = y_pred.select(proba_cols).sum_horizontal()
        for s in row_sums:
            assert abs(s - 1.0) < 1e-6

    def test_probabilities_in_bounds(self, class_proba_data):
        """All probability values in [0, 1]."""
        y_train, _, X_train, _ = class_proba_data
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y_train, X_train, forecasting_horizon=1)
        y_pred = forecaster.predict_class_proba(forecasting_horizon=1)

        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        for col in proba_cols:
            assert y_pred[col].min() >= 0.0
            assert y_pred[col].max() <= 1.0

    def test_direct_strategy(self, class_proba_data):
        """Direct strategy produces correct output."""
        y_train, _, X_train, _ = class_proba_data
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
            reduction_strategy="direct",
        )
        forecaster.fit(y_train, X_train, forecasting_horizon=3)
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
        y_train, y_test, X_train, X_test = class_proba_data
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y_train, X_train, forecasting_horizon=3)

        y_pred = forecaster.observe_predict_class_proba(
            y=y_test[:3], X=X_test[:3], forecasting_horizon=3
        )
        assert "observed_time" in y_pred.columns
        assert "time" in y_pred.columns
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) == 3

    def test_observe_predict_returns_labels(self, class_proba_data):
        """observe_predict returns argmax labels."""
        y_train, y_test, X_train, X_test = class_proba_data
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y_train, X_train, forecasting_horizon=3)

        y_pred = forecaster.observe_predict(
            y=y_test[:3], X=X_test[:3], forecasting_horizon=3
        )
        assert "weather" in y_pred.columns
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) == 0


class TestTags:
    """Tests for tag reporting."""

    def test_forecaster_type_tag(self):
        """forecaster_type should be 'class_proba'."""
        forecaster = ClassProbaReductionForecaster()
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.forecaster_type == "class_proba"

    def test_uses_reduction_tag(self):
        """uses_reduction should be True."""
        forecaster = ClassProbaReductionForecaster()
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.uses_reduction is True
