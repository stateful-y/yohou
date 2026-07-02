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
    """Tests for fit/predict lifecycle.

    Probability sums, bounds, prediction structure, label output, the
    ``classes_`` attribute, and the forecaster-type / uses-reduction tags are all
    covered unconditionally by the systematic suite in
    ``TestClassProbaReductionSystematic`` (``check_class_proba_prediction_sums``,
    ``check_class_proba_prediction_bounds``, ``check_class_proba_prediction_structure``,
    ``check_class_proba_predict_returns_labels``, ``check_class_proba_classes_attribute``,
    ``check_class_proba_prediction_types``, and
    ``check_forecaster_tags_match_capabilities``), so they are not re-asserted here.
    """

    def test_nan_handling_drop_warns_and_predicts(self, class_proba_data):
        """nan_handling='drop' removes NaN feature rows with a warning."""
        import warnings

        y_train, _, X_actual_train, _ = class_proba_data
        # Inject NaN into two feature rows so the drop branch removes them.
        X_with_nan = X_actual_train.with_columns(
            pl.when(pl.int_range(pl.len()).is_in([5, 10])).then(None).otherwise(pl.col("temp")).alias("temp")
        )

        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
            nan_handling="drop",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            forecaster.fit(y_train, X_with_nan, forecasting_horizon=1)
            messages = [str(w.message) for w in caught]

        assert any("NaN handling dropped" in m for m in messages)
        y_pred = forecaster.predict_class_proba(forecasting_horizon=1)
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) == 3

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

    def test_rewind_encodes_categorical_string_labels(self, class_proba_data):
        """rewind() accepts raw string-label y, encoding it before validation.

        ``BaseClassProbaForecaster.rewind`` overrides the base to call
        ``_encode_y_input`` first; passing raw string labels (not pre-encoded)
        pins that encode-then-super contract and leaves a predictable buffer.
        """
        y_train, y_test, X_actual_train, X_actual_test = class_proba_data
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(random_state=42),
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)

        # Observe new ground truth, then rewind with raw string-label data.
        forecaster.observe(y=y_test[:5], X_actual=X_actual_test[:5])
        forecaster.rewind(y=y_train[-3:], X_actual=X_actual_train[-3:])

        y_pred = forecaster.predict_class_proba(forecasting_horizon=3)
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) == 3
        row_sums = y_pred.select(proba_cols).sum_horizontal()
        for s in row_sums:
            assert abs(s - 1.0) < 1e-6


class TestClassProbaNotFitted:
    """predict_class_proba / observe_predict_class_proba guard against unfitted use.

    ``check_forecaster_methods_call_check_is_fitted`` only exercises ``predict``
    and ``observe_predict``; the class-proba variants have their own
    ``check_is_fitted`` guards that the systematic suite does not reach.
    """

    def test_predict_class_proba_raises_not_fitted(self):
        """predict_class_proba on an unfitted instance raises NotFittedError."""
        from sklearn.exceptions import NotFittedError

        forecaster = ClassProbaReductionForecaster(estimator=DecisionTreeClassifier(random_state=42))
        with pytest.raises(NotFittedError):
            forecaster.predict_class_proba(forecasting_horizon=3)

    def test_observe_predict_class_proba_raises_not_fitted(self, class_proba_data):
        """observe_predict_class_proba on an unfitted instance raises NotFittedError."""
        from sklearn.exceptions import NotFittedError

        _, y_test, _, X_actual_test = class_proba_data
        forecaster = ClassProbaReductionForecaster(estimator=DecisionTreeClassifier(random_state=42))
        with pytest.raises(NotFittedError):
            forecaster.observe_predict_class_proba(y=y_test[:3], X_actual=X_actual_test[:3], forecasting_horizon=3)


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

        ``_encode_observation`` must look up ``label_to_code_`` by the base
        target name, so panel columns like ``group_0__weather`` resolve
        correctly (the dict is keyed by the base target name ``weather``)
        rather than raising ``KeyError`` on the raw column name. Recursive
        prediction (horizon > fit horizon, no X_actual) is the path that
        exercises it.
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


class TestEstimatorPredictProbaDispatch:
    """_estimator_predict_proba_one rejects estimators that mismatch the reduction strategy."""

    def test_direct_strategy_requires_list(self):
        """The 'direct' strategy expects a list of estimators, not a single one."""
        forecaster = ClassProbaReductionForecaster(estimator=DecisionTreeClassifier(), reduction_strategy="direct")
        with pytest.raises(TypeError, match="list of estimators for the 'direct' strategy"):
            forecaster._estimator_predict_proba_one(estimator=DecisionTreeClassifier(), groups=[])

    def test_multi_output_strategy_requires_single_estimator(self):
        """The 'multi-output' strategy expects a single estimator, not a list."""
        forecaster = ClassProbaReductionForecaster(
            estimator=DecisionTreeClassifier(), reduction_strategy="multi-output"
        )
        with pytest.raises(TypeError, match="single estimator for the 'multi-output' strategy"):
            forecaster._estimator_predict_proba_one(estimator=[DecisionTreeClassifier()], groups=[])
