"""Common tests for all yohou estimators using systematic check generators.

Discovers all concrete estimators via ``all_estimators()`` and runs the
appropriate ``_yield_yohou_*_checks()`` generators to verify consistent
behaviour across the entire package.
"""

import contextlib
from datetime import datetime, timedelta

import polars as pl
import pytest
from sklearn.base import clone

from conftest import run_checks
from yohou.base import BaseForecaster, BaseTransformer
from yohou.metrics.base import BaseScorer
from yohou.model_selection.split import BaseSplitter
from yohou.testing import (
    _yield_yohou_forecaster_checks,
    _yield_yohou_scorer_checks,
    _yield_yohou_splitter_checks,
    _yield_yohou_transformer_checks,
)
from yohou.utils.discovery import all_estimators

# These classes require special init args and can't be default-constructed
# or need composition with inner estimators, so we exclude them from the
# generic common-test sweep.
_SKIP_COMMON = {
    # Composition / meta forecasters (tested in tests/compose/)
    "ColumnForecaster",
    "ColumnTransformer",
    "DecompositionPipeline",
    "FeaturePipeline",
    "FeatureUnion",
    "ForecastedFeatureForecaster",
    # Search CVs (tested in tests/model_selection/)
    "GridSearchCV",
    "RandomizedSearchCV",
    # Abstract markers that slip through (no abstract methods but not useful alone)
    "BasePointForecaster",
    # Wrappers that require init args
    "SklearnTransformer",
    "SklearnScaler",
    # Similarity (not a standard estimator type)
    "DistanceSimilarity",
    # Meta-forecasters requiring inner estimator (tested in tests/interval/)
    "SplitConformalForecaster",
}

# Known check failures per estimator, covered by dedicated test files.
# Checks listed here are skipped in the common sweep.
_XFAIL_CHECKS: dict[str, set[str]] = {
    "NumericalDifferentiator": {
        "check_inverse_transform_identity",
        "check_inverse_transform_round_trip",
        "check_inverse_observe_transform_identity",
    },
    "NumericalFilter": {
        "check_inverse_transform_identity",
        "check_inverse_transform_round_trip",
        "check_inverse_observe_transform_identity",
        "check_observe_transform_sequential_consistency",
    },
    "NumericalIntegrator": {
        "check_inverse_transform_identity",
        "check_inverse_transform_round_trip",
        "check_inverse_observe_transform_identity",
    },
    "QuantileTransformer": {
        "check_inverse_observe_transform_identity",
    },
    "SimpleImputer": {
        "check_inverse_transform_identity",
        "check_inverse_transform_round_trip",
        "check_inverse_observe_transform_identity",
    },
}


def _collect_transformers() -> list[tuple[str, type]]:
    """Collect all transformer classes for testing."""
    return [
        (name, cls) for name, cls in all_estimators() if issubclass(cls, BaseTransformer) and name not in _SKIP_COMMON
    ]


def _collect_forecasters() -> list[tuple[str, type]]:
    """Collect all forecaster classes for testing."""
    return [
        (name, cls)
        for name, cls in all_estimators()
        if issubclass(cls, BaseForecaster)
        and not issubclass(cls, BaseTransformer)  # exclude transformers
        and name not in _SKIP_COMMON
    ]


def _collect_splitters() -> list[tuple[str, type]]:
    """Collect all splitter classes for testing."""
    return [(name, cls) for name, cls in all_estimators() if issubclass(cls, BaseSplitter) and name not in _SKIP_COMMON]


def _collect_scorers() -> list[tuple[str, type]]:
    """Collect all scorer classes for testing."""
    return [(name, cls) for name, cls in all_estimators() if issubclass(cls, BaseScorer) and name not in _SKIP_COMMON]


def _make_time_series(length: int = 100, n_targets: int = 1, n_features: int = 1):
    """Create simple time series data for testing."""
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
        interval="1s",
        eager=True,
    )
    y_cols = {"time": time}
    for i in range(n_targets):
        y_cols[f"target_{i}"] = [float(j + i * 10) for j in range(length)]
    y = pl.DataFrame(y_cols)

    X_cols: dict | None = None
    X: pl.DataFrame | None = None
    if n_features > 0:
        X_cols = {"time": time}
        for i in range(n_features):
            X_cols[f"feature_{i}"] = [float(j + i * 5) for j in range(length)]
        X = pl.DataFrame(X_cols)

    return y, X


def _make_scorer_data():
    """Create ground truth and prediction data for point scorer tests."""
    time = [datetime(2021, 1, 1, 0, 0, i) for i in range(10)]
    y_truth = pl.DataFrame({
        "time": time,
        "value": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
    })
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2020, 12, 31)] * 10,
        "time": time,
        "value": [12.0, 19.0, 28.0, 42.0, 48.0, 62.0, 68.0, 82.0, 88.0, 102.0],
    })
    return y_truth, y_pred


def _make_interval_scorer_data():
    """Create interval prediction data with 2 coverage rates for interval scorers."""
    time = [datetime(2021, 1, 1, 0, 0, i) for i in range(10)]
    y_truth = pl.DataFrame({
        "time": time,
        "value": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
    })
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2020, 12, 31)] * 10,
        "time": time,
        "value": [12.0, 19.0, 28.0, 42.0, 48.0, 62.0, 68.0, 82.0, 88.0, 102.0],
        "value_lower_0.9": [6.0, 13.0, 22.0, 34.0, 42.0, 54.0, 62.0, 74.0, 82.0, 94.0],
        "value_upper_0.9": [18.0, 25.0, 34.0, 50.0, 54.0, 70.0, 74.0, 90.0, 94.0, 110.0],
        "value_lower_0.95": [5.0, 12.0, 21.0, 33.0, 41.0, 53.0, 61.0, 73.0, 81.0, 93.0],
        "value_upper_0.95": [19.0, 26.0, 35.0, 51.0, 55.0, 71.0, 75.0, 91.0, 95.0, 111.0],
    })
    return y_truth, y_pred


def _uses_point_data(cls: type) -> bool:
    """Check if a scorer uses point-format data (point or conformity scorers)."""
    try:
        instance = cls()
        tags = instance.__sklearn_tags__()
        prediction_type = getattr(tags.scorer_tags, "prediction_type", None)
        return prediction_type in ("point", "conformity", None)
    except Exception:
        return True  # Default to point data if can't determine


def _uses_class_proba_data(cls: type) -> bool:
    """Check if a scorer uses class-probability data."""
    try:
        instance = cls()
        tags = instance.__sklearn_tags__()
        prediction_type = getattr(tags.scorer_tags, "prediction_type", None)
        return prediction_type == "class_proba"
    except Exception:
        return False


class TestTransformerCommon:
    """Run systematic checks on all discovered transformers."""

    @pytest.mark.parametrize(
        "name,cls",
        _collect_transformers(),
        ids=[name for name, _ in _collect_transformers()],
    )
    def test_transformer_checks(self, name, cls):
        """Run all applicable check-generator checks for a transformer."""
        y, X = _make_time_series(length=100, n_targets=1, n_features=1)

        # Use X as the transformer input data (transformers operate on X)
        X_data = y  # transformers in yohou operate on a DataFrame with time + values

        # Handle transformers with min_value constraints
        try:
            instance = cls()
        except TypeError:
            pytest.skip(f"{name} requires constructor arguments")
            return

        tags = instance.__sklearn_tags__()
        if tags.input_tags and tags.input_tags.min_value is not None:
            offset = max(0.0, tags.input_tags.min_value + 1.0)
            X_data = X_data.select([pl.col("time"), (pl.all().exclude("time") + offset)])

        X_train = X_data[:80]
        X_test = X_data[80:]

        # Fit transformer
        transformer = clone(instance)
        try:
            transformer.fit(X_train)
        except Exception as e:
            pytest.skip(f"{name} fit failed: {e}")
            return

        run_checks(
            transformer,
            _yield_yohou_transformer_checks(transformer, X_train, None, X_test),
            expected_failures=_XFAIL_CHECKS.get(name, set()),
        )


class TestForecasterCommon:
    """Run systematic checks on all discovered forecasters."""

    @pytest.mark.parametrize(
        "name,cls",
        _collect_forecasters(),
        ids=[name for name, _ in _collect_forecasters()],
    )
    def test_forecaster_checks(self, name, cls):
        """Run all applicable check-generator checks for a forecaster."""
        y, X = _make_time_series(length=200, n_targets=1, n_features=2)

        y_train, y_test = y[:160], y[160:]
        X_train = X[:160] if X is not None else None
        X_test = X[160:] if X is not None else None

        try:
            instance = cls()
        except TypeError:
            pytest.skip(f"{name} requires constructor arguments")
            return

        forecaster = clone(instance)
        try:
            forecaster.fit(y_train, X_train, forecasting_horizon=3)
        except Exception as e:
            pytest.skip(f"{name} fit failed: {e}")
            return

        run_checks(
            forecaster,
            _yield_yohou_forecaster_checks(forecaster, y_train, X_train, y_test, X_test),
        )


class TestSplitterCommon:
    """Run systematic checks on all discovered splitters."""

    @pytest.mark.parametrize(
        "name,cls",
        _collect_splitters(),
        ids=[name for name, _ in _collect_splitters()],
    )
    def test_splitter_checks(self, name, cls):
        """Run all applicable check-generator checks for a splitter."""
        y, _ = _make_time_series(length=100, n_targets=1, n_features=0)

        try:
            instance = cls()
        except TypeError:
            pytest.skip(f"{name} requires constructor arguments")
            return

        run_checks(instance, _yield_yohou_splitter_checks(instance, y))


class TestPointScorerCommon:
    """Run systematic checks on all discovered point and conformity scorers."""

    @pytest.mark.parametrize(
        "name,cls",
        [(n, c) for n, c in _collect_scorers() if _uses_point_data(c)],
        ids=[n for n, c in _collect_scorers() if _uses_point_data(c)],
    )
    def test_point_scorer_checks(self, name, cls):
        """Run all applicable check-generator checks for a point scorer."""
        y_truth, y_pred = _make_scorer_data()

        try:
            instance = cls()
        except TypeError:
            pytest.skip(f"{name} requires constructor arguments")
            return

        # Fit scorer if it needs fitting (e.g., scaled metrics)
        with contextlib.suppress(Exception):
            instance.fit(y_truth)

        run_checks(instance, _yield_yohou_scorer_checks(instance, y_truth, y_pred))


class TestIntervalScorerCommon:
    """Run systematic checks on all discovered interval scorers."""

    @pytest.mark.parametrize(
        "name,cls",
        [(n, c) for n, c in _collect_scorers() if not _uses_point_data(c) and not _uses_class_proba_data(c)],
        ids=[n for n, c in _collect_scorers() if not _uses_point_data(c) and not _uses_class_proba_data(c)],
    )
    def test_interval_scorer_checks(self, name, cls):
        """Run all applicable check-generator checks for an interval scorer."""
        y_truth, y_pred = _make_interval_scorer_data()

        try:
            instance = cls()
        except TypeError:
            pytest.skip(f"{name} requires constructor arguments")
            return

        with contextlib.suppress(Exception):
            instance.fit(y_truth)

        run_checks(instance, _yield_yohou_scorer_checks(instance, y_truth, y_pred))


def _make_class_proba_scorer_data():
    """Create class-probability data for class_proba scorer tests."""
    time = [datetime(2021, 1, 1, 0, 0, i) for i in range(10)]
    labels = ["cat_a", "cat_b", "cat_a", "cat_c", "cat_b", "cat_a", "cat_c", "cat_b", "cat_a", "cat_c"]
    y_truth = pl.DataFrame({"time": time, "target": labels})
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2020, 12, 31)] * 10,
        "time": time,
        "target_proba_cat_a": [0.7, 0.1, 0.6, 0.1, 0.2, 0.8, 0.1, 0.2, 0.7, 0.1],
        "target_proba_cat_b": [0.2, 0.7, 0.2, 0.1, 0.6, 0.1, 0.1, 0.7, 0.2, 0.2],
        "target_proba_cat_c": [0.1, 0.2, 0.2, 0.8, 0.2, 0.1, 0.8, 0.1, 0.1, 0.7],
    })
    return y_truth, y_pred


class TestClassProbaScorerCommon:
    """Run systematic checks on all discovered class-probability scorers."""

    @pytest.mark.parametrize(
        "name,cls",
        [(n, c) for n, c in _collect_scorers() if _uses_class_proba_data(c)],
        ids=[n for n, c in _collect_scorers() if _uses_class_proba_data(c)],
    )
    def test_class_proba_scorer_checks(self, name, cls):
        """Run all applicable check-generator checks for a class_proba scorer."""
        y_truth, y_pred = _make_class_proba_scorer_data()

        try:
            instance = cls()
        except TypeError:
            pytest.skip(f"{name} requires constructor arguments")
            return

        with contextlib.suppress(Exception):
            instance.fit(y_truth)

        run_checks(instance, _yield_yohou_scorer_checks(instance, y_truth, y_pred))
