"""Tests for VotingClassProbaForecaster ensemble.

Tests VotingClassProbaForecaster using both the check generator pattern
and specific tests for soft/hard voting behavior, probability correctness,
error handling, and sklearn compatibility.
"""

from datetime import datetime

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier

from conftest import run_checks
from yohou.class_proba import ClassProbaReductionForecaster
from yohou.ensemble import VotingClassProbaForecaster
from yohou.testing import _yield_yohou_forecaster_checks


def _make_class_proba_forecaster(**kwargs):
    """Create a ClassProbaReductionForecaster with deterministic defaults."""
    defaults = {
        "estimator": DecisionTreeClassifier(random_state=42),
        "reduction_strategy": "direct",
    }
    defaults.update(kwargs)
    return ClassProbaReductionForecaster(**defaults)


class TestVotingClassProbaSystematicChecks:
    """Systematic validation via check generators."""

    @pytest.mark.parametrize(
        "forecaster",
        [
            VotingClassProbaForecaster(
                forecasters=[
                    ("dt_1", _make_class_proba_forecaster()),
                    (
                        "dt_2",
                        _make_class_proba_forecaster(
                            estimator=DecisionTreeClassifier(random_state=123),
                        ),
                    ),
                ],
                method="soft",
            ),
            VotingClassProbaForecaster(
                forecasters=[
                    ("dt_1", _make_class_proba_forecaster()),
                    (
                        "dt_2",
                        _make_class_proba_forecaster(
                            estimator=DecisionTreeClassifier(random_state=123),
                        ),
                    ),
                ],
                method="hard",
            ),
            VotingClassProbaForecaster(
                forecasters=[
                    ("dt_1", _make_class_proba_forecaster()),
                    (
                        "dt_2",
                        _make_class_proba_forecaster(
                            estimator=DecisionTreeClassifier(random_state=123),
                        ),
                    ),
                ],
                method="soft",
                weights=[0.3, 0.7],
            ),
        ],
        ids=["soft", "hard", "weighted_soft"],
    )
    def test_systematic_checks(self, forecaster, class_proba_y_X_factory):
        """Run systematic checks on VotingClassProbaForecaster."""
        y, X_actual = class_proba_y_X_factory(length=200, n_targets=1, n_features=2, n_classes=3, seed=42)

        y_train, y_test = y[:160], y[160:]
        X_actual_train, X_actual_test = X_actual[:160], X_actual[160:]

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_actual_train, forecasting_horizon=3)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(forecaster_fitted, y_train, X_actual_train, y_test, X_actual_test),
            expected_failures=set(),
        )


class TestVotingClassProbaFitPredict:
    """Tests for fit/predict lifecycle and correctness."""

    def test_soft_voting_probabilities(self, class_proba_y_X_factory):
        """Test that soft voting produces averaged probabilities."""
        y, X_actual = class_proba_y_X_factory(length=100, n_targets=1, n_features=2, n_classes=3, seed=42)
        y_train = y[:80]
        X_actual_train = X_actual[:80]

        base_1 = _make_class_proba_forecaster()
        base_2 = _make_class_proba_forecaster(
            estimator=DecisionTreeClassifier(random_state=123),
        )

        forecaster = VotingClassProbaForecaster(
            forecasters=[("dt_1", base_1), ("dt_2", base_2)],
            method="soft",
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)
        y_proba = forecaster.predict_class_proba(forecasting_horizon=3)

        # Compute expected mean from individual predictions
        b1 = clone(base_1).fit(y_train, X_actual_train, forecasting_horizon=3)
        b2 = clone(base_2).fit(y_train, X_actual_train, forecasting_horizon=3)
        p1 = b1.predict_class_proba(forecasting_horizon=3)
        p2 = b2.predict_class_proba(forecasting_horizon=3)

        proba_cols = [c for c in y_proba.columns if c not in ("vintage_time", "time")]
        for col in proba_cols:
            expected = (p1[col].to_numpy() + p2[col].to_numpy()) / 2
            np.testing.assert_allclose(y_proba[col].to_numpy(), expected, rtol=1e-10)

    def test_weighted_soft_voting_probabilities(self, class_proba_y_X_factory):
        """Test that weighted soft voting uses numpy.average."""
        y, X_actual = class_proba_y_X_factory(length=100, n_targets=1, n_features=2, n_classes=3, seed=42)
        y_train = y[:80]
        X_actual_train = X_actual[:80]

        base_1 = _make_class_proba_forecaster()
        base_2 = _make_class_proba_forecaster(
            estimator=DecisionTreeClassifier(random_state=123),
        )
        weights = [1.0, 3.0]

        forecaster = VotingClassProbaForecaster(
            forecasters=[("dt_1", base_1), ("dt_2", base_2)],
            method="soft",
            weights=weights,
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)
        y_proba = forecaster.predict_class_proba(forecasting_horizon=3)

        # Compute expected weighted average
        b1 = clone(base_1).fit(y_train, X_actual_train, forecasting_horizon=3)
        b2 = clone(base_2).fit(y_train, X_actual_train, forecasting_horizon=3)
        p1 = b1.predict_class_proba(forecasting_horizon=3)
        p2 = b2.predict_class_proba(forecasting_horizon=3)

        proba_cols = [c for c in y_proba.columns if c not in ("vintage_time", "time")]
        for col in proba_cols:
            values = np.column_stack([p1[col].to_numpy(), p2[col].to_numpy()])
            expected = np.average(values, axis=1, weights=weights)
            np.testing.assert_allclose(y_proba[col].to_numpy(), expected, rtol=1e-10)

    def test_hard_voting_predict(self, class_proba_y_X_factory):
        """Test that hard voting predict returns valid class labels."""
        y, X_actual = class_proba_y_X_factory(length=100, n_targets=1, n_features=2, n_classes=3, seed=42)
        y_train = y[:80]
        X_actual_train = X_actual[:80]

        forecaster = VotingClassProbaForecaster(
            forecasters=[
                ("dt_1", _make_class_proba_forecaster()),
                (
                    "dt_2",
                    _make_class_proba_forecaster(
                        estimator=DecisionTreeClassifier(random_state=123),
                    ),
                ),
                (
                    "dt_3",
                    _make_class_proba_forecaster(
                        estimator=DecisionTreeClassifier(random_state=999),
                    ),
                ),
            ],
            method="hard",
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3)

        target_col = [c for c in y_pred.columns if c not in ("vintage_time", "time")][0]
        all_classes = set()
        for classes in forecaster.classes_.values():
            all_classes.update(classes)

        pred_labels = set(y_pred[target_col].to_list())
        assert pred_labels.issubset(all_classes)

    def test_hard_voting_one_hot_probabilities(self, class_proba_y_X_factory):
        """Test that hard voting probabilities are one-hot."""
        y, X_actual = class_proba_y_X_factory(length=100, n_targets=1, n_features=2, n_classes=3, seed=42)
        y_train = y[:80]
        X_actual_train = X_actual[:80]

        forecaster = VotingClassProbaForecaster(
            forecasters=[
                ("dt_1", _make_class_proba_forecaster()),
                (
                    "dt_2",
                    _make_class_proba_forecaster(
                        estimator=DecisionTreeClassifier(random_state=123),
                    ),
                ),
                (
                    "dt_3",
                    _make_class_proba_forecaster(
                        estimator=DecisionTreeClassifier(random_state=999),
                    ),
                ),
            ],
            method="hard",
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)
        y_proba = forecaster.predict_class_proba(forecasting_horizon=3)

        proba_cols = [c for c in y_proba.columns if c not in ("vintage_time", "time")]
        for col in proba_cols:
            values = y_proba[col].to_numpy()
            assert np.all((values == 0.0) | (values == 1.0)), f"Non-one-hot values in {col}: {values}"

        # Each row should sum to 1
        row_sums = np.sum([y_proba[col].to_numpy() for col in proba_cols], axis=0)
        np.testing.assert_allclose(row_sums, 1.0)

    def test_soft_predict_returns_argmax(self, class_proba_y_X_factory):
        """Test that soft predict returns argmax of probabilities."""
        y, X_actual = class_proba_y_X_factory(length=100, n_targets=1, n_features=2, n_classes=3, seed=42)
        y_train = y[:80]
        X_actual_train = X_actual[:80]

        forecaster = VotingClassProbaForecaster(
            forecasters=[
                ("dt_1", _make_class_proba_forecaster()),
                (
                    "dt_2",
                    _make_class_proba_forecaster(
                        estimator=DecisionTreeClassifier(random_state=123),
                    ),
                ),
            ],
            method="soft",
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3)
        y_proba = forecaster.predict_class_proba(forecasting_horizon=3)

        target_col = [c for c in y_pred.columns if c not in ("vintage_time", "time")][0]
        proba_cols = [c for c in y_proba.columns if c not in ("vintage_time", "time")]
        class_labels = forecaster.classes_[target_col]

        # Verify predict returns the class with highest probability
        proba_matrix = np.column_stack([y_proba[col].to_numpy() for col in proba_cols])
        expected_labels = [class_labels[idx] for idx in np.argmax(proba_matrix, axis=1)]
        actual_labels = y_pred[target_col].to_list()
        assert actual_labels == expected_labels

    def test_weights_ignored_with_hard_voting(self, class_proba_y_X_factory):
        """Test that weights are silently ignored with method='hard'."""
        y, X_actual = class_proba_y_X_factory(length=100, n_targets=1, n_features=2, n_classes=3, seed=42)
        y_train = y[:80]
        X_actual_train = X_actual[:80]

        forecasters = [
            ("dt_1", _make_class_proba_forecaster()),
            (
                "dt_2",
                _make_class_proba_forecaster(
                    estimator=DecisionTreeClassifier(random_state=123),
                ),
            ),
            (
                "dt_3",
                _make_class_proba_forecaster(
                    estimator=DecisionTreeClassifier(random_state=999),
                ),
            ),
        ]

        f_no_w = VotingClassProbaForecaster(
            forecasters=forecasters,
            method="hard",
        )
        f_w = VotingClassProbaForecaster(
            forecasters=forecasters,
            method="hard",
            weights=[0.1, 0.2, 0.7],
        )

        f_no_w.fit(y_train, X_actual_train, forecasting_horizon=3)
        f_w.fit(y_train, X_actual_train, forecasting_horizon=3)

        p_no_w = f_no_w.predict(forecasting_horizon=3)
        p_w = f_w.predict(forecasting_horizon=3)

        target_col = [c for c in p_no_w.columns if c not in ("vintage_time", "time")][0]
        assert p_no_w[target_col].to_list() == p_w[target_col].to_list()


class TestVotingClassProbaPanelData:
    """Tests for panel data support."""

    def test_panel_data_predictions(self, class_proba_y_X_factory):
        """Test VotingClassProbaForecaster works with panel data."""
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
        X_actual_train = X_actual[:80]

        forecaster = VotingClassProbaForecaster(
            forecasters=[
                ("dt_1", _make_class_proba_forecaster()),
                (
                    "dt_2",
                    _make_class_proba_forecaster(
                        estimator=DecisionTreeClassifier(random_state=123),
                    ),
                ),
            ],
            method="soft",
        )
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)
        y_proba = forecaster.predict_class_proba(forecasting_horizon=3)
        y_pred = forecaster.predict(forecasting_horizon=3)

        assert forecaster.groups_ is not None
        assert len(y_proba) == 3
        assert len(y_pred) == 3
        # Panel proba columns should include group prefixes
        proba_cols = [c for c in y_proba.columns if "_proba_" in c]
        assert len(proba_cols) > 0


class TestVotingClassProbaRouting:
    """Metadata-routing guards on predict and predict_class_proba."""

    @pytest.mark.parametrize("method", ["soft", "hard"])
    def test_predict_rejects_unrouted_metadata(self, class_proba_y_X_factory, method):
        """predict/predict_class_proba reject metadata no child requested.

        These methods must call ``_raise_for_params``/``process_routing`` so that
        ``**params`` labelled as routing metadata is validated; a stray param must
        be rejected rather than passed through to children unchecked.
        """
        y, X_actual = class_proba_y_X_factory(length=100, n_targets=1, n_features=2, n_classes=3, seed=42)
        forecaster = VotingClassProbaForecaster(
            forecasters=[("dt_1", _make_class_proba_forecaster()), ("dt_2", _make_class_proba_forecaster())],
            method=method,
        )
        forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=3)

        with pytest.raises(TypeError, match="not routed"):
            forecaster.predict(forecasting_horizon=3, sample_weight=[1.0] * 3)
        with pytest.raises(TypeError, match="not routed"):
            forecaster.predict_class_proba(forecasting_horizon=3, sample_weight=[1.0] * 3)


class TestVotingClassProbaErrorHandling:
    """Tests for error handling behavior."""

    def test_single_forecaster_failure_skipped(self, class_proba_y_X_factory):
        """Test that a single failing forecaster is skipped with warning."""
        y, X_actual = class_proba_y_X_factory(length=100, n_targets=1, n_features=2, n_classes=3, seed=42)
        y_train = y[:80]
        X_actual_train = X_actual[:80]

        # Use a tiny data slice so one forecaster with large lag requirements
        # fails while the other succeeds
        from sklearn.base import BaseEstimator, ClassifierMixin

        class AlwaysFailClassifier(ClassifierMixin, BaseEstimator):
            def fit(self, X_actual, y):
                raise ValueError("Intentional failure in classifier")

            def predict(self, X_actual):
                pass

            def predict_proba(self, X_actual):
                pass

        forecaster = VotingClassProbaForecaster(
            forecasters=[
                ("good", _make_class_proba_forecaster()),
                (
                    "bad",
                    ClassProbaReductionForecaster(
                        estimator=AlwaysFailClassifier(),
                        reduction_strategy="direct",
                    ),
                ),
            ],
            method="soft",
        )

        with pytest.warns(UserWarning, match="failed during fit"):
            forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)

        assert len(forecaster.forecasters_) == 1
        y_proba = forecaster.predict_class_proba(forecasting_horizon=3)
        assert len(y_proba) == 3

    def test_all_forecasters_fail_raises(self, class_proba_y_X_factory):
        """Test that ensemble raises when all forecasters fail."""
        y, _ = class_proba_y_X_factory(length=10, n_targets=1, n_features=0, n_classes=3, seed=42)

        # Very short data likely to fail
        forecaster = VotingClassProbaForecaster(
            forecasters=[
                (
                    "bad1",
                    ClassProbaReductionForecaster(
                        estimator=DecisionTreeClassifier(random_state=42),
                        reduction_strategy="direct",
                    ),
                ),
                (
                    "bad2",
                    ClassProbaReductionForecaster(
                        estimator=DecisionTreeClassifier(random_state=42),
                        reduction_strategy="direct",
                    ),
                ),
            ],
            method="soft",
        )

        with pytest.raises((RuntimeError, ValueError)):
            forecaster.fit(y[:3], forecasting_horizon=10)

    def test_weights_length_mismatch_raises(self, class_proba_y_X_factory):
        """Test that mismatched weights length raises ValueError."""
        y, X_actual = class_proba_y_X_factory(length=100, n_targets=1, n_features=2, n_classes=3, seed=42)

        forecaster = VotingClassProbaForecaster(
            forecasters=[
                ("dt_1", _make_class_proba_forecaster()),
                (
                    "dt_2",
                    _make_class_proba_forecaster(
                        estimator=DecisionTreeClassifier(random_state=123),
                    ),
                ),
            ],
            weights=[0.5],
            method="soft",
        )

        with pytest.raises(ValueError, match="Number of weights"):
            forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=3)

    def test_duplicate_names_raises(self, class_proba_y_X_factory):
        """Test that duplicate forecaster names raise ValueError."""
        y, X_actual = class_proba_y_X_factory(length=100, n_targets=1, n_features=2, n_classes=3, seed=42)

        forecaster = VotingClassProbaForecaster(
            forecasters=[
                ("same", _make_class_proba_forecaster()),
                (
                    "same",
                    _make_class_proba_forecaster(
                        estimator=DecisionTreeClassifier(random_state=123),
                    ),
                ),
            ],
            method="soft",
        )

        with pytest.raises(ValueError, match="Duplicate forecaster names"):
            forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=3)

    def test_double_underscore_name_raises(self, class_proba_y_X_factory):
        """A forecaster name containing '__' is rejected.

        A name like 'foo__bar' must fail validation; otherwise it corrupts
        nested-parameter routing.
        """
        y, X_actual = class_proba_y_X_factory(length=100, n_targets=1, n_features=2, n_classes=3, seed=42)

        forecaster = VotingClassProbaForecaster(
            forecasters=[
                ("foo__bar", _make_class_proba_forecaster()),
                (
                    "dt_2",
                    _make_class_proba_forecaster(
                        estimator=DecisionTreeClassifier(random_state=123),
                    ),
                ),
            ],
            method="soft",
        )

        with pytest.raises(ValueError, match="must not contain '__'"):
            forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=3)

    def test_wrong_family_forecaster_raises(self, class_proba_y_X_factory):
        """A non-class-probability forecaster is rejected at fit with a clear error."""
        from yohou.point import SeasonalNaive

        y, X_actual = class_proba_y_X_factory(length=100, n_targets=1, n_features=2, n_classes=3, seed=42)

        forecaster = VotingClassProbaForecaster(
            forecasters=[
                ("cp", _make_class_proba_forecaster()),
                # SeasonalNaive is a point forecaster, not a class-probability one.
                ("point", SeasonalNaive(seasonality=1)),
            ],
            method="soft",
        )

        with pytest.raises(ValueError, match="BaseClassProbaForecaster"):
            forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=3)


class TestVotingClassProbaConsistency:
    """Tests for class consistency validation across base forecasters."""

    def test_mismatched_classes_raises(self):
        """Test that forecasters discovering different classes raise ValueError."""
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 4, 10),
            interval="1d",
            eager=True,
        )
        n = len(time)

        # Two datasets with different class labels
        y_ab = pl.DataFrame({
            "time": time,
            "label": [["cat_a", "cat_b"][i % 2] for i in range(n)],
        })
        y_cd = pl.DataFrame({
            "time": time,
            "label": [["cat_c", "cat_d"][i % 2] for i in range(n)],
        })

        # Fit two forecasters on data with different classes
        f1 = _make_class_proba_forecaster()
        f2 = _make_class_proba_forecaster(
            estimator=DecisionTreeClassifier(random_state=123),
        )

        f1.fit(y_ab[:80], forecasting_horizon=3)
        f2.fit(y_cd[:80], forecasting_horizon=3)

        # They should have different classes_
        assert f1.classes_ != f2.classes_

        # Build an ensemble that already has fitted forecasters injected
        # by fitting normally with same data, then swapping one out
        ensemble = VotingClassProbaForecaster(
            forecasters=[
                ("f1", _make_class_proba_forecaster()),
                (
                    "f2",
                    _make_class_proba_forecaster(
                        estimator=DecisionTreeClassifier(random_state=123),
                    ),
                ),
            ],
            method="soft",
        )

        # Directly test _validate_classes_consistent by simulating the state
        ensemble.forecasters_ = [("f1", f1), ("f2", f2)]
        with pytest.raises(ValueError, match="discovered classes"):
            ensemble._validate_classes_consistent()


class TestVotingClassProbaObserveRewind:
    """Tests for observe, rewind, and observe_predict delegation."""

    def test_observe_and_predict(self, class_proba_y_X_factory):
        """Test that observe delegates to all base forecasters."""
        y, X_actual = class_proba_y_X_factory(length=120, n_targets=1, n_features=2, n_classes=3, seed=42)

        forecaster = VotingClassProbaForecaster(
            forecasters=[
                ("dt_1", _make_class_proba_forecaster()),
                (
                    "dt_2",
                    _make_class_proba_forecaster(
                        estimator=DecisionTreeClassifier(random_state=123),
                    ),
                ),
            ],
            method="soft",
        )
        forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=3)
        forecaster.observe(y=y[80:100], X_actual=X_actual[80:100])
        y_pred = forecaster.predict(forecasting_horizon=3)
        assert len(y_pred) == 3

    def test_rewind_and_predict(self, class_proba_y_X_factory):
        """Test that rewind delegates to all base forecasters."""
        y, X_actual = class_proba_y_X_factory(length=120, n_targets=1, n_features=2, n_classes=3, seed=42)

        forecaster = VotingClassProbaForecaster(
            forecasters=[
                ("dt_1", _make_class_proba_forecaster()),
                (
                    "dt_2",
                    _make_class_proba_forecaster(
                        estimator=DecisionTreeClassifier(random_state=123),
                    ),
                ),
            ],
            method="soft",
        )
        forecaster.fit(y[:90], X_actual[:90], forecasting_horizon=3)
        forecaster.observe(y=y[90:100], X_actual=X_actual[90:100])
        forecaster.rewind(y=y[:90], X_actual=X_actual[:90])
        y_pred = forecaster.predict(forecasting_horizon=3)
        assert len(y_pred) == 3

    def test_observe_predict_class_proba_soft(self, class_proba_y_X_factory):
        """Test observe_predict_class_proba exercises _predict_class_proba_one (soft)."""
        y, X_actual = class_proba_y_X_factory(length=120, n_targets=1, n_features=2, n_classes=3, seed=42)

        forecaster = VotingClassProbaForecaster(
            forecasters=[
                ("dt_1", _make_class_proba_forecaster()),
                (
                    "dt_2",
                    _make_class_proba_forecaster(
                        estimator=DecisionTreeClassifier(random_state=123),
                    ),
                ),
            ],
            method="soft",
        )
        forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=3)
        y_proba = forecaster.observe_predict_class_proba(y=y[80:90], X_actual=X_actual[80:90], forecasting_horizon=3)
        assert len(y_proba) > 0
        proba_cols = [c for c in y_proba.columns if "_proba_" in c]
        assert len(proba_cols) > 0

    def test_observe_predict_class_proba_hard(self, class_proba_y_X_factory):
        """Test observe_predict_class_proba exercises _predict_class_proba_one (hard)."""
        y, X_actual = class_proba_y_X_factory(length=120, n_targets=1, n_features=2, n_classes=3, seed=42)

        forecaster = VotingClassProbaForecaster(
            forecasters=[
                ("dt_1", _make_class_proba_forecaster()),
                (
                    "dt_2",
                    _make_class_proba_forecaster(
                        estimator=DecisionTreeClassifier(random_state=123),
                    ),
                ),
                (
                    "dt_3",
                    _make_class_proba_forecaster(
                        estimator=DecisionTreeClassifier(random_state=999),
                    ),
                ),
            ],
            method="hard",
        )
        forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=3)
        y_proba = forecaster.observe_predict_class_proba(y=y[80:90], X_actual=X_actual[80:90], forecasting_horizon=3)
        assert len(y_proba) > 0
        proba_cols = [c for c in y_proba.columns if "_proba_" in c]
        assert len(proba_cols) > 0

    def test_predict_class_proba_one_soft(self, class_proba_y_X_factory):
        """Test _predict_class_proba_one directly for soft voting."""
        y, X_actual = class_proba_y_X_factory(length=100, n_targets=1, n_features=2, n_classes=3, seed=42)

        forecaster = VotingClassProbaForecaster(
            forecasters=[
                ("dt_1", _make_class_proba_forecaster()),
                (
                    "dt_2",
                    _make_class_proba_forecaster(
                        estimator=DecisionTreeClassifier(random_state=123),
                    ),
                ),
            ],
            method="soft",
            weights=[1.0, 3.0],
        )
        forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=3)

        y_proba = forecaster._predict_class_proba_one(groups=None)
        proba_cols = [c for c in y_proba.columns if "_proba_" in c]
        assert len(proba_cols) > 0
        assert "time" in y_proba.columns

    def test_predict_class_proba_one_hard(self, class_proba_y_X_factory):
        """Test _predict_class_proba_one directly for hard voting."""
        y, X_actual = class_proba_y_X_factory(length=100, n_targets=1, n_features=2, n_classes=3, seed=42)

        forecaster = VotingClassProbaForecaster(
            forecasters=[
                ("dt_1", _make_class_proba_forecaster()),
                (
                    "dt_2",
                    _make_class_proba_forecaster(
                        estimator=DecisionTreeClassifier(random_state=123),
                    ),
                ),
                (
                    "dt_3",
                    _make_class_proba_forecaster(
                        estimator=DecisionTreeClassifier(random_state=999),
                    ),
                ),
            ],
            method="hard",
        )
        forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=3)

        y_proba = forecaster._predict_class_proba_one(groups=None)
        proba_cols = [c for c in y_proba.columns if "_proba_" in c]
        assert len(proba_cols) > 0
        # Hard voting should produce one-hot probabilities
        for col in proba_cols:
            values = y_proba[col].to_numpy()
            assert np.all((values == 0.0) | (values == 1.0))


class TestVotingClassProbaSklearn:
    """Tests for sklearn compatibility."""

    def test_clone(self, class_proba_y_X_factory):
        """Test that VotingClassProbaForecaster can be cloned."""
        y, X_actual = class_proba_y_X_factory(length=100, n_targets=1, n_features=2, n_classes=3, seed=42)

        forecaster = VotingClassProbaForecaster(
            forecasters=[
                ("dt_1", _make_class_proba_forecaster()),
                (
                    "dt_2",
                    _make_class_proba_forecaster(
                        estimator=DecisionTreeClassifier(random_state=123),
                    ),
                ),
            ],
            method="soft",
        )
        forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=3)

        cloned = clone(forecaster)
        with pytest.raises(NotFittedError):
            cloned.predict(forecasting_horizon=3)

    def test_get_set_params(self):
        """Test get_params and set_params with nested estimators."""
        forecaster = VotingClassProbaForecaster(
            forecasters=[
                ("dt_1", _make_class_proba_forecaster()),
                (
                    "dt_2",
                    _make_class_proba_forecaster(
                        estimator=DecisionTreeClassifier(random_state=123),
                    ),
                ),
            ],
            method="soft",
        )

        params = forecaster.get_params(deep=True)
        assert params["method"] == "soft"
        assert params["dt_1__estimator__random_state"] == 42
        assert params["dt_2__estimator__random_state"] == 123

        forecaster.set_params(dt_1__estimator__random_state=99)
        assert forecaster.get_params()["dt_1__estimator__random_state"] == 99

    def test_predict_class_proba_not_fitted_error(self):
        """predict_class_proba raises NotFittedError before fit.

        ``predict`` is covered by ``check_forecaster_methods_call_check_is_fitted``;
        only ``predict_class_proba`` needs this dedicated guard test.
        """
        forecaster = VotingClassProbaForecaster(
            forecasters=[
                ("dt_1", _make_class_proba_forecaster()),
            ],
            method="soft",
        )

        with pytest.raises(NotFittedError):
            forecaster.predict_class_proba(forecasting_horizon=3)


class TestVotingClassProbaTags:
    """Tag propagation from child forecasters, matching sibling voters."""

    def test_stateful_propagates_from_children(self):
        """stateful is True when any child is stateful, like the point/interval voters."""
        from yohou.preprocessing import LagTransformer

        stateful_child = _make_class_proba_forecaster(actual_transformer=LagTransformer(lag=1))
        assert stateful_child.__sklearn_tags__().forecaster_tags.stateful is True

        forecaster = VotingClassProbaForecaster(
            forecasters=[
                ("stateful", stateful_child),
                ("plain", _make_class_proba_forecaster()),
            ],
            method="soft",
        )
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.stateful is True

    def test_stateful_false_when_no_child_stateful(self):
        """stateful is False when no child is stateful."""
        forecaster = VotingClassProbaForecaster(
            forecasters=[
                ("dt_1", _make_class_proba_forecaster()),
                ("dt_2", _make_class_proba_forecaster()),
            ],
            method="soft",
        )
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.stateful is False
