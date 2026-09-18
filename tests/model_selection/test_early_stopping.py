"""Tests for the early-stopping adapters used by ``validation="cv"`` searches.

Each built-in adapter is checked against its library's own behaviour: the
stopping curve against the library's reported best round, truncation against
the library's own truncated prediction, and refit preparation against a model
trained for the same number of rounds.
"""

import pickle
import sys
from unittest import mock

import catboost
import lightgbm
import numpy as np
import pytest
import xgboost
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from yohou.model_selection import (
    BaseEarlyStoppingAdapter,
    CatBoostEarlyStoppingAdapter,
    HistGradientBoostingEarlyStoppingAdapter,
    LightGBMEarlyStoppingAdapter,
    XGBoostEarlyStoppingAdapter,
)
from yohou.model_selection.early_stopping import _resolve_early_stopping_adapter
from yohou.model_selection.utils import _select_shared_rounds

PATIENCE = 10


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(0)
    X = rng.random((500, 5))
    y = 3 * X[:, 0] + np.sin(8 * X[:, 1]) + rng.normal(0, 0.5, 500)
    return X[:350], y[:350], X[350:], y[350:]


@pytest.fixture(scope="module")
def classification_data():
    rng = np.random.default_rng(1)
    X = rng.random((500, 5))
    y = (X[:, 0] + rng.normal(0, 0.3, 500) > 0.5).astype(int)
    return X[:350], y[:350], X[350:], y[350:]


def _regressor(library: str, **overrides):
    if library == "lightgbm":
        params = {
            "n_estimators": 400,
            "learning_rate": 0.3,
            "early_stopping_round": PATIENCE,
            "verbose": -1,
            "random_state": 0,
        }
        return lightgbm.LGBMRegressor(**{**params, **overrides})
    if library == "xgboost":
        params = {
            "n_estimators": 400,
            "learning_rate": 0.3,
            "early_stopping_rounds": PATIENCE,
            "random_state": 0,
            "n_jobs": 1,
        }
        return xgboost.XGBRegressor(**{**params, **overrides})
    if library == "sklearn":
        params = {
            "max_iter": 400,
            "learning_rate": 0.3,
            "early_stopping": True,
            "n_iter_no_change": PATIENCE,
            "random_state": 0,
        }
        return HistGradientBoostingRegressor(**{**params, **overrides})
    params = {
        "iterations": 400,
        "learning_rate": 0.3,
        "early_stopping_rounds": PATIENCE,
        "verbose": False,
        "random_seed": 0,
    }
    return catboost.CatBoostRegressor(**{**params, **overrides})


def _classifier(library: str):
    if library == "lightgbm":
        return lightgbm.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.3,
            early_stopping_round=PATIENCE,
            metric="auc",
            verbose=-1,
            random_state=0,
            n_jobs=1,
        )
    if library == "xgboost":
        return xgboost.XGBClassifier(
            n_estimators=300,
            learning_rate=0.3,
            early_stopping_rounds=PATIENCE,
            eval_metric="auc",
            random_state=0,
            n_jobs=1,
        )
    if library == "sklearn":
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.3,
            early_stopping=True,
            n_iter_no_change=PATIENCE,
            scoring="roc_auc",
            random_state=0,
        )
    return catboost.CatBoostClassifier(
        iterations=300,
        learning_rate=0.3,
        early_stopping_rounds=PATIENCE,
        eval_metric="AUC",
        verbose=False,
        random_seed=0,
    )


ADAPTERS = {
    "lightgbm": LightGBMEarlyStoppingAdapter,
    "xgboost": XGBoostEarlyStoppingAdapter,
    "catboost": CatBoostEarlyStoppingAdapter,
    "sklearn": HistGradientBoostingEarlyStoppingAdapter,
}
LIBRARIES = list(ADAPTERS)
# scikit-learn stores a score on its validation set, the other three store a
# loss, so the regression curve's direction is per library.
CURVE_HIGHER_IS_BETTER = {"lightgbm": False, "xgboost": False, "catboost": False, "sklearn": True}
# The parameter each library spells its round ceiling with.
ROUND_PARAM = {"lightgbm": "n_estimators", "xgboost": "n_estimators", "catboost": "iterations", "sklearn": "max_iter"}


def _fit_eval(estimator, fit_params, data):
    X_train, y_train, X_eval, y_eval = data
    if isinstance(estimator, HistGradientBoostingRegressor | HistGradientBoostingClassifier):
        # scikit-learn speaks the third dialect: X_val/y_val, not eval_set.
        return estimator.fit(X_train, y_train, X_val=X_eval, y_val=y_eval, **fit_params)
    extra = {"verbose": False} if isinstance(estimator, xgboost.XGBModel) else {}
    return estimator.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], **fit_params, **extra)


def _rounds(fitted) -> int:
    if isinstance(fitted, HistGradientBoostingRegressor | HistGradientBoostingClassifier):
        return fitted.n_iter_
    if isinstance(fitted, lightgbm.LGBMModel):
        return fitted.booster_.current_iteration()
    if isinstance(fitted, xgboost.XGBModel):
        return fitted.get_booster().num_boosted_rounds()
    return fitted.tree_count_


def _explicit_truncated(fitted, X, k, method="predict"):
    if isinstance(fitted, HistGradientBoostingRegressor | HistGradientBoostingClassifier):
        # No truncated-prediction argument exists; the staged sequence is the
        # library's own answer for "the model after k rounds".
        staged = getattr(fitted, f"staged_{method}")(X)
        return next(step for i, step in enumerate(staged, start=1) if i == k)
    if isinstance(fitted, lightgbm.LGBMModel):
        return getattr(fitted, method)(X, num_iteration=k)
    if isinstance(fitted, xgboost.XGBModel):
        return getattr(fitted, method)(X, iteration_range=(0, k))
    return getattr(fitted, method)(X, ntree_end=k)


def _library_best_round(library, data) -> int:
    """The best round the library itself reports with its own early stopping."""
    estimator = _regressor(library)
    fitted = _fit_eval(estimator, {}, data)
    if library == "sklearn":
        # Entry 0 is the score before any iteration, and higher is better.
        return int(np.argmax(np.asarray(fitted.validation_score_)[1:])) + 1
    if library == "lightgbm":
        return fitted.best_iteration_
    if library == "xgboost":
        return fitted.best_iteration + 1
    return fitted.get_best_iteration() + 1


def _best_round(curve, higher_is_better) -> int:
    return int(np.argmax(curve) if higher_is_better else np.argmin(curve)) + 1


class TestResolution:
    @pytest.mark.parametrize("library", LIBRARIES)
    def test_regressors_resolve(self, library):
        assert isinstance(_resolve_early_stopping_adapter(_regressor(library)), ADAPTERS[library])

    @pytest.mark.parametrize("library", LIBRARIES)
    def test_classifiers_resolve(self, library):
        assert isinstance(_resolve_early_stopping_adapter(_classifier(library)), ADAPTERS[library])

    @pytest.mark.parametrize("library", LIBRARIES)
    def test_pipeline_final_step_resolves(self, library):
        pipeline = Pipeline([("scale", StandardScaler()), ("model", _regressor(library))])
        assert isinstance(_resolve_early_stopping_adapter(pipeline), ADAPTERS[library])

    def test_unsupported_estimator(self):
        # GradientBoostingRegressor takes no validation data at all (only a
        # monitor callback), so no adapter can reach it.
        with pytest.raises(ValueError, match=r"GradientBoostingRegressor.*early_stopping_adapter"):
            _resolve_early_stopping_adapter(GradientBoostingRegressor())

    def test_rejection_names_every_supported_library(self):
        with pytest.raises(ValueError) as excinfo:
            _resolve_early_stopping_adapter(GradientBoostingRegressor())
        message = str(excinfo.value)
        for library in ("LightGBM", "XGBoost", "CatBoost", "histogram gradient boosting"):
            assert library in message

    def test_missing_library_does_not_import(self):
        with mock.patch.dict(sys.modules, {"xgboost": None, "catboost": None}):
            assert isinstance(_resolve_early_stopping_adapter(_regressor("lightgbm")), LightGBMEarlyStoppingAdapter)

    def test_explicit_adapter_skips_builtins(self):
        custom = mock.create_autospec(BaseEarlyStoppingAdapter, instance=True)
        with mock.patch.object(
            LightGBMEarlyStoppingAdapter, "supports", side_effect=AssertionError("built-in consulted")
        ):
            assert _resolve_early_stopping_adapter(_regressor("lightgbm"), custom) is custom


class TestPreparationDoesNotMutate:
    @pytest.mark.parametrize("library", LIBRARIES)
    def test_prepare_fold_fit_and_refit(self, library):
        estimator = _regressor(library)
        before = estimator.get_params()
        adapter = ADAPTERS[library]()
        prepared, _ = adapter.prepare_fold_fit(estimator)
        refit = adapter.prepare_refit(estimator, 7)
        assert estimator.get_params() == before
        assert prepared is not estimator
        assert refit is not estimator


class TestFoldFitReachesTheCeiling:
    @pytest.mark.parametrize("library", LIBRARIES)
    def test_early_stopping_settings_do_not_shorten_the_fit(self, library, regression_data):
        assert _library_best_round(library, regression_data) + PATIENCE < 400, "the library alone must stop early"
        adapter = ADAPTERS[library]()
        prepared, fit_params = adapter.prepare_fold_fit(_regressor(library))
        fitted = _fit_eval(prepared, fit_params, regression_data)
        curve, _ = adapter.stopping_curve(fitted)
        assert _rounds(fitted) == 400
        assert len(curve) == 400

    @pytest.mark.parametrize("library", LIBRARIES)
    def test_curve_best_round_matches_library(self, library, regression_data):
        adapter = ADAPTERS[library]()
        prepared, fit_params = adapter.prepare_fold_fit(_regressor(library))
        fitted = _fit_eval(prepared, fit_params, regression_data)
        curve, higher_is_better = adapter.stopping_curve(fitted)
        library_best = _library_best_round(library, regression_data)
        assert higher_is_better is CURVE_HIGHER_IS_BETTER[library]
        assert _best_round(curve[: library_best + PATIENCE], higher_is_better) == library_best

    def test_xgboost_early_stopping_callback_keeps_its_direction(self, regression_data):
        callback = xgboost.callback.EarlyStopping(rounds=5, metric_name="rmse", maximize=True, save_best=True)
        estimator = xgboost.XGBRegressor(n_estimators=60, learning_rate=0.3, callbacks=[callback], n_jobs=1)
        adapter = XGBoostEarlyStoppingAdapter()
        adapter.validate(estimator)
        prepared, fit_params = adapter.prepare_fold_fit(estimator)
        fitted = _fit_eval(prepared, fit_params, regression_data)
        curve, higher_is_better = adapter.stopping_curve(fitted)
        assert higher_is_better is True
        assert len(curve) == _rounds(fitted) == 60
        assert estimator.get_params()["callbacks"] == [callback]

    @pytest.mark.parametrize("library", LIBRARIES)
    def test_maximized_metric_direction(self, library, classification_data):
        adapter = ADAPTERS[library]()
        prepared, fit_params = adapter.prepare_fold_fit(_classifier(library))
        fitted = _fit_eval(prepared, fit_params, classification_data)
        curve, higher_is_better = adapter.stopping_curve(fitted)
        assert higher_is_better is True
        assert _rounds(fitted) == len(curve)


class TestTruncation:
    @pytest.mark.parametrize("library", LIBRARIES)
    def test_predict_equals_library_truncation(self, library, regression_data):
        adapter = ADAPTERS[library]()
        prepared, fit_params = adapter.prepare_fold_fit(adapter.prepare_refit(_regressor(library), 30))
        fitted = _fit_eval(prepared, fit_params, regression_data)
        X_eval = regression_data[2]
        expected = _explicit_truncated(fitted, X_eval, 5)
        adapter.truncate(fitted, 5)
        np.testing.assert_array_equal(fitted.predict(X_eval), expected)

    @pytest.mark.parametrize("library", LIBRARIES)
    def test_predict_proba_equals_library_truncation(self, library, classification_data):
        adapter = ADAPTERS[library]()
        if library == "catboost":
            estimator = catboost.CatBoostClassifier(
                iterations=30, learning_rate=0.3, verbose=False, random_seed=0, thread_count=1
            )
        else:
            estimator = clone(_classifier(library)).set_params(**{ROUND_PARAM[library]: 30})
        prepared, fit_params = adapter.prepare_fold_fit(estimator)
        fitted = _fit_eval(prepared, fit_params, classification_data)
        X_eval = classification_data[2]
        expected = _explicit_truncated(fitted, X_eval, 5, method="predict_proba")
        adapter.truncate(fitted, 5)
        np.testing.assert_array_equal(fitted.predict_proba(X_eval), expected)

    @pytest.mark.parametrize("library", LIBRARIES)
    @pytest.mark.parametrize("n_rounds", [0, 10_000])
    def test_out_of_range_rejected(self, library, n_rounds, regression_data):
        adapter = ADAPTERS[library]()
        prepared, fit_params = adapter.prepare_fold_fit(_regressor(library))
        fitted = _fit_eval(prepared, fit_params, regression_data)
        with pytest.raises(ValueError, match="Cannot truncate"):
            adapter.truncate(fitted, n_rounds)


class TestRefit:
    @pytest.mark.parametrize("library", LIBRARIES)
    def test_fits_without_evaluation_set(self, library, regression_data):
        adapter = ADAPTERS[library]()
        refit = adapter.prepare_refit(_regressor(library), 15)
        X_train, y_train, _, _ = regression_data
        fitted = refit.fit(X_train, y_train)
        assert _rounds(fitted) == 15

    @pytest.mark.parametrize("library", LIBRARIES)
    def test_round_prefix_property(self, library, regression_data):
        adapter = ADAPTERS[library]()
        X_train, y_train, X_eval, _ = regression_data
        long = adapter.prepare_refit(_regressor(library), 50).fit(X_train, y_train)
        adapter.truncate(long, 20)
        short = adapter.prepare_refit(_regressor(library), 20).fit(X_train, y_train)
        np.testing.assert_allclose(long.predict(X_eval), short.predict(X_eval), rtol=0, atol=1e-12)


class TestValidate:
    def test_catboost_requires_explicit_learning_rate(self):
        with pytest.raises(ValueError, match="explicit learning_rate"):
            CatBoostEarlyStoppingAdapter().validate(catboost.CatBoostRegressor(iterations=1000, thread_count=1))
        CatBoostEarlyStoppingAdapter().validate(_regressor("catboost"))

    def test_lightgbm_dart_rejected(self):
        with pytest.raises(ValueError, match="dart"):
            LightGBMEarlyStoppingAdapter().validate(lightgbm.LGBMRegressor(boosting_type="dart"))
        LightGBMEarlyStoppingAdapter().validate(_regressor("lightgbm"))

    def test_xgboost_dart_rejected(self):
        with pytest.raises(ValueError, match="dart"):
            XGBoostEarlyStoppingAdapter().validate(xgboost.XGBRegressor(booster="dart"))
        XGBoostEarlyStoppingAdapter().validate(_regressor("xgboost"))


class TestSharedCallbackConcurrency:
    """One LightGBM callback instance serves every per-step fit of a forecaster."""

    @staticmethod
    def _series():
        from datetime import datetime, timedelta

        import polars as pl

        rng = np.random.default_rng(3)
        n = 400
        values = np.sin(np.arange(n) / 6.0) * 3 + rng.normal(0, 0.8, n)
        times = pl.datetime_range(
            datetime(2021, 1, 1), datetime(2021, 1, 1) + timedelta(seconds=n - 1), "1s", eager=True
        )
        y = pl.DataFrame({"time": times, "value": values})
        return y[:320], y[320:]

    @staticmethod
    def _forecaster(n_jobs):
        from yohou.point import PointReductionForecaster
        from yohou.preprocessing import LagTransformer

        return PointReductionForecaster(
            estimator=lightgbm.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.3,
                early_stopping_round=PATIENCE,
                min_child_samples=5,
                n_jobs=1,
                verbose=-1,
            ),
            reduction_strategy="direct",
            actual_transformer=LagTransformer(lag=[1, 2, 3, 6]),
            n_jobs=n_jobs,
        )

    def _fit(self, n_jobs, backend=None):
        import joblib

        head, window = self._series()
        forecaster = self._forecaster(n_jobs)
        adapter = LightGBMEarlyStoppingAdapter()
        prepared, fit_params = adapter.prepare_fold_fit(forecaster.estimator)
        forecaster.set_params(estimator=prepared)
        if backend is None:
            forecaster.fit(y=head, forecasting_horizon=3, y_val=window, **fit_params)
        else:
            with joblib.parallel_backend(backend):
                forecaster.fit(y=head, forecasting_horizon=3, y_val=window, **fit_params)
        return adapter, forecaster

    def _reference(self):
        """Each step's training call refitted alone, with its own callback instance."""
        calls = []
        original_fit = lightgbm.LGBMRegressor.fit

        def record(self, X, y, **kwargs):
            calls.append((X, y, kwargs["eval_set"]))
            return original_fit(self, X, y, **kwargs)

        with mock.patch.object(lightgbm.LGBMRegressor, "fit", record):
            self._fit(n_jobs=1)
        adapter = LightGBMEarlyStoppingAdapter()
        reference = []
        for X, y, eval_set in calls:
            prepared, fit_params = adapter.prepare_fold_fit(self._forecaster(1).estimator)
            fitted = prepared.fit(X, y, eval_set=eval_set, **fit_params)
            curve, higher_is_better = adapter.stopping_curve(fitted)
            reference.append((curve.tolist(), higher_is_better, fitted.booster_.current_iteration()))
        return reference

    @pytest.mark.parametrize(
        ("n_jobs", "backend"), [(1, None), (3, None), (3, "threading")], ids=["sequential", "processes", "threads"]
    )
    def test_each_step_matches_its_own_callback(self, n_jobs, backend):
        reference = self._reference()
        adapter, forecaster = self._fit(n_jobs, backend)
        observed = []
        for _, est in forecaster._fitted_estimator_positions():
            curve, higher_is_better = adapter.stopping_curve(est)
            observed.append((curve.tolist(), higher_is_better, est.booster_.current_iteration()))
        assert observed == reference
        assert all(len(curve) == trees == 300 for curve, _, trees in observed)


class TestAdapterErrorPaths:
    """Adapters fail with a named cause when a model cannot give what they need."""

    def test_lightgbm_recorder_needs_an_evaluation_set(self):
        from types import SimpleNamespace

        from yohou.model_selection.early_stopping import _StoppingMetricRecorder

        env = SimpleNamespace(
            iteration=0, begin_iteration=0, model=object(), evaluation_result_list=[("training", "l2", 1.0, False)]
        )
        with pytest.raises(ValueError, match="evaluation set other than the training data"):
            _StoppingMetricRecorder()(env)

    def test_lightgbm_curve_needs_the_recorder(self, regression_data):
        X_train, y_train, X_eval, y_eval = regression_data
        model = lightgbm.LGBMRegressor(n_estimators=10, verbose=-1, n_jobs=1).fit(
            X_train, y_train, eval_set=[(X_eval, y_eval)]
        )
        with pytest.raises(ValueError, match="not fitted with the callbacks"):
            LightGBMEarlyStoppingAdapter().stopping_curve(model)

    def test_catboost_curve_needs_an_evaluation_set(self, regression_data):
        X_train, y_train, _, _ = regression_data
        model = catboost.CatBoostRegressor(iterations=10, learning_rate=0.3, verbose=False, thread_count=1).fit(
            X_train, y_train
        )
        with pytest.raises(ValueError, match="without an evaluation set"):
            CatBoostEarlyStoppingAdapter().stopping_curve(model)

    def test_catboost_curve_needs_the_metric(self, regression_data):
        adapter = CatBoostEarlyStoppingAdapter()
        prepared, fit_params = adapter.prepare_fold_fit(
            catboost.CatBoostRegressor(iterations=10, learning_rate=0.3, verbose=False, thread_count=1)
        )
        model = _fit_eval(prepared, fit_params, regression_data)
        with (
            mock.patch.object(type(model), "get_all_params", return_value={"eval_metric": "Poisson"}),
            pytest.raises(ValueError, match="no validation values for 'Poisson'"),
        ):
            adapter.stopping_curve(model)

    def test_catboost_truncate_to_all_rounds_keeps_the_model(self, regression_data):
        adapter = CatBoostEarlyStoppingAdapter()
        prepared, fit_params = adapter.prepare_fold_fit(
            catboost.CatBoostRegressor(iterations=12, learning_rate=0.3, verbose=False, thread_count=1)
        )
        model = _fit_eval(prepared, fit_params, regression_data)
        before = model.predict(regression_data[2])
        adapter.truncate(model, 12)
        assert model.tree_count_ == 12
        np.testing.assert_array_equal(model.predict(regression_data[2]), before)

    def test_xgboost_callbacks_ignored_when_library_not_loaded(self):
        estimator = xgboost.XGBRegressor(callbacks=[xgboost.callback.EarlyStopping(rounds=5)])
        with mock.patch.dict(sys.modules, {"xgboost": None}):
            assert XGBoostEarlyStoppingAdapter._early_stopping_callbacks(estimator) == []

    def test_xgboost_refit_removes_only_the_early_stopping_callback(self, regression_data):
        monitor = xgboost.callback.EvaluationMonitor(period=1000)
        estimator = xgboost.XGBRegressor(
            n_estimators=200,
            learning_rate=0.3,
            n_jobs=1,
            callbacks=[xgboost.callback.EarlyStopping(rounds=5), monitor],
        )
        refit = XGBoostEarlyStoppingAdapter().prepare_refit(estimator, 15)
        kept = refit.get_params()["callbacks"]
        assert len(kept) == 1 and isinstance(kept[0], xgboost.callback.EvaluationMonitor)
        X_train, y_train, _, _ = regression_data
        assert refit.fit(X_train, y_train).get_booster().num_boosted_rounds() == 15

    def test_xgboost_refit_with_only_early_stopping_clears_callbacks(self):
        estimator = xgboost.XGBRegressor(callbacks=[xgboost.callback.EarlyStopping(rounds=5)])
        assert XGBoostEarlyStoppingAdapter().prepare_refit(estimator, 15).get_params()["callbacks"] is None


class TestHistGradientBoostingSpecifics:
    """Behaviour particular to the scikit-learn adapter."""

    def test_truncation_survives_pickling(self, regression_data):
        adapter = HistGradientBoostingEarlyStoppingAdapter()
        prepared, fit_params = adapter.prepare_fold_fit(_regressor("sklearn", max_iter=40))
        fitted = _fit_eval(prepared, fit_params, regression_data)
        X_eval = regression_data[2]
        adapter.truncate(fitted, 12)
        before = fitted.predict(X_eval)
        restored = pickle.loads(pickle.dumps(fitted))
        np.testing.assert_array_equal(restored.predict(X_eval), before)
        assert restored.n_iter_ == 12

    def test_selected_round_follows_the_reported_direction(self):
        """A wrong direction would pick the opposite end of the curve.

        The adapter reports higher-is-better, so the shared-round selection must
        take the curve's maximum. This pins the orientation itself: nothing else
        in the suite fails if it is reported backwards.
        """
        # Best value at round 3 (index 2) on every fold.
        curve = np.array([0.1, 0.5, 0.9, 0.4, 0.2])
        rounds, at_boundary = _select_shared_rounds({"pos": [(curve, True), (curve, True)]})
        assert rounds["pos"] == 3
        assert at_boundary["pos"] is False
        # The same curve read as a loss would choose round 1, so the two
        # directions are genuinely distinguishable on this input.
        flipped, _ = _select_shared_rounds({"pos": [(curve, False), (curve, False)]})
        assert flipped["pos"] == 1

    def test_fold_fit_reports_higher_is_better(self, regression_data):
        adapter = HistGradientBoostingEarlyStoppingAdapter()
        prepared, fit_params = adapter.prepare_fold_fit(_regressor("sklearn", max_iter=30))
        fitted = _fit_eval(prepared, fit_params, regression_data)
        _, higher_is_better = adapter.stopping_curve(fitted)
        assert higher_is_better is True

    def test_curve_without_an_evaluation_set_is_rejected(self, regression_data):
        """With early_stopping='auto' below the threshold, no curve is recorded."""
        X_train, y_train, X_eval, y_eval = regression_data
        estimator = HistGradientBoostingRegressor(max_iter=20, early_stopping="auto")
        estimator.fit(X_train, y_train, X_val=X_eval, y_val=y_eval)
        with pytest.raises(ValueError, match="no validation curve"):
            HistGradientBoostingEarlyStoppingAdapter().stopping_curve(estimator)
