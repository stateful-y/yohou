"""Tests for ``validation="cv"`` on GridSearchCV and RandomizedSearchCV."""

import warnings
from datetime import datetime, timedelta
from unittest import mock

import lightgbm
import numpy as np
import polars as pl
import pytest
from sklearn.base import clone

from yohou.class_proba import ClassProbaReductionForecaster
from yohou.compose import DecompositionPipeline
from yohou.interval import IntervalReductionForecaster
from yohou.metrics import IntervalScore, LogLoss, MeanAbsoluteError, RootMeanSquaredError
from yohou.model_selection import ExpandingWindowSplitter, GridSearchCV, RandomizedSearchCV
from yohou.model_selection import utils as ms_utils
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer, MinMaxScaler

from .shared_round_stubs import (
    CallbackCurveEarlyStoppingAdapter,
    CurveEarlyStoppingAdapter,
    CurveRegressor,
    QuantileCurveRegressor,
)

N_SPLITS = 3
TEST_SIZE = 12
HORIZON = 3


def _series(n: int = 132, panel: bool = False) -> pl.DataFrame:
    rng = np.random.default_rng(11)
    times = pl.datetime_range(datetime(2023, 1, 1), datetime(2023, 1, 1) + timedelta(hours=n - 1), "1h", eager=True)
    level = 10.0 + np.arange(n) * 0.2
    if panel:
        return pl.DataFrame({
            "time": times,
            "a__value": level + rng.normal(0, 0.3, n),
            "b__value": 1000.0 + level + rng.normal(0, 0.3, n),
        })
    return pl.DataFrame({"time": times, "value": level + rng.normal(0, 0.3, n)})


def _cv():
    return ExpandingWindowSplitter(n_splits=N_SPLITS, test_size=TEST_SIZE)


def _point(estimator=None, **kwargs):
    return PointReductionForecaster(
        estimator=estimator if estimator is not None else CurveRegressor(),
        reduction_strategy=kwargs.pop("reduction_strategy", "direct"),
        actual_transformer=kwargs.pop("actual_transformer", LagTransformer(lag=[1, 2])),
        **kwargs,
    )


def _search(forecaster=None, param_grid=None, **kwargs):
    return GridSearchCV(
        forecaster=forecaster if forecaster is not None else _point(),
        param_grid=param_grid if param_grid is not None else {"estimator__patience": [4, 8]},
        scoring=kwargs.pop("scoring", MeanAbsoluteError()),
        cv=_cv(),
        validation=kwargs.pop("validation", "cv"),
        early_stopping_adapter=kwargs.pop("early_stopping_adapter", CurveEarlyStoppingAdapter()),
        **kwargs,
    )


def _captured_folds(search, y, **fit_kwargs):
    folds = []
    original = ms_utils._score_fold

    def capture(fold, **kwargs):
        folds.append(fold)
        return original(fold, **kwargs)

    with mock.patch.object(ms_utils, "_score_fold", capture):
        search.fit(y, forecasting_horizon=HORIZON, **fit_kwargs)
    return folds


class TestDefaultMode:
    def test_no_round_columns_without_validation(self):
        y = _series()
        search = _search(validation=None, early_stopping_adapter=None, forecaster=_point(estimator=CurveRegressor()))
        search.fit(y, forecasting_horizon=HORIZON)
        assert not {"rounds", "rounds_at_boundary"} & set(search.cv_results_)
        assert not any(key.endswith("_curve_length") for key in search.cv_results_)
        assert not hasattr(search, "best_rounds_")

    def test_parameters_round_trip(self):
        adapter = CurveEarlyStoppingAdapter()
        for search in (
            _search(early_stopping_adapter=adapter),
            RandomizedSearchCV(
                _point(),
                {"estimator__patience": [4, 8]},
                n_iter=1,
                validation="cv",
                early_stopping_adapter=adapter,
            ),
        ):
            params = search.get_params()
            assert params["validation"] == "cv"
            assert params["early_stopping_adapter"] is adapter
            assert clone(search).get_params()["validation"] == "cv"


class TestFoldWindows:
    def test_evaluation_targets_are_the_scored_fold(self):
        y = _series()
        search = _search(refit=False)
        folds = _captured_folds(search, y)
        splits = list(_cv().split(y))
        assert len(folds) == 2 * N_SPLITS
        for idx, fold in enumerate(folds):
            _, test = splits[idx % N_SPLITS]
            pl.testing.assert_frame_equal(fold.y_test, y[test])
            window = set(y["value"].to_numpy()[test].tolist())
            for _, est in fold.forecaster._fitted_estimator_positions():
                assert set(est.received_eval_targets_.tolist()) <= window

    def test_transformers_see_only_the_training_window(self):
        y = _series()
        forecaster = _point(target_transformer=MinMaxScaler())
        folds = _captured_folds(_search(forecaster=forecaster, refit=False, param_grid={"estimator__patience": [6]}), y)
        for fold in folds:
            plain = clone(forecaster).fit(y=fold.y_train, forecasting_horizon=HORIZON)
            for (_, est), plain_est in zip(
                fold.forecaster._fitted_estimator_positions(), plain.estimator_, strict=True
            ):
                assert est.train_mean_ == plain_est.train_mean_

    def test_panel_groups_evaluate_on_their_own_windows(self):
        y = _series(panel=True)
        folds = _captured_folds(_search(refit=False, param_grid={"estimator__patience": [6]}), y)
        splits = list(_cv().split(y))
        for fold, (train, test) in zip(folds, splits, strict=True):
            window = set(y["a__value"].to_numpy()[test].tolist()) | set(y["b__value"].to_numpy()[test].tolist())
            history = set(y["a__value"].to_numpy()[train].tolist()) | set(y["b__value"].to_numpy()[train].tolist())
            for _, est in fold.forecaster._fitted_estimator_positions():
                targets = set(est.received_eval_targets_.tolist())
                assert targets <= window
                assert not targets & history


class TestSharedRounds:
    def test_rounds_follow_the_fold_average_curve(self):
        y = _series()
        search = _search(refit=False, param_grid={"estimator__patience": [6]})
        folds = _captured_folds(search, y)
        rounds = search.cv_results_["rounds"][0]
        assert list(rounds) == ["step_1", "step_2", "step_3"]
        for position in rounds:
            curves = [dict(f.forecaster._fitted_estimator_positions())[position].curve_ for f in folds]
            shortest = min(len(c) for c in curves)
            mean = np.mean([c[:shortest] for c in curves], axis=0)
            assert rounds[position] == int(np.argmin(mean)) + 1
            assert all(
                dict(f.forecaster._fitted_estimator_positions())[position].rounds_used_ == rounds[position]
                for f in folds
            )
        assert len({len(dict(f.forecaster._fitted_estimator_positions())["step_1"].curve_) for f in folds}) > 1

    @staticmethod
    def _evaluate_interval(strategy, forecasting_horizon):
        y = _series()
        forecaster = IntervalReductionForecaster(
            estimator=QuantileCurveRegressor(patience=6),
            reduction_strategy=strategy,
            actual_transformer=LagTransformer(lag=[1, 2]),
        )
        return ms_utils._evaluate_candidate_shared_rounds(
            forecaster,
            y,
            None,
            forecasting_horizon,
            splits=list(_cv().split(y)),
            parameters=None,
            early_stopping_adapter=CurveEarlyStoppingAdapter(),
            scorer=IntervalScore(coverage_rates=[0.9]),
            verbose=0,
            fit_params={},
            predict_func_params={},
            score_params={},
            coverage_rates=[0.9],
        )

    def test_interval_bounds_multi_output(self):
        results, record = self._evaluate_interval("multi-output", HORIZON)
        assert list(record["rounds"]) == ["coverage_rate_0.9_lower", "coverage_rate_0.9_upper"]
        assert record["rounds"]["coverage_rate_0.9_lower"] != record["rounds"]["coverage_rate_0.9_upper"]
        assert all(np.isfinite(result["test_scores"]) for result in results)

    def test_interval_bounds_with_per_step_estimators(self):
        _, record = self._evaluate_interval("direct", 2)
        assert list(record["rounds"]) == [
            "coverage_rate_0.9_lower/step_1",
            "coverage_rate_0.9_lower/step_2",
            "coverage_rate_0.9_upper/step_1",
            "coverage_rate_0.9_upper/step_2",
        ]

    def test_round_at_shortest_curve_is_flagged_and_warned(self):
        y = _series()
        search = _search(
            refit=False,
            forecaster=_point(estimator=CurveRegressor(n_rounds=8)),
            param_grid={"estimator__patience": [6]},
        )
        with pytest.warns(UserWarning, match=r"'step_1'.*last round every fold trained"):
            search.fit(y, forecasting_horizon=HORIZON)
        assert list(search.cv_results_["rounds_at_boundary"]) == [True]

    def test_boundary_warning_points_at_the_caller(self):
        y = _series()
        search = _search(
            refit=False,
            forecaster=_point(estimator=CurveRegressor(n_rounds=8)),
            param_grid={"estimator__patience": [6]},
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            search.fit(y, forecasting_horizon=HORIZON)
        boundary = [w for w in caught if "last round every fold trained" in str(w.message)]
        assert len(boundary) == 1
        assert boundary[0].filename == __file__

    def test_train_scores_of_the_truncated_folds(self):
        y = _series()
        search = _search(param_grid={"estimator__patience": [4, 8]}, return_train_score=True, refit=False)
        search.fit(y, forecasting_horizon=HORIZON)
        results = search.cv_results_
        assert np.isfinite(results["mean_train_score"]).all()
        for i in range(N_SPLITS):
            assert np.isfinite(results[f"split{i}_train_score"]).all()
        assert len(results["rounds"]) == len(results["rounds_at_boundary"]) == 2

    def test_round_record_is_consistent(self):
        y = _series()
        search = _search(param_grid={"estimator__patience": [4, 8]})
        search.fit(y, forecasting_horizon=2)
        results = search.cv_results_
        for idx, rounds in enumerate(results["rounds"]):
            assert list(rounds) == ["step_1", "step_2"]
            for position, chosen in rounds.items():
                assert chosen <= min(results[f"split{i}_curve_length"][idx][position] for i in range(N_SPLITS))
        assert search.best_rounds_ == results["rounds"][search.best_index_]

    def test_verbose_progress_lines(self, capsys):
        y = _series()
        _search(refit=False, verbose=3, validation=None, early_stopping_adapter=None).fit(
            y, forecasting_horizon=HORIZON
        )
        default_lines = [line for line in capsys.readouterr().out.splitlines() if " END " in line]
        _search(refit=False, verbose=3).fit(y, forecasting_horizon=HORIZON)
        cv_lines = [line for line in capsys.readouterr().out.splitlines() if " END " in line]
        assert len(cv_lines) == len(default_lines) == 2 * N_SPLITS
        assert [line.split(" END ")[0] for line in cv_lines] == [line.split(" END ")[0] for line in default_lines]


class TestMultimetric:
    @staticmethod
    def _search(refit):
        return _search(
            scoring={"mae": MeanAbsoluteError(), "rmse": RootMeanSquaredError()},
            refit=refit,
            param_grid={"estimator__patience": [4, 8]},
        )

    def test_refit_metric_sets_best_rounds(self):
        search = self._search(refit="mae")
        search.fit(_series(), forecasting_horizon=HORIZON)
        assert search.best_rounds_ == search.cv_results_["rounds"][search.best_index_]
        assert {"mean_test_mae", "mean_test_rmse", "rounds", "rounds_at_boundary"} <= set(search.cv_results_)

    def test_refit_false_leaves_best_rounds_unset(self):
        search = self._search(refit=False)
        search.fit(_series(), forecasting_horizon=HORIZON)
        assert not hasattr(search, "best_index_")
        assert not hasattr(search, "best_rounds_")
        results = search.cv_results_
        assert len(results["rounds"]) == 2
        assert len(results["rounds_at_boundary"]) == 2
        for i in range(N_SPLITS):
            assert [set(lengths) for lengths in results[f"split{i}_curve_length"]] == [
                {"step_1", "step_2", "step_3"}
            ] * 2


class TestAdapterFitParams:
    def test_caller_fit_params_merge_with_the_adapters(self):
        y = _series()
        forecaster = _point(estimator=CurveRegressor(patience=6).set_fit_request(callbacks=True))
        search = _search(
            forecaster=forecaster,
            early_stopping_adapter=CallbackCurveEarlyStoppingAdapter(),
            param_grid={"estimator__patience": [6]},
            refit=False,
        )
        folds = _captured_folds(search, y, callbacks=["caller"])
        assert len(folds) == N_SPLITS
        for fold in folds:
            for _, est in fold.forecaster._fitted_estimator_positions():
                assert est.received_callbacks_ == ["caller", "adapter"]


class TestFailedFolds:
    def test_one_failing_fold(self):
        y = _series()
        smallest_train = len(next(iter(_cv().split(y)))[0]) - HORIZON
        forecaster = _point(estimator=CurveRegressor(fail_below_train_rows=smallest_train + 1))
        search = _search(forecaster=forecaster, param_grid={"estimator__patience": [6]}, refit=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(y, forecasting_horizon=HORIZON)
        results = search.cv_results_
        assert np.isnan(results["split0_test_score"][0])
        assert np.isfinite(results["split1_test_score"][0]) and np.isfinite(results["split2_test_score"][0])
        assert results["split0_curve_length"][0] is None
        assert set(results["rounds"][0]) == {"step_1", "step_2", "step_3"}


class TestRefit:
    def test_refit_uses_chosen_rounds_without_evaluation_set(self):
        y = _series()
        adapter = CurveEarlyStoppingAdapter()
        search = _search(early_stopping_adapter=adapter, param_grid={"estimator__patience": [6]})
        fit_calls = []
        original_fit = PointReductionForecaster.fit

        def record(self, *args, **kwargs):
            fit_calls.append(kwargs)
            return original_fit(self, *args, **kwargs)

        with mock.patch.object(PointReductionForecaster, "fit", record):
            search.fit(y, forecasting_horizon=HORIZON)
        refit_kwargs = fit_calls[-1]
        assert "y_val" not in refit_kwargs
        best = search.best_forecaster_
        assert best.estimator.n_rounds == max(search.best_rounds_.values())
        for position, est in best._fitted_estimator_positions():
            assert est.received_eval_targets_ is None
            assert est.rounds_trained_ == max(search.best_rounds_.values())
            assert est.rounds_used_ == search.best_rounds_[position]

    def test_lightgbm_refit_with_early_stopping_configured(self):
        y = _series(n=200)
        forecaster = _point(
            estimator=lightgbm.LGBMRegressor(
                n_estimators=300, learning_rate=0.3, early_stopping_round=5, min_child_samples=5, n_jobs=1, verbose=-1
            ),
            actual_transformer=LagTransformer(lag=[1, 2, 3]),
        )
        search = GridSearchCV(
            forecaster, {"estimator__num_leaves": [7, 15]}, scoring=MeanAbsoluteError(), cv=_cv(), validation="cv"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            search.fit(y, forecasting_horizon=HORIZON)
        for position, est in search.best_forecaster_._fitted_estimator_positions():
            assert est.booster_.current_iteration() == max(search.best_rounds_.values())
            X = np.zeros((1, est.n_features_in_))
            np.testing.assert_array_equal(est.predict(X), est.predict(X, num_iteration=search.best_rounds_[position]))


class TestEndToEnd:
    def test_class_proba_lightgbm_with_unseen_class_fold(self):
        n = 150
        rng = np.random.default_rng(2)
        times = pl.datetime_range(datetime(2023, 1, 1), datetime(2023, 1, 1) + timedelta(hours=n - 1), "1h", eager=True)
        states = [["low", "high"][int(v)] for v in rng.integers(0, 2, n)]
        splits = list(_cv().split(pl.DataFrame({"time": times, "state": states})))
        first_test = splits[0][1]
        # "rare" appears for the first time inside the first test window.
        states[int(first_test[3])] = "rare"
        y = pl.DataFrame({"time": times, "state": states})
        forecaster = ClassProbaReductionForecaster(
            estimator=lightgbm.LGBMClassifier(
                n_estimators=200, learning_rate=0.2, early_stopping_round=5, min_child_samples=3, n_jobs=1, verbose=-1
            ),
            reduction_strategy="direct",
            actual_transformer=LagTransformer(lag=[1, 2]),
        )
        search = GridSearchCV(forecaster, {"estimator__num_leaves": [7]}, scoring=LogLoss(), cv=_cv(), validation="cv")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(y, forecasting_horizon=1)
        results = search.cv_results_
        assert np.isnan(results["split0_test_score"][0])
        assert results["split0_curve_length"][0] is None
        assert np.isfinite(results["split1_test_score"][0])
        best = search.best_forecaster_
        [(position, est)] = best._fitted_estimator_positions()
        X = np.zeros((1, est.n_features_in_))
        np.testing.assert_array_equal(
            est.predict_proba(X), est.predict_proba(X, num_iteration=search.best_rounds_[position])
        )

    def test_class_proba_unseen_class_error_names_the_class(self):
        n = 150
        times = pl.datetime_range(datetime(2023, 1, 1), datetime(2023, 1, 1) + timedelta(hours=n - 1), "1h", eager=True)
        states = [["low", "high"][i % 2] for i in range(n)]
        splits = list(_cv().split(pl.DataFrame({"time": times, "state": states})))
        states[int(splits[0][1][3])] = "rare"
        y = pl.DataFrame({"time": times, "state": states})
        forecaster = ClassProbaReductionForecaster(
            estimator=lightgbm.LGBMClassifier(n_estimators=20, min_child_samples=3, n_jobs=1, verbose=-1),
            reduction_strategy="direct",
        )
        folds = _captured_folds(
            GridSearchCV(
                forecaster, {"estimator__num_leaves": [7]}, scoring=LogLoss(), cv=_cv(), validation="cv", refit=False
            ),
            y,
        )
        assert "['rare']" in folds[0].fit_error and "y_val window" in folds[0].fit_error

    def test_search_parallelism_does_not_change_results(self):
        y = _series(n=200)
        forecaster = _point(
            estimator=lightgbm.LGBMRegressor(
                n_estimators=200, learning_rate=0.3, early_stopping_round=5, min_child_samples=5, n_jobs=1, verbose=-1
            ),
        )
        outputs = []
        for n_jobs in (1, 2):
            search = GridSearchCV(
                forecaster,
                {"estimator__num_leaves": [7, 15]},
                scoring=MeanAbsoluteError(),
                cv=_cv(),
                validation="cv",
                n_jobs=n_jobs,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                search.fit(y, forecasting_horizon=HORIZON)
            outputs.append(search)
        sequential, parallel = outputs
        for key in ["mean_test_score", *[f"split{i}_test_score" for i in range(N_SPLITS)]]:
            np.testing.assert_array_equal(sequential.cv_results_[key], parallel.cv_results_[key])
        assert list(sequential.cv_results_["rounds"]) == list(parallel.cv_results_["rounds"])
        assert sequential.best_rounds_ == parallel.best_rounds_


class TestRejectedConfigurations:
    def test_composite_forecaster(self):
        y = _series()
        search = _search(forecaster=DecompositionPipeline(forecasters=[("trend", _point())]), param_grid={})
        with (
            mock.patch.object(ms_utils, "_fit_fold") as fit_fold,
            pytest.raises(ValueError, match="requires a reduction forecaster"),
        ):
            search.fit(y, forecasting_horizon=HORIZON)
        fit_fold.assert_not_called()

    def test_dir_rec_in_grid(self):
        y = _series()
        search = _search(param_grid={"reduction_strategy": ["direct", "dir-rec"]})
        with pytest.raises(ValueError, match="earlier steps' predictions"):
            search.fit(y, forecasting_horizon=HORIZON)

    @pytest.mark.parametrize(
        ("template", "grid"),
        [
            ({"reduction_strategy": "dir-rec"}, {"reduction_strategy": ["direct"]}),
            ({"validation_size": 48}, {"validation_size": [None]}),
        ],
        ids=["dir-rec", "validation_size"],
    )
    def test_grid_overrides_template_configuration(self, template, grid):
        y = _series()
        search = _search(forecaster=_point(**template), param_grid=grid)
        search.fit(y, forecasting_horizon=HORIZON)
        assert search.best_rounds_

    def test_validation_size_on_forecaster(self):
        y = _series()
        search = _search(forecaster=_point(validation_size=48))
        with (
            mock.patch.object(ms_utils, "_fit_fold") as fit_fold,
            pytest.raises(ValueError, match="validation_size=48"),
        ):
            search.fit(y, forecasting_horizon=HORIZON)
        fit_fold.assert_not_called()

    @pytest.mark.parametrize("key", ["eval_set", "y_val"])
    def test_evaluation_data_in_fit_params(self, key):
        with pytest.raises(ValueError, match=key):
            _search()._check_shared_round_setup({key: object()})

    def test_lightgbm_several_fit_time_metrics(self):
        y = _series()
        estimator = lightgbm.LGBMRegressor(n_estimators=20, n_jobs=1, verbose=-1).set_fit_request(eval_metric=True)
        search = _search(forecaster=_point(estimator=estimator), early_stopping_adapter=None, param_grid={})
        with (
            mock.patch.object(ms_utils, "_fit_fold") as fit_fold,
            pytest.raises(ValueError, match="first_metric_only"),
        ):
            search.fit(y, forecasting_horizon=HORIZON, eval_metric=["l1", "l2"])
        fit_fold.assert_not_called()

    def test_no_adapter_for_estimator(self):
        y = _series()
        search = _search(forecaster=_point(estimator=CurveRegressor()), early_stopping_adapter=None)
        with pytest.raises(ValueError, match="no early-stopping adapter for CurveRegressor"):
            search.fit(y, forecasting_horizon=HORIZON)

    def test_catboost_default_learning_rate(self):
        catboost = pytest.importorskip("catboost")
        y = _series()
        search = _search(
            forecaster=_point(estimator=catboost.CatBoostRegressor(iterations=100, verbose=False, thread_count=1)),
            early_stopping_adapter=None,
            param_grid={"estimator__depth": [3]},
        )
        with pytest.raises(ValueError, match="explicit learning_rate"):
            search.fit(y, forecasting_horizon=HORIZON)


class TestRandomizedSearch:
    """Shared rounds are recorded per sampled candidate, not per grid point."""

    def test_each_sampled_candidate_records_its_own_rounds(self):
        search = RandomizedSearchCV(
            _point(estimator=CurveRegressor(patience=6)),
            {"estimator__n_rounds": [4, 6, 8, 10]},
            n_iter=3,
            random_state=0,
            scoring=MeanAbsoluteError(),
            cv=_cv(),
            validation="cv",
            early_stopping_adapter=CurveEarlyStoppingAdapter(),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            search.fit(_series(), forecasting_horizon=HORIZON)
        results = search.cv_results_
        assert len(results["params"]) == len(results["rounds"]) == 3
        for params, rounds in zip(results["params"], results["rounds"], strict=True):
            assert list(rounds) == ["step_1", "step_2", "step_3"]
            # Every curve still improves at these ceilings, so each candidate
            # is cut at its own sampled ceiling.
            assert set(rounds.values()) == {params["estimator__n_rounds"]}
        assert len({tuple(rounds.values()) for rounds in results["rounds"]}) == 3
        assert search.best_rounds_ == results["rounds"][search.best_index_]


class TestSystematicChecks:
    """The generic search checks hold for a ``validation="cv"`` search."""

    @pytest.mark.slow
    def test_systematic_search_checks(self, y_X_factory):
        from conftest import run_checks
        from yohou.testing import _yield_yohou_search_checks

        y, X_actual, X_future, X_forecast = y_X_factory(
            length=200,
            n_targets=1,
            n_features=2,
            seed=42,
            n_future_features=2,
            n_forecast_features=2,
            return_exogenous=True,
        )
        y_train, y_test = y[:180], y[180:]
        X_actual_train, X_actual_test = X_actual[:180], X_actual[180:]
        search = GridSearchCV(
            forecaster=_point(estimator=CurveRegressor(patience=6)),
            param_grid={"estimator__patience": [4, 8]},
            scoring=MeanAbsoluteError(),
            cv=2,
            validation="cv",
            early_stopping_adapter=CurveEarlyStoppingAdapter(),
        )
        fitted = clone(search)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fitted.fit(y_train, X_actual_train, forecasting_horizon=3, X_future=X_future, X_forecast=X_forecast)
            run_checks(
                fitted,
                _yield_yohou_search_checks(
                    fitted,
                    y_train,
                    X_actual_train,
                    y_test,
                    X_actual_test,
                    tags={"search_type": "grid", "refit": True, "multimetric": False},
                    X_future_train=X_future,
                    X_future_test=X_future,
                    X_forecast_train=X_forecast,
                    X_forecast_test=X_forecast,
                ),
            )


class TestBuiltinFoldFitsReachTheCeiling:
    def test_lightgbm_curves_cover_every_round(self):
        y = _series(n=200)
        forecaster = _point(
            estimator=lightgbm.LGBMRegressor(
                n_estimators=80, learning_rate=0.3, early_stopping_round=3, min_child_samples=5, n_jobs=1, verbose=-1
            ),
        )
        search = GridSearchCV(
            forecaster,
            {"estimator__num_leaves": [7]},
            scoring=MeanAbsoluteError(),
            cv=_cv(),
            validation="cv",
            refit=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            search.fit(y, forecasting_horizon=HORIZON)
        for i in range(N_SPLITS):
            assert set(search.cv_results_[f"split{i}_curve_length"][0].values()) == {80}


class TestIntervalSearches:
    """validation="cv" reaches interval-only forecasters through the searches."""

    @staticmethod
    def _search(strategy, **kwargs):
        return GridSearchCV(
            IntervalReductionForecaster(
                estimator=QuantileCurveRegressor(patience=6),
                reduction_strategy=strategy,
                actual_transformer=LagTransformer(lag=[1, 2]),
            ),
            {"estimator__n_rounds": [60]},
            scoring=IntervalScore(coverage_rates=[0.9]),
            cv=_cv(),
            validation="cv",
            early_stopping_adapter=CurveEarlyStoppingAdapter(),
            **kwargs,
        )

    def test_multi_output_bounds(self):
        search = self._search("multi-output")
        search.fit(_series(), forecasting_horizon=HORIZON)
        assert list(search.best_rounds_) == ["coverage_rate_0.9_lower", "coverage_rate_0.9_upper"]
        assert search.best_rounds_["coverage_rate_0.9_lower"] != search.best_rounds_["coverage_rate_0.9_upper"]
        assert len(search.predict_interval(coverage_rates=[0.9])) == HORIZON

    def test_direct_bounds_and_steps_refit_cut(self):
        search = self._search("direct")
        search.fit(_series(), forecasting_horizon=2)
        assert list(search.cv_results_["rounds"][0]) == [
            "coverage_rate_0.9_lower/step_1",
            "coverage_rate_0.9_lower/step_2",
            "coverage_rate_0.9_upper/step_1",
            "coverage_rate_0.9_upper/step_2",
        ]
        assert search.best_rounds_ == search.cv_results_["rounds"][0]
        for position, est in search.best_forecaster_._fitted_estimator_positions():
            assert est.rounds_used_ == search.best_rounds_[position]
            assert est.received_eval_targets_ is None
        assert len(search.predict_interval(coverage_rates=[0.9])) == 2

    def test_boundary_warning_names_an_interval_bound(self):
        search = self._search("multi-output", refit=False)
        search.set_params(param_grid={"estimator__n_rounds": [8]})
        with pytest.warns(UserWarning, match=r"coverage_rate_0\.9_lower.*last round every fold trained"):
            search.fit(_series(), forecasting_horizon=HORIZON)
        assert list(search.cv_results_["rounds_at_boundary"]) == [True]

    def test_lightgbm_quantile_end_to_end(self):
        y = _series(n=200)
        forecaster = IntervalReductionForecaster(
            estimator=lightgbm.LGBMRegressor(
                objective="quantile",
                alpha=0.5,
                n_estimators=60,
                learning_rate=0.2,
                min_child_samples=5,
                n_jobs=1,
                verbose=-1,
            ),
            reduction_strategy="direct",
            actual_transformer=LagTransformer(lag=[1, 2, 3]),
        )
        search = GridSearchCV(
            forecaster,
            {"estimator__num_leaves": [7]},
            scoring=IntervalScore(coverage_rates=[0.9]),
            cv=_cv(),
            validation="cv",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            search.fit(y, forecasting_horizon=2)
        for i in range(N_SPLITS):
            assert set(search.cv_results_[f"split{i}_curve_length"][0].values()) == {60}
        X = np.zeros((1, 3))
        for position, est in search.best_forecaster_._fitted_estimator_positions():
            X = np.zeros((1, est.n_features_in_))
            np.testing.assert_array_equal(est.predict(X), est.predict(X, num_iteration=search.best_rounds_[position]))
        assert len(search.predict_interval(coverage_rates=[0.9])) == 2


class TestPipelineEstimator:
    """The adapter applies to a Pipeline's final step, in fold fits and in the refit."""

    def test_pipeline_final_step_in_cv_mode(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        forecaster = _point(estimator=Pipeline([("scale", StandardScaler()), ("model", CurveRegressor(patience=6))]))
        search = _search(forecaster=forecaster, param_grid={"estimator__model__n_rounds": [60]})
        folds = _captured_folds(search, _series())
        for fold in folds:
            for _, est in fold.forecaster._fitted_estimator_positions():
                assert est.received_eval_targets_ is not None
        best = search.best_forecaster_
        assert isinstance(best.estimator, Pipeline)
        assert best.estimator.named_steps["model"].n_rounds == max(search.best_rounds_.values())
        for position, est in best._fitted_estimator_positions():
            assert est.rounds_used_ == search.best_rounds_[position]


class TestRefitWithoutRounds:
    def test_refit_needs_chosen_rounds(self):
        search = _search()
        search.best_rounds_ = {}
        with pytest.raises(ValueError, match="cannot refit: no fold of the best candidate fitted successfully"):
            search._prepare_shared_round_refit(_point())

    def test_fit_refuses_refit_when_best_candidate_has_no_rounds(self):
        """A failing candidate ranked best by a high error_score reaches the refit error through fit."""
        search = _search(param_grid={"estimator__fail_below_train_rows": [0, 10_000]}, error_score=1e9)
        with pytest.raises(ValueError, match="cannot refit: no fold of the best candidate fitted successfully"):
            search.fit(_series(), forecasting_horizon=HORIZON)


class TestHistGradientBoostingCandidate:
    """An end-to-end search over scikit-learn's histogram gradient boosting."""

    @staticmethod
    def _forecaster(**kwargs):
        from sklearn.ensemble import HistGradientBoostingRegressor

        return _point(
            estimator=HistGradientBoostingRegressor(max_iter=25, min_samples_leaf=2, random_state=0),
            **kwargs,
        )

    def test_search_picks_a_shared_round_and_refits(self):
        search = GridSearchCV(
            forecaster=self._forecaster(),
            param_grid={"estimator__learning_rate": [0.1, 0.3]},
            scoring=MeanAbsoluteError(),
            cv=_cv(),
            validation="cv",
        )
        search.fit(y=_series(), forecasting_horizon=HORIZON)

        assert search.best_rounds_
        for rounds in search.best_rounds_.values():
            assert 1 <= rounds <= 25
        # The refit trains exactly the shared count with early stopping off.
        for position, estimator in zip(search.best_rounds_, search.best_forecaster_.estimator_, strict=True):
            assert estimator.get_params()["early_stopping"] is False
            assert estimator.n_iter_ == search.best_rounds_[position]
        assert search.predict().height == HORIZON

    def test_adapter_resolves_without_being_named(self):
        """No early_stopping_adapter is passed; resolution finds the built-in."""
        search = GridSearchCV(
            forecaster=self._forecaster(),
            param_grid={"estimator__learning_rate": [0.2]},
            scoring=MeanAbsoluteError(),
            cv=_cv(),
            validation="cv",
        )
        search.fit(y=_series(), forecasting_horizon=HORIZON)
        assert "rounds" in search.cv_results_

    def test_default_early_stopping_is_not_rejected_in_cv_mode(self):
        """The holdout path rejects early_stopping='auto'; cv mode overrides it."""
        from sklearn.ensemble import HistGradientBoostingRegressor

        forecaster = _point(
            estimator=HistGradientBoostingRegressor(
                max_iter=20, min_samples_leaf=2, random_state=0, early_stopping="auto"
            )
        )
        search = GridSearchCV(
            forecaster=forecaster,
            param_grid={"estimator__learning_rate": [0.2]},
            scoring=MeanAbsoluteError(),
            cv=_cv(),
            validation="cv",
        )
        search.fit(y=_series(), forecasting_horizon=HORIZON)
        assert search.best_rounds_


class TestCvModeRejectsSuppliedEvaluationKeys:
    """The cv guard refuses every key the mode supplies itself."""

    @pytest.mark.parametrize(
        "key",
        [
            "eval_set",
            "eval_X",
            "eval_y",
            "X_val",
            "y_val",
            "sample_weight_val",
            "eval_sample_weight",
            "sample_weight_eval_set",
            "X_actual_val",
            "X_forecast_val",
        ],
    )
    def test_key_rejected_before_any_fold(self, key):
        search = _search()
        with pytest.raises(ValueError, match=key):
            search.fit(y=_series(), forecasting_horizon=HORIZON, **{key: "anything"})
        assert not hasattr(search, "best_forecaster_")
