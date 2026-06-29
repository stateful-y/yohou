"""Tests for search CV check functions in yohou.testing.search."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from conftest import run_checks
from yohou.interval import SplitConformalForecaster
from yohou.metrics.interval import EmpiricalCoverage
from yohou.metrics.point import MeanAbsoluteError, MeanSquaredError
from yohou.model_selection import ExpandingWindowSplitter, GridSearchCV, RandomizedSearchCV
from yohou.point.naive import SeasonalNaive
from yohou.testing import _yield_yohou_search_checks
from yohou.testing.search import (
    check_grid_search_exhaustive,
    check_grid_search_param_grid_validation,
    check_randomized_search_distributions,
    check_randomized_search_n_iter,
    check_randomized_search_reproducibility,
    check_search_clone_preserves_params,
    check_search_cv_results_structure,
    check_search_error_score_handling,
    check_search_fit_sets_attributes,
    check_search_interval_predict_delegates,
    check_search_method_availability,
    check_search_multimetric_scoring,
    check_search_not_fitted_error,
    check_search_observe_delegates,
    check_search_panel_data,
    check_search_predict_delegates,
    check_search_refit_false_no_forecaster,
    check_search_return_train_score,
    check_search_rewind_delegates,
)


@pytest.fixture(scope="module")
def search_data():
    """Training data for search CV tests."""
    n = 100
    times = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({
        "time": times,
        "val": [float(i % 7) for i in range(n)],
    })
    return y


@pytest.fixture(scope="module")
def grid_search_cv():
    """Unfitted GridSearchCV with SeasonalNaive."""
    return GridSearchCV(
        forecaster=SeasonalNaive(seasonality=1),
        param_grid={"seasonality": [1, 7]},
        cv=ExpandingWindowSplitter(n_splits=2, test_size=10),
        scoring=MeanAbsoluteError(),
    )


@pytest.fixture(scope="module")
def randomized_search_cv():
    """Unfitted RandomizedSearchCV with SeasonalNaive."""
    return RandomizedSearchCV(
        forecaster=SeasonalNaive(seasonality=1),
        param_distributions={"seasonality": [1, 7, 14]},
        n_iter=2,
        cv=ExpandingWindowSplitter(n_splits=2, test_size=10),
        scoring=MeanAbsoluteError(),
        random_state=42,
    )


@pytest.fixture(scope="module")
def fitted_grid_search(grid_search_cv, search_data):
    """Fitted GridSearchCV for delegation tests."""
    from sklearn.base import clone

    y_train = search_data.head(80)
    gs = clone(grid_search_cv)
    gs.fit(y_train, forecasting_horizon=3)
    return gs, search_data


@pytest.fixture(scope="module")
def multimetric_search_cv():
    """Unfitted GridSearchCV with dict scoring."""
    return GridSearchCV(
        forecaster=SeasonalNaive(seasonality=1),
        param_grid={"seasonality": [1, 7]},
        cv=ExpandingWindowSplitter(n_splits=2, test_size=10),
        scoring={"mae": MeanAbsoluteError(), "mse": MeanSquaredError()},
        refit="mae",
    )


class TestSearchCheckFunctionsNoFit:
    """Tests for search check functions that do not require fitting."""

    def test_clone_preserves_params_grid(self, grid_search_cv):
        """check_search_clone_preserves_params works for GridSearchCV."""
        check_search_clone_preserves_params(grid_search_cv)

    def test_clone_preserves_params_randomized(self, randomized_search_cv):
        """check_search_clone_preserves_params works for RandomizedSearchCV."""
        check_search_clone_preserves_params(randomized_search_cv)

    def test_param_grid_validation(self, grid_search_cv):
        """check_grid_search_param_grid_validation validates grid format."""
        check_grid_search_param_grid_validation(grid_search_cv)

    def test_not_fitted_error(self, grid_search_cv, search_data):
        """check_search_not_fitted_error on unfitted search raises."""
        check_search_not_fitted_error(grid_search_cv, search_data)

    def test_not_fitted_distinguishes_error_paths(self, grid_search_cv):
        """Disambiguate the two not-fitted paths the check accepts together.

        ``check_search_not_fitted_error`` accepts either ``NotFittedError`` or
        ``AttributeError`` for ``predict()``. Pin which is which: accessing a
        fitted attribute raises ``NotFittedError``, while ``predict()`` on an
        unfitted search is gated off by ``@available_if`` and raises
        ``AttributeError`` (best_forecaster_ does not exist yet).
        """
        from sklearn.base import clone
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted

        unfitted = clone(grid_search_cv)

        with pytest.raises(NotFittedError):
            check_is_fitted(unfitted, "best_forecaster_")

        with pytest.raises(AttributeError):
            unfitted.predict(forecasting_horizon=1)


class TestSearchCheckFunctionsFit:
    """Tests for search check functions that require fitting."""

    @pytest.mark.slow
    def test_fit_sets_attributes(self, grid_search_cv, search_data):
        """check_search_fit_sets_attributes validates fitted state."""
        check_search_fit_sets_attributes(grid_search_cv, search_data, forecasting_horizon=3)

    @pytest.mark.slow
    def test_cv_results_structure(self, grid_search_cv, search_data):
        """check_search_cv_results_structure validates dict keys."""
        check_search_cv_results_structure(grid_search_cv, search_data, forecasting_horizon=3)

    @pytest.mark.slow
    def test_grid_search_exhaustive(self, grid_search_cv, search_data):
        """check_grid_search_exhaustive verifies all combos evaluated."""
        check_grid_search_exhaustive(grid_search_cv, search_data, forecasting_horizon=3)

    @pytest.mark.slow
    def test_refit_false_no_forecaster(self, grid_search_cv, search_data):
        """check_search_refit_false_no_forecaster with refit=False."""
        check_search_refit_false_no_forecaster(grid_search_cv, search_data, forecasting_horizon=3)

    @pytest.mark.slow
    def test_method_availability(self, grid_search_cv, search_data):
        """check_search_method_availability validates refit behavior."""
        check_search_method_availability(grid_search_cv, search_data, forecasting_horizon=3)

    @pytest.mark.slow
    def test_return_train_score(self, grid_search_cv, search_data):
        """check_search_return_train_score validates train keys added."""
        check_search_return_train_score(grid_search_cv, search_data, forecasting_horizon=3)

    @pytest.mark.slow
    def test_error_score_handling(self, grid_search_cv, search_data):
        """check_search_error_score_handling validates error tolerance."""
        check_search_error_score_handling(grid_search_cv, search_data, forecasting_horizon=3)


class TestSearchCheckFunctionsRandomized:
    """Tests for RandomizedSearchCV-specific check functions."""

    @pytest.mark.slow
    def test_n_iter(self, randomized_search_cv, search_data):
        """check_randomized_search_n_iter validates combo count."""
        check_randomized_search_n_iter(randomized_search_cv, search_data, forecasting_horizon=3)

    @pytest.mark.slow
    def test_reproducibility(self, randomized_search_cv, search_data):
        """check_randomized_search_reproducibility validates determinism."""
        check_randomized_search_reproducibility(randomized_search_cv, search_data, forecasting_horizon=3)

    @pytest.mark.slow
    def test_distributions(self, randomized_search_cv, search_data):
        """check_randomized_search_distributions validates sampling."""
        check_randomized_search_distributions(randomized_search_cv, search_data, forecasting_horizon=3)


class TestSearchCheckFunctionsDelegation:
    """Tests for delegation check functions that need fitted search."""

    @pytest.mark.slow
    def test_predict_delegates(self, fitted_grid_search):
        """check_search_predict_delegates verifies prediction delegation."""
        gs, y = fitted_grid_search
        y_test = y.tail(20)
        check_search_predict_delegates(gs, y_test)

    @pytest.mark.slow
    def test_observe_delegates(self, fitted_grid_search):
        """check_search_observe_delegates verifies observation delegation."""
        gs, y = fitted_grid_search
        y_update = y.slice(80, 10)
        check_search_observe_delegates(gs, y_update)

    @pytest.mark.slow
    def test_rewind_delegates(self, fitted_grid_search):
        """check_search_rewind_delegates verifies rewind delegation."""
        gs, y = fitted_grid_search
        y_reset = y.tail(10)
        check_search_rewind_delegates(gs, y_reset)


class TestSearchCheckFunctionsIntervalDelegation:
    """Tests for the interval-scoring delegation check (conditionally yielded)."""

    @pytest.fixture(scope="class")
    def interval_search_data(self):
        """Training target with enough history for a calibrated conformal search."""
        n = 120
        times = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=n - 1),
            interval="1d",
            eager=True,
        )
        return pl.DataFrame({"time": times, "val": [float(i % 7) for i in range(n)]})

    @pytest.fixture(scope="class")
    def fitted_interval_search(self, interval_search_data):
        """Fitted GridSearchCV scored with an interval metric (refit=True)."""
        gs = GridSearchCV(
            forecaster=SplitConformalForecaster(
                point_forecaster=SeasonalNaive(seasonality=7),
                calibration_size=20,
            ),
            param_grid={"calibration_size": [20, 30]},
            cv=ExpandingWindowSplitter(n_splits=2, test_size=10),
            scoring=EmpiricalCoverage(),
        )
        gs.fit(interval_search_data.head(100), forecasting_horizon=3)
        return gs

    @pytest.mark.slow
    def test_interval_predict_delegates(self, fitted_interval_search):
        """check_search_interval_predict_delegates exercises predict_interval delegation."""
        check_search_interval_predict_delegates(fitted_interval_search)

    def test_interval_predict_delegates_requires_vintage_time(self):
        """check_search_interval_predict_delegates rejects predictions lacking vintage_time."""
        from sklearn.base import BaseEstimator

        class _StubSearchNoVintage(BaseEstimator):
            """Fitted-looking stub whose predict_interval omits vintage_time."""

            def fit(self, *args, **kwargs):
                # Set a trailing-underscore attr so check_is_fitted() passes.
                self.best_forecaster_ = object()
                return self

            def predict_interval(self, coverage_rates=None, X_future=None, X_forecast=None):
                return pl.DataFrame({
                    "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
                    "val_lower_0.9": [0.0, 1.0],
                    "val_upper_0.9": [2.0, 3.0],
                })

        with pytest.raises(AssertionError, match="vintage_time"):
            check_search_interval_predict_delegates(_StubSearchNoVintage().fit())


class TestSearchCheckFunctionsPanel:
    """Tests for the panel-data subselection check (conditionally yielded)."""

    @pytest.fixture(scope="class")
    def panel_search_data(self):
        """Panel training target with two store groups."""
        n = 100
        times = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=n - 1),
            interval="1d",
            eager=True,
        )
        return pl.DataFrame({
            "time": times,
            "store_1__val": [float(i % 7) for i in range(n)],
            "store_2__val": [float(i % 5) + 1 for i in range(n)],
        })

    @pytest.fixture(scope="class")
    def fitted_panel_search(self, panel_search_data):
        """Fitted GridSearchCV over panel data."""
        gs = GridSearchCV(
            forecaster=SeasonalNaive(seasonality=1),
            param_grid={"seasonality": [1, 7]},
            cv=ExpandingWindowSplitter(n_splits=2, test_size=10),
            scoring=MeanAbsoluteError(),
        )
        gs.fit(panel_search_data.head(80), forecasting_horizon=3)
        return gs, panel_search_data

    @pytest.mark.slow
    def test_panel_data_subselection(self, fitted_panel_search):
        """check_search_panel_data verifies the groups parameter propagates through predict()."""
        gs, y = fitted_panel_search
        check_search_panel_data(gs, y.tail(20), groups=["store_1"])


class TestSearchCheckFunctionsMultimetric:
    """Tests for multi-metric search check functions."""

    @pytest.mark.slow
    def test_multimetric_scoring(self, multimetric_search_cv, search_data):
        """check_search_multimetric_scoring validates multi-metric setup."""
        check_search_multimetric_scoring(multimetric_search_cv, search_data, forecasting_horizon=3)

    @pytest.mark.slow
    def test_multimetric_scoring_refit_false(self, search_data):
        """check_search_multimetric_scoring with refit=False skips best_score_ assertion."""
        gs = GridSearchCV(
            forecaster=SeasonalNaive(seasonality=1),
            param_grid={"seasonality": [1, 7]},
            cv=ExpandingWindowSplitter(n_splits=2, test_size=10),
            scoring={"mae": MeanAbsoluteError(), "mse": MeanSquaredError()},
            refit=False,
        )
        check_search_multimetric_scoring(gs, search_data, forecasting_horizon=3)


class TestSearchSystematicGridChecks:
    """Systematic tests using _yield_yohou_search_checks for GridSearchCV."""

    @pytest.mark.slow
    def test_grid_search_all_checks(self, search_data):
        """Run all yielded checks for GridSearchCV."""
        y_train = search_data.head(80)
        y_test = search_data.tail(20)
        gs = GridSearchCV(
            forecaster=SeasonalNaive(seasonality=1),
            param_grid={"seasonality": [1, 7]},
            cv=ExpandingWindowSplitter(n_splits=2, test_size=10),
            scoring=MeanAbsoluteError(),
        )
        gs.fit(y_train, forecasting_horizon=3)

        run_checks(
            gs,
            _yield_yohou_search_checks(gs, y_train, None, y_test, None),
        )


class TestSearchSystematicRandomizedChecks:
    """Systematic tests using _yield_yohou_search_checks for RandomizedSearchCV."""

    @pytest.mark.slow
    def test_randomized_search_all_checks(self, search_data):
        """Run all yielded checks for RandomizedSearchCV."""
        y_train = search_data.head(80)
        y_test = search_data.tail(20)
        rs = RandomizedSearchCV(
            forecaster=SeasonalNaive(seasonality=1),
            param_distributions={"seasonality": [1, 7, 14]},
            n_iter=2,
            cv=ExpandingWindowSplitter(n_splits=2, test_size=10),
            scoring=MeanAbsoluteError(),
            random_state=42,
        )
        rs.fit(y_train, forecasting_horizon=3)

        run_checks(
            rs,
            _yield_yohou_search_checks(rs, y_train, None, y_test, None),
        )

    @pytest.mark.slow
    def test_randomized_search_refit_false_checks(self, search_data):
        """Run all yielded checks for RandomizedSearchCV with refit=False."""
        y_train = search_data.head(80)
        y_test = search_data.tail(20)
        rs = RandomizedSearchCV(
            forecaster=SeasonalNaive(seasonality=1),
            param_distributions={"seasonality": [1, 7, 14]},
            n_iter=2,
            cv=ExpandingWindowSplitter(n_splits=2, test_size=10),
            scoring=MeanAbsoluteError(),
            random_state=42,
            refit=False,
        )
        rs.fit(y_train, forecasting_horizon=3)

        run_checks(
            rs,
            _yield_yohou_search_checks(rs, y_train, None, y_test, None),
        )


class TestSearchSystematicMultimetricChecks:
    """Systematic tests for multimetric search via generator."""

    @pytest.mark.slow
    def test_multimetric_grid_all_checks(self, search_data):
        """Run all yielded checks for multi-metric GridSearchCV."""
        y_train = search_data.head(80)
        y_test = search_data.tail(20)
        gs = GridSearchCV(
            forecaster=SeasonalNaive(seasonality=1),
            param_grid={"seasonality": [1, 7]},
            cv=ExpandingWindowSplitter(n_splits=2, test_size=10),
            scoring={"mae": MeanAbsoluteError(), "mse": MeanSquaredError()},
            refit="mae",
        )
        gs.fit(y_train, forecasting_horizon=3)

        run_checks(
            gs,
            _yield_yohou_search_checks(gs, y_train, None, y_test, None),
        )


class TestSearchCheckTypeGuards:
    """Tests for type guard ValueError raises in check functions."""

    def test_grid_exhaustive_rejects_randomized(self, randomized_search_cv):
        """check_grid_search_exhaustive raises ValueError for RandomizedSearchCV."""
        with pytest.raises(ValueError, match="GridSearchCV"):
            check_grid_search_exhaustive(randomized_search_cv, pl.DataFrame({"time": [], "val": []}))

    def test_grid_param_validation_rejects_randomized(self, randomized_search_cv):
        """check_grid_search_param_grid_validation raises for RandomizedSearchCV."""
        with pytest.raises(ValueError, match="GridSearchCV"):
            check_grid_search_param_grid_validation(randomized_search_cv)

    def test_randomized_n_iter_rejects_grid(self, grid_search_cv, search_data):
        """check_randomized_search_n_iter raises ValueError for GridSearchCV."""
        with pytest.raises(ValueError, match="RandomizedSearchCV"):
            check_randomized_search_n_iter(grid_search_cv, search_data)

    def test_randomized_reproducibility_rejects_grid(self, grid_search_cv, search_data):
        """check_randomized_search_reproducibility raises for GridSearchCV."""
        with pytest.raises(ValueError, match="RandomizedSearchCV"):
            check_randomized_search_reproducibility(grid_search_cv, search_data)

    def test_randomized_reproducibility_rejects_no_seed(self, search_data):
        """check_randomized_search_reproducibility raises for no random_state."""
        rs = RandomizedSearchCV(
            forecaster=SeasonalNaive(seasonality=1),
            param_distributions={"seasonality": [1, 7]},
            n_iter=2,
            cv=ExpandingWindowSplitter(n_splits=2, test_size=10),
            scoring=MeanAbsoluteError(),
            random_state=None,
        )
        with pytest.raises(ValueError, match="random_state"):
            check_randomized_search_reproducibility(rs, search_data)

    def test_randomized_distributions_rejects_grid(self, grid_search_cv, search_data):
        """check_randomized_search_distributions raises for GridSearchCV."""
        with pytest.raises(ValueError, match="RandomizedSearchCV"):
            check_randomized_search_distributions(grid_search_cv, search_data)

    def test_multimetric_scoring_rejects_non_dict(self, search_data):
        """check_search_multimetric_scoring raises for non-dict scoring."""
        gs = GridSearchCV(
            forecaster=SeasonalNaive(seasonality=1),
            param_grid={"seasonality": [1, 7]},
            cv=ExpandingWindowSplitter(n_splits=2, test_size=10),
            scoring=MeanAbsoluteError(),
        )
        with pytest.raises(ValueError, match="dict"):
            check_search_multimetric_scoring(gs, search_data)
