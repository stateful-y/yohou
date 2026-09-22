"""Tests for yohou.testing.reduction check functions."""

import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from yohou.base.reduction import _EvalSet
from yohou.interval.reduction import IntervalReductionForecaster
from yohou.point.reduction import PointReductionForecaster
from yohou.testing.reduction import (
    check_estimator_parameter,
    check_reduction_strategy,
    check_validation_holdout_default_noop,
    check_validation_holdout_fit,
    check_validation_holdout_parameters,
)


class TestReductionChecks:
    """Tests for reduction forecaster check functions."""

    def test_check_estimator_parameter_point(self):
        """Test check_estimator_parameter passes for point reduction forecaster."""
        forecaster = PointReductionForecaster(estimator=LinearRegression())

        # Should not raise
        check_estimator_parameter(forecaster)

    def test_check_estimator_parameter_interval(self):
        """Test check_estimator_parameter passes for interval reduction forecaster."""
        forecaster = IntervalReductionForecaster(estimator=LinearRegression())

        # Should not raise
        check_estimator_parameter(forecaster)

    def test_check_estimator_parameter_default(self):
        """Test check validates default estimator is set."""
        forecaster = PointReductionForecaster()

        # Should not raise - default estimator is valid
        check_estimator_parameter(forecaster)

    def test_check_reduction_strategy_point(self):
        """Test check_reduction_strategy passes for point reduction forecaster."""
        forecaster = PointReductionForecaster(reduction_strategy="direct")

        # Should not raise
        check_reduction_strategy(forecaster)

    def test_check_reduction_strategy_multi_output(self):
        """Test check validates multi-output strategy."""
        forecaster = PointReductionForecaster(reduction_strategy="multi-output")

        # Should not raise
        check_reduction_strategy(forecaster)

    def test_check_reduction_strategy_interval(self):
        """Test check validates interval reduction forecaster."""
        forecaster = IntervalReductionForecaster(reduction_strategy="direct")

        # Should not raise
        check_reduction_strategy(forecaster)

    def test_check_reduction_strategy_dir_rec(self):
        """Test check validates dir-rec strategy."""
        forecaster = PointReductionForecaster(reduction_strategy="dir-rec")

        # Should not raise
        check_reduction_strategy(forecaster)

    def test_check_reduction_strategy_no_attribute(self):
        """Test check returns early when forecaster lacks reduction_strategy."""
        from sklearn.linear_model import LinearRegression

        estimator = LinearRegression()
        # LinearRegression has no reduction_strategy attribute, so check returns early
        check_reduction_strategy(estimator)


class _RewritingEvalForecaster(PointReductionForecaster):
    """Reduction forecaster that rewrites the evaluation pair before delivery."""

    def _rewrite(self, eval_data, X_tab, y_tab):
        raise NotImplementedError

    def _estimator_fit_one(
        self, y_t, X_t, forecasting_horizon, estimator_params=None, estimator_fit_params=None, eval_data=None
    ):
        X_tab, y_tab = self._get_stacked_tabularized_data(y_t, X_t, forecasting_horizon)
        return super()._estimator_fit_one(
            y_t,
            X_t,
            forecasting_horizon,
            estimator_params=estimator_params,
            estimator_fit_params=estimator_fit_params,
            eval_data=self._rewrite(eval_data, X_tab, y_tab),
        )


class _DropsEvalSet(_RewritingEvalForecaster):
    def _rewrite(self, eval_data, X_tab, y_tab):
        return None


class _ShortEvalSet(_RewritingEvalForecaster):
    def _rewrite(self, eval_data, X_tab, y_tab):
        return _EvalSet(eval_data.X[:-1], eval_data.y[:-1], None)


class _TrainingRowsEvalSet(_RewritingEvalForecaster):
    def _rewrite(self, eval_data, X_tab, y_tab):
        n = len(eval_data.X)
        return _EvalSet(X_tab.head(n), y_tab.head(n), None)


class _LeakingTraining(PointReductionForecaster):
    """Reduction forecaster whose holdout fit shifts every training feature."""

    def _estimator_fit_one(
        self, y_t, X_t, forecasting_horizon, estimator_params=None, estimator_fit_params=None, eval_data=None
    ):
        if eval_data is not None:
            X_t = X_t.with_columns(pl.exclude("time") + 1.0)
        return super()._estimator_fit_one(
            y_t,
            X_t,
            forecasting_horizon,
            estimator_params=estimator_params,
            estimator_fit_params=estimator_fit_params,
            eval_data=eval_data,
        )


class _AlwaysDeliversEvalSet(_RewritingEvalForecaster):
    def _rewrite(self, eval_data, X_tab, y_tab):
        return _EvalSet(X_tab.head(1), y_tab.head(1), None)


class TestValidationHoldoutChecks:
    """The validation-holdout checks pass on a correct forecaster and fail on broken ones."""

    @pytest.fixture
    def y(self, y_X_factory):
        y, _ = y_X_factory(length=60, n_targets=1, n_features=0)
        return y

    def test_checks_pass(self, y):
        forecaster = PointReductionForecaster()
        check_validation_holdout_parameters(forecaster)
        check_validation_holdout_fit(forecaster, y)
        check_validation_holdout_default_noop(forecaster, y)

    def test_parameters_missing(self):
        with pytest.raises(AssertionError, match="validation_size must be a constructor parameter"):
            check_validation_holdout_parameters(LinearRegression())

    def test_fit_no_eval_set(self, y):
        with pytest.raises(AssertionError, match="no eval_set reached the estimator"):
            check_validation_holdout_fit(_DropsEvalSet(), y)

    def test_fit_wrong_row_count(self, y):
        with pytest.raises(AssertionError, match="evaluation rows"):
            check_validation_holdout_fit(_ShortEvalSet(), y)

    def test_fit_training_rows(self, y):
        with pytest.raises(AssertionError, match="also appear in the training matrix"):
            check_validation_holdout_fit(_TrainingRowsEvalSet(), y)

    def test_fit_tail_leak(self, y):
        with pytest.raises(AssertionError, match="differs from a head-only fit"):
            check_validation_holdout_fit(_LeakingTraining(), y)

    def test_default_noop_delivers_eval_set(self, y):
        with pytest.raises(AssertionError, match="must not deliver an eval_set"):
            check_validation_holdout_default_noop(_AlwaysDeliversEvalSet(), y)
