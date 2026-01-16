"""Tests for SearchCV forecaster."""

from datetime import datetime

import optuna
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge

from yohou.metrics import MeanAbsoluteError
from yohou.model_selection import SearchCV
from yohou.point_forecaster import PointReductionForecaster, SeasonalNaive

# Add parent directory to path for imports
from yohou.testing import _yield_yohou_forecaster_checks

length = 52

y = pl.DataFrame(
    {
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 0, 0, length - 1),
            interval="1s",
            eager=True,
        ),
        "a": range(length),
        "b": range(10, length + 10),
    }
)

X = pl.DataFrame(
    {
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 0, 0, length - 1),
            interval="1s",
            eager=True,
        ),
        "c": range(length),
        "d": range(10, length + 10),
        "e": range(20, length + 20),
    }
)


def test_search():
    search = SearchCV(
        forecaster=SeasonalNaive(),
        param_distributions={"seasonality": optuna.distributions.IntDistribution(1, 20)},
        scoring=MeanAbsoluteError(),
        error_score="raise",
        n_warmup_trials=5,
        n_trials=10,
        n_jobs=2,
    )

    search.fit(y, X, forecasting_horizon=1)


@pytest.mark.parametrize(
    "base_forecaster,expected_failures",
    [
        (
            PointReductionForecaster(estimator=Ridge(alpha=1.0)),
            [
                # SearchCV is a meta-forecaster that delegates to best_forecaster_
                # so it doesn't maintain its own observation buffers
                "check_fit_sets_forecaster_attributes",
                "check_update_extends_observation_buffers",
            ],
        ),
    ],
)
def test_search_cv_forecaster_checks(base_forecaster, expected_failures, y_X_factory):
    """Run systematic forecaster checks on SearchCV.

    SearchCV is a fully-fledged forecaster that delegates to best_forecaster_
    after fitting, so it should pass all standard forecaster checks.
    """
    # Generate data
    y, X = y_X_factory(length=100, seed=42)
    y_train, y_test = y[:80], y[80:]
    X_train, X_test = X[:80], X[80:]

    # Create SearchCV with minimal trials for testing
    search_cv = SearchCV(
        forecaster=base_forecaster,
        param_distributions={"estimator__alpha": optuna.distributions.FloatDistribution(0.1, 1.0)},
        scoring=MeanAbsoluteError(),
        n_warmup_trials=1,
        n_trials=2,
        n_jobs=1,
        refit=True,
        cv=2,
        verbose=0,
    )

    # Fit SearchCV
    search_cv_fitted = clone(search_cv)
    search_cv_fitted.fit(y_train, X_train, forecasting_horizon=3)

    # Run all generated checks
    expected_failures_set = set(expected_failures)
    for check_name, check_func, check_kwargs in _yield_yohou_forecaster_checks(
        search_cv_fitted,
        y_train,
        X_train,
        y_test,
        X_test,
    ):
        if check_name in expected_failures_set:
            pytest.skip(f"Expected failure: {check_name}")
        else:
            check_func(search_cv_fitted, **check_kwargs)


def test_search_cv_best_forecaster_attributes(y_X_factory):
    """Test that SearchCV properly exposes best_forecaster_ attributes."""
    y, X = y_X_factory(length=50, seed=42)

    search_cv = SearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        param_distributions={"estimator__alpha": optuna.distributions.FloatDistribution(0.1, 10.0)},
        scoring=MeanAbsoluteError(),
        n_warmup_trials=1,
        n_trials=2,
        cv=2,
        refit=True,
    )

    search_cv.fit(y, X, forecasting_horizon=3)

    # Verify best_forecaster_ exists and is fitted
    assert hasattr(search_cv, "best_forecaster_")
    assert hasattr(search_cv.best_forecaster_, "fit_forecasting_horizon_")

    # Verify cv_results_ attributes
    assert hasattr(search_cv, "cv_results_")
    assert "mean_test_score" in search_cv.cv_results_
    assert "params" in search_cv.cv_results_

    # Verify best attributes
    assert hasattr(search_cv, "best_params_")
    assert hasattr(search_cv, "best_index_")
    assert hasattr(search_cv, "best_score_")


def test_search_cv_return_train_score(y_X_factory):
    """Test that return_train_score parameter correctly stores training scores."""
    y, X = y_X_factory(length=50, seed=42)

    # With return_train_score=True
    search_cv_with_train = SearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        param_distributions={"estimator__alpha": optuna.distributions.FloatDistribution(0.1, 10.0)},
        scoring=MeanAbsoluteError(),
        n_warmup_trials=1,
        n_trials=2,
        cv=2,
        refit=True,
        return_train_score=True,
    )

    search_cv_with_train.fit(y, X, forecasting_horizon=3)

    # Verify train scores are present
    assert "mean_train_score" in search_cv_with_train.cv_results_
    assert "std_train_score" in search_cv_with_train.cv_results_

    # With return_train_score=False (default)
    search_cv_without_train = SearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        param_distributions={"estimator__alpha": optuna.distributions.FloatDistribution(0.1, 10.0)},
        scoring=MeanAbsoluteError(),
        n_warmup_trials=1,
        n_trials=2,
        cv=2,
        refit=True,
        return_train_score=False,
    )

    search_cv_without_train.fit(y, X, forecasting_horizon=3)

    # Verify train scores are NOT present
    assert "mean_train_score" not in search_cv_without_train.cv_results_
    assert "std_train_score" not in search_cv_without_train.cv_results_


def test_search_cv_predict_delegates(y_X_factory):
    """Test that SearchCV.predict() properly delegates to best_forecaster_."""
    y, X = y_X_factory(length=50, seed=42)
    y_train, y_test = y[:40], y[40:]
    X_train, X_test = X[:40], X[40:]

    search_cv = SearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        param_distributions={"estimator__alpha": optuna.distributions.FloatDistribution(0.1, 10.0)},
        scoring=MeanAbsoluteError(),
        n_warmup_trials=1,
        n_trials=2,
        cv=2,
        refit=True,
    )

    search_cv.fit(y_train, X_train, forecasting_horizon=3)

    # Predict with SearchCV (using new signature with X)
    y_pred_search = search_cv.predict(X=None, forecasting_horizon=3)

    # Predict directly with best_forecaster_
    y_pred_direct = search_cv.best_forecaster_.predict(X=None, forecasting_horizon=3)

    # Both should produce identical predictions
    assert y_pred_search.equals(y_pred_direct)


def test_search_cv_predict_with_x(y_X_factory):
    """Test that SearchCV.predict() correctly passes X to best_forecaster_."""
    y, X = y_X_factory(length=50, seed=42)
    y_train = y[:40]
    X_train = X[:40]

    search_cv = SearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        param_distributions={"estimator__alpha": optuna.distributions.FloatDistribution(0.1, 10.0)},
        scoring=MeanAbsoluteError(),
        n_warmup_trials=1,
        n_trials=2,
        cv=2,
        refit=True,
    )

    search_cv.fit(y_train, X_train, forecasting_horizon=3)

    # Create future X for prediction
    X_future = X[40:43]

    # Predict with SearchCV passing X
    y_pred_search = search_cv.predict(X=X_future, forecasting_horizon=3)

    # Predict directly with best_forecaster_
    y_pred_direct = search_cv.best_forecaster_.predict(X=X_future, forecasting_horizon=3)

    # Both should produce identical predictions
    assert y_pred_search.equals(y_pred_direct)
    assert "time" in y_pred_search.columns


def test_search_cv_update_with_x(y_X_factory):
    """Test that SearchCV.update() correctly passes X to best_forecaster_."""
    y, X = y_X_factory(length=50, seed=42)
    y_train = y[:40]
    X_train = X[:40]

    search_cv = SearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        param_distributions={"estimator__alpha": optuna.distributions.FloatDistribution(0.1, 10.0)},
        scoring=MeanAbsoluteError(),
        n_warmup_trials=1,
        n_trials=2,
        cv=2,
        refit=True,
    )

    search_cv.fit(y_train, X_train, forecasting_horizon=3)

    # Update with new data including X
    y_update = y[40:45]
    X_update = X[40:45]

    # Update SearchCV
    search_cv_result = search_cv.update(y_update, X_update)

    # Verify returns self
    assert search_cv_result is search_cv


def test_search_cv_update_predict_delegates(y_X_factory):
    """Test that SearchCV.update_predict() properly delegates all parameters."""
    y, X = y_X_factory(length=50, seed=42)
    y_train = y[:40]
    X_train = X[:40]

    search_cv = SearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        param_distributions={"estimator__alpha": optuna.distributions.FloatDistribution(0.1, 10.0)},
        scoring=MeanAbsoluteError(),
        n_warmup_trials=1,
        n_trials=2,
        cv=2,
        refit=True,
    )

    search_cv.fit(y_train, X_train, forecasting_horizon=3)

    # Prepare update data
    y_update = y[40:45]
    X_update = X[40:45]

    # Use update_predict
    y_pred = search_cv.update_predict(
        y=y_update,
        X=X_update,
        forecasting_horizon=3,
        stride=1,
    )

    # Verify predictions are returned
    assert isinstance(y_pred, pl.DataFrame)
    assert "time" in y_pred.columns


def test_search_cv_reset_delegates(y_X_factory):
    """Test that SearchCV.reset() properly delegates to best_forecaster_."""
    y, X = y_X_factory(length=50, seed=42)
    y_train = y[:40]
    X_train = X[:40]

    search_cv = SearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        param_distributions={"estimator__alpha": optuna.distributions.FloatDistribution(0.1, 10.0)},
        scoring=MeanAbsoluteError(),
        n_warmup_trials=1,
        n_trials=2,
        cv=2,
        refit=True,
    )

    search_cv.fit(y_train, X_train, forecasting_horizon=3)

    # Reset with new observation data
    y_reset = y[35:40]
    X_reset = X[35:40]

    # Reset SearchCV
    search_cv_result = search_cv.reset(y_reset, X_reset)

    # Verify returns self
    assert search_cv_result is search_cv


@pytest.mark.skip(
    reason="SearchCV does not yet support stateful scorers - needs scorer.fit() implementation"
)
def test_search_cv_with_stateful_scorer(y_X_factory):
    """Test SearchCV with RootMeanSquaredScaledError scorer that requires fit().

    RootMeanSquaredScaledError is a stateful scorer that requires fit() to compute scaling factors
    from training data. This test verifies that SearchCV properly handles
    stateful scorers by:
    1. Calling scorer.fit(y_train) during cross-validation
    2. Properly routing training data through the scoring pipeline
    3. Computing correct scores with per-fold fitted scorers
    """
    from yohou.metrics import RootMeanSquaredScaledError

    y, X = y_X_factory(length=100, seed=42)
    y_train = y[:80]
    X_train = X[:80]

    # Create RootMeanSquaredScaledError scorer that requires fit
    rmsse = RootMeanSquaredScaledError(seasonality=3)

    # SearchCV should handle stateful scorer automatically
    search_cv = SearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        param_distributions={"estimator__alpha": optuna.distributions.FloatDistribution(0.1, 10.0)},
        scoring=rmsse,
        n_warmup_trials=2,
        n_trials=5,
        cv=3,
        refit=True,
        return_train_score=True,  # Test that train scores are computed correctly
    )

    # Fit should succeed - SearchCV handles scorer.fit() internally
    search_cv.fit(y_train, X_train, forecasting_horizon=5)

    # Verify SearchCV completed successfully
    assert hasattr(search_cv, "best_forecaster_")
    assert hasattr(search_cv, "best_score_")
    assert hasattr(search_cv, "cv_results_")

    # Verify cross-validation scores were computed
    assert "mean_test_score" in search_cv.cv_results_
    assert "std_test_score" in search_cv.cv_results_
    assert len(search_cv.cv_results_["mean_test_score"]) == 5  # n_trials

    # Verify train scores were computed (RootMeanSquaredScaledError requires calibration)
    assert "mean_train_score" in search_cv.cv_results_
    assert "std_train_score" in search_cv.cv_results_

    # Verify all scores are positive floats
    for score in search_cv.cv_results_["mean_test_score"]:
        assert isinstance(score, float)
        assert score >= 0  # RootMeanSquaredScaledError is non-negative

    # Verify best_score_ is a valid RootMeanSquaredScaledError value
    assert isinstance(search_cv.best_score_, float)
    assert search_cv.best_score_ >= 0

    # Predictions should still work
    y_pred = search_cv.predict(X=None, forecasting_horizon=5)
    assert isinstance(y_pred, pl.DataFrame)
    assert "time" in y_pred.columns


# ============================================================================
# Aggregate Scorer Tests
# ============================================================================


def test_search_cv_raises_with_non_timewise_scorer():
    """SearchCV should raise ValueError when used with aggregate != 'timewise' scorers."""
    from yohou.metrics import MeanAbsoluteError
    from yohou.preprocessing import LagTransformer

    # Create simple training data
    y_train = pl.DataFrame(
        {
            "time": pl.datetime_range(
                start=datetime(2021, 1, 1),
                end=datetime(2021, 1, 1, 0, 0, 49),
                interval="1s",
                eager=True,
            ),
            "value": range(50),
        }
    )

    # Point forecaster with lag transformer for lag features
    forecaster = PointReductionForecaster(
        estimator=Ridge(),
        feature_transformer=LagTransformer(lag=[1, 2, 3]),
    )

    # SearchCV with aggregation_method=['componentwise'] scorer (should fail)
    search = SearchCV(
        forecaster=forecaster,
        param_distributions={
            "estimator__alpha": optuna.distributions.FloatDistribution(0.1, 10.0),
        },
        scoring=MeanAbsoluteError(
            aggregation_method=["componentwise"]
        ),  # aggregation_method != 'all' should raise error
        n_trials=2,
        cv=2,
    )

    # Should raise ValueError during fit
    with pytest.raises(ValueError, match="aggregation_method.*all.*SearchCV"):
        search.fit(y=y_train, X=None, forecasting_horizon=5)
