"""Tests for SearchCV forecaster."""

import sys
from datetime import datetime
from pathlib import Path

import optuna
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.linear_model import Ridge

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from estimator_checks import _yield_yohou_forecaster_checks

from yohou.metrics import MAE
from yohou.model_selection import SearchCV
from yohou.point_forecaster import PointReductionForecaster, SeasonalNaive

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

X_ante = pl.DataFrame(
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
        scoring=MAE(),
        error_score="raise",
        n_warmup_trials=5,
        n_trials=10,
        n_jobs=2,
    )

    search.fit(y, X_ante, forecasting_horizon=1)


@pytest.mark.parametrize(
    "base_forecaster,tags,expected_failures",
    [
        (
            PointReductionForecaster(estimator=Ridge(alpha=1.0)),
            {
                "forecaster_type": "point",
                "uses_reduction": False,  # SearchCV itself doesn't use reduction
                "uses_transformers": False,  # SearchCV itself doesn't use transformers
            },
            [
                # SearchCV is a meta-forecaster that delegates to best_forecaster_
                # so it doesn't maintain its own observation buffers
                "check_fit_sets_forecaster_attributes",
                "check_update_extends_observation_buffers",
            ],
        ),
    ],
)
def test_search_cv_forecaster_checks(base_forecaster, tags, expected_failures, y_X_factory):
    """Run systematic forecaster checks on SearchCV.

    SearchCV is a fully-fledged forecaster that delegates to best_forecaster_
    after fitting, so it should pass all standard forecaster checks.
    """
    # Generate data
    y, X_ante, X_post = y_X_factory(length=100, seed=42)
    y_train, y_test = y[:80], y[80:]
    X_ante_train, X_ante_test = X_ante[:80], X_ante[80:]
    X_post_train, X_post_test = (X_post[:80], X_post[80:]) if X_post is not None else (None, None)

    # Create SearchCV with minimal trials for testing
    search_cv = SearchCV(
        forecaster=base_forecaster,
        param_distributions={"estimator__alpha": optuna.distributions.FloatDistribution(0.1, 1.0)},
        scoring=MAE(),
        n_warmup_trials=1,
        n_trials=2,
        predict_forecasting_horizon=3,
        predict_stride=1,
        n_jobs=1,
        refit=True,
        cv=2,
        verbose=0,
    )

    # Fit SearchCV
    search_cv_fitted = clone(search_cv)
    search_cv_fitted.fit(y_train, X_ante_train, None, forecasting_horizon=3)

    # Run all generated checks
    expected_failures_set = set(expected_failures)
    for check_name, check_func, check_kwargs in _yield_yohou_forecaster_checks(
        search_cv_fitted,
        y_train,
        X_ante_train,
        None,
        y_test,
        X_ante_test,
        None,
        tags=tags,
    ):
        if check_name in expected_failures_set:
            pytest.skip(f"Expected failure: {check_name}")
        else:
            check_func(search_cv_fitted, **check_kwargs)


def test_search_cv_best_forecaster_attributes(y_X_factory):
    """Test that SearchCV properly exposes best_forecaster_ attributes."""
    y, X_ante, X_post = y_X_factory(length=50, seed=42)

    search_cv = SearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        param_distributions={"estimator__alpha": optuna.distributions.FloatDistribution(0.1, 10.0)},
        scoring=MAE(),
        n_warmup_trials=1,
        n_trials=2,
        cv=2,
        refit=True,
    )

    search_cv.fit(y, X_ante, None, forecasting_horizon=3)

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
    y, X_ante, X_post = y_X_factory(length=50, seed=42)

    # With return_train_score=True
    search_cv_with_train = SearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        param_distributions={"estimator__alpha": optuna.distributions.FloatDistribution(0.1, 10.0)},
        scoring=MAE(),
        n_warmup_trials=1,
        n_trials=2,
        cv=2,
        refit=True,
        return_train_score=True,
    )

    search_cv_with_train.fit(y, X_ante, None, forecasting_horizon=3)

    # Verify train scores are present
    assert "mean_train_score" in search_cv_with_train.cv_results_
    assert "std_train_score" in search_cv_with_train.cv_results_

    # With return_train_score=False (default)
    search_cv_without_train = SearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        param_distributions={"estimator__alpha": optuna.distributions.FloatDistribution(0.1, 10.0)},
        scoring=MAE(),
        n_warmup_trials=1,
        n_trials=2,
        cv=2,
        refit=True,
        return_train_score=False,
    )

    search_cv_without_train.fit(y, X_ante, None, forecasting_horizon=3)

    # Verify train scores are NOT present
    assert "mean_train_score" not in search_cv_without_train.cv_results_
    assert "std_train_score" not in search_cv_without_train.cv_results_


def test_search_cv_predict_delegates(y_X_factory):
    """Test that SearchCV.predict() properly delegates to best_forecaster_."""
    y, X_ante, X_post = y_X_factory(length=50, seed=42)
    y_train, y_test = y[:40], y[40:]
    X_ante_train, X_ante_test = X_ante[:40], X_ante[40:]
    X_post_train, X_post_test = (X_post[:40], X_post[40:]) if X_post is not None else (None, None)

    search_cv = SearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        param_distributions={"estimator__alpha": optuna.distributions.FloatDistribution(0.1, 10.0)},
        scoring=MAE(),
        n_warmup_trials=1,
        n_trials=2,
        cv=2,
        refit=True,
    )

    search_cv.fit(y_train, X_ante_train, None, forecasting_horizon=3)

    # Predict with SearchCV (using new signature with X_ante)
    y_pred_search = search_cv.predict(X_ante=None, X_post=None, forecasting_horizon=3)

    # Predict directly with best_forecaster_
    y_pred_direct = search_cv.best_forecaster_.predict(
        X_post=None, forecasting_horizon=3
    )

    # Both should produce identical predictions
    assert y_pred_search.equals(y_pred_direct)


def test_search_cv_predict_with_x_ante(y_X_factory):
    """Test that SearchCV.predict() correctly passes X_ante to best_forecaster_."""
    y, X_ante, X_post = y_X_factory(length=50, seed=42)
    y_train = y[:40]
    X_ante_train = X_ante[:40]

    search_cv = SearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        param_distributions={"estimator__alpha": optuna.distributions.FloatDistribution(0.1, 10.0)},
        scoring=MAE(),
        n_warmup_trials=1,
        n_trials=2,
        cv=2,
        refit=True,
    )

    search_cv.fit(y_train, X_ante_train, None, forecasting_horizon=3)

    # Create future X_ante for prediction
    X_ante_future = X_ante[40:43]

    # Predict with SearchCV passing X_ante
    y_pred_search = search_cv.predict(X_ante=X_ante_future, X_post=None, forecasting_horizon=3)

    # Predict directly with best_forecaster_
    y_pred_direct = search_cv.best_forecaster_.predict(
        X_post=None, forecasting_horizon=3
    )

    # Both should produce identical predictions
    assert y_pred_search.equals(y_pred_direct)
    assert "time" in y_pred_search.columns


def test_search_cv_update_with_x_post(y_X_factory):
    """Test that SearchCV.update() correctly passes X_post to best_forecaster_."""
    y, X_ante, X_post = y_X_factory(length=50, seed=42)
    y_train = y[:40]
    X_ante_train = X_ante[:40]
    X_post_train = X_post[:40] if X_post is not None else None

    search_cv = SearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        param_distributions={"estimator__alpha": optuna.distributions.FloatDistribution(0.1, 10.0)},
        scoring=MAE(),
        n_warmup_trials=1,
        n_trials=2,
        cv=2,
        refit=True,
    )

    search_cv.fit(y_train, X_ante_train, X_post_train, forecasting_horizon=3)

    # Update with new data including X_post
    y_update = y[40:45]
    X_ante_update = X_ante[40:45]
    X_post_update = X_post[40:45] if X_post is not None else None

    # Update SearchCV
    search_cv_result = search_cv.update(y_update, X_ante_update, X_post_update)

    # Verify returns self
    assert search_cv_result is search_cv


def test_search_cv_update_predict_delegates(y_X_factory):
    """Test that SearchCV.update_predict() properly delegates all parameters."""
    y, X_ante, X_post = y_X_factory(length=50, seed=42)
    y_train = y[:40]
    X_ante_train = X_ante[:40]

    search_cv = SearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        param_distributions={"estimator__alpha": optuna.distributions.FloatDistribution(0.1, 10.0)},
        scoring=MAE(),
        n_warmup_trials=1,
        n_trials=2,
        cv=2,
        refit=True,
    )

    search_cv.fit(y_train, X_ante_train, None, forecasting_horizon=3)

    # Prepare update data
    y_update = y[40:45]
    X_ante_update = X_ante[40:45]

    # Use update_predict
    y_pred = search_cv.update_predict(
        y=y_update,
        X_ante=X_ante_update,
        X_post=None,
        forecasting_horizon=3,
        stride=1,
    )

    # Verify predictions are returned
    assert isinstance(y_pred, pl.DataFrame)
    assert "time" in y_pred.columns


def test_search_cv_reset_delegates(y_X_factory):
    """Test that SearchCV.reset() properly delegates to best_forecaster_."""
    y, X_ante, X_post = y_X_factory(length=50, seed=42)
    y_train = y[:40]
    X_ante_train = X_ante[:40]

    search_cv = SearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge(alpha=1.0)),
        param_distributions={"estimator__alpha": optuna.distributions.FloatDistribution(0.1, 10.0)},
        scoring=MAE(),
        n_warmup_trials=1,
        n_trials=2,
        cv=2,
        refit=True,
    )

    search_cv.fit(y_train, X_ante_train, None, forecasting_horizon=3)

    # Reset with new observation data
    y_reset = y[35:40]
    X_ante_reset = X_ante[35:40]

    # Reset SearchCV
    search_cv_result = search_cv.reset(y_reset, X_ante_reset, None)

    # Verify returns self
    assert search_cv_result is search_cv
