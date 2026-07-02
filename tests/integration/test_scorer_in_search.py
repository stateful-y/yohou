"""Integration test: scorer.fit() receives the fitted forecaster during CV scoring.

Verifies that ``_score()`` passes ``forecaster=forecaster`` to ``scorer.fit()``,
so scorers relying on ``forecaster_horizon_`` or ``interval_`` work correctly
inside ``GridSearchCV`` cross-validation folds.
"""

import polars as pl
import pytest
from sklearn.utils.validation import check_is_fitted

from yohou.metrics.base import BasePointScorer
from yohou.model_selection import GridSearchCV, SlidingWindowSplitter
from yohou.point import SeasonalNaive


class _HorizonAwareScorer(BasePointScorer):
    """Scorer that requires ``forecaster_horizon_`` from the forecaster.

    Raises ``AttributeError`` during ``score()`` if ``fit()`` was not called
    with a forecaster (because ``forecaster_horizon_`` would be missing).
    """

    _parameter_constraints: dict = {
        **BasePointScorer._parameter_constraints,
    }

    _metric_name = "horizon_aware_mae"

    def __init__(
        self,
        aggregation_method="all",
        groups=None,
        components=None,
    ):
        super().__init__(
            aggregation_method=aggregation_method,
            groups=groups,
            components=components,
        )

    def _compute_raw_errors(self, y_truth, y_pred):
        # Access forecaster_horizon_ to verify it was set by fit(forecaster=...)
        check_is_fitted(self, ["forecaster_horizon_"])
        _ = self.forecaster_horizon_
        return (y_truth - y_pred).select(pl.all().abs())


@pytest.fixture
def daily_series():
    n = 100
    return pl.DataFrame({
        "time": pl.datetime_range(
            pl.lit("2020-01-01").str.to_datetime(),
            pl.lit("2020-01-01").str.to_datetime() + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        ),
        "value": [float(i % 7 + 1) for i in range(n)],
    })


@pytest.mark.integration
class TestScorerReceivesForecasterInCV:
    """Verify scorer.fit() receives the fitted forecaster during CV folds."""

    def test_single_metric_scorer_gets_forecaster_horizon(self, daily_series):
        """search.fit() completes without _HorizonAwareScorer raising NotFittedError.

        _HorizonAwareScorer._compute_raw_errors asserts forecaster_horizon_ was set,
        which only happens if _score() passed forecaster=... to scorer.fit() inside
        each CV fold. If routing were broken, fit() would raise here. (Generic
        attribute presence is already covered by check_search_fit_sets_attributes.)
        """
        scorer = _HorizonAwareScorer()
        cv = SlidingWindowSplitter(n_splits=2, test_size=10)

        search = GridSearchCV(
            SeasonalNaive(),
            param_grid={"seasonality": [7, 14]},
            scoring=scorer,
            cv=cv,
        )
        # Must not raise: every fold scored the forecaster-aware scorer successfully.
        search.fit(daily_series, forecasting_horizon=5)

        # The selected scorer_ carries the forecaster horizon from the refit fold,
        # proving the forecaster reached scorer.fit() (not merely that fit ran).
        assert search.scorer_.forecaster_horizon_ == 5

    def test_scorer_in_cv_with_refit_false(self, daily_series):
        """With refit=False, CV scoring still receives the forecaster per fold.

        Exercises the refit=False branch of GridSearchCV._fit, where the
        ``if self.refit:`` block (which refits best_forecaster_ and re-fits the
        scorer with it) is skipped. The forecaster-aware scorer must still succeed
        during the CV folds (otherwise fit() would raise), and the no-refit contract
        must hold: best_params_/best_score_ are set but best_forecaster_ is not, so
        predict() is unavailable.
        """
        scorer = _HorizonAwareScorer()
        cv = SlidingWindowSplitter(n_splits=2, test_size=10)

        search = GridSearchCV(
            SeasonalNaive(),
            param_grid={"seasonality": [7, 14]},
            scoring=scorer,
            cv=cv,
            refit=False,
        )
        # Must not raise: each fold scored the forecaster-aware scorer successfully.
        search.fit(daily_series, forecasting_horizon=5)

        assert hasattr(search, "best_params_")
        assert hasattr(search, "best_score_")
        # refit=False skips the refit block, so no best_forecaster_ is produced.
        assert not hasattr(search, "best_forecaster_")
        with pytest.raises(AttributeError):
            search.predict(forecasting_horizon=3)

    def test_scorer_has_forecaster_horizon_after_refit(self, daily_series):
        """After refit, scorer_ should have forecaster_horizon_ from best_forecaster_."""
        scorer = _HorizonAwareScorer()
        cv = SlidingWindowSplitter(n_splits=2, test_size=10)

        search = GridSearchCV(
            SeasonalNaive(),
            param_grid={"seasonality": [7]},
            scoring=scorer,
            cv=cv,
        )
        search.fit(daily_series, forecasting_horizon=5)

        assert hasattr(search.scorer_, "forecaster_horizon_")
        assert search.scorer_.forecaster_horizon_ == 5

    def test_multi_metric_scorer_gets_forecaster_horizon(self, daily_series):
        """Multi-metric dict scoring should propagate forecaster to each scorer."""
        scoring = {
            "horizon_mae": _HorizonAwareScorer(),
        }
        cv = SlidingWindowSplitter(n_splits=2, test_size=10)

        search = GridSearchCV(
            SeasonalNaive(),
            param_grid={"seasonality": [7, 14]},
            scoring=scoring,
            refit="horizon_mae",
            cv=cv,
        )
        search.fit(daily_series, forecasting_horizon=5)

        assert hasattr(search, "best_params_")
        # scorer_ is a dict for multi-metric
        assert "horizon_mae" in search.scorer_
        assert hasattr(search.scorer_["horizon_mae"], "forecaster_horizon_")
        assert search.scorer_["horizon_mae"].forecaster_horizon_ == 5
