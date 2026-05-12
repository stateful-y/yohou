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
        """_HorizonAwareScorer should not raise during CV scoring."""
        scorer = _HorizonAwareScorer()
        cv = SlidingWindowSplitter(n_splits=2, test_size=10)

        search = GridSearchCV(
            SeasonalNaive(),
            param_grid={"seasonality": [7, 14]},
            scoring=scorer,
            cv=cv,
        )
        search.fit(daily_series, forecasting_horizon=5)

        assert hasattr(search, "best_params_")
        assert hasattr(search, "best_score_")
        assert search.best_params_["seasonality"] in (7, 14)

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
