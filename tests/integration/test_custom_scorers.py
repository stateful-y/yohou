"""Integration tests for custom scorer extensibility.

Verifies that custom ``_compute_raw_errors``, ``_collapse_rows``, and
``_transform_scores`` override patterns work end to end with systematic
check generators and manual assertions.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from conftest import run_checks
from yohou.metrics.base import BasePointScorer
from yohou.model_selection import GridSearchCV, SlidingWindowSplitter
from yohou.testing import _yield_yohou_scorer_checks

# ---------------------------------------------------------------------------
# Custom estimators
# ---------------------------------------------------------------------------


class _MaxAbsoluteError(BasePointScorer):
    """Max absolute error (custom _collapse_rows using agg_fn='max')."""

    _metric_name = "max_ae"

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
        return (y_truth - y_pred).select(pl.all().abs())

    def _collapse_rows(self, df, context, dims):
        return self._collapse_rows_with(df, context, dims, agg_fn="max")


class _RootMeanSquaredError(BasePointScorer):
    """RMSE (custom _transform_scores with sqrt)."""

    _metric_name = "rmse"

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
        return (y_truth - y_pred).select(pl.all().pow(2))

    def _transform_scores(self, df):
        return df.select(pl.all().sqrt())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scorer_data():
    """y_truth and y_pred for scorer testing."""
    n = 20
    times = pl.datetime_range(
        pl.lit("2020-01-01").str.to_datetime(),
        pl.lit("2020-01-01").str.to_datetime() + pl.duration(days=n - 1),
        interval="1d",
        eager=True,
    )
    y_truth = pl.DataFrame({
        "time": times,
        "value": list(range(n)),
    }).cast({"value": pl.Float64})

    y_pred = pl.DataFrame({
        "vintage_time": [times[0]] * n,
        "time": times,
        "value": [float(v + 1) for v in range(n)],
    })
    return y_truth, y_pred


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMaxAbsoluteError:
    """Verify custom scorer with overridden _collapse_rows."""

    def test_systematic_checks(self, scorer_data):
        y_truth, y_pred = scorer_data
        scorer = _MaxAbsoluteError()
        scorer.fit(y_truth)
        run_checks(
            scorer,
            _yield_yohou_scorer_checks(scorer, y_truth, y_pred),
        )

    def test_score_value(self, scorer_data):
        y_truth, y_pred = scorer_data
        scorer = _MaxAbsoluteError()
        scorer.fit(y_truth)
        result = scorer.score(y_truth, y_pred)
        # Every row has |actual - pred| = 1.0, max is 1.0
        assert result == pytest.approx(1.0)

    def test_score_asymmetric_errors(self):
        times = pl.datetime_range(
            pl.lit("2020-01-01").str.to_datetime(),
            pl.lit("2020-01-03").str.to_datetime(),
            interval="1d",
            eager=True,
        )
        y_truth = pl.DataFrame({"time": times, "value": [10.0, 20.0, 30.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [times[0]] * 3,
            "time": times,
            "value": [10.0, 20.0, 25.0],
        })
        scorer = _MaxAbsoluteError()
        scorer.fit(y_truth)
        result = scorer.score(y_truth, y_pred)
        assert result == pytest.approx(5.0)


@pytest.mark.integration
class TestRootMeanSquaredError:
    """Verify custom scorer with overridden _transform_scores."""

    def test_systematic_checks(self, scorer_data):
        y_truth, y_pred = scorer_data
        scorer = _RootMeanSquaredError()
        scorer.fit(y_truth)
        run_checks(
            scorer,
            _yield_yohou_scorer_checks(scorer, y_truth, y_pred),
        )

    def test_score_value(self, scorer_data):
        y_truth, y_pred = scorer_data
        scorer = _RootMeanSquaredError()
        scorer.fit(y_truth)
        result = scorer.score(y_truth, y_pred)
        # Every row has error = 1.0, so MSE = 1.0, RMSE = 1.0
        assert result == pytest.approx(1.0)

    def test_score_known_rmse(self):
        times = pl.datetime_range(
            pl.lit("2020-01-01").str.to_datetime(),
            pl.lit("2020-01-04").str.to_datetime(),
            interval="1d",
            eager=True,
        )
        y_truth = pl.DataFrame({"time": times, "value": [0.0, 0.0, 0.0, 0.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [times[0]] * 4,
            "time": times,
            "value": [1.0, 2.0, 3.0, 4.0],
        })
        scorer = _RootMeanSquaredError()
        scorer.fit(y_truth)
        result = scorer.score(y_truth, y_pred)
        expected = math.sqrt((1 + 4 + 9 + 16) / 4)
        assert result == pytest.approx(expected)


@pytest.mark.integration
class TestCustomScorerInSearch:
    """Verify custom scorer works as scoring in GridSearchCV."""

    def test_max_ae_in_grid_search(self):
        from yohou.point import SeasonalNaive

        rng = np.random.default_rng(42)
        n = 100
        y = pl.DataFrame({
            "time": pl.datetime_range(
                pl.lit("2020-01-01").str.to_datetime(),
                pl.lit("2020-01-01").str.to_datetime() + pl.duration(days=n - 1),
                interval="1d",
                eager=True,
            ),
            "value": rng.normal(10, 1, n).tolist(),
        })

        cv = SlidingWindowSplitter(n_splits=2, test_size=10)
        search = GridSearchCV(
            SeasonalNaive(),
            param_grid={"seasonality": [1, 7]},
            scoring=_MaxAbsoluteError(),
            cv=cv,
        )
        search.fit(y, forecasting_horizon=5)

        assert hasattr(search, "best_score_")
        assert np.isfinite(search.best_score_)
        assert "mean_test_score" in search.cv_results_
