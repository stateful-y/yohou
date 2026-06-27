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
from yohou.testing import (
    _yield_yohou_scorer_checks,
    check_scorer_component_subselection,
    check_scorer_panel_subselection,
)


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


@pytest.fixture
def panel_scorer_data():
    """Two-group panel y_truth and contract-shaped y_pred for subselection checks."""
    n = 20
    times = pl.datetime_range(
        pl.lit("2020-01-01").str.to_datetime(),
        pl.lit("2020-01-01").str.to_datetime() + pl.duration(days=n - 1),
        interval="1d",
        eager=True,
    )
    y_truth = pl.DataFrame({
        "time": times,
        "g1__value": [float(i) for i in range(n)],
        "g2__value": [float(i * 2) for i in range(n)],
    })
    y_pred = pl.DataFrame({
        "vintage_time": [times[0]] * n,
        "time": times,
        "g1__value": [float(i + 1) for i in range(n)],
        "g2__value": [float(i * 2 + 1) for i in range(n)],
    })
    return y_truth, y_pred


@pytest.mark.integration
@pytest.mark.parametrize("scorer_cls", [_MaxAbsoluteError, _RootMeanSquaredError])
class TestCustomScorerSubselection:
    """Cover panel/component subselection for custom scorers.

    check_scorer_panel_subselection and check_scorer_component_subselection are
    exported from yohou.testing but are not yielded by _yield_yohou_scorer_checks,
    so the test_systematic_checks tests below do not exercise groups/components
    filtering. These tests call those checks directly to close that gap, ensuring a
    custom scorer that crashed on groups=[...] or components=[...] would be caught.
    """

    def test_panel_subselection(self, panel_scorer_data, scorer_cls):
        y_truth, y_pred = panel_scorer_data
        scorer = scorer_cls()
        scorer.fit(y_truth)
        check_scorer_panel_subselection(scorer, y_truth, y_pred, groups=["g1"])

    def test_component_subselection(self, scorer_data, scorer_cls):
        y_truth, y_pred = scorer_data
        components = [c for c in y_truth.columns if c != "time"]
        scorer = scorer_cls()
        scorer.fit(y_truth)
        check_scorer_component_subselection(scorer, y_truth, y_pred, components=components)


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

    # NOTE: a uniform-error (all errors == 1.0) score_value test cannot distinguish
    # max from sqrt-mean (both equal 1.0), so it was removed. The differentiating
    # behavior is covered by test_score_asymmetric_errors (max picks the largest
    # error), and the no-crash/aggregation path by test_systematic_checks.

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

    # NOTE: a uniform-error (all errors == 1.0) score_value test cannot distinguish
    # sqrt-mean from max (both equal 1.0), so it was removed. The distinguishing
    # sqrt(mean(e^2)) behavior is covered by test_score_known_rmse, and the
    # no-crash/aggregation path by test_systematic_checks.

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
