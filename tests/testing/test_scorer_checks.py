"""Tests for scorer check functions in yohou.testing.scorer."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from yohou.interval import SplitConformalForecaster
from yohou.metrics.interval import EmpiricalCoverage
from yohou.metrics.point import MeanAbsoluteError
from yohou.point.naive import SeasonalNaive
from yohou.testing.scorer import (
    check_scorer_aggregation_methods,
    check_scorer_component_subselection,
    check_scorer_coverage_rate_subselection,
    check_scorer_lower_is_better,
    check_scorer_panel_subselection,
    check_scorer_prediction_type_compatibility,
)


@pytest.fixture(scope="module")
def scorer_panel_data():
    """Scorer with panel y_truth and point y_pred of matching length."""
    n = 10
    times = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )
    y_truth = pl.DataFrame({
        "time": times,
        "g1__val": [float(i) for i in range(n)],
        "g2__val": [float(i) * 2 for i in range(n)],
    })
    y_pred = pl.DataFrame({
        "observed_time": [times[0]] * n,
        "time": times,
        "g1__val": [float(i) + 0.5 for i in range(n)],
        "g2__val": [float(i) * 2 + 0.5 for i in range(n)],
    })
    scorer = MeanAbsoluteError()
    scorer.fit(y_truth)
    return scorer, y_truth, y_pred


@pytest.fixture(scope="module")
def interval_scorer_data():
    """Interval scorer with y_truth and interval y_pred."""
    n = 50
    times = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )
    y_truth = pl.DataFrame({
        "time": times,
        "val": [float(i) for i in range(n)],
    })
    y_pred_interval = pl.DataFrame({
        "observed_time": [times[-1]] * 5,
        "time": pl.datetime_range(
            start=datetime(2020, 1, 1) + timedelta(days=n),
            end=datetime(2020, 1, 1) + timedelta(days=n + 4),
            interval="1d",
            eager=True,
        ),
        "val": [float(i) for i in range(50, 55)],
        "val_lower_0.9": [float(i) - 2 for i in range(50, 55)],
        "val_upper_0.9": [float(i) + 2 for i in range(50, 55)],
        "coverage_rate": [0.9] * 5,
    })
    scorer = EmpiricalCoverage()
    scorer.fit(y_truth)
    return scorer, y_truth, y_pred_interval


@pytest.fixture(scope="module")
def fitted_point_forecaster():
    """Fitted point forecaster for compatibility checks."""
    n = 80
    times = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": times, "val": [float(i % 7) for i in range(n)]})
    forecaster = SeasonalNaive(seasonality=7)
    forecaster.fit(y, forecasting_horizon=5)
    return forecaster, y


class TestScorerCheckFunctions:
    """Tests that directly call scorer check functions from testing.scorer."""

    def test_check_lower_is_better(self):
        """check_scorer_lower_is_better validates boolean tag."""
        scorer = MeanAbsoluteError()
        check_scorer_lower_is_better(scorer)

    def test_check_aggregation_methods(self, scorer_panel_data):
        """check_scorer_aggregation_methods validates each method."""
        scorer, y_truth, y_pred = scorer_panel_data
        check_scorer_aggregation_methods(
            scorer, y_truth, y_pred, aggregation_methods=["stepwise", "vintagewise", "componentwise"]
        )

    def test_check_panel_subselection(self, scorer_panel_data):
        """check_scorer_panel_subselection clones internally (exposes unfitted clone bug)."""
        scorer, y_truth, y_pred = scorer_panel_data
        scorer_copy = MeanAbsoluteError()
        scorer_copy.fit(y_truth)
        with pytest.raises(AssertionError, match="filtering failed"):
            check_scorer_panel_subselection(scorer_copy, y_truth, y_pred, groups=["g1"])

    def test_check_component_subselection(self, scorer_panel_data):
        """check_scorer_component_subselection clones internally (exposes unfitted clone bug)."""
        scorer, y_truth, y_pred = scorer_panel_data
        scorer_copy = MeanAbsoluteError()
        scorer_copy.fit(y_truth)
        with pytest.raises(AssertionError, match="filtering failed"):
            check_scorer_component_subselection(scorer_copy, y_truth, y_pred, components=["val"])

    def test_check_coverage_rate_subselection(self, interval_scorer_data):
        """check_scorer_coverage_rate_subselection filters interval predictions."""
        scorer, y_truth, y_pred_interval = interval_scorer_data
        check_scorer_coverage_rate_subselection(scorer, y_truth, y_pred_interval, coverage_rates=[0.9])

    def test_check_coverage_rate_subselection_point_scorer_skips(self):
        """Point scorer skips coverage rate subselection check entirely."""
        n = 10
        times = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=n - 1),
            interval="1d",
            eager=True,
        )
        y_truth = pl.DataFrame({"time": times, "val": [float(i) for i in range(n)]})
        y_pred = pl.DataFrame({
            "observed_time": [times[0]] * n,
            "time": times,
            "val": [float(i) + 0.5 for i in range(n)],
        })
        scorer = MeanAbsoluteError()
        scorer.fit(y_truth)
        # For a point scorer, prediction_type != "interval" so the check returns early
        check_scorer_coverage_rate_subselection(scorer, y_truth, y_pred, coverage_rates=[0.9])

    def test_check_prediction_type_compatibility(self, fitted_point_forecaster):
        """check_scorer_prediction_type_compatibility pairs scorer with forecaster."""
        forecaster, y = fitted_point_forecaster
        scorer = MeanAbsoluteError()
        scorer.fit(y)
        check_scorer_prediction_type_compatibility(scorer, forecaster, y)


class TestScorerCheckFunctionsInterval:
    """Tests that call scorer check functions with interval scorer."""

    @pytest.fixture(scope="class")
    def interval_forecaster_and_data(self):
        """Fitted interval forecaster with y_truth and interval y_pred."""
        n = 200
        times = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=n - 1),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": times, "val": [float(i % 7) for i in range(n)]})
        forecaster = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
        )
        forecaster.fit(y, forecasting_horizon=5)
        y_pred_interval = forecaster.predict_interval(
            forecasting_horizon=5,
            coverage_rates=[0.9],
        )
        return forecaster, y, y_pred_interval

    def test_interval_prediction_type_compatibility(self, interval_forecaster_and_data):
        """Interval scorer paired with interval forecaster succeeds."""
        forecaster, y, _ = interval_forecaster_and_data
        scorer = EmpiricalCoverage()
        scorer.fit(y)
        check_scorer_prediction_type_compatibility(scorer, forecaster, y)

    def test_interval_aggregation_methods(self):
        """Point scorer stepwise+vintagewise aggregation returns DataFrame (covers DataFrame branch)."""
        n = 10
        times = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=n - 1),
            interval="1d",
            eager=True,
        )
        y_truth = pl.DataFrame({
            "time": times,
            "a": [float(i) for i in range(n)],
            "b": [float(i) * 2 for i in range(n)],
        })
        y_pred = pl.DataFrame({
            "observed_time": [times[0]] * n,
            "time": times,
            "a": [float(i) + 0.5 for i in range(n)],
            "b": [float(i) * 2 + 0.5 for i in range(n)],
        })
        scorer = MeanAbsoluteError()
        scorer.fit(y_truth)
        check_scorer_aggregation_methods(
            scorer,
            y_truth,
            y_pred,
            aggregation_methods=["stepwise", "vintagewise"],
        )


class TestScorerCheckFunctionsClassProba:
    """Tests that call scorer check functions with class-proba scorer."""

    @pytest.fixture(scope="class")
    def class_proba_forecaster_and_data(self):
        """Fitted class-proba forecaster with y_truth."""
        from sklearn.linear_model import LogisticRegression

        from yohou.class_proba import ClassProbaReductionForecaster

        n = 80
        times = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=n - 1),
            interval="1d",
            eager=True,
        )
        labels = ["sun", "rain", "cloud"]
        y = pl.DataFrame({
            "time": times,
            "weather": [labels[i % 3] for i in range(n)],
        })
        forecaster = ClassProbaReductionForecaster(
            estimator=LogisticRegression(max_iter=200),
        )
        forecaster.fit(y, forecasting_horizon=1)
        return forecaster, y

    def test_class_proba_prediction_type_compatibility(self, class_proba_forecaster_and_data):
        """Class-proba scorer paired with class-proba forecaster succeeds."""
        from yohou.metrics.class_proba import LogLoss

        forecaster, y = class_proba_forecaster_and_data
        scorer = LogLoss()
        scorer.fit(y)
        check_scorer_prediction_type_compatibility(scorer, forecaster, y)
