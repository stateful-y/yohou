"""Tests for cross-validation functions and _fit_and_score changes."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from yohou.metrics.point import MeanAbsoluteError, MeanSquaredError
from yohou.model_selection import cross_val_predict, cross_val_score, cross_validate
from yohou.model_selection.utils import (
    _fit_and_score,
    _predict,
)
from yohou.point.naive import SeasonalNaive


@pytest.fixture()
def cv_data():
    """Dataset for cross-validation tests."""
    length = 50
    fh = 5
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
        interval="1s",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "target": [float(i) for i in range(length)]})
    return y, fh


@pytest.fixture()
def fit_and_score_data():
    """Dataset with train/test indices for _fit_and_score tests."""
    length = 30
    fh = 5
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
        interval="1s",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "target": [float(i) for i in range(length)]})
    train = np.arange(0, length - fh)
    test = np.arange(length - fh, length)
    return y, train, test, fh


class TestCrossValidateSingleMetric:
    """Tests for cross_validate with a single scorer."""

    def test_returns_dataframe_with_expected_columns(self, cv_data):
        """Result DataFrame contains split, test_score, fit_time, score_time."""
        y, fh = cv_data
        result = cross_validate(
            SeasonalNaive(seasonality=1),
            y,
            forecasting_horizon=fh,
            scoring=MeanAbsoluteError(),
            cv=3,
        )
        assert isinstance(result, pl.DataFrame)
        assert "split" in result.columns
        assert "test_score" in result.columns
        assert "fit_time" in result.columns
        assert "score_time" in result.columns

    def test_row_count_matches_splits(self, cv_data):
        """DataFrame has one row per fold."""
        y, fh = cv_data
        n_splits = 3
        result = cross_validate(
            SeasonalNaive(seasonality=1),
            y,
            forecasting_horizon=fh,
            scoring=MeanAbsoluteError(),
            cv=n_splits,
        )
        assert isinstance(result, pl.DataFrame)
        assert len(result) == n_splits
        assert result["split"].to_list() == list(range(n_splits))

    def test_scores_are_negated_for_lower_is_better(self, cv_data):
        """MAE (lower_is_better=True) scores are negated."""
        y, fh = cv_data
        result = cross_validate(
            SeasonalNaive(seasonality=1),
            y,
            forecasting_horizon=fh,
            scoring=MeanAbsoluteError(),
            cv=3,
        )
        # MAE is lower_is_better, so scores should be negated
        assert all(s <= 0 for s in result["test_score"].to_list())


class TestCrossValidateMultiMetric:
    """Tests for cross_validate with multiple scorers."""

    def test_multi_metric_columns(self, cv_data):
        """Result DataFrame contains test_{name} columns for each scorer."""
        y, fh = cv_data
        result = cross_validate(
            SeasonalNaive(seasonality=1),
            y,
            forecasting_horizon=fh,
            scoring={"mae": MeanAbsoluteError(), "mse": MeanSquaredError()},
            cv=3,
        )
        assert isinstance(result, pl.DataFrame)
        assert "test_mae" in result.columns
        assert "test_mse" in result.columns
        assert "test_score" not in result.columns

    def test_multi_metric_row_count(self, cv_data):
        """Each metric column has one value per fold."""
        y, fh = cv_data
        result = cross_validate(
            SeasonalNaive(seasonality=1),
            y,
            forecasting_horizon=fh,
            scoring={"mae": MeanAbsoluteError(), "mse": MeanSquaredError()},
            cv=3,
        )
        assert len(result) == 3


class TestCrossValidateReturnOptions:
    """Tests for cross_validate return_* flags."""

    def test_return_train_score(self, cv_data):
        """return_train_score=True includes train_score column."""
        y, fh = cv_data
        result = cross_validate(
            SeasonalNaive(seasonality=1),
            y,
            forecasting_horizon=fh,
            scoring=MeanAbsoluteError(),
            cv=3,
            return_train_score=True,
        )
        assert isinstance(result, pl.DataFrame)
        assert "train_score" in result.columns
        assert len(result) == 3

    def test_return_train_score_multi_metric(self, cv_data):
        """return_train_score=True with multi-metric includes train_{name} columns."""
        y, fh = cv_data
        result = cross_validate(
            SeasonalNaive(seasonality=1),
            y,
            forecasting_horizon=fh,
            scoring={"mae": MeanAbsoluteError(), "mse": MeanSquaredError()},
            cv=3,
            return_train_score=True,
        )
        assert isinstance(result, pl.DataFrame)
        assert "train_mae" in result.columns
        assert "train_mse" in result.columns

    def test_return_forecaster(self, cv_data):
        """return_forecaster=True returns dict with results DataFrame and forecaster list."""
        y, fh = cv_data
        result = cross_validate(
            SeasonalNaive(seasonality=1),
            y,
            forecasting_horizon=fh,
            scoring=MeanAbsoluteError(),
            cv=3,
            return_forecaster=True,
        )
        assert isinstance(result, dict)
        assert isinstance(result["results"], pl.DataFrame)
        assert "forecaster" in result
        assert len(result["forecaster"]) == 3
        assert all(isinstance(f, SeasonalNaive) for f in result["forecaster"])

    def test_return_indices(self, cv_data):
        """return_indices=True returns dict with results DataFrame and indices."""
        y, fh = cv_data
        result = cross_validate(
            SeasonalNaive(seasonality=1),
            y,
            forecasting_horizon=fh,
            scoring=MeanAbsoluteError(),
            cv=3,
            return_indices=True,
        )
        assert isinstance(result, dict)
        assert isinstance(result["results"], pl.DataFrame)
        assert "indices" in result
        assert "train" in result["indices"]
        assert "test" in result["indices"]
        assert len(result["indices"]["train"]) == 3
        assert len(result["indices"]["test"]) == 3


class TestCrossValidateInputValidation:
    """Tests for cross_validate input validation."""

    def test_scoring_required(self, cv_data):
        """Omitting scoring raises TypeError."""
        y, fh = cv_data
        with pytest.raises(TypeError):
            cross_validate(
                SeasonalNaive(seasonality=1),
                y,
                forecasting_horizon=fh,
                cv=3,
            )


class TestCrossValidatePredictParams:
    """Tests for custom predict_forecasting_horizon and predict_stride."""

    def test_custom_predict_forecasting_horizon(self, cv_data):
        """Custom predict_forecasting_horizon does not error."""
        y, fh = cv_data
        result = cross_validate(
            SeasonalNaive(seasonality=1),
            y,
            forecasting_horizon=fh,
            scoring=MeanAbsoluteError(),
            cv=3,
            predict_forecasting_horizon=2,
        )
        assert len(result) == 3

    def test_custom_predict_stride(self, cv_data):
        """Custom predict_stride does not error."""
        y, fh = cv_data
        result = cross_validate(
            SeasonalNaive(seasonality=1),
            y,
            forecasting_horizon=fh,
            scoring=MeanAbsoluteError(),
            cv=3,
            predict_stride=1,
        )
        assert len(result) == 3


class TestCrossValScore:
    """Tests for cross_val_score."""

    def test_returns_dataframe(self, cv_data):
        """cross_val_score returns a DataFrame with split and score columns."""
        y, fh = cv_data
        scores = cross_val_score(
            SeasonalNaive(seasonality=1),
            y,
            forecasting_horizon=fh,
            scoring=MeanAbsoluteError(),
            cv=3,
        )
        assert isinstance(scores, pl.DataFrame)
        assert scores.columns == ["split", "score"]
        assert len(scores) == 3
        assert scores["split"].to_list() == [0, 1, 2]
        assert scores["score"].dtype == pl.Float64

    def test_multi_metric_rejected(self, cv_data):
        """Dict scoring raises ValueError."""
        y, fh = cv_data
        with pytest.raises(ValueError, match="does not accept dict"):
            cross_val_score(
                SeasonalNaive(seasonality=1),
                y,
                forecasting_horizon=fh,
                scoring={"mae": MeanAbsoluteError()},
                cv=3,
            )

    def test_scoring_required(self, cv_data):
        """Omitting scoring raises TypeError."""
        y, fh = cv_data
        with pytest.raises(TypeError):
            cross_val_score(
                SeasonalNaive(seasonality=1),
                y,
                forecasting_horizon=fh,
                cv=3,
            )

    def test_lower_is_better_negation(self, cv_data):
        """MAE scores are negated (lower_is_better)."""
        y, fh = cv_data
        scores = cross_val_score(
            SeasonalNaive(seasonality=1),
            y,
            forecasting_horizon=fh,
            scoring=MeanAbsoluteError(),
            cv=3,
        )
        assert all(s <= 0 for s in scores["score"].to_list())


class TestCrossValPredict:
    """Tests for cross_val_predict."""

    def test_returns_dataframe_with_split_column(self, cv_data):
        """Result is a DataFrame with a 'split' column."""
        y, fh = cv_data
        preds = cross_val_predict(
            SeasonalNaive(seasonality=1),
            y,
            forecasting_horizon=fh,
            cv=3,
        )
        assert isinstance(preds, pl.DataFrame)
        assert "split" in preds.columns
        assert "time" in preds.columns

    def test_split_column_values(self, cv_data):
        """split column contains 0-based fold indices."""
        y, fh = cv_data
        n_splits = 3
        preds = cross_val_predict(
            SeasonalNaive(seasonality=1),
            y,
            forecasting_horizon=fh,
            cv=n_splits,
        )
        unique_splits = sorted(preds["split"].unique().to_list())
        assert unique_splits == list(range(n_splits))

    def test_point_predictions_columns(self, cv_data):
        """Point predictions contain vintage_time, time, target, split."""
        y, fh = cv_data
        preds = cross_val_predict(
            SeasonalNaive(seasonality=1),
            y,
            forecasting_horizon=fh,
            cv=3,
            method="predict",
        )
        assert "vintage_time" in preds.columns
        assert "time" in preds.columns
        assert "target" in preds.columns
        assert "split" in preds.columns

    def test_overlapping_test_sets_preserved(self, cv_data):
        """Overlapping predictions from different folds are all present."""
        y, fh = cv_data
        preds = cross_val_predict(
            SeasonalNaive(seasonality=1),
            y,
            forecasting_horizon=fh,
            cv=3,
        )
        # With expanding window and overlapping test folds, we should
        # have more rows than the original y (or at least the test portions)
        n_rows = len(preds)
        # Each fold produces predictions; with 3 folds and potential overlap
        # we should have predictions distinguishable by split
        per_split = preds.group_by("split").len()
        assert len(per_split) == 3
        assert n_rows == per_split["len"].sum()

    def test_invalid_method_raises(self, cv_data):
        """Invalid method raises ValueError."""
        y, fh = cv_data
        with pytest.raises(ValueError, match="method must be one of"):
            cross_val_predict(
                SeasonalNaive(seasonality=1),
                y,
                forecasting_horizon=fh,
                cv=3,
                method="invalid",
            )


class TestFitAndScoreReturnPredictions:
    """Tests for _fit_and_score return_predictions parameter."""

    def test_train_score_params_match_predicted_rows(self, fit_and_score_data):
        """Train-side score_params must align with the scored train-test rows.

        ``score_params`` were previously sliced with the full ``train`` indices
        (len(train) entries) while ``_score`` evaluates only the last len(test)
        rows of the training window, producing a length mismatch.
        """
        y, train, test, fh = fit_and_score_data

        captured = {}

        class _RecordingMAE(MeanAbsoluteError):
            def __call__(self, y_truth, y_pred, **params):
                if "sample_weight" in params:
                    captured.setdefault("lengths", []).append(len(params["sample_weight"]))
                params.pop("sample_weight", None)
                return super().__call__(y_truth, y_pred, **params)

        sample_weight = np.arange(len(y), dtype=float)

        _fit_and_score(
            SeasonalNaive(seasonality=1),
            y,
            None,
            fh,
            scorer=_RecordingMAE(),
            train=train,
            test=test,
            verbose=0,
            parameters=None,
            fit_params=None,
            predict_func_params=None,
            score_params={"sample_weight": sample_weight},
            return_train_score=True,
        )

        # Both the test scoring and the train scoring evaluate len(test) rows,
        # so every captured weight array must have len(test) entries.
        assert captured["lengths"]
        assert all(length == len(test) for length in captured["lengths"])

    def test_return_predictions_with_scorer(self, fit_and_score_data):
        """return_predictions=True with scorer returns both scores and predictions."""
        y, train, test, fh = fit_and_score_data
        result = _fit_and_score(
            SeasonalNaive(seasonality=1),
            y,
            None,
            fh,
            scorer=MeanAbsoluteError(),
            train=train,
            test=test,
            verbose=0,
            parameters=None,
            fit_params=None,
            predict_func_params=None,
            score_params=None,
            return_predictions=True,
        )
        assert "predictions" in result
        assert isinstance(result["predictions"], pl.DataFrame)
        assert "test_scores" in result

    def test_return_predictions_with_scorer_none(self, fit_and_score_data):
        """return_predictions=True with scorer=None returns predictions without scores."""
        y, train, test, fh = fit_and_score_data
        result = _fit_and_score(
            SeasonalNaive(seasonality=1),
            y,
            None,
            fh,
            scorer=None,
            train=train,
            test=test,
            verbose=0,
            parameters=None,
            fit_params=None,
            predict_func_params=None,
            score_params=None,
            return_predictions=True,
            predict_method="predict",
        )
        assert "predictions" in result
        assert isinstance(result["predictions"], pl.DataFrame)
        assert "test_scores" not in result

    def test_scorer_none_without_return_predictions(self, fit_and_score_data):
        """scorer=None with return_predictions=False produces no predictions or scores."""
        y, train, test, fh = fit_and_score_data
        result = _fit_and_score(
            SeasonalNaive(seasonality=1),
            y,
            None,
            fh,
            scorer=None,
            train=train,
            test=test,
            verbose=0,
            parameters=None,
            fit_params=None,
            predict_func_params=None,
            score_params=None,
            return_predictions=False,
        )
        assert "predictions" not in result
        assert "test_scores" not in result

    def test_predictions_dataframe_structure(self, fit_and_score_data):
        """Predictions DataFrame contains expected columns."""
        y, train, test, fh = fit_and_score_data
        result = _fit_and_score(
            SeasonalNaive(seasonality=1),
            y,
            None,
            fh,
            scorer=None,
            train=train,
            test=test,
            verbose=0,
            parameters=None,
            fit_params=None,
            predict_func_params=None,
            score_params=None,
            return_predictions=True,
            predict_method="predict",
        )
        preds = result["predictions"]
        assert "vintage_time" in preds.columns
        assert "time" in preds.columns
        assert "target" in preds.columns


class TestPredictFunction:
    """Tests for the _predict standalone function."""

    def test_predict_with_scorer(self):
        """_predict uses scorer to resolve response method."""
        length = 30
        fh = 5
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
            interval="1s",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "target": [float(i) for i in range(length)]})
        y_train = y.head(length - fh)
        y_test = y.tail(fh)

        forecaster = SeasonalNaive(seasonality=1)
        forecaster.fit(y_train, forecasting_horizon=fh)

        y_pred = _predict(forecaster, y_test, None, MeanAbsoluteError())
        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns

    def test_predict_with_method(self):
        """_predict uses explicit method parameter."""
        length = 30
        fh = 5
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
            interval="1s",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "target": [float(i) for i in range(length)]})
        y_train = y.head(length - fh)
        y_test = y.tail(fh)

        forecaster = SeasonalNaive(seasonality=1)
        forecaster.fit(y_train, forecasting_horizon=fh)

        y_pred = _predict(forecaster, y_test, None, method="predict")
        assert isinstance(y_pred, pl.DataFrame)
        assert "time" in y_pred.columns

    def test_predict_no_scorer_no_method_raises(self):
        """_predict raises when neither scorer nor method is provided."""
        forecaster = MagicMock()
        y_test = pl.DataFrame({"time": [datetime(2021, 1, 1)], "target": [1.0]})
        with pytest.raises(ValueError, match="Either scorer or method"):
            _predict(forecaster, y_test, None)
