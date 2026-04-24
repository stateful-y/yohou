"""Tests for similarity measures in interval forecasting."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone

from yohou.interval.similarity import CompositeSimilarity, DistanceSimilarity, TemporalSimilarity


@pytest.fixture
def train_data():
    """Create training data for similarity tests."""
    time_train = pl.datetime_range(
        start=datetime(2021, 12, 16),
        end=datetime(2021, 12, 16, 0, 0, 7),
        interval="1s",
        eager=True,
    )
    y = pl.DataFrame({
        "time": time_train,
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    })
    y_pred = pl.DataFrame({
        "time": time_train,
        "value": [1.1, 2.1, 2.9, 4.2, 4.8, 6.1, 7.0, 8.1],
    })
    return y, y_pred


@pytest.fixture
def prediction_data():
    """Create test data for similarity prediction."""
    time_test = pl.datetime_range(
        start=datetime(2021, 12, 16, 0, 0, 8),
        end=datetime(2021, 12, 16, 0, 0, 9),
        interval="1s",
        eager=True,
    )
    y_pred = pl.DataFrame({
        "time": time_test,
        "value": [8.5, 9.2],
    })
    return y_pred


class TestDistanceSimilarityBasic:
    """Basic tests for DistanceSimilarity."""

    def test_fit_returns_self(self, train_data):
        """Test that fit returns the estimator."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        result = sim.fit(y, y_pred)
        assert result is sim

    def test_predict_returns_ndarray(self, train_data, prediction_data):
        """Test that predict returns a numpy array."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        sim.fit(y, y_pred)
        weights = sim.predict(prediction_data)
        assert isinstance(weights, np.ndarray)

    def test_predict_shape(self, train_data, prediction_data):
        """Test that predict returns correct shape."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        sim.fit(y, y_pred)
        weights = sim.predict(prediction_data)
        # Shape: (n_test_samples, n_train_samples)
        assert weights.shape == (2, 8)

    def test_weights_are_positive(self, train_data, prediction_data):
        """Test that all weights are positive."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        sim.fit(y, y_pred)
        weights = sim.predict(prediction_data)
        assert np.all(weights > 0)

    def test_weights_are_finite(self, train_data, prediction_data):
        """Test that all weights are finite."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        sim.fit(y, y_pred)
        weights = sim.predict(prediction_data)
        assert np.all(np.isfinite(weights))

    def test_default_metric(self):
        """Test default metric is euclidean."""
        sim = DistanceSimilarity()
        assert sim.metric == "euclidean"

    def test_default_metric_params(self):
        """Test default metric_params is empty dict."""
        sim = DistanceSimilarity()
        assert sim.metric_params == {}


class TestDistanceSimilarityMetrics:
    """Tests for different distance metrics."""

    @pytest.mark.parametrize("metric", ["euclidean", "cityblock", "cosine"])
    def test_different_metrics(self, train_data, prediction_data, metric):
        """Test that different metrics produce valid weights."""
        y, y_pred = train_data
        sim = DistanceSimilarity(metric=metric)
        sim.fit(y, y_pred)
        weights = sim.predict(prediction_data)
        assert weights.shape == (2, 8)
        assert np.all(np.isfinite(weights))

    def test_custom_metric_params(self, train_data, prediction_data):
        """Test with custom metric parameters."""
        y, y_pred = train_data
        sim = DistanceSimilarity(metric="minkowski", metric_params={"p": 3})
        sim.fit(y, y_pred)
        weights = sim.predict(prediction_data)
        assert weights.shape == (2, 8)

    def test_different_metrics_give_different_weights(self, train_data, prediction_data):
        """Test that different metrics produce valid but potentially different weights."""
        y, y_pred = train_data

        sim_euclidean = DistanceSimilarity(metric="euclidean")
        sim_euclidean.fit(y, y_pred)
        w_euclidean = sim_euclidean.predict(prediction_data)

        sim_cityblock = DistanceSimilarity(metric="cityblock")
        sim_cityblock.fit(y, y_pred)
        w_cityblock = sim_cityblock.predict(prediction_data)

        # Both should produce valid weights
        assert np.all(np.isfinite(w_euclidean))
        assert np.all(np.isfinite(w_cityblock))


class TestDistanceSimilarityObserve:
    """Tests for observe method."""

    def test_observe_returns_self(self, train_data):
        """Test that observe returns the estimator."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        sim.fit(y, y_pred)
        result = sim.observe(y[:3], y_pred[:3])
        assert result is sim

    def test_observe_extends_observations(self, train_data, prediction_data):
        """Test that observe adds new observations."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        sim.fit(y, y_pred)

        # Get initial weights
        weights_before = sim.predict(prediction_data)

        # Observe new data
        sim.observe(y[:2], y_pred[:2])

        # Weights shape should change (more training points)
        weights_after = sim.predict(prediction_data)
        assert weights_after.shape[1] == weights_before.shape[1] + 2


class TestDistanceSimilarityWithExogenous:
    """Tests for similarity with exogenous features."""

    def test_fit_with_X(self, train_data):
        """Test fitting with exogenous features."""
        y, y_pred = train_data
        # X should NOT have a 'time' column to avoid duplicate with y_pred
        X = pl.DataFrame({
            "feature": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        })
        sim = DistanceSimilarity()
        sim.fit(y, y_pred, X=X)
        assert sim._X_observed.shape[1] > 1  # time + value + feature cols

    def test_predict_with_X(self, train_data, prediction_data):
        """Test prediction with exogenous features."""
        y, y_pred = train_data
        X_train = pl.DataFrame({
            "feature": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        })
        X_test = pl.DataFrame({
            "feature": [85.0, 92.0],
        })
        sim = DistanceSimilarity()
        sim.fit(y, y_pred, X=X_train)
        weights = sim.predict(prediction_data, X=X_test)
        assert weights.shape == (2, 8)


class TestDistanceSimilarityProperties:
    """Tests for properties and attributes."""

    def test_clone(self):
        """Test that DistanceSimilarity can be cloned."""
        sim = DistanceSimilarity(metric="cityblock")
        cloned = clone(sim)
        assert cloned.metric == "cityblock"
        assert cloned is not sim

    def test_get_params(self):
        """Test get_params returns expected parameters."""
        sim = DistanceSimilarity(metric="cosine", metric_params={"p": 2})
        params = sim.get_params()
        assert params["metric"] == "cosine"
        assert params["metric_params"] == {"p": 2}

    def test_set_params(self):
        """Test set_params updates parameters."""
        sim = DistanceSimilarity()
        sim.set_params(metric="cityblock")
        assert sim.metric == "cityblock"

    def test_tags(self):
        """Test sklearn tags are set correctly."""
        sim = DistanceSimilarity()
        tags = sim.__sklearn_tags__()
        assert tags.estimator_type == "similarity"
        assert tags.requires_fit is True


@pytest.fixture
def daily_data():
    """Create 10 weeks of daily data with weekly seasonality."""
    n = 70
    dates = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(n)]
    np.random.seed(42)
    values = [10.0 + 5.0 * np.sin(2 * np.pi * i / 7) for i in range(n)]
    y = pl.DataFrame({"time": dates, "value": values})
    y_pred = pl.DataFrame({"time": dates, "value": [v + 0.1 for v in values]})
    return y, y_pred


@pytest.fixture
def daily_prediction_data():
    """Create a single prediction row for the 71st day."""
    dates = [datetime(2021, 1, 1) + timedelta(days=70)]
    y_pred = pl.DataFrame({"time": dates, "value": [10.5]})
    return y_pred


class TestDistanceSimilarityNullRejection:
    """Tests that DistanceSimilarity rejects null and NaN data."""

    def test_fit_rejects_null_y_pred(self, train_data):
        """Test that fit raises ValueError when y_pred contains null."""
        y, y_pred = train_data
        y_pred_null = y_pred.with_columns(
            pl.when(pl.col("value") > 5).then(None).otherwise(pl.col("value")).alias("value")
        )
        sim = DistanceSimilarity()
        with pytest.raises(ValueError, match="null or NaN"):
            sim.fit(y, y_pred_null)

    def test_fit_rejects_nan_y_pred(self, train_data):
        """Test that fit raises ValueError when y_pred contains NaN."""
        y, y_pred = train_data
        y_pred_nan = y_pred.with_columns(
            pl.when(pl.col("value") > 5).then(float("nan")).otherwise(pl.col("value")).alias("value")
        )
        sim = DistanceSimilarity()
        with pytest.raises(ValueError, match="null or NaN"):
            sim.fit(y, y_pred_nan)

    def test_observe_rejects_null(self, train_data):
        """Test that observe raises ValueError when data contains null."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        sim.fit(y, y_pred)
        y_null = y.head(1).with_columns(pl.lit(None, dtype=pl.Float64).alias("value"))
        y_pred_null = y_pred.head(1).with_columns(pl.lit(None, dtype=pl.Float64).alias("value"))
        with pytest.raises(ValueError, match="null or NaN"):
            sim.observe(y_null, y_pred_null)

    def test_observe_rejects_nan(self, train_data):
        """Test that observe raises ValueError when data contains NaN."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        sim.fit(y, y_pred)
        y_pred_nan = y_pred.head(1).with_columns(pl.lit(float("nan")).alias("value"))
        with pytest.raises(ValueError, match="null or NaN"):
            sim.observe(y.head(1), y_pred_nan)

    def test_predict_rejects_null(self, train_data):
        """Test that predict raises ValueError when data contains null."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        sim.fit(y, y_pred)
        y_pred_null = y_pred.head(1).with_columns(pl.lit(None, dtype=pl.Float64).alias("value"))
        with pytest.raises(ValueError, match="null or NaN"):
            sim.predict(y_pred_null)

    def test_predict_rejects_nan(self, train_data):
        """Test that predict raises ValueError when data contains NaN."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        sim.fit(y, y_pred)
        y_pred_nan = y_pred.head(1).with_columns(pl.lit(float("nan")).alias("value"))
        with pytest.raises(ValueError, match="null or NaN"):
            sim.predict(y_pred_nan)

    def test_error_message_includes_column_name(self, train_data):
        """Test that the error message includes the offending column name."""
        y, y_pred = train_data
        y_pred_null = y_pred.with_columns(pl.lit(None, dtype=pl.Float64).alias("value"))
        sim = DistanceSimilarity()
        with pytest.raises(ValueError, match="value"):
            sim.fit(y, y_pred_null)


class TestTemporalSimilarityBasic:
    """Basic tests for TemporalSimilarity."""

    def test_fit_returns_self(self, daily_data):
        """Test that fit returns the estimator."""
        y, y_pred = daily_data
        sim = TemporalSimilarity(seasonalities=[7.0])
        result = sim.fit(y, y_pred)
        assert result is sim

    def test_predict_returns_ndarray(self, daily_data, daily_prediction_data):
        """Test that predict returns a numpy array."""
        y, y_pred = daily_data
        sim = TemporalSimilarity(seasonalities=[7.0])
        sim.fit(y, y_pred)
        weights = sim.predict(daily_prediction_data)
        assert isinstance(weights, np.ndarray)

    def test_predict_shape(self, daily_data, daily_prediction_data):
        """Test that predict returns correct shape."""
        y, y_pred = daily_data
        sim = TemporalSimilarity(seasonalities=[7.0])
        sim.fit(y, y_pred)
        weights = sim.predict(daily_prediction_data)
        assert weights.shape == (1, 70)

    def test_predict_shape_multi_row(self, daily_data):
        """Test predict shape with multiple prediction rows."""
        y, y_pred = daily_data
        sim = TemporalSimilarity(seasonalities=[7.0])
        sim.fit(y, y_pred)

        dates = [datetime(2021, 3, 12) + timedelta(days=i) for i in range(3)]
        y_pred_new = pl.DataFrame({"time": dates, "value": [1.0, 2.0, 3.0]})
        weights = sim.predict(y_pred_new)
        assert weights.shape == (3, 70)

    def test_weights_are_positive(self, daily_data, daily_prediction_data):
        """Test that all weights are positive."""
        y, y_pred = daily_data
        sim = TemporalSimilarity(seasonalities=[7.0])
        sim.fit(y, y_pred)
        weights = sim.predict(daily_prediction_data)
        assert np.all(weights > 0)

    def test_weights_are_finite(self, daily_data, daily_prediction_data):
        """Test that all weights are finite."""
        y, y_pred = daily_data
        sim = TemporalSimilarity(seasonalities=[7.0])
        sim.fit(y, y_pred)
        weights = sim.predict(daily_prediction_data)
        assert np.all(np.isfinite(weights))

    def test_weights_row_sum_less_than_one(self, daily_data, daily_prediction_data):
        """Test that weight rows sum to less than 1 (uniform component reserved)."""
        y, y_pred = daily_data
        sim = TemporalSimilarity(seasonalities=[7.0])
        sim.fit(y, y_pred)
        weights = sim.predict(daily_prediction_data)
        assert np.all(np.sum(weights, axis=1) < 1.0)

    def test_empty_seasonalities_raises(self, daily_data):
        """Test that empty seasonalities raises ValueError."""
        y, y_pred = daily_data
        sim = TemporalSimilarity(seasonalities=[])
        with pytest.raises(ValueError, match="seasonalities"):
            sim.fit(y, y_pred)

    def test_none_seasonalities_raises(self, daily_data):
        """Test that None seasonalities raises ValueError."""
        y, y_pred = daily_data
        sim = TemporalSimilarity(seasonalities=None)
        with pytest.raises(ValueError, match="seasonalities"):
            sim.fit(y, y_pred)


class TestTemporalSimilaritySeasonalProximity:
    """Tests verifying that same-season observations get higher weight."""

    def test_same_weekday_gets_higher_weight(self, daily_data):
        """Test that observations on the same weekday get higher weight."""
        y, y_pred = daily_data
        sim = TemporalSimilarity(seasonalities=[7.0])
        sim.fit(y, y_pred)

        # Predict for a specific day
        test_date = datetime(2021, 3, 15)  # Monday
        y_pred_test = pl.DataFrame({"time": [test_date], "value": [10.0]})
        weights = sim.predict(y_pred_test)

        # Check that Mondays in calibration get higher weight
        calib_dates = y_pred["time"].to_list()
        same_weekday = [i for i, d in enumerate(calib_dates) if d.weekday() == test_date.weekday()]
        other_weekday = [i for i, d in enumerate(calib_dates) if d.weekday() != test_date.weekday()]

        avg_same = np.mean(weights[0, same_weekday])
        avg_other = np.mean(weights[0, other_weekday])
        assert avg_same > avg_other

    def test_weekday_weight_ratio(self, daily_data):
        """Test that same-weekday weight is significantly higher."""
        y, y_pred = daily_data
        sim = TemporalSimilarity(seasonalities=[7.0])
        sim.fit(y, y_pred)

        test_date = datetime(2021, 3, 15)  # Monday
        y_pred_test = pl.DataFrame({"time": [test_date], "value": [10.0]})
        weights = sim.predict(y_pred_test)

        calib_dates = y_pred["time"].to_list()
        same_weekday = [i for i, d in enumerate(calib_dates) if d.weekday() == test_date.weekday()]
        other_weekday = [i for i, d in enumerate(calib_dates) if d.weekday() != test_date.weekday()]

        ratio = np.mean(weights[0, same_weekday]) / np.mean(weights[0, other_weekday])
        # Should be at least 2x higher for weekly seasonality
        assert ratio > 2.0


class TestTemporalSimilarityMultiSeasonality:
    """Tests for multiple seasonalities."""

    def test_multi_seasonality_feature_count(self, daily_data):
        """Test feature count with multiple seasonalities."""
        y, y_pred = daily_data
        sim = TemporalSimilarity(seasonalities=[7.0, 365.25])
        sim.fit(y, y_pred)
        # 2 features per seasonality (sin + cos), 2 seasonalities = 4
        assert sim._n_features == 4

    def test_multi_seasonality_predict_shape(self, daily_data, daily_prediction_data):
        """Test that multi-seasonality prediction shape is correct."""
        y, y_pred = daily_data
        sim = TemporalSimilarity(seasonalities=[7.0, 365.25])
        sim.fit(y, y_pred)
        weights = sim.predict(daily_prediction_data)
        assert weights.shape == (1, 70)


class TestTemporalSimilarityHarmonics:
    """Tests for custom harmonics."""

    def test_custom_harmonics(self, daily_data, daily_prediction_data):
        """Test with custom harmonics."""
        y, y_pred = daily_data
        sim = TemporalSimilarity(
            seasonalities=[7.0],
            harmonics={7.0: [1, 2, 3]},
        )
        sim.fit(y, y_pred)
        # 3 harmonics x 2 (sin+cos) = 6 features
        assert sim._n_features == 6

        weights = sim.predict(daily_prediction_data)
        assert weights.shape == (1, 70)
        assert np.all(np.isfinite(weights))

    def test_harmonics_more_selective(self, daily_data):
        """Test that more harmonics give sharper weighting."""
        y, y_pred = daily_data

        sim_1 = TemporalSimilarity(seasonalities=[7.0], harmonics={7.0: [1]})
        sim_3 = TemporalSimilarity(seasonalities=[7.0], harmonics={7.0: [1, 2, 3]})
        sim_1.fit(y, y_pred)
        sim_3.fit(y, y_pred)

        test_date = datetime(2021, 3, 15)
        y_pred_test = pl.DataFrame({"time": [test_date], "value": [10.0]})
        w_1 = sim_1.predict(y_pred_test)
        w_3 = sim_3.predict(y_pred_test)

        # Both should have valid weights
        assert np.all(np.isfinite(w_1))
        assert np.all(np.isfinite(w_3))
        # More harmonics should give different (typically sharper) weights
        assert not np.allclose(w_1, w_3)


class TestTemporalSimilarityObserve:
    """Tests for observe method."""

    def test_observe_returns_self(self, daily_data):
        """Test that observe returns the estimator."""
        y, y_pred = daily_data
        sim = TemporalSimilarity(seasonalities=[7.0])
        sim.fit(y, y_pred)
        result = sim.observe(y[:3], y_pred[:3])
        assert result is sim

    def test_observe_extends_features(self, daily_data, daily_prediction_data):
        """Test that observe extends the reference feature matrix."""
        y, y_pred = daily_data
        sim = TemporalSimilarity(seasonalities=[7.0])
        sim.fit(y, y_pred)

        weights_before = sim.predict(daily_prediction_data)
        n_before = weights_before.shape[1]

        # Observe 5 new rows
        new_dates = [datetime(2021, 3, 12) + timedelta(days=i) for i in range(5)]
        y_new = pl.DataFrame({"time": new_dates, "value": [1.0] * 5})
        y_pred_new = pl.DataFrame({"time": new_dates, "value": [1.1] * 5})
        sim.observe(y_new, y_pred_new)

        weights_after = sim.predict(daily_prediction_data)
        assert weights_after.shape[1] == n_before + 5


class TestTemporalSimilarityProperties:
    """Tests for properties and sklearn compatibility."""

    def test_clone(self):
        """Test that TemporalSimilarity can be cloned."""
        sim = TemporalSimilarity(seasonalities=[7.0, 365.25], metric="cityblock")
        cloned = clone(sim)
        assert cloned.seasonalities == [7.0, 365.25]
        assert cloned.metric == "cityblock"
        assert cloned is not sim

    def test_get_params(self):
        """Test get_params returns expected parameters."""
        sim = TemporalSimilarity(
            seasonalities=[7.0],
            harmonics={7.0: [1, 2]},
            metric="cosine",
        )
        params = sim.get_params()
        assert params["seasonalities"] == [7.0]
        assert params["harmonics"] == {7.0: [1, 2]}
        assert params["metric"] == "cosine"

    def test_set_params(self):
        """Test set_params updates parameters."""
        sim = TemporalSimilarity(seasonalities=[7.0])
        sim.set_params(metric="cityblock")
        assert sim.metric == "cityblock"

    def test_tags(self):
        """Test sklearn tags are set correctly."""
        sim = TemporalSimilarity(seasonalities=[7.0])
        tags = sim.__sklearn_tags__()
        assert tags.estimator_type == "similarity"
        assert tags.requires_fit is True

    def test_auto_detect_interval(self, daily_data):
        """Test that interval is auto-detected from timestamps."""
        y, y_pred = daily_data
        sim = TemporalSimilarity(seasonalities=[7.0])
        sim.fit(y, y_pred)
        assert sim.interval_td_ == timedelta(days=1)

    def test_first_time_stored(self, daily_data):
        """Test that first_time_ is stored from calibration data."""
        y, y_pred = daily_data
        sim = TemporalSimilarity(seasonalities=[7.0])
        sim.fit(y, y_pred)
        assert sim.first_time_ == datetime(2021, 1, 1)


class TestTemporalSimilarityIntegration:
    """Integration tests with SplitConformalForecaster."""

    def test_with_split_conformal(self):
        """Test TemporalSimilarity plugged into SplitConformalForecaster."""
        from yohou.interval import SplitConformalForecaster
        from yohou.metrics.conformity import AbsoluteResidual
        from yohou.point import SeasonalNaive

        n = 200
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        np.random.seed(42)
        values = [10.0 + 5.0 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 0.5) for i in range(n)]
        y = pl.DataFrame({"time": dates, "value": values})
        y_train = y[:180]

        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
            similarity=TemporalSimilarity(seasonalities=[7.0]),
        )
        scf.fit(y_train, forecasting_horizon=3, coverage_rates=[0.9])

        intervals = scf.predict_interval(forecasting_horizon=3, coverage_rates=[0.9])
        assert len(intervals) == 3
        assert "time" in intervals.columns

    def test_produces_different_intervals_than_unweighted(self):
        """Test that temporal similarity changes intervals vs unweighted."""
        from yohou.interval import SplitConformalForecaster
        from yohou.metrics.conformity import AbsoluteResidual
        from yohou.point import SeasonalNaive

        n = 200
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        np.random.seed(42)
        values = [10.0 + 5.0 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 0.5) for i in range(n)]
        y = pl.DataFrame({"time": dates, "value": values})
        y_train = y[:180]

        scf_weighted = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
            similarity=TemporalSimilarity(seasonalities=[7.0]),
        )
        scf_weighted.fit(y_train, forecasting_horizon=1, coverage_rates=[0.9])

        scf_unweighted = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
        )
        scf_unweighted.fit(y_train, forecasting_horizon=1, coverage_rates=[0.9])

        iv_w = scf_weighted.predict_interval(forecasting_horizon=1, coverage_rates=[0.9])
        iv_u = scf_unweighted.predict_interval(forecasting_horizon=1, coverage_rates=[0.9])

        # Intervals should differ (temporal weighting changes quantiles)
        value_cols = [c for c in iv_w.columns if c != "time" and iv_w[c].dtype.is_numeric()]
        any_differ = any(not np.isclose(float(iv_w[col][0]), float(iv_u[col][0])) for col in value_cols)
        assert any_differ, "At least one interval bound should differ"

    def test_with_gamma_residual(self):
        """Test that similarity + GammaResidual produces weighted intervals."""
        from yohou.interval import SplitConformalForecaster
        from yohou.metrics.conformity import GammaResidual
        from yohou.point import SeasonalNaive

        n = 200
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        np.random.seed(42)
        values = [10.0 + 5.0 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 0.5) for i in range(n)]
        y = pl.DataFrame({"time": dates, "value": values})
        y_train = y[:180]

        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            conformity_scorer=GammaResidual(),
            similarity=TemporalSimilarity(seasonalities=[7.0]),
        )
        scf.fit(y_train, forecasting_horizon=1, coverage_rates=[0.9])

        intervals = scf.predict_interval(forecasting_horizon=1, coverage_rates=[0.9])
        assert len(intervals) == 1
        # Should have lower and upper bound columns
        non_time_cols = [c for c in intervals.columns if c != "time"]
        assert len(non_time_cols) >= 2

    def test_with_absolute_gamma_residual(self):
        """Test that similarity + AbsoluteGammaResidual produces weighted intervals."""
        from yohou.interval import SplitConformalForecaster
        from yohou.metrics.conformity import AbsoluteGammaResidual
        from yohou.point import SeasonalNaive

        n = 200
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        np.random.seed(42)
        values = [10.0 + 5.0 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 0.5) for i in range(n)]
        y = pl.DataFrame({"time": dates, "value": values})
        y_train = y[:180]

        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            conformity_scorer=AbsoluteGammaResidual(),
            similarity=TemporalSimilarity(seasonalities=[7.0]),
        )
        scf.fit(y_train, forecasting_horizon=1, coverage_rates=[0.9])

        intervals = scf.predict_interval(forecasting_horizon=1, coverage_rates=[0.9])
        assert len(intervals) == 1
        non_time_cols = [c for c in intervals.columns if c != "time"]
        assert len(non_time_cols) >= 2


# ── CompositeSimilarity ──────────────────────────────────────────────


@pytest.fixture
def composite_data():
    """Create daily data with weekly seasonality for composite tests."""
    n = 56  # 8 weeks
    dates = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(n)]
    np.random.seed(42)
    values = [10.0 + 5.0 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 0.5) for i in range(n)]
    y = pl.DataFrame({"time": dates, "value": values})
    y_pred = pl.DataFrame({"time": dates, "value": [v + np.random.normal(0, 0.3) for v in values]})
    return y, y_pred


class TestCompositeSimilarityBasic:
    """Core fit/predict behaviour."""

    def test_fit_returns_self(self, composite_data):
        y, y_pred = composite_data
        comp = CompositeSimilarity(
            similarities=[
                DistanceSimilarity(metric="euclidean"),
                TemporalSimilarity(seasonalities=[7.0]),
            ],
        )
        result = comp.fit(y, y_pred)
        assert result is comp

    def test_predict_returns_ndarray(self, composite_data):
        y, y_pred = composite_data
        comp = CompositeSimilarity(
            similarities=[
                DistanceSimilarity(metric="euclidean"),
                TemporalSimilarity(seasonalities=[7.0]),
            ],
        )
        comp.fit(y, y_pred)
        new_pred = y_pred.tail(1)
        weights = comp.predict(new_pred)
        assert isinstance(weights, np.ndarray)

    def test_predict_shape(self, composite_data):
        y, y_pred = composite_data
        comp = CompositeSimilarity(
            similarities=[
                DistanceSimilarity(metric="euclidean"),
                TemporalSimilarity(seasonalities=[7.0]),
            ],
        )
        comp.fit(y, y_pred)
        new_pred = y_pred.tail(3)
        weights = comp.predict(new_pred)
        assert weights.shape == (3, len(y_pred))

    def test_weights_are_positive(self, composite_data):
        y, y_pred = composite_data
        comp = CompositeSimilarity(
            similarities=[
                DistanceSimilarity(metric="euclidean"),
                TemporalSimilarity(seasonalities=[7.0]),
            ],
        )
        comp.fit(y, y_pred)
        weights = comp.predict(y_pred.tail(1))
        assert np.all(weights > 0)

    def test_weights_are_finite(self, composite_data):
        y, y_pred = composite_data
        comp = CompositeSimilarity(
            similarities=[
                DistanceSimilarity(metric="euclidean"),
                TemporalSimilarity(seasonalities=[7.0]),
            ],
        )
        comp.fit(y, y_pred)
        weights = comp.predict(y_pred.tail(1))
        assert np.all(np.isfinite(weights))

    def test_row_sum_less_than_one_multiply(self, composite_data):
        y, y_pred = composite_data
        comp = CompositeSimilarity(
            similarities=[
                DistanceSimilarity(metric="euclidean"),
                TemporalSimilarity(seasonalities=[7.0]),
            ],
            combination="multiply",
        )
        comp.fit(y, y_pred)
        weights = comp.predict(y_pred.tail(2))
        assert np.all(weights.sum(axis=1) < 1.0)

    def test_mean_combination(self, composite_data):
        y, y_pred = composite_data
        comp = CompositeSimilarity(
            similarities=[
                DistanceSimilarity(metric="euclidean"),
                TemporalSimilarity(seasonalities=[7.0]),
            ],
            combination="mean",
        )
        comp.fit(y, y_pred)
        weights = comp.predict(y_pred.tail(1))
        assert weights.shape == (1, len(y_pred))
        assert np.all(weights > 0)


class TestCompositeSimilarityValidation:
    """Parameter validation."""

    def test_fewer_than_two_raises(self):
        comp = CompositeSimilarity(similarities=[DistanceSimilarity()])
        with pytest.raises(ValueError, match="at least 2"):
            comp.fit(
                pl.DataFrame({"time": [datetime(2021, 1, 1)], "value": [1.0]}),
                pl.DataFrame({"time": [datetime(2021, 1, 1)], "value": [1.0]}),
            )

    def test_none_similarities_raises(self):
        comp = CompositeSimilarity(similarities=None)
        with pytest.raises(ValueError, match="at least 2"):
            comp.fit(
                pl.DataFrame({"time": [datetime(2021, 1, 1)], "value": [1.0]}),
                pl.DataFrame({"time": [datetime(2021, 1, 1)], "value": [1.0]}),
            )

    def test_invalid_combination_raises(self):
        comp = CompositeSimilarity(
            similarities=[DistanceSimilarity(), DistanceSimilarity()],
            combination="invalid",
        )
        with pytest.raises(ValueError, match="combination"):
            comp.fit(
                pl.DataFrame({"time": [datetime(2021, 1, 1)], "value": [1.0]}),
                pl.DataFrame({"time": [datetime(2021, 1, 1)], "value": [1.0]}),
            )

    def test_weights_length_mismatch_raises(self):
        comp = CompositeSimilarity(
            similarities=[DistanceSimilarity(), DistanceSimilarity()],
            weights=[1.0, 2.0, 3.0],
        )
        with pytest.raises(ValueError, match="weights length"):
            comp.fit(
                pl.DataFrame({"time": [datetime(2021, 1, 1)], "value": [1.0]}),
                pl.DataFrame({"time": [datetime(2021, 1, 1)], "value": [1.0]}),
            )


class TestCompositeSimilarityWeights:
    """Custom weight behaviour."""

    def test_custom_weights_multiply(self, composite_data):
        y, y_pred = composite_data
        comp = CompositeSimilarity(
            similarities=[
                DistanceSimilarity(metric="euclidean"),
                TemporalSimilarity(seasonalities=[7.0]),
            ],
            combination="multiply",
            weights=[2.0, 0.5],
        )
        comp.fit(y, y_pred)
        weights = comp.predict(y_pred.tail(1))
        assert weights.shape == (1, len(y_pred))
        assert np.all(weights > 0)

    def test_custom_weights_mean(self, composite_data):
        y, y_pred = composite_data
        comp = CompositeSimilarity(
            similarities=[
                DistanceSimilarity(metric="euclidean"),
                TemporalSimilarity(seasonalities=[7.0]),
            ],
            combination="mean",
            weights=[0.3, 0.7],
        )
        comp.fit(y, y_pred)
        weights = comp.predict(y_pred.tail(1))
        assert weights.shape == (1, len(y_pred))
        assert np.all(weights > 0)

    def test_multiply_differs_from_individual(self, composite_data):
        """Composite multiply should differ from either sub-similarity alone."""
        y, y_pred = composite_data
        new_pred = y_pred.tail(1)

        dist_sim = DistanceSimilarity(metric="euclidean")
        dist_sim.fit(y, y_pred)
        w_dist = dist_sim.predict(new_pred)

        temp_sim = TemporalSimilarity(seasonalities=[7.0])
        temp_sim.fit(y, y_pred)
        w_temp = temp_sim.predict(new_pred)

        comp = CompositeSimilarity(
            similarities=[
                DistanceSimilarity(metric="euclidean"),
                TemporalSimilarity(seasonalities=[7.0]),
            ],
            combination="multiply",
        )
        comp.fit(y, y_pred)
        w_comp = comp.predict(new_pred)

        assert not np.allclose(w_comp, w_dist)
        assert not np.allclose(w_comp, w_temp)


class TestCompositeSimilarityObserveRewind:
    """Observe and rewind delegation."""

    def test_observe_returns_self(self, composite_data):
        y, y_pred = composite_data
        comp = CompositeSimilarity(
            similarities=[
                DistanceSimilarity(metric="euclidean"),
                TemporalSimilarity(seasonalities=[7.0]),
            ],
        )
        comp.fit(y, y_pred)
        result = comp.observe(y.tail(1), y_pred.tail(1))
        assert result is comp

    def test_observe_extends_sub_similarities(self, composite_data):
        y, y_pred = composite_data
        comp = CompositeSimilarity(
            similarities=[
                DistanceSimilarity(metric="euclidean"),
                TemporalSimilarity(seasonalities=[7.0]),
            ],
        )
        comp.fit(y, y_pred)

        n_dist_before = len(comp.similarities_[0]._X_observed)
        n_temp_before = comp.similarities_[1]._features_observed.shape[0]

        comp.observe(y.tail(1), y_pred.tail(1))

        assert len(comp.similarities_[0]._X_observed) == n_dist_before + 1
        assert comp.similarities_[1]._features_observed.shape[0] == n_temp_before + 1

    def test_rewind_restores_sub_similarities(self, composite_data):
        y, y_pred = composite_data
        comp = CompositeSimilarity(
            similarities=[
                DistanceSimilarity(metric="euclidean"),
                TemporalSimilarity(seasonalities=[7.0]),
            ],
        )
        comp.fit(y, y_pred)

        n_dist_before = len(comp.similarities_[0]._X_observed)
        n_temp_before = comp.similarities_[1]._features_observed.shape[0]

        comp.observe(y.tail(1), y_pred.tail(1))
        comp.rewind(y.tail(1), y_pred.tail(1))

        assert len(comp.similarities_[0]._X_observed) == n_dist_before
        assert comp.similarities_[1]._features_observed.shape[0] == n_temp_before


class TestCompositeSimilarityProperties:
    """sklearn estimator interface."""

    def test_clone(self):
        comp = CompositeSimilarity(
            similarities=[
                DistanceSimilarity(metric="euclidean"),
                TemporalSimilarity(seasonalities=[7.0]),
            ],
            combination="mean",
            weights=[0.3, 0.7],
        )
        cloned = clone(comp)
        assert cloned.combination == "mean"
        assert cloned.weights == [0.3, 0.7]
        assert len(cloned.similarities) == 2

    def test_get_params(self):
        comp = CompositeSimilarity(
            similarities=[DistanceSimilarity(), TemporalSimilarity(seasonalities=[7.0])],
            combination="multiply",
        )
        params = comp.get_params()
        assert params["combination"] == "multiply"
        assert params["weights"] is None
        assert len(params["similarities"]) == 2


class TestCompositeSimilarityIntegration:
    """Integration with SplitConformalForecaster."""

    def test_with_split_conformal(self):
        from yohou.interval import SplitConformalForecaster
        from yohou.metrics import AbsoluteResidual
        from yohou.point import SeasonalNaive

        n = 200
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        np.random.seed(42)
        values = [10.0 + 5.0 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 0.5) for i in range(n)]
        y = pl.DataFrame({"time": dates, "value": values})
        y_train = y[:180]

        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            conformity_scorer=AbsoluteResidual(),
            similarity=CompositeSimilarity(
                similarities=[
                    DistanceSimilarity(metric="euclidean"),
                    TemporalSimilarity(seasonalities=[7.0]),
                ],
                combination="multiply",
            ),
        )
        scf.fit(y_train, forecasting_horizon=1, coverage_rates=[0.9])

        intervals = scf.predict_interval(forecasting_horizon=1, coverage_rates=[0.9])
        assert len(intervals) == 1
        non_time_cols = [c for c in intervals.columns if c != "time"]
        assert len(non_time_cols) >= 2


class TestTemporalSimilaritySingleTimestamp:
    """Tests for TemporalSimilarity with a single observation."""

    def test_fit_single_timestamp(self):
        """Test fit with a single timestamp sets interval_td_ to zero."""
        y = pl.DataFrame({"time": [datetime(2021, 1, 1)], "value": [1.0]})
        y_pred = pl.DataFrame({"time": [datetime(2021, 1, 1)], "value": [1.1]})
        sim = TemporalSimilarity(seasonalities=[7.0])
        sim.fit(y, y_pred)
        assert sim.interval_td_ == timedelta(0)

    def test_predict_after_single_timestamp_fit(self):
        """Test predict works after fitting with a single timestamp (zero interval)."""
        y = pl.DataFrame({"time": [datetime(2021, 1, 1)], "value": [1.0]})
        y_pred = pl.DataFrame({"time": [datetime(2021, 1, 1)], "value": [1.1]})
        sim = TemporalSimilarity(seasonalities=[7.0])
        sim.fit(y, y_pred)

        weights = sim.predict(y_pred)
        assert weights.shape == (1, 1)
        assert np.all(np.isfinite(weights))
