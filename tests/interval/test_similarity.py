"""Tests for similarity measures in interval forecasting."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from yohou.interval.similarity import CompositeSimilarity, DistanceSimilarity, SeasonalSimilarity


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

    def test_default_metric(self):
        """Test default metric is euclidean."""
        sim = DistanceSimilarity()
        assert sim.metric == "euclidean"

    def test_default_metric_params(self):
        """Test default metric_params is None."""
        sim = DistanceSimilarity()
        assert sim.metric_params is None


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

    def test_rewind_before_fit_raises_not_fitted(self, train_data):
        """Test that rewind before fit raises NotFittedError (not AttributeError)."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        with pytest.raises(NotFittedError):
            sim.rewind(y[:2], y_pred[:2])


class TestDistanceSimilarityWithExogenous:
    """Tests for similarity with exogenous features."""

    def test_fit_with_X(self, train_data):
        """Test fitting with exogenous features."""
        y, y_pred = train_data
        # X_actual should NOT have a 'time' column to avoid duplicate with y_pred
        X_actual = pl.DataFrame({
            "feature": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        })
        sim = DistanceSimilarity()
        sim.fit(y, y_pred, X_actual=X_actual)
        assert sim._X_observed.shape[1] > 1  # time + value + feature cols

    def test_predict_with_X(self, train_data, prediction_data):
        """Test prediction with exogenous features."""
        y, y_pred = train_data
        X_actual_train = pl.DataFrame({
            "feature": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        })
        X_actual_test = pl.DataFrame({
            "feature": [85.0, 92.0],
        })
        sim = DistanceSimilarity()
        sim.fit(y, y_pred, X_actual=X_actual_train)
        weights = sim.predict(prediction_data, X_actual=X_actual_test)
        assert weights.shape == (2, 8)


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
        """fit raises ValueError naming the offending column when y_pred contains null."""
        y, y_pred = train_data
        y_pred_null = y_pred.with_columns(
            pl.when(pl.col("value") > 5).then(None).otherwise(pl.col("value")).alias("value")
        )
        sim = DistanceSimilarity()
        with pytest.raises(ValueError, match=r"value.*null or NaN"):
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

    def test_error_reports_all_bad_columns(self, train_data):
        """Error message lists every offending column, not just the first."""
        y, y_pred = train_data
        y_pred_bad = y_pred.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("value"),
        )
        x_actual_bad = y_pred.select("time").with_columns(
            pl.lit(None, dtype=pl.Float64).alias("extra"),
        )
        sim = DistanceSimilarity()
        with pytest.raises(ValueError) as exc_info:
            sim.fit(y, y_pred_bad, X_actual=x_actual_bad)
        message = str(exc_info.value)
        assert "value" in message
        assert "extra" in message

    def test_predict_rejects_nan(self, train_data):
        """Test that predict raises ValueError when data contains NaN."""
        y, y_pred = train_data
        sim = DistanceSimilarity()
        sim.fit(y, y_pred)
        y_pred_nan = y_pred.head(1).with_columns(pl.lit(float("nan")).alias("value"))
        with pytest.raises(ValueError, match="null or NaN"):
            sim.predict(y_pred_nan)


class TestSeasonalSimilarityParamName:
    """The public seasonal-period parameter is singular ``seasonality``."""

    def test_param_is_singular_seasonality(self):
        """SeasonalSimilarity exposes ``seasonality`` (not ``seasonalities``)."""
        sim = SeasonalSimilarity(seasonality=[7.0])
        params = sim.get_params()
        assert "seasonality" in params
        assert "seasonalities" not in params
        assert sim.seasonality == [7.0]

    def test_legacy_seasonalities_kwarg_rejected(self, daily_data):
        """The old plural keyword is no longer accepted (clean pre-1.0 rename)."""
        with pytest.raises(TypeError):
            SeasonalSimilarity(**{"seasonalities": [7.0]})


class TestSeasonalSimilarityBasic:
    """Basic tests for SeasonalSimilarity."""

    def test_predict_shape_multi_row(self, daily_data):
        """Test predict shape with multiple prediction rows."""
        y, y_pred = daily_data
        sim = SeasonalSimilarity(seasonality=[7.0])
        sim.fit(y, y_pred)

        dates = [datetime(2021, 3, 12) + timedelta(days=i) for i in range(3)]
        y_pred_new = pl.DataFrame({"time": dates, "value": [1.0, 2.0, 3.0]})
        weights = sim.predict(y_pred_new)
        assert weights.shape == (3, 70)

    def test_empty_seasonalities_raises(self, daily_data):
        """Test that empty seasonalities raises ValueError."""
        y, y_pred = daily_data
        sim = SeasonalSimilarity(seasonality=[])
        with pytest.raises(ValueError, match="seasonality"):
            sim.fit(y, y_pred)

    def test_none_seasonalities_raises(self, daily_data):
        """Test that None seasonalities raises ValueError."""
        y, y_pred = daily_data
        sim = SeasonalSimilarity(seasonality=None)
        with pytest.raises(ValueError, match="seasonality"):
            sim.fit(y, y_pred)


class TestSeasonalSimilaritySeasonalProximity:
    """Tests verifying that same-season observations get higher weight."""

    def test_weekday_weight_ratio(self, daily_data):
        """Test that same-weekday weight is significantly higher."""
        y, y_pred = daily_data
        sim = SeasonalSimilarity(seasonality=[7.0])
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


class TestSeasonalSimilarityMultiSeasonality:
    """Tests for multiple seasonalities."""

    def test_multi_seasonality_feature_count(self, daily_data):
        """Test feature count with multiple seasonalities."""
        y, y_pred = daily_data
        sim = SeasonalSimilarity(seasonality=[7.0, 365.25])
        sim.fit(y, y_pred)
        # 2 features per seasonality (sin + cos), 2 seasonalities = 4
        assert sim._features_observed.shape[1] == 4

    def test_multi_seasonality_predict_shape(self, daily_data, daily_prediction_data):
        """Test that multi-seasonality prediction shape is correct."""
        y, y_pred = daily_data
        sim = SeasonalSimilarity(seasonality=[7.0, 365.25])
        sim.fit(y, y_pred)
        weights = sim.predict(daily_prediction_data)
        assert weights.shape == (1, 70)


class TestSeasonalSimilarityHarmonics:
    """Tests for custom harmonics."""

    def test_custom_harmonics(self, daily_data):
        """Custom harmonics produce two features (sin + cos) per requested harmonic.

        The predict-shape and finiteness invariants are covered generically by
        the similarity systematic checks; this test pins only the feature count
        unique to the custom-harmonics configuration.
        """
        y, y_pred = daily_data
        sim = SeasonalSimilarity(
            seasonality=[7.0],
            harmonics={7.0: [1, 2, 3]},
        )
        sim.fit(y, y_pred)
        # 3 harmonics x 2 (sin+cos) = 6 features
        assert sim._features_observed.shape[1] == 6

    def test_harmonics_more_selective(self, daily_data):
        """Test that more harmonics give sharper weighting."""
        y, y_pred = daily_data

        sim_1 = SeasonalSimilarity(seasonality=[7.0], harmonics={7.0: [1]})
        sim_3 = SeasonalSimilarity(seasonality=[7.0], harmonics={7.0: [1, 2, 3]})
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


class TestSeasonalSimilarityObserve:
    """Tests for observe method."""

    def test_observe_returns_self(self, daily_data):
        """Test that observe returns the estimator."""
        y, y_pred = daily_data
        sim = SeasonalSimilarity(seasonality=[7.0])
        sim.fit(y, y_pred)
        result = sim.observe(y[:3], y_pred[:3])
        assert result is sim

    def test_observe_extends_features(self, daily_data, daily_prediction_data):
        """Test that observe extends the reference feature matrix."""
        y, y_pred = daily_data
        sim = SeasonalSimilarity(seasonality=[7.0])
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

    def test_rewind_before_fit_raises_not_fitted(self, daily_data):
        """Test that rewind before fit raises NotFittedError (not AttributeError)."""
        y, y_pred = daily_data
        sim = SeasonalSimilarity(seasonality=[7.0])
        with pytest.raises(NotFittedError):
            sim.rewind(y[:2], y_pred[:2])


class TestSeasonalSimilarityProperties:
    """Tests for properties and sklearn compatibility."""

    def test_auto_detect_interval(self, daily_data):
        """Test that interval is auto-detected from timestamps."""
        y, y_pred = daily_data
        sim = SeasonalSimilarity(seasonality=[7.0])
        sim.fit(y, y_pred)
        assert sim.interval_td_ == timedelta(days=1)

    def test_first_time_stored(self, daily_data):
        """Test that first_time_ is stored from calibration data."""
        y, y_pred = daily_data
        sim = SeasonalSimilarity(seasonality=[7.0])
        sim.fit(y, y_pred)
        assert sim.first_time_ == datetime(2021, 1, 1)


class TestSeasonalSimilarityIntegration:
    """Integration tests with SplitConformalForecaster."""

    def test_with_split_conformal(self):
        """Test SeasonalSimilarity plugged into SplitConformalForecaster."""
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
            similarity=SeasonalSimilarity(seasonality=[7.0]),
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
            similarity=SeasonalSimilarity(seasonality=[7.0]),
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
            similarity=SeasonalSimilarity(seasonality=[7.0]),
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
            similarity=SeasonalSimilarity(seasonality=[7.0]),
        )
        scf.fit(y_train, forecasting_horizon=1, coverage_rates=[0.9])

        intervals = scf.predict_interval(forecasting_horizon=1, coverage_rates=[0.9])
        assert len(intervals) == 1
        non_time_cols = [c for c in intervals.columns if c != "time"]
        assert len(non_time_cols) >= 2


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

    def test_mean_combination(self, composite_data):
        y, y_pred = composite_data
        comp = CompositeSimilarity(
            similarities=[
                ("dist", DistanceSimilarity(metric="euclidean")),
                ("seasonal", SeasonalSimilarity(seasonality=[7.0])),
            ],
            combination="mean",
        )
        comp.fit(y, y_pred)
        weights = comp.predict(y_pred.tail(1))
        assert weights.shape == (1, len(y_pred))
        assert np.all(weights > 0)


class TestCompositeSimilarityValidation:
    """Invalid composition parameters raise descriptive errors at fit."""

    def test_none_similarities_raises(self, composite_data):
        """``similarities=None`` is rejected by the parameter constraint."""
        y, y_pred = composite_data
        comp = CompositeSimilarity(similarities=None)
        with pytest.raises(ValueError, match="must be an instance of 'list'"):
            comp.fit(y, y_pred)

    def test_single_similarity_raises(self, composite_data):
        """Fewer than two sub-similarities is rejected."""
        y, y_pred = composite_data
        comp = CompositeSimilarity(similarities=[("dist", DistanceSimilarity())])
        with pytest.raises(ValueError, match="at least 2 sub-similarities"):
            comp.fit(y, y_pred)

    def test_invalid_combination_raises(self, composite_data):
        """An unknown ``combination`` value is rejected."""
        y, y_pred = composite_data
        comp = CompositeSimilarity(
            similarities=[
                ("dist", DistanceSimilarity()),
                ("seasonal", SeasonalSimilarity(seasonality=[7.0])),
            ],
            combination="invalid",
        )
        with pytest.raises(ValueError, match="combination must be"):
            comp.fit(y, y_pred)

    def test_weights_length_mismatch_raises(self, composite_data):
        """A ``weights`` list that does not match the sub-similarity count is rejected."""
        y, y_pred = composite_data
        comp = CompositeSimilarity(
            similarities=[
                ("dist", DistanceSimilarity()),
                ("seasonal", SeasonalSimilarity(seasonality=[7.0])),
            ],
            weights=[1.0, 2.0, 3.0],
        )
        with pytest.raises(ValueError, match="weights length"):
            comp.fit(y, y_pred)


class TestCompositeSimilarityWeights:
    """Custom weight behaviour."""

    def test_custom_weights_multiply(self, composite_data):
        y, y_pred = composite_data
        comp = CompositeSimilarity(
            similarities=[
                ("dist", DistanceSimilarity(metric="euclidean")),
                ("seasonal", SeasonalSimilarity(seasonality=[7.0])),
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
                ("dist", DistanceSimilarity(metric="euclidean")),
                ("seasonal", SeasonalSimilarity(seasonality=[7.0])),
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

        temp_sim = SeasonalSimilarity(seasonality=[7.0])
        temp_sim.fit(y, y_pred)
        w_temp = temp_sim.predict(new_pred)

        comp = CompositeSimilarity(
            similarities=[
                ("dist", DistanceSimilarity(metric="euclidean")),
                ("seasonal", SeasonalSimilarity(seasonality=[7.0])),
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
                ("dist", DistanceSimilarity(metric="euclidean")),
                ("seasonal", SeasonalSimilarity(seasonality=[7.0])),
            ],
        )
        comp.fit(y, y_pred)
        result = comp.observe(y.tail(1), y_pred.tail(1))
        assert result is comp

    def test_observe_extends_sub_similarities(self, composite_data):
        y, y_pred = composite_data
        comp = CompositeSimilarity(
            similarities=[
                ("dist", DistanceSimilarity(metric="euclidean")),
                ("seasonal", SeasonalSimilarity(seasonality=[7.0])),
            ],
        )
        comp.fit(y, y_pred)

        n_dist_before = len(comp.similarities_[0][1]._X_observed)
        n_temp_before = comp.similarities_[1][1]._features_observed.shape[0]

        comp.observe(y.tail(1), y_pred.tail(1))

        assert len(comp.similarities_[0][1]._X_observed) == n_dist_before + 1
        assert comp.similarities_[1][1]._features_observed.shape[0] == n_temp_before + 1

    def test_rewind_restores_sub_similarities(self, composite_data):
        y, y_pred = composite_data
        comp = CompositeSimilarity(
            similarities=[
                ("dist", DistanceSimilarity(metric="euclidean")),
                ("seasonal", SeasonalSimilarity(seasonality=[7.0])),
            ],
        )
        comp.fit(y, y_pred)

        n_dist_before = len(comp.similarities_[0][1]._X_observed)
        n_temp_before = comp.similarities_[1][1]._features_observed.shape[0]

        comp.observe(y.tail(1), y_pred.tail(1))
        comp.rewind(y.tail(1), y_pred.tail(1))

        assert len(comp.similarities_[0][1]._X_observed) == n_dist_before
        assert comp.similarities_[1][1]._features_observed.shape[0] == n_temp_before

    def test_rewind_before_fit_raises_not_fitted(self, composite_data):
        """rewind before fit raises NotFittedError, not a bare AttributeError."""
        from sklearn.exceptions import NotFittedError

        y, y_pred = composite_data
        comp = CompositeSimilarity(
            similarities=[
                ("dist", DistanceSimilarity(metric="euclidean")),
                ("seasonal", SeasonalSimilarity(seasonality=[7.0])),
            ],
        )
        with pytest.raises(NotFittedError):
            comp.rewind(y.tail(1), y_pred.tail(1))


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
                    ("dist", DistanceSimilarity(metric="euclidean")),
                    ("seasonal", SeasonalSimilarity(seasonality=[7.0])),
                ],
                combination="multiply",
            ),
        )
        scf.fit(y_train, forecasting_horizon=1, coverage_rates=[0.9])

        intervals = scf.predict_interval(forecasting_horizon=1, coverage_rates=[0.9])
        assert len(intervals) == 1
        non_time_cols = [c for c in intervals.columns if c != "time"]
        assert len(non_time_cols) >= 2


class TestSeasonalSimilaritySingleTimestamp:
    """Tests for SeasonalSimilarity with a single observation."""

    def test_fit_single_timestamp(self):
        """Test fit with a single timestamp sets interval_td_ to zero."""
        y = pl.DataFrame({"time": [datetime(2021, 1, 1)], "value": [1.0]})
        y_pred = pl.DataFrame({"time": [datetime(2021, 1, 1)], "value": [1.1]})
        sim = SeasonalSimilarity(seasonality=[7.0])
        sim.fit(y, y_pred)
        assert sim.interval_td_ == timedelta(0)

    def test_predict_after_single_timestamp_fit(self):
        """Test predict works after fitting with a single timestamp (zero interval)."""
        y = pl.DataFrame({"time": [datetime(2021, 1, 1)], "value": [1.0]})
        y_pred = pl.DataFrame({"time": [datetime(2021, 1, 1)], "value": [1.1]})
        sim = SeasonalSimilarity(seasonality=[7.0])
        sim.fit(y, y_pred)

        weights = sim.predict(y_pred)
        assert weights.shape == (1, 1)
        assert np.all(np.isfinite(weights))


class TestSimilarityTuningThroughForecaster:
    """Sub-similarity params are tunable through SplitConformalForecaster."""

    def _data(self):
        n = 160
        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
        np.random.seed(0)
        values = [10.0 + 5.0 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 0.5) for i in range(n)]
        return pl.DataFrame({"time": dates, "value": values})

    def test_set_params_through_forecaster(self):
        from yohou.interval import SplitConformalForecaster

        f = SplitConformalForecaster(
            similarity=CompositeSimilarity(
                similarities=[
                    ("dist", DistanceSimilarity()),
                    ("seasonal", SeasonalSimilarity(seasonality=[7.0])),
                ]
            )
        )
        assert "similarity__dist__metric" in f.get_params(deep=True)
        f.set_params(similarity__dist__metric="cosine")
        assert dict(f.similarity.similarities)["dist"].metric == "cosine"

    def test_grid_search_over_similarity_param(self):
        from yohou.interval import SplitConformalForecaster
        from yohou.metrics import EmpiricalCoverage
        from yohou.model_selection import GridSearchCV
        from yohou.point import SeasonalNaive

        y = self._data()
        scf = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=40,
            similarity=DistanceSimilarity(),
        )
        search = GridSearchCV(
            scf,
            param_grid={"similarity__metric": ["euclidean", "cityblock"]},
            scoring=EmpiricalCoverage(coverage_rates=[0.95]),
            cv=2,
        )
        search.fit(y, forecasting_horizon=1)
        assert search.best_params_["similarity__metric"] in ("euclidean", "cityblock")


class TestSimilarityNumericalStability:
    """Regression: large distances must not overflow exp() or produce NaN weights."""

    def test_to_weights_large_distances_finite(self):
        from yohou.interval.base import BaseSimilarity

        # Distances large enough that exp(+max) would overflow without stabilisation.
        distances = np.array([[0.0, 500.0, 1000.0], [800.0, 0.0, 1600.0]])
        w = BaseSimilarity._to_weights(distances)
        assert np.all(np.isfinite(w))
        assert np.all(w >= 0)
        assert np.all(w.sum(axis=1) < 1.0)
        # Closest (distance 0) keeps the largest weight in each row.
        assert w[0, 0] == w[0].max()
        assert w[1, 1] == w[1].max()

    def test_distance_similarity_large_values_no_overflow(self):
        import warnings

        dates = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(20)]
        y = pl.DataFrame({"time": dates, "value": [float(i) for i in range(20)]})
        # Predictions spanning a large magnitude range -> large pairwise distances.
        y_pred = pl.DataFrame({"time": dates, "value": [float(i) * 1000.0 for i in range(20)]})
        sim = DistanceSimilarity().fit(y, y_pred)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            w = sim.predict(y_pred.tail(1))
        assert np.all(np.isfinite(w))
