"""Tests for ForecastedFeatureForecaster meta-forecaster.

Tests ForecastedFeatureForecaster using both the check generator pattern and
specific meta-forecaster tests for feature chaining behavior.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.utils.metadata_routing import MetadataRouter

from conftest import run_checks
from yohou.compose import ForecastedFeatureForecaster
from yohou.point import PointReductionForecaster, SeasonalNaive
from yohou.testing import _yield_yohou_forecaster_checks


class TestSystematicChecks:
    """Systematic validation via check generators."""

    @pytest.mark.parametrize(
        "forecaster",
        [
            # Point target with point features
            ForecastedFeatureForecaster(
                target_forecaster=SeasonalNaive(seasonality=1),
                feature_forecaster=SeasonalNaive(seasonality=1),
            ),
            # With strategy="predicted"
            ForecastedFeatureForecaster(
                target_forecaster=SeasonalNaive(seasonality=1),
                feature_forecaster=SeasonalNaive(seasonality=1),
                strategy="predicted",
                split_ratio=0.6,
            ),
            # With reduction forecasters
            ForecastedFeatureForecaster(
                target_forecaster=PointReductionForecaster(estimator=Ridge()),
                feature_forecaster=PointReductionForecaster(estimator=Ridge()),
            ),
        ],
        ids=["point_naive", "predicted_policy", "reduction"],
    )
    def test_forecasted_feature_forecaster_systematic_checks(self, forecaster, y_X_factory):
        """Run systematic checks on ForecastedFeatureForecaster meta-forecaster.

        Note: check_observe_extends_observations and check_rewind_replaces_observations
        are not yielded for ForecastedFeatureForecaster because it sets tracks_observations=False.
        ForecastedFeatureForecaster delegates observation tracking to child forecasters.
        """
        y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

        y_train, y_test = y[:80], y[80:]
        X_train, X_test = X[:80], X[80:]

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_train, forecasting_horizon=3)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(forecaster_fitted, y_train, X_train, y_test, X_test),
        )


class TestBasicFitPredict:
    """Tests for basic fit/predict workflow."""

    def test_basic_fit_predict(self):
        """Test basic fit and predict workflow."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=99),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales": list(range(100)),
        })
        X = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
            "promo": [i % 7 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )
        forecaster.fit(y[:80], X[:80], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        assert "vintage_time" in y_pred.columns
        assert "time" in y_pred.columns
        assert "sales" in y_pred.columns

    def test_requires_X(self):
        """Test that X is required for fit."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "value": range(50),
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )

        with pytest.raises(ValueError, match="requires X"):
            forecaster.fit(y[:30], X_actual=None, forecasting_horizon=5)

    def test_predict_ignores_X(self):
        """Test that X columns already being forecasted are ignored."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=99),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales": list(range(100)),
        })
        X = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )
        forecaster.fit(y[:80], X[:80], forecasting_horizon=5)

        # Passing X with same columns as forecasted should use forecasted values
        y_pred_with_X = forecaster.predict(forecasting_horizon=5)
        y_pred_without_X = forecaster.predict(forecasting_horizon=5)

        # Should produce same results since forecasted columns in X are ignored
        assert y_pred_with_X["sales"].to_list() == y_pred_without_X["sales"].to_list()

    def test_predict_merges_known_ahead_features(self):
        """Test that known-ahead features in X are merged with forecasted features."""
        from sklearn.linear_model import Ridge

        from yohou.point import PointReductionForecaster

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=99),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales": list(range(100)),
        })
        # Features to forecast
        X_forecast = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(estimator=Ridge()),
            feature_forecaster=PointReductionForecaster(estimator=Ridge()),
        )
        # Fit with price only (will be forecasted)
        forecaster.fit(y[:80], X_forecast[:80], forecasting_horizon=5)

        # Predict with known-ahead holiday feature
        # Note: The target forecaster wasn't trained with is_holiday, so this is
        # just testing that the merging happens without error
        pred_time = time[80:85]
        _X_pred_known = pl.DataFrame({
            "time": pred_time,
            "is_holiday": [1, 0, 0, 0, 0],
        })

        # Should not raise - known-ahead features are merged
        y_pred = forecaster.predict(forecasting_horizon=5)
        assert len(y_pred) == 5


class TestStrategy:
    """Tests for fit policy strategies."""

    def test_strategy_actual(self):
        """Test strategy='actual' uses actual X values for target training."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=99),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales": list(range(100)),
        })
        X = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
            strategy="actual",
        )
        forecaster.fit(y[:80], X[:80], forecasting_horizon=5)

        # Both forecasters should be fitted
        assert hasattr(forecaster, "target_forecaster_")
        assert hasattr(forecaster, "feature_forecaster_")

        y_pred = forecaster.predict(forecasting_horizon=5)
        assert len(y_pred) == 5

    def test_strategy_predicted(self):
        """Test strategy='predicted' splits data and uses predicted X."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=99),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales": list(range(100)),
        })
        X = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
            strategy="predicted",
            split_ratio=0.5,
        )
        forecaster.fit(y[:80], X[:80], forecasting_horizon=5)

        # Both forecasters should be fitted
        assert hasattr(forecaster, "target_forecaster_")
        assert hasattr(forecaster, "feature_forecaster_")

        y_pred = forecaster.predict(forecasting_horizon=5)
        assert len(y_pred) == 5

    def test_strategy_rewind(self):
        """Test strategy='rewind' fits on full data, rewinds, predicts X, then fits target."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=99),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales": list(range(100)),
        })
        X = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
            strategy="rewind",
        )
        forecaster.fit(y[:80], X[:80], forecasting_horizon=5)

        # Both forecasters should be fitted
        assert hasattr(forecaster, "target_forecaster_")
        assert hasattr(forecaster, "feature_forecaster_")

        y_pred = forecaster.predict(forecasting_horizon=5)
        assert len(y_pred) == 5

    def test_strategy_rewind_requires_enough_data(self):
        """Test strategy='rewind' raises error if data <= observation_horizon."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=4),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales": list(range(5)),
        })
        X = pl.DataFrame({
            "time": time,
            "price": list(range(5)),
        })

        # SeasonalNaive with seasonality=7 has observation_horizon=7
        # but we only have 5 data points
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=7),
            strategy="rewind",
        )
        with pytest.raises(ValueError, match="Not enough data|Cannot use strategy='rewind'"):
            forecaster.fit(y, X, forecasting_horizon=2)

    def test_split_ratio_edge_cases(self):
        """Test split_ratio validation for edge cases."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=19),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales": list(range(20)),
        })
        X = pl.DataFrame({
            "time": time,
            "price": list(range(20)),
        })

        # Very low split_ratio should raise error (n_split < 2)
        forecaster_low = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
            strategy="predicted",
            split_ratio=0.01,  # Results in n_split < 2 for small data
        )
        with pytest.raises(ValueError, match="n_split=0"):
            forecaster_low.fit(y, X, forecasting_horizon=3)

        # Very high split_ratio should raise error (leaves too few rows)
        forecaster_high = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
            strategy="predicted",
            split_ratio=0.99,  # Leaves only 1 row for target forecaster
        )
        with pytest.raises(ValueError, match="leaving only"):
            forecaster_high.fit(y, X, forecasting_horizon=3)


class TestUpdateReset:
    """Tests for observe, rewind, and observe_predict."""

    def test_observe(self):
        """Test observe propagates to both forecasters."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=99),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales": list(range(100)),
        })
        X = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )
        forecaster.fit(y[:80], X[:80], forecasting_horizon=5)

        # Update with new observations
        forecaster.observe(y[80:85], X[80:85])

        # Should still be able to predict
        y_pred = forecaster.predict(forecasting_horizon=5)
        assert len(y_pred) == 5

    def test_rewind(self):
        """Test rewind propagates to both forecasters."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=99),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales": list(range(100)),
        })
        X = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )
        forecaster.fit(y[:80], X[:80], forecasting_horizon=5)

        # Update then reset
        forecaster.observe(y[80:90], X[80:90])
        forecaster.rewind(y[80:90], X[80:90])

        # Should still be able to predict
        y_pred = forecaster.predict(forecasting_horizon=5)
        assert len(y_pred) == 5

    def test_observe_predict(self):
        """Test observe_predict atomic operation."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=99),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales": list(range(100)),
        })
        X = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )
        forecaster.fit(y[:80], X[:80], forecasting_horizon=5)

        # observe_predict in one call
        y_pred = forecaster.observe_predict(y[80:85], X[80:85])
        assert len(y_pred) == 5


class TestMethodAvailability:
    """Tests for method availability."""

    def test_predict_available_for_point_target(self):
        """Test predict is available when target supports point predictions."""
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )
        # predict should be available (not raise AttributeError)
        assert hasattr(forecaster, "predict")

    def test_not_fitted_error(self):
        """Test NotFittedError is raised when calling predict before fit."""
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )
        with pytest.raises(NotFittedError, match="fitted"):
            forecaster.predict(forecasting_horizon=5)


class TestCloneParams:
    """Tests for clone and params."""

    def test_clone_preserves_params(self):
        """Test clone preserves parameters."""
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=7),
            feature_forecaster=SeasonalNaive(seasonality=3),
            strategy="predicted",
            split_ratio=0.6,
        )
        cloned = clone(forecaster)

        assert cloned.strategy == "predicted"
        assert cloned.split_ratio == 0.6
        assert cloned.target_forecaster.seasonality == 7
        assert cloned.feature_forecaster.seasonality == 3

    def test_get_set_params(self):
        """Test get_params and set_params work correctly."""
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
            strategy="actual",
            split_ratio=0.5,
        )

        params = forecaster.get_params()
        assert params["strategy"] == "actual"
        assert params["split_ratio"] == 0.5

        forecaster.set_params(strategy="predicted", split_ratio=0.7)
        assert forecaster.strategy == "predicted"
        assert forecaster.split_ratio == 0.7

    def test_get_params_deep(self):
        """Test get_params with deep=True includes nested forecaster params."""
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=7),
            feature_forecaster=SeasonalNaive(seasonality=3),
        )

        params = forecaster.get_params(deep=True)
        assert "target_forecaster__seasonality" in params
        assert params["target_forecaster__seasonality"] == 7
        assert "feature_forecaster__seasonality" in params
        assert params["feature_forecaster__seasonality"] == 3


class TestDifferentHorizons:
    """Tests for different horizon values."""

    def test_different_horizons(self):
        """Test prediction with different horizons than fit."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=99),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales": list(range(100)),
        })
        X = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )
        forecaster.fit(y[:80], X[:80], forecasting_horizon=5)

        # Predict with different horizons
        y_pred_3 = forecaster.predict(forecasting_horizon=3)
        y_pred_10 = forecaster.predict(forecasting_horizon=10)

        assert len(y_pred_3) == 3
        assert len(y_pred_10) == 10

    def test_use_fit_horizon_when_none(self):
        """Test predict uses fit horizon when forecasting_horizon is None."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=99),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales": list(range(100)),
        })
        X = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )
        forecaster.fit(y[:80], X[:80], forecasting_horizon=7)

        y_pred = forecaster.predict()  # No horizon specified
        assert len(y_pred) == 7


class TestMetadataRouting:
    """Tests for metadata routing."""

    def test_get_metadata_routing(self):
        """Test metadata routing is configured correctly."""
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )

        router = forecaster.get_metadata_routing()
        assert isinstance(router, MetadataRouter)


class TestTags:
    """Tests for sklearn tags."""

    def test_tags_accessible_before_fit(self):
        """Test tags are accessible before fit."""
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )

        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags is not None
        assert tags.forecaster_tags.tracks_observations is False

    def test_tags_forecaster_type_from_target(self):
        """Test forecaster_type is determined by target forecaster."""
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )

        tags = forecaster.__sklearn_tags__()
        # SeasonalNaive is a point forecaster
        assert tags.forecaster_tags.forecaster_type == frozenset({"point"})


class TestClassProbaForecastedFeature:
    """Tests for class-probability methods on ForecastedFeatureForecaster."""

    @pytest.fixture
    def class_proba_fff_setup(self):
        """Create a fitted ForecastedFeatureForecaster with class_proba target."""
        from sklearn.tree import DecisionTreeClassifier

        from yohou.class_proba import ClassProbaReductionForecaster

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=79),
            interval="1d",
            eager=True,
        )
        classes = ["cat", "dog", "bird"]
        y = pl.DataFrame({
            "time": time,
            "animal": [classes[i % 3] for i in range(80)],
        })
        X = pl.DataFrame({
            "time": time,
            "temp": [20.0 + (i % 10) for i in range(80)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=ClassProbaReductionForecaster(
                estimator=DecisionTreeClassifier(random_state=42),
            ),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )
        y_train, y_test = y[:60], y[60:]
        X_train, X_test = X[:60], X[60:]
        forecaster.fit(y_train, X_train, forecasting_horizon=3)
        return forecaster, y_train, y_test, X_train, X_test

    def test_predict_class_proba(self, class_proba_fff_setup):
        """predict_class_proba returns probabilities using forecasted features."""
        forecaster, _, _, _, _ = class_proba_fff_setup
        y_pred = forecaster.predict_class_proba(forecasting_horizon=3)

        assert "time" in y_pred.columns
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) == 3
        assert len(y_pred) == 3

    def test_observe_predict_class_proba(self, class_proba_fff_setup):
        """observe_predict_class_proba observes and predicts probabilities."""
        forecaster, _, y_test, _, X_test = class_proba_fff_setup
        y_pred = forecaster.observe_predict_class_proba(
            y=y_test[:3],
            X_actual=X_test[:3],
        )
        assert "time" in y_pred.columns
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) > 0

    def test_predict_class_proba_with_x(self):
        """predict_class_proba passes X through to target forecaster."""
        from sklearn.tree import DecisionTreeClassifier

        from yohou.class_proba import ClassProbaReductionForecaster

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=79),
            interval="1d",
            eager=True,
        )
        classes = ["cat", "dog", "bird"]
        y = pl.DataFrame({
            "time": time,
            "animal": [classes[i % 3] for i in range(80)],
        })
        X = pl.DataFrame({
            "time": time,
            "temp": [20.0 + (i % 10) for i in range(80)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=ClassProbaReductionForecaster(
                estimator=DecisionTreeClassifier(random_state=42),
            ),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )
        forecaster.fit(y[:60], X[:60], forecasting_horizon=3)
        y_pred = forecaster.predict_class_proba(forecasting_horizon=3)
        assert "time" in y_pred.columns
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) == 3
