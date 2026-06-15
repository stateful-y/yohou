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
            # With feature_stride > 1 (feature forecast refreshed every 2 steps)
            ForecastedFeatureForecaster(
                target_forecaster=PointReductionForecaster(estimator=Ridge()),
                feature_forecaster=PointReductionForecaster(estimator=Ridge()),
                feature_stride=2,
            ),
        ],
        ids=["point_naive", "predicted_policy", "reduction", "feature_stride"],
    )
    def test_forecasted_feature_forecaster_systematic_checks(self, forecaster, y_X_factory):
        """Run systematic checks on ForecastedFeatureForecaster meta-forecaster.

        Note: check_observe_extends_observations and check_rewind_replaces_observations
        are not yielded for ForecastedFeatureForecaster because it sets tracks_observations=False.
        ForecastedFeatureForecaster delegates observation tracking to child forecasters.
        """
        y, X_actual = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)

        y_train, y_test = y[:80], y[80:]
        X_actual_train, X_actual_test = X_actual[:80], X_actual[80:]

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_actual_train, forecasting_horizon=3)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(forecaster_fitted, y_train, X_actual_train, y_test, X_actual_test),
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
        X_actual = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
            "promo": [i % 7 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )
        forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        assert "vintage_time" in y_pred.columns
        assert "time" in y_pred.columns
        assert "sales" in y_pred.columns

    def test_requires_X(self):
        """Test that X_actual is required for fit."""
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

        with pytest.raises(ValueError, match="requires X_actual"):
            forecaster.fit(y[:30], X_actual=None, forecasting_horizon=5)

    def test_predict_forwards_x_future(self):
        """Known-ahead features passed via X_future are forwarded to the target.

        ``predict`` no longer accepts ``X_actual`` (forecasted features flow through
        the X_forecast channel); a genuinely known-ahead feature travels through
        X_future and must be forwarded to the target without error.
        """
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
        X_actual = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )
        forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=5)

        X_future = pl.DataFrame({
            "time": time[80:85],
            "is_holiday": [1, 0, 0, 0, 0],
        })
        y_pred = forecaster.predict(forecasting_horizon=5, X_future=X_future)
        assert len(y_pred) == 5


class TestStrategy:
    """Tests for fit policy strategies."""

    def test_strategy_actual(self):
        """Test strategy='actual' uses actual X_actual values for target training."""
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
        X_actual = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
            strategy="actual",
        )
        forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=5)

        # Both forecasters should be fitted
        assert hasattr(forecaster, "target_forecaster_")
        assert hasattr(forecaster, "feature_forecaster_")

        y_pred = forecaster.predict(forecasting_horizon=5)
        assert len(y_pred) == 5

    def test_strategy_predicted(self):
        """Test strategy='predicted' splits data and uses predicted X_actual."""
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
        X_actual = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
            strategy="predicted",
            split_ratio=0.5,
        )
        forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=5)

        # Both forecasters should be fitted
        assert hasattr(forecaster, "target_forecaster_")
        assert hasattr(forecaster, "feature_forecaster_")

        y_pred = forecaster.predict(forecasting_horizon=5)
        assert len(y_pred) == 5

    def test_strategy_rewind(self):
        """Test strategy='rewind' fits on full data, rewinds, predicts X_actual, then fits target."""
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
        X_actual = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
            strategy="rewind",
        )
        forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=5)

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
        X_actual = pl.DataFrame({
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
            forecaster.fit(y, X_actual, forecasting_horizon=2)

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
        X_actual = pl.DataFrame({
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
            forecaster_low.fit(y, X_actual, forecasting_horizon=3)

        # Very high split_ratio should raise error (leaves too few rows)
        forecaster_high = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
            strategy="predicted",
            split_ratio=0.99,  # Leaves only 1 row for target forecaster
        )
        with pytest.raises(ValueError, match="leaving only"):
            forecaster_high.fit(y, X_actual, forecasting_horizon=3)


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
        X_actual = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )
        forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=5)

        # Update with new observations
        forecaster.observe(y[80:85], X_actual[80:85])

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
        X_actual = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )
        forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=5)

        # Update then reset
        forecaster.observe(y[80:90], X_actual[80:90])
        forecaster.rewind(y[80:90], X_actual[80:90])

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
        X_actual = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )
        forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=5)

        # observe_predict rolls over the block: an initial prediction plus one per
        # stride-sized slice, producing multiple vintages (not a single end-of-block one).
        y_pred = forecaster.observe_predict(y[80:90], X_actual[80:90], stride=5)
        assert y_pred["vintage_time"].n_unique() > 1
        assert "sales" in y_pred.columns


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
        X_actual = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )
        forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=5)

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
        X_actual = pl.DataFrame({
            "time": time,
            "price": [10 + i % 5 for i in range(100)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )
        forecaster.fit(y[:80], X_actual[:80], forecasting_horizon=7)

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
        X_actual = pl.DataFrame({
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
        X_actual_train, X_actual_test = X_actual[:60], X_actual[60:]
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)
        return forecaster, y_train, y_test, X_actual_train, X_actual_test

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
        forecaster, _, y_test, _, X_actual_test = class_proba_fff_setup
        y_pred = forecaster.observe_predict_class_proba(
            y=y_test[:3],
            X_actual=X_actual_test[:3],
        )
        assert "time" in y_pred.columns
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) > 0

    def test_predict_class_proba_with_x(self):
        """predict_class_proba passes X_actual through to target forecaster."""
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
        X_actual = pl.DataFrame({
            "time": time,
            "temp": [20.0 + (i % 10) for i in range(80)],
        })

        forecaster = ForecastedFeatureForecaster(
            target_forecaster=ClassProbaReductionForecaster(
                estimator=DecisionTreeClassifier(random_state=42),
            ),
            feature_forecaster=SeasonalNaive(seasonality=1),
        )
        forecaster.fit(y[:60], X_actual[:60], forecasting_horizon=3)
        y_pred = forecaster.predict_class_proba(forecasting_horizon=3)
        assert "time" in y_pred.columns
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) == 3


def _make_contemporaneous_data(n: int = 120, seed: int = 0):
    """Build data where the target depends on a future feature value.

    sales[t] = 3 * price[t] + small noise, so the feature forecast genuinely
    drives the target prediction (contemporaneous dependence).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )
    price = 10.0 + 5.0 * np.sin(2 * np.pi * np.arange(n) / 7) + 0.05 * np.arange(n)
    sales = 3.0 * price + rng.normal(0, 0.5, n)
    y = pl.DataFrame({"time": time, "sales": sales})
    X_actual = pl.DataFrame({"time": time, "price": price})
    return y, X_actual


class TestXForecastRouting:
    """Tests for the corrected predict-time data flow (feature forecast via X_forecast)."""

    def test_default_strategy_is_rewind(self):
        """Default strategy is 'rewind' (no train/serve mismatch)."""
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(estimator=Ridge()),
            feature_forecaster=PointReductionForecaster(estimator=Ridge()),
        )
        assert forecaster.strategy == "rewind"

    def test_feature_forecast_changes_predictions(self):
        """Two different feature forecasters yield different target predictions."""
        y, X_actual = _make_contemporaneous_data()
        y_train, X_train = y[:100], X_actual[:100]

        common = {
            "target_forecaster": PointReductionForecaster(estimator=Ridge()),
            "strategy": "rewind",
        }
        fc_a = ForecastedFeatureForecaster(feature_forecaster=PointReductionForecaster(estimator=Ridge()), **common)
        fc_b = ForecastedFeatureForecaster(feature_forecaster=SeasonalNaive(seasonality=1), **common)
        fc_a.fit(y_train, X_train, forecasting_horizon=7)
        fc_b.fit(y_train, X_train, forecasting_horizon=7)

        pred_a = fc_a.predict(forecasting_horizon=7)
        pred_b = fc_b.predict(forecasting_horizon=7)
        assert not pred_a["sales"].equals(pred_b["sales"])

    @pytest.mark.parametrize("strategy", ["actual", "predicted", "rewind"])
    def test_feature_forecaster_invoked_at_predict(self, strategy):
        """feature_forecaster_ is consulted at predict time in every strategy."""
        from unittest.mock import patch

        y, X_actual = _make_contemporaneous_data()
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(estimator=Ridge()),
            feature_forecaster=PointReductionForecaster(estimator=Ridge()),
            strategy=strategy,
            split_ratio=0.6,
        )
        forecaster.fit(y[:100], X_actual[:100], forecasting_horizon=5)

        real_predict = forecaster.feature_forecaster_.predict
        with patch.object(forecaster.feature_forecaster_, "predict", wraps=real_predict) as spy:
            forecaster.predict(forecasting_horizon=5)
        assert spy.called

    @pytest.mark.parametrize("strategy", ["actual", "predicted", "rewind"])
    def test_observed_time_parity_after_fit(self, strategy):
        """feature and target forecasters share the same observed_time_ after fit."""
        y, X_actual = _make_contemporaneous_data()
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(estimator=Ridge()),
            feature_forecaster=PointReductionForecaster(estimator=Ridge()),
            strategy=strategy,
            split_ratio=0.6,
        )
        forecaster.fit(y[:100], X_actual[:100], forecasting_horizon=5)
        assert forecaster.feature_forecaster_.observed_time_ == forecaster.target_forecaster_.observed_time_

    def test_merge_forecasts_disjoint_columns(self):
        """_merge_forecasts joins disjoint external-forecast columns on (vintage_time, time)."""
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(estimator=Ridge()),
            feature_forecaster=PointReductionForecaster(estimator=Ridge()),
        )
        t0 = datetime(2020, 1, 1)
        feat = pl.DataFrame({
            "vintage_time": [t0, t0],
            "time": [t0 + timedelta(days=1), t0 + timedelta(days=2)],
            "price": [1.0, 2.0],
        })
        user = pl.DataFrame({
            "vintage_time": [t0, t0],
            "time": [t0 + timedelta(days=1), t0 + timedelta(days=2)],
            "promo": [1.0, 0.0],
        })
        merged = forecaster._merge_forecasts(feat, user)
        assert set(merged.columns) == {"vintage_time", "time", "price", "promo"}
        assert merged.height == 2
        # None passthrough returns the feature forecast unchanged.
        assert forecaster._merge_forecasts(feat, None) is feat

    def test_user_x_forecast_collision_raises(self):
        """A caller X_forecast colliding with a forecasted feature raises ValueError."""
        y, X_actual = _make_contemporaneous_data()
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(estimator=Ridge()),
            feature_forecaster=PointReductionForecaster(estimator=Ridge()),
            strategy="rewind",
        )
        forecaster.fit(y[:100], X_actual[:100], forecasting_horizon=5)

        origin = forecaster.target_forecaster_.observed_time_
        future_times = pl.datetime_range(
            start=origin + timedelta(days=1),
            end=origin + timedelta(days=5),
            interval="1d",
            eager=True,
        )
        colliding = pl.DataFrame({
            "vintage_time": [origin] * 5,
            "time": future_times,
            "price": [1.0, 2.0, 3.0, 4.0, 5.0],  # same name as the forecasted feature
        })
        with pytest.raises(ValueError, match="collides"):
            forecaster.predict(forecasting_horizon=5, X_forecast=colliding)

    def test_non_exogenous_target_degrades_gracefully(self):
        """A requires_exogenous=False target fits and predicts without error."""
        y, X_actual = _make_contemporaneous_data()
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=7),
            feature_forecaster=PointReductionForecaster(estimator=Ridge()),
            strategy="rewind",
        )
        forecaster.fit(y[:100], X_actual[:100], forecasting_horizon=5)
        pred = forecaster.predict(forecasting_horizon=5)
        assert len(pred) == 5


def _make_panel_contemporaneous_data(n: int = 120, seed: int = 0):
    """Two-group panel where each group's target depends on its feature.

    sales[g, t] = 3 * price[g, t] + noise, with `group__column` naming.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )
    y_cols = {"time": time}
    x_cols = {"time": time}
    for g, phase in (("s1", 0.0), ("s2", 1.5)):
        price = 10.0 + 5.0 * np.sin(2 * np.pi * np.arange(n) / 7 + phase) + 0.05 * np.arange(n)
        sales = 3.0 * price + rng.normal(0, 0.5, n)
        y_cols[f"{g}__sales"] = sales
        x_cols[f"{g}__price"] = price
    return pl.DataFrame(y_cols), pl.DataFrame(x_cols)


class TestHardening:
    """Tests for review-hardening fixes (panel actual, state parity, merge, errors)."""

    def test_panel_actual_builds_group_prefixed_step_columns(self):
        """Panel + strategy='actual' + reduction target feeds the forecast (not empty)."""
        from yohou.preprocessing import LagTransformer

        y, X = _make_panel_contemporaneous_data()
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(
                estimator=Ridge(), feature_transformer=LagTransformer(lag=[1, 2])
            ),
            feature_forecaster=PointReductionForecaster(
                estimator=Ridge(), feature_transformer=LagTransformer(lag=[1, 2])
            ),
            strategy="actual",
        )
        forecaster.fit(y[:100], X[:100], forecasting_horizon=3)

        step_cols = forecaster.target_forecaster_._step_column_names_
        assert step_cols, "panel strategy='actual' must not drop the feature forecast"
        assert any(c.startswith("s1__price") for c in step_cols)
        assert any(c.startswith("s2__price") for c in step_cols)

    def test_observe_requires_x_actual(self):
        """observe(y, X_actual=None) raises, keeping state parity."""
        y, X_actual = _make_contemporaneous_data()
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(estimator=Ridge()),
            feature_forecaster=PointReductionForecaster(estimator=Ridge()),
        )
        forecaster.fit(y[:100], X_actual[:100], forecasting_horizon=5)
        with pytest.raises(ValueError, match="requires X_actual"):
            forecaster.observe(y[100:105], X_actual=None)

    def test_observe_predict_requires_x_actual(self):
        """observe_predict(y, X_actual=None) raises (delegates to observe)."""
        y, X_actual = _make_contemporaneous_data()
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(estimator=Ridge()),
            feature_forecaster=PointReductionForecaster(estimator=Ridge()),
        )
        forecaster.fit(y[:100], X_actual[:100], forecasting_horizon=5)
        with pytest.raises(ValueError, match="requires X_actual"):
            forecaster.observe_predict(y[100:105], X_actual=None)

    def test_rewind_requires_x_actual(self):
        """rewind(y, X_actual=None) raises."""
        y, X_actual = _make_contemporaneous_data()
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(estimator=Ridge()),
            feature_forecaster=PointReductionForecaster(estimator=Ridge()),
        )
        forecaster.fit(y[:100], X_actual[:100], forecasting_horizon=5)
        with pytest.raises(ValueError, match="requires X_actual"):
            forecaster.rewind(y[95:100], X_actual=None)

    def test_observe_keeps_observed_time_parity(self):
        """observe with aligned y + X_actual keeps feature/target observed_time_ equal."""
        y, X_actual = _make_contemporaneous_data()
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(estimator=Ridge()),
            feature_forecaster=PointReductionForecaster(estimator=Ridge()),
        )
        forecaster.fit(y[:100], X_actual[:100], forecasting_horizon=5)
        forecaster.observe(y[100:105], X_actual[100:105])
        assert forecaster.feature_forecaster_.observed_time_ == forecaster.target_forecaster_.observed_time_

    def test_merge_forecasts_does_not_introduce_null_rows(self):
        """_merge_forecasts keeps exactly the meta forecast's (vintage_time, time) rows."""
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(estimator=Ridge()),
            feature_forecaster=PointReductionForecaster(estimator=Ridge()),
        )
        t0 = datetime(2020, 1, 1)
        meta = pl.DataFrame({
            "vintage_time": [t0, t0],
            "time": [t0 + timedelta(days=1), t0 + timedelta(days=2)],
            "price": [1.0, 2.0],
        })
        # Caller forecast with one aligning row and one non-aligning row.
        user = pl.DataFrame({
            "vintage_time": [t0, t0],
            "time": [t0 + timedelta(days=2), t0 + timedelta(days=99)],
            "promo": [1.0, 0.0],
        })
        merged = forecaster._merge_forecasts(meta, user)
        assert merged.height == 2  # no spurious row from the non-aligning user row
        assert set(merged["time"].to_list()) == {t0 + timedelta(days=1), t0 + timedelta(days=2)}
        assert set(merged.columns) == {"vintage_time", "time", "price", "promo"}

    def test_rewind_short_series_actionable_error(self):
        """strategy='rewind' on a too-short series raises an actionable error.

        The feature forecaster fits on the 3 rows but the rewind guard
        (len(X_actual) <= observation_horizon + 1) still fires.
        """
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=2),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "sales": [float(i) for i in range(3)]})
        X_actual = pl.DataFrame({"time": time, "price": [float(10 + i) for i in range(3)]})
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=SeasonalNaive(seasonality=2),
            strategy="rewind",
        )
        with pytest.raises(ValueError, match="strategy='actual'"):
            forecaster.fit(y, X_actual, forecasting_horizon=1)


def _make_fs_data(n: int = 140, seed: int = 0):
    """Univariate series where the target depends on a future feature value."""
    import numpy as np

    rng = np.random.default_rng(seed)
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=n - 1),
        interval="1d",
        eager=True,
    )
    price = 10.0 + 5.0 * np.sin(2 * np.pi * np.arange(n) / 7) + 0.05 * np.arange(n)
    sales = 3.0 * price + rng.normal(0, 0.5, n)
    return (
        pl.DataFrame({"time": time, "sales": sales}),
        pl.DataFrame({"time": time, "price": price}),
    )


def _reduction_fff(strategy="rewind", feature_stride=1):
    from yohou.preprocessing import LagTransformer

    return ForecastedFeatureForecaster(
        target_forecaster=PointReductionForecaster(estimator=Ridge(), feature_transformer=LagTransformer(lag=[1, 2])),
        feature_forecaster=PointReductionForecaster(estimator=Ridge(), feature_transformer=LagTransformer(lag=[1, 2])),
        strategy=strategy,
        feature_stride=feature_stride,
        split_ratio=0.5,
    )


class TestFeatureStride:
    """Tests for the feature_stride cadence parameter."""

    def test_default_is_one(self):
        forecaster = _reduction_fff()
        assert forecaster.feature_stride == 1

    @pytest.mark.parametrize("strategy", ["actual", "predicted", "rewind"])
    def test_fit_feature_stride_extends_feature_horizon_no_nan(self, strategy):
        """feature_stride>1 fits a Ridge target without NaN; feature horizon is H+F-1."""
        y, X = _make_fs_data()
        H, F = 5, 3
        forecaster = _reduction_fff(strategy=strategy, feature_stride=F)
        forecaster.fit(y[:110], X[:110], forecasting_horizon=H)
        # feature forecaster fit at the extended horizon
        assert forecaster.feature_forecaster_.fit_forecasting_horizon_ == H + F - 1
        # target trained with full H step columns (no cadence-induced nulls -> Ridge ok)
        assert len(forecaster.target_forecaster_._step_column_names_) == H

    def test_fit_vintages_spaced_by_feature_stride(self):
        """The in-sample feature forecast (actual strategy) has vintages on the F grid."""
        y, X = _make_fs_data()
        F = 4
        forecaster = _reduction_fff(strategy="actual", feature_stride=F)
        forecaster.fit(y[:120], X[:120], forecasting_horizon=5)
        feat = forecaster._actuals_as_forecast(X[:120].select("time", "price"), 5 + F - 1, F)
        vintages = feat["vintage_time"].unique().sort()
        # consecutive vintages are F steps apart
        gaps = vintages.diff().drop_nulls().dt.total_days().unique().to_list()
        assert gaps == [F]

    def test_serve_rolling_multi_vintage_at_feature_stride(self):
        """observe_predict at feature_stride>1 rolls and clears the buffer afterward."""
        y, X = _make_fs_data()
        forecaster = _reduction_fff(strategy="rewind", feature_stride=3)
        forecaster.fit(y[:110], X[:110], forecasting_horizon=5)
        pred = forecaster.observe_predict(y[110:130], X[110:130], stride=5)
        assert pred["vintage_time"].n_unique() > 1
        # buffer state is cleaned up
        assert getattr(forecaster, "_feature_forecast_buffer", None) is None

    def test_feature_stride_changes_predictions(self):
        """feature_stride>1 (reused aging forecast) yields different serve output than F=1."""
        y, X = _make_fs_data()
        fc1 = _reduction_fff(strategy="rewind", feature_stride=1)
        fc3 = _reduction_fff(strategy="rewind", feature_stride=3)
        fc1.fit(y[:110], X[:110], forecasting_horizon=5)
        fc3.fit(y[:110], X[:110], forecasting_horizon=5)
        p1 = fc1.observe_predict(y[110:130], X[110:130], stride=5)
        p3 = fc3.observe_predict(y[110:130], X[110:130], stride=5)
        assert not p1["sales"].equals(p3["sales"])

    def test_non_exogenous_target_uses_base_loop(self):
        """A requires_exogenous=False target uses the base loop even at feature_stride>1."""
        y, X = _make_fs_data()
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=1),
            feature_forecaster=PointReductionForecaster(estimator=Ridge()),
            strategy="rewind",
            feature_stride=3,
        )
        forecaster.fit(y[:110], X[:110], forecasting_horizon=5)
        pred = forecaster.observe_predict(y[110:130], X[110:130], stride=5)
        assert pred["vintage_time"].n_unique() > 1
        assert getattr(forecaster, "_feature_forecast_buffer", None) is None

    def test_panel_feature_stride(self):
        """feature_stride>1 works on panel data with an exogenous target."""
        from yohou.preprocessing import LagTransformer

        y, X = _make_panel_contemporaneous_data()
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(
                estimator=Ridge(), feature_transformer=LagTransformer(lag=[1, 2])
            ),
            feature_forecaster=PointReductionForecaster(
                estimator=Ridge(), feature_transformer=LagTransformer(lag=[1, 2])
            ),
            strategy="rewind",
            feature_stride=3,
        )
        forecaster.fit(y[:100], X[:100], forecasting_horizon=3)
        assert forecaster.target_forecaster_._step_column_names_
        pred = forecaster.observe_predict(y[100:115], X[100:115], stride=3)
        assert pred["vintage_time"].n_unique() > 1

    def test_invalid_feature_stride_rejected(self):
        """feature_stride < 1 is rejected by parameter validation."""
        y, X = _make_fs_data()
        forecaster = _reduction_fff(feature_stride=0)
        with pytest.raises((ValueError, Exception)):
            forecaster.fit(y[:110], X[:110], forecasting_horizon=5)

    def test_short_series_for_feature_stride_raises_actionably(self):
        """A feature_stride too large for the data raises an error naming feature_stride."""
        y, X = _make_fs_data()
        # H + feature_stride - 1 >= len(X) is guarded with an actionable message.
        forecaster = _reduction_fff(feature_stride=20)
        with pytest.raises(ValueError, match="feature_stride"):
            forecaster.fit(y[:18], X[:18], forecasting_horizon=5)

    def test_short_series_for_feature_stride_one_raises(self):
        """The length guard also fires for feature_stride=1, not just > 1.

        With feature_stride=1, feature_horizon == forecasting_horizon, so the
        guard must still reject a forecasting_horizon that meets or exceeds the
        available data length.
        """
        y, X = _make_fs_data()
        forecaster = _reduction_fff(feature_stride=1)
        with pytest.raises(ValueError, match="feature_stride"):
            forecaster.fit(y[:5], X[:5], forecasting_horizon=10)

    @pytest.mark.parametrize("variant", ["observe_predict", "observe_predict_interval"])
    def test_observe_predict_variants_roll(self, variant):
        """Point and interval observe_predict produce rolling multi-vintage output."""
        from yohou.interval import IntervalReductionForecaster
        from yohou.preprocessing import LagTransformer

        y, X = _make_fs_data()
        if variant == "observe_predict":
            target = PointReductionForecaster(estimator=Ridge(), feature_transformer=LagTransformer(lag=[1, 2]))
        else:
            target = IntervalReductionForecaster(feature_transformer=LagTransformer(lag=[1, 2]))
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=target,
            feature_forecaster=PointReductionForecaster(
                estimator=Ridge(), feature_transformer=LagTransformer(lag=[1, 2])
            ),
            strategy="rewind",
        )
        forecaster.fit(y[:110], X[:110], forecasting_horizon=5)
        pred = getattr(forecaster, variant)(y[110:125], X[110:125], stride=5)
        assert pred["vintage_time"].n_unique() > 1

    def test_cross_val_predict_rolls_per_origin(self):
        """cross_val_predict on an FFF yields rolling per-origin blocks (a split column)."""
        from yohou.model_selection import SlidingWindowSplitter, cross_val_predict

        y, X = _make_fs_data(n=160)
        forecaster = _reduction_fff(strategy="rewind")
        splitter = SlidingWindowSplitter(train_size=60, test_size=15, n_splits=2)
        preds = cross_val_predict(
            forecaster,
            y,
            X_actual=X,
            forecasting_horizon=5,
            cv=splitter,
            predict_stride=5,
        )
        # one fold-tagged frame with multiple vintages per fold (rolling, not single-shot)
        assert "split" in preds.columns
        assert preds["vintage_time"].n_unique() > 2


class _RecordingPredictFeatureForecaster(PointReductionForecaster):
    """Feature forecaster that records whether predict received routed time_weight."""

    def predict(self, forecasting_horizon=None, groups=None, time_weight=None, **kwargs):
        self.__dict__.setdefault("_tw_seen", [])
        self._tw_seen.append(time_weight is not None)
        return super().predict(forecasting_horizon=forecasting_horizon, groups=groups, **kwargs)


class _CountingObservePredictFeatureForecaster(PointReductionForecaster):
    """Feature forecaster that counts observe_predict calls (rolling-forecast probe)."""

    def observe_predict(self, *args, **kwargs):
        self.__dict__["_op_calls"] = self.__dict__.get("_op_calls", 0) + 1
        return super().observe_predict(*args, **kwargs)


class TestFeatureForecasterMetadataRouting:
    """The feature forecaster's predict metadata is honored regardless of feature_stride."""

    @pytest.mark.parametrize("feature_stride", [1, 3])
    def test_predict_metadata_routed_under_feature_stride(self, feature_stride):
        """Routed predict-time metadata reaches the feature forecaster during a strided roll.

        Before the fix, the feature_stride>1 buffer refresh used a bare predict() that
        dropped routed metadata, while feature_stride==1 honored it. This pins parity.
        """
        import numpy as np

        y, X = _make_fs_data()
        feature_forecaster = _RecordingPredictFeatureForecaster(estimator=Ridge())
        feature_forecaster.set_predict_request(time_weight=True)
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=PointReductionForecaster(estimator=Ridge()),
            feature_forecaster=feature_forecaster,
            feature_stride=feature_stride,
        )
        forecaster.fit(y[:110], X[:110], forecasting_horizon=5)
        # Isolate serve-time predicts (fit also predicts, without routed metadata).
        forecaster.feature_forecaster_._tw_seen.clear()

        time_weight = np.linspace(0.1, 1.0, 5)
        forecaster.observe_predict(y[110:130], X[110:130], stride=5, time_weight=time_weight)

        seen = forecaster.feature_forecaster_._tw_seen
        assert seen, "feature forecaster predict was never called during the roll"
        # Every serve-time feature-forecast generation received the routed metadata.
        assert all(seen)


class TestNonExogenousFitSkipsRollingForecast:
    """A non-exogenous target skips the in-sample forecast but keeps parity and validation."""

    @pytest.mark.parametrize("strategy", ["rewind", "predicted"])
    def test_no_rolling_forecast_for_non_exogenous_target(self, strategy):
        """The feature forecaster is not rolled (observe_predict) when the target ignores it."""
        y, X_actual = _make_contemporaneous_data()
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=7),
            feature_forecaster=_CountingObservePredictFeatureForecaster(estimator=Ridge()),
            strategy=strategy,
            split_ratio=0.6,
        )
        forecaster.fit(y[:100], X_actual[:100], forecasting_horizon=5)
        assert forecaster.feature_forecaster_.__dict__.get("_op_calls", 0) == 0

    def test_parity_preserved_for_non_exogenous_target(self):
        """Observed-time parity holds even though the rolling forecast is skipped."""
        y, X_actual = _make_contemporaneous_data()
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=7),
            feature_forecaster=PointReductionForecaster(estimator=Ridge()),
            strategy="rewind",
        )
        forecaster.fit(y[:100], X_actual[:100], forecasting_horizon=5)
        assert forecaster.feature_forecaster_.observed_time_ == forecaster.target_forecaster_.observed_time_

    def test_predictions_match_bare_target(self):
        """The target ignores the feature forecast, so predictions equal the bare target's."""
        y, X_actual = _make_contemporaneous_data()
        forecaster = ForecastedFeatureForecaster(
            target_forecaster=SeasonalNaive(seasonality=7),
            feature_forecaster=PointReductionForecaster(estimator=Ridge()),
            strategy="rewind",
        )
        forecaster.fit(y[:100], X_actual[:100], forecasting_horizon=5)
        meta_pred = forecaster.predict(forecasting_horizon=5)

        bare = SeasonalNaive(seasonality=7)
        bare.fit(y[:100], forecasting_horizon=5)
        bare_pred = bare.predict(forecasting_horizon=5)

        assert meta_pred["sales"].to_list() == bare_pred["sales"].to_list()
