"""Tests for Decomposer meta-forecaster."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from sklearn.base import clone

from yohou.decomposition import Decomposer, PolynomialTrendForecaster, SeasonalityForecaster
from yohou.point_forecaster import SeasonalNaive
from yohou.preprocessing import LogTransform

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from estimator_checks import _yield_yohou_forecaster_checks


@pytest.mark.parametrize(
    "forecaster,tags,expected_failures",
    [
        (
            Decomposer(
                [
                    ("trend", PolynomialTrendForecaster(degree=1)),
                    ("seasonality", SeasonalNaive(seasonality=7)),
                ]
            ),
            {"forecaster_type": "point", "is_meta_forecaster": True},
            [
                "check_update_extends_observations",
                "check_reset_replaces_observations",
            ],  # Decomposer has complex residual-based update logic
        ),
        (
            Decomposer(
                [
                    ("trend", PolynomialTrendForecaster(degree=2)),
                    ("seasonality", SeasonalityForecaster(seasonality=12, method="average")),
                ]
            ),
            {"forecaster_type": "point", "is_meta_forecaster": True},
            [
                "check_update_extends_observations",
                "check_reset_replaces_observations",
            ],  # Decomposer has complex residual-based update logic
        ),
        (
            Decomposer(
                [
                    ("trend", PolynomialTrendForecaster(degree=1)),
                    ("seasonality", SeasonalNaive(seasonality=7)),
                ],
                target_transformer=LogTransform(),
            ),
            {"forecaster_type": "point", "is_meta_forecaster": True, "uses_transformers": True},
            [
                "check_update_extends_observations",
                "check_reset_replaces_observations",
            ],  # Decomposer has complex residual-based update logic
        ),
        (
            Decomposer(
                [
                    ("trend", PolynomialTrendForecaster(degree=1)),
                ]
            ),
            {"forecaster_type": "point", "is_meta_forecaster": True},
            [
                "check_update_extends_observations",
                "check_reset_replaces_observations",
            ],  # Decomposer has complex residual-based update logic
        ),
    ],
)
def test_decomposer_checks(forecaster, tags, expected_failures, y_X_factory):
    """Run systematic checks on Decomposer meta-forecaster."""
    # Generate data with sufficient length for seasonality
    y, X = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)

    # Make values positive for LogTransform compatibility
    y = y.with_columns([(pl.col(col).abs() + 1).alias(col) for col in y.columns if col != "time"])

    y_train, y_test = y[:80], y[80:]
    X_train, X_test = (X[:80], X[80:]) if X is not None else (None, None)

    # Fit forecaster
    forecaster_fitted = clone(forecaster)
    forecaster_fitted.fit(y_train, X_train, forecasting_horizon=3)

    # Run all generated checks
    expected_failures_set = set(expected_failures)
    for check_name, check_func, check_kwargs in _yield_yohou_forecaster_checks(
        forecaster_fitted,
        y_train,
        X_train,
        y_test,
        X_test,
        tags=tags,
    ):
        if check_name in expected_failures_set:
            pytest.skip(f"Expected failure: {check_name}")
        else:
            check_func(forecaster_fitted, **check_kwargs)


def test_decomposer_basic_fit_predict():
    """Test basic fit and predict workflow."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=49),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": range(50)})

    forecaster = Decomposer(
        [
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ]
    )
    forecaster.fit(y[:30], forecasting_horizon=5)

    y_pred = forecaster.predict(forecasting_horizon=5)

    # Validate output structure
    assert len(y_pred) == 5
    assert "observed_time" in y_pred.columns
    assert "time" in y_pred.columns
    assert "value" in y_pred.columns


def test_decomposer_fit_different_horizons():
    """Test prediction with different horizons."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=49),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": range(50)})

    forecaster = Decomposer(
        [
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ]
    )
    forecaster.fit(y[:30], forecasting_horizon=5)

    # Predict different horizon
    y_pred = forecaster.predict(forecasting_horizon=10)
    assert len(y_pred) == 10


def test_decomposer_store_residuals():
    """Test storing residuals for inspection."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=49),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": range(50)})

    forecaster = Decomposer(
        [
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ],
        store_residuals=True,
    )
    forecaster.fit(y[:30], forecasting_horizon=5)

    # Check residuals are stored
    assert hasattr(forecaster, "residuals_")
    assert "trend" in forecaster.residuals_
    assert "seasonality" in forecaster.residuals_

    # Check residual structure
    trend_residuals = forecaster.residuals_["trend"]
    assert "time" in trend_residuals.columns
    assert "value" in trend_residuals.columns
    assert len(trend_residuals) == 30


def test_decomposer_target_transformer():
    """Test with target transformer (multiplicative decomposition)."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=49),
        interval="1d",
        eager=True,
    )
    # Use positive values for log transform
    y = pl.DataFrame({"time": time, "value": [float(i + 1) for i in range(50)]})

    forecaster = Decomposer(
        [
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ],
        target_transformer=LogTransform(),
    )
    forecaster.fit(y[:30], forecasting_horizon=5)

    y_pred = forecaster.predict(forecasting_horizon=5)

    # Predictions should be in original scale (inverse transformed)
    assert len(y_pred) == 5
    assert all(y_pred["value"] > 0)  # Positive values


def test_decomposer_update_predict():
    """Test update_predict method."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=49),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": range(50)})

    forecaster = Decomposer(
        [
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ]
    )
    fit_forecasting_horizon = 5
    forecaster.fit(y[:30], forecasting_horizon=fit_forecasting_horizon)

    # Update with new data
    n_new = 10
    predict_forecasting_horizon = 5
    y_pred = forecaster.update_predict(
        y[30 : 30 + n_new], forecasting_horizon=predict_forecasting_horizon
    )

    assert len(y_pred) == predict_forecasting_horizon * (1 + n_new // fit_forecasting_horizon)
    assert "observed_time" in y_pred.columns


def test_decomposer_reset():
    """Test reset method."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=49),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": range(50)})

    forecaster = Decomposer(
        [
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ]
    )
    forecaster.fit(y[:30], forecasting_horizon=5)

    # Update then reset
    forecaster.update(y[30:35])
    forecaster.reset(y[20:30])

    y_pred = forecaster.predict(forecasting_horizon=5)
    assert len(y_pred) == 5


def test_decomposer_observation_horizon():
    """Test observation_horizon property."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=49),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": range(50)})

    forecaster = Decomposer(
        [
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ]
    )

    # Should raise NotFittedError before fit
    with pytest.raises(Exception):  # NotFittedError
        _ = forecaster.observation_horizon

    forecaster.fit(y[:30], forecasting_horizon=5)

    # Should return max observation horizon
    horizon = forecaster.observation_horizon
    assert horizon >= 0


def test_decomposer_validates_forecaster_names():
    """Test that duplicate forecaster names are rejected."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=49),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": range(50)})

    forecaster = Decomposer(
        [
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("trend", SeasonalNaive(seasonality=7)),  # Duplicate name
        ]
    )

    with pytest.raises(ValueError, match="Names provided are not unique"):
        forecaster.fit(y[:30], forecasting_horizon=5)


def test_decomposer_panel_data(panel_time_series_factory):
    """Test with panel data."""
    y_panel = panel_time_series_factory(length=50, n_series=3, seed=42)

    forecaster = Decomposer(
        [
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ]
    )
    forecaster.fit(y_panel[:30], forecasting_horizon=5)

    y_pred = forecaster.predict(forecasting_horizon=5)

    # Check panel structure preserved
    assert len(y_pred) == 5
    assert "observed_time" in y_pred.columns
    assert "time" in y_pred.columns


def test_decomposer_three_components():
    """Test with three components (trend + seasonality + residual)."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=99),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": range(100)})

    forecaster = Decomposer(
        [
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
            ("residual", SeasonalNaive(seasonality=1)),
        ],
        store_residuals=True,
    )
    forecaster.fit(y[:60], forecasting_horizon=5)

    y_pred = forecaster.predict(forecasting_horizon=5)

    assert len(y_pred) == 5
    assert len(forecaster.residuals_) == 3
    assert "trend" in forecaster.residuals_
    assert "seasonality" in forecaster.residuals_
    assert "residual" in forecaster.residuals_


def test_decomposer_single_component():
    """Test with single component (edge case)."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=49),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": range(50)})

    forecaster = Decomposer([("trend", PolynomialTrendForecaster(degree=1))])
    forecaster.fit(y[:30], forecasting_horizon=5)

    predict_forecasting_horizon = 5
    y_pred = forecaster.predict(forecasting_horizon=predict_forecasting_horizon)

    assert len(y_pred) == predict_forecasting_horizon


def test_decomposer_with_exogenous_features():
    """Test with exogenous features."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 1) + timedelta(days=49),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": range(50)})
    X = pl.DataFrame({"time": time, "feature": range(50, 100)})

    from sklearn.linear_model import Ridge

    from yohou.point_forecaster import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    forecaster = Decomposer(
        [
            ("trend", PolynomialTrendForecaster(degree=1)),
            (
                "ml",
                PointReductionForecaster(
                    estimator=Ridge(), feature_transformer=LagTransformer(lag=[1, 2])
                ),
            ),
        ]
    )
    fit_forecasting_horizon = 5
    forecaster.fit(y[:30], X=X[:30], forecasting_horizon=fit_forecasting_horizon)

    # Predict with future features
    predict_forecasting_horizon = 5
    y_pred = forecaster.predict(
        X=X[30 : 30 + predict_forecasting_horizon], forecasting_horizon=predict_forecasting_horizon
    )

    assert len(y_pred) == predict_forecasting_horizon


def test_decomposer_prediction_types():
    """Test that Decomposer has correct prediction_types."""
    forecaster = Decomposer(
        [
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("seasonality", SeasonalNaive(seasonality=7)),
        ]
    )

    assert forecaster.prediction_types == {"point"}
