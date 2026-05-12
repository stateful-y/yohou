"""Integration tests for custom transformer extensibility.

Verifies that stateless/invertible and stateful transformer extension
patterns work end to end with systematic check generators and manual
assertions.
"""

from __future__ import annotations

import numbers

import numpy as np
import polars as pl
import polars.selectors as cs
import pytest
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from conftest import run_checks
from yohou.base.transformer import BaseTransformer
from yohou.compose import FeaturePipeline
from yohou.testing import _yield_yohou_transformer_checks

# ---------------------------------------------------------------------------
# Custom estimators
# ---------------------------------------------------------------------------


class _ScaleTransformer(BaseTransformer):
    """Multiplies all numeric columns by a constant factor (stateless, invertible)."""

    _tags = {"invertible": True}

    _parameter_constraints: dict = {
        **BaseTransformer._parameter_constraints,
        "factor": [Interval(numbers.Real, 0, None, closed="neither")],
    }

    def __init__(self, factor: float = 2.0):
        self.factor = factor

    def _fit(self, X, y=None):
        self._observation_horizon = 0

    def _transform(self, X):
        return X.select(
            pl.col("time"),
            (cs.numeric() & ~cs.by_name("time")) * self.factor,
        )

    def _inverse_transform(self, X_t, X_p=None):
        return X_t.select(
            pl.col("time"),
            (cs.numeric() & ~cs.by_name("time")) / self.factor,
        )

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            check_is_fitted(self)
            input_features = self.feature_names_in_
        return list(input_features)


class _WindowStatsTransformer(BaseTransformer):
    """Computes rolling mean over the last ``window_size`` rows (stateful, non-invertible)."""

    _tags = {"stateful": True}

    _parameter_constraints: dict = {
        **BaseTransformer._parameter_constraints,
        "window_size": [Interval(numbers.Integral, 2, None, closed="left")],
    }

    def __init__(self, window_size: int = 3):
        self.window_size = window_size

    def _fit(self, X, y=None):
        self._observation_horizon = self.window_size

    def _transform(self, X):
        num_cols = [c for c in X.columns if c != "time"]
        rolling = X.select(
            pl.col("time"),
            *[pl.col(c).rolling_mean(window_size=self.window_size).alias(c) for c in num_cols],
        )
        # Drop the first observation_horizon rows (warmup)
        return rolling.slice(self.observation_horizon)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            check_is_fitted(self)
            input_features = self.feature_names_in_
        return list(input_features)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def numeric_series():
    rng = np.random.default_rng(42)
    n = 100
    return pl.DataFrame({
        "time": pl.datetime_range(
            pl.lit("2020-01-01").str.to_datetime(),
            pl.lit("2020-01-01").str.to_datetime() + pl.duration(days=n - 1),
            interval="1d",
            eager=True,
        ),
        "value": rng.normal(10, 1, n).tolist(),
    })


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestScaleTransformer:
    """Verify stateless invertible transformer extension."""

    def test_systematic_checks(self, numeric_series):
        X_train, X_test = numeric_series[:80], numeric_series[80:]
        transformer = _ScaleTransformer(factor=3.0)
        transformer.fit(X_train)

        run_checks(
            transformer,
            _yield_yohou_transformer_checks(transformer, X_train, None, X_test),
        )

    def test_transform_values(self, numeric_series):
        t = _ScaleTransformer(factor=2.0)
        t.fit(numeric_series)
        X_t = t.transform(numeric_series)
        original = numeric_series.select(cs.numeric() & ~cs.by_name("time")).to_numpy()
        transformed = X_t.select(cs.numeric() & ~cs.by_name("time")).to_numpy()
        np.testing.assert_allclose(transformed, original * 2.0)

    def test_inverse_recovers_original(self, numeric_series):
        t = _ScaleTransformer(factor=5.0)
        t.fit(numeric_series)
        X_t = t.transform(numeric_series)
        X_inv = t.inverse_transform(X_t)
        original = numeric_series.select(cs.numeric() & ~cs.by_name("time")).to_numpy()
        recovered = X_inv.select(cs.numeric() & ~cs.by_name("time")).to_numpy()
        np.testing.assert_allclose(recovered, original, atol=1e-12)

    def test_observation_horizon_is_zero(self):
        t = _ScaleTransformer(factor=2.0)
        # Must be fitted to access observation_horizon property
        X = pl.DataFrame({
            "time": pl.datetime_range(
                pl.lit("2020-01-01").str.to_datetime(),
                pl.lit("2020-01-10").str.to_datetime(),
                interval="1d",
                eager=True,
            ),
            "v": list(range(10)),
        })
        t.fit(X)
        assert t.observation_horizon == 0


@pytest.mark.integration
class TestWindowStatsTransformer:
    """Verify stateful non-invertible transformer extension."""

    def test_systematic_checks(self, numeric_series):
        X_train, X_test = numeric_series[:80], numeric_series[80:]
        transformer = _WindowStatsTransformer(window_size=5)
        transformer.fit(X_train)

        run_checks(
            transformer,
            _yield_yohou_transformer_checks(transformer, X_train, None, X_test),
        )

    def test_observation_horizon_matches_window(self):
        t = _WindowStatsTransformer(window_size=7)
        X = pl.DataFrame({
            "time": pl.datetime_range(
                pl.lit("2020-01-01").str.to_datetime(),
                pl.lit("2020-01-01").str.to_datetime() + pl.duration(days=49),
                interval="1d",
                eager=True,
            ),
            "v": list(range(50)),
        })
        t.fit(X)
        assert t.observation_horizon == 7

    def test_transform_drops_warmup_rows(self, numeric_series):
        t = _WindowStatsTransformer(window_size=5)
        t.fit(numeric_series)
        X_t = t.transform(numeric_series)
        assert len(X_t) == len(numeric_series) - 5  # observation_horizon rows dropped

    def test_get_feature_names_out(self, numeric_series):
        t = _WindowStatsTransformer(window_size=3)
        t.fit(numeric_series)
        names = t.get_feature_names_out()
        assert "value" in names
        assert "time" not in names


@pytest.mark.integration
class TestCustomTransformerInPipeline:
    """Verify custom transformer works inside FeaturePipeline."""

    def test_scale_in_pipeline(self, numeric_series):
        pipe = FeaturePipeline([
            ("scale", _ScaleTransformer(factor=3.0)),
        ])
        pipe.fit(numeric_series[:80])
        X_t = pipe.transform(numeric_series[:80])

        assert "time" in X_t.columns
        original = numeric_series[:80].select(cs.numeric() & ~cs.by_name("time")).to_numpy()
        transformed = X_t.select(cs.numeric() & ~cs.by_name("time")).to_numpy()
        np.testing.assert_allclose(transformed, original * 3.0)

    def test_stateful_in_pipeline(self, numeric_series):
        pipe = FeaturePipeline([
            ("stats", _WindowStatsTransformer(window_size=3)),
        ])
        pipe.fit(numeric_series[:80])
        assert pipe.observation_horizon == 3
