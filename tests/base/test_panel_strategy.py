"""Tests for panel_strategy dispatch on BaseForecaster.

Validates that panel_strategy="global" (default) detects panel groups and
pools data, while panel_strategy="multivariate" bypasses panel detection
and treats __-prefixed columns as regular multivariate features.
"""

import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer


@pytest.fixture
def panel_y_X(y_X_factory):
    """Panel data with 2 groups, 2 targets, 2 features."""
    return y_X_factory(length=100, n_targets=2, n_features=2, panel=True, n_groups=2)


@pytest.fixture
def standard_y_X(y_X_factory):
    """Standard (non-panel) data with 2 targets, 2 features."""
    return y_X_factory(length=100, n_targets=2, n_features=2, panel=False)


def _make_forecaster(panel_strategy):
    """Create a concrete PointReductionForecaster with given strategy."""
    return PointReductionForecaster(
        estimator=LinearRegression(),
        feature_transformer=LagTransformer(lag=[1, 2]),
        panel_strategy=panel_strategy,
    )


class TestDispatchAttributes:
    """Test that panel_strategy controls detection and attribute setting."""

    def test_global_detects_panel_groups(self, panel_y_X):
        """Global strategy detects panel groups and sets panel_group_names_."""
        y, X = panel_y_X
        forecaster = _make_forecaster("global")
        forecaster.fit(y[:80], X[:80], forecasting_horizon=5)

        assert forecaster.panel_group_names_ is not None
        assert set(forecaster.panel_group_names_) == {"group_0", "group_1"}
        # local_y_schema_ uses unprefixed names
        assert "y_0" in forecaster.local_y_schema_
        assert "y_1" in forecaster.local_y_schema_
        # target_transformer_ is a dict
        if forecaster.target_transformer_ is not None:
            assert isinstance(forecaster.target_transformer_, dict)
        # feature_transformer_ is a dict
        assert isinstance(forecaster.feature_transformer_, dict)
        assert set(forecaster.feature_transformer_.keys()) == {"group_0", "group_1"}

    def test_multivariate_skips_panel_detection(self, panel_y_X):
        """Multivariate strategy skips panel detection on __-prefixed data."""
        y, X = panel_y_X
        forecaster = _make_forecaster("multivariate")
        forecaster.fit(y[:80], X[:80], forecasting_horizon=5)

        assert forecaster.panel_group_names_ is None
        # local_y_schema_ keeps full prefixed column names
        assert "group_0__y_0" in forecaster.local_y_schema_
        assert "group_1__y_1" in forecaster.local_y_schema_
        # Transformer is a single instance, not a dict
        assert not isinstance(forecaster.feature_transformer_, dict)

    def test_global_on_standard_data_falls_through(self, standard_y_X):
        """Global strategy on non-panel data takes the standard path."""
        y, X = standard_y_X
        forecaster = _make_forecaster("global")
        forecaster.fit(y[:80], X[:80], forecasting_horizon=5)

        assert forecaster.panel_group_names_ is None
        assert "y_0" in forecaster.local_y_schema_
        assert not isinstance(forecaster.feature_transformer_, dict)

    def test_multivariate_on_standard_data(self, standard_y_X):
        """Multivariate strategy on standard data also takes the standard path."""
        y, X = standard_y_X
        forecaster = _make_forecaster("multivariate")
        forecaster.fit(y[:80], X[:80], forecasting_horizon=5)

        assert forecaster.panel_group_names_ is None
        assert "y_0" in forecaster.local_y_schema_


class TestValidation:
    """Test parameter constraint validation."""

    def test_invalid_panel_strategy_raises(self):
        """Non-allowed panel_strategy values raise ValueError."""
        forecaster = _make_forecaster("global")
        forecaster.panel_strategy = "local"  # invalid
        y = pl.DataFrame({
            "time": pl.datetime_range(
                start=pl.datetime(2020, 1, 1),
                end=pl.datetime(2020, 4, 9),
                interval="1d",
                eager=True,
            ),
        }).with_columns(pl.lit(1.0).alias("value"))
        with pytest.raises(ValueError, match="panel_strategy"):
            forecaster.fit(y, forecasting_horizon=1)


class TestPredictLifecycle:
    """Test that predict/observe/rewind work under both strategies."""

    def test_multivariate_predict_returns_prefixed_columns(self, panel_y_X):
        """Multivariate predict output keeps __-prefixed column names."""
        y, X = panel_y_X
        forecaster = _make_forecaster("multivariate")
        forecaster.fit(y[:80], X[:80], forecasting_horizon=5)
        y_pred = forecaster.predict(X[:80], forecasting_horizon=5)

        # Columns should include the prefixed names
        pred_cols = set(y_pred.columns) - {"time", "observed_time"}
        y_cols = set(y.columns) - {"time"}
        assert pred_cols == y_cols

    def test_global_predict_returns_prefixed_columns(self, panel_y_X):
        """Global predict output has same panel structure as input."""
        y, X = panel_y_X
        forecaster = _make_forecaster("global")
        forecaster.fit(y[:80], X[:80], forecasting_horizon=5)
        y_pred = forecaster.predict(X[:80], forecasting_horizon=5)

        pred_cols = set(y_pred.columns) - {"time", "observed_time"}
        y_cols = set(y.columns) - {"time"}
        assert pred_cols == y_cols

    def test_multivariate_observe_predict_cycle(self, panel_y_X):
        """Observe-predict works with multivariate strategy."""
        y, X = panel_y_X
        forecaster = _make_forecaster("multivariate")
        forecaster.fit(y[:80], X[:80], forecasting_horizon=5)

        # Observe new data
        forecaster.observe(y[80:90], X[80:90])
        y_pred = forecaster.predict(X[80:90], forecasting_horizon=5)
        assert len(y_pred) == 5

    def test_multivariate_rewind(self, panel_y_X):
        """Rewind works with multivariate strategy."""
        y, X = panel_y_X
        forecaster = _make_forecaster("multivariate")
        forecaster.fit(y[:80], X[:80], forecasting_horizon=5)
        forecaster.observe(y[80:90], X[80:90])
        forecaster.rewind(y[80:90], X[80:90])
        y_pred = forecaster.predict(X[80:90], forecasting_horizon=5)
        assert len(y_pred) == 5


class TestDifferentBehavior:
    """Test that the two strategies produce genuinely different results."""

    def test_multivariate_vs_global_different_predictions(self, panel_y_X):
        """Global and multivariate strategies yield different predictions on panel data."""
        y, X = panel_y_X
        y_train, X_train = y[:80], X[:80]
        fh = 5

        f_global = _make_forecaster("global")
        f_global.fit(y_train, X_train, forecasting_horizon=fh)
        p_global = f_global.predict(X_train, forecasting_horizon=fh)

        f_multi = _make_forecaster("multivariate")
        f_multi.fit(y_train, X_train, forecasting_horizon=fh)
        p_multi = f_multi.predict(X_train, forecasting_horizon=fh)

        # Both should have same columns, but different values
        assert set(p_global.columns) == set(p_multi.columns)
        # Values differ because global pools per-group, multivariate sees cross-group features
        value_cols = [c for c in p_global.columns if c not in {"time", "observed_time"}]
        any_different = False
        for col in value_cols:
            if p_global[col].to_list() != p_multi[col].to_list():
                any_different = True
                break
        assert any_different, "Global and multivariate should produce different predictions"
