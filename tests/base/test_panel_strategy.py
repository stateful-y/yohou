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
        """Global strategy detects panel groups and sets groups_."""
        y, X = panel_y_X
        forecaster = _make_forecaster("global")
        forecaster.fit(y[:80], X[:80], forecasting_horizon=5)

        assert forecaster.groups_ is not None
        assert set(forecaster.groups_) == {"group_0", "group_1"}
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

        assert forecaster.groups_ is None
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

        assert forecaster.groups_ is None
        assert "y_0" in forecaster.local_y_schema_
        assert not isinstance(forecaster.feature_transformer_, dict)

    def test_multivariate_on_standard_data(self, standard_y_X):
        """Multivariate strategy on standard data also takes the standard path."""
        y, X = standard_y_X
        forecaster = _make_forecaster("multivariate")
        forecaster.fit(y[:80], X[:80], forecasting_horizon=5)

        assert forecaster.groups_ is None
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
        pred_cols = set(y_pred.columns) - {"time", "vintage_time"}
        y_cols = set(y.columns) - {"time"}
        assert pred_cols == y_cols

    def test_global_predict_returns_prefixed_columns(self, panel_y_X):
        """Global predict output has same panel structure as input."""
        y, X = panel_y_X
        forecaster = _make_forecaster("global")
        forecaster.fit(y[:80], X[:80], forecasting_horizon=5)
        y_pred = forecaster.predict(X[:80], forecasting_horizon=5)

        pred_cols = set(y_pred.columns) - {"time", "vintage_time"}
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
        value_cols = [c for c in p_global.columns if c not in {"time", "vintage_time"}]
        any_different = False
        for col in value_cols:
            if p_global[col].to_list() != p_multi[col].to_list():
                any_different = True
                break
        assert any_different, "Global and multivariate should produce different predictions"


class TestPanelInverseTransform:
    """Tests for _inverse_transform_target with panel data and target_transformer."""

    def test_predict_with_target_transformer_on_panel_data(self, y_X_factory):
        """Panel prediction with target_transformer exercises panel inverse path."""
        from yohou.stationarity import LogTransformer

        y, X = y_X_factory(length=80, n_targets=1, n_features=2, panel=True, n_groups=2)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            target_transformer=LogTransformer(offset=1.0),
            feature_transformer=LagTransformer(lag=[1, 2]),
            panel_strategy="global",
        )
        forecaster.fit(y[:60], X[:60], forecasting_horizon=3)

        assert isinstance(forecaster.target_transformer_, dict)

        y_pred = forecaster.predict(X[:60], forecasting_horizon=3)

        assert len(y_pred) == 3
        assert "vintage_time" in y_pred.columns
        target_cols = [c for c in y_pred.columns if c not in {"time", "vintage_time"}]
        assert len(target_cols) == 2
        assert all("__" in c for c in target_cols)

    def test_observation_horizon_with_panel_dict_transformer(self, y_X_factory):
        """observation_horizon reads from dict of transformers for panel data."""
        from yohou.stationarity import SeasonalDifferencing

        y, X = y_X_factory(length=80, n_targets=1, n_features=2, panel=True, n_groups=2)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            target_transformer=SeasonalDifferencing(seasonality=7),
            feature_transformer=LagTransformer(lag=[1, 2]),
            panel_strategy="global",
        )
        forecaster.fit(y[:60], X[:60], forecasting_horizon=3)

        assert isinstance(forecaster.target_transformer_, dict)
        assert forecaster.observation_horizon >= 7

    def test_panel_inverse_transform_preserves_dtype(self, y_X_factory):
        """Panel inverse transform casts predictions back to original dtypes."""
        from yohou.stationarity import LogTransformer

        y, X = y_X_factory(length=80, n_targets=1, n_features=2, panel=True, n_groups=2)

        forecaster = PointReductionForecaster(
            estimator=LinearRegression(),
            target_transformer=LogTransformer(offset=1.0),
            feature_transformer=LagTransformer(lag=[1, 2]),
            panel_strategy="global",
        )
        forecaster.fit(y[:60], X[:60], forecasting_horizon=3)
        y_pred = forecaster.predict(X[:60], forecasting_horizon=3)

        target_cols = [c for c in y_pred.columns if c not in {"time", "vintage_time"}]
        for col in target_cols:
            assert y_pred[col].dtype.is_float()


class TestGlobalOnlyExogenous:
    """Test global-only X (no panel prefixes) with panel y."""

    @pytest.fixture
    def panel_y_global_X(self, y_X_factory):
        """Panel y with global-only X (no __ prefixes)."""
        import numpy as np

        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, panel=True, n_groups=2)
        rng = np.random.default_rng(42)
        time = y["time"]
        X = pl.DataFrame({
            "time": time,
            "weather": rng.random(len(time)),
            "holiday": rng.choice([0.0, 1.0], size=len(time)),
        })
        return y, X

    @pytest.fixture
    def panel_y_mixed_X(self, y_X_factory):
        """Panel y with mixed X (panel + global columns)."""
        import numpy as np

        y, X_panel = y_X_factory(length=100, n_targets=1, n_features=1, panel=True, n_groups=2)
        rng = np.random.default_rng(99)
        X = X_panel.with_columns(
            pl.Series("weather", rng.random(len(y))),
        )
        return y, X

    def test_point_forecaster_fit_predict(self, panel_y_global_X):
        """PointReductionForecaster fit and predict with global-only X."""
        y, X = panel_y_global_X
        f = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=[1, 2]),
        )
        f.fit(y[:80], X[:80], forecasting_horizon=5)

        assert f.groups_ is not None
        assert f.local_X_actual_schema_ == {}
        assert f.shared_X_actual_schema_ is not None
        assert "weather" in f.shared_X_actual_schema_
        assert "holiday" in f.shared_X_actual_schema_

        y_pred = f.predict(X=X[80:85], forecasting_horizon=5)
        assert y_pred.shape[0] == 5
        pred_cols = set(y_pred.columns) - {"time", "vintage_time"}
        y_cols = set(y.columns) - {"time"}
        assert pred_cols == y_cols

    def test_point_forecaster_full_lifecycle(self, panel_y_global_X):
        """Full lifecycle: fit, predict, observe, rewind, predict."""
        y, X = panel_y_global_X
        f = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=[1, 2]),
        )
        f.fit(y[:80], X[:80], forecasting_horizon=5)

        y_pred_1 = f.predict(X=X[80:85], forecasting_horizon=5)
        assert y_pred_1.shape[0] == 5

        f.observe(y[80:85], X_actual=X[80:85])
        y_pred_2 = f.predict(X=X[85:90], forecasting_horizon=5)
        assert y_pred_2.shape[0] == 5

        f.rewind(y[:80], X_actual=X[:80])
        y_pred_3 = f.predict(X=X[80:85], forecasting_horizon=5)
        assert y_pred_3.shape[0] == 5

        # Predictions after rewind should match the first prediction
        for col in [c for c in y_pred_1.columns if c not in {"time", "vintage_time"}]:
            assert y_pred_1[col].to_list() == y_pred_3[col].to_list()

    def test_interval_forecaster_fit_predict(self, panel_y_global_X):
        """IntervalReductionForecaster fit and predict with global-only X."""
        from yohou.interval import SplitConformalForecaster

        y, X = panel_y_global_X
        f = SplitConformalForecaster(
            point_forecaster=PointReductionForecaster(
                estimator=LinearRegression(),
                feature_transformer=LagTransformer(lag=[1, 2]),
            ),
            calibration_size=20,
        )
        f.fit(y[:80], X[:80], forecasting_horizon=5)
        y_pred = f.predict(X=X[80:85], forecasting_horizon=5, coverage_rates=[0.9])
        assert y_pred.shape[0] == 5

    def test_grid_search_cv(self, panel_y_global_X):
        """GridSearchCV with global-only X and panel y."""
        from yohou.metrics import MeanAbsoluteError
        from yohou.model_selection import ExpandingWindowSplitter, GridSearchCV

        y, X = panel_y_global_X
        f = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=[1, 2]),
        )
        cv = ExpandingWindowSplitter(n_splits=2, test_size=5)
        search = GridSearchCV(
            forecaster=f,
            param_grid={"feature_transformer__lag": [[1, 2], [1, 2, 3]]},
            scoring=MeanAbsoluteError(),
            cv=cv,
        )
        search.fit(y[:80], X[:80])
        assert hasattr(search, "best_params_")

    def test_local_panel_forecaster(self, panel_y_global_X):
        """LocalPanelForecaster with global-only X and panel y."""
        from yohou.compose import LocalPanelForecaster

        y, X = panel_y_global_X
        f = LocalPanelForecaster(
            forecaster=PointReductionForecaster(
                estimator=LinearRegression(),
                feature_transformer=LagTransformer(lag=[1, 2]),
            ),
        )
        f.fit(y[:80], X[:80], forecasting_horizon=5)
        y_pred = f.predict(forecasting_horizon=5)
        assert y_pred.shape[0] == 5
        pred_cols = set(y_pred.columns) - {"time", "vintage_time"}
        y_cols = set(y.columns) - {"time"}
        assert pred_cols == y_cols

    def test_splitter_with_global_X(self, panel_y_global_X):
        """ExpandingWindowSplitter split with global-only X and panel y."""
        from yohou.model_selection import ExpandingWindowSplitter

        y, X = panel_y_global_X
        splitter = ExpandingWindowSplitter(n_splits=3, test_size=5)
        splits = list(splitter.split(y[:80], X[:80]))
        assert len(splits) == 3

    def test_mixed_X_regression(self, panel_y_mixed_X):
        """Mixed X (panel + global columns) still works (regression test)."""
        y, X = panel_y_mixed_X
        f = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=[1, 2]),
        )
        f.fit(y[:80], X[:80], forecasting_horizon=5)

        assert f.local_X_actual_schema_ is not None
        assert len(f.local_X_actual_schema_) > 0
        assert f.shared_X_actual_schema_ is not None
        assert "weather" in f.shared_X_actual_schema_

        y_pred = f.predict(X=X[80:85], forecasting_horizon=5)
        assert y_pred.shape[0] == 5

    def test_lifecycle_with_target_transformer(self, panel_y_global_X):
        """Observe and rewind with target_transformer exercises dict lookup paths."""
        from yohou.stationarity import LogTransformer

        y, X = panel_y_global_X
        f = PointReductionForecaster(
            estimator=LinearRegression(),
            target_transformer=LogTransformer(offset=1.0),
            feature_transformer=LagTransformer(lag=[1, 2]),
        )
        f.fit(y[:80], X[:80], forecasting_horizon=5)
        assert isinstance(f.target_transformer_, dict)

        y_pred_1 = f.predict(X=X[80:85], forecasting_horizon=5)
        assert y_pred_1.shape[0] == 5

        f.observe(y[80:85], X_actual=X[80:85])
        y_pred_2 = f.predict(X=X[85:90], forecasting_horizon=5)
        assert y_pred_2.shape[0] == 5

        f.rewind(y[:80], X_actual=X[:80])
        y_pred_3 = f.predict(X=X[80:85], forecasting_horizon=5)
        assert y_pred_3.shape[0] == 5

        for col in [c for c in y_pred_1.columns if c not in {"time", "vintage_time"}]:
            assert y_pred_1[col].to_list() == y_pred_3[col].to_list()

    def test_mismatched_X_panel_suffixes_raises(self, y_X_factory):
        """X with panel columns whose suffixes differ across groups raises."""
        import numpy as np

        y, _ = y_X_factory(length=100, n_targets=1, n_features=0, panel=True, n_groups=2)
        groups = sorted({col.split("__")[0] for col in y.columns if "__" in col})
        rng = np.random.default_rng(0)
        time = y["time"]
        X = pl.DataFrame({
            "time": time,
            f"{groups[0]}__temp": rng.random(len(time)),
            f"{groups[1]}__humidity": rng.random(len(time)),
        })
        f = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=[1, 2]),
        )
        with pytest.raises(ValueError, match="do not have the same column suffixes"):
            f.fit(y[:80], X[:80], forecasting_horizon=5)
