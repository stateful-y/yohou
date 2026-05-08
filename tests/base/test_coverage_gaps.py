"""Tests targeting specific uncovered branches in base/panel/forecaster.

Covers:
- panel.py line 330: global step columns in _pre_fit_panel
- panel.py lines 589, 594, 598: _observe_with_precomputed_steps_panel with transformers
- forecaster.py line 664: predict with step override on panel (else branch)
- local_panel_forecaster.py lines 769-770, 775, 780: on-the-fly schema derivation
"""

from datetime import datetime, timedelta

import polars as pl
from sklearn.linear_model import LinearRegression

from yohou.compose import LocalPanelForecaster
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer
from yohou.stationarity import LogTransformer


def _make_panel_data_with_global_X(length=100, forecasting_horizon=5):
    """Build panel y with global X_actual and global X_future.

    X_actual has both panel-prefixed and global (shared) columns.
    X_future has a global column (no __ prefix) to produce global step columns.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(hours=length - 1),
        interval="1h",
        eager=True,
    )
    y = pl.DataFrame({
        "time": time,
        "grp_a__value": rng.random(length) * 10 + 10,
        "grp_b__value": rng.random(length) * 10 + 20,
    })

    # X_actual with both panel and shared columns
    X_actual = pl.DataFrame({
        "time": time,
        "grp_a__feat": rng.random(length),
        "grp_b__feat": rng.random(length),
        "weather": rng.random(length),  # shared/global column
    })

    # X_future with a global column (no __) to produce global step columns
    time_ext = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(hours=length + forecasting_horizon - 1),
        interval="1h",
        eager=True,
    )
    X_future = pl.DataFrame({
        "time": time_ext,
        "temperature": rng.random(len(time_ext)),  # global step column
    })

    return y, X_actual, X_future


class TestPanelGlobalStepColumns:
    """Panel fit with global (non-prefixed) step columns."""

    def test_fit_with_global_step_columns(self):
        """Fit panel forecaster with X_future that has global columns (line 330)."""
        y, X_actual, X_future = _make_panel_data_with_global_X()
        f = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=[1, 2]),
        )
        f.fit(y[:80], X_actual[:80], forecasting_horizon=5, X_future=X_future)

        assert f._step_column_names_
        assert f._step_schema_per_group_ is not None
        # The global column "temperature" should appear in the step schema
        assert any("temperature" in col for col in f._step_schema_per_group_)

        y_pred = f.predict(forecasting_horizon=5)
        assert y_pred.shape[0] == 5


class TestPanelObservePredictWithTransformers:
    """observe_predict on panel forecaster with transformers and step columns.

    Exercises _observe_with_precomputed_steps_panel (lines 589, 594, 598).
    """

    def test_observe_predict_panel_with_transformers_and_shared_X(self):
        """observe_predict with target_transformer, feature_transformer, and shared X."""
        y, X_actual, X_future = _make_panel_data_with_global_X()
        f = PointReductionForecaster(
            estimator=LinearRegression(),
            target_transformer=LogTransformer(offset=1.0),
            feature_transformer=LagTransformer(lag=[1, 2]),
        )
        f.fit(y[:80], X_actual[:80], forecasting_horizon=5, X_future=X_future)

        # Verify shared schema exists
        assert f.shared_X_actual_schema_ is not None
        assert "weather" in f.shared_X_actual_schema_
        assert isinstance(f.target_transformer_, dict)
        assert isinstance(f.feature_transformer_, dict)

        # observe_predict triggers _observe_with_precomputed_steps_panel
        y_pred = f.observe_predict(
            y[80:90],
            X_actual=X_actual[80:90],
            stride=5,
            X_future=X_future,
        )
        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) >= 5

    def test_observe_predict_panel_with_transformers_no_shared_X(self):
        """observe_predict with transformers but no shared columns."""
        import numpy as np

        rng = np.random.default_rng(7)
        length = 100
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1) + timedelta(hours=length - 1),
            interval="1h",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "grp_a__value": rng.random(length) * 10 + 10,
            "grp_b__value": rng.random(length) * 10 + 20,
        })
        X_actual = pl.DataFrame({
            "time": time,
            "grp_a__feat": rng.random(length),
            "grp_b__feat": rng.random(length),
        })
        time_ext = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1) + timedelta(hours=length + 4),
            interval="1h",
            eager=True,
        )
        X_future = pl.DataFrame({
            "time": time_ext,
            "grp_a__temp": rng.random(len(time_ext)),
            "grp_b__temp": rng.random(len(time_ext)),
        })

        f = PointReductionForecaster(
            estimator=LinearRegression(),
            target_transformer=LogTransformer(offset=1.0),
            feature_transformer=LagTransformer(lag=[1, 2]),
        )
        f.fit(y[:80], X_actual[:80], forecasting_horizon=5, X_future=X_future)
        assert isinstance(f.target_transformer_, dict)

        y_pred = f.observe_predict(
            y[80:90],
            X_actual=X_actual[80:90],
            stride=5,
            X_future=X_future,
        )
        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) >= 5


class TestPredictWithStepOverridePanel:
    """Test predict with step column override on panel forecaster (line 664)."""

    def test_predict_panel_step_override_first_call(self):
        """predict(X_future=...) on panel forecaster exercises else branch for step swap.

        After removing step columns from _X_t_observed manually, predict
        with override should use the else branch (no cols_to_drop).
        """
        y, X_actual, X_future = _make_panel_data_with_global_X()
        f = PointReductionForecaster(
            estimator=LinearRegression(),
            feature_transformer=LagTransformer(lag=[1, 2]),
        )
        f.fit(y[:80], X_actual[:80], forecasting_horizon=5, X_future=X_future)

        # Remove step columns from _X_t_observed to simulate the else branch
        step_cols = sorted(f._step_schema_per_group_) if f._step_schema_per_group_ else []
        if step_cols and isinstance(f._X_t_observed, dict):
            for group_name in f._X_t_observed:
                group_df = f._X_t_observed[group_name]
                cols_present = [c for c in step_cols if c in group_df.columns]
                if cols_present:
                    f._X_t_observed[group_name] = group_df.drop(cols_present)

        # Now predict with X_future override: triggers else branch (line 664)
        y_pred = f.predict(forecasting_horizon=5, X_future=X_future)
        assert isinstance(y_pred, pl.DataFrame)
        assert y_pred.shape[0] == 5


class TestLocalPanelForecasterOnTheFlySchema:
    """LocalPanelForecaster _split_exogenous_for_group on-the-fly schema derivation.

    Covers lines 769-770, 775, 780 in local_panel_forecaster.py.
    """

    def test_predict_with_x_future_when_fit_without(self):
        """predict(X_future=..., X_forecast=...) when fit had neither triggers schema derivation."""
        import numpy as np

        rng = np.random.default_rng(11)
        length = 100
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1) + timedelta(hours=length - 1),
            interval="1h",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "grp_a__value": rng.random(length) * 10 + 10,
            "grp_b__value": rng.random(length) * 10 + 20,
        })
        X_actual = pl.DataFrame({
            "time": time,
            "grp_a__feat": rng.random(length),
            "grp_b__feat": rng.random(length),
        })

        # Fit WITHOUT X_future or X_forecast
        f = LocalPanelForecaster(
            forecaster=PointReductionForecaster(
                estimator=LinearRegression(),
                feature_transformer=LagTransformer(lag=[1, 2]),
            ),
        )
        f.fit(y[:80], X_actual[:80], forecasting_horizon=5)
        assert f._local_X_future_schema_ is None
        assert f._local_X_forecast_schema_ is None

        # Predict WITH both X_future and X_forecast
        time_ext = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1) + timedelta(hours=length + 4),
            interval="1h",
            eager=True,
        )
        X_future = pl.DataFrame({
            "time": time_ext,
            "grp_a__temp": rng.random(len(time_ext)),
            "grp_b__temp": rng.random(len(time_ext)),
        })

        # Build X_forecast with vintage_time and time
        obs_time = y[:80]["time"][-1]
        forecast_times = pl.datetime_range(
            start=obs_time + timedelta(hours=1),
            end=obs_time + timedelta(hours=5),
            interval="1h",
            eager=True,
        )
        X_forecast = pl.DataFrame({
            "vintage_time": [obs_time] * 5,
            "time": forecast_times,
            "grp_a__rain": rng.random(5),
            "grp_b__rain": rng.random(5),
        })

        y_pred = f.predict(forecasting_horizon=5, X_future=X_future, X_forecast=X_forecast)
        assert isinstance(y_pred, pl.DataFrame)
        assert y_pred.shape[0] == 5

    def test_predict_with_x_future_global_columns(self):
        """predict(X_future=...) with global columns triggers on-the-fly schema."""
        import numpy as np

        rng = np.random.default_rng(12)
        length = 100
        time = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1) + timedelta(hours=length - 1),
            interval="1h",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "grp_a__value": rng.random(length) * 10 + 10,
            "grp_b__value": rng.random(length) * 10 + 20,
        })
        X_actual = pl.DataFrame({
            "time": time,
            "grp_a__feat": rng.random(length),
            "grp_b__feat": rng.random(length),
        })

        # Fit WITHOUT X_future
        f = LocalPanelForecaster(
            forecaster=PointReductionForecaster(
                estimator=LinearRegression(),
                feature_transformer=LagTransformer(lag=[1, 2]),
            ),
        )
        f.fit(y[:80], X_actual[:80], forecasting_horizon=5)
        assert f._local_X_future_schema_ is None

        # X_future with a global column (no __ prefix)
        time_ext = pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1) + timedelta(hours=length + 4),
            interval="1h",
            eager=True,
        )
        X_future = pl.DataFrame({
            "time": time_ext,
            "temperature": rng.random(len(time_ext)),  # global column
        })
        y_pred = f.predict(forecasting_horizon=5, X_future=X_future)
        assert isinstance(y_pred, pl.DataFrame)
        assert y_pred.shape[0] == 5
