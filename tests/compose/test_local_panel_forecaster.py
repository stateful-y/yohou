"""Tests for LocalPanelForecaster meta-forecaster.

Tests per-group independent forecasting via LocalPanelForecaster.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge

from yohou.compose import LocalPanelForecaster
from yohou.interval import SplitConformalForecaster
from yohou.point import PointReductionForecaster, SeasonalNaive
from yohou.stationarity import PolynomialTrendForecaster
from yohou.utils.panel import inspect_panel


@pytest.fixture
def panel_y():
    """Simple panel target with two groups."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 4, 9),
        interval="1d",
        eager=True,
    )
    return pl.DataFrame({
        "time": time,
        "store_a__sales": [float(i) for i in range(100)],
        "store_b__sales": [float(i + 100) for i in range(100)],
    })


@pytest.fixture
def panel_y_X():
    """Panel target and exogenous with two groups."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 4, 9),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({
        "time": time,
        "store_a__sales": [float(i) for i in range(100)],
        "store_b__sales": [float(i + 100) for i in range(100)],
    })
    X_actual = pl.DataFrame({
        "time": time,
        "store_a__promo": [float(i % 7) for i in range(100)],
        "store_b__promo": [float(i % 5) for i in range(100)],
    })
    return y, X_actual


class TestBasicFitPredict:
    """Tests for basic fit/predict workflow."""

    def test_fit_predict_point(self, panel_y):
        """Fit and predict with SeasonalNaive on panel data."""
        y = panel_y
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        f.fit(y[:80], forecasting_horizon=5)

        y_pred = f.predict(forecasting_horizon=5)

        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) == 5
        assert "time" in y_pred.columns
        assert "store_a__sales" in y_pred.columns
        assert "store_b__sales" in y_pred.columns

    def test_fit_predict_with_exogenous(self, panel_y_X):
        """Fit and predict with exogenous features."""
        y, X_actual = panel_y_X
        f = LocalPanelForecaster(
            forecaster=PointReductionForecaster(estimator=Ridge()),
        )
        f.fit(y[:80], X_actual[:80], forecasting_horizon=5)
        y_pred = f.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        assert "store_a__sales" in y_pred.columns
        assert "store_b__sales" in y_pred.columns

    def test_per_group_clones_are_independent(self, panel_y):
        """Each group should have its own independent forecaster clone."""
        y = panel_y
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        f.fit(y[:80], forecasting_horizon=5)

        assert len(f.forecasters_) == 2
        assert "store_a" in f.forecasters_
        assert "store_b" in f.forecasters_
        assert f.forecasters_["store_a"] is not f.forecasters_["store_b"]

    def test_rejects_non_panel_data(self):
        """Should raise ValueError on global (non-panel) data."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 4, 9),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "sales": range(100)})

        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        with pytest.raises(ValueError, match="requires panel data"):
            f.fit(y, forecasting_horizon=5)

    def test_not_fitted_error(self, panel_y):
        """Should raise NotFittedError before fit."""
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        with pytest.raises(NotFittedError, match="is not fitted"):
            f.predict(forecasting_horizon=5)


class TestGroupSubsetting:
    """Tests for groups subsetting."""

    def test_predict_subset_groups(self, panel_y):
        """predict(groups=...) should return only requested groups."""
        y = panel_y
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        f.fit(y[:80], forecasting_horizon=5)

        y_pred = f.predict(forecasting_horizon=5, groups=["store_a"])

        non_time = [c for c in y_pred.columns if c not in ("time", "vintage_time")]
        assert all(c.startswith("store_a__") for c in non_time)

    def test_observe_subset_groups(self, panel_y):
        """observe(groups=...) should update only requested groups."""
        y = panel_y
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        f.fit(y[:80], forecasting_horizon=5)

        # Observe only store_a
        f.observe(y=y[80:85], groups=["store_a"])

        # store_a should have been updated, store_b should still be at fit state
        # We verify by checking prediction differs for store_a
        y_pred_a = f.predict(forecasting_horizon=3, groups=["store_a"])
        assert len(y_pred_a) == 3


class TestObserveRewind:
    """Tests for observe and rewind methods."""

    def test_observe_predict_cycle(self, panel_y):
        """Observe then predict should use new observations."""
        y = panel_y
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        f.fit(y[:80], forecasting_horizon=5)

        y_pred_before = f.predict(forecasting_horizon=5)
        f.observe(y=y[80:85])
        y_pred_after = f.predict(forecasting_horizon=5)

        # Predictions should differ after observing new data
        assert y_pred_before["store_a__sales"].to_list() != y_pred_after["store_a__sales"].to_list()

    def test_observe_predict_combined(self, panel_y):
        """observe_predict delegates to each clone's rolling loop."""
        y = panel_y
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        f.fit(y[:80], forecasting_horizon=5)

        y_pred = f.observe_predict(y=y[80:85])
        assert isinstance(y_pred, pl.DataFrame)
        # Rolling loop: initial predict (5) + 1 stride window (5) = 10
        assert len(y_pred) == 10

    def test_rewind(self, panel_y):
        """Rewind should reset observation state per group."""
        y = panel_y
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        f.fit(y[:80], forecasting_horizon=5)
        f.observe(y=y[80:85])

        # Rewind to a subset of the data
        f.rewind(y=y[70:80])

        y_pred = f.predict(forecasting_horizon=5)
        assert len(y_pred) == 5

    def test_observe_rewind_reject_unexpected_params(self, panel_y):
        """observe/rewind reject stray kwargs at the LocalPanelForecaster boundary.

        Regression for the 2026-06-15 QA finding: observe/rewind accepted and
        forwarded ``**params`` raw to child forecasters, so a stray keyword was
        passed through and surfaced as a confusing error deep in the base class.
        Like the base ``observe``/``rewind``, these methods take no routable
        metadata, so an unexpected keyword is rejected up front.
        """
        y = panel_y
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        f.fit(y[:80], forecasting_horizon=5)

        with pytest.raises(TypeError, match="observe"):
            f.observe(y=y[80:85], sample_weight=[1.0] * 5)
        with pytest.raises(TypeError, match="rewind"):
            f.rewind(y=y[70:80], sample_weight=[1.0] * 10)


class TestLocalVsGlobal:
    """Verify that LocalPanelForecaster handles heterogeneous groups better than pooled."""

    def test_captures_per_group_trends(self):
        """Groups with different slopes: local should capture each individually."""
        n = 100
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=n - 1),
            interval="1d",
            eager=True,
        )

        # Group A: slope=1, Group B: slope=10
        y = pl.DataFrame({
            "time": time,
            "gA__value": [1.0 * i for i in range(n)],
            "gB__value": [10.0 * i for i in range(n)],
        })

        f = LocalPanelForecaster(
            forecaster=PolynomialTrendForecaster(degree=1),
        )
        f.fit(y[:80], forecasting_horizon=5)
        y_pred = f.predict(forecasting_horizon=5)

        # Group A predictions should reflect slope=1
        gA_pred = y_pred["gA__value"].to_numpy()
        gA_expected = np.array([1.0 * i for i in range(80, 85)])
        np.testing.assert_allclose(gA_pred, gA_expected, rtol=0.1)

        # Group B predictions should reflect slope=10
        gB_pred = y_pred["gB__value"].to_numpy()
        gB_expected = np.array([10.0 * i for i in range(80, 85)])
        np.testing.assert_allclose(gB_pred, gB_expected, rtol=0.1)


class TestTags:
    """Tests for sklearn tag propagation."""

    def test_inherits_forecaster_type_point(self):
        """Should inherit forecaster_type from wrapped point forecaster."""
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        tags = f.__sklearn_tags__()
        assert tags.forecaster_tags.forecaster_type == frozenset({"point"})

    def test_inherits_forecaster_type_interval(self):
        """Should inherit forecaster_type from wrapped interval forecaster."""
        f = LocalPanelForecaster(
            forecaster=SplitConformalForecaster(
                point_forecaster=SeasonalNaive(seasonality=7),
            ),
        )
        tags = f.__sklearn_tags__()
        assert "interval" in tags.forecaster_tags.forecaster_type

    def test_supports_panel_data(self):
        """Should declare panel data support."""
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        tags = f.__sklearn_tags__()
        assert tags.forecaster_tags.supports_panel_data is True

    def test_tracks_observations_false(self):
        """Should delegate observation tracking to children."""
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        tags = f.__sklearn_tags__()
        assert tags.forecaster_tags.tracks_observations is False


class TestCloneParams:
    """Tests for sklearn clone and parameter management."""

    def test_clone(self):
        """Clone should produce equivalent unfitted forecaster."""
        f = LocalPanelForecaster(
            forecaster=SeasonalNaive(seasonality=7),
            n_jobs=2,
        )
        f_cloned = clone(f)

        assert f_cloned.n_jobs == 2
        assert isinstance(f_cloned.forecaster, SeasonalNaive)
        assert f_cloned.forecaster is not f.forecaster

    def test_get_params(self):
        """get_params should include nested forecaster parameters."""
        f = LocalPanelForecaster(
            forecaster=SeasonalNaive(seasonality=7),
            n_jobs=2,
        )
        params = f.get_params(deep=True)

        assert params["n_jobs"] == 2
        assert params["forecaster__seasonality"] == 7

    def test_set_params(self):
        """set_params should update nested parameters."""
        f = LocalPanelForecaster(
            forecaster=SeasonalNaive(seasonality=7),
        )
        f.set_params(forecaster__seasonality=14)

        assert f.forecaster.seasonality == 14


class TestPredictInterval:
    """Tests for predict_interval delegation."""

    def test_predict_interval_available(self, panel_y):
        """predict_interval should be available when wrapped forecaster supports it."""
        y = panel_y
        f = LocalPanelForecaster(
            forecaster=SplitConformalForecaster(
                point_forecaster=SeasonalNaive(seasonality=7),
                calibration_size=20,
            ),
        )
        f.fit(y[:80], forecasting_horizon=5)
        y_pred = f.predict_interval(forecasting_horizon=5, coverage_rates=[0.9])

        assert isinstance(y_pred, pl.DataFrame)
        assert len(y_pred) == 5

    def test_predict_interval_not_available(self):
        """predict_interval should not be available for point-only forecasters."""
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        assert not hasattr(f, "predict_interval") or not callable(getattr(f, "predict_interval", None))


class TestMultipleGroups:
    """Tests with more than two panel groups."""

    def test_five_groups(self, panel_time_series_factory):
        """Should handle 5+ panel groups."""
        y = panel_time_series_factory(length=100, n_series=2, n_groups=5)
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        f.fit(y[:80], forecasting_horizon=5)

        assert len(f.forecasters_) == 5
        assert len(f.groups_) == 5

        y_pred = f.predict(forecasting_horizon=5)
        _, panel_groups = inspect_panel(y_pred)
        assert len(panel_groups) == 5


class TestLocalPanelForecasterExogenous:
    """Tests for LocalPanelForecaster with X_future and X_forecast."""

    def test_fit_predict_with_X_future(self, panel_y):
        """LocalPanelForecaster passes X_future to per-group clones."""
        y = panel_y

        # X_future with panel prefix covering beyond y range
        time_ext = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 4, 19),
            interval="1d",
            eager=True,
        )
        X_future = pl.DataFrame({
            "time": time_ext,
            "store_a__holiday": [1.0 if t.weekday() == 6 else 0.0 for t in time_ext],
            "store_b__holiday": [1.0 if t.weekday() == 6 else 0.0 for t in time_ext],
        })

        f = LocalPanelForecaster(forecaster=PointReductionForecaster(estimator=Ridge()))
        f.fit(y[:80], forecasting_horizon=5, X_future=X_future)

        y_pred = f.predict(forecasting_horizon=5)
        assert len(y_pred) == 5
        assert "store_a__sales" in y_pred.columns
        assert "store_b__sales" in y_pred.columns

    def test_predict_X_forecast_override(self, panel_y):
        """LocalPanelForecaster passes X_forecast override to per-group predict."""
        y = panel_y
        rng = np.random.default_rng(42)
        time = y["time"]
        n_obs = len(time)
        fh = 5

        # Build X_forecast with panel prefix
        rows = []
        for v_idx in range(n_obs):
            for step in range(1, fh + 1):
                target_idx = v_idx + step
                if target_idx < n_obs + fh + 5:
                    tt = time[target_idx] if target_idx < n_obs else time[-1] + timedelta(days=target_idx - n_obs + 1)
                    rows.append({
                        "vintage_time": time[v_idx],
                        "time": tt,
                        "store_a__wx": float(rng.random()),
                        "store_b__wx": float(rng.random()),
                    })
        X_forecast = pl.DataFrame(rows)

        f = LocalPanelForecaster(forecaster=PointReductionForecaster(estimator=Ridge()))
        f.fit(y[:80], forecasting_horizon=fh, X_forecast=X_forecast)

        pred1 = f.predict()
        pred2 = f.predict(X_forecast=X_forecast)

        assert len(pred1) == fh
        assert len(pred2) == fh

    def test_exogenous_column_isolation(self, panel_y):
        """Per-group clones see only their own local + global step columns."""
        y = panel_y

        time_ext = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 4, 19),
            interval="1d",
            eager=True,
        )
        X_future = pl.DataFrame({
            "time": time_ext,
            "store_a__promo": [float(i % 3) for i in range(len(time_ext))],
            "store_b__promo": [float(i % 5) for i in range(len(time_ext))],
            "holiday": [1.0 if t.weekday() == 6 else 0.0 for t in time_ext],
        })

        f = LocalPanelForecaster(forecaster=PointReductionForecaster(estimator=Ridge()))
        f.fit(y[:80], forecasting_horizon=3, X_future=X_future)

        # Each clone should have unprefixed local + global step columns
        for gn, fc in f.forecasters_.items():
            step_cols = [c for c in fc._X_t_observed.columns if "_step_" in c]
            # Should have promo_step_* and holiday_step_* (unprefixed)
            promo_steps = [c for c in step_cols if c.startswith("promo_")]
            holiday_steps = [c for c in step_cols if c.startswith("holiday_")]
            assert len(promo_steps) > 0, f"Clone {gn} missing promo step columns"
            assert len(holiday_steps) > 0, f"Clone {gn} missing holiday step columns"
            # Should NOT have other group's prefixed columns
            other_prefix = [c for c in step_cols if "__" in c]
            assert len(other_prefix) == 0, f"Clone {gn} has prefixed columns: {other_prefix}"

    def test_global_only_X_future(self, panel_y):
        """Global-only X_future (no prefixed columns) works for all groups."""
        y = panel_y

        time_ext = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 4, 19),
            interval="1d",
            eager=True,
        )
        X_future = pl.DataFrame({
            "time": time_ext,
            "holiday": [1.0 if t.weekday() == 6 else 0.0 for t in time_ext],
        })

        f = LocalPanelForecaster(forecaster=PointReductionForecaster(estimator=Ridge()))
        f.fit(y[:80], forecasting_horizon=3, X_future=X_future)

        # Both clones should have identical holiday step columns
        cols_a = sorted(c for c in f.forecasters_["store_a"]._X_t_observed.columns if "holiday" in c)
        cols_b = sorted(c for c in f.forecasters_["store_b"]._X_t_observed.columns if "holiday" in c)
        assert cols_a == cols_b
        assert len(cols_a) > 0

        y_pred = f.predict(forecasting_horizon=3)
        assert len(y_pred) == 3

    def test_predict_X_future_override_after_fit_none(self, panel_y):
        """fit(X_future=None) then predict(X_future=override) derives schema on the fly."""
        y = panel_y

        f = LocalPanelForecaster(forecaster=PointReductionForecaster(estimator=Ridge()))
        f.fit(y[:80], forecasting_horizon=3)
        assert f._local_X_future_schema_ is None

        time_ext = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 4, 19),
            interval="1d",
            eager=True,
        )
        X_future = pl.DataFrame({
            "time": time_ext,
            "store_a__promo": [float(i % 3) for i in range(len(time_ext))],
            "store_b__promo": [float(i % 5) for i in range(len(time_ext))],
        })

        y_pred = f.predict(forecasting_horizon=3, X_future=X_future)
        assert len(y_pred) == 3

    def test_observe_predict_with_stride(self, panel_y):
        """observe_predict with stride produces rolling multi-window output."""
        y = panel_y
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=7))
        f.fit(y[:80], forecasting_horizon=3)

        y_new = y[80:86]  # 6 rows
        y_pred = f.observe_predict(y=y_new, stride=2)

        # initial predict (3) + 3 stride windows (3*3=9) = 12 rows
        assert len(y_pred) == 12
        assert "vintage_time" in y_pred.columns
        assert "store_a__sales" in y_pred.columns
        assert "store_b__sales" in y_pred.columns

    def test_observe_predict_with_exogenous(self, panel_y):
        """observe_predict splits X_future per group and forwards stride."""
        y = panel_y

        time_ext = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 4, 19),
            interval="1d",
            eager=True,
        )
        X_future = pl.DataFrame({
            "time": time_ext,
            "store_a__promo": [float(i % 3) for i in range(len(time_ext))],
            "store_b__promo": [float(i % 5) for i in range(len(time_ext))],
        })

        f = LocalPanelForecaster(forecaster=PointReductionForecaster(estimator=Ridge()))
        f.fit(y[:80], forecasting_horizon=3, X_future=X_future)

        y_new = y[80:86]
        y_pred = f.observe_predict(y=y_new, stride=3, X_future=X_future)

        # initial predict (3) + 2 stride windows (2*3=6) = 9 rows
        assert len(y_pred) == 9
        assert "store_a__sales" in y_pred.columns

    def test_observe_predict_interval_with_stride(self, panel_y):
        """observe_predict_interval delegates to each clone's rolling loop."""
        y = panel_y
        f = LocalPanelForecaster(
            forecaster=SplitConformalForecaster(
                point_forecaster=SeasonalNaive(seasonality=7),
                calibration_size=20,
            ),
        )
        f.fit(y[:80], forecasting_horizon=3)

        y_new = y[80:86]
        y_pred = f.observe_predict_interval(y=y_new, stride=2, coverage_rates=[0.9])

        # initial predict (3) + 3 stride windows (3*3=9) = 12 rows
        assert len(y_pred) == 12
        # Should have lower/upper bound columns
        bound_cols = [c for c in y_pred.columns if "lower" in c or "upper" in c]
        assert len(bound_cols) > 0


class TestReassemblePanelPredictions:
    """Unit tests for _reassemble_panel_predictions stitching logic."""

    def _forecaster(self):
        return LocalPanelForecaster(SeasonalNaive(seasonality=1))

    def test_empty_mapping_returns_empty_frame(self):
        """No group predictions yields an empty DataFrame."""
        result = self._forecaster()._reassemble_panel_predictions({})
        assert result.height == 0
        assert result.width == 0

    def test_aligned_groups_join_and_sort(self):
        """Aligned time grids join on time keys and sort by vintage/time."""
        time = [datetime(2020, 1, 2), datetime(2020, 1, 1)]
        vintage = [datetime(2020, 1, 1), datetime(2020, 1, 1)]
        preds = {
            "a": pl.DataFrame({"vintage_time": vintage, "time": time, "value": [2.0, 1.0]}),
            "b": pl.DataFrame({"vintage_time": vintage, "time": time, "value": [20.0, 10.0]}),
        }
        result = self._forecaster()._reassemble_panel_predictions(preds)

        assert result["time"].to_list() == [datetime(2020, 1, 1), datetime(2020, 1, 2)]
        assert result["a__value"].to_list() == [1.0, 2.0]
        assert result["b__value"].to_list() == [10.0, 20.0]

    def test_mismatched_row_counts_raise(self):
        """Groups producing different numbers of rows raise a clear error."""
        preds = {
            "a": pl.DataFrame({"time": [datetime(2020, 1, 1), datetime(2020, 1, 2)], "value": [1.0, 2.0]}),
            "b": pl.DataFrame({"time": [datetime(2020, 1, 1)], "value": [10.0]}),
        }
        with pytest.raises(ValueError, match="must align to form a panel"):
            self._forecaster()._reassemble_panel_predictions(preds)

    def test_misaligned_time_grids_raise(self):
        """Same row count but disjoint timestamps cannot form a panel."""
        preds = {
            "a": pl.DataFrame({"time": [datetime(2020, 1, 1), datetime(2020, 1, 2)], "value": [1.0, 2.0]}),
            "b": pl.DataFrame({"time": [datetime(2020, 1, 3), datetime(2020, 1, 4)], "value": [10.0, 20.0]}),
        }
        with pytest.raises(ValueError, match="does not align"):
            self._forecaster()._reassemble_panel_predictions(preds)

    def test_groups_without_time_columns_concat_horizontally(self):
        """Frames without time keys are concatenated by position."""
        preds = {
            "a": pl.DataFrame({"value": [1.0, 2.0]}),
            "b": pl.DataFrame({"value": [10.0, 20.0]}),
        }
        result = self._forecaster()._reassemble_panel_predictions(preds)

        assert result["a__value"].to_list() == [1.0, 2.0]
        assert result["b__value"].to_list() == [10.0, 20.0]


class TestHeterogeneousPanelSchema:
    """fit() must reject panels whose groups have mismatched local schemas."""

    def test_mismatched_group_dtypes_raise(self):
        """Groups with different local target dtypes raise instead of silently using group[0]."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=29),
            interval="1d",
            eager=True,
        )
        # store_a sales are Float64, store_b sales are Int64: same local name, different dtype.
        y = pl.DataFrame({
            "time": time,
            "store_a__sales": [float(i) for i in range(30)],
            "store_b__sales": list(range(30)),
        })
        f = LocalPanelForecaster(forecaster=SeasonalNaive(seasonality=1))
        with pytest.raises(ValueError, match="same local"):
            f.fit(y, forecasting_horizon=3)
