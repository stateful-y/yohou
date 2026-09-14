"""HorizonRollingStatisticsTransformer inside reduction forecasters: alignment end to end."""

import pickle
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from yohou.compose import FeatureUnion
from yohou.interval import SplitConformalForecaster
from yohou.point import PointReductionForecaster
from yohou.point.reduction import PointReductionForecaster as _Reduction
from yohou.preprocessing import HorizonRollingStatisticsTransformer, LagTransformer
from yohou.testing.forecaster import check_step_feature_alignment_filters

K, N, H = 24, 2, 48


def _hourly_target(length: int, n_groups: int | None = None, seed: int = 0) -> pl.DataFrame:
    """Hourly target with a daily cycle; panel columns ``group_i__y_0`` when ``n_groups`` is set."""
    rng = np.random.default_rng(seed)
    times = pl.datetime_range(
        datetime(2021, 1, 1), datetime(2021, 1, 1) + timedelta(hours=length - 1), interval="1h", eager=True
    )
    cycle = np.sin(2 * np.pi * np.arange(length) / K)
    if n_groups is None:
        return pl.DataFrame({"time": times, "y_0": cycle + 0.1 * rng.normal(size=length)})
    return pl.DataFrame({
        "time": times,
        **{f"group_{g}__y_0": cycle * (g + 1) + 0.1 * rng.normal(size=length) for g in range(n_groups)},
    })


def _forecaster(alignment: str = "matched", **kwargs) -> PointReductionForecaster:
    """Direct forecaster whose features are a lag and per-step seasonal means of the target."""
    return PointReductionForecaster(
        estimator=LinearRegression(),
        actual_transformer=FeatureUnion([
            ("lag", LagTransformer(lag=1)),
            ("seasonal", HorizonRollingStatisticsTransformer(seasonality=K, n_seasons=N)),
        ]),
        target_as_feature="transformed",
        reduction_strategy="direct",
        step_feature_alignment=alignment,
        **kwargs,
    )


def _widths(forecaster) -> list[int]:
    return [estimator.n_features_in_ for estimator in forecaster.estimator_]


class TestConformanceCheck:
    """check_step_feature_alignment_filters covers step-output columns."""

    def test_passes_with_step_output_transformer(self):
        """No X_future or X_forecast needed: the actual transformer provides the step columns."""
        y = _hourly_target(200)
        check_step_feature_alignment_filters(_forecaster("all"), y, None, forecasting_horizon=6)

    def test_fails_when_step_output_columns_are_not_recognised(self, monkeypatch):
        """With recognition limited to derived columns, the check reports the silent no-op."""
        y = _hourly_target(200)
        original = _Reduction._is_step_column

        def derived_only(self, name, *, derived_only=False):
            return original(self, name, derived_only=True)

        monkeypatch.setattr(_Reduction, "_is_step_column", derived_only)
        with pytest.raises((AssertionError, RuntimeError)):
            check_step_feature_alignment_filters(_forecaster("all"), y, None, forecasting_horizon=6)


class TestAlignment:
    """Each per-step model reads only its own step's seasonal column."""

    @pytest.mark.parametrize(
        ("n_groups", "panel_strategy", "n_series"),
        [(None, "global", 1), (2, "global", 1), (2, "multivariate", 2)],
        ids=["standard", "panel-global", "panel-multivariate"],
    )
    def test_matched_and_cumulative_at_fit_predict_and_observe(self, n_groups, panel_strategy, n_series):
        """Widths at fit follow the alignment, and predict/observe/predict run on the same columns."""
        y = _hourly_target(400, n_groups=n_groups)
        train, new = y[:360], y[360:372]

        matched = _forecaster("matched", panel_strategy=panel_strategy).fit(train, forecasting_horizon=H)
        # lag column(s) + the step's own seasonal column(s)
        assert _widths(matched) == [2 * n_series] * H

        row = (
            matched._X_t_observed
            if not isinstance(matched._X_t_observed, dict)
            else next(iter(matched._X_t_observed.values()))
        )
        for step in (4, 30):
            kept = matched._filter_step_features(row.drop("time"), step).columns
            seasonal = [c for c in kept if "_s24_mean_step_" in c]
            assert seasonal and all(c.endswith(f"_step_{step}") for c in seasonal)
            assert [c for c in kept if "_lag_1" in c]

        cumulative = _forecaster("cumulative", panel_strategy=panel_strategy).fit(train, forecasting_horizon=H)
        assert _widths(cumulative)[3] == n_series + 4 * n_series

        everything = _forecaster("all", panel_strategy=panel_strategy).fit(train, forecasting_horizon=H)
        assert _widths(everything) == [n_series + H * n_series] * H

        first = matched.predict(forecasting_horizon=H)
        matched.observe(new)
        second = matched.predict(forecasting_horizon=H)
        assert first.height == second.height == H
        assert second["time"][0] > first["time"][0]


class TestConformalCalibration:
    """The bulk calibration path builds the same per-step features as the rolling path."""

    def test_bulk_matches_non_bulk_replay(self):
        """Conformity scores agree whether calibration replays in bulk or not."""

        class _NotBatchInvariant(HorizonRollingStatisticsTransformer):
            _tags = {"batch_invariant": False}

        def conformal(seasonal_cls) -> SplitConformalForecaster:
            point = PointReductionForecaster(
                estimator=LinearRegression(),
                actual_transformer=FeatureUnion([
                    ("lag", LagTransformer(lag=1)),
                    ("seasonal", seasonal_cls(seasonality=K, n_seasons=N)),
                ]),
                reduction_strategy="direct",
                step_feature_alignment="matched",
            )
            return SplitConformalForecaster(point_forecaster=point, calibration_size=60)

        y = _hourly_target(420)
        bulk = conformal(HorizonRollingStatisticsTransformer).fit(y, forecasting_horizon=12)
        other = conformal(_NotBatchInvariant).fit(y, forecasting_horizon=12)

        assert bulk.replay_path_ == "bulk"
        assert other.replay_path_ != "bulk"
        a = bulk.conformity_scores_.sort(["step", "time"])
        b = other.conformity_scores_.sort(["step", "time"])
        assert a.columns == b.columns and a.height == b.height
        value_columns = [c for c in a.columns if c not in ("time", "step", "vintage_time", "observed_time")]
        for column in value_columns:
            np.testing.assert_allclose(a[column].to_numpy(), b[column].to_numpy(), rtol=1e-9, atol=1e-12)


def test_pickle_round_trip_predicts_identically():
    """A fitted forecaster with a step-output transformer survives pickling unchanged."""
    y = _hourly_target(300)
    forecaster = _forecaster().fit(y, forecasting_horizon=H)
    restored = pickle.loads(pickle.dumps(forecaster))  # noqa: S301

    assert restored._actual_step_column_names_ == forecaster._actual_step_column_names_
    assert restored.predict(forecasting_horizon=H).equals(forecaster.predict(forecasting_horizon=H))
