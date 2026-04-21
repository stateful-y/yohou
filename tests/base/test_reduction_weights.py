"""Tests for sample_weight_alignment strategies and time_weight integration in reduction forecasters."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from yohou.point import PointReductionForecaster


@pytest.fixture
def reduction_data():
    """Create minimal time series data for reduction forecaster tests."""
    n = 50
    times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    values = [float(i) + np.sin(i * 0.5) for i in range(n)]
    y = pl.DataFrame({"time": times, "value": values})
    return y


@pytest.fixture
def linear_weight_fn():
    """Return a 1-param weight function giving linearly increasing weights."""

    def _weight_fn(time: pl.Series) -> pl.Series:
        n = len(time)
        return pl.Series("w", list(range(1, n + 1)), dtype=pl.Float64)

    return _weight_fn


def _make_forecaster():
    """Create a default PointReductionForecaster with LinearRegression."""
    return PointReductionForecaster(estimator=LinearRegression())


class TestSampleWeightAlignmentStrategies:
    """Tests for the five sample_weight_alignment strategies."""

    @pytest.mark.parametrize(
        "alignment",
        ["first_step", "mean_step", "weighted_mean_step", "max_weight_step", "min_weight_step"],
    )
    def test_alignment_strategy_produces_fitted_forecaster(self, reduction_data, linear_weight_fn, alignment):
        """Each alignment strategy should produce a fitted forecaster without error."""
        f = _make_forecaster()
        f.set_fit_request(time_weight=True, sample_weight_alignment=True)
        f.fit(
            reduction_data,
            forecasting_horizon=3,
            time_weight=linear_weight_fn,
            sample_weight_alignment=alignment,
        )
        result = f.predict(forecasting_horizon=3)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3

    def test_different_alignments_produce_different_predictions(self, reduction_data, linear_weight_fn):
        """Different alignment strategies should generally produce different predictions."""
        results = {}
        for alignment in ["first_step", "mean_step", "weighted_mean_step", "max_weight_step", "min_weight_step"]:
            f = _make_forecaster()
            f.set_fit_request(time_weight=True, sample_weight_alignment=True)
            f.fit(
                reduction_data,
                forecasting_horizon=3,
                time_weight=linear_weight_fn,
                sample_weight_alignment=alignment,
            )
            pred = f.predict(forecasting_horizon=3)
            results[alignment] = pred["value"].to_numpy()

        # At least some pairs should differ (not all alignments give identical results)
        pairs_differ = 0
        strategies = list(results.keys())
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i + 1 :]:
                if not np.allclose(results[s1], results[s2]):
                    pairs_differ += 1

        assert pairs_differ > 0, "All alignment strategies produced identical predictions"

    def test_weighted_mean_step_vs_mean_step_differ(self, reduction_data, linear_weight_fn):
        """weighted_mean_step and mean_step should produce different results with non-uniform weights."""
        forecasters = {}
        for alignment in ["mean_step", "weighted_mean_step"]:
            f = _make_forecaster()
            f.set_fit_request(time_weight=True, sample_weight_alignment=True)
            f.fit(
                reduction_data,
                forecasting_horizon=5,
                time_weight=linear_weight_fn,
                sample_weight_alignment=alignment,
            )
            forecasters[alignment] = f.predict(forecasting_horizon=5)

        pred_mean = forecasters["mean_step"]["value"].to_numpy()
        pred_wmean = forecasters["weighted_mean_step"]["value"].to_numpy()
        assert not np.allclose(pred_mean, pred_wmean)


class TestTimeWeightCallableFit:
    """Tests for time_weight callable integration with fit()."""

    def test_time_weight_none_matches_no_kwarg(self, reduction_data):
        """Passing time_weight=None should match omitting it entirely."""
        f1 = _make_forecaster()
        f1.fit(reduction_data, forecasting_horizon=3)
        pred1 = f1.predict(forecasting_horizon=3)

        f2 = _make_forecaster()
        f2.set_fit_request(time_weight=True)
        f2.fit(reduction_data, forecasting_horizon=3, time_weight=None)
        pred2 = f2.predict(forecasting_horizon=3)

        np.testing.assert_array_almost_equal(
            pred1["value"].to_numpy(),
            pred2["value"].to_numpy(),
        )

    def test_nonuniform_weight_changes_predictions(self, reduction_data, linear_weight_fn):
        """Non-uniform time_weight should change predictions vs no weight."""
        f_plain = _make_forecaster()
        f_plain.fit(reduction_data, forecasting_horizon=3)
        pred_plain = f_plain.predict(forecasting_horizon=3)

        f_weighted = _make_forecaster()
        f_weighted.set_fit_request(time_weight=True)
        f_weighted.fit(
            reduction_data,
            forecasting_horizon=3,
            time_weight=linear_weight_fn,
        )
        pred_weighted = f_weighted.predict(forecasting_horizon=3)

        assert not np.allclose(
            pred_plain["value"].to_numpy(),
            pred_weighted["value"].to_numpy(),
        )


class TestTimeWeightDataFrameFit:
    """Tests for time_weight as pl.DataFrame in fit()."""

    def test_dataframe_weight_works(self, reduction_data):
        """DataFrame time_weight with 'time' and 'weight' columns should work."""
        tw_df = pl.DataFrame({
            "time": reduction_data["time"].to_list(),
            "weight": list(range(1, len(reduction_data) + 1)),
        }).cast({"weight": pl.Float64})

        f = _make_forecaster()
        f.set_fit_request(time_weight=True)
        f.fit(reduction_data, forecasting_horizon=3, time_weight=tw_df)
        result = f.predict(forecasting_horizon=3)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3


class TestTimeWeightValidation:
    """Tests for time_weight error paths in reduction forecasters."""

    def test_zero_sum_weights_raises(self, reduction_data):
        """Weights summing to zero should raise ValueError."""

        def zero_weights(time: pl.Series) -> pl.Series:
            return pl.Series("w", [0.0] * len(time), dtype=pl.Float64)

        f = _make_forecaster()
        f.set_fit_request(time_weight=True)

        with pytest.raises(ValueError, match="zero"):
            f.fit(reduction_data, forecasting_horizon=3, time_weight=zero_weights)

    def test_estimator_without_sample_weight_raises(self, reduction_data, linear_weight_fn):
        """Estimator that doesn't support sample_weight should raise ValueError."""
        from sklearn.neighbors import KNeighborsRegressor

        f = PointReductionForecaster(estimator=KNeighborsRegressor())
        f.set_fit_request(time_weight=True)

        with pytest.raises(ValueError, match="sample_weight"):
            f.fit(reduction_data, forecasting_horizon=3, time_weight=linear_weight_fn)


class TestVintageWeightFit:
    """Tests for vintage_weight integration with fit()."""

    def test_vintage_weight_none_matches_no_kwarg(self, reduction_data):
        """Passing vintage_weight=None should match omitting it entirely."""
        f1 = _make_forecaster()
        f1.fit(reduction_data, forecasting_horizon=3)
        pred1 = f1.predict(forecasting_horizon=3)

        f2 = _make_forecaster()
        f2.set_fit_request(vintage_weight=True)
        f2.fit(reduction_data, forecasting_horizon=3, vintage_weight=None)
        pred2 = f2.predict(forecasting_horizon=3)

        np.testing.assert_array_almost_equal(
            pred1["value"].to_numpy(),
            pred2["value"].to_numpy(),
        )

    def test_vintage_weight_callable(self, reduction_data):
        """Callable vintage_weight should produce a fitted forecaster."""

        def vw_fn(time: pl.Series) -> pl.Series:
            n = len(time)
            return pl.Series("w", list(range(1, n + 1)), dtype=pl.Float64)

        f = _make_forecaster()
        f.set_fit_request(vintage_weight=True)
        f.fit(reduction_data, forecasting_horizon=3, vintage_weight=vw_fn)
        result = f.predict(forecasting_horizon=3)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3

    def test_vintage_weight_dataframe(self, reduction_data):
        """DataFrame vintage_weight should work."""
        vw_df = pl.DataFrame({
            "time": reduction_data["time"].to_list(),
            "weight": list(range(1, len(reduction_data) + 1)),
        }).cast({"weight": pl.Float64})

        f = _make_forecaster()
        f.set_fit_request(vintage_weight=True)
        f.fit(reduction_data, forecasting_horizon=3, vintage_weight=vw_df)
        result = f.predict(forecasting_horizon=3)
        assert isinstance(result, pl.DataFrame)

    def test_vintage_weight_dict(self, reduction_data):
        """Dict vintage_weight should work."""
        times = reduction_data["time"].to_list()
        vw_dict = {t: float(i + 1) for i, t in enumerate(times)}

        f = _make_forecaster()
        f.set_fit_request(vintage_weight=True)
        f.fit(reduction_data, forecasting_horizon=3, vintage_weight=vw_dict)
        result = f.predict(forecasting_horizon=3)
        assert isinstance(result, pl.DataFrame)

    def test_vintage_weight_changes_predictions(self, reduction_data):
        """Non-uniform vintage_weight should change predictions vs no weight."""
        f_plain = _make_forecaster()
        f_plain.fit(reduction_data, forecasting_horizon=3)
        pred_plain = f_plain.predict(forecasting_horizon=3)

        # Weight only the first 5 observations, zero out the rest
        def vw_fn(time: pl.Series) -> pl.Series:
            n = len(time)
            weights = [0.0] * n
            for i in range(min(5, n)):
                weights[i] = 1.0
            return pl.Series("w", weights, dtype=pl.Float64)

        f_weighted = _make_forecaster()
        f_weighted.set_fit_request(vintage_weight=True)
        f_weighted.fit(reduction_data, forecasting_horizon=3, vintage_weight=vw_fn)
        pred_weighted = f_weighted.predict(forecasting_horizon=3)

        assert not np.allclose(
            pred_plain["value"].to_numpy(),
            pred_weighted["value"].to_numpy(),
        )


class TestCombinedTimeAndVintageWeight:
    """Tests for combining time_weight and vintage_weight."""

    def test_both_weights_work_together(self, reduction_data, linear_weight_fn):
        """Providing both time_weight and vintage_weight should produce fitted forecaster."""

        def vw_fn(time: pl.Series) -> pl.Series:
            n = len(time)
            return pl.Series("w", list(range(1, n + 1)), dtype=pl.Float64)

        f = _make_forecaster()
        f.set_fit_request(time_weight=True, vintage_weight=True)
        f.fit(
            reduction_data,
            forecasting_horizon=3,
            time_weight=linear_weight_fn,
            vintage_weight=vw_fn,
        )
        result = f.predict(forecasting_horizon=3)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3

    def test_combined_differs_from_individual(self, reduction_data, linear_weight_fn):
        """Combined weights should differ from either individual weight alone."""

        def vw_fn(time: pl.Series) -> pl.Series:
            n = len(time)
            weights = [0.01] * n
            weights[-1] = 100.0
            return pl.Series("w", weights, dtype=pl.Float64)

        f_tw = _make_forecaster()
        f_tw.set_fit_request(time_weight=True)
        f_tw.fit(reduction_data, forecasting_horizon=3, time_weight=linear_weight_fn)
        pred_tw = f_tw.predict(forecasting_horizon=3)["value"].to_numpy()

        f_vw = _make_forecaster()
        f_vw.set_fit_request(vintage_weight=True)
        f_vw.fit(reduction_data, forecasting_horizon=3, vintage_weight=vw_fn)
        pred_vw = f_vw.predict(forecasting_horizon=3)["value"].to_numpy()

        f_both = _make_forecaster()
        f_both.set_fit_request(time_weight=True, vintage_weight=True)
        f_both.fit(
            reduction_data,
            forecasting_horizon=3,
            time_weight=linear_weight_fn,
            vintage_weight=vw_fn,
        )
        pred_both = f_both.predict(forecasting_horizon=3)["value"].to_numpy()

        differs_from_tw = not np.allclose(pred_both, pred_tw)
        differs_from_vw = not np.allclose(pred_both, pred_vw)
        assert differs_from_tw or differs_from_vw

    def test_zero_vintage_weight_raises(self, reduction_data):
        """All-zero vintage_weight should raise ValueError."""

        def zero_vw(time: pl.Series) -> pl.Series:
            return pl.Series("w", [0.0] * len(time), dtype=pl.Float64)

        f = _make_forecaster()
        f.set_fit_request(vintage_weight=True)
        with pytest.raises(ValueError, match="zero"):
            f.fit(reduction_data, forecasting_horizon=3, vintage_weight=zero_vw)


class TestVintageWeightDirectStrategies:
    """Tests for vintage_weight with different reduction strategies."""

    @pytest.mark.parametrize("strategy", ["multi-output", "direct"])
    def test_vintage_weight_with_strategy(self, reduction_data, strategy):
        """vintage_weight should work with each reduction strategy."""

        def vw_fn(time: pl.Series) -> pl.Series:
            n = len(time)
            return pl.Series("w", list(range(1, n + 1)), dtype=pl.Float64)

        f = PointReductionForecaster(estimator=LinearRegression(), reduction_strategy=strategy)
        f.set_fit_request(vintage_weight=True)
        f.fit(reduction_data, forecasting_horizon=3, vintage_weight=vw_fn)
        result = f.predict(forecasting_horizon=3)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
