"""Tests for sample_weight_alignment strategies and weighter integration in reduction forecasters."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from yohou.point import PointReductionForecaster
from yohou.utils.weighting import LinearDecayWeighter, LookupWeighter, TableWeighter


@pytest.fixture
def reduction_data():
    """Create minimal time series data for reduction forecaster tests."""
    n = 50
    times = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    values = [float(i) + np.sin(i * 0.5) for i in range(n)]
    y = pl.DataFrame({"time": times, "value": values})
    return y


def _increasing_table(reduction_data: pl.DataFrame) -> TableWeighter:
    """A TableWeighter giving linearly increasing weights over the data times."""
    frame = pl.DataFrame({
        "time": reduction_data["time"].to_list(),
        "weight": [float(i + 1) for i in range(len(reduction_data))],
    })
    return TableWeighter(frame=frame, on="time")


def _make_forecaster(**kwargs):
    """Create a default PointReductionForecaster with LinearRegression."""
    return PointReductionForecaster(estimator=LinearRegression(), **kwargs)


class TestSampleWeightAlignmentStrategies:
    """Tests for the five sample_weight_alignment strategies."""

    @pytest.mark.parametrize(
        "alignment",
        ["first_step", "mean_step", "weighted_mean_step", "max_weight_step", "min_weight_step"],
    )
    def test_alignment_strategy_produces_fitted_forecaster(self, reduction_data, alignment):
        """Each alignment strategy should produce a fitted forecaster without error."""
        f = _make_forecaster(time_weighter=LinearDecayWeighter(), sample_weight_alignment=alignment)
        f.fit(reduction_data, forecasting_horizon=3)
        result = f.predict(forecasting_horizon=3)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3

    def test_different_alignments_produce_different_predictions(self, reduction_data):
        """Different alignment strategies should generally produce different predictions."""
        results = {}
        for alignment in ["first_step", "mean_step", "weighted_mean_step", "max_weight_step", "min_weight_step"]:
            f = _make_forecaster(time_weighter=LinearDecayWeighter(), sample_weight_alignment=alignment)
            f.fit(reduction_data, forecasting_horizon=3)
            pred = f.predict(forecasting_horizon=3)
            results[alignment] = pred["value"].to_numpy()

        pairs_differ = 0
        strategies = list(results.keys())
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i + 1 :]:
                if not np.allclose(results[s1], results[s2]):
                    pairs_differ += 1

        assert pairs_differ > 0, "All alignment strategies produced identical predictions"

    def test_weighted_mean_step_vs_mean_step_differ(self, reduction_data):
        """weighted_mean_step and mean_step should produce different results with non-uniform weights."""
        forecasters = {}
        for alignment in ["mean_step", "weighted_mean_step"]:
            f = _make_forecaster(time_weighter=LinearDecayWeighter(), sample_weight_alignment=alignment)
            f.fit(reduction_data, forecasting_horizon=5)
            forecasters[alignment] = f.predict(forecasting_horizon=5)

        pred_mean = forecasters["mean_step"]["value"].to_numpy()
        pred_wmean = forecasters["weighted_mean_step"]["value"].to_numpy()
        assert not np.allclose(pred_mean, pred_wmean)


class TestTimeWeighterFit:
    """Tests for time_weighter integration with fit()."""

    def test_time_weighter_none_matches_no_weighter(self, reduction_data):
        """Passing time_weighter=None should match omitting it entirely."""
        f1 = _make_forecaster()
        f1.fit(reduction_data, forecasting_horizon=3)
        pred1 = f1.predict(forecasting_horizon=3)

        f2 = _make_forecaster(time_weighter=None)
        f2.fit(reduction_data, forecasting_horizon=3)
        pred2 = f2.predict(forecasting_horizon=3)

        np.testing.assert_array_almost_equal(pred1["value"].to_numpy(), pred2["value"].to_numpy())

    def test_nonuniform_weighter_changes_predictions(self, reduction_data):
        """Non-uniform time_weighter should change predictions vs no weight."""
        f_plain = _make_forecaster()
        f_plain.fit(reduction_data, forecasting_horizon=3)
        pred_plain = f_plain.predict(forecasting_horizon=3)

        f_weighted = _make_forecaster(time_weighter=LinearDecayWeighter())
        f_weighted.fit(reduction_data, forecasting_horizon=3)
        pred_weighted = f_weighted.predict(forecasting_horizon=3)

        assert not np.allclose(pred_plain["value"].to_numpy(), pred_weighted["value"].to_numpy())


class TestTableWeighterFit:
    """Tests for TableWeighter (DataFrame-backed) time weighting in fit()."""

    def test_table_weighter_works(self, reduction_data):
        """A TableWeighter with 'time' and 'weight' columns should work."""
        f = _make_forecaster(time_weighter=_increasing_table(reduction_data))
        f.fit(reduction_data, forecasting_horizon=3)
        result = f.predict(forecasting_horizon=3)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3


class TestTimeWeighterValidation:
    """Tests for time_weighter error paths in reduction forecasters."""

    def test_zero_sum_weights_raises(self, reduction_data):
        """Weights summing to zero should raise ValueError."""
        f = _make_forecaster(time_weighter=LookupWeighter(mapping={}, default=0.0))
        with pytest.raises(ValueError, match="zero"):
            f.fit(reduction_data, forecasting_horizon=3)

    def test_estimator_without_sample_weight_raises(self, reduction_data):
        """Estimator that doesn't support sample_weight should raise ValueError."""
        from sklearn.neighbors import KNeighborsRegressor

        f = PointReductionForecaster(estimator=KNeighborsRegressor(), time_weighter=LinearDecayWeighter())
        with pytest.raises(ValueError, match="sample_weight"):
            f.fit(reduction_data, forecasting_horizon=3)


class TestVintageWeighterFit:
    """Tests for vintage_weighter integration with fit()."""

    def test_vintage_weighter_none_matches_no_weighter(self, reduction_data):
        """Passing vintage_weighter=None should match omitting it entirely."""
        f1 = _make_forecaster()
        f1.fit(reduction_data, forecasting_horizon=3)
        pred1 = f1.predict(forecasting_horizon=3)

        f2 = _make_forecaster(vintage_weighter=None)
        f2.fit(reduction_data, forecasting_horizon=3)
        pred2 = f2.predict(forecasting_horizon=3)

        np.testing.assert_array_almost_equal(pred1["value"].to_numpy(), pred2["value"].to_numpy())

    def test_vintage_weighter_works(self, reduction_data):
        """A vintage TableWeighter should produce a fitted forecaster."""
        f = _make_forecaster(vintage_weighter=_increasing_table(reduction_data))
        f.fit(reduction_data, forecasting_horizon=3)
        result = f.predict(forecasting_horizon=3)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3


class TestInvalidSampleWeightAlignment:
    """Tests for invalid sample_weight_alignment values."""

    def test_invalid_alignment_raises(self, reduction_data):
        """Invalid sample_weight_alignment should raise an error mentioning the param."""
        f = _make_forecaster(time_weighter=LinearDecayWeighter(), sample_weight_alignment="invalid_strategy")
        with pytest.raises(ValueError, match="sample_weight_alignment"):
            f.fit(reduction_data, forecasting_horizon=3)

    def test_vintage_weighter_dict(self, reduction_data):
        """A LookupWeighter (dict-backed) vintage weighter should work."""
        times = reduction_data["time"].to_list()
        vw = LookupWeighter(mapping={t: float(i + 1) for i, t in enumerate(times)})
        f = _make_forecaster(vintage_weighter=vw)
        f.fit(reduction_data, forecasting_horizon=3)
        result = f.predict(forecasting_horizon=3)
        assert isinstance(result, pl.DataFrame)

    def test_vintage_weighter_changes_predictions(self, reduction_data):
        """Non-uniform vintage_weighter should change predictions vs no weight."""
        f_plain = _make_forecaster()
        f_plain.fit(reduction_data, forecasting_horizon=3)
        pred_plain = f_plain.predict(forecasting_horizon=3)

        # Weight only the first 5 observations, zero out the rest.
        times = reduction_data["time"].to_list()
        weights = [1.0 if i < 5 else 0.0 for i in range(len(times))]
        vw = TableWeighter(frame=pl.DataFrame({"time": times, "weight": weights}), on="time")

        f_weighted = _make_forecaster(vintage_weighter=vw)
        f_weighted.fit(reduction_data, forecasting_horizon=3)
        pred_weighted = f_weighted.predict(forecasting_horizon=3)

        assert not np.allclose(pred_plain["value"].to_numpy(), pred_weighted["value"].to_numpy())


class TestCombinedTimeAndVintageWeighter:
    """Tests for combining time_weighter and vintage_weighter."""

    def test_both_weighters_work_together(self, reduction_data):
        """Providing both time_weighter and vintage_weighter should produce a fitted forecaster."""
        f = _make_forecaster(
            time_weighter=LinearDecayWeighter(),
            vintage_weighter=_increasing_table(reduction_data),
        )
        f.fit(reduction_data, forecasting_horizon=3)
        result = f.predict(forecasting_horizon=3)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3

    def test_combined_differs_from_individual(self, reduction_data):
        """Combined weights should differ from either individual weight alone."""
        times = reduction_data["time"].to_list()
        weights = [0.01] * len(times)
        weights[-1] = 100.0
        vw = TableWeighter(frame=pl.DataFrame({"time": times, "weight": weights}), on="time")

        f_tw = _make_forecaster(time_weighter=LinearDecayWeighter())
        f_tw.fit(reduction_data, forecasting_horizon=3)
        pred_tw = f_tw.predict(forecasting_horizon=3)["value"].to_numpy()

        f_vw = _make_forecaster(vintage_weighter=vw)
        f_vw.fit(reduction_data, forecasting_horizon=3)
        pred_vw = f_vw.predict(forecasting_horizon=3)["value"].to_numpy()

        f_both = _make_forecaster(time_weighter=LinearDecayWeighter(), vintage_weighter=vw)
        f_both.fit(reduction_data, forecasting_horizon=3)
        pred_both = f_both.predict(forecasting_horizon=3)["value"].to_numpy()

        assert not np.allclose(pred_both, pred_tw) or not np.allclose(pred_both, pred_vw)

    def test_zero_vintage_weight_raises(self, reduction_data):
        """All-zero vintage_weighter should raise ValueError."""
        f = _make_forecaster(vintage_weighter=LookupWeighter(mapping={}, default=0.0))
        with pytest.raises(ValueError, match="zero"):
            f.fit(reduction_data, forecasting_horizon=3)


class TestVintageWeighterDirectStrategies:
    """Tests for vintage_weighter with different reduction strategies."""

    @pytest.mark.parametrize("strategy", ["multi-output", "direct"])
    def test_vintage_weighter_with_strategy(self, reduction_data, strategy):
        """vintage_weighter should work with each reduction strategy."""
        f = PointReductionForecaster(
            estimator=LinearRegression(),
            reduction_strategy=strategy,
            vintage_weighter=_increasing_table(reduction_data),
        )
        f.fit(reduction_data, forecasting_horizon=3)
        result = f.predict(forecasting_horizon=3)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
