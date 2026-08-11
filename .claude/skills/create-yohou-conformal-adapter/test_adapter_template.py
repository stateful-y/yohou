"""Template for testing a new conformal adapter.

Two layers:
1. The systematic checks via ``_yield_yohou_conformal_adapter_checks`` (wired
   in ``tests/test_common.py`` ``_conformal_adapter_instances`` +
   ``TestConformalAdapterCommon``); register your instance there.
2. Behavioral / integration tests below, in ``tests/interval/test_adapter.py``.
"""

import pytest
from sklearn.exceptions import NotFittedError

from yohou.interval import SplitConformalForecaster
from yohou.interval.adapter import MyAdapter  # rename to your class
from yohou.point import SeasonalNaive


class TestMyAdapterUnit:
    """The adapter in isolation."""

    def test_fit_seeds_target_level(self):
        """Seed one level per rate at 1 - coverage_rate."""
        adapter = MyAdapter().fit([0.9], symmetric=True)
        assert adapter.predict()[0.9] == pytest.approx(0.1)

    def test_observe_rewind_round_trip(self):
        """Observe then rewind restores the levels exactly."""
        adapter = MyAdapter().fit([0.9, 0.8], symmetric=True)
        before = adapter.predict()
        adapter.observe([{0.9: 1.0, 0.8: 0.0}, {0.9: 0.0, 0.8: 1.0}])
        assert adapter.predict() != before
        adapter.rewind(2)
        assert adapter.predict() == before

    def test_asymmetric_two_tails(self):
        """Asymmetric scorers seed a (lower, upper) level per rate."""
        adapter = MyAdapter().fit([0.9], symmetric=False)
        lower, upper = adapter.predict()[0.9]
        assert lower == pytest.approx(0.05)
        assert upper == pytest.approx(0.05)

    def test_unfitted_methods_raise(self):
        """predict/observe/rewind raise NotFittedError before fit."""
        adapter = MyAdapter()
        with pytest.raises(NotFittedError):
            adapter.predict()
        with pytest.raises(NotFittedError):
            adapter.observe([{0.9: 1.0}])
        with pytest.raises(NotFittedError):
            adapter.rewind(1)


class TestMyAdapterInForecaster:
    """The adapter wired into SplitConformalForecaster."""

    def test_lifecycle_end_to_end(self):
        """The adapter adapts and predicts through the forecaster lifecycle."""
        # Build ~200 rows of your representative series here.
        y = ...  # pl.DataFrame with "time" + value column
        forecaster = SplitConformalForecaster(
            point_forecaster=SeasonalNaive(seasonality=7),
            calibration_size=50,
            adapter=MyAdapter(),
        )
        forecaster.fit(y[:180], forecasting_horizon=3, coverage_rates=[0.9])
        before = forecaster.adapters_["step_1"].predict()
        forecaster.observe(y[180:188])
        assert forecaster.adapters_["step_1"].predict() != before
        intervals = forecaster.predict_interval()
        assert "value_lower_0.9" in intervals.columns
