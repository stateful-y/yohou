"""End-to-end integration tests for X_actual + X_future + X_forecast API.

Tests the full lifecycle of the three-parameter exogenous feature API using
a synthetic electricity price forecasting scenario with known relationships:

- y: hourly prices = base_load + weather_effect + holiday_effect
- X_actual: realized temperature readings (observation features)
- X_future: holiday indicators (deterministic, known for all dates)
- X_forecast: weather forecasts with vintage_time (uncertain, multi-vintage)

The synthetic data has exact linear relationships so we can verify:
1. Step columns are correctly derived and injected
2. Different X_forecast vintages produce different predictions
3. observe/predict cycle preserves state correctly
4. observe_predict loop handles all three parameter types
5. Pickle round-trip preserves step column state
"""

import pickle
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import NotFittedError

from yohou.metrics import MeanAbsoluteError
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer

FREQ = "1h"
N_TRAIN = 200
N_TEST = 48
H = 6
SEED = 42


def _make_times(n: int, start: datetime | None = None) -> pl.Series:
    """Generate hourly datetime series."""
    s = start or datetime(2024, 1, 1)
    return pl.Series("time", [s + timedelta(hours=i) for i in range(n)])


def _make_holiday(times: pl.Series) -> pl.DataFrame:
    """Holiday indicator: 1.0 on Sundays, 0.0 otherwise."""
    return pl.DataFrame({
        "time": times,
        "is_holiday": [1.0 if t.weekday() == 6 else 0.0 for t in times],
    })


def _make_actual_temp(times: pl.Series, rng: np.random.Generator) -> pl.DataFrame:
    """Realized temperature: sinusoidal daily cycle + noise."""
    n = len(times)
    t = np.arange(n, dtype=float)
    temp = 15.0 + 5.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, n)
    return pl.DataFrame({"time": times, "temperature": temp})


def _make_weather_forecast(
    vintage_times: pl.Series,
    forecast_times: pl.Series,
    bias: float,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Weather forecast with vintage_time: actual temp + bias + noise.

    Each vintage covers all forecast_times. Different biases simulate
    different forecast vintages (earlier forecasts are less accurate).
    """
    n_vintages = len(vintage_times)
    n_steps = len(forecast_times)
    t = np.arange(n_steps, dtype=float)
    base_temp = 15.0 + 5.0 * np.sin(2 * np.pi * t / 24)

    rows = []
    for i in range(n_vintages):
        vt = vintage_times[i]
        for j in range(n_steps):
            rows.append({
                "vintage_time": vt,
                "time": forecast_times[j],
                "wx_temp": float(base_temp[j] + bias + rng.normal(0, 0.3)),
            })
    return pl.DataFrame(rows)


def _make_price(
    times: pl.Series,
    actual_temp: pl.DataFrame,
    holiday: pl.DataFrame,
) -> pl.DataFrame:
    """Synthetic price = 50 + 2*temperature + 10*is_holiday + small noise."""
    rng = np.random.default_rng(SEED + 1)
    temp = actual_temp["temperature"].to_numpy()
    hol = holiday["is_holiday"].to_numpy()
    price = 50.0 + 2.0 * temp + 10.0 * hol + rng.normal(0, 0.1, len(times))
    return pl.DataFrame({"time": times, "price": price})


@pytest.fixture
def electricity_data():
    """Complete electricity dataset with train/test split.

    Returns dict with keys: y_train, y_test, X_actual_train, X_actual_test,
    X_future_full, X_forecast_train, X_forecast_test_accurate,
    X_forecast_test_biased, all_times.
    """
    rng = np.random.default_rng(SEED)
    n_total = N_TRAIN + N_TEST
    times = _make_times(n_total)

    holiday = _make_holiday(times)
    actual_temp = _make_actual_temp(times, rng)
    y = _make_price(times, actual_temp, holiday)

    # Train/test split
    y_train = y[:N_TRAIN]
    y_test = y[N_TRAIN:]

    X_actual_train = actual_temp[:N_TRAIN]
    X_actual_test = actual_temp[N_TRAIN:]

    # X_future covers the full range (deterministic, always available)
    X_future_full = holiday

    # X_forecast for training: one vintage per observation time
    # Each training vintage anchored at that observation time, covering H steps
    train_times = times[:N_TRAIN]
    forecast_rows = []
    for i in range(H, N_TRAIN):
        vt = train_times[i]
        for step in range(1, H + 1):
            future_idx = i + step
            if future_idx < n_total:
                t_arr = np.array([future_idx], dtype=float)
                forecast_rows.append({
                    "vintage_time": vt,
                    "time": times[future_idx],
                    "wx_temp": float(15.0 + 5.0 * np.sin(2 * np.pi * t_arr[0] / 24) + 0.5 + rng.normal(0, 0.3)),
                })
    X_forecast_train = pl.DataFrame(forecast_rows)

    # X_forecast for test: two vintages at the last training time
    last_obs = times[N_TRAIN - 1]
    test_times = times[N_TRAIN : N_TRAIN + H]

    # Accurate vintage (small bias)
    X_forecast_test_accurate = _make_weather_forecast(
        pl.Series("vintage_time", [last_obs]),
        test_times,
        bias=0.1,
        rng=np.random.default_rng(100),
    )

    # Biased vintage (large bias)
    X_forecast_test_biased = _make_weather_forecast(
        pl.Series("vintage_time", [last_obs]),
        test_times,
        bias=5.0,
        rng=np.random.default_rng(200),
    )

    return {
        "y_train": y_train,
        "y_test": y_test,
        "X_actual_train": X_actual_train,
        "X_actual_test": X_actual_test,
        "X_future_full": X_future_full,
        "X_forecast_train": X_forecast_train,
        "X_forecast_test_accurate": X_forecast_test_accurate,
        "X_forecast_test_biased": X_forecast_test_biased,
        "all_times": times,
    }


def _build_forecaster(**overrides):
    """Build a standard PointReductionForecaster for electricity tests."""
    defaults = {
        "estimator": HistGradientBoostingRegressor(
            max_iter=50,
            max_depth=3,
            random_state=SEED,
        ),
        "feature_transformer": LagTransformer([1, 2, 3]),
        "reduction_strategy": "direct",
        "forecasting_horizon": H,
    }
    defaults.update(overrides)
    fh = defaults.pop("forecasting_horizon")
    f = PointReductionForecaster(**defaults)
    return f, fh


@pytest.mark.integration
class TestFitPredictWithExogenous:
    """Test fit and predict with all three exogenous parameter types."""

    def test_fit_with_all_three_params(self, electricity_data):
        """Fit accepts X_actual, X_future, and X_forecast simultaneously."""
        d = electricity_data
        f, fh = _build_forecaster()

        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
            X_forecast=d["X_forecast_train"],
        )

        # Generic fitted attributes (observed_time_, fit_forecasting_horizon_)
        # are covered by check_fit_sets_forecaster_attributes in the systematic
        # suite; here we assert only the exogenous-specific fit state that the
        # three-parameter API must populate.
        assert len(f._step_column_names_) > 0
        assert f._X_future_raw_ is not None
        assert f._X_forecast_raw_ is not None

    def test_predict_bare_after_fit(self, electricity_data):
        """Bare predict() uses stored raws for step columns."""
        d = electricity_data
        f, fh = _build_forecaster()
        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
            X_forecast=d["X_forecast_train"],
        )

        y_pred = f.predict()
        assert len(y_pred) == H
        assert "vintage_time" in y_pred.columns
        assert "time" in y_pred.columns
        assert "price" in y_pred.columns

    def test_multi_vintage_predictions_differ(self, electricity_data):
        """Different X_forecast vintages produce different predictions."""
        d = electricity_data
        f, fh = _build_forecaster()
        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
            X_forecast=d["X_forecast_train"],
        )

        pred_accurate = f.predict(X_forecast=d["X_forecast_test_accurate"])
        pred_biased = f.predict(X_forecast=d["X_forecast_test_biased"])

        # Predictions must differ (different weather forecasts)
        prices_accurate = pred_accurate["price"].to_numpy()
        prices_biased = pred_biased["price"].to_numpy()
        assert not np.allclose(prices_accurate, prices_biased, atol=0.01), "Multi-vintage predictions should differ"

    def test_predict_does_not_mutate_state(self, electricity_data):
        """predict(X_forecast=...) does not change internal state.

        Stronger than check_predict_X_forecast_override (which only checks the
        _X_forecast_raw_ pointer is unchanged): this asserts the actual baseline
        prediction array is bit-for-bit identical before and after an override,
        on the realistic electricity scenario.
        """
        d = electricity_data
        f, fh = _build_forecaster()
        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
            X_forecast=d["X_forecast_train"],
        )

        # Get baseline prediction
        pred_before = f.predict()

        # Override with biased vintage
        _ = f.predict(X_forecast=d["X_forecast_test_biased"])

        # Baseline must be unchanged
        pred_after = f.predict()
        np.testing.assert_array_almost_equal(
            pred_before["price"].to_numpy(),
            pred_after["price"].to_numpy(),
            decimal=10,
        )

    def test_accurate_vintage_beats_biased(self, electricity_data):
        """Accurate weather forecast produces better price predictions."""
        d = electricity_data
        f, fh = _build_forecaster()
        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
            X_forecast=d["X_forecast_train"],
        )

        pred_acc = f.predict(X_forecast=d["X_forecast_test_accurate"])
        pred_bias = f.predict(X_forecast=d["X_forecast_test_biased"])

        scorer = MeanAbsoluteError()
        scorer.fit(d["y_train"])
        y_test_h = d["y_test"][:H]

        score_acc = scorer.score(y_test_h, pred_acc)
        score_bias = scorer.score(y_test_h, pred_bias)

        # Accurate forecast should produce lower error
        assert score_acc < score_bias, f"Accurate vintage ({score_acc:.4f}) should beat biased ({score_bias:.4f})"


@pytest.mark.integration
class TestObserveWithExogenous:
    """Test observe and rewind with step column state."""

    def test_observe_updates_state(self, electricity_data):
        """observe() with X_actual advances observed_time_ and re-derives step cols."""
        d = electricity_data
        f, fh = _build_forecaster()
        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
            X_forecast=d["X_forecast_train"],
        )
        obs_time_before = f.observed_time_

        # Observe one new row
        new_y = d["y_test"][:1]
        new_X = d["X_actual_test"][:1]
        f.observe(y=new_y, X_actual=new_X)

        assert f.observed_time_ > obs_time_before

    def test_observe_then_predict_uses_new_state(self, electricity_data):
        """After observe, predict uses the updated observation buffer."""
        d = electricity_data
        f, fh = _build_forecaster()
        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
            X_forecast=d["X_forecast_train"],
        )

        pred_before = f.predict()

        # Observe a few rows
        n_obs = 3
        f.observe(
            y=d["y_test"][:n_obs],
            X_actual=d["X_actual_test"][:n_obs],
        )

        pred_after = f.predict()

        # Predictions should differ (different observation state)
        assert not np.allclose(
            pred_before["price"].to_numpy(),
            pred_after["price"].to_numpy(),
            atol=0.01,
        )

    def test_observe_X_forecast_all_future_vintages_clears_raw(self, electricity_data):
        """observe() with only future-dated vintages narrows _X_forecast_raw_ to empty.

        observe() keeps the single latest vintage at or before observed_time_.
        When every supplied vintage is later than the new observed_time_, the
        as-of filter matches nothing and _X_forecast_raw_ becomes an empty
        frame; predict() must still return H rows (step columns may be null).
        """
        d = electricity_data
        f, fh = _build_forecaster()
        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
            X_forecast=d["X_forecast_train"],
        )

        # Observe one new row to advance observed_time_ into the test range.
        f.observe(y=d["y_test"][:1], X_actual=d["X_actual_test"][:1])

        # Build an X_forecast whose vintages all lie strictly after the current
        # observed_time_, so the as-of filter (latest vintage <= observed_time_)
        # matches nothing.
        all_times = d["all_times"]
        future_vintage_start = N_TRAIN + 10
        future_only = _make_weather_forecast(
            pl.Series("vintage_time", list(all_times[future_vintage_start : future_vintage_start + 3])),
            all_times[future_vintage_start + 3 : future_vintage_start + 3 + H],
            bias=0.1,
            rng=np.random.default_rng(700),
        )
        assert future_only.height > 0
        assert future_only["vintage_time"].min() > f.observed_time_

        f.observe(
            y=d["y_test"][1:2],
            X_actual=d["X_actual_test"][1:2],
            X_forecast=future_only,
        )

        # No vintage qualified, so the stored raw is narrowed to an empty frame.
        assert f._X_forecast_raw_ is not None
        assert f._X_forecast_raw_.height == 0

        # predict() still produces a correctly shaped block.
        y_pred = f.predict()
        assert len(y_pred) == H
        assert "vintage_time" in y_pred.columns
        assert "time" in y_pred.columns

    def test_rewind_restores_state(self, electricity_data):
        """rewind() restores the forecaster to a prior observation point."""
        d = electricity_data
        f, fh = _build_forecaster()
        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
            X_forecast=d["X_forecast_train"],
        )

        pred_original = f.predict()
        obs_time_original = f.observed_time_

        # Observe
        f.observe(
            y=d["y_test"][:5],
            X_actual=d["X_actual_test"][:5],
        )
        assert f.observed_time_ != obs_time_original

        # Rewind back
        f.rewind(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
        )

        pred_rewound = f.predict()
        np.testing.assert_array_almost_equal(
            pred_original["price"].to_numpy(),
            pred_rewound["price"].to_numpy(),
            decimal=8,
        )


@pytest.mark.integration
class TestObservePredictLoop:
    """Test the observe_predict walk-forward loop with exogenous features."""

    def test_observe_predict_with_X_future(self, electricity_data):
        """observe_predict loop handles X_future correctly."""
        d = electricity_data
        f, fh = _build_forecaster()
        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
        )

        preds = f.observe_predict(
            y=d["y_test"],
            X_actual=d["X_actual_test"],
            X_future=d["X_future_full"],
            stride=fh,
        )

        # Should produce predictions for the test range
        assert len(preds) > 0
        assert "vintage_time" in preds.columns
        assert "time" in preds.columns
        assert "price" in preds.columns

    def test_observe_predict_with_all_three(self, electricity_data):
        """observe_predict loop handles X_actual, X_future, and X_forecast."""
        d = electricity_data
        f, fh = _build_forecaster()

        # Build X_forecast for the full range (train + test)
        rng = np.random.default_rng(300)
        all_times = d["all_times"]
        forecast_rows = []
        for i in range(H, len(all_times)):
            vt = all_times[i]
            for step in range(1, H + 1):
                future_idx = i + step
                if future_idx < len(all_times):
                    t_val = float(future_idx)
                    forecast_rows.append({
                        "vintage_time": vt,
                        "time": all_times[future_idx],
                        "wx_temp": float(15.0 + 5.0 * np.sin(2 * np.pi * t_val / 24) + 0.5 + rng.normal(0, 0.3)),
                    })
        X_forecast_full = pl.DataFrame(forecast_rows)

        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
            X_forecast=d["X_forecast_train"],
        )

        preds = f.observe_predict(
            y=d["y_test"],
            X_actual=d["X_actual_test"],
            X_future=d["X_future_full"],
            X_forecast=X_forecast_full,
            stride=fh,
        )

        assert len(preds) > 0
        assert "vintage_time" in preds.columns


@pytest.mark.integration
class TestStepFeatureAlignment:
    """Test step_feature_alignment modes with exogenous features."""

    @pytest.mark.parametrize("alignment", ["all", "matched", "cumulative"])
    def test_alignment_modes_produce_predictions(self, electricity_data, alignment):
        """All three alignment modes produce valid predictions."""
        d = electricity_data
        f, fh = _build_forecaster(step_feature_alignment=alignment)
        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
            X_forecast=d["X_forecast_train"],
        )

        y_pred = f.predict()
        assert len(y_pred) == H
        # Every alignment mode must still emit the mandatory time columns.
        assert "vintage_time" in y_pred.columns
        assert "time" in y_pred.columns


@pytest.mark.integration
class TestPickleRoundTrip:
    """Test that pickle preserves step column state."""

    def test_pickle_preserves_predictions(self, electricity_data):
        """Pickle round-trip produces identical predictions."""
        d = electricity_data
        f, fh = _build_forecaster()
        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
            X_forecast=d["X_forecast_train"],
        )

        pred_before = f.predict()

        # Round-trip
        buf = pickle.dumps(f)
        f2 = pickle.loads(buf)  # noqa: S301

        pred_after = f2.predict()
        np.testing.assert_array_almost_equal(
            pred_before["price"].to_numpy(),
            pred_after["price"].to_numpy(),
            decimal=10,
        )

    def test_pickle_preserves_step_column_names(self, electricity_data):
        """Pickle preserves _step_column_names_ and stored raws."""
        d = electricity_data
        f, fh = _build_forecaster()
        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
            X_forecast=d["X_forecast_train"],
        )

        buf = pickle.dumps(f)
        f2 = pickle.loads(buf)  # noqa: S301

        assert f2._step_column_names_ == f._step_column_names_
        assert f2._X_future_raw_ is not None
        assert f2._X_forecast_raw_ is not None

    def test_pickle_preserves_multi_vintage(self, electricity_data):
        """After pickle, different vintages still produce different predictions."""
        d = electricity_data
        f, fh = _build_forecaster()
        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
            X_forecast=d["X_forecast_train"],
        )

        buf = pickle.dumps(f)
        f2 = pickle.loads(buf)  # noqa: S301

        pred_acc = f2.predict(X_forecast=d["X_forecast_test_accurate"])
        pred_bias = f2.predict(X_forecast=d["X_forecast_test_biased"])

        assert not np.allclose(
            pred_acc["price"].to_numpy(),
            pred_bias["price"].to_numpy(),
            atol=0.01,
        )


@pytest.mark.integration
class TestFitOnlySubsets:
    """Test fit with only some of the three exogenous params."""

    def test_fit_with_X_actual_only(self, electricity_data):
        """Fit with X_actual only (no step columns)."""
        d = electricity_data
        f, fh = _build_forecaster()
        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
        )

        y_pred = f.predict()
        assert len(y_pred) == H
        assert len(f._step_column_names_) == 0

    def test_fit_with_X_future_only(self, electricity_data):
        """Fit with X_future only (step columns from holidays)."""
        d = electricity_data
        f, fh = _build_forecaster()
        f.fit(
            y=d["y_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
        )

        y_pred = f.predict()
        assert len(y_pred) == H
        assert len(f._step_column_names_) > 0
        assert f._X_future_raw_ is not None
        assert f._X_forecast_raw_ is None

    def test_fit_with_X_forecast_only(self, electricity_data):
        """Fit with X_forecast only (step columns from weather)."""
        d = electricity_data
        f, fh = _build_forecaster()
        f.fit(
            y=d["y_train"],
            forecasting_horizon=fh,
            X_forecast=d["X_forecast_train"],
        )

        y_pred = f.predict()
        assert len(y_pred) == H
        assert len(f._step_column_names_) > 0
        assert f._X_future_raw_ is None
        assert f._X_forecast_raw_ is not None

    def test_fit_with_X_actual_and_X_future(self, electricity_data):
        """Fit with X_actual + X_future (no X_forecast)."""
        d = electricity_data
        f, fh = _build_forecaster()
        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
        )

        y_pred = f.predict()
        assert len(y_pred) == H


@pytest.mark.integration
class TestEdgeCases:
    """Edge cases for the exogenous forecast API."""

    def test_step_columns_in_local_X_t_schema(self, electricity_data):
        """Step columns are registered in local_X_t_schema_ after fit."""
        d = electricity_data
        f, fh = _build_forecaster()
        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
            X_forecast=d["X_forecast_train"],
        )

        schema_cols = set(f.local_X_t_schema_.keys())
        # Step columns must be a subset of the schema
        assert f._step_column_names_.issubset(schema_cols), f"Missing step cols: {f._step_column_names_ - schema_cols}"

    def test_recursive_predict_raises_with_X_forecast(self, electricity_data):
        """Recursive predict raises ValueError when X_forecast was used at fit.

        Covers the raise side of the guard in _recursive_predict
        (BasePointForecaster, "forecasting_horizon > fit_forecasting_horizon_
        AND _X_forecast_raw_ is not None"). The no-raise side is covered by
        test_recursive_predict_allowed_with_X_future_only below.
        """
        d = electricity_data
        f, fh = _build_forecaster()
        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
            X_forecast=d["X_forecast_train"],
        )

        with pytest.raises(ValueError, match="Recursive prediction"):
            f.predict(forecasting_horizon=fh + 1)

    def test_recursive_predict_allowed_with_X_future_only(self, electricity_data):
        """Recursive predict works when only X_future was used (no X_forecast)."""
        d = electricity_data
        _, fh = _build_forecaster()
        # Build a forecaster without X_actual (recursive observe cannot
        # supply X_actual, so the feature_transformer must not depend on
        # external actual features).
        f = PointReductionForecaster(
            estimator=HistGradientBoostingRegressor(random_state=SEED),
            feature_transformer=LagTransformer([1, 2, 3]),
            reduction_strategy="direct",
        )
        f.fit(
            y=d["y_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
        )

        # Should NOT raise
        result = f.predict(forecasting_horizon=fh + 1)
        assert len(result) == fh + 1


@pytest.mark.integration
class TestEdgeCasesExtended:
    """Additional edge cases for exogenous forecast API."""

    def test_target_as_feature_none_with_X_future_only(self, electricity_data):
        """target_as_feature=None with X_actual and X_future produces step columns alongside actual features."""
        d = electricity_data
        f = PointReductionForecaster(
            estimator=HistGradientBoostingRegressor(random_state=SEED, max_iter=50),
            target_as_feature=None,
            reduction_strategy="direct",
        )
        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=H,
            X_future=d["X_future_full"],
        )

        y_pred = f.predict()
        assert len(y_pred) == H
        assert len(f._step_column_names_) > 0

    def test_predict_X_future_on_unfitted_raises(self, electricity_data):
        """predict(X_future=...) on unfitted forecaster raises NotFittedError."""
        d = electricity_data
        f, _ = _build_forecaster()
        with pytest.raises(NotFittedError):
            f.predict(X_future=d["X_future_full"])

    def test_predict_X_forecast_on_unfitted_raises(self, electricity_data):
        """predict(X_forecast=...) on unfitted forecaster raises NotFittedError."""
        d = electricity_data
        f, _ = _build_forecaster()
        with pytest.raises(NotFittedError):
            f.predict(X_forecast=d["X_forecast_test_accurate"])

    def test_predict_partial_override_X_future_only(self, electricity_data):
        """Fitted with both X_future+X_forecast, predict with only X_future override uses stored X_forecast."""
        d = electricity_data
        f, fh = _build_forecaster()
        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
            X_forecast=d["X_forecast_train"],
        )

        pred_default = f.predict()

        # Override only X_future (all holidays)
        all_holiday = d["X_future_full"].with_columns(pl.lit(1.0).alias("is_holiday"))
        pred_override = f.predict(X_future=all_holiday)

        # Predictions differ (X_future changed, X_forecast unchanged)
        assert not np.allclose(
            pred_default["price"].to_numpy(),
            pred_override["price"].to_numpy(),
            atol=0.01,
        )
        # Stored X_forecast_raw_ unchanged
        assert f._X_forecast_raw_ is not None


@pytest.mark.integration
class TestMismatchedVintageOverride:
    """Tests for X_forecast overrides whose vintage_time differs from observed_time_.

    The existing TestFitPredictWithExogenous.test_multi_vintage_predictions_differ
    uses vintage_time == observed_time_ (last training timestamp). These tests
    advance observed_time_ via observe() so that the override vintage_time no
    longer matches, exercising the remap path in _predict_with_step_override.
    """

    def _fit_and_observe(self, electricity_data):
        """Fit a forecaster and observe a few test rows to shift observed_time_."""
        d = electricity_data
        f, fh = _build_forecaster()
        f.fit(
            y=d["y_train"],
            X_actual=d["X_actual_train"],
            forecasting_horizon=fh,
            X_future=d["X_future_full"],
            X_forecast=d["X_forecast_train"],
        )

        # Advance observed_time_ past the last training timestamp
        n_obs = 3
        f.observe(
            y=d["y_test"][:n_obs],
            X_actual=d["X_actual_test"][:n_obs],
        )
        return f, fh, d

    def _make_mismatched_vintages(self, d, fh):
        """Build two single-vintage X_forecast overrides whose vintage_time
        differs from the forecaster's observed_time_ (uses the last training
        timestamp as vintage_time, while observed_time_ has been advanced).
        """
        last_train = d["y_train"]["time"].max()
        test_times = d["all_times"][N_TRAIN : N_TRAIN + fh]

        X_a = _make_weather_forecast(
            pl.Series("vintage_time", [last_train]),
            test_times,
            bias=0.1,
            rng=np.random.default_rng(500),
        )
        X_b = _make_weather_forecast(
            pl.Series("vintage_time", [last_train]),
            test_times,
            bias=5.0,
            rng=np.random.default_rng(600),
        )
        return X_a, X_b

    def test_predict_with_mismatched_vintage_time_produces_different_results(self, electricity_data):
        """Two X_forecast overrides with different weather values but
        vintage_time != observed_time_ must produce different predictions."""
        f, fh, d = self._fit_and_observe(electricity_data)
        X_a, X_b = self._make_mismatched_vintages(d, fh)

        # Confirm vintage_time differs from observed_time_
        assert X_a["vintage_time"][0] != f.observed_time_

        pred_a = f.predict(X_forecast=X_a)
        pred_b = f.predict(X_forecast=X_b)

        assert not np.allclose(
            pred_a["price"].to_numpy(),
            pred_b["price"].to_numpy(),
            atol=0.01,
        ), "Predictions with different X_forecast overrides must differ even when vintage_time != observed_time_"

    def test_predict_with_mismatched_vintage_restores_state(self, electricity_data):
        """predict(X_forecast=X_v) with vintage_time != observed_time_ must
        restore _X_t_observed step columns to their original values."""
        f, fh, d = self._fit_and_observe(electricity_data)
        X_a, _ = self._make_mismatched_vintages(d, fh)

        step_cols = sorted(f._step_column_names_)

        # Snapshot step columns before override predict
        step_before = f._X_t_observed.select([c for c in step_cols if c in f._X_t_observed.columns])

        _ = f.predict(X_forecast=X_a)

        # Step columns must be identical after predict returns
        step_after = f._X_t_observed.select([c for c in step_cols if c in f._X_t_observed.columns])

        assert step_before.equals(step_after), (
            "Step columns in _X_t_observed were not restored after predict(X_forecast=...) returned"
        )


@pytest.mark.integration
class TestPanelExogenousDispatch:
    """Panel-path coverage for the step-column observe/predict dispatch.

    The other tests exercise only non-panel PointReductionForecaster. This one
    feeds group__column panel data with a global X_future so the panel branches
    of step-column injection and observe/predict are exercised end to end.
    """

    def _make_panel(self):
        n = N_TRAIN + N_TEST
        times = _make_times(n)
        t = np.arange(n, dtype=float)
        rng = np.random.default_rng(SEED)
        y = pl.DataFrame({
            "time": times,
            "store_1__price": 50.0 + 2.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, n),
            "store_2__price": 30.0 + 3.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, n),
        })
        # Global (non-grouped) future feature shared by all groups.
        x_future = _make_holiday(times)
        return y, x_future

    def test_panel_observe_predict_with_X_future(self):
        """fit -> observe -> predict on panel data with a global X_future.

        Exercises the panel branch of step-column injection and observe(): the
        forecast must keep both groups, carry the time-column contract, and
        shift forward after observing new ground truth.
        """
        from yohou.utils.panel import inspect_panel

        y, x_future = self._make_panel()
        f = PointReductionForecaster(
            estimator=HistGradientBoostingRegressor(max_iter=30, max_depth=3, random_state=SEED),
            feature_transformer=LagTransformer([1, 2, 3]),
            reduction_strategy="direct",
        )

        y_train = y[:N_TRAIN]
        f.fit(y_train, forecasting_horizon=H, X_future=x_future)
        assert f.groups_ == ["store_1", "store_2"]
        assert len(f._step_column_names_) > 0

        pred_before = f.predict()
        assert "vintage_time" in pred_before.columns
        assert "time" in pred_before.columns
        _, panel_groups = inspect_panel(pred_before)
        assert set(panel_groups.keys()) == {"store_1", "store_2"}

        # Observe newly arrived ground truth (panel observe branch), then predict.
        f.observe(y[N_TRAIN : N_TRAIN + H], X_future=x_future)
        pred_after = f.predict()

        assert len(pred_after) == H
        assert pred_after["vintage_time"][0] > pred_before["vintage_time"][0]
        assert pred_after["time"].min() > pred_before["time"].min()


@pytest.mark.integration
class TestSparseVintageSchedule:
    """Integration test: 6h vintage schedule with hourly observations."""

    def test_fit_observe_predict_with_sparse_vintages(self):
        """Fit with 6h vintages, observe hourly, predict with non-null step columns."""
        rng = np.random.default_rng(SEED)
        n_train = 200
        n_test = 12
        h = 6
        vintage_interval_hours = 6
        n_total = n_train + n_test

        times = pl.Series("time", [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n_total)])
        t = np.arange(n_total, dtype=float)
        y_vals = 50.0 + 2.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, n_total)
        y = pl.DataFrame({"time": times, "price": y_vals})

        # Build X_forecast with 6h vintage schedule (vintages at hours 0, 6, 12, 18, ...)
        forecast_rows = []
        for i in range(0, n_total, vintage_interval_hours):
            vt = times[i]
            for step in range(1, h + 1):
                target_idx = i + step
                if target_idx < n_total:
                    forecast_rows.append({
                        "vintage_time": vt,
                        "time": times[target_idx],
                        "wx_temp": float(np.sin(2 * np.pi * target_idx / 24) + rng.normal(0, 0.3)),
                    })
        x_forecast = pl.DataFrame(forecast_rows)

        y_train = y[:n_train]
        x_fc_train = x_forecast.filter(pl.col("vintage_time").is_in(times[:n_train]))

        f = PointReductionForecaster(
            estimator=HistGradientBoostingRegressor(max_iter=50, max_depth=3, random_state=SEED),
            feature_transformer=LagTransformer([1, 2, 3]),
            reduction_strategy="direct",
        )
        f.fit(y=y_train, forecasting_horizon=h, X_forecast=x_fc_train)

        assert len(f._step_column_names_) > 0
        assert f._X_forecast_raw_ is not None

        # Observe hourly for the test period (sparse vintages mean as-of matching)
        obs_time = times[n_train]
        y_obs = pl.DataFrame({"time": [obs_time], "price": [y_vals[n_train]]})
        x_fc_obs = x_forecast.filter(
            pl.col("vintage_time").is_in(times[n_train - vintage_interval_hours : n_train + 1])
        )
        f.observe(y=y_obs, X_forecast=x_fc_obs)

        # Step columns should be non-null (as-of should find a vintage)
        step_cols_in_X = [c for c in f._X_t_observed.columns if "_step_" in c]
        assert len(step_cols_in_X) > 0

        pred = f.predict()
        assert len(pred) == h
        assert "vintage_time" in pred.columns
        assert "time" in pred.columns
        assert "price" in pred.columns
