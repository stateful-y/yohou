"""Tests for the X_future rank-deficiency diagnostic.

A clock feature routed through X_future expands into H step columns that carry
no information its value at the observation point did not already carry. The
warning under test reports that; these tests pin both that it fires when it
should and, more importantly, that it stays quiet when it should.
"""

import contextlib
import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from yohou.point import PointReductionForecaster

H = 12


@pytest.fixture
def y_series():
    """A year of daily observations with a weekly cycle."""
    time = pl.datetime_range(datetime(2021, 1, 1), datetime(2021, 12, 31), interval="1d", eager=True)
    t = np.arange(len(time), dtype=float)
    return pl.DataFrame({"time": time, "value": np.sin(2 * np.pi * t / 7) + 0.01 * t})


@pytest.fixture
def future_time():
    """Timestamps covering the observations plus room beyond every target."""
    return pl.datetime_range(datetime(2021, 1, 1), datetime(2022, 6, 30), interval="1d", eager=True)


@pytest.fixture
def fourier_future(future_time):
    """A clock feature: its value at T determines its value at every T+h."""
    t = np.arange(len(future_time), dtype=float)
    return pl.DataFrame({
        "time": future_time,
        "f_sin": np.sin(2 * np.pi * t / 7),
        "f_cos": np.cos(2 * np.pi * t / 7),
    })


@pytest.fixture
def event_future(future_time):
    """An event calendar: not derivable from the timestamp."""
    rng = np.random.default_rng(0)
    return pl.DataFrame({
        "time": future_time,
        "promo": rng.integers(0, 2, len(future_time)).astype(float),
    })


def _rank_warnings(record):
    """Select the rank-deficiency warnings out of a warning record."""
    return [w for w in record if "expands to" in str(w.message)]


@contextlib.contextmanager
def _captured_warnings():
    """Record every warning without requiring one.

    ``pytest.warns`` asserts that at least one warning is raised, so it cannot
    express "this must not warn" when nothing warns at all.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        yield record


class TestRankDeficiencyWarning:
    """The diagnostic fires on clock features and only on clock features."""

    def test_fourier_payload_warns_with_measurements(self, y_series, fourier_future):
        """A Fourier payload warns per column, reporting step count and rank."""
        with pytest.warns(UserWarning) as record:
            PointReductionForecaster(estimator=LinearRegression()).fit(
                y=y_series, forecasting_horizon=H, X_future=fourier_future
            )
        messages = [str(w.message) for w in _rank_warnings(record)]
        assert len(messages) == 2
        assert any("'f_sin'" in m for m in messages)
        assert any("'f_cos'" in m for m in messages)
        assert all(f"expands to {H} step columns" in m for m in messages)
        assert all("rank of only 2" in m for m in messages)
        assert all("actual_transformer" in m for m in messages)

    def test_event_calendar_does_not_warn(self, y_series, event_future):
        """An irregular event calendar is what the channel is for."""
        with _captured_warnings() as record:
            PointReductionForecaster(estimator=LinearRegression()).fit(
                y=y_series, forecasting_horizon=H, X_future=event_future
            )
        assert _rank_warnings(record) == []

    def test_undercovering_event_calendar_does_not_warn(self, y_series, future_time):
        """Null step columns must not be mistaken for a clock feature.

        An X_future that stops short of the last required target produces null
        step values. Those rows are excluded before ranking; without that, an
        under-covering event calendar would be reported as clock-derived, which
        is the exact opposite of the truth.
        """
        short_time = future_time.filter(future_time <= datetime(2021, 12, 25))
        rng = np.random.default_rng(1)
        short_future = pl.DataFrame({
            "time": short_time,
            "promo": rng.integers(0, 2, len(short_time)).astype(float),
        })
        with _captured_warnings() as record:
            PointReductionForecaster(estimator=LinearRegression(), nan_handling="drop").fit(
                y=y_series, forecasting_horizon=H, X_future=short_future
            )
        assert _rank_warnings(record) == []

    def test_short_frame_skips_the_check(self, fourier_future):
        """Too few rows to rank means no claim is made."""
        time = pl.datetime_range(datetime(2021, 1, 1), datetime(2021, 1, 20), interval="1d", eager=True)
        t = np.arange(len(time), dtype=float)
        y_short = pl.DataFrame({"time": time, "value": np.sin(2 * np.pi * t / 7)})
        with _captured_warnings() as record:
            PointReductionForecaster(estimator=LinearRegression()).fit(
                y=y_short, forecasting_horizon=H, X_future=fourier_future
            )
        assert _rank_warnings(record) == []


class TestDiagnosticFiresOnceAtFit:
    """The condition is a property of routing, so it is said once."""

    def test_no_repeat_across_observe_predict_and_predict(self, y_series, fourier_future):
        """Walk-forward must not repeat the warning once per stride."""
        with pytest.warns(UserWarning):
            forecaster = PointReductionForecaster(estimator=LinearRegression())
            forecaster.fit(y=y_series, forecasting_horizon=H, X_future=fourier_future)

        new_time = pl.datetime_range(datetime(2022, 1, 1), datetime(2022, 2, 15), interval="1d", eager=True)
        y_new = pl.DataFrame({
            "time": new_time,
            "value": np.sin(2 * np.pi * np.arange(len(new_time)) / 7),
        })
        with _captured_warnings() as record:
            forecaster.observe_predict(y=y_new, X_future=fourier_future, stride=H)
            forecaster.predict(X_future=fourier_future)
        assert _rank_warnings(record) == []


class TestWarningsPointAtTheCaller:
    """Step-column warnings blame the user's call, not library internals.

    Step columns are derived from fit, observe, and predict, whose chains reach
    the warn site at different depths, and fit adds a frame for scikit-learn's
    _fit_context decorator. A constant stacklevel cannot serve all three: the
    X_forecast coverage warning shipped with ``stacklevel=5``, which was right
    for observe, pointed at ``sklearn/base.py`` from fit, and overshot the
    caller from predict. These tests pin the measured behaviour so the numbers
    cannot silently drift again.
    """

    def test_rank_warning_points_at_the_caller(self, y_series, fourier_future):
        """The rank diagnostic blames this test, not yohou or sklearn."""
        with pytest.warns(UserWarning) as record:
            PointReductionForecaster(estimator=LinearRegression()).fit(
                y=y_series, forecasting_horizon=H, X_future=fourier_future
            )
        warning = _rank_warnings(record)[0]
        assert warning.filename == __file__

    @pytest.mark.parametrize("phase", ["fit", "observe", "predict"])
    def test_x_forecast_coverage_warning_points_at_the_caller(self, phase):
        """The X_forecast coverage warning blames the caller from every phase."""
        n = 60
        time = pl.datetime_range(datetime(2021, 1, 1), datetime(2021, 3, 1), interval="1d", eager=True)[:n]
        y = pl.DataFrame({"time": time, "value": np.arange(len(time), dtype=float)})
        # Cover only 2 of 4 steps, so the under-coverage warning fires.
        forecast = pl.DataFrame([
            {"vintage_time": vintage, "time": vintage + timedelta(days=step), "wx": float(i + step)}
            for i, vintage in enumerate(time)
            for step in (1, 2)
        ])
        # The warning needs the last step column to be entirely null, which
        # nan_handling="drop" would strip every row for. DummyRegressor ignores
        # X, so the real fit/observe/predict chains run without an estimator
        # that has an opinion about NaN.
        forecaster = PointReductionForecaster(estimator=DummyRegressor(), nan_handling="pass")

        with _captured_warnings():
            forecaster.fit(y=y, forecasting_horizon=4, X_forecast=forecast)

        later = pl.datetime_range(datetime(2021, 3, 2), datetime(2021, 3, 11), interval="1d", eager=True)
        y_later = pl.DataFrame({"time": later, "value": np.arange(n, n + len(later), dtype=float)})
        calls = {
            "fit": lambda: PointReductionForecaster(estimator=DummyRegressor(), nan_handling="pass").fit(
                y=y, forecasting_horizon=4, X_forecast=forecast
            ),
            "observe": lambda: forecaster.observe(y=y_later, X_forecast=forecast),
            "predict": lambda: forecaster.predict(X_forecast=forecast),
        }
        with _captured_warnings() as record:
            calls[phase]()
        # Fit reports coverage per column ("X_forecast column '...'"); observe and
        # predict use the per-call warning ("X_forecast covers ..."). Both must
        # blame the caller.
        coverage = [w for w in record if "X_forecast covers" in str(w.message) or "X_forecast column" in str(w.message)]
        assert coverage, f"expected a coverage warning from {phase}"
        assert coverage[0].filename == __file__


class TestDiagnosticDoesNotMutate:
    """Report, never repair."""

    def test_step_columns_survive_the_warning(self, y_series, fourier_future):
        """Every generated step column is still present after the warning."""
        with pytest.warns(UserWarning):
            forecaster = PointReductionForecaster(estimator=LinearRegression())
            forecaster.fit(y=y_series, forecasting_horizon=H, X_future=fourier_future)
        assert len(forecaster._step_column_names_) == 2 * H
        for col in ("f_sin", "f_cos"):
            for step in range(1, H + 1):
                assert f"{col}_step_{step}" in forecaster._step_column_names_


def _zero_coverage_warnings(record):
    """Select the serve-path dead-channel warnings out of a warning record."""
    return [w for w in record if "X_forecast covers 0 of" in str(w.message)]


def _partial_coverage_warnings(record):
    """Select the serve-path partial-coverage warnings, excluding dead-channel ones."""
    return [
        w for w in record if "X_forecast covers" in str(w.message) and "X_forecast covers 0 of" not in str(w.message)
    ]


def _fit_coverage_warnings(record):
    """Select the fit-path per-column coverage warnings out of a warning record."""
    return [w for w in record if "X_forecast column" in str(w.message)]


class TestForecastCoverageDiagnostic:
    """A dead forecast channel reads differently from a short-range one.

    Zero coverage is not the extreme of partial coverage. Under partial coverage
    some step features carry values; at zero every one is null and a model fitted
    on them is predicting without them. Sharing a message would train readers to
    ignore both.
    """

    HORIZON = 3

    @staticmethod
    def _y(n: int = 40) -> pl.DataFrame:
        time = pl.datetime_range(
            datetime(2021, 1, 1), datetime(2021, 1, 1) + timedelta(days=n - 1), interval="1d", eager=True
        )
        return pl.DataFrame({"time": time, "value": np.arange(n, dtype=float)})

    @classmethod
    def _forecast(cls, n_vintages: int, span: int | None = None) -> pl.DataFrame:
        """Vintages issued daily, each carrying ``span`` steps of its own future."""
        span = cls.HORIZON if span is None else span
        rows = []
        for i in range(n_vintages):
            vintage = datetime(2021, 1, 1) + timedelta(days=i)
            for step in range(1, span + 1):
                rows.append({
                    "vintage_time": vintage,
                    "time": vintage + timedelta(days=step),
                    "temp": float(i + step),
                })
        return pl.DataFrame(rows)

    def _fitted(self, n_vintages: int, span: int | None = None):
        forecaster = PointReductionForecaster(estimator=DummyRegressor(), nan_handling="pass")
        with _captured_warnings():
            forecaster.fit(y=self._y(), forecasting_horizon=self.HORIZON, X_forecast=self._forecast(n_vintages, span))
        return forecaster

    def test_exhausted_cache_reports_a_dead_channel(self):
        """Driven the way a caller reaches it: serving past the vintages held.

        No ``X_forecast`` is passed at observe, so the forecaster falls back to its
        cached frame. Once the observation point is a full horizon past the newest
        cached vintage, as-of selection resolves to a vintage that covers nothing.
        """
        forecaster = self._fitted(n_vintages=20)

        later = pl.DataFrame({
            "time": pl.datetime_range(
                datetime(2021, 1, 21) + timedelta(days=self.HORIZON),
                datetime(2021, 1, 21) + timedelta(days=self.HORIZON + 1),
                interval="1d",
                eager=True,
            ),
            "value": np.arange(2, dtype=float),
        })
        with _captured_warnings() as record:
            forecaster.observe(y=later)

        assert _zero_coverage_warnings(record), "expected the dead-channel warning"
        assert not _partial_coverage_warnings(record)

    def test_partial_coverage_reports_the_column_at_fit(self):
        """A vintage carrying fewer steps than the horizon is short-range, not dead.

        Every observation is covered to the same depth (< H), so the fit-path
        per-column diagnostic names the column and reports its worst depth, and
        the serve-path zero/partial warning stays silent at fit.
        """
        forecaster = PointReductionForecaster(estimator=DummyRegressor(), nan_handling="pass")
        with _captured_warnings() as record:
            forecaster.fit(y=self._y(), forecasting_horizon=self.HORIZON, X_forecast=self._forecast(40, span=2))

        fit_warnings = _fit_coverage_warnings(record)
        assert fit_warnings, "expected the fit-path per-column warning"
        assert any(f"covers 2 of {self.HORIZON}" in str(w.message) and "'temp'" in str(w.message) for w in fit_warnings)
        assert not _zero_coverage_warnings(record)
        assert not _partial_coverage_warnings(record)

    def test_full_coverage_is_silent(self):
        """No message fires when every observation is fully covered.

        The forecast archive spans every observation time (one vintage per
        observation, each carrying the full horizon), so no observation is
        under-covered and neither diagnostic fires.
        """
        forecaster = PointReductionForecaster(estimator=DummyRegressor(), nan_handling="pass")
        with _captured_warnings() as record:
            forecaster.fit(y=self._y(), forecasting_horizon=self.HORIZON, X_forecast=self._forecast(40))

        assert not _fit_coverage_warnings(record)
        assert not _zero_coverage_warnings(record)
        assert not _partial_coverage_warnings(record)

    @pytest.mark.parametrize("span", [2, 3])
    def test_no_coverage_message_recommends_an_estimator_or_asserts_normality(self, span):
        """The message reports a measurement; it does not prescribe a remedy.

        A null-tolerant estimator makes a dead channel survivable rather than
        correct, so naming one directs a reader from a visible failure toward an
        invisible one. And the same measurement has several causes, only some of
        which are routine, so the warn site cannot call it normal.
        """
        forecaster = PointReductionForecaster(estimator=DummyRegressor(), nan_handling="pass")
        with _captured_warnings() as record:
            forecaster.fit(y=self._y(), forecasting_horizon=self.HORIZON, X_forecast=self._forecast(20, span=span))
            later = pl.DataFrame({
                "time": pl.datetime_range(datetime(2021, 2, 1), datetime(2021, 2, 2), interval="1d", eager=True),
                "value": np.arange(2, dtype=float),
            })
            forecaster.observe(y=later)

        messages = [str(w.message) for w in record if "X_forecast covers" in str(w.message)]
        assert messages, "expected at least one coverage warning to inspect"
        for message in messages:
            for estimator in ("XGBoost", "LightGBM", "HistGradientBoosting", "estimator"):
                assert estimator not in message, f"message recommends {estimator}: {message}"
            assert "is normal" not in message, f"message asserts normality: {message}"

    @pytest.mark.parametrize("age", [1, 2, 3, 4])
    def test_coverage_falls_one_step_per_interval_of_age(self, age):
        """The bound this change relies on: coverage reaches zero at age == horizon.

        A vintage carries the ``forecasting_horizon`` timestamps after its own
        ``vintage_time``, so an observation ``age`` intervals later retains
        ``horizon - age`` of them. That is why no staleness parameter is needed.
        """
        forecaster = self._fitted(n_vintages=20)
        newest_vintage = datetime(2021, 1, 1) + timedelta(days=19)

        later = pl.DataFrame({
            "time": pl.datetime_range(
                newest_vintage + timedelta(days=age),
                newest_vintage + timedelta(days=age),
                interval="1d",
                eager=True,
            ),
            "value": np.zeros(1),
        })
        with _captured_warnings() as record:
            forecaster.observe(y=later)

        expected_covered = max(self.HORIZON - age, 0)
        if expected_covered == 0:
            assert _zero_coverage_warnings(record), f"age={age} should exhaust coverage"
        else:
            partial = _partial_coverage_warnings(record)
            assert partial, f"age={age} should report partial coverage"
            assert f"covers {expected_covered} of {self.HORIZON}" in str(partial[0].message)


class TestFitCoverageIsPerObservationPerColumn:
    """Fit measures coverage per observation, not existentially across the batch.

    The batch-wide test it replaces answered "was this column ever covered?", so a
    channel null in all but one training row read as fully covered and nothing
    warned. The fit-path diagnostic names each under-covered column and reports how
    many observations it fails, so a channel dead for most of the batch is caught.
    """

    HORIZON = 3
    N = 60

    def _y(self) -> pl.DataFrame:
        time = pl.datetime_range(
            datetime(2021, 1, 1), datetime(2021, 1, 1) + timedelta(days=self.N - 1), interval="1d", eager=True
        )
        return pl.DataFrame({"time": time, "value": np.arange(self.N, dtype=float)})

    def _two_channel_forecast(self) -> pl.DataFrame:
        """``fast`` issued at every vintage; ``slow`` only at the first.

        ``slow`` covers only observations near the first vintage and is null for
        the rest, which is the batch the old existential test could not see.
        """
        rows = []
        for i in range(self.N):
            vintage = datetime(2021, 1, 1) + timedelta(days=i)
            for step in range(1, self.HORIZON + 1):
                rows.append({
                    "vintage_time": vintage,
                    "time": vintage + timedelta(days=step),
                    "fast": float(i + step),
                    "slow": float(step) if i == 0 else None,
                })
        return pl.DataFrame(rows)

    def test_channel_dead_for_most_of_the_batch_is_named_at_fit(self):
        """The reproduction as a test: ``slow`` is named, ``fast`` is not."""
        forecaster = PointReductionForecaster(estimator=DummyRegressor(), nan_handling="pass")
        with _captured_warnings() as record:
            forecaster.fit(y=self._y(), forecasting_horizon=self.HORIZON, X_forecast=self._two_channel_forecast())

        fit_warnings = [str(w.message) for w in _fit_coverage_warnings(record)]
        assert any("'slow'" in m for m in fit_warnings), "expected 'slow' to be named at fit"
        assert not any("'fast'" in m for m in fit_warnings), "'fast' is fully covered and must not warn"
        # The message reports how many observations the channel fails to cover.
        slow = next(m for m in fit_warnings if "'slow'" in m)
        assert f"of {self.N} training observations" in slow

    def test_no_serve_warning_fires_at_fit(self):
        """The per-call zero/partial warning is suppressed on the fit path."""
        forecaster = PointReductionForecaster(estimator=DummyRegressor(), nan_handling="pass")
        with _captured_warnings() as record:
            forecaster.fit(y=self._y(), forecasting_horizon=self.HORIZON, X_forecast=self._two_channel_forecast())
        assert not _zero_coverage_warnings(record)
        assert not _partial_coverage_warnings(record)

    def test_fit_diagnostic_does_not_repeat_in_walk_forward(self):
        """The fit-path warning is said once; walk-forward uses the serve path.

        The condition the fit diagnostic reports is a property of the training
        assembly, so it must not re-fire per stride. The serve-path warning still
        fires when a stride resolves a dead channel, which is a runtime condition.
        """
        forecaster = PointReductionForecaster(estimator=DummyRegressor(), nan_handling="pass")
        with _captured_warnings():
            forecaster.fit(y=self._y(), forecasting_horizon=self.HORIZON, X_forecast=self._two_channel_forecast())

        later = pl.DataFrame({
            "time": pl.datetime_range(
                datetime(2021, 1, 1) + timedelta(days=self.N),
                datetime(2021, 1, 1) + timedelta(days=self.N + self.HORIZON),
                interval="1d",
                eager=True,
            ),
            "value": np.arange(self.HORIZON + 1, dtype=float),
        })
        with _captured_warnings() as record:
            forecaster.observe_predict(y=later, stride=1)

        assert not _fit_coverage_warnings(record), "the fit diagnostic must not repeat during walk-forward"


class TestCoverageMeasurementDoesNotAlterDerivation:
    """Changing how coverage is measured changes what is reported, never derived."""

    def test_warn_flag_does_not_change_derived_step_columns(self):
        """``warn_coverage`` gates the warning only; the frame is identical."""
        from yohou.base.utils import _derive_step_columns

        rows = []
        for i in range(20):
            vintage = datetime(2021, 1, 1) + timedelta(days=i)
            for step in range(1, 3):  # span 2 of horizon 3: guarantees under-coverage
                rows.append({
                    "vintage_time": vintage,
                    "time": vintage + timedelta(days=step),
                    "wx": float(i + step),
                })
        forecast = pl.DataFrame(rows)
        obs = pl.Series([datetime(2021, 1, 1) + timedelta(days=i) for i in range(20)])

        with _captured_warnings():
            warned = _derive_step_columns(None, forecast, obs, 3, "1d", warn_coverage=True)
            silent = _derive_step_columns(None, forecast, obs, 3, "1d", warn_coverage=False)

        assert warned is not None and silent is not None
        assert warned.equals(silent)
