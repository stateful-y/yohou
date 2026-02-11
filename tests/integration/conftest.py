"""Shared fixtures for analytical integration tests.

This module provides analytical data generators that produce time series with
known closed-form solutions for forecasting. These fixtures enable verification
of exact numerical correctness rather than just crash-testing.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def constant_series():
    """Generate constant time series: y = c for all t.

    Returns
    -------
    callable
        Function: constant_series(value, length, freq="1d") -> pl.DataFrame

    """

    def _generator(value: float, length: int, freq: str = "1d") -> pl.DataFrame:
        start = datetime(2020, 1, 1)
        time = [start + timedelta(days=i) for i in range(length)]
        return pl.DataFrame({
            "time": time,
            "value": [value] * length,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    return _generator


@pytest.fixture
def linear_series():
    """Generate linear time series: y = m*t + b.

    Returns
    -------
    callable
        Function: linear_series(slope, intercept, length, freq="1d") -> pl.DataFrame

    """

    def _generator(slope: float, intercept: float, length: int, freq: str = "1d") -> pl.DataFrame:
        start = datetime(2020, 1, 1)
        time = [start + timedelta(days=i) for i in range(length)]
        t = np.arange(length)
        values = slope * t + intercept
        return pl.DataFrame({
            "time": time,
            "value": values,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    return _generator


@pytest.fixture
def ar1_series():
    """Generate AR(1) process: y_t = phi * y_{t-1} + c + epsilon.

    Parameters
    ----------
    phi : float
        Autoregressive coefficient (stability requires |phi| < 1).
    c : float
        Constant term.
    length : int
        Number of time steps.
    seed : int, default=42
        Random seed for reproducibility.
    noise_std : float, default=0.0
        Standard deviation of Gaussian noise.

    Returns
    -------
    pl.DataFrame
        Time series with analytical AR(1) structure.

    Notes
    -----
    Closed-form k-step forecast:
    y_{t+k} = phi^k * y_t + c * (1 - phi^k) / (1 - phi)

    """

    def _generator(
        phi: float,
        c: float,
        length: int,
        seed: int = 42,
        noise_std: float = 0.0,
        freq: str = "1d",
    ) -> pl.DataFrame:
        np.random.seed(seed)
        start = datetime(2020, 1, 1)
        time = [start + timedelta(days=i) for i in range(length)]

        # Initialize at stationary mean if |phi| < 1, else at 0
        y = np.zeros(length)
        if abs(phi) < 1:
            y[0] = c / (1 - phi)
        else:
            y[0] = 0

        # Generate AR(1) process
        for t in range(1, length):
            epsilon = np.random.normal(0, noise_std) if noise_std > 0 else 0
            y[t] = phi * y[t - 1] + c + epsilon

        return pl.DataFrame({
            "time": time,
            "value": y,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    return _generator


@pytest.fixture
def ar_p_series():
    """Generate AR(p) process: y_t = sum(phi_i * y_{t-i}) + c + epsilon.

    Parameters
    ----------
    coeffs : list of float
        AR coefficients [phi_1, phi_2, ..., phi_p].
    c : float
        Constant term.
    length : int
        Number of time steps.
    seed : int, default=42
        Random seed.
    noise_std : float, default=0.0
        Standard deviation of noise.

    Returns
    -------
    pl.DataFrame
        Time series with AR(p) structure.

    Examples
    --------
    AR(2): y_t = 0.5*y_{t-1} + 0.3*y_{t-2} + 2.0

    """

    def _generator(
        coeffs: list[float],
        c: float,
        length: int,
        seed: int = 42,
        noise_std: float = 0.0,
        freq: str = "1d",
    ) -> pl.DataFrame:
        np.random.seed(seed)
        start = datetime(2020, 1, 1)
        time = [start + timedelta(days=i) for i in range(length)]

        p = len(coeffs)
        y = np.zeros(length)

        # Initialize first p values
        for i in range(p):
            y[i] = c

        # Generate AR(p) process
        for t in range(p, length):
            y_t = c
            for i, phi in enumerate(coeffs):
                y_t += phi * y[t - i - 1]
            epsilon = np.random.normal(0, noise_std) if noise_std > 0 else 0
            y[t] = y_t + epsilon

        return pl.DataFrame({
            "time": time,
            "value": y,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    return _generator


@pytest.fixture
def seasonal_series():
    """Generate seasonal time series with Fourier components.

    Parameters
    ----------
    period : int
        Seasonality period.
    amplitudes : list of float
        Amplitudes for each harmonic [A_1, A_2, ...].
    length : int
        Number of time steps.

    Returns
    -------
    pl.DataFrame
        y = sum(A_k * sin(2*pi*k*t / period))

    """

    def _generator(
        period: int,
        amplitudes: list[float],
        length: int,
        freq: str = "1d",
    ) -> pl.DataFrame:
        start = datetime(2020, 1, 1)
        time = [start + timedelta(days=i) for i in range(length)]
        t = np.arange(length)

        y = np.zeros(length)
        for k, A in enumerate(amplitudes, start=1):
            y += A * np.sin(2 * np.pi * k * t / period)

        return pl.DataFrame({
            "time": time,
            "value": y,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    return _generator


@pytest.fixture
def trend_plus_seasonal():
    """Generate trend + seasonal: y = m*t + b + A*sin(2*pi*t/P).

    Returns
    -------
    callable
        Function with parameters (slope, intercept, period, amplitude, length).

    """

    def _generator(
        slope: float,
        intercept: float,
        period: int,
        amplitude: float,
        length: int,
        freq: str = "1d",
    ) -> pl.DataFrame:
        start = datetime(2020, 1, 1)
        time = [start + timedelta(days=i) for i in range(length)]
        t = np.arange(length)

        trend = slope * t + intercept
        seasonal = amplitude * np.sin(2 * np.pi * t / period)
        y = trend + seasonal

        return pl.DataFrame({
            "time": time,
            "value": y,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    return _generator


@pytest.fixture
def exponential_series():
    """Generate exponential series: y = exp(a*t + b).

    Useful for testing LogTransformer with known inverse.

    """

    def _generator(
        a: float,
        b: float,
        length: int,
        freq: str = "1d",
    ) -> pl.DataFrame:
        start = datetime(2020, 1, 1)
        time = [start + timedelta(days=i) for i in range(length)]
        t = np.arange(length)

        y = np.exp(a * t + b)

        return pl.DataFrame({
            "time": time,
            "value": y,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    return _generator


@pytest.fixture
def panel_analytical():
    """Generate panel data where each group follows a different analytical pattern.

    Parameters
    ----------
    generator : callable
        One of the analytical generators (linear_series, ar1_series, etc.).
    n_groups : int
        Number of panel groups.
    params_per_group : list of dict
        List of parameter dicts, one per group.

    Returns
    -------
    callable
        Function that returns panel DataFrame with __ separator.

    """

    def _generator(
        generator_fn,
        n_groups: int,
        params_per_group: list[dict],
        prefix: str = "value",
    ) -> pl.DataFrame:
        if len(params_per_group) != n_groups:
            raise ValueError(f"params_per_group must have {n_groups} entries, got {len(params_per_group)}")

        # Generate each group's series
        group_dfs = []
        for i, params in enumerate(params_per_group):
            df_i = generator_fn(**params)
            # Rename value column to group-specific name
            df_i = df_i.rename({"value": f"{prefix}__group_{i}"})
            group_dfs.append(df_i)

        # Join all on time
        result = group_dfs[0]
        for df_i in group_dfs[1:]:
            result = result.join(df_i, on="time", how="inner")

        return result

    return _generator


@pytest.fixture
def analytical_forecast():
    """Compute closed-form forecast for known series types.

    Returns
    -------
    callable
        Function: analytical_forecast(series_type, params, horizon, last_obs) -> np.ndarray

    """

    def _forecast(
        series_type: str,
        params: dict,
        horizon: int,
        last_obs: float | None = None,
        last_index: int | None = None,
    ) -> np.ndarray:
        if series_type == "constant":
            value = params["value"]
            return np.full(horizon, value)

        elif series_type == "linear":
            slope = params["slope"]
            intercept = params["intercept"]
            if last_index is None:
                raise ValueError("linear forecast requires last_index")
            # Next horizon steps: y = m*(t+1), m*(t+2), ...
            t = np.arange(last_index + 1, last_index + 1 + horizon)
            return slope * t + intercept

        elif series_type == "ar1":
            phi = params["phi"]
            c = params["c"]
            if last_obs is None:
                raise ValueError("AR(1) forecast requires last_obs")
            # y_{t+k} = phi^k * y_t + c * (1 - phi^k) / (1 - phi)
            k = np.arange(1, horizon + 1)
            phi_k = phi**k
            if abs(phi - 1.0) < 1e-10:
                # Unit root case: y_{t+k} = y_t + k*c
                return last_obs + k * c
            else:
                return phi_k * last_obs + c * (1 - phi_k) / (1 - phi)

        elif series_type == "trend_seasonal":
            slope = params["slope"]
            intercept = params["intercept"]
            period = params["period"]
            amplitude = params["amplitude"]
            if last_index is None:
                raise ValueError("trend_seasonal forecast requires last_index")
            t = np.arange(last_index + 1, last_index + 1 + horizon)
            trend = slope * t + intercept
            seasonal = amplitude * np.sin(2 * np.pi * t / period)
            return trend + seasonal

        elif series_type == "exponential":
            a = params["a"]
            b = params["b"]
            if last_index is None:
                raise ValueError("exponential forecast requires last_index")
            t = np.arange(last_index + 1, last_index + 1 + horizon)
            return np.exp(a * t + b)

        else:
            raise ValueError(f"Unknown series_type: {series_type}")

    return _forecast


@pytest.fixture
def single_row_series():
    """Generate single-row DataFrame for edge case testing."""

    def _generator(value: float = 1.0) -> pl.DataFrame:
        return pl.DataFrame({
            "time": [datetime(2020, 1, 1)],
            "value": [value],
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    return _generator


@pytest.fixture
def empty_series():
    """Generate empty DataFrame with correct schema."""

    def _generator() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "time": [],
                "value": [],
            },
            schema={"time": pl.Datetime("us"), "value": pl.Float64},
        )

    return _generator


@pytest.fixture
def all_nan_series():
    """Generate series with all NaN values."""

    def _generator(length: int) -> pl.DataFrame:
        start = datetime(2020, 1, 1)
        time = [start + timedelta(days=i) for i in range(length)]
        return pl.DataFrame({
            "time": time,
            "value": [float("nan")] * length,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    return _generator


@pytest.fixture
def extreme_value_series():
    """Generate series with extreme numeric values."""

    def _generator(magnitude: str = "large") -> pl.DataFrame:
        start = datetime(2020, 1, 1)
        time = [start + timedelta(days=i) for i in range(10)]

        if magnitude == "large":
            values = [1e100] * 10
        elif magnitude == "small":
            values = [1e-100] * 10
        elif magnitude == "mixed":
            values = [1e100, 1e-100, 1e100, 1e-100, 1e100, 1e-100, 1e100, 1e-100, 1e100, 1e-100]
        else:
            raise ValueError(f"Unknown magnitude: {magnitude}")

        return pl.DataFrame({
            "time": time,
            "value": values,
        }).with_columns(pl.col("time").cast(pl.Datetime("us")))

    return _generator
