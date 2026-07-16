"""Signal processing transformers for integration and differentiation."""

import numbers
from typing import Literal, cast

import numpy as np
import polars as pl
import polars.selectors as cs
import scipy.integrate
import scipy.signal
from pydantic import StrictInt
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseTransformer
from yohou.utils import validate_transformer_data
from yohou.utils._compat import Interval, StrOptions, _check_feature_names_in
from yohou.utils.validation import check_interval_consistency, interval_to_timedelta

__all__ = ["NumericalDifferentiator", "NumericalFilter", "NumericalIntegrator"]


class NumericalFilter(BaseTransformer):
    """Apply digital IIR or FIR filters to time series data.

    Applies standard digital filters (Butterworth, Chebyshev, Bessel, etc.)
    for lowpass, highpass, bandpass, or bandstop filtering. Useful for noise
    removal, drift correction, and signal preprocessing.

    Parameters
    ----------
    design : {"butterworth", "chebyshev1", "chebyshev2", "elliptic", "bessel"}, default="butterworth"
        Filter design method:
        - "butterworth": Butterworth (maximally flat passband)
        - "chebyshev1": Chebyshev Type I (passband ripple)
        - "chebyshev2": Chebyshev Type II (stopband ripple)
        - "elliptic": Elliptic/Cauer (passband and stopband ripple)
        - "bessel": Bessel (linear phase)
    mode : {"lowpass", "highpass", "bandpass", "bandstop"}, default="lowpass"
        Filter mode. For bandpass/bandstop, cutoff_frequency should be a 2-tuple.
    order : int, default=4
        Filter order. Higher order = sharper cutoff but more phase distortion.
    cutoff_frequency : float or tuple of float, default=0.1
        Cutoff frequency as fraction of Nyquist (0 to 1). For bandpass/bandstop,
        provide (low_freq, high_freq).
    passband_ripple : float or None, default=None
        Passband ripple in dB (for chebyshev1, elliptic). Defaults to 1.0 if required.
    stopband_attenuation : float or None, default=None
        Stopband attenuation in dB (for chebyshev2, elliptic). Defaults to 40.0 if required.

    Attributes
    ----------
    b_ : ndarray
        Numerator coefficients of the filter.
    a_ : ndarray
        Denominator coefficients of the filter.
    zi_ : dict of ndarray
        Filter delay state per column. Updated after each transform call
        to enable streaming.

    Notes
    -----
    **Statefulness**: The filter maintains internal state (delay line values)
    between transform calls. This enables streaming/chunked processing without
    transients at chunk boundaries.

    Use ``rewind()`` to clear the filter state and start fresh.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> import numpy as np
    >>> from yohou.preprocessing import NumericalFilter

    >>> # Generate noisy signal
    >>> times = pl.datetime_range(
    ...     start=datetime(2020, 1, 1), end=datetime(2020, 1, 1, 0, 1), interval="1s", eager=True
    ... )
    >>> t = np.arange(len(times))
    >>> np.random.seed(42)  # deterministic noise for a reproducible example
    >>> signal = np.sin(2 * np.pi * 0.05 * t) + 0.5 * np.random.randn(len(t))
    >>> X = pl.DataFrame({"time": times, "signal": signal.tolist()})

    >>> # Apply lowpass filter (causal, stateful)
    >>> transformer = NumericalFilter(design="butterworth", mode="lowpass", order=4, cutoff_frequency=0.2)
    >>> transformer.fit(X)
    NumericalFilter(...)
    >>> X_filtered = transformer.transform(X)
    >>> "time" in X_filtered.columns
    True
    >>> # Filter state preserved for subsequent chunks
    >>> # Use transformer.rewind() to clear state

    See Also
    --------
    - [`NumericalIntegrator`][yohou.preprocessing.signal.NumericalIntegrator] : Numerical integration.
    - [`NumericalDifferentiator`][yohou.preprocessing.signal.NumericalDifferentiator] : Numerical differentiation.
    - [`scipy.signal.butter`][scipy.signal.butter] : Butterworth filter design.

    """

    _valid_designs = {"butterworth", "chebyshev1", "chebyshev2", "elliptic", "bessel"}
    _valid_modes = {"lowpass", "highpass", "bandpass", "bandstop"}

    _parameter_constraints: dict = {
        "design": [StrOptions(_valid_designs)],
        "mode": [StrOptions(_valid_modes)],
        "order": [Interval(numbers.Integral, 1, None, closed="left")],
        "cutoff_frequency": [Interval(numbers.Real, 0.0, 1.0, closed="neither"), tuple, list],
        "passband_ripple": [Interval(numbers.Real, 0.0, None, closed="neither"), None],
        "stopband_attenuation": [Interval(numbers.Real, 0.0, None, closed="neither"), None],
    }

    _tags = {"stateful": True}

    def __init__(
        self,
        design: str = "butterworth",
        mode: str = "lowpass",
        order: int = 4,
        cutoff_frequency: float | tuple[float, float] = 0.1,
        passband_ripple: float | None = None,
        stopband_attenuation: float | None = None,
    ):
        self.design = design
        self.mode = mode
        self.order = order
        self.cutoff_frequency = cutoff_frequency
        self.passband_ripple = passband_ripple
        self.stopband_attenuation = stopband_attenuation

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Fit the internal model."""
        # Get filter design function
        filter_funcs = {
            "butterworth": scipy.signal.butter,
            "chebyshev1": scipy.signal.cheby1,
            "chebyshev2": scipy.signal.cheby2,
            "elliptic": scipy.signal.ellip,
            "bessel": scipy.signal.bessel,
        }
        filter_func = filter_funcs[self.design]

        # Build filter kwargs
        kwargs = {
            "N": self.order,
            "Wn": self.cutoff_frequency,
            "btype": self.mode,
            "output": "ba",
        }

        # Add ripple parameters for specific filter types
        if self.design == "chebyshev1":
            kwargs["rp"] = self.passband_ripple if self.passband_ripple is not None else 1.0
        elif self.design == "chebyshev2":
            kwargs["rs"] = self.stopband_attenuation if self.stopband_attenuation is not None else 40.0
        elif self.design == "elliptic":
            kwargs["rp"] = self.passband_ripple if self.passband_ripple is not None else 1.0
            kwargs["rs"] = self.stopband_attenuation if self.stopband_attenuation is not None else 40.0

        # Design filter
        self.b_, self.a_ = filter_func(**kwargs)

        # Initialize filter state dict
        self.zi_: dict[str, np.ndarray] = {}

    def rewind(self, X: pl.DataFrame) -> "NumericalFilter":
        """Rewind the filter state and observation horizon.

        Clears the stored filter delay state and rewinds the observation
        window, so the next transform call starts fresh.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series to set new observation window.

        Returns
        -------
        self

        """
        # Rewind filter delay state
        self.zi_ = {}
        # Call parent rewind
        BaseTransformer.rewind(self, X)
        return self

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Apply digital filter to time series.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.

        Returns
        -------
        pl.DataFrame
            Filtered time series.

        """
        # Get data columns
        data_cols = [c for c in X.columns if c != "time"]

        # Apply filter to each column
        result_cols = {"time": X["time"]}

        for col_name in data_cols:
            signal_data = X[col_name].to_numpy()

            # Causal filtering with state preservation
            if col_name in self.zi_:
                # Use stored state from previous transform
                zi = self.zi_[col_name]
            else:
                # Initialize state from first sample
                zi = scipy.signal.lfilter_zi(self.b_, self.a_) * signal_data[0]

            filtered, zf = scipy.signal.lfilter(self.b_, self.a_, signal_data, zi=zi)
            # Store final state for next transform
            self.zi_[col_name] = zf

            result_cols[col_name] = pl.Series(filtered)

        return pl.DataFrame(result_cols)

    def observe_transform(self, X: pl.DataFrame, **params) -> pl.DataFrame:
        """Transform new data and update state without clearing filter delays.

        The base ``observe_transform`` calls ``observe()``, which in turn calls
        ``rewind()``, clearing the filter delay state ``zi_`` as a side effect and
        reintroducing chunk-boundary transients. This override bypasses that chain
        and preserves ``zi_`` so streaming/chunked processing continues seamlessly.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column (datetime) and one or
            more numeric columns.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Transformed time series with a ``"time"`` column and transformed
            value columns.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=True)

        X_t = self.transform(X, **params)
        # Update observed window without clearing the filter delay state.
        self._update_X_observed(X)
        return X_t

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : list of str or None, default=None
            Column names of the input features.  If ``None``, uses the
            feature names seen during ``fit``.

        Returns
        -------
        list of str
            Output feature names after transformation.

        """
        check_is_fitted(self, ["feature_names_in_"])
        input_features = _check_feature_names_in(self, input_features)
        return list(input_features)


class NumericalIntegrator(BaseTransformer):
    """Numerical integration transformer for time series signals.

    Integrates each feature column using scipy's cumulative integration methods.
    The integration uses the time column to determine the sampling interval.

    This transformer is **stateful**: it maintains a running integral offset
    between transform calls, enabling streaming/chunked processing. Each
    transform call continues the integration from where the previous chunk
    left off.

    Parameters
    ----------
    method : {"cumulative_trapezoid", "cumulative_simpson"}, default="cumulative_trapezoid"
        The scipy.integrate method to use for cumulative integration.
        - "cumulative_trapezoid": Trapezoidal rule integration (faster, less accurate)
        - "cumulative_simpson": Simpson's rule integration (slower, more accurate)

    Attributes
    ----------
    interval_ : str
        Detected time interval string (e.g., '1d', '1h', '1s').

    sampling_interval_ : float
        Sampling interval in seconds derived from interval_.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> time = [datetime(2020, 1, 1) + timedelta(seconds=i * 0.001) for i in range(100)]
    >>> X = pl.DataFrame({"time": time, "signal": [float(i) for i in range(100)]})
    >>> transformer = NumericalIntegrator(method="cumulative_trapezoid")
    >>> transformer.fit(X)
    NumericalIntegrator(...)
    >>> X_t = transformer.transform(X)
    >>> "time" in X_t.columns
    True

    Notes
    -----
    **Statefulness**: The integrator maintains the last transformed value
    between transform calls via ``_X_t_observed_``. This enables accurate
    streaming integration where each chunk continues seamlessly from the
    previous one. The last input values are stored in ``_last_X_value_`` for
    proper trapezoid boundary calculation (this is a per-column last-input
    buffer, distinct from the parent-class ``_X_observed`` observation window).

    Use ``rewind()`` to clear the integration state and start fresh.

    - Both methods produce output of the same length as the input because
      ``initial=0.0`` is always supplied to scipy
    - Inverse transform uses numerical differentiation (np.gradient)

    See Also
    --------
    - [`NumericalDifferentiator`][yohou.preprocessing.signal.NumericalDifferentiator] : Numerical differentiation transformer.
    - [`scipy.integrate.cumulative_trapezoid`][scipy.integrate.cumulative_trapezoid] : Trapezoidal integration.
    - [`scipy.integrate.cumulative_simpson`][scipy.integrate.cumulative_simpson] : Simpson's rule integration.

    """

    _parameter_constraints: dict = {
        "method": [StrOptions({"cumulative_trapezoid", "cumulative_simpson"})],
    }

    _tags = {"stateful": True, "invertible": True}

    def __init__(
        self,
        method: Literal["cumulative_trapezoid", "cumulative_simpson"] = "cumulative_trapezoid",
    ):
        self.method = method

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Fit the internal model."""
        # Detect interval using utility function
        self.interval_ = check_interval_consistency(X)
        td = interval_to_timedelta(self.interval_)
        if td is None:
            raise ValueError(
                f"NumericalIntegrator requires fixed-length intervals, but got variable interval: {self.interval_}"
            )
        self.sampling_interval_ = td.total_seconds()

        # Initialize state tracking
        # _last_X_value_: Last input value per column (for trapezoid boundary)
        # _X_t_observed_: Last transformed value per column (running integral offset)
        self._last_X_value_: dict[str, float] = {}
        self._X_t_observed_: dict[str, float] = {}

    def rewind(self, X: pl.DataFrame) -> "NumericalIntegrator":
        """Rewind the integration state and observation horizon.

        Clears the running integral offset, so the next transform call
        starts integration from zero.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series to set new observation window.

        Returns
        -------
        self

        """
        # Rewind state tracking
        self._last_X_value_ = {}
        self._X_t_observed_ = {}
        # Call parent rewind
        BaseTransformer.rewind(self, X)
        return self

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Integrate each feature column.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.

        Returns
        -------
        pl.DataFrame
            Integrated time series.

        """
        time = X.select(cs.by_name("time"))
        data = X.select(~cs.by_name("time"))

        integrator_func = getattr(scipy.integrate, self.method)
        dt = self.sampling_interval_

        # Integrate each column
        result_cols = {}
        for col_name in data.columns:
            col_values = data[col_name].to_numpy()

            # Handle the single inter-chunk interval with the trapezoid rule.
            # This bridge applies regardless of the intra-chunk integration
            # order (trapezoid or simpson); a single interval has no Simpson
            # form, so the trapezoidal half-step is the correct correction.
            boundary_integral = 0.0
            if col_name in self._last_X_value_:
                # Add half-step for proper boundary: 0.5 * (last + first) * dt
                last_val = self._last_X_value_[col_name]
                boundary_integral = 0.5 * (last_val + col_values[0]) * dt

            # Get offset from previous chunks
            offset = self._X_t_observed_.get(col_name, 0.0) + boundary_integral

            # Perform integration with initial=0, then add offset
            integrated = integrator_func(col_values, x=None, dx=dt, initial=0.0)
            integrated = integrated + offset

            # Store state for next chunk
            self._X_t_observed_[col_name] = float(integrated[-1])
            self._last_X_value_[col_name] = float(col_values[-1])

            result_cols[col_name] = integrated

        X_t = pl.DataFrame(result_cols)
        feature_names = self.get_feature_names_out()
        X_t = X_t.rename(dict(zip(X_t.columns, feature_names, strict=False)))
        X_t = pl.concat([time, X_t], how="horizontal")

        return X_t

    def observe_transform(self, X: pl.DataFrame, **params) -> pl.DataFrame:
        """Transform new data and update integration state without rewind.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column (datetime) and one or
            more numeric columns.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Transformed time series with a ``"time"`` column and transformed
            value columns.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_", "sampling_interval_"])
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=True)

        X_t = self.transform(X, **params)
        # Update observed time without clearing integration state
        self._update_X_observed(X)
        return X_t

    def _inverse_transform(self, X_t: pl.DataFrame, X_p: pl.DataFrame | None = None) -> pl.DataFrame:
        """Differentiate to reverse integration.

        Parameters
        ----------
        X_t : pl.DataFrame
            Integrated time series.
        X_p : pl.DataFrame or None
            Not used; the inverse is applied statically regardless of the
            running integration state.

        Returns
        -------
        pl.DataFrame
            Inverse-transformed time series.

        """
        X_t, _ = validate_transformer_data(
            self,
            X=X_t,
            reset=False,
            inverse=True,
            X_p=X_p,
            observation_horizon=0,
        )

        time = X_t.select(cs.by_name("time"))
        data = X_t.select(~cs.by_name("time"))

        dt = self.sampling_interval_

        # Differentiate each column
        result_cols = {}
        for col_name in data.columns:
            col_values = data[col_name].to_numpy()
            differentiated = np.gradient(col_values, dt)
            result_cols[col_name] = differentiated

        X = pl.DataFrame(result_cols)
        X = X.rename(dict(zip(X.columns, self.feature_names_in_, strict=False)))
        X = pl.concat([time, X], how="horizontal")

        return X

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Column names of the input features.  If ``None``, uses the
            feature names seen during ``fit``.

        Returns
        -------
        list of str
            Output feature names after transformation.

        """
        input_features = _check_feature_names_in(self, input_features)
        return [f"{col}_integrated" for col in input_features]


class NumericalDifferentiator(BaseTransformer):
    """Numerical differentiation transformer for time series signals.

    Differentiates each feature column using np.gradient, which computes
    the derivative using central differences in the interior and first
    differences at the boundaries.

    Parameters
    ----------
    order : {1, 2}, default=1
        Gradient is calculated using N-th order accurate differences at
        the boundaries:
        - 1: First-order accurate (uses 2 points at boundary)
        - 2: Second-order accurate (uses 3 points at boundary)

    Attributes
    ----------
    interval_ : str
        Detected time interval string (e.g., '1d', '1h', '1s').

    sampling_interval_ : float
        Sampling interval in seconds derived from interval_.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> time = [datetime(2020, 1, 1) + timedelta(seconds=i * 0.001) for i in range(100)]
    >>> X = pl.DataFrame({"time": time, "signal": [float(i) for i in range(100)]})
    >>> transformer = NumericalDifferentiator(order=1)
    >>> transformer.fit(X)
    NumericalDifferentiator(...)
    >>> X_t = transformer.transform(X)
    >>> "time" in X_t.columns
    True

    Notes
    -----
    - Output has the same length as input
    - Uses central differences in the interior (more accurate)
    - Uses one-sided differences at boundaries (order controls accuracy)
    - Inverse transform uses cumulative trapezoidal integration

    See Also
    --------
    - [`NumericalIntegrator`][yohou.preprocessing.signal.NumericalIntegrator] : Numerical integration transformer.
    - [`numpy.gradient`][numpy.gradient] : NumPy gradient function.

    """

    _parameter_constraints: dict = {
        "order": [Interval(numbers.Integral, 1, 2, closed="both")],
    }

    _tags = {"invertible": True}

    def __init__(self, order: StrictInt = 1):
        self.order = order

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Fit the internal model."""
        # Detect interval using utility function
        self.interval_ = check_interval_consistency(X)
        td = interval_to_timedelta(self.interval_)
        if td is None:
            raise ValueError(
                f"NumericalDifferentiator requires fixed-length intervals, but got variable interval: {self.interval_}"
            )
        self.sampling_interval_ = td.total_seconds()

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Differentiate each feature column.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.

        Returns
        -------
        pl.DataFrame
            Differentiated time series.

        """
        time = X.select(cs.by_name("time"))
        data = X.select(~cs.by_name("time"))

        dt = self.sampling_interval_

        # Differentiate each column
        result_cols = {}
        for col_name in data.columns:
            col_values = data[col_name].to_numpy()
            # Cast order to Literal[1, 2] for numpy typing (validated in parameter_constraints)
            differentiated = np.gradient(col_values, dt, edge_order=cast(Literal[1, 2], self.order))
            result_cols[col_name] = differentiated

        X_t = pl.DataFrame(result_cols)
        feature_names = self.get_feature_names_out()
        X_t = X_t.rename(dict(zip(X_t.columns, feature_names, strict=False)))
        X_t = pl.concat([time, X_t], how="horizontal")

        return X_t

    def _inverse_transform(self, X_t: pl.DataFrame, X_p: pl.DataFrame | None = None) -> pl.DataFrame:
        """Integrate to reverse differentiation.

        Parameters
        ----------
        X_t : pl.DataFrame
            Differentiated time series.
        X_p : pl.DataFrame or None
            Not used for this stateless transformer.

        Returns
        -------
        pl.DataFrame
            Inverse-transformed time series.

        """
        X_t, _ = validate_transformer_data(
            self,
            X=X_t,
            reset=False,
            inverse=True,
            X_p=X_p,
            observation_horizon=0,
        )

        time = X_t.select(cs.by_name("time"))
        data = X_t.select(~cs.by_name("time"))

        dt = self.sampling_interval_

        # Integrate each column
        result_cols = {}
        for col_name in data.columns:
            col_values = data[col_name].to_numpy()
            integrated = scipy.integrate.cumulative_trapezoid(col_values, x=None, dx=dt, initial=0.0)
            result_cols[col_name] = integrated

        X = pl.DataFrame(result_cols)
        X = X.rename(dict(zip(X.columns, self.feature_names_in_, strict=False)))
        X = pl.concat([time, X], how="horizontal")

        return X

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Column names of the input features.  If ``None``, uses the
            feature names seen during ``fit``.

        Returns
        -------
        list of str
            Output feature names after transformation.

        """
        input_features = _check_feature_names_in(self, input_features)
        return [f"{col}_differentiated" for col in input_features]
