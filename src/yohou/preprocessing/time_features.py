"""Fourier and time index feature transformers for time series."""

import math
import numbers

import numpy as np
import polars as pl
from pydantic import StrictInt
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseActualTransformer
from yohou.utils._compat import Interval
from yohou.utils.validation import interval_to_timedelta


class FourierFeatureTransformer(BaseActualTransformer):
    r"""Generate Fourier harmonic features from the time column.

    Creates sin/cos feature pairs at specified harmonics of a given
    seasonal period, useful for encoding cyclical patterns as inputs
    to reduction forecasters. The output contains only the ``"time"``
    column and the generated Fourier columns (prefixed with ``fourier_``);
    all original input feature columns are dropped.

    For each harmonic $k$ and seasonal period $S$, the features at
    time step $t$ are:

    $$\text{sin}_k(t) = \sin\!\left(\frac{2\pi k\, t}{S}\right), \quad \text{cos}_k(t) = \cos\!\left(\frac{2\pi k\, t}{S}\right)$$

    where $t$ is the number of time steps since the first observed
    timestamp, $S$ is the ``seasonality`` parameter, and $k$ ranges
    over the selected ``harmonics``. Harmonics must satisfy
    $k \leq S / 2$ (Nyquist limit).

    Parameters
    ----------
    seasonality : float, default=7.0
        Seasonal period length in number of time steps. Can be
        non-integer (e.g., ``365.25`` for yearly on daily data).
    harmonics : list of int or None, default=None
        Which Fourier harmonics to include. For example,
        ``[1, 2, 3]`` uses the first three harmonics. If ``None``,
        defaults to ``[1]``.

    Attributes
    ----------
    harmonics_ : list of int
        Effective list of harmonics used for feature generation.
    first_time_ : datetime
        First observed timestamp, used as reference for index
        computation.

    Raises
    ------
    ValueError
        At fit time if the effective ``harmonics`` list is empty; if any
        harmonic is less than 1; or if the maximum harmonic exceeds
        ``seasonality / 2`` (the Nyquist limit).

    See Also
    --------
    - [`CalendarFeatureTransformer`][yohou.preprocessing.calendar.CalendarFeatureTransformer] : Calendar features (month, day of week, etc.).
    - [`HolidayFeatureTransformer`][yohou.preprocessing.calendar.HolidayFeatureTransformer] : Binary holiday indicator.
    - [`DaylightSavingFeatureTransformer`][yohou.preprocessing.calendar.DaylightSavingFeatureTransformer] : Daylight-saving offset and transition-day features.
    - [`TimeIndexTransformer`][yohou.preprocessing.time_features.TimeIndexTransformer] : Numeric time index for trend features.
    - [`FourierSeasonalityForecaster`][yohou.stationarity.seasonality.FourierSeasonalityForecaster] : Forecaster-level Fourier seasonality.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> time = pl.datetime_range(
    ...     start=datetime(2020, 1, 1), end=datetime(2020, 1, 15), interval="1d", eager=True
    ... )
    >>> X = pl.DataFrame({"time": time, "value": range(len(time))})
    >>> transformer = FourierFeatureTransformer(seasonality=7.0, harmonics=[1, 2])
    >>> transformer.fit(X)
    FourierFeatureTransformer(harmonics=[1, 2])
    >>> X_t = transformer.transform(X)
    >>> "fourier_7.0_sin_1" in X_t.columns
    True

    """

    # Batch invariant: a pure function of each row's timestamp, so chunking cannot change a value.
    _tags = {"batch_invariant": True}

    _parameter_constraints: dict = {
        "seasonality": [Interval(numbers.Real, 0, None, closed="neither")],
        "harmonics": [list, None],
    }

    def __init__(
        self,
        seasonality: float = 7.0,
        harmonics: list[int] | None = None,
    ):
        self.seasonality = seasonality
        self.harmonics = harmonics

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Fit the internal model."""
        self.harmonics_ = self.harmonics if self.harmonics is not None else [1]

        if not self.harmonics_:
            raise ValueError("harmonics list cannot be empty")
        if any(h < 1 for h in self.harmonics_):
            raise ValueError("All harmonics must be positive integers")

        max_harmonic = max(self.harmonics_)
        if max_harmonic > self.seasonality / 2:
            raise ValueError(
                f"Maximum harmonic ({max_harmonic}) cannot exceed seasonality/2 "
                f"({self.seasonality / 2:.1f}) due to Nyquist sampling theorem."
            )

        self.first_time_ = X["time"][0]

        # Fixed sampling step (seconds) from the fitted interval, used so that
        # transform produces consistent Fourier phases regardless of which
        # sub-range of the series is passed (including single-row chunks).
        step = interval_to_timedelta(self.interval_)
        self._step_seconds_ = step.total_seconds() if step is not None else None

        s = f"{self.seasonality}"
        generated_names = []
        for k in self.harmonics_:
            generated_names.append(f"fourier_{s}_sin_{k}")
            generated_names.append(f"fourier_{s}_cos_{k}")
        existing = set(X.columns) - {"time"}
        conflicts = set(generated_names) & existing
        if conflicts:
            raise ValueError(f"Generated column names {sorted(conflicts)} conflict with existing columns in X.")

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Generate Fourier harmonic features from the time column.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.

        Returns
        -------
        pl.DataFrame
            DataFrame with ``"time"`` column and Fourier feature columns.

        """
        interval = self.interval_
        first_time = self.first_time_
        if interval.endswith("mo") or interval.endswith("y"):
            # Anchor month/year offsets to the fitted first timestamp via
            # calendar arithmetic, so phases continue past the fit origin.
            months = (
                ((X["time"].dt.year() - first_time.year) * 12 + (X["time"].dt.month() - first_time.month))
                .to_numpy()
                .astype(np.float64)
            )
            if interval.endswith("y"):
                t = months / 12.0
            else:
                mo_count = int(interval.replace("mo", ""))
                t = months / mo_count
        else:
            t = (X["time"] - first_time).dt.total_seconds().to_numpy().astype(np.float64)
            # Normalise by the fixed sampling step from fit (not the spacing of
            # the incoming chunk), avoiding wrong angles on single-row chunks.
            if self._step_seconds_ and self._step_seconds_ != 0:
                t = t / self._step_seconds_

        feature_cols = []
        s = f"{self.seasonality}"
        for k in self.harmonics_:
            angle = 2.0 * math.pi * k * t / self.seasonality
            sin_vals = np.sin(angle)
            cos_vals = np.cos(angle)
            feature_cols.append(pl.Series(f"fourier_{s}_sin_{k}", sin_vals, dtype=pl.Float64))
            feature_cols.append(pl.Series(f"fourier_{s}_cos_{k}", cos_vals, dtype=pl.Float64))

        return X.select(pl.col("time")).with_columns(*feature_cols)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names (unused, for API compatibility).

        Returns
        -------
        list of str
            Generated Fourier feature column names.

        """
        check_is_fitted(self, ["harmonics_"])
        s = f"{self.seasonality}"
        generated = []
        for k in self.harmonics_:
            generated.append(f"fourier_{s}_sin_{k}")
            generated.append(f"fourier_{s}_cos_{k}")
        return generated


class TimeIndexTransformer(BaseActualTransformer):
    r"""Convert the time column to a numeric index with optional polynomial terms.

    Produces integer or normalized time step indices starting from the
    first observed timestamp, useful as trend features for reduction
    forecasters. The output contains only the ``"time"`` column and the
    generated index columns; all original input feature columns are dropped.

    The base index at time step $t$ is:

    $$x(t) = \frac{t - t_0}{\Delta t}$$

    where $t_0$ is the first observed timestamp and $\Delta t$ is the
    detected time interval. When ``normalize=True``, the index is
    scaled by the number of fit steps:

    $$\tilde{x}(t) = \frac{x(t)}{N - 1}$$

    where $N$ is ``n_steps_``. Polynomial features of degree $d$ are
    then $\tilde{x}(t),\, \tilde{x}(t)^2,\, \ldots,\, \tilde{x}(t)^d$.

    Parameters
    ----------
    normalize : bool, default=False
        If ``True``, scale the index to ``[0, 1]`` based on the number
        of steps in the fit data. Values can exceed this range for
        data beyond the fit range.
    degree : int, default=1
        Polynomial degree. ``1`` produces a linear index only, ``2``
        adds a quadratic term, and so on.

    Attributes
    ----------
    first_time_ : datetime
        First observed timestamp, used as reference for index
        computation.
    n_steps_ : int
        Number of time steps in the fit data. Used as normalization
        denominator (``n_steps_ - 1``).

    See Also
    --------
    - [`CalendarFeatureTransformer`][yohou.preprocessing.calendar.CalendarFeatureTransformer] : Calendar features (month, day of week, etc.).
    - [`FourierFeatureTransformer`][yohou.preprocessing.time_features.FourierFeatureTransformer] : Sin/cos harmonics for cyclical encoding.
    - [`PolynomialTrendForecaster`][yohou.stationarity.trend.PolynomialTrendForecaster] : Forecaster-level polynomial trend estimation.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> time = pl.datetime_range(
    ...     start=datetime(2020, 1, 1), end=datetime(2020, 1, 11), interval="1d", eager=True
    ... )
    >>> X = pl.DataFrame({"time": time, "value": range(len(time))})
    >>> transformer = TimeIndexTransformer(degree=2)
    >>> transformer.fit(X)
    TimeIndexTransformer(degree=2)
    >>> X_t = transformer.transform(X)
    >>> X_t["time_index"][0]
    0
    >>> "time_index_2" in X_t.columns
    True

    """

    # Batch invariant: a pure function of each row's timestamp, so chunking cannot change a value.
    _tags = {"batch_invariant": True}

    _parameter_constraints: dict = {
        "normalize": ["boolean"],
        "degree": [Interval(numbers.Integral, 1, None, closed="left")],
    }

    _PREFIX = "time_index"

    def __init__(
        self,
        normalize: bool = False,
        degree: StrictInt = 1,
    ):
        self.normalize = normalize
        self.degree = degree

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Fit the internal model."""
        self.first_time_ = X["time"][0]
        self.n_steps_ = len(X)

        # Cache the step size in seconds so the step index is computed
        # consistently regardless of how many rows a transform chunk contains.
        step = interval_to_timedelta(self.interval_)
        self._step_seconds_ = step.total_seconds() if step is not None else None

        generated_names = []
        for d in range(1, self.degree + 1):
            col_name = self._PREFIX if d == 1 else f"{self._PREFIX}_{d}"
            generated_names.append(col_name)
        existing = set(X.columns) - {"time"}
        conflicts = set(generated_names) & existing
        if conflicts:
            raise ValueError(f"Generated column names {sorted(conflicts)} conflict with existing columns in X.")

    def _compute_index(self, X: pl.DataFrame) -> np.ndarray:
        """Compute the numeric time index for the given data.

        Parameters
        ----------
        X : pl.DataFrame
            DataFrame with ``"time"`` column.

        Returns
        -------
        np.ndarray
            Numeric time index array.

        """
        interval = self.interval_
        time_diff = X["time"] - self.first_time_

        if interval.endswith("mo") or interval.endswith("y"):
            first_time = self.first_time_
            months = (
                ((X["time"].dt.year() - first_time.year) * 12 + (X["time"].dt.month() - first_time.month))
                .to_numpy()
                .astype(np.float64)
            )
            if interval.endswith("y"):
                t = months / 12.0
            else:
                mo_count = int(interval.replace("mo", ""))
                t = months / mo_count
        else:
            t = time_diff.dt.total_seconds().to_numpy().astype(np.float64)
            if self._step_seconds_ and self._step_seconds_ != 0:
                t = t / self._step_seconds_

        return t

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Generate time index features from the time column.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.

        Returns
        -------
        pl.DataFrame
            DataFrame with ``"time"`` column and time index features.

        """
        t = self._compute_index(X)

        feature_cols = []
        for d in range(1, self.degree + 1):
            col_name = self._PREFIX if d == 1 else f"{self._PREFIX}_{d}"
            if d == 1 and not self.normalize:
                values = t.astype(np.int64)
                feature_cols.append(pl.Series(col_name, values, dtype=pl.Int64))
            elif d == 1 and self.normalize:
                denom = max(self.n_steps_ - 1, 1)
                values = t / denom
                feature_cols.append(pl.Series(col_name, values, dtype=pl.Float64))
            else:
                if self.normalize:
                    denom = max(self.n_steps_ - 1, 1)
                    base = t / denom
                else:
                    base = t
                values = base**d
                feature_cols.append(pl.Series(col_name, values, dtype=pl.Float64))

        return X.select(pl.col("time")).with_columns(*feature_cols)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names (unused, for API compatibility).

        Returns
        -------
        list of str
            Generated time index feature column names.

        """
        check_is_fitted(self, ["first_time_"])
        generated = []
        for d in range(1, self.degree + 1):
            col_name = self._PREFIX if d == 1 else f"{self._PREFIX}_{d}"
            generated.append(col_name)
        return generated
