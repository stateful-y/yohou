"""Function-based transformers for time series."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Literal

import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.utils.metadata_routing import UNUSED
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseActualTransformer
from yohou.utils import validate_transformer_data
from yohou.utils._compat import StrOptions, _check_feature_names_in, _fit_context

__all__ = ["FunctionTransformer"]


class FunctionTransformer(BaseActualTransformer):
    """Constructs a transformer from an arbitrary callable.

    A FunctionTransformer forwards its X arguments to a user-defined function
    and returns the result. This is useful for both stateless transformations
    such as log scaling and stateful transformations such as diff that require
    past values.

    The function receives a polars DataFrame (excluding the "time" column) and
    should return a polars DataFrame or numpy array with the same number of rows.

    For stateful functions that produce NaN values at the beginning (e.g., diff),
    use ``fit_transform`` which auto-detects the warmup period by counting
    leading NaN rows. The transformer stores these initial rows for use in
    ``inverse_transform``.

    Parameters
    ----------
    func : callable or None, default=None
        The callable to use for the transformation. This will be passed the
        DataFrame excluding the "time" column. If func is None, then func
        will be the identity function.
    inverse_func : callable or None, default=None
        The callable to use for the inverse transformation.  For stateful
        transforms (with auto-detected warmup), the past observations are
        passed as the second positional argument:
        ``inverse_func(X_t, X_p, **inv_kw_args)``.  For stateless transforms,
        only ``inverse_func(X_t, **inv_kw_args)`` is called.  If ``None``,
        the identity function is used.
    check_inverse : bool, default=True
        Whether to check that ``func`` followed by ``inverse_func`` leads
        to the original inputs. It can be used for a sanity check, raising
        a warning when the condition is not fulfilled.
    kw_args : dict or None, default=None
        Dictionary of additional keyword arguments to pass to func.
    inv_kw_args : dict or None, default=None
        Dictionary of additional keyword arguments to pass to inverse_func.
    feature_names_out : callable, 'one-to-one' or None, default=None
        Determines the list of feature names that will be returned by the
        ``get_feature_names_out`` method. If 'one-to-one', the output feature
        names equal input feature names. If a callable, it must take two
        arguments (self, input_features) and return output feature names.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : list of str
        Names of features seen during fit.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> import numpy as np
    >>> from yohou.preprocessing import FunctionTransformer

    >>> times = pl.datetime_range(
    ...     start=datetime(2020, 1, 1), end=datetime(2020, 1, 10), interval="1d", eager=True
    ... )
    >>> X = pl.DataFrame({"time": times, "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})

    >>> # Log transformation
    >>> transformer = FunctionTransformer(func=np.log, inverse_func=np.exp)
    >>> transformer.fit(X)
    FunctionTransformer(func=<ufunc 'log'>, inverse_func=<ufunc 'exp'>)
    >>> X_t = transformer.transform(X)
    >>> "time" in X_t.columns
    True

    >>> # Identity transformation (when func is None)
    >>> transformer = FunctionTransformer()
    >>> transformer.fit(X)
    FunctionTransformer()
    >>> X_t = transformer.transform(X)
    >>> X_t["value"].to_list() == X["value"].to_list()
    True

    >>> # Stateful transformation (diff) with auto-detected warmup
    >>> def diff_func(df):
    ...     return df.select(pl.all().diff())
    >>> def cumsum_func(df, X_p=None):
    ...     r = df.select(pl.all().cum_sum())
    ...     if X_p is not None:
    ...         for c in r.columns:
    ...             r = r.with_columns((pl.col(c) + X_p[c][-1]).alias(c))
    ...     return r
    >>> transformer = FunctionTransformer(func=diff_func, inverse_func=cumsum_func)
    >>> X_t = transformer.fit_transform(X)  # Auto-detects warmup=1 from NaN
    >>> len(X_t) == len(X) - 1  # First row sliced off
    True

    See Also
    --------
    - [`SlidingWindowFunctionTransformer`][yohou.preprocessing.window.SlidingWindowFunctionTransformer] : Apply function over sliding windows.
    - [`RollingStatisticsTransformer`][yohou.preprocessing.window.RollingStatisticsTransformer] : Compute rolling statistics.

    Notes
    -----
    Warmup detection is automatic: during ``fit``, the function is applied
    and leading all-NaN rows are counted to set ``_observation_horizon``.
    NaN rows must be contiguous at the start; a ``ValueError`` is raised if
    NaN appears at discontinuous positions.

    For stateful transforms (e.g., diff), ``inverse_func`` receives the
    past observations as a second positional argument:
    ``inverse_func(X_t, X_p, **inv_kw_args)``. For stateless transforms,
    only ``inverse_func(X_t, **inv_kw_args)`` is called.

    The ``check_inverse`` flag samples up to 100 rows and warns if the
    round-trip ``inverse_func(func(X))`` does not recover the original
    values within a tolerance of ``rtol=1e-5``, ``atol=1e-8``.

    """

    # ``transform(X, X_p)`` carries a second data frame; UNUSED keeps it off
    # the request surface. Only X_p: this signature has no X_t, and UNUSED on
    # a nonexistent key raises at class creation.
    __metadata_request__transform = {"X_p": UNUSED}

    _parameter_constraints: dict = {
        "func": [callable, None],
        "inverse_func": [callable, None],
        "check_inverse": ["boolean"],
        "kw_args": [dict, None],
        "inv_kw_args": [dict, None],
        "feature_names_out": [callable, StrOptions({"one-to-one"}), None],
    }

    # NOT batch invariant. `func` is caller supplied and receives the whole frame, so
    # it may read rows ahead of the one it writes. The tag is a class level claim and
    # this class cannot make it for every `func`, even though a row local `func`
    # would satisfy it. Contrast `SlidingWindowFunctionTransformer`, which can make
    # the claim because it hands `func` a trailing window rather than the frame.
    _tags = {"stateful": True, "batch_invariant": False}

    def __init__(
        self,
        func: Callable | None = None,
        inverse_func: Callable | None = None,
        *,
        check_inverse: bool = True,
        kw_args: dict | None = None,
        inv_kw_args: dict | None = None,
        feature_names_out: Callable | Literal["one-to-one"] | None = None,
    ):
        self.func = func
        self.inverse_func = inverse_func
        self.check_inverse = check_inverse
        self.kw_args = kw_args
        self.inv_kw_args = inv_kw_args
        self.feature_names_out = feature_names_out

    def _check_inverse_transform(self, X: pl.DataFrame) -> None:
        """Check that func and inverse_func are inverses."""
        # Sample a subset of the data for efficiency. For stateful transforms
        # the round-trip needs contiguous history (e.g. finite differences), so
        # take a contiguous tail slice rather than spread-out rows.
        n_samples = min(100, len(X))
        if self._observation_horizon > 0:
            X_sample = X.tail(n_samples)
        else:
            idx = np.linspace(0, len(X) - 1, n_samples, dtype=int)
            X_sample = X[idx.tolist()]

        X_transformed = self.transform(X_sample)
        # Provide warmup rows as X_p so inverse_func can reconstruct
        X_p = X_sample.head(self._observation_horizon) if self._observation_horizon > 0 else None
        X_round_trip = self.inverse_transform(X_transformed, X_p=X_p)

        # Compare numeric columns only
        numeric_cols = [c for c in X.columns if c != "time"]
        for col in numeric_cols:
            original = X_sample[col].to_numpy()
            recovered = X_round_trip[col].to_numpy()
            if len(original) != len(recovered):
                min_len = min(len(original), len(recovered))
                original = original[-min_len:]
                recovered = recovered[-min_len:]
            if not np.allclose(original, recovered, rtol=1e-5, atol=1e-8, equal_nan=True):
                warnings.warn(
                    "The provided functions are not strictly inverse of each other. "
                    "If you are sure you want to proceed regardless, set 'check_inverse=False'.",
                    UserWarning,
                    stacklevel=2,
                )
                return

    def _detect_warmup(self, X: pl.DataFrame) -> int:
        """Detect warmup by counting leading NaN rows after applying func.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with "time" column.

        Returns
        -------
        int
            Number of leading NaN rows (observation horizon).

        Raises
        ------
        ValueError
            If ``func`` generates NaN values at discontinuous time stamps.

        """
        # Apply transformation
        X_t = self._apply_func(X, self.func, self.kw_args)

        # Auto-detect observation_horizon from NaN rows at the beginning
        data_cols = X_t.select(~cs.by_name("time"))

        # Find rows where all values are NaN. `is_nan` is defined for floats only, so
        # cast first: a Decimal column (a Postgres NUMERIC round-tripped through
        # parquet, say) otherwise raises InvalidOperationError. Nulls are already
        # caught by `is_null`, and a column that will not cast yields a null from
        # `is_nan`, so those fill to False rather than poisoning the row's mask.
        nan_mask = pl.all().cast(pl.Float64, strict=False).is_nan().fill_null(False)
        all_nan_mask = data_cols.select(pl.all_horizontal(pl.all().is_null() | nan_mask)).to_series()

        # Count consecutive NaN rows at the beginning: the first non-NaN row
        # index is the leading NaN count.
        non_nan = (~all_nan_mask).arg_true()
        obs_horizon = int(non_nan[0]) if len(non_nan) > 0 else len(all_nan_mask)

        # Verify NaN only at the beginning
        if obs_horizon > 0:
            # Check that remaining rows don't have all-NaN
            remaining_nan = all_nan_mask[obs_horizon:]
            if remaining_nan.any():
                raise ValueError(
                    "`func` generates NaN values at non-leading positions. "
                    "`func` is only allowed to generate NaN at the beginning of "
                    "X (contiguous leading rows)."
                )

        return obs_horizon

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> FunctionTransformer:
        """Fit transformer by validating X and auto-detecting warmup.

        The warmup is auto-detected by applying the function and counting
        leading NaN rows.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column (datetime) and one or
            more numeric columns.
        y : pl.DataFrame or None, default=None
            Ignored.  Present for API compatibility.
        **params : dict
            Accepted for signature compatibility and ignored: this transformer
            routes no metadata to nested estimators.

        Returns
        -------
        self
            The fitted transformer instance.

        """
        X = validate_transformer_data(self, X=X, reset=True)
        BaseActualTransformer.fit(self, X, y, **params)

        # Auto-detect warmup from NaN rows. BaseActualTransformer.fit ran
        # _update_X_observed while _observation_horizon was still 0, so the
        # buffer is empty; re-sync it now that the real horizon is known,
        # otherwise observe_transform would have no warmup history to prepend.
        self._observation_horizon = self._detect_warmup(X)
        self._update_X_observed(X)

        if self.check_inverse and self.func is not None and self.inverse_func is not None:
            self._check_inverse_transform(X)

        return self

    def fit_transform(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> pl.DataFrame:
        """Fit and transform, handling stateful functions with warmup.

        For stateful functions that produce NaN at the beginning (e.g., diff),
        this method fits the transformer (auto-detecting warmup if needed),
        transforms the data, and slices off the warmup rows.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column (datetime) and one or
            more numeric columns.
        y : pl.DataFrame or None, default=None
            Ignored.  Present for API compatibility.
        **params : dict
            Accepted for signature compatibility and ignored: this transformer
            routes no metadata to nested estimators.

        Returns
        -------
        pl.DataFrame
            Transformed time series with a ``"time"`` column and transformed
            value columns.

        """
        # Fit (includes warmup detection)
        self.fit(X, y, **params)

        # Apply transformation
        X_t = self._apply_func(X, self.func, self.kw_args)

        # Store initial rows for inverse transform and slice output
        if self._observation_horizon > 0:
            self._X_observed = X.head(self._observation_horizon)
            X_t = X_t.slice(self._observation_horizon)

        return X_t

    def _apply_func(
        self,
        X: pl.DataFrame,
        func: Callable | None,
        kw_args: dict | None,
        X_p: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Apply function to DataFrame, preserving structure.

        Parameters
        ----------
        X : pl.DataFrame
            Input DataFrame with "time" column.
        func : callable or None
            Function to apply.
        kw_args : dict or None
            Keyword arguments for function.
        X_p : pl.DataFrame or None, default=None
            Past observations (with or without ``"time"`` column).
            When not ``None``, the data portion is passed as the second
            positional argument to *func*.

        Returns
        -------
        pl.DataFrame
            Transformed DataFrame.

        """
        if func is None:
            return X

        # Separate time column
        time_col = X.select("time")
        data_cols = X.select(~cs.by_name("time"))

        # Apply function
        kwargs = kw_args if kw_args is not None else {}
        if X_p is not None:
            data_p = X_p.select(~cs.by_name("time")) if "time" in X_p.columns else X_p
            result = func(data_cols, data_p, **kwargs)
        else:
            result = func(data_cols, **kwargs)

        # Handle different return types
        if isinstance(result, pl.DataFrame):
            result_df = result
        elif isinstance(result, np.ndarray):
            # Reconstruct DataFrame from numpy array
            if result.ndim == 1:
                result = result.reshape(-1, 1)
            result_df = pl.DataFrame(result, schema=data_cols.columns)
        else:
            # Try to convert to DataFrame
            result_df = pl.DataFrame(result, schema=data_cols.columns)

        # Recombine with time
        return pl.concat([time_col, result_df], how="horizontal")

    def transform(self, X: pl.DataFrame, X_p: pl.DataFrame | None = None, **params) -> pl.DataFrame:
        """Transform X using the forward function.

        For stateful functions, provide past observations `X_p` to
        enable proper transformation without NaN at the beginning.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column (datetime) and one or
            more numeric columns.
        X_p : pl.DataFrame or None, default=None
            Past observations to prepend before transformation. If provided
            and ``_observation_horizon > 0``, the function is applied to the
            concatenation of X_p and X, then only the X portion is returned.
        **params : dict
            Accepted for signature compatibility and ignored: this transformer
            routes no metadata to nested estimators.

        Returns
        -------
        pl.DataFrame
            Transformed time series with a ``"time"`` column and transformed
            value columns.

        Notes
        -----
        When ``X_p`` is ``None`` and ``_observation_horizon > 0``, the first
        ``_observation_horizon`` rows (which contain NaN from insufficient
        history) are dropped, so the output is shorter than the input. Pass
        ``X_p`` to avoid this row loss.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_", "_observation_horizon"])
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=False)

        if X_p is not None and self._observation_horizon > 0:
            # Concatenate past observations with current data
            X_full = pl.concat([X_p.tail(self._observation_horizon), X], how="vertical")
            X_t_full = self._apply_func(X_full, self.func, self.kw_args)
            # Return only the X portion (after warmup)
            return X_t_full.slice(self._observation_horizon)

        X_t = self._apply_func(X, self.func, self.kw_args)

        # Drop warmup rows (contain NaN from insufficient history)
        if self._observation_horizon > 0:
            X_t = X_t[self._observation_horizon :]

        return X_t

    def inverse_transform(self, X_t: pl.DataFrame, X_p: pl.DataFrame | None = None, **params) -> pl.DataFrame:
        """Transform X using the inverse function.

        For stateful transforms (``_observation_horizon > 0``), the past
        observations ``X_p`` are passed as the second positional argument
        to ``inverse_func``.  The ``inverse_func`` signature should be
        ``inverse_func(X_t, X_p, **inv_kw_args)`` for stateful transforms
        and ``inverse_func(X_t, **inv_kw_args)`` for stateless transforms.

        Parameters
        ----------
        X_t : pl.DataFrame
            Transformed time series with "time" column.
        X_p : pl.DataFrame or None, default=None
            Past observations in original scale.  When ``None`` and
            ``_observation_horizon > 0``, the stored ``_X_observed`` is
            used instead.
        **params : dict
            Accepted for signature compatibility and ignored: this transformer
            routes no metadata to nested estimators.

        Returns
        -------
        pl.DataFrame
            Inverse-transformed time series, reversing the transformation
            applied by ``transform``.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_", "_observation_horizon"])

        # Validate input data (inverse=True skips strict schema check since
        # predictions from an ML model may have different dtypes, e.g. Float64
        # when the original target was Int64)
        X_t, X_p = validate_transformer_data(
            self,
            X=X_t,
            reset=False,
            inverse=True,
            X_p=X_p,
            observation_horizon=self._observation_horizon,
        )

        # For stateful transformers, pass past observations to inverse_func
        if self._observation_horizon > 0:
            # Use provided X_p or stored _X_observed
            X_observed = X_p if X_p is not None else getattr(self, "_X_observed", None)
            if X_observed is not None:
                return self._apply_func(
                    X_t,
                    self.inverse_func,
                    self.inv_kw_args,
                    X_p=X_observed.tail(self._observation_horizon),
                )

        return self._apply_func(X_t, self.inverse_func, self.inv_kw_args)

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
        input_features = _check_feature_names_in(self, input_features)

        if self.feature_names_out is None or self.feature_names_out == "one-to-one":
            return list(input_features)
        else:
            return list(np.asarray(self.feature_names_out(self, input_features), dtype=object))
