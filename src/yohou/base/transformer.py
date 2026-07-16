"""Base classes for time series transformers.

The transformer hierarchy has a private shared root, ``_BaseTransformer``, which
holds the sklearn scaffolding common to every transformer (parameter-constraint
merging, tag construction, ``fit_transform``, and the ``_fit``/``_transform``
hooks). Two public bases specialise it:

- ``BaseActualTransformer`` -- single-axis transformers that operate on a frame
  with a mandatory ``"time"`` column, supporting stateful windowing via
  ``observe``/``rewind``. This is the base for the whole preprocessing and
  stationarity catalogue.
- ``BaseForecastTransformer`` -- transformers over ``X_forecast`` frames (two
  time axes: ``"vintage_time"`` and ``"time"``); see
  ``yohou.base.forecast_transformer``.
"""

import abc
from typing import Any

import polars as pl
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from yohou.base.utils import _require_actual_memory_api
from yohou.utils import (
    Tags,
    validate_transformer_data,
)
from yohou.utils._compat import _fit_context

__all__ = [
    "BaseActualTransformer",
]


class _BaseTransformer(BaseEstimator, metaclass=abc.ABCMeta):
    """Private shared base for all yohou transformers.

    Holds the estimator scaffolding common to every transformer kind:
    parameter-constraint merging across the MRO, tag construction (including
    the ``kind`` tag), the ``fit_transform`` convenience method, and the
    ``_fit``/``_transform`` subclass hooks. Concrete behaviour (single-axis vs
    forecast-frame validation, memory management) lives in the public
    ``BaseActualTransformer`` and ``BaseForecastTransformer`` subclasses.

    Attributes
    ----------
    feature_names_in_ : list[str]
        Names of the non-index columns seen during ``fit``.
    n_features_in_ : int
        Number of non-index columns seen during ``fit``.
    X_schema_ : dict[str, pl.DataType]
        Column name to dtype mapping (non-index columns) seen during ``fit``.

    """

    _parameter_constraints: dict = {}

    # Fitted attributes shared by every transformer kind (set during fit()).
    feature_names_in_: list[str]
    n_features_in_: int
    X_schema_: dict[str, pl.DataType]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Merge parameter constraints from all classes in the MRO."""
        super().__init_subclass__(**kwargs)
        # Auto-merge _parameter_constraints from all classes in the MRO.
        merged: dict = {}
        for klass in reversed(cls.__mro__):
            own = klass.__dict__.get("_parameter_constraints")
            if own and isinstance(own, dict):
                merged.update(own)
        cls._parameter_constraints = merged

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        # Create Tags with transformer-specific defaults
        tags = Tags(estimator_type="transformer", requires_fit=True)

        if tags.transformer_tags is None:
            raise RuntimeError('Tags object has no transformer_tags; estimator_type="transformer" was not honoured.')

        # Default to non-invertible; subclasses set _tags = {"invertible": True}
        tags.transformer_tags.invertible = False

        # Merge class-level _tags dict (flat keys) into tag dataclasses.
        # Walk MRO in reverse so most-derived class wins.
        merged_tags: dict[str, Any] = {}
        for klass in reversed(type(self).__mro__):
            class_tags = klass.__dict__.get("_tags")
            if class_tags and isinstance(class_tags, dict):
                merged_tags.update(class_tags)

        if merged_tags:
            for key, value in merged_tags.items():
                if tags.transformer_tags is not None and hasattr(tags.transformer_tags, key):
                    setattr(tags.transformer_tags, key, value)
                elif tags.input_tags is not None and hasattr(tags.input_tags, key):
                    setattr(tags.input_tags, key, value)
                elif tags.target_tags is not None and hasattr(tags.target_tags, key):
                    setattr(tags.target_tags, key, value)
                elif hasattr(tags, key):
                    setattr(tags, key, value)

        return tags

    @abc.abstractmethod
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> "_BaseTransformer":
        """Fit the transformer to input data (implemented by the kind-specific base)."""

    @abc.abstractmethod
    def transform(self, X: pl.DataFrame, **params) -> pl.DataFrame:
        """Transform the input (implemented by the kind-specific base)."""

    def fit_transform(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> pl.DataFrame:
        """Fit the transformer and return transformed data.

        Equivalent to calling ``fit(X).transform(X)``.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series.
        y : pl.DataFrame or None, default=None
            Ignored.  Present for API compatibility.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Transformed time series.

        """
        self.fit(X, y, **params)
        return self.transform(X, **params)

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Subclass hook called at the end of ``fit()``.

        Override this to implement custom fitting logic.  The default
        implementation does nothing.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.
        y : pl.DataFrame or None
            Ignored.  Present for API compatibility.

        """

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Subclass hook called by ``transform()``.

        Override this to implement the core transformation logic. The
        input ``X`` has already been validated.

        Parameters
        ----------
        X : pl.DataFrame
            Validated input time series.

        Returns
        -------
        pl.DataFrame
            Transformed time series.

        """
        raise NotImplementedError(f"{type(self).__name__} must implement _transform() or override transform().")

    @abc.abstractmethod
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


class BaseActualTransformer(_BaseTransformer, metaclass=abc.ABCMeta):
    """Base class for single-axis (``"actual"``-kind) time series transformers.

    Yohou transformers operate on polars DataFrames with a mandatory
    ``"time"`` column and support stateful windowing via ``observe``,
    ``rewind``, ``observe_transform``, and ``rewind_transform`` methods.
    ``observe_transform`` observes new data and then transforms it, while
    ``rewind_transform`` rewinds state to a window and then transforms it.

    Attributes
    ----------
    feature_names_in_ : list[str]
        Names of the non-time columns seen during ``fit``.
    n_features_in_ : int
        Number of non-time columns seen during ``fit``.
    X_schema_ : dict[str, pl.DataType]
        Column name to dtype mapping seen during ``fit``.
    interval_ : str
        Detected time interval of the training data (e.g., ``"1d"``,
        ``"1h"``).

    Notes
    -----
    Transformers can be **stateful** (``observation_horizon > 0``) or
    **stateless** (``observation_horizon == 0``).  Stateful transformers
    maintain an internal memory buffer of the most recent
    ``observation_horizon`` rows, which is updated by ``observe()`` and
    replaced by ``rewind()`` (which sets the memory to the last
    ``observation_horizon`` rows of the provided data).

    All transformers preserve the ``"time"`` column through
    ``transform()`` and ``inverse_transform()``.

    See Also
    --------
    - [`BaseForecastTransformer`][yohou.base.forecast_transformer.BaseForecastTransformer] : Base class for X_forecast transformers.
    - [`BaseForecaster`][yohou.base.forecaster.BaseForecaster] : Base class for forecasters.
    - [`LagTransformer`][yohou.preprocessing.window.LagTransformer] : Creates lagged features from time series.
    - [`SeasonalDifferencing`][yohou.stationarity.transformers.SeasonalDifferencing] : Stateful seasonal differencing transformer.

    """

    _tags: dict = {"kind": "actual"}

    # Fitted attributes specific to single-axis transformers (set during fit()).
    _observation_horizon: int
    interval_: str

    @property
    def observation_horizon(self) -> int:
        """Get the number of time steps needed for stateful operations.

        The observation horizon defines how many recent observations the transformer
        needs to maintain in its memory.

        Returns
        -------
        int
            Number of time steps to retain.

        Raises
        ------
        NotFittedError
            If the transformer has not been fitted yet.

        """
        check_is_fitted(self, "_observation_horizon")
        return self._observation_horizon

    def _update_X_observed(self, X: pl.DataFrame) -> None:
        """Update stored observed data for stateful transformations.

        Parameters
        ----------
        X : pl.DataFrame
            Feature time series.

        """
        if self.observation_horizon > 0:
            if self.observation_horizon > len(X):
                raise ValueError(
                    f"Not enough input data to set the transformer memory: "
                    f"observation_horizon={self.observation_horizon} but X has {len(X)} rows."
                )

            self._X_observed = X[-self.observation_horizon :]
            self.observed_time_ = X["time"][-1]
        else:
            self._X_observed = X[:0]
            # For stateless transformers, only update observed_time_ if X is non-empty
            if len(X) > 0:
                self.observed_time_ = X["time"][-1]

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> "BaseActualTransformer":
        """Fit the transformer to input data.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column (datetime) and one or
            more numeric columns.
        y : pl.DataFrame or None, default=None
            Ignored.  Present for API compatibility.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted transformer instance.

        Raises
        ------
        ValueError
            If ``X`` does not have a ``"time"`` column, or if time intervals
            are inconsistent.

        """
        # Validate inputs and set fitted attributes (feature_names_in_, n_features_in_, X_schema_, interval_)
        X = validate_transformer_data(self, X=X, reset=True)

        if not hasattr(self, "_observation_horizon"):
            self._observation_horizon = 0

        self._fit(X, y)

        # Sync _observation_horizon with the property after _fit() completes.
        # This handles @property overrides that compute from constructor params.
        self._observation_horizon = self.observation_horizon

        self._update_X_observed(X)

        return self

    def rewind(self, X: pl.DataFrame) -> "BaseActualTransformer":
        """Rewind internal memory to the last ``observation_horizon`` rows.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column (datetime) and one or
            more numeric columns.

        Returns
        -------
        self
            The transformer with internal memory rewound to the last
            ``observation_horizon`` rows of the provided data.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the transformer has not been fitted yet.
        ValueError
            If the transformer reports ``kind == "forecast"``.

        """
        _require_actual_memory_api(self, "rewind")
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        # Validate against fitted state (no continuity check - rewind sets new window)
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=False)

        self._update_X_observed(X)

        return self

    def observe(self, X: pl.DataFrame) -> "BaseActualTransformer":
        """Observe new data and update internal memory.

        Extends the internal memory buffer with new observations, then
        calls ``rewind()`` to maintain the fixed ``observation_horizon``
        window.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column (datetime) and one or
            more numeric columns containing new observations.

        Returns
        -------
        self
            The transformer with updated internal memory from new
            observations.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the transformer has not been fitted.
        ValueError
            If ``X`` contains overlapping data with existing observations, or if
            the transformer reports ``kind == "forecast"``.

        """
        _require_actual_memory_api(self, "observe")
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        # Validate against fitted state (includes continuity check)
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=True)

        self.rewind(pl.concat([self._X_observed, X]))

        return self

    def transform(self, X: pl.DataFrame, **params) -> pl.DataFrame:
        """Transform the input time series.

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
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=False)
        return self._transform(X)

    def inverse_transform(self, X_t: pl.DataFrame, X_p: pl.DataFrame | None = None) -> pl.DataFrame:
        """Inverse-transform the data back to the original space.

        Parameters
        ----------
        X_t : pl.DataFrame
            Transformed time series to invert.
        X_p : pl.DataFrame or None, default=None
            Past observations needed by stateful transformers.

        Returns
        -------
        pl.DataFrame
            Data in the original (pre-transform) space.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        return self._inverse_transform(X_t, X_p=X_p)

    def _inverse_transform(self, X_t: pl.DataFrame, X_p: pl.DataFrame | None = None) -> pl.DataFrame:
        """Subclass hook called by ``inverse_transform()``.

        Override this to implement the inverse transformation logic.

        Parameters
        ----------
        X_t : pl.DataFrame
            Transformed time series to invert.
        X_p : pl.DataFrame or None
            Past observations needed by stateful transformers.

        Returns
        -------
        pl.DataFrame
            Data in the original space.

        """
        raise NotImplementedError(f"{type(self).__name__} does not support inverse_transform.")

    def observe_transform(self, X: pl.DataFrame, **params) -> pl.DataFrame:
        """Transform using pre-existing memory, then observe state.

        Performs a stateful transformation by concatenating stored
        observations with the new input, applying the transformation,
        and then updating the internal state.

        Transforms using pre-existing memory first, then updates state with
        ``observe(X)``. This is NOT equivalent to ``observe(X); transform(X)``,
        since the transform uses memory from before the observation.

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

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the transformer has not been fitted yet.
        ValueError
            If ``X`` has invalid structure or non-contiguous time index.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        # Validate against fitted state (includes continuity check)
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=True)

        # Route all params to transform only (observe is memory management)
        if self.observation_horizon > 0:
            X_full = pl.concat([self._X_observed, X])
            X_t = self.transform(X_full, **params)
            X_t = X_t[-len(X) :]
        else:
            X_t = self.transform(X, **params)

        self.observe(X)

        return X_t

    def rewind_transform(self, X: pl.DataFrame, **params) -> pl.DataFrame:
        """Transform the input and rewind state (stateless transform).

        Applies the transformation to the full input and then rewinds
        internal state. Stateful subclasses typically discard the first
        ``observation_horizon`` rows inside ``_transform()``, so the result
        usually has ``len(X) - observation_horizon`` rows, but the exact count
        depends on the transformer implementation.

        Equivalent to calling ``rewind(X)`` then ``transform(X)``.

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
            Transformed time series with the first ``observation_horizon``
            rows discarded (by ``transform()``).

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the transformer has not been fitted yet.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        # Validate against fitted state (no continuity check - rewind sets new window)
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=False)

        # Apply transformation without using pre-existing observations.
        # transform() already drops the first observation_horizon rows for
        # stateful transformers, so no additional slicing is needed.
        X_t = self.transform(X, **params)

        # Rewind internal state with the input
        self.rewind(X)

        return X_t
