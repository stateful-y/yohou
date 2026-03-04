"""Base class for time series transformers."""

import abc

import polars as pl
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from yohou.utils import (
    Tags,
    validate_transformer_data,
)
from yohou.utils._compat import _fit_context

__all__ = [
    "BaseTransformer",
]


class BaseTransformer(BaseEstimator, metaclass=abc.ABCMeta):
    """Base class for time series transformers.

    Yohou transformers operate on polars DataFrames with a mandatory
    ``"time"`` column and support stateful windowing via ``observe``,
    ``rewind``, and ``observe_transform`` methods.

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
    truncated by ``rewind()``.

    All transformers preserve the ``"time"`` column through
    ``transform()`` and ``inverse_transform()``.

    See Also
    --------
    BaseForecaster : Base class for forecasters.
    LagTransformer : Creates lagged features from time series.
    SeasonalDifferencing : Stateful seasonal differencing transformer.

    """

    _parameter_constraints: dict = {}

    # Fitted attributes (set during fit())
    _observation_horizon: int
    feature_names_in_: list[str]
    n_features_in_: int
    X_schema_: dict[str, pl.DataType]
    interval_: str

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        # Create Tags with transformer-specific defaults
        tags = Tags(estimator_type="transformer", requires_fit=True)

        assert tags.transformer_tags is not None

        # Auto-detect invertible: check if inverse_transform method exists
        tags.transformer_tags.invertible = hasattr(self, "inverse_transform")

        return tags

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
                raise ValueError("Not enough input data to set the transformer memory.")

            self._X_observed = X[-self.observation_horizon :]
            self.observed_time_ = X["time"][-1]
        else:
            self._X_observed = X[:0]
            # For stateless transformers, only update observed_time_ if X is non-empty
            if len(X) > 0:
                self.observed_time_ = X["time"][-1]

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> "BaseTransformer":
        """Fit the transformer to input data.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column (datetime) and one or
            more numeric columns.
        y : pl.DataFrame or None, default=None
            Ignored.  Present for API compatibility with yohou pipelines.
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

        # Router transformers would call process_routing() in their fit function

        self._update_X_observed(X)

        return self

    def fit_transform(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> pl.DataFrame:
        """Fit the transformer and return transformed data.

        Equivalent to calling ``fit(X).transform(X)``.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with a ``"time"`` column (datetime) and one or
            more numeric columns.
        y : pl.DataFrame or None, default=None
            Ignored.  Present for API compatibility with yohou pipelines.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Transformed time series with a ``"time"`` column and transformed
            value columns.

        Raises
        ------
        ValueError
            If ``X`` is missing the ``"time"`` column or contains invalid data.

        """
        self.fit(X, y, **params)
        return self.transform(X, **params)

    def rewind(self, X: pl.DataFrame) -> "BaseTransformer":
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

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        # Validate against fitted state (no continuity check - rewind sets new window)
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=False)

        self._update_X_observed(X)

        return self

    def observe(self, X: pl.DataFrame) -> "BaseTransformer":
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
            If ``X`` contains overlapping data with existing observations.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        # Validate against fitted state (includes continuity check)
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=True)

        self.rewind(pl.concat([self._X_observed, X]))

        return self

    @abc.abstractmethod
    def transform(self, X: pl.DataFrame, **params) -> pl.DataFrame:
        """Transform the input time series (stateless).

        Performs a stateless transformation, treating the input independently
        without using any pre-existing observations from the transformer's
        memory.

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

    def observe_transform(self, X: pl.DataFrame, **params) -> pl.DataFrame:
        """Transform using pre-existing memory, then observe state.

        Performs a stateful transformation by concatenating stored
        observations with the new input, applying the transformation,
        and then updating the internal state.

        Equivalent to calling ``observe(X)`` then ``transform(X)``, but
        uses pre-existing memory for the transform.

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
        internal state.  Because ``transform()`` already drops the first
        ``observation_horizon`` rows for stateful transformers, the result
        has ``len(X) - observation_horizon`` rows.

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
