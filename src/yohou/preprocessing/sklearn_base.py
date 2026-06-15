"""Base sklearn transformer and scaler wrappers for polars DataFrames with time column preservation.

This module provides ``SklearnTransformer`` and ``SklearnScaler``, wrappers that integrate
sklearn transformers and scalers into the Yohou pipeline. It preserves polars DataFrame
structure and the "time" column while applying sklearn transformations to numeric
columns.
"""

import polars as pl
import polars.selectors as cs
from sklearn.base import TransformerMixin
from sklearn.utils._param_validation import HasMethods
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted
from sklearn_wrap.base import BaseClassWrapper, _fit_context

from yohou.base import BaseTransformer
from yohou.utils import validate_transformer_data

__all__ = ["SklearnScaler", "SklearnTransformer"]


def _transformer_has_inverse(self) -> bool:
    """Check if the wrapped transformer has inverse_transform method.

    This function is used with sklearn's available_if decorator to conditionally
    expose the inverse_transform method only when the underlying transformer has it.

    Note: Some transformers (e.g., SimpleImputer) have inverse_transform that only
    works under certain conditions.

    Parameters
    ----------
    self : SklearnTransformer
        The transformer wrapper instance.

    Returns
    -------
    bool
        True if the wrapped transformer has inverse_transform method.

    """
    # Check if fitted (instance_ exists)
    if not hasattr(self, "instance_"):
        # Before fit, check the default class
        default_class = getattr(self, "_estimator_default_class", None)
        if default_class is not None:
            return hasattr(default_class, "inverse_transform")
        # Fall back to checking if transformer param was provided
        transformer = getattr(self, "transformer", None)
        if transformer is not None:
            return hasattr(transformer, "inverse_transform")
        return False
    return hasattr(self.instance_, "inverse_transform")


class SklearnTransformer(BaseClassWrapper, BaseTransformer):
    """Wrapper to integrate sklearn transformers into the Yohou pipeline.

    Preserves the polars DataFrame structure and "time" column while applying
    sklearn transformations to numeric columns.

    This class can be used to:

    1. Wrap any sklearn-compatible transformer for use in yohou pipelines
    2. Serve as a base class for creating yohou transformer extensions

    Parameters
    ----------
    transformer : type, default=None
        The sklearn transformer class to wrap. Must be a subclass of
        ``sklearn.base.TransformerMixin``. If not provided,
        ``_estimator_default_class`` is used (subclasses define this).

    **params : dict
        Parameters passed to the underlying sklearn transformer constructor.
        See the documentation of the specific transformer for available parameters.

    Attributes
    ----------
    instance_ : TransformerMixin
        The fitted sklearn transformer instance (created by ``BaseClassWrapper``).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
    >>> from yohou.preprocessing import SklearnTransformer
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2024, 1, i) for i in range(1, 6)],
    ...     "value": [10.0, 20.0, 30.0, 40.0, 50.0],
    ... })
    >>> transformer = SklearnTransformer(transformer=SklearnStandardScaler, with_mean=True)
    >>> transformer.fit(X)  # doctest: +ELLIPSIS
    SklearnTransformer(...)
    >>> X_transformed = transformer.transform(X)
    >>> "time" in X_transformed.columns
    True

    See Also
    --------
    - [`StandardScaler`][yohou.preprocessing.sklearn_wrappers.StandardScaler] : Pre-configured wrapper for sklearn's StandardScaler.
    - [`MinMaxScaler`][yohou.preprocessing.sklearn_wrappers.MinMaxScaler] : Pre-configured wrapper for sklearn's MinMaxScaler.
    - [`RobustScaler`][yohou.preprocessing.sklearn_wrappers.RobustScaler] : Pre-configured wrapper for sklearn's RobustScaler.
    - [`MaxAbsScaler`][yohou.preprocessing.sklearn_wrappers.MaxAbsScaler] : Pre-configured wrapper for sklearn's MaxAbsScaler.

    """

    _estimator_name = "transformer"
    _estimator_base_class = TransformerMixin
    _estimator_default_class: type | None = None

    _parameter_constraints: dict = {
        "transformer": [HasMethods(["fit", "transform"]), None],
    }

    def __init__(self, transformer=None, **params):
        if transformer is not None:
            super().__init__(transformer=transformer, **params)
        else:
            super().__init__(**params)

    def __sklearn_tags__(self):
        """Get estimator tags.

        Override to ensure stateful=False before and after fit. The invertible tag
        is set dynamically based on whether the wrapped transformer has inverse_transform.

        Returns
        -------
        Tags
            Estimator tags with stateful=False and invertible based on underlying transformer.

        """
        tags = super().__sklearn_tags__()
        # transformers are always stateless (no memory / observation horizon)
        if tags.transformer_tags is not None:
            tags.transformer_tags.stateful = False
            # Invertible only if underlying transformer has inverse_transform
            tags.transformer_tags.invertible = _transformer_has_inverse(self)
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> "SklearnTransformer":
        """Fit the transformer to the data.

        Fits the underlying sklearn transformer on the training data,
        excluding the "time" column.

        Parameters
        ----------
        X : pl.DataFrame
            Input time series with "time" column.

        y : pl.DataFrame or None, default=None
            Target time series. Ignored and only present for API consistency.

        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            Fitted transformer.

        Raises
        ------
        ValueError
            If X does not have a "time" column.

        """
        # Validate input data (checks time column, schema, etc.)
        X = validate_transformer_data(self, X=X, reset=True)

        # Call parent fit (stores schema, memory, etc.)
        BaseTransformer.fit(self, X, y, **params)

        # Strip time column before fitting sklearn transformer
        X_no_time = X.select(~cs.by_name("time"))

        # Configure transformer output and fit (instance_ created by _fit_context)
        self.instance_.set_output(transform="polars")
        self.instance_.fit(X_no_time)

        return self

    def transform(self, X: pl.DataFrame, **params) -> pl.DataFrame:
        """Transform the input time series.

        Applies the learned scaling transformation to each feature.

        Parameters
        ----------
        X : pl.DataFrame
            Feature time series with "time" column.

        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Transformed time series with "time" column preserved.

        """
        check_is_fitted(self, ["instance_", "X_schema_", "feature_names_in_"])

        # Validate input data
        X = validate_transformer_data(self, X=X, reset=False, check_continuity=False)

        # Strip time column before transforming
        time = X.select(cs.by_name("time"))
        X_no_time = X.select(~cs.by_name("time"))

        # An empty observe window (e.g. observation_horizon == 0 during rewind)
        # yields a zero-row frame. sklearn's check_array rejects 0 samples, so
        # short-circuit: an empty input transforms to an empty output.
        if X_no_time.height == 0:
            return X

        # Apply scaling transformation
        X_scaled_no_time = self.instance_.transform(X_no_time)

        # Reattach time column to the scaled features
        return pl.concat([time, X_scaled_no_time], how="horizontal")

    @available_if(_transformer_has_inverse)
    def inverse_transform(self, X_t: pl.DataFrame, X_p: pl.DataFrame | None = None, **params) -> pl.DataFrame:
        """Apply the inverse transformer transformation to the data.

        This method is only available if the underlying sklearn transformer
        supports inverse_transform (e.g., StandardScaler, PowerTransformer).

        Reverts the scaling transformation, restoring the original data scale.

        Parameters
        ----------
        X_t : pl.DataFrame
            Scaled features with "time" column.

        X_p : pl.DataFrame or None, default=None
            Past observations for stateful inverse transformation. Ignored for
            sklearn wrappers since sklearn transformers are stateless.

        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Unscaled features with "time" column preserved.

        """
        check_is_fitted(self, ["instance_"])
        X_t, _ = validate_transformer_data(self, X=X_t, reset=False, inverse=True)

        # Strip time column before inverse transforming
        time = X_t.select(cs.by_name("time"))
        X_no_time = X_t.select(~cs.by_name("time"))

        # Apply inverse scaling transformation (returns numpy array)
        X_unscaled_array = self.instance_.inverse_transform(X_no_time)

        # Convert back to DataFrame with original column names
        X_unscaled_no_time = pl.DataFrame(X_unscaled_array, schema=X_no_time.columns, orient="row")

        # Reattach time column to the unscaled features
        return pl.concat([time, X_unscaled_no_time], how="horizontal")

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : list of str or None, default=None
            Input features. If None, uses feature names from fit.

        Returns
        -------
        list of str
            Transformed feature names (same as input features for transformers).

        """
        check_is_fitted(self, ["instance_"])
        return list(self.instance_.get_feature_names_out(input_features))


class SklearnScaler(SklearnTransformer):
    """Wrapper to integrate sklearn scalers into the Yohou pipeline.

    Preserves the polars DataFrame structure and "time" column while applying
    sklearn scaling transformations to numeric columns.

    This class can be used to:

    1. Wrap any sklearn-compatible scaler for use in yohou pipelines
    2. Serve as a base class for creating yohou scaler extensions

    Parameters
    ----------
    scaler : type, default=None
        The sklearn scaler class to wrap. Must be a subclass of
        ``sklearn.base.TransformerMixin``. If not provided,
        ``_estimator_default_class`` is used (subclasses define this).

    **params : dict
        Parameters passed to the underlying sklearn scaler constructor.
        See the documentation of the specific scaler for available parameters.

    Attributes
    ----------
    instance_ : TransformerMixin
        The fitted sklearn scaler instance (created by ``BaseClassWrapper``).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
    >>> from yohou.preprocessing import SklearnScaler
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2024, 1, i) for i in range(1, 6)],
    ...     "value": [10.0, 20.0, 30.0, 40.0, 50.0],
    ... })
    >>> scaler = SklearnScaler(scaler=SklearnStandardScaler, with_mean=True)
    >>> scaler.fit(X)  # doctest: +ELLIPSIS
    SklearnScaler(...)
    >>> X_scaled = scaler.transform(X)
    >>> "time" in X_scaled.columns
    True

    See Also
    --------
    - [`StandardScaler`][yohou.preprocessing.sklearn_wrappers.StandardScaler] : Pre-configured wrapper for sklearn's StandardScaler.
    - [`MinMaxScaler`][yohou.preprocessing.sklearn_wrappers.MinMaxScaler] : Pre-configured wrapper for sklearn's MinMaxScaler.
    - [`RobustScaler`][yohou.preprocessing.sklearn_wrappers.RobustScaler] : Pre-configured wrapper for sklearn's RobustScaler.
    - [`MaxAbsScaler`][yohou.preprocessing.sklearn_wrappers.MaxAbsScaler] : Pre-configured wrapper for sklearn's MaxAbsScaler.

    """

    _estimator_name = "scaler"
    _estimator_base_class = TransformerMixin
    _estimator_default_class: type | None = None

    _parameter_constraints: dict = {
        "scaler": [HasMethods(["fit", "transform"]), None],
    }

    _tags = {"stateful": False, "invertible": True}

    def __init__(self, scaler=None, **params):
        if scaler is not None:
            super().__init__(scaler=scaler, **params)
        else:
            super().__init__(**params)
