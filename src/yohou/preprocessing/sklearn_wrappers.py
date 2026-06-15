"""Pre-configured sklearn transformer and scaler wrappers.

This module provides pre-configured transformer classes (Normalizer, PolynomialFeatures,
PowerTransformer, QuantileTransformer, SplineTransformer) and scaler classes (StandardScaler,
MinMaxScaler, RobustScaler, MaxAbsScaler) that default to their sklearn counterparts but can
accept any sklearn-compatible transformer class.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MaxAbsScaler as sklearn_MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler as sklearn_MinMaxScaler
from sklearn.preprocessing import Normalizer as sklearn_Normalizer
from sklearn.preprocessing import PolynomialFeatures as sklearn_PolynomialFeatures
from sklearn.preprocessing import PowerTransformer as sklearn_PowerTransformer
from sklearn.preprocessing import QuantileTransformer as sklearn_QuantileTransformer
from sklearn.preprocessing import RobustScaler as sklearn_RobustScaler
from sklearn.preprocessing import SplineTransformer as sklearn_SplineTransformer
from sklearn.preprocessing import StandardScaler as sklearn_StandardScaler
from sklearn.utils.validation import check_is_fitted

from yohou.preprocessing.sklearn_base import SklearnScaler, SklearnTransformer
from yohou.utils._compat import _filter_estimator_params

__all__ = [
    "MaxAbsScaler",
    "MinMaxScaler",
    "Normalizer",
    "PolynomialFeatures",
    "PowerTransformer",
    "QuantileTransformer",
    "RobustScaler",
    "SplineTransformer",
    "StandardScaler",
]


class StandardScaler(SklearnScaler):
    """Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample ``x`` is calculated as::

        z = (x - u) / s

    where ``u`` is the mean of the training samples or zero if ``with_mean=False``,
    and ``s`` is the standard deviation of the training samples or one if
    ``with_std=False``.

    Centering and scaling happen independently on each feature by computing the
    relevant statistics on the samples in the training set. Mean and standard
    deviation are then stored to be used on later data using ``transform()``.

    Standardization of a dataset is a common requirement for many machine learning
    estimators: they might behave badly if the individual features do not more or
    less look like standard normally distributed data (e.g. Gaussian with 0 mean
    and unit variance).

    ``StandardScaler`` is sensitive to outliers, and the features may scale
    differently from each other in the presence of outliers. For outlier-robust
    scaling, use ``RobustScaler`` instead.

    This is a Yohou wrapper that preserves the polars DataFrame structure and
    "time" column.

    Parameters
    ----------
    with_mean : bool, default=True
        If True, center the data before scaling.

    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently, unit standard
        deviation).

    Attributes
    ----------
    instance_ : sklearn.preprocessing.StandardScaler
        The fitted sklearn StandardScaler instance.

    scale_ : ndarray of shape (n_features,) or None
        Per feature relative scaling of the data to achieve zero mean and unit
        variance. Equal to ``None`` when ``with_std=False``.

    mean_ : ndarray of shape (n_features,) or None
        The mean value for each feature in the training set. Equal to ``None``
        when ``with_mean=False`` and ``with_std=False``.

    var_ : ndarray of shape (n_features,) or None
        The variance for each feature in the training set. Equal to ``None``
        when ``with_mean=False`` and ``with_std=False``.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.preprocessing import StandardScaler
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2024, 1, i) for i in range(1, 6)],
    ...     "value": [10.0, 20.0, 30.0, 40.0, 50.0],
    ... })
    >>> scaler = StandardScaler()
    >>> scaler.fit(X)  # doctest: +ELLIPSIS
    StandardScaler(...)
    >>> X_scaled = scaler.transform(X)
    >>> # Values are standardized (mean=0, std=1)
    >>> round(X_scaled["value"].mean(), 10)
    0.0

    See Also
    --------
    - [`MinMaxScaler`][yohou.preprocessing.sklearn_wrappers.MinMaxScaler] : Scale features to a given range.
    - [`RobustScaler`][yohou.preprocessing.sklearn_wrappers.RobustScaler] : Scale using statistics robust to outliers.

    """

    _estimator_default_class = sklearn_StandardScaler

    def __init__(self, with_mean=True, with_std=True, copy=True, **kwargs):
        super().__init__(with_mean=with_mean, with_std=with_std, copy=copy, **kwargs)

    @property
    def scale_(self) -> np.ndarray:
        """Per feature relative scaling of the data."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.scale_

    @property
    def mean_(self) -> np.ndarray:
        """The mean value for each feature in the training set."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.mean_

    @property
    def var_(self) -> np.ndarray:
        """The variance for each feature in the training set."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.var_


class MinMaxScaler(SklearnScaler):
    """Transform features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such that it
    is in the given range on the training set, e.g. between zero and one.

    The transformation is given by::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where ``min, max = feature_range``.

    This transformation is often used as an alternative to zero mean, unit
    variance scaling.

    ``MinMaxScaler`` doesn't reduce the effect of outliers, but it linearly
    scales them down into a fixed range, where the largest occurring data point
    corresponds to the maximum value and the smallest one corresponds to the
    minimum value.

    This is a Yohou wrapper that preserves the polars DataFrame structure and
    "time" column.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    clip : bool, default=False
        Set to True to clip transformed values of held-out data to the provided
        ``feature_range``.

    Attributes
    ----------
    instance_ : sklearn.preprocessing.MinMaxScaler
        The fitted sklearn MinMaxScaler instance.

    min_ : ndarray of shape (n_features,)
        Per feature adjustment for minimum.

    scale_ : ndarray of shape (n_features,)
        Per feature relative scaling of the data.

    data_min_ : ndarray of shape (n_features,)
        Per feature minimum seen in the data.

    data_max_ : ndarray of shape (n_features,)
        Per feature maximum seen in the data.

    data_range_ : ndarray of shape (n_features,)
        Per feature range ``(data_max_ - data_min_)`` seen in the data.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.preprocessing import MinMaxScaler
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2024, 1, i) for i in range(1, 6)],
    ...     "value": [10.0, 20.0, 30.0, 40.0, 50.0],
    ... })
    >>> scaler = MinMaxScaler()
    >>> scaler.fit(X)  # doctest: +ELLIPSIS
    MinMaxScaler(...)
    >>> X_scaled = scaler.transform(X)
    >>> # Values are scaled to [0, 1]
    >>> X_scaled["value"].min()
    0.0
    >>> X_scaled["value"].max()
    1.0

    See Also
    --------
    - [`StandardScaler`][yohou.preprocessing.sklearn_wrappers.StandardScaler] : Standardize features by removing mean and scaling to unit variance.
    - [`MaxAbsScaler`][yohou.preprocessing.sklearn_wrappers.MaxAbsScaler] : Scale each feature by its maximum absolute value.

    """

    _estimator_default_class = sklearn_MinMaxScaler

    def __init__(self, feature_range=(0, 1), copy=True, clip=False, **kwargs):
        super().__init__(feature_range=feature_range, copy=copy, clip=clip, **kwargs)

    @property
    def min_(self) -> np.ndarray:
        """Per feature adjustment for minimum."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.min_

    @property
    def scale_(self) -> np.ndarray:
        """Per feature relative scaling of the data."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.scale_

    @property
    def data_min_(self) -> np.ndarray:
        """Per feature minimum seen in the data."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.data_min_

    @property
    def data_max_(self) -> np.ndarray:
        """Per feature maximum seen in the data."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.data_max_

    @property
    def data_range_(self) -> np.ndarray:
        """Per feature range seen in the data."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.data_range_


class RobustScaler(SklearnScaler):
    """Scale features using statistics that are robust to outliers.

    This Scaler removes the median and scales the data according to the
    quantile range (defaults to IQR: Interquartile Range). The IQR is the range
    between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).

    Centering and scaling happen independently on each feature by computing the
    relevant statistics on the samples in the training set. Median and
    interquartile range are then stored to be used on later data using the
    ``transform()`` method.

    Standardization of a dataset is a common preprocessing for many machine
    learning estimators. Typically this is done by removing the mean and scaling
    to unit variance. However, outliers can often influence the sample mean /
    variance in a negative way. In such cases, using the median and the
    interquartile range often give better results.

    This is a Yohou wrapper that preserves the polars DataFrame structure and
    "time" column.

    Parameters
    ----------
    with_centering : bool, default=True
        If True, center the data before scaling.

    with_scaling : bool, default=True
        If True, scale the data to interquartile range.

    quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0, default=(25.0, 75.0)
        Quantile range used to calculate ``scale_``. By default this is equal
        to the IQR, i.e., ``q_min`` is the first quantile and ``q_max`` is the
        third quantile.

    unit_variance : bool, default=False
        If True, scale data so that normally distributed features have a
        variance of 1.

    Attributes
    ----------
    instance_ : sklearn.preprocessing.RobustScaler
        The fitted sklearn RobustScaler instance.

    center_ : array of floats
        The median value for each feature in the training set.

    scale_ : array of floats
        The (scaled) interquartile range for each feature in the training set.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.preprocessing import RobustScaler
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2024, 1, i) for i in range(1, 6)],
    ...     "value": [10.0, 20.0, 30.0, 100.0, 50.0],  # 100 is an outlier
    ... })
    >>> scaler = RobustScaler()
    >>> scaler.fit(X)  # doctest: +ELLIPSIS
    RobustScaler(...)
    >>> X_scaled = scaler.transform(X)
    >>> # Median-centered and scaled by IQR
    >>> "time" in X_scaled.columns
    True

    See Also
    --------
    - [`StandardScaler`][yohou.preprocessing.sklearn_wrappers.StandardScaler] : Scale using mean and standard deviation (sensitive to outliers).

    """

    _estimator_default_class = sklearn_RobustScaler

    def __init__(
        self,
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        copy=True,
        unit_variance=False,
        **kwargs,
    ):
        super().__init__(
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=quantile_range,
            copy=copy,
            unit_variance=unit_variance,
            **kwargs,
        )

    @property
    def center_(self) -> np.ndarray:
        """The median value for each feature in the training set."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.center_

    @property
    def scale_(self) -> np.ndarray:
        """The (scaled) interquartile range for each feature."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.scale_


class MaxAbsScaler(SklearnScaler):
    """Scale each feature by its maximum absolute value.

    This estimator scales and translates each feature individually such that
    the maximal absolute value of each feature in the training set will be 1.0.
    It does not shift/center the data, and thus does not destroy any sparsity.

    ``MaxAbsScaler`` doesn't reduce the effect of outliers; it only linearly
    scales them down.

    This is a Yohou wrapper that preserves the polars DataFrame structure and
    "time" column.

    Parameters
    ----------
    clip : bool, default=False
        Set to True to clip transformed values of held-out data to [-1, 1].

    Attributes
    ----------
    instance_ : sklearn.preprocessing.MaxAbsScaler
        The fitted sklearn MaxAbsScaler instance.

    scale_ : ndarray of shape (n_features,)
        Per feature relative scaling of the data.

    max_abs_ : ndarray of shape (n_features,)
        Per feature maximum absolute value.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.preprocessing import MaxAbsScaler
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2024, 1, i) for i in range(1, 6)],
    ...     "value": [-50.0, -25.0, 0.0, 25.0, 50.0],
    ... })
    >>> scaler = MaxAbsScaler()
    >>> scaler.fit(X)  # doctest: +ELLIPSIS
    MaxAbsScaler(...)
    >>> X_scaled = scaler.transform(X)
    >>> # Values scaled to [-1, 1] range
    >>> X_scaled["value"].max()
    1.0
    >>> X_scaled["value"].min()
    -1.0

    See Also
    --------
    - [`MinMaxScaler`][yohou.preprocessing.sklearn_wrappers.MinMaxScaler] : Scale features to a given range.
    - [`StandardScaler`][yohou.preprocessing.sklearn_wrappers.StandardScaler] : Standardize features by removing mean and scaling to unit variance.

    """

    _estimator_default_class = sklearn_MaxAbsScaler

    def __init__(self, copy=True, clip=False, **kwargs):
        params = _filter_estimator_params(sklearn_MaxAbsScaler, {"copy": copy, "clip": clip})
        super().__init__(**params, **kwargs)

    @property
    def scale_(self) -> np.ndarray:
        """Per feature relative scaling of the data."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.scale_

    @property
    def max_abs_(self) -> np.ndarray:
        """Per feature maximum absolute value."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.max_abs_


class Normalizer(SklearnTransformer):
    """Normalize samples individually to unit norm.

    Each sample (i.e. each row of the data matrix) with at least one non-zero
    component is rescaled independently of other samples so that its norm
    (l1, l2 or max) equals one.

    This normalizer can be useful as a preprocessing step for classifiers or
    other algorithms that rely on the angle between vectors, such as cosine
    similarity for document classification.

    This is a Yohou wrapper that preserves the polars DataFrame structure and
    "time" column.

    Parameters
    ----------
    norm : {'l1', 'l2', 'max'}, default='l2'
        The norm to use to normalize each non zero sample. If norm='max' is
        used, values will be rescaled by the maximum of the absolute values.

    Attributes
    ----------
    instance_ : sklearn.preprocessing.Normalizer
        The fitted sklearn Normalizer instance.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.preprocessing import Normalizer
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2024, 1, i) for i in range(1, 4)],
    ...     "a": [1.0, 2.0, 3.0],
    ...     "b": [2.0, 4.0, 6.0],
    ... })
    >>> normalizer = Normalizer(norm="l2")
    >>> normalizer.fit(X)  # doctest: +ELLIPSIS
    Normalizer(...)
    >>> X_norm = normalizer.transform(X)
    >>> # Each row normalized to unit L2 norm
    >>> "time" in X_norm.columns
    True

    See Also
    --------
    - [`StandardScaler`][yohou.preprocessing.sklearn_wrappers.StandardScaler] : Standardize features by removing mean and scaling to unit variance.

    """

    _estimator_default_class = sklearn_Normalizer

    def __init__(self, norm="l2", copy=True, **kwargs):
        super().__init__(norm=norm, copy=copy, **kwargs)


class PolynomialFeatures(SklearnTransformer):
    """Generate polynomial and interaction features.

    Generate a new feature matrix consisting of all polynomial combinations of
    the features with degree less than or equal to the specified degree.

    For example, if an input sample is two dimensional and of the form
    ``[a, b]``, the degree-2 polynomial features are ``[1, a, b, a^2, ab, b^2]``.

    This is a Yohou wrapper that preserves the polars DataFrame structure and
    "time" column.

    Parameters
    ----------
    degree : int or tuple (min_degree, max_degree), default=2
        If a single int is given, it specifies the maximal degree of the
        polynomial features. If a tuple ``(min_degree, max_degree)`` is passed,
        then ``min_degree`` is the minimum and ``max_degree`` is the maximum
        polynomial degree of the generated features.

    interaction_only : bool, default=False
        If True, only interaction features are produced: features that are
        products of at most ``degree`` *distinct* input features.

    include_bias : bool, default=True
        If True, then include a bias column, the feature in which all
        polynomial powers are zero.

    order : {'C', 'F'}, default='C'
        Order of output array in the dense case. 'F' order is faster to compute,
        but may slow down subsequent estimators.

    Attributes
    ----------
    instance_ : sklearn.preprocessing.PolynomialFeatures
        The fitted sklearn PolynomialFeatures instance.

    n_features_in_ : int
        Number of features seen during fit.

    n_output_features_ : int
        The total number of polynomial output features.

    powers_ : ndarray of shape (n_output_features_, n_features_in_)
        Exponent for each of the inputs in the output.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.preprocessing import PolynomialFeatures
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2024, 1, i) for i in range(1, 4)],
    ...     "a": [1.0, 2.0, 3.0],
    ...     "b": [2.0, 3.0, 4.0],
    ... })
    >>> poly = PolynomialFeatures(degree=2, include_bias=False)
    >>> poly.fit(X)  # doctest: +ELLIPSIS
    PolynomialFeatures(...)
    >>> X_poly = poly.transform(X)
    >>> # Generates: a, b, a^2, a*b, b^2
    >>> len(X_poly.columns) > len(X.columns)
    True

    See Also
    --------
    - [`SplineTransformer`][yohou.preprocessing.sklearn_wrappers.SplineTransformer] : Generate spline basis features.

    """

    _estimator_default_class = sklearn_PolynomialFeatures

    def __init__(self, degree=2, interaction_only=False, include_bias=True, order="C", **kwargs):
        super().__init__(
            degree=degree, interaction_only=interaction_only, include_bias=include_bias, order=order, **kwargs
        )

    @property
    def n_output_features_(self) -> int:
        """The total number of polynomial output features."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.n_output_features_

    @property
    def powers_(self) -> np.ndarray:
        """Exponent for each of the inputs in the output."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.powers_


class PowerTransformer(SklearnTransformer):
    """Apply a power transform featurewise to make data more Gaussian-like.

    Power transforms are a family of parametric, monotonic transformations that
    are applied to make data more Gaussian-like. This is useful for modeling
    issues related to heteroscedasticity (non-constant variance), or other
    situations where normality is desired.

    Currently, PowerTransformer supports the Box-Cox transform and the
    Yeo-Johnson transform. The optimal parameter for stabilizing variance and
    minimizing skewness is estimated through maximum likelihood.

    Box-Cox requires input data to be strictly positive, while Yeo-Johnson
    supports both positive and negative data.

    This is a Yohou wrapper that preserves the polars DataFrame structure and
    "time" column.

    Parameters
    ----------
    method : {'yeo-johnson', 'box-cox'}, default='yeo-johnson'
        The power transform method. Available methods are:

        - 'yeo-johnson': Works with positive and negative values.
        - 'box-cox': Only works with strictly positive values.

    standardize : bool, default=True
        Set to True to apply zero-mean, unit-variance normalization to the
        transformed output.

    Attributes
    ----------
    instance_ : sklearn.preprocessing.PowerTransformer
        The fitted sklearn PowerTransformer instance.

    lambdas_ : ndarray of float, shape (n_features,)
        The parameters of the power transform for the selected features.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.preprocessing import PowerTransformer
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2024, 1, i) for i in range(1, 6)],
    ...     "value": [1.0, 4.0, 9.0, 16.0, 25.0],  # Skewed data
    ... })
    >>> pt = PowerTransformer(method="yeo-johnson")
    >>> pt.fit(X)  # doctest: +ELLIPSIS
    PowerTransformer(...)
    >>> X_transformed = pt.transform(X)
    >>> # Data is transformed to be more Gaussian-like
    >>> "time" in X_transformed.columns
    True
    >>> # Can be inverted
    >>> X_inv = pt.inverse_transform(X_transformed)
    >>> abs(X_inv["value"][0] - 1.0) < 1e-10
    True

    See Also
    --------
    - [`QuantileTransformer`][yohou.preprocessing.sklearn_wrappers.QuantileTransformer] : Transform features using quantiles information.

    """

    _estimator_default_class = sklearn_PowerTransformer

    def __init__(self, method="yeo-johnson", standardize=True, copy=True, **kwargs):
        super().__init__(method=method, standardize=standardize, copy=copy, **kwargs)

    @property
    def lambdas_(self) -> np.ndarray:
        """The parameters of the power transform for the selected features."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.lambdas_


class QuantileTransformer(SklearnTransformer):
    """Transform features using quantiles information.

    This method transforms the features to follow a uniform or a normal
    distribution. Therefore, for a given feature, this transformation tends
    to spread out the most frequent values. It also reduces the impact of
    (marginal) outliers: this is therefore a robust preprocessing scheme.

    The transformation is applied on each feature independently. First an
    estimate of the cumulative distribution function of a feature is used to
    map the original values to a uniform distribution. The obtained values are
    then mapped to the desired output distribution using the associated
    quantile function.

    This is a Yohou wrapper that preserves the polars DataFrame structure and
    "time" column.

    Parameters
    ----------
    n_quantiles : int, default=1000 or n_samples
        Number of quantiles to be computed. It corresponds to the number of
        landmarks used to discretize the cumulative distribution function.
        If n_quantiles is larger than the number of samples, n_quantiles is set
        to the number of samples.

    output_distribution : {'uniform', 'normal'}, default='uniform'
        Marginal distribution for the transformed data. The choices are
        'uniform' (default) or 'normal'.

    ignore_implicit_zeros : bool, default=False
        Only applies to sparse matrices. If True, the sparse entries of the
        matrix are discarded to compute the quantile statistics.

    subsample : int, default=10_000
        Maximum number of samples used to estimate the quantiles for
        computational efficiency.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for subsampling and smoothing
        noise.

    Attributes
    ----------
    instance_ : sklearn.preprocessing.QuantileTransformer
        The fitted sklearn QuantileTransformer instance.

    n_quantiles_ : int
        The actual number of quantiles used to discretize the cumulative
        distribution function.

    quantiles_ : ndarray of shape (n_quantiles, n_features)
        The values corresponding to the quantiles of reference.

    references_ : ndarray of shape (n_quantiles,)
        Quantiles of references.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.preprocessing import QuantileTransformer
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2024, 1, i) for i in range(1, 11)],
    ...     "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0],  # 100 is outlier
    ... })
    >>> qt = QuantileTransformer(n_quantiles=10, output_distribution="uniform")
    >>> qt.fit(X)  # doctest: +ELLIPSIS
    QuantileTransformer(...)
    >>> X_transformed = qt.transform(X)
    >>> # Outlier impact is reduced
    >>> "time" in X_transformed.columns
    True

    See Also
    --------
    - [`PowerTransformer`][yohou.preprocessing.sklearn_wrappers.PowerTransformer] : Apply a power transform to make data more Gaussian-like.

    """

    _estimator_default_class = sklearn_QuantileTransformer

    def __init__(
        self,
        n_quantiles=1000,
        output_distribution="uniform",
        ignore_implicit_zeros=False,
        subsample=10_000,
        random_state=None,
        copy=True,
        **kwargs,
    ):
        super().__init__(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            ignore_implicit_zeros=ignore_implicit_zeros,
            subsample=subsample,
            random_state=random_state,
            copy=copy,
            **kwargs,
        )

    @property
    def n_quantiles_(self) -> int:
        """The actual number of quantiles used."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.n_quantiles_

    @property
    def quantiles_(self) -> np.ndarray:
        """The values corresponding to the quantiles of reference."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.quantiles_

    @property
    def references_(self) -> np.ndarray:
        """Quantiles of references."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.references_


class SplineTransformer(SklearnTransformer):
    """Generate univariate B-spline bases for features.

    Generate a new feature matrix consisting of ``n_splines=n_knots + degree - 1``
    spline basis functions (B-splines) of polynomial order ``degree`` for each
    feature.

    This is a Yohou wrapper that preserves the polars DataFrame structure and
    "time" column.

    Parameters
    ----------
    n_knots : int, default=5
        Number of knots of the splines if ``knots`` equals one of
        {'uniform', 'quantile'}. Must be larger than or equal to 2.

    degree : int, default=3
        The polynomial degree of the spline basis. Must be a non-negative
        integer.

    knots : {'uniform', 'quantile'} or array-like of shape (n_knots, n_features), default='uniform'
        Set knot positions such that first and last knots are the 1st percentile
        and 99th percentile of the data respectively.

    extrapolation : {'error', 'constant', 'linear', 'continue', 'periodic'}, default='constant'
        If 'error', values outside the min and max values of the training
        features will raise an error.

    include_bias : bool, default=True
        If False, then the last spline element inside the data range of each
        feature is dropped. Set to True (default) to keep it and include a
        redundant basis column.

    order : {'C', 'F'}, default='C'
        Order of output array.

    sparse_output : bool, default=False
        If True, transform will return sparse CSC format. Otherwise,
        transform will return dense array.

    handle_missing : {'error', 'missing-as-zero'}, default='error'
        How to handle missing values during transform. If 'error', a ValueError
        is raised if missing values are present. If 'missing-as-zero', missing
        values are treated as zeros in the spline basis.

    Attributes
    ----------
    instance_ : sklearn.preprocessing.SplineTransformer
        The fitted sklearn SplineTransformer instance.

    bsplines_ : list of shape (n_features,)
        List of BSplines objects, one for each feature.

    n_features_out_ : int
        Number of output features.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.preprocessing import SplineTransformer
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2024, 1, i) for i in range(1, 11)],
    ...     "value": [float(i) for i in range(10)],
    ... })
    >>> spline = SplineTransformer(n_knots=4, degree=3)
    >>> spline.fit(X)  # doctest: +ELLIPSIS
    SplineTransformer(...)
    >>> X_spline = spline.transform(X)
    >>> # Generates spline basis features
    >>> len(X_spline.columns) > len(X.columns)
    True

    See Also
    --------
    - [`PolynomialFeatures`][yohou.preprocessing.sklearn_wrappers.PolynomialFeatures] : Generate polynomial and interaction features.

    """

    _estimator_default_class = sklearn_SplineTransformer

    def __init__(
        self,
        n_knots=5,
        degree=3,
        knots="uniform",
        extrapolation="constant",
        include_bias=True,
        order="C",
        sparse_output=False,
        handle_missing="error",
        **kwargs,
    ):
        params = _filter_estimator_params(
            sklearn_SplineTransformer,
            {
                "n_knots": n_knots,
                "degree": degree,
                "knots": knots,
                "extrapolation": extrapolation,
                "include_bias": include_bias,
                "order": order,
                "sparse_output": sparse_output,
                "handle_missing": handle_missing,
            },
        )
        super().__init__(**params, **kwargs)

    @property
    def bsplines_(self) -> list:
        """List of BSplines objects, one for each feature."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.bsplines_

    @property
    def n_features_out_(self) -> int:
        """Number of output features."""
        check_is_fitted(self, ["instance_"])
        return self.instance_.n_features_out_
