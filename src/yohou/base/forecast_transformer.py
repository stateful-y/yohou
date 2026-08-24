"""Base class for transformers over ``X_forecast`` frames.

An ``X_forecast`` frame carries two time axes -- ``"vintage_time"`` (when a
forecast was issued) and ``"time"`` (what it forecasts) -- so it is a stack of
short per-vintage series rather than a single monotonic series. Single-axis
transformers (``BaseActualTransformer``) cannot consume such a frame;
``BaseForecastTransformer`` is the base for those that can.

Forecast transformers are stateless: each call to ``transform`` is a pure
function of its input frame, so there is no ``observe``/``rewind`` memory API.
Serving a single fresh vintage is simply a one-group input.
"""

import abc

import polars as pl
from sklearn.utils.validation import check_is_fitted

from yohou.base.transformer import _BaseTransformer
from yohou.utils._compat import _fit_context
from yohou.utils.validate_data import _validate_X_forecast_structure

__all__ = [
    "BaseForecastTransformer",
]

#: Index columns of an ``X_forecast`` frame; every other column is a feature.
FORECAST_INDEX_COLS: tuple[str, str] = ("vintage_time", "time")


def _validate_forecast_transformer_data(
    transformer: "BaseForecastTransformer",
    X: pl.DataFrame,
    *,
    reset: bool,
) -> pl.DataFrame:
    """Validate an ``X_forecast`` frame for a forecast transformer.

    Checks the two-column index (``"vintage_time"`` and ``"time"``) and, in a
    fit context, records the feature schema from the non-index columns.

    Parameters
    ----------
    transformer : BaseForecastTransformer
        The transformer whose fitted attributes are set (``reset=True``) or
        checked against (``reset=False``).
    X : pl.DataFrame
        Input ``X_forecast`` frame.
    reset : bool
        Whether this is a fit context (records the schema) or a transform
        context (checks the schema).

    Returns
    -------
    pl.DataFrame
        The validated frame (unchanged).

    Raises
    ------
    ValueError
        If ``X`` is ``None``, is missing an index column, or (``reset=False``)
        its feature columns differ from those seen during ``fit``.

    """
    if X is None:
        raise ValueError("`X` cannot be None.")

    _validate_X_forecast_structure(X)

    feature_names = [c for c in X.columns if c not in FORECAST_INDEX_COLS]

    if reset:
        transformer.feature_names_in_ = feature_names
        transformer.n_features_in_ = len(feature_names)
        transformer.X_schema_ = {c: X.schema[c] for c in feature_names}
    else:
        expected = getattr(transformer, "feature_names_in_", None)
        if expected is not None and feature_names != expected:
            raise ValueError(f"Feature columns of X {feature_names} do not match those seen during fit {expected}.")

    return X


class BaseForecastTransformer(_BaseTransformer, metaclass=abc.ABCMeta):
    """Base class for ``"forecast"``-kind transformers over ``X_forecast`` frames.

    ``fit`` and ``transform`` operate on a frame carrying ``"vintage_time"`` and
    ``"time"`` index columns plus one or more feature columns. ``transform``
    preserves the two index columns on output and is stateless. A frame may
    contain any number of vintages, including one (serving a single fresh
    vintage) or none (an empty frame).

    Attributes
    ----------
    feature_names_in_ : list[str]
        Names of the non-index columns seen during ``fit``.
    n_features_in_ : int
        Number of non-index columns seen during ``fit``.
    X_schema_ : dict[str, pl.DataType]
        Feature column name to dtype mapping seen during ``fit``.

    See Also
    --------
    - [`BaseActualTransformer`][yohou.base.transformer.BaseActualTransformer] : Base class for single-axis transformers.
    - [`PerVintageActualTransformer`][yohou.compose.per_vintage.PerVintageActualTransformer] : Lift a stateless actual transformer onto the vintage axis.

    """

    _tags: dict = {"kind": "forecast"}

    @property
    def min_vintage_rows(self) -> int:
        """Get the smallest vintage length that yields non-empty output.

        A vintage shorter than this contributes no rows to ``transform`` output.
        The default is ``1``: a transformer imposing no per-vintage minimum emits
        output for any non-empty vintage. Implementations that consume rows within
        a vintage, or that need a minimum number of rows to fit on, report their
        own larger value.

        This is deliberately not named ``observation_horizon``. On
        [`BaseActualTransformer`][yohou.base.transformer.BaseActualTransformer]
        that name means a buffer carried *between* calls, sized so the next
        ``transform`` has enough history behind it. A forecast transformer keeps
        no such buffer: each vintage is self-contained, so this quantity is a
        length requirement *within* one vintage rather than a memory depth across
        calls.

        A forecaster holding this transformer in its ``forecast_transformer`` slot
        asserts ``forecasting_horizon >= min_vintage_rows`` at fit, because a
        served vintage spans the forecasting horizon.

        Returns
        -------
        int
            Smallest vintage length yielding non-empty output, at least ``1``.

        Raises
        ------
        NotFittedError
            If the transformer has not been fitted yet.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        return 1

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> "BaseForecastTransformer":
        """Fit the transformer to an ``X_forecast`` frame.

        Parameters
        ----------
        X : pl.DataFrame
            Input ``X_forecast`` frame with ``"vintage_time"`` and ``"time"``
            columns and one or more feature columns.
        y : pl.DataFrame or None, default=None
            Ignored.  Present for API compatibility.
        **params : dict
            Accepted for signature compatibility and ignored: this transformer
            routes no metadata to nested estimators.

        Returns
        -------
        self
            The fitted transformer instance.

        Raises
        ------
        ValueError
            If ``X`` is missing the ``"vintage_time"`` or ``"time"`` index
            columns.

        """
        X = _validate_forecast_transformer_data(self, X, reset=True)
        self._fit(X, y)
        return self

    def transform(self, X: pl.DataFrame, **params) -> pl.DataFrame:
        """Transform an ``X_forecast`` frame.

        Parameters
        ----------
        X : pl.DataFrame
            Input ``X_forecast`` frame with ``"vintage_time"`` and ``"time"``
            columns and one or more feature columns.
        **params : dict
            Accepted for signature compatibility and ignored: this transformer
            routes no metadata to nested estimators.

        Returns
        -------
        pl.DataFrame
            Transformed ``X_forecast`` frame with the ``"vintage_time"`` and
            ``"time"`` index columns preserved.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the transformer has not been fitted yet.
        ValueError
            If ``X`` is missing an index column or its feature columns differ
            from those seen during ``fit``.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        X = _validate_forecast_transformer_data(self, X, reset=False)
        return self._transform(X)
