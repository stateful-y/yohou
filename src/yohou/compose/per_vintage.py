"""PerVintageActualTransformer: lift a stateless actual transformer onto the vintage axis."""

from __future__ import annotations

import polars as pl
from sklearn.base import clone

from yohou.base.forecast_transformer import FORECAST_INDEX_COLS, BaseForecastTransformer
from yohou.base.transformer import BaseActualTransformer

__all__ = ["PerVintageActualTransformer"]

_VINTAGE_COL = "vintage_time"


class PerVintageActualTransformer(BaseForecastTransformer):
    """Apply a stateless single-axis transformer to each vintage of an ``X_forecast`` frame.

    Wraps a `BaseActualTransformer` and applies it independently to every
    vintage: the frame is grouped by ``"vintage_time"``, each group's single-axis
    ``["time", ...]`` slice is transformed by the wrapped transformer, and the
    results are re-stacked with ``"vintage_time"`` restored. Because each vintage
    is transformed using only its own rows, order-dependent inner transforms
    (lags, differences, rolling statistics within the forecast horizon) never
    bleed across vintage boundaries.

    The wrapped transformer must be **stateless** (measured
    ``observation_horizon == 0`` after fitting). Stateful transformers need
    contiguous memory across a single series, which the discontinuous vintage
    axis cannot provide, so they are rejected.

    Parameters
    ----------
    transformer : BaseActualTransformer
        The single-axis transformer to apply per vintage. Must be stateless.

    Attributes
    ----------
    transformer_ : BaseActualTransformer
        The fitted clone of ``transformer``.
    feature_names_in_ : list[str]
        Feature (non-index) column names seen during ``fit``.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.preprocessing import FunctionTransformer
    >>> from yohou.compose import PerVintageActualTransformer
    >>>
    >>> X_forecast = pl.DataFrame({
    ...     "vintage_time": [datetime(2020, 1, 1)] * 2 + [datetime(2020, 1, 2)] * 2,
    ...     "time": [
    ...         datetime(2020, 1, 2),
    ...         datetime(2020, 1, 3),
    ...         datetime(2020, 1, 3),
    ...         datetime(2020, 1, 4),
    ...     ],
    ...     "load": [100.0, 110.0, 120.0, 130.0],
    ...     "wind": [10.0, 20.0, 15.0, 25.0],
    ... })
    >>> def net_load(df):
    ...     return df.select((pl.col("load") - pl.col("wind")).alias("net_load"))
    >>> tx = PerVintageActualTransformer(
    ...     FunctionTransformer(func=net_load, feature_names_out=lambda self, names: ["net_load"])
    ... )
    >>> out = tx.fit_transform(X_forecast)
    >>> out.columns
    ['vintage_time', 'time', 'net_load']
    >>> out["net_load"].to_list()
    [90.0, 90.0, 105.0, 105.0]

    See Also
    --------
    - [`BaseForecastTransformer`][yohou.base.forecast_transformer.BaseForecastTransformer] : Base class for X_forecast transformers.
    - [`BaseActualTransformer`][yohou.base.transformer.BaseActualTransformer] : Base class for single-axis transformers.

    """

    _required_parameters = ["transformer"]

    _parameter_constraints: dict = {
        "transformer": [BaseActualTransformer],
    }

    def __init__(self, transformer: BaseActualTransformer):
        self.transformer = transformer

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Fit the wrapped transformer once and enforce statelessness.

        The wrapped transformer is fitted on a single representative vintage (a
        valid single-axis frame); the ``X_forecast`` frame as a whole is not a
        single monotonic series, so it cannot be fed to a single-axis
        transformer directly.
        """
        if X.is_empty():
            raise ValueError("Cannot fit PerVintageActualTransformer on an empty X_forecast frame.")

        # Fit the inner on the first vintage's slice (a valid single-axis series).
        first_vintage = X[_VINTAGE_COL][0]
        slice_0 = X.filter(pl.col(_VINTAGE_COL) == first_vintage).drop(_VINTAGE_COL)

        self.transformer_ = clone(self.transformer)
        self.transformer_.fit(slice_0)

        if self.transformer_.observation_horizon != 0:
            raise ValueError(
                "PerVintageActualTransformer only lifts stateless transformers onto the vintage "
                f"axis, but {type(self.transformer).__name__} measured "
                f"observation_horizon={self.transformer_.observation_horizon} after fitting. "
                "A stateful transformer needs contiguous memory that the vintage axis cannot provide."
            )

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Transform each vintage independently and re-stack with vintage_time."""
        vintage_dtype = X[_VINTAGE_COL].dtype

        if X.is_empty():
            # Transform the empty single-axis slice to recover the output columns.
            out = self.transformer_.transform(X.drop(_VINTAGE_COL))
            out = out.with_columns(pl.lit(None, dtype=vintage_dtype).alias(_VINTAGE_COL))
            return self._order_columns(out)

        parts: list[pl.DataFrame] = []
        for part in X.partition_by(_VINTAGE_COL, maintain_order=True):
            vintage_value = part[_VINTAGE_COL][0]
            transformed = self.transformer_.transform(part.drop(_VINTAGE_COL))
            transformed = transformed.with_columns(pl.lit(vintage_value, dtype=vintage_dtype).alias(_VINTAGE_COL))
            parts.append(transformed)

        return self._order_columns(pl.concat(parts))

    @staticmethod
    def _order_columns(df: pl.DataFrame) -> pl.DataFrame:
        """Return ``df`` with ``vintage_time``, ``time`` first, then feature columns."""
        feature_cols = [c for c in df.columns if c not in FORECAST_INDEX_COLS]
        return df.select(*FORECAST_INDEX_COLS, *feature_cols)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return the wrapped transformer's output feature names.

        Parameters
        ----------
        input_features : list of str or None, default=None
            Ignored; the fitted wrapped transformer determines the output names.

        Returns
        -------
        list of str
            Output feature names, delegated to the wrapped transformer.

        """
        return list(self.transformer_.get_feature_names_out(input_features))
