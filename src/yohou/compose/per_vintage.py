"""PerVintageActualTransformer: lift a stateless actual transformer onto the vintage axis."""

from __future__ import annotations

import warnings

import polars as pl
from sklearn.base import clone

from yohou.base.forecast_transformer import FORECAST_INDEX_COLS, BaseForecastTransformer
from yohou.base.transformer import BaseActualTransformer

__all__ = ["PerVintageActualTransformer"]

_VINTAGE_COL = "vintage_time"

#: A vintage needs at least this many rows to be a fittable single-axis series
#: (two timestamps are the minimum to infer the series interval). Smaller
#: vintages, such as the truncated tail of a forecast frame, cannot be fitted
#: on their own and are dropped with a warning.
_MIN_VINTAGE_ROWS = 2


class PerVintageActualTransformer(BaseForecastTransformer):
    """Apply a stateless single-axis transformer to each vintage of an ``X_forecast`` frame.

    Wraps a `BaseActualTransformer` and applies it independently to every
    vintage: the frame is grouped by ``"vintage_time"``, and each group's
    single-axis ``["time", ...]`` slice is **fitted and transformed on its own**,
    then re-stacked with ``"vintage_time"`` restored. Because a vintage is fitted
    using only its own rows, a value-dependent inner self-references per vintage:
    a scaler standardizes each vintage to that vintage's own scale, a mean
    imputer fills each vintage's gaps from that vintage's own values. Order
    dependent transforms (cumulative sums, differences within the horizon) also
    stay contained.

    The transform is leakage-free: a vintage's output depends only on its own
    rows, all of which are known at that vintage's ``vintage_time``. This is
    distinct from normalizing ``X_forecast`` by the training distribution, which
    is done with a scaler inside the estimator pipeline instead.

    ``fit`` does not fit the inner for the values it will produce. It fits a
    single representative clone on one vintage only to expose output feature
    names and to check statelessness, both value-independent; the per-vintage
    fits happen in ``transform``.

    The wrapped transformer must be **stateless** (measured
    ``observation_horizon == 0`` after fitting). Stateful transformers need
    contiguous memory across a single series, which the discontinuous vintage
    axis cannot provide, so they are rejected.

    A vintage with fewer than two rows cannot be fitted on its own and has no
    per-vintage statistic to compute. Such vintages, typically the truncated
    tail of a forecast frame, are dropped from the output with a warning rather
    than transformed with borrowed parameters.

    Parameters
    ----------
    transformer : BaseActualTransformer
        The single-axis transformer to apply per vintage. Must be stateless.

    Attributes
    ----------
    transformer_ : BaseActualTransformer
        A representative clone fitted on the first vintage, used only for
        ``get_feature_names_out`` and the statelessness check, never to produce
        transformed values.
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
        """Fit a representative clone for feature names and the statelessness check.

        The wrapped transformer is **not** fitted here for the values it will
        produce; those come from a per-vintage fit in ``_transform``. This fit
        exists only to expose ``get_feature_names_out`` and to measure
        ``observation_horizon``, both of which are value-independent (they depend
        on the wrapped transformer's parameters and the frame schema, not on any
        vintage's data). Any single vintage's slice is therefore a valid
        representative; the first is used.
        """
        if X.is_empty():
            raise ValueError("Cannot fit PerVintageActualTransformer on an empty X_forecast frame.")

        representative = self._first_fittable_vintage(X)
        if representative is None:
            raise ValueError(
                "PerVintageActualTransformer needs at least one vintage with "
                f"{_MIN_VINTAGE_ROWS} or more rows to fit, but every vintage in the frame is smaller."
            )

        self.transformer_ = clone(self.transformer)
        self.transformer_.fit(representative)

        if self.transformer_.observation_horizon != 0:
            raise ValueError(
                "PerVintageActualTransformer only lifts stateless transformers onto the vintage "
                f"axis, but {type(self.transformer).__name__} measured "
                f"observation_horizon={self.transformer_.observation_horizon} after fitting. "
                "A stateful transformer needs contiguous memory that the vintage axis cannot provide."
            )

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Fit and transform each vintage independently, then re-stack.

        Each vintage is a self-contained single-axis series, so a fresh clone is
        fitted and applied to that vintage alone. A vintage's output depends only
        on its own rows, which makes the transform leakage-free: every row of a
        vintage is known at its ``vintage_time``.

        A vintage with fewer than ``_MIN_VINTAGE_ROWS`` rows cannot be fitted on
        its own (a single-axis series needs two timestamps to infer its interval)
        and has no meaningful per-vintage statistic to compute. Such vintages,
        typically the truncated tail of a forecast frame, are dropped from the
        output and reported in a warning.
        """
        vintage_dtype = X[_VINTAGE_COL].dtype

        if X.is_empty():
            # No vintage to fit; use the representative to recover output columns.
            return self._empty_output(X, vintage_dtype)

        parts: list[pl.DataFrame] = []
        dropped = 0
        for part in X.partition_by(_VINTAGE_COL, maintain_order=True):
            if part.height < _MIN_VINTAGE_ROWS:
                dropped += 1
                continue
            vintage_value = part[_VINTAGE_COL][0]
            transformed = clone(self.transformer).fit_transform(part.drop(_VINTAGE_COL))
            transformed = transformed.with_columns(pl.lit(vintage_value, dtype=vintage_dtype).alias(_VINTAGE_COL))
            parts.append(transformed)

        if dropped:
            warnings.warn(
                f"PerVintageActualTransformer dropped {dropped} vintage(s) with fewer than "
                f"{_MIN_VINTAGE_ROWS} rows, which cannot be fitted per vintage. This is expected "
                f"for the truncated tail of a forecast frame; those vintages are absent from the output.",
                UserWarning,
                stacklevel=3,
            )

        if not parts:
            return self._empty_output(X, vintage_dtype)

        return self._order_columns(pl.concat(parts))

    def _empty_output(self, X: pl.DataFrame, vintage_dtype: pl.DataType) -> pl.DataFrame:
        """Return a zero-row frame with the wrapped transformer's output schema."""
        out = self.transformer_.transform(X.head(0).drop(_VINTAGE_COL))
        out = out.with_columns(pl.lit(None, dtype=vintage_dtype).alias(_VINTAGE_COL))
        return self._order_columns(out)

    def _first_fittable_vintage(self, X: pl.DataFrame) -> pl.DataFrame | None:
        """Return the first vintage's single-axis slice that has enough rows to fit, or None."""
        for part in X.partition_by(_VINTAGE_COL, maintain_order=True):
            if part.height >= _MIN_VINTAGE_ROWS:
                return part.drop(_VINTAGE_COL)
        return None

    @staticmethod
    def _order_columns(df: pl.DataFrame) -> pl.DataFrame:
        """Return ``df`` with ``vintage_time``, ``time`` first, then feature columns."""
        feature_cols = [c for c in df.columns if c not in FORECAST_INDEX_COLS]
        return df.select(*FORECAST_INDEX_COLS, *feature_cols)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return the wrapped transformer's output feature names.

        Delegates to the representative clone, which was fitted on one vintage,
        while ``transform`` produces values from a separate clone per vintage.
        This relies on output names being value-independent: they follow from the
        wrapped transformer's parameters and the frame schema, which every
        vintage shares, not from any vintage's data. A wrapped transformer whose
        ``feature_names_out`` varied with the values would break that invariant
        and is not supported.

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
