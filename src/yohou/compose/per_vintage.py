"""PerVintageActualTransformer: lift a single-axis actual transformer onto the vintage axis."""

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
    """Apply a single-axis transformer to each vintage of an ``X_forecast`` frame.

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
    names and to measure ``observation_horizon``, both value-independent; the
    per-vintage fits happen in ``transform``.

    The wrapped transformer **may be stateful**. A vintage is internally
    contiguous (its forecast steps at the series interval), so a lag or a
    difference is well defined within one, and a fresh clone per vintage means
    its history never reaches across a vintage boundary. Only *cross-vintage*
    memory is impossible, which is why the composite itself stays stateless and
    exposes no ``observe``/``rewind``: it carries no buffer between calls
    whatever the inner does inside one vintage.

    A stateful inner consumes ``observation_horizon`` rows from the **start** of
    every vintage, which are its nearest-term forecast steps. If it consumes the
    whole vintage the composite raises, since that is a configuration error. Note
    that a lifted lag means "the forecast for step h-1 issued at the same
    ``vintage_time``", a trajectory ramp; it is *not* the previous vintage's
    forecast for the same ``time``, which this wrapper does not provide.

    A vintage with fewer than two rows cannot be fitted on its own and has no
    per-vintage statistic to compute. Such vintages, typically the truncated
    tail of a forecast frame, are dropped from the output with a warning rather
    than transformed with borrowed parameters.

    Parameters
    ----------
    transformer : BaseActualTransformer
        The single-axis transformer to apply per vintage. May be stateful, in
        which case it must leave at least one row per vintage.

    Attributes
    ----------
    transformer_ : BaseActualTransformer
        A representative clone fitted on the first vintage, used only for
        ``get_feature_names_out`` and to measure ``observation_horizon``, never
        to produce transformed values.
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
        """Fit a representative clone for feature names and the horizon.

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
        try:
            self.transformer_.fit(representative)
        except Exception as err:
            # Vintages are short, so an inner needing more history than a vintage
            # holds raises inside its own fit. Its message mentions neither
            # vintages nor lifting, so name the sizing constraint and keep the
            # diagnosis. This is not a statelessness problem: stateful inners are
            # supported, this one simply does not fit in a vintage.
            raise ValueError(
                f"PerVintageActualTransformer fits its wrapped transformer on one vintage at a "
                f"time, and {type(self.transformer).__name__} could not fit a vintage of "
                f"{representative.height} rows. The wrapped transformer requires more history "
                f"than a single vintage provides; see the chained error for its own diagnosis."
            ) from err

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

        A stateful inner consumes its ``observation_horizon`` in rows from each
        vintage's start. If that consumes the whole vintage, the composite raises
        rather than returning the vintage's absence silently: unlike the tail
        drop, which is a property of the data, this is a configuration error.
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
            if transformed.is_empty():
                raise ValueError(
                    f"{type(self.transformer).__name__} consumed an entire vintage: it left no rows "
                    f"from a vintage of {part.height}, because its observation_horizon "
                    f"({self.transformer_.observation_horizon}) is not smaller than the vintage "
                    f"length. A wrapped transformer must leave at least one row per vintage; "
                    f"reduce its horizon or use longer vintages."
                )
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
