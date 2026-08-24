"""Base class for transformers over the derived step frame.

A step frame is the wide, step-indexed frame that ``_derive_step_columns``
produces from ``X_future`` and ``X_forecast``: one row per observation time and,
for every value column, the columns ``{base}_step_1`` through ``{base}_step_H``
holding that column's values over the forecasting horizon ahead of the row's
time. It is the only place in the pipeline where "the H values ahead of
observation time T" exist as a single aligned row, and so the only place a
transformation along the horizon can be expressed.

Structurally a step frame carries one index column, ``"time"``, exactly as a
single-axis ``X_actual`` frame does. The distinct ``"step"`` kind exists for what
the columns mean rather than for their shape: a transformer written to reduce
``{base}_step_1..H`` blocks finds nothing to do on an ``X_actual`` frame, and the
kind tag turns that silent no-op into a slot error.

Step transformers are stateless. The step frame is re-derived from scratch at
every ``observe`` and ``predict``, so no buffer persists between calls and the
memory API has nothing to accumulate; it is refused rather than merely unused.
"""

import abc
import re

import polars as pl
from sklearn.utils.validation import check_is_fitted

from yohou.base.transformer import _BaseTransformer
from yohou.utils._compat import _fit_context

__all__ = [
    "BaseStepTransformer",
]

#: Index columns of a step frame; every other column is a feature.
STEP_INDEX_COLS: tuple[str] = ("time",)

#: Matches the trailing ``_step_<integer>`` of a horizon-indexed column name.
#: Anchored at the end so a base column that already contains ``_step_`` (for
#: example ``foo_step_3`` expanding to ``foo_step_3_step_1``) resolves on its
#: real trailing index rather than on the one inside its name.
_STEP_INDEX_RE = re.compile(r"_step_(\d+)$")


def _is_step_indexed(name: str) -> bool:
    """Return whether a column name carries a horizon step index.

    This is the single site in the library that parses a step index out of a
    column name. Everything else either builds names from a known step number or
    tests set membership.

    A step-indexed column belongs to one horizon step and can therefore be
    filtered per estimator by ``step_feature_alignment``. A column produced by a
    step transformer (``temp_step_mean``, ``wx_step_pc1``) is horizon-agnostic:
    it summarises the whole block, has no step to align to, and must reach every
    estimator.

    Parameters
    ----------
    name : str
        A column name from a step frame.

    Returns
    -------
    bool
        ``True`` if the name ends in ``_step_<integer>``.

    Examples
    --------
    >>> _is_step_indexed("temp_step_3")
    True
    >>> _is_step_indexed("temp_step_mean")
    False
    >>> _is_step_indexed("foo_step_3_step_1")
    True

    """
    return _STEP_INDEX_RE.search(name) is not None


def _step_index(name: str) -> int | None:
    """Return the horizon step a column name carries, or ``None``.

    Parameters
    ----------
    name : str
        A column name from a step frame.

    Returns
    -------
    int or None
        The trailing step number, or ``None`` when the name is horizon-agnostic.

    Examples
    --------
    >>> _step_index("temp_step_3")
    3
    >>> _step_index("temp_step_mean") is None
    True

    """
    match = _STEP_INDEX_RE.search(name)
    return int(match.group(1)) if match else None


def _validate_step_transformer_data(
    transformer: "BaseStepTransformer",
    X: pl.DataFrame,
    *,
    reset: bool,
) -> pl.DataFrame:
    """Validate a step frame for a step transformer.

    Checks the single ``"time"`` index column and, in a fit context, records the
    feature schema from the non-index columns.

    Parameters
    ----------
    transformer : BaseStepTransformer
        The transformer whose fitted attributes are set (``reset=True``) or
        checked against (``reset=False``).
    X : pl.DataFrame
        Input step frame.
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
        If ``X`` is ``None``, lacks a ``"time"`` column, that column is not a
        date or datetime, or (``reset=False``) its feature columns differ from
        those seen during ``fit``.

    """
    if X is None:
        raise ValueError("`X` cannot be None.")

    if "time" not in X.columns:
        raise ValueError(f"A step frame must contain a 'time' column. Found columns: {list(X.columns)}")
    if not isinstance(X["time"].dtype, pl.Datetime | pl.Date):
        raise ValueError(f"The 'time' column of a step frame must be Date or Datetime, got {X['time'].dtype}.")

    feature_names = [c for c in X.columns if c not in STEP_INDEX_COLS]

    if reset:
        transformer.feature_names_in_ = feature_names
        transformer.n_features_in_ = len(feature_names)
        transformer.X_schema_ = {c: X.schema[c] for c in feature_names}
    else:
        expected = getattr(transformer, "feature_names_in_", None)
        if expected is not None and feature_names != expected:
            raise ValueError(f"Feature columns of X {feature_names} do not match those seen during fit {expected}.")

    return X


class BaseStepTransformer(_BaseTransformer, metaclass=abc.ABCMeta):
    """Base class for ``"step"``-kind transformers over the derived step frame.

    ``fit`` and ``transform`` operate on a frame carrying a ``"time"`` index
    column plus step-indexed feature columns (``{base}_step_1`` through
    ``{base}_step_H``). ``transform`` preserves the index column on output and is
    stateless.

    Output columns that summarise a whole block are named ``{base}_step_{name}``
    where ``name`` is not a decimal integer. That is what separates them from
    horizon-indexed columns, so ``step_feature_alignment`` filters the latter per
    estimator and leaves the former alone.

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
    - [`BaseForecastTransformer`][yohou.base.forecast_transformer.BaseForecastTransformer] : Base class for X_forecast transformers.

    """

    _tags: dict = {"kind": "step"}

    @property
    def min_steps(self) -> int:
        """Get the smallest forecasting horizon this transformer can work on.

        The default is ``1``: a transformer imposing no minimum block length
        accepts any horizon. Implementations that need a minimum number of step
        columns to be meaningful (a projection onto ``k`` components needs at
        least ``k``) report their own larger value.

        This is deliberately not named ``observation_horizon``. On
        [`BaseActualTransformer`][yohou.base.transformer.BaseActualTransformer]
        that name means a buffer carried *between* calls, sized so the next
        ``transform`` has enough history behind it. A step transformer keeps no
        such buffer: each row is self-contained, so this quantity is a width
        requirement *within* one row rather than a memory depth across calls.

        A forecaster holding this transformer in its ``step_transformer`` slot
        asserts ``forecasting_horizon >= min_steps`` at fit, because the horizon
        is exactly how many step columns each base column expands to.

        Readable before ``fit``, unlike
        [`BaseForecastTransformer.min_vintage_rows`][yohou.base.forecast_transformer.BaseForecastTransformer.min_vintage_rows],
        which it otherwise mirrors. The forecaster's assertion has to run *before*
        the slot is fitted to be worth anything: a transformer whose inner
        estimator needs more columns than the horizon provides fails during its own
        fit, and the resulting message describes matrix shapes rather than naming
        the horizon. A minimum derived from constructor parameters needs no fitted
        state to report, so requiring one would only delay the clearer error.

        Returns
        -------
        int
            Smallest usable forecasting horizon, at least ``1``.

        """
        return 1

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params) -> "BaseStepTransformer":
        """Fit the transformer to a step frame.

        Parameters
        ----------
        X : pl.DataFrame
            Input step frame with a ``"time"`` column and one or more
            step-indexed feature columns.
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
            If ``X`` is missing the ``"time"`` index column.

        """
        X = _validate_step_transformer_data(self, X, reset=True)
        self._fit(X, y)
        return self

    def transform(self, X: pl.DataFrame, **params) -> pl.DataFrame:
        """Transform a step frame.

        Parameters
        ----------
        X : pl.DataFrame
            Input step frame with a ``"time"`` column and one or more
            step-indexed feature columns.
        **params : dict
            Accepted for signature compatibility and ignored: this transformer
            routes no metadata to nested estimators.

        Returns
        -------
        pl.DataFrame
            Transformed step frame with the ``"time"`` index column preserved.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the transformer has not been fitted yet.
        ValueError
            If ``X`` is missing the ``"time"`` index column or its feature
            columns differ from those seen during ``fit``.

        """
        check_is_fitted(self, ["X_schema_", "feature_names_in_", "n_features_in_"])
        X = _validate_step_transformer_data(self, X, reset=False)
        return self._transform(X)
