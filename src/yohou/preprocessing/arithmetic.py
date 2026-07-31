"""Generic invertible arithmetic transformers.

``ArithmeticTransformer`` computes one output column from a binary operation
(``add``/``sub``/``mul``/``div``) between two named columns, and
``ReduceTransformer`` reduces an arbitrary-length list of columns with ``sum`` or
``product``. Both are stateless (``observation_horizon == 0``) and row-local, so
they lift onto an ``X_forecast`` frame through
[`PerVintageActualTransformer`][yohou.compose.per_vintage.PerVintageActualTransformer]
unchanged.

Every operation here is a binary operation of an abelian group -- additive
``(ℝ, +)`` for ``add``/``sub``, multiplicative ``(ℝ₊, ×)`` for ``mul``/``div`` --
which is why they invert and what dictates the retention rule: an operand is
recoverable only if the *other* operand (its sibling) survives the forward pass.
With ``keep_inputs=False`` (the default) the operands are dropped and the
transform is not invertible; with ``keep_inputs=True`` they are retained and
``inverse_transform`` recovers the operand named by ``invert_wrt``.
"""

from __future__ import annotations

import functools
import operator

import polars as pl
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseActualTransformer
from yohou.utils import Tags
from yohou.utils._compat import StrOptions

__all__ = ["ArithmeticTransformer", "ReduceTransformer"]

#: Forward polars-expression builders for each binary operation.
_BINARY_OPS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
}


def _product_expr(cols: list[str]) -> pl.Expr:
    """Row-wise product of ``cols`` as a single Polars expression."""
    return functools.reduce(operator.mul, (pl.col(c) for c in cols))


class ArithmeticTransformer(BaseActualTransformer):
    r"""Invertible column-wise arithmetic between two columns.

    Computes ``output_name = left_col <op> right_col`` for ``op`` in
    ``{"add", "sub", "mul", "div"}``. Stateless and row-local: each row's output
    depends only on that row, so the transformer composes in a ``FeatureUnion``
    and lifts onto ``X_forecast`` via ``PerVintageActualTransformer``.

    Invertibility follows the abelian-group inverse and requires the sibling
    operand to be retained. With ``keep_inputs=True`` the operands are carried
    through ``transform``; ``inverse_transform`` then recovers the operand named
    by ``invert_wrt`` from ``output_name`` plus the retained sibling:

    ========  ================  =============================  =============================
    ``op``    forward           recover ``left`` (keep right)  recover ``right`` (keep left)
    ========  ================  =============================  =============================
    ``add``   ``a + b``         ``a = c - b``                  ``b = c - a``
    ``sub``   ``a - b``         ``a = c + b``                  ``b = a - c``
    ``mul``   ``a * b``         ``a = c / b``                  ``b = c / a``
    ``div``   ``a / b``         ``a = c * b``                  ``b = a / c``
    ========  ================  =============================  =============================

    Parameters
    ----------
    left_col : str
        Column name of the left operand ``a``.
    right_col : str
        Column name of the right operand ``b``.
    op : {"add", "sub", "mul", "div"}
        The binary operation.
    output_name : str, default="arithmetic"
        Name of the emitted output column.
    keep_inputs : bool, default=False
        When ``False``, ``transform`` emits only ``["time", output_name]`` (the
        lean feature-construction convention) and the transform is not
        invertible. When ``True``, the operand columns are retained so
        ``inverse_transform`` can recover one of them.
    invert_wrt : {"left", "right"}, default="left"
        Which operand ``inverse_transform`` recovers. For ``sub``/``div`` the two
        directions are distinct formulas; for ``add``/``mul`` they coincide.

    Examples
    --------
    >>> from datetime import datetime
    >>> import polars as pl
    >>> from yohou.preprocessing import ArithmeticTransformer
    >>> time = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 3), interval="1d", eager=True)
    >>> X = pl.DataFrame({"time": time, "revenue": [30.0, 40.0, 50.0], "cost": [25.0, 30.0, 35.0]})
    >>> t = ArithmeticTransformer("revenue", "cost", op="sub", output_name="margin", keep_inputs=True)
    >>> X_t = t.fit_transform(X)
    >>> X_t["margin"].to_list()
    [5.0, 10.0, 15.0]
    >>> t.inverse_transform(X_t)["revenue"].to_list()
    [30.0, 40.0, 50.0]
    """

    # The binary operation is row-local: each row's output reads only that row's two
    # operands, never the spacing of the time axis. The strict interval-consistency check
    # is therefore inapplicable, and a sparse per-vintage forward slice lifted via
    # `PerVintageActualTransformer` would otherwise be rejected for spacing the
    # computation never looks at. Merged across the MRO, so subclasses inherit it.
    _tags = {"accepts_irregular_grid": True}

    _parameter_constraints: dict = {
        "left_col": [str],
        "right_col": [str],
        "op": [StrOptions(set(_BINARY_OPS))],
        "output_name": [str],
        "keep_inputs": ["boolean"],
        "invert_wrt": [StrOptions({"left", "right"})],
    }

    def __init__(
        self,
        left_col: str,
        right_col: str,
        op: str,
        output_name: str = "arithmetic",
        *,
        keep_inputs: bool = False,
        invert_wrt: str = "left",
    ):
        self.left_col = left_col
        self.right_col = right_col
        self.op = op
        self.output_name = output_name
        self.keep_inputs = keep_inputs
        self.invert_wrt = invert_wrt

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Validate that both operand columns are present in ``X``."""
        missing = [c for c in (self.left_col, self.right_col) if c not in X.columns]
        if missing:
            raise ValueError(f"input is missing required column(s): {missing}")

    def _retained_inputs(self) -> list[str]:
        """Operand columns carried through ``transform`` when ``keep_inputs``."""
        kept: list[str] = []
        if self.keep_inputs:
            for c in (self.left_col, self.right_col):
                if c not in kept and c != self.output_name:
                    kept.append(c)
        return kept

    def __sklearn_tags__(self) -> Tags:
        """Mark the transform invertible only when the operands are retained.

        Invertibility is a constructor-time property here, not a class-time one: the
        inverse needs the sibling operand, so ``keep_inputs=False`` has no inverse. A
        class-level ``_tags`` dict cannot express that, and leaving the base default of
        ``False`` hides a working inverse from every ``available_if`` gate that reads the
        tag rather than probing for the method (`FeaturePipeline.inverse_transform`).
        """
        tags = super().__sklearn_tags__()
        if tags.transformer_tags is not None:
            tags.transformer_tags.invertible = bool(self.keep_inputs)
        return tags

    @property
    def target_output_name(self) -> str:
        """The operand ``inverse_transform`` reconstructs, per ``invert_wrt``.

        ``inverse_transform`` emits the recovered operand *and* the retained sibling it
        needed to recover it. Only the former is the value; the sibling is scaffolding.
        Naming it here lets a consumer that combines transformed series pick the target
        out rather than treating the sibling as a second contribution.
        """
        return self.left_col if self.invert_wrt == "left" else self.right_col

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Emit ``"time"``, the retained operands, and the output column."""
        forward = _BINARY_OPS[self.op](pl.col(self.left_col), pl.col(self.right_col)).alias(self.output_name)
        return X.select("time", *self._retained_inputs(), forward)

    def _inverse_transform(self, X_t: pl.DataFrame, X_p: pl.DataFrame | None = None) -> pl.DataFrame:
        """Recover the ``invert_wrt`` operand from the output and its sibling.

        Requires the output column and the sibling operand to be present in
        ``X_t``; raises otherwise. ``X_p`` is unused (the transform is stateless).
        """
        target_col = self.left_col if self.invert_wrt == "left" else self.right_col
        sibling_col = self.right_col if self.invert_wrt == "left" else self.left_col
        missing = [c for c in (self.output_name, sibling_col) if c not in X_t.columns]
        if missing:
            raise ValueError(
                f"inverse_transform requires column(s) {missing} to recover '{target_col}' "
                f"(invert_wrt='{self.invert_wrt}'); retain them with keep_inputs=True or supply them."
            )
        recovered = self._inverse_expr(pl.col(self.output_name), pl.col(sibling_col)).alias(target_col)
        return X_t.select("time", pl.col(sibling_col), recovered)

    def _inverse_expr(self, c: pl.Expr, sibling: pl.Expr) -> pl.Expr:
        """Group-inverse expression recovering the ``invert_wrt`` operand."""
        left = self.invert_wrt == "left"
        if self.op == "add":  # a = c - b ; b = c - a
            return c - sibling
        if self.op == "sub":  # a = c + b ; b = a - c
            return c + sibling if left else sibling - c
        if self.op == "mul":  # a = c / b ; b = c / a
            return c / sibling
        # div: a = c * b ; b = a / c
        return c * sibling if left else sibling / c

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return the emitted feature-column names (retained operands + output)."""
        check_is_fitted(self, "X_schema_")
        return [*self._retained_inputs(), self.output_name]


class ReduceTransformer(BaseActualTransformer):
    r"""Invertible n-ary reduction of several columns with ``sum`` or ``product``.

    Reduces ``input_cols`` row-wise into a single ``output_name`` column. An n-ary
    reduction collapses ``n`` columns into one and is lossy unless the other
    ``n-1`` parts are retained, so the inverse is **partial**: given the output
    and the ``n-1`` siblings of a designated ``invert_col``, it recovers that
    part (``sum``: ``part = output - Σ others``; ``product``:
    ``part = output / Π others``). Full multi-way *target* reconstruction is the
    combining forecaster's job; this partial inverse is the feature-frame
    counterpart for frames that keep their siblings.

    Parameters
    ----------
    input_cols : list of str
        Column names reduced row-wise.
    op : {"sum", "product"}, default="sum"
        The reduction operation.
    output_name : str, default="reduce"
        Name of the emitted output column.
    keep_inputs : bool, default=False
        When ``True``, the input columns are retained so the partial inverse can
        recover ``invert_col``. When ``False``, only ``["time", output_name]`` is
        emitted and no inverse is possible.
    invert_col : str or None, default=None
        The part ``inverse_transform`` recovers. Must be one of ``input_cols``.
        Required for ``inverse_transform``.

    Examples
    --------
    >>> from datetime import datetime
    >>> import polars as pl
    >>> from yohou.preprocessing import ReduceTransformer
    >>> time = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 2), interval="1d", eager=True)
    >>> X = pl.DataFrame({"time": time, "a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
    >>> t = ReduceTransformer(
    ...     ["a", "b", "c"], op="sum", output_name="total", keep_inputs=True, invert_col="a"
    ... )
    >>> X_t = t.fit_transform(X)
    >>> X_t["total"].to_list()
    [9.0, 12.0]
    >>> t.inverse_transform(X_t)["a"].to_list()
    [1.0, 2.0]
    """

    # The reduction is row-local: each row's output reads only that row's `input_cols`,
    # never the spacing of the time axis. See `ArithmeticTransformer._tags`.
    _tags = {"accepts_irregular_grid": True}

    _parameter_constraints: dict = {
        "input_cols": [list],
        "op": [StrOptions({"sum", "product"})],
        "output_name": [str],
        "keep_inputs": ["boolean"],
        "invert_col": [str, None],
    }

    def __init__(
        self,
        input_cols: list[str],
        op: str = "sum",
        output_name: str = "reduce",
        *,
        keep_inputs: bool = False,
        invert_col: str | None = None,
    ):
        self.input_cols = input_cols
        self.op = op
        self.output_name = output_name
        self.keep_inputs = keep_inputs
        self.invert_col = invert_col

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Validate input columns are present and ``invert_col`` is among them."""
        if not self.input_cols:
            raise ValueError("input_cols must be a non-empty list of column names")
        missing = [c for c in self.input_cols if c not in X.columns]
        if missing:
            raise ValueError(f"input is missing required column(s): {missing}")
        if self.invert_col is not None and self.invert_col not in self.input_cols:
            raise ValueError(f"invert_col '{self.invert_col}' must be one of input_cols {self.input_cols}")

    def _retained_inputs(self) -> list[str]:
        """Input columns carried through ``transform`` when ``keep_inputs``."""
        if not self.keep_inputs:
            return []
        return [c for c in self.input_cols if c != self.output_name]

    def __sklearn_tags__(self) -> Tags:
        """Mark the transform invertible only when an ``invert_col`` is designated.

        See `ArithmeticTransformer.__sklearn_tags__`; here the inverse additionally needs
        a nominated column to recover, so both conditions must hold.
        """
        tags = super().__sklearn_tags__()
        if tags.transformer_tags is not None:
            tags.transformer_tags.invertible = bool(self.keep_inputs and self.invert_col is not None)
        return tags

    @property
    def target_output_name(self) -> str | None:
        """The part ``inverse_transform`` recovers, or ``None`` when not invertible.

        The inverse emits the recovered ``invert_col`` alongside the ``n-1`` siblings it
        needed; only the former is the value. ``None`` when ``invert_col`` is unset, since
        there is then no inverse and no designated target.
        """
        return self.invert_col

    def _forward_expr(self) -> pl.Expr:
        """Row-wise reduction of ``input_cols``."""
        if self.op == "sum":
            return pl.sum_horizontal(self.input_cols)
        return _product_expr(self.input_cols)

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Emit ``"time"``, the retained inputs, and the reduced output column."""
        return X.select("time", *self._retained_inputs(), self._forward_expr().alias(self.output_name))

    def _inverse_transform(self, X_t: pl.DataFrame, X_p: pl.DataFrame | None = None) -> pl.DataFrame:
        """Recover ``invert_col`` from the output and its ``n-1`` retained siblings."""
        if self.invert_col is None:
            raise ValueError("inverse_transform requires invert_col to be set")
        others = [c for c in self.input_cols if c != self.invert_col]
        missing = [c for c in (self.output_name, *others) if c not in X_t.columns]
        if missing:
            raise ValueError(
                f"inverse_transform requires column(s) {missing} to recover '{self.invert_col}'; "
                f"retain them with keep_inputs=True or supply them."
            )
        c = pl.col(self.output_name)
        if not others:
            recovered = c  # single-column reduction is the identity
        elif self.op == "sum":
            recovered = c - pl.sum_horizontal(others)
        else:
            recovered = c / _product_expr(others)
        return X_t.select("time", *[pl.col(o) for o in others], recovered.alias(self.invert_col))

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return the emitted feature-column names (retained inputs + output)."""
        check_is_fitted(self, "X_schema_")
        return [*self._retained_inputs(), self.output_name]
