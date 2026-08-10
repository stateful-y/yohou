"""Step-kind transformers over the derived ``{base}_step_1..H`` frame."""

import re
from typing import Any, Literal

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, clone

from yohou.base.step_transformer import BaseStepTransformer, _is_step_indexed
from yohou.utils._compat import StrOptions

__all__ = [
    "StepAggregator",
    "StepColumnReducer",
    "StepFrameReducer",
]

#: Aggregations `StepAggregator` accepts. Closed by design: anything outside this
#: set is reached through `StepColumnReducer` with a `FunctionTransformer`, so the
#: library keeps one extension mechanism rather than two.
_AGGREGATIONS: dict[str, Any] = {
    "min": pl.min_horizontal,
    "max": pl.max_horizontal,
    "mean": pl.mean_horizontal,
    "sum": pl.sum_horizontal,
    "std": None,  # handled separately; polars has no std_horizontal
}

#: An output name that is a bare integer would render as ``{base}_step_3``, which
#: is indistinguishable from a real horizon step column.
_NUMERIC_NAME_RE = re.compile(r"^\d+$")


def _step_blocks(columns: list[str]) -> dict[str, list[str]]:
    """Group step-indexed column names into ``{base: [members in step order]}``.

    Parameters
    ----------
    columns : list of str
        Column names from a step frame, index column included or not.

    Returns
    -------
    dict
        Base name to its step columns, ordered by step index. Columns that carry
        no step index are ignored.

    """
    blocks: dict[str, list[str]] = {}
    for name in columns:
        if name == "time" or not _is_step_indexed(name):
            continue
        base = name.rsplit("_step_", 1)[0]
        blocks.setdefault(base, []).append(name)
    for base, members in blocks.items():
        blocks[base] = sorted(members, key=lambda c: int(c.rsplit("_step_", 1)[1]))
    return blocks


def _require_step_blocks(blocks: dict[str, list[str]], transformer: object, columns: list[str]) -> None:
    """Raise when a step transformer is handed a frame with nothing to reduce.

    Every step transformer reduces ``{base}_step_{h}`` blocks, so a frame carrying
    none of them leaves it with no work. Returning an empty frame there is worse
    than failing: the design matrix silently loses every feature the slot was
    meant to produce, and nothing says so.

    The realistic cause is chaining. A ``FeaturePipeline`` of a reducer followed
    by an aggregator hands the second stage the first stage's output, which is
    horizon-agnostic by construction (``temp_step_c0``), so the second stage has
    nothing to reduce. That is a misconfiguration rather than a data condition,
    and it is worth reporting as one.

    Parameters
    ----------
    blocks : dict
        Base name to its step columns, as returned by :func:`_step_blocks`.
    transformer : object
        The transformer raising, used in the message.
    columns : list of str
        The input frame's columns, used to describe what was received.

    Raises
    ------
    ValueError
        If ``blocks`` is empty.

    """
    if blocks:
        return
    received = [c for c in columns if c != "time"]
    raise ValueError(
        f"{type(transformer).__name__} received a frame with no step blocks to reduce. "
        f"It looks for columns named '{{base}}_step_{{h}}' with an integer step, and found "
        f"none among {received!r}. The usual cause is chaining two step transformers: the "
        f"first one's output is horizon-agnostic, so the second has nothing left to work "
        f"on. Put them in a FeatureUnion if you want both applied to the raw block."
    )


def _reject_numeric_name(name: str) -> None:
    """Raise if an output name would collide with a horizon step index.

    Parameters
    ----------
    name : str
        The proposed output suffix.

    Raises
    ------
    ValueError
        If ``name`` is a bare decimal integer.

    """
    if _NUMERIC_NAME_RE.match(name):
        raise ValueError(
            f"Output name {name!r} is a bare integer, which would produce a column "
            f"named '{{base}}_step_{name}' that is indistinguishable from horizon "
            f"step {name}. Choose a non-numeric name."
        )


class StepAggregator(BaseStepTransformer):
    """Reduce each base column's step block to one column per aggregation.

    Collapses ``{base}_step_1 .. {base}_step_H`` to ``{base}_step_{aggregation}``,
    turning a wide forward window into the handful of numbers a model usually
    needs from it. Output columns carry no horizon index, so they reach every
    per-step estimator regardless of ``step_feature_alignment``.

    Parameters
    ----------
    aggregations : tuple of str, default=("mean",)
        Which summaries to emit, drawn from ``"min"``, ``"max"``, ``"mean"``,
        ``"std"``, and ``"sum"``. The vocabulary is closed; for anything else use
        [`StepColumnReducer`][yohou.preprocessing.step.StepColumnReducer] with a
        `sklearn.preprocessing.FunctionTransformer`.
    null_policy : {"ignore", "propagate"}, default="ignore"
        How to treat a partially covered block. ``"ignore"`` aggregates over the
        steps that carry a value, so a row covering 12 of 48 steps is summarised
        over those 12. ``"propagate"`` yields null for any row with a missing
        step. Neither is right in general, which is why it is a parameter: see
        ``emit_coverage``.
    emit_coverage : bool, default=False
        Whether to emit a companion ``{base}_step_n_covered`` column counting the
        steps that contributed. Off by default because under full coverage it is
        a constant column. Turn it on when coverage varies, so the model can tell
        a mean over 48 steps from a mean over 12.

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
    - [`StepColumnReducer`][yohou.preprocessing.step.StepColumnReducer] : Lift an sklearn transformer per base column.
    - [`StepFrameReducer`][yohou.preprocessing.step.StepFrameReducer] : Lift an sklearn transformer over the whole step frame.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.preprocessing import StepAggregator
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
    ...     "temp_step_1": [1.0, 4.0],
    ...     "temp_step_2": [3.0, 6.0],
    ... })
    >>> StepAggregator(aggregations=("min", "max")).fit_transform(X).columns
    ['time', 'temp_step_min', 'temp_step_max']

    """

    _parameter_constraints: dict = {
        "aggregations": [tuple, list],
        "null_policy": [StrOptions({"ignore", "propagate"})],
        "emit_coverage": ["boolean"],
    }

    def __init__(
        self,
        aggregations: tuple[str, ...] = ("mean",),
        *,
        null_policy: Literal["ignore", "propagate"] = "ignore",
        emit_coverage: bool = False,
    ) -> None:
        self.aggregations = aggregations
        self.null_policy = null_policy
        self.emit_coverage = emit_coverage

    def _check_aggregations(self) -> tuple[str, ...]:
        """Validate the configured aggregations and return them as a tuple.

        Returns
        -------
        tuple of str
            The validated aggregation names.

        Raises
        ------
        ValueError
            If an aggregation is outside the closed vocabulary, or its name would
            collide with a horizon step index.

        """
        aggregations = tuple(self.aggregations)
        if not aggregations:
            raise ValueError("aggregations must name at least one summary.")
        for name in aggregations:
            if name not in _AGGREGATIONS:
                raise ValueError(
                    f"Unknown aggregation {name!r}. Available: {sorted(_AGGREGATIONS)}. "
                    f"The vocabulary is closed; for anything else wrap a "
                    f"FunctionTransformer in a StepColumnReducer."
                )
            _reject_numeric_name(name)
        return aggregations

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Validate configuration and input; the transform itself learns nothing."""
        self._check_aggregations()
        _require_step_blocks(_step_blocks(X.columns), self, X.columns)

    def _aggregate_expr(self, name: str, members: list[str], base: str) -> pl.Expr:
        """Build the polars expression for one aggregation over one block.

        Parameters
        ----------
        name : str
            Aggregation name from the closed vocabulary.
        members : list of str
            The block's step columns, in step order.
        base : str
            Base column name, used for the output alias.

        Returns
        -------
        pl.Expr
            Expression producing ``{base}_step_{name}``.

        """
        alias = f"{base}_step_{name}"
        # No std_horizontal in polars: concat the block into a list column and reduce
        # it, which honours skip_nulls the same way the horizontal helpers do.
        expr = pl.concat_list(members).list.std() if name == "std" else _AGGREGATIONS[name](members)

        if self.null_policy == "propagate":
            # Any missing step invalidates the row's summary. The horizontal
            # helpers skip nulls by default, so the condition is stated explicitly
            # rather than relying on their behaviour.
            any_null = pl.any_horizontal([pl.col(c).is_null() for c in members])
            expr = pl.when(any_null).then(None).otherwise(expr)
        return expr.alias(alias)

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Collapse every step block to its configured aggregations."""
        aggregations = self._check_aggregations()
        blocks = _step_blocks(X.columns)

        exprs: list[pl.Expr] = []
        for base, members in blocks.items():
            for name in aggregations:
                exprs.append(self._aggregate_expr(name, members, base))
            if self.emit_coverage:
                exprs.append(
                    pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in members]).alias(
                        f"{base}_step_n_covered"
                    )
                )

        return X.select(pl.col("time"), *exprs)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names.

        Parameters
        ----------
        input_features : list of str or None, default=None
            Input column names. Defaults to those seen during ``fit``.

        Returns
        -------
        list of str
            One name per (base column, aggregation) pair, plus a coverage
            companion per base column when ``emit_coverage`` is set.

        """
        features = list(input_features) if input_features is not None else list(self.feature_names_in_)
        names: list[str] = []
        for base in _step_blocks(features):
            names.extend(f"{base}_step_{name}" for name in self.aggregations)
            if self.emit_coverage:
                names.append(f"{base}_step_n_covered")
        return names


class _BaseStepReducer(BaseStepTransformer):
    """Shared machinery for the sklearn-wrapping step transformers.

    Owns cloning, the width guards, the null guard, and feature naming. Subclasses
    supply only how the step frame is reshaped into the tables the inner estimator
    sees, and how outputs are named.
    """

    _parameter_constraints: dict = {
        "reducer": [BaseEstimator],
    }

    #: Set by every concrete subclass's ``__init__``.
    reducer: BaseEstimator
    #: Output column names, recorded at fit by every concrete subclass.
    feature_names_out_: list[str]

    @property
    def min_steps(self) -> int:
        """Get the smallest forecasting horizon this wrapper can work on.

        Read from the inner estimator's ``n_components`` where it exposes one, so
        a projection onto ``k`` components reports ``k``: fewer step columns than
        components is not a shortfall the estimator can absorb. Anything without
        that attribute reports ``1``, which is the honest answer rather than a
        guess, since no other attribute states a width requirement.

        Introspection is best-effort for the same reason as in
        :meth:`_check_fixed_width`: ``n_components`` is a convention across
        sklearn's reducers rather than part of any interface. What it misses is
        caught at fit by the inner estimator itself.

        Returns
        -------
        int
            Smallest usable forecasting horizon, at least ``1``.

        """
        n_components = getattr(self.reducer, "n_components", None)
        if isinstance(n_components, int) and not isinstance(n_components, bool) and n_components >= 1:
            return n_components
        return 1

    def _check_fixed_width(self) -> None:
        """Reject an inner estimator whose output width is data-determined.

        Best-effort and deliberately narrow: ``n_components`` is a convention
        across sklearn's reducers, not part of any interface, and width-preserving
        transformers do not carry it at all. So this catches the common mistake
        early and the post-fit uniformity check carries the real guarantee.

        Raises
        ------
        ValueError
            If ``n_components`` is present and is not a positive integer.

        """
        # hasattr, not getattr-with-default: an estimator carrying
        # ``n_components=None`` (PCA's "keep everything") is data-determined and must
        # be refused, whereas one with no such attribute at all (a scaler) is
        # width-preserving and fine. A default of None conflates the two.
        if not hasattr(self.reducer, "n_components"):
            return
        n_components = self.reducer.n_components
        if not isinstance(n_components, int) or isinstance(n_components, bool) or n_components < 1:
            raise ValueError(
                f"{type(self).__name__} requires an inner estimator whose output width is "
                f"fixed before fit, but {type(self.reducer).__name__}.n_components is "
                f"{n_components!r}. A data-determined width (a float variance ratio, None, "
                f"or 'mle') cannot satisfy the panel schema contract, which derives one "
                f"local schema from the first group and applies it to every group. Pass a "
                f"positive integer instead."
            )

    def _fit_block(self, values: np.ndarray, label: str, horizon: int) -> tuple[Any, np.ndarray]:
        """Fit a clone of ``reducer`` on one block, diagnosing partial coverage.

        Partial coverage is a supported state upstream, so a block legitimately
        arrives with missing steps. Whether that is a problem is the inner
        estimator's business, and the answer is obtained by asking it to fit
        rather than by reading its ``allow_nan`` tag.

        The tag cannot answer this. scikit-learn takes a ``Pipeline``'s input tag
        from its **last** step, so ``Pipeline([SimpleImputer(), PCA()])`` reports
        ``allow_nan=False`` while fitting missing values perfectly well. A tag gate
        would reject the very composition this method recommends as the fix, and
        would misjudge any other wrapper that aggregates tags the same way.

        Trying is therefore both simpler and strictly more accurate: nothing is
        refused that would have worked. The cost is that the diagnosis is added
        after the failure rather than before the attempt.

        Parameters
        ----------
        values : np.ndarray
            The block's values, one row per observation and one column per step.
        label : str
            Base column name (or the frame prefix), used in the message.
        horizon : int
            Number of step columns per base column.

        Returns
        -------
        tuple
            The fitted clone and its output array.

        Raises
        ------
        ValueError
            If the fit fails on a block that carries missing values. The inner
            exception is chained, so the estimator's own error stays visible.

        """
        try:
            fitted = clone(self.reducer)
            out = np.asarray(fitted.fit_transform(values))
        except Exception as exc:
            if not _has_missing(values):
                # Nothing to do with coverage; let the real error through untouched.
                raise
            n_rows = int(np.isnan(values).any(axis=1).sum())
            worst = int((~np.isnan(values)).sum(axis=1).min())
            raise ValueError(
                f"Step block {label!r} is only partially covered: {n_rows} rows carry a "
                f"missing step, reaching {worst} of {horizon} steps at worst, and "
                f"{type(self.reducer).__name__} failed on it. Either compose imputation "
                f"into the inner estimator (Pipeline([SimpleImputer(), ...])), or use "
                f"StepAggregator, whose null_policy handles partial coverage directly. "
                f"The estimator's own error is chained below."
            ) from exc
        return fitted, out

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names recorded at ``fit``.

        Parameters
        ----------
        input_features : list of str or None, default=None
            Ignored; names depend on the fitted inner estimators.

        Returns
        -------
        list of str
            Output column names in order.

        """
        return list(self.feature_names_out_)


def _has_missing(values: np.ndarray) -> bool:
    """Return whether a block carries missing values.

    Only float and complex arrays can hold them: polars upcasts an integer or
    boolean column to float when it contains nulls, so an integer array is
    missing-free by construction and ``np.isnan`` would raise on it.

    Parameters
    ----------
    values : np.ndarray
        A block's values.

    Returns
    -------
    bool
        ``True`` if any entry is NaN.

    """
    if values.dtype.kind not in "fc":
        return False
    return bool(np.isnan(values).any())


class StepColumnReducer(_BaseStepReducer):
    r"""Lift an sklearn transformer onto the step axis, one estimator per base column.

    Each base column's ``H`` step columns become an ``(n_observations, H)`` table,
    and one clone of ``reducer`` is fitted per base column. A wrapped
    `sklearn.preprocessing.StandardScaler` therefore standardises each step
    position independently; a wrapped `sklearn.decomposition.PCA` reduces each
    variable's horizon profile on its own terms, without blending variables.

    Output columns are named ``{base}_step_c{k}`` for ``k`` in ``0..n-1``. The
    ``c`` prefix is load-bearing rather than decorative: a bare integer would make
    the name match the ``_step_(\\d+)$`` horizon pattern, and
    ``step_feature_alignment`` would then filter learned components as though they
    were horizon steps. Per-variable provenance survives into the design matrix.

    Parameters
    ----------
    reducer : sklearn transformer
        Fitted per base column on an ``(n_observations, H)`` table. Its output
        width must be fixed before fit; see Notes.

    Attributes
    ----------
    reducers_ : dict[str, sklearn transformer]
        Fitted clone per base column.
    feature_names_out_ : list[str]
        Output column names, in order.

    Notes
    -----
    Output width must not depend on the data. Under ``panel_strategy="global"`` a
    forecaster fits one step transformer per panel group and derives a single
    local schema from the first group, so groups producing different widths would
    break extraction. An ``n_components`` that is a float, ``None``, or ``"mle"``
    is rejected at fit.

    A partially covered block is refused only when the inner estimator actually
    fails on it, which is established by attempting the fit rather than by reading
    its ``allow_nan`` tag. Compose a `sklearn.impute.SimpleImputer` into the inner
    estimator, or use [`StepAggregator`][yohou.preprocessing.step.StepAggregator],
    whose ``null_policy`` handles partial coverage directly.

    See Also
    --------
    - [`StepFrameReducer`][yohou.preprocessing.step.StepFrameReducer] : One estimator over the whole step frame.
    - [`StepAggregator`][yohou.preprocessing.step.StepAggregator] : Closed-vocabulary arithmetic reduction.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from sklearn.preprocessing import StandardScaler
    >>> from yohou.preprocessing import StepColumnReducer
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
    ...     "temp_step_1": [1.0, 3.0],
    ...     "temp_step_2": [2.0, 6.0],
    ... })
    >>> StepColumnReducer(reducer=StandardScaler()).fit_transform(X).columns
    ['time', 'temp_step_c0', 'temp_step_c1']

    """

    def __init__(self, reducer: BaseEstimator) -> None:
        self.reducer = reducer

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Fit one clone per base column and record the output names."""
        self._check_fixed_width()
        blocks = _step_blocks(X.columns)
        _require_step_blocks(blocks, self, X.columns)

        horizon = len(next(iter(blocks.values())))

        self.reducers_ = {}
        widths: dict[str, int] = {}
        for base, members in blocks.items():
            fitted, out = self._fit_block(X.select(members).to_numpy(), base, horizon)
            self.reducers_[base] = fitted
            widths[base] = out.shape[1]

        self.feature_names_out_ = [f"{base}_step_c{k}" for base, w in widths.items() for k in range(w)]

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Apply each base column's fitted clone to its block."""
        blocks = _step_blocks(X.columns)
        out = X.select(pl.col("time"))
        for base, members in blocks.items():
            values = np.asarray(self.reducers_[base].transform(X.select(members).to_numpy()))
            out = out.hstack(
                pl.DataFrame(
                    {f"{base}_step_c{k}": values[:, k].astype(float) for k in range(values.shape[1])},
                )
            )
        return out


class StepFrameReducer(_BaseStepReducer):
    """Lift an sklearn transformer onto the step axis, one estimator for the whole frame.

    Every step column of every base column becomes one
    ``(n_observations, n_base * H)`` table, and a single clone of ``reducer`` is
    fitted on it. This captures structure across variables that
    [`StepColumnReducer`][yohou.preprocessing.step.StepColumnReducer] cannot see,
    at the cost of provenance: an output column describes no single base column.

    Parameters
    ----------
    reducer : sklearn transformer
        Fitted once on the whole step frame. Its output width must be fixed
        before fit; see Notes.
    prefix : str
        Names the output block. Output columns are ``{prefix}_step_c{k}``, which
        keeps the ``{base}_step_{name}`` convention intact for output that has no
        natural base column. Required, because a default would silently collide
        between two frame reducers in one composition.

    Attributes
    ----------
    reducer_ : sklearn transformer
        The fitted clone.
    feature_names_out_ : list[str]
        Output column names, in order.

    Notes
    -----
    Under panel data with ``panel_strategy="global"``, global (shared) step
    columns are folded into every group's frame and are blended into the
    components here, so a shared channel cannot be recovered downstream. That is
    inherent to whole-frame reduction rather than a defect; use
    [`StepColumnReducer`][yohou.preprocessing.step.StepColumnReducer] when
    per-variable provenance matters.

    The fixed-width and null guards are the same as on
    [`StepColumnReducer`][yohou.preprocessing.step.StepColumnReducer].

    See Also
    --------
    - [`StepColumnReducer`][yohou.preprocessing.step.StepColumnReducer] : One estimator per base column.
    - [`StepAggregator`][yohou.preprocessing.step.StepAggregator] : Closed-vocabulary arithmetic reduction.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from sklearn.preprocessing import StandardScaler
    >>> from yohou.preprocessing import StepFrameReducer
    >>> X = pl.DataFrame({
    ...     "time": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
    ...     "temp_step_1": [1.0, 3.0],
    ...     "rain_step_1": [0.0, 2.0],
    ... })
    >>> StepFrameReducer(reducer=StandardScaler(), prefix="wx").fit_transform(X).columns
    ['time', 'wx_step_c0', 'wx_step_c1']

    """

    _parameter_constraints: dict = {
        "prefix": [str],
    }

    def __init__(self, reducer: BaseEstimator, *, prefix: str) -> None:
        self.reducer = reducer
        self.prefix = prefix

    def _ordered_members(self, X: pl.DataFrame) -> list[str]:
        """Return every step column in a deterministic order."""
        blocks = _step_blocks(X.columns)
        return [c for base in sorted(blocks) for c in blocks[base]]

    def _fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> None:
        """Fit one clone on the whole step frame and record the output names."""
        if not self.prefix:
            raise ValueError("prefix must be a non-empty string; it names the output block.")
        _reject_numeric_name(self.prefix)
        self._check_fixed_width()

        blocks = _step_blocks(X.columns)
        _require_step_blocks(blocks, self, X.columns)

        horizon = len(next(iter(blocks.values())))

        members = self._ordered_members(X)
        self.reducer_, out = self._fit_block(X.select(members).to_numpy(), self.prefix, horizon)
        self.feature_names_out_ = [f"{self.prefix}_step_c{k}" for k in range(out.shape[1])]

    def _transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Apply the fitted clone to the whole step frame."""
        members = self._ordered_members(X)
        values = np.asarray(self.reducer_.transform(X.select(members).to_numpy()))
        return X.select(pl.col("time")).hstack(
            pl.DataFrame({f"{self.prefix}_step_c{k}": values[:, k].astype(float) for k in range(values.shape[1])})
        )
