"""Utility functions for compose module."""

from typing import Any, Literal, cast

import polars as pl
import polars.selectors as cs

from yohou.base import BaseTransformer

#: The kind of frame a transformer consumes/produces.
Kind = Literal["actual", "forecast"]

#: Reserved index columns, in canonical order. A single-axis ("actual") frame
#: carries ``["time"]``; an ``X_forecast`` ("forecast") frame carries
#: ``["vintage_time", "time"]``. Every other column is a feature.
_INDEX_COLUMNS: tuple[str, str] = ("vintage_time", "time")


def index_columns(df: pl.DataFrame) -> list[str]:
    """Return the reserved index columns present in ``df``, in canonical order.

    Parameters
    ----------
    df : pl.DataFrame
        A transformer input or output frame.

    Returns
    -------
    list of str
        ``["time"]`` for a single-axis frame, ``["vintage_time", "time"]`` for
        an ``X_forecast`` frame.

    """
    return [c for c in _INDEX_COLUMNS if c in df.columns]


def transformer_kind(transformer: Any) -> Kind:
    """Return a transformer's ``kind`` tag (``"actual"`` or ``"forecast"``).

    Defaults to ``"actual"`` for anything without a readable ``kind`` tag.

    Parameters
    ----------
    transformer : Any
        A transformer instance.

    Returns
    -------
    {"actual", "forecast"}
        The transformer's kind.

    """
    # Anything without readable transformer tags (e.g. an invalid step) defaults
    # to "actual" so homogeneity checking never masks the real validation error.
    get_tags = getattr(transformer, "__sklearn_tags__", None)
    if get_tags is None:
        return "actual"
    tags = get_tags()
    transformer_tags = getattr(tags, "transformer_tags", None)
    if transformer_tags is None:
        return "actual"
    return cast(Kind, getattr(transformer_tags, "kind", "actual"))


def _real_transformers(named_transformers: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """Drop ``"drop"``/``"passthrough"``/``None`` entries, keeping real transformers."""
    return [(name, t) for name, t in named_transformers if t is not None and t not in ("drop", "passthrough")]


def common_kind(named_transformers: list[tuple[str, Any]]) -> Kind:
    """Return the shared kind of the children, or ``"actual"`` if empty or mixed.

    Non-raising; use for tag computation. Use :func:`check_homogeneous_kinds`
    to enforce homogeneity with a descriptive error at fit time.

    Parameters
    ----------
    named_transformers : list of (str, Any)
        ``(name, transformer)`` pairs; ``"drop"``/``"passthrough"``/``None`` are ignored.

    Returns
    -------
    str
        The single child kind, or ``"actual"`` when there are no real children
        or the children disagree.

    """
    kinds = {transformer_kind(t) for _, t in _real_transformers(named_transformers)}
    return kinds.pop() if len(kinds) == 1 else "actual"


def check_homogeneous_kinds(named_transformers: list[tuple[str, Any]], composer_name: str) -> Kind:
    """Enforce that all children share one kind, returning it.

    Parameters
    ----------
    named_transformers : list of (str, Any)
        ``(name, transformer)`` pairs; ``"drop"``/``"passthrough"``/``None`` are ignored.
    composer_name : str
        Name of the composing estimator, used in the error message.

    Returns
    -------
    str
        The single shared kind (``"actual"`` when there are no real children).

    Raises
    ------
    ValueError
        If the children mix ``"actual"`` and ``"forecast"`` kinds.

    """
    kinds = {name: transformer_kind(t) for name, t in _real_transformers(named_transformers)}
    if len(set(kinds.values())) > 1:
        actual = sorted(n for n, k in kinds.items() if k == "actual")
        forecast = sorted(n for n, k in kinds.items() if k == "forecast")
        raise ValueError(
            f"{composer_name} cannot mix actual and forecast transformers in one composition: "
            f"actual={actual}, forecast={forecast}. A composition must be homogeneous in kind."
        )
    return next(iter(kinds.values())) if kinds else "actual"


def _hstack(Xs: list[pl.DataFrame], column_names: list[list[str]]) -> pl.DataFrame:
    """Stack transformed features horizontally, aligning on the index columns.

    Aligns transformer outputs by their index columns (``["time"]`` for
    single-axis frames, ``["vintage_time", "time"]`` for ``X_forecast`` frames),
    keeping only the intersection of index rows across all DataFrames.  This
    handles transformers that may drop different numbers of initial rows (e.g.,
    due to different ``observation_horizon`` values).

    Parameters
    ----------
    Xs : list of pl.DataFrame
        List of transformed DataFrames, each carrying the same index columns.

    column_names : list of list of str
        New column names for the non-index columns of each DataFrame in
        ``Xs``; each inner list must have the same length as the non-index
        columns of the corresponding DataFrame.

    Returns
    -------
    pl.DataFrame
        Horizontally concatenated features aligned on the index columns.

    """
    index = index_columns(Xs[0])

    # Find the common index rows across all transformer outputs.
    common = Xs[0].select(index)
    for X in Xs[1:]:
        common = common.join(X.select(index), on=index, how="inner")

    # Align each output to the common index, then rename its feature columns.
    index_frame = Xs[0].join(common, on=index, how="semi").select(index)

    Xs_renamed = []
    for X, cols in zip(Xs, column_names, strict=False):
        X_aligned = X.join(common, on=index, how="semi")
        X_no_index = X_aligned.select(pl.exclude(index))
        rename_map = dict(zip(X_no_index.columns, cols, strict=False))
        X_renamed = X_no_index.rename(rename_map)
        Xs_renamed.append(X_renamed)

    Xs_concat = pl.concat(Xs_renamed, how="horizontal")
    result = pl.concat([index_frame, Xs_concat], how="horizontal")

    return result


def _observe_transform_one(
    transformer: Any, X: pl.DataFrame, y: None, weight: float | None, params: Any
) -> pl.DataFrame:
    """Observe and transform data using a single transformer.

    Parameters
    ----------
    transformer : estimator
        The transformer to observe and transform with.
    X : pl.DataFrame
        Input data to observe and transform.
    y : None
        Not used, present for API consistency.
    weight : float | None
        Weight to apply to the feature (non-``"time"``) columns of the
        transformed output; the ``"time"`` column is never scaled so its
        datetime dtype is preserved.
    params : Any
        Routed parameters for the transformer.

    Returns
    -------
    pl.DataFrame
        Transformed data.

    Raises
    ------
    AttributeError
        If ``transformer`` is a ``BaseTransformer`` subclass that does not
        expose ``observe_transform``.

    """
    # Stateful BaseTransformers must expose observe_transform; the transform()
    # fallback is only legitimate for stateless transformers (e.g.
    # FunctionTransformer used for passthrough). A BaseTransformer that reaches
    # the fallback has lost its stateful method and would silently use stale
    # state, so surface that as an error.
    if hasattr(transformer, "observe_transform"):
        X_transformed = transformer.observe_transform(X, **params.get("observe_transform", {}))
    elif isinstance(transformer, BaseTransformer):
        raise AttributeError(
            f"{type(transformer).__name__} is a BaseTransformer but has no "
            "'observe_transform' method; cannot observe-transform it statefully."
        )
    else:
        X_transformed = transformer.transform(X)

    if weight is None:
        return X_transformed
    # Scale only feature columns: multiplying the whole DataFrame would cast the
    # mandatory datetime "time" column to f64, violating the data contract.
    return X_transformed.with_columns(cs.exclude("time") * weight)


def _rewind_transform_one(
    transformer: Any, X: pl.DataFrame, y: None, weight: float | None, params: Any
) -> pl.DataFrame:
    """Rewind and transform data using a single transformer.

    Delegates to ``transformer.rewind_transform()``, which transforms from
    scratch without using pre-existing memory, discards warmup rows, and
    rewinds the internal state with the input data. This wrapper does not
    perform the row-discarding itself.

    Parameters
    ----------
    transformer : estimator
        The transformer to rewind and transform with.
    X : pl.DataFrame
        Input data to transform and use for rewinding state.
    y : None
        Not used, present for API consistency.
    weight : float | None
        Weight to apply to the feature (non-``"time"``) columns of the
        transformed output; the ``"time"`` column is never scaled so its
        datetime dtype is preserved.
    params : Any
        Routed parameters for the transformer.

    Returns
    -------
    pl.DataFrame
        Transformed data; for stateful transformers (``BaseTransformer``
        subclasses) the first ``observation_horizon`` rows are discarded by
        ``rewind_transform``; stateless fallback transformers return all rows.

    Raises
    ------
    AttributeError
        If ``transformer`` is a ``BaseTransformer`` subclass that does not
        expose ``rewind_transform``.

    """
    # As in _observe_transform_one, the transform() fallback is only legitimate
    # for stateless transformers; a BaseTransformer missing rewind_transform has
    # lost its stateful method and must not silently fall back.
    if hasattr(transformer, "rewind_transform"):
        X_transformed = transformer.rewind_transform(X, **params.get("rewind_transform", {}))
    elif isinstance(transformer, BaseTransformer):
        raise AttributeError(
            f"{type(transformer).__name__} is a BaseTransformer but has no "
            "'rewind_transform' method; cannot rewind-transform it statefully."
        )
    else:
        X_transformed = transformer.transform(X)

    if weight is None:
        return X_transformed
    # Scale only feature columns: multiplying the whole DataFrame would cast the
    # mandatory datetime "time" column to f64, violating the data contract.
    return X_transformed.with_columns(cs.exclude("time") * weight)
