"""Utility functions for compose module."""

from typing import Any

import polars as pl
import polars.selectors as cs

from yohou.base import BaseTransformer


def _hstack(Xs: list[pl.DataFrame], column_names: list[list[str]], observation_horizons: list[int]) -> pl.DataFrame:
    """Stack transformed features horizontally, aligning observation horizons.

    Aligns transformer outputs by their ``"time"`` column, keeping only the
    intersection of timestamps across all DataFrames.  This handles
    transformers that may drop different numbers of initial rows (e.g.,
    due to different ``observation_horizon`` values).

    Parameters
    ----------
    Xs : list of pl.DataFrame
        List of transformed DataFrames, each containing a ``"time"`` column.

    column_names : list of list of str
        Column names for each DataFrame.

    observation_horizons : list of int
        Observation horizon for each transformer. Accepted for API
        consistency; currently unused because alignment is driven entirely
        by the ``"time"`` column intersection.

    Returns
    -------
    pl.DataFrame
        Horizontally concatenated features aligned by time.

    """
    # Find common time range across all transformer outputs
    common_times = Xs[0].select(cs.by_name("time"))
    for X in Xs[1:]:
        common_times = common_times.join(X.select(cs.by_name("time")), on="time", how="inner")

    # Align each output to common times, then rename columns
    time = Xs[0].join(common_times, on="time", how="semi").select(cs.by_name("time"))

    Xs_renamed = []
    for X, cols in zip(Xs, column_names, strict=False):
        X_aligned = X.join(common_times, on="time", how="semi")
        X_no_time = X_aligned.select(~cs.by_name("time"))
        rename_map = dict(zip(X_no_time.columns, cols, strict=False))
        X_renamed = X_no_time.rename(rename_map)
        Xs_renamed.append(X_renamed)

    Xs_concat = pl.concat(Xs_renamed, how="horizontal")
    result = pl.concat([time, Xs_concat], how="horizontal")

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
        Weight to apply to transformed output.
    params : Any
        Routed parameters for the transformer.

    Returns
    -------
    pl.DataFrame
        Transformed data.

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
        Weight to apply to transformed output.
    params : Any
        Routed parameters for the transformer.

    Returns
    -------
    pl.DataFrame
        Transformed data with warmup rows discarded.

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
