"""Utility functions for compose module."""

from typing import Any

import polars as pl
import polars.selectors as cs


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
        Observation horizon for each transformer (used for fallback when
        ``"time"`` column is absent).

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
    # Handle passthrough/FunctionTransformer which doesn't have observe_transform
    if hasattr(transformer, "observe_transform"):
        X_transformed = transformer.observe_transform(X, **params.get("observe_transform", {}))
    else:
        # Fall back to transform for stateless transformers (e.g., FunctionTransformer for passthrough)
        X_transformed = transformer.transform(X)

    if weight is None:
        return X_transformed
    return X_transformed * weight


def _rewind_transform_one(
    transformer: Any, X: pl.DataFrame, y: None, weight: float | None, params: Any
) -> pl.DataFrame:
    """Rewind and transform data using a single transformer.

    Applies rewind_transform semantics: transforms from scratch without using
    pre-existing memory, discards the first observation_horizon rows, and
    rewinds the internal state with the input data.

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
    # Handle passthrough/transformers that might not have rewind_transform
    if hasattr(transformer, "rewind_transform"):
        X_transformed = transformer.rewind_transform(X, **params.get("rewind_transform", {}))
    else:
        # Fall back to transform for stateless transformers without rewind_transform
        X_transformed = transformer.transform(X)

    if weight is None:
        return X_transformed
    return X_transformed * weight
