"""Utility functions for compose module."""

from typing import Any

import polars as pl
import polars.selectors as cs


def _hstack(Xs: list[pl.DataFrame], column_names: list[list[str]], observation_horizons: list[int]) -> pl.DataFrame:
    """Stack transformed features horizontally, aligning observation horizons.

    Parameters
    ----------
    Xs : list of pl.DataFrame
        List of transformed DataFrames.

    column_names : list of list of str
        Column names for each DataFrame.

    observation_horizons : list of int
        Observation horizon for each transformer.

    Returns
    -------
    pl.DataFrame
        Horizontally concatenated features.

    """
    ref_observation_horizon = max(observation_horizons)
    time = Xs[0].select(cs.by_name("time"))[ref_observation_horizon - observation_horizons[0] :]

    # Rename columns before concat to avoid duplicates
    Xs_renamed = []
    col_idx = 0
    for X, observation_horizon, cols in zip(Xs, observation_horizons, column_names, strict=False):
        X_no_time = X.select(~cs.by_name("time"))[ref_observation_horizon - observation_horizon :]
        # Create rename mapping for this transformer's columns
        rename_map = dict(zip(X_no_time.columns, cols, strict=False))
        X_renamed = X_no_time.rename(rename_map)
        Xs_renamed.append(X_renamed)
        col_idx += len(cols)

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
