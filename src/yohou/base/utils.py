"""Utility functions for forecaster transformations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs
from sklearn.base import clone

if TYPE_CHECKING:
    from yohou.base import BaseTransformer


def _fit_transform_transformers_one(
    y: pl.DataFrame,
    X: pl.DataFrame | None,
    target_transformer: BaseTransformer | None,
    feature_transformer: BaseTransformer | None,
    target_as_feature: str | None,
) -> tuple[pl.DataFrame, pl.DataFrame | None, BaseTransformer | None, BaseTransformer | None]:
    """Fit and apply target and feature transformers to a single time series.

    Orchestrates the transformation pipeline: target transformer first (if any),
    then feature transformer (if any). Handles observation horizon alignment to
    ensure transformed data matches temporally.

    Parameters
    ----------
    y : pl.DataFrame
        Target time series with "time" column.
    X : pl.DataFrame or None
        Feature time series with "time" column.
    target_transformer : BaseTransformer or None
        Target transformer to apply.
    feature_transformer : BaseTransformer or None
        Feature transformer to apply.
    target_as_feature : {"transformed", "raw"} or None
        Controls whether the target is included as a feature.
        ``"transformed"`` includes the target after ``target_transformer``,
        ``"raw"`` includes the original target, and ``None`` uses only
        exogenous features.

    Returns
    -------
    y_t : pl.DataFrame
        Transformed target time series.
    X_t : pl.DataFrame or None
        Transformed feature matrix (includes transformed y if no separate X provided).
    target_transformer : BaseTransformer or None
        Fitted target transformer.
    feature_transformer : BaseTransformer or None
        Fitted feature transformer.

    Notes
    -----
    Transformation order matters:
    1. Apply target_transformer to y → y_t
    2. Concatenate y_t with X (aligned by observation horizon)
    3. Apply feature_transformer to combined → X_t
    4. Trim y_t if feature transformer has its own observation horizon

    This ensures features can include lagged versions of the transformed target.

    See Also
    --------
    `BaseTransformer` : Base class for transformers

    """
    y_t = y
    target_transformer_fitted = None
    if target_transformer is not None:
        target_transformer_fitted = clone(target_transformer)
        y_t = target_transformer_fitted.fit_transform(y)

    X_feat_in = _build_feature_input(y, y_t, X, target_as_feature, feature_transformer)

    X_t = X_feat_in
    feature_transformer_fitted = None
    if feature_transformer is not None and X_feat_in is not None:
        feature_transformer_fitted = clone(feature_transformer)
        X_t = feature_transformer_fitted.fit_transform(X_feat_in)
        feature_observation_horizon = feature_transformer_fitted.observation_horizon
        # Trim y_t to align with X_t
        # First, align by feature transformer's observation horizon (handles transformers that don't drop rows)
        y_t = y_t[feature_observation_horizon:]
        # Also trim X_t: drop null rows produced by transformers that keep all rows
        # but fill initial positions with null (e.g., LagTransformer, RollingStatisticsTransformer)
        data_cols = [c for c in X_t.columns if c != "time"]
        X_t = X_t.drop_nulls(subset=data_cols)
        # Then, align by timestamps (handles transformers that DO drop rows)
        y_t = y_t.join(X_t.select("time"), on="time", how="semi")

    return y_t, X_t, target_transformer_fitted, feature_transformer_fitted


def _build_feature_input(
    y: pl.DataFrame,
    y_t: pl.DataFrame,
    X: pl.DataFrame | None,
    target_as_feature: str | None,
    feature_transformer: BaseTransformer | None,
) -> pl.DataFrame | None:
    """Build feature input based on target_as_feature parameter.

    Constructs the input to the feature_transformer by combining original y,
    transformed y_t, and exogenous features X according to the
    target_as_feature configuration.

    Parameters
    ----------
    y : pl.DataFrame
        Original target time series (untransformed).
    y_t : pl.DataFrame
        Transformed target time series.
    X : pl.DataFrame or None
        Exogenous feature time series.
    target_as_feature : {"transformed", "raw"} or None
        Controls whether the target is included as a feature.
        ``"transformed"`` includes the target after ``target_transformer``,
        ``"raw"`` includes the original target, and ``None`` uses only
        exogenous features.
    feature_transformer : BaseTransformer or None
        Feature transformer (used for validation when
        ``target_as_feature=None``).

    Returns
    -------
    pl.DataFrame or None
        Feature input for feature_transformer.

    Notes
    -----
    The target_as_feature parameter controls what features are available:
    - ``"transformed"``: Transformed target + exogenous features (default)
    - ``"raw"``: Original target + exogenous features
    - ``None``: Only exogenous features (no target)

    For ``"raw"``, the original y is aligned with y_t by taking rows from
    target_observation_horizon onwards to match the transformed data.

    """
    if target_as_feature == "transformed":
        # Default: use transformed target
        X_feat_in = y_t
        if X is not None:
            # Align X to y_t timestamps before concatenation (y_t may be shorter after transformations)
            X_aligned = X.join(y_t.select("time"), on="time", how="semi")
            X_feat_in = pl.concat(
                [y_t, X_aligned.select(~cs.by_name("time"))],
                how="horizontal",
            )
    elif target_as_feature == "raw":
        # Use original target (aligned with transformed data)
        # Align y to y_t length (y_t might be shorter after transformations)
        y_aligned = y.join(y_t.select("time"), on="time", how="semi")
        X_feat_in = y_aligned
        if X is not None:
            # Also align X to y_t timestamps
            X_aligned = X.join(y_t.select("time"), on="time", how="semi")
            X_feat_in = pl.concat(
                [y_aligned, X_aligned.select(~cs.by_name("time"))],
                how="horizontal",
            )
    elif target_as_feature is None:
        # Only exogenous features
        if X is None:
            if feature_transformer is not None:
                # This should not happen since _validate_pre_fit checks at fit
                # time, but guard against direct calls.
                raise ValueError(
                    "target_as_feature=None requires X to be provided when a feature_transformer is set, but X is None."
                )
            else:
                X_feat_in = None
        else:
            # Align X to y_t timestamps (y_t may be shorter after target
            # transformer), consistent with the "transformed"/"raw" branches.
            X_aligned = X.join(y_t.select("time"), on="time", how="semi")
            X_feat_in = X_aligned
    else:
        raise ValueError(
            f"Invalid target_as_feature={target_as_feature!r}. Must be one of: 'transformed', 'raw', or None."
        )

    return X_feat_in


def _observe_transformers_one(
    y: pl.DataFrame,
    X: pl.DataFrame | None,
    target_transformer: BaseTransformer | None,
    feature_transformer: BaseTransformer | None,
    target_as_feature: str | None,
) -> pl.DataFrame | None:
    """Observe new data through transformers.

    Parameters
    ----------
    y : pl.DataFrame
        New target observations.
    X : pl.DataFrame or None
        New features.
    target_transformer : BaseTransformer or None
        Target transformer to observe.
    feature_transformer : BaseTransformer or None
        Feature transformer to observe.
    target_as_feature : {"transformed", "raw"} or None
        Controls whether the target is included as a feature.

    Returns
    -------
    pl.DataFrame or None
        Transformed new observations.

    """
    y_t = y
    if target_transformer is not None:
        y_t = target_transformer.observe_transform(y)

    X_feat_in = _build_feature_input(y, y_t, X, target_as_feature, feature_transformer)

    X_t = X_feat_in
    if feature_transformer is not None and X_feat_in is not None:
        X_t = feature_transformer.observe_transform(X_feat_in)

    return X_t


def _rewind_transformers_one(
    y: pl.DataFrame,
    X: pl.DataFrame | None,
    target_transformer: BaseTransformer | None,
    feature_transformer: BaseTransformer | None,
    observation_horizon: int,
    target_as_feature: str | None,
) -> pl.DataFrame | None:
    """Rewind transformers.

    Parameters
    ----------
    y : pl.DataFrame
        New target observations.
    X : pl.DataFrame or None
        New features.
    target_transformer : BaseTransformer or None
        Target transformer to rewind.
    feature_transformer : BaseTransformer or None
        Feature transformer to rewind.
    observation_horizon : int
        Number of time steps to retain in observation horizon.
    target_as_feature : {"transformed", "raw"} or None
        Controls whether the target is included as a feature.

    Returns
    -------
    pl.DataFrame or None
        Transformed new observations.

    """
    y_t = y

    if target_transformer is not None:
        target_transformer.rewind(X=y[:-observation_horizon])
        y_t = target_transformer.observe_transform(y[-observation_horizon:])

    X_feat_in = _build_feature_input(y, y_t, X, target_as_feature, feature_transformer)

    X_t = X_feat_in
    if feature_transformer is not None and X_feat_in is not None:
        feature_observation_horizon = feature_transformer.observation_horizon

        # X_feat_in is aligned to y_t timestamps (observation_horizon rows)
        # but the feature transformer may need more rows for its own rewind.
        # Widen by transforming earlier rows through the target transformer.
        needed = feature_observation_horizon + 1
        if len(X_feat_in) < needed and target_transformer is not None:
            target_obs = target_transformer.observation_horizon
            deficit = needed - len(X_feat_in)
            # Extra rows just before the y_t window. These go through
            # rewind_transform so the target transformer state is unchanged.
            start = max(0, len(y) - observation_horizon - deficit - target_obs)
            end = len(y) - observation_horizon
            y_extra = y[start:end]
            X_extra = X[start:end] if X is not None else None
            y_t_extra = target_transformer.rewind_transform(y_extra) if len(y_extra) > target_obs else y_extra
            X_feat_extra = _build_feature_input(y_extra, y_t_extra, X_extra, target_as_feature, feature_transformer)
            if X_feat_extra is not None:
                # Only keep the tail; rewind_transform may drop observation_horizon rows
                X_feat_extra = X_feat_extra.tail(deficit)
                X_feat_in = pl.concat([X_feat_extra, X_feat_in], how="vertical")

        # Use rewind_transform (combined operation) instead of separate
        # rewind() + observe_transform().  Composite transformers such as
        # FeatureUnion only implement the combined method; calling rewind()
        # directly on them raises NotFittedError because their fit() does
        # not set the base-class fitted attributes.
        X_t_all = feature_transformer.rewind_transform(X_feat_in)
        X_t = X_t_all[[-1]]

    return X_t
