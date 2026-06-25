"""Utility functions for forecaster transformations."""

from __future__ import annotations

import re
import warnings
from datetime import timedelta
from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs
from sklearn.base import clone

if TYPE_CHECKING:
    from yohou.base import BaseTransformer


def _fit_transform_transformers_one(
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None,
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
    X_actual : pl.DataFrame or None
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
        Transformed feature matrix (includes transformed y if no separate X_actual provided).
    target_transformer : BaseTransformer or None
        Fitted target transformer.
    feature_transformer : BaseTransformer or None
        Fitted feature transformer.

    Notes
    -----
    Transformation order matters:
    1. Apply target_transformer to y → y_t
    2. Concatenate y_t with X_actual (aligned by observation horizon)
    3. Apply feature_transformer to combined → X_t
    4. Trim y_t if feature transformer has its own observation horizon

    This ensures features can include lagged versions of the transformed target.

    See Also
    --------
    - [`BaseTransformer`][yohou.base.transformer.BaseTransformer] : Base class for transformers

    """
    y_t = y
    target_transformer_fitted = None
    if target_transformer is not None:
        target_transformer_fitted = clone(target_transformer)
        y_t = target_transformer_fitted.fit_transform(y)

    X_feat_in = _build_feature_input(y, y_t, X_actual, target_as_feature, feature_transformer)

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
        X_t = X_t.drop_nulls(subset=~cs.by_name("time"))
        # Then, align by timestamps (handles transformers that DO drop rows)
        y_t = y_t.join(X_t.select("time"), on="time", how="semi")

    return y_t, X_t, target_transformer_fitted, feature_transformer_fitted


def _build_feature_input(
    y: pl.DataFrame,
    y_t: pl.DataFrame,
    X_actual: pl.DataFrame | None,
    target_as_feature: str | None,
    feature_transformer: BaseTransformer | None,
) -> pl.DataFrame | None:
    """Build feature input based on target_as_feature parameter.

    Constructs the input to the feature_transformer by combining original y,
    transformed y_t, and exogenous features X_actual according to the
    target_as_feature configuration.

    Parameters
    ----------
    y : pl.DataFrame
        Original target time series (untransformed).
    y_t : pl.DataFrame
        Transformed target time series.
    X_actual : pl.DataFrame or None
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
        if X_actual is not None:
            # Align X_actual to y_t timestamps before concatenation (y_t may be shorter after transformations)
            X_aligned = X_actual.join(y_t.select("time"), on="time", how="semi")
            X_feat_in = pl.concat(
                [y_t, X_aligned.select(~cs.by_name("time"))],
                how="horizontal",
            )
    elif target_as_feature == "raw":
        # Use original target (aligned with transformed data)
        # Align y to y_t length (y_t might be shorter after transformations)
        y_aligned = y.join(y_t.select("time"), on="time", how="semi")
        X_feat_in = y_aligned
        if X_actual is not None:
            # Also align X_actual to y_t timestamps
            X_aligned = X_actual.join(y_t.select("time"), on="time", how="semi")
            X_feat_in = pl.concat(
                [y_aligned, X_aligned.select(~cs.by_name("time"))],
                how="horizontal",
            )
    elif target_as_feature is None:
        # Only exogenous features
        if X_actual is None:
            if feature_transformer is not None:
                # This should not happen since _validate_pre_fit checks at fit
                # time, but guard against direct calls.
                raise ValueError(
                    "target_as_feature=None requires X_actual to be provided when a feature_transformer is set, but X_actual is None."
                )
            else:
                X_feat_in = None
        else:
            # Align X_actual to y_t timestamps (y_t may be shorter after target
            # transformer), consistent with the "transformed"/"raw" branches.
            X_aligned = X_actual.join(y_t.select("time"), on="time", how="semi")
            X_feat_in = X_aligned
    else:
        raise ValueError(
            f"Invalid target_as_feature={target_as_feature!r}. Must be one of: 'transformed', 'raw', or None."
        )

    return X_feat_in


def _observe_transformers_one(
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None,
    target_transformer: BaseTransformer | None,
    feature_transformer: BaseTransformer | None,
    target_as_feature: str | None,
) -> pl.DataFrame | None:
    """Observe new data through transformers.

    Parameters
    ----------
    y : pl.DataFrame
        New target observations.
    X_actual : pl.DataFrame or None
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

    X_feat_in = _build_feature_input(y, y_t, X_actual, target_as_feature, feature_transformer)

    X_t = X_feat_in
    if feature_transformer is not None and X_feat_in is not None:
        X_t = feature_transformer.observe_transform(X_feat_in)

    return X_t


def _rewind_transformers_one(
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None,
    target_transformer: BaseTransformer | None,
    feature_transformer: BaseTransformer | None,
    observation_horizon: int,
    target_as_feature: str | None,
) -> pl.DataFrame | None:
    """Rewind transformers.

    Parameters
    ----------
    y : pl.DataFrame
        Historical target time series to rewind state from.
    X_actual : pl.DataFrame or None
        Historical feature observations to rewind state from.
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
        # Use an explicit split index rather than negative slicing: when
        # observation_horizon == 0, y[:-0] is empty and y[-0:] is the full
        # frame, which inverts the rewind/observe windows. len(y) - 0 == len(y)
        # rewinds over all rows and leaves an empty observation window.
        split = len(y) - observation_horizon
        if split < 0:
            raise ValueError(
                f"observation_horizon={observation_horizon} exceeds the number of "
                f"available rows ({len(y)}); not enough data to rewind."
            )
        target_transformer.rewind(X=y[:split])
        y_t = target_transformer.observe_transform(y[split:])

    X_feat_in = _build_feature_input(y, y_t, X_actual, target_as_feature, feature_transformer)

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
            X_extra = X_actual[start:end] if X_actual is not None else None
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
        # Keep the row aligned to the most-recent observation timestamp rather
        # than blindly taking the last row: when the feature transformer drops
        # rows (its own observation_horizon), the surviving tail may not line up
        # with the latest observation.
        last_time = y["time"][-1]
        if "time" in X_t_all.columns:
            X_t = X_t_all.filter(pl.col("time") == last_time)
            if X_t.is_empty():
                X_t = X_t_all.tail(1)
        else:
            X_t = X_t_all.tail(1)

    return X_t


def _derive_step_columns(
    X_future: pl.DataFrame | None,
    X_forecast: pl.DataFrame | None,
    observation_times: pl.Series,
    forecasting_horizon: int,
    interval: str | timedelta,
    *,
    existing_columns: set[str] | None = None,
) -> pl.DataFrame | None:
    """Derive step-indexed columns from X_future and X_forecast.

    Pure function with no forecaster dependency. Pivots raw X_future
    (via windowing) and X_forecast (via ordinal ranking) into wide
    step-indexed columns (``col_step_1`` through ``col_step_H``).

    Parameters
    ----------
    X_future : pl.DataFrame or None
        Known-future features with a ``"time"`` column. Deterministic
        values that are windowed forward from each observation time.
    X_forecast : pl.DataFrame or None
        External forecasts with ``"vintage_time"`` and ``"time"`` columns.
        Before pivoting, each vintage is filtered to timestamps within
        ``(vintage_time, vintage_time + H * interval]``. Timestamps outside
        this window are discarded. The remaining timestamps are pivoted by
        ordinal rank within each vintage group. If filtering produces fewer
        than H step columns, the missing columns are padded with null and
        a ``UserWarning`` is emitted.
    observation_times : pl.Series
        Observation timestamps to derive step columns from.
    forecasting_horizon : int
        Number of forward steps (H) per observation time.
    interval : str or timedelta
        Time frequency between consecutive steps.
    existing_columns : set of str or None, default=None
        Column names already present (e.g., X_actual columns). Used for
        collision detection against generated step column names.

    Returns
    -------
    pl.DataFrame or None
        Wide DataFrame with ``[time, <col>_step_1, ..., <col>_step_H]``
        combining step columns from both sources. Step columns always
        span exactly ``1..H`` per value column: over-long forecasts are
        clipped, and under-coverage is padded with null. Returns ``None``
        when both ``X_future`` and ``X_forecast`` are ``None``.

    Raises
    ------
    ValueError
        If any generated step column name collides with ``existing_columns``
        or appears in both X_future and X_forecast sources.

    """
    from yohou.utils.pivot import window_forecasts, window_futures  # noqa: PLC0415

    if X_future is None and X_forecast is None:
        return None

    parts: list[pl.DataFrame] = []
    source_names: dict[str, str] = {}  # col_name → source label

    if X_future is not None:
        future_pivoted = window_futures(X_future, observation_times, forecasting_horizon, interval)
        step_cols = [c for c in future_pivoted.columns if c != "time"]
        for c in step_cols:
            source_names[c] = "X_future"
        parts.append(future_pivoted)

    if X_forecast is not None:
        # Use as-of vintage selection: for each observation time T, select
        # the latest vintage V <= T, then extract values at T+1..T+H.
        forecast_pivoted = window_forecasts(
            X_forecast,
            observation_times,
            forecasting_horizon,
            interval,
        )

        # Determine value columns and their dtypes for padding
        value_cols_info = {c: X_forecast[c].dtype for c in X_forecast.columns if c not in ("vintage_time", "time")}

        # Warn if any value column has step columns that are entirely null,
        # indicating the matched vintage(s) don't cover the full horizon.
        step_cols_forecast = [c for c in forecast_pivoted.columns if c != "time" and re.search(r"_step_\d+$", c)]
        if step_cols_forecast:
            # Coverage is per value column: a column whose later steps are all
            # null is under-covered even if another column reaches the horizon.
            # Track the highest covered step per base column and take the worst
            # (minimum) so a single under-covered column still warns.
            n_rows = len(forecast_pivoted)
            null_counts = forecast_pivoted.select(step_cols_forecast).null_count().row(0)
            per_col_max: dict[str, int] = {}
            for c, null_count in zip(step_cols_forecast, null_counts, strict=True):
                m = re.search(r"^(.*)_step_(\d+)$", c)
                if m is None:  # pragma: no cover - step_cols_forecast is pre-filtered by `_step_\d+$`
                    continue
                base = m.group(1)
                per_col_max.setdefault(base, 0)
                if null_count < n_rows:
                    per_col_max[base] = max(per_col_max[base], int(m.group(2)))
            max_covered = min(per_col_max.values(), default=0)
            if max_covered < forecasting_horizon:
                warnings.warn(
                    f"X_forecast covers {max_covered} of {forecasting_horizon} "
                    f"forecast steps. The remaining step features will be null. "
                    f"This is normal for short-range forecasts or when the "
                    f"observation point has advanced past some forecast "
                    f"timestamps. Tree-based estimators (e.g. XGBoost, LightGBM, "
                    f"HistGradientBoosting) handle null features natively.",
                    UserWarning,
                    # user -> fit/observe -> _pre_fit(_standard) -> _derive_step_columns
                    # -> warn; point at the user's fit()/observe() call.
                    stacklevel=5,
                )

        # Pad missing step columns to H (partial coverage → null columns)
        missing_exprs = []
        for col, dtype in value_cols_info.items():
            for h in range(1, forecasting_horizon + 1):
                step_name = f"{col}_step_{h}"
                if step_name not in forecast_pivoted.columns:
                    missing_exprs.append(pl.lit(None).cast(dtype).alias(step_name))
        if missing_exprs:
            forecast_pivoted = forecast_pivoted.with_columns(missing_exprs)

        step_cols = [c for c in forecast_pivoted.columns if c != "time"]
        for c in step_cols:
            if c in source_names:
                msg = f"Column name collision between X_future and X_forecast: '{c}' is produced by both sources."
                raise ValueError(msg)
            source_names[c] = "X_forecast"
        parts.append(forecast_pivoted)

    # Check collisions against existing columns (e.g., X_actual)
    if existing_columns is not None:
        collisions = set(source_names.keys()) & existing_columns
        if collisions:
            details = ", ".join(f"'{c}' (from {source_names[c]})" for c in sorted(collisions))
            msg = (
                f"Step column names collide with existing columns: {details}. "
                f"Rename the source columns to avoid conflicts."
            )
            raise ValueError(msg)

    # Combine parts horizontally
    if len(parts) == 1:
        return parts[0]

    # Both X_future and X_forecast present: concat horizontally
    result = pl.concat(
        [parts[0], parts[1].select(~cs.by_name("time"))],
        how="horizontal",
    )
    return result
