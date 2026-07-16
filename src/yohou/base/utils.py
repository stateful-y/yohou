"""Utility functions for forecaster transformations."""

from __future__ import annotations

import inspect
import re
import warnings
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import polars.selectors as cs
import sklearn
from sklearn.base import clone

if TYPE_CHECKING:
    from yohou.base import BaseActualTransformer

_YOHOU_ROOT = str(Path(__file__).resolve().parents[1])
_SKLEARN_ROOT = str(Path(sklearn.__file__).resolve().parent)


def _caller_stacklevel() -> int:
    """Return the ``stacklevel`` that points at the nearest frame outside the library.

    Step columns are derived from ``fit``, ``observe``, and ``predict``, whose
    call chains reach this module at different depths, and ``fit`` adds another
    frame for scikit-learn's ``_fit_context`` decorator. No constant
    ``stacklevel`` points at the caller for all three, so it is measured instead
    of assumed: walk outward from the warn site and stop at the first frame that
    belongs to neither yohou nor scikit-learn.

    Call from the frame that calls :func:`warnings.warn`.
    """
    frame = inspect.currentframe()
    if frame is not None:
        frame = frame.f_back  # the warn site itself
    level = 1
    while frame is not None:
        filename = frame.f_code.co_filename
        if not filename.startswith(_YOHOU_ROOT) and not filename.startswith(_SKLEARN_ROOT):
            return level
        frame = frame.f_back
        level += 1
    return 1


def _is_forecast_kind(obj: object) -> bool:
    """Return whether ``obj`` reports ``kind == "forecast"``.

    Reads the tag rather than testing the base class. An ``isinstance`` check
    cannot express this: the composition estimators stay ``BaseActualTransformer``
    subclasses and discriminate their kind by tag, so a ``FeatureUnion`` of
    forecast transformers is an instance of the actual base while reporting
    ``kind="forecast"``. Anything without readable transformer tags counts as
    actual, so a real validation error surfaces instead of this one.

    Parameters
    ----------
    obj : object
        A transformer instance (or anything, including ``None``).

    Returns
    -------
    bool
        ``True`` only if the object explicitly reports the forecast kind.

    """
    get_tags = getattr(obj, "__sklearn_tags__", None)
    if get_tags is None:
        return False
    transformer_tags = getattr(get_tags(), "transformer_tags", None)
    return transformer_tags is not None and getattr(transformer_tags, "kind", "actual") == "forecast"


def _require_actual_memory_api(transformer: object, method: str) -> None:
    """Raise if the ``observe``/``rewind`` memory API is used on a forecast-kind transformer.

    The memory API maintains a buffer of the most recent contiguous rows. The
    vintage axis is discontinuous, so no such buffer exists and the call has no
    meaning rather than merely being unsupported.

    Parameters
    ----------
    transformer : object
        The transformer the method was called on.
    method : str
        Method name, used in the error message.

    Raises
    ------
    ValueError
        If ``transformer`` reports ``kind == "forecast"``.

    """
    if _is_forecast_kind(transformer):
        raise ValueError(
            f"{type(transformer).__name__}.{method}() is unavailable on a forecast-kind "
            f"transformer, which must be an actual-kind transformer to maintain memory. "
            f"The observe/rewind buffer holds contiguous recent rows, which the vintage "
            f"axis of an X_forecast frame cannot provide. Forecast transformers are "
            f"stateless, so there is no memory to update or rewind."
        )


def _require_actual_transformer(transformer: object, slot: str) -> None:
    """Raise if ``transformer`` is a forecast-kind transformer in an actual slot.

    A forecaster's ``target_transformer`` / ``feature_transformer`` operate on
    the single-axis target / ``X_actual`` frames, so they must be actual-kind.
    Leaf forecast transformers are already rejected by the parameter constraint
    (they are not ``BaseActualTransformer`` instances); this catches a
    forecast-kind composition (e.g. a ``FeatureUnion`` of forecast transformers),
    which is structurally an actual transformer but reports ``kind="forecast"``.

    Parameters
    ----------
    transformer : object
        The transformer assigned to the slot (or ``None``).
    slot : str
        Slot name, used in the error message.

    Raises
    ------
    ValueError
        If ``transformer`` reports ``kind == "forecast"``.

    """
    if _is_forecast_kind(transformer):
        raise ValueError(
            f"{slot} must be an actual-kind transformer (operating on the single-axis "
            f"target/X_actual frame), but got a forecast-kind transformer. Forecast "
            f"transformers belong on the X_forecast channel, not in {slot}."
        )


def _fit_transform_transformers_one(
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None,
    target_transformer: BaseActualTransformer | None,
    feature_transformer: BaseActualTransformer | None,
    target_as_feature: str | None,
) -> tuple[pl.DataFrame, pl.DataFrame | None, BaseActualTransformer | None, BaseActualTransformer | None]:
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
    target_transformer : BaseActualTransformer or None
        Target transformer to apply.
    feature_transformer : BaseActualTransformer or None
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
    target_transformer : BaseActualTransformer or None
        Fitted target transformer.
    feature_transformer : BaseActualTransformer or None
        Fitted feature transformer.

    Notes
    -----
    Transformation order matters:
    1. Apply target_transformer to y → y_t
    2. Align X_actual to y_t timestamps via a semi-join, then concatenate with y_t
    3. Apply feature_transformer to combined → X_t
    4. Trim y_t if feature transformer has its own observation horizon

    This ensures features can include lagged versions of the transformed target.

    See Also
    --------
    - [`BaseActualTransformer`][yohou.base.transformer.BaseActualTransformer] : Base class for transformers

    """
    _require_actual_transformer(target_transformer, "target_transformer")
    _require_actual_transformer(feature_transformer, "feature_transformer")

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
    feature_transformer: BaseActualTransformer | None,
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
    feature_transformer : BaseActualTransformer or None
        Feature transformer (used for validation when
        ``target_as_feature=None``).

    Returns
    -------
    pl.DataFrame or None
        Feature input for feature_transformer.

    Notes
    -----
    The target_as_feature parameter controls what features are available:
    - ``"transformed"``: Transformed target + exogenous features
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
    target_transformer: BaseActualTransformer | None,
    feature_transformer: BaseActualTransformer | None,
    target_as_feature: str | None,
) -> pl.DataFrame | None:
    """Observe new data through transformers.

    Parameters
    ----------
    y : pl.DataFrame
        New target observations.
    X_actual : pl.DataFrame or None
        New features.
    target_transformer : BaseActualTransformer or None
        Target transformer to observe.
    feature_transformer : BaseActualTransformer or None
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
    target_transformer: BaseActualTransformer | None,
    feature_transformer: BaseActualTransformer | None,
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
    target_transformer : BaseActualTransformer or None
        Target transformer to rewind.
    feature_transformer : BaseActualTransformer or None
        Feature transformer to rewind.
    observation_horizon : int
        Number of time steps to retain in observation horizon.
    target_as_feature : {"transformed", "raw"} or None
        Controls whether the target is included as a feature.

    Returns
    -------
    pl.DataFrame or None
        Feature matrix containing the single transformed row aligned to the
        latest observation timestamp after rewinding state.

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

        # Use rewind_transform (combined operation): this path needs the
        # transformed output X_t_all, so the combined rewind-and-transform is
        # the right call rather than a state-only rewind() followed by a
        # separate transform.
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


def _rank_deficient_step_columns(
    X_step: pl.DataFrame,
    value_columns: list[str],
    forecasting_horizon: int,
) -> list[tuple[str, int, int]]:
    """Find X_future value columns whose step expansion is rank deficient.

    A column whose value at ``T`` already determines its value at ``T+h`` (a
    Fourier term, a calendar cycle) expands into ``H`` step columns that span
    fewer than ``H`` dimensions, because each step column is a fixed function of
    the value at ``T``. A genuine event calendar spans the full ``H``.

    Rank rather than pairwise similarity is the right test: Fourier step columns
    are rotations of the base pair, not copies, so they are neither duplicated
    nor strongly pairwise correlated while still spanning only two dimensions.

    Parameters
    ----------
    X_step : pl.DataFrame
        The derived step frame, as returned by :func:`_derive_step_columns`.
    value_columns : list of str
        Base column names from ``X_future``, excluding ``"time"``.
    forecasting_horizon : int
        Number of forward steps (H) per observation time.

    Returns
    -------
    list of (str, int, int)
        One ``(column, n_step_columns, rank)`` triple per rank-deficient column.
        Empty when nothing is deficient or nothing can be checked.

    Notes
    -----
    The check is deliberately incomplete. It reports only what it can prove from
    the step block alone, and stays silent whenever the measurement would not
    support a conclusion:

    - Rows carrying nulls are dropped first. Null step columns are a supported
      state (an ``X_future`` that does not cover every observation time), and
      ranking them would report an under-covering event calendar as a clock
      feature, which is the opposite of the truth.
    - A column is skipped when fewer than ``2 * H`` usable rows remain, since
      rank is bounded by row count and a short frame would otherwise look
      deficient for a reason unrelated to the feature.
    - Non-numeric columns are skipped; rank is not defined for them.

    """
    findings: list[tuple[str, int, int]] = []
    for col in value_columns:
        step_names = [f"{col}_step_{h}" for h in range(1, forecasting_horizon + 1)]
        if any(name not in X_step.columns for name in step_names):
            continue

        block = X_step.select(step_names)
        if block.select(cs.numeric()).width != len(step_names):
            continue

        block = block.drop_nulls()
        if len(block) < 2 * forecasting_horizon:
            continue

        array = block.to_numpy().astype(float)
        array = array[np.isfinite(array).all(axis=1)]
        if len(array) < 2 * forecasting_horizon:
            continue

        rank = int(np.linalg.matrix_rank(array))
        if rank < forecasting_horizon:
            findings.append((col, len(step_names), rank))

    return findings


def _warn_rank_deficient_step_columns(
    X_step: pl.DataFrame | None,
    X_future: pl.DataFrame | None,
    forecasting_horizon: int,
) -> None:
    """Warn when an X_future column's step expansion carries no extra information.

    Call from the fit path only. ``_derive_step_columns`` also serves observe and
    predict, so a warning raised there would repeat once per stride of a
    walk-forward loop, and the condition being reported is a property of the
    caller's feature routing that cannot change between strides.

    The message reports the measurement rather than asserting a diagnosis: a
    constant column is rank deficient too, and this check cannot tell it apart
    from a clock feature.
    """
    if X_step is None or X_future is None:
        return

    value_columns = [c for c in X_future.columns if c != "time"]
    for col, n_steps, rank in _rank_deficient_step_columns(X_step, value_columns, forecasting_horizon):
        warnings.warn(
            f"X_future column '{col}' expands to {n_steps} step columns spanning "
            f"a rank of only {rank}, so {n_steps - rank} of them add no information "
            f"the others do not already carry. This is what a deterministic clock "
            f"feature looks like (a Fourier term, or a calendar cycle): its value "
            f"at the observation point already determines its value at every "
            f"forecast step, so the forward window buys nothing and the collinear "
            f"copies dilute any regularisation applied to them. If '{col}' is "
            f"computable from the timestamp alone, generate it with a "
            f"feature_transformer instead of passing it through X_future. If it is "
            f"constant, or genuinely needs an external table, this warning does not "
            f"apply and can be silenced.",
            UserWarning,
            stacklevel=_caller_stacklevel(),
        )


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
                    stacklevel=_caller_stacklevel(),
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
