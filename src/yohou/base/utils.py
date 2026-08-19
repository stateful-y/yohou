"""Utility functions for forecaster transformations."""

from __future__ import annotations

import inspect
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import polars.selectors as cs
import sklearn
from sklearn.base import clone

if TYPE_CHECKING:
    from collections.abc import Callable

    from yohou.base import BaseActualTransformer

_YOHOU_ROOT = str(Path(__file__).resolve().parents[1])
_SKLEARN_ROOT = str(Path(sklearn.__file__).resolve().parent)


class ForecastCoverageWarning(UserWarning):
    """Raised when ``X_forecast`` covers fewer steps than the forecasting horizon.

    Carries the per-column breakdown the check computed rather than only the
    worst number, because which channel is starved is what tells a reader what
    to do. A consumer aggregating these across a long run can then report the
    affected columns instead of only a count.

    Subclasses ``UserWarning`` so existing ``pytest.warns(UserWarning)`` and
    application ``filterwarnings`` entries keep matching.

    Attributes
    ----------
    coverage : dict[str, int]
        Base column name mapped to the worst number of covered forecast steps
        observed for it. A value of 0 means every step feature derived from that
        column is null.
    forecasting_horizon : int
        The number of steps full coverage would mean, so a reader can judge the
        counts without holding the call's configuration in mind.
    """

    def __init__(self, message: str, *, coverage: dict[str, int], forecasting_horizon: int) -> None:
        super().__init__(message)
        self.coverage = coverage
        self.forecasting_horizon = forecasting_horizon


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


def _kind_of(obj: object) -> str:
    """Return ``obj``'s transformer ``kind`` tag, defaulting to ``"actual"``.

    Reads the tag rather than testing the base class. An ``isinstance`` check
    cannot express this: the composition estimators stay ``BaseActualTransformer``
    subclasses and discriminate their kind by tag, so a ``FeatureUnion`` of
    forecast transformers is an instance of the actual base while reporting
    ``kind="forecast"``. Anything without readable transformer tags counts as
    actual, so a real validation error surfaces instead of a kind error.

    Parameters
    ----------
    obj : object
        A transformer instance (or anything, including ``None``).

    Returns
    -------
    str
        The reported kind, or ``"actual"`` when no kind tag is readable.

    """
    get_tags = getattr(obj, "__sklearn_tags__", None)
    if get_tags is None:
        return "actual"
    transformer_tags = getattr(get_tags(), "transformer_tags", None)
    if transformer_tags is None:
        return "actual"
    return getattr(transformer_tags, "kind", "actual")


def _is_forecast_kind(obj: object) -> bool:
    """Return whether ``obj`` reports ``kind == "forecast"``.

    Parameters
    ----------
    obj : object
        A transformer instance (or anything, including ``None``).

    Returns
    -------
    bool
        ``True`` only if the object explicitly reports the forecast kind.

    """
    return _kind_of(obj) == "forecast"


def _is_step_kind(obj: object) -> bool:
    """Return whether ``obj`` reports ``kind == "step"``.

    The mirror of :func:`_is_forecast_kind` for the step frame, which is the
    wide ``{base}_step_1..H`` frame derived from ``X_future`` and ``X_forecast``.
    A step frame carries the same single ``"time"`` index a single-axis frame
    does, so the kind tag is the only thing separating the two; an ``isinstance``
    check would let a step transformer into an actual slot, where it would find
    no step columns and silently emit nothing.

    Parameters
    ----------
    obj : object
        A transformer instance (or anything, including ``None``).

    Returns
    -------
    bool
        ``True`` only if the object explicitly reports the step kind.

    """
    return _kind_of(obj) == "step"


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
        If ``transformer`` reports ``kind == "forecast"`` or ``kind == "step"``.

    """
    if _is_forecast_kind(transformer):
        raise ValueError(
            f"{type(transformer).__name__}.{method}() is unavailable on a forecast-kind "
            f"transformer, which must be an actual-kind transformer to maintain memory. "
            f"The observe/rewind buffer holds contiguous recent rows, which the vintage "
            f"axis of an X_forecast frame cannot provide. Forecast transformers are "
            f"stateless, so there is no memory to update or rewind."
        )
    if _is_step_kind(transformer):
        raise ValueError(
            f"{type(transformer).__name__}.{method}() is unavailable on a step-kind "
            f"transformer, which must be an actual-kind transformer to maintain memory. "
            f"The observe/rewind buffer holds contiguous recent rows carried between "
            f"calls, but a step frame is re-derived from scratch at every observe and "
            f"predict, so nothing accumulates across them. Step transformers are "
            f"stateless, so there is no memory to update or rewind."
        )


def _require_actual_transformer(transformer: object, slot: str) -> None:
    """Raise if ``transformer`` is a forecast-kind transformer in an actual slot.

    A forecaster's ``target_transformer`` / ``actual_transformer`` operate on
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
            f"transformers belong in the forecast_transformer slot, which applies them "
            f"to the X_forecast frame, not in {slot}."
        )
    if _is_step_kind(transformer):
        raise ValueError(
            f"{slot} must be an actual-kind transformer (operating on the single-axis "
            f"target/X_actual frame), but got a step-kind transformer. Step transformers "
            f"belong in the step_transformer slot, which applies them to the derived "
            f"{{base}}_step_1..H frame, not in {slot}. A step transformer placed here "
            f"would find no step columns and silently emit nothing."
        )


def _require_forecast_transformer(transformer: object, slot: str) -> None:
    """Raise if ``transformer`` is an actual-kind transformer in the forecast slot.

    The mirror of :func:`_require_actual_transformer`. A forecaster's
    ``forecast_transformer`` operates on the vintage-indexed ``X_forecast`` frame,
    so it must be forecast-kind.

    The parameter constraint cannot carry this on its own. A forecast-kind
    composition (e.g. a ``FeatureUnion`` of forecast transformers) is a
    ``BaseActualTransformer`` subclass reporting ``kind="forecast"``, so the
    constraint has to admit ``BaseActualTransformer`` to let compositions through,
    which also lets a genuine actual transformer past. The kind tag is what
    separates them, exactly as on the actual side.

    Parameters
    ----------
    transformer : object
        The transformer assigned to the slot (or ``None``).
    slot : str
        Slot name, used in the error message.

    Raises
    ------
    ValueError
        If ``transformer`` is not ``None`` and does not report ``kind == "forecast"``.

    """
    if transformer is not None and not _is_forecast_kind(transformer):
        raise ValueError(
            f"{slot} must be a forecast-kind transformer (operating on the "
            f"vintage-indexed X_forecast frame), but got a {_kind_of(transformer)}-kind "
            f"transformer. Lift it onto the vintage axis with PerVintageActualTransformer, "
            f"or pass it to actual_transformer to apply it to the single-axis X_actual frame."
        )


def _require_step_transformer(transformer: object, slot: str) -> None:
    """Raise if ``transformer`` is not a step-kind transformer in the step slot.

    The mirror of :func:`_require_forecast_transformer`. A forecaster's
    ``step_transformer`` operates on the wide ``{base}_step_1..H`` frame derived
    from ``X_future`` and ``X_forecast``, so it must be step-kind.

    As on the forecast side, the parameter constraint cannot carry this alone. A
    step-kind composition (a ``FeatureUnion`` of step transformers) is a
    ``BaseActualTransformer`` subclass reporting ``kind="step"``, so the
    constraint has to admit ``BaseActualTransformer`` to let compositions
    through, which also lets a genuine actual transformer past. The kind tag is
    what separates them.

    Parameters
    ----------
    transformer : object
        The transformer assigned to the slot (or ``None``).
    slot : str
        Slot name, used in the error message.

    Raises
    ------
    ValueError
        If ``transformer`` is not ``None`` and does not report ``kind == "step"``.

    """
    if transformer is not None and not _is_step_kind(transformer):
        raise ValueError(
            f"{slot} must be a step-kind transformer (operating on the derived "
            f"{{base}}_step_1..H frame), but got a {_kind_of(transformer)}-kind "
            f"transformer. Pass it to actual_transformer to apply it to the single-axis "
            f"X_actual frame, or to forecast_transformer to apply it per vintage to the "
            f"X_forecast frame."
        )


def _fit_transform_transformers_one(
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None,
    target_transformer: BaseActualTransformer | None,
    actual_transformer: BaseActualTransformer | None,
    target_as_feature: str | None,
) -> tuple[pl.DataFrame, pl.DataFrame | None, BaseActualTransformer | None, BaseActualTransformer | None]:
    """Fit and apply target and actual transformers to a single time series.

    Orchestrates the transformation pipeline: target transformer first (if any),
    then actual transformer (if any). Handles observation horizon alignment to
    ensure transformed data matches temporally.

    Parameters
    ----------
    y : pl.DataFrame
        Target time series with "time" column.
    X_actual : pl.DataFrame or None
        Feature time series with "time" column.
    target_transformer : BaseActualTransformer or None
        Target transformer to apply.
    actual_transformer : BaseActualTransformer or None
        Actual transformer to apply.
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
    actual_transformer : BaseActualTransformer or None
        Fitted actual transformer.

    Notes
    -----
    Transformation order matters:
    1. Apply target_transformer to y → y_t
    2. Align X_actual to y_t timestamps via a semi-join, then concatenate with y_t
    3. Apply actual_transformer to combined → X_t
    4. Trim y_t if actual transformer has its own observation horizon

    This ensures features can include lagged versions of the transformed target.

    See Also
    --------
    - [`BaseActualTransformer`][yohou.base.transformer.BaseActualTransformer] : Base class for transformers

    """
    _require_actual_transformer(target_transformer, "target_transformer")
    _require_actual_transformer(actual_transformer, "actual_transformer")

    y_t = y
    target_transformer_fitted = None
    if target_transformer is not None:
        target_transformer_fitted = clone(target_transformer)
        y_t = target_transformer_fitted.fit_transform(y)

    X_feat_in = _build_feature_input(y, y_t, X_actual, target_as_feature, actual_transformer)

    X_t = X_feat_in
    actual_transformer_fitted = None
    if actual_transformer is not None and X_feat_in is not None:
        actual_transformer_fitted = clone(actual_transformer)
        X_t = actual_transformer_fitted.fit_transform(X_feat_in)
        feature_observation_horizon = actual_transformer_fitted.observation_horizon
        # Trim y_t to align with X_t
        # First, align by actual transformer's observation horizon (handles transformers that don't drop rows)
        y_t = y_t[feature_observation_horizon:]
        # Also trim X_t: drop null rows produced by transformers that keep all rows
        # but fill initial positions with null (e.g., LagTransformer, RollingStatisticsTransformer)
        X_t = X_t.drop_nulls(subset=~cs.by_name("time"))
        # Then, align by timestamps (handles transformers that DO drop rows)
        y_t = y_t.join(X_t.select("time"), on="time", how="semi")

    return y_t, X_t, target_transformer_fitted, actual_transformer_fitted


def _build_feature_input(
    y: pl.DataFrame,
    y_t: pl.DataFrame,
    X_actual: pl.DataFrame | None,
    target_as_feature: str | None,
    actual_transformer: BaseActualTransformer | None,
) -> pl.DataFrame | None:
    """Build feature input based on target_as_feature parameter.

    Constructs the input to the actual_transformer by combining original y,
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
    actual_transformer : BaseActualTransformer or None
        Actual transformer (used for validation when
        ``target_as_feature=None``).

    Returns
    -------
    pl.DataFrame or None
        Feature input for actual_transformer.

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
            if actual_transformer is not None:
                # This should not happen since _validate_pre_fit checks at fit
                # time, but guard against direct calls.
                raise ValueError(
                    "target_as_feature=None requires X_actual to be provided when an actual_transformer is set, but X_actual is None."
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
    actual_transformer: BaseActualTransformer | None,
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
    actual_transformer : BaseActualTransformer or None
        Actual transformer to observe.
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

    X_feat_in = _build_feature_input(y, y_t, X_actual, target_as_feature, actual_transformer)

    X_t = X_feat_in
    if actual_transformer is not None and X_feat_in is not None:
        X_t = actual_transformer.observe_transform(X_feat_in)

    return X_t


def _rewind_transformers_one(
    y: pl.DataFrame,
    X_actual: pl.DataFrame | None,
    target_transformer: BaseActualTransformer | None,
    actual_transformer: BaseActualTransformer | None,
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
    actual_transformer : BaseActualTransformer or None
        Actual transformer to rewind.
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

    X_feat_in = _build_feature_input(y, y_t, X_actual, target_as_feature, actual_transformer)

    X_t = X_feat_in
    if actual_transformer is not None and X_feat_in is not None:
        feature_observation_horizon = actual_transformer.observation_horizon

        # X_feat_in is aligned to y_t timestamps (observation_horizon rows)
        # but the actual transformer may need more rows for its own rewind.
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
            X_feat_extra = _build_feature_input(y_extra, y_t_extra, X_extra, target_as_feature, actual_transformer)
            if X_feat_extra is not None:
                # Only keep the tail; rewind_transform may drop observation_horizon rows
                X_feat_extra = X_feat_extra.tail(deficit)
                X_feat_in = pl.concat([X_feat_extra, X_feat_in], how="vertical")

        # Use rewind_transform (combined operation): this path needs the
        # transformed output X_t_all, so the combined rewind-and-transform is
        # the right call rather than a state-only rewind() followed by a
        # separate transform.
        X_t_all = actual_transformer.rewind_transform(X_feat_in)
        # Keep the row aligned to the most-recent observation timestamp rather
        # than blindly taking the last row: when the actual transformer drops
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

    ``X_step`` is the frame that reaches the estimator, so a ``step_transformer``
    that reduced a base column's block away suppresses the warning for that
    column without a separate rule: :func:`_rank_deficient_step_columns` skips any
    base column whose ``{col}_step_1..H`` names are absent. A transform that keeps
    the block one-for-one (a wrapped scaler) leaves the names in place and does
    not remove collinearity, so the warning correctly still fires. Suppression is
    therefore decided by column presence, never by the transformer's identity.
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
            f"actual_transformer instead of passing it through X_future; if you "
            f"want to keep it in X_future, reduce the block along the horizon with "
            f"a step_transformer, which collapses the collinear copies. If it is "
            f"constant, or genuinely needs an external table, this warning does not "
            f"apply and can be silenced.",
            UserWarning,
            stacklevel=_caller_stacklevel(),
        )


def _forecast_step_coverage(
    frame: pl.DataFrame,
    value_cols: list[str],
    forecasting_horizon: int,
) -> dict[str, list[int]]:
    """Count covered forecast steps per observation row, per base column.

    For each base column in ``value_cols`` and each row of ``frame``, the
    covered-step count is the number of its ``_step_1..H`` columns carrying a
    value. Coverage from a single as-of vintage is a contiguous prefix, so the
    count equals the depth reached for that observation.

    Parameters
    ----------
    frame : pl.DataFrame
        A frame carrying ``{base}_step_{h}`` columns (the forecast-pivoted frame
        or the combined step frame). One row per observation.
    value_cols : list of str
        Base column names to measure. Names not present as step columns in
        ``frame`` are skipped.
    forecasting_horizon : int
        Number of forward steps (H) per observation.

    Returns
    -------
    dict[str, list[int]]
        Base column name to a per-row list of covered-step counts. Base columns
        with no step columns in ``frame`` are omitted.

    """
    coverage: dict[str, list[int]] = {}
    for base in value_cols:
        step_names = [f"{base}_step_{h}" for h in range(1, forecasting_horizon + 1)]
        present = [c for c in step_names if c in frame.columns]
        if not present:
            continue
        counts = frame.select(pl.sum_horizontal(pl.col(c).is_not_null().cast(pl.Int32) for c in present).alias("_n"))[
            "_n"
        ].to_list()
        coverage[base] = counts
    return coverage


def _warn_forecast_coverage_at_fit(
    X_step: pl.DataFrame | None,
    X_forecast: pl.DataFrame | None,
    forecasting_horizon: int,
) -> None:
    """Warn per X_forecast base column that under-covers the training batch.

    Call from the fit path only, alongside
    :func:`_warn_rank_deficient_step_columns`. Fit-time coverage is a property
    of how the training frame was assembled (a forecast archive that starts
    later than the target series, or a channel issued on a slower cadence than
    others in the frame), so it is stated once per column rather than once per
    stride of a walk-forward loop. The per-call zero/partial warning inside
    :func:`_derive_step_columns` carries the observe and predict paths instead.

    Measurement, not diagnosis: the message reports how many of the training
    observations a column fails to fully cover and the worst depth it reaches.
    It recommends no estimator and asserts no normality, mirroring the rank
    diagnostic, because the same measurement has more than one cause.

    Parameters
    ----------
    X_step : pl.DataFrame or None
        The combined step frame returned by :func:`_derive_step_columns`.
    X_forecast : pl.DataFrame or None
        The raw ``X_forecast`` frame, used only for its base column names.
    forecasting_horizon : int
        Number of forward steps (H) per observation.

    """
    if X_step is None or X_forecast is None:
        return

    value_cols = [c for c in X_forecast.columns if c not in ("vintage_time", "time")]
    coverage = _forecast_step_coverage(X_step, value_cols, forecasting_horizon)
    n_obs = len(X_step)
    for base, counts in coverage.items():
        worst = min(counts, default=forecasting_horizon)
        if worst >= forecasting_horizon:
            continue
        n_under = sum(1 for c in counts if c < forecasting_horizon)
        warnings.warn(
            f"X_forecast column '{base}' covers {worst} of {forecasting_horizon} forecast steps "
            f"at its worst across {n_under} of {n_obs} training observations, and those "
            f"observations get null step features for this channel. A model fitted on them "
            f"is trained without '{base}' where it is null. This is what a forecast archive "
            f"that starts later than the target series looks like, and what a channel issued "
            f"on a slower cadence than others in the same X_forecast frame looks like.",
            UserWarning,
            stacklevel=_caller_stacklevel(),
        )


def _densify_forecast_vintages(X_forecast: pl.DataFrame) -> pl.DataFrame:
    """Fill each vintage row so every base column carries its own newest value.

    For a given vintage ``V`` and base column ``c``, the values come from the
    newest vintage at or before ``V`` that carries ``c`` (has a non-null value).
    All of ``V``'s rows for ``c`` therefore originate in a single source vintage,
    so a column's forecast trajectory is never spliced across vintages. Where
    that source vintage does not reach a target time, the value stays null.

    The wide ``X_forecast`` shape implicitly assumes every column is issued at
    every vintage. When sources publish on different schedules that assumption
    fails, and the frame-wide as-of in windowing resolves the slower channel to
    null. Densifying first lets each channel resolve against its own newest
    vintage while a single as-of still serves the whole row, so windowing and
    any row-wise ``forecast_transformer`` both see a dense frame.

    Densification is a no-op on a uniform-cadence frame (every vintage already
    carries every column, so each column's source vintage is the vintage itself)
    and is idempotent (a dense frame densifies to itself). Rows are filled in
    place: the frame's own ``(vintage_time, time)`` index is preserved, so a
    sparse-but-uniform vintage schedule (one coarse cadence, finer observations)
    is unaffected. Because windowing picks the frame-wide newest vintage at or
    before an observation, no carrying vintage can sit between it and the
    observation, so filling per vintage reproduces per-column as-of exactly.

    Parameters
    ----------
    X_forecast : pl.DataFrame
        A raw ``X_forecast`` frame with ``vintage_time`` and ``time`` columns.

    Returns
    -------
    pl.DataFrame
        A frame with the same ``(vintage_time, time)`` rows, every base column
        filled from its own newest applicable vintage.

    """
    value_cols = [c for c in X_forecast.columns if c not in ("vintage_time", "time")]
    if not value_cols or X_forecast.is_empty():
        return X_forecast

    vintages = X_forecast.select("vintage_time").unique().sort("vintage_time")
    result = X_forecast.select("vintage_time", "time")
    for col in value_cols:
        dtype = X_forecast.schema[col]
        carrying = (
            X_forecast
            .filter(pl.col(col).is_not_null())
            .select(pl.col("vintage_time").alias("_src_vintage"))
            .unique()
            .sort("_src_vintage")
        )
        if carrying.is_empty():
            result = result.with_columns(pl.lit(None).cast(dtype).alias(col))
            continue
        # For each vintage, the newest source vintage carrying this column.
        vmap = vintages.join_asof(
            carrying, left_on="vintage_time", right_on="_src_vintage", strategy="backward"
        ).select("vintage_time", "_src_vintage")
        src = X_forecast.filter(pl.col(col).is_not_null()).select(
            pl.col("vintage_time").alias("_src_vintage"), "time", col
        )
        result = (
            result
            .join(vmap, on="vintage_time", how="left")
            .join(src, on=["_src_vintage", "time"], how="left")
            .drop("_src_vintage")
        )
    return result.select("vintage_time", "time", *value_cols)


def _retained_forecast_vintages(X_forecast: pl.DataFrame, observed_time: datetime) -> list:
    """Return the ``vintage_time`` values to retain after an observe or rewind.

    For each base column, the newest vintage at or before ``observed_time`` that
    both carries the column (has a non-null value) and still reaches a target
    beyond ``observed_time``. The union of those per-column vintages is retained.

    Keying per column, rather than keeping the single newest vintage in the
    frame, is what lets channels issued on different schedules each survive: a
    vintage carrying only the fast channel does not evict the slow channel's
    older but still-current vintage. The retained set is bounded by the number
    of base columns, holding at most one vintage per column.

    A vintage is dropped once it is a full horizon old, detected without any
    interval arithmetic: its rows carry its own target times, so a vintage whose
    latest target is at or before ``observed_time`` covers no future step and is
    evicted. This also drops a channel whose only surviving vintage has gone
    stale, so its step features become null and the coverage diagnostic fires.

    Parameters
    ----------
    X_forecast : pl.DataFrame
        A raw ``X_forecast`` frame with ``vintage_time`` and ``time`` columns.
        Channel membership is read from this frame's columns.
    observed_time : object
        The observation point (a scalar datetime). Only vintages at or before
        it are eligible.

    Returns
    -------
    list
        The ``vintage_time`` values to retain, possibly empty. Filter both the
        raw and the transformed cache by this set to keep them in lockstep.

    """
    value_cols = [c for c in X_forecast.columns if c not in ("vintage_time", "time")]
    eligible = X_forecast.filter(pl.col("vintage_time") <= observed_time)
    if eligible.is_empty():
        return []

    keep: list = []
    seen: set = set()
    for col in value_cols:
        live = (
            eligible
            .filter(pl.col(col).is_not_null())
            .group_by("vintage_time")
            .agg(pl.col("time").max().alias("_max_target"))
            .filter(pl.col("_max_target") > observed_time)
        )
        if live.is_empty():
            continue
        newest = live["vintage_time"].max()
        if newest not in seen:
            seen.add(newest)
            keep.append(newest)
    return keep


def _derive_step_columns(
    X_future: pl.DataFrame | None,
    X_forecast: pl.DataFrame | None,
    observation_times: pl.Series,
    forecasting_horizon: int,
    interval: str | timedelta,
    *,
    warn_coverage: bool = True,
    warn_coverage_at_fit: bool = False,
    existing_columns: set[str] | None = None,
    step_transform: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
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
        For each observation time ``T`` the newest vintage at or before ``T``
        is selected (as-of), and step columns are taken at ``T + 1..H`` steps,
        anchored to the observation time rather than the vintage time. Vintages
        are not clipped: a value at a target time beyond one observation's
        horizon simply serves an earlier observation instead. Where the
        resolved vintage carries no value at a step's target time, that step
        column is null and a ``UserWarning`` is emitted.
    observation_times : pl.Series
        Observation timestamps to derive step columns from.
    forecasting_horizon : int
        Number of forward steps (H) per observation time.
    interval : str or timedelta
        Time frequency between consecutive steps.
    warn_coverage : bool, default=True
        Whether to emit the per-call zero/partial coverage warning for
        ``X_forecast``. The fit path passes ``False`` and reports coverage
        per column instead via :func:`_warn_forecast_coverage_at_fit`, so the
        per-call warning carries the observe and predict paths only.
    warn_coverage_at_fit : bool, default=False
        Whether to emit the per-column fit-time coverage diagnostic. Set by the
        fit path only. Measured here, before ``step_transform`` runs, because a
        step transformer renames its output and the measurement needs the
        ``{base}_step_h`` names; delegating it to derivation is what keeps it on
        the pre-transform frame.
    existing_columns : set of str or None, default=None
        Column names already present (e.g., X_actual columns). Used for
        collision detection against generated step column names.
    step_transform : callable or None, default=None
        Applied to the combined step frame before it is returned, and before the
        collision check, so callers see only post-transform names. Taken as a
        parameter rather than applied by callers because there are six derivation
        sites, four of them transform-only; applying it outside would leave four
        places to miss, and a missed one yields a design matrix whose columns
        disagree with what the estimator was fitted on. Kept as a plain callable
        so this function stays free of any forecaster dependency: a forecaster
        passes its own ``_transform_X_step``, which resolves the panel dict.

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

        # Per-call coverage warning for the observe and predict paths. Coverage
        # is measured per observation and per base column: for each row, how many
        # forecast steps of each base column carry a value. The worst (minimum)
        # over rows and columns drives the warning, so a column dead for even one
        # observation is not hidden by another that happens to cover it. This
        # replaces a batch-wide existential test that answered "covered for any
        # observation?" and so missed a channel null in all but one row. The fit
        # path passes warn_coverage=False and reports per column instead.
        if warn_coverage:
            coverage = _forecast_step_coverage(forecast_pivoted, list(value_cols_info), forecasting_horizon)
            # Worst per column, kept rather than collapsed. The single ``worst``
            # below still decides which branch fires, but the breakdown rides on
            # the warning so a reader learns which channel is starved and not
            # merely that one is.
            per_column = {base: min(counts) for base, counts in coverage.items()}
            worst = min(per_column.values(), default=forecasting_horizon)
            starved = sorted(base for base, count in per_column.items() if count < forecasting_horizon)
            if coverage and worst == 0:
                # Distinct from partial coverage rather than its extreme: nothing
                # derived from X_forecast carries a value, so a model relying on
                # that channel is predicting without it. Reachable from ordinary
                # use, because a vintage covers only the forecasting_horizon
                # timestamps after its own vintage_time, so as-of selection yields
                # zero coverage once the observation point is a full horizon past
                # the newest available vintage. Stated as a measurement rather than
                # an error: a caller may reach this deliberately.
                warnings.warn(
                    ForecastCoverageWarning(
                        f"X_forecast covers 0 of {forecasting_horizon} forecast steps for "
                        f"{', '.join(starved)}, so every step feature derived from those columns is "
                        f"null and a model relying on them is predicting without them. This happens "
                        f"when the newest usable vintage is at least {forecasting_horizon} intervals "
                        f"older than the observation point, which a cached frame reaches once serving "
                        f"advances past the vintages it holds.",
                        coverage=per_column,
                        forecasting_horizon=forecasting_horizon,
                    ),
                    stacklevel=_caller_stacklevel(),
                )
            elif coverage and worst < forecasting_horizon:
                warnings.warn(
                    ForecastCoverageWarning(
                        f"X_forecast covers {worst} of {forecasting_horizon} forecast steps, worst "
                        f"over {', '.join(starved)}. The remaining step features will be null. This "
                        f"arises for short-range forecasts and when the observation point has "
                        f"advanced past some forecast timestamps.",
                        coverage=per_column,
                        forecasting_horizon=forecasting_horizon,
                    ),
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

    # Combine parts horizontally
    if len(parts) == 1:
        result = parts[0]
    else:
        # Both X_future and X_forecast present: concat horizontally
        result = pl.concat(
            [parts[0], parts[1].select(~cs.by_name("time"))],
            how="horizontal",
        )

    # Coverage is measured here, on the pre-transform frame, because it reports
    # the nulls a step transformer must have a policy about. Measuring it after
    # would hide them behind whatever that policy did, and a transformer renames
    # its output anyway, so the {base}_step_h columns the measurement needs would
    # be gone. This is why the fit path delegates the per-column diagnostic to
    # derivation rather than calling it on the returned frame.
    if warn_coverage_at_fit:
        _warn_forecast_coverage_at_fit(result, X_forecast, forecasting_horizon)

    # Apply the step transform before the collision check. A step transformer
    # emits names derivation never generated (temp_step_mean), and those are what
    # actually join the design matrix, so checking the pre-transform names would
    # let a post-transform collision through unnoticed.
    if step_transform is not None:
        result = step_transform(result)

    # Check collisions against existing columns (e.g., X_actual)
    if existing_columns is not None:
        derived = {c for c in result.columns if c != "time"}
        collisions = derived & existing_columns
        if collisions:
            details = ", ".join(f"'{c}' (from {source_names.get(c, 'step_transformer')})" for c in sorted(collisions))
            msg = (
                f"Step column names collide with existing columns: {details}. "
                f"Rename the source columns to avoid conflicts."
            )
            raise ValueError(msg)

    return result
