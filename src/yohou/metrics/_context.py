"""Scoring context for carrying metadata through the scorer pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class ScoringContext:
    """Immutable metadata extracted during data validation for use in aggregation.

    Carries time values, observed-time (vintage), and forecasting-step
    information through the scoring pipeline without polluting the
    value DataFrames used for raw score computation.

    Parameters
    ----------
    time_values : list
        Aligned time values (one per scored row). Used by componentwise
        aggregation to label output rows.
    observed_time : pl.Series or None
        Observed-time (vintage origin) for each scored row. ``None`` when
        y_pred has no ``"observed_time"`` column (e.g. single-vintage
        scoring or conformity inverse path).
    forecasting_step : pl.Series or None
        Integer ordinal forecasting step for each scored row, derived as
        ``(time - observed_time) / interval``. ``None`` when
        ``observed_time`` is ``None`` or the scorer has no ``interval_``
        attribute.

    """

    time_values: list
    observed_time: pl.Series | None = None
    forecasting_step: pl.Series | None = None
