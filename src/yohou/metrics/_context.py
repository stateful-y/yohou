"""Scoring context for carrying metadata through the scorer pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class ScoringContext:
    """Immutable metadata extracted during data validation for use in aggregation.

    Carries time values, vintage time, and forecasting-step
    information through the scoring pipeline without polluting the
    value DataFrames used for raw score computation.

    Parameters
    ----------
    time_values : list
        Aligned time values (one per scored row). Used by componentwise
        aggregation to label output rows.
    vintage_time : pl.Series or None
        Vintage origin time for each scored row. ``None`` when
        y_pred has no ``"vintage_time"`` column (e.g. single-vintage
        scoring or conformity inverse path).
    forecasting_step : pl.Series or None
        Integer ordinal forecasting step for each scored row, derived as
        ``(time - vintage_time) / interval``. ``None`` when
        ``vintage_time`` is ``None`` or the scorer has no ``interval_``
        attribute.
    vintage_weight : np.ndarray or None
        Per-unique-vintage weights for cross-vintage aggregation.
        Length matches the number of unique vintages. Unlike
        ``vintage_time``, which has one entry per scored row,
        ``vintage_weight`` has one entry per unique vintage value and is
        indexed in the same order as ``vintage_time.unique(maintain_order=True)``.
        ``None`` when no vintage_weight was provided by the caller.

    """

    time_values: list
    vintage_time: pl.Series | None = None
    forecasting_step: pl.Series | None = None
    vintage_weight: np.ndarray | None = None
