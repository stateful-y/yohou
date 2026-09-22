"""Shared helpers for preprocessing tests."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl


def hourly_frame(length: int, columns: tuple[str, ...] = ("price",), seed: int = 0) -> pl.DataFrame:
    """Hourly frame from 2021-01-01 with standard normal values in each column."""
    rng = np.random.default_rng(seed)
    times = pl.datetime_range(
        datetime(2021, 1, 1), datetime(2021, 1, 1) + timedelta(hours=length - 1), interval="1h", eager=True
    )
    return pl.DataFrame({"time": times, **{c: rng.normal(size=length) for c in columns}})
