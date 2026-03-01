"""Hypothesis strategies for generating valid yohou time-series data.

Provides reusable composite strategies that produce polars DataFrames
conforming to yohou's conventions (``"time"`` column, numeric features,
panel-data prefixes with ``__`` separator).

Usage
-----
>>> from strategies import st_time_series, st_panel_time_series
>>> from hypothesis import given
>>>
>>> @given(df=st_time_series())
... def test_transform_preserves_time(df):
...     assert "time" in df.columns
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
from hypothesis import strategies as st


def st_forecasting_horizon(min_value: int = 1, max_value: int = 30) -> st.SearchStrategy[int]:
    """Strategy for valid forecasting horizons."""
    return st.integers(min_value=min_value, max_value=max_value)


def st_seed() -> st.SearchStrategy[int]:
    """Strategy for NumPy random seeds."""
    return st.integers(min_value=0, max_value=2**31 - 1)


@st.composite
def st_time_series(
    draw: st.DrawFn,
    *,
    min_length: int = 3,
    max_length: int = 200,
    min_columns: int = 1,
    max_columns: int = 5,
    allow_nulls: bool = False,
) -> pl.DataFrame:
    """Generate a valid yohou time-series DataFrame.

    The returned DataFrame always has a ``"time"`` column of type
    ``pl.Datetime`` with 1-second resolution, plus *n* ``Float64``
    feature columns named ``col_0``, ``col_1``, etc.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function (injected by ``@st.composite``).
    min_length : int
        Minimum number of rows.
    max_length : int
        Maximum number of rows.
    min_columns : int
        Minimum number of numeric feature columns.
    max_columns : int
        Maximum number of numeric feature columns.
    allow_nulls : bool
        Whether to randomly insert null values.

    Returns
    -------
    pl.DataFrame

    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    n_cols = draw(st.integers(min_value=min_columns, max_value=max_columns))
    seed = draw(st_seed())

    rng = np.random.default_rng(seed)

    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
        interval="1s",
        eager=True,
    )

    data: dict[str, object] = {"time": time}
    for i in range(n_cols):
        values = rng.standard_normal(length)
        if allow_nulls:
            # Randomly null-out ~10 % of entries
            mask = rng.random(length) < 0.1
            values = values.tolist()
            for idx in np.where(mask)[0]:
                values[idx] = None  # type: ignore[index]
        data[f"col_{i}"] = values

    return pl.DataFrame(data)


@st.composite
def st_panel_time_series(
    draw: st.DrawFn,
    *,
    min_length: int = 3,
    max_length: int = 100,
    min_groups: int = 2,
    max_groups: int = 4,
    min_columns_per_group: int = 1,
    max_columns_per_group: int = 3,
) -> pl.DataFrame:
    """Generate panel time-series data with ``__``-separated column names.

    Column names follow the yohou panel convention:
    ``group_<i>__col_<j>``.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function.
    min_length : int
        Minimum number of rows.
    max_length : int
        Maximum number of rows.
    min_groups : int
        Minimum number of panel groups.
    max_groups : int
        Maximum number of panel groups.
    min_columns_per_group : int
        Minimum feature columns per group.
    max_columns_per_group : int
        Maximum feature columns per group.

    Returns
    -------
    pl.DataFrame

    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    n_groups = draw(st.integers(min_value=min_groups, max_value=max_groups))
    n_cols = draw(st.integers(min_value=min_columns_per_group, max_value=max_columns_per_group))
    seed = draw(st_seed())

    rng = np.random.default_rng(seed)

    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
        interval="1s",
        eager=True,
    )

    data: dict[str, object] = {"time": time}
    for g in range(n_groups):
        for c in range(n_cols):
            data[f"group_{g}__col_{c}"] = rng.standard_normal(length)

    return pl.DataFrame(data)


@st.composite
def st_y_X(
    draw: st.DrawFn,
    *,
    min_length: int = 10,
    max_length: int = 200,
    min_targets: int = 1,
    max_targets: int = 3,
    min_features: int = 0,
    max_features: int = 3,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Generate a ``(y, X)`` pair suitable for forecaster ``fit``.

    Parameters
    ----------
    draw : st.DrawFn
        Hypothesis draw function.
    min_length : int
        Minimum number of rows.
    max_length : int
        Maximum number of rows.
    min_targets : int
        Minimum number of target columns in ``y``.
    max_targets : int
        Maximum number of target columns in ``y``.
    min_features : int
        Minimum number of feature columns in ``X`` (0 means ``X=None``).
    max_features : int
        Maximum number of feature columns in ``X``.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame | None]

    """
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    n_targets = draw(st.integers(min_value=min_targets, max_value=max_targets))
    n_features = draw(st.integers(min_value=min_features, max_value=max_features))
    seed = draw(st_seed())

    rng = np.random.default_rng(seed)

    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
        interval="1s",
        eager=True,
    )

    y_data: dict[str, object] = {"time": time}
    for i in range(n_targets):
        y_data[f"y_{i}"] = rng.standard_normal(length)
    y = pl.DataFrame(y_data)

    X: pl.DataFrame | None = None
    if n_features > 0:
        x_data: dict[str, object] = {"time": time}
        for i in range(n_features):
            x_data[f"X_{i}"] = rng.standard_normal(length)
        X = pl.DataFrame(x_data)

    return y, X
