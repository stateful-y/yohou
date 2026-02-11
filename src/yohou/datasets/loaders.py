"""Dataset loader functions for bundled time series datasets."""

import importlib.resources

import polars as pl

from yohou.datasets import data

__all__ = [
    "load_air_passengers",
    "load_australian_tourism",
    "load_ett_m1",
    "load_store_sales",
    "load_sunspots",
    "load_vic_electricity",
    "load_walmart_sales",
]


def _load_dataset(name: str) -> pl.DataFrame:
    """Helper to load a parquet dataset from the package data directory."""
    # Use resource access that works for zipped packages/wheels
    with importlib.resources.as_file(importlib.resources.files(data).joinpath(f"{name}.parquet")) as p:
        df = pl.read_parquet(p)

    # Ensure "time" is always Datetime (some datasets store Date on disk)
    if df.schema.get("time") == pl.Date:
        df = df.with_columns(pl.col("time").cast(pl.Datetime))

    return df


def load_air_passengers() -> pl.DataFrame:
    """Load the Air Passengers dataset.

    Univariate monthly airline passengers series with strong trend and
    multiplicative seasonality. Contains 144 observations spanning
    12 years.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``"time"`` (Datetime) and
        ``"Passengers"`` (Int64).

    Notes
    -----
    Frequency: monthly. Source: `Airline Passengers CSV
    <https://github.com/jbrownlee/Datasets/blob/master/airline-passengers.csv>`_.

    Examples
    --------
    >>> from yohou.datasets import load_air_passengers
    >>> df = load_air_passengers()
    >>> df.columns
    ['time', 'Passengers']
    >>> len(df)
    144

    """
    return _load_dataset("air_passengers")


def load_sunspots() -> pl.DataFrame:
    """Load the Monthly Sunspots dataset.

    Long univariate series of international monthly sunspot counts with
    approximately 11-year cyclic patterns (solar cycles). Contains
    2820 observations spanning 1749 to 1983.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``"time"`` (Datetime) and
        ``"Sunspots"`` (Float64).

    Notes
    -----
    Frequency: monthly. Source: `Monthly Sunspots CSV
    <https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv>`_.

    Examples
    --------
    >>> from yohou.datasets import load_sunspots
    >>> df = load_sunspots()
    >>> df.columns
    ['time', 'Sunspots']
    >>> len(df)
    2820

    """
    return _load_dataset("sunspots")


def load_australian_tourism() -> pl.DataFrame:
    """Load the Australian Tourism dataset.

    Panel quarterly tourism trips aggregated by Australian state.
    Contains 8 states and 80 quarterly observations spanning 1998
    to 2017, pre-aggregated from regional data.

    Returns
    -------
    pl.DataFrame
        Panel DataFrame with ``"time"`` column and 8 panel columns using
        the ``__`` separator convention: ``"act__trips"``,
        ``"new_south_wales__trips"``, ``"northern_territory__trips"``,
        ``"queensland__trips"``, ``"south_australia__trips"``,
        ``"tasmania__trips"``, ``"victoria__trips"``,
        ``"western_australia__trips"``.

    Notes
    -----
    Frequency: quarterly. Source: `Australia Tourism CSV
    <https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/australia_tourism.csv>`_.

    Examples
    --------
    >>> from yohou.datasets import load_australian_tourism
    >>> df = load_australian_tourism()
    >>> len([c for c in df.columns if c != "time"])
    8

    """
    return _load_dataset("australian_tourism")


def load_vic_electricity() -> pl.DataFrame:
    """Load the Victoria Electricity dataset.

    Multivariate household/grid electricity consumption with demand,
    temperature, and holiday indicator at 30-minute intervals.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``"time"``, ``"Demand"``,
        ``"Temperature"``, ``"Holiday"``.

    Notes
    -----
    Frequency: 30-minute intervals. Source: `Victoria Electricity CSV
    <https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/vic_electricity.csv>`_.

    Examples
    --------
    >>> from yohou.datasets import load_vic_electricity
    >>> df = load_vic_electricity()
    >>> sorted(df.columns)
    ['Demand', 'Holiday', 'Temperature', 'time']

    """
    return _load_dataset("vic_electricity")


def load_store_sales() -> pl.DataFrame:
    """Load the Store Sales dataset.

    Panel daily retail sales for 3 stores and 3 items. Contains 9
    panel groups and 1826 daily observations spanning 2013 to 2017.

    Returns
    -------
    pl.DataFrame
        Panel DataFrame with ``"time"`` column and 9 panel columns using
        the ``__`` separator convention (3 stores x 3 items):
        ``"store_1_item_1__sales"``, ``"store_1_item_2__sales"``, ...,
        ``"store_3_item_3__sales"``.

    Notes
    -----
    Frequency: daily. Source: `Store Sales CSV
    <https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/store_sales.csv>`_.

    Examples
    --------
    >>> from yohou.datasets import load_store_sales
    >>> df = load_store_sales()
    >>> len([c for c in df.columns if c != "time"])
    9

    """
    return _load_dataset("store_sales")


def load_walmart_sales() -> pl.DataFrame:
    """Load the Walmart / Supermarket Sales dataset.

    Panel daily sales and ratings for 3 branches (A, B, C) with
    89 days of observations spanning January to March 2019.

    Returns
    -------
    pl.DataFrame
        Panel DataFrame with ``"time"`` column and 6 panel columns using
        the ``__`` separator convention: ``"branch_a__total"``,
        ``"branch_b__total"``, ``"branch_c__total"``,
        ``"branch_a__rating"``, ``"branch_b__rating"``,
        ``"branch_c__rating"``.

    Notes
    -----
    Frequency: daily. Source: `Supermarket Sales CSV
    <https://raw.githubusercontent.com/selva86/datasets/master/supermarket_sales.csv>`_.

    Examples
    --------
    >>> from yohou.datasets import load_walmart_sales
    >>> df = load_walmart_sales()
    >>> len([c for c in df.columns if c != "time"])
    6

    """
    return _load_dataset("walmart_sales")


def load_ett_m1() -> pl.DataFrame:
    """Load the ETTm1 (Electricity Transformer Temperature) dataset.

    Multivariate electricity transformer temperature at 15-minute
    intervals with 7 features. The target is OT (Oil Temperature),
    commonly used as a forecasting benchmark.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns ``"time"``, ``"OT"`` (target), and
        covariates (``"HUFL"``, ``"HULL"``, ``"MUFL"``, ``"MULL"``,
        ``"LUFL"``, ``"LULL"``).

    Notes
    -----
    Frequency: 15-minute intervals. Source: `ETTm1 CSV
    <https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/ETTm1.csv>`_.

    Examples
    --------
    >>> from yohou.datasets import load_ett_m1
    >>> df = load_ett_m1()
    >>> "OT" in df.columns
    True

    """
    return _load_dataset("ett_m1")
