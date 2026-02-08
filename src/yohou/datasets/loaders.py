import polars as pl
import importlib.resources
from yohou.datasets import data

def _load_dataset(name: str) -> pl.DataFrame:
    """Helper to load a parquet dataset from the package data directory."""
    # Use resource access that works for zipped packages/wheels
    with importlib.resources.as_file(importlib.resources.files(data).joinpath(f"{name}.parquet")) as p:
        return pl.read_parquet(p)

def load_air_passengers() -> pl.DataFrame:
    """
    Load the Air Passengers dataset.

    Returns
    -------
    pl.DataFrame
        Dataframe with columns "time" (Date) and "y" (Int64).

    Notes
    -----
    - name: Air Passengers
    - description: Univariate monthly airline passengers series
    - type: univariate
    - frequency: monthly
    - source_url: https://github.com/jbrownlee/Datasets/blob/master/airline-passengers.csv
    - format: parquet
    - notes: Classic Box & Jenkins airline data. Monthly totals of international airline passengers, 1949 to 1960.
    """
    return _load_dataset("air_passengers")

def load_sunspots() -> pl.DataFrame:
    """
    Load the Monthly Sunspots dataset.

    Returns
    -------
    pl.DataFrame
        Dataframe with columns "time" (Date) and "y" (Float64).

    Notes
    -----
    - name: Sunspots
    - description: Long univariate series of international monthly sunspot counts
    - type: univariate
    - frequency: monthly
    - source_url: https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv
    - format: parquet
    - notes: Monthly mean total sunspot number, from 1749 to 1983.
    """
    return _load_dataset("sunspots")

def load_m4_monthly() -> pl.DataFrame:
    """
    Load a subset of the M4 Monthly dataset.

    Returns
    -------
    pl.DataFrame
        Dataframe with columns "unique_id" (String), "time" (Datetime), and "y" (Float64).

    Notes
    -----
    - name: M4 Monthly (Subset)
    - description: Partial M4 dataset with monthly time series
    - type: panel
    - frequency: monthly
    - source_url: https://zenodo.org/records/4656480
    - format: parquet
    - notes: Contains first 50 series from M4 Monthly Train set. Times are synthetic starting 2000-01-01.
    """
    return _load_dataset("m4_monthly")

def load_m4_quarterly() -> pl.DataFrame:
    """
    Load a subset of the M4 Quarterly dataset.

    Returns
    -------
    pl.DataFrame
        Dataframe with columns "unique_id" (String), "time" (Datetime), and "y" (Float64).

    Notes
    -----
    - name: M4 Quarterly (Subset)
    - description: Partial M4 dataset with quarterly time series
    - type: panel
    - frequency: quarterly
    - source_url: https://zenodo.org/records/4656410
    - format: parquet
    - notes: Contains first 50 series from M4 Quarterly Train set. Times are synthetic starting 2000-01-01.
    """
    return _load_dataset("m4_quarterly")

def load_m4_hourly() -> pl.DataFrame:
    """
    Load a subset of the M4 Hourly dataset.

    Returns
    -------
    pl.DataFrame
        Dataframe with columns "unique_id" (String), "time" (Datetime), and "y" (Float64).

    Notes
    -----
    - name: M4 Hourly (Subset)
    - description: Partial M4 dataset with hourly time series
    - type: panel
    - frequency: hourly
    - source_url: https://zenodo.org/records/4656589
    - format: parquet
    - notes: Contains first 50 series from M4 Hourly Train set. Times are synthetic starting 2000-01-01.
    """
    return _load_dataset("m4_hourly")

def load_australian_tourism() -> pl.DataFrame:
    """
    Load the Australian Tourism dataset.

    Returns
    -------
    pl.DataFrame
        Dataframe with columns "time", "Region", "State", "Purpose", "y".

    Notes
    -----
    - name: Australian Tourism
    - description: Panel of tourism demand series by region and frequency
    - type: panel / hierarchical
    - frequency: monthly
    - source_url: https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/australia_tourism.csv
    - format: parquet
    - notes: Sourced from skforecast-datasets.
    """
    return _load_dataset("australian_tourism")

def load_vic_electricity() -> pl.DataFrame:
    """
    Load the Victoria Electricity dataset.

    Returns
    -------
    pl.DataFrame
        Dataframe with columns "time", "y" (Demand), "Temperature", "Holiday".

    Notes
    -----
    - name: Victoria Electricity
    - description: Multivariate household / grid electricity consumption
    - type: multivariate
    - frequency: 30-min
    - source_url: https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/vic_electricity.csv
    - format: parquet
    - notes: Contains 30-minute electricity demand for Victoria, Australia.
    """
    return _load_dataset("vic_electricity")

def load_store_sales() -> pl.DataFrame:
    """
    Load the Store Sales dataset (Rossmann substitute).

    Returns
    -------
    pl.DataFrame
        Dataframe with columns "time", "store", "item", "y" (sales).

    Notes
    -----
    - name: Store Sales
    - description: Panel retail sales with exogenous features
    - type: panel
    - frequency: daily
    - source_url: https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/store_sales.csv
    - format: parquet
    - notes: Includes 10 stores and 50 items. Similar structure to Rossmann.
    """
    return _load_dataset("store_sales")

def load_walmart_sales() -> pl.DataFrame:
    """
    Load the Walmart / Supermarket Sales dataset.

    Returns
    -------
    pl.DataFrame
        Dataframe with columns "time", "y", "Branch", "City", etc.

    Notes
    -----
    - name: Walmart Sales (Supermarket)
    - description: Panel sales with covariates
    - type: panel / covariates
    - frequency: daily/ex-ante
    - source_url: https://raw.githubusercontent.com/selva86/datasets/master/supermarket_sales.csv
    - format: parquet
    - notes: Sales data from 3 branches of a supermarket. One quarter of data (2019).
    """
    return _load_dataset("walmart_sales")

def load_ett_m1() -> pl.DataFrame:
    """
    Load the ETTm1 (Electricity Transformer Temperature) dataset.

    Returns
    -------
    pl.DataFrame
        Dataframe with columns "time", "y" (OT), and covariates (HUFL, HULL, etc).

    Notes
    -----
    - name: ETTm1
    - description: Multivariate high-frequency dataset for load forecasting
    - type: multivariate
    - frequency: 15-min
    - source_url: https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/ETTm1.csv
    - format: parquet
    - notes: Target is OT (Oil Temperature).
    """
    return _load_dataset("ett_m1")
