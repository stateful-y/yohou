import importlib.resources

import polars as pl

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
        Dataframe with columns "time" (Date) and "Passengers" (Int64).

    Notes
    -----
    - **name**: Air Passengers
    - **description**: Univariate monthly airline passengers series
    - **type**: univariate
    - **frequency**: monthly
    - **source_url**: https://github.com/jbrownlee/Datasets/blob/master/airline-passengers.csv
    - **format**: parquet
    - **characteristics**: Strong trend and multiplicative seasonality, 144 observations (12 years)
    - **use_cases**: Trend analysis, seasonal decomposition, forecasting, transformation techniques
    - **example_notebooks**: `examples/datasets/air_passengers.py` - Comprehensive demonstration of 13 plotting functions
    """
    return _load_dataset("air_passengers")


def load_sunspots() -> pl.DataFrame:
    """
    Load the Monthly Sunspots dataset.

    Returns
    -------
    pl.DataFrame
        Dataframe with columns "time" (Date) and "Sunspots" (Float64).

    Notes
    -----
    - **name**: Sunspots
    - **description**: Long univariate series of international monthly sunspot counts
    - **type**: univariate
    - **frequency**: monthly
    - **source_url**: https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv
    - **format**: parquet
    - **characteristics**: ~11-year cyclic patterns (solar cycles), 2820 observations (1749-1983)
    - **use_cases**: Cyclic pattern detection, frequency analysis, smoothing techniques, periodogram analysis
    - **example_notebooks**: `examples/datasets/sunspots.py` - Smoothing and frequency domain demonstrations
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
    - **name**: M4 Monthly (Subset)
    - **description**: Partial M4 dataset with monthly time series
    - **type**: panel
    - **frequency**: monthly
    - **source_url**: https://zenodo.org/records/4656480
    - **format**: parquet
    - **characteristics**: 50 heterogeneous time series with varying patterns and scales
    - **use_cases**: Panel visualization with facets, dropdown selection, aggregated statistics
    - **example_notebooks**: `examples/datasets/m4_monthly.py` - Panel data with facets and interactive dropdowns
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
    - **name**: M4 Quarterly (Subset)
    - **description**: Partial M4 dataset with quarterly time series
    - **type**: panel
    - **frequency**: quarterly
    - **source_url**: https://zenodo.org/records/4656410
    - **format**: parquet
    - **characteristics**: 50 quarterly time series, ideal for seasonal subseries analysis
    - **use_cases**: Quarterly seasonality patterns, panel forecasting, faceted comparisons
    - **example_notebooks**: `examples/datasets/m4_quarterly.py` - Quarterly seasonality and year-over-year comparison
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
    - **name**: M4 Hourly (Subset)
    - **description**: Partial M4 dataset with hourly time series
    - **type**: panel
    - **frequency**: hourly
    - **source_url**: https://zenodo.org/records/4656589
    - **format**: parquet
    - **characteristics**: 50 high-frequency hourly series, may contain missing values
    - **use_cases**: High-frequency analysis, missing data visualization, intraday patterns
    - **example_notebooks**: `examples/datasets/m4_hourly.py` - Intraday patterns and hour-of-day seasonality
    """
    return _load_dataset("m4_hourly")


def load_australian_tourism() -> pl.DataFrame:
    """
    Load the Australian Tourism dataset.

    Returns
    -------
    pl.DataFrame
        Dataframe with columns "time", "Region", "State", "Purpose", "Trips".

    Notes
    -----
    - **name**: Australian Tourism
    - **description**: Panel of tourism demand series by region and frequency
    - **type**: panel / hierarchical
    - **frequency**: monthly
    - **source_url**: https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/australia_tourism.csv
    - **format**: parquet
    - **characteristics**: Hierarchical structure with Region, State, Purpose groupings
    - **use_cases**: Hierarchical forecasting, multi-level aggregation, regional comparisons
    - **example_notebooks**: `examples/datasets/australian_tourism.py` - Hierarchical aggregation and regional analysis
    """
    return _load_dataset("australian_tourism")


def load_vic_electricity() -> pl.DataFrame:
    """
    Load the Victoria Electricity dataset.

    Returns
    -------
    pl.DataFrame
        Dataframe with columns "time", "Demand", "Temperature", "Holiday".

    Notes
    -----
    - **name**: Victoria Electricity
    - **description**: Multivariate household / grid electricity consumption
    - **type**: multivariate
    - **frequency**: 30-minute intervals
    - **source_url**: https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/vic_electricity.csv
    - **format**: parquet
    - **characteristics**: High-frequency data with demand, temperature, and holiday indicator
    - **use_cases**: Multivariate visualization, cross-correlation analysis, weather impact studies
    - **example_notebooks**: `examples/datasets/vic_electricity.py` - Cross-correlation and rolling statistics
    """
    return _load_dataset("vic_electricity")


def load_store_sales() -> pl.DataFrame:
    """
    Load the Store Sales dataset (Rossmann substitute).

    Returns
    -------
    pl.DataFrame
        Dataframe with columns "time", "store", "item", "sales".

    Notes
    -----
    - **name**: Store Sales
    - **description**: Panel retail sales with exogenous features
    - **type**: panel
    - **frequency**: daily
    - **source_url**: https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/store_sales.csv
    - **format**: parquet
    - **characteristics**: 10 stores × 50 items (500 series), daily retail transaction data
    - **use_cases**: Calendar heatmaps, weekday patterns, store/item ranking, retail forecasting
    - **example_notebooks**: `examples/datasets/store_sales.py` - Calendar heatmaps and retail pattern analysis
    """
    return _load_dataset("store_sales")


def load_walmart_sales() -> pl.DataFrame:
    """
    Load the Walmart / Supermarket Sales dataset.

    Returns
    -------
    pl.DataFrame
        Dataframe with columns "time", "Total", "Branch", "City", etc.

    Notes
    -----
    - **name**: Walmart Sales (Supermarket)
    - **description**: Panel sales with covariates
    - **type**: panel / covariates
    - **frequency**: daily/ex-ante
    - **source_url**: https://raw.githubusercontent.com/selva86/datasets/master/supermarket_sales.csv
    - **format**: parquet
    - **characteristics**: 3 branches with multiple product categories and customer segments
    - **use_cases**: Exogenous variable visualization, branch comparison, customer behavior analysis
    - **example_notebooks**: `examples/datasets/walmart_sales.py` - Transaction aggregation and covariate analysis
    """
    return _load_dataset("walmart_sales")


def load_ett_m1() -> pl.DataFrame:
    """
    Load the ETTm1 (Electricity Transformer Temperature) dataset.

    Returns
    -------
    pl.DataFrame
        Dataframe with columns "time", "OT", and covariates (HUFL, HULL, etc).

    Notes
    -----
    - **name**: ETTm1
    - **description**: Electricity Transformer Temperature (multivariate)
    - **type**: multivariate
    - **frequency**: 15-minute intervals
    - **source_url**: https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/ETTm1.csv
    - **format**: parquet
    - **characteristics**: 7 temperature features, target is OT (Oil Temperature), long-term dependencies
    - **use_cases**: Transformer forecasting benchmark, multivariate dependencies, industrial IoT
    - **example_notebooks**: `examples/datasets/ett_m1.py` - Multivariate analysis and cross-correlation
    """
    return _load_dataset("ett_m1")
