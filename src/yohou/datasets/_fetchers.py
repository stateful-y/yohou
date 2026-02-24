"""Dataset fetcher functions for Monash/Zenodo time series datasets."""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import polars as pl
from sklearn.datasets import fetch_file
from sklearn.utils import Bunch

from yohou.datasets._registry import (
    DOMINICK,
    ELECTRICITY_DEMAND,
    HOSPITAL,
    PEDESTRIAN_COUNTS,
    SUNSPOT,
    TOURISM_MONTHLY,
    TOURISM_QUARTERLY,
    RemoteFileMetadata,
)
from yohou.datasets._tsf_parser import parse_tsf


def get_data_home(data_home: str | os.PathLike | None = None) -> str:
    """Return the path of the yohou data directory.

    By default the data directory is set to a folder named ``yohou_data``
    in the user home folder. Alternatively, it can be set programmatically
    by giving an explicit folder path. The ``YOHOU_DATA`` environment
    variable has the same effect. If the folder does not already exist,
    it is automatically created.

    Parameters
    ----------
    data_home : str, PathLike, or None
        The path to the data directory. If ``None``, the default path
        is ``~/yohou_data``.

    Returns
    -------
    str
        The path to the data directory.

    """
    if data_home is None:
        data_home = os.environ.get("YOHOU_DATA", os.path.join("~", "yohou_data"))
    data_home = os.path.expanduser(str(data_home))
    os.makedirs(data_home, exist_ok=True)
    return data_home


def clear_data_home(data_home: str | os.PathLike | None = None) -> None:
    """Delete all the content of the data home cache.

    Parameters
    ----------
    data_home : str, PathLike, or None
        The path to the data directory. If ``None``, the default path
        is ``~/yohou_data``.

    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def _fetch_dataset(
    metadata: RemoteFileMetadata,
    dataset_name: str,
    *,
    value_column_name: str = "value",
    n_series: int | None = None,
    data_home: str | os.PathLike | None = None,
    download_if_missing: bool = True,
    n_retries: int = 3,
    delay: float = 1.0,
) -> Bunch:
    """Download, parse, cache, and return a Monash dataset.

    Parameters
    ----------
    metadata : RemoteFileMetadata
        Registry entry for the dataset.
    dataset_name : str
        Subdirectory name under data_home for caching.
    value_column_name : str
        Name for the value column(s) in the DataFrame.
    n_series : int or None
        Maximum number of series to load.  ``None`` loads every series
        in the dataset.  When set, a separate parquet cache is kept so
        that the full dataset and any subset coexist on disk.
    data_home : str, PathLike, or None
        Cache directory.
    download_if_missing : bool
        If False, raise OSError when data is not cached.
    n_retries : int
        Number of download retries.
    delay : float
        Delay between retries in seconds.

    Returns
    -------
    Bunch
        Dictionary-like object with ``frame``, ``feature_names``,
        ``DESCR``, ``frequency``, ``n_series``, ``filename`` keys.

    """
    data_home_str = get_data_home(data_home)
    dataset_dir = os.path.join(data_home_str, dataset_name)

    suffix = "" if n_series is None else f"_n{n_series}"
    parquet_path = os.path.join(dataset_dir, f"{dataset_name}{suffix}.parquet")

    if os.path.exists(parquet_path):
        frame = pl.read_parquet(parquet_path)
    else:
        if not download_if_missing:
            msg = f"Data not found and download_if_missing=False. Expected: {parquet_path}"
            raise OSError(msg)

        os.makedirs(dataset_dir, exist_ok=True)

        zip_path = fetch_file(
            url=metadata.url,
            folder=dataset_dir,
            local_filename=metadata.zip_filename,
            sha256=metadata.sha256,
            n_retries=n_retries,
            delay=delay,
        )

        frame = _extract_and_parse(
            zip_path,
            metadata.tsf_filename,
            value_column_name=value_column_name,
            n_series=n_series,
        )

        frame.write_parquet(parquet_path)

    feature_names = [c for c in frame.columns if c != "time"]

    return Bunch(
        frame=frame,
        feature_names=feature_names,
        DESCR=metadata.descr,
        frequency=metadata.frequency,
        n_series=len(feature_names),
        filename=parquet_path,
    )


def _extract_and_parse(
    zip_path: str | Path,
    tsf_filename: str,
    *,
    value_column_name: str,
    n_series: int | None = None,
) -> pl.DataFrame:
    """Extract a TSF file from a ZIP and parse it to a polars DataFrame."""
    with zipfile.ZipFile(zip_path, "r") as zf, zf.open(tsf_filename) as f:
        frame, _metadata = parse_tsf(f, value_column_name=value_column_name, n_series=n_series)

    if frame.schema.get("time") == pl.Date:
        frame = frame.with_columns(pl.col("time").cast(pl.Datetime))

    return frame


def fetch_tourism_monthly(
    *,
    n_series: int | None = None,
    data_home: str | os.PathLike | None = None,
    download_if_missing: bool = True,
    n_retries: int = 3,
    delay: float = 1.0,
) -> Bunch:
    """Fetch the Tourism Monthly dataset from Monash/Zenodo.

    366 monthly tourism time series from the tourism forecasting
    competition.

    Parameters
    ----------
    n_series : int or None, default=None
        Maximum number of series to include.  ``None`` loads all 366
        series.  A smaller value reduces memory usage and speeds up
        parsing.
    data_home : str, PathLike, or None
        Specify another download and cache folder for the datasets.
        By default all yohou data is stored in ``~/yohou_data/``.
    download_if_missing : bool, default=True
        If ``False``, raise an ``OSError`` if the data is not locally
        available instead of trying to download it.
    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.
    delay : float, default=1.0
        Number of seconds between retries.

    Returns
    -------
    Bunch
        Dictionary-like object with the following attributes:

        frame : pl.DataFrame
            DataFrame with ``"time"`` (Datetime) and up to 366 series
            columns using the ``__`` separator convention
            (e.g. ``"T1__tourists"``).
        feature_names : list of str
            Non-time column names.
        DESCR : str
            Full description of the dataset.
        frequency : str
            ``"1mo"``.
        n_series : int
            Number of series actually loaded.
        filename : str
            Path to the cached parquet file.

    References
    ----------
    .. [1] Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., &
       Montero-Manso, P. (2021). "Monash Time Series Forecasting Archive."
       Neural Information Processing Systems Track on Datasets and
       Benchmarks. https://doi.org/10.5281/zenodo.4656096

    Examples
    --------
    >>> from yohou.datasets import fetch_tourism_monthly
    >>> bunch = fetch_tourism_monthly()  # doctest: +SKIP
    >>> bunch.frame.columns[:2]  # doctest: +SKIP
    ['time', 'T1__tourists']

    """
    return _fetch_dataset(
        metadata=TOURISM_MONTHLY,
        dataset_name="tourism_monthly",
        value_column_name="tourists",
        n_series=n_series,
        data_home=data_home,
        download_if_missing=download_if_missing,
        n_retries=n_retries,
        delay=delay,
    )


def fetch_sunspot(
    *,
    data_home: str | os.PathLike | None = None,
    download_if_missing: bool = True,
    n_retries: int = 3,
    delay: float = 1.0,
) -> Bunch:
    """Fetch the Sunspot dataset (without missing values) from Monash/Zenodo.

    Single daily time series of sunspot numbers from 1818-01-08 to
    2020-05-31 (73 924 observations).

    Parameters
    ----------
    data_home : str, PathLike, or None
        Specify another download and cache folder for the datasets.
        By default all yohou data is stored in ``~/yohou_data/``.
    download_if_missing : bool, default=True
        If ``False``, raise an ``OSError`` if the data is not locally
        available instead of trying to download it.
    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.
    delay : float, default=1.0
        Number of seconds between retries.

    Returns
    -------
    Bunch
        Dictionary-like object with the following attributes:

        frame : pl.DataFrame
            DataFrame with ``"time"`` (Datetime) and
            ``"sunspot_number"`` (Float64).
        feature_names : list of str
            ``["sunspot_number"]``.
        DESCR : str
            Full description of the dataset.
        frequency : str
            ``"1d"``.
        n_series : int
            ``1``.
        filename : str
            Path to the cached parquet file.

    References
    ----------
    .. [1] Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., &
       Montero-Manso, P. (2021). "Monash Time Series Forecasting Archive."
       Neural Information Processing Systems Track on Datasets and
       Benchmarks. https://doi.org/10.5281/zenodo.4654722

    Examples
    --------
    >>> from yohou.datasets import fetch_sunspot
    >>> bunch = fetch_sunspot()  # doctest: +SKIP
    >>> bunch.frame.columns  # doctest: +SKIP
    ['time', 'sunspot_number']

    """
    return _fetch_dataset(
        metadata=SUNSPOT,
        dataset_name="sunspot",
        value_column_name="sunspot_number",
        data_home=data_home,
        download_if_missing=download_if_missing,
        n_retries=n_retries,
        delay=delay,
    )


def fetch_tourism_quarterly(
    *,
    n_series: int | None = None,
    data_home: str | os.PathLike | None = None,
    download_if_missing: bool = True,
    n_retries: int = 3,
    delay: float = 1.0,
) -> Bunch:
    """Fetch the Tourism Quarterly dataset from Monash/Zenodo.

    427 quarterly tourism time series from the tourism forecasting
    competition.

    Parameters
    ----------
    n_series : int or None, default=None
        Maximum number of series to include.  ``None`` loads all 427
        series.  A smaller value reduces memory usage and speeds up
        parsing.
    data_home : str, PathLike, or None
        Specify another download and cache folder for the datasets.
        By default all yohou data is stored in ``~/yohou_data/``.
    download_if_missing : bool, default=True
        If ``False``, raise an ``OSError`` if the data is not locally
        available instead of trying to download it.
    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.
    delay : float, default=1.0
        Number of seconds between retries.

    Returns
    -------
    Bunch
        Dictionary-like object with the following attributes:

        frame : pl.DataFrame
            DataFrame with ``"time"`` (Datetime) and up to 427 series
            columns using the ``__`` separator convention
            (e.g. ``"T1__tourists"``).
        feature_names : list of str
            Non-time column names.
        DESCR : str
            Full description of the dataset.
        frequency : str
            ``"3mo"``.
        n_series : int
            Number of series actually loaded.
        filename : str
            Path to the cached parquet file.

    References
    ----------
    .. [1] Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., &
       Montero-Manso, P. (2021). "Monash Time Series Forecasting Archive."
       Neural Information Processing Systems Track on Datasets and
       Benchmarks. https://doi.org/10.5281/zenodo.4656093

    Examples
    --------
    >>> from yohou.datasets import fetch_tourism_quarterly
    >>> bunch = fetch_tourism_quarterly()  # doctest: +SKIP
    >>> bunch.frame.columns[:2]  # doctest: +SKIP
    ['time', 'T1__tourists']

    """
    return _fetch_dataset(
        metadata=TOURISM_QUARTERLY,
        dataset_name="tourism_quarterly",
        value_column_name="tourists",
        n_series=n_series,
        data_home=data_home,
        download_if_missing=download_if_missing,
        n_retries=n_retries,
        delay=delay,
    )


def fetch_electricity_demand(
    *,
    data_home: str | os.PathLike | None = None,
    download_if_missing: bool = True,
    n_retries: int = 3,
    delay: float = 1.0,
) -> Bunch:
    """Fetch the Australian Electricity Demand dataset from Monash/Zenodo.

    5 half-hourly time series of electricity demand from five
    Australian states: New South Wales, Queensland, South Australia,
    Tasmania, and Victoria (232 272 time steps).

    Parameters
    ----------
    data_home : str, PathLike, or None
        Specify another download and cache folder for the datasets.
        By default all yohou data is stored in ``~/yohou_data/``.
    download_if_missing : bool, default=True
        If ``False``, raise an ``OSError`` if the data is not locally
        available instead of trying to download it.
    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.
    delay : float, default=1.0
        Number of seconds between retries.

    Returns
    -------
    Bunch
        Dictionary-like object with the following attributes:

        frame : pl.DataFrame
            DataFrame with ``"time"`` (Datetime) and 5 state columns
            using the ``__`` separator convention
            (e.g. ``"nsw__demand"``).
        feature_names : list of str
            Non-time column names.
        DESCR : str
            Full description of the dataset.
        frequency : str
            ``"30m"``.
        n_series : int
            ``5``.
        filename : str
            Path to the cached parquet file.

    References
    ----------
    .. [1] Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., &
       Montero-Manso, P. (2021). "Monash Time Series Forecasting Archive."
       Neural Information Processing Systems Track on Datasets and
       Benchmarks. https://doi.org/10.5281/zenodo.4659727

    Examples
    --------
    >>> from yohou.datasets import fetch_electricity_demand
    >>> bunch = fetch_electricity_demand()  # doctest: +SKIP
    >>> bunch.frame.columns[:2]  # doctest: +SKIP
    ['time', 'nsw__demand']

    """
    return _fetch_dataset(
        metadata=ELECTRICITY_DEMAND,
        dataset_name="electricity_demand",
        value_column_name="demand",
        data_home=data_home,
        download_if_missing=download_if_missing,
        n_retries=n_retries,
        delay=delay,
    )


def fetch_dominick(
    *,
    n_series: int | None = 50,
    data_home: str | os.PathLike | None = None,
    download_if_missing: bool = True,
    n_retries: int = 3,
    delay: float = 1.0,
) -> Bunch:
    """Fetch the Dominick dataset from Monash/Zenodo.

    Weekly time series representing the profit of individual stock
    keeping units from a retailer (Dominick's Finer Foods).  The full
    dataset contains 115 704 series; by default only the first 50 are
    loaded to keep memory usage reasonable.

    Parameters
    ----------
    n_series : int or None, default=50
        Maximum number of series to include.  ``None`` loads all
        115 704 series (several GB of memory).  The default of 50
        keeps the dataset small enough for interactive examples.
    data_home : str, PathLike, or None
        Specify another download and cache folder for the datasets.
        By default all yohou data is stored in ``~/yohou_data/``.
    download_if_missing : bool, default=True
        If ``False``, raise an ``OSError`` if the data is not locally
        available instead of trying to download it.
    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.
    delay : float, default=1.0
        Number of seconds between retries.

    Returns
    -------
    Bunch
        Dictionary-like object with the following attributes:

        frame : pl.DataFrame
            DataFrame with ``"time"`` (Datetime) and up to 115 704
            series columns using the ``__`` separator convention
            (e.g. ``"T1__profit"``).
        feature_names : list of str
            Non-time column names.
        DESCR : str
            Full description of the dataset.
        frequency : str
            ``"1w"``.
        n_series : int
            Number of series actually loaded.
        filename : str
            Path to the cached parquet file.

    References
    ----------
    .. [1] Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., &
       Montero-Manso, P. (2021). "Monash Time Series Forecasting Archive."
       Neural Information Processing Systems Track on Datasets and
       Benchmarks. https://doi.org/10.5281/zenodo.4654802

    Examples
    --------
    >>> from yohou.datasets import fetch_dominick
    >>> bunch = fetch_dominick()  # doctest: +SKIP
    >>> bunch.frame.columns[:2]  # doctest: +SKIP
    ['time', 'T1__profit']

    """
    return _fetch_dataset(
        metadata=DOMINICK,
        dataset_name="dominick",
        value_column_name="profit",
        n_series=n_series,
        data_home=data_home,
        download_if_missing=download_if_missing,
        n_retries=n_retries,
        delay=delay,
    )


def fetch_pedestrian_counts(
    *,
    n_series: int | None = 20,
    data_home: str | os.PathLike | None = None,
    download_if_missing: bool = True,
    n_retries: int = 3,
    delay: float = 1.0,
) -> Bunch:
    """Fetch the Melbourne Pedestrian Counts dataset from Monash/Zenodo.

    Hourly time series of pedestrian counts captured from sensors in
    Melbourne city from May 2009 to April 2020.  The full dataset
    contains 66 sensors; by default only the first 20 are loaded.

    Parameters
    ----------
    n_series : int or None, default=20
        Maximum number of sensor series to include.  ``None`` loads
        all 66 series.  The default of 20 keeps memory usage
        manageable for interactive examples.
    data_home : str, PathLike, or None
        Specify another download and cache folder for the datasets.
        By default all yohou data is stored in ``~/yohou_data/``.
    download_if_missing : bool, default=True
        If ``False``, raise an ``OSError`` if the data is not locally
        available instead of trying to download it.
    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.
    delay : float, default=1.0
        Number of seconds between retries.

    Returns
    -------
    Bunch
        Dictionary-like object with the following attributes:

        frame : pl.DataFrame
            DataFrame with ``"time"`` (Datetime) and up to 66 sensor
            columns using the ``__`` separator convention
            (e.g. ``"T1__count"``).
        feature_names : list of str
            Non-time column names.
        DESCR : str
            Full description of the dataset.
        frequency : str
            ``"1h"``.
        n_series : int
            Number of series actually loaded.
        filename : str
            Path to the cached parquet file.

    References
    ----------
    .. [1] Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., &
       Montero-Manso, P. (2021). "Monash Time Series Forecasting Archive."
       Neural Information Processing Systems Track on Datasets and
       Benchmarks. https://doi.org/10.5281/zenodo.4656626

    Examples
    --------
    >>> from yohou.datasets import fetch_pedestrian_counts
    >>> bunch = fetch_pedestrian_counts()  # doctest: +SKIP
    >>> bunch.frame.columns[:2]  # doctest: +SKIP
    ['time', 'T1__count']

    """
    return _fetch_dataset(
        metadata=PEDESTRIAN_COUNTS,
        dataset_name="pedestrian_counts",
        value_column_name="count",
        n_series=n_series,
        data_home=data_home,
        download_if_missing=download_if_missing,
        n_retries=n_retries,
        delay=delay,
    )


def fetch_hospital(
    *,
    n_series: int | None = None,
    data_home: str | os.PathLike | None = None,
    download_if_missing: bool = True,
    n_retries: int = 3,
    delay: float = 1.0,
) -> Bunch:
    """Fetch the Hospital dataset from Monash/Zenodo.

    767 monthly time series representing patient counts related to
    medical products from January 2000 to December 2006.

    Parameters
    ----------
    n_series : int or None, default=None
        Maximum number of series to include.  ``None`` loads all 767
        series.  A smaller value reduces memory usage and speeds up
        parsing.
    data_home : str, PathLike, or None
        Specify another download and cache folder for the datasets.
        By default all yohou data is stored in ``~/yohou_data/``.
    download_if_missing : bool, default=True
        If ``False``, raise an ``OSError`` if the data is not locally
        available instead of trying to download it.
    n_retries : int, default=3
        Number of retries when HTTP errors are encountered.
    delay : float, default=1.0
        Number of seconds between retries.

    Returns
    -------
    Bunch
        Dictionary-like object with the following attributes:

        frame : pl.DataFrame
            DataFrame with ``"time"`` (Datetime) and up to 767 series
            columns using the ``__`` separator convention
            (e.g. ``"T1__patients"``).
        feature_names : list of str
            Non-time column names.
        DESCR : str
            Full description of the dataset.
        frequency : str
            ``"1mo"``.
        n_series : int
            Number of series actually loaded.
        filename : str
            Path to the cached parquet file.

    References
    ----------
    .. [1] Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., &
       Montero-Manso, P. (2021). "Monash Time Series Forecasting Archive."
       Neural Information Processing Systems Track on Datasets and
       Benchmarks. https://doi.org/10.5281/zenodo.4656014

    Examples
    --------
    >>> from yohou.datasets import fetch_hospital
    >>> bunch = fetch_hospital()  # doctest: +SKIP
    >>> bunch.frame.columns[:2]  # doctest: +SKIP
    ['time', 'T1__patients']

    """
    return _fetch_dataset(
        metadata=HOSPITAL,
        dataset_name="hospital",
        value_column_name="patients",
        n_series=n_series,
        data_home=data_home,
        download_if_missing=download_if_missing,
        n_retries=n_retries,
        delay=delay,
    )
