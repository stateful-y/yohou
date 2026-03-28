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
    KDD_CUP_2018,
    PEDESTRIAN_COUNTS,
    SUNSPOT,
    TOURISM_MONTHLY,
    TOURISM_QUARTERLY,
    RemoteFileMetadata,
)
from yohou.datasets._tsf_parser import parse_tsf

_KDD_CUP_MEASUREMENTS = frozenset({"pm2.5", "pm10", "no2", "co", "o3", "so2"})


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

    See Also
    --------
    `clear_data_home` : Delete all cached datasets.

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

    See Also
    --------
    `get_data_home` : Return the path of the data directory.

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

    See Also
    --------
    `fetch_tourism_quarterly` : Quarterly tourism series from the same competition.
    `fetch_hospital` : Monthly hospital patient count series.
    `get_data_home` : Return the path of the data directory.

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

    See Also
    --------
    `fetch_tourism_monthly` : Monthly tourism series.
    `fetch_electricity_demand` : Half-hourly electricity demand series.
    `get_data_home` : Return the path of the data directory.

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

    See Also
    --------
    `fetch_tourism_monthly` : Monthly tourism series from the same competition.
    `fetch_hospital` : Monthly hospital patient count series.
    `get_data_home` : Return the path of the data directory.

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

    See Also
    --------
    `fetch_pedestrian_counts` : Hourly pedestrian sensor series.
    `fetch_kdd_cup` : Hourly air quality series.
    `get_data_home` : Return the path of the data directory.

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

    See Also
    --------
    `fetch_tourism_monthly` : Monthly tourism series.
    `fetch_hospital` : Monthly hospital patient count series.
    `get_data_home` : Return the path of the data directory.

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

    See Also
    --------
    `fetch_electricity_demand` : Half-hourly electricity demand series.
    `fetch_kdd_cup` : Hourly air quality series.
    `get_data_home` : Return the path of the data directory.

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

    See Also
    --------
    `fetch_tourism_monthly` : Monthly tourism series.
    `fetch_dominick` : Weekly retail profit series.
    `get_data_home` : Return the path of the data directory.

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


def _restructure_kdd_cup_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Rename KDD Cup columns from ``station_measurement__value`` to ``station__measurement``.

    The raw TSF parser produces column names like
    ``beijing_aotizhongxin_aq_pm2.5__value`` where the measurement type
    is embedded in the group prefix. This function restructures them
    into ``beijing_aotizhongxin_aq__pm2.5`` so that each station becomes
    a proper multivariate panel group with one member per measurement.

    Columns that do not match the expected pattern (e.g. ``"time"`` or
    mock columns like ``"T1__value"``) are left unchanged.
    """
    rename_map: dict[str, str] = {}
    for col in frame.columns:
        if "__" not in col:
            continue
        prefix, suffix = col.rsplit("__", 1)
        if suffix != "value":
            continue
        for m in _KDD_CUP_MEASUREMENTS:
            if prefix.endswith(f"_{m}"):
                station = prefix[: -len(m) - 1]
                rename_map[col] = f"{station}__{m}"
                break
    if rename_map:
        frame = frame.rename(rename_map)
    return frame


def fetch_kdd_cup(
    *,
    n_groups: int | None = 5,
    data_home: str | os.PathLike | None = None,
    download_if_missing: bool = True,
    n_retries: int = 3,
    delay: float = 1.0,
) -> Bunch:
    """Fetch the KDD Cup 2018 air quality dataset from Monash/Zenodo.

    Hourly time series of air quality measurements (PM2.5, PM10, NO2,
    CO, O3, SO2) from 59 monitoring stations in Beijing and London.
    This is a multivariate panel dataset: each station (panel group)
    contains multiple measurement columns.

    Column names use yohou's ``__`` separator convention with the
    station as group prefix and the measurement as member suffix,
    e.g. ``"beijing_dongsi_aq__pm2.5"``.

    Parameters
    ----------
    n_groups : int or None, default=5
        Maximum number of station groups to include. Each station has
        6 measurement series (PM2.5, PM10, NO2, CO, O3, SO2), so
        ``n_groups=5`` loads 30 raw series. ``None`` loads all 59
        stations (270 series).
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
            DataFrame with ``"time"`` (Datetime) and up to 270 series
            columns using the ``__`` separator convention
            (e.g. ``"beijing_dongsi_aq__pm2.5"``).
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

    See Also
    --------
    `fetch_electricity_demand` : Half-hourly electricity demand series.
    `fetch_pedestrian_counts` : Hourly pedestrian sensor series.
    `get_data_home` : Return the path of the data directory.

    References
    ----------
    .. [1] Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., &
       Montero-Manso, P. (2021). "Monash Time Series Forecasting Archive."
       Neural Information Processing Systems Track on Datasets and
       Benchmarks. https://doi.org/10.5281/zenodo.4656756

    Examples
    --------
    >>> from yohou.datasets import fetch_kdd_cup
    >>> bunch = fetch_kdd_cup()  # doctest: +SKIP
    >>> bunch.frame.columns[:3]  # doctest: +SKIP
    ['time', 'beijing_aotizhongxin_aq__pm2.5', 'beijing_aotizhongxin_aq__pm10']

    """
    _n_measurements = len(_KDD_CUP_MEASUREMENTS)
    n_series = n_groups * _n_measurements if n_groups is not None else None
    bunch = _fetch_dataset(
        metadata=KDD_CUP_2018,
        dataset_name="kdd_cup_2018",
        value_column_name="value",
        n_series=n_series,
        data_home=data_home,
        download_if_missing=download_if_missing,
        n_retries=n_retries,
        delay=delay,
    )
    bunch.frame = _restructure_kdd_cup_columns(bunch.frame)
    bunch.feature_names = [c for c in bunch.frame.columns if c != "time"]
    return bunch


_WHO_PM25_THRESHOLDS = (15.0, 35.0, 75.0)
_WHO_PM25_LABELS = ("good", "moderate", "unhealthy", "hazardous")


def fetch_air_quality_classification(
    *,
    data_home: str | os.PathLike | None = None,
    download_if_missing: bool = True,
    n_retries: int = 3,
    delay: float = 1.0,
) -> Bunch:
    """Fetch a categorical air quality dataset derived from KDD Cup 2018.

    Downloads the KDD Cup 2018 dataset (single station) and bins its
    PM2.5 values into four WHO air quality categories using guideline
    thresholds of 15, 35, and 75 ug/m3. The remaining five pollutant
    measurements (PM10, NO2, CO, O3, SO2) become exogenous features.
    Rows with null PM2.5 values are dropped.

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

        y : pl.DataFrame
            DataFrame with ``"time"`` (Datetime) and ``"air_quality"``
            (Utf8) columns. The ``"air_quality"`` column contains one
            of ``"good"``, ``"moderate"``, ``"unhealthy"``, or
            ``"hazardous"``.
        X : pl.DataFrame
            DataFrame with ``"time"`` and 5 pollutant feature columns
            (``"pm10"``, ``"no2"``, ``"co"``, ``"o3"``, ``"so2"``).
        feature_names : list of str
            Feature column names (excludes ``"time"``).
        target_names : list of str
            ``["air_quality"]``.
        classes : list of str
            ``["good", "hazardous", "moderate", "unhealthy"]`` (sorted).
        DESCR : str
            Human-readable dataset description.

    See Also
    --------
    `fetch_kdd_cup` : Full KDD Cup 2018 air quality dataset.
    `fetch_demand_classification` : Categorical electricity demand dataset.

    Examples
    --------
    >>> from yohou.datasets import fetch_air_quality_classification
    >>> data = fetch_air_quality_classification()  # doctest: +SKIP
    >>> data.y.columns  # doctest: +SKIP
    ['time', 'air_quality']
    >>> sorted(data.classes)  # doctest: +SKIP
    ['good', 'hazardous', 'moderate', 'unhealthy']

    """
    bunch = fetch_kdd_cup(
        n_groups=1,
        data_home=data_home,
        download_if_missing=download_if_missing,
        n_retries=n_retries,
        delay=delay,
    )
    frame = bunch.frame

    # Identify station prefix from first non-time column
    non_time = [c for c in frame.columns if c != "time"]
    station_prefix = non_time[0].split("__")[0]

    pm25_col = f"{station_prefix}__pm2.5"
    feature_measurements = ["pm10", "no2", "co", "o3", "so2"]
    feature_cols = [f"{station_prefix}__{m}" for m in feature_measurements]

    # Drop rows with null PM2.5
    frame = frame.drop_nulls(subset=[pm25_col])

    # Bin PM2.5 into WHO categories
    air_quality = (
        pl.when(pl.col(pm25_col) < _WHO_PM25_THRESHOLDS[0])
        .then(pl.lit("good"))
        .when(pl.col(pm25_col) < _WHO_PM25_THRESHOLDS[1])
        .then(pl.lit("moderate"))
        .when(pl.col(pm25_col) < _WHO_PM25_THRESHOLDS[2])
        .then(pl.lit("unhealthy"))
        .otherwise(pl.lit("hazardous"))
    )

    y = frame.select("time", air_quality.alias("air_quality"))
    X = frame.select("time", *feature_cols).rename(
        {c: c.split("__")[1] for c in feature_cols}
    )

    classes = sorted(set(y["air_quality"].to_list()))

    return Bunch(
        y=y,
        X=X,
        feature_names=[c for c in X.columns if c != "time"],
        target_names=["air_quality"],
        classes=classes,
        DESCR=(
            "Air quality classification dataset derived from KDD Cup 2018. "
            "PM2.5 values are binned into four WHO air quality categories: "
            "good (<15 ug/m3), moderate (15-35), unhealthy (35-75), "
            "hazardous (>75). Features are the remaining five pollutant "
            "measurements (PM10, NO2, CO, O3, SO2) from the same station."
        ),
    )


def fetch_demand_classification(
    *,
    data_home: str | os.PathLike | None = None,
    download_if_missing: bool = True,
    n_retries: int = 3,
    delay: float = 1.0,
) -> Bunch:
    """Fetch a categorical electricity demand dataset from Monash/Zenodo.

    Downloads the Australian Electricity Demand dataset and bins
    Victoria's half-hourly demand into three levels (low, medium, high)
    using tercile thresholds. The remaining four states (NSW, QLD, SA,
    TAS) become exogenous features. Rows with null Victoria demand are
    dropped.

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

        y : pl.DataFrame
            DataFrame with ``"time"`` (Datetime) and ``"demand_level"``
            (Utf8) columns. The ``"demand_level"`` column contains one
            of ``"low"``, ``"medium"``, or ``"high"``.
        X : pl.DataFrame
            DataFrame with ``"time"`` and 4 state demand columns
            (``"nsw__demand"``, ``"qun__demand"``, ``"sa__demand"``,
            ``"tas__demand"``).
        feature_names : list of str
            Feature column names (excludes ``"time"``).
        target_names : list of str
            ``["demand_level"]``.
        classes : list of str
            ``["high", "low", "medium"]`` (sorted).
        DESCR : str
            Human-readable dataset description.

    See Also
    --------
    `fetch_electricity_demand` : Full Australian electricity demand dataset.
    `fetch_air_quality_classification` : Categorical air quality dataset.

    Examples
    --------
    >>> from yohou.datasets import fetch_demand_classification
    >>> data = fetch_demand_classification()  # doctest: +SKIP
    >>> data.y.columns  # doctest: +SKIP
    ['time', 'demand_level']
    >>> sorted(data.classes)  # doctest: +SKIP
    ['high', 'low', 'medium']

    """
    bunch = fetch_electricity_demand(
        data_home=data_home,
        download_if_missing=download_if_missing,
        n_retries=n_retries,
        delay=delay,
    )
    frame = bunch.frame

    target_col = "vic__demand"
    feature_cols = ["nsw__demand", "qun__demand", "sa__demand", "tas__demand"]

    # Drop rows with null target
    frame = frame.drop_nulls(subset=[target_col])

    # Compute tercile thresholds
    q33 = frame[target_col].quantile(1 / 3)
    q66 = frame[target_col].quantile(2 / 3)

    demand_level = (
        pl.when(pl.col(target_col) < q33)
        .then(pl.lit("low"))
        .when(pl.col(target_col) < q66)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("high"))
    )

    y = frame.select("time", demand_level.alias("demand_level"))
    X = frame.select("time", *feature_cols)

    classes = sorted(set(y["demand_level"].to_list()))

    return Bunch(
        y=y,
        X=X,
        feature_names=[c for c in X.columns if c != "time"],
        target_names=["demand_level"],
        classes=classes,
        DESCR=(
            "Electricity demand classification dataset derived from the "
            "Australian Electricity Demand dataset (Monash/Zenodo). "
            "Victoria's half-hourly demand is binned into three tercile-based "
            "levels: low, medium, high. Features are the demand series from "
            "the remaining four Australian states (NSW, QLD, SA, TAS)."
        ),
    )
