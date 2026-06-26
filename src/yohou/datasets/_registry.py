"""Remote file metadata registry for Monash/Zenodo datasets."""

from collections import namedtuple

RemoteFileMetadata = namedtuple(
    "RemoteFileMetadata",
    ["url", "zip_filename", "tsf_filename", "sha256", "descr", "frequency", "n_series"],
)

_ZENODO_URL = "https://zenodo.org/records/{record_id}/files/{filename}?download=1"

TOURISM_MONTHLY = RemoteFileMetadata(
    url=_ZENODO_URL.format(record_id="4656096", filename="tourism_monthly_dataset.zip"),
    zip_filename="tourism_monthly_dataset.zip",
    tsf_filename="tourism_monthly_dataset.tsf",
    sha256="6e434ad8a0ef6acb55ebee0e1d1ffe22707ea4e62c05e091326267f3a5b4a93c",
    descr="""\
Tourism Monthly Dataset
=======================

366 monthly tourism time series from the tourism forecasting competition.

Source: Monash Time Series Forecasting Archive (forecastingdata.org)
License: CC BY 4.0
DOI: 10.5281/zenodo.4656096
Citation: Godahewa et al. (2021), "Monash Time Series Forecasting Archive"
""",
    frequency="1mo",
    n_series=366,
)

SUNSPOT = RemoteFileMetadata(
    url=_ZENODO_URL.format(
        record_id="4654722",
        filename="sunspot_dataset_without_missing_values.zip",
    ),
    zip_filename="sunspot_dataset_without_missing_values.zip",
    tsf_filename="sunspot_dataset_without_missing_values.tsf",
    sha256="e47ca44f2046a95eb539ada9f1eab7c0091483b4b872f8b6682987f67342cca8",
    descr="""\
Sunspot Dataset (without missing values)
=========================================

Single daily time series of sunspot numbers from 1818-01-08 to
2020-05-31. Missing values were replaced by carrying forward the
corresponding last observations (LOCF method).

Source: Monash Time Series Forecasting Archive (forecastingdata.org)
License: CC BY 4.0
DOI: 10.5281/zenodo.4654722
Citation: Godahewa et al. (2021), "Monash Time Series Forecasting Archive"
""",
    frequency="1d",
    n_series=1,
)

TOURISM_QUARTERLY = RemoteFileMetadata(
    url=_ZENODO_URL.format(record_id="4656093", filename="tourism_quarterly_dataset.zip"),
    zip_filename="tourism_quarterly_dataset.zip",
    tsf_filename="tourism_quarterly_dataset.tsf",
    sha256="fa16da64f29aa04df447756cb65a5d58f8bc3d860fc22751c76850a3ffb048c3",
    descr="""\
Tourism Quarterly Dataset
=========================

427 quarterly tourism time series from the tourism forecasting
competition.

Source: Monash Time Series Forecasting Archive (forecastingdata.org)
License: CC BY 4.0
DOI: 10.5281/zenodo.4656093
Citation: Godahewa et al. (2021), "Monash Time Series Forecasting Archive"
""",
    frequency="3mo",
    n_series=427,
)

ELECTRICITY_DEMAND = RemoteFileMetadata(
    url=_ZENODO_URL.format(
        record_id="4659727",
        filename="australian_electricity_demand_dataset.zip",
    ),
    zip_filename="australian_electricity_demand_dataset.zip",
    tsf_filename="australian_electricity_demand_dataset.tsf",
    sha256="b1439c28a631766bd05ee327f1f430a887b9f53b38a8cd5c0769ded1fd4aaf5a",
    descr="""\
Australian Electricity Demand Dataset
=====================================

5 half-hourly time series of electricity demand from five Australian
states: New South Wales, Queensland, South Australia, Tasmania, and
Victoria.

Source: Monash Time Series Forecasting Archive (forecastingdata.org)
License: CC BY 4.0
DOI: 10.5281/zenodo.4659727
Citation: Godahewa et al. (2021), "Monash Time Series Forecasting Archive"
""",
    frequency="30m",
    n_series=5,
)

DOMINICK = RemoteFileMetadata(
    url=_ZENODO_URL.format(record_id="4654802", filename="dominick_dataset.zip"),
    zip_filename="dominick_dataset.zip",
    tsf_filename="dominick_dataset.tsf",
    sha256="3025a5e6abc152b700b13cc84d9056c99398651e508803b20f2fe76156f7bd96",
    descr="""\
Dominick Dataset
================

115704 weekly time series representing the profit of individual stock
keeping units from a retailer (Dominick's Finer Foods).

Source: Monash Time Series Forecasting Archive (forecastingdata.org)
License: CC BY 4.0
DOI: 10.5281/zenodo.4654802
Citation: Godahewa et al. (2021), "Monash Time Series Forecasting Archive"
""",
    frequency="1w",
    n_series=115704,
)

PEDESTRIAN_COUNTS = RemoteFileMetadata(
    url=_ZENODO_URL.format(record_id="4656626", filename="pedestrian_counts_dataset.zip"),
    zip_filename="pedestrian_counts_dataset.zip",
    tsf_filename="pedestrian_counts_dataset.tsf",
    sha256="6e81cb8cad43650e7e0f754a1103fc75c960e03387bae9b91146aa3241c9aa50",
    descr="""\
Melbourne Pedestrian Counts Dataset
====================================

66 hourly time series of pedestrian counts captured from sensors in
Melbourne city from May 2009 to April 2020.

Source: Monash Time Series Forecasting Archive (forecastingdata.org)
License: CC BY 4.0
DOI: 10.5281/zenodo.4656626
Citation: Godahewa et al. (2021), "Monash Time Series Forecasting Archive"
""",
    frequency="1h",
    n_series=66,
)

HOSPITAL = RemoteFileMetadata(
    url=_ZENODO_URL.format(record_id="4656014", filename="hospital_dataset.zip"),
    zip_filename="hospital_dataset.zip",
    tsf_filename="hospital_dataset.tsf",
    sha256="8fd4085efb4517508e30fc69247eee9b74b300765bc6e94482c5d250efba66e5",
    descr="""\
Hospital Dataset
================

767 monthly time series representing patient counts related to
medical products from January 2000 to December 2006.

Source: Monash Time Series Forecasting Archive (forecastingdata.org)
License: CC BY 4.0
DOI: 10.5281/zenodo.4656014
Citation: Godahewa et al. (2021), "Monash Time Series Forecasting Archive"
""",
    frequency="1mo",
    n_series=767,
)

KDD_CUP_2018 = RemoteFileMetadata(
    url=_ZENODO_URL.format(
        record_id="4656756",
        filename="kdd_cup_2018_dataset_without_missing_values.zip",
    ),
    zip_filename="kdd_cup_2018_dataset_without_missing_values.zip",
    tsf_filename="kdd_cup_2018_dataset_without_missing_values.tsf",
    sha256="2ababea948fb1179eb3c5e0bed63611251278f5e8dca30dd8e8d658652720b01",
    descr="""\
KDD Cup 2018 Air Quality Dataset
=================================

270 hourly time series representing air quality measurements from
59 monitoring stations across two cities: Beijing (35 stations) and
London (24 stations), from January 2017 to March 2018.

Each station records up to six pollutant measurements: PM2.5, PM10,
NO2, CO, O3, and SO2. All Beijing stations have all six measurements
(35 x 6 = 210 series); the remaining 60 series come from the 24 London
stations, which record a subset (on average 2-3 measurements per
station: typically PM2.5 and PM10, sometimes NO2).

This is the only multivariate panel dataset in yohou: each panel
group (station) contains multiple member columns (measurements),
making it ideal for demonstrating groupwise aggregation, multivariate
panel strategies, and composition patterns.

Source: Monash Time Series Forecasting Archive (forecastingdata.org)
License: CC BY 4.0
DOI: 10.5281/zenodo.4656756
Citation: Godahewa et al. (2021), "Monash Time Series Forecasting Archive"
""",
    frequency="1h",
    n_series=270,
)
