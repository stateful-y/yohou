# yohou.datasets

Remote time series dataset fetchers. Each function downloads data from [Monash/Zenodo](https://forecastingdata.org) (CC BY 4.0) and returns a `sklearn.utils.Bunch` with a `.frame` attribute containing a `polars.DataFrame` with a `"time"` column. Data is cached locally after the first download.

**User guide**: See the [Core Concepts](../user-guide/core-concepts.md) section for data format details.

## Dataset Fetchers

::: yohou.datasets.fetch_tourism_monthly
    options:
      show_root_heading: true
      show_source: false

::: yohou.datasets.fetch_sunspot
    options:
      show_root_heading: true
      show_source: false

::: yohou.datasets.fetch_tourism_quarterly
    options:
      show_root_heading: true
      show_source: false

::: yohou.datasets.fetch_electricity_demand
    options:
      show_root_heading: true
      show_source: false

::: yohou.datasets.fetch_dominick
    options:
      show_root_heading: true
      show_source: false

::: yohou.datasets.fetch_pedestrian_counts
    options:
      show_root_heading: true
      show_source: false

::: yohou.datasets.fetch_hospital
    options:
      show_root_heading: true
      show_source: false

## Utilities

::: yohou.datasets.get_data_home
    options:
      show_root_heading: true
      show_source: false

::: yohou.datasets.clear_data_home
    options:
      show_root_heading: true
      show_source: false
