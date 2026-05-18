---
template: api-submodule.html
---

# yohou.datasets

Remote time series dataset fetchers and related utilities.

**User guide**: See the [Core Concepts](../explanation/core-concepts.md) section for data format details.

### Loaders

Each function downloads data from [Monash/Zenodo](https://forecastingdata.org) (CC BY 4.0) and returns a `sklearn.utils.Bunch` with a `.frame` attribute containing a `polars.DataFrame` with a `"time"` column. Data is cached locally after the first download.

| Name | Description |
| --- | --- |
| [`fetch_dominick`](generated/yohou.datasets._fetchers.fetch_dominick.md) | Fetch the Dominick dataset from Monash/Zenodo. |
| [`fetch_electricity_demand`](generated/yohou.datasets._fetchers.fetch_electricity_demand.md) | Fetch the Australian Electricity Demand dataset from Monash/Zenodo. |
| [`fetch_hospital`](generated/yohou.datasets._fetchers.fetch_hospital.md) | Fetch the Hospital dataset from Monash/Zenodo. |
| [`fetch_kdd_cup`](generated/yohou.datasets._fetchers.fetch_kdd_cup.md) | Fetch the KDD Cup 2018 air quality dataset from Monash/Zenodo. |
| [`fetch_pedestrian_counts`](generated/yohou.datasets._fetchers.fetch_pedestrian_counts.md) | Fetch the Melbourne Pedestrian Counts dataset from Monash/Zenodo. |
| [`fetch_sunspot`](generated/yohou.datasets._fetchers.fetch_sunspot.md) | Fetch the Sunspot dataset (without missing values) from Monash/Zenodo. |
| [`fetch_tourism_monthly`](generated/yohou.datasets._fetchers.fetch_tourism_monthly.md) | Fetch the Tourism Monthly dataset from Monash/Zenodo. |
| [`fetch_tourism_quarterly`](generated/yohou.datasets._fetchers.fetch_tourism_quarterly.md) | Fetch the Tourism Quarterly dataset from Monash/Zenodo. |
| [`fetch_air_quality_classification`](generated/yohou.datasets._fetchers.fetch_air_quality_classification.md) | Fetch a categorical air quality dataset derived from KDD Cup 2018. |
| [`fetch_demand_classification`](generated/yohou.datasets._fetchers.fetch_demand_classification.md) | Fetch a categorical electricity demand dataset from Monash/Zenodo. |

### Utilities

| Name | Description |
| --- | --- |
| [`clear_data_home`](generated/yohou.datasets._fetchers.clear_data_home.md) | Delete all the content of the data home cache. |
| [`get_data_home`](generated/yohou.datasets._fetchers.get_data_home.md) | Return the path of the yohou data directory. |
| [`parse_tsf`](generated/yohou.datasets._tsf_parser.parse_tsf.md) | Parse a Monash `.tsf` file into a wide polars DataFrame. |

### Synthetic generators

Parameterized generators that create synthetic time series with all three exogenous feature types (`X_actual`, `X_future`, `X_forecast`). No download required.

| Name | Description |
| --- | --- |
| [`make_exogenous_regression`](generated/yohou.datasets._generators.make_exogenous_regression.md) | Generate synthetic regression data with exogenous features. |
| [`make_exogenous_classification`](generated/yohou.datasets._generators.make_exogenous_classification.md) | Generate synthetic classification data with exogenous features. |
