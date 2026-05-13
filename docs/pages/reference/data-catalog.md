# Data Catalog

Bundled datasets available in `yohou.datasets`. Each dataset is downloaded on first use and cached locally. The interactive notebooks below show how to load, inspect, and visualize each one.

| Dataset | Fetch function | Frequency | Series |
|---|---|---|---|
| Sunspots | `fetch_sunspot()` | Daily | 1 univariate |
| Tourism Monthly | `fetch_tourism_monthly()` | Monthly | 366 |
| Tourism Quarterly | `fetch_tourism_quarterly()` | Quarterly | 427 |
| Hospital | `fetch_hospital()` | Monthly | 767 |
| Melbourne Pedestrian Counts | `fetch_pedestrian_counts()` | Hourly | 66 sensors |
| KDD Cup Air Quality | `fetch_kdd_cup()` | Hourly | 59 stations, 6 pollutants |
| Australian Electricity Demand | `fetch_electricity_demand()` | Half-hourly | 5 states |
| Dominick Store Sales | `fetch_dominick()` | Weekly | 115 704 (50 by default) |
| Air Quality (classification) | `fetch_air_quality_classification()` | Hourly | 1 station |
| Electricity Demand (classification) | `fetch_demand_classification()` | Half-hourly | 5 states |

<!-- GALLERY:section:data-catalog -->
