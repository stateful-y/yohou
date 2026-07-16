# Data Catalog

Bundled datasets available in `yohou.datasets`. Each dataset is downloaded on first use and cached locally as Parquet files. The cache directory defaults to `~/yohou_data/` and can be changed via the `data_home` parameter or the `YOHOU_DATA` environment variable.

## Quick Reference

| Function | Shape | Frequency | Series | Observations | Domain |
|---|---|---|---|---|---|
| [`fetch_sunspot()`](/pages/api/generated/yohou.datasets.fetch_sunspot/) | Univariate | Daily | 1 | 73,924 | Astronomy |
| [`fetch_tourism_monthly()`](/pages/api/generated/yohou.datasets.fetch_tourism_monthly/) | Panel | Monthly | 366 | ~80/series | Tourism |
| [`fetch_tourism_quarterly()`](/pages/api/generated/yohou.datasets.fetch_tourism_quarterly/) | Panel | Quarterly | 427 | ~32/series | Tourism |
| [`fetch_hospital()`](/pages/api/generated/yohou.datasets.fetch_hospital/) | Panel | Monthly | 767 | 84/series | Healthcare |
| [`fetch_pedestrian_counts()`](/pages/api/generated/yohou.datasets.fetch_pedestrian_counts/) | Panel | Hourly | 66 | ~10,000/series | Transport |
| [`fetch_kdd_cup()`](/pages/api/generated/yohou.datasets.fetch_kdd_cup/) | Panel | Hourly | 270 (59 stations × up to 6 pollutants) | ~8,760/series | Environment |
| [`fetch_electricity_demand()`](/pages/api/generated/yohou.datasets.fetch_electricity_demand/) | Panel | Half-hourly | 5 | ~46,000/series | Energy |
| [`fetch_dominick()`](/pages/api/generated/yohou.datasets.fetch_dominick/) | Panel | Weekly | 115,704 | ~412/series | Retail |
| [`fetch_air_quality_classification()`](/pages/api/generated/yohou.datasets.fetch_air_quality_classification/) | Classification | Hourly | 1 | ~8,000 | Environment |
| [`fetch_demand_classification()`](/pages/api/generated/yohou.datasets.fetch_demand_classification/) | Classification | Half-hourly | 1 | ~46,000 | Energy |

## Return Type

All numeric fetch functions return a `sklearn.utils.Bunch` (a dict subclass with attribute access) containing:

| Attribute | Type | Description |
|---|---|---|
| `frame` | `pl.DataFrame` | Time column (`time`, Datetime) followed by one or more target columns (Float64) |
| `feature_names` | `list[str]` | Names of all non-time columns |
| `DESCR` | `str` | Full dataset description |
| `frequency` | `str` | Polars duration string (`"1d"`, `"1mo"`, `"3mo"`, `"1h"`, `"30m"`, `"1w"`) |
| `n_series` | `int` | Number of series loaded |
| `filename` | `str` | Path to the cached Parquet file |

Classification fetch functions return a `Bunch` with different attributes:

| Attribute | Type | Description |
|---|---|---|
| `y` | `pl.DataFrame` | Time column + target column (Utf8 class labels) |
| `X_actual` | `pl.DataFrame` | Time column + numeric feature columns (Float64) |
| `feature_names` | `list[str]` | Names of feature columns in `X_actual` |
| `target_names` | `list[str]` | Names of target columns in `y` |
| `classes` | `list[str]` | Unique class labels |
| `DESCR` | `str` | Full dataset description |

## Common Parameters

All fetch functions accept the following keyword-only parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data_home` | `str`, `PathLike`, or `None` | `None` | Cache directory. `None` resolves to `~/yohou_data/` or the `YOHOU_DATA` environment variable |
| `download_if_missing` | `bool` | `True` | If `False`, raises `OSError` when data is not cached locally |
| `n_retries` | `int` | `3` | Number of download retry attempts |
| `delay` | `float` | `1.0` | Seconds between retries |

Panel datasets also accept:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_series` | `int` or `None` | varies | Maximum number of series to load. `None` loads all series |

## Numeric Datasets

### `fetch_sunspot()`

[API Reference](/pages/api/generated/yohou.datasets.fetch_sunspot/){ .md-button .md-button--primary .md-button--small }

Daily sunspot numbers, 1818 to 2020. Single univariate series with 73,924 observations.

| Property | Value |
|---|---|
| Frequency | `"1d"` (Daily) |
| Time column | `time` (Datetime) |
| Target column | `sunspot_number` (Float64) |
| Series count | 1 |
| Shape | Univariate |

```python
from yohou.datasets import fetch_sunspot

bunch = fetch_sunspot()
bunch.frame.shape  # (73924, 2)
```

### `fetch_tourism_monthly()`

[API Reference](/pages/api/generated/yohou.datasets.fetch_tourism_monthly/){ .md-button .md-button--primary .md-button--small }

Monthly Australian tourism visitor counts, 1992 to 2011. Up to 366 panel series, ~80 observations each.

| Property | Value |
|---|---|
| Frequency | `"1mo"` (Monthly) |
| Time column | `time` (Datetime) |
| Target columns | `T1__tourists`, `T2__tourists`, ... (Float64) |
| Default `n_series` | `None` (all 366) |
| Shape | Panel |

```python
from yohou.datasets import fetch_tourism_monthly

bunch = fetch_tourism_monthly(n_series=10)
bunch.n_series  # 10
```

### `fetch_tourism_quarterly()`

[API Reference](/pages/api/generated/yohou.datasets.fetch_tourism_quarterly/){ .md-button .md-button--primary .md-button--small }

Quarterly Australian tourism visitor counts, 1992 to 2011. Up to 427 panel series, ~32 observations each.

| Property | Value |
|---|---|
| Frequency | `"3mo"` (Quarterly) |
| Time column | `time` (Datetime) |
| Target columns | `T1__tourists`, `T2__tourists`, ... (Float64) |
| Default `n_series` | `None` (all 427) |
| Shape | Panel |

```python
from yohou.datasets import fetch_tourism_quarterly

bunch = fetch_tourism_quarterly(n_series=5)
bunch.frame.columns[:3]  # ['time', 'T1__tourists', 'T2__tourists']
```

### `fetch_hospital()`

[API Reference](/pages/api/generated/yohou.datasets.fetch_hospital/){ .md-button .md-button--primary .md-button--small }

Monthly hospital patient counts, 2000 to 2006. Up to 767 panel series, 84 observations each.

| Property | Value |
|---|---|
| Frequency | `"1mo"` (Monthly) |
| Time column | `time` (Datetime) |
| Target columns | `T1__patients`, `T2__patients`, ... (Float64) |
| Default `n_series` | `None` (all 767) |
| Shape | Panel |

```python
from yohou.datasets import fetch_hospital

bunch = fetch_hospital(n_series=20)
bunch.feature_names[:2]  # ['T1__patients', 'T2__patients']
```

### `fetch_pedestrian_counts()`

[API Reference](/pages/api/generated/yohou.datasets.fetch_pedestrian_counts/){ .md-button .md-button--primary .md-button--small }

Hourly Melbourne pedestrian sensor counts, 2009 to 2020. Up to 66 sensors, ~10,000 observations each.

| Property | Value |
|---|---|
| Frequency | `"1h"` (Hourly) |
| Time column | `time` (Datetime) |
| Target columns | `T1__count`, `T2__count`, ... (Float64) |
| Default `n_series` | `20` |
| Shape | Panel |

```python
from yohou.datasets import fetch_pedestrian_counts

bunch = fetch_pedestrian_counts()          # default 20 sensors
bunch_all = fetch_pedestrian_counts(n_series=None)  # all 66 sensors
```

### `fetch_kdd_cup()`

[API Reference](/pages/api/generated/yohou.datasets.fetch_kdd_cup/){ .md-button .md-button--primary .md-button--small }

Hourly air quality measurements from 59 monitoring stations (Beijing and London), 2017 to 2018. Each station reports up to 6 pollutants (PM2.5, PM10, NO2, CO, O3, SO2), so the total loaded series is at most `n_groups × 6`; London stations carry fewer measurements, so the actual count when loading all groups is 270. Column names follow the `station__measurement` format.

| Property | Value |
|---|---|
| Frequency | `"1h"` (Hourly) |
| Time column | `time` (Datetime) |
| Target columns | `beijing_dongsi_aq__pm2.5`, `beijing_dongsi_aq__pm10`, ... (Float64) |
| Default `n_groups` | `5` (30 series) |
| Shape | Panel |

This function accepts `n_groups` instead of `n_series`. Each group is one station with all 6 pollutant measurements.

```python
from yohou.datasets import fetch_kdd_cup

bunch = fetch_kdd_cup(n_groups=2)
bunch.n_series  # 12 (2 stations × 6 pollutants)
```

### `fetch_electricity_demand()`

[API Reference](/pages/api/generated/yohou.datasets.fetch_electricity_demand/){ .md-button .md-button--primary .md-button--small }

Half-hourly electricity demand for 5 Australian states, 2008 to 2015. Fixed panel dataset (`group__column` naming) with ~46,000 observations per series.

| Property | Value |
|---|---|
| Frequency | `"30m"` (Half-hourly) |
| Time column | `time` (Datetime) |
| Target columns | `nsw__demand`, `qun__demand`, `sa__demand`, `tas__demand`, `vic__demand` (Float64) |
| Series count | 5 (fixed) |
| Shape | Panel |

```python
from yohou.datasets import fetch_electricity_demand

bunch = fetch_electricity_demand()
bunch.feature_names  # ['nsw__demand', 'qun__demand', 'sa__demand', 'tas__demand', 'vic__demand']
```

### `fetch_dominick()`

[API Reference](/pages/api/generated/yohou.datasets.fetch_dominick/){ .md-button .md-button--primary .md-button--small }

Weekly store-level profit data from Dominick's Finer Foods, 1989 to 1997. Up to 115,704 panel series, ~412 observations each. Default `n_series=50` limits memory usage. Pass `n_series=None` to load all series (several GB).

| Property | Value |
|---|---|
| Frequency | `"1w"` (Weekly) |
| Time column | `time` (Datetime) |
| Target columns | `T1__profit`, `T2__profit`, ... (Float64) |
| Default `n_series` | `50` |
| Shape | Panel |

```python
from yohou.datasets import fetch_dominick

bunch = fetch_dominick(n_series=100)
bunch.n_series  # 100
```

## Classification Datasets

Classification fetch functions return `y` (target labels) and `X_actual` (numeric features) instead of `frame`. Both are `pl.DataFrame` instances with a shared `time` column.

### `fetch_air_quality_classification()`

[API Reference](/pages/api/generated/yohou.datasets.fetch_air_quality_classification/){ .md-button .md-button--primary .md-button--small }

Hourly air quality classification derived from `fetch_kdd_cup(n_groups=1)`. PM2.5 values are binned into 4 WHO categories using thresholds at 15, 35, and 75 µg/m³. Rows with null PM2.5 are dropped. ~8,000 observations.

| Property | Value |
|---|---|
| Frequency | Hourly |
| Target column (`y`) | `air_quality` (Utf8: `"good"`, `"moderate"`, `"unhealthy"`, `"hazardous"`) |
| Feature columns (`X_actual`) | `pm10`, `no2`, `co`, `o3`, `so2` (Float64) |
| Shape | Multivariate |

```python
from yohou.datasets import fetch_air_quality_classification

bunch = fetch_air_quality_classification()
bunch.classes  # ['good', 'hazardous', 'moderate', 'unhealthy']
bunch.y.head()
```

### `fetch_demand_classification()`

[API Reference](/pages/api/generated/yohou.datasets.fetch_demand_classification/){ .md-button .md-button--primary .md-button--small }

Half-hourly electricity demand classification derived from `fetch_electricity_demand()`. Victoria's demand is binned into 3 tercile-based levels. Features are the remaining 4 states. ~46,000 observations.

| Property | Value |
|---|---|
| Frequency | Half-hourly |
| Target column (`y`) | `demand_level` (Utf8: `"low"`, `"medium"`, `"high"`) |
| Feature columns (`X_actual`) | `nsw__demand`, `qun__demand`, `sa__demand`, `tas__demand` (Float64) |
| Shape | Multivariate |

```python
from yohou.datasets import fetch_demand_classification

bunch = fetch_demand_classification()
bunch.classes  # ['high', 'low', 'medium']
bunch.X_actual.columns  # ['time', 'nsw__demand', 'qun__demand', 'sa__demand', 'tas__demand']
```

## Synthetic Generators

Generator functions create synthetic datasets locally (no download required). Useful for testing and examples involving exogenous features.

### `make_exogenous_regression()`

[API Reference](/pages/api/generated/yohou.datasets.make_exogenous_regression/){ .md-button .md-button--primary .md-button--small }

Generates synthetic hourly electricity prices driven by temperature, holiday, and weather forecast features.

```python
def make_exogenous_regression(
    *,
    n_samples: int = 200,
    forecasting_horizon: int = 6,
    noise: float = 0.1,
    forecast_bias: float = 0.5,
    random_state: int = 42,
) -> Bunch
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_samples` | `int` | `200` | Number of time steps |
| `forecasting_horizon` | `int` | `6` | Length of the forecast period |
| `noise` | `float` | `0.1` | Scale of random noise |
| `forecast_bias` | `float` | `0.5` | Bias added to forecast features |
| `random_state` | `int` | `42` | Random seed for reproducibility |

Returns a `Bunch` with:

| Attribute | Description |
|---|---|
| `y` | `pl.DataFrame` with `time` and `price` (Float64) |
| `X_actual` | `pl.DataFrame` with `time` and `temperature` (Float64) |
| `X_future` | `pl.DataFrame` with `time` and `is_holiday` (Float64) |
| `X_forecast` | `pl.DataFrame` with `vintage_time`, `time`, and `wx_temp` (Float64) |
| `frame` | `pl.DataFrame` joining `y`, `X_actual`, and `X_future` on `time` |
| `feature_names` | `["temperature", "is_holiday", "wx_temp"]` |
| `target_names` | `["price"]` |
| `frequency` | `"1h"` |
| `DESCR` | Full dataset description (`str`) |

```python
from yohou.datasets import make_exogenous_regression

bunch = make_exogenous_regression(n_samples=500)
bunch.y.shape      # (500, 2)
bunch.X_actual.shape  # (500, 2)
```

### `make_exogenous_classification()`

[API Reference](/pages/api/generated/yohou.datasets.make_exogenous_classification/){ .md-button .md-button--primary .md-button--small }

Generates synthetic hourly air quality labels driven by pollutant level, weekend indicator, and pollutant forecast features.

```python
def make_exogenous_classification(
    *,
    n_samples: int = 400,
    forecasting_horizon: int = 6,
    noise: float = 0.1,
    forecast_bias: float = 0.3,
    random_state: int = 42,
) -> Bunch
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_samples` | `int` | `400` | Number of time steps |
| `forecasting_horizon` | `int` | `6` | Length of the forecast period |
| `noise` | `float` | `0.1` | Scale of random noise |
| `forecast_bias` | `float` | `0.3` | Bias added to forecast features |
| `random_state` | `int` | `42` | Random seed for reproducibility |

Returns a `Bunch` with:

| Attribute | Description |
|---|---|
| `y` | `pl.DataFrame` with `time` and `air_quality` (Utf8: `"good"`, `"moderate"`, `"poor"`) |
| `X_actual` | `pl.DataFrame` with `time` and `pollutant` (Float64) |
| `X_future` | `pl.DataFrame` with `time` and `is_weekend` (Float64) |
| `X_forecast` | `pl.DataFrame` with `vintage_time`, `time`, and `pollutant_forecast` (Float64) |
| `frame` | `pl.DataFrame` joining `y`, `X_actual`, and `X_future` on `time` |
| `feature_names` | `["pollutant", "is_weekend", "pollutant_forecast"]` |
| `target_names` | `["air_quality"]` |
| `classes` | `["good", "moderate", "poor"]` |
| `frequency` | `"1h"` |
| `DESCR` | Full dataset description (`str`) |

```python
from yohou.datasets import make_exogenous_classification

bunch = make_exogenous_classification(n_samples=300)
bunch.classes  # ['good', 'moderate', 'poor']
```

## Utility Functions

### `get_data_home()`

[API Reference](/pages/api/generated/yohou.datasets.get_data_home/){ .md-button .md-button--primary .md-button--small }

```python
from yohou.datasets import get_data_home

get_data_home()           # ~/yohou_data/ (default)
get_data_home("/tmp/data")  # /tmp/data (custom path, created if missing)
```

Returns the path to the data cache directory. If `data_home` is `None`, resolves to the `YOHOU_DATA` environment variable or `~/yohou_data/`. Creates the directory if it does not exist.

### `clear_data_home()`

[API Reference](/pages/api/generated/yohou.datasets.clear_data_home/){ .md-button .md-button--primary .md-button--small }

```python
from yohou.datasets import clear_data_home

clear_data_home()  # deletes all files in ~/yohou_data/
```

Deletes all cached data files in the data home directory.

<!-- GALLERY:section:data-catalog -->

## See Also

- [Extensions](extensions.md): base classes and extension packages for custom components
- [Tags](tags.md): tag system for declaring component capabilities
