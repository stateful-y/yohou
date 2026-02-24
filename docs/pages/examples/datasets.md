# Datasets

Each dataset notebook fetches data from [Monash/Zenodo](https://forecastingdata.org) using `yohou.datasets` functions. See the [API reference](../api/datasets.md) for full function signatures.

### Tourism Monthly ([View](/examples/datasets/tourism_monthly/) | [Editable](/examples/datasets/tourism_monthly/edit/))

**Trend and Seasonality Analysis** — `fetch_tourism_monthly`

Monthly tourism series from the Monash forecasting competition. Explores the first series T1 with `plot_time_series`, `plot_rolling_statistics`, and `plot_seasonality`.

### Tourism Quarterly ([View](/examples/datasets/australian_tourism/) | [Editable](/examples/datasets/australian_tourism/edit/))

**Panel Data Exploration** — `fetch_tourism_quarterly`

Quarterly tourism trips from the Monash forecasting competition (427 series). Explores 8 series with `inspect_locality`, `plot_time_series`, `plot_seasonality`, and `plot_boxplot`.

### Tourism Quarterly Forecasting ([View](/examples/datasets/australian_tourism_forecasting/) | [Editable](/examples/datasets/australian_tourism_forecasting/edit/))

**Panel Forecasting Workflow** — `fetch_tourism_quarterly`

End-to-end panel forecasting on tourism quarterly data: fit, predict, observe-predict, per-group scoring, rolling evaluation, and selective group observation.

### Sunspots ([View](/examples/datasets/sunspots/) | [Editable](/examples/datasets/sunspots/edit/))

**Cyclic Pattern Analysis** — `fetch_sunspot`

Daily sunspot numbers (1818–2020), resampled to monthly. Demonstrates 11-year solar cycles with `plot_time_series`, `plot_rolling_statistics`, `plot_autocorrelation`, and `plot_spectrum`.

### Electricity Demand ([View](/examples/datasets/vic_electricity/) | [Editable](/examples/datasets/vic_electricity/edit/))

**Multi-State Panel Analysis** — `fetch_electricity_demand`

Half-hourly electricity demand from 5 Australian states. Demonstrates `inspect_locality`, `plot_time_series`, `plot_cross_correlation`, `plot_seasonality`, and `plot_rolling_statistics`.

### Dominick Store Sales ([View](/examples/datasets/store_sales/) | [Editable](/examples/datasets/store_sales/edit/))

**Weekly Retail Panel Analysis** — `fetch_dominick`

Weekly retail profit for SKUs from Dominick's Finer Foods in panel format. Explores 9 of 50 series with `inspect_locality`, `plot_time_series`, `plot_boxplot`, and `plot_seasonality`.

### Pedestrian Counts ([View](/examples/datasets/pedestrian_counts/) | [Editable](/examples/datasets/pedestrian_counts/edit/))

**Sensor-Level Panel Analysis** — `fetch_pedestrian_counts`

Hourly pedestrian counts from Melbourne sensors (20 series). Explores 6 series with `inspect_locality`, `plot_time_series`, `plot_boxplot`, and `plot_seasonality`.

### Pedestrian Counts Forecasting ([View](/examples/datasets/pedestrian_counts_forecasting/) | [Editable](/examples/datasets/pedestrian_counts_forecasting/edit/))

**Panel Forecasting Workflow** — `fetch_pedestrian_counts`

Panel forecasting on hourly pedestrian data: per-sensor evaluation, rolling observe-predict, and selective group observation.

### Hospital ([View](/examples/datasets/hospital/) | [Editable](/examples/datasets/hospital/edit/))

**Monthly Patient Counts Panel** — `fetch_hospital`

Monthly patient counts for medical products (767 series, 2000–2006). Explores 6 series with `plot_time_series`, `plot_cross_correlation`, and `plot_seasonality`.

### Hospital Multivariate ([View](/examples/datasets/hospital_multivariate/) | [Editable](/examples/datasets/hospital_multivariate/edit/))

**Multivariate Forecasting** — `fetch_hospital`

Multivariate forecasting using `ForecastedFeatureForecaster`: target-only baseline vs. known exogenous features vs. forecasted features.
