# Dataset Acquisition and Preprocessing

## Overview

The `yohou.datasets` module provides a collection of standard time series datasets for testing, benchmarking, and tutorials. These datasets are bundled with the package as **Parquet** files using `zstd` compression to ensure:
- **Strict Typing**: Dates are strictly typed as `Date` or `Datetime`, avoiding parsing ambiguity.
- **Efficiency**: Fast loading with minimal overhead.
- **Portability**: No runtime dependency on external URLs.

## Dataset Inventory

| Dataset | Type | Frequency | Rows | Columns | Description | Source |
|---------|------|-----------|------|---------|-------------|--------|
| **Air Passengers** | Univariate | Monthly | 144 | `time` (Date), `y` (Int64) | Airline passengers (1949-1960) | [J. Brownlee](https://github.com/jbrownlee/Datasets/blob/master/airline-passengers.csv) |
| **Sunspots** | Univariate | Monthly | 2820 | `time` (Date), `y` (Float64) | Monthly sunspot numbers (1749-1983) | [J. Brownlee](https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv) |
| **Australian Tourism** | Panel/Hierarch | Monthly | 24,320 | `time`, `y`, `Region`, `State`, `Purpose` | Tourism demand by region | [Skforecast](https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/australia_tourism.csv) |
| **Victoria Electricity** | Multivariate | 30-min | 52,608 | `time`, `y`, `Temperature`, `Holiday` | Electricity demand (Victoria, AU) | [Skforecast](https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/vic_electricity.csv) |
| **Store Sales** | Panel/Exog | Daily | 913,000 | `time`, `store`, `item`, `y` | Retail sales (10 stores, 50 items) | [Skforecast](https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/store_sales.csv) |
| **Walmart Sales** | Panel | Daily | 1,000 | `time`, `y`, `Branch`, `City`, ... | Supermarket sales data | [Selva86](https://raw.githubusercontent.com/selva86/datasets/master/supermarket_sales.csv) |
| **ETTm1** | Multivariate | 15-min | 69,680 | `time`, `y` (OT), `HUFL`, `HULL`, ... | Electricity Transformer Temp | [Skforecast](https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/ETTm1.csv) |
| **M4 Monthly** | Panel | Monthly | ~22k | `unique_id`, `time`*, `y` | Subset of M4 Competition (Monthly) | [Zenodo](https://zenodo.org/records/4656480) |
| **M4 Quarterly** | Panel | Quarterly | ~3k | `unique_id`, `time`*, `y` | Subset of M4 Competition (Quarterly) | [Zenodo](https://zenodo.org/records/4656410) |
| **M4 Hourly** | Panel | Hourly | ~35k | `unique_id`, `time`*, `y` | Subset of M4 Competition (Hourly) | [Zenodo](https://zenodo.org/records/4656589) |

*\*Times for M4 datasets are synthetic, starting from 2000-01-01.*

## Processing Methodology

The datasets were processed using a reproducible script (`scripts/download_datasets.py`) that performs the following transformations:

### 1. Standardization
- **Time Column**: Renamed to `"time"`.
    - Converted to `pl.Date` for low-frequency data (Daily, Monthly).
    - Converted to `pl.Datetime` for high-frequency data (Hourly, 15-min).
- **Target Column**: Renamed to `"y"`.
    - Strictly cast to `pl.Float64` or `pl.Int64`.
    - M4 data (originally string numbers) was cleaned and cast.

### 2. M4 Dataset Handling
M4 datasets are provided in "Wide" format (one row per series, columns V2-V2500 for steps).
- **Transformation**: Unpivoted (melted) to "Long" format.
- **Time Generation**: Since M4 data is anonymized without dates, synthetic datetime indices were generated starting from `2000-01-01` based on the series frequency.
- **Subsetting**: Only the first 50 series of each subset were processed to keep package size manageable.

### 3. Localization
- Timezones were removed (`dt.replace_time_zone(None)`) to ensure simple datetime handling in polars.
- "Victoria Electricity" originally had an ISO format `YYYY-MM-DDTHH:MM:SSZ`.

### 4. Storage
- **Format**: Parquet.
- **Compression**: `zstd` (provides high compression ratio).
- **Location**: `src/yohou/datasets/data/`.
- **Git LFS**: All `*.parquet` files are tracked via Git LFS to keep the repository lightweight.

## Adding New Datasets

To add a new dataset:
1.  Identify a stable public URL (Raw CSV preferred).
2.  Add an entry to the `DATASETS` config in `scripts/download_datasets.py` (if available) or create a processing function that:
    - Loads raw data with `polars`.
    - Renames time -> "time", target -> "y".
    - Enforces schema.
    - Saves as `.parquet` in `src/yohou/datasets/data/`.
3.  Add a loader function in `src/yohou/datasets/loaders.py` using `_load_dataset`.
4.  Export the loader in `src/yohou/datasets/__init__.py`.
5.  Track the new file: `git lfs track "src/yohou/datasets/data/new_dataset.parquet"`.
