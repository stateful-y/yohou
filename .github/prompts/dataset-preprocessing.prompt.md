---
description: "Dataset inventory and preprocessing methodology for yohou.datasets module. Use when adding new datasets or understanding bundled data."
---

# Dataset Acquisition and Preprocessing

## Dataset Inventory

| Dataset | Type | Frequency | Key Columns | Source |
|---------|------|-----------|-------------|--------|
| Air Passengers | Univariate | Monthly | `time` (Date), `y` (Int64) | J. Brownlee |
| Sunspots | Univariate | Monthly | `time` (Date), `y` (Float64) | J. Brownlee |
| Australian Tourism | Panel | Monthly | `time`, `y`, `Region`, `State`, `Purpose` | Skforecast |
| Victoria Electricity | Multivariate | 30-min | `time`, `y`, `Temperature`, `Holiday` | Skforecast |
| Store Sales | Panel | Daily | `time`, `store`, `item`, `y` | Skforecast |
| Walmart Sales | Panel | Daily | `time`, `y`, `Branch`, `City`, ... | Selva86 |
| ETTm1 | Multivariate | 15-min | `time`, `y` (OT), `HUFL`, `HULL`, ... | Skforecast |
| M4 Monthly | Panel | Monthly | `unique_id`, `time`*, `y` | Zenodo |
| M4 Quarterly | Panel | Quarterly | `unique_id`, `time`*, `y` | Zenodo |
| M4 Hourly | Panel | Hourly | `unique_id`, `time`*, `y` | Zenodo |

*M4 times are synthetic (starting 2000-01-01).*

## Processing Standards

1. **Time column**: Renamed to `"time"`. `pl.Date` for low-frequency, `pl.Datetime` for high-frequency
2. **Target column**: Renamed to `"y"`. Cast to `pl.Float64` or `pl.Int64`
3. **M4 handling**: Unpivoted wide→long, synthetic datetime indices, first 50 series per subset
4. **Timezones removed**: `dt.replace_time_zone(None)`
5. **Storage**: Parquet + zstd compression in `src/yohou/datasets/data/`, tracked via Git LFS

## Adding New Datasets

1. Identify stable public URL (raw CSV preferred)
2. Create processing function: load → rename time/y → enforce schema → save `.parquet`
3. Add loader in `src/yohou/datasets/loaders.py` using `_load_dataset()`
4. Export in `src/yohou/datasets/__init__.py`
5. Track: `git lfs track "src/yohou/datasets/data/new_dataset.parquet"`
