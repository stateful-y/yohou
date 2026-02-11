---
description: "Guide for adding new datasets to Yohou. Use when contributing datasets to the built-in collection."
---

# Creating New Datasets

## Overview

Yohou includes bundled datasets for examples, testing, and benchmarking. All datasets:
- Stored as **Parquet files** with **zstd compression**
- Tracked via **Git LFS** (large file storage)
- Have **loader functions** in `src/yohou/datasets/loaders.py`
- Follow **consistent schema conventions**
- Include **comprehensive metadata** in docstrings

**Location**: `src/yohou/datasets/data/`

---

## Current Dataset Inventory

**10 bundled datasets** (as of current version):

| Dataset | Type | Frequency | Size | Use Case |
|---------|------|-----------|------|----------|
| `air_passengers` | Univariate | Monthly | 144 obs | Trend + seasonality |
| `sunspots` | Univariate | Monthly | 2820 obs | Cyclic patterns |
| `m4_monthly` | Panel | Monthly | 50 series | Panel forecasting |
| `m4_quarterly` | Panel | Quarterly | 50 series | Quarterly seasonality |
| `m4_hourly` | Panel | Hourly | 50 series | High-frequency panel |
| `australian_tourism` | Panel | Quarterly | 304 series | Tourism demand |
| `vic_electricity` | Univariate | 30-min | 52,416 obs | High-frequency energy |
| `store_sales` | Panel | Daily | 54,000 rows | Retail forecasting |
| `walmart_sales` | Panel | Weekly | 421,570 rows | Large-scale retail |
| `ett_m1` | Multivariate | 15-min | 69,680 obs | Energy transformer temperature |

---

## Adding a New Dataset (Step-by-Step)

### 1. Obtain and Prepare Data

```python
import polars as pl

# Load raw data (from CSV, API, etc.)
df = pl.read_csv("raw_data.csv")

# CRITICAL: Ensure 'time' column with datetime type
df = df.with_columns(
    pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("time")
)

# For panel data: Use "__" separator (prefix__suffix)
# Example: "sales__store_1", "sales__store_2"
df = df.rename({
    "store_1_sales": "sales__store_1",
    "store_2_sales": "sales__store_2",
})

# Drop unnecessary columns (metadata, duplicates, etc.)
df = df.select(["time", "sales__store_1", "sales__store_2"])

# Sort by time
df = df.sort("time")

# Validate time consistency
from yohou.utils.validation import check_interval_consistency
interval = check_interval_consistency(df)
print(f"Detected interval: {interval}")
```

### 2. Export to Parquet with Compression

```python
# Export with zstd compression (high compression, fast decompression)
df.write_parquet(
    "src/yohou/datasets/data/my_dataset.parquet",
    compression="zstd",  # REQUIRED
    compression_level=9,  # Optional: higher = smaller file
)
```

### 3. Add to Git LFS Tracking

```bash
# Initialize Git LFS (if not already done)
git lfs install

# Track parquet files (should already be configured in .gitattributes)
# Verify with:
cat .gitattributes | grep parquet
# Should show: *.parquet filter=lfs diff=lfs merge=lfs -text

# Stage and commit (Git LFS handles automatically)
git add src/yohou/datasets/data/my_dataset.parquet
git commit -m "Add my_dataset to bundled datasets"

# Verify it's tracked by LFS
git lfs ls-files
# Should list my_dataset.parquet
```

### 4. Create Loader Function

Add to `src/yohou/datasets/loaders.py`:

```python
def load_my_dataset() -> pl.DataFrame:
    """
    Load My Dataset.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns "time" (Datetime) and target/feature columns.

    Notes
    -----
    - **name**: My Dataset
    - **description**: Brief description of the dataset (1-2 sentences)
    - **type**: univariate | multivariate | panel
    - **frequency**: hourly | daily | weekly | monthly | quarterly | yearly | 15-min | etc.
    - **source_url**: https://original-source.com/data
    - **format**: parquet
    - **characteristics**: Key features (length, seasonality, trend, noise level, etc.)
    - **use_cases**: Specific forecasting scenarios this dataset is good for
    - **example_notebooks**: `examples/my_dataset.py` - Description of example (if exists)
    """
    return _load_dataset("my_dataset")
```

### 5. Add to Exports

Update `src/yohou/datasets/__init__.py`:

```python
from .loaders import (
    # ... existing loaders
    load_my_dataset,
)

__all__ = [
    # ... existing loaders
    "load_my_dataset",
]
```

### 6. Create Example Notebook (Optional but Recommended)

Create `examples/my_dataset.py` as a Marimo notebook:

```python
import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def __():
    import polars as pl
    from yohou.datasets import load_my_dataset
    from yohou.plotting import plot_timeseries

    df = load_my_dataset()
    return df, load_my_dataset, plot_timeseries


@app.cell
def __(df, plot_timeseries):
    """Display the dataset."""
    fig = plot_timeseries(df)
    fig
    return


if __name__ == "__main__":
    app.run()
```

---

## Dataset Schema Conventions

### Univariate Time Series

```python
df = pl.DataFrame({
    "time": [...],     # Datetime column (REQUIRED)
    "value": [...],    # Target variable (any name, typically singular)
})
```

### Multivariate Time Series

```python
df = pl.DataFrame({
    "time": [...],          # Datetime column (REQUIRED)
    "target": [...],        # Primary target
    "feature_1": [...],     # Exogenous feature 1
    "feature_2": [...],     # Exogenous feature 2
})
```

### Panel Data

```python
df = pl.DataFrame({
    "time": [...],                  # Datetime column (REQUIRED)
    "sales__store_1": [...],        # Prefix: sales, Suffix: store_1
    "sales__store_2": [...],        # Prefix: sales, Suffix: store_2
    "sales__store_3": [...],
})
```

**Critical**: Panel data uses `"prefix__suffix"` naming with **double underscore** separator.

---

## File Size Guidelines

**Git LFS quota**: GitHub free accounts have 1 GB storage, 1 GB bandwidth/month.

**Compression guidelines**:
- **Small** (<1 MB compressed): No concern
- **Medium** (1-10 MB compressed): Acceptable for examples
- **Large** (10-100 MB compressed): Consider sampling or external hosting
- **Very large** (>100 MB compressed): External hosting required (Zenodo, Hugging Face, etc.)

**Check compressed size**:
```bash
ls -lh src/yohou/datasets/data/my_dataset.parquet
```

**Reduce size if needed**:
```python
# Sample data
df_sample = df.sample(fraction=0.1, seed=42)

# Or: Subset time range
df_subset = df.filter(pl.col("time") >= pl.datetime(2020, 1, 1))

# Or: Select fewer series (panel data)
df_subset = df.select(["time", "sales__store_1", "sales__store_2"])
```

---

## Data Sources and Licensing

**Preferred sources**:
- Public domain datasets (no license restrictions)
- CC0 / CC-BY licensed data
- Aggregated/anonymized data from public APIs

**License metadata**: Include in loader docstring:
```python
Notes
-----
- **license**: CC0 | CC-BY 4.0 | Public Domain | etc.
- **citation**: "Author (Year). Dataset Name. URL."
```

**Avoid**:
- Proprietary datasets without permission
- Personal/sensitive data
- Data with restrictive licenses

---

## Testing Patterns

Add test in `tests/datasets/test_loaders.py`:

```python
import polars as pl
from yohou.datasets import load_my_dataset


def test_load_my_dataset():
    """Test my_dataset loads correctly."""
    df = load_my_dataset()

    # Check type
    assert isinstance(df, pl.DataFrame)

    # Check required columns
    assert "time" in df.columns
    assert df["time"].dtype == pl.Datetime

    # Check not empty
    assert len(df) > 0

    # Check expected schema (adjust based on dataset)
    expected_cols = ["time", "value"]  # Or panel columns
    assert set(df.columns) == set(expected_cols)

    # Check time ordering
    assert df["time"].is_sorted()

    # Check no missing values in time column
    assert df["time"].null_count() == 0
```

---

## Checklist Before Committing

1. Data exported to `src/yohou/datasets/data/my_dataset.parquet` with zstd compression
2. Git LFS tracking verified (`git lfs ls-files` shows dataset)
3. Loader function added to `loaders.py` with complete docstring metadata
4. Exports updated in `__init__.py`
5. Test added to `tests/datasets/test_loaders.py`
6. Example notebook created (optional but recommended)
7. File size checked (<10 MB compressed preferred)
8. License/citation included in docstring
9. `uvx nox -s fix` passes
10. `uv run pytest tests/datasets/ -v` passes

---

## Common Pitfalls

- **Missing time column**: All datasets MUST have `"time"` column
- **Wrong time type**: Must be `pl.Datetime`, not `pl.Date` or string
- **Unsorted time**: Always sort by `"time"` column
- **Not using Git LFS**: Parquet files must be tracked by LFS (check `.gitattributes`)
- **No compression**: Always use `compression="zstd"` when writing parquet
- **Panel data naming**: Must use double underscore `"prefix__suffix"`, not single `"prefix_suffix"`
- **Missing metadata**: Loader docstring MUST include all Notes fields
- **File too large**: Check compressed size, consider sampling if >10 MB
- **No test**: Every loader needs a test in `tests/datasets/test_loaders.py`

---

## Real-World Examples to Study

**Loader functions**:
- `src/yohou/datasets/loaders.py` - All 10 built-in datasets

**Example notebooks**:
- `examples/air_passengers.py` - Comprehensive univariate example
- `examples/m4_monthly.py` - Panel data with facets
- `examples/vic_electricity.py` - High-frequency univariate

**Tests**:
- `tests/datasets/test_loaders.py` - Dataset loading tests

**Dataset overview**:
- `examples/datasets_overview.py` - Marimo notebook showcasing all datasets

---

## External Dataset Hosting (For Large Datasets)

If dataset is >100 MB compressed, host externally:

### Zenodo (Recommended)

```python
def load_my_large_dataset() -> pl.DataFrame:
    """
    Load My Large Dataset.

    Notes
    -----
    - **name**: My Large Dataset
    - **description**: Large-scale dataset description
    - **type**: panel
    - **frequency**: daily
    - **source_url**: https://zenodo.org/records/1234567
    - **download_url**: https://zenodo.org/records/1234567/files/data.parquet
    - **format**: parquet (hosted externally)
    - **size**: 250 MB compressed
    - **characteristics**: Very large panel data with 1000+ series
    - **use_cases**: Scalability testing, large-scale forecasting
    """
    import urllib.request
    from pathlib import Path

    cache_dir = Path.home() / ".yohou" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / "my_large_dataset.parquet"

    if not cache_file.exists():
        print("Downloading dataset (this may take a while)...")
        url = "https://zenodo.org/records/1234567/files/data.parquet"
        urllib.request.urlretrieve(url, cache_file)
        print(f"Downloaded to {cache_file}")

    return pl.read_parquet(cache_file)
```

### Hugging Face Datasets

```python
def load_my_hf_dataset() -> pl.DataFrame:
    """Load dataset from Hugging Face."""
    from datasets import load_dataset

    ds = load_dataset("username/dataset-name", split="train")
    df = pl.from_pandas(ds.to_pandas())
    return df
```

---

## Dataset Contribution Workflow

1. **Open issue**: Propose dataset with description, source, use case
2. **Prepare data**: Follow schema conventions, compress with zstd
3. **Create PR**: Include loader, test, example notebook
4. **Review**: Maintainers verify size, licensing, quality
5. **Merge**: Dataset becomes part of official collection
