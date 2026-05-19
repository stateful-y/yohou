"""Generate pre-processed parquet files for all datasets.

These parquets are hosted on the orphan `datasets` branch and served
via jsDelivr for WASM/Pyodide environments where Zenodo CORS blocks
direct downloads.

Usage:
    uv run python scripts/generate_dataset_parquets.py [output_dir]

Default output directory: ./dataset_parquets/
"""

from __future__ import annotations

import sys
from pathlib import Path

from yohou.datasets import (
    fetch_air_quality_classification,
    fetch_demand_classification,
    fetch_dominick,
    fetch_electricity_demand,
    fetch_hospital,
    fetch_kdd_cup,
    fetch_pedestrian_counts,
    fetch_sunspot,
    fetch_tourism_monthly,
    fetch_tourism_quarterly,
)


def main() -> None:  # noqa: D103
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset_parquets")
    output_dir.mkdir(parents=True, exist_ok=True)

    standard_datasets = {
        "sunspot": fetch_sunspot,
        "tourism_monthly": fetch_tourism_monthly,
        "tourism_quarterly": fetch_tourism_quarterly,
        "hospital": fetch_hospital,
        "electricity_demand": fetch_electricity_demand,
        "dominick": fetch_dominick,
        "pedestrian_counts": fetch_pedestrian_counts,
        "kdd_cup": fetch_kdd_cup,
    }

    for name, fetcher in standard_datasets.items():
        print(f"Generating {name}...")  # noqa: T201
        bunch = fetcher()
        path = output_dir / f"{name}.parquet"
        bunch.frame.write_parquet(path, compression="zstd", compression_level=9)
        size_kb = path.stat().st_size / 1000
        print(f"  {path} ({size_kb:.0f} KB, shape={bunch.frame.shape})")  # noqa: T201

    # Classification datasets: separate y and X_actual parquets
    print("Generating air_quality_classification...")  # noqa: T201
    aq = fetch_air_quality_classification()
    aq_y_path = output_dir / "air_quality_classification_y.parquet"
    aq_x_path = output_dir / "air_quality_classification_X.parquet"
    aq.y.write_parquet(aq_y_path, compression="zstd", compression_level=9)
    aq.X_actual.write_parquet(aq_x_path, compression="zstd", compression_level=9)
    print(f"  {aq_y_path} ({aq_y_path.stat().st_size / 1000:.0f} KB)")  # noqa: T201
    print(f"  {aq_x_path} ({aq_x_path.stat().st_size / 1000:.0f} KB)")  # noqa: T201

    print("Generating demand_classification...")  # noqa: T201
    dc = fetch_demand_classification()
    dc_y_path = output_dir / "demand_classification_y.parquet"
    dc_x_path = output_dir / "demand_classification_X.parquet"
    dc.y.write_parquet(dc_y_path, compression="zstd", compression_level=9)
    dc.X_actual.write_parquet(dc_x_path, compression="zstd", compression_level=9)
    print(f"  {dc_y_path} ({dc_y_path.stat().st_size / 1000:.0f} KB)")  # noqa: T201
    print(f"  {dc_x_path} ({dc_x_path.stat().st_size / 1000:.0f} KB)")  # noqa: T201

    total_bytes = sum(f.stat().st_size for f in output_dir.glob("*.parquet"))
    print(f"\nTotal: {total_bytes / 1_000_000:.1f} MB across {len(list(output_dir.glob('*.parquet')))} files")  # noqa: T201


if __name__ == "__main__":
    main()
