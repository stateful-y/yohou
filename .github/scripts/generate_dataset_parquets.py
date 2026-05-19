"""Generate pre-processed dataset files for WASM/Pyodide loading.

These binary files are hosted on the orphan ``datasets`` branch and served
via jsDelivr for WASM/Pyodide environments where Zenodo CORS blocks
direct downloads.

The files use Polars binary serialization (``DataFrame.serialize``) rather
than Parquet because marimo patches all Polars I/O functions
(``read_parquet``, ``read_ipc``, ``read_csv``, etc.) in WASM to require
PyArrow, which is not available. ``DataFrame.deserialize`` is not patched
and works directly.

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
        path = output_dir / f"{name}.bin"
        path.write_bytes(bunch.frame.serialize(format="binary"))
        size_kb = path.stat().st_size / 1000
        print(f"  {path} ({size_kb:.0f} KB, shape={bunch.frame.shape})")  # noqa: T201

    # Classification datasets: separate y and X_actual files
    print("Generating air_quality_classification...")  # noqa: T201
    aq = fetch_air_quality_classification()
    aq_y_path = output_dir / "air_quality_classification_y.bin"
    aq_x_path = output_dir / "air_quality_classification_X.bin"
    aq_y_path.write_bytes(aq.y.serialize(format="binary"))
    aq_x_path.write_bytes(aq.X_actual.serialize(format="binary"))
    print(f"  {aq_y_path} ({aq_y_path.stat().st_size / 1000:.0f} KB)")  # noqa: T201
    print(f"  {aq_x_path} ({aq_x_path.stat().st_size / 1000:.0f} KB)")  # noqa: T201

    print("Generating demand_classification...")  # noqa: T201
    dc = fetch_demand_classification()
    dc_y_path = output_dir / "demand_classification_y.bin"
    dc_x_path = output_dir / "demand_classification_X.bin"
    dc_y_path.write_bytes(dc.y.serialize(format="binary"))
    dc_x_path.write_bytes(dc.X_actual.serialize(format="binary"))
    print(f"  {dc_y_path} ({dc_y_path.stat().st_size / 1000:.0f} KB)")  # noqa: T201
    print(f"  {dc_x_path} ({dc_x_path.stat().st_size / 1000:.0f} KB)")  # noqa: T201

    total_bytes = sum(f.stat().st_size for f in output_dir.glob("*.bin"))
    print(f"\nTotal: {total_bytes / 1_000_000:.1f} MB across {len(list(output_dir.glob('*.bin')))} files")  # noqa: T201


if __name__ == "__main__":
    main()
