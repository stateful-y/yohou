"""Pre-download and cache all yohou datasets as parquet files.

Run this script *before* the documentation build so that
``marimo export html`` subprocesses read from the parquet cache
instead of downloading and parsing large TSF archives.

Usage
-----
::

    uv run python docs/precache_datasets.py

"""

from __future__ import annotations

from yohou.datasets import (
    fetch_air_quality_classification,
    fetch_demand_classification,
    fetch_dominick,
    fetch_electricity_demand,
    fetch_hospital,
    fetch_pedestrian_counts,
    fetch_sunspot,
    fetch_tourism_monthly,
    fetch_tourism_quarterly,
)

_FETCHERS = [
    ("tourism_monthly", fetch_tourism_monthly),
    ("tourism_quarterly", fetch_tourism_quarterly),
    ("sunspot", fetch_sunspot),
    ("hospital", fetch_hospital),
    ("electricity_demand", fetch_electricity_demand),
    ("pedestrian_counts", fetch_pedestrian_counts),
    ("dominick", fetch_dominick),
    ("air_quality_classification", fetch_air_quality_classification),
    ("demand_classification", fetch_demand_classification),
]

if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_one(name, fetcher):
        bunch = fetcher()
        if hasattr(bunch, "n_series"):
            return name, f"{bunch.n_series} series, {len(bunch.frame)} rows, cached to {bunch.filename}"
        else:
            return name, f"{len(bunch.y)} rows, {len(bunch.classes)} classes"

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_fetch_one, name, fn): name for name, fn in _FETCHERS}
        for future in as_completed(futures):
            name = futures[future]
            try:
                name, info = future.result()
                print(f"[precache] {name}: {info}", flush=True)
            except Exception as exc:
                print(f"[precache] {name}: FAILED ({exc})", flush=True)

    print("[precache] All datasets cached.")
