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
]

if __name__ == "__main__":
    for name, fetcher in _FETCHERS:
        print(f"[precache] Fetching {name} ...", flush=True)
        bunch = fetcher()
        print(f"[precache]   -> {bunch.n_series} series, {len(bunch.frame)} rows, cached to {bunch.filename}")
    print("[precache] All datasets cached.")
