"""Time series forecasting datasets and loaders."""

from yohou.datasets.loaders import (
    load_air_passengers,
    load_australian_tourism,
    load_ett_m1,
    load_store_sales,
    load_sunspots,
    load_vic_electricity,
    load_walmart_sales,
)

__all__ = [
    "load_air_passengers",
    "load_sunspots",
    "load_australian_tourism",
    "load_vic_electricity",
    "load_store_sales",
    "load_walmart_sales",
    "load_ett_m1",
]
