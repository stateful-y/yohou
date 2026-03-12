"""Synthetic dataset generators for categorical time series."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
from sklearn.utils import Bunch


def make_weather_classification(
    *,
    length: int = 365,
    start: datetime | None = None,
    interval: str = "1d",
    seed: int = 42,
) -> Bunch:
    """Generate a synthetic categorical weather classification dataset.

    Creates a deterministic weather time series with three target classes
    (``"sunny"``, ``"rainy"``, ``"cloudy"``) driven by seasonal temperature
    and humidity features.  Suitable for testing and demonstrating
    class-probability forecasters.

    Parameters
    ----------
    length : int, default=365
        Number of time steps to generate.
    start : datetime or None, default=None
        Start datetime.  If ``None``, defaults to ``2020-01-01``.
    interval : str, default="1d"
        Time interval between observations (polars duration string).
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    sklearn.utils.Bunch
        Dictionary-like object with keys:

        - ``y`` : pl.DataFrame with ``"time"`` and ``"weather"`` columns.
        - ``X`` : pl.DataFrame with ``"time"``, ``"temperature"``, and
          ``"humidity"`` columns.
        - ``feature_names`` : list of str, feature column names.
        - ``target_names`` : list of str, target column names.
        - ``classes`` : list of str, unique class labels (sorted).
        - ``DESCR`` : str, human-readable dataset description.

    See Also
    --------
    `fetch_tourism_monthly` : Real-world tourism forecasting dataset.

    Examples
    --------
    >>> from yohou.datasets import make_weather_classification
    >>> data = make_weather_classification(length=100, seed=0)
    >>> data.y.columns
    ['time', 'weather']
    >>> sorted(data.classes)
    ['cloudy', 'rainy', 'sunny']

    """
    if start is None:
        start = datetime(2020, 1, 1)

    rng = np.random.default_rng(seed)

    times = pl.datetime_range(
        start=start,
        end=pl.Series([start]).cast(pl.Datetime).item(),
        interval=interval,
        eager=True,
    )
    if len(times) < length:
        times = pl.datetime_range(
            start=start,
            interval=interval,
            eager=True,
            end=pl.Series([start]).cast(pl.Datetime).item() + pl.duration(days=length * 2),
        )
    times = times[:length]

    t = np.arange(length, dtype=np.float64)
    temperature = 15.0 + 10.0 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 2, length)
    humidity = 60.0 - 15.0 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 5, length)
    humidity = np.clip(humidity, 10, 100)

    sunny_score = temperature / 30.0 - humidity / 100.0 + rng.normal(0, 0.3, length)
    rainy_score = humidity / 80.0 - temperature / 40.0 + rng.normal(0, 0.3, length)
    cloudy_score = rng.normal(0, 0.3, length)

    scores = np.stack([sunny_score, rainy_score, cloudy_score], axis=1)
    labels_idx = scores.argmax(axis=1)
    label_map = {0: "sunny", 1: "rainy", 2: "cloudy"}
    weather = [label_map[i] for i in labels_idx]

    y = pl.DataFrame({"time": times, "weather": weather})
    X = pl.DataFrame({
        "time": times,
        "temperature": temperature,
        "humidity": humidity,
    })

    return Bunch(
        y=y,
        X=X,
        feature_names=["temperature", "humidity"],
        target_names=["weather"],
        classes=sorted(label_map.values()),
        DESCR=(
            "Synthetic weather classification dataset with 3 classes "
            "(sunny, rainy, cloudy) driven by seasonal temperature and "
            "humidity features."
        ),
    )
