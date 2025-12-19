from datetime import datetime

import numpy as np
import polars as pl
import pytest

from yohou.preprocessing.stationarization import (
    SeasonalDifferencing,
    SeasonalLogDifferencing,
)

length = 52

X = pl.DataFrame(
    {
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 0, 0, length - 1),
            interval="1s",
            eager=True,
        ),
        "a": np.random.rand(length),
        "b": np.random.rand(length),
    }
)


@pytest.mark.parametrize(
    "transformer",
    [
        SeasonalDifferencing(1),
        SeasonalDifferencing(5),
        SeasonalLogDifferencing(1, 2),
        SeasonalLogDifferencing(3, 1),
    ],
)
def test_identity(transformer):
    X_t = transformer.fit_transform(X)
    memory_size = transformer.memory_size

    X_it = transformer.inverse_transform(X_t=X_t, X_p=X[:memory_size])

    pl.testing.assert_frame_equal(X_it, X[memory_size:])
