from datetime import datetime

import numpy as np
import polars as pl
import pytest

from yohou.pipeline import ColumnTransformer, FeatureUnion, Pipeline
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
        "a": range(1, 1 + length),
        "b": np.random.rand(length),
    }
)


@pytest.mark.skip(reason="Pipeline inverse_transform needs redesign")
@pytest.mark.parametrize(
    "transformer",
    [
        Pipeline(
            [
                ("1", SeasonalDifferencing(6)),
                ("5", SeasonalLogDifferencing(5)),
            ]
        )
    ],
)
def test_pipeline_identity(transformer):
    params = transformer.get_params()
    transformer.set_params(**params)
    X_t = transformer.fit_transform(X)
    observation_horizon = transformer.observation_horizon

    X_it = transformer.inverse_transform(X_t=X_t, X_p=X[:observation_horizon])

    pl.testing.assert_frame_equal(X_it, X[observation_horizon:])


@pytest.mark.parametrize(
    "transformer",
    [
        FeatureUnion(
            [
                ("1", SeasonalDifferencing(6)),
                ("5", SeasonalLogDifferencing(5)),
            ]
        )
    ],
)
def test_feature_union(transformer):
    params = transformer.get_params()
    transformer.set_params(**params)
    X_t = transformer.fit_transform(X)
    observation_horizon = transformer.observation_horizon

    pl.testing.assert_frame_equal(X_t[["time"]], X[observation_horizon:][["time"]])


@pytest.mark.parametrize(
    "transformer",
    [
        ColumnTransformer(
            [
                ("1", SeasonalDifferencing(6), "a"),
                ("5", SeasonalLogDifferencing(5), "b"),
            ]
        )
    ],
)
def test_column_transformer(transformer):
    params = transformer.get_params()
    transformer.set_params(**params)
    X_t = transformer.fit_transform(X)
    observation_horizon = transformer.observation_horizon

    pl.testing.assert_frame_equal(X_t[["time"]], X[observation_horizon:][["time"]])
