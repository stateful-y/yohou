from datetime import datetime

import optuna
import polars as pl

from yohou.metrics import MAE
from yohou.model_selection import SearchCV
from yohou.point_forecaster import SeasonalNaive

length = 52

y = pl.DataFrame(
    {
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 0, 0, length - 1),
            interval="1s",
            eager=True,
        ),
        "a": range(length),
        "b": range(10, length + 10),
    }
)

X_ante = pl.DataFrame(
    {
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 0, 0, length - 1),
            interval="1s",
            eager=True,
        ),
        "c": range(length),
        "d": range(10, length + 10),
        "e": range(20, length + 20),
    }
)


def test_search():
    search = SearchCV(
        forecaster=SeasonalNaive(),
        param_distributions={"seasonality": optuna.distributions.IntDistribution(1, 20)},
        scoring=MAE(),
        error_score="raise",
        n_warmup_trials=5,
        n_trials=10,
        n_jobs=2,
    )

    search.fit(y, X_ante, forecasting_horizon=1)
