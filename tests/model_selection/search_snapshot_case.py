"""A fixed ``validation=None`` search whose results are pinned by a JSON snapshot.

The snapshot was recorded before the ``validation="cv"`` mode was added to the
search, so comparing against it shows the default mode is unchanged.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from yohou.metrics import MeanAbsoluteError, RootMeanSquaredError
from yohou.model_selection import ExpandingWindowSplitter, GridSearchCV
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer

TIMING_KEYS = ("mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time")


def build_search() -> tuple[GridSearchCV, pl.DataFrame]:
    """Return the pinned search and its data."""
    rng = np.random.default_rng(7)
    n = 160
    times = pl.datetime_range(datetime(2022, 1, 1), datetime(2022, 1, 1) + timedelta(hours=n - 1), "1h", eager=True)
    y = pl.DataFrame({"time": times, "value": np.sin(np.arange(n) / 5.0) * 4 + rng.normal(0, 0.5, n)})
    search = GridSearchCV(
        forecaster=PointReductionForecaster(estimator=Ridge(), actual_transformer=LagTransformer(lag=[1, 2])),
        param_grid={"estimator__alpha": [0.1, 10.0], "reduction_strategy": ["multi-output", "direct"]},
        scoring={"mae": MeanAbsoluteError(), "rmse": RootMeanSquaredError()},
        refit="mae",
        cv=ExpandingWindowSplitter(n_splits=3, test_size=12),
        return_train_score=True,
    )
    return search, y


def _jsonable(value):
    if isinstance(value, np.ma.MaskedArray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float | int | str | bool) or value is None:
        return value
    return repr(value)


def summarize(search: GridSearchCV) -> dict:
    """Return everything the snapshot pins, in JSON-compatible form."""
    results = {k: _jsonable(v) for k, v in search.cv_results_.items() if k not in TIMING_KEYS}
    predictions = search.best_forecaster_.predict()
    return {
        "cv_results_keys": sorted(search.cv_results_),
        "cv_results": results,
        "best_params": _jsonable(search.best_params_),
        "best_score": float(search.best_score_),
        "best_index": int(search.best_index_),
        "predictions": {
            "time": [t.isoformat() for t in predictions["time"].to_list()],
            "value": predictions["value"].to_list(),
        },
    }
