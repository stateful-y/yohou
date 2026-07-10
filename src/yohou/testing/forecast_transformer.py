"""Contract checks for ``BaseForecastTransformer`` subclasses.

These mirror the single-axis checks in :mod:`yohou.testing.transformer` but for
transformers over ``X_forecast`` frames (two time axes: ``"vintage_time"`` and
``"time"``). Each check takes an *unfitted* forecast transformer and a sample
``X_forecast`` frame with at least two vintages, and asserts one clause of the
forecast-transformer contract.
"""

import polars as pl

from yohou.base.forecast_transformer import BaseForecastTransformer
from yohou.base.transformer import BaseActualTransformer

__all__ = [
    "check_forecast_kind_tag",
    "check_missing_vintage_index_raises",
    "check_not_an_actual_transformer",
    "check_single_and_empty_vintage_handled",
    "check_vintage_time_preserved",
]


def check_forecast_kind_tag(transformer: BaseForecastTransformer, X: pl.DataFrame) -> None:
    """Assert the transformer reports ``kind == "forecast"``, readable before fit."""
    tags = transformer.__sklearn_tags__()
    assert tags.transformer_tags is not None
    assert tags.transformer_tags.kind == "forecast", f"expected kind='forecast', got {tags.transformer_tags.kind!r}"


def check_not_an_actual_transformer(transformer: BaseForecastTransformer, X: pl.DataFrame) -> None:
    """Assert the transformer is not a ``BaseActualTransformer`` (so actual-only slots reject it)."""
    assert not isinstance(transformer, BaseActualTransformer)
    assert isinstance(transformer, BaseForecastTransformer)


def check_vintage_time_preserved(transformer: BaseForecastTransformer, X: pl.DataFrame) -> None:
    """Assert ``transform`` preserves the vintage/time index and every input vintage."""
    out = transformer.fit_transform(X)
    assert "vintage_time" in out.columns
    assert "time" in out.columns
    assert set(out["vintage_time"].unique().to_list()) == set(X["vintage_time"].unique().to_list())


def check_missing_vintage_index_raises(transformer: BaseForecastTransformer, X: pl.DataFrame) -> None:
    """Assert a frame missing the ``vintage_time`` index column is rejected with a ValueError."""
    transformer.fit(X)
    try:
        transformer.transform(X.drop("vintage_time"))
    except ValueError:
        return
    raise AssertionError("transform did not raise ValueError on a frame missing 'vintage_time'")


def check_single_and_empty_vintage_handled(transformer: BaseForecastTransformer, X: pl.DataFrame) -> None:
    """Assert a single-vintage frame and an empty frame both transform without error."""
    transformer.fit(X)

    first_vintage = X["vintage_time"][0]
    single = X.filter(pl.col("vintage_time") == first_vintage)
    single_out = transformer.transform(single)
    assert single_out["vintage_time"].n_unique() == 1

    empty_out = transformer.transform(X.clear())
    assert len(empty_out) == 0
    assert "vintage_time" in empty_out.columns and "time" in empty_out.columns


#: All forecast-transformer contract checks, for iterating in a parametrized test.
FORECAST_TRANSFORMER_CHECKS = [
    check_forecast_kind_tag,
    check_not_an_actual_transformer,
    check_vintage_time_preserved,
    check_missing_vintage_index_raises,
    check_single_and_empty_vintage_handled,
]
