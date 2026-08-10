"""Contract checks for ``BaseStepTransformer`` subclasses.

These mirror the forecast-frame checks in `yohou.testing.forecast_transformer`
but for transformers over the derived step frame, which carries one ``"time"``
index plus ``{base}_step_1..H`` columns. Each check takes an *unfitted* step
transformer and a sample step frame, and asserts one clause of the
step-transformer contract.

The memory-API checks are the sharpest ones here. A step frame has the same index
as a single-axis frame, so nothing about its shape stops a stateful transformer
being used in a step slot; only the kind tag and these guards do.
"""

import polars as pl

from yohou.base.step_transformer import BaseStepTransformer, _is_step_indexed

__all__ = [
    "check_horizon_agnostic_output_naming",
    "check_memory_api_refused",
    "check_min_steps_contract",
    "check_missing_time_index_raises",
    "check_step_kind_tag",
    "check_time_preserved",
    "check_transform_is_stateless",
]


def check_step_kind_tag(transformer: BaseStepTransformer, X: pl.DataFrame) -> None:
    """Assert the transformer reports ``kind == "step"``, readable before fit."""
    tags = transformer.__sklearn_tags__()
    assert tags.transformer_tags is not None
    assert tags.transformer_tags.kind == "step", f"expected kind='step', got {tags.transformer_tags.kind!r}"


def check_time_preserved(transformer: BaseStepTransformer, X: pl.DataFrame) -> None:
    """Assert ``transform`` preserves the ``"time"`` index and its row count."""
    out = transformer.fit_transform(X)
    assert "time" in out.columns, "transform must preserve the 'time' index column"
    assert len(out) == len(X), f"transform changed the row count: {len(X)} -> {len(out)}"
    assert out["time"].to_list() == X["time"].to_list(), "transform must not reorder or alter 'time'"


def check_missing_time_index_raises(transformer: BaseStepTransformer, X: pl.DataFrame) -> None:
    """Assert a frame missing the ``"time"`` index column is rejected with a ValueError."""
    transformer.fit(X)
    try:
        transformer.transform(X.drop("time"))
    except ValueError:
        return
    raise AssertionError("transform did not raise ValueError on a frame missing 'time'")


def check_transform_is_stateless(transformer: BaseStepTransformer, X: pl.DataFrame) -> None:
    """Assert repeated transforms are identical, with an unrelated call interleaved.

    A step transformer that accumulated anything between calls would drift here,
    which matters because the step frame is re-derived from scratch at every
    observe and predict rather than extended.
    """
    transformer.fit(X)
    first = transformer.transform(X)
    transformer.transform(X.head(1))
    second = transformer.transform(X)
    assert first.equals(second), "transform is not a pure function of its input"


def check_memory_api_refused(transformer: BaseStepTransformer, X: pl.DataFrame) -> None:
    """Assert all four memory methods refuse a step-kind transformer.

    Covers ``observe_transform`` and ``rewind_transform`` as well as
    ``observe``/``rewind``: the composers override the paired ``*_transform``
    methods separately, and guarding only the first pair is a gap this library has
    already had to close once for forecast-kind transformers.
    """
    transformer.fit(X)
    for method in ("observe", "rewind", "observe_transform", "rewind_transform"):
        fn = getattr(transformer, method, None)
        if fn is None:
            continue
        try:
            fn(X)
        except ValueError:
            continue
        raise AssertionError(
            f"{type(transformer).__name__}.{method}() must raise ValueError on a step-kind transformer"
        )


def check_min_steps_contract(transformer: BaseStepTransformer, X: pl.DataFrame) -> None:
    """Assert ``min_steps`` is a positive int and readable before fit.

    Deliberately unlike ``min_vintage_rows``, which requires fitting: a
    forecaster asserts the horizon against this *before* fitting the slot, so
    that a transformer needing more step columns than the horizon provides is
    reported in terms of the horizon rather than by whatever its inner estimator
    says about matrix shapes.
    """
    name = type(transformer).__name__
    before = transformer.min_steps
    assert isinstance(before, int) and not isinstance(before, bool), (
        f"{name}.min_steps must be an int, got {type(before).__name__}"
    )
    assert before >= 1, f"{name}.min_steps must be at least 1, got {before}"

    transformer.fit(X)
    assert transformer.min_steps == before, f"{name}.min_steps changed across fit: {before} -> {transformer.min_steps}"


def check_horizon_agnostic_output_naming(transformer: BaseStepTransformer, X: pl.DataFrame) -> None:
    r"""Assert every output column is either step-indexed or named ``{base}_step_{name}``.

    The ``_step_(\\d+)$`` partition is what lets ``step_feature_alignment`` filter
    per-step columns while leaving whole-block summaries alone, so an output name
    outside the convention would be silently dropped from every direct estimator.
    """
    out = transformer.fit_transform(X)
    for column in out.columns:
        if column == "time" or _is_step_indexed(column):
            continue
        assert "_step_" in column, (
            f"output column {column!r} is neither step-indexed nor named '{{base}}_step_{{name}}'; "
            f"step_feature_alignment could not classify it"
        )


#: All step-transformer contract checks, for iterating in a parametrized test.
STEP_TRANSFORMER_CHECKS = [
    check_step_kind_tag,
    check_time_preserved,
    check_missing_time_index_raises,
    check_transform_is_stateless,
    check_memory_api_refused,
    check_min_steps_contract,
    check_horizon_agnostic_output_naming,
]
