"""Unit tests for the failure branches of the weighter check functions."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from yohou.testing.weighter import (
    check_weighter_default_constructible,
    check_weighter_resolved_array_validation,
    check_weighter_tags_accessible_before_fit,
)
from yohou.weighting.weighters import BaseWeighter, ExponentialDecayWeighter


class _NeedsArgWeighter(BaseWeighter):
    """Weighter that is not default-constructible."""

    def __init__(self, scale):
        self.scale = scale

    def compute_weights(self, key: pl.Series, group_name: str | None = None) -> pl.Series:
        return pl.Series([1.0] * len(key), dtype=pl.Float64).alias("weight")


class _BadTagsWeighter(BaseWeighter):
    """Weighter whose tag access raises."""

    def compute_weights(self, key: pl.Series, group_name: str | None = None) -> pl.Series:
        return pl.Series([1.0] * len(key), dtype=pl.Float64).alias("weight")

    def __sklearn_tags__(self):
        raise RuntimeError("boom")


def test_default_constructible_flags_required_args():
    with pytest.raises(AssertionError, match="not default-constructible"):
        check_weighter_default_constructible(_NeedsArgWeighter(scale="position"))


def test_tags_accessible_flags_raising_tags():
    with pytest.raises(AssertionError, match="__sklearn_tags__"):
        check_weighter_tags_accessible_before_fit(_BadTagsWeighter())


def test_resolved_array_validation_passes_for_concrete_weighter():
    """check_weighter_resolved_array_validation rejects all four invalid resolutions.

    The check builds its own stub weighters resolving to NaN/negative/inf/all-zero
    arrays and asserts each is rejected by the resolution path. Passing a concrete
    weighter exercises every sub-case end-to-end; the check returns None when all
    four invalid arrays are correctly rejected and raises AssertionError otherwise.
    """
    key = pl.Series("time", [datetime(2024, 1, d) for d in range(1, 6)])
    assert check_weighter_resolved_array_validation(ExponentialDecayWeighter(), key) is None
