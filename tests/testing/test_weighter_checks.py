"""Unit tests for the failure branches of the weighter check functions."""

from __future__ import annotations

import polars as pl
import pytest

from yohou.testing.weighter import (
    check_weighter_default_constructible,
    check_weighter_tags_accessible_before_fit,
)
from yohou.weighting.weighters import BaseWeighter


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
