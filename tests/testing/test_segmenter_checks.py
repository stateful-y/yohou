"""Unit tests for the failure branches of the segmenter check functions."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from yohou.segmenting import BaseSegmenter, CalendarSegmenter
from yohou.testing import _yield_yohou_segmenter_checks
from yohou.testing.segmenter import (
    check_segmenter_compute_labels_alignment,
    check_segmenter_default_constructible,
    check_segmenter_deterministic,
    check_segmenter_fit_noop_returns_self,
    check_segmenter_row_wise,
    check_segmenter_tags_accessible_before_fit,
    check_segmenter_tags_static_after_fit,
)


@pytest.fixture
def time() -> pl.Series:
    """Twelve hourly timestamps."""
    return pl.Series("time", [datetime(2024, 1, 1) + timedelta(hours=h) for h in range(12)])


class _ShortSegmenter(BaseSegmenter):
    """Segmenter that drops the last label."""

    def compute_labels(self, time: pl.Series) -> pl.Series:
        return pl.Series("label", [1] * (len(time) - 1))


class _MutatingFitSegmenter(BaseSegmenter):
    """Segmenter whose fit changes a parameter."""

    def __init__(self, level: int = 0):
        self.level = level

    def fit(self, time=None, **params):
        self.level += 1
        return self

    def compute_labels(self, time: pl.Series) -> pl.Series:
        return pl.Series("label", [self.level] * len(time))


class _CountingSegmenter(BaseSegmenter):
    """Segmenter whose labels change on every call."""

    def __init__(self):
        self._calls = 0

    def compute_labels(self, time: pl.Series) -> pl.Series:
        self._calls += 1
        return pl.Series("label", [self._calls] * len(time))


class _PositionalSegmenter(BaseSegmenter):
    """Segmenter that labels rows by their position in the batch."""

    def compute_labels(self, time: pl.Series) -> pl.Series:
        return pl.Series("label", list(range(len(time))))


class _NeedsArgSegmenter(BaseSegmenter):
    """Segmenter that is not default-constructible."""

    def __init__(self, feature):
        self.feature = feature

    def compute_labels(self, time: pl.Series) -> pl.Series:
        return pl.Series("label", [1] * len(time))


class _BadTagsSegmenter(BaseSegmenter):
    """Segmenter whose tag access raises."""

    def compute_labels(self, time: pl.Series) -> pl.Series:
        return pl.Series("label", [1] * len(time))

    def __sklearn_tags__(self):
        raise RuntimeError("boom")


class _ShiftingTagsSegmenter(BaseSegmenter):
    """Segmenter whose tags change after fit."""

    def __init__(self):
        self._fitted = False

    def fit(self, time=None, **params):
        self._fitted = True
        return self

    def compute_labels(self, time: pl.Series) -> pl.Series:
        return pl.Series("label", [1] * len(time))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = self._fitted
        return tags


def test_alignment_flags_short_output(time):
    with pytest.raises(AssertionError, match="expected 12"):
        check_segmenter_compute_labels_alignment(_ShortSegmenter(), time)


def test_fit_noop_flags_mutation(time):
    with pytest.raises(AssertionError, match="changed parameter values"):
        check_segmenter_fit_noop_returns_self(_MutatingFitSegmenter(), time)


def test_deterministic_flags_changing_labels(time):
    with pytest.raises(AssertionError, match="not deterministic"):
        check_segmenter_deterministic(_CountingSegmenter(), time)


def test_row_wise_flags_positional_labels(time):
    with pytest.raises(AssertionError, match="depends on the other rows"):
        check_segmenter_row_wise(_PositionalSegmenter(), time)


def test_default_constructible_flags_required_args():
    with pytest.raises(AssertionError, match="not default-constructible"):
        check_segmenter_default_constructible(_NeedsArgSegmenter(feature="hour"))


def test_tags_accessible_flags_raising_tags():
    with pytest.raises(AssertionError, match="raised RuntimeError"):
        check_segmenter_tags_accessible_before_fit(_BadTagsSegmenter())


def test_tags_static_flags_changing_tags(time):
    with pytest.raises(AssertionError, match="tags changed after fit"):
        check_segmenter_tags_static_after_fit(_ShiftingTagsSegmenter(), time)


def test_harness_yields_runnable_checks(time):
    """Test the generator yields named checks that pass on a working segmenter."""
    seg = CalendarSegmenter("hour", values=range(3, 9))
    checks = list(_yield_yohou_segmenter_checks(seg, time))
    names = {name for name, _, _ in checks}
    assert "check_segmenter_row_wise" in names
    for _, func, kwargs in checks:
        func(seg, **kwargs)
