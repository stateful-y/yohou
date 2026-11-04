"""Tests for the intersection, union and complement segmenters."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from yohou.segmenting import CalendarSegmenter, ComplementSegmenter, IntersectionSegmenter, UnionSegmenter


@pytest.fixture
def hourly_week() -> pl.Series:
    """Hourly timestamps from Monday 2024-01-01 00:00 to Sunday 23:00."""
    return pl.Series("time", [datetime(2024, 1, 1) + timedelta(hours=h) for h in range(7 * 24)])


def _business_hours() -> IntersectionSegmenter:
    return IntersectionSegmenter([
        ("days", CalendarSegmenter("day_of_week", values=[1, 2, 3, 4, 5])),
        ("hours", CalendarSegmenter("hour", values=list(range(8, 19)))),
    ])


def _morning_or_weekend(morning=(7, 8, 9)) -> UnionSegmenter:
    return UnionSegmenter([
        ("morning", CalendarSegmenter("hour", values=list(morning))),
        ("weekend", CalendarSegmenter("day_of_week", values=[6, 7])),
    ])


def _label_at(segmenter, when: datetime):
    return segmenter.compute_labels(pl.Series("time", [when])).to_list()[0]


class TestIntersection:
    """Tests for IntersectionSegmenter."""

    def test_weekdays_during_business_hours(self, hourly_week):
        """Test only weekday rows from 08:00 to 18:00 are inside."""
        labels = _business_hours().compute_labels(hourly_week)
        inside = [t for t, lab in zip(hourly_week.to_list(), labels.to_list(), strict=True) if lab is not None]
        assert len(inside) == 5 * 11
        assert all(t.weekday() < 5 and 8 <= t.hour <= 18 for t in inside)
        assert labels.is_not_null().sum() == 5 * 11

    def test_struct_label(self):
        """Test Monday 09:00 carries one field per component."""
        assert _label_at(_business_hours(), datetime(2024, 1, 1, 9)) == {"days": 1, "hours": 9}

    def test_outside_row_is_null_not_struct(self):
        """Test a row failing one component is a real null."""
        labels = _business_hours().compute_labels(pl.Series("time", [datetime(2024, 1, 6, 9)]))
        assert labels.is_null().to_list() == [True]

    def test_nested_set_params(self):
        """Test component parameters are addressable by name."""
        seg = _business_hours()
        seg.set_params(hours__values=[9])
        assert seg.get_params(deep=True)["hours__values"] == [9]
        assert seg.named_segmenters["hours"].values == [9]

    def test_bare_list_rejected(self, hourly_week):
        """Test components without names are rejected."""
        seg = IntersectionSegmenter([CalendarSegmenter("hour"), CalendarSegmenter("day_of_week")])
        with pytest.raises(ValueError, match="tuple"):
            seg.compute_labels(hourly_week)

    def test_empty_rejected(self, hourly_week):
        """Test a missing or empty component list is rejected."""
        with pytest.raises(ValueError, match="at least one"):
            IntersectionSegmenter().compute_labels(hourly_week)


class TestUnion:
    """Tests for UnionSegmenter."""

    def test_one_match(self):
        """Test Monday 08:00 matches only the morning component."""
        assert _label_at(_morning_or_weekend(), datetime(2024, 1, 1, 8)) == {"morning": 8, "weekend": None}

    def test_both_match(self):
        """Test Saturday 08:00 gets its own label, distinct from Monday 08:00 and Saturday 14:00."""
        seg = _morning_or_weekend()
        both = _label_at(seg, datetime(2024, 1, 6, 8))
        assert both == {"morning": 8, "weekend": 6}
        assert both != _label_at(seg, datetime(2024, 1, 1, 8))
        assert both != _label_at(seg, datetime(2024, 1, 6, 14))

    def test_equal_values_do_not_collide(self):
        """Test hour 6 and Saturday (6) from different components give different labels."""
        seg = _morning_or_weekend(morning=(6, 7, 8))
        assert _label_at(seg, datetime(2024, 1, 1, 6)) == {"morning": 6, "weekend": None}
        assert _label_at(seg, datetime(2024, 1, 6, 14)) == {"morning": None, "weekend": 6}

    def test_no_match_is_null(self):
        """Test Monday 14:00 is a real null, not a struct of nulls."""
        labels = _morning_or_weekend().compute_labels(pl.Series("time", [datetime(2024, 1, 1, 14)]))
        assert labels.is_null().to_list() == [True]


class TestComplement:
    """Tests for ComplementSegmenter."""

    def test_outside_business_hours(self):
        """Test rows from 19:00 to 07:00 are inside with label True."""
        day = pl.Series("time", [datetime(2024, 1, 1, h) for h in range(24)])
        labels = ComplementSegmenter(CalendarSegmenter("hour", values=list(range(8, 19)))).compute_labels(day)
        assert labels.to_list() == [None if 8 <= h <= 18 else True for h in range(24)]

    def test_missing_segmenter(self, hourly_week):
        """Test a missing inner segmenter is rejected."""
        with pytest.raises(ValueError, match="segmenter"):
            ComplementSegmenter().compute_labels(hourly_week)

    def test_excluding_a_composite(self, hourly_week):
        """Test the complement of business hours keeps every other row."""
        labels = ComplementSegmenter(_business_hours()).compute_labels(hourly_week)
        assert labels.is_not_null().sum() == 7 * 24 - 5 * 11


class TestNesting:
    """Tests for composites whose components are composites."""

    def test_union_of_intersection_label(self):
        """Test Monday 09:00 carries a struct inside a struct."""
        seg = UnionSegmenter([
            ("business", _business_hours()),
            ("weekend", CalendarSegmenter("day_of_week", values=[6, 7])),
        ])
        assert _label_at(seg, datetime(2024, 1, 1, 9)) == {"business": {"days": 1, "hours": 9}, "weekend": None}
        assert _label_at(seg, datetime(2024, 1, 6, 3)) == {"business": None, "weekend": 6}
        assert _label_at(seg, datetime(2024, 1, 1, 3)) is None
