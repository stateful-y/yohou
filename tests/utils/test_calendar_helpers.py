"""Tests for the shared calendar feature helpers."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from yohou.utils._calendar import _FEATURE_DOMAINS, ALL_FEATURES, _check_feature_values, _extract_feature

EXPECTED_DOMAINS = {
    "hour": (0, 23),
    "minute": (0, 59),
    "day_of_week": (1, 7),
    "day_of_month": (1, 31),
    "day_of_year": (1, 366),
    "week": (1, 53),
    "month": (1, 12),
    "quarter": (1, 4),
    "is_weekend": (0, 1),
    "is_month_start": (0, 1),
    "is_month_end": (0, 1),
    "is_quarter_start": (0, 1),
    "is_quarter_end": (0, 1),
    "is_year_start": (0, 1),
    "is_year_end": (0, 1),
    "year": None,
}


class TestFeatureDomains:
    """Tests for the per-feature value ranges."""

    def test_every_feature_has_a_domain(self):
        """Test the domain table covers exactly the calendar features."""
        assert set(_FEATURE_DOMAINS) == set(ALL_FEATURES)

    @pytest.mark.parametrize(("feature", "domain"), EXPECTED_DOMAINS.items())
    def test_domain(self, feature, domain):
        """Test each feature's range matches the documented one."""
        assert _FEATURE_DOMAINS[feature] == domain

    @pytest.mark.parametrize("feature", [f for f, d in EXPECTED_DOMAINS.items() if d is not None])
    def test_extracted_values_stay_in_domain(self, feature):
        """Test the extractor never yields a value outside the feature's range over two leap years."""
        start = datetime(2023, 1, 1)
        times = [start + timedelta(minutes=37 * i) for i in range(2 * 366 * 24 * 60 // 37)]
        values = pl.DataFrame({"time": times}).select(_extract_feature(feature)).to_series()
        low, high = _FEATURE_DOMAINS[feature]
        assert values.min() >= low
        assert values.max() <= high


class TestCheckFeatureValues:
    """Tests for _check_feature_values."""

    @pytest.mark.parametrize(("feature", "domain"), [(f, d) for f, d in EXPECTED_DOMAINS.items() if d is not None])
    def test_bounds_accepted(self, feature, domain):
        """Test both ends of each range are accepted."""
        assert _check_feature_values(feature, list(domain)) == list(domain)

    @pytest.mark.parametrize(("feature", "domain"), [(f, d) for f, d in EXPECTED_DOMAINS.items() if d is not None])
    def test_out_of_range_rejected(self, feature, domain):
        """Test values just outside each range are rejected with the range in the message."""
        low, high = domain
        with pytest.raises(ValueError, match=rf"\[{low - 1}, {high + 1}\].*{low} to {high}"):
            _check_feature_values(feature, [low - 1, high + 1])

    def test_year_unbounded(self):
        """Test year accepts any integer."""
        assert _check_feature_values("year", [-5, 0, 3000]) == [-5, 0, 3000]

    def test_range_and_numpy_ints(self):
        """Test a range and numpy integers come back as Python ints."""
        result = _check_feature_values("hour", [*range(8, 10), np.int64(12)])
        assert result == [8, 9, 12]
        assert all(type(v) is int for v in result)

    def test_non_integer_rejected(self):
        """Test a float value is rejected and named."""
        with pytest.raises(ValueError, match=r"must be integers, got \[8.5\]"):
            _check_feature_values("hour", [8, 8.5])

    def test_unknown_feature_rejected(self):
        """Test an unknown feature name is rejected."""
        with pytest.raises(ValueError, match="Unknown feature"):
            _check_feature_values("fortnight", [1])
