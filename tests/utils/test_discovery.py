import pytest

from yohou.utils.discovery import all_displays, all_estimators, all_functions


class TestAllEstimators:
    """Tests for all_estimators discovery function."""

    def test_all_estimators_total_count(self):
        """Test all_estimators returns correct total count."""
        estimators = all_estimators()
        assert len(estimators) == 71

    def test_all_estimators_forecaster_filter(self):
        """Test forecaster type filter (matches estimator_type == 'forecaster')."""
        forecasters = all_estimators(type_filter="forecaster")
        assert len(forecasters) == 6

    def test_all_estimators_point_filter(self):
        """Test point forecaster sub-type filter."""
        points = all_estimators(type_filter="point")
        assert len(points) == 5

    def test_all_estimators_interval_filter(self):
        """Test interval forecaster sub-type filter."""
        intervals = all_estimators(type_filter="interval")
        assert len(intervals) == 2

    def test_all_estimators_scorer_filter(self):
        """Test scorer type filter."""
        scorers = all_estimators(type_filter="scorer")
        assert len(scorers) == 17

    def test_all_estimators_transformer_filter(self):
        """Test transformer type filter."""
        transformers = all_estimators(type_filter="transformer")
        assert len(transformers) == 29

    def test_all_estimators_splitter_filter(self):
        """Test splitter type filter."""
        splitters = all_estimators(type_filter="splitter")
        assert len(splitters) == 2

    def test_all_estimators_multiple_filters(self):
        """Test multiple type filters combined."""
        multi = all_estimators(type_filter=["forecaster", "transformer"])
        assert len(multi) == 35  # 6 forecasters + 29 transformers

    def test_all_estimators_invalid_filter(self):
        """Test invalid filter raises error with proper message."""
        err_msg = "Invalid type_filter values"
        with pytest.raises(ValueError, match=err_msg):
            all_estimators(type_filter="xxxx")


class TestAllDisplays:
    """Tests for all_displays discovery function."""

    def test_all_displays(self):
        """Test all_displays returns empty list (no displays currently)."""
        displays = all_displays()
        assert len(displays) == 0


class TestAllFunctions:
    """Tests for all_functions discovery function."""

    def test_all_functions(self):
        """Test all_functions returns correct count."""
        functions = all_functions()
        assert len(functions) == 179
