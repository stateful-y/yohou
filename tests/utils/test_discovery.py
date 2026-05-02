import importlib.util

import pytest

from yohou.utils.discovery import (
    _is_checked_function,
    all_displays,
    all_estimators,
    all_functions,
)

_HAS_PLOTLY = importlib.util.find_spec("plotly") is not None


class TestAllEstimators:
    """Tests for all_estimators discovery function."""

    def test_all_estimators_total_count(self):
        """Test all_estimators returns correct total count."""
        estimators = all_estimators()
        assert len(estimators) == 87

    def test_all_estimators_forecaster_filter(self):
        """Test forecaster type filter (matches estimator_type == 'forecaster')."""
        forecasters = all_estimators(type_filter="forecaster")
        assert len(forecasters) == 8

    def test_all_estimators_point_filter(self):
        """Test point forecaster sub-type filter."""
        points = all_estimators(type_filter="point")
        assert len(points) == 6

    def test_all_estimators_interval_filter(self):
        """Test interval forecaster sub-type filter."""
        intervals = all_estimators(type_filter="interval")
        assert len(intervals) == 2

    def test_all_estimators_scorer_filter(self):
        """Test scorer type filter."""
        scorers = all_estimators(type_filter="scorer")
        assert len(scorers) == 21

    def test_all_estimators_transformer_filter(self):
        """Test transformer type filter."""
        transformers = all_estimators(type_filter="transformer")
        assert len(transformers) == 34

    def test_all_estimators_splitter_filter(self):
        """Test splitter type filter."""
        splitters = all_estimators(type_filter="splitter")
        assert len(splitters) == 2

    def test_all_estimators_multiple_filters(self):
        """Test multiple type filters combined."""
        multi = all_estimators(type_filter=["forecaster", "transformer"])
        assert len(multi) == 42  # 8 forecasters + 34 transformers  (class_proba scorers not included)

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
        expected = 214 if _HAS_PLOTLY else 170
        assert len(functions) == expected


class TestAllEstimatorsAdvancedFilters:
    """Tests for type_filter variants that exercise additional branches."""

    def test_conformity_scorer_filter(self):
        """conformity_scorer filter matches conformity-type scorers."""
        result = all_estimators(type_filter="conformity_scorer")
        assert len(result) > 0
        from yohou.metrics.conformity_base import BaseConformityScorer

        for _, cls in result:
            assert issubclass(cls, BaseConformityScorer)

    def test_generic_scorer_filter(self):
        """Generic 'scorer' filter matches all scorer types."""
        all_scorers = all_estimators(type_filter="scorer")
        point_scorers = all_estimators(type_filter="point_scorer")
        interval_scorers = all_estimators(type_filter="interval_scorer")
        conformity_scorers = all_estimators(type_filter="conformity_scorer")
        class_proba_scorers = all_estimators(type_filter="class_proba_scorer")
        combined = len(point_scorers) + len(interval_scorers) + len(conformity_scorers) + len(class_proba_scorers)
        assert len(all_scorers) == combined

    def test_generic_forecaster_filter(self):
        """Generic 'forecaster' filter matches all forecaster types."""
        all_forecasters = all_estimators(type_filter="forecaster")
        assert len(all_forecasters) > 0
        point_forecasters = all_estimators(type_filter="point")
        interval_forecasters = all_estimators(type_filter="interval")
        class_proba_forecasters = all_estimators(type_filter="class_proba")
        point_names = {name for name, _ in point_forecasters}
        interval_names = {name for name, _ in interval_forecasters}
        class_proba_names = {name for name, _ in class_proba_forecasters}
        all_names = {name for name, _ in all_forecasters}
        assert all_names == point_names | interval_names | class_proba_names

    def test_list_filter_with_scorer(self):
        """List filter with scorer type returns matching estimators."""
        result = all_estimators(type_filter=["scorer", "splitter"])
        scorers = all_estimators(type_filter="scorer")
        splitters = all_estimators(type_filter="splitter")
        assert len(result) == len(scorers) + len(splitters)


class TestIsCheckedFunction:
    """Tests for _is_checked_function utility."""

    def test_non_function_returns_false(self):
        """Non-function objects are rejected."""
        assert _is_checked_function("not_a_function") is False
        assert _is_checked_function(42) is False

    def test_private_function_returns_false(self):
        """Private functions starting with _ are rejected."""

        def _private():
            pass

        _private.__module__ = "yohou.utils.panel"
        assert _is_checked_function(_private) is False

    def test_non_yohou_module_returns_false(self):
        """Functions from non-yohou modules are rejected."""

        def my_func():
            pass

        my_func.__module__ = "sklearn.base"
        assert _is_checked_function(my_func) is False

    def test_estimator_checks_module_rejected(self):
        """Functions from estimator_checks modules are rejected."""

        def check_something():
            pass

        check_something.__module__ = "yohou.testing.estimator_checks"
        assert _is_checked_function(check_something) is False

    def test_valid_yohou_function_accepted(self):
        """Public yohou functions pass the check."""

        def my_func():
            pass

        my_func.__module__ = "yohou.utils.panel"
        assert _is_checked_function(my_func) is True


class TestAllDisplaysContent:
    """Tests for all_displays content validation."""

    def test_displays_returns_list(self):
        """all_displays returns a list of tuples."""
        displays = all_displays()
        assert isinstance(displays, list)
        for item in displays:
            assert isinstance(item, tuple)
            assert len(item) == 2


class TestAllFunctionsContent:
    """Tests for all_functions content validation."""

    def test_functions_are_callable(self):
        """All discovered functions are callable."""
        functions = all_functions()
        for name, func in functions:
            assert callable(func), f"{name} is not callable"
            assert not name.startswith("_"), f"{name} starts with underscore"


class TestAllEstimatorsMultiTypeForecaster:
    """Tests for type_filter capturing forecasters with multiple capabilities."""

    def test_point_filter_includes_point_interval_type(self):
        """Point filter includes forecasters with 'point' in forecaster_type."""
        point = all_estimators(type_filter="point")
        point_names = {name for name, _ in point}
        assert "SplitConformalForecaster" in point_names

    def test_interval_filter_includes_point_interval_type(self):
        """Interval filter includes forecasters with 'interval' in forecaster_type."""
        interval = all_estimators(type_filter="interval")
        interval_names = {name for name, _ in interval}
        assert "SplitConformalForecaster" in interval_names
