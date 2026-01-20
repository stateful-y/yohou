import pytest

from yohou.utils.discovery import all_displays, all_estimators, all_functions


def test_all_estimators():
    """Test all_estimators with yohou's tag-based filtering."""
    # Test total count
    estimators = all_estimators()
    assert len(estimators) == 39

    # Test forecaster filter (matches estimator_type == "forecaster")
    forecasters = all_estimators(type_filter="forecaster")
    assert len(forecasters) == 6

    # Test point forecaster sub-type filter
    point_forecasters = all_estimators(type_filter="point_forecaster")
    assert len(point_forecasters) == 5

    # Test interval forecaster sub-type filter
    interval_forecasters = all_estimators(type_filter="interval_forecaster")
    assert len(interval_forecasters) == 2

    # Test scorer filter
    scorers = all_estimators(type_filter="scorer")
    assert len(scorers) == 15

    # Test multiple filters
    multi = all_estimators(type_filter=["forecaster", "transformer"])
    assert len(multi) == 10  # 6 forecasters + 4 transformers

    # Test transformer filter
    transformers = all_estimators(type_filter="transformer")
    assert len(transformers) == 4

    # Test splitter filter
    splitters = all_estimators(type_filter="splitter")
    assert len(splitters) == 2

    # Test invalid filter raises error with proper message
    err_msg = "Invalid type_filter values"
    with pytest.raises(ValueError, match=err_msg):
        all_estimators(type_filter="xxxx")


def test_all_displays():
    displays = all_displays()
    assert len(displays) == 0


def test_all_functions():
    functions = all_functions()
    assert len(functions) == 105
