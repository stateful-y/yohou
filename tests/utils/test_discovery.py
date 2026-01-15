import pytest

from yohou.utils.discovery import all_displays, all_estimators, all_functions


def test_all_estimators():
    estimators = all_estimators()
    assert len(estimators) == 26

    estimators = all_estimators(type_filter="classifier")
    assert len(estimators) == 0

    estimators = all_estimators(type_filter=["classifier", "transformer"])
    assert len(estimators) == 7

    err_msg = "Parameter type_filter must be"
    with pytest.raises(ValueError, match=err_msg):
        all_estimators(type_filter="xxxx")


def test_all_displays():
    displays = all_displays()
    assert len(displays) == 0


def test_all_functions():
    functions = all_functions()
    # Updated count includes 55 new functions from yohou.testing module (49 checks + 2 generators + 4 utilities)
    assert len(functions) == 78
