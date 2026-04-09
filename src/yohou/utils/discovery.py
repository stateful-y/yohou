"""Utilities for discovering estimators, displays, and functions in yohou."""

# Adapted from scikit-learn
# Authors: scikit-learn-contrib developers
# License: BSD 3 clause

import inspect
import logging
import pkgutil
import warnings
from importlib import import_module
from operator import itemgetter
from pathlib import Path

from sklearn.base import BaseEstimator

__all__ = ["all_displays", "all_estimators", "all_functions"]

_MODULE_TO_IGNORE = {"tests"}


def all_estimators(type_filter: str | list[str] | None = None) -> list[tuple[str, type]]:
    """Get a list of all estimators from `yohou`.

    This function crawls the module and gets all classes that inherit
    from `BaseEstimator`. Classes that are defined in test-modules are not
    included.

    Parameters
    ----------
    type_filter : {"forecaster", "point", "interval", \
            "transformer", "splitter", "scorer", "point_scorer", \
            "interval_scorer", "conformity_scorer"} or list of such str, default=None
        Which kind of estimators should be returned. If None, no filter is
        applied and all estimators are returned. Possible values are:

        - 'forecaster': All forecasters (point, interval, or both)
        - 'point': Only point forecasters
        - 'interval': Only interval forecasters
        - 'transformer': Transformers
        - 'splitter': Cross-validation splitters
        - 'scorer': All scorers
        - 'point_scorer': Only point scorers
        - 'interval_scorer': Only interval scorers
        - 'conformity_scorer': Only conformity scorers

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class.

    Examples
    --------
    >>> from yohou.utils.discovery import all_estimators
    >>> estimators = all_estimators()
    >>> type(estimators)
    <class 'list'>
    >>> forecasters = all_estimators(type_filter='forecaster')
    >>> points = all_estimators(type_filter='point')

    See Also
    --------
    `all_displays` : Get all display classes from yohou.
    `all_functions` : Get all public functions from yohou.
    """

    def is_abstract(c: type) -> bool:
        """Check if a class is abstract."""
        if not (hasattr(c, "__abstractmethods__")):
            return False
        abstract_methods = c.__abstractmethods__
        return bool(abstract_methods)

    all_classes = []
    root = str(Path(__file__).parent.parent)  # yohou package
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        for _, module_name, _ in pkgutil.walk_packages(path=[root], prefix="yohou."):
            module_parts = module_name.split(".")
            if any(part in _MODULE_TO_IGNORE for part in module_parts):
                continue
            try:
                module = import_module(module_name)
            except ImportError:
                continue
            classes = inspect.getmembers(module, inspect.isclass)
            classes = [
                (name, est_cls)
                for name, est_cls in classes
                if not name.startswith("_") and est_cls.__module__ == module_name
            ]

            all_classes.extend(classes)

    all_classes_set = set(all_classes)

    estimators = [c for c in all_classes_set if (issubclass(c[1], BaseEstimator) and c[0] != "BaseEstimator")]
    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]

    if type_filter is not None:
        type_filter = [type_filter] if not isinstance(type_filter, list) else list(type_filter)

        filtered_estimators = []

        # Define valid filter types
        valid_filters = {
            "forecaster",
            "point",
            "interval",
            "transformer",
            "splitter",
            "scorer",
            "point_scorer",
            "interval_scorer",
            "conformity_scorer",
        }

        # Check for invalid filters
        invalid_filters = set(type_filter) - valid_filters
        if invalid_filters:
            raise ValueError(
                f"Invalid type_filter values: {sorted(invalid_filters)}. Valid options are: {sorted(valid_filters)}"
            )

        # Filter estimators by tags
        for name, est_cls in estimators:
            try:
                # Get tags from instance (tags may be instance-dependent)
                # Try default initialization
                try:
                    instance = est_cls()
                except TypeError:
                    # Skip estimators that require constructor arguments
                    continue

                tags = instance.__sklearn_tags__()
                est_type = tags.estimator_type

                # Check base estimator type
                if est_type in type_filter:
                    filtered_estimators.append((name, est_cls))
                    continue

                # Check forecaster sub-types
                if est_type == "forecaster" and hasattr(tags, "forecaster_tags"):
                    forecaster_type = tags.forecaster_tags.forecaster_type
                    if (
                        forecaster_type == "point"
                        and "point" in type_filter
                        or forecaster_type == "interval"
                        and "interval" in type_filter
                    ):
                        filtered_estimators.append((name, est_cls))
                    elif forecaster_type == "both":
                        if "point" in type_filter or "interval" in type_filter:
                            filtered_estimators.append((name, est_cls))
                    elif "forecaster" in type_filter:
                        # Generic forecaster filter matches all forecasters
                        filtered_estimators.append((name, est_cls))

                # Check scorer sub-types
                elif est_type == "scorer" and hasattr(tags, "scorer_tags"):
                    prediction_type = tags.scorer_tags.prediction_type
                    if (
                        prediction_type == "point"
                        and "point_scorer" in type_filter
                        or prediction_type == "interval"
                        and "interval_scorer" in type_filter
                        or prediction_type == "conformity"
                        and "conformity_scorer" in type_filter
                    ):
                        filtered_estimators.append((name, est_cls))
                    elif "scorer" in type_filter:
                        # Generic scorer filter matches all scorers
                        filtered_estimators.append((name, est_cls))

            except Exception as e:
                # Skip estimators that can't be instantiated or don't have proper tags
                logging.getLogger(__name__).debug("Skipped %s: %s", name, e)
                continue

        estimators = filtered_estimators

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators), key=itemgetter(0))


def all_displays() -> list[tuple[str, type]]:
    """Get a list of all displays from `yohou`.

    Returns
    -------
    displays : list of tuples
        List of (name, class), where ``name`` is the display class name as
        string and ``class`` is the actual type of the class.

    Examples
    --------
    >>> from yohou.utils.discovery import all_displays
    >>> displays = all_displays()

    See Also
    --------
    `all_estimators` : Get all estimator classes from yohou.
    `all_functions` : Get all public functions from yohou.
    """
    all_classes = []
    root = str(Path(__file__).parent.parent)  # yohou package
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        for _, module_name, _ in pkgutil.walk_packages(path=[root], prefix="yohou."):
            module_parts = module_name.split(".")
            if any(part in _MODULE_TO_IGNORE for part in module_parts):
                continue
            try:
                module = import_module(module_name)
            except ImportError:
                continue
            classes = inspect.getmembers(module, inspect.isclass)
            classes = [
                (name, display_class)
                for name, display_class in classes
                if not name.startswith("_") and name.endswith("Display")
            ]
            all_classes.extend(classes)

    return sorted(set(all_classes), key=itemgetter(0))


def _is_checked_function(item: object) -> bool:
    """Check if a function should be included in discovery."""
    if not inspect.isfunction(item):
        return False

    if item.__name__.startswith("_"):
        return False

    mod = item.__module__
    return not (not mod.startswith("yohou.") or mod.endswith("estimator_checks"))


def all_functions() -> list[tuple[str, object]]:
    """Get a list of all functions from `yohou`.

    Returns
    -------
    functions : list of tuples
        List of (name, function), where ``name`` is the function name as
        string and ``function`` is the actual function.

    Examples
    --------
    >>> from yohou.utils.discovery import all_functions
    >>> functions = all_functions()

    See Also
    --------
    `all_estimators` : Get all estimator classes from yohou.
    `all_displays` : Get all display classes from yohou.
    """
    all_functions = []
    root = str(Path(__file__).parent.parent)  # yohou package
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        for _, module_name, _ in pkgutil.walk_packages(path=[root], prefix="yohou."):
            module_parts = module_name.split(".")
            if any(part in _MODULE_TO_IGNORE for part in module_parts):
                continue

            try:
                module = import_module(module_name)
            except ImportError:
                continue
            functions = inspect.getmembers(module, _is_checked_function)
            functions = [(func.__name__, func) for name, func in functions if not name.startswith("_")]
            all_functions.extend(functions)

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(all_functions), key=itemgetter(0))
