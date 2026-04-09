"""Compatibility shim for private scikit-learn APIs.

All private sklearn imports used by yohou are centralized here so that
version-specific breakage is isolated to a single file. When sklearn changes
an internal API, only this module needs version-guarded ``try/except`` blocks.

Public sklearn imports (e.g. ``BaseEstimator``, ``clone``, ``check_is_fitted``,
``MetadataRouter``, ``process_routing``) remain imported directly at call sites
since they are part of sklearn's stable public surface.
"""

import timeit
from contextlib import contextmanager

from sklearn.base import _fit_context
from sklearn.compose._column_transformer import ColumnTransformer as sklearn_ColumnTransformer
from sklearn.compose._column_transformer import _check_X
from sklearn.model_selection._validation import (
    _aggregate_score_dicts,
    _insert_error_scores,
    _normalize_score_results,
    _warn_or_raise_about_fit_failures,
)
from sklearn.pipeline import _fit_one, _fit_transform_one, _transform_one
from sklearn.utils import _safe_indexing
from sklearn.utils._metadata_requests import COMPOSITE_METHODS, METHODS, SIMPLE_METHODS
from sklearn.utils._param_validation import HasMethods, Hidden, Interval, InvalidParameterError, StrOptions
from sklearn.utils._set_output import _get_output_config
from sklearn.utils.metadata_routing import _raise_for_params
from sklearn.utils.metaestimators import _BaseComposition, _safe_split
from sklearn.utils.validation import (
    _check_feature_names,
    _check_feature_names_in,
    _check_method_params,
    _check_n_features,
    _get_feature_names,
    _num_samples,
)


def _message_with_time(source, message, time):
    """Create one line message for logging purposes.

    Hardcoded from ``sklearn.utils._user_interface._message_with_time``
    to avoid depending on the private ``_user_interface`` module.

    Parameters
    ----------
    source : str
        String indicating the source or the reference of the message.

    message : str
        Short message.

    time : int
        Time in seconds.
    """
    start_message = f"[{source}] "

    time_str = "%4.1fmin" % (time / 60) if time > 60 else f" {time:5.1f}s"
    end_message = f" {message}, total={time_str}"
    dots_len = 70 - len(start_message) - len(end_message)
    return "{}{}{}".format(start_message, dots_len * ".", end_message)


@contextmanager
def _print_elapsed_time(source, message=None):
    """Log elapsed time to stdout when the context is exited.

    Hardcoded from ``sklearn.utils._user_interface._print_elapsed_time``
    to avoid depending on the private ``_user_interface`` module.

    Parameters
    ----------
    source : str
        String indicating the source or the reference of the message.

    message : str, default=None
        Short message. If None, nothing will be printed.

    Yields
    ------
    None
    """
    if message is None:
        yield
    else:
        start = timeit.default_timer()
        yield
        duration = timeit.default_timer() - start
        print(_message_with_time(source, message, duration))  # noqa: T201


def _filter_estimator_params(estimator_class: type, params: dict) -> dict:
    """Filter params to only those accepted by the estimator class constructor.

    Parameters added in newer sklearn versions (e.g. ``clip`` for ``MaxAbsScaler``
    in 1.7, ``handle_missing`` for ``SplineTransformer`` in 1.7) are silently
    dropped when running against an older sklearn that does not support them.

    Parameters
    ----------
    estimator_class : type
        The sklearn estimator class to check against.
    params : dict
        Parameters to filter.

    Returns
    -------
    dict
        Filtered parameters accepted by the class constructor.

    """
    import inspect  # noqa: PLC0415

    sig = inspect.signature(estimator_class.__init__)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_keyword:
        return params
    valid = {k for k in sig.parameters if k != "self"}
    return {k: v for k, v in params.items() if k in valid}


def _is_pandas_df(X: object) -> bool:
    """Return True if *X* is a pandas DataFrame.

    Duck-type check so we never need to import pandas.

    Parameters
    ----------
    X : object
        Object to check.

    Returns
    -------
    bool
        True when *X* looks like a pandas DataFrame.

    """
    return hasattr(X, "iloc") and hasattr(X, "columns")


#: Default value for ``ColumnTransformer.force_int_remainder_cols``.
#: sklearn <1.8 reads this attribute inside ``_get_remainder_cols``;
#: sklearn >=1.8 deprecated and then removed it.
FORCE_INT_REMAINDER_COLS = True


__all__ = [
    "COMPOSITE_METHODS",
    "HasMethods",
    "Hidden",
    "Interval",
    "InvalidParameterError",
    "METHODS",
    "SIMPLE_METHODS",
    "StrOptions",
    "_BaseComposition",
    "_aggregate_score_dicts",
    "_check_X",
    "_check_feature_names",
    "_check_feature_names_in",
    "_check_method_params",
    "_check_n_features",
    "_fit_context",
    "_fit_one",
    "_fit_transform_one",
    "_get_feature_names",
    "_get_output_config",
    "_insert_error_scores",
    "_normalize_score_results",
    "_num_samples",
    "_print_elapsed_time",
    "_raise_for_params",
    "_safe_indexing",
    "_safe_split",
    "_transform_one",
    "_warn_or_raise_about_fit_failures",
    "_filter_estimator_params",
    "_is_pandas_df",
    "FORCE_INT_REMAINDER_COLS",
    "sklearn_ColumnTransformer",
]
