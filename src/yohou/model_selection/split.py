"""Time series cross-validation splitters for model selection."""

import numbers
from collections.abc import Iterable, Iterator

import numpy as np
import polars as pl
from sklearn.model_selection._split import BaseCrossValidator, TimeSeriesSplit


class Splitter(TimeSeriesSplit):  # type: ignore[misc]
    """Time Series splitter.

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus
    shuffling in cross validator is inappropriate.

    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.

    max_train_size : int, default=None
        Maximum size for a single training set.

    test_size : int, default=None
        Used to limit the size of the test set. Defaults to
        ``n_samples // (n_splits + 1)``, which is the maximum allowed
        value.

    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        max_train_size: int | None = None,
        test_size: int | None = None,
    ) -> None:
        TimeSeriesSplit.__init__(
            self,
            n_splits=n_splits,
            max_train_size=max_train_size,
            test_size=test_size,
        )

    def split(
        self, y: pl.DataFrame
    ) -> Iterator[tuple[np.ndarray[int, np.dtype[np.intp]], np.ndarray[int, np.dtype[np.intp]]]]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        y : pl.DataFrame
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """

        return TimeSeriesSplit.split(self, y)  # type: ignore[no-any-return]


class _CVIterableWrapper(BaseCrossValidator):  # type: ignore[misc]
    """Wrapper class for old style cv objects and iterables."""

    def __init__(
        self, cv: Iterable[tuple[np.ndarray[object, object], np.ndarray[object, object]]]
    ) -> None:
        self.cv = list(cv)

    def get_n_splits(
        self,
        X: pl.DataFrame | None = None,
        y: pl.DataFrame | None = None,
        groups: object = None,
    ) -> int:
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return len(self.cv)

    def split(
        self, y: pl.DataFrame
    ) -> Iterator[tuple[np.ndarray[object, object], np.ndarray[object, object]]]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        y : pl.DataFrame
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        for train, test in self.cv:
            yield train, test


def check_cv(
    cv: int
    | Splitter
    | Iterable[tuple[np.ndarray[object, object], np.ndarray[object, object]]]
    | None = 5,
    forecasting_horizon: int = 1,
) -> Splitter | _CVIterableWrapper:
    """Input checker utility for building a cross-validator.

    Parameters
    ----------
    cv : int, cross-validation generator or an iterable, default=5
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold time series cross validation,
        - integer, to specify the number of folds in a time series `Splitter`,
        - :class:`yohou.model_selection.Splitter` instance,
        - An iterable yielding (train, test) splits as arrays of indices.

    forecasting_horizon : int >= 1, default=1
        Horizon to forecast recursively.

    Returns
    -------
    checked_cv : a cross-validator instance.
        The return value is a cross-validator which generates the train/test
        splits via the ``split`` method.

    """
    cv = 5 if cv is None else cv
    if isinstance(cv, numbers.Integral):
        return Splitter(cv, test_size=forecasting_horizon)

    if not isinstance(cv, Splitter):
        if isinstance(cv, Iterable):
            return _CVIterableWrapper(cv)

        else:
            raise ValueError(
                "Expected cv as an integer, cross-validation "
                "object (from yohou.model_selection) "
                "or an iterable. Got %s." % cv
            )

    return cv
