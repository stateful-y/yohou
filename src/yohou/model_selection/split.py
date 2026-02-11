"""Time series cross-validation splitters for model selection."""

import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import Interval

from yohou.utils import validate_splitter_data
from yohou.utils.tags import Tags


class BaseSplitter(BaseEstimator, ABC):
    """Base class for yohou time series cross-validation splitters.

    Extends sklearn's BaseCrossValidator with time series-specific
    functionality including polars DataFrame support and panel data awareness.

    All concrete splitters should inherit from this class and implement
    the `_iter_test_indices()` method.

    Notes
    -----
    This is an abstract base class. Concrete splitters should inherit from
    this class and implement `_iter_test_indices()` and `get_n_splits()`.

    """

    _parameter_constraints: dict = {}

    # TODO: In the case of panel data, each group might have a different amount of data
    # Should we add panel_group_name(s) paramter to split and get_n_splits methods?
    @abstractmethod
    def split(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> Iterator[tuple[np.ndarray[Any, np.dtype[np.intp]], np.ndarray[Any, np.dtype[np.intp]]]]:
        """Generate indices to split time series data.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with mandatory "time" column.
        X : pl.DataFrame, optional
            Exogenous features (for signature compatibility, not used in splitting logic).
        Yields
        ------
        train : ndarray
            Training set row indices for that split.
        test : ndarray
            Test set row indices for that split.

        """

    @abstractmethod
    def _iter_test_indices(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> Iterator[np.ndarray[Any, np.dtype[np.intp]]]:
        """Generate test indices for each split.

        Must be implemented by concrete splitter classes.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame, optional
            Exogenous features.
        Yields
        ------
        test : ndarray
            Test set indices for this split.

        """

    @abstractmethod
    def get_n_splits(
        self,
        y: pl.DataFrame | None = None,
        X: pl.DataFrame | None = None,
    ) -> int:
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        y : pl.DataFrame, optional
            Target time series.
        X : pl.DataFrame, optional
            Exogenous features.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.

        """

    def __sklearn_tags__(self):
        """Get metadata tags for this splitter.

        Returns
        -------
        tags : Tags
            Metadata tags describing splitter capabilities.

        """
        tags = Tags(estimator_type="splitter")
        tags.splitter_tags.supports_panel_data = True
        return tags


class ExpandingWindowSplitter(BaseSplitter):
    """Expanding window time series cross-validation splitter.

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus
    shuffling in cross validator is inappropriate.

    The training set grows with each split (expanding window), meaning
    successive training sets are supersets of those that come before them.
    This is useful when more data generally leads to better models and
    when you want to simulate accumulating historical data over time.

    Parameters
    ----------
    n_splits : int, default=3
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum size for a single training set. If None, all available
        training data is used.
    test_size : int, default=None
        Used to limit the size of the test set. Defaults to
        ``n_samples // (n_splits + 1)``, which is the maximum allowed
        value with no overlap between test sets.
    gap : int, default=None
        Number of time steps to skip between the end of the training set
        and the start of the test set. If None or 0, no gap is applied.
        Use gap > 0 to simulate forecasting lag or prevent data leakage.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> from yohou.model_selection import ExpandingWindowSplitter
    >>>
    >>> # Create time series
    >>> time = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
    >>> y = pl.DataFrame({"time": time, "value": range(100)})
    >>>
    >>> # 3 splits with 10-day test windows
    >>> splitter = ExpandingWindowSplitter(n_splits=3, test_size=10)
    >>> splits = list(splitter.split(y))
    >>> len(splits)
    3
    >>>
    >>> # First split: train on [0:70], test on [70:80]
    >>> train, test = splits[0]
    >>> len(train), len(test)
    (70, 10)
    >>>
    >>> # Second split: train on [0:80], test on [80:90] (training set grows)
    >>> train, test = splits[1]
    >>> len(train), len(test)
    (80, 10)
    >>>
    >>> # With 5-day gap between train and test
    >>> splitter_gap = ExpandingWindowSplitter(n_splits=3, test_size=10, gap=5)
    >>> train, test = list(splitter_gap.split(y))[0]
    >>> bool(test[0] - train[-1] == 6)  # 5-day gap + 1 for next index
    True

    Notes
    -----
    - Training sets grow with each split (expanding window)
    - Test sets do not overlap
    - All data is used in temporal order
    - For panel data, splits all groups together using row indices

    See Also
    --------
    SlidingWindowSplitter : Fixed-size rolling window splitter

    """

    _parameter_constraints: dict = {
        **BaseSplitter._parameter_constraints,
        "n_splits": [Interval(numbers.Integral, 2, None, closed="left")],
        "max_train_size": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "test_size": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "gap": [Interval(numbers.Integral, 0, None, closed="left"), None],
    }

    def __init__(
        self,
        n_splits: int = 3,
        *,
        max_train_size: int | None = None,
        test_size: int | None = None,
        gap: int | None = None,
    ) -> None:
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

        # Validate parameters
        self._validate_params()

    def split(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> Iterator[tuple[np.ndarray[Any, np.dtype[np.intp]], np.ndarray[Any, np.dtype[np.intp]]]]:
        """Generate indices to split time series data with expanding windows.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with mandatory "time" column.
        X : pl.DataFrame, optional
            Exogenous features (for signature compatibility, not used in splitting logic).

        Yields
        ------
        train : ndarray
            Training set row indices for that split.
        test : ndarray
            Test set row indices for that split.

        """
        # Validate data
        y, X = validate_splitter_data(self, y=y, X=X)

        n_samples = len(y)
        indices = np.arange(n_samples)
        max_train_size = self.max_train_size
        gap = self.gap if self.gap is not None else 0

        # Delegate to concrete implementation
        for test_index in self._iter_test_indices(y, X):
            # test_index already has gap incorporated from _iter_test_indices
            # Train indices are all samples before (test_start - gap)
            train_end = test_index[0] - gap
            train_index = indices[indices < train_end]

            # Apply max_train_size if specified
            if max_train_size is not None and len(train_index) > max_train_size:
                train_index = train_index[-max_train_size:]

            yield train_index, test_index

    def _iter_test_indices(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> Iterator[np.ndarray[Any, np.dtype[np.intp]]]:
        """Generate test indices for expanding window splits.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame, optional
            Exogenous features (not used).

        Yields
        ------
        test : ndarray
            Test set indices for this split.

        """
        n_samples = len(y)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        test_size = self.test_size if self.test_size is not None else n_samples // n_folds
        gap = self.gap if self.gap is not None else 0

        if n_folds > n_samples:
            raise ValueError(f"Cannot have number of folds={n_folds} greater than the number of samples={n_samples}.")

        if test_size >= n_samples:
            raise ValueError(f"test_size={test_size} should be less than the number of samples={n_samples}.")

        # Calculate test start positions accounting for gap
        # The last test must end at n_samples, working backwards n_splits times
        # Each test is test_size long, and we need gap between train and test
        # So test positions are shifted by gap compared to no-gap case
        test_starts = range(n_samples - gap - n_splits * test_size, n_samples - gap, test_size)

        for test_start in test_starts:
            # Add gap to shift test forward (train ends gap samples before test)
            if test_start + gap < 0:
                continue
            yield np.arange(test_start + gap, test_start + gap + test_size, dtype=np.intp)

    def get_n_splits(
        self,
        y: pl.DataFrame | None = None,
        X: pl.DataFrame | None = None,
    ) -> int:
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        y : pl.DataFrame, optional
            Required when gap > 0 to compute number of valid splits.

        X : pl.DataFrame, optional
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.

        """
        # Gap is incorporated into index calculation, doesn't affect count
        return self.n_splits

    def __sklearn_tags__(self):
        """Get metadata tags for this splitter.

        Returns
        -------
        tags : Tags
            Metadata tags describing splitter capabilities.

        """
        tags = super().__sklearn_tags__()
        tags.splitter_tags.splitter_type = "expanding"
        tags.splitter_tags.requires_data_for_n_splits = False
        return tags


class SlidingWindowSplitter(BaseSplitter):
    """Sliding window time series cross-validation splitter.

    Both training and test windows slide forward with fixed sizes.
    This is useful when recent data is more relevant than distant past
    (concept drift), or when simulating production scenarios with
    fixed-size training windows.

    Parameters
    ----------
    train_size : int, default=50
        Number of samples in each training window.
    test_size : int, default=10
        Number of samples in each test window.
    stride : int, default=None
        Number of samples to move forward between splits. If None,
        defaults to `test_size` (non-overlapping windows).
    gap : int, default=None
        Number of time steps to skip between the end of the training set
        and the start of the test set. If None or 0, no gap is applied.
        Use gap > 0 to simulate forecasting lag or prevent data leakage.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> from yohou.model_selection import SlidingWindowSplitter
    >>>
    >>> # Create time series
    >>> time = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
    >>> y = pl.DataFrame({"time": time, "value": range(100)})
    >>>
    >>> # Sliding windows: 50-day train, 10-day test
    >>> splitter = SlidingWindowSplitter(train_size=50, test_size=10)
    >>> splits = list(splitter.split(y))
    >>> len(splits)
    5
    >>>
    >>> # First split: train on [0:50], test on [50:60]
    >>> train, test = splits[0]
    >>> len(train), len(test)
    (50, 10)
    >>>
    >>> # Second split: train on [10:60], test on [60:70] (windows slide)
    >>> train, test = splits[1]
    >>> len(train), len(test)
    (50, 10)
    >>>
    >>> # With 3-day gap between train and test
    >>> splitter_gap = SlidingWindowSplitter(train_size=50, test_size=10, gap=3)
    >>> train, test = list(splitter_gap.split(y))[0]
    >>> bool(test[0] - train[-1] == 4)  # 3-day gap + 1 for next index
    True

    Notes
    -----
    - Training and test windows have fixed sizes
    - Windows slide forward by `stride` samples
    - Useful for concept drift scenarios
    - Number of splits determined automatically from data length

    See Also
    --------
    ExpandingWindowSplitter : Growing training window splitter

    """

    _parameter_constraints: dict = {
        **BaseSplitter._parameter_constraints,
        "train_size": [Interval(numbers.Integral, 1, None, closed="left")],
        "test_size": [Interval(numbers.Integral, 1, None, closed="left")],
        "stride": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "gap": [Interval(numbers.Integral, 0, None, closed="left"), None],
    }

    # TODO: Find better defaults for train_size and test_size
    def __init__(
        self,
        train_size: int = 50,
        test_size: int = 10,
        stride: int | None = None,
        gap: int | None = None,
    ) -> None:
        self.train_size = train_size
        self.test_size = test_size
        self.stride = stride
        self.gap = gap

        # Validate parameters
        self._validate_params()

    def split(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> Iterator[tuple[np.ndarray[Any, np.dtype[np.intp]], np.ndarray[Any, np.dtype[np.intp]]]]:
        """Generate indices to split time series data with sliding windows.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with mandatory "time" column.
        X : pl.DataFrame, optional
            Exogenous features (for signature compatibility, not used in splitting logic).

        Yields
        ------
        train : ndarray
            Training set row indices for that split.
        test : ndarray
            Test set row indices for that split.

        """
        # Validate data
        y, X = validate_splitter_data(self, y=y, X=X)

        train_size = self.train_size
        gap = self.gap if self.gap is not None else 0

        # Delegate to concrete implementation for test indices
        for test_index in self._iter_test_indices(y, X):
            # For sliding window, train indices are the fixed-size window
            # ending gap samples before the test set
            train_end = test_index[0] - gap
            train_start = train_end - train_size

            train_index = np.arange(train_start, train_end, dtype=np.intp)
            yield train_index, test_index

    def _iter_test_indices(
        self,
        y: pl.DataFrame,
        X: pl.DataFrame | None = None,
    ) -> Iterator[np.ndarray[Any, np.dtype[np.intp]]]:
        """Generate test indices for sliding window splits.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series.
        X : pl.DataFrame, optional
            Exogenous features (not used).

        Yields
        ------
        test : ndarray
            Test set indices for this split.

        """
        n_samples = len(y)
        train_size = self.train_size
        test_size = self.test_size
        stride = self.stride if self.stride is not None else test_size

        gap = self.gap if self.gap is not None else 0

        if train_size + gap + test_size > n_samples:
            raise ValueError(
                f"train_size ({train_size}) + gap ({gap}) + test_size ({test_size}) = "
                f"{train_size + gap + test_size} is greater than n_samples ({n_samples})."
            )

        # Start from the first position where we have enough data (including gap)
        test_start = train_size + gap

        while test_start + test_size <= n_samples:
            yield np.arange(test_start, test_start + test_size, dtype=np.intp)
            test_start += stride

    def get_n_splits(
        self,
        y: pl.DataFrame | None = None,
        X: pl.DataFrame | None = None,
    ) -> int:
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        y : pl.DataFrame, optional
            Target time series. Required to compute number of splits.
        X : pl.DataFrame, optional
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.

        """
        if y is None:
            raise ValueError("y is required to compute the number of splits for SlidingWindowSplitter.")

        y, X = validate_splitter_data(self, y=y, X=X)

        n_samples = len(y)
        train_size = self.train_size
        test_size = self.test_size
        stride = self.stride if self.stride is not None else test_size
        gap = self.gap if self.gap is not None else 0

        if train_size + gap + test_size > n_samples:
            return 0

        # Calculate number of complete windows (accounting for gap)
        available_positions = n_samples - train_size - gap - test_size
        n_splits = (available_positions // stride) + 1

        return n_splits

    def __sklearn_tags__(self):
        """Get metadata tags for this splitter.

        Returns
        -------
        tags : Tags
            Metadata tags describing splitter capabilities.

        """
        tags = super().__sklearn_tags__()
        tags.splitter_tags.splitter_type = "sliding"
        tags.splitter_tags.requires_data_for_n_splits = True
        return tags


def check_cv(
    cv: int | BaseSplitter | Iterable[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] | None = 5,
    forecasting_horizon: int = 1,
) -> BaseSplitter:
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
        return ExpandingWindowSplitter(cv, test_size=forecasting_horizon)

    if not isinstance(cv, BaseSplitter):
        raise ValueError("Expected cv as an integer or a splitter object (from yohou.model_selection).")

    return cv
