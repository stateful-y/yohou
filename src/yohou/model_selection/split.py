"""Time series cross-validation splitters for model selection."""

import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator

from yohou.utils import Tags, validate_splitter_data
from yohou.utils._compat import Interval

__all__ = [
    "BaseSplitter",
    "ExpandingWindowSplitter",
    "SlidingWindowSplitter",
    "check_cv",
]


class BaseSplitter(BaseEstimator, ABC):
    """Base class for yohou time series cross-validation splitters.

    Extends sklearn's BaseCrossValidator with time series-specific
    functionality including polars DataFrame support and panel data awareness.

    All concrete splitters should inherit from this class and implement
    the ``_iter_test_indices()`` method.

    Attributes
    ----------
    interval_ : str
        Detected time interval of the data, set during ``split()``.

    Notes
    -----
    This is an abstract base class. Concrete splitters should inherit from
    this class and implement ``_iter_test_indices()`` and ``get_n_splits()``.

    See Also
    --------
    ExpandingWindowSplitter : Expanding-window cross-validation.
    SlidingWindowSplitter : Sliding-window cross-validation.

    """

    _parameter_constraints: dict = {}

    # Fitted attributes (set during split())
    interval_: str

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
            Target time series used to generate train/test split indices.
            Must have a ``"time"`` column.
        X : pl.DataFrame or None, default=None
            Exogenous features.  Not used for splitting but accepted for
            API compatibility.

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
        """Return the number of cross-validation folds.

        Parameters
        ----------
        y : pl.DataFrame or None, default=None
            Not used.  Accepted for API compatibility.
        X : pl.DataFrame or None, default=None
            Not used.  Accepted for API compatibility.

        Returns
        -------
        int
            The number of cross-validation folds.

        """

    def __sklearn_tags__(self):
        """Get metadata tags for this splitter.

        Returns
        -------
        tags : Tags
            Metadata tags describing splitter capabilities.

        """
        tags = Tags(estimator_type="splitter")
        if tags.splitter_tags is not None:
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
    gap : int, default=0
        Number of time steps to skip between the end of the training set
        and the start of the test set. Must be non-negative.
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
        "gap": [Interval(numbers.Integral, 0, None, closed="left")],
    }

    def __init__(
        self,
        n_splits: int = 3,
        *,
        max_train_size: int | None = None,
        test_size: int | None = None,
        gap: int = 0,
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
            Target time series used to generate train/test split indices.
            Must have a ``"time"`` column.
        X : pl.DataFrame or None, default=None
            Exogenous features.  Not used for splitting but accepted for
            API compatibility.

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
        gap = self.gap

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
        gap = self.gap

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
        """Return the number of cross-validation folds.

        Parameters
        ----------
        y : pl.DataFrame or None, default=None
            Not used.  Accepted for API compatibility.
        X : pl.DataFrame or None, default=None
            Not used.  Accepted for API compatibility.

        Returns
        -------
        int
            The number of cross-validation folds.

        """
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
        return tags


class SlidingWindowSplitter(BaseSplitter):
    """Sliding window time series cross-validation splitter.

    Both training and test windows slide forward with fixed sizes.
    This is useful when recent data is more relevant than distant past
    (concept drift), or when simulating production scenarios with
    fixed-size training windows.

    Parameters
    ----------
    n_splits : int, default=3
        Number of cross-validation folds. Must be at least 2.
    train_size : int, default=None
        Number of samples in each training window.  When ``None``
        (default), the training size is computed automatically so that
        exactly ``n_splits`` folds fit the data:
        ``train_size = n_samples - gap - test_size - (n_splits - 1) * stride``.
        When both ``n_splits`` and ``train_size`` are provided explicitly,
        they must be consistent with the data length; otherwise a
        ``ValueError`` is raised during ``split()``.
    test_size : int, default=10
        Number of samples in each test window.
    stride : int, default=None
        Number of samples to move forward between splits. If None,
        defaults to `test_size` (non-overlapping windows).
    gap : int, default=0
        Number of time steps to skip between the end of the training set
        and the start of the test set. Must be non-negative.
        Use gap > 0 to simulate forecasting lag or prevent data leakage.

    Attributes
    ----------
    train_size_ : int
        Resolved training window size, set during ``split()``.
        Equal to ``train_size`` when provided explicitly, or computed
        from ``n_splits`` and the data length.

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
    >>> # 5-fold sliding window with 10-day test windows
    >>> splitter = SlidingWindowSplitter(n_splits=5, test_size=10)
    >>> splits = list(splitter.split(y))
    >>> len(splits)
    5
    >>>
    >>> # First split: training size computed automatically
    >>> train, test = splits[0]
    >>> len(test)
    10
    >>>
    >>> # All training windows have the same size
    >>> all(len(tr) == len(splits[0][0]) for tr, _ in splits)
    True
    >>>
    >>> # With explicit train_size and gap
    >>> splitter_gap = SlidingWindowSplitter(n_splits=6, train_size=30, test_size=10, gap=3)
    >>> train, test = list(splitter_gap.split(y))[0]
    >>> bool(test[0] - train[-1] == 4)  # 3-day gap + 1 for next index
    True

    Notes
    -----
    - Training and test windows have fixed sizes
    - Windows slide forward by `stride` samples
    - Useful for concept drift scenarios
    - When ``train_size`` is omitted, it is computed from ``n_splits``
      and the data length in the first call to ``split()``

    See Also
    --------
    ExpandingWindowSplitter : Growing training window splitter

    """

    _parameter_constraints: dict = {
        **BaseSplitter._parameter_constraints,
        "n_splits": [Interval(numbers.Integral, 2, None, closed="left")],
        "train_size": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "test_size": [Interval(numbers.Integral, 1, None, closed="left")],
        "stride": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "gap": [Interval(numbers.Integral, 0, None, closed="left")],
    }

    def __init__(
        self,
        n_splits: int = 3,
        *,
        train_size: int | None = None,
        test_size: int = 10,
        stride: int | None = None,
        gap: int = 0,
    ) -> None:
        self.n_splits = n_splits
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
            Target time series used to generate train/test split indices.
            Must have a ``"time"`` column.
        X : pl.DataFrame or None, default=None
            Exogenous features.  Not used for splitting but accepted for
            API compatibility.

        Yields
        ------
        train : ndarray
            Training set row indices for that split.
        test : ndarray
            Test set row indices for that split.

        """
        # Validate data
        y, X = validate_splitter_data(self, y=y, X=X)

        gap = self.gap

        # Resolve train_size (may compute from n_splits) and store as fitted attr
        self.train_size_ = self._resolve_train_size(len(y))

        # Delegate to concrete implementation for test indices
        for test_index in self._iter_test_indices(y, X):
            # For sliding window, train indices are the fixed-size window
            # ending gap samples before the test set
            train_end = test_index[0] - gap
            train_start = train_end - self.train_size_

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
        train_size = self._resolve_train_size(n_samples)
        test_size = self.test_size
        stride = self.stride if self.stride is not None else test_size
        gap = self.gap

        if train_size + gap + test_size > n_samples:
            raise ValueError(
                f"train_size ({train_size}) + gap ({gap}) + test_size ({test_size}) = "
                f"{train_size + gap + test_size} is greater than n_samples ({n_samples})."
            )

        # Fixed iteration: produce exactly n_splits test windows
        test_start = train_size + gap
        for _ in range(self.n_splits):
            if test_start + test_size > n_samples:
                break
            yield np.arange(test_start, test_start + test_size, dtype=np.intp)
            test_start += stride

    def _resolve_train_size(self, n_samples: int) -> int:
        """Compute or validate the training window size.

        Parameters
        ----------
        n_samples : int
            Number of samples in the dataset.

        Returns
        -------
        int
            Resolved training window size.

        Raises
        ------
        ValueError
            If the computed train_size is less than 1 or if an explicit
            train_size is inconsistent with n_splits and the data length.

        """
        test_size = self.test_size
        stride = self.stride if self.stride is not None else test_size
        gap = self.gap
        n_splits = self.n_splits

        if self.train_size is None:
            # Compute train_size from n_splits
            train_size = n_samples - gap - test_size - (n_splits - 1) * stride
            if train_size < 1:
                raise ValueError(
                    f"Not enough data for n_splits={n_splits} with "
                    f"test_size={test_size}, stride={stride}, gap={gap}: "
                    f"computed train_size={train_size} (must be >= 1). "
                    f"Reduce n_splits or provide more data."
                )
            return train_size

        # Both n_splits and train_size are explicit; check consistency
        train_size = self.train_size
        if train_size + gap + test_size > n_samples:
            raise ValueError(
                f"train_size ({train_size}) + gap ({gap}) + test_size ({test_size}) = "
                f"{train_size + gap + test_size} is greater than n_samples ({n_samples})."
            )

        available = n_samples - train_size - gap - test_size
        max_splits = (available // stride) + 1
        if n_splits > max_splits:
            raise ValueError(
                f"Inconsistent parameters: n_splits={n_splits} but the data "
                f"(n_samples={n_samples}) with train_size={train_size}, "
                f"test_size={test_size}, stride={stride}, gap={gap} supports "
                f"at most {max_splits} splits. Set train_size=None to "
                f"auto-compute or reduce n_splits to at most {max_splits}."
            )
        return train_size

    def get_n_splits(
        self,
        y: pl.DataFrame | None = None,
        X: pl.DataFrame | None = None,
    ) -> int:
        """Return the number of cross-validation folds.

        Parameters
        ----------
        y : pl.DataFrame or None, default=None
            Not used.  Accepted for API compatibility.
        X : pl.DataFrame or None, default=None
            Not used.  Accepted for API compatibility.

        Returns
        -------
        int
            The number of cross-validation folds.

        """
        return self.n_splits

    def __sklearn_tags__(self):
        """Get metadata tags for this splitter.

        Returns
        -------
        tags : Tags
            Metadata tags describing splitter capabilities.

        """
        tags = super().__sklearn_tags__()
        tags.splitter_tags.splitter_type = "sliding"
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
        - `BaseSplitter` instance,
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
        return ExpandingWindowSplitter(int(cv), test_size=forecasting_horizon)

    if not isinstance(cv, BaseSplitter):
        raise ValueError("Expected cv as an integer or a splitter object (from yohou.model_selection).")

    return cv
