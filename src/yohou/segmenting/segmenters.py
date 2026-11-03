"""Estimator-based segmentation of the time axis.

This module provides :class:`BaseSegmenter` and concrete segmenter estimators that
assign each timestamp a segment label, or mark it as outside every segment with a
null label. Weighters and outlier handlers take a segmenter to restrict their
behaviour to part of the calendar, such as specific hours of each day. Segmenters
are scikit-learn estimators: their parameters are introspectable via
``get_params``/``set_params`` and therefore tunable with
:class:`~yohou.model_selection.GridSearchCV`.

Here "segmentation" means a fixed calendar rule applied row by row. It is not the
change-point detection that sktime and aeon call segmentation: a segmenter never
looks at the values of the series, and a row's label depends only on its own
timestamp.
"""

from __future__ import annotations

import abc
from typing import Any

import polars as pl
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch

from yohou.utils._calendar import (
    _TIME_OF_DAY_FEATURES,
    ALL_FEATURES,
    _check_feature_values,
    _extract_feature,
)
from yohou.utils._compat import HasMethods, StrOptions, _BaseComposition

__all__ = [
    "BaseSegmenter",
    "CalendarSegmenter",
    "ComplementSegmenter",
    "IntersectionSegmenter",
    "UnionSegmenter",
]


class BaseSegmenter(BaseEstimator, metaclass=abc.ABCMeta):
    """Base class for time-axis segmenter estimators.

    A segmenter maps a series of timestamps to a series of labels, one per element.
    A null label means the element is outside every segment. Concrete subclasses
    declare their tunable parameters in ``_parameter_constraints`` and implement
    :meth:`compute_labels`. Because segmenters are scikit-learn estimators, their
    parameters can be addressed through a host with the ``__`` syntax (e.g.
    ``segmenter__values``) and tuned by search.

    Notes
    -----
    A row's label must depend only on that row's timestamp and the segmenter's
    parameters, never on the other rows. Labelling a series in consecutive parts
    then gives the same labels as labelling it whole, which the outlier handlers
    rely on to stay batch invariant. The segmenter check harness verifies it.

    Segmenters are stateless by default: :meth:`fit` is a no-op that exists so a
    future data-dependent segmenter can hook into the host estimator's ``fit``
    without an API change.

    See Also
    --------
    - [`CalendarSegmenter`][yohou.segmenting.segmenters.CalendarSegmenter] : Segments by calendar feature.
    - [`IntersectionSegmenter`][yohou.segmenting.segmenters.IntersectionSegmenter] : Rows inside every component.
    - [`UnionSegmenter`][yohou.segmenting.segmenters.UnionSegmenter] : Rows inside any component.
    - [`ComplementSegmenter`][yohou.segmenting.segmenters.ComplementSegmenter] : Rows outside a segmenter.

    """

    _parameter_constraints: dict = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Merge parameter constraints from all classes in the MRO."""
        super().__init_subclass__(**kwargs)
        merged: dict = {}
        for klass in reversed(cls.__mro__):
            own = klass.__dict__.get("_parameter_constraints")
            if own and isinstance(own, dict):
                merged.update(own)
        cls._parameter_constraints = merged

    def fit(self, time: pl.Series | None = None, **params: Any) -> BaseSegmenter:
        """Fit the segmenter (no-op by default).

        Parameters
        ----------
        time : pl.Series or None, default=None
            Training timestamps. Ignored by stateless segmenters.
        **params : dict
            Ignored. Present for routing compatibility.

        Returns
        -------
        self

        """
        return self

    @abc.abstractmethod
    def compute_labels(self, time: pl.Series) -> pl.Series:
        """Compute one segment label per element of ``time``.

        Parameters
        ----------
        time : pl.Series
            Timestamps to label.

        Returns
        -------
        pl.Series
            Label series aligned to ``time`` (alias ``"label"``). A null label marks
            an element outside every segment.

        """
        raise NotImplementedError


def _check_label_length(labels: pl.Series, time: pl.Series, owner: str) -> pl.Series:
    """Raise if a component segmenter returned a mis-sized label series."""
    if len(labels) != len(time):
        msg = f"{owner} component returned {len(labels)} labels, expected {len(time)}"
        raise ValueError(msg)
    return labels


class CalendarSegmenter(BaseSegmenter):
    """Segments timestamps by the value of a calendar feature.

    Each timestamp is labelled with its calendar feature value as an integer (0 or 1
    for boolean features). With ``values`` set, only timestamps whose feature value is
    in ``values`` are inside; all others get a null label.

    Parameters
    ----------
    feature : str, default="hour"
        Calendar feature, one of the features of
        [`CalendarFeatureTransformer`][yohou.preprocessing.calendar.CalendarFeatureTransformer]:
        ``"year"``, ``"month"``, ``"week"``, ``"day_of_week"`` (Monday is 1),
        ``"day_of_month"``, ``"day_of_year"``, ``"hour"``, ``"minute"``,
        ``"quarter"``, or a boolean feature such as ``"is_weekend"``.
    values : list, tuple, range, or None, default=None
        Feature values that are inside. ``None`` puts every timestamp inside, each
        labelled by its feature value; an empty list puts every timestamp outside.
        Each value must be an integer the feature can take (e.g. 0 to 23 for
        ``"hour"``), or ``compute_labels`` raises ``ValueError``.
    time_zone : str or None, default=None
        IANA zone in which the feature is read, so ``"hour"`` is a local hour. The
        timestamps must then be timezone-aware. The input series is not modified.

    Notes
    -----
    A window across midnight is a plain list, e.g. ``values=[22, 23, 0, 1]``. With
    ``time_zone`` set, a daylight-saving transition day has 23 or 25 local hours, so
    an hour segment holds one row more or less on those days.

    See Also
    --------
    - [`IntersectionSegmenter`][yohou.segmenting.segmenters.IntersectionSegmenter] : Combine conditions, e.g. weekdays during business hours.
    - [`SegmentWeighter`][yohou.weighting.weighters.SegmentWeighter] : Weight rows by segment membership.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> times = pl.Series("time", [datetime(2024, 1, 1, h) for h in (6, 9, 12, 20)])
    >>> CalendarSegmenter("hour", values=range(8, 19)).compute_labels(times).to_list()
    [None, 9, 12, None]

    """

    _parameter_constraints: dict = {
        "feature": [StrOptions(set(ALL_FEATURES))],
        "values": [list, tuple, range, None],
        "time_zone": [str, None],
    }

    def __init__(
        self,
        feature: str = "hour",
        values: list[int] | tuple[int, ...] | range | None = None,
        time_zone: str | None = None,
    ) -> None:
        self.feature = feature
        self.values = values
        self.time_zone = time_zone

    def _check_key(self, time: pl.Series) -> None:
        """Reject keys the feature cannot be read from."""
        dtype = time.dtype
        if not isinstance(dtype, pl.Datetime | pl.Date):
            msg = (
                f"CalendarSegmenter requires a Date or Datetime series, got dtype {dtype}. "
                "Integer forecasting steps carry no calendar information."
            )
            raise ValueError(msg)
        if isinstance(dtype, pl.Date) and self.feature in _TIME_OF_DAY_FEATURES:
            msg = f"CalendarSegmenter feature {self.feature!r} needs a time of day, but the series has dtype Date"
            raise ValueError(msg)
        if self.time_zone is not None:
            source_zone = dtype.time_zone if isinstance(dtype, pl.Datetime) else None
            if source_zone is None:
                msg = (
                    "CalendarSegmenter with time_zone set requires a timezone-aware series "
                    "(a zone conversion is undefined without an instant and a source zone); "
                    f"got dtype {dtype}. Localize the source first, e.g. dt.replace_time_zone('UTC')."
                )
                raise ValueError(msg)

    def compute_labels(self, time: pl.Series) -> pl.Series:
        """Label each timestamp with its feature value, null when outside ``values``."""
        self._validate_params()
        allowed = None if self.values is None else _check_feature_values(self.feature, self.values)
        self._check_key(time)
        feature = _extract_feature(self.feature, self.time_zone)
        if allowed is not None:
            feature = pl.when(feature.is_in(pl.Series(allowed, dtype=pl.Int32).implode())).then(feature)
        return pl.DataFrame({"time": time}).select(feature.alias("label")).to_series()


class _NamedSegmenterComposition(BaseSegmenter, _BaseComposition):
    """Shared plumbing for segmenters built from named ``(name, segmenter)`` tuples."""

    _parameter_constraints: dict = {"segmenters": [list, None]}

    def __init__(self, segmenters: list[tuple[str, BaseSegmenter]] | None = None) -> None:
        self.segmenters = segmenters

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters, including nested component parameters.

        Parameters
        ----------
        deep : bool, default=True
            If True, include component parameters as ``<name>__<param>``.

        Returns
        -------
        dict
            Parameter names mapped to values.

        """
        return self._get_params("segmenters", deep=deep)

    def set_params(self, **params: Any) -> _NamedSegmenterComposition:
        """Set parameters, routing ``<name>__<param>`` to named sub-segmenters.

        Parameters
        ----------
        **params : dict
            Parameters to set.

        Returns
        -------
        self

        """
        self._set_params("segmenters", **params)
        return self

    @property
    def named_segmenters(self) -> Bunch:
        """Access sub-segmenters by name.

        No trailing underscore: the sub-segmenters are used in place rather than
        cloned, so this is a view over parameters, not fitted state.

        Returns
        -------
        Bunch
            Dictionary-like object with sub-segmenter names as keys.

        """
        return Bunch(**dict(self._check_segmenters()))

    def _check_segmenters(self) -> list[tuple[str, BaseSegmenter]]:
        """Validate the composition parameters and return the named segmenters.

        Not the sklearn ``_validate_params``.
        """
        owner = type(self).__name__
        if not self.segmenters:
            raise ValueError(f"{owner} requires at least one segmenter")
        for item in self.segmenters:
            if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)):
                raise ValueError(f"Each entry in `segmenters` must be a (name, segmenter) tuple, got {item!r}")
        self._validate_names([name for name, _ in self.segmenters])
        return self.segmenters

    @abc.abstractmethod
    def _membership(self, names: list[str]) -> pl.Expr:
        """Expression that is true for rows inside the composite segment."""

    def compute_labels(self, time: pl.Series) -> pl.Series:
        """Label inside rows with a struct of the component labels, null otherwise."""
        self._validate_params()
        segmenters = self._check_segmenters()
        owner = type(self).__name__
        components = pl.DataFrame({
            name: _check_label_length(segmenter.compute_labels(time), time, owner) for name, segmenter in segmenters
        })
        names = [name for name, _ in segmenters]
        # A struct is never null as a whole, even when every field is null, so the
        # outer null that marks an outside row has to be applied explicitly.
        label = pl.when(self._membership(names)).then(pl.struct(names))
        return components.select(label.alias("label")).to_series()


class IntersectionSegmenter(_NamedSegmenterComposition):
    """Rows inside every component segmenter.

    A row is inside only if every component labels it non-null. An inside row's
    label is a struct with one field per component name, holding that component's
    label. Component parameters are addressable as ``<name>__<param>``.

    Parameters
    ----------
    segmenters : list of (str, BaseSegmenter) tuples or None, default=None
        Named component segmenters. Must be non-empty with unique names when labels
        are computed.

    See Also
    --------
    - [`UnionSegmenter`][yohou.segmenting.segmenters.UnionSegmenter] : Rows inside any component.
    - [`ComplementSegmenter`][yohou.segmenting.segmenters.ComplementSegmenter] : Rows outside a segmenter.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> times = pl.Series("time", [datetime(2024, 1, 1, 9), datetime(2024, 1, 6, 9)])  # Monday, Saturday
    >>> business = IntersectionSegmenter([
    ...     ("days", CalendarSegmenter("day_of_week", values=[1, 2, 3, 4, 5])),
    ...     ("hours", CalendarSegmenter("hour", values=range(8, 19))),
    ... ])
    >>> business.compute_labels(times).to_list()
    [{'days': 1, 'hours': 9}, None]

    """

    def _membership(self, names: list[str]) -> pl.Expr:
        """Select rows that every component labels non-null."""
        return pl.all_horizontal([pl.col(name).is_not_null() for name in names])


class UnionSegmenter(_NamedSegmenterComposition):
    """Rows inside any component segmenter.

    A row is inside if at least one component labels it non-null. An inside row's
    label is a struct with one field per component name, null for components that
    do not match the row, so rows matching different combinations of components get
    different labels and equal values from different components never collide.

    Parameters
    ----------
    segmenters : list of (str, BaseSegmenter) tuples or None, default=None
        Named component segmenters. Must be non-empty with unique names when labels
        are computed.

    See Also
    --------
    - [`IntersectionSegmenter`][yohou.segmenting.segmenters.IntersectionSegmenter] : Rows inside every component.
    - [`ComplementSegmenter`][yohou.segmenting.segmenters.ComplementSegmenter] : Rows outside a segmenter.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> times = pl.Series(
    ...     "time", [datetime(2024, 1, 1, 8), datetime(2024, 1, 6, 8), datetime(2024, 1, 1, 14)]
    ... )
    >>> union = UnionSegmenter([
    ...     ("morning", CalendarSegmenter("hour", values=[7, 8, 9])),
    ...     ("weekend", CalendarSegmenter("day_of_week", values=[6, 7])),
    ... ])
    >>> union.compute_labels(times).to_list()
    [{'morning': 8, 'weekend': None}, {'morning': 8, 'weekend': 6}, None]

    """

    def _membership(self, names: list[str]) -> pl.Expr:
        """Select rows that at least one component labels non-null."""
        return pl.any_horizontal([pl.col(name).is_not_null() for name in names])


class ComplementSegmenter(BaseSegmenter):
    """Rows outside a segmenter.

    A row is inside exactly when the wrapped segmenter labels it null. Every inside
    row carries the label ``True``. The wrapped segmenter's parameters are
    addressable as ``segmenter__<param>``.

    Parameters
    ----------
    segmenter : BaseSegmenter or None, default=None
        Segmenter to invert. Required when labels are computed.

    See Also
    --------
    - [`IntersectionSegmenter`][yohou.segmenting.segmenters.IntersectionSegmenter] : Rows inside every component.
    - [`UnionSegmenter`][yohou.segmenting.segmenters.UnionSegmenter] : Rows inside any component.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> times = pl.Series("time", [datetime(2024, 1, 1, h) for h in (6, 12, 20)])
    >>> night = ComplementSegmenter(CalendarSegmenter("hour", values=range(8, 19)))
    >>> night.compute_labels(times).to_list()
    [True, None, True]

    """

    _parameter_constraints: dict = {"segmenter": [HasMethods(["compute_labels"]), None]}

    def __init__(self, segmenter: BaseSegmenter | None = None) -> None:
        self.segmenter = segmenter

    def compute_labels(self, time: pl.Series) -> pl.Series:
        """Label rows the wrapped segmenter leaves outside with ``True``, null otherwise."""
        self._validate_params()
        if self.segmenter is None:
            raise ValueError("ComplementSegmenter requires a non-None `segmenter` to compute labels")
        inner = _check_label_length(self.segmenter.compute_labels(time), time, "ComplementSegmenter")
        return (
            pl
            .DataFrame({"inner": inner})
            .select(pl.when(pl.col("inner").is_null()).then(pl.lit(True)).alias("label"))
            .to_series()
        )
