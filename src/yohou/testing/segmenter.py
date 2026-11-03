"""Check functions for yohou segmenters.

Systematic validation functions for the ``BaseSegmenter`` estimator family
(time-axis segmenters). All check functions raise AssertionError on failure.
"""

import polars as pl

__all__ = [
    "check_segmenter_compute_labels_alignment",
    "check_segmenter_default_constructible",
    "check_segmenter_deterministic",
    "check_segmenter_fit_noop_returns_self",
    "check_segmenter_row_wise",
    "check_segmenter_tags_accessible_before_fit",
    "check_segmenter_tags_static_after_fit",
]


def check_segmenter_compute_labels_alignment(segmenter, time: pl.Series) -> None:
    """Check ``compute_labels`` returns one label per timestamp.

    Parameters
    ----------
    segmenter : BaseSegmenter
        Segmenter instance.
    time : pl.Series
        Timestamps to label.

    Raises
    ------
    AssertionError
        If the output is not a ``pl.Series`` aligned 1:1 with ``time``.

    """
    name = type(segmenter).__name__
    labels = segmenter.compute_labels(time)
    assert isinstance(labels, pl.Series), f"{name}.compute_labels must return pl.Series, got {type(labels)}"
    assert len(labels) == len(time), (
        f"{name}.compute_labels returned {len(labels)} labels, expected {len(time)} (one per timestamp)"
    )


def check_segmenter_fit_noop_returns_self(segmenter, time: pl.Series) -> None:
    """Check ``fit`` is a no-op returning self that leaves params unchanged.

    Parameters
    ----------
    segmenter : BaseSegmenter
        Segmenter instance.
    time : pl.Series
        Timestamps passed to ``fit``.

    Raises
    ------
    AssertionError
        If ``fit`` does not return ``self`` or alters constructor parameters.

    """
    name = type(segmenter).__name__
    before = segmenter.get_params(deep=False)
    returned = segmenter.fit(time)
    assert returned is segmenter, f"{name}.fit() must return self"
    after = segmenter.get_params(deep=False)
    assert before == after, f"{name}.fit() changed parameter values: {before} -> {after}"


def check_segmenter_deterministic(segmenter, time: pl.Series) -> None:
    """Check two ``compute_labels`` calls on the same series agree.

    Parameters
    ----------
    segmenter : BaseSegmenter
        Segmenter instance.
    time : pl.Series
        Timestamps to label.

    Raises
    ------
    AssertionError
        If the two label series differ.

    """
    name = type(segmenter).__name__
    first = segmenter.compute_labels(time)
    second = segmenter.compute_labels(time)
    assert first.equals(second, null_equal=True), f"{name}.compute_labels is not deterministic"


def check_segmenter_row_wise(segmenter, time: pl.Series) -> None:
    """Check a row's label depends only on its own timestamp.

    Labels the two consecutive halves of ``time`` separately and compares them with
    the labels of the whole series. The outlier handlers rely on this property to
    stay batch invariant.

    Parameters
    ----------
    segmenter : BaseSegmenter
        Segmenter instance.
    time : pl.Series
        Timestamps to label; at least two.

    Raises
    ------
    AssertionError
        If labelling in parts gives different labels from labelling the whole.

    """
    name = type(segmenter).__name__
    half = len(time) // 2
    whole = segmenter.compute_labels(time)
    parts = pl.concat([segmenter.compute_labels(time[:half]), segmenter.compute_labels(time[half:])])
    assert whole.equals(parts, null_equal=True), (
        f"{name}.compute_labels depends on the other rows of the series: labelling two halves "
        "separately differs from labelling the whole"
    )


def check_segmenter_default_constructible(segmenter) -> None:
    """Check the segmenter class is constructible with no arguments.

    Parameters
    ----------
    segmenter : BaseSegmenter
        Segmenter instance (used for its type).

    Raises
    ------
    AssertionError
        If ``type(segmenter)()`` raises.

    """
    cls = type(segmenter)
    try:
        cls()
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"{cls.__name__} is not default-constructible: {type(e).__name__}: {e}") from e


def check_segmenter_tags_accessible_before_fit(segmenter) -> None:
    """Check ``__sklearn_tags__()`` is callable on an unfitted segmenter.

    Parameters
    ----------
    segmenter : BaseSegmenter
        Segmenter instance.

    Raises
    ------
    AssertionError
        If tags are not accessible.

    """
    name = type(segmenter).__name__
    assert hasattr(segmenter, "__sklearn_tags__"), f"{name} must have __sklearn_tags__()"
    try:
        segmenter.__sklearn_tags__()
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"{name}.__sklearn_tags__() raised {type(e).__name__}: {e}") from e


def check_segmenter_tags_static_after_fit(segmenter, time: pl.Series) -> None:
    """Check tags are unchanged by the no-op ``fit``.

    Parameters
    ----------
    segmenter : BaseSegmenter
        Unfitted segmenter instance.
    time : pl.Series
        Timestamps passed to ``fit``.

    Raises
    ------
    AssertionError
        If tags change after ``fit``.

    """
    name = type(segmenter).__name__
    before = segmenter.__sklearn_tags__()
    segmenter.fit(time)
    after = segmenter.__sklearn_tags__()
    assert before == after, f"{name} tags changed after fit() - tags should be static"
