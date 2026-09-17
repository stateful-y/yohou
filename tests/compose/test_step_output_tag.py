"""Composites declare ``produces_step_columns`` exactly when a child does."""

import pytest

from yohou.compose import ColumnTransformer, FeaturePipeline, FeatureUnion
from yohou.preprocessing.window import LagTransformer, RollingStatisticsTransformer


class _StepOutputProbe(LagTransformer):
    """LagTransformer that declares step-column output, for tag aggregation only."""

    _tags = {"produces_step_columns": True}


def _declares(estimator) -> bool:
    return estimator.__sklearn_tags__().transformer_tags.produces_step_columns


def test_leaf_default_is_false():
    """Ordinary transformers do not declare step-column output."""
    assert not _declares(LagTransformer(lag=1))
    assert _declares(_StepOutputProbe(lag=1))


@pytest.mark.parametrize(
    ("make_composite", "expected"),
    [
        (lambda: FeatureUnion([("probe", _StepOutputProbe(lag=1)), ("lag", LagTransformer(lag=2))]), True),
        (lambda: FeatureUnion([("lag", LagTransformer(lag=1)), ("roll", RollingStatisticsTransformer())]), False),
        (
            lambda: ColumnTransformer([("probe", _StepOutputProbe(lag=1), ["a"]), ("lag", LagTransformer(), ["b"])]),
            True,
        ),
        (lambda: ColumnTransformer([("lag", LagTransformer(lag=1), ["a"])]), False),
        (lambda: FeaturePipeline([("roll", RollingStatisticsTransformer()), ("probe", _StepOutputProbe(lag=1))]), True),
        (lambda: FeaturePipeline([("lag", LagTransformer(lag=1)), ("roll", RollingStatisticsTransformer())]), False),
        (
            lambda: FeatureUnion([
                ("nested", FeaturePipeline([("probe", _StepOutputProbe(lag=1))])),
                ("lag", LagTransformer(lag=1)),
            ]),
            True,
        ),
    ],
    ids=[
        "union-with-probe",
        "union-without",
        "column-with-probe",
        "column-without",
        "pipeline-with-probe",
        "pipeline-without",
        "nested",
    ],
)
def test_composite_declares_iff_a_child_does(make_composite, expected):
    """A composite's tag is the disjunction of its children's, before fit."""
    assert _declares(make_composite()) is expected
