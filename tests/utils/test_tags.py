"""Tests for the yohou tag system (Tags, InputTags, ForecasterTags, etc.)."""

import pytest

from yohou.utils.tags import (
    ForecasterTags,
    InputTags,
    ScorerTags,
    SimilarityTags,
    SplitterTags,
    Tags,
    TargetTags,
    TransformerTags,
)


class TestTagsPostInit:
    """Tests for Tags.__post_init__ auto-initialization logic."""

    def test_input_and_target_tags_always_initialized(self):
        """InputTags and TargetTags are always created regardless of type."""
        tags = Tags()
        assert isinstance(tags.input_tags, InputTags)
        assert isinstance(tags.target_tags, TargetTags)

    @pytest.mark.parametrize(
        "estimator_type,expected_attr",
        [
            pytest.param("transformer", "transformer_tags", id="transformer"),
            pytest.param("forecaster", "forecaster_tags", id="forecaster"),
            pytest.param("scorer", "scorer_tags", id="scorer"),
            pytest.param("similarity", "similarity_tags", id="similarity"),
            pytest.param("splitter", "splitter_tags", id="splitter"),
        ],
    )
    def test_type_specific_tags_auto_created(self, estimator_type, expected_attr):
        """Type-specific tag group is auto-initialized from estimator_type."""
        tags = Tags(estimator_type=estimator_type)
        assert getattr(tags, expected_attr) is not None

    @pytest.mark.parametrize(
        "estimator_type,absent_attrs",
        [
            pytest.param(
                "transformer",
                ["forecaster_tags", "scorer_tags", "similarity_tags", "splitter_tags"],
                id="transformer-excludes-others",
            ),
            pytest.param(
                "forecaster",
                ["transformer_tags", "scorer_tags", "similarity_tags", "splitter_tags"],
                id="forecaster-excludes-others",
            ),
            pytest.param(
                "scorer",
                ["transformer_tags", "forecaster_tags", "similarity_tags", "splitter_tags"],
                id="scorer-excludes-others",
            ),
        ],
    )
    def test_other_type_tags_remain_none(self, estimator_type, absent_attrs):
        """Non-matching type-specific tag groups stay None."""
        tags = Tags(estimator_type=estimator_type)
        for attr in absent_attrs:
            assert getattr(tags, attr) is None

    def test_no_estimator_type_creates_no_specific_tags(self):
        """Without estimator_type, no type-specific tags are created."""
        tags = Tags()
        assert tags.transformer_tags is None
        assert tags.forecaster_tags is None
        assert tags.scorer_tags is None
        assert tags.similarity_tags is None
        assert tags.splitter_tags is None


class TestTagDefaults:
    """Tests for default values of all tag dataclasses."""

    def test_input_tags_defaults(self):
        """InputTags defaults are correct."""
        t = InputTags()
        assert t.requires_time_column is True
        assert t.pairwise is False
        assert t.allow_nan is False
        assert t.min_value is None

    def test_target_tags_defaults(self):
        """TargetTags defaults are correct."""
        t = TargetTags()
        assert t.required is False
        assert t.min_value is None

    def test_transformer_tags_defaults(self):
        """TransformerTags defaults are correct."""
        t = TransformerTags()
        assert t.stateful is False
        assert t.invertible is False
        assert t.preserves_dtype is False

    def test_forecaster_tags_defaults(self):
        """ForecasterTags defaults are correct."""
        t = ForecasterTags()
        assert t.forecaster_type is None
        assert t.stateful is False
        assert t.uses_reduction is False
        assert t.uses_target_transformer is False
        assert t.uses_feature_transformer is False
        assert t.supports_panel_data is True
        assert t.supports_time_weight is False
        assert t.tracks_observations is True

    def test_scorer_tags_defaults(self):
        """ScorerTags defaults are correct."""
        t = ScorerTags()
        assert t.prediction_type is None
        assert t.lower_is_better is True
        assert t.requires_calibration is False

    def test_similarity_tags_defaults(self):
        """SimilarityTags defaults are correct."""
        t = SimilarityTags()
        assert t.symmetric is True
        assert t.requires_predictions is True
        assert t.produces_weights is True

    def test_splitter_tags_defaults(self):
        """SplitterTags defaults are correct."""
        t = SplitterTags()
        assert t.splitter_type is None
        assert t.supports_panel_data is False
        assert t.produces_non_overlapping_tests is True
        assert t.stateful is False


class TestTagMutability:
    """Tests for tag field mutation."""

    def test_forecaster_tags_mutable(self):
        """ForecasterTags fields can be set after creation."""
        tags = Tags(estimator_type="forecaster")
        assert tags.forecaster_tags is not None
        tags.forecaster_tags.forecaster_type = "point"
        assert tags.forecaster_tags.forecaster_type == "point"

    def test_transformer_tags_mutable(self):
        """TransformerTags fields can be set after creation."""
        tags = Tags(estimator_type="transformer")
        assert tags.transformer_tags is not None
        tags.transformer_tags.stateful = True
        tags.transformer_tags.invertible = True
        assert tags.transformer_tags.stateful is True
        assert tags.transformer_tags.invertible is True

    def test_scorer_tags_mutable(self):
        """ScorerTags fields can be set after creation."""
        tags = Tags(estimator_type="scorer")
        assert tags.scorer_tags is not None
        tags.scorer_tags.prediction_type = "interval"
        tags.scorer_tags.lower_is_better = False
        assert tags.scorer_tags.prediction_type == "interval"
        assert tags.scorer_tags.lower_is_better is False

    def test_top_level_fields_mutable(self):
        """Top-level Tags fields can be set after creation."""
        tags = Tags()
        tags.requires_fit = False
        tags.non_deterministic = True
        assert tags.requires_fit is False
        assert tags.non_deterministic is True


class TestEstimatorTags:
    """Tests that real estimators produce correct tags."""

    def test_point_forecaster_tags(self):
        """Point forecasters have forecaster_type='point'."""
        from yohou.point import SeasonalNaive

        tags = SeasonalNaive().__sklearn_tags__()
        assert tags.estimator_type == "forecaster"
        assert tags.forecaster_tags is not None
        assert tags.forecaster_tags.forecaster_type == "point"
        assert tags.forecaster_tags.supports_panel_data is True

    def test_reduction_forecaster_tags(self):
        """Reduction forecasters have uses_reduction=True."""
        from yohou.point import PointReductionForecaster

        tags = PointReductionForecaster().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        assert tags.forecaster_tags.uses_reduction is True
        assert tags.forecaster_tags.supports_time_weight is True

    def test_interval_forecaster_tags(self):
        """Interval forecasters have forecaster_type='interval' or 'both'."""
        from yohou.interval import SplitConformalForecaster

        tags = SplitConformalForecaster().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        assert tags.forecaster_tags.forecaster_type in ("interval", "both")

    def test_transformer_tags(self):
        """Transformers have correct type and stateful detection."""
        from yohou.stationarity import SeasonalDifferencing

        tags = SeasonalDifferencing(seasonality=3).__sklearn_tags__()
        assert tags.estimator_type == "transformer"
        assert tags.transformer_tags is not None
        assert tags.transformer_tags.stateful is True
        assert tags.transformer_tags.invertible is True

    def test_stateless_transformer_tags(self):
        """Stateless transformers have stateful=False."""
        from yohou.stationarity import LogTransformer

        tags = LogTransformer().__sklearn_tags__()
        assert tags.transformer_tags is not None
        assert tags.transformer_tags.stateful is False

    def test_point_scorer_tags(self):
        """Point scorers have prediction_type='point'."""
        from yohou.metrics import MeanAbsoluteError

        tags = MeanAbsoluteError().__sklearn_tags__()
        assert tags.estimator_type == "scorer"
        assert tags.scorer_tags is not None
        assert tags.scorer_tags.prediction_type == "point"
        assert tags.scorer_tags.lower_is_better is True

    def test_interval_scorer_tags(self):
        """Interval scorers have prediction_type='interval'."""
        from yohou.metrics import IntervalScore

        tags = IntervalScore().__sklearn_tags__()
        assert tags.scorer_tags is not None
        assert tags.scorer_tags.prediction_type == "interval"

    def test_splitter_tags(self):
        """Splitters have correct type and capabilities."""
        from yohou.model_selection import ExpandingWindowSplitter

        tags = ExpandingWindowSplitter().__sklearn_tags__()
        assert tags.estimator_type == "splitter"
        assert tags.splitter_tags is not None
        assert tags.splitter_tags.splitter_type == "expanding"
        assert tags.splitter_tags.supports_panel_data is True
