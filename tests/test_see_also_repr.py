"""Tests for the numpydoc See Also parser patch in yohou/__init__.py."""

import pytest

# The patch is applied at import time, so just importing yohou activates it.
import yohou  # noqa: F401
from yohou import _BACKTICK_NAME_RE, _BULLET_PREFIX_RE, _MKDOCS_LINK_RE

_has_repr_html = hasattr(__import__("sklearn.utils", fromlist=["_repr_html"]), "_repr_html")


class TestSeeAlsoRegexTransformations:
    """Unit tests for the regex transformations applied to See Also lines."""

    def test_mkdocs_link_with_backticks_collapsed_to_path(self):
        """MkDocs link with backtick display name becomes qualified path."""
        line = (
            "- [`FeaturePipeline`][yohou.compose.feature_pipeline.FeaturePipeline] : Sequential transformer chaining."
        )
        result = _MKDOCS_LINK_RE.sub(r"\1", line)
        assert result == "- yohou.compose.feature_pipeline.FeaturePipeline : Sequential transformer chaining."

    def test_mkdocs_link_without_backticks_collapsed_to_path(self):
        """MkDocs link without backticks in display name becomes qualified path."""
        line = "- [FeaturePipeline][yohou.compose.feature_pipeline.FeaturePipeline] : Description"
        result = _MKDOCS_LINK_RE.sub(r"\1", line)
        assert result == "- yohou.compose.feature_pipeline.FeaturePipeline : Description"

    def test_bullet_prefix_stripped_preserving_indent(self):
        """Leading '- ' is removed while preserving indentation."""
        line = "    - yohou.compose.FeaturePipeline : Description"
        result = _BULLET_PREFIX_RE.sub(r"\1", line)
        assert result == "    yohou.compose.FeaturePipeline : Description"

    def test_bullet_prefix_no_indent(self):
        """Leading '- ' at column 0 is removed."""
        line = "- yohou.path.Class : Desc"
        result = _BULLET_PREFIX_RE.sub(r"\1", line)
        assert result == "yohou.path.Class : Desc"

    def test_backtick_name_stripped(self):
        """Backtick-wrapped names become bare names."""
        line = "`sklearn.pipeline.FeatureUnion` : Description"
        result = _BACKTICK_NAME_RE.sub(r"\1", line)
        assert result == "sklearn.pipeline.FeatureUnion : Description"

    def test_full_pipeline_bullet_mkdocs_link(self):
        """Full three-step pipeline transforms bullet MkDocs link to numpydoc format."""
        line = "    - [`BaseActualTransformer`][yohou.base.transformer.BaseActualTransformer] : Base class."
        # Step 1: collapse MkDocs link
        result = _MKDOCS_LINK_RE.sub(r"\1", line)
        # Step 2: strip bullet
        result = _BULLET_PREFIX_RE.sub(r"\1", result)
        # Step 3: strip backticks (no-op in this case since step 1 removed them)
        result = _BACKTICK_NAME_RE.sub(r"\1", result)
        assert result == "    yohou.base.transformer.BaseActualTransformer : Base class."

    def test_backtick_only_line_unchanged_by_mkdocs_and_bullet_regexes(self):
        """A backtick-only line (no bullet, no MkDocs link) passes through steps 1-2 unchanged."""
        line = "`sklearn.preprocessing.PowerTransformer` : sklearn's power transforms."
        result = _MKDOCS_LINK_RE.sub(r"\1", line)
        result = _BULLET_PREFIX_RE.sub(r"\1", result)
        assert result == "`sklearn.preprocessing.PowerTransformer` : sklearn's power transforms."
        # Step 3 handles it
        result = _BACKTICK_NAME_RE.sub(r"\1", result)
        assert result == "sklearn.preprocessing.PowerTransformer : sklearn's power transforms."


@pytest.mark.skipif(not _has_repr_html, reason="sklearn.utils._repr_html not available (sklearn < 1.8)")
class TestSeeAlsoHtmlRepr:
    """Integration tests that HTML repr works for estimators with various See Also formats."""

    def test_bullet_mkdocs_see_also_repr(self):
        """Estimator with bullet-style MkDocs See Also links renders HTML without error."""
        from sklearn.utils._repr_html.estimator import estimator_html_repr

        from yohou.stationarity import PatternSeasonalityForecaster

        obj = PatternSeasonalityForecaster(seasonality=7)
        html = estimator_html_repr(obj)
        assert len(html) > 0

    def test_mixed_format_see_also_repr(self):
        """Estimator with mixed See Also (backtick-only + bullet MkDocs links) renders HTML."""
        from sklearn.utils._repr_html.estimator import estimator_html_repr

        from yohou.compose import FeatureUnion

        obj = FeatureUnion(transformer_list=[])
        html = estimator_html_repr(obj)
        assert len(html) > 0

    def test_docstrings_unchanged_after_import(self):
        """The __doc__ attributes retain original MkDocs link syntax."""
        from yohou.compose import FeatureUnion

        doc = FeatureUnion.__doc__
        assert "[`FeaturePipeline`][yohou.compose.feature_pipeline.FeaturePipeline]" in doc

    def test_docstrings_unchanged_stationarity(self):
        """Stationarity docstrings retain original format."""
        from yohou.stationarity import PatternSeasonalityForecaster

        doc = PatternSeasonalityForecaster.__doc__
        assert "[`FourierSeasonalityForecaster`]" in doc
