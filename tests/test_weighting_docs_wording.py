"""Wording checks for the sample-weight alignment documentation.

The weighting pages and the time-weighted reduction notebook used to say that a
row's per-timestep weights "must be collapsed" and that the combined weights sum
to "the number of samples". Both statements are false for
``sample_weight_alignment="matched_step"``, which keeps one weight per step and
normalizes per step and per panel group. These tests keep every remaining mention
qualified to the collapse values, and check that the pages state what the
``matched_step`` documentation requirements ask for.

Text is compared with whitespace normalized, so reflowing a paragraph never breaks
a test; rewording one of the allowed phrases does, on purpose. Update the phrase
here in the same change, after checking that the new sentence is still qualified.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WEIGHTING = "docs/pages/explanation/weighting.md"
REDUCTION = "docs/pages/explanation/reduction-forecasting.md"
HOWTO = "docs/pages/how-to/time-weighting.md"
NOTEBOOK = "examples/forecasting-models/time_weighted_reduction.py"

STALE = re.compile(r"collaps|number of samples|five", re.IGNORECASE)

# Every match of STALE in a file must lie inside one of that file's phrases.
ALLOWED = {
    WEIGHTING: [
        "the collapse values reduce them to one scalar",
        "the first five strategies collapse the window into a scalar for it",
        "so they need no collapse at all",
        "No collapse: the model for step `h`",
        "Under a collapse strategy, every per-step model",
        "A collapse strategy only hid that",
        "to a per-sample weight without collapsing",
        "Under a collapse strategy, the combined weights are then renormalized so that their sum equals "
        "the number of samples",
    ],
    REDUCTION: [
        "so the collapse values reduce the window to a single weight",
        "instead of a collapsed one",
        "Under a collapse value they are normalized so their sum equals the number of samples",
    ],
    HOWTO: [
        'With the default `"multi-output"` strategy those weights must be collapsed into one weight per sample',
        'With a collapse value such as `"first_step"`',
        "or use a collapse value",
    ],
    NOTEBOOK: [
        "so the first five strategies collapse the window into a single `sample_weight`",
        'No collapse: under `"direct"` or `"dir-rec"`',
        'runs the five collapse strategies under the default `"multi-output"` strategy',
        "Under a collapse strategy they all receive the same row weights",
    ],
}


def _text(path: str) -> str:
    return re.sub(r"\s+", " ", (ROOT / path).read_text())


def _section(path: str, heading: str, next_heading: str) -> str:
    text = _text(path)
    start = text.index(heading)
    return text[start : text.index(next_heading, start)]


@pytest.mark.parametrize("path", list(ALLOWED))
def test_stale_wording_is_qualified(path):
    """Every mention of collapsing, the sample count, or "five" sits in an allowed phrase."""
    text = _text(path)
    spans = []
    for phrase in ALLOWED[path]:
        found = [m.span() for m in re.finditer(re.escape(phrase), text)]
        assert found, f"allowed phrase no longer in {path}: {phrase!r}"
        spans.extend(found)
    stray = [
        text[max(0, m.start() - 60) : m.end() + 40]
        for m in STALE.finditer(text)
        if not any(start <= m.start() and m.end() <= end for start, end in spans)
    ]
    assert stray == [], f"unqualified wording in {path}: {stray}"


class TestExplanationPages:
    """Both explanation pages describe the per-step alignment and its parameter."""

    @pytest.mark.parametrize("path", [WEIGHTING, REDUCTION])
    def test_name_the_value_and_the_decay_parameter(self, path):
        text = _text(path)
        assert '"matched_step"' in text
        assert "sample_weight_horizon_decay" in text

    def test_weighting_table_has_six_rows(self):
        section = _section(WEIGHTING, "## The Alignment Problem", "## Composition and Normalization")
        assert "Six strategies are available" in section
        assert len(re.findall(r'\| `"\w+_step"`', section)) == 6

    def test_weighting_states_normalization_and_multi_output_limit(self):
        text = _text(WEIGHTING)
        assert "the same rule applies per step and per panel group" in text
        assert 'It cannot apply to `"multi-output"`, because that estimator receives a single weight per row' in text

    def test_weighting_alignment_section_explains_the_zero_step_error(self):
        section = _section(WEIGHTING, "## The Alignment Problem", "## Composition and Normalization")
        assert "raises a `ValueError` at fit, because that step has nothing to learn from" in section

    def test_reduction_states_normalization_and_multi_output_limit(self):
        text = _text(REDUCTION)
        assert '`"matched_step"` the normalization is per step and per panel group' in text
        assert 'A `"multi-output"` estimator takes one weight per sample' in text
        assert 'it falls back to `"first_step"` with a warning' in text


class TestHowTo:
    """Section 5 of the time-weighting how-to carries the matched_step recipe."""

    SECTION = ("## 5. Apply Weights During Training", "## 6. Apply Weights During Scoring")

    def test_recipe_uses_direct_seasonal_and_matched_step(self):
        section = _section(HOWTO, *self.SECTION)
        assert 'reduction_strategy="direct"' in section
        assert "time_weighter=SeasonalEmphasisWeighter(" in section
        assert 'sample_weight_alignment="matched_step"' in section

    def test_explains_shift_strategy_requirement_and_zero_step(self):
        section = _section(HOWTO, *self.SECTION)
        assert "the model for step 12 emphasizes the rows eleven steps before the marked season" in section
        assert '`"matched_step"` requires `reduction_strategy="direct"` or `"dir-rec"`' in section
        assert "that step has nothing to learn from" in section


class TestNotebook:
    """The companion notebook compares first_step with matched_step under direct."""

    def test_direct_cell_compares_first_and_matched_step(self):
        text = _text(NOTEBOOK)
        assert 'for _align in ["first_step", "matched_step"]:' in text
        assert 'reduction_strategy="direct",' in text

    def test_looped_comparison_stays_at_the_five_collapse_values(self):
        text = _text(NOTEBOOK)
        loop = text[text.index("_alignments = [") : text.index("]", text.index("_alignments = ["))]
        assert len(re.findall(r'"\w+_step"', loop)) == 5
        assert "matched_step" not in loop
        assert 'where `"matched_step"` would fall back to `"first_step"` with a warning' in text
