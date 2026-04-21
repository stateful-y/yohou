"""Unit tests for facet_figure column-mode faceting."""

import pytest

pytest.importorskip("plotly", reason="plotting extra not installed")

import polars as pl
from plotly import graph_objects as go

from yohou.plotting._utils import RenderContext, build_category_map, facet_figure


@pytest.fixture
def col_df():
    """Small DataFrame with 3 value columns for column-mode faceting tests."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), "1d", eager=True),
        "a": list(range(10)),
        "b": [x * 2 for x in range(10)],
        "c": [x * 3 for x in range(10)],
    })


class TestFacetFigureColumnsDispatch:
    """Tests that facet_figure dispatches to column mode when groups is None."""

    def test_dispatch_without_groups(self, col_df):
        """Omitting groups triggers column-mode faceting."""
        contexts: list[RenderContext] = []

        def _capture(ctx: RenderContext) -> None:
            contexts.append(ctx)

        facet_figure(col_df, _capture, columns=["a", "b"])
        assert len(contexts) == 2
        assert all(ctx.facet_by == "column" for ctx in contexts)

    def test_dispatch_with_groups(self):
        """Providing groups triggers panel-mode faceting."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 5), "1d", eager=True),
            "y__a": [1, 2, 3, 4, 5],
            "y__b": [5, 4, 3, 2, 1],
        })
        contexts: list[RenderContext] = []

        def _capture(ctx: RenderContext) -> None:
            contexts.append(ctx)

        facet_figure(df, _capture, groups=["y"])
        assert len(contexts) == 2
        assert all(ctx.facet_by == "member" for ctx in contexts)


class TestFacetFigureColumnsSingleColumn:
    """Tests for column-mode faceting with a single column."""

    def test_single_column_string(self, col_df):
        """A single column string produces one subplot."""
        contexts: list[RenderContext] = []

        def _capture(ctx: RenderContext) -> None:
            contexts.append(ctx)

        fig = facet_figure(col_df, _capture, columns="a")
        assert len(contexts) == 1
        ctx = contexts[0]
        assert ctx.display_name == "a"
        assert ctx.row == 1
        assert ctx.col == 1
        assert ctx.facet_by == "column"
        assert ctx.entity_idx == 0
        assert list(ctx.sub_df.columns) == ["time", "a"]
        assert isinstance(fig, go.Figure)

    def test_single_column_list(self, col_df):
        """A single-element list produces one subplot."""
        contexts: list[RenderContext] = []

        def _capture(ctx: RenderContext) -> None:
            contexts.append(ctx)

        facet_figure(col_df, _capture, columns=["b"])
        assert len(contexts) == 1
        assert contexts[0].display_name == "b"


class TestFacetFigureColumnsMultiColumn:
    """Tests for column-mode faceting with multiple columns."""

    def test_two_columns_grid(self, col_df):
        """Two columns produce a 1-row x 2-col grid."""
        contexts: list[RenderContext] = []

        def _capture(ctx: RenderContext) -> None:
            contexts.append(ctx)

        facet_figure(col_df, _capture, columns=["a", "b"], facet_n_cols=2)
        assert len(contexts) == 2
        assert (contexts[0].row, contexts[0].col) == (1, 1)
        assert (contexts[1].row, contexts[1].col) == (1, 2)

    def test_three_columns_wraps_to_rows(self, col_df):
        """Three columns with facet_n_cols=2 wraps to a 2x2 grid."""
        contexts: list[RenderContext] = []

        def _capture(ctx: RenderContext) -> None:
            contexts.append(ctx)

        facet_figure(col_df, _capture, columns=["a", "b", "c"], facet_n_cols=2)
        assert len(contexts) == 3
        assert (contexts[0].row, contexts[0].col) == (1, 1)
        assert (contexts[1].row, contexts[1].col) == (1, 2)
        assert (contexts[2].row, contexts[2].col) == (2, 1)

    def test_sub_df_contains_correct_column(self, col_df):
        """Each context sub_df contains the expected column plus time."""
        contexts: list[RenderContext] = []

        def _capture(ctx: RenderContext) -> None:
            contexts.append(ctx)

        facet_figure(col_df, _capture, columns=["a", "c"])
        assert list(contexts[0].sub_df.columns) == ["time", "a"]
        assert list(contexts[1].sub_df.columns) == ["time", "c"]

    def test_display_names_match_column_names(self, col_df):
        """display_name equals the column name for each facet."""
        contexts: list[RenderContext] = []

        def _capture(ctx: RenderContext) -> None:
            contexts.append(ctx)

        facet_figure(col_df, _capture, columns=["a", "b", "c"])
        names = [ctx.display_name for ctx in contexts]
        assert names == ["a", "b", "c"]


class TestFacetFigureColumnsDefault:
    """Tests for column-mode faceting with no columns specified (all non-time)."""

    def test_default_all_columns(self, col_df):
        """Omitting columns uses all non-time columns."""
        contexts: list[RenderContext] = []

        def _capture(ctx: RenderContext) -> None:
            contexts.append(ctx)

        facet_figure(col_df, _capture)
        names = [ctx.display_name for ctx in contexts]
        assert names == ["a", "b", "c"]

    def test_default_single_column(self):
        """Single-value-column DataFrame still creates one subplot."""
        df = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 5), "1d", eager=True),
            "y": [1, 2, 3, 4, 5],
        })
        contexts: list[RenderContext] = []

        def _capture(ctx: RenderContext) -> None:
            contexts.append(ctx)

        facet_figure(df, _capture)
        assert len(contexts) == 1
        assert contexts[0].display_name == "y"


class TestFacetFigureColumnGroups:
    """Tests for column_groups parameter (grouped columns per subplot)."""

    def test_column_groups_basic(self, col_df):
        """column_groups maps facet names to multiple columns in sub_df."""
        contexts: list[RenderContext] = []

        def _capture(ctx: RenderContext) -> None:
            contexts.append(ctx)

        groups = {"first": ["a", "b"], "second": ["c"]}
        facet_figure(col_df, _capture, column_groups=groups)
        assert len(contexts) == 2
        assert contexts[0].display_name == "first"
        assert list(contexts[0].sub_df.columns) == ["time", "a", "b"]
        assert contexts[1].display_name == "second"
        assert list(contexts[1].sub_df.columns) == ["time", "c"]

    def test_column_groups_single_group(self, col_df):
        """A single column_group produces one subplot with multiple columns."""
        contexts: list[RenderContext] = []

        def _capture(ctx: RenderContext) -> None:
            contexts.append(ctx)

        facet_figure(col_df, _capture, column_groups={"all": ["a", "b", "c"]})
        assert len(contexts) == 1
        assert list(contexts[0].sub_df.columns) == ["time", "a", "b", "c"]

    def test_columns_and_column_groups_mutual_exclusion(self, col_df):
        """Specifying both columns and column_groups raises ValueError."""

        def _noop(ctx: RenderContext) -> None:
            pass

        with pytest.raises(ValueError, match="Cannot specify both"):
            facet_figure(col_df, _noop, columns=["a"], column_groups={"x": ["b"]})


class TestFacetFigureColumnsLayout:
    """Tests for layout properties produced by column-mode faceting."""

    def test_title_applied(self, col_df):
        """Title is set on the figure layout."""
        fig = facet_figure(col_df, lambda ctx: None, columns=["a"], title="My Title")
        assert fig.layout.title.text == "My Title"

    def test_default_height_scales_with_rows(self, col_df):
        """Height grows with the number of subplot rows."""
        fig_1row = facet_figure(col_df, lambda ctx: None, columns=["a", "b"], facet_n_cols=2)
        fig_2rows = facet_figure(col_df, lambda ctx: None, columns=["a", "b", "c"], facet_n_cols=2)
        assert fig_2rows.layout.height > fig_1row.layout.height

    def test_explicit_height_overrides(self, col_df):
        """Explicit height overrides the row_height-based default."""
        fig = facet_figure(col_df, lambda ctx: None, columns=["a", "b", "c"], height=500)
        assert fig.layout.height == 500

    def test_explicit_width(self, col_df):
        """Explicit width is applied to the figure."""
        fig = facet_figure(col_df, lambda ctx: None, columns=["a"], width=800)
        assert fig.layout.width == 800

    def test_subplot_titles_match_column_names(self, col_df):
        """Subplot annotation titles match the column names."""
        fig = facet_figure(col_df, lambda ctx: None, columns=["a", "b"])
        annotation_texts = [a.text for a in fig.layout.annotations]
        assert "a" in annotation_texts
        assert "b" in annotation_texts

    def test_column_groups_subplot_titles(self, col_df):
        """Subplot titles match the column_groups keys."""
        groups = {"First Group": ["a", "b"], "Second Group": ["c"]}
        fig = facet_figure(col_df, lambda ctx: None, column_groups=groups)
        annotation_texts = [a.text for a in fig.layout.annotations]
        assert "First Group" in annotation_texts
        assert "Second Group" in annotation_texts

    def test_shared_xaxes_default(self, col_df):
        """shared_xaxes=True by default produces a valid figure."""
        fig = facet_figure(col_df, lambda ctx: None, columns=["a", "b", "c"], facet_n_cols=1)
        assert isinstance(fig, go.Figure)

    def test_render_callback_adds_traces(self, col_df):
        """Traces added in the render callback appear in the figure."""

        def _render(ctx: RenderContext) -> None:
            ctx.fig.add_scatter(
                x=ctx.sub_df["time"],
                y=ctx.sub_df[ctx.display_name],
                name=ctx.display_name,
                row=ctx.row,
                col=ctx.col,
            )

        fig = facet_figure(col_df, _render, columns=["a", "b"])
        assert len(fig.data) == 2
        trace_names = {t.name for t in fig.data}
        assert trace_names == {"a", "b"}


class TestFacetFigureColumnsContextFields:
    """Tests that RenderContext fields are set correctly in column mode."""

    def test_facet_name_equals_column_name(self, col_df):
        """facet_name equals the column name in column mode."""
        contexts: list[RenderContext] = []

        def _capture(ctx: RenderContext) -> None:
            contexts.append(ctx)

        facet_figure(col_df, _capture, columns=["a", "c"])
        assert contexts[0].facet_name == "a"
        assert contexts[1].facet_name == "c"

    def test_group_and_member_name_equal_column(self, col_df):
        """group_name and member_name both equal the column name in column mode."""
        contexts: list[RenderContext] = []

        def _capture(ctx: RenderContext) -> None:
            contexts.append(ctx)

        facet_figure(col_df, _capture, columns=["b"])
        ctx = contexts[0]
        assert ctx.group_name == "b"
        assert ctx.member_name == "b"

    def test_context_is_frozen(self, col_df):
        """RenderContext is a frozen dataclass (immutable)."""
        contexts: list[RenderContext] = []

        def _capture(ctx: RenderContext) -> None:
            contexts.append(ctx)

        facet_figure(col_df, _capture, columns=["a"])
        with pytest.raises(AttributeError, match="cannot assign"):
            contexts[0].display_name = "modified"

    def test_column_groups_display_name_is_group_key(self, col_df):
        """display_name is the column_groups key, not the individual column names."""
        contexts: list[RenderContext] = []

        def _capture(ctx: RenderContext) -> None:
            contexts.append(ctx)

        facet_function_groups = {"combo": ["a", "b"]}
        facet_figure(col_df, _capture, column_groups=facet_function_groups)
        assert contexts[0].display_name == "combo"
        assert contexts[0].facet_name == "combo"


class TestBuildCategoryMap:
    """Tests for the build_category_map utility function."""

    def test_single_series(self):
        """Returns sorted categories and correct mapping from one series."""
        s = pl.Series("x", ["B", "A", "C", "A"])
        sorted_cats, cat_to_int = build_category_map(s)
        assert sorted_cats == ["A", "B", "C"]
        assert cat_to_int == {"A": 0, "B": 1, "C": 2}

    def test_multiple_series(self):
        """Collects unique categories across multiple series."""
        s1 = pl.Series("a", ["X", "Y"])
        s2 = pl.Series("b", ["Y", "Z"])
        sorted_cats, cat_to_int = build_category_map(s1, s2)
        assert sorted_cats == ["X", "Y", "Z"]
        assert cat_to_int == {"X": 0, "Y": 1, "Z": 2}

    def test_nulls_dropped(self):
        """Null values are excluded from the category map."""
        s = pl.Series("x", ["A", None, "B", None])
        sorted_cats, cat_to_int = build_category_map(s)
        assert sorted_cats == ["A", "B"]
        assert cat_to_int == {"A": 0, "B": 1}

    def test_empty_series(self):
        """Empty series returns empty results."""
        s = pl.Series("x", [], dtype=pl.String)
        sorted_cats, cat_to_int = build_category_map(s)
        assert sorted_cats == []
        assert cat_to_int == {}

    def test_duplicates_across_series(self):
        """Duplicate values across series produce unique categories."""
        s1 = pl.Series("a", ["A", "B", "A"])
        s2 = pl.Series("b", ["B", "C", "B"])
        sorted_cats, cat_to_int = build_category_map(s1, s2)
        assert sorted_cats == ["A", "B", "C"]
        assert len(cat_to_int) == 3

    def test_single_value(self):
        """Single value produces mapping with one entry."""
        s = pl.Series("x", ["only"])
        sorted_cats, cat_to_int = build_category_map(s)
        assert sorted_cats == ["only"]
        assert cat_to_int == {"only": 0}
