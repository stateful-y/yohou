"""Tests for cross-validation plotting functions."""

import pytest

pytest.importorskip("plotly", reason="plotting extra not installed")


import polars as pl
import pytest

from yohou.model_selection import ExpandingWindowSplitter, SlidingWindowSplitter
from yohou.plotting import plot_cv_results_scatter, plot_nested_splits, plot_splits

from .conftest import assert_figure_valid, assert_layout


@pytest.fixture
def sample_y():
    """Create a sample time series DataFrame for CV split plots."""
    return pl.DataFrame({
        "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), "1d", eager=True),
        "value": list(range(366)),
    })


@pytest.fixture
def sample_cv_results():
    """Create a sample cv_results_ dictionary."""
    return {
        "param_alpha": [0.01, 0.1, 1.0, 10.0],
        "mean_test_score": [-0.5, -0.3, -0.2, -0.4],
        "std_test_score": [0.05, 0.03, 0.02, 0.06],
        "rank_test_score": [3, 2, 1, 4],
    }


class TestPlotSplits:
    """Tests for plot_splits function."""

    def test_basic_expanding_window(self, sample_y):
        """Test basic expanding window split visualization."""
        splitter = ExpandingWindowSplitter(n_splits=3, test_size=30)
        fig = plot_splits(sample_y, splitter)
        assert_figure_valid(fig)

    def test_basic_sliding_window(self, sample_y):
        """Test basic sliding window split visualization."""
        splitter = SlidingWindowSplitter(n_splits=3, test_size=30)
        fig = plot_splits(sample_y, splitter)
        assert_figure_valid(fig)

    def test_has_train_and_test_traces(self, sample_y):
        """Test that figure has both train and test traces."""
        splitter = ExpandingWindowSplitter(n_splits=2, test_size=30)
        fig = plot_splits(sample_y, splitter)
        # Each fold has at least a train trace and a test trace
        # 2 folds × 2 traces (train + test) = 4 traces minimum
        assert len(fig.data) >= 4

    def test_custom_colors(self, sample_y):
        """Test custom train/test colors."""
        splitter = ExpandingWindowSplitter(n_splits=2, test_size=30)
        fig = plot_splits(
            sample_y,
            splitter,
            train_color="#FF0000",
            test_color="#0000FF",
        )
        assert len(fig.data) > 0

    def test_custom_title_and_labels(self, sample_y):
        """Test custom title and axis labels."""
        splitter = ExpandingWindowSplitter(n_splits=2, test_size=30)
        fig = plot_splits(
            sample_y,
            splitter,
            title="My Splits",
            x_label="Date",
            y_label="Split",
        )
        assert_layout(fig, title="My Splits")

    def test_custom_dimensions(self, sample_y):
        """Test custom width and height."""
        splitter = ExpandingWindowSplitter(n_splits=2, test_size=30)
        fig = plot_splits(sample_y, splitter, width=800, height=400)
        assert_layout(fig, width=800, height=400)

    def test_kwargs_line_width(self, sample_y):
        """Test line_width kwarg."""
        splitter = ExpandingWindowSplitter(n_splits=2, test_size=30)
        fig = plot_splits(sample_y, splitter, line_width=5.0)
        assert len(fig.data) > 0

    def test_with_exogenous(self, sample_y):
        """Test split plot with X parameter."""
        X = pl.DataFrame({
            "time": sample_y["time"],
            "feature": list(range(len(sample_y))),
        })
        splitter = ExpandingWindowSplitter(n_splits=2, test_size=30)
        fig = plot_splits(sample_y, splitter, X_actual=X)
        assert len(fig.data) > 0

    def test_invalid_splitter_type(self, sample_y):
        """Test that invalid splitter type raises TypeError."""
        with pytest.raises(TypeError, match="Expected BaseSplitter"):
            plot_splits(sample_y, "not_a_splitter")

    def test_empty_dataframe(self):
        """Test that empty DataFrame raises error."""
        empty_df = pl.DataFrame({"time": [], "value": []}).cast({"time": pl.Date})
        splitter = ExpandingWindowSplitter(n_splits=2, test_size=5)
        with pytest.raises(ValueError, match="empty|rows|split"):
            plot_splits(empty_df, splitter)

    def test_two_splits(self, sample_y):
        """Test with two splits (minimum for ExpandingWindowSplitter)."""
        splitter = ExpandingWindowSplitter(n_splits=2, test_size=30)
        fig = plot_splits(sample_y, splitter)
        assert len(fig.data) >= 4  # At least 2 × (train + test)

    def test_default_legend_names_unchanged(self, sample_y):
        """The default legend labels remain 'Train' / 'Test'."""
        splitter = ExpandingWindowSplitter(n_splits=2, test_size=30)
        fig = plot_splits(sample_y, splitter)
        legend_names = [t.name for t in fig.data if t.showlegend]
        assert legend_names == ["Train", "Test"]

    def test_custom_train_test_names(self, sample_y):
        """train_name / test_name override the legend labels without renaming traces."""
        splitter = ExpandingWindowSplitter(n_splits=2, test_size=30)
        fig = plot_splits(sample_y, splitter, train_name="Fit", test_name="Scored")
        legend_names = [t.name for t in fig.data if t.showlegend]
        assert legend_names == ["Fit", "Scored"]


class TestPlotNestedSplits:
    """Tests for plot_nested_splits function."""

    def test_basic_single_fold(self, sample_y):
        """Single outer fold renders inner folds plus the outer evaluation row."""
        fig = plot_nested_splits(
            sample_y,
            ExpandingWindowSplitter(n_splits=3, test_size=30),
            ExpandingWindowSplitter(n_splits=2, test_size=20),
        )
        # 2 inner folds × (train + test) + outer row (train + test) = 6 traces
        assert_figure_valid(fig, min_traces=6)

    def test_outer_row_present(self, sample_y):
        """The outer evaluation row shows the refit-train and the scored outer test."""
        fig = plot_nested_splits(
            sample_y,
            ExpandingWindowSplitter(n_splits=3, test_size=30),
            ExpandingWindowSplitter(n_splits=2, test_size=20),
        )
        legend_names = [t.name for t in fig.data if t.showlegend]
        assert "Outer train (refit, best params)" in legend_names
        assert "Outer test (scored)" in legend_names
        rows = {label for t in fig.data for label in t.y}
        assert "Outer fold" in rows

    def test_outer_train_spans_full_training_window(self, sample_y):
        """The outer refit-train segment covers the whole outer training window."""
        outer = ExpandingWindowSplitter(n_splits=3, test_size=30)
        train_idx, _ = list(outer.split(sample_y))[-1]
        expected_start = sample_y["time"][int(train_idx[0])]
        expected_end = sample_y["time"][int(train_idx[-1])]
        fig = plot_nested_splits(sample_y, outer, ExpandingWindowSplitter(n_splits=2, test_size=20))
        outer_train = next(t for t in fig.data if t.name == "Outer train (refit, best params)")
        assert outer_train.x[0] == expected_start
        assert outer_train.x[-1] == expected_end

    def test_faceted_all_folds(self, sample_y):
        """outer_fold='all' renders one subplot per outer fold, legend shown once."""
        outer = ExpandingWindowSplitter(n_splits=3, test_size=30)
        fig = plot_nested_splits(
            sample_y,
            outer,
            ExpandingWindowSplitter(n_splits=2, test_size=20),
            outer_fold="all",
        )
        assert_figure_valid(fig)
        # One subplot per outer fold (3) -> at least 3 x-axes in the layout.
        x_axes = [k for k in fig.layout if k.startswith("xaxis")]
        assert len(x_axes) >= 3
        # Each of the four legend entries appears once across the subplots.
        legend_names = [t.name for t in fig.data if t.showlegend]
        assert sorted(legend_names) == [
            "Inner train",
            "Inner validation",
            "Outer test (scored)",
            "Outer train (refit, best params)",
        ]

    def test_custom_colors_and_names(self, sample_y):
        """Colors and segment names are configurable."""
        fig = plot_nested_splits(
            sample_y,
            ExpandingWindowSplitter(n_splits=3, test_size=30),
            ExpandingWindowSplitter(n_splits=2, test_size=20),
            train_color="#111111",
            test_color="#222222",
            outer_train_color="#333333",
            outer_test_color="#444444",
            train_name="itr",
            test_name="ival",
            outer_train_name="otr",
            outer_test_name="otest",
        )
        legend_names = [t.name for t in fig.data if t.showlegend]
        assert legend_names == ["itr", "ival", "otr", "otest"]

    def test_sliding_window_outer(self, sample_y):
        """A non-prefix outer training set (sliding window) renders without error."""
        fig = plot_nested_splits(
            sample_y,
            SlidingWindowSplitter(n_splits=3, test_size=30),
            ExpandingWindowSplitter(n_splits=2, test_size=20),
            outer_fold=-1,
        )
        assert_figure_valid(fig)

    def test_invalid_outer_fold_index(self, sample_y):
        """An out-of-range integer outer_fold raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            plot_nested_splits(
                sample_y,
                ExpandingWindowSplitter(n_splits=3, test_size=30),
                ExpandingWindowSplitter(n_splits=2, test_size=20),
                outer_fold=99,
            )

    def test_invalid_splitter_type(self, sample_y):
        """A non-BaseSplitter argument raises TypeError naming the argument."""
        with pytest.raises(TypeError, match="inner_splitter"):
            plot_nested_splits(
                sample_y,
                ExpandingWindowSplitter(n_splits=3, test_size=30),
                "not_a_splitter",
            )

    def test_invalid_outer_fold_value(self, sample_y):
        """A non-integer outer_fold other than 'all' raises ValueError."""
        with pytest.raises(ValueError, match="must be an int or 'all'"):
            plot_nested_splits(
                sample_y,
                ExpandingWindowSplitter(n_splits=3, test_size=30),
                ExpandingWindowSplitter(n_splits=2, test_size=20),
                outer_fold="last",
            )

    def test_faceted_inner_split_too_small_raises(self, sample_y):
        """An outer fold whose training slice is too small for the inner splitter raises a fold-indexed error."""
        with pytest.raises(ValueError, match="inner split failed for outer fold"):
            plot_nested_splits(
                sample_y,
                ExpandingWindowSplitter(n_splits=3, test_size=30),
                ExpandingWindowSplitter(n_splits=2, test_size=200),  # needs 400 rows, first fold has ~276
                outer_fold="all",
            )

    def test_with_x_actual(self, sample_y):
        """X_actual is sliced to the outer training window in both single-fold and faceted modes."""
        x_actual = pl.DataFrame({"time": sample_y["time"], "feature": list(range(len(sample_y)))})
        single = plot_nested_splits(
            sample_y,
            ExpandingWindowSplitter(n_splits=3, test_size=30),
            ExpandingWindowSplitter(n_splits=2, test_size=20),
            X_actual=x_actual,
        )
        assert_figure_valid(single)
        faceted = plot_nested_splits(
            sample_y,
            ExpandingWindowSplitter(n_splits=3, test_size=30),
            ExpandingWindowSplitter(n_splits=2, test_size=20),
            X_actual=x_actual,
            outer_fold="all",
        )
        assert_figure_valid(faceted)


class TestPlotCvResultsScatter:
    """Tests for plot_cv_results_scatter function."""

    def test_basic_scatter(self, sample_cv_results):
        """Test basic CV results scatter plot."""
        fig = plot_cv_results_scatter(sample_cv_results, param_name="alpha")
        assert_figure_valid(fig)

    def test_highlight_best(self, sample_cv_results):
        """Test highlighting of best result."""
        fig = plot_cv_results_scatter(sample_cv_results, param_name="alpha", highlight_best=True)
        # Should have the scatter trace + best marker trace
        assert len(fig.data) >= 2

    def test_no_highlight(self, sample_cv_results):
        """Test without highlighting best result."""
        fig = plot_cv_results_scatter(sample_cv_results, param_name="alpha", highlight_best=False)
        # Should have just the scatter trace
        assert len(fig.data) >= 1

    def test_custom_scorer_name(self, sample_cv_results):
        """Test with explicit scorer name."""
        fig = plot_cv_results_scatter(sample_cv_results, param_name="alpha", scorer_name="score")
        assert len(fig.data) > 0

    def test_auto_detect_scorer(self, sample_cv_results):
        """Test auto-detection of scorer name from keys."""
        fig = plot_cv_results_scatter(sample_cv_results, param_name="alpha")
        assert len(fig.data) > 0

    def test_custom_title_and_labels(self, sample_cv_results):
        """Test custom title and axis labels."""
        fig = plot_cv_results_scatter(
            sample_cv_results,
            param_name="alpha",
            title="Custom Title",
            x_label="Alpha",
            y_label="Score",
        )
        assert_layout(fig, title="Custom Title")

    def test_custom_dimensions(self, sample_cv_results):
        """Test custom width and height."""
        fig = plot_cv_results_scatter(
            sample_cv_results,
            param_name="alpha",
            width=800,
            height=600,
        )
        assert_layout(fig, width=800, height=600)

    def test_with_error_bars(self, sample_cv_results):
        """Test that error bars are shown when std scores available."""
        fig = plot_cv_results_scatter(sample_cv_results, param_name="alpha", show_std=True)
        assert len(fig.data) > 0

    def test_without_error_bars(self, sample_cv_results):
        """Test without error bars."""
        fig = plot_cv_results_scatter(sample_cv_results, param_name="alpha", show_std=False)
        assert len(fig.data) > 0

    def test_missing_param_key(self, sample_cv_results):
        """Test error when parameter key not found."""
        with pytest.raises(ValueError, match="not found"):
            plot_cv_results_scatter(sample_cv_results, param_name="nonexistent")

    def test_missing_mean_score_key(self):
        """Test error when mean score key not found."""
        cv_results = {"param_alpha": [0.1, 1.0]}
        with pytest.raises(ValueError, match="not found"):
            plot_cv_results_scatter(cv_results, param_name="alpha", scorer_name="nonexistent")

    def test_custom_color_palette(self, sample_cv_results):
        """Test custom color palette."""
        fig = plot_cv_results_scatter(
            sample_cv_results,
            param_name="alpha",
            color_palette=["#FF0000", "#0000FF"],
        )
        assert len(fig.data) > 0

    def test_custom_marker_styling(self, sample_cv_results):
        """Test custom marker styling kwargs."""
        fig = plot_cv_results_scatter(
            sample_cv_results,
            param_name="alpha",
            marker_size=15.0,
            best_marker_size=20.0,
            best_marker_color="#00FF00",
        )
        assert len(fig.data) > 0

    def test_without_rank_key(self):
        """Test highlighting works without rank_test_score."""
        cv_results = {
            "param_alpha": [0.01, 0.1, 1.0],
            "mean_test_score": [-0.5, -0.3, -0.2],
        }
        fig = plot_cv_results_scatter(cv_results, param_name="alpha", highlight_best=True)
        assert len(fig.data) >= 2  # scatter + best marker

    def test_without_std_key(self):
        """Test works without std_test_score."""
        cv_results = {
            "param_alpha": [0.01, 0.1, 1.0],
            "mean_test_score": [-0.5, -0.3, -0.2],
            "rank_test_score": [3, 2, 1],
        }
        plot_cv_results_scatter(cv_results, param_name="alpha")

    def test_higher_is_better_false(self):
        """Test that scores are negated when higher_is_better=False."""
        cv_results = {
            "param_alpha": [0.01, 0.1, 1.0],
            "mean_test_score": [-0.5, -0.3, -0.2],
            "rank_test_score": [3, 2, 1],
        }
        fig = plot_cv_results_scatter(cv_results, param_name="alpha", higher_is_better=False)
        # Scores should be negated: [0.5, 0.3, 0.2]
        scatter_y = list(fig.data[0].y)
        assert scatter_y == [0.5, 0.3, 0.2]
        # Best marker should still be at rank=1 (alpha=1.0, displayed score=0.2)
        assert len(fig.data) >= 2
        assert fig.data[1].y[0] == 0.2

    def test_higher_is_better_true(self):
        """Test that scores are unchanged when higher_is_better=True (default)."""
        cv_results = {
            "param_alpha": [0.01, 0.1, 1.0],
            "mean_test_score": [-0.5, -0.3, -0.2],
            "rank_test_score": [3, 2, 1],
        }
        fig = plot_cv_results_scatter(cv_results, param_name="alpha", higher_is_better=True)
        scatter_y = list(fig.data[0].y)
        assert scatter_y == [-0.5, -0.3, -0.2]
        assert len(fig.data) > 0


class TestSplitsEmptyError:
    """plot_splits raises on an empty split set instead of returning a blank figure."""

    def test_zero_splits_raises(self):
        from yohou.model_selection import BaseSplitter

        class _EmptySplitter(BaseSplitter):
            def split(self, y, X_actual=None):  # noqa: ARG002
                return iter(())

            def _iter_test_indices(self, y, X_actual=None):  # noqa: ARG002
                return iter(())

            def get_n_splits(self, y=None, X_actual=None):  # noqa: ARG002
                return 0

        y = pl.DataFrame({
            "time": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 5), "1d", eager=True),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        with pytest.raises(ValueError, match="no splits"):
            plot_splits(y, _EmptySplitter())


class TestCvResultsNoMeanKeys:
    """plot_cv_results_scatter raises a clear error when no mean_test score key is present."""

    def test_missing_mean_test_keys_raises_clear_error(self):
        cv_results = {"param_alpha": [0.1, 1.0]}
        with pytest.raises(ValueError, match="mean_test"):
            plot_cv_results_scatter(cv_results, param_name="alpha")
