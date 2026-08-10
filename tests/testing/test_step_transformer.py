"""Tests for the yohou.testing.step_transformer check functions."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from yohou.preprocessing import StepAggregator
from yohou.testing.step_transformer import check_memory_api_refused


def _frame() -> pl.DataFrame:
    """A step frame with one base block over a horizon of two."""
    times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(4)]
    return pl.DataFrame({
        "time": times,
        "temp_step_1": [1.0, 2.0, 3.0, 4.0],
        "temp_step_2": [2.0, 3.0, 4.0, 5.0],
    })


class TestMemoryApiRefused:
    """The check has to fail a transformer that answers a memory call.

    Concrete step transformers define none of the four methods, so the check walks
    past all of them and passes without asserting anything. That is the right result
    but it is also indistinguishable from a check that cannot fail, which is worth
    pinning: the guard exists because a composer can reintroduce one of these methods
    by inheritance, and the library has already had to close that gap once for
    forecast-kind transformers.
    """

    def test_a_transformer_with_no_memory_methods_passes(self):
        """The ordinary case: nothing to refuse, so nothing to report."""
        check_memory_api_refused(StepAggregator(aggregations=("mean",)), _frame())

    @pytest.mark.parametrize("method", ["observe", "rewind", "observe_transform", "rewind_transform"])
    def test_a_method_that_answers_is_reported(self, method):
        """Any of the four silently accepting a frame is the failure being guarded."""

        class _Answers(StepAggregator):
            """A step transformer that wrongly services one memory call."""

        setattr(_Answers, method, lambda self, X, *a, **k: X)

        with pytest.raises(AssertionError, match=f"{method}\\(\\) must raise ValueError"):
            check_memory_api_refused(_Answers(aggregations=("mean",)), _frame())

    @pytest.mark.parametrize("method", ["observe", "rewind", "observe_transform", "rewind_transform"])
    def test_a_method_that_refuses_with_valueerror_passes(self, method):
        """Refusing is what the contract asks for, so the check moves on."""

        class _Refuses(StepAggregator):
            """A step transformer that refuses one memory call the required way."""

        def _refuse(self, X, *a, **k):
            raise ValueError("step-kind transformers hold no memory")

        setattr(_Refuses, method, _refuse)

        check_memory_api_refused(_Refuses(aggregations=("mean",)), _frame())


class TestStepFrameValidation:
    """The shared validator's refusals, which the concrete transformers inherit.

    Each raise names what it received. A step frame that reaches a transformer without
    a usable time index, or with columns that drifted since fit, is a wiring mistake
    upstream, and the message is the only place that says so.
    """

    def test_none_is_refused(self):
        with pytest.raises(ValueError, match="`X` cannot be None"):
            StepAggregator(aggregations=("mean",)).fit(None)

    def test_a_frame_without_time_is_refused_naming_its_columns(self):
        frame = _frame().drop("time")
        with pytest.raises(ValueError, match="must contain a 'time' column"):
            StepAggregator(aggregations=("mean",)).fit(frame)

    def test_a_non_temporal_time_column_is_refused(self):
        """A row counter in the index slot would order the frame silently wrong."""
        frame = _frame().with_columns(time=pl.int_range(pl.len()))
        with pytest.raises(ValueError, match="must be Date or Datetime"):
            StepAggregator(aggregations=("mean",)).fit(frame)

    def test_columns_that_drifted_since_fit_are_refused(self):
        """Transform against a different block is a mismatch, not a re-fit."""
        transformer = StepAggregator(aggregations=("mean",)).fit(_frame())
        with pytest.raises(ValueError, match="do not match those seen during fit"):
            transformer.transform(_frame().rename({"temp_step_1": "other_step_1"}))
