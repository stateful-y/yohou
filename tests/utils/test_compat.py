"""Tests for the sklearn compatibility shim."""

from sklearn.preprocessing import StandardScaler

from yohou.utils._compat import (
    FORCE_INT_REMAINDER_COLS,
    _filter_estimator_params,
    _is_pandas_df,
    _message_with_time,
    _print_elapsed_time,
)


class TestMessageWithTime:
    """Tests for ``_message_with_time``."""

    def test_short_duration(self):
        """Format a message for a short (<= 60 s) duration."""
        result = _message_with_time("src", "done", 1.5)
        assert result.startswith("[src]")
        assert "done" in result
        assert "1.5s" in result

    def test_long_duration(self):
        """Format a message for a long (> 60 s) duration."""
        result = _message_with_time("src", "done", 120)
        assert "2.0min" in result


class TestPrintElapsedTime:
    """Tests for ``_print_elapsed_time``."""

    def test_none_message_prints_nothing(self, capsys):
        """No output when *message* is None."""
        with _print_elapsed_time("src", None):
            pass
        assert capsys.readouterr().out == ""

    def test_message_prints_timing(self, capsys):
        """Prints a timing line when *message* is given."""
        with _print_elapsed_time("src", "done"):
            pass
        out = capsys.readouterr().out
        assert "[src]" in out
        assert "done" in out


class TestFilterEstimatorParams:
    """Tests for ``_filter_estimator_params``."""

    def test_keeps_valid_params(self):
        """Accepted params are preserved."""
        result = _filter_estimator_params(StandardScaler, {"with_mean": True, "with_std": False})
        assert result == {"with_mean": True, "with_std": False}

    def test_drops_unknown_params(self):
        """Params not in the constructor signature are dropped."""

        class _NoClip:
            def __init__(self, copy=True):
                self.copy = copy

        result = _filter_estimator_params(_NoClip, {"copy": True, "clip": False})
        assert result == {"copy": True}

    def test_allows_all_when_var_keyword(self):
        """All params pass through when the class accepts **kwargs."""

        class _Open:
            def __init__(self, **kwargs):
                pass

        result = _filter_estimator_params(_Open, {"any": 1, "thing": 2})
        assert result == {"any": 1, "thing": 2}


class TestIsPandasDf:
    """Tests for ``_is_pandas_df``."""

    def test_plain_object_is_false(self):
        """An ordinary object is not a pandas DataFrame."""
        assert _is_pandas_df(42) is False

    def test_dict_is_false(self):
        """A dict has no iloc/columns."""
        assert _is_pandas_df({"a": 1}) is False

    def test_duck_typed_true(self):
        """An object with iloc and columns attributes is detected."""

        class _FakeDF:
            iloc = None
            columns = None

        assert _is_pandas_df(_FakeDF()) is True


class TestForceIntRemainderCols:
    """Tests for ``FORCE_INT_REMAINDER_COLS``."""

    def test_value_is_true(self):
        """Constant must be True for sklearn <1.8 compatibility."""
        assert FORCE_INT_REMAINDER_COLS is True
