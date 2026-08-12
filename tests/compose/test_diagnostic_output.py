"""The library must not write to standard output unless the caller asks.

`ColumnTransformer` and `FeatureUnion` were ported from sklearn without the
opening guard in `_log_message`, so both printed on every fit regardless of
`verbose`, which defaults to False. `_print_elapsed_time` prints whenever the
message it is handed is not None, so returning a string unconditionally is
enough to make it print.

The output went to raw stdout rather than through logging, so no handler, level
or filter in a consuming application could suppress it. One measured Azure ML
tuning rank carried 1,588 such lines, about 90% of that step's log.

The bug existed because nothing checked, which is what these tests are for.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from conftest import SimpleTransformer  # noqa: E402
from yohou.compose import ColumnTransformer, FeatureUnion  # noqa: E402


@pytest.fixture
def frame() -> pl.DataFrame:
    """A small three-column series, mirroring the other compose tests."""
    length = 20
    return pl.DataFrame({
        "time": pl.datetime_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
            interval="1s",
            eager=True,
        ),
        "a": [float(x) for x in range(length)],
        "b": [float(x) * 10 for x in range(length)],
        "c": [float(x) * 100 for x in range(length)],
    })


def _column_transformer(**kwargs):
    return ColumnTransformer(
        transformers=[("t1", SimpleTransformer(observation_horizon=0), ["a"])],
        remainder="drop",
        **kwargs,
    )


def _feature_union(**kwargs):
    return FeatureUnion(
        transformer_list=[("t1", SimpleTransformer(observation_horizon=0))],
        **kwargs,
    )


class TestDefaultConstructedEstimatorsAreSilent:
    """`verbose` defaults to False, so the documented default must hold."""

    def test_column_transformer_is_silent_on_fit(self, frame, capsys):
        _column_transformer().fit(frame)

        assert capsys.readouterr().out == ""

    def test_column_transformer_is_silent_on_fit_transform(self, frame, capsys):
        estimator = _column_transformer()
        estimator.fit(frame)
        estimator.transform(frame)

        assert capsys.readouterr().out == ""

    def test_feature_union_is_silent_on_fit(self, frame, capsys):
        _feature_union().fit(frame)

        assert capsys.readouterr().out == ""

    def test_feature_union_is_silent_on_fit_transform(self, frame, capsys):
        estimator = _feature_union()
        estimator.fit(frame)
        estimator.transform(frame)

        assert capsys.readouterr().out == ""

    def test_silence_holds_under_composition(self, frame, capsys):
        """A nested composition must be silent at every level, not just the outer one."""
        nested = ColumnTransformer(
            transformers=[
                (
                    "inner",
                    FeatureUnion(transformer_list=[("t1", SimpleTransformer(observation_horizon=0))]),
                    ["a"],
                )
            ],
            remainder="drop",
        )
        nested.fit(frame)
        nested.transform(frame)

        assert capsys.readouterr().out == ""


class TestTheDocumentedParameterStillWorks:
    """Restoring the guard must not delete the feature it guards."""

    def test_column_transformer_reports_when_asked(self, frame, capsys):
        _column_transformer(verbose=True).fit(frame)

        out = capsys.readouterr().out
        assert "ColumnTransformer" in out
        assert "Processing t1" in out

    def test_feature_union_reports_when_asked(self, frame, capsys):
        _feature_union(verbose=True).fit(frame)

        out = capsys.readouterr().out
        assert "FeatureUnion" in out
        assert "Processing t1" in out

    @pytest.mark.parametrize("factory", [_column_transformer, _feature_union])
    def test_the_message_is_none_at_the_default(self, factory):
        """`None` is the mechanism: `_print_elapsed_time` prints anything else."""
        assert factory()._log_message("t1", 1, 1) is None

    @pytest.mark.parametrize("factory", [_column_transformer, _feature_union])
    def test_the_message_is_a_string_when_verbose(self, factory):
        message = factory(verbose=True)._log_message("t1", 1, 1)

        assert isinstance(message, str)
        assert "t1" in message


class TestFeaturePipelineWasNeverAffected:
    """It delegates to sklearn's implementation, which kept the guard.

    Which is why `[FeaturePipeline]` lines never appeared in captured logs while
    the other two did. Pinned so a future port does not lose the delegation.
    """

    def test_it_delegates_rather_than_reimplementing(self):
        import inspect

        from yohou.compose import FeaturePipeline

        source = inspect.getsource(FeaturePipeline._log_message)
        assert "sklearn_Pipeline._log_message" in source


class TestTheLoggerNamespace:
    """The library emits; the application decides where it goes."""

    def test_the_package_logger_has_a_null_handler(self):
        handlers = logging.getLogger("yohou").handlers

        assert any(isinstance(h, logging.NullHandler) for h in handlers)

    def test_the_library_sets_no_level(self):
        """The application's configuration must be authoritative."""
        assert logging.getLogger("yohou").level == logging.NOTSET

    def test_submodules_are_separately_controllable(self):
        """`yohou.compose` must be levellable without silencing the rest."""
        compose = logging.getLogger("yohou.compose")
        model_selection = logging.getLogger("yohou.model_selection")
        try:
            compose.setLevel(logging.ERROR)
            assert compose.getEffectiveLevel() == logging.ERROR
            assert model_selection.getEffectiveLevel() != logging.ERROR
        finally:
            compose.setLevel(logging.NOTSET)

    def test_an_unconfigured_application_sees_no_fallback_output(self, capsys):
        """The null handler exists to prevent the 'no handlers' message."""
        logger = logging.getLogger("yohou.compose")
        logger.debug("a diagnostic nobody configured a handler for")

        captured = capsys.readouterr()
        assert captured.out == ""
        assert "No handlers could be found" not in captured.err
