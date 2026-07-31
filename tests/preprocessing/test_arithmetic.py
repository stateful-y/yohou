"""Tests for ArithmeticTransformer and ReduceTransformer."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from yohou.compose import PerVintageActualTransformer
from yohou.preprocessing import ArithmeticTransformer, ReduceTransformer


def _frame(n: int = 5) -> pl.DataFrame:
    """Two positive, non-zero columns ``a`` and ``b`` on a daily grid."""
    time = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, n), interval="1d", eager=True)
    return pl.DataFrame({
        "time": time,
        "a": [float(i + 2) for i in range(n)],
        "b": [float(i + 1) for i in range(n)],
    })


class TestArithmeticForward:
    """Forward evaluation of the four binary operations."""

    @pytest.mark.parametrize(
        ("op", "expected"),
        [
            ("add", lambda a, b: a + b),
            ("sub", lambda a, b: a - b),
            ("mul", lambda a, b: a * b),
            ("div", lambda a, b: a / b),
        ],
    )
    def test_each_op(self, op, expected):
        """Each op emits the expected output column."""
        X = _frame()
        t = ArithmeticTransformer("a", "b", op=op, output_name="c")
        X_t = t.fit_transform(X)
        a, b = X["a"].to_numpy(), X["b"].to_numpy()
        assert np.allclose(X_t["c"].to_numpy(), expected(a, b))

    def test_keep_inputs_toggle(self):
        """keep_inputs controls whether the operand columns survive."""
        X = _frame()
        lean = ArithmeticTransformer("a", "b", op="sub", output_name="c").fit_transform(X)
        assert lean.columns == ["time", "c"]

        kept = ArithmeticTransformer("a", "b", op="sub", output_name="c", keep_inputs=True).fit_transform(X)
        assert kept.columns == ["time", "a", "b", "c"]

    def test_feature_names_out(self):
        """get_feature_names_out mirrors the emitted columns."""
        X = _frame()
        t = ArithmeticTransformer("a", "b", op="sub", output_name="c", keep_inputs=True).fit(X)
        assert t.get_feature_names_out() == ["a", "b", "c"]
        t2 = ArithmeticTransformer("a", "b", op="sub", output_name="c").fit(X)
        assert t2.get_feature_names_out() == ["c"]

    def test_missing_input_raises(self):
        """A missing operand column raises at fit."""
        X = _frame().drop("b")
        with pytest.raises(ValueError, match="missing required column"):
            ArithmeticTransformer("a", "b", op="sub").fit(X)


class TestArithmeticInverse:
    """Group-inverse recovery in both directions."""

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    def test_roundtrip_left(self, op):
        """invert_wrt='left' recovers the left operand from output + right."""
        X = _frame()
        t = ArithmeticTransformer("a", "b", op=op, output_name="c", keep_inputs=True, invert_wrt="left")
        X_t = t.fit_transform(X)
        recovered = t.inverse_transform(X_t)
        assert np.allclose(recovered["a"].to_numpy(), X["a"].to_numpy())

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    def test_roundtrip_right(self, op):
        """invert_wrt='right' recovers the right operand from output + left."""
        X = _frame()
        t = ArithmeticTransformer("a", "b", op=op, output_name="c", keep_inputs=True, invert_wrt="right")
        X_t = t.fit_transform(X)
        recovered = t.inverse_transform(X_t)
        assert np.allclose(recovered["b"].to_numpy(), X["b"].to_numpy())

    def test_inverse_from_output_plus_sibling_only(self):
        """The decomposition shape: output + sibling (aggregate operand absent) reconstructs it."""
        X = _frame()
        t = ArithmeticTransformer(
            "revenue", "cost", op="sub", output_name="margin", keep_inputs=True, invert_wrt="left"
        )
        frame = X.rename({"a": "revenue", "b": "cost"})
        X_t = t.fit_transform(frame)
        # Drop the aggregate operand; keep only time + sibling + output, as a forecaster would hold.
        decomposed = X_t.select("time", "cost", "margin")
        recovered = t.inverse_transform(decomposed)
        assert np.allclose(recovered["revenue"].to_numpy(), frame["revenue"].to_numpy())

    def test_missing_sibling_raises(self):
        """Inverse without the retained sibling raises instead of guessing."""
        X = _frame()
        t = ArithmeticTransformer("a", "b", op="sub", output_name="c").fit(X)  # keep_inputs=False
        lean = t.transform(X)
        with pytest.raises(ValueError, match="inverse_transform requires"):
            t.inverse_transform(lean)


class TestReduce:
    """N-ary reduction and its partial inverse."""

    def _frame3(self) -> pl.DataFrame:
        time = pl.datetime_range(datetime(2020, 1, 1), datetime(2020, 1, 4), interval="1d", eager=True)
        return pl.DataFrame({
            "time": time,
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [5.0, 6.0, 7.0, 8.0],
            "c": [2.0, 2.0, 2.0, 2.0],
        })

    @pytest.mark.parametrize(("op", "reduce_fn"), [("sum", np.add), ("product", np.multiply)])
    def test_forward(self, op, reduce_fn):
        """sum/product reduce the input columns row-wise."""
        X = self._frame3()
        X_t = ReduceTransformer(["a", "b", "c"], op=op, output_name="out").fit_transform(X)
        expected = reduce_fn(reduce_fn(X["a"].to_numpy(), X["b"].to_numpy()), X["c"].to_numpy())
        assert np.allclose(X_t["out"].to_numpy(), expected)

    @pytest.mark.parametrize("op", ["sum", "product"])
    def test_partial_inverse(self, op):
        """The partial inverse recovers invert_col from output + the n-1 siblings."""
        X = self._frame3()
        t = ReduceTransformer(["a", "b", "c"], op=op, output_name="out", keep_inputs=True, invert_col="a")
        X_t = t.fit_transform(X)
        recovered = t.inverse_transform(X_t)
        assert np.allclose(recovered["a"].to_numpy(), X["a"].to_numpy())

    def test_partial_inverse_without_siblings_raises(self):
        """Without the retained siblings the partial inverse raises."""
        X = self._frame3()
        t = ReduceTransformer(["a", "b", "c"], op="sum", output_name="out", invert_col="a").fit(X)
        lean = t.transform(X)  # inputs dropped
        with pytest.raises(ValueError, match="inverse_transform requires"):
            t.inverse_transform(lean)

    def test_invert_col_must_be_input(self):
        """invert_col outside input_cols raises at fit."""
        X = self._frame3()
        with pytest.raises(ValueError, match="must be one of input_cols"):
            ReduceTransformer(["a", "b"], invert_col="z").fit(X)


class TestForecastLift:
    """Statelessness => lift onto X_forecast via PerVintageActualTransformer."""

    def _x_forecast(self) -> pl.DataFrame:
        rows = []
        for vt in (datetime(2020, 1, 1), datetime(2020, 1, 2)):
            for h in range(3):
                t = vt + timedelta(days=h)
                rows.append((vt, t, float(10 + h), float(1 + h)))
        return pl.DataFrame(rows, schema=["vintage_time", "time", "a", "b"], orient="row")

    def test_lift_per_vintage(self):
        """The lifted transformer is per-vintage correct and preserves both index columns."""
        X = self._x_forecast()
        lifted = PerVintageActualTransformer(ArithmeticTransformer("a", "b", op="sub", output_name="d"))
        out = lifted.fit(X).transform(X)
        assert {"vintage_time", "time", "d"}.issubset(set(out.columns))
        # d = a - b holds row-wise within every vintage (here a constant 9.0).
        merged = out.join(X, on=["vintage_time", "time"], how="inner")
        assert np.allclose(merged["d"].to_numpy(), merged["a"].to_numpy() - merged["b"].to_numpy())


class TestInvertibleTag:
    """The ``invertible`` tag must track the constructor, not the class.

    Gates like `FeaturePipeline.inverse_transform` read the tag rather than probing for
    the method, so a wrong tag hides a working inverse from every composition.
    """

    def test_arithmetic_tag_follows_keep_inputs(self):
        kept = ArithmeticTransformer("a", "b", op="div", output_name="c", keep_inputs=True)
        lean = ArithmeticTransformer("a", "b", op="div", output_name="c")
        assert kept.__sklearn_tags__().transformer_tags.invertible is True
        assert lean.__sklearn_tags__().transformer_tags.invertible is False

    def test_reduce_tag_needs_both_keep_inputs_and_invert_col(self):
        full = ReduceTransformer(["a", "b"], op="sum", output_name="c", keep_inputs=True, invert_col="a")
        no_col = ReduceTransformer(["a", "b"], op="sum", output_name="c", keep_inputs=True)
        no_keep = ReduceTransformer(["a", "b"], op="sum", output_name="c", invert_col="a")
        assert full.__sklearn_tags__().transformer_tags.invertible is True
        assert no_col.__sklearn_tags__().transformer_tags.invertible is False
        assert no_keep.__sklearn_tags__().transformer_tags.invertible is False

    def test_tagged_transformer_is_invertible_inside_a_pipeline(self):
        from yohou.compose import FeaturePipeline
        from yohou.stationarity import ASinhTransformer

        X = _frame()
        pipe = FeaturePipeline([
            ("ratio", ArithmeticTransformer("a", "b", op="div", output_name="c", keep_inputs=True)),
            ("compress", ASinhTransformer()),
        ]).fit(X)
        assert hasattr(pipe, "inverse_transform")
        back = pipe.inverse_transform(X_t=pipe.transform(X), X_p=X)
        assert np.allclose(back["a"].to_numpy(), X["a"].to_numpy())


class TestAcceptsIrregularGrid:
    """Both transformers are row-local, so the time-axis spacing is inert.

    Each row's output reads only that row's operands. Declaring
    ``accepts_irregular_grid`` is what lets `validate_data` skip the strict
    interval-consistency check, so a sparse per-vintage forward slice is not rejected
    for spacing the computation never looks at.
    """

    @pytest.mark.parametrize(
        "tx",
        [
            ArithmeticTransformer("a", "b", op="sub", output_name="c"),
            ArithmeticTransformer("a", "b", op="div", output_name="c"),
            ReduceTransformer(["a", "b"], op="sum", output_name="c"),
            ReduceTransformer(["a", "b"], op="product", output_name="c"),
        ],
    )
    def test_reports_the_tag(self, tx):
        assert tx.__sklearn_tags__().transformer_tags.accepts_irregular_grid is True

    @pytest.mark.parametrize(
        "make",
        [
            lambda: ArithmeticTransformer("a", "b", op="sub", output_name="c"),
            lambda: ReduceTransformer(["a", "b"], op="sum", output_name="c"),
        ],
    )
    def test_irregular_grid_matches_regular(self, make):
        """Same values, different spacing: a 1d-then-3d gap must not change the output."""
        vals = {"a": [3.0, 4.0, 5.0], "b": [1.0, 2.0, 3.0]}
        regular = pl.DataFrame({"time": [datetime(2020, 1, d) for d in (1, 2, 3)], **vals})
        irregular = pl.DataFrame({"time": [datetime(2020, 1, d) for d in (1, 2, 5)], **vals})
        out_reg = make().fit_transform(regular)
        out_irr = make().fit_transform(irregular)
        assert np.allclose(out_irr["c"].to_numpy(), out_reg["c"].to_numpy())

    def test_sparse_vintages_lift_without_dropping_rows(self):
        """A per-vintage forward slice with gaps fits and keeps every row."""
        rows = []
        for vt in (datetime(2020, 1, 1), datetime(2020, 1, 2)):
            # Forward steps at +0, +1 and +4 days: a deliberately gapped slice.
            for h in (0, 1, 4):
                t = vt + timedelta(days=h)
                rows.append((vt, t, float(10 + h), float(1 + h)))
        X = pl.DataFrame(rows, schema=["vintage_time", "time", "a", "b"], orient="row")
        lifted = PerVintageActualTransformer(ArithmeticTransformer("a", "b", op="sub", output_name="c"))
        out = lifted.fit(X).transform(X)
        assert out.height == X.height
        merged = out.join(X, on=["vintage_time", "time"], how="inner")
        assert merged.height == X.height
        assert np.allclose(merged["c"].to_numpy(), merged["a"].to_numpy() - merged["b"].to_numpy())


class _FrozenDiv(ArithmeticTransformer):
    """A specialization freezing ``op`` by narrowing its own constructor."""

    def __init__(self, left_col, right_col, output_name="frozen"):
        super().__init__(left_col, right_col, op="div", output_name=output_name)


class _FrozenSum(ReduceTransformer):
    """A specialization freezing ``op`` on the n-ary reducer."""

    def __init__(self, input_cols, output_name="frozen"):
        super().__init__(input_cols, op="sum", output_name=output_name)


class TestFrozenOperationSubclass:
    """A subclass may fix ``op`` by narrowing its ``__init__``.

    ``get_params`` reads the subclass constructor signature, so the frozen parameter
    leaves the parameter surface entirely: it is absent from ``get_params``, absent from
    anything serialized from the instance, and rejected by ``set_params``. ``clone``
    still reconstructs the subclass with the operation intact.
    """

    def test_frozen_param_absent_from_get_params(self):
        assert set(_FrozenDiv("a", "b").get_params()) == {"left_col", "right_col", "output_name"}
        assert set(_FrozenSum(["a", "b"]).get_params()) == {"input_cols", "output_name"}

    def test_frozen_param_rejected_by_set_params(self):
        with pytest.raises(ValueError):
            _FrozenDiv("a", "b").set_params(op="mul")
        with pytest.raises(ValueError):
            _FrozenSum(["a", "b"]).set_params(op="product")

    def test_clone_preserves_the_freeze(self):
        from sklearn.base import clone

        X = _frame()
        original = _FrozenDiv("a", "b", output_name="c")
        copy = clone(original)
        assert copy.op == "div"
        assert np.allclose(
            copy.fit_transform(X)["c"].to_numpy(),
            original.fit_transform(X)["c"].to_numpy(),
        )

    def test_frozen_subclass_inherits_the_irregular_grid_tag(self):
        assert _FrozenDiv("a", "b").__sklearn_tags__().transformer_tags.accepts_irregular_grid is True
        assert _FrozenSum(["a", "b"]).__sklearn_tags__().transformer_tags.accepts_irregular_grid is True
