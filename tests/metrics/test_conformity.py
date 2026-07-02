"""Tests for conformity scorers."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from conftest import run_checks as _run_checks_base
from yohou.metrics.conformity import (
    AbsoluteGammaResidual,
    AbsoluteQuantileResidual,
    AbsoluteResidual,
    GammaResidual,
    QuantileResidual,
    Residual,
)
from yohou.testing import _yield_yohou_scorer_checks


class ConcreteGammaResidual(GammaResidual):
    """Concrete subclass of abstract GammaResidual for testing purposes."""

    def inverse_score(self, y_pred, conformity_scores, coverage_rate):
        """Dummy implementation to allow instantiation."""
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, ["_is_fitted"])
        return y_pred


class ConcreteAbsoluteGammaResidual(AbsoluteGammaResidual):
    """Concrete subclass of abstract AbsoluteGammaResidual for testing purposes."""

    def inverse_score(self, y_pred, conformity_scores, coverage_rate):
        """Dummy implementation to allow instantiation."""
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, ["_is_fitted"])
        return y_pred


@pytest.fixture
def data():
    """Return synthetic (y_truth, y_pred) with proper types."""
    start = datetime(2020, 1, 1)
    dates = [start + timedelta(days=i) for i in range(3)]

    y_truth = pl.DataFrame({"time": dates, "y": [1.0, 2.0, 3.0]})

    y_pred = pl.DataFrame({"vintage_time": [start - timedelta(days=1)] * 3, "time": dates, "y": [1.1, 1.9, 3.2]})
    return y_truth, y_pred


def run_checks(scorer, y_truth, y_pred):
    """Run all systematic scorer checks."""
    _run_checks_base(scorer, _yield_yohou_scorer_checks(scorer, y_truth, y_pred))


class TestSystematicChecks:
    def test_residual(self, data):
        """Test standard checks for Residual scorer."""
        y_truth, y_pred = data
        scorer = Residual()
        run_checks(scorer, y_truth, y_pred)

    def test_absolute_residual(self, data):
        """Test standard checks for AbsoluteResidual scorer."""
        y_truth, y_pred = data
        scorer = AbsoluteResidual()
        run_checks(scorer, y_truth, y_pred)

    def test_gamma_residual(self, data):
        """Test standard checks for GammaResidual scorer (using concrete subclass)."""
        y_truth, y_pred = data
        scorer = ConcreteGammaResidual()
        run_checks(scorer, y_truth, y_pred)

    def test_absolute_gamma_residual(self, data):
        """Test standard checks for AbsoluteGammaResidual scorer (using concrete subclass)."""
        y_truth, y_pred = data
        scorer = ConcreteAbsoluteGammaResidual()
        run_checks(scorer, y_truth, y_pred)


class TestResidualValues:
    def test_residual_values(self, data):
        """Test Residual specific value calculation."""
        y_truth, y_pred = data
        scorer = Residual()
        scorer.fit(y_truth)
        scores = scorer.score(y_truth, y_pred)

        expected = y_truth["y"] - y_pred["y"]
        assert np.allclose(scores["y"].to_numpy(), expected.to_numpy())

    def test_absolute_residual_values(self, data):
        """Test AbsoluteResidual specific value calculation."""
        y_truth, y_pred = data
        scorer = AbsoluteResidual()
        scorer.fit(y_truth)
        scores = scorer.score(y_truth, y_pred)

        expected = (y_truth["y"] - y_pred["y"]).abs()
        assert np.allclose(scores["y"].to_numpy(), expected.to_numpy())

    def test_gamma_residual_values(self, data):
        """Test GammaResidual specific value calculation."""
        y_truth, y_pred = data
        eps = 1e-8
        scorer = ConcreteGammaResidual(epsilon=eps)
        scorer.fit(y_truth)
        scores = scorer.score(y_truth, y_pred)

        expected = (y_truth["y"] - y_pred["y"]) / (y_pred["y"] + eps)
        assert np.allclose(scores["y"].to_numpy(), expected.to_numpy())

    def test_absolute_gamma_residual_values(self, data):
        """Test AbsoluteGammaResidual specific value calculation."""
        y_truth, y_pred = data
        eps = 1e-8
        scorer = ConcreteAbsoluteGammaResidual(epsilon=eps)
        scorer.fit(y_truth)
        scores = scorer.score(y_truth, y_pred)

        expected = ((y_truth["y"] - y_pred["y"]) / (y_pred["y"] + eps)).abs()
        assert np.allclose(scores["y"].to_numpy(), expected.to_numpy())


class TestAbstractClasses:
    def test_quantile_residual_abstract(self):
        """Test that QuantileResidual is abstract."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            QuantileResidual()

    def test_absolute_quantile_residual_abstract(self):
        """Test that AbsoluteQuantileResidual is abstract."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            AbsoluteQuantileResidual()


class TestInverseScore:
    def test_residual_inverse_score(self):
        """Test inverse_score mechanism for Residual."""
        scorer = Residual()

        y_pred = pl.DataFrame({"time": [datetime(2020, 1, 1)], "y": [10.0]})
        conformity_scores = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "y": [-1.0, 0.0, 1.0],
        })
        coverage_rate = 0.5

        scorer.fit(conformity_scores)
        intervals = scorer.inverse_score(y_pred, conformity_scores, coverage_rate)

        assert "time" in intervals.columns
        assert intervals["time"][0] == datetime(2020, 1, 1)

        cols = intervals.columns
        lower_col = next(c for c in cols if "lower" in c)
        upper_col = next(c for c in cols if "upper" in c)

        assert intervals.height == 1

        lower = intervals[lower_col][0]
        upper = intervals[upper_col][0]

        # coverage_rate=0.5 -> alpha=0.5; lower quantile of [-1,0,1] at 0.25
        # (method="lower") is -1.0, upper quantile at 0.75 (method="higher")
        # is 1.0. Bounds are y_pred (10.0) plus those quantiles.
        assert np.isfinite(lower)
        assert np.isfinite(upper)
        assert lower == pytest.approx(9.0)
        assert upper == pytest.approx(11.0)
        assert lower <= upper

    def test_absolute_residual_inverse_score(self):
        """AbsoluteResidual builds symmetric intervals centred on y_pred."""
        scorer = AbsoluteResidual()

        y_pred = pl.DataFrame({"time": [datetime(2020, 1, 1)], "y": [10.0]})
        conformity_scores = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "y": [0.0, 1.0, 2.0],
        })

        scorer.fit(conformity_scores)
        intervals = scorer.inverse_score(y_pred, conformity_scores, 0.5)

        lower_col = next(c for c in intervals.columns if "lower" in c)
        upper_col = next(c for c in intervals.columns if "upper" in c)
        lower = intervals[lower_col][0]
        upper = intervals[upper_col][0]

        # Symmetric quantile of [0,1,2] at coverage_rate=0.5 (method="lower")
        # is 1.0, so bounds are y_pred +/- 1.0 and equidistant from y_pred.
        assert lower == pytest.approx(9.0)
        assert upper == pytest.approx(11.0)
        assert (upper - 10.0) == pytest.approx(10.0 - lower)

    def test_gamma_residual_inverse_score(self):
        """GammaResidual scales interval half-widths by the prediction magnitude."""
        eps = 1e-8
        scorer = GammaResidual(epsilon=eps)

        y_pred = pl.DataFrame({"time": [datetime(2020, 1, 1)], "y": [10.0]})
        conformity_scores = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "y": [-0.2, 0.0, 0.2],
        })

        scorer.fit(conformity_scores)
        intervals = scorer.inverse_score(y_pred, conformity_scores, 0.5)

        lower_col = next(c for c in intervals.columns if "lower" in c)
        upper_col = next(c for c in intervals.columns if "upper" in c)
        lower = intervals[lower_col][0]
        upper = intervals[upper_col][0]

        # Asymmetric quantiles of [-0.2,0,0.2] at 0.25/0.75 are -0.2/0.2.
        # Bounds are y_pred + q * (y_pred + epsilon), i.e. 10 +/- 0.2*(10+eps).
        denom = 10.0 + eps
        assert lower == pytest.approx(10.0 - 0.2 * denom)
        assert upper == pytest.approx(10.0 + 0.2 * denom)


class TestConformityScorerTags:
    """Test that conformity scorer tags correctly declare symmetric and multiplicative."""

    def test_residual_tags(self):
        """Residual is asymmetric and additive."""
        tags = Residual().__sklearn_tags__()
        assert tags.scorer_tags is not None
        assert tags.scorer_tags.symmetric is False
        assert tags.scorer_tags.multiplicative is False

    def test_absolute_residual_tags(self):
        """AbsoluteResidual is symmetric and additive."""
        tags = AbsoluteResidual().__sklearn_tags__()
        assert tags.scorer_tags is not None
        assert tags.scorer_tags.symmetric is True
        assert tags.scorer_tags.multiplicative is False

    def test_gamma_residual_tags(self):
        """GammaResidual is asymmetric and multiplicative."""
        tags = ConcreteGammaResidual().__sklearn_tags__()
        assert tags.scorer_tags is not None
        assert tags.scorer_tags.symmetric is False
        assert tags.scorer_tags.multiplicative is True

    def test_absolute_gamma_residual_tags(self):
        """AbsoluteGammaResidual is symmetric and multiplicative."""
        tags = AbsoluteGammaResidual().__sklearn_tags__()
        assert tags.scorer_tags is not None
        assert tags.scorer_tags.symmetric is True
        assert tags.scorer_tags.multiplicative is True

    def test_quantile_residual_tags(self):
        """QuantileResidual is asymmetric (signed variant)."""

        class _ConcreteQuantileResidual(QuantileResidual):
            def score(self, y_truth, y_pred, /, **score_params):
                return y_truth

            def inverse_score(self, y_pred, conformity_scores, coverage_rate):
                return y_pred

        tags = _ConcreteQuantileResidual().__sklearn_tags__()
        assert tags.scorer_tags is not None
        assert tags.scorer_tags.symmetric is False

    def test_absolute_quantile_residual_tags(self):
        """AbsoluteQuantileResidual is symmetric, matching the 'absolute' convention."""

        class _ConcreteAbsoluteQuantileResidual(AbsoluteQuantileResidual):
            def score(self, y_truth, y_pred, /, **score_params):
                return y_truth

            def inverse_score(self, y_pred, conformity_scores, coverage_rate):
                return y_pred

        tags = _ConcreteAbsoluteQuantileResidual().__sklearn_tags__()
        assert tags.scorer_tags is not None
        assert tags.scorer_tags.symmetric is True
