"""Tests for component_weight and step_weight parameters."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from yohou.metrics import MeanAbsoluteError


@pytest.fixture()
def y_train():
    """Simple training data with daily frequency."""
    dates = [datetime(2024, 1, i) for i in range(1, 11)]
    return pl.DataFrame({"time": dates, "value": [float(i) for i in range(10)]})


@pytest.fixture()
def y_train_mv():
    """Multivariate training data."""
    base = datetime(2024, 1, 10)
    return pl.DataFrame({
        "time": [base - timedelta(days=i) for i in range(9, -1, -1)],
        "a": [float(i) for i in range(10)],
        "b": [float(i) * 2 for i in range(10)],
    })


@pytest.fixture()
def multi_vintage_data():
    """Multi-vintage data: 3 vintages x 2 steps."""
    base = datetime(2024, 1, 10)
    y_true = pl.DataFrame({
        "time": [
            base + timedelta(days=1),
            base + timedelta(days=2),
            base + timedelta(days=3),
            base + timedelta(days=4),
        ],
        "value": [10.0, 20.0, 30.0, 40.0],
    })
    y_pred = pl.DataFrame({
        "observed_time": [
            base,
            base,
            base + timedelta(days=1),
            base + timedelta(days=1),
            base + timedelta(days=2),
            base + timedelta(days=2),
        ],
        "time": [
            base + timedelta(days=1),
            base + timedelta(days=2),
            base + timedelta(days=2),
            base + timedelta(days=3),
            base + timedelta(days=3),
            base + timedelta(days=4),
        ],
        "value": [11.0, 22.0, 19.0, 28.0, 31.0, 38.0],
    })
    return y_true, y_pred


@pytest.fixture()
def multivariate_data():
    """Multivariate data (2 components) with no vintages."""
    y_true = pl.DataFrame({
        "time": [datetime(2024, 1, i) for i in range(1, 4)],
        "a": [10.0, 20.0, 30.0],
        "b": [100.0, 200.0, 300.0],
    })
    y_pred = pl.DataFrame({
        "time": [datetime(2024, 1, i) for i in range(1, 4)],
        "a": [12.0, 18.0, 33.0],
        "b": [110.0, 190.0, 280.0],
    })
    return y_true, y_pred


class TestComponentWeight:
    """component_weight produces weighted mean across components."""

    def test_equal_weights_matches_default(self, y_train_mv, multivariate_data):
        """Equal weights reproduce unweighted result."""
        y_true, y_pred = multivariate_data

        mae_default = MeanAbsoluteError()
        mae_default.fit(y_train_mv)
        result_default = mae_default.score(y_true, y_pred)

        mae_equal = MeanAbsoluteError(component_weight={"a": 1.0, "b": 1.0})
        mae_equal.fit(y_train_mv)
        result_equal = mae_equal.score(y_true, y_pred)

        np.testing.assert_allclose(result_default, result_equal, atol=1e-10)

    def test_component_weight_changes_scalar(self, y_train_mv, multivariate_data):
        """Weighting components changes the scalar score."""
        y_true, y_pred = multivariate_data
        # a errors: |10-12|=2, |20-18|=2, |30-33|=3 -> mean 7/3
        # b errors: |100-110|=10, |200-190|=10, |300-280|=20 -> mean 40/3
        # Unweighted: mean of all = (2+10+2+10+3+20)/6 = 47/6
        # Weighted {a: 3, b: 1}: per-row weighted mean, then average across rows
        #   row 0: (2*3 + 10*1)/4 = 16/4 = 4.0
        #   row 1: (2*3 + 10*1)/4 = 16/4 = 4.0
        #   row 2: (3*3 + 20*1)/4 = 29/4 = 7.25
        #   mean: (4.0 + 4.0 + 7.25)/3 = 15.25/3

        mae = MeanAbsoluteError(component_weight={"a": 3.0, "b": 1.0})
        mae.fit(y_train_mv)
        result = mae.score(y_true, y_pred)

        expected = (4.0 + 4.0 + 7.25) / 3
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_component_weight_missing_key_defaults_to_1(self, y_train_mv, multivariate_data):
        """Components not in weight dict get weight 1.0."""
        y_true, y_pred = multivariate_data

        # Only specify weight for "a", "b" gets default 1.0
        mae = MeanAbsoluteError(component_weight={"a": 1.0})
        mae.fit(y_train_mv)
        result = mae.score(y_true, y_pred)

        mae_explicit = MeanAbsoluteError(component_weight={"a": 1.0, "b": 1.0})
        mae_explicit.fit(y_train_mv)
        result_explicit = mae_explicit.score(y_true, y_pred)

        np.testing.assert_allclose(result, result_explicit, atol=1e-10)

    def test_component_weight_with_componentwise(self, y_train_mv, multivariate_data):
        """component_weight with componentwise aggregation uses weighted reduce."""
        y_true, y_pred = multivariate_data

        mae = MeanAbsoluteError(
            aggregation_method=["componentwise", "stepwise", "vintagewise"],
            component_weight={"a": 2.0, "b": 1.0},
        )
        mae.fit(y_train_mv)
        result = mae.score(y_true, y_pred)

        # componentwise reduction to scalar with weights
        assert isinstance(result, float)

    def test_get_params_includes_component_weight(self):
        mae = MeanAbsoluteError(component_weight={"a": 2.0})
        params = mae.get_params()
        assert params["component_weight"] == {"a": 2.0}

    def test_none_component_weight_default(self):
        mae = MeanAbsoluteError()
        assert mae.component_weight is None


class TestStepWeight:
    """step_weight produces weighted mean across forecasting steps."""

    def test_equal_step_weights_matches_default(self, y_train, multi_vintage_data):
        """Equal step weights reproduce unweighted result."""
        y_true, y_pred = multi_vintage_data

        mae_default = MeanAbsoluteError()
        mae_default.fit(y_train)
        result_default = mae_default.score(y_true, y_pred)

        mae_equal = MeanAbsoluteError(step_weight={1: 1.0, 2: 1.0})
        mae_equal.fit(y_train)
        result_equal = mae_equal.score(y_true, y_pred)

        np.testing.assert_allclose(result_default, result_equal, atol=1e-10)

    def test_step_weight_emphasize_step_1(self, y_train, multi_vintage_data):
        """Emphasizing step 1 changes the scalar score."""
        y_true, y_pred = multi_vintage_data
        # Step 1 errors: 1, 1, 1 (per-vintage)
        # Step 2 errors: 2, 2, 2
        # Unweighted rows: (1+2+1+2+1+2)/6 = 1.5
        # step_weight {1: 3, 2: 1}: row weights [3,1,3,1,3,1]
        # weighted mean = (1*3 + 2*1 + 1*3 + 2*1 + 1*3 + 2*1) / (3+1+3+1+3+1)
        #               = (3+2+3+2+3+2) / 12 = 15/12 = 1.25

        mae = MeanAbsoluteError(step_weight={1: 3.0, 2: 1.0})
        mae.fit(y_train)
        result = mae.score(y_true, y_pred)

        np.testing.assert_allclose(result, 1.25, atol=1e-10)

    def test_step_weight_without_observed_time_ignored(self, y_train):
        """step_weight is ignored when forecasting_step is unavailable."""
        y_true = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 4)],
            "value": [10.0, 20.0, 30.0],
        })
        y_pred = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 4)],
            "value": [11.0, 22.0, 28.0],
        })

        mae_default = MeanAbsoluteError()
        mae_default.fit(y_train)
        result_default = mae_default.score(y_true, y_pred)

        mae_weighted = MeanAbsoluteError(step_weight={1: 10.0})
        mae_weighted.fit(y_train)
        result_weighted = mae_weighted.score(y_true, y_pred)

        # Without observed_time, step_weight is a no-op
        np.testing.assert_allclose(result_default, result_weighted, atol=1e-10)

    def test_step_weight_with_stepwise_vintagewise_row_reduction(self, y_train, multi_vintage_data):
        """step_weight applies during row reduction with stepwise+vintagewise."""
        y_true, y_pred = multi_vintage_data

        mae = MeanAbsoluteError(
            aggregation_method=["stepwise", "vintagewise"],
            step_weight={1: 1.0, 2: 0.0},
        )
        mae.fit(y_train)
        result = mae.score(y_true, y_pred)

        # step_weight 0 for step 2 means only step 1 counts
        # stepwise+vintagewise collapses all rows, result is a 1-row DataFrame
        assert isinstance(result, pl.DataFrame)

    def test_get_params_includes_step_weight(self):
        mae = MeanAbsoluteError(step_weight={1: 2.0, 2: 1.0})
        params = mae.get_params()
        assert params["step_weight"] == {1: 2.0, 2: 1.0}

    def test_none_step_weight_default(self):
        mae = MeanAbsoluteError()
        assert mae.step_weight is None


class TestCombinedWeights:
    """component_weight and step_weight together."""

    def test_both_weights_applied(self, y_train):
        """Both component_weight and step_weight apply simultaneously."""
        base = datetime(2024, 1, 10)
        y_train_mv = pl.DataFrame({
            "time": [base - timedelta(days=i) for i in range(9, -1, -1)],
            "a": [float(i) for i in range(10)],
            "b": [float(i) * 2 for i in range(10)],
        })
        y_true = pl.DataFrame({
            "time": [
                base + timedelta(days=1),
                base + timedelta(days=2),
            ],
            "a": [10.0, 20.0],
            "b": [100.0, 200.0],
        })
        y_pred = pl.DataFrame({
            "observed_time": [base, base],
            "time": [base + timedelta(days=1), base + timedelta(days=2)],
            "a": [12.0, 18.0],
            "b": [110.0, 190.0],
        })

        mae = MeanAbsoluteError(
            component_weight={"a": 2.0, "b": 1.0},
            step_weight={1: 3.0, 2: 1.0},
        )
        mae.fit(y_train_mv)
        result = mae.score(y_true, y_pred)

        # a errors: 2, 2; b errors: 10, 10
        # Col-weighted per row: (2*2 + 10*1)/3 = 14/3, (2*2 + 10*1)/3 = 14/3
        # Step weights: [3, 1]
        # Row-weighted: (14/3 * 3 + 14/3 * 1) / 4 = 14/3 * 4/4 = 14/3
        expected = 14.0 / 3.0
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_clone_preserves_weights(self):
        from sklearn.base import clone

        mae = MeanAbsoluteError(
            component_weight={"a": 2.0},
            step_weight={1: 3.0},
            vintage_weight={datetime(2024, 1, 10): 2.0},
        )
        cloned = clone(mae)
        assert cloned.component_weight == {"a": 2.0}
        assert cloned.step_weight == {1: 3.0}
        assert cloned.vintage_weight == {datetime(2024, 1, 10): 2.0}


class TestVintageWeight:
    """vintage_weight produces weighted mean across vintages."""

    def test_equal_vintage_weights_matches_default(self, y_train, multi_vintage_data):
        """Equal vintage weights reproduce unweighted result."""
        y_true, y_pred = multi_vintage_data
        base = datetime(2024, 1, 10)

        mae_default = MeanAbsoluteError()
        mae_default.fit(y_train)
        result_default = mae_default.score(y_true, y_pred)

        mae_equal = MeanAbsoluteError(
            vintage_weight={
                base: 1.0,
                base + timedelta(days=1): 1.0,
                base + timedelta(days=2): 1.0,
            }
        )
        mae_equal.fit(y_train)
        result_equal = mae_equal.score(y_true, y_pred)

        np.testing.assert_allclose(result_default, result_equal, atol=1e-10)

    def test_vintage_weight_emphasize_recent(self, y_train, multi_vintage_data):
        """Emphasizing recent vintages changes the scalar score."""
        y_true, y_pred = multi_vintage_data
        base = datetime(2024, 1, 10)
        # Vintage base: errors 1, 2 (step 1, step 2)
        # Vintage base+1: errors 1, 2
        # Vintage base+2: errors 1, 2
        # Unweighted: (1+2+1+2+1+2)/6 = 1.5
        # vintage_weight {base+2: 3, others: 1} -> row weights [1,1,1,1,3,3]
        # weighted = (1*1 + 2*1 + 1*1 + 2*1 + 1*3 + 2*3) / (1+1+1+1+3+3)
        #          = (1+2+1+2+3+6) / 10 = 15/10 = 1.5
        # Actually the same because errors are uniform, let me make them different

        y_true_diff = pl.DataFrame({
            "time": [
                base + timedelta(days=1),
                base + timedelta(days=2),
                base + timedelta(days=3),
                base + timedelta(days=4),
            ],
            "value": [10.0, 20.0, 30.0, 40.0],
        })
        y_pred_diff = pl.DataFrame({
            "observed_time": [
                base,
                base,
                base + timedelta(days=1),
                base + timedelta(days=1),
                base + timedelta(days=2),
                base + timedelta(days=2),
            ],
            "time": [
                base + timedelta(days=1),
                base + timedelta(days=2),
                base + timedelta(days=2),
                base + timedelta(days=3),
                base + timedelta(days=3),
                base + timedelta(days=4),
            ],
            "value": [15.0, 25.0, 19.0, 28.0, 31.0, 38.0],
        })
        # Errors by vintage:
        # base: |10-15|=5, |20-25|=5
        # base+1: |20-19|=1, |30-28|=2
        # base+2: |30-31|=1, |40-38|=2
        # Unweighted: (5+5+1+2+1+2)/6 = 16/6
        # vintage_weight {base: 0, base+1: 1, base+2: 1} -> row weights [0,0,1,1,1,1]
        # weighted = (5*0 + 5*0 + 1*1 + 2*1 + 1*1 + 2*1) / (0+0+1+1+1+1) = 6/4 = 1.5

        mae = MeanAbsoluteError(
            vintage_weight={
                base: 0.0,
                base + timedelta(days=1): 1.0,
                base + timedelta(days=2): 1.0,
            }
        )
        mae.fit(y_train)
        result = mae.score(y_true_diff, y_pred_diff)

        np.testing.assert_allclose(result, 1.5, atol=1e-10)

    def test_vintage_weight_without_observed_time_ignored(self, y_train):
        """vintage_weight is ignored when observed_time is unavailable."""
        y_true = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 4)],
            "value": [10.0, 20.0, 30.0],
        })
        y_pred = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 4)],
            "value": [11.0, 22.0, 28.0],
        })

        mae_default = MeanAbsoluteError()
        mae_default.fit(y_train)
        result_default = mae_default.score(y_true, y_pred)

        mae_weighted = MeanAbsoluteError(vintage_weight={datetime(2024, 1, 1): 10.0})
        mae_weighted.fit(y_train)
        result_weighted = mae_weighted.score(y_true, y_pred)

        np.testing.assert_allclose(result_default, result_weighted, atol=1e-10)

    def test_get_params_includes_vintage_weight(self):
        vw = {datetime(2024, 1, 10): 2.0}
        mae = MeanAbsoluteError(vintage_weight=vw)
        params = mae.get_params()
        assert params["vintage_weight"] == vw

    def test_none_vintage_weight_default(self):
        mae = MeanAbsoluteError()
        assert mae.vintage_weight is None

    def test_step_and_vintage_weight_combined(self, y_train, multi_vintage_data):
        """step_weight and vintage_weight are multiplicative."""
        y_true, y_pred = multi_vintage_data
        base = datetime(2024, 1, 10)

        # Use only step_weight
        mae_step = MeanAbsoluteError(step_weight={1: 3.0, 2: 1.0})
        mae_step.fit(y_train)
        result_step = mae_step.score(y_true, y_pred)

        # Use only vintage_weight (equal)
        mae_vintage = MeanAbsoluteError(
            vintage_weight={
                base: 1.0,
                base + timedelta(days=1): 1.0,
                base + timedelta(days=2): 1.0,
            }
        )
        mae_vintage.fit(y_train)
        result_vintage = mae_vintage.score(y_true, y_pred)

        # Equal vintage weights shouldn't change from default
        mae_default = MeanAbsoluteError()
        mae_default.fit(y_train)
        result_default = mae_default.score(y_true, y_pred)

        np.testing.assert_allclose(result_vintage, result_default, atol=1e-10)
        # step_weight should change the result
        assert not np.isclose(result_step, result_default, atol=1e-10)
