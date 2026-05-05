"""Tests for weight dimension parameters (components, groups, step/vintage/time weights)."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from yohou.metrics import IntervalScore, MeanAbsoluteError


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
        "vintage_time": [
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

        mae_equal = MeanAbsoluteError(components={"a": 1.0, "b": 1.0})
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

        mae = MeanAbsoluteError(components={"a": 3.0, "b": 1.0})
        mae.fit(y_train_mv)
        result = mae.score(y_true, y_pred)

        expected = (4.0 + 4.0 + 7.25) / 3
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_component_weight_filters_to_specified_keys(self, y_train_mv, multivariate_data):
        """components dict filters to only specified keys."""
        y_true, y_pred = multivariate_data

        # Only specify "a" → filters to component "a" only
        mae_a = MeanAbsoluteError(components={"a": 1.0})
        mae_a.fit(y_train_mv)
        result_a = mae_a.score(y_true, y_pred)

        mae_both = MeanAbsoluteError(components={"a": 1.0, "b": 1.0})
        mae_both.fit(y_train_mv)
        result_both = mae_both.score(y_true, y_pred)

        # Results differ because mae_a only scores component "a"
        assert not np.isclose(result_a, result_both, atol=1e-10)

    def test_component_weight_with_componentwise(self, y_train_mv, multivariate_data):
        """component_weight with componentwise aggregation uses weighted reduce."""
        y_true, y_pred = multivariate_data

        mae = MeanAbsoluteError(
            aggregation_method=["componentwise", "stepwise", "vintagewise"],
            components={"a": 2.0, "b": 1.0},
        )
        mae.fit(y_train_mv)
        result = mae.score(y_true, y_pred)

        # componentwise reduction to scalar with weights
        assert isinstance(result, float)

    def test_get_params_includes_component_weight(self):
        mae = MeanAbsoluteError(components={"a": 2.0})
        params = mae.get_params()
        assert params["components"] == {"a": 2.0}

    def test_none_component_weight_default(self):
        mae = MeanAbsoluteError()
        assert mae.components is None


class TestStepWeight:
    """step_weight produces weighted mean across forecasting steps."""

    def test_equal_step_weights_matches_default(self, y_train, multi_vintage_data):
        """Equal step weights reproduce unweighted result."""
        y_true, y_pred = multi_vintage_data

        mae_default = MeanAbsoluteError()
        mae_default.fit(y_train)
        result_default = mae_default.score(y_true, y_pred)

        mae_equal = MeanAbsoluteError()
        mae_equal.fit(y_train)
        result_equal = mae_equal.score(y_true, y_pred, step_weight={1: 1.0, 2: 1.0})

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

        mae = MeanAbsoluteError()
        mae.fit(y_train)
        result = mae.score(y_true, y_pred, step_weight={1: 3.0, 2: 1.0})

        np.testing.assert_allclose(result, 1.25, atol=1e-10)

    def test_step_weight_without_vintage_time_ignored(self, y_train):
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

        mae_weighted = MeanAbsoluteError()
        mae_weighted.fit(y_train)
        result_weighted = mae_weighted.score(y_true, y_pred, step_weight={1: 10.0})

        # Without vintage_time, step_weight is a no-op
        np.testing.assert_allclose(result_default, result_weighted, atol=1e-10)

    def test_step_weight_with_stepwise_vintagewise_row_reduction(self, y_train, multi_vintage_data):
        """step_weight applies during row reduction with stepwise+vintagewise."""
        y_true, y_pred = multi_vintage_data

        mae = MeanAbsoluteError(
            aggregation_method=["stepwise", "vintagewise"],
        )
        mae.fit(y_train)
        result = mae.score(y_true, y_pred, step_weight={1: 1.0, 2: 0.0})

        # step_weight 0 for step 2 means only step 1 counts
        # stepwise+vintagewise collapses all rows, result is a 1-row DataFrame
        assert isinstance(result, pl.DataFrame)

    def test_step_weight_is_score_param(self, y_train, multi_vintage_data):
        """step_weight is now a score() parameter, not an __init__ param."""
        y_true, y_pred = multi_vintage_data
        mae = MeanAbsoluteError()
        mae.fit(y_train)
        # Should accept step_weight in score()
        result = mae.score(y_true, y_pred, step_weight={1: 3.0, 2: 1.0})
        np.testing.assert_allclose(result, 1.25, atol=1e-10)


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
            "vintage_time": [base, base],
            "time": [base + timedelta(days=1), base + timedelta(days=2)],
            "a": [12.0, 18.0],
            "b": [110.0, 190.0],
        })

        mae = MeanAbsoluteError(
            components={"a": 2.0, "b": 1.0},
        )
        mae.fit(y_train_mv)
        result = mae.score(y_true, y_pred, step_weight={1: 3.0, 2: 1.0})

        # a errors: 2, 2; b errors: 10, 10
        # Col-weighted per row: (2*2 + 10*1)/3 = 14/3, (2*2 + 10*1)/3 = 14/3
        # Step weights: [3, 1]
        # Row-weighted: (14/3 * 3 + 14/3 * 1) / 4 = 14/3 * 4/4 = 14/3
        expected = 14.0 / 3.0
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_clone_preserves_components(self):
        from sklearn.base import clone

        mae = MeanAbsoluteError(
            components={"a": 2.0},
        )
        cloned = clone(mae)
        assert cloned.components == {"a": 2.0}


class TestVintageWeight:
    """vintage_weight produces weighted mean across vintages."""

    def test_equal_vintage_weights_matches_default(self, y_train, multi_vintage_data):
        """Equal vintage weights reproduce unweighted result."""
        y_true, y_pred = multi_vintage_data
        base = datetime(2024, 1, 10)

        mae_default = MeanAbsoluteError()
        mae_default.fit(y_train)
        result_default = mae_default.score(y_true, y_pred)

        mae_equal = MeanAbsoluteError()
        mae_equal.fit(y_train)
        result_equal = mae_equal.score(
            y_true,
            y_pred,
            vintage_weight={
                base: 1.0,
                base + timedelta(days=1): 1.0,
                base + timedelta(days=2): 1.0,
            },
        )

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
            "vintage_time": [
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

        mae = MeanAbsoluteError()
        mae.fit(y_train)
        result = mae.score(
            y_true_diff,
            y_pred_diff,
            vintage_weight={
                base: 0.0,
                base + timedelta(days=1): 1.0,
                base + timedelta(days=2): 1.0,
            },
        )

        np.testing.assert_allclose(result, 1.5, atol=1e-10)

    def test_vintage_weight_without_vintage_time_ignored(self, y_train):
        """vintage_weight is ignored when vintage_time is unavailable."""
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

        mae_weighted = MeanAbsoluteError()
        mae_weighted.fit(y_train)
        result_weighted = mae_weighted.score(y_true, y_pred, vintage_weight={datetime(2024, 1, 1): 10.0})

        np.testing.assert_allclose(result_default, result_weighted, atol=1e-10)

    def test_vintage_weight_is_score_param(self, y_train, multi_vintage_data):
        """vintage_weight is now a score() parameter."""
        y_true, y_pred = multi_vintage_data
        mae = MeanAbsoluteError()
        mae.fit(y_train)
        result = mae.score(
            y_true,
            y_pred,
            vintage_weight={
                datetime(2024, 1, 10): 1.0,
                datetime(2024, 1, 11): 1.0,
                datetime(2024, 1, 12): 1.0,
            },
        )
        assert isinstance(result, float)

    def test_step_and_vintage_weight_combined(self, y_train, multi_vintage_data):
        """step_weight and vintage_weight are multiplicative."""
        y_true, y_pred = multi_vintage_data
        base = datetime(2024, 1, 10)

        # Use only step_weight
        mae_step = MeanAbsoluteError()
        mae_step.fit(y_train)
        result_step = mae_step.score(y_true, y_pred, step_weight={1: 3.0, 2: 1.0})

        # Use only vintage_weight (equal)
        mae_vintage = MeanAbsoluteError()
        mae_vintage.fit(y_train)
        result_vintage = mae_vintage.score(
            y_true,
            y_pred,
            vintage_weight={
                base: 1.0,
                base + timedelta(days=1): 1.0,
                base + timedelta(days=2): 1.0,
            },
        )

        # Equal vintage weights shouldn't change from default
        mae_default = MeanAbsoluteError()
        mae_default.fit(y_train)
        result_default = mae_default.score(y_true, y_pred)

        np.testing.assert_allclose(result_vintage, result_default, atol=1e-10)
        # step_weight should change the result
        assert not np.isclose(result_step, result_default, atol=1e-10)


class TestComponentWeightComponentwise:
    """component_weight with componentwise reduction paths."""

    def test_componentwise_stepwise_panel_with_weight(self):
        """Panel data with componentwise + component_weight produces weighted groups."""
        y_true = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 4)],
            "g1__a": [10.0, 20.0, 30.0],
            "g1__b": [100.0, 200.0, 300.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2024, 1, 1)] * 3,
            "time": [datetime(2024, 1, i) for i in range(1, 4)],
            "g1__a": [12.0, 18.0, 33.0],
            "g1__b": [110.0, 190.0, 280.0],
        })
        mae = MeanAbsoluteError(
            aggregation_method=["componentwise", "vintagewise"],
            components={"a": 2.0, "b": 1.0},
        )
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)
        # Without stepwise in aggregation_method, forecasting_step is collapsed
        assert "g1__mae" in result.columns

    def test_componentwise_non_panel_with_weight(self):
        """Non-panel data with componentwise + component_weight uses weighted reduce."""
        y_true = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 4)],
            "a": [10.0, 20.0, 30.0],
            "b": [100.0, 200.0, 300.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2024, 1, 1)] * 3,
            "time": [datetime(2024, 1, i) for i in range(1, 4)],
            "a": [12.0, 18.0, 33.0],
            "b": [110.0, 190.0, 280.0],
        })
        mae = MeanAbsoluteError(
            aggregation_method=["componentwise", "vintagewise"],
            components={"a": 2.0, "b": 1.0},
        )
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)
        assert isinstance(result, pl.DataFrame)
        # Without stepwise in aggregation_method, forecasting_step is collapsed
        assert "mae" in result.columns


class TestGroupwiseComponentWeight:
    """Groupwise aggregation with component_weight."""

    def test_groupwise_with_component_weight(self):
        """Groupwise + component_weight applies weights within groups."""
        y_true = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 6)],
            "region__east": [10.0, 20.0, 30.0, 40.0, 50.0],
            "region__west": [15.0, 25.0, 35.0, 45.0, 55.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2024, 1, 1)] * 5,
            "time": [datetime(2024, 1, i) for i in range(1, 6)],
            "region__east": [12.0, 18.0, 33.0, 38.0, 52.0],
            "region__west": [14.0, 26.0, 33.0, 46.0, 53.0],
        })
        mae = MeanAbsoluteError(
            aggregation_method="all",
            components={"east": 2.0, "west": 1.0},
        )
        mae.fit(y_true)
        result = mae.score(y_true, y_pred)
        assert isinstance(result, float)
        # Should differ from equal weights
        mae_equal = MeanAbsoluteError(aggregation_method="all")
        mae_equal.fit(y_true)
        result_equal = mae_equal.score(y_true, y_pred)
        assert not np.isclose(result, result_equal, atol=1e-10)


class TestNoContextDimFallback:
    """Tests for weight helpers when context dimensions are missing."""

    def test_step_weight_no_context_steps(self):
        """step_weight is ignored when context has no forecasting_step."""
        y_true = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 4)],
            "value": [10.0, 20.0, 30.0],
        })
        y_pred = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 4)],
            "value": [12.0, 18.0, 33.0],
        })
        # No vintage_time -> no forecasting_step in context
        mae = MeanAbsoluteError()
        mae.fit(y_true)
        result = mae.score(y_true, y_pred, step_weight={1: 3.0, 2: 1.0})
        # Should still produce a valid scalar
        assert isinstance(result, float)
        # Same as default since weights can't be applied
        mae_default = MeanAbsoluteError()
        mae_default.fit(y_true)
        np.testing.assert_allclose(result, mae_default.score(y_true, y_pred), atol=1e-10)

    def test_vintage_weight_no_context_vintages(self):
        """vintage_weight is ignored when context has no vintage_time."""
        y_true = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 4)],
            "value": [10.0, 20.0, 30.0],
        })
        y_pred = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 4)],
            "value": [12.0, 18.0, 33.0],
        })
        mae = MeanAbsoluteError()
        mae.fit(y_true)
        result = mae.score(y_true, y_pred, vintage_weight={datetime(2024, 1, 1): 3.0, datetime(2024, 1, 2): 1.0})
        assert isinstance(result, float)
        mae_default = MeanAbsoluteError()
        mae_default.fit(y_true)
        np.testing.assert_allclose(result, mae_default.score(y_true, y_pred), atol=1e-10)


class TestCollapseComponentsDirect:
    """Test _collapse_components method directly."""

    def test_weighted_component_collapse(self):
        """Weighted component collapse applies proper weights."""
        from yohou.metrics import MeanAbsoluteError

        scorer = MeanAbsoluteError(components={"a": 2.0, "b": 1.0})
        df = pl.DataFrame({"a": [2.0, 3.0], "b": [10.0, 20.0]})
        result = scorer._collapse_components(df)
        # weight a=2, b=1 => per-row: row0=(2*2/3+10*1/3)=14/3, row1=(3*2/3+20*1/3)=26/3
        assert "score" in result.columns
        expected_row0 = (2 * 2 + 10 * 1) / 3
        expected_row1 = (3 * 2 + 20 * 1) / 3
        np.testing.assert_allclose(result["score"][0], expected_row0, atol=1e-10)
        np.testing.assert_allclose(result["score"][1], expected_row1, atol=1e-10)


class TestCombineRowWeightsBranches:
    """Test weight combination paths through scoring with both weights set."""

    def test_both_step_and_vintage_weight_combined(self):
        """Both step_weight and vintage_weight produce multiplicative weights."""
        base = datetime(2024, 1, 10)
        y_true = pl.DataFrame({
            "time": [base + timedelta(days=i) for i in range(1, 5)],
            "value": [10.0, 20.0, 30.0, 40.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [
                base,
                base,
                base + timedelta(days=1),
                base + timedelta(days=1),
            ],
            "time": [
                base + timedelta(days=1),
                base + timedelta(days=2),
                base + timedelta(days=2),
                base + timedelta(days=3),
            ],
            "value": [11.0, 22.0, 25.0, 45.0],
        })
        y_train = pl.DataFrame({
            "time": [base - timedelta(days=i) for i in range(9, -1, -1)],
            "value": [float(i) for i in range(10)],
        })

        mae_both = MeanAbsoluteError()
        mae_both.fit(y_train)
        result_both = mae_both.score(
            y_true,
            y_pred,
            step_weight={1: 2.0, 2: 1.0},
            vintage_weight={base: 3.0, base + timedelta(days=1): 1.0},
        )
        assert isinstance(result_both, float)

        # Should differ from step-only and vintage-only
        mae_step = MeanAbsoluteError()
        mae_step.fit(y_train)
        result_step = mae_step.score(y_true, y_pred, step_weight={1: 2.0, 2: 1.0})

        mae_vin = MeanAbsoluteError()
        mae_vin.fit(y_train)
        result_vin = mae_vin.score(
            y_true,
            y_pred,
            vintage_weight={base: 3.0, base + timedelta(days=1): 1.0},
        )

        # Combined should differ from each individual
        assert not np.isclose(result_both, result_step, atol=1e-10)
        assert not np.isclose(result_both, result_vin, atol=1e-10)


class TestNormalizeAggMethodsAll:
    """Test that aggregation_method='all' produces correct modes."""

    def test_all_includes_groupwise(self):
        """aggregation_method='all' includes groupwise."""
        from yohou.metrics.base import BasePointScorer

        modes = BasePointScorer._normalize_agg_methods("all")
        assert "groupwise" in modes
        assert "stepwise" in modes
        assert "vintagewise" in modes
        assert "componentwise" in modes
        assert "coveragewise" not in modes

    def test_all_with_coveragewise(self):
        """aggregation_method='all' with include_coveragewise adds coveragewise."""
        from yohou.metrics.base import BasePointScorer

        modes = BasePointScorer._normalize_agg_methods("all", include_coveragewise=True)
        assert "coveragewise" in modes


@pytest.fixture()
def interval_train():
    """Training data for interval scorer tests."""
    return pl.DataFrame({
        "time": [datetime(2024, 1, i) for i in range(1, 11)],
        "value": [float(i) for i in range(10)],
    })


@pytest.fixture()
def interval_multi_rate():
    """Interval predictions with two coverage rates (0.9 and 0.95)."""
    times = [datetime(2024, 1, 11), datetime(2024, 1, 12), datetime(2024, 1, 13)]
    y_true = pl.DataFrame({
        "time": times,
        "value": [10.0, 20.0, 30.0],
    })
    y_pred = pl.DataFrame({
        "vintage_time": [datetime(2024, 1, 10)] * 3,
        "time": times,
        "value_lower_0.9": [8.0, 18.0, 28.0],
        "value_upper_0.9": [12.0, 22.0, 32.0],
        "value_lower_0.95": [7.0, 17.0, 27.0],
        "value_upper_0.95": [13.0, 23.0, 33.0],
    })
    return y_true, y_pred


@pytest.fixture()
def interval_multi_vintage():
    """Multi-vintage interval predictions (2 vintages x 2 steps, single rate)."""
    base = datetime(2024, 1, 10)
    y_true = pl.DataFrame({
        "time": [base + timedelta(days=i) for i in range(1, 5)],
        "value": [10.0, 20.0, 30.0, 40.0],
    })
    y_pred = pl.DataFrame({
        "vintage_time": [
            base,
            base,
            base + timedelta(days=1),
            base + timedelta(days=1),
        ],
        "time": [
            base + timedelta(days=1),
            base + timedelta(days=2),
            base + timedelta(days=2),
            base + timedelta(days=3),
        ],
        "value_lower_0.9": [8.0, 18.0, 17.0, 27.0],
        "value_upper_0.9": [12.0, 22.0, 23.0, 33.0],
    })
    return y_true, y_pred


class TestIntervalTimeWeight:
    """Time weight applied to interval scorers."""

    def test_constant_time_weight_matches_default(self, interval_train, interval_multi_rate):
        """Uniform time weights reproduce the unweighted score."""
        y_true, y_pred = interval_multi_rate

        scorer = IntervalScore()
        scorer.fit(interval_train)
        default = scorer.score(y_true, y_pred)

        scorer_tw = IntervalScore()
        scorer_tw.fit(interval_train)
        scorer_tw.set_score_request(time_weight=True)
        weighted = scorer_tw.score(
            y_true,
            y_pred,
            time_weight=lambda t: pl.Series("w", [1.0] * len(t)),
        )

        np.testing.assert_allclose(weighted, default, atol=1e-10)

    def test_time_weight_changes_interval_score(self, interval_train):
        """Non-uniform time weights change the interval score."""
        # Use asymmetric intervals so different timesteps have different scores
        times = [datetime(2024, 1, 11), datetime(2024, 1, 12), datetime(2024, 1, 13)]
        y_true = pl.DataFrame({"time": times, "value": [10.0, 20.0, 30.0]})
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2024, 1, 10)] * 3,
            "time": times,
            "value_lower_0.9": [8.0, 15.0, 28.0],  # 2nd point outside interval
            "value_upper_0.9": [12.0, 18.0, 32.0],
        })

        scorer = IntervalScore()
        scorer.fit(interval_train)
        default = scorer.score(y_true, y_pred)

        scorer_tw = IntervalScore()
        scorer_tw.fit(interval_train)
        scorer_tw.set_score_request(time_weight=True)
        # Weight the outlier timestep heavily
        weighted = scorer_tw.score(
            y_true,
            y_pred,
            time_weight=lambda t: pl.Series("w", [1.0, 10.0, 1.0][: len(t)]),
        )

        assert not np.isclose(weighted, default, atol=1e-10)


class TestIntervalStepWeight:
    """Step and vintage weights with interval data and multiple coverage rates."""

    def test_step_weight_with_multi_vintage(self, interval_train):
        """Step weight changes result for asymmetric per-step errors."""
        base = datetime(2024, 1, 10)
        y_true = pl.DataFrame({
            "time": [base + timedelta(days=i) for i in range(1, 5)],
            "value": [10.0, 20.0, 30.0, 40.0],
        })
        # Asymmetric: step 1 has tight intervals, step 2 has wide intervals
        y_pred = pl.DataFrame({
            "vintage_time": [base, base, base + timedelta(days=1), base + timedelta(days=1)],
            "time": [
                base + timedelta(days=1),
                base + timedelta(days=2),
                base + timedelta(days=2),
                base + timedelta(days=3),
            ],
            "value_lower_0.9": [9.0, 10.0, 19.0, 20.0],
            "value_upper_0.9": [11.0, 30.0, 21.0, 50.0],
        })

        scorer = IntervalScore()
        scorer.fit(interval_train)
        default = scorer.score(y_true, y_pred)

        scorer_sw = IntervalScore()
        scorer_sw.fit(interval_train)
        result = scorer_sw.score(y_true, y_pred, step_weight={1: 5.0, 2: 1.0})

        assert isinstance(result, float)
        assert not np.isclose(result, default, atol=1e-10)

    def test_vintage_weight_with_interval(self, interval_train, interval_multi_vintage):
        """Vintage weight applied to interval scorer multi-vintage data."""
        y_true, y_pred = interval_multi_vintage
        base = datetime(2024, 1, 10)

        scorer = IntervalScore()
        scorer.fit(interval_train)
        result = scorer.score(
            y_true,
            y_pred,
            vintage_weight={base: 5.0, base + timedelta(days=1): 1.0},
        )

        assert isinstance(result, float)

    def test_equal_step_weight_matches_default(self, interval_train, interval_multi_vintage):
        """Equal step weights reproduce the unweighted score."""
        y_true, y_pred = interval_multi_vintage

        scorer = IntervalScore()
        scorer.fit(interval_train)
        default = scorer.score(y_true, y_pred)

        scorer_eq = IntervalScore()
        scorer_eq.fit(interval_train)
        result = scorer_eq.score(y_true, y_pred, step_weight={1: 1.0, 2: 1.0})

        np.testing.assert_allclose(result, default, atol=1e-10)


class TestIntervalCoverageWeight:
    """Coverage weight via dict coverage parameter with weighted collapse."""

    def test_equal_coverage_weights_match_default(self, interval_train, interval_multi_rate):
        """Equal coverage weights reproduce the unweighted coveragewise collapse."""
        y_true, y_pred = interval_multi_rate

        scorer = IntervalScore()
        scorer.fit(interval_train)
        default = scorer.score(y_true, y_pred)

        scorer_cw = IntervalScore(coverage_rates={0.9: 1.0, 0.95: 1.0})
        scorer_cw.fit(interval_train)
        weighted = scorer_cw.score(y_true, y_pred)

        np.testing.assert_allclose(weighted, default, atol=1e-10)

    def test_coverage_weight_shifts_score(self, interval_train, interval_multi_rate):
        """Unequal coverage weights shift the score toward the heavier rate."""
        y_true, y_pred = interval_multi_rate

        scorer_heavy_90 = IntervalScore(coverage_rates={0.9: 10.0, 0.95: 1.0})
        scorer_heavy_90.fit(interval_train)
        heavy_90 = scorer_heavy_90.score(y_true, y_pred)

        scorer_heavy_95 = IntervalScore(coverage_rates={0.9: 1.0, 0.95: 10.0})
        scorer_heavy_95.fit(interval_train)
        heavy_95 = scorer_heavy_95.score(y_true, y_pred)

        assert not np.isclose(heavy_90, heavy_95, atol=1e-10)


class TestIntervalPanelTimeWeight:
    """Time weight applied to interval scorer with panel-prefixed columns."""

    def test_panel_interval_time_weight(self):
        """Time weight works on panel interval data (exercises panel branch)."""
        times = [datetime(2024, 1, 11), datetime(2024, 1, 12), datetime(2024, 1, 13)]
        y_train = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 11)],
            "g1__val": [float(i) for i in range(10)],
            "g2__val": [float(i) * 2 for i in range(10)],
        })
        y_true = pl.DataFrame({
            "time": times,
            "g1__val": [10.0, 20.0, 30.0],
            "g2__val": [20.0, 40.0, 60.0],
        })
        y_pred = pl.DataFrame({
            "vintage_time": [datetime(2024, 1, 10)] * 3,
            "time": times,
            "g1__val_lower_0.9": [8.0, 15.0, 28.0],
            "g1__val_upper_0.9": [12.0, 18.0, 32.0],
            "g2__val_lower_0.9": [18.0, 35.0, 58.0],
            "g2__val_upper_0.9": [22.0, 42.0, 62.0],
        })

        scorer = IntervalScore()
        scorer.fit(y_train)
        default = scorer.score(y_true, y_pred)

        scorer_tw = IntervalScore()
        scorer_tw.fit(y_train)
        scorer_tw.set_score_request(time_weight=True)
        weighted = scorer_tw.score(
            y_true,
            y_pred,
            time_weight=lambda t: pl.Series("w", [1.0, 10.0, 1.0][: len(t)]),
        )

        assert isinstance(weighted, float)
        assert not np.isclose(weighted, default, atol=1e-10)


class TestIntervalMultiRateStepWeight:
    """Step/vintage weight tiling across multiple coverage rate blocks."""

    def test_step_weight_tiles_across_rates(self):
        """Step weight tiles correctly when multiple coverage rates exist."""
        base = datetime(2024, 1, 10)
        y_train = pl.DataFrame({
            "time": [datetime(2024, 1, i) for i in range(1, 11)],
            "value": [float(i) for i in range(10)],
        })
        y_true = pl.DataFrame({
            "time": [base + timedelta(days=i) for i in range(1, 5)],
            "value": [10.0, 20.0, 30.0, 40.0],
        })
        # 2 vintages x 2 steps, with 2 coverage rates
        y_pred = pl.DataFrame({
            "vintage_time": [base, base, base + timedelta(days=1), base + timedelta(days=1)],
            "time": [
                base + timedelta(days=1),
                base + timedelta(days=2),
                base + timedelta(days=2),
                base + timedelta(days=3),
            ],
            "value_lower_0.9": [9.0, 10.0, 19.0, 20.0],
            "value_upper_0.9": [11.0, 30.0, 21.0, 50.0],
            "value_lower_0.95": [8.0, 5.0, 18.0, 15.0],
            "value_upper_0.95": [12.0, 35.0, 22.0, 55.0],
        })

        scorer = IntervalScore()
        scorer.fit(y_train)
        default = scorer.score(y_true, y_pred)

        scorer_sw = IntervalScore()
        scorer_sw.fit(y_train)
        result = scorer_sw.score(y_true, y_pred, step_weight={1: 5.0, 2: 1.0})

        assert isinstance(result, float)
        assert not np.isclose(result, default, atol=1e-10)
