"""Tests for ColumnForecaster meta-forecaster.

Tests ColumnForecaster using both the check generator pattern and
specific meta-forecaster tests for column-wise composition behavior.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.metadata_routing import MetadataRouter

from conftest import run_checks
from yohou.compose import ColumnForecaster
from yohou.point import PointReductionForecaster, SeasonalNaive
from yohou.stationarity import PolynomialTrendForecaster
from yohou.testing import _yield_yohou_forecaster_checks


class TestSystematicChecks:
    """Systematic validation via check generators."""

    @pytest.mark.parametrize(
        "forecaster",
        [
            # Single forecaster covering all columns
            ColumnForecaster([
                ("all", SeasonalNaive(seasonality=1), ["y_0", "y_1"]),
            ]),
            # Multiple forecasters for different columns
            ColumnForecaster([
                ("model_a", SeasonalNaive(seasonality=1), "y_0"),
                ("model_b", SeasonalNaive(seasonality=2), "y_1"),
            ]),
            # With custom remainder
            ColumnForecaster(
                [("main", SeasonalNaive(seasonality=1), "y_0")],
                remainder=SeasonalNaive(seasonality=3),
            ),
            # Different forecaster types
            ColumnForecaster([
                ("trend", PolynomialTrendForecaster(degree=1), "y_0"),
                ("naive", SeasonalNaive(seasonality=1), "y_1"),
            ]),
        ],
        ids=["single_all", "multiple", "custom_remainder", "mixed_types"],
    )
    def test_column_forecaster_systematic_checks(self, forecaster, y_X_factory):
        """Run systematic checks on ColumnForecaster meta-forecaster.

        Note: check_observe_extends_observations and check_rewind_replaces_observations
        are not yielded for ColumnForecaster because it sets tracks_observations=False.
        ColumnForecaster delegates observation tracking to child forecasters.
        """
        y, X_actual = y_X_factory(length=100, n_targets=2, n_features=0, seed=42)

        y_train, y_test = y[:80], y[80:]
        X_actual_train, X_actual_test = (X_actual[:80], X_actual[80:]) if X_actual is not None else (None, None)

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, X_actual_train, forecasting_horizon=3)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(
                forecaster_fitted,
                y_train,
                X_actual_train,
                y_test,
                X_actual_test,
            ),
            expected_failures=set(),
        )


class TestBasicFitPredict:
    """Tests for basic fit/predict workflow."""

    def test_basic_fit_predict(self):
        """Test basic fit and predict workflow."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales": range(50),
            "inventory": range(50, 100),
        })

        forecaster = ColumnForecaster([
            ("sales_model", SeasonalNaive(seasonality=1), "sales"),
            ("inv_model", SeasonalNaive(seasonality=1), "inventory"),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        assert "vintage_time" in y_pred.columns
        assert "time" in y_pred.columns
        assert "sales" in y_pred.columns
        assert "inventory" in y_pred.columns

    def test_single_forecaster_multiple_columns(self):
        """Test single forecaster handling multiple columns."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "a": range(50),
            "b": range(50, 100),
            "c": range(100, 150),
        })

        forecaster = ColumnForecaster([
            ("combined", SeasonalNaive(seasonality=1), ["a", "b", "c"]),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        assert "a" in y_pred.columns
        assert "b" in y_pred.columns
        assert "c" in y_pred.columns

    def test_different_horizons(self):
        """Test prediction with different horizons than fit."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": range(50)})

        forecaster = ColumnForecaster([
            ("main", SeasonalNaive(seasonality=1), "value"),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=10)
        assert len(y_pred) == 10

    def test_single_column_as_string(self):
        """Test that single column can be specified as string (not list)."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": range(50)})

        forecaster = ColumnForecaster([
            ("model", SeasonalNaive(seasonality=1), "value"),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=5)
        assert "value" in y_pred.columns
        assert len(y_pred) == 5


class TestRemainder:
    """Tests for remainder forecaster handling."""

    def test_default_remainder(self):
        """Test default remainder='drop' excludes unassigned columns."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "assigned": range(50),
            "unassigned": range(50, 100),
        })

        forecaster = ColumnForecaster([
            ("main", SeasonalNaive(seasonality=1), "assigned"),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=5)

        # Only assigned column should be predicted (remainder="drop")
        assert "assigned" in y_pred.columns
        assert "unassigned" not in y_pred.columns
        assert len(y_pred) == 5
        assert forecaster.remainder_forecaster_ is None

    def test_custom_remainder(self):
        """Test custom remainder forecaster."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "assigned": range(50),
            "unassigned": range(50, 100),
        })

        custom_remainder = SeasonalNaive(seasonality=7)
        forecaster = ColumnForecaster(
            [("main", SeasonalNaive(seasonality=1), "assigned")],
            remainder=custom_remainder,
        )
        forecaster.fit(y[:30], forecasting_horizon=5)

        assert forecaster.remainder_cols_ == ["unassigned"]
        assert forecaster.remainder_forecaster_ is not None

        y_pred = forecaster.predict(forecasting_horizon=5)
        assert "unassigned" in y_pred.columns

    def test_no_remainder_columns(self):
        """Test when all columns are explicitly assigned."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "a": range(50),
            "b": range(50, 100),
        })

        forecaster = ColumnForecaster([
            ("model_a", SeasonalNaive(seasonality=1), "a"),
            ("model_b", SeasonalNaive(seasonality=1), "b"),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)

        assert forecaster.remainder_cols_ == []
        assert forecaster.remainder_forecaster_ is None

    def test_remainder_multiple_unassigned(self):
        """Test remainder handles multiple unassigned columns."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "assigned": range(50),
            "extra_1": range(50, 100),
            "extra_2": range(100, 150),
        })

        forecaster = ColumnForecaster([
            ("main", SeasonalNaive(seasonality=1), "assigned"),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)

        assert set(forecaster.remainder_cols_) == {"extra_1", "extra_2"}

        y_pred = forecaster.predict(forecasting_horizon=5)
        assert "assigned" in y_pred.columns
        assert "extra_1" not in y_pred.columns
        assert "extra_2" not in y_pred.columns


class TestColumnAssignmentValidation:
    """Tests for column assignment validation."""

    def test_duplicate_names_raises(self):
        """Test that duplicate forecaster names raise ValueError."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "a": range(50), "b": range(50)})

        forecaster = ColumnForecaster([
            ("model", SeasonalNaive(seasonality=1), "a"),
            ("model", SeasonalNaive(seasonality=2), "b"),
        ])

        with pytest.raises(ValueError, match="Duplicate forecaster names"):
            forecaster.fit(y[:30], forecasting_horizon=5)

    def test_overlapping_columns_raises(self):
        """Test that overlapping column assignments raise ValueError."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "a": range(50), "b": range(50)})

        forecaster = ColumnForecaster([
            ("model_1", SeasonalNaive(seasonality=1), ["a", "b"]),
            ("model_2", SeasonalNaive(seasonality=1), "a"),
        ])

        with pytest.raises(ValueError, match="assigned to multiple forecasters"):
            forecaster.fit(y[:30], forecasting_horizon=5)

    def test_nonexistent_column_raises(self):
        """Test that referencing non-existent columns raises ValueError."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "a": range(50)})

        forecaster = ColumnForecaster([
            ("model", SeasonalNaive(seasonality=1), "nonexistent"),
        ])

        with pytest.raises(ValueError, match="non-existent columns"):
            forecaster.fit(y[:30], forecasting_horizon=5)


class TestNamedForecasters:
    """Tests for named_forecasters property."""

    def test_named_forecasters_property(self):
        """Test named_forecasters_ property access."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales": range(50),
            "inventory": range(50, 100),
        })

        forecaster = ColumnForecaster([
            ("sales_model", SeasonalNaive(seasonality=1), "sales"),
            ("inv_model", SeasonalNaive(seasonality=7), "inventory"),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)

        named = forecaster.named_forecasters_
        assert "sales_model" in named
        assert "inv_model" in named
        assert isinstance(named["sales_model"], SeasonalNaive)
        assert isinstance(named["inv_model"], SeasonalNaive)

    def test_named_forecasters_before_fit_raises(self):
        """Test named_forecasters_ raises NotFittedError before fit."""
        forecaster = ColumnForecaster([
            ("model", SeasonalNaive(seasonality=1), "value"),
        ])

        with pytest.raises(NotFittedError, match="fitted"):
            _ = forecaster.named_forecasters_


class TestGetSetParams:
    """Tests for get_params and set_params."""

    def test_get_params(self):
        """Test get_params returns nested forecaster parameters."""
        forecaster = ColumnForecaster([
            ("naive", SeasonalNaive(seasonality=5), "value"),
        ])

        params = forecaster.get_params(deep=True)

        assert "naive__seasonality" in params
        assert params["naive__seasonality"] == 5
        assert "remainder" in params
        assert "n_jobs" in params

    def test_get_params_shallow(self):
        """Test get_params with deep=False."""
        forecaster = ColumnForecaster([
            ("naive", SeasonalNaive(seasonality=5), "value"),
        ])

        params = forecaster.get_params(deep=False)

        # Shallow params should NOT include nested params
        assert "naive__seasonality" not in params
        assert "forecasters" in params
        assert "remainder" in params

    def test_set_params(self):
        """Test set_params modifies nested forecaster parameters."""
        forecaster = ColumnForecaster([
            ("naive", SeasonalNaive(seasonality=1), "value"),
        ])

        forecaster.set_params(naive__seasonality=10)

        params = forecaster.get_params(deep=True)
        assert params["naive__seasonality"] == 10

    def test_clone_preserves_params(self):
        """Test that clone preserves all parameters."""
        forecaster = ColumnForecaster(
            [("model", SeasonalNaive(seasonality=7), ["a", "b"])],
            remainder=SeasonalNaive(seasonality=14),
            n_jobs=2,
            verbose_feature_names_out=True,
        )

        cloned = clone(forecaster)

        assert cloned.n_jobs == 2
        assert cloned.verbose_feature_names_out is True
        assert len(cloned.forecasters) == 1
        assert cloned.forecasters[0][0] == "model"

        # Cloned should not be fitted
        with pytest.raises(NotFittedError, match="fitted"):
            _ = cloned.named_forecasters_


class TestUpdateReset:
    """Tests for observe, rewind, and observe_predict."""

    def test_observe(self):
        """Test observe method propagates to child forecasters."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": range(50)})

        forecaster = ColumnForecaster([
            ("main", SeasonalNaive(seasonality=1), "value"),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)

        forecaster.observe(y[30:35])

        y_pred = forecaster.predict(forecasting_horizon=5)
        assert len(y_pred) == 5

    def test_observe_multiple_forecasters(self):
        """Test observe dispatches correct column subsets to each forecaster."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "a": range(50),
            "b": range(50, 100),
        })

        forecaster = ColumnForecaster([
            ("model_a", SeasonalNaive(seasonality=1), "a"),
            ("model_b", SeasonalNaive(seasonality=1), "b"),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)

        # Update with both columns
        forecaster.observe(y[30:35])

        y_pred = forecaster.predict(forecasting_horizon=5)
        assert "a" in y_pred.columns
        assert "b" in y_pred.columns
        assert len(y_pred) == 5

    def test_observe_with_remainder(self):
        """Test observe propagates to remainder forecaster."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "assigned": range(50),
            "unassigned": range(50, 100),
        })

        forecaster = ColumnForecaster(
            [("main", SeasonalNaive(seasonality=1), "assigned")],
            remainder=SeasonalNaive(seasonality=1),
        )
        forecaster.fit(y[:30], forecasting_horizon=5)
        forecaster.observe(y[30:35])

        y_pred = forecaster.predict(forecasting_horizon=5)
        assert "assigned" in y_pred.columns
        assert "unassigned" in y_pred.columns

    def test_observe_predict(self):
        """Test observe_predict combined method."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": range(50)})

        forecaster = ColumnForecaster([
            ("main", SeasonalNaive(seasonality=1), "value"),
        ])
        fit_fh = 5
        forecaster.fit(y[:30], forecasting_horizon=fit_fh)

        n_new = 10
        y_pred = forecaster.observe_predict(y[30 : 30 + n_new], forecasting_horizon=5)

        # Initial predict + n_new/stride updates = 1 + 10/5 = 3 predictions of size 5
        assert len(y_pred) == fit_fh * (1 + n_new // fit_fh)
        assert "vintage_time" in y_pred.columns

    def test_rewind(self):
        """Test rewind method."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": range(50)})

        forecaster = ColumnForecaster([
            ("main", SeasonalNaive(seasonality=1), "value"),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)

        forecaster.observe(y[30:35])
        forecaster.rewind(y[20:30])

        y_pred = forecaster.predict(forecasting_horizon=5)
        assert len(y_pred) == 5

    def test_rewind_multiple_forecasters(self):
        """Test rewind dispatches correct column subsets to each forecaster."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "a": range(50),
            "b": range(50, 100),
        })

        forecaster = ColumnForecaster([
            ("model_a", SeasonalNaive(seasonality=1), "a"),
            ("model_b", SeasonalNaive(seasonality=1), "b"),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)

        forecaster.observe(y[30:35])
        forecaster.rewind(y[20:30])

        y_pred = forecaster.predict(forecasting_horizon=5)
        assert "a" in y_pred.columns
        assert "b" in y_pred.columns

    def test_not_fitted_predict_raises(self):
        """Test predict raises NotFittedError before fit."""
        forecaster = ColumnForecaster([
            ("model", SeasonalNaive(seasonality=1), "value"),
        ])

        with pytest.raises(NotFittedError, match="fitted"):
            forecaster.predict(forecasting_horizon=5)

    def test_not_fitted_observe_raises(self):
        """Test observe raises NotFittedError before fit."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=9),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": range(10)})

        forecaster = ColumnForecaster([
            ("model", SeasonalNaive(seasonality=1), "value"),
        ])

        with pytest.raises(NotFittedError, match="fitted"):
            forecaster.observe(y)

    def test_not_fitted_rewind_raises(self):
        """Test rewind raises NotFittedError before fit."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=9),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": range(10)})

        forecaster = ColumnForecaster([
            ("model", SeasonalNaive(seasonality=1), "value"),
        ])

        with pytest.raises(NotFittedError, match="fitted"):
            forecaster.rewind(y)


class TestPanelData:
    """Tests for panel data support."""

    def test_panel_data(self, y_X_factory):
        """Test with panel data (prefixed column names with __ separator)."""
        y, _ = y_X_factory(length=50, n_targets=2, n_features=0, seed=42, panel=True, n_groups=2)

        panel_cols = [c for c in y.columns if c != "time"]
        forecaster = ColumnForecaster([
            ("model", SeasonalNaive(seasonality=1), panel_cols),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        assert "vintage_time" in y_pred.columns
        assert "time" in y_pred.columns
        for col in panel_cols:
            assert col in y_pred.columns

    def test_panel_data_split_across_forecasters(self, y_X_factory):
        """Test panel columns split across different forecasters."""
        y, _ = y_X_factory(length=50, n_targets=2, n_features=0, seed=42, panel=True, n_groups=2)

        # Split panel columns between forecasters
        panel_cols = [c for c in y.columns if c != "time"]
        mid = len(panel_cols) // 2
        group_a = panel_cols[:mid]
        group_b = panel_cols[mid:]

        forecaster = ColumnForecaster([
            ("group_a", SeasonalNaive(seasonality=1), group_a),
            ("group_b", SeasonalNaive(seasonality=2), group_b),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        for col in panel_cols:
            assert col in y_pred.columns


class TestParallelFitting:
    """Tests for parallel fitting (n_jobs)."""

    @pytest.mark.slow
    def test_parallel_fitting(self):
        """Test parallel fitting with n_jobs."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "a": range(50),
            "b": range(50, 100),
            "c": range(100, 150),
        })

        forecaster = ColumnForecaster(
            [
                ("model_a", SeasonalNaive(seasonality=1), "a"),
                ("model_b", SeasonalNaive(seasonality=2), "b"),
                ("model_c", SeasonalNaive(seasonality=3), "c"),
            ],
            n_jobs=2,
        )
        forecaster.fit(y[:30], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        assert "a" in y_pred.columns
        assert "b" in y_pred.columns
        assert "c" in y_pred.columns

    def test_parallel_same_result_as_sequential(self):
        """Test parallel fitting produces same results as sequential."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "a": range(50),
            "b": range(50, 100),
        })

        forecasters_spec = [
            ("model_a", SeasonalNaive(seasonality=1), "a"),
            ("model_b", SeasonalNaive(seasonality=2), "b"),
        ]

        sequential = ColumnForecaster(forecasters_spec, n_jobs=1)
        sequential.fit(y[:30], forecasting_horizon=5)
        y_seq = sequential.predict(forecasting_horizon=5)

        parallel = clone(ColumnForecaster(forecasters_spec, n_jobs=2))
        parallel.fit(y[:30], forecasting_horizon=5)
        y_par = parallel.predict(forecasting_horizon=5)

        assert y_seq.select(["a", "b"]).equals(y_par.select(["a", "b"]))


class TestSklearnTags:
    """Tests for sklearn tags."""

    def test_tags_tracks_observations_false(self):
        """Test that ColumnForecaster has tracks_observations=False."""
        forecaster = ColumnForecaster([
            ("model", SeasonalNaive(seasonality=1), "value"),
        ])

        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.tracks_observations is False

    def test_tags_aggregates_from_children(self):
        """Test that tags are aggregated from child forecasters."""
        forecaster = ColumnForecaster([
            ("trend", PolynomialTrendForecaster(degree=1), "a"),
            ("naive", SeasonalNaive(seasonality=1), "b"),
        ])

        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.forecaster_type == frozenset({"point"})

    def test_tags_accessible_before_fit(self):
        """Test tags accessible before fitting."""
        forecaster = ColumnForecaster([
            ("model", SeasonalNaive(seasonality=1), "value"),
        ])

        # Should not raise
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags is not None

    def test_tags_include_remainder(self):
        """Test tags aggregate includes remainder forecaster."""
        forecaster = ColumnForecaster(
            [("main", SeasonalNaive(seasonality=1), "a")],
            remainder=SeasonalNaive(seasonality=7),
        )

        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.forecaster_type == frozenset({"point"})


class TestExogenousFeatures:
    """Tests for exogenous feature support."""

    def test_with_exogenous(self, y_X_factory):
        """Test with exogenous features."""
        y, X_actual = y_X_factory(length=50, n_targets=2, n_features=2, seed=42)

        target_cols = [c for c in y.columns if c != "time"]

        forecaster = ColumnForecaster([
            ("model", SeasonalNaive(seasonality=1), target_cols),
        ])
        forecaster.fit(y[:30], X_actual=X_actual[:30], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=5)

        assert len(y_pred) == 5

    def test_with_exogenous_multiple_forecasters(self, y_X_factory):
        """Test exogenous features shared across multiple forecasters."""
        y, X_actual = y_X_factory(length=50, n_targets=2, n_features=2, seed=42)

        target_cols = [c for c in y.columns if c != "time"]

        forecaster = ColumnForecaster([
            ("model_0", SeasonalNaive(seasonality=1), target_cols[0]),
            ("model_1", SeasonalNaive(seasonality=1), target_cols[1]),
        ])
        forecaster.fit(y[:30], X_actual=X_actual[:30], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        for col in target_cols:
            assert col in y_pred.columns


class TestFittedAttributes:
    """Tests for fitted attributes."""

    def test_forecasters_attribute_structure(self):
        """Test forecasters_ fitted attribute structure."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "a": range(50),
            "b": range(50, 100),
        })

        forecaster = ColumnForecaster([
            ("model_a", SeasonalNaive(seasonality=1), "a"),
            ("model_b", SeasonalNaive(seasonality=2), "b"),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)

        assert len(forecaster.forecasters_) == 2
        name_0, fitted_0, cols_0 = forecaster.forecasters_[0]
        assert name_0 == "model_a"
        assert isinstance(fitted_0, SeasonalNaive)
        assert cols_0 == ["a"]

        name_1, fitted_1, cols_1 = forecaster.forecasters_[1]
        assert name_1 == "model_b"
        assert isinstance(fitted_1, SeasonalNaive)
        assert cols_1 == ["b"]

    def test_column_map_attribute(self):
        """Test column_map_ attribute."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "sales": range(50),
            "inventory": range(50, 100),
            "price": range(100, 150),
        })

        forecaster = ColumnForecaster([
            ("sales_inv", SeasonalNaive(seasonality=1), ["sales", "inventory"]),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)

        assert forecaster.column_map_ == {"sales_inv": ["sales", "inventory"]}
        assert forecaster.remainder_cols_ == ["price"]

    def test_fit_forecasting_horizon_stored(self):
        """Test fit_forecasting_horizon_ is stored after fit."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": range(50)})

        forecaster = ColumnForecaster([
            ("main", SeasonalNaive(seasonality=1), "value"),
        ])
        forecaster.fit(y[:30], forecasting_horizon=7)

        assert forecaster.fit_forecasting_horizon_ == 7

    def test_interval_attribute(self):
        """Test interval_ is set from first forecaster."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "value": range(50)})

        forecaster = ColumnForecaster([
            ("main", SeasonalNaive(seasonality=1), "value"),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)

        assert forecaster.interval_ == "1d"

    def test_schemas_combined(self):
        """Test local_y_schema_ combines schemas from all forecasters."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "a": range(50),
            "b": range(50, 100),
            "c": range(100, 150),
        })

        forecaster = ColumnForecaster([
            ("model_a", SeasonalNaive(seasonality=1), "a"),
            ("model_b", SeasonalNaive(seasonality=1), "b"),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)

        # Schema should include columns from all forecasters + remainder
        assert "a" in forecaster.local_y_schema_
        assert "b" in forecaster.local_y_schema_
        assert "c" not in forecaster.local_y_schema_  # remainder="drop" excludes it


class TestMetadataRouting:
    """Tests for metadata routing."""

    def test_metadata_routing(self):
        """Test get_metadata_routing returns valid router."""
        forecaster = ColumnForecaster([
            ("naive_a", SeasonalNaive(seasonality=1), "a"),
            ("naive_b", SeasonalNaive(seasonality=2), "b"),
        ])

        router = forecaster.get_metadata_routing()

        assert isinstance(router, MetadataRouter)


class TestIntervalForecasterComposition:
    """Tests for interval forecaster composition."""

    def test_with_interval_forecaster(self):
        """Test ColumnForecaster with interval forecasters."""
        from yohou.interval import SplitConformalForecaster

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=99),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "value": [float(i) for i in range(100)],
        })

        forecaster = ColumnForecaster([
            (
                "interval_model",
                SplitConformalForecaster(
                    point_forecaster=SeasonalNaive(seasonality=1),
                    calibration_size=20,
                ),
                "value",
            ),
        ])
        forecaster.fit(y[:70], forecasting_horizon=5)

        y_pred = forecaster.predict_interval(forecasting_horizon=5, coverage_rates=[0.9])

        assert len(y_pred) == 5
        assert any("lower" in col for col in y_pred.columns)
        assert any("upper" in col for col in y_pred.columns)

    def test_mixed_point_interval_tags(self):
        """Test tags when mixing point and interval forecasters."""
        from yohou.interval import SplitConformalForecaster

        forecaster = ColumnForecaster([
            ("point", SeasonalNaive(seasonality=1), "a"),
            (
                "interval",
                SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=1)),
                "b",
            ),
        ])

        tags = forecaster.__sklearn_tags__()
        # Mixed should be frozenset({"point", "interval"})
        assert tags.forecaster_tags.forecaster_type == frozenset({"point", "interval"})


class TestConditionalMethodAvailability:
    """Tests for conditional method availability."""

    def test_point_forecaster_has_predict(self):
        """Test that point-only ColumnForecaster exposes predict and observe_predict."""
        forecaster = ColumnForecaster([
            ("naive", SeasonalNaive(seasonality=1), "a"),
        ])
        assert hasattr(forecaster, "predict")
        assert hasattr(forecaster, "observe_predict")

    def test_point_forecaster_no_predict_interval(self):
        """Test that point-only ColumnForecaster does not expose predict_interval."""
        forecaster = ColumnForecaster([
            ("naive", SeasonalNaive(seasonality=1), "a"),
        ])
        assert not hasattr(forecaster, "predict_interval")
        assert not hasattr(forecaster, "observe_predict_interval")

    def test_interval_forecaster_has_predict_and_predict_interval(self):
        """Test that all-interval ColumnForecaster exposes all methods.

        When both specified forecasters and the remainder support
        predict_interval, all prediction methods are available.
        """
        from yohou.interval import SplitConformalForecaster

        remainder = SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=1))
        forecaster = ColumnForecaster(
            [("interval", SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=1)), "a")],
            remainder=remainder,
        )
        assert hasattr(forecaster, "predict")
        assert hasattr(forecaster, "observe_predict")
        assert hasattr(forecaster, "predict_interval")
        assert hasattr(forecaster, "observe_predict_interval")

    def test_mixed_point_and_interval_no_predict_interval(self):
        """Test that mixing point and interval forecasters hides predict_interval.

        When some children lack predict_interval, the method is not exposed
        on ColumnForecaster. predict remains available since all forecasters
        have it.
        """
        from yohou.interval import SplitConformalForecaster

        forecaster = ColumnForecaster([
            ("point", SeasonalNaive(seasonality=1), "a"),
            ("interval", SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=1)), "b"),
        ])
        assert hasattr(forecaster, "predict")
        assert hasattr(forecaster, "observe_predict")
        assert not hasattr(forecaster, "predict_interval")
        assert not hasattr(forecaster, "observe_predict_interval")

    def test_drop_remainder_does_not_block_predict_interval(self):
        """Test that remainder='drop' does not block predict_interval.

        When remainder is a string ("drop" or "passthrough"), it has no
        forecaster and does not participate in method availability checks.
        """
        from yohou.interval import SplitConformalForecaster

        forecaster = ColumnForecaster([
            ("interval", SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=1)), "a"),
        ])
        # remainder="drop" (default) has no forecaster, so predict_interval is available
        assert hasattr(forecaster, "predict_interval")
        assert hasattr(forecaster, "observe_predict_interval")

    def test_forecaster_remainder_blocks_predict_interval(self):
        """Test that a point-only remainder forecaster blocks predict_interval.

        When remainder is a BaseForecaster without predict_interval,
        the method is not exposed.
        """
        from yohou.interval import SplitConformalForecaster

        forecaster = ColumnForecaster(
            [("interval", SplitConformalForecaster(point_forecaster=SeasonalNaive(seasonality=1)), "a")],
            remainder=SeasonalNaive(seasonality=1),
        )
        # Point-only remainder blocks predict_interval
        assert not hasattr(forecaster, "predict_interval")


class TestEdgeCases:
    """Tests for edge cases."""

    def test_passthrough_remainder(self):
        """Test remainder='passthrough' excludes unassigned columns from predictions."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "assigned": range(50),
            "unassigned": range(50, 100),
        })

        forecaster = ColumnForecaster(
            [("main", SeasonalNaive(seasonality=1), "assigned")],
            remainder="passthrough",
        )
        forecaster.fit(y[:30], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=5)
        assert "assigned" in y_pred.columns
        assert "unassigned" not in y_pred.columns
        assert forecaster.remainder_forecaster_ is None

    def test_many_forecasters(self):
        """Test with many forecasters."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        cols = {f"col_{i}": range(i * 10, 50 + i * 10) for i in range(10)}
        y = pl.DataFrame({"time": time, **cols})

        forecasters_spec = [(f"model_{i}", SeasonalNaive(seasonality=1), f"col_{i}") for i in range(10)]

        forecaster = ColumnForecaster(forecasters_spec)
        forecaster.fit(y[:30], forecasting_horizon=5)

        y_pred = forecaster.predict(forecasting_horizon=5)

        assert len(y_pred) == 5
        for i in range(10):
            assert f"col_{i}" in y_pred.columns


class TestColumnForecasterIntervalObservePredict:
    """Tests for observe_predict_interval and remainder interval prediction."""

    @pytest.fixture
    def interval_forecaster_and_data(self):
        """Fitted interval ColumnForecaster and data for observe_predict_interval."""
        from yohou.interval import SplitConformalForecaster

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=99),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "value": [float(i) for i in range(100)],
            "extra": [float(i * 2) for i in range(100)],
        })

        forecaster = ColumnForecaster([
            (
                "interval_model",
                SplitConformalForecaster(
                    point_forecaster=SeasonalNaive(seasonality=1),
                    calibration_size=20,
                ),
                "value",
            ),
            (
                "interval_extra",
                SplitConformalForecaster(
                    point_forecaster=SeasonalNaive(seasonality=1),
                    calibration_size=20,
                ),
                "extra",
            ),
        ])
        forecaster.fit(y[:70], forecasting_horizon=5)
        return forecaster, y

    def test_observe_predict_interval(self, interval_forecaster_and_data):
        """observe_predict_interval alternates observe and predict."""
        forecaster, y = interval_forecaster_and_data
        y_pred = forecaster.observe_predict_interval(
            y=y[70:80],
            forecasting_horizon=5,
            coverage_rates=[0.9],
        )
        assert len(y_pred) > 5
        assert any("lower" in c for c in y_pred.columns)
        assert any("upper" in c for c in y_pred.columns)

    def test_observe_predict_interval_with_stride(self, interval_forecaster_and_data):
        """observe_predict_interval respects custom stride."""
        forecaster, y = interval_forecaster_and_data
        y_pred = forecaster.observe_predict_interval(
            y=y[70:80],
            forecasting_horizon=5,
            coverage_rates=[0.9],
            stride=2,
        )
        assert len(y_pred) > 5

    def test_remainder_interval_prediction(self):
        """Remainder forecaster contributes interval predictions."""
        from yohou.interval import SplitConformalForecaster

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=99),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "main_col": [float(i) for i in range(100)],
            "other_col": [float(i * 3) for i in range(100)],
        })

        forecaster = ColumnForecaster(
            [
                (
                    "main",
                    SplitConformalForecaster(
                        point_forecaster=SeasonalNaive(seasonality=1),
                        calibration_size=20,
                    ),
                    "main_col",
                ),
            ],
            remainder=SplitConformalForecaster(
                point_forecaster=SeasonalNaive(seasonality=1),
                calibration_size=20,
            ),
        )
        forecaster.fit(y[:70], forecasting_horizon=5)
        y_pred = forecaster.predict_interval(forecasting_horizon=5, coverage_rates=[0.9])

        assert any("main_col" in c for c in y_pred.columns)
        assert any("other_col" in c for c in y_pred.columns)
        assert any("lower" in c for c in y_pred.columns)


class TestColumnForecasterTagsBeforeFit:
    """Tests for __sklearn_tags__() before fit."""

    def test_tags_before_fit_with_remainder_forecaster(self):
        """Tags before fit includes remainder forecaster when it is a BaseForecaster."""
        forecaster = ColumnForecaster(
            [("main", SeasonalNaive(seasonality=1), "a")],
            remainder=SeasonalNaive(seasonality=1),
        )
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags is not None
        assert tags.forecaster_tags.forecaster_type == frozenset({"point"})

    def test_tags_before_fit_without_remainder(self):
        """Tags before fit without remainder forecaster."""
        forecaster = ColumnForecaster(
            [("main", SeasonalNaive(seasonality=1), "a")],
        )
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags is not None
        assert tags.forecaster_tags.forecaster_type == frozenset({"point"})


class TestColumnForecasterRewindWithRemainder:
    """Tests for rewind with remainder forecaster."""

    def test_rewind_with_remainder_forecaster(self):
        """Rewind dispatches to both main and remainder forecasters."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "a": range(50),
            "b": range(50, 100),
        })
        forecaster = ColumnForecaster(
            [("main", SeasonalNaive(seasonality=1), "a")],
            remainder=SeasonalNaive(seasonality=1),
        )
        forecaster.fit(y[:30], forecasting_horizon=5)

        forecaster.observe(y[30:35])
        result = forecaster.rewind(y[20:30])
        assert result is forecaster

        y_pred = forecaster.predict(forecasting_horizon=5)
        assert "a" in y_pred.columns
        assert "b" in y_pred.columns


class TestColumnForecasterObservePredictWithX:
    """Tests for observe_predict with exogenous features."""

    def test_observe_predict_with_exogenous(self):
        """observe_predict passes X_future to individual forecasters."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "a": range(50),
        })
        X_actual = pl.DataFrame({
            "time": time,
            "feat": list(range(50)),
        })
        forecaster = ColumnForecaster(
            [("main", PointReductionForecaster(), "a")],
        )
        forecaster.fit(y[:30], X_actual=X_actual[:30], forecasting_horizon=5)

        time_ext = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=54),
            interval="1d",
            eager=True,
        )
        X_ext = pl.DataFrame({
            "time": time_ext,
            "feat": list(range(55)),
        })
        y_pred = forecaster.observe_predict(y[30:35], X_actual=X_ext)
        assert len(y_pred) > 0
        assert "a" in y_pred.columns


class TestColumnForecasterMixedTags:
    """Tests for tags with mixed point and interval forecasters."""

    def test_tags_both_when_point_and_interval(self):
        """Tags report 'both' when mixing point and interval forecasters."""
        from yohou.interval import SplitConformalForecaster

        forecaster = ColumnForecaster(
            [
                ("point_col", SeasonalNaive(seasonality=1), "a"),
                (
                    "interval_col",
                    SplitConformalForecaster(
                        point_forecaster=SeasonalNaive(),
                        calibration_size=5,
                    ),
                    "b",
                ),
            ],
        )
        tags = forecaster.__sklearn_tags__()
        assert tags.forecaster_tags.forecaster_type == frozenset({"point", "interval"})


class TestClassProbaColumnForecaster:
    """Tests for class-probability methods on ColumnForecaster."""

    @pytest.fixture
    def class_proba_column_setup(self):
        """Create a fitted ColumnForecaster with class_proba forecasters."""
        from sklearn.tree import DecisionTreeClassifier

        from yohou.class_proba import ClassProbaReductionForecaster

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=79),
            interval="1d",
            eager=True,
        )
        classes_a = ["cat", "dog", "bird"]
        classes_b = ["red", "blue", "green"]
        y = pl.DataFrame({
            "time": time,
            "animal": [classes_a[i % 3] for i in range(80)],
            "color": [classes_b[i % 3] for i in range(80)],
        })
        X_actual = pl.DataFrame({
            "time": time,
            "temp": [20.0 + (i % 10) for i in range(80)],
        })

        forecaster = ColumnForecaster([
            (
                "animal_model",
                ClassProbaReductionForecaster(
                    estimator=DecisionTreeClassifier(random_state=42),
                ),
                "animal",
            ),
            (
                "color_model",
                ClassProbaReductionForecaster(
                    estimator=DecisionTreeClassifier(random_state=42),
                ),
                "color",
            ),
        ])
        y_train, y_test = y[:60], y[60:]
        X_actual_train, X_actual_test = X_actual[:60], X_actual[60:]
        forecaster.fit(y_train, X_actual_train, forecasting_horizon=3)
        return forecaster, y_train, y_test, X_actual_train, X_actual_test

    def test_predict_class_proba(self, class_proba_column_setup):
        """predict_class_proba returns probabilities from all forecasters."""
        forecaster, _, _, _, X_actual_test = class_proba_column_setup
        y_pred = forecaster.predict_class_proba(forecasting_horizon=3)

        assert "time" in y_pred.columns
        animal_proba = [c for c in y_pred.columns if "animal_proba_" in c]
        color_proba = [c for c in y_pred.columns if "color_proba_" in c]
        assert len(animal_proba) == 3
        assert len(color_proba) == 3
        assert len(y_pred) == 3

    def test_observe_predict_class_proba(self, class_proba_column_setup):
        """observe_predict_class_proba observes and predicts probabilities."""
        forecaster, _, y_test, _, X_actual_test = class_proba_column_setup
        y_pred = forecaster.observe_predict_class_proba(
            y=y_test[:3],
            X_actual=X_actual_test[:3],
            forecasting_horizon=3,
        )
        assert "time" in y_pred.columns
        proba_cols = [c for c in y_pred.columns if "_proba_" in c]
        assert len(proba_cols) > 0

    def test_predict_class_proba_with_remainder(self):
        """predict_class_proba includes remainder forecaster predictions."""
        from sklearn.tree import DecisionTreeClassifier

        from yohou.class_proba import ClassProbaReductionForecaster

        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=79),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "animal": [["cat", "dog", "bird"][i % 3] for i in range(80)],
            "color": [["red", "blue"][i % 2] for i in range(80)],
        })
        forecaster = ColumnForecaster(
            [
                (
                    "animal_model",
                    ClassProbaReductionForecaster(
                        estimator=DecisionTreeClassifier(random_state=42),
                    ),
                    "animal",
                ),
            ],
            remainder=ClassProbaReductionForecaster(
                estimator=DecisionTreeClassifier(random_state=42),
            ),
        )
        forecaster.fit(y, forecasting_horizon=3)
        y_pred = forecaster.predict_class_proba(forecasting_horizon=3)

        animal_proba = [c for c in y_pred.columns if "animal_proba_" in c]
        color_proba = [c for c in y_pred.columns if "color_proba_" in c]
        assert len(animal_proba) == 3
        assert len(color_proba) == 2
        assert len(y_pred) == 3


class TestColumnForecasterExogenous:
    """Tests for ColumnForecaster with X_future and X_forecast."""

    def test_fit_predict_with_X_future_and_X_forecast(self):
        """ColumnForecaster passes X_future/X_forecast to child forecasters."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=59),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "a": [float(i) for i in range(60)],
        })

        # X_future covers beyond y's range
        time_ext = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=69),
            interval="1d",
            eager=True,
        )
        X_future = pl.DataFrame({
            "time": time_ext,
            "holiday": [1.0 if t.weekday() == 6 else 0.0 for t in time_ext],
        })

        forecaster = ColumnForecaster(
            [("main", PointReductionForecaster(), "a")],
        )
        forecaster.fit(y[:40], forecasting_horizon=5, X_future=X_future)

        y_pred = forecaster.predict()
        assert len(y_pred) == 5
        assert "a" in y_pred.columns

    def test_predict_X_forecast_override(self):
        """ColumnForecaster passes X_forecast override to children's predict."""
        rng = np.random.default_rng(42)
        n_obs = 60
        fh = 5
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=n_obs - 1),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({
            "time": time,
            "a": rng.random(n_obs).tolist(),
        })

        # Build X_forecast with vintage coverage beyond y range
        rows = []
        for v_idx in range(n_obs):
            vt = time[v_idx]
            for step in range(1, fh + 1):
                target_idx = v_idx + step
                if target_idx < n_obs + fh + 5:
                    tt = time[target_idx] if target_idx < n_obs else time[-1] + timedelta(days=target_idx - n_obs + 1)
                    rows.append({"vintage_time": vt, "time": tt, "wx": float(rng.random())})
        X_forecast = pl.DataFrame(rows)

        forecaster = ColumnForecaster(
            [("main", PointReductionForecaster(), "a")],
        )
        forecaster.fit(y[:40], forecasting_horizon=fh, X_forecast=X_forecast)

        pred1 = forecaster.predict()
        pred2 = forecaster.predict(X_forecast=X_forecast)

        # Both should produce valid predictions
        assert len(pred1) == fh
        assert len(pred2) == fh


class TestVerboseFeatureNamesOut:
    """Tests for the verbose_feature_names_out prefixing behavior."""

    def _make_y(self):
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=49),
            interval="1d",
            eager=True,
        )
        return pl.DataFrame({"time": time, "sales": range(50), "inventory": range(50, 100)})

    def test_verbose_false_keeps_bare_names(self):
        """Default verbose_feature_names_out=False keeps original column names."""
        y = self._make_y()
        forecaster = ColumnForecaster([
            ("sales_model", SeasonalNaive(seasonality=1), "sales"),
            ("inv_model", SeasonalNaive(seasonality=1), "inventory"),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        assert "sales" in y_pred.columns
        assert "inventory" in y_pred.columns
        assert "sales_model__sales" not in y_pred.columns

    def test_verbose_true_prefixes_with_forecaster_name(self):
        """verbose_feature_names_out=True prefixes columns with the forecaster name."""
        y = self._make_y()
        forecaster = ColumnForecaster(
            [
                ("sales_model", SeasonalNaive(seasonality=1), "sales"),
                ("inv_model", SeasonalNaive(seasonality=1), "inventory"),
            ],
            verbose_feature_names_out=True,
        )
        forecaster.fit(y[:30], forecasting_horizon=5)
        y_pred = forecaster.predict(forecasting_horizon=5)

        assert "sales_model__sales" in y_pred.columns
        assert "inv_model__inventory" in y_pred.columns
        assert "sales" not in y_pred.columns
        assert "inventory" not in y_pred.columns


class TestObserveBuffersBounded:
    """observe() must not accumulate an unbounded observation buffer."""

    def test_observe_buffer_holds_only_latest_batch(self):
        """Repeated observe() keeps _y_observed bounded to the latest batch."""
        time = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=199),
            interval="1d",
            eager=True,
        )
        y = pl.DataFrame({"time": time, "sales": range(200), "inventory": range(200, 400)})

        forecaster = ColumnForecaster([
            ("sales_model", SeasonalNaive(seasonality=1), "sales"),
            ("inv_model", SeasonalNaive(seasonality=1), "inventory"),
        ])
        forecaster.fit(y[:30], forecasting_horizon=5)

        for start in range(30, 90, 10):
            forecaster.observe(y[start : start + 10])

        # Buffer holds only the most recent batch, not the concatenation of all.
        assert forecaster._y_observed.height == 10
