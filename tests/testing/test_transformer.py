"""Tests for yohou.testing.transformer check functions."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from conftest import run_checks
from yohou.preprocessing.function import FunctionTransformer
from yohou.preprocessing.window import LagTransformer
from yohou.stationarity.transformers import LogTransformer, SeasonalDifferencing
from yohou.testing import _yield_yohou_transformer_checks
from yohou.testing.transformer import (
    check_feature_names_out_match,
    check_fit_idempotent,
    check_fit_sets_attributes,
    check_fit_transform_equivalence,
    check_insufficient_data_raises,
    check_inverse_observe_transform_identity,
    check_inverse_transform_identity,
    check_inverse_transform_round_trip,
    check_memory_bounded,
    check_observation_horizon_after_fit,
    check_observation_horizon_not_fitted,
    check_observe_concatenates_memory,
    check_observe_transform_equivalence,
    check_observe_transform_sequential_consistency,
    check_panel_data_support,
    check_panel_group_preservation,
    check_rewind_transform_behavior,
    check_rewind_updates_memory,
    check_tags_accessible_before_fit,
    check_tags_match_capabilities,
    check_tags_static_after_fit,
    check_transform_output_structure,
    check_transformer_preserve_dtypes,
    check_transformers_unfitted_stateless,
)


class TestTransformerFitChecks:
    """Tests for transformer fit-related check functions."""

    def test_fit_sets_attributes(self, y_X_factory):
        """Test check_fit_sets_attributes passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)

        # Should not raise
        check_fit_sets_attributes(transformer, X[:40], y[:40])

    def test_fit_sets_attributes_no_y(self, y_X_factory):
        """Test check validates transformer that doesn't use y."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)

        # Should not raise - y is optional
        check_fit_sets_attributes(transformer, X[:40], y=None)

    def test_fit_idempotent(self, y_X_factory):
        """Test check_fit_idempotent passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:40], y[:40])

        # Should not raise
        check_fit_idempotent(transformer, X[:40], y[:40])

    def test_fit_transform_equivalence(self, y_X_factory):
        """Test check_fit_transform_equivalence passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)

        # Should not raise
        check_fit_transform_equivalence(transformer, X[:40], y[:40])


class TestTransformerObservationHorizonChecks:
    """Tests for transformer observation horizon check functions."""

    def test_not_fitted(self, y_X_factory):
        """Test check_observation_horizon_not_fitted passes for unfitted transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)

        # Should not raise - unfitted transformer correctly raises NotFittedError
        check_observation_horizon_not_fitted(transformer, X)

    def test_after_fit(self, y_X_factory):
        """Test check_observation_horizon_after_fit passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:40], y[:40])

        # Should not raise
        check_observation_horizon_after_fit(transformer, X[:40], y[:40])

    def test_insufficient_data_raises(self, y_X_factory):
        """Test check_insufficient_data_raises passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=10)

        # Pass data long enough to fit (lag=10 needs >10 rows)
        check_insufficient_data_raises(transformer, X[:30], y[:30])


class TestTransformerMemoryChecks:
    """Tests for transformer memory and state check functions."""

    def test_rewind_updates_memory(self, y_X_factory):
        """Test check_rewind_updates_memory passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:30], y[:30])

        # Should not raise
        check_rewind_updates_memory(transformer, X[20:40], y[20:40])

    def test_observe_concatenates_memory(self, y_X_factory):
        """Test check_observe_concatenates_memory passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:30], y[:30])

        # Should not raise
        check_observe_concatenates_memory(transformer, X[30:40], y[30:40])

    def test_observe_transform_equivalence(self, y_X_factory):
        """Test check_observe_transform_equivalence passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:30], y[:30])

        # Should not raise
        check_observe_transform_equivalence(transformer, X[30:40], y[30:40])

    def test_rewind_transform_behavior(self, y_X_factory):
        """Test check_rewind_transform_behavior passes for valid transformer."""
        y, X = y_X_factory(length=100, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=5)

        # Should not raise
        check_rewind_transform_behavior(transformer, X, y)

    def test_memory_bounded(self, y_X_factory):
        """Test check_memory_bounded passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42, panel=False)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:20], y[:20])

        # Should not raise - check validates memory stays bounded
        check_memory_bounded(transformer, X[:20], X[20:40], y[:20])

    def test_unfitted_stateless(self, y_X_factory):
        """Test check_transformers_unfitted_stateless passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)

        # Should not raise
        check_transformers_unfitted_stateless(transformer, X)


class TestTransformerOutputChecks:
    """Tests for transformer output check functions."""

    def test_transform_output_structure(self, y_X_factory):
        """Test check_transform_output_structure passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:40], y[:40])

        # Should not raise
        check_transform_output_structure(transformer, X[:40], y[:40])

    def test_feature_names_out_match(self, y_X_factory):
        """Test check_feature_names_out_match passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:40], y[:40])

        # Should not raise
        check_feature_names_out_match(transformer, X[:40], y[:40])

    def test_preserve_dtypes(self, y_X_factory):
        """Test check_transformer_preserve_dtypes passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:40], y[:40])

        # Should not raise
        check_transformer_preserve_dtypes(transformer, X[:40], y[:40])

    def test_panel_data_support(self, y_X_factory):
        """Test check_panel_data_support passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42, panel=True)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:40], y[:40])

        # Should not raise
        check_panel_data_support(transformer, X[:40], y[:40])


class TestTransformerInverseChecks:
    """Tests for transformer inverse transform check functions."""

    def test_inverse_transform_identity(self, y_X_factory):
        """Test check_inverse_transform_identity passes for invertible transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LogTransformer()
        transformer.fit(X[:40], y[:40])

        # Should not raise
        check_inverse_transform_identity(transformer, X[:40], y[:40])

    def test_inverse_transform_identity_skips_non_invertible(self, y_X_factory):
        """Test check skips transformers without inverse_transform."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:40], y[:40])

        # Should not raise - check skips non-invertible transformers
        check_inverse_transform_identity(transformer, X[:40], y[:40])

    def test_inverse_transform_round_trip(self, y_X_factory):
        """Test check_inverse_transform_round_trip passes for invertible transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LogTransformer()
        transformer.fit(X[:40], y[:40])

        # Should not raise
        check_inverse_transform_round_trip(transformer, X[:40], y[:40])


class TestTransformerTagChecks:
    """Tests for transformer tag check functions."""

    def test_tags_accessible_before_fit(self, y_X_factory):
        """Test check_tags_accessible_before_fit passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)

        # Should not raise
        check_tags_accessible_before_fit(transformer, X)

    def test_tags_static_after_fit(self, y_X_factory):
        """Test check_tags_static_after_fit passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:40], y[:40])

        # Should not raise
        check_tags_static_after_fit(transformer, X[:40], y[:40])

    def test_tags_match_capabilities(self, y_X_factory):
        """Test check_tags_match_capabilities passes for valid transformer."""
        y, X = y_X_factory(length=50, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X[:40], y[:40])

        # Should not raise
        check_tags_match_capabilities(transformer, X[:40])


class TestTransformerPanelGroupPreservation:
    """Tests for check_panel_group_preservation with panel data."""

    def test_panel_groups_preserved(self):
        """Transformer preserves panel group prefixes after transform."""
        n = 30
        times = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=n - 1),
            interval="1d",
            eager=True,
        )
        X_panel = pl.DataFrame({
            "time": times,
            "grp1__val_a": [float(i) for i in range(n)],
            "grp1__val_b": [float(i) * 2 for i in range(n)],
            "grp2__val_a": [float(i) + 1 for i in range(n)],
            "grp2__val_b": [float(i) * 3 for i in range(n)],
        })
        transformer = LogTransformer()
        check_panel_group_preservation(transformer, X_panel)

    def test_non_panel_data_skipped(self, y_X_factory):
        """Non-panel data is skipped gracefully."""
        y, X = y_X_factory(length=30, n_targets=1, n_features=2, seed=42)
        transformer = LogTransformer()
        check_panel_group_preservation(transformer, X)


class TestTransformerStatelessWithoutFit:
    """Tests for check_transformers_unfitted_stateless with stateless transformer."""

    def test_stateless_transform_without_fit(self, y_X_factory):
        """Stateless transformer works without explicit fit."""
        _, X = y_X_factory(length=20, n_targets=1, n_features=2, seed=42)
        transformer = FunctionTransformer(func=lambda df: df * 2)
        check_transformers_unfitted_stateless(transformer, X)

    def test_stateful_transformer_skipped(self, y_X_factory):
        """Stateful transformer is skipped (horizon != 0)."""
        _, X = y_X_factory(length=20, n_targets=1, n_features=2, seed=42)
        transformer = LagTransformer(lag=3)
        transformer.fit(X)
        check_transformers_unfitted_stateless(transformer, X)


class TestTransformerInverseRoundTrip:
    """Tests for check_inverse_transform_round_trip with invertible transformers."""

    def test_log_transformer_round_trip(self, y_X_factory):
        """LogTransformer round trip via check function succeeds."""
        _, X = y_X_factory(length=30, n_targets=1, n_features=2, seed=42)
        transformer = LogTransformer()
        check_inverse_transform_round_trip(transformer, X)


class TestStatefulInvertibleSystematicChecks:
    """Systematic checks for stateful + invertible transformer via run_checks."""

    @pytest.fixture
    def train_test_data(self, time_series_factory):
        """Create train/test split for seasonal differencing."""
        X = time_series_factory(length=60, n_components=2, seed=42)
        return X[:40], X[40:]

    def test_seasonal_differencing_all_checks(self, train_test_data):
        """Run all yields from _yield_yohou_transformer_checks on SeasonalDifferencing."""
        X_train, X_test = train_test_data
        transformer = SeasonalDifferencing(seasonality=3)
        transformer.fit(X_train)

        run_checks(
            transformer,
            _yield_yohou_transformer_checks(transformer, X_train, None, X_test),
        )

    def test_inverse_observe_transform_identity(self, train_test_data):
        """Inverse of observe_transform recovers original data."""
        X_train, _ = train_test_data
        transformer = SeasonalDifferencing(seasonality=3)
        check_inverse_observe_transform_identity(transformer, X_train)

    def test_observe_transform_sequential_consistency(self, train_test_data):
        """Sequential observe_transforms produce consistent results."""
        X_train, _ = train_test_data
        transformer = SeasonalDifferencing(seasonality=3)
        transformer.fit(X_train[:20])
        check_observe_transform_sequential_consistency(transformer, X_train)


class TestStatelessInvertibleSystematicChecks:
    """Systematic checks for stateless + invertible transformer via run_checks."""

    @pytest.fixture
    def train_test_data(self, time_series_factory):
        """Create train/test split for LogTransformer."""
        X = time_series_factory(length=60, n_components=2, seed=42)
        return X[:40], X[40:]

    def test_log_transformer_all_checks(self, train_test_data):
        """Run all yields from _yield_yohou_transformer_checks on LogTransformer."""
        X_train, X_test = train_test_data
        transformer = LogTransformer()
        transformer.fit(X_train)

        run_checks(
            transformer,
            _yield_yohou_transformer_checks(transformer, X_train, None, X_test),
        )


class TestStatefulTransformerSystematicChecks:
    """Systematic checks for stateful (non-invertible) Lag transformer."""

    @pytest.fixture
    def train_test_data(self, time_series_factory):
        """Create train/test split for LagTransformer."""
        X = time_series_factory(length=60, n_components=2, seed=42)
        return X[:40], X[40:]

    def test_lag_transformer_all_checks(self, train_test_data):
        """Run all yields from _yield_yohou_transformer_checks on LagTransformer."""
        X_train, X_test = train_test_data
        transformer = LagTransformer(lag=3)
        transformer.fit(X_train)

        run_checks(
            transformer,
            _yield_yohou_transformer_checks(transformer, X_train, None, X_test),
        )


class TestPanelTransformerSystematicChecks:
    """Systematic checks with panel data columns."""

    @pytest.fixture
    def panel_train_test_data(self):
        """Create panel data train/test split."""
        n = 60
        times = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=n - 1),
            interval="1d",
            eager=True,
        )
        X = pl.DataFrame({
            "time": times,
            "grp1__val_a": [float(i) + 1 for i in range(n)],
            "grp1__val_b": [float(i) * 2 + 1 for i in range(n)],
            "grp2__val_a": [float(i) + 3 for i in range(n)],
            "grp2__val_b": [float(i) * 3 + 1 for i in range(n)],
        })
        return X[:40], X[40:]

    def test_log_transformer_panel_all_checks(self, panel_train_test_data):
        """Run all checks on LogTransformer with panel data."""
        X_train, X_test = panel_train_test_data
        transformer = LogTransformer()
        transformer.fit(X_train)

        run_checks(
            transformer,
            _yield_yohou_transformer_checks(transformer, X_train, None, X_test),
        )

    def test_lag_transformer_panel_all_checks(self, panel_train_test_data):
        """Run all checks on LagTransformer with panel data."""
        X_train, X_test = panel_train_test_data
        transformer = LagTransformer(lag=3)
        transformer.fit(X_train)

        run_checks(
            transformer,
            _yield_yohou_transformer_checks(transformer, X_train, None, X_test),
        )


class TestTransformerCheckEarlyReturns:
    """Tests for early return branches in check functions.

    These test that check functions gracefully handle edge cases like
    short data, non-invertible transformers, and non-panel data.
    """

    @pytest.fixture
    def short_data(self):
        """Create a DataFrame with only 5 rows (too short for many checks)."""
        times = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 5),
            interval="1d",
            eager=True,
        )
        return pl.DataFrame({"time": times, "val": [1.0, 2.0, 3.0, 4.0, 5.0]})

    @pytest.fixture
    def medium_data(self):
        """Create a DataFrame with 20 rows for transformer checks."""
        n = 20
        times = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=n - 1),
            interval="1d",
            eager=True,
        )
        return pl.DataFrame({
            "time": times,
            "val": [float(i) + 1.0 for i in range(n)],
        })

    @pytest.fixture
    def non_panel_data(self):
        """Create simple non-panel DataFrame."""
        n = 30
        times = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=n - 1),
            interval="1d",
            eager=True,
        )
        return pl.DataFrame({
            "time": times,
            "val": [float(i) + 1.0 for i in range(n)],
        })

    def test_observe_transform_sequential_short_data(self, short_data):
        """check_observe_transform_sequential_consistency returns early for short data."""
        t = LagTransformer(lag=2)
        check_observe_transform_sequential_consistency(t, short_data)

    def test_rewind_transform_short_data(self, short_data):
        """check_rewind_transform_behavior returns early for short data."""
        t = LagTransformer(lag=3)
        check_rewind_transform_behavior(t, short_data)

    def test_insufficient_data_stateless(self, medium_data):
        """check_insufficient_data_raises returns early for stateless transformers."""
        t = FunctionTransformer(func=lambda X: X)
        check_insufficient_data_raises(t, medium_data)

    def test_inverse_observe_transform_non_invertible(self, medium_data):
        """check_inverse_observe_transform_identity returns for non-invertible."""
        t = LagTransformer(lag=2)
        t.fit(medium_data)
        check_inverse_observe_transform_identity(t, medium_data)

    def test_inverse_observe_transform_short_data(self):
        """check_inverse_observe_transform_identity returns for short data."""
        times = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 8),
            interval="1d",
            eager=True,
        )
        X = pl.DataFrame({"time": times, "val": [float(i) + 1 for i in range(8)]})
        t = LogTransformer()
        t.fit(X)
        check_inverse_observe_transform_identity(t, X.head(5))

    def test_inverse_round_trip_non_invertible(self, medium_data):
        """check_inverse_transform_round_trip returns for non-invertible."""
        t = LagTransformer(lag=2)
        t.fit(medium_data)
        check_inverse_transform_round_trip(t, medium_data)

    def test_panel_data_support_non_panel(self, non_panel_data):
        """check_panel_data_support returns early for non-panel data."""
        t = LagTransformer(lag=2)
        check_panel_data_support(t, non_panel_data)

    def test_unfitted_stateless_with_stateful_transformer(self, medium_data):
        """check_transformers_unfitted_stateless returns early for stateful transformer."""
        t = LagTransformer(lag=2)
        check_transformers_unfitted_stateless(t, medium_data)

    def test_unfitted_stateless_with_stateless_transformer(self, medium_data):
        """check_transformers_unfitted_stateless succeeds for stateless transformer."""
        t = FunctionTransformer(func=lambda X: X)
        check_transformers_unfitted_stateless(t, medium_data)

    def test_memory_bounded_short_test_data(self):
        """check_memory_bounded hits break for short test data."""
        n = 30
        times = pl.datetime_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 1) + timedelta(days=n - 1),
            interval="1d",
            eager=True,
        )
        X = pl.DataFrame({"time": times, "val": [float(i) + 1 for i in range(n)]})
        X_train = X[:20]
        X_test = X[20:22]
        t = LagTransformer(lag=2)
        check_memory_bounded(t, X_train, X_test, n_updates=10)
