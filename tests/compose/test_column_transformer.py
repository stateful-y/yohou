"""Tests for ColumnTransformer composition class."""

import sys
from pathlib import Path

import polars as pl
import pytest
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).parent))

from conftest import SimpleTransformer, StatelessTransformer, run_checks
from yohou.compose import ColumnTransformer
from yohou.preprocessing import Downsampler
from yohou.preprocessing.window import LagTransformer
from yohou.testing import _yield_yohou_transformer_checks


def _make_3col(length: int = 20) -> pl.DataFrame:
    """Build a 3-column (a, b, c) time series of the requested length."""
    from datetime import datetime, timedelta

    time = pl.datetime_range(
        start=datetime(2021, 1, 1),
        end=datetime(2021, 1, 1) + timedelta(seconds=length - 1),
        interval="1s",
        eager=True,
    )
    return pl.DataFrame({
        "time": time,
        "a": [float(x) for x in range(length)],
        "b": [float(x) * 10 for x in range(length)],
        "c": [float(x) * 100 for x in range(length)],
    })


@pytest.fixture
def time_series_3col():
    """Create a 3-column time series for column transformer tests."""
    return _make_3col(20)


class TestColumnTransformerBasic:
    """Basic ColumnTransformer functionality."""

    def test_init(self):
        """Test ColumnTransformer initialization."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=1), ["a"]),
            ]
        )
        assert ct.transformers is not None

    def test_fit_transform_single_transformer(self, time_series_3col):
        """Test fit and transform with a single transformer."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0, add_constant=1.0), ["a", "b"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        result = ct.transform(time_series_3col)
        assert isinstance(result, pl.DataFrame)
        assert "time" in result.columns

    def test_remainder_drop(self, time_series_3col):
        """Test that remainder='drop' excludes unspecified columns."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0), ["a"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        result = ct.transform(time_series_3col)
        # "b" and "c" should be dropped
        assert "c" not in result.columns

    def test_remainder_passthrough(self, time_series_3col):
        """Test that remainder='passthrough' preserves unspecified columns."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0), ["a"]),
            ],
            remainder="passthrough",
        )
        ct.fit(time_series_3col)
        result = ct.transform(time_series_3col)
        # "b" and "c" should be passed through
        assert len(result.columns) >= 3  # time + a + b + c at minimum

    def test_multiple_transformers(self, time_series_3col):
        """Test with multiple transformers on different columns."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0, add_constant=1.0), ["a"]),
                ("t2", SimpleTransformer(observation_horizon=0, add_constant=2.0), ["b"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        result = ct.transform(time_series_3col)
        assert isinstance(result, pl.DataFrame)

    def test_clone(self, time_series_3col):
        """Test that ColumnTransformer can be cloned."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0), ["a"]),
            ],
        )
        ct_clone = clone(ct)
        assert ct_clone is not ct


class TestColumnTransformerObservationHorizon:
    """Test observation_horizon property."""

    def test_observation_horizon_max(self, time_series_3col):
        """Test that observation_horizon is max of component horizons."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=2), ["a"]),
                ("t2", SimpleTransformer(observation_horizon=5), ["b"]),
            ],
        )
        ct.fit(time_series_3col)
        # ColumnTransformer takes max of component observation horizons
        assert ct.observation_horizon >= 2

    def test_observation_horizon_stateless(self, time_series_3col):
        """Test observation_horizon with stateless transformers."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0), ["a"]),
                ("t2", SimpleTransformer(observation_horizon=0), ["b"]),
            ],
        )
        ct.fit(time_series_3col)
        assert ct.observation_horizon == 0


class TestColumnTransformerUpdateReset:
    """Test observe_transform and rewind_transform methods."""

    def test_observe_transform(self, time_series_3col):
        """Test observe_transform method."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", "passthrough", ["a", "b"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        # observe_transform should work on new data
        result = ct.observe_transform(time_series_3col)
        assert isinstance(result, pl.DataFrame)

    def test_rewind_transform(self, time_series_3col):
        """Test rewind_transform method."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", StatelessTransformer(), ["a", "b"]),
            ],
            remainder="passthrough",
        )
        ct.fit(time_series_3col)
        result = ct.rewind_transform(time_series_3col)
        assert isinstance(result, pl.DataFrame)

    def test_fit_transform_method(self, time_series_3col):
        """Test fit_transform method."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0), ["a", "b"]),
            ],
            remainder="drop",
        )
        result = ct.fit_transform(time_series_3col)
        assert isinstance(result, pl.DataFrame)


class TestColumnTransformerFeatureNames:
    """Test get_feature_names_out."""

    def test_feature_names_out(self, time_series_3col):
        """Test that get_feature_names_out returns feature names."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0), ["a", "b"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        feature_names = ct.get_feature_names_out()
        # Default verbose_feature_names_out=True prefixes each column with the
        # transformer name, so the two transformed columns become t1_a, t1_b.
        assert isinstance(feature_names, list)
        assert feature_names == ["t1_a", "t1_b"]

    def test_verbose_feature_names(self, time_series_3col):
        """Test verbose_feature_names_out=True prefixes names."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", StatelessTransformer(), ["a"]),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )
        ct.fit(time_series_3col)
        feature_names = ct.get_feature_names_out()
        assert feature_names is not None
        # With verbose, names should be prefixed with transformer name using _
        assert any("t1_" in str(n) for n in feature_names)

    def test_feature_names_out_panel_matches_transformed_columns(self):
        """get_feature_names_out uses group__column naming matching the output.

        Regression for the 2026-06-21 QA finding: panel feature names were
        built with a naive ``{name}_{col}`` join, producing
        ``t1_store_1__sales`` instead of the panel-aware
        ``store_1__t1_sales`` that the transformed DataFrame actually emits.
        """
        from datetime import datetime

        time = pl.datetime_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 5),
            interval="1d",
            eager=True,
        )
        panel = pl.DataFrame({
            "time": time,
            "store_1__sales": [1.0, 2.0, 3.0, 4.0, 5.0],
            "store_2__sales": [6.0, 7.0, 8.0, 9.0, 10.0],
        })
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0), ["store_1__sales", "store_2__sales"]),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )
        ct.fit(panel)
        transformed = ct.transform(panel)
        emitted_cols = [c for c in transformed.columns if c != "time"]
        feature_names = ct.get_feature_names_out()
        assert feature_names == emitted_cols
        assert feature_names == ["store_1__t1_sales", "store_2__t1_sales"]


class TestColumnTransformerPassthrough:
    """Test passthrough and drop special cases."""

    def test_passthrough_transformer(self, time_series_3col):
        """Test using 'passthrough' as transformer."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", "passthrough", ["a", "b"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        result = ct.transform(time_series_3col)
        assert isinstance(result, pl.DataFrame)

    def test_drop_transformer(self, time_series_3col):
        """Test that 'drop' is not accepted as transformer string."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", "passthrough", ["b"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        result = ct.transform(time_series_3col)
        assert isinstance(result, pl.DataFrame)


class TestColumnTransformerObserveRewindTransform:
    """Tests for ColumnTransformer observe_transform and rewind_transform."""

    def test_observe_transform(self, time_series_3col):
        """observe_transform produces results with time column."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0), ["a", "b"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        result = ct.observe_transform(time_series_3col)
        assert isinstance(result, pl.DataFrame)
        assert "time" in result.columns

    def test_rewind_transform(self, time_series_3col):
        """rewind_transform produces results with time column."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0), ["a", "b"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        result = ct.rewind_transform(time_series_3col)
        assert isinstance(result, pl.DataFrame)
        assert "time" in result.columns

    def test_observe_transform_updates_stateful_component_buffer(self, time_series_3col):
        """observe_transform propagates the stateful update into the component.

        With a real stateful component (observation_horizon=2), the buffered
        state of the fitted child transformer must advance to the newly observed
        data, not stay pinned at the fit-time tail. This exercises the
        _observe_transform_one stateful path rather than the passthrough one.
        """
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=2), ["a", "b"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col[:10])

        component = ct["t1"]
        fit_observed_time = component.observed_time_
        assert component._X_observed.height == 2

        new_batch = time_series_3col[10:]
        ct.observe_transform(new_batch)

        # The component buffer advanced to the latest observed timestamp and
        # still holds exactly observation_horizon rows.
        assert component.observed_time_ == new_batch["time"][-1]
        assert component.observed_time_ > fit_observed_time
        assert component._X_observed.height == 2

    def test_getitem_by_name(self, time_series_3col):
        """Accessing transformer by name via __getitem__."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0), ["a"]),
                ("t2", SimpleTransformer(observation_horizon=0), ["b"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        assert isinstance(ct["t1"], SimpleTransformer)
        assert isinstance(ct["t2"], SimpleTransformer)

    def test_getitem_by_index(self, time_series_3col):
        """Accessing transformer by integer index."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0), ["a"]),
                ("t2", SimpleTransformer(observation_horizon=0), ["b"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        assert isinstance(ct[0], SimpleTransformer)

    def test_getitem_by_slice(self, time_series_3col):
        """Accessing transformers by slice."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0), ["a"]),
                ("t2", SimpleTransformer(observation_horizon=0), ["b"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        sliced = ct[0:2]
        assert len(sliced.transformers) == 2

    def test_remainder_as_estimator(self, time_series_3col):
        """Remainder set to an estimator transforms unspecified columns."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0), ["a"]),
            ],
            remainder=SimpleTransformer(observation_horizon=0, add_constant=5.0),
        )
        ct.fit(time_series_3col)
        result = ct.transform(time_series_3col)
        assert isinstance(result, pl.DataFrame)

    def test_get_metadata_routing(self, time_series_3col):
        """get_metadata_routing returns a MetadataRouter."""
        from sklearn.utils.metadata_routing import MetadataRouter

        ct = ColumnTransformer(
            transformers=[
                ("t1", SimpleTransformer(observation_horizon=0), ["a"]),
            ],
            remainder="drop",
        )
        router = ct.get_metadata_routing()
        assert isinstance(router, MetadataRouter)


class TestColumnTransformerEdgeCases:
    """Tests for ColumnTransformer edge cases and less-visited branches."""

    def test_getitem_int_on_unfitted(self):
        """Integer indexing on unfitted CT returns transformer from list."""
        t1 = StatelessTransformer()
        ct = ColumnTransformer(
            transformers=[("t1", t1, ["a"])],
        )
        assert ct[0] is t1

    def test_getitem_slice_with_step_raises(self, time_series_3col):
        """Slicing with a step raises ValueError."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", StatelessTransformer(), ["a"]),
                ("t2", StatelessTransformer(), ["b"]),
            ],
        )
        with pytest.raises(ValueError, match="step"):
            ct[::2]

    def test_getitem_int_on_fitted(self, time_series_3col):
        """Integer indexing on a fitted CT returns fitted transformer."""
        ct = ColumnTransformer(
            transformers=[("t1", StatelessTransformer(), ["a"])],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        result = ct[0]
        assert hasattr(result, "X_schema_")

    def test_getitem_str_on_fitted(self, time_series_3col):
        """String indexing on a fitted CT returns fitted transformer by name."""
        ct = ColumnTransformer(
            transformers=[("t1", StatelessTransformer(), ["a"])],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        result = ct["t1"]
        assert hasattr(result, "X_schema_")

    def test_getitem_str_not_found_raises(self):
        """String indexing with unknown name raises KeyError."""
        ct = ColumnTransformer(
            transformers=[("t1", StatelessTransformer(), ["a"])],
        )
        with pytest.raises(KeyError, match="nonexistent"):
            ct["nonexistent"]

    def test_fit_transform_drops_remainder(self, time_series_3col):
        """fit_transform with remainder='drop' excludes unspecified columns."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", StatelessTransformer(), ["a"]),
            ],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        result = ct.transform(time_series_3col)
        assert "time" in result.columns
        non_time = [c for c in result.columns if c != "time"]
        assert len(non_time) == 1
        assert "b" not in result.columns
        assert "c" not in result.columns

    def test_verbose_feature_names_out_false_no_duplicates(self, time_series_3col):
        """verbose_feature_names_out=False works when no duplicate names."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", StatelessTransformer(), ["a"]),
                ("t2", StatelessTransformer(), ["b"]),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        result = ct.fit_transform(time_series_3col)
        assert "a" in result.columns
        assert "b" in result.columns

    def test_verbose_feature_names_out_false_duplicates_raises(self, time_series_3col):
        """verbose_feature_names_out=False with duplicate names raises ValueError."""
        ct = ColumnTransformer(
            transformers=[
                ("t1", "passthrough", ["a"]),
                ("t2", "passthrough", ["a"]),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        with pytest.raises(ValueError, match="Duplicated feature names"):
            ct.fit_transform(time_series_3col)

    def test_transformers_property(self, time_series_3col):
        """_transformers property returns fitted transformer list."""
        ct = ColumnTransformer(
            transformers=[("t1", StatelessTransformer(), ["a"])],
            remainder="drop",
        )
        ct.fit(time_series_3col)
        result = ct._transformers
        assert isinstance(result, list)
        assert len(result) >= 1


class TestColumnTransformerCoveragePaths:
    """Tests targeting uncovered branches in ColumnTransformer."""

    def test_set_params_empty_transformers(self):
        """set_params on empty transformers list returns self unchanged."""
        ct = ColumnTransformer(transformers=[], remainder="drop")
        result = ct.set_params()
        assert result is ct

    def test_getitem_slice(self):
        """Slicing without step returns a new ColumnTransformer."""
        t1 = StatelessTransformer()
        t2 = StatelessTransformer()
        ct = ColumnTransformer(
            transformers=[("t1", t1, ["a"]), ("t2", t2, ["b"])],
        )
        sliced = ct[0:1]
        assert isinstance(sliced, ColumnTransformer)
        assert len(sliced.transformers) == 1

    def test_validate_transformers_non_base_raises(self):
        """Non-BaseActualTransformer estimator in transformers raises TypeError."""
        from sklearn.preprocessing import StandardScaler

        ct = ColumnTransformer(
            transformers=[("bad", StandardScaler(), ["a"])],
        )
        with pytest.raises(TypeError, match="BaseActualTransformer"):
            ct._validate_transformers()

    def test_remainder_passthrough(self, time_series_3col):
        """remainder='passthrough' includes unspecified columns."""
        ct = ColumnTransformer(
            transformers=[("t1", StatelessTransformer(), ["a"])],
            remainder="passthrough",
        )
        result = ct.fit_transform(time_series_3col)
        non_time = [c for c in result.columns if c != "time"]
        assert len(non_time) == 3

    def test_fit_transform_all_empty_returns_time_column(self, time_series_3col):
        """When no transformers are specified, fit_transform returns time-only."""
        ct = ColumnTransformer(transformers=[], remainder="drop")
        result = ct.fit_transform(time_series_3col)
        assert result.columns == ["time"]
        assert result.shape[0] == time_series_3col.shape[0]

    def test_observe_transform_all_empty_returns_time_column(self, time_series_3col):
        """When no transformers are specified, observe_transform returns time-only."""
        ct = ColumnTransformer(transformers=[], remainder="drop")
        ct.fit(time_series_3col)
        result = ct.observe_transform(time_series_3col)
        assert result.columns == ["time"]

    def test_rewind_transform_all_empty_returns_time_column(self, time_series_3col):
        """When no transformers are specified, rewind_transform returns time-only."""
        ct = ColumnTransformer(transformers=[], remainder="drop")
        ct.fit(time_series_3col)
        result = ct.rewind_transform(time_series_3col)
        assert result.columns == ["time"]


class TestColumnTransformerTagsAggregation:
    """Tests for __sklearn_tags__ aggregation with various transformer combos."""

    def test_stateful_tag_propagated(self):
        """Stateful tag is True when any transformer is stateful."""
        ct = ColumnTransformer(
            transformers=[
                ("stateful", LagTransformer(lag=[1]), ["a"]),
                ("stateless", StatelessTransformer(), ["b"]),
            ],
        )
        tags = ct.__sklearn_tags__()
        assert tags.transformer_tags.stateful is True

    def test_all_stateless_tag(self):
        """Stateful tag is False when all transformers are stateless."""
        ct = ColumnTransformer(
            transformers=[
                ("s1", StatelessTransformer(), ["a"]),
                ("s2", StatelessTransformer(), ["b"]),
            ],
        )
        tags = ct.__sklearn_tags__()
        assert tags.transformer_tags.stateful is False

    def test_invertible_always_false(self):
        """ColumnTransformer is never invertible."""
        ct = ColumnTransformer(
            transformers=[
                ("s1", StatelessTransformer(), ["a"]),
            ],
        )
        tags = ct.__sklearn_tags__()
        assert tags.transformer_tags.invertible is False

    def test_accepts_irregular_grid_all_tolerant(self):
        """accepts_irregular_grid is True when every child tolerates an irregular grid."""
        ct = ColumnTransformer(
            transformers=[
                ("d1", Downsampler(interval="1h", aggregation="mean"), ["a"]),
                ("d2", Downsampler(interval="1h", aggregation="mean"), ["b"]),
            ],
        )
        assert ct.__sklearn_tags__().transformer_tags.accepts_irregular_grid is True

    def test_accepts_irregular_grid_one_strict_child(self):
        """One interval-dependent child makes the whole composition strict."""
        ct = ColumnTransformer(
            transformers=[
                ("tolerant", Downsampler(interval="1h", aggregation="mean"), ["a"]),
                ("strict", LagTransformer(lag=[1]), ["b"]),
            ],
        )
        assert ct.__sklearn_tags__().transformer_tags.accepts_irregular_grid is False

    def test_accepts_irregular_grid_empty_keeps_base_default(self):
        """A passthrough/drop-only composition keeps the base default, not all([]) == True."""
        ct = ColumnTransformer(
            transformers=[("keep", "passthrough", ["a"])],
            remainder="drop",
        )
        assert ct.__sklearn_tags__().transformer_tags.accepts_irregular_grid is False

    def test_remainder_estimator_included_in_tags(self):
        """Remainder estimator contributes to tag aggregation."""
        ct = ColumnTransformer(
            transformers=[("s1", StatelessTransformer(), ["a"])],
            remainder=LagTransformer(lag=[1, 2]),
        )
        tags = ct.__sklearn_tags__()
        assert tags.transformer_tags.stateful is True

    def test_min_value_aggregated_from_transformers(self):
        """min_value is the maximum (most restrictive) across transformers."""
        ct = ColumnTransformer(
            transformers=[
                ("s1", StatelessTransformer(), ["a"]),
                ("s2", StatelessTransformer(), ["b"]),
            ],
        )
        tags = ct.__sklearn_tags__()
        assert tags.input_tags.min_value is None or isinstance(tags.input_tags.min_value, int | float)


class TestColumnTransformerNestedParams:
    """Nested ``<name>__<param>`` get/set over the 3-tuple ``transformers`` form."""

    def test_get_params_exposes_nested(self):
        ct = ColumnTransformer(transformers=[("s", SimpleTransformer(add_constant=1.0), ["a"])])
        params = ct.get_params(deep=True)
        assert "s__add_constant" in params
        assert params["s__add_constant"] == 1.0

    def test_set_params_routes_nested(self):
        ct = ColumnTransformer(transformers=[("s", SimpleTransformer(add_constant=1.0), ["a"])])
        ct.set_params(s__add_constant=5.0)
        assert {n: t for n, t, _ in ct.transformers}["s"].add_constant == 5.0

    def test_replace_component_preserves_columns(self):
        ct = ColumnTransformer(transformers=[("s", SimpleTransformer(), ["a"])])
        # Replacing a whole component routes through the _transformers setter,
        # which rebuilds the 3-tuple and preserves the original columns.
        ct.set_params(s=StatelessTransformer())
        assert len(ct.transformers[0]) == 3
        assert ct.transformers[0][2] == ["a"]
        assert isinstance(ct.transformers[0][1], StatelessTransformer)


class TestColumnTransformerSystematicChecks:
    """Run the generic transformer contract over representative configurations.

    ColumnTransformer requires constructor arguments, so it is excluded from
    the auto-discovered ``TestTransformerCommon`` suite in ``tests/test_common.py``
    (it sits in ``_SKIP_COMMON``). This dedicated parametrization wires it into
    ``_yield_yohou_transformer_checks`` so the data-driven transformer contract
    actually runs against it.
    """

    @staticmethod
    def _data(length: int = 100):
        return _make_3col(length)

    @pytest.mark.parametrize(
        "name,factory,extra_xfail",
        [
            (
                "stateless-only",
                lambda: ColumnTransformer(
                    transformers=[("t1", StatelessTransformer(), ["a", "b"])],
                    remainder="drop",
                ),
                set(),
            ),
            (
                "stateful",
                lambda: ColumnTransformer(
                    transformers=[("t1", SimpleTransformer(observation_horizon=2, add_constant=1.0), ["a"])],
                    remainder="drop",
                ),
                # The composition reassembles per-column outputs and does not
                # drop warmup rows from the combined frame; only relevant when a
                # component is stateful.
                {"check_transform_drops_warmup_rows"},
            ),
            (
                "passthrough-remainder",
                lambda: ColumnTransformer(
                    transformers=[("t1", SimpleTransformer(observation_horizon=0, add_constant=1.0), ["a"])],
                    remainder="passthrough",
                ),
                set(),
            ),
        ],
    )
    def test_column_transformer_checks(self, name, factory, extra_xfail):
        """Run the systematic transformer checks for a ColumnTransformer config."""
        data = self._data()
        X_train = data[:80]
        X_test = data[80:]

        transformer = clone(factory())
        transformer.fit(X_train)

        # ColumnTransformer computes observation_horizon dynamically from its
        # components (a property) rather than storing _observation_horizon at
        # fit, so check_fit_sets_attributes does not apply by design.
        expected_failures = {"check_fit_sets_attributes"} | extra_xfail

        run_checks(
            transformer,
            _yield_yohou_transformer_checks(transformer, X_train, None, X_test),
            expected_failures=expected_failures,
        )


class _TimeOnlyTransformer(SimpleTransformer):
    """Transformer whose output is the time column only (no feature columns)."""

    def transform(self, X):
        return X.select(pl.col("time"))

    def get_feature_names_out(self, input_features=None):
        return []


class TestHstackPrefixPairing:
    """Regression: prefixes must stay paired with the right columns in _hstack."""

    def test_time_only_transformer_does_not_shift_prefixes(self, time_series_3col):
        """A time-only transformer output must not mis-pair feature prefixes.

        When an earlier transformer emits only the time column (shape[1] == 1)
        and verbose_feature_names_out=True, the surviving feature columns must
        still be prefixed with their own transformer's name, not a shifted one.
        """
        ct = ColumnTransformer(
            transformers=[
                ("empty", _TimeOnlyTransformer(observation_horizon=0), ["a"]),
                ("real", SimpleTransformer(observation_horizon=0), ["b"]),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )
        result = ct.fit_transform(time_series_3col)

        # The real transformer's column must survive and carry its own "real"
        # prefix (single underscore), never the time-only "empty" transformer's.
        assert "real_b" in result.columns
        assert "empty_b" not in result.columns
        assert "time" in result.columns
