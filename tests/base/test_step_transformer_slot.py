"""Forecaster-level behaviour of the ``step_transformer`` slot.

Covers what the transformer unit tests cannot: that the slot is applied at every
derivation path, survives the alignment filter, works per group under panel data,
and interacts correctly with the fit-time diagnostics.
"""

import warnings
from datetime import datetime, timedelta

import polars as pl
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from yohou.compose import DecompositionPipeline, FeatureUnion
from yohou.point import PointReductionForecaster
from yohou.preprocessing import StepAggregator, StepColumnReducer

HORIZON = 4
N = 48


def _times(n: int) -> list[datetime]:
    return [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]


@pytest.fixture
def standard_data():
    """A univariate series with a known-future feature covering the horizon."""
    y = pl.DataFrame({"time": _times(N), "y": [float(i % 7) + i * 0.1 for i in range(N)]})
    X_future = pl.DataFrame({
        "time": _times(N + HORIZON),
        "temp": [float((i * 3) % 11) for i in range(N + HORIZON)],
    })
    return y, X_future


@pytest.fixture
def panel_data():
    """Two groups plus a global known-future column."""
    y = pl.DataFrame({
        "time": _times(N),
        "A__y": [float(i % 5) for i in range(N)],
        "B__y": [float(i % 3) for i in range(N)],
    })
    X_future = pl.DataFrame({
        "time": _times(N + HORIZON),
        "A__temp": [float((i * 3) % 11) for i in range(N + HORIZON)],
        "B__temp": [float((i * 7) % 13) for i in range(N + HORIZON)],
        "holiday": [float(i % 4 == 0) for i in range(N + HORIZON)],
    })
    return y, X_future


class TestSlotIsAppliedEverywhere:
    """The slot must reach every derivation path, not just fit."""

    def test_fit_reduces_the_block(self, standard_data):
        """Raw step columns are replaced by the reduced ones."""
        y, X_future = standard_data
        forecaster = PointReductionForecaster(estimator=Ridge(), step_transformer=StepAggregator())
        forecaster.fit(y, forecasting_horizon=HORIZON, X_future=X_future)
        assert sorted(forecaster._step_column_names_) == ["temp_step_mean"]

    def test_without_slot_the_block_survives(self, standard_data):
        """The unset slot leaves derivation exactly as it was."""
        y, X_future = standard_data
        forecaster = PointReductionForecaster(estimator=Ridge())
        forecaster.fit(y, forecasting_horizon=HORIZON, X_future=X_future)
        assert sorted(forecaster._step_column_names_) == [f"temp_step_{h}" for h in range(1, HORIZON + 1)]

    def test_same_columns_on_every_path(self, standard_data):
        """fit, observe, predict, and predict-with-override agree on column names.

        A derivation site that forgot to apply the slot would produce a design
        matrix disagreeing with what the estimator was fitted on, which is the
        failure this slot's single application point exists to prevent.
        """
        y, X_future = standard_data
        forecaster = PointReductionForecaster(estimator=Ridge(), step_transformer=StepAggregator())
        forecaster.fit(y[:40], forecasting_horizon=HORIZON, X_future=X_future)
        after_fit = set(forecaster._step_column_names_)

        forecaster.predict()
        forecaster.observe(y[40:44])
        assert set(forecaster._step_column_names_) == after_fit

        forecaster.predict(X_future=X_future)
        assert set(forecaster._step_column_names_) == after_fit

    def test_observe_predict_loop(self, standard_data):
        """The pre-computed step frame in the observe-predict loop is transformed too."""
        y, X_future = standard_data
        forecaster = PointReductionForecaster(estimator=Ridge(), step_transformer=StepAggregator())
        forecaster.fit(y[:36], forecasting_horizon=HORIZON, X_future=X_future)
        predictions = forecaster.observe_predict(y[36:], X_future=X_future, stride=4)
        assert len(predictions) > 0

    def test_post_transform_collision_refused(self, standard_data):
        """A reduced name colliding with an X_actual column is caught.

        The collision check runs on post-transform names; checking the raw block
        would miss this entirely, since ``temp_step_mean`` is never a derived name.
        """
        y, X_future = standard_data
        X_actual = pl.DataFrame({"time": _times(N), "temp_step_mean": [float(i) for i in range(N)]})
        forecaster = PointReductionForecaster(estimator=Ridge(), step_transformer=StepAggregator())
        with pytest.raises(ValueError, match="collide with existing columns"):
            forecaster.fit(y, X_actual, forecasting_horizon=HORIZON, X_future=X_future)


class TestStepFeatureAlignment:
    """Horizon-agnostic columns survive every alignment."""

    @pytest.mark.parametrize("alignment", ["all", "matched", "cumulative"])
    def test_aggregates_reach_every_direct_estimator(self, standard_data, alignment):
        """A summary has no step index to align against, so it is never filtered."""
        y, X_future = standard_data
        forecaster = PointReductionForecaster(
            estimator=Ridge(),
            reduction_strategy="direct",
            step_feature_alignment=alignment,
            step_transformer=StepAggregator(aggregations=("mean", "max")),
            nan_handling="drop",
        )
        forecaster.fit(y, forecasting_horizon=HORIZON, X_future=X_future)
        for step in range(1, HORIZON + 1):
            kept = forecaster._filter_step_features(
                pl.DataFrame({"temp_step_mean": [1.0], "temp_step_max": [2.0], "other": [3.0]}),
                step,
            )
            assert "temp_step_mean" in kept.columns
            assert "temp_step_max" in kept.columns

    @pytest.mark.parametrize("alignment", ["matched", "cumulative"])
    def test_step_indexed_columns_are_still_filtered(self, standard_data, alignment):
        """Filtering of genuinely step-indexed columns is unchanged."""
        y, X_future = standard_data
        forecaster = PointReductionForecaster(
            estimator=Ridge(),
            reduction_strategy="direct",
            step_feature_alignment=alignment,
            nan_handling="drop",
        )
        forecaster.fit(y, forecasting_horizon=HORIZON, X_future=X_future)
        tab = pl.DataFrame({f"temp_step_{h}": [float(h)] for h in range(1, HORIZON + 1)})
        kept = forecaster._filter_step_features(tab, 2)
        expected = ["temp_step_2"] if alignment == "matched" else ["temp_step_1", "temp_step_2"]
        assert sorted(kept.columns) == sorted(expected)

    def test_multi_output_gets_step_transformation(self, standard_data):
        """multi-output cannot use step_feature_alignment but does get the slot.

        This is capability the alignment parameter cannot offer, since it warns and
        no-ops outside the direct strategy.
        """
        y, X_future = standard_data
        forecaster = PointReductionForecaster(
            estimator=Ridge(),
            reduction_strategy="multi-output",
            step_transformer=StepAggregator(),
        )
        forecaster.fit(y, forecasting_horizon=HORIZON, X_future=X_future)
        assert sorted(forecaster._step_column_names_) == ["temp_step_mean"]


class TestAugmentationByComposition:
    """Keeping raw steps alongside summaries is a composition, not a flag."""

    def test_union_emits_both(self, standard_data):
        """A FeatureUnion of passthrough and aggregator yields both sets."""
        y, X_future = standard_data
        union = FeatureUnion([("raw", "passthrough"), ("agg", StepAggregator())])
        forecaster = PointReductionForecaster(estimator=Ridge(), step_transformer=union)
        forecaster.fit(y, forecasting_horizon=HORIZON, X_future=X_future)
        names = forecaster._step_column_names_
        assert any(n.endswith("_step_mean") for n in names), sorted(names)
        assert any(n.endswith("_step_1") for n in names), sorted(names)


class TestPanel:
    """Per-group fitting under panel_strategy='global'."""

    def test_slot_is_a_per_group_dict(self, panel_data):
        """One fitted clone per group, keyed by group name."""
        y, X_future = panel_data
        forecaster = PointReductionForecaster(estimator=Ridge(), step_transformer=StepAggregator())
        forecaster.fit(y, forecasting_horizon=HORIZON, X_future=X_future)
        assert isinstance(forecaster.step_transformer_, dict)
        assert sorted(forecaster.step_transformer_) == ["A", "B"]

    def test_global_columns_are_localized(self, panel_data):
        """A shared column returns per group, as designed.

        ``get_group_df`` folds global columns into every group's slice and the dict
        path re-prefixes every output, so ``holiday`` comes back as
        ``A__holiday_step_mean`` and ``B__holiday_step_mean``. This matches how the
        forecast_transformer slot already behaves.
        """
        y, X_future = panel_data
        forecaster = PointReductionForecaster(estimator=Ridge(), step_transformer=StepAggregator())
        forecaster.fit(y, forecasting_horizon=HORIZON, X_future=X_future)
        assert sorted(forecaster._step_column_names_) == [
            "A__holiday_step_mean",
            "A__temp_step_mean",
            "B__holiday_step_mean",
            "B__temp_step_mean",
        ]

    def test_global_columns_stay_global_without_a_slot(self, panel_data):
        """Localization is a consequence of the slot, not of panel fitting.

        This is why the global/local clash check in the panel path is still live.
        """
        y, X_future = panel_data
        forecaster = PointReductionForecaster(estimator=Ridge())
        forecaster.fit(y, forecasting_horizon=HORIZON, X_future=X_future)
        assert [n for n in forecaster._step_column_names_ if "__" not in n]

    def test_predict_for_a_subset_of_groups(self, panel_data):
        """Prediction for fewer groups than were fitted still works."""
        y, X_future = panel_data
        forecaster = PointReductionForecaster(estimator=Ridge(), step_transformer=StepAggregator())
        forecaster.fit(y, forecasting_horizon=HORIZON, X_future=X_future)
        predictions = forecaster.predict(groups=["A"])
        assert "A__y" in predictions.columns


class TestDecompositionPipeline:
    """Summary columns must be stripped before components re-derive their own."""

    @pytest.mark.parametrize("panel", [False, True], ids=["standard", "panel"])
    def test_no_duplicate_column_error(self, standard_data, panel_data, panel):
        """A summary column left in place would collide and fail the fit."""
        y, X_future = panel_data if panel else standard_data
        pipeline = DecompositionPipeline(
            forecasters=[
                ("a", PointReductionForecaster(estimator=Ridge(), nan_handling="drop")),
                ("b", PointReductionForecaster(estimator=Ridge(), nan_handling="drop")),
            ],
            step_transformer=StepAggregator(),
        )
        pipeline.fit(y, forecasting_horizon=HORIZON, X_future=X_future)
        assert len(pipeline.predict()) == HORIZON


class TestFitDiagnostics:
    """Interaction with the rank and coverage warnings."""

    @pytest.fixture
    def rank_deficient_data(self):
        """An X_future column whose step expansion is rank deficient.

        A constant column is the simplest such case: its H step columns span a
        rank of one.
        """
        y = pl.DataFrame({"time": _times(N), "y": [float(i % 7) + i * 0.1 for i in range(N)]})
        X_future = pl.DataFrame({"time": _times(N + HORIZON), "flat": [1.0] * (N + HORIZON)})
        return y, X_future

    def test_reducing_the_block_silences_the_rank_warning(self, rank_deficient_data):
        """Aggregating away the collinear copies resolves the condition."""
        y, X_future = rank_deficient_data
        forecaster = PointReductionForecaster(estimator=Ridge(), step_transformer=StepAggregator())
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            forecaster.fit(y, forecasting_horizon=HORIZON, X_future=X_future)
        assert not [w for w in record if "rank of only" in str(w.message)]

    def test_suppression_does_not_discriminate_between_transformers(self, rank_deficient_data):
        """A rescaling transformer silences it too, not only a reducing one.

        Suppression asks whether the block's original names survive, and every
        step transformer renames its output, so both consume the block as far as
        the diagnostic can tell. Discriminating would need output columns
        attributed back to their source variable, which the whole-frame reducer
        deliberately makes impossible. Documented rather than worked around.
        """
        y, X_future = rank_deficient_data
        forecaster = PointReductionForecaster(
            estimator=Ridge(),
            step_transformer=StepColumnReducer(reducer=SklearnStandardScaler()),
        )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            forecaster.fit(y, forecasting_horizon=HORIZON, X_future=X_future)
        assert not [w for w in record if "rank of only" in str(w.message)]

    def test_warning_fires_without_a_step_transformer(self, rank_deficient_data):
        """The population this diagnostic was written for is unaffected."""
        y, X_future = rank_deficient_data
        forecaster = PointReductionForecaster(estimator=Ridge())
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            forecaster.fit(y, forecasting_horizon=HORIZON, X_future=X_future)
        assert [w for w in record if "rank of only" in str(w.message)]

    def test_rank_message_names_the_step_transformer_remedy(self, rank_deficient_data):
        """The message offers reduction alongside actual_transformer."""
        y, X_future = rank_deficient_data
        forecaster = PointReductionForecaster(estimator=Ridge())
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            forecaster.fit(y, forecasting_horizon=HORIZON, X_future=X_future)
        messages = [str(w.message) for w in record if "rank of only" in str(w.message)]
        assert messages
        assert "step_transformer" in messages[0]

    def test_coverage_message_recommends_no_estimator(self):
        """The coverage warning states the measurement only.

        ``forecast-coverage-diagnostics`` forbids naming an estimator as a
        response, so the coverage-companion flag is documented rather than
        advertised here.
        """
        y = pl.DataFrame({"time": _times(N), "y": [float(i % 7) + i * 0.1 for i in range(N)]})
        vintage = _times(N)[20]
        X_forecast = pl.DataFrame({
            "vintage_time": [vintage, vintage],
            "time": [vintage + timedelta(days=1), vintage + timedelta(days=2)],
            "temp": [1.0, 2.0],
        })
        forecaster = PointReductionForecaster(estimator=Ridge(), step_transformer=StepAggregator(), nan_handling="drop")
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            forecaster.fit(y, forecasting_horizon=HORIZON, X_forecast=X_forecast)
        coverage = [str(w.message) for w in record if "covers" in str(w.message)]
        assert coverage
        for message in coverage:
            assert "StepAggregator" not in message
            assert "HistGradientBoosting" not in message
