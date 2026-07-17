"""Tests for the ``forecast_transformer`` slot: panel split, caches, and horizon guard."""

from datetime import datetime, timedelta

import polars as pl
import pytest
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from yohou.base.utils import _derive_step_columns
from yohou.compose import (
    ColumnForecaster,
    ColumnTransformer,
    DecompositionPipeline,
    FeaturePipeline,
    FeatureUnion,
    ForecastedFeatureForecaster,
    LocalPanelForecaster,
    PerVintageActualTransformer,
)
from yohou.point import PointReductionForecaster
from yohou.preprocessing import FunctionTransformer, LagTransformer

H = 3
N_TRAIN = 25


def _y(cols: list[str], n: int = N_TRAIN) -> pl.DataFrame:
    times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    return pl.DataFrame({"time": times, **{c: [float(i) for i in range(n)] for c in cols}})


def _x_forecast(cols: list[str], n: int = 30, n_steps: int = H, start: int = 0) -> pl.DataFrame:
    rows = []
    for i in range(start, start + n):
        vintage = datetime(2024, 1, 1) + timedelta(days=i)
        for step in range(1, n_steps + 1):
            rows.append({
                "vintage_time": vintage,
                "time": vintage + timedelta(days=step),
                **{c: float(i + step) for c in cols},
            })
    return pl.DataFrame(rows)


def _net_load() -> PerVintageActualTransformer:
    """Lifted ``load - wind``, which only works if each group's slice is unprefixed."""
    return PerVintageActualTransformer(
        FunctionTransformer(
            func=lambda df: df.select((pl.col("load") - pl.col("wind")).alias("net")),
            feature_names_out=lambda self, names: ["net"],
        )
    )


def _passthrough() -> PerVintageActualTransformer:
    return PerVintageActualTransformer(
        FunctionTransformer(
            func=lambda df: df.select(pl.col("temp")),
            feature_names_out=lambda self, names: ["temp"],
        )
    )


def _passthrough_load() -> PerVintageActualTransformer:
    """Stateless, load-only: consumes no rows from any vintage."""
    return PerVintageActualTransformer(
        FunctionTransformer(
            func=lambda df: df.select(pl.col("load")),
            feature_names_out=lambda self, names: ["load"],
        )
    )


def _forecaster(**kwargs) -> PointReductionForecaster:
    # "direct" fits one single-output model per step, which HistGradientBoosting
    # supports; it also tolerates the nulls a partial-loss transform leaves.
    return PointReductionForecaster(
        HistGradientBoostingRegressor(max_iter=5, random_state=42),
        reduction_strategy="direct",
        **kwargs,
    )


PANEL_COLS = ["a__load", "a__wind", "b__load", "b__wind"]


def test_step_columns_are_named_for_transformer_output():
    """The slot applies before derivation, so step columns carry output names."""
    forecaster = _forecaster(forecast_transformer=_net_load())
    forecaster.fit(_y(["y"]), X_forecast=_x_forecast(["load", "wind"]), forecasting_horizon=H)

    assert forecaster._step_column_names_ == {f"net_step_{h}" for h in range(1, H + 1)}


def test_slot_is_inert_when_unset():
    """An unset slot leaves X_forecast handling exactly as it was."""
    with_slot = _forecaster()
    with_slot.fit(_y(["y"]), X_forecast=_x_forecast(["load"]), forecasting_horizon=H)

    assert with_slot._step_column_names_ == {f"load_step_{h}" for h in range(1, H + 1)}
    assert with_slot.forecast_transformer_ is None


def test_actual_kind_transformer_is_rejected_by_the_kind_guard():
    """A LagTransformer is actual-kind, so the tag guard rejects it at fit."""
    forecaster = _forecaster(forecast_transformer=LagTransformer(lag=1))

    with pytest.raises(ValueError, match="forecast_transformer must be a forecast-kind"):
        forecaster.fit(_y(["y"]), X_forecast=_x_forecast(["load"]), forecasting_horizon=H)


def test_actual_kind_transformer_is_rejected_even_without_x_forecast():
    """The kind guard is a static check, so it fires at fit with no X_forecast passed.

    Otherwise a wrong-kind slot is silently accepted on a fit that omits X_forecast
    and only surfaces if X_forecast later appears at predict.
    """
    forecaster = _forecaster(forecast_transformer=LagTransformer(lag=1))

    with pytest.raises(ValueError, match="forecast_transformer must be a forecast-kind"):
        forecaster.fit(_y(["y"]), forecasting_horizon=H)


def test_actual_kind_composition_is_rejected_by_the_kind_guard():
    """The constraint admits BaseActualTransformer, so the tag is what rejects this."""
    forecaster = _forecaster(forecast_transformer=FeaturePipeline([("lag", LagTransformer(lag=1))]))

    with pytest.raises(ValueError, match="forecast_transformer must be a forecast-kind"):
        forecaster.fit(_y(["y"]), X_forecast=_x_forecast(["load"]), forecasting_horizon=H)


def test_forecast_kind_composition_is_accepted():
    """A forecast-kind FeatureUnion is a BaseActualTransformer reporting kind='forecast'.

    A ``[BaseForecastTransformer, None]`` constraint would reject it, which is why
    the constraint is loose and the kind guard is what discriminates.
    """
    union = FeatureUnion([("net", _net_load())])
    forecaster = _forecaster(forecast_transformer=union)
    forecaster.fit(_y(["y"]), X_forecast=_x_forecast(["load", "wind"]), forecasting_horizon=H)

    assert any("net" in c for c in forecaster._step_column_names_)


def test_panel_group_slices_reach_the_transformer_unprefixed():
    """The case that fails without the per-group split.

    ``load - wind`` cannot be expressed against ``a__load`` / ``b__load``; it works
    only because each group's slice carries its columns unprefixed.
    """
    forecaster = _forecaster(forecast_transformer=_net_load(), panel_strategy="global")
    forecaster.fit(_y(["a__y", "b__y"]), X_forecast=_x_forecast(PANEL_COLS), forecasting_horizon=H)

    assert isinstance(forecaster.forecast_transformer_, dict)
    assert sorted(forecaster.forecast_transformer_) == ["a", "b"]
    assert {"a__net_step_1", "b__net_step_1"} <= forecaster._step_column_names_


def test_panel_cache_is_a_single_wide_frame_not_a_dict():
    """``_X_forecast_t_`` holds the re-stacked frame that derivation consumes."""
    forecaster = _forecaster(forecast_transformer=_net_load(), panel_strategy="global")
    forecaster.fit(_y(["a__y", "b__y"]), X_forecast=_x_forecast(PANEL_COLS), forecasting_horizon=H)

    cache = forecaster._X_forecast_t_
    assert isinstance(cache, pl.DataFrame)
    assert set(cache.columns) == {"vintage_time", "time", "a__net", "b__net"}


def test_global_x_forecast_columns_are_localized_per_group():
    """A global column fed through a per-group-fitted transformer emits one column per group."""
    forecaster = _forecaster(forecast_transformer=_passthrough(), panel_strategy="global")
    forecaster.fit(
        _y(["a__y", "b__y"]),
        X_forecast=_x_forecast(["temp"]),
        forecasting_horizon=H,
    )

    assert {"a__temp_step_1", "b__temp_step_1"} <= forecaster._step_column_names_


def test_schema_stays_raw_while_step_columns_are_transformed():
    """The invariant the caches turn on: input contract raw, derived names transformed."""
    forecaster = _forecaster(forecast_transformer=_net_load())
    forecaster.fit(_y(["y"]), X_forecast=_x_forecast(["load", "wind"]), forecasting_horizon=H)

    # Raw: it is the contract checked against user frames at observe/predict.
    assert sorted(forecaster._X_forecast_schema_) == ["load", "wind"]
    # Transformed: derived from the transformer's output.
    assert all(c.startswith("net_step_") for c in forecaster._step_column_names_)


def test_observe_with_a_raw_frame_validates_against_the_raw_schema():
    """Users pass raw frames after fit; caching a transformed schema would break that."""
    forecaster = _forecaster(forecast_transformer=_net_load())
    forecaster.fit(_y(["y"]), X_forecast=_x_forecast(["load", "wind"]), forecasting_horizon=H)

    new_y = _y(["y"], n=2).with_columns(time=pl.col("time") + timedelta(days=N_TRAIN))
    forecaster.observe(new_y, X_forecast=_x_forecast(["load", "wind"], n=30))

    assert sorted(forecaster._X_forecast_schema_) == ["load", "wind"]


def test_transformed_cache_holds_transformed_values_not_raw():
    """The fit-time cache stores the transformed frame, so fallbacks reuse it."""
    forecaster = _forecaster(forecast_transformer=_net_load())
    forecaster.fit(_y(["y"]), X_forecast=_x_forecast(["load", "wind"]), forecasting_horizon=H)

    # net = load - wind: the transformed cache carries one net column, the raw two.
    assert "net" in forecaster._X_forecast_t_.columns
    assert set(forecaster._X_forecast_t_.columns) == {"vintage_time", "time", "net"}
    assert sorted(c for c in forecaster._X_forecast_raw_.columns if c not in ("vintage_time", "time")) == [
        "load",
        "wind",
    ]


def test_fallback_reuses_cache_without_re_transforming(monkeypatch):
    """observe() without X_forecast reads the transformed cache, not a re-transform.

    A no-arg fallback that re-ran the transform would show a transform call; the
    cache-read path must show none. This is what makes the cache load-bearing
    rather than decorative.
    """
    forecaster = _forecaster(forecast_transformer=_net_load())
    forecaster.fit(_y(["y"]), X_forecast=_x_forecast(["load", "wind"]), forecasting_horizon=H)

    calls = {"n": 0}
    original = type(forecaster)._transform_X_forecast

    def counting(self, X):
        calls["n"] += 1
        return original(self, X)

    monkeypatch.setattr(type(forecaster), "_transform_X_forecast", counting)

    new_y = _y(["y"], n=2).with_columns(time=pl.col("time") + timedelta(days=N_TRAIN))
    forecaster.observe(new_y)
    assert calls["n"] == 0, "observe() without X_forecast must reuse the cache, not re-transform"

    forecaster.observe(
        _y(["y"], n=2).with_columns(time=pl.col("time") + timedelta(days=N_TRAIN + 2)),
        X_forecast=_x_forecast(["load", "wind"], n=30),
    )
    assert calls["n"] == 1, "observe() with a supplied X_forecast transforms it exactly once"


def test_predict_override_is_transformed_exactly_once(monkeypatch):
    """A predict-time X_forecast override is transformed once, never double."""
    forecaster = _forecaster(forecast_transformer=_net_load())
    forecaster.fit(_y(["y"]), X_forecast=_x_forecast(["load", "wind"]), forecasting_horizon=H)

    calls = {"n": 0}
    original = type(forecaster)._transform_X_forecast

    def counting(self, X):
        calls["n"] += 1
        return original(self, X)

    monkeypatch.setattr(type(forecaster), "_transform_X_forecast", counting)

    forecaster.predict(X_forecast=_x_forecast(["load", "wind"], n=1, start=N_TRAIN - 1), forecasting_horizon=H)
    assert calls["n"] == 1, "a predict override must be transformed exactly once"


def test_rewind_narrows_the_transformed_cache_to_transformed_values():
    """After rewind with a supplied frame, the cache stays transformed and narrowed."""
    forecaster = _forecaster(forecast_transformer=_net_load())
    forecaster.fit(_y(["y"]), X_forecast=_x_forecast(["load", "wind"]), forecasting_horizon=H)

    new_y = _y(["y"], n=3).with_columns(time=pl.col("time") + timedelta(days=N_TRAIN))
    forecaster.rewind(new_y, X_forecast=_x_forecast(["load", "wind"], n=30))

    assert set(forecaster._X_forecast_t_.columns) == {"vintage_time", "time", "net"}
    assert forecaster._X_forecast_t_["vintage_time"].n_unique() == 1


@pytest.mark.parametrize("strategy", ["global", "multivariate"])
def test_walk_forward_transforms_at_predict_and_observe_predict(strategy):
    """The path the helper exists for.

    Under ``panel_strategy="global"`` the fitted slot is a per-group dict, and both
    of these sites live in code shared by standard and panel forecasters. A
    caller-side ``forecast_transformer_.transform(...)`` would raise
    ``AttributeError`` on the dict here.
    """
    cols = PANEL_COLS if strategy == "global" else ["load", "wind"]
    # "multivariate" keeps a__y and b__y as two wide targets, so it needs a
    # multi-output estimator; "global" pools them into one column.
    estimator = (
        HistGradientBoostingRegressor(max_iter=5, random_state=42)
        if strategy == "global"
        else MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=5, random_state=42))
    )
    forecaster = PointReductionForecaster(
        estimator,
        reduction_strategy="direct",
        forecast_transformer=_net_load(),
        panel_strategy=strategy,
    )
    forecaster.fit(_y(["a__y", "b__y"]), X_forecast=_x_forecast(cols), forecasting_horizon=H)

    assert isinstance(forecaster.forecast_transformer_, dict) is (strategy == "global")

    y_pred = forecaster.predict(X_forecast=_x_forecast(cols, n=1, start=N_TRAIN - 1), forecasting_horizon=H)
    assert len(y_pred) == H

    new_y = _y(["a__y", "b__y"], n=3).with_columns(time=pl.col("time") + timedelta(days=N_TRAIN))
    y_op = forecaster.observe_predict(new_y, X_forecast=_x_forecast(cols, n=30), forecasting_horizon=H)
    assert y_op["vintage_time"].n_unique() >= 1


def test_clone_of_a_slot_bearing_forecaster_refits():
    """A cloned forecaster carries the slot config and fits it from scratch."""
    cloned = clone(_forecaster(forecast_transformer=_net_load()))
    cloned.fit(_y(["y"]), X_forecast=_x_forecast(["load", "wind"]), forecasting_horizon=H)

    assert cloned.forecast_transformer_ is not None
    assert "net" in cloned._X_forecast_t_.columns
    assert len(cloned.predict(forecasting_horizon=H)) == H


def test_extended_horizon_predict_with_a_slot_raises():
    """Recursive predict beyond the fit horizon is unsupported once X_forecast was used."""
    forecaster = _forecaster(forecast_transformer=_net_load())
    forecaster.fit(_y(["y"], n=40), X_forecast=_x_forecast(["load", "wind"], n=40), forecasting_horizon=H)

    with pytest.raises(ValueError, match="Recursive prediction"):
        forecaster.predict(forecasting_horizon=H * 2)


def test_empty_vintage_narrowing_clears_the_transformed_cache():
    """Observing an X_forecast with no vintage at or before now clears the cache.

    The cache stays transformed-schema (an empty ``net`` frame), never reverting to
    raw columns, so a later fallback reads a consistent shape.
    """
    forecaster = _forecaster(forecast_transformer=_net_load())
    forecaster.fit(_y(["y"]), X_forecast=_x_forecast(["load", "wind"]), forecasting_horizon=H)

    new_y = _y(["y"], n=2).with_columns(time=pl.col("time") + timedelta(days=N_TRAIN))
    forecaster.observe(new_y, X_forecast=_x_forecast(["load", "wind"], n=5, start=200))

    assert len(forecaster._X_forecast_t_) == 0
    assert set(forecaster._X_forecast_t_.columns) == {"vintage_time", "time", "net"}


def test_rewind_under_panel_global_with_a_slot():
    """Covers the rewind path with a per-group dict slot."""
    forecaster = _forecaster(forecast_transformer=_net_load(), panel_strategy="global")
    forecaster.fit(_y(["a__y", "b__y"]), X_forecast=_x_forecast(PANEL_COLS), forecasting_horizon=H)

    new_y = _y(["a__y", "b__y"], n=3).with_columns(time=pl.col("time") + timedelta(days=N_TRAIN))
    forecaster.observe(new_y, X_forecast=_x_forecast(PANEL_COLS, n=30))
    forecaster.rewind(new_y, X_forecast=_x_forecast(PANEL_COLS, n=30))

    y_pred = forecaster.predict(forecasting_horizon=H)
    assert len(y_pred) == H


def test_horizon_below_the_minimum_raises_at_fit():
    """lag=5 needs 6-row vintages; a horizon of 5 could never serve one."""
    forecaster = _forecaster(forecast_transformer=PerVintageActualTransformer(LagTransformer(lag=5)))

    with pytest.raises(ValueError, match=r"forecasting_horizon \(5\).*\(6\)"):
        forecaster.fit(_y(["y"]), X_forecast=_x_forecast(["load"], n_steps=8), forecasting_horizon=5)


def test_stateless_slot_at_horizon_one_raises():
    """The case a ``forecasting_horizon > observation_horizon`` guard would admit.

    The horizon term is 0 for a stateless inner, but a one-row served vintage is
    below the fit minimum of two and would be dropped to an empty frame.
    """
    forecaster = _forecaster(forecast_transformer=_passthrough())

    with pytest.raises(ValueError, match=r"forecasting_horizon \(1\).*\(2\)"):
        forecaster.fit(_y(["y"]), X_forecast=_x_forecast(["temp"], n_steps=5), forecasting_horizon=1)


def test_stateless_slot_at_horizon_two_fits():
    """Two is the fit minimum, so a horizon of two is admissible."""
    forecaster = _forecaster(forecast_transformer=_passthrough())
    forecaster.fit(_y(["y"]), X_forecast=_x_forecast(["temp"], n_steps=5), forecasting_horizon=2)

    assert forecaster.forecast_transformer_ is not None


def test_unset_slot_checks_no_minimum():
    """A horizon of one is fine when no slot is set."""
    forecaster = _forecaster()
    forecaster.fit(_y(["y"]), X_forecast=_x_forecast(["load"], n_steps=1), forecasting_horizon=1)

    assert forecaster.forecast_transformer_ is None


def test_partial_loss_pads_the_consumed_steps_entirely_null():
    """A lifted lag consumes each vintage's nearest steps, padding them all-null.

    The guard admits this: ``lag=2`` reports a minimum of 3, which a horizon of 5
    meets. What it cannot see is that *every* vintage loses its first two rows, so
    the two nearest-term step columns are null for every row rather than null for
    some. An entirely-null feature is not merely lossy: it defeats
    ``nan_handling="drop"`` (every row carries a null, so every row drops) and
    breaks estimators that otherwise tolerate NaN, including
    ``HistGradientBoostingRegressor``, whose binner rejects an all-NaN column. The
    horizon guard covers total loss, not this.
    """
    transformer = PerVintageActualTransformer(LagTransformer(lag=2)).fit(_x_forecast(["load"], n=10, n_steps=5))
    assert transformer.min_vintage_rows == 3

    transformed = transformer.transform(_x_forecast(["load"], n=10, n_steps=5))
    assert transformed.group_by("vintage_time").len()["len"].unique().to_list() == [3]

    observation_times = pl.Series([datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)])
    step_columns = _derive_step_columns(None, transformed, observation_times, 5, "1d")

    # Steps 1 and 2 are padded back, entirely null; steps 3 to 5 carry values.
    for step in (1, 2):
        column = step_columns[f"load_lag_2_step_{step}"]
        assert column.null_count() == step_columns.height
    for step in (3, 4, 5):
        assert step_columns[f"load_lag_2_step_{step}"].null_count() == 0


def test_usage_tag_reports_the_slot():
    """``uses_forecast_transformer`` mirrors ``uses_actual_transformer``."""
    assert _forecaster(forecast_transformer=_passthrough()).__sklearn_tags__().forecaster_tags.uses_forecast_transformer
    assert not _forecaster().__sklearn_tags__().forecaster_tags.uses_forecast_transformer


def test_forecast_transformer_does_not_make_a_forecaster_stateful():
    """A forecast transformer carries no forecaster-level memory.

    Statefulness is about a buffer held *between* calls. A lifted lag consumes rows
    inside one vintage and keeps nothing across calls, so the forecaster stays
    stateless however stateful the inner is.
    """
    forecaster = _forecaster(forecast_transformer=PerVintageActualTransformer(LagTransformer(lag=2)))
    tags = forecaster.__sklearn_tags__().forecaster_tags

    assert tags.stateful is False
    assert tags.uses_forecast_transformer is True


def test_router_registers_predict_transform_but_no_memory_api():
    """The slot's transform is called at predict, so the mapping must exist.

    Omitting ``predict -> transform`` would make ``process_routing`` reject a
    request for a call that demonstrably happens. No ``observe``/``rewind``
    mapping: ``BaseForecastTransformer`` has neither method, and that absence is
    what keeps the slot clear of the forecast-kind memory guard.
    """
    forecaster = _forecaster(forecast_transformer=_passthrough())
    route = forecaster.get_metadata_routing()._route_mappings["forecast_transformer"]

    pairs = {(m.caller, m.callee) for m in route.mapping}
    assert pairs == {("fit", "fit"), ("fit", "transform"), ("predict", "transform")}
    assert not {"observe", "rewind"} & {m.callee for m in route.mapping}


def test_decomposition_pipeline_exposes_both_slots():
    """It declares the actual slot and calls ``_pre_fit``, so it gains the forecast slot."""
    pipeline = DecompositionPipeline([("a", _forecaster())])

    assert {"actual_transformer", "forecast_transformer"} <= set(pipeline.get_params())


@pytest.mark.parametrize("name", ["forecasted_feature", "local_panel", "column"])
def test_metas_that_never_exposed_a_slot_gain_neither(name):
    """``get_params`` must not advertise a parameter that would never fire."""
    inner = _forecaster()
    estimator = {
        "forecasted_feature": lambda: ForecastedFeatureForecaster(
            target_forecaster=inner, feature_forecaster=_forecaster()
        ),
        "local_panel": lambda: LocalPanelForecaster(forecaster=inner),
        "column": lambda: ColumnForecaster(forecasters=[("a", inner, ["y"])]),
    }[name]()

    assert not {"actual_transformer", "forecast_transformer"} & set(estimator.get_params())


@pytest.mark.parametrize("name", ["forecasted_feature", "local_panel", "column"])
def test_meta_propagates_forecast_usage_tag_from_child(name):
    """A meta wrapping a slot-bearing child reports ``uses_forecast_transformer``.

    The two transformer-usage tags propagate symmetrically: a meta must not report
    ``uses_actual_transformer`` while dropping ``uses_forecast_transformer``.
    """
    child = _forecaster(forecast_transformer=_passthrough_load())
    meta = {
        "forecasted_feature": lambda: ForecastedFeatureForecaster(
            target_forecaster=child, feature_forecaster=_forecaster()
        ),
        "local_panel": lambda: LocalPanelForecaster(forecaster=child),
        "column": lambda: ColumnForecaster(forecasters=[("a", child, ["y"])]),
    }[name]()

    assert meta.__sklearn_tags__().forecaster_tags.uses_forecast_transformer is True

    plain = {
        "forecasted_feature": lambda: ForecastedFeatureForecaster(
            target_forecaster=_forecaster(), feature_forecaster=_forecaster()
        ),
        "local_panel": lambda: LocalPanelForecaster(forecaster=_forecaster()),
        "column": lambda: ColumnForecaster(forecasters=[("a", _forecaster(), ["y"])]),
    }[name]()

    assert plain.__sklearn_tags__().forecaster_tags.uses_forecast_transformer is False


def test_slot_check_runs_across_reduction_families():
    """The systematic slot check fits every family that exposes the slot.

    Point reduction is exercised through the generator; interval and
    class-probability reduction forecasters use ``predict_interval`` /
    ``predict_class_proba`` rather than the point ``predict`` the generator's
    X_forecast block assumes, so the fit-only slot check covers them here.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeClassifier

    from yohou.class_proba import ClassProbaReductionForecaster
    from yohou.interval import IntervalReductionForecaster
    from yohou.testing.forecaster import check_forecast_transformer_slot

    numeric_y = _y(["y"], n=40)
    x_forecast = _x_forecast(["load"], n=40)

    check_forecast_transformer_slot(
        PointReductionForecaster(LinearRegression()), numeric_y, None, x_forecast, forecasting_horizon=H
    )
    check_forecast_transformer_slot(IntervalReductionForecaster(), numeric_y, None, x_forecast, forecasting_horizon=H)

    classes = ["a", "b", "c"]
    categorical_y = pl.DataFrame({
        "time": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(40)],
        "y": [classes[i % 3] for i in range(40)],
    })
    check_forecast_transformer_slot(
        ClassProbaReductionForecaster(estimator=DecisionTreeClassifier(random_state=42)),
        categorical_y,
        None,
        x_forecast,
        forecasting_horizon=H,
    )


def test_dead_step_warning_fires_for_a_lifted_lag():
    """A lifted lag empties the nearest-term steps, and fit says so.

    The horizon guard cannot cover this: the transform still yields rows, so there
    is nothing to assert against, and no horizon avoids it because the loss scales
    with the transformer. The warning is what stops a search over the inner's lag
    from trading the nearest steps away silently.
    """
    forecaster = _forecaster(forecast_transformer=PerVintageActualTransformer(LagTransformer(lag=2)))

    # The warning lands, and then the estimator fails on the all-null columns it
    # warned about: HistGradientBoosting cannot bin an all-NaN feature. Both halves
    # are pinned here, because the warning exists precisely to explain this failure.
    with (
        pytest.warns(UserWarning, match=r"consumes 2 row\(s\) from the start of every vintage"),
        pytest.raises(ValueError, match="window shape cannot be larger"),
    ):
        forecaster.fit(_y(["y"], n=60), X_forecast=_x_forecast(["load"], n=60, n_steps=5), forecasting_horizon=5)


def test_dead_step_warning_does_not_fire_for_a_stateless_slot(recwarn):
    """A stateless inner consumes nothing, so no step column is dead."""
    forecaster = _forecaster(forecast_transformer=_passthrough())
    forecaster.fit(_y(["y"], n=60), X_forecast=_x_forecast(["temp"], n=60, n_steps=5), forecasting_horizon=5)

    assert not [w for w in recwarn if "start of every vintage" in str(w.message)]


def test_dead_step_warning_fires_for_a_composition():
    """Wrapping the lag in a union must not hide the cost."""
    union = FeatureUnion([("lagged", PerVintageActualTransformer(LagTransformer(lag=2)))])
    forecaster = _forecaster(forecast_transformer=union)

    with pytest.warns(UserWarning, match="start of every vintage"), pytest.raises(ValueError):
        forecaster.fit(_y(["y"], n=60), X_forecast=_x_forecast(["load"], n=60, n_steps=5), forecasting_horizon=5)


def test_dead_step_warning_does_not_fire_for_a_whole_vintage_drop(recwarn):
    """A vintage dropped whole is loss in one vintage, not leading-row loss in all.

    A stateless slot consumes no leading rows from any full vintage, so nothing is
    dead. A single short vintage that PerVintage drops whole must not be mistaken
    for total loss: measuring the maximum row loss would let that one dropped tail
    vintage assert that every step column is null. The minimum measurement reads
    zero, so no dead-step warning fires.
    """
    rows = _x_forecast(["load"], n=25, n_steps=H).to_dicts()
    short_vintage = datetime(2024, 1, 1) + timedelta(days=25)
    rows.append({"vintage_time": short_vintage, "time": short_vintage + timedelta(days=1), "load": 99.0})
    x_forecast = pl.DataFrame(rows)

    forecaster = _forecaster(forecast_transformer=_passthrough_load())
    forecaster.fit(_y(["y"]), X_forecast=x_forecast, forecasting_horizon=H)

    assert not [w for w in recwarn if "start of every vintage" in str(w.message)]


def test_feature_union_reports_the_strictest_child_minimum():
    """Branches see the same vintage, so the strictest binds."""
    union = FeatureUnion([
        ("short", PerVintageActualTransformer(LagTransformer(lag=2))),
        ("long", PerVintageActualTransformer(LagTransformer(lag=5))),
    ]).fit(_x_forecast(["load"], n=12, n_steps=10))

    assert union.min_vintage_rows == 6


def test_column_transformer_reports_the_strictest_child_minimum():
    """Column groups see the same rows per vintage, so the strictest child binds."""
    column_transformer = ColumnTransformer([
        ("short", PerVintageActualTransformer(LagTransformer(lag=2)), ["load"]),
        ("long", PerVintageActualTransformer(LagTransformer(lag=5)), ["wind"]),
    ]).fit(_x_forecast(["load", "wind"], n=12, n_steps=10))

    assert column_transformer.min_vintage_rows == 6


def test_feature_pipeline_accumulates_and_never_understates():
    """Sequential steps consume rows in turn, so the requirements add up.

    The reported value must never fall below the shortest vintage that actually
    survives the chain: under-reporting would admit a configuration that silently
    empties every vintage, which is the failure the minimum exists to prevent.
    """
    pipeline = FeaturePipeline([
        ("first", PerVintageActualTransformer(LagTransformer(lag=2))),
        ("second", PerVintageActualTransformer(LagTransformer(lag=2))),
    ])
    reported = pipeline.fit(_x_forecast(["load"], n=12, n_steps=10)).min_vintage_rows

    def _survives(n: int) -> bool:
        chain = FeaturePipeline([
            ("first", PerVintageActualTransformer(LagTransformer(lag=2))),
            ("second", PerVintageActualTransformer(LagTransformer(lag=2))),
        ])
        try:
            return chain.fit_transform(_x_forecast(["load"], n=12, n_steps=n)).height > 0
        except ValueError:
            # A vintage too short is rejected outright rather than emptied.
            return False

    shortest_surviving = next(n for n in range(2, 10) if _survives(n))

    assert reported >= shortest_surviving
    assert reported == 5


def test_composition_is_guarded_exactly_as_its_leaf_is():
    """Wrapping a transformer in a union must not disable the horizon guard."""
    union = FeatureUnion([("lagged", PerVintageActualTransformer(LagTransformer(lag=5)))])
    forecaster = _forecaster(forecast_transformer=union)

    with pytest.raises(ValueError, match=r"forecasting_horizon \(5\).*\(6\)"):
        forecaster.fit(_y(["y"], n=60), X_forecast=_x_forecast(["load"], n=60, n_steps=8), forecasting_horizon=5)


def _ragged_x_forecast(n: int = 60, n_steps: int = 5) -> pl.DataFrame:
    """Full-length vintages plus a one-row tail, the shape a real forecast frame has.

    The horizon runs off the end of the series, so the last vintage is truncated.
    ``PerVintageActualTransformer`` documents this as the expected tail case and
    drops such vintages with a warning.
    """
    full = _x_forecast(["load"], n=n, n_steps=n_steps)
    tail_vintage = datetime(2024, 1, 1) + timedelta(days=n)
    tail = pl.DataFrame({
        "vintage_time": [tail_vintage],
        "time": [tail_vintage + timedelta(days=1)],
        "load": [1.0],
    })
    return pl.concat([full, tail])


def test_dead_step_warning_ignores_a_dropped_tail_vintage(recwarn):
    """A vintage dropped whole is not row consumption, and must not be reported as it.

    A truncated tail vintage is dropped entirely by the fit minimum, which joins to
    null and reads as total row loss. Taking the worst vintage rather than the least
    affected one would let that single dropped vintage assert that a transformer
    consuming nothing empties every step column, which is false in every particular:
    the step columns below have no nulls at all.
    """
    forecaster = _forecaster(forecast_transformer=_passthrough_load())
    forecaster.fit(_y(["y"], n=60), X_forecast=_ragged_x_forecast(), forecasting_horizon=5)

    assert not [w for w in recwarn if "start of every vintage" in str(w.message)]


def test_dead_step_warning_still_fires_when_a_tail_is_dropped_too(recwarn):
    """A real per-vintage loss is still reported when a dropped tail is also present.

    The dropped tail must not mask a genuine loss any more than it may invent one.
    """
    forecaster = _forecaster(forecast_transformer=PerVintageActualTransformer(LagTransformer(lag=1)))

    # The estimator then fails on the all-null step column the warning describes,
    # which is the behaviour pinned by test_dead_step_warning_fires_for_a_lifted_lag.
    with (
        pytest.warns(UserWarning, match=r"consumes 1 row\(s\) from the start of every vintage"),
        pytest.raises(ValueError, match="window shape cannot be larger"),
    ):
        forecaster.fit(_y(["y"], n=60), X_forecast=_ragged_x_forecast(), forecasting_horizon=5)
