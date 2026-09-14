"""Tests for HorizonRollingStatisticsTransformer."""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone

from conftest import run_checks
from yohou.compose import ColumnTransformer, FeaturePipeline, FeatureUnion
from yohou.point import MeanSeasonalNaive
from yohou.preprocessing import HorizonRollingStatisticsTransformer, LagTransformer
from yohou.testing import _yield_yohou_transformer_checks
from yohou.testing.common import check_metadata_routing_default_request


def _hourly(length: int, columns=("price",), seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    times = pl.datetime_range(
        datetime(2021, 1, 1), datetime(2021, 1, 1) + timedelta(hours=length - 1), interval="1h", eager=True
    )
    return pl.DataFrame({"time": times, **{c: rng.normal(size=length) for c in columns}})


def _expected(values: np.ndarray, t: int, h: int, k: int, n: int, stat=np.mean) -> float:
    """Reference value of step ``h`` at origin ``t`` from the spec formula."""
    first = math.ceil(h / k)
    return float(stat([values[t + h - k * j] for j in range(first, first + n)]))


class TestOutputs:
    """Columns, names and values."""

    def test_names_and_order(self):
        """Two columns x two statistics x 48 steps, ordered by column, statistic, step."""
        X = _hourly(400, columns=("price", "load"))
        transformer = HorizonRollingStatisticsTransformer(seasonality=24, n_seasons=7, statistics=["mean", "std"])
        X_t = transformer.fit(X, forecasting_horizon=48).transform(X)

        names = [c for c in X_t.columns if c != "time"]
        assert X_t.columns[0] == "time"
        assert len(names) == 192
        expected = [f"{c}_s24_{s}_step_{h}" for c in ("price", "load") for s in ("mean", "std") for h in range(1, 49)]
        assert names == expected
        assert transformer.get_feature_names_out() == expected
        assert {
            "price_s24_mean_step_1",
            "price_s24_mean_step_48",
            "price_s24_std_step_4",
            "load_s24_mean_step_12",
        } <= set(names)

    def test_string_statistic_equals_list(self):
        """A single statistic given as a string is the same as a one-item list."""
        X = _hourly(200)
        a = HorizonRollingStatisticsTransformer(seasonality=24, n_seasons=3, statistics="median")
        b = HorizonRollingStatisticsTransformer(seasonality=24, n_seasons=3, statistics=["median"])
        assert a.fit(X, forecasting_horizon=6).transform(X).equals(b.fit(X, forecasting_horizon=6).transform(X))

    def test_first_season_value_table(self):
        """k=4, n=2, H=4: step h averages x[t-(4-h)] and x[t-(8-h)]."""
        X = _hourly(60)
        X_t = HorizonRollingStatisticsTransformer(seasonality=4, n_seasons=2).fit(X, forecasting_horizon=4).transform(X)
        x = X["price"].to_numpy()
        offset = X.height - X_t.height
        for row, t in enumerate(range(offset, X.height)):
            got = X_t.row(row)[1:]
            want = (
                np.mean([x[t - 3], x[t - 7]]),
                np.mean([x[t - 2], x[t - 6]]),
                np.mean([x[t - 1], x[t - 5]]),
                np.mean([x[t], x[t - 4]]),
            )
            np.testing.assert_allclose(got, want, rtol=1e-12)

    @pytest.mark.parametrize(
        ("stat", "reference"),
        [
            ("mean", np.mean),
            ("std", lambda v: np.std(v, ddof=1)),
            ("min", np.min),
            ("max", np.max),
            ("median", np.median),
            ("sum", np.sum),
            ("var", lambda v: np.var(v, ddof=1)),
        ],
    )
    def test_matches_spec_formula(self, stat, reference):
        """Every step of every row equals the spec formula, across a season boundary."""
        k, n, horizon = 5, 3, 12
        X = _hourly(120)
        X_t = (
            HorizonRollingStatisticsTransformer(seasonality=k, n_seasons=n, statistics=stat)
            .fit(X, forecasting_horizon=horizon)
            .transform(X)
        )
        x = X["price"].to_numpy()
        offset = X.height - X_t.height
        for row, t in enumerate(range(offset, X.height)):
            want = [_expected(x, t, h, k, n, reference) for h in range(1, horizon + 1)]
            np.testing.assert_allclose(X_t.row(row)[1:], want, rtol=1e-10, atol=1e-12)

    def test_profile_repeats_past_one_season(self):
        """Steps h and h + k share a seasonal position and carry equal values."""
        X = _hourly(300)
        X_t = (
            HorizonRollingStatisticsTransformer(seasonality=24, n_seasons=2).fit(X, forecasting_horizon=48).transform(X)
        )
        assert X_t["price_s24_mean_step_28"].equals(X_t["price_s24_mean_step_4"], check_names=False)
        assert X_t["price_s24_mean_step_48"].equals(X_t["price_s24_mean_step_24"], check_names=False)

    def test_future_values_do_not_change_a_row(self):
        """Perturbing values after origin t leaves the row at t unchanged."""
        X = _hourly(200)
        transformer = HorizonRollingStatisticsTransformer(seasonality=24, n_seasons=3, statistics=["mean", "max"])
        transformer.fit(X, forecasting_horizon=30)
        before = transformer.transform(X)

        t_index = 150
        perturbed = X.with_columns(
            pl.when(pl.int_range(pl.len()) > t_index).then(pl.col("price") + 1000.0).otherwise(pl.col("price"))
        )
        after = transformer.transform(perturbed)
        origin = X["time"][t_index]
        assert before.filter(pl.col("time") == origin).equals(after.filter(pl.col("time") == origin))


class TestHorizonAndData:
    """Warm-up, data requirements and the fit-metadata horizon."""

    def test_warm_up(self):
        """observation_horizon is k*n - 1, independent of the horizon."""
        X = _hourly(500)
        transformer = HorizonRollingStatisticsTransformer(seasonality=24, n_seasons=7)
        X_t = transformer.fit(X, forecasting_horizon=48).transform(X)
        assert transformer.observation_horizon == 167
        assert X_t.height == 333
        assert X_t["time"][0] == X["time"][167]
        assert transformer.forecasting_horizon_ == 48

    def test_too_little_data(self):
        """Fewer than k*n rows cannot fill a window."""
        X = _hourly(167)
        with pytest.raises(ValueError, match="at least seasonality \\* n_seasons = 168 rows"):
            HorizonRollingStatisticsTransformer(seasonality=24, n_seasons=7).fit(X, forecasting_horizon=48)

    def test_missing_horizon(self):
        """Fitting without forecasting_horizon names the metadata and how to supply it."""
        X = _hourly(100)
        with pytest.raises(ValueError, match="requires `forecasting_horizon` as fit metadata"):
            HorizonRollingStatisticsTransformer(seasonality=24).fit(X)

    @pytest.mark.parametrize("horizon", [0, -3, 2.5, True])
    def test_invalid_horizon(self, horizon):
        """The horizon must be a positive integer."""
        X = _hourly(100)
        with pytest.raises(ValueError, match="forecasting_horizon"):
            HorizonRollingStatisticsTransformer(seasonality=24).fit(X, forecasting_horizon=horizon)

    @pytest.mark.parametrize(
        ("params", "name"),
        [
            ({"seasonality": 1}, "seasonality"),
            ({"seasonality": 24, "n_seasons": 0}, "n_seasons"),
            ({"seasonality": 24, "statistics": "bogus"}, "statistics"),
        ],
    )
    def test_invalid_parameters(self, params, name):
        """Invalid constructor parameters are rejected at fit, naming the parameter."""
        X = _hourly(100)
        with pytest.raises(ValueError, match=name if name != "statistics" else "Invalid statistics"):
            HorizonRollingStatisticsTransformer(**params).fit(X, forecasting_horizon=4)


class TestMeanSeasonalNaiveEquivalence:
    """The mean statistic reproduces MeanSeasonalNaive step for step."""

    @pytest.mark.parametrize("origin", [120, 200, 287])
    def test_equivalence(self, origin):
        """Step values at origin t equal MeanSeasonalNaive's 48-step forecast from t."""
        y = _hourly(288, columns=("y",), seed=3)
        history = y[: origin + 1]

        forecaster = MeanSeasonalNaive(seasonality=24, n_seasons=3).fit(y=history, forecasting_horizon=48)
        forecast = forecaster.predict(forecasting_horizon=48)["y"].to_numpy()

        transformer = HorizonRollingStatisticsTransformer(seasonality=24, n_seasons=3, statistics="mean")
        row = transformer.fit(history, forecasting_horizon=48).transform(history).tail(1)
        steps = row.select([f"y_s24_mean_step_{h}" for h in range(1, 49)]).row(0)

        np.testing.assert_allclose(steps, forecast, rtol=1e-9)


class TestSystematic:
    """Conformance checks with the horizon supplied as fit metadata."""

    @pytest.mark.parametrize(
        ("k", "n", "horizon"), [(4, 2, 4), (4, 3, 10), (24, 2, 48)], ids=["k4n2h4", "k4n3h10", "k24n2h48"]
    )
    def test_systematic_checks(self, k, n, horizon, time_series_train_test_factory):
        """All applicable transformer checks pass, including batch invariance and observe/rewind."""
        observation_horizon = k * n - 1
        X_train, X_test = time_series_train_test_factory(
            train_length=2 * observation_horizon + 60, test_length=observation_horizon + 30
        )
        transformer = HorizonRollingStatisticsTransformer(seasonality=k, n_seasons=n, statistics=["mean", "q25"])
        transformer.fit(X_train, forecasting_horizon=horizon)

        run_checks(
            transformer,
            _yield_yohou_transformer_checks(
                transformer, X_train, None, X_test, fit_params={"forecasting_horizon": horizon}
            ),
        )

    def test_default_request_exemption_is_scoped_to_the_tag(self):
        """An untagged transformer declaring the same default request still fails the check."""

        class _Untagged(LagTransformer):
            __metadata_request__fit = {"forecasting_horizon": True}

        check_metadata_routing_default_request(HorizonRollingStatisticsTransformer(seasonality=4))
        with pytest.raises(AssertionError, match="non-empty requests"):
            check_metadata_routing_default_request(_Untagged(lag=1))


class TestRoutingThroughComposites:
    """The fit horizon reaches the transformer inside every composite."""

    @staticmethod
    def _inner(composite):
        if isinstance(composite, FeatureUnion):
            return composite.transformer_list[0][1]
        if isinstance(composite, ColumnTransformer):
            return composite.named_transformers_["seasonal"]
        return composite.steps[-1][1]

    @pytest.mark.parametrize(
        "make",
        [
            lambda: FeatureUnion([
                ("seasonal", HorizonRollingStatisticsTransformer(seasonality=24)),
                ("lag", LagTransformer()),
            ]),
            lambda: ColumnTransformer([("seasonal", HorizonRollingStatisticsTransformer(seasonality=24), ["price"])]),
            lambda: FeaturePipeline([("seasonal", HorizonRollingStatisticsTransformer(seasonality=24))]),
        ],
        ids=["feature_union", "column_transformer", "feature_pipeline"],
    )
    def test_horizon_routed(self, make):
        """fit_transform with forecasting_horizon=48 sets the inner horizon and emits steps 1..48."""
        X = _hourly(200)
        composite = clone(make())
        X_t = composite.fit_transform(X, forecasting_horizon=48)

        assert self._inner(composite).forecasting_horizon_ == 48
        steps = sorted(int(c.rsplit("_step_", 1)[1]) for c in X_t.columns if "_s24_mean_step_" in c)
        assert steps == list(range(1, 49))
