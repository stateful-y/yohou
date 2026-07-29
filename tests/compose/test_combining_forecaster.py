"""Tests for the CombiningForecaster meta-forecaster."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone

from conftest import run_checks
from yohou.base import BaseActualTransformer
from yohou.compose import CombiningForecaster
from yohou.point import SeasonalNaive
from yohou.stationarity import PolynomialTrendForecaster
from yohou.testing import _yield_yohou_forecaster_checks

# ``check_clone_preserves_forecaster_params`` asserts that the trailing element of
# every 3-tuple parameter survives clone unchanged (``_safe_equal``). That assertion
# was written for ``ColumnForecaster``'s ``(name, forecaster, columns)`` shape, whose
# trailing ``columns`` is a clone-stable string/list. ``CombiningForecaster``'s term is
# ``(name, extractor, forecaster)``: its trailing element is an estimator, which
# ``clone`` correctly freshens into a new instance, so ``_safe_equal`` (identity-based
# for estimators) returns False. The clone is otherwise faithful (verified by the
# shared ``check_clone_preserves_params``, which checks the estimator at index 1 and
# the residual by type), so this single systematic check is a structural
# incompatibility of the layout, not a defect, and is marked expected-failing.
_CLONE_CHECK = "check_clone_preserves_forecaster_params"


# --------------------------------------------------------------------------------------
# Extractors used across the tests
# --------------------------------------------------------------------------------------


class _SelectColumn(BaseActualTransformer):
    """Stateless extractor selecting one column and renaming it as the term target."""

    def __init__(self, column: str, out_name: str):
        self.column = column
        self.out_name = out_name
        self._observation_horizon = 0

    @property
    def observation_horizon(self) -> int:
        """Return the (zero) observation horizon of this stateless extractor."""
        return 0

    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> "_SelectColumn":
        """Fit against the input frame, setting the transformer schema."""
        BaseActualTransformer.fit(self, X, y)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Return ``time`` plus the selected column renamed to ``out_name``."""
        return X.select([pl.col("time"), pl.col(self.column).alias(self.out_name)])

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return the single output feature name."""
        return [self.out_name]


class _ScaleColumn(BaseActualTransformer):
    """Stateless extractor producing ``factor * column`` as the term target."""

    def __init__(self, column: str, factor: float, out_name: str):
        self.column = column
        self.factor = factor
        self.out_name = out_name
        self._observation_horizon = 0

    @property
    def observation_horizon(self) -> int:
        """Return the (zero) observation horizon of this stateless extractor."""
        return 0

    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> "_ScaleColumn":
        """Fit against the input frame, setting the transformer schema."""
        BaseActualTransformer.fit(self, X, y)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Return ``time`` plus the scaled column."""
        return X.select([pl.col("time"), (pl.col(self.column) * self.factor).alias(self.out_name)])

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return the single output feature name."""
        return [self.out_name]


class _DifferenceExtractor(BaseActualTransformer):
    """Stateless extractor computing ``minuend - subtrahend`` as one global term."""

    def __init__(self, minuend: str, subtrahend: str, out_name: str):
        self.minuend = minuend
        self.subtrahend = subtrahend
        self.out_name = out_name
        self._observation_horizon = 0

    @property
    def observation_horizon(self) -> int:
        """Return the (zero) observation horizon of this stateless extractor."""
        return 0

    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> "_DifferenceExtractor":
        """Fit against the input frame, setting the transformer schema."""
        BaseActualTransformer.fit(self, X, y)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Return ``time`` plus the elementwise difference of the two columns."""
        return X.select([pl.col("time"), (pl.col(self.minuend) - pl.col(self.subtrahend)).alias(self.out_name)])

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return the single output feature name."""
        return [self.out_name]


class _PanelMean(BaseActualTransformer):
    """Stateless extractor: mean across panel columns, a single global anchor series."""

    def __init__(self, columns: list[str], out_name: str = "anchor"):
        self.columns = columns
        self.out_name = out_name
        self._observation_horizon = 0

    @property
    def observation_horizon(self) -> int:
        """Return the (zero) observation horizon of this stateless extractor."""
        return 0

    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> "_PanelMean":
        """Fit against the input frame, setting the transformer schema."""
        BaseActualTransformer.fit(self, X, y)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Return ``time`` plus the row-wise mean across the configured columns."""
        return X.select([pl.col("time"), pl.mean_horizontal([pl.col(c) for c in self.columns]).alias(self.out_name)])

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return the single output feature name."""
        return [self.out_name]


class _PanelIdentity(BaseActualTransformer):
    """Stateless extractor selecting the given panel columns verbatim (a panel term)."""

    def __init__(self, columns: list[str]):
        self.columns = columns
        self._observation_horizon = 0

    @property
    def observation_horizon(self) -> int:
        """Return the (zero) observation horizon of this stateless extractor."""
        return 0

    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> "_PanelIdentity":
        """Fit against the input frame, setting the transformer schema."""
        BaseActualTransformer.fit(self, X, y)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Return ``time`` plus the selected panel columns unchanged."""
        return X.select([pl.col("time"), *[pl.col(c) for c in self.columns]])

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return the selected panel column names."""
        return list(self.columns)


class _SpyExtractor(BaseActualTransformer):
    """Identity extractor recording, at class level, how many times transform runs."""

    n_transform_calls = 0

    def __init__(self, column: str, out_name: str):
        self.column = column
        self.out_name = out_name
        self._observation_horizon = 0

    @property
    def observation_horizon(self) -> int:
        """Return the (zero) observation horizon of this stateless extractor."""
        return 0

    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> "_SpyExtractor":
        """Fit against the input frame, setting the transformer schema."""
        BaseActualTransformer.fit(self, X, y)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Record the call and return ``time`` plus the renamed column."""
        type(self).n_transform_calls += 1
        return X.select([pl.col("time"), pl.col(self.column).alias(self.out_name)])

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return the single output feature name."""
        return [self.out_name]


class _StatefulExtractor(BaseActualTransformer):
    """Extractor tagged stateful, used to verify CombiningForecaster rejects it."""

    _tags = {"stateful": True}

    def __init__(self, column: str, out_name: str):
        self.column = column
        self.out_name = out_name
        self._observation_horizon = 1

    @property
    def observation_horizon(self) -> int:
        """Return the (nonzero) observation horizon of this stateful extractor."""
        return 1

    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None) -> "_StatefulExtractor":
        """Fit against the input frame, setting the transformer schema."""
        BaseActualTransformer.fit(self, X, y)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Return ``time`` plus the renamed column."""
        return X.select([pl.col("time"), pl.col(self.column).alias(self.out_name)])

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return the single output feature name."""
        return [self.out_name]


# --------------------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------------------


def _daily(n: int, start: datetime = datetime(2020, 1, 1)) -> pl.Series:
    """Return a length-``n`` daily datetime series."""
    return pl.datetime_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)


def _global_y(n: int = 60) -> pl.DataFrame:
    """Return a single-column non-panel target with trend plus a weekly cycle."""
    time = _daily(n)
    return pl.DataFrame({"time": time, "total": [10.0 + i + (i % 7) for i in range(n)]})


def _panel_y(n: int = 60) -> pl.DataFrame:
    """Return a two settlement-point panel target."""
    time = _daily(n)
    return pl.DataFrame({
        "time": time,
        "p0__spp": [10.0 + i + (i % 7) for i in range(n)],
        "p1__spp": [20.0 + 2 * i + (i % 5) for i in range(n)],
    })


# --------------------------------------------------------------------------------------
# Systematic conformance harness
# --------------------------------------------------------------------------------------


class TestSystematicChecks:
    """Systematic validation of CombiningForecaster via the shared check generators."""

    @pytest.mark.parametrize(
        "forecaster",
        [
            CombiningForecaster(
                terms=[
                    ("season", _SelectColumn("y_0", "season_part"), SeasonalNaive(seasonality=7)),
                    ("trend", _SelectColumn("y_0", "trend_part"), PolynomialTrendForecaster(degree=1)),
                ]
            ),
            CombiningForecaster(
                terms=[("trend", _SelectColumn("y_0", "trend_part"), PolynomialTrendForecaster(degree=1))],
                residual_forecaster=SeasonalNaive(seasonality=7),
            ),
        ],
    )
    @pytest.mark.slow
    def test_checks(self, forecaster, y_X_factory):
        """Run the systematic forecaster checks on CombiningForecaster configs.

        ``check_clone_preserves_forecaster_params`` is expected-failing because the
        term layout is ``(name, extractor, forecaster)`` whose trailing element is an
        estimator (see the module-level note). Every other check must pass.
        """
        y, X_actual = y_X_factory(length=100, n_targets=1, n_features=0, seed=42)
        y = y.with_columns([(pl.col(col).abs() + 1).alias(col) for col in y.columns if col != "time"])

        y_train, y_test = y[:80], y[80:]

        forecaster_fitted = clone(forecaster)
        forecaster_fitted.fit(y_train, None, forecasting_horizon=3)

        run_checks(
            forecaster_fitted,
            _yield_yohou_forecaster_checks(forecaster_fitted, y_train, None, y_test, None),
            expected_failures={_CLONE_CHECK},
        )


# --------------------------------------------------------------------------------------
# Global summation
# --------------------------------------------------------------------------------------


class TestGlobalSum:
    """Two global terms sum to the expected total (non-panel target)."""

    def test_two_global_terms_sum(self):
        """The prediction equals the elementwise sum of the two terms' forecasts."""
        y = _global_y()
        y_train = y[:45]

        forecaster = CombiningForecaster(
            terms=[
                ("season", _SelectColumn("total", "season_part"), SeasonalNaive(seasonality=7)),
                ("trend", _SelectColumn("total", "trend_part"), PolynomialTrendForecaster(degree=1)),
            ]
        )
        forecaster.fit(y_train, forecasting_horizon=3)
        pred = forecaster.predict(forecasting_horizon=3)

        assert pred.columns == ["vintage_time", "time", "total"]
        assert pred.height == 3

        season = SeasonalNaive(seasonality=7)
        season.fit(y_train.rename({"total": "season_part"}), forecasting_horizon=3)
        trend = PolynomialTrendForecaster(degree=1)
        trend.fit(y_train.rename({"total": "trend_part"}), forecasting_horizon=3)
        expected = (
            season.predict(forecasting_horizon=3)["season_part"].to_numpy()
            + trend.predict(forecasting_horizon=3)["trend_part"].to_numpy()
        )
        assert np.allclose(pred["total"].to_numpy(), expected)


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


class TestValidation:
    """Fit-time validation of term forecasters, names, and extractors."""

    def test_non_point_term_forecaster_rejected(self):
        """A non-point term forecaster raises ValueError naming the offending term."""

        class _NotAForecaster:
            """A stand-in object that is not a BasePointForecaster."""

        y = _global_y()[:30]
        forecaster = CombiningForecaster(terms=[("bad", _SelectColumn("total", "part"), _NotAForecaster())])
        with pytest.raises(ValueError, match="Term 'bad'"):
            forecaster.fit(y, forecasting_horizon=3)

    def test_non_point_residual_forecaster_rejected(self):
        """A non-point residual forecaster raises ValueError."""

        class _NotAForecaster:
            """A stand-in object that is not a BasePointForecaster."""

        y = _global_y()[:30]
        forecaster = CombiningForecaster(
            terms=[("a", _SelectColumn("total", "part"), SeasonalNaive(seasonality=7))],
            residual_forecaster=_NotAForecaster(),
        )
        with pytest.raises(ValueError, match="residual_forecaster"):
            forecaster.fit(y, forecasting_horizon=3)

    def test_duplicate_term_names_rejected(self):
        """Two terms sharing a name raise ValueError."""
        y = _global_y()[:30]
        forecaster = CombiningForecaster(
            terms=[
                ("dup", _SelectColumn("total", "a"), SeasonalNaive(seasonality=7)),
                ("dup", _SelectColumn("total", "b"), PolynomialTrendForecaster(degree=1)),
            ]
        )
        with pytest.raises(ValueError, match="not unique"):
            forecaster.fit(y, forecasting_horizon=3)

    def test_stateful_extractor_rejected(self):
        """A stateful extractor is rejected at fit, naming the term."""
        y = _global_y()[:30]
        forecaster = CombiningForecaster(
            terms=[("stateful", _StatefulExtractor("total", "part"), SeasonalNaive(seasonality=7))]
        )
        with pytest.raises(ValueError, match="Term 'stateful'"):
            forecaster.fit(y, forecasting_horizon=3)


# --------------------------------------------------------------------------------------
# Extractor semantics
# --------------------------------------------------------------------------------------


class TestExtractorSemantics:
    """Extractors manufacture term targets at fit and never run at predict."""

    def test_difference_extractor_builds_term_label(self):
        """A difference extractor fits the term forecaster on the derived series."""
        n = 40
        time = _daily(n)
        y = pl.DataFrame({"time": time, "spp": [30.0 + i for i in range(n)]})
        X_actual = pl.DataFrame({"time": time, "system_lambda_dam": [10.0 + 0.5 * i for i in range(n)]})

        forecaster = CombiningForecaster(
            terms=[
                (
                    "basis",
                    _DifferenceExtractor("spp", "system_lambda_dam", "basis"),
                    SeasonalNaive(seasonality=7),
                )
            ]
        )
        forecaster.fit(y[:30], X_actual=X_actual[:30], forecasting_horizon=3)

        # The term forecaster was fitted on the derived "basis" series, not raw "spp".
        name, term_forecaster, granularity = forecaster.forecasters_[0]
        assert name == "basis"
        assert granularity == "global"
        assert list(term_forecaster.local_y_schema_.keys()) == ["basis"]

        # Cross-check the fitted basis against the manual difference.
        expected_basis = (y["spp"] - X_actual["system_lambda_dam"]).to_numpy()[:30]
        derived = forecaster.extractors_["basis"].transform(y[:30].join(X_actual[:30], on="time"))["basis"].to_numpy()
        assert np.allclose(derived, expected_basis)

    def test_extractor_not_invoked_at_predict(self):
        """The extractor runs at fit but not during predict (call count)."""
        y = _global_y()[:40]

        _SpyExtractor.n_transform_calls = 0
        forecaster = CombiningForecaster(terms=[("spy", _SpyExtractor("total", "part"), SeasonalNaive(seasonality=7))])
        forecaster.fit(y[:30], forecasting_horizon=3)
        calls_after_fit = _SpyExtractor.n_transform_calls
        assert calls_after_fit >= 1, "extractor must run at fit"

        forecaster.predict(forecasting_horizon=3)
        assert _SpyExtractor.n_transform_calls == calls_after_fit, "extractor must not run at predict"

        # observe re-extracts, confirming the extractor is still wired for streaming.
        forecaster.observe(y[30:33])
        assert _SpyExtractor.n_transform_calls > calls_after_fit


# --------------------------------------------------------------------------------------
# Residual closure
# --------------------------------------------------------------------------------------


class TestResidualClosure:
    """The optional residual forecaster closes the gap the declared terms leave."""

    def test_exact_split_reconstructs_target(self):
        """An exact scaled split plus residual reconstructs the direct forecast of y.

        With a linear ``SeasonalNaive`` term and residual, forecasting a scaled part
        (``0.3 * total``) and its complement (the residual ``0.7 * total``) sums back
        to forecasting the full ``total`` directly.
        """
        y = _global_y()
        y_train = y[:45]

        forecaster = CombiningForecaster(
            terms=[("part", _ScaleColumn("total", 0.3, "part"), SeasonalNaive(seasonality=7))],
            residual_forecaster=SeasonalNaive(seasonality=7),
        )
        forecaster.fit(y_train, forecasting_horizon=3)
        pred = forecaster.predict(forecasting_horizon=3)

        direct = SeasonalNaive(seasonality=7)
        direct.fit(y_train, forecasting_horizon=3)
        assert np.allclose(pred["total"].to_numpy(), direct.predict(forecasting_horizon=3)["total"].to_numpy())

    def test_approximate_split_residual_closes_gap(self):
        """The residual forecast is added on top of the term forecast to close a gap."""
        y = _global_y()
        y_train = y[:45]

        term_extractor = _ScaleColumn("total", 0.5, "part")
        forecaster = CombiningForecaster(
            terms=[("part", term_extractor, PolynomialTrendForecaster(degree=1))],
            residual_forecaster=SeasonalNaive(seasonality=7),
        )
        forecaster.fit(y_train, forecasting_horizon=3)
        pred_with_residual = forecaster.predict(forecasting_horizon=3)

        # Manual reconstruction: term forecast + residual forecast.
        part = PolynomialTrendForecaster(degree=1)
        part.fit(_ScaleColumn("total", 0.5, "part").fit_transform(y_train), forecasting_horizon=3)
        term_pred = part.predict(forecasting_horizon=3)["part"].to_numpy()

        eps = y_train.select([pl.col("time"), (pl.col("total") - 0.5 * pl.col("total")).alias("total")])
        residual = SeasonalNaive(seasonality=7)
        residual.fit(eps, forecasting_horizon=3)
        residual_pred = residual.predict(forecasting_horizon=3)["total"].to_numpy()

        assert np.allclose(pred_with_residual["total"].to_numpy(), term_pred + residual_pred)

        # Without the residual, the prediction is only the term forecast (a real gap).
        forecaster_no_residual = CombiningForecaster(
            terms=[("part", _ScaleColumn("total", 0.5, "part"), PolynomialTrendForecaster(degree=1))]
        )
        forecaster_no_residual.fit(y_train, forecasting_horizon=3)
        pred_no_residual = forecaster_no_residual.predict(forecasting_horizon=3)["total"].to_numpy()
        assert np.allclose(pred_no_residual, term_pred)
        assert not np.allclose(pred_with_residual["total"].to_numpy(), pred_no_residual)

    def test_no_residual_is_plain_sum(self):
        """residual_forecaster=None yields the plain sum of the term forecasts."""
        y = _global_y()
        y_train = y[:45]

        forecaster = CombiningForecaster(
            terms=[
                ("a", _ScaleColumn("total", 0.4, "a"), SeasonalNaive(seasonality=7)),
                ("b", _ScaleColumn("total", 0.6, "b"), SeasonalNaive(seasonality=7)),
            ]
        )
        forecaster.fit(y_train, forecasting_horizon=3)
        pred = forecaster.predict(forecasting_horizon=3)

        fa = SeasonalNaive(seasonality=7)
        fa.fit(_ScaleColumn("total", 0.4, "a").fit_transform(y_train), forecasting_horizon=3)
        fb = SeasonalNaive(seasonality=7)
        fb.fit(_ScaleColumn("total", 0.6, "b").fit_transform(y_train), forecasting_horizon=3)
        expected = fa.predict(forecasting_horizon=3)["a"].to_numpy() + fb.predict(forecasting_horizon=3)["b"].to_numpy()
        assert np.allclose(pred["total"].to_numpy(), expected)


# --------------------------------------------------------------------------------------
# Panel behavior
# --------------------------------------------------------------------------------------


class TestPanel:
    """Global terms broadcast across a panel; panel terms align per group."""

    def test_global_term_broadcasts_across_groups(self):
        """A global term's single forecast is added to every settlement point."""
        y = _panel_y()
        y_train = y[:45]

        forecaster = CombiningForecaster(
            terms=[("anchor", _PanelMean(["p0__spp", "p1__spp"]), SeasonalNaive(seasonality=7))]
        )
        forecaster.fit(y_train, forecasting_horizon=3)
        pred = forecaster.predict(forecasting_horizon=3)

        assert set(pred.columns) == {"vintage_time", "time", "p0__spp", "p1__spp"}
        anchor = SeasonalNaive(seasonality=7)
        anchor.fit(_PanelMean(["p0__spp", "p1__spp"]).fit_transform(y_train), forecasting_horizon=3)
        anchor_pred = anchor.predict(forecasting_horizon=3)["anchor"].to_numpy()
        # Both groups receive the same broadcast anchor forecast.
        assert np.allclose(pred["p0__spp"].to_numpy(), anchor_pred)
        assert np.allclose(pred["p1__spp"].to_numpy(), anchor_pred)

    def test_panel_term_aligns_per_group(self):
        """Each group's panel term forecast is added only to that group."""
        y = _panel_y()
        y_train = y[:45]

        forecaster = CombiningForecaster(
            terms=[("basis", _PanelIdentity(["p0__spp", "p1__spp"]), SeasonalNaive(seasonality=7))]
        )
        forecaster.fit(y_train, forecasting_horizon=3)
        pred = forecaster.predict(forecasting_horizon=3)

        # A single identity panel term reproduces a plain per-group SeasonalNaive.
        base = SeasonalNaive(seasonality=7)
        base.fit(y_train, forecasting_horizon=3)
        base_pred = base.predict(forecasting_horizon=3)
        assert np.allclose(pred["p0__spp"].to_numpy(), base_pred["p0__spp"].to_numpy())
        assert np.allclose(pred["p1__spp"].to_numpy(), base_pred["p1__spp"].to_numpy())

    def test_mixed_global_panel_residual_reconstructs(self):
        """A global anchor plus a per-point residual reconstructs the panel target."""
        y = _panel_y()
        y_train = y[:45]

        forecaster = CombiningForecaster(
            terms=[("anchor", _PanelMean(["p0__spp", "p1__spp"]), PolynomialTrendForecaster(degree=1))],
            residual_forecaster=SeasonalNaive(seasonality=7),
        )
        forecaster.fit(y_train, forecasting_horizon=3)
        pred = forecaster.predict(forecasting_horizon=3)

        anchor = PolynomialTrendForecaster(degree=1)
        anchor.fit(_PanelMean(["p0__spp", "p1__spp"]).fit_transform(y_train), forecasting_horizon=3)
        anchor_pred = anchor.predict(forecasting_horizon=3)["anchor"].to_numpy()

        eps = y_train.select([
            pl.col("time"),
            (pl.col("p0__spp") - pl.mean_horizontal(pl.col("p0__spp"), pl.col("p1__spp"))).alias("p0__spp"),
            (pl.col("p1__spp") - pl.mean_horizontal(pl.col("p0__spp"), pl.col("p1__spp"))).alias("p1__spp"),
        ])
        residual = SeasonalNaive(seasonality=7)
        residual.fit(eps, forecasting_horizon=3)
        residual_pred = residual.predict(forecasting_horizon=3)

        assert np.allclose(pred["p0__spp"].to_numpy(), anchor_pred + residual_pred["p0__spp"].to_numpy())
        assert np.allclose(pred["p1__spp"].to_numpy(), anchor_pred + residual_pred["p1__spp"].to_numpy())

    def test_missing_group_in_panel_term_raises(self):
        """A panel term forecast that omits a group raises, naming the group."""

        class _DropGroup(BaseActualTransformer):
            """Panel extractor that drops p1, yielding an incomplete panel term."""

            def __init__(self):
                self._observation_horizon = 0

            @property
            def observation_horizon(self) -> int:
                """Return the (zero) observation horizon of this stateless extractor."""
                return 0

            def fit(self, X, y=None):
                """Fit against the input frame, setting the transformer schema."""
                BaseActualTransformer.fit(self, X, y)
                return self

            def transform(self, X):
                """Return ``time`` plus only the p0 column (dropping p1)."""
                return X.select([pl.col("time"), pl.col("p0__spp")])

            def get_feature_names_out(self, input_features=None):
                """Return the single retained panel column name."""
                return ["p0__spp"]

        y = _panel_y()
        forecaster = CombiningForecaster(terms=[("drop", _DropGroup(), SeasonalNaive(seasonality=7))])
        forecaster.fit(y[:45], forecasting_horizon=3)
        with pytest.raises(ValueError, match="p1"):
            forecaster.predict(forecasting_horizon=3)

    def test_group_subselection_filters_output(self):
        """Predicting a single group returns only that group's column."""
        y = _panel_y()
        forecaster = CombiningForecaster(
            terms=[("anchor", _PanelMean(["p0__spp", "p1__spp"]), SeasonalNaive(seasonality=7))],
            residual_forecaster=SeasonalNaive(seasonality=7),
        )
        forecaster.fit(y[:45], forecasting_horizon=3)
        pred = forecaster.predict(forecasting_horizon=3, groups=["p0"])
        assert set(pred.columns) == {"vintage_time", "time", "p0__spp"}


# --------------------------------------------------------------------------------------
# observe / rewind parity
# --------------------------------------------------------------------------------------


class TestObserveRewind:
    """Streamed observe matches refit-on-window; rewind restores the fit state."""

    def test_global_observe_matches_refit(self):
        """Global streamed-observe matches a refit on the concatenated window."""
        y = _global_y()
        y_train, y_new = y[:45], y[45:51]

        forecaster = CombiningForecaster(
            terms=[
                ("a", _ScaleColumn("total", 0.5, "a"), SeasonalNaive(seasonality=7)),
                ("b", _ScaleColumn("total", 0.5, "b"), SeasonalNaive(seasonality=1)),
            ]
        )
        forecaster.fit(y_train, forecasting_horizon=3)
        forecaster.observe(y_new)
        streamed = forecaster.predict(forecasting_horizon=3)

        refit = clone(forecaster)
        refit.fit(pl.concat([y_train, y_new]), forecasting_horizon=3)
        assert np.allclose(streamed["total"].to_numpy(), refit.predict(forecasting_horizon=3)["total"].to_numpy())

    def test_panel_observe_matches_refit(self):
        """Panel streamed-observe (with residual) matches a refit on the window."""
        y = _panel_y()
        y_train, y_new = y[:45], y[45:51]

        forecaster = CombiningForecaster(
            terms=[("anchor", _PanelMean(["p0__spp", "p1__spp"]), SeasonalNaive(seasonality=7))],
            residual_forecaster=SeasonalNaive(seasonality=7),
        )
        forecaster.fit(y_train, forecasting_horizon=3)
        forecaster.observe(y_new)
        streamed = forecaster.predict(forecasting_horizon=3)

        refit = clone(forecaster)
        refit.fit(pl.concat([y_train, y_new]), forecasting_horizon=3)
        refit_pred = refit.predict(forecasting_horizon=3)
        assert np.allclose(streamed["p0__spp"].to_numpy(), refit_pred["p0__spp"].to_numpy())
        assert np.allclose(streamed["p1__spp"].to_numpy(), refit_pred["p1__spp"].to_numpy())

    def test_rewind_restores_fit_state(self):
        """Observe then rewind to the training window matches the original fit."""
        y = _global_y()
        y_train, y_new = y[:45], y[45:51]

        forecaster = CombiningForecaster(terms=[("a", _SelectColumn("total", "a"), SeasonalNaive(seasonality=7))])
        forecaster.fit(y_train, forecasting_horizon=3)
        forecaster.observe(y_new)
        forecaster.rewind(y_train)
        rewound = forecaster.predict(forecasting_horizon=3)

        base = CombiningForecaster(terms=[("a", _SelectColumn("total", "a"), SeasonalNaive(seasonality=7))])
        base.fit(y_train, forecasting_horizon=3)
        assert np.allclose(rewound["total"].to_numpy(), base.predict(forecasting_horizon=3)["total"].to_numpy())

    def test_observe_predict_runs(self):
        """observe_predict produces stacked vintages via the custom observe."""
        y = _global_y()
        forecaster = CombiningForecaster(terms=[("a", _SelectColumn("total", "a"), SeasonalNaive(seasonality=7))])
        forecaster.fit(y[:30], forecasting_horizon=3)
        pred = forecaster.observe_predict(y[30:39], forecasting_horizon=3)
        assert "vintage_time" in pred.columns
        assert pred["vintage_time"].n_unique() >= 2


# --------------------------------------------------------------------------------------
# get_params / set_params
# --------------------------------------------------------------------------------------


class TestParams:
    """Nested term parameters are addressable through get_params/set_params."""

    def test_nested_params_round_trip(self):
        """A nested term parameter is exposed and settable."""
        forecaster = CombiningForecaster(
            terms=[("trend", _SelectColumn("total", "part"), PolynomialTrendForecaster(degree=1))]
        )
        params = forecaster.get_params()
        assert "trend__degree" in params
        assert params["trend__degree"] == 1

        forecaster.set_params(trend__degree=3)
        # The extractor survives the set_params merge (still a 3-tuple).
        name, extractor, sub_forecaster = forecaster.terms[0]
        assert name == "trend"
        assert isinstance(extractor, _SelectColumn)
        assert sub_forecaster.degree == 3


# --------------------------------------------------------------------------------------
# Extractor state guard and empty terms
# --------------------------------------------------------------------------------------


class TestStateAndEmptyValidation:
    """The extractor guard keys on observation state, and empty terms are rejected."""

    def test_stateless_functiontransformer_accepted(self):
        """A stateless FunctionTransformer (observation_horizon 0) is a valid extractor."""
        from yohou.preprocessing import FunctionTransformer

        y = _global_y()[:45]
        anchor = FunctionTransformer(func=lambda df: df.select((pl.col("total") * 0.5).alias("anchor")))
        forecaster = CombiningForecaster(terms=[("anchor", anchor, SeasonalNaive(seasonality=7))])
        forecaster.fit(y, forecasting_horizon=3)  # must not raise
        pred = forecaster.predict(forecasting_horizon=3)
        assert pred.columns == ["vintage_time", "time", "total"]

    def test_empty_terms_rejected(self):
        """An empty terms list raises at fit rather than crashing later at predict."""
        y = _global_y()[:30]
        forecaster = CombiningForecaster(terms=[])
        with pytest.raises(ValueError, match="at least one term"):
            forecaster.fit(y, forecasting_horizon=3)


# --------------------------------------------------------------------------------------
# Exogenous forwarding (X_actual features and X_forecast) to term forecasters
# --------------------------------------------------------------------------------------


class TestExogenousForwarding:
    """X_actual features and X_forecast reach the term forecaster unchanged."""

    def test_x_actual_reaches_term_forecaster(self, y_X_factory):
        """A term forecaster consuming X_actual features matches a standalone fit."""
        from sklearn.linear_model import Ridge

        from yohou.point import PointReductionForecaster

        y, X_actual = y_X_factory(length=100, n_targets=1, n_features=2, seed=3)
        y_train, X_train = y[:80], X_actual[:80]

        forecaster = CombiningForecaster(
            terms=[("t", _SelectColumn("y_0", "y_0"), PointReductionForecaster(estimator=Ridge()))]
        )
        forecaster.fit(y_train, X_train, forecasting_horizon=3)
        pred = forecaster.predict(forecasting_horizon=3)

        standalone = PointReductionForecaster(estimator=Ridge())
        standalone.fit(y_train, X_train, forecasting_horizon=3)
        expected = standalone.predict(forecasting_horizon=3)["y_0"].to_numpy()
        # Equality only holds if X_actual reached the term forecaster.
        assert np.allclose(pred["y_0"].to_numpy(), expected)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_x_forecast_reaches_term_forecaster(self, y_X_factory):
        """X_forecast forwarded to a term forecaster matches a standalone fit (the day-ahead path)."""
        from sklearn.linear_model import Ridge

        from yohou.point import PointReductionForecaster

        y, X_actual, _X_future, X_forecast = y_X_factory(
            length=100,
            n_targets=1,
            n_features=1,
            n_forecast_features=1,
            forecasting_horizon=3,
            return_exogenous=True,
            seed=5,
        )
        y_train, X_train = y[:80], X_actual[:80]

        forecaster = CombiningForecaster(
            terms=[("t", _SelectColumn("y_0", "y_0"), PointReductionForecaster(estimator=Ridge()))]
        )
        forecaster.fit(y_train, X_train, forecasting_horizon=3, X_forecast=X_forecast)
        pred = forecaster.predict(forecasting_horizon=3, X_forecast=X_forecast)

        standalone = PointReductionForecaster(estimator=Ridge())
        standalone.fit(y_train, X_train, forecasting_horizon=3, X_forecast=X_forecast)
        expected = standalone.predict(forecasting_horizon=3, X_forecast=X_forecast)["y_0"].to_numpy()
        # Equality only holds if X_forecast reached the term forecaster.
        assert np.allclose(pred["y_0"].to_numpy(), expected)


# --------------------------------------------------------------------------------------
# Metadata routing
# --------------------------------------------------------------------------------------


class TestRouting:
    """Metadata routing is wired to each named term and the residual forecaster."""

    def test_metadata_routing_wires_terms_and_residual(self):
        """get_metadata_routing routes fit to every named term and the residual."""
        from sklearn.utils.metadata_routing import MetadataRouter

        forecaster = CombiningForecaster(
            terms=[
                ("a", _SelectColumn("total", "a"), SeasonalNaive(seasonality=7)),
                ("b", _SelectColumn("total", "b"), SeasonalNaive(seasonality=1)),
            ],
            residual_forecaster=SeasonalNaive(seasonality=7),
        )
        router = forecaster.get_metadata_routing()
        assert isinstance(router, MetadataRouter)

        routed = router.route_params(caller="fit", params={})
        assert {"a", "b", "residual_forecaster"} <= set(routed)


# --------------------------------------------------------------------------------------
# Multiplicative combination (combine="product")
# --------------------------------------------------------------------------------------


class TestProductCombine:
    """Group-parameterized reconstruction with combine='product'."""

    def test_two_global_terms_product(self):
        """The prediction equals the elementwise product of the two terms' forecasts."""
        y = _global_y()  # strictly positive
        y_train = y[:45]

        forecaster = CombiningForecaster(
            terms=[
                ("a", _ScaleColumn("total", 0.5, "a_part"), SeasonalNaive(seasonality=7)),
                ("b", _ScaleColumn("total", 2.0, "b_part"), SeasonalNaive(seasonality=7)),
            ],
            combine="product",
        )
        forecaster.fit(y_train, forecasting_horizon=3)
        pred = forecaster.predict(forecasting_horizon=3)

        assert pred.columns == ["vintage_time", "time", "total"]

        fa = SeasonalNaive(seasonality=7)
        fa.fit(y_train.select("time", (pl.col("total") * 0.5).alias("a_part")), forecasting_horizon=3)
        fb = SeasonalNaive(seasonality=7)
        fb.fit(y_train.select("time", (pl.col("total") * 2.0).alias("b_part")), forecasting_horizon=3)
        expected = (
            fa.predict(forecasting_horizon=3)["a_part"].to_numpy()
            * fb.predict(forecasting_horizon=3)["b_part"].to_numpy()
        )
        assert np.allclose(pred["total"].to_numpy(), expected)

    def test_product_residual_reconstructs_exact_split(self):
        """With a multiplicative residual eps = y / prod, the split reconstructs the target."""
        y = _global_y()
        y_train = y[:45]
        forecaster = CombiningForecaster(
            terms=[("part", _ScaleColumn("total", 0.5, "part"), SeasonalNaive(seasonality=7))],
            combine="product",
            residual_forecaster=SeasonalNaive(seasonality=7),
        )
        forecaster.fit(y_train, forecasting_horizon=3)
        pred = forecaster.predict(forecasting_horizon=3)
        # SeasonalNaive on both the 0.5*y term and the y/(0.5*y)=2 residual is exact here,
        # so the product reconstructs the SeasonalNaive forecast of the target itself.
        direct = SeasonalNaive(seasonality=7)
        direct.fit(y_train, forecasting_horizon=3)
        assert np.allclose(pred["total"].to_numpy(), direct.predict(forecasting_horizon=3)["total"].to_numpy())

    def test_non_positive_target_rejected(self):
        """combine='product' rejects a non-positive target at fit."""
        y = _global_y()[:30].with_columns(pl.col("total") - 100.0)  # now <= 0
        forecaster = CombiningForecaster(
            terms=[("a", _SelectColumn("total", "part"), SeasonalNaive(seasonality=7))],
            combine="product",
        )
        with pytest.raises(ValueError, match="positive.*target"):
            forecaster.fit(y, forecasting_horizon=3)

    def test_non_positive_term_rejected(self):
        """combine='product' rejects a non-positive term target, naming the term."""
        y = _global_y()[:30]  # positive target
        forecaster = CombiningForecaster(
            terms=[("neg", _ScaleColumn("total", -1.0, "part"), SeasonalNaive(seasonality=7))],
            combine="product",
        )
        with pytest.raises(ValueError, match="term 'neg'"):
            forecaster.fit(y, forecasting_horizon=3)


class TestRetainedOperandsAreNotCombined:
    """A term whose target transform keeps operands must contribute only its target.

    ``ArithmeticTransformer(keep_inputs=True)`` retains the sibling operand so its
    inverse is defined, which makes the term forecaster emit two columns per group: the
    recovered value and the sibling. Only the first is the term's contribution. Before
    ``target_output_name`` existed, the sibling was summed into the aggregate too, which
    is silent numerical corruption: the aggregate came out wrong with no error raised.
    """

    @staticmethod
    def _frame(n: int = 60) -> pl.DataFrame:
        time = pl.datetime_range(
            datetime(2024, 1, 1), datetime(2024, 1, 1) + timedelta(hours=n - 1), interval="1h", eager=True
        )
        return pl.DataFrame({
            "time": time,
            "total": [40.0 + (i % 5) for i in range(n)],
            "gas": [3.0 + (i % 3) * 0.1 for i in range(n)],
        })

    def _fitted(self):
        from yohou.preprocessing import ArithmeticTransformer, FunctionTransformer

        y = self._frame().select("time", "total")
        x_actual = self._frame().select("time", "gas")
        # The term's target is `anchor`; `gas_ref` rides along only so `anchor` can be
        # rebuilt as heat_rate * gas_ref at predict.
        extractor = FunctionTransformer(
            func=lambda df: df.select(pl.col("total").alias("anchor"), pl.col("gas").alias("gas_ref")),
            check_inverse=False,
        )
        term = PolynomialTrendForecaster(
            degree=1,
            target_transformer=ArithmeticTransformer(
                "anchor",
                "gas_ref",
                op="div",
                output_name="heat_rate",
                keep_inputs=True,
                invert_wrt="left",
            ),
        )
        forecaster = CombiningForecaster(terms=[("anchor", extractor, term)], combine="sum")
        return forecaster.fit(y, X_actual=x_actual, forecasting_horizon=4), y

    def test_declared_contribution_member_is_recorded(self):
        forecaster, _ = self._fitted()
        assert forecaster.contribution_members_["anchor"] == "anchor"

    def test_aggregate_excludes_the_retained_operand(self):
        forecaster, _ = self._fitted()
        term = {n: f for n, f, _ in forecaster.forecasters_}["anchor"]
        term_pred = term.predict(forecasting_horizon=4)
        assert {"anchor", "gas_ref"} <= set(term_pred.columns), "the term emits both, by design"

        agg = forecaster.predict(forecasting_horizon=4)
        # The aggregate must equal the target alone. Summing the operand too would add
        # roughly the gas level (~3) to every prediction.
        assert agg["total"].to_list() == pytest.approx(term_pred["anchor"].to_list(), abs=1e-9)
        gas_polluted = [a + g for a, g in zip(term_pred["anchor"], term_pred["gas_ref"], strict=True)]
        assert agg["total"].to_list() != pytest.approx(gas_polluted, abs=1e-9)

    def test_value_preserving_transform_declares_nothing(self):
        from yohou.stationarity import LogTransformer

        assert LogTransformer().target_output_name is None
        assert _SelectColumn("a", "out").target_output_name is None


class TestNamedForecasters:
    """Terms are summed, so their position is not a handle; their name is.

    Reaching a term through ``forecasters_[i][1]`` reads a different term the moment
    anyone reorders ``terms``, and reordering is meant to be free.
    """

    @staticmethod
    def _frame(n: int = 40) -> pl.DataFrame:
        time = pl.datetime_range(
            datetime(2024, 1, 1), datetime(2024, 1, 1) + timedelta(hours=n - 1), interval="1h", eager=True
        )
        return pl.DataFrame({"time": time, "total": [40.0 + (i % 5) for i in range(n)]})

    def _terms(self):
        return [
            ("trend", _ScaleColumn("total", 0.6, "trend_part"), PolynomialTrendForecaster(degree=1)),
            ("season", _ScaleColumn("total", 0.4, "season_part"), SeasonalNaive(seasonality=4)),
        ]

    def _fitted(self, terms):
        y = self._frame()
        return CombiningForecaster(terms=terms, combine="sum").fit(y, forecasting_horizon=3)

    def test_returns_the_fitted_term_under_its_name(self):
        fitted = self._fitted(self._terms())
        assert set(fitted.named_forecasters_) == {"trend", "season"}
        assert isinstance(fitted.named_forecasters_["trend"], PolynomialTrendForecaster)
        assert isinstance(fitted.named_forecasters_["season"], SeasonalNaive)

    def test_term_order_is_not_a_handle(self):
        """The same name resolves to the same kind of forecaster in either order."""
        forward = self._fitted(self._terms())
        reversed_ = self._fitted(list(reversed(self._terms())))
        for name in ("trend", "season"):
            assert type(forward.named_forecasters_[name]) is type(reversed_.named_forecasters_[name])
        # ...while the positional lookup does not survive the reorder.
        assert type(forward.forecasters_[0][1]) is not type(reversed_.forecasters_[0][1])

    def test_requires_fit(self):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            _ = CombiningForecaster(terms=self._terms()).named_forecasters_

    def test_agrees_with_the_positional_list(self):
        """The named view is a view, not a copy: same instances, nothing removed."""
        fitted = self._fitted(self._terms())
        for name, forecaster, _granularity in fitted.forecasters_:
            assert fitted.named_forecasters_[name] is forecaster


class TestGlobalTermWithPanelExogenous:
    """A global term must not be handed the panel columns it cannot use.

    Every term receives the same ``X_actual``. ``_validate_pre_fit`` compares a term's panel
    groups against ``X_actual``'s, so a global target paired with a panel ``X_actual`` raised
    "Panel groups mismatch" before the term's own feature routing ran. That made a
    decomposition combining a global term with a panel one unfittable whenever ``X_actual``
    carried group columns, which it must whenever a panel term's extractor reads its target
    from there. The mix was untested: the broadcast tests pass no ``X_actual`` at all, and the
    exogenous tests use a global target with a global ``X_actual``.
    """

    @staticmethod
    def _panel_x(n: int = 60) -> pl.DataFrame:
        time = _daily(n)
        return pl.DataFrame({
            "time": time,
            "gas": [3.0 + 0.01 * i for i in range(n)],
            "p0__basis": [0.5 * i for i in range(n)],
            "p1__basis": [0.25 * i for i in range(n)],
        })

    def test_a_global_term_fits_beside_a_panel_one(self):
        y = _panel_y()
        x = self._panel_x()
        forecaster = CombiningForecaster(
            terms=[
                ("anchor", _PanelMean(["p0__spp", "p1__spp"]), SeasonalNaive(seasonality=7)),
                ("basis", _PanelIdentity(["p0__spp", "p1__spp"]), SeasonalNaive(seasonality=7)),
            ]
        )
        forecaster.fit(y[:45], X_actual=x[:45], forecasting_horizon=3)

        granularities = {name: g for name, _fc, g in forecaster.forecasters_}
        assert granularities == {"anchor": "global", "basis": "panel"}

    def test_the_global_term_predicts_across_every_group(self):
        y = _panel_y()
        x = self._panel_x()
        forecaster = CombiningForecaster(
            terms=[
                ("anchor", _PanelMean(["p0__spp", "p1__spp"]), SeasonalNaive(seasonality=7)),
                ("basis", _PanelIdentity(["p0__spp", "p1__spp"]), SeasonalNaive(seasonality=7)),
            ]
        )
        forecaster.fit(y[:45], X_actual=x[:45], forecasting_horizon=3)
        pred = forecaster.predict(forecasting_horizon=3)

        assert set(pred.columns) == {"vintage_time", "time", "p0__spp", "p1__spp"}
        assert pred["p0__spp"].is_finite().all()

    def test_the_panel_term_still_receives_the_group_columns(self):
        """Narrowing applies to the global term only; a panel term keeps the whole frame."""
        y = _panel_y()
        x = self._panel_x()
        seen: dict[str, list[str]] = {}

        class _Recording(SeasonalNaive):
            def __init__(self, seasonality: int, label: str = ""):
                super().__init__(seasonality=seasonality)
                self.label = label

            def fit(self, y, X_actual=None, **kw):  # noqa: ANN001, ANN003
                seen[self.label] = [] if X_actual is None else list(X_actual.columns)
                return super().fit(y, X_actual=X_actual, **kw)

        CombiningForecaster(
            terms=[
                ("anchor", _PanelMean(["p0__spp", "p1__spp"]), _Recording(7, "anchor")),
                ("basis", _PanelIdentity(["p0__spp", "p1__spp"]), _Recording(7, "basis")),
            ]
        ).fit(y[:45], X_actual=x[:45], forecasting_horizon=3)

        assert seen["anchor"] == ["time", "gas"], "the global term was handed panel columns"
        assert "p0__basis" in seen["basis"], "the panel term lost its group columns"

    def test_a_global_only_exogenous_frame_is_untouched(self):
        """Nothing to narrow, so the global term sees exactly what it was given."""
        y = _panel_y()
        time = _daily(60)
        x = pl.DataFrame({"time": time, "gas": [3.0 + 0.01 * i for i in range(60)]})
        forecaster = CombiningForecaster(
            terms=[("anchor", _PanelMean(["p0__spp", "p1__spp"]), SeasonalNaive(seasonality=7))]
        )
        forecaster.fit(y[:45], X_actual=x[:45], forecasting_horizon=3)

        assert forecaster.forecasters_[0][2] == "global"
