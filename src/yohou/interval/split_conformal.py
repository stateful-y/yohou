"""Implementation of conformal forecasters."""

import numbers
import warnings
from typing import Literal

import numpy as np
import polars as pl
from pydantic import StrictFloat, StrictInt
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from yohou.base.panel import BasePanelForecaster
from yohou.metrics import BaseConformityScorer, Residual
from yohou.point import BasePointForecaster, SeasonalNaive
from yohou.utils import POINT_INTERVAL, Tags, validate_forecaster_data
from yohou.utils._compat import Interval, StrOptions, _fit_context

from .base import BaseConformalAdapter, BaseIntervalForecaster, BaseSimilarity
from .utils import pooled_weights, warn_if_calibration_too_small, warn_if_weights_collapsed, weighted_quantile

__all__ = ["SplitConformalForecaster"]


class SplitConformalForecaster(BaseIntervalForecaster):
    """Split conformal forecaster implementation.

    Wraps a point forecaster and calibrates prediction intervals using
    split conformal prediction.  A held-out calibration set is used to
    compute conformity scores whose quantiles define the interval width.

    Parameters
    ----------
    point_forecaster : BasePointForecaster, default=SeasonalNaive()
        Point forecaster used to generate point predictions.
    calibration_size : int >= 1, default=100
        Number of observations to use for calibration.
    conformity_scorer : BaseConformityScorer, default=Residual()
        Scorer used to compute conformity scores.
    similarity : BaseSimilarity or None, default=None
        Similarity measure used to weight conformity scores.
    adapter : BaseConformalAdapter or None, default=None
        Adaptive conformal inference adapter. When ``None`` (the default),
        intervals use the static calibrated level. When supplied, the
        forecaster tracks a time-varying effective miscoverage level per
        coverage rate and horizon step, updating it online from realized
        coverage. Composes with ``similarity``: the adapter sets the level,
        the similarity sets the weights.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data. See `BaseForecaster` for details. This
        governs the wrapped *point model* and the per-group state, not the
        calibration.
    calibration_strategy : {"local", "global"}, default="local"
        Which conformity scores a value column's quantile is drawn from.
        ``"local"`` (the default) uses that column's own scores, so each
        entity is calibrated independently. ``"global"`` draws one quantile
        from every column's scores together and rebuilds each column's bound
        with its own scale, which lifts the ceiling on the coverage rates a
        short series can express.

        Note that ``panel_strategy`` and ``calibration_strategy`` both accept
        ``"global"`` and neither implies the other. ``panel_strategy="global"``
        shares the *point model* across entities; ``calibration_strategy="global"``
        shares the *calibration*. The default pairing, a shared point model with
        per-entity calibration, is a sensible and common configuration.

        ``"global"`` requires a conformity scorer whose scores are comparable
        across columns, and raises at ``fit`` otherwise: pooling scores of
        different magnitude or volatility produces a systematically
        miscalibrated interval rather than a merely imprecise one. Whether
        pooling helps at all depends on how correlated your entities are; see
        the interval-forecasting explanation page.

    Attributes
    ----------
    fit_coverage_rates_ : list of float
        Coverage rates used during fit.

    Notes
    -----
    The data is split into a training portion and a calibration portion
    of size ``calibration_size``.  The point forecaster is fit on the
    training portion, then conformity scores are computed on the
    calibration portion.  At prediction time, interval bounds are
    derived from the empirical quantiles of these scores.

    See Also
    --------
    - [`BaseSimilarity`][yohou.interval.base.BaseSimilarity] : Similarity weighting for adaptive intervals.
    - [`Residual`][yohou.metrics.conformity.Residual] : Default conformity scorer.
    - [`IntervalReductionForecaster`][yohou.interval.reduction.IntervalReductionForecaster] : Alternative interval forecaster.

    """

    _parameter_constraints: dict = {
        **BaseIntervalForecaster._parameter_constraints,
        "point_forecaster": [BasePointForecaster],
        "calibration_size": [Interval(numbers.Integral, 1, None, closed="left")],
        "conformity_scorer": [BaseConformityScorer],
        "similarity": [BaseSimilarity, None],
        "adapter": [BaseConformalAdapter, None],
        "calibration_strategy": [StrOptions({"local", "global"})],
    }

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with forecaster_type set to POINT_INTERVAL since this
            forecaster produces both point predictions and intervals.

        """
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None
        # SplitConformal wraps a point forecaster and adds intervals
        tags.forecaster_tags.forecaster_type = POINT_INTERVAL
        return tags

    def __init__(
        self,
        point_forecaster: BasePointForecaster = SeasonalNaive(),
        calibration_size: StrictInt = 100,
        conformity_scorer: BaseConformityScorer = Residual(),
        similarity: BaseSimilarity | None = None,
        adapter: BaseConformalAdapter | None = None,
        panel_strategy: Literal["global", "multivariate"] = "global",
        calibration_strategy: Literal["local", "global"] = "local",
    ):
        BaseIntervalForecaster.__init__(self, panel_strategy=panel_strategy)

        self.point_forecaster = point_forecaster
        self.conformity_scorer = conformity_scorer
        self.similarity = similarity
        self.adapter = adapter
        self.calibration_size = calibration_size
        self.calibration_strategy = calibration_strategy

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        coverage_rates: list[StrictFloat] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> "SplitConformalForecaster":
        """Fit the forecaster to historical data.

        Trains the wrapped point forecaster, calibrates conformity scores
        on a held-out calibration set, and optionally fits similarity
        weights.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Passed directly to the wrapped
            ``point_forecaster_``; any feature transformation is the
            responsibility of that inner estimator. If ``None``, only
            target-derived features are used.
        forecasting_horizon : int, default=1
            Number of time steps to forecast into the future.
        coverage_rates : list of float or None, default=None
            Coverage levels for prediction intervals (e.g., ``[0.9, 0.95]``
            for 90 % and 95 % intervals).  If ``None``, defaults to
            ``[0.95]``.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column. Deterministic
            values available for past and future dates.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns. Passed to the wrapped ``point_forecaster_``; any
            feature transformation is the responsibility of that inner
            estimator, reachable as ``point_forecaster__forecast_transformer``.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted forecaster instance.

        """
        # Validate data and set interval
        y, X_actual, _ = validate_forecaster_data(self, y, X_actual, reset=True)

        # _pre_fit: set schemas/panel attributes, fit transformers
        # (target_transformer=None, actual_transformer=None → no-ops),
        # and populate observation buffers (observed_time_, _y_observed,
        # _X_t_observed).  Called on the full y before the
        # train/calibration split so base-class state reflects the
        # complete training history.
        self._pre_fit(y, X_actual, forecasting_horizon, X_future=X_future, X_forecast=X_forecast)

        # Validate interval-specific parameters (coverage rates)
        _, self.fit_coverage_rates_ = self._validate_interval_fit_params(self.fit_forecasting_horizon_, coverage_rates)

        # Handle splitting with optional X
        if X_actual is None:
            y_train, y_calib = train_test_split(y, test_size=self.calibration_size, shuffle=False)
            X_actual_train, X_actual_calib = None, None
        else:
            y_train, y_calib, X_actual_train, X_actual_calib = train_test_split(
                y, X_actual, test_size=self.calibration_size, shuffle=False
            )

        self.point_forecaster_ = clone(self.point_forecaster).fit(
            y=y_train,
            X_actual=X_actual_train,
            forecasting_horizon=forecasting_horizon,
            X_future=X_future,
            X_forecast=X_forecast,
        )

        # stride=1: each row of y_calib produces one prediction window of length
        # forecasting_horizon.  This yields calibration_size - step + 1 conformity
        # scores for each horizon step k, instead of the ~2-3 scores that result
        # from stride=forecasting_horizon.  More calibration scores per step gives
        # quantiles that are stable and well-separated.
        #
        # The stride is not a cost knob, which is why there is no trade to make here.
        # Each origin yields exactly one score per horizon step, so scores, origins and
        # predictions are the same number: raising the stride buys speed only by
        # discarding scores one for one.  With a weighted quantile over
        # similarity-concentrated scores, that budget is already close to its floor.
        #
        # X_future and X_forecast are forwarded, not withheld.  Given either,
        # `_observe_predict_loop` derives step columns once over every observation time
        # and each origin selects its row; given neither, it falls through to `observe`,
        # which re-derives them one timestamp at a time.  Withholding them cost one
        # `_derive_step_columns` call per calibration origin where one call covers them
        # all.  The two branches resolve vintages as-of per observation time either way,
        # so the predictions are identical rather than merely close.
        # The point forecaster is frozen for the whole replay, so no origin depends on
        # having predicted at the one before it. Where the wrapped forecaster can say so,
        # the origins are recorded first and predicted in one pass per horizon step, so
        # the estimator call count stops scaling with the origin count. That is the same
        # rows through the same estimators, and it is bit-identical for tree models.
        # Anything that cannot make the guarantee keeps the rolling path.
        point = self.point_forecaster_
        direct = getattr(point, "reduction_strategy", None) == "direct"
        bulk = getattr(point, "_observe_predict_bulk_origins", None)
        batched = getattr(point, "_observe_predict_batched_origins", None)

        # Three paths, most to least aggressive. The bulk one additionally skips the
        # rolling observe, which is sound only where every transformer declares
        # `batch_invariant`; that path moves rolling-statistic columns by about one ULP,
        # so it is taken only on a declared stack. Anything undeclared keeps a
        # bit-identical path, which is what makes a missing declaration cost speed alone.
        # Recorded as a fitted attribute, not just chosen. Which path ran decides what
        # state the forecaster is left in, so a state defect is only attributable to a
        # path if the fitted object can say which one it took. Establishing that for the
        # 2026-08-08 regression cost a bisect across two submodule bumps.
        if direct and bulk is not None and point._chains_are_batch_invariant():
            replay_path = "bulk"
        elif direct and batched is not None:
            replay_path = "batched"
        else:
            replay_path = "rolling"
        self.replay_path_ = replay_path

        if direct and bulk is not None and point._chains_are_batch_invariant():
            y_pred_calib = bulk(
                y=y_calib,
                X_actual=X_actual_calib,
                groups=point.groups_ or [],
                X_future=X_future,
                X_forecast=X_forecast,
                predict_transformed=False,
            )
        elif direct and batched is not None:
            y_pred_calib = batched(
                y=y_calib,
                X_actual=X_actual_calib,
                groups=point.groups_ or [],
                stride=1,
                X_future=X_future,
                X_forecast=X_forecast,
                predict_transformed=False,
            )
        else:
            y_pred_calib = self.point_forecaster_.observe_predict(
                y=y_calib,
                X_actual=X_actual_calib,
                forecasting_horizon=None,
                stride=1,
                predict_transformed=False,
                X_future=X_future,
                X_forecast=X_forecast,
            )

        # Pooling scores that are not comparable across columns produces a
        # systematically miscalibrated interval, not a merely imprecise one, so
        # this refuses rather than warns.
        if self.calibration_strategy == "global":
            scorer_tags = self.conformity_scorer.__sklearn_tags__()
            assert scorer_tags.scorer_tags is not None
            if not scorer_tags.scorer_tags.comparable_across_columns:
                raise ValueError(
                    f"calibration_strategy='global' pools conformity scores across value columns, "
                    f"which requires a scorer whose scores are comparable across them. "
                    f"{type(self.conformity_scorer).__name__} does not declare that: its scores carry "
                    f"each column's own magnitude or volatility, so one pooled quantile would be too "
                    f"wide for some columns and too narrow for others. Use a dispersion-normalized "
                    f"scorer such as NormalizedResidual, or keep calibration_strategy='local'."
                )

        conformity_scorers = {}
        conformity_scores_list: list[pl.DataFrame] = []
        similarities = {}

        for step in range(1, 1 + forecasting_horizon):
            y_pred_calib_step = y_pred_calib[step - 1 :: forecasting_horizon]
            y_truth_step = y_calib

            conformity_scorer_step = clone(self.conformity_scorer).fit(y_calib, forecaster=self.point_forecaster_)
            conformity_scores_step = conformity_scorer_step.score(y_truth_step, y_pred_calib_step)

            conformity_scores_step = conformity_scores_step.with_columns(step=step)
            conformity_scores_list.append(conformity_scores_step)

            conformity_scorers[f"step_{step}"] = conformity_scorer_step

            # Fit similarity on the same scored subset to ensure length alignment
            if self.similarity is not None:
                scored_times_df = conformity_scores_step.drop("step").select("time")
                y_pred_for_sim = y_pred_calib_step.drop("vintage_time", strict=False).join(
                    scored_times_df, on="time", how="semi"
                )

                similarity_step = clone(self.similarity)
                similarity_step.fit(y=y_calib, y_pred=y_pred_for_sim)

                similarities[f"step_{step}"] = similarity_step

        conformity_scores = pl.concat(conformity_scores_list)

        self.conformity_scorers_ = conformity_scorers
        self.conformity_scores_ = conformity_scores

        if self.similarity is not None:
            self.similarities_ = similarities
            # Track fit-time counts for correct rewind arithmetic
            self._fit_score_counts_ = {}
            for step in range(1, 1 + forecasting_horizon):
                key = f"step_{step}"
                self._fit_score_counts_[key] = conformity_scores.filter(pl.col("step") == step).height

        self._check_replay_left_the_block_observed(y, replay_path)

        if self.adapter is not None:
            # One cloned adapter per horizon step, mirroring similarities_.
            # Each clone is told the tracked coverage rates and the scorer's
            # symmetry so it can seed one effective level per (rate, tail).
            scorer_tags = self.conformity_scorer.__sklearn_tags__()
            assert scorer_tags.scorer_tags is not None
            self._adapter_symmetric_ = scorer_tags.scorer_tags.symmetric
            # One clone per distinct pooling slot. The level modulates a quantile
            # that is already per column, so a level shared across columns would
            # let one entity's misses widen every other entity's interval; the
            # column axis is therefore never pooled. The step axis is, when
            # alpha_pooling="shared", in which case every step key points at the
            # same object rather than at horizon-many identical copies of it.
            adapter_columns = [c for c in conformity_scores.columns if c not in ("time", "step")]
            self.adapter_pooling_ = self.adapter.alpha_pooling
            steps = range(1, 1 + forecasting_horizon)

            def _new_adapter():
                """Return one freshly seeded adapter clone for a pooling slot."""
                return clone(self.adapter).fit(self.fit_coverage_rates_, symmetric=self._adapter_symmetric_)

            if self.adapter_pooling_ == "shared":
                shared = {column: _new_adapter() for column in adapter_columns}
                self.adapters_ = {f"step_{step}": dict(shared) for step in steps}
            else:
                self.adapters_ = {
                    f"step_{step}": {column: _new_adapter() for column in adapter_columns} for step in steps
                }

        return self

    def _check_replay_left_the_block_observed(self, y: pl.DataFrame, replay_path: str) -> None:
        """Fail here if the replay did not leave the point forecaster where fit ends.

        The replay consumes the calibration window to produce conformity scores, and
        whatever it does internally the forecaster it leaves behind must have observed
        that window. A path that does not is fitted but unpredictable: the first
        ``predict`` assembles a frame with a ``calibration_size``-sized hole in it.

        Without this check that defect surfaces several layers away, as an
        inconsistent-interval error raised from whichever transformer inverts first, and
        it describes the *input* as irregular when the input was regular and the gap was
        introduced by the fit. Attributing it took a cloud step log and a bisect. One
        timestamp comparison against a fit that costs minutes buys the attribution back.

        Raises
        ------
        RuntimeError
            If the wrapped point forecaster's observation state stops short of the data
            the forecaster was fitted on.
        """
        observed = getattr(self.point_forecaster_, "observed_time_", None)
        if observed is None:
            return

        # Under a panel strategy this is one timestamp per entity, not a scalar.
        stamps = observed.values() if isinstance(observed, dict) else [observed]
        expected = y["time"].max()
        behind = sorted({stamp for stamp in stamps if stamp is not None and stamp != expected})
        if not behind:
            return

        msg = (
            f"The '{replay_path}' calibration replay left the point forecaster at {behind} "
            f"but fit ended at {expected}, so it is fitted and not predictable: the next "
            f"predict would assemble a frame with a {expected - behind[0]} hole in it. "
            "The replay must leave the forecaster having observed the calibration block."
        )
        raise RuntimeError(msg)

    def _observe_conformity(
        self,
        y: pl.DataFrame,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> None:
        """Update similarity state and conformity scores for new observations.

        This conformity/similarity update is panel-agnostic: it operates on
        the full-frame ``y`` and delegates to ``point_forecaster_.predict``
        (which handles panel dispatch internally). It is therefore called
        unconditionally by ``observe()`` *before* the panel/standard branch,
        so the panel path does not silently skip it.

        Called *before* the point forecaster absorbs the new data so that
        ``predict()`` still reflects the pre-observe state.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.
        X_future : pl.DataFrame or None, default=None
            Known future features for the point forecaster's pre-observe
            ``predict`` call. Note that ``observe()`` does not forward its
            own ``X_future`` here, so this argument is ``None`` in practice;
            the caller's ``X_future`` is applied only by the subsequent
            ``point_forecaster_.observe()`` call.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts for the point forecaster's pre-observe
            ``predict`` call. As with ``X_future``, ``observe()`` does not
            forward its own ``X_forecast`` here, so this argument is ``None``
            in practice.

        """
        if not hasattr(self, "similarities_"):
            return

        # Generate predictions *before* the point forecaster is updated
        y_pred = self.point_forecaster_.predict(
            forecasting_horizon=self.fit_forecasting_horizon_,
            X_future=X_future,
            X_forecast=X_forecast,
        )

        for step in range(1, 1 + self.fit_forecasting_horizon_):
            key = f"step_{step}"
            similarity_step = self.similarities_[key]
            conformity_scorer_step = self.conformity_scorers_[key]

            y_pred_step = y_pred[step - 1 :: self.fit_forecasting_horizon_]
            # Drop vintage_time if present (conformity scorer expects time + value cols)
            y_pred_step = y_pred_step.drop("vintage_time", strict=False)

            conformity_scores_step = conformity_scorer_step.score(y, y_pred_step)

            # Observe the similarity on the scored subset, the same alignment ``fit``
            # performs. Scoring aligns y and y_pred by an inner join on time, so it
            # can return fewer rows than it was given, and none at all for a step
            # whose prediction reaches past the observed truth. Observing the full
            # y_pred_step then grew the similarity's state faster than the conformity
            # scores, and predict_interval later paired an N-weight array with fewer
            # than N scores. A step that scored nothing observes nothing, so the two
            # stay in step.
            if conformity_scores_step.height > 0:
                scored_times = conformity_scores_step.select("time")
                y_pred_for_sim = y_pred_step.join(scored_times, on="time", how="semi")
                similarity_step.observe(y=y, y_pred=y_pred_for_sim)

            conformity_scores_step = conformity_scores_step.with_columns(step=step)
            self.conformity_scores_ = pl.concat([self.conformity_scores_, conformity_scores_step])

    def _rewind_conformity(self, y: pl.DataFrame, X_actual: pl.DataFrame | None) -> None:
        """Rewind similarity state and conformity scores.

        Removes post-fit observations from similarity and conformity
        score state.  The number of rows removed equals the number
        added by ``_observe_conformity`` since fit (capped so we never
        remove fit-time data).

        Like ``_observe_conformity``, this is panel-agnostic and is called
        unconditionally by ``rewind()`` before the panel/standard branch, so
        the panel path does not skip the score rollback.

        Parameters
        ----------
        y : pl.DataFrame
            Target observations to rewind (used for row count).
        X_actual : pl.DataFrame or None
            Exogenous features (passed through to similarity rewind).

        """
        if not hasattr(self, "similarities_"):
            return

        n_rewind = len(y)

        for step in range(1, 1 + self.fit_forecasting_horizon_):
            key = f"step_{step}"
            fit_count = self._fit_score_counts_[key]
            step_scores = self.conformity_scores_.filter(pl.col("step") == step)
            n_post_fit = len(step_scores) - fit_count
            n_remove = min(n_rewind, n_post_fit)

            if n_remove > 0:
                # Remove the last n_remove rows for this step
                other_steps = self.conformity_scores_.filter(pl.col("step") != step)
                kept = step_scores.head(len(step_scores) - n_remove)
                self.conformity_scores_ = pl.concat([other_steps, kept])

                # Rewind similarity state
                similarity_step = self.similarities_[key]
                # Build a dummy y/y_pred with correct length for rewind
                y_rewind = y.head(n_remove)
                similarity_step.rewind(y=y_rewind, y_pred=y_rewind, X_actual=X_actual)

    @staticmethod
    def _adapter_thresholds(
        calib_col: np.ndarray,
        weights: np.ndarray,
        level: object,
        symmetric: bool,
    ) -> object:
        """Compute score-space interval thresholds at an effective level.

        Parameters
        ----------
        calib_col : np.ndarray
            Calibration conformity scores for one value column.
        weights : np.ndarray
            Per-calibration weights (uniform when no similarity is used).
        level : float or tuple of float
            Effective level: a single miscoverage ``alpha`` for symmetric
            scorers, or ``(beta_lower, beta_upper)`` per-tail miscoverage for
            asymmetric scorers.
        symmetric : bool
            Whether the conformity scorer is symmetric.

        Returns
        -------
        float or tuple of float
            For symmetric scorers, the half-width quantile ``q`` (a truth is
            covered when its score is ``<= q``). For asymmetric scorers, the
            ``(lower_q, upper_q)`` score bounds (a truth is covered when its
            score lies within them).

        """
        if symmetric:
            alpha_eff = float(level)  # ty: ignore[invalid-argument-type]
            return weighted_quantile(calib_col, alpha_eff, weights)

        beta_lower, beta_upper = level  # ty: ignore[not-iterable]
        lower_q = weighted_quantile(calib_col, 1.0 - float(beta_lower), weights)
        upper_q = weighted_quantile(calib_col, float(beta_upper), weights)
        return (lower_q, upper_q)

    def _adapter_step_errors(
        self,
        y: pl.DataFrame,
        scores_new: pl.DataFrame,
        calib: pl.DataFrame,
        y_pred_step: pl.DataFrame,
        step_key: str,
        coverage_rates: list[float],
        symmetric: bool,
        n_rows: int,
    ) -> list[dict[str, dict[float, object]]]:
        """Build one per-row, per-column miscoverage dict for a step.

        Rows this step could not score (its prediction reached past the
        observed truth) receive a zero-update sentinel so every step advances
        exactly ``n_rows`` times, keeping the rewind arithmetic exact.

        Parameters
        ----------
        y : pl.DataFrame
            The newly observed batch (defines row order and count).
        scores_new : pl.DataFrame
            New conformity scores for this step (``"time"`` plus value cols).
        calib : pl.DataFrame
            Pre-observe calibration scores for this step.
        y_pred_step : pl.DataFrame
            Pre-observe point predictions for this step.
        step_key : str
            The ``"step_k"`` key into ``similarities_``.
        coverage_rates : list of float
            Tracked coverage rates.
        symmetric : bool
            Whether the conformity scorer is symmetric.
        n_rows : int
            Number of rows in ``y``.

        Returns
        -------
        list of dict
            Length ``n_rows``; each entry maps a value column to a dict from
            coverage rate to that column's own miscoverage signal.

        """
        value_cols = [c for c in calib.columns if c != "time"]
        calib_cols = {c: calib[c].to_numpy().astype(np.float64) for c in value_cols}
        n_calib = calib.height
        levels = {column: adapter.predict() for column, adapter in self.adapters_[step_key].items()}

        def sentinel(coverage_rate: float) -> object:
            """Return the zero-update error for a row this step could not score."""
            alpha_target = 1.0 - coverage_rate
            return alpha_target if symmetric else (alpha_target / 2.0, alpha_target / 2.0)

        # Default every row to the zero-update sentinel, then fill scored rows.
        errors: list[dict[str, dict[float, object]]] = [
            {c: {cr: sentinel(cr) for cr in coverage_rates} for c in value_cols} for _ in range(n_rows)
        ]

        if scores_new.height == 0:
            return errors

        # Weights per scored row (uniform when no similarity is configured).
        scored_pred = y_pred_step.join(scores_new.select("time"), on="time", how="semi")
        if self.similarity is not None and hasattr(self, "similarities_"):
            weight_matrix = self.similarities_[step_key].predict(y_pred=scored_pred).astype(np.float64)
        else:
            # Uniform weights that reserve mass for the test point, mirroring
            # BaseSimilarity._reserve_mass. weighted_quantile does not
            # renormalize, so 1/(n+1) each is what reproduces the standard
            # ceil((n+1) * q) conformal order statistic.
            weight_matrix = np.full((scores_new.height, n_calib), 1.0 / (n_calib + 1), dtype=np.float64)

        row_index = {t: i for i, t in enumerate(y["time"].to_list())}
        scored_values = {c: scores_new[c].to_numpy().astype(np.float64) for c in value_cols}

        for j, t in enumerate(scores_new["time"].to_list()):
            i = row_index[t]
            weights = weight_matrix[j]
            for coverage_rate in coverage_rates:
                # Each column is judged against its own level and its own
                # threshold, and reports its own binary outcome. Averaging the
                # outcomes across columns, as this once did, fed a fraction into
                # a recursion the spec defines with an indicator, and coupled
                # every entity's level to every other entity's misses.
                for c in value_cols:
                    level = levels[c][coverage_rate]
                    score = scored_values[c][j]
                    if symmetric:
                        q = float(self._adapter_thresholds(calib_cols[c], weights, level, True))  # ty: ignore[invalid-argument-type]
                        errors[i][c][coverage_rate] = 1.0 if score > q else 0.0
                    else:
                        lower_q, upper_q = self._adapter_thresholds(calib_cols[c], weights, level, False)  # ty: ignore[not-iterable]
                        errors[i][c][coverage_rate] = (
                            1.0 if score < lower_q else 0.0,
                            1.0 if score > upper_q else 0.0,
                        )

        return errors

    @staticmethod
    def _pool_adapter_errors(
        per_step_errors: dict[str, list[dict[str, dict[float, object]]]],
        coverage_rates: list[float],
        symmetric: bool,
        n_rows: int,
    ) -> list[dict[str, dict[float, object]]]:
        """Average per-step miscoverage into one shared per-row trajectory.

        Used for ``alpha_pooling="shared"``: a single pooled update is fed to
        every step's adapter, so the per-step dict alone (which cannot see
        across steps) still yields one shared level.

        Pooling runs along the horizon-step axis only. Each value column is
        pooled with itself across steps and never with another column, so two
        columns' levels stay free to diverge under ``"shared"``.

        Parameters
        ----------
        per_step_errors : dict
            Maps each ``"step_k"`` key to its length-``n_rows`` error list,
            each entry keyed by value column then coverage rate.
        coverage_rates : list of float
            Tracked coverage rates.
        symmetric : bool
            Whether the conformity scorer is symmetric.
        n_rows : int
            Number of observed rows.

        Returns
        -------
        list of dict
            Length ``n_rows`` pooled error list, still keyed by value column.

        """
        keys = list(per_step_errors)
        columns = list(per_step_errors[keys[0]][0]) if keys and n_rows else []
        pooled: list[dict[str, dict[float, object]]] = []
        for i in range(n_rows):
            row: dict[str, dict[float, object]] = {}
            for column in columns:
                column_row: dict[float, object] = {}
                for coverage_rate in coverage_rates:
                    if symmetric:
                        column_row[coverage_rate] = float(
                            np.mean([per_step_errors[k][i][column][coverage_rate] for k in keys])
                        )
                    else:
                        lowers = [per_step_errors[k][i][column][coverage_rate][0] for k in keys]  # ty: ignore[not-subscriptable]
                        uppers = [per_step_errors[k][i][column][coverage_rate][1] for k in keys]  # ty: ignore[not-subscriptable]
                        column_row[coverage_rate] = (float(np.mean(lowers)), float(np.mean(uppers)))
                row[column] = column_row
            pooled.append(row)
        return pooled

    def _observe_adapter(self, y: pl.DataFrame) -> None:
        """Advance the per-step adapters from newly observed coverage.

        Runs before ``_observe_conformity`` (and before the point forecaster
        absorbs ``y``) so the effective level is updated against the
        calibration set and similarity state as they stood pre-observe.
        Panel-agnostic, like ``_observe_conformity``.

        Parameters
        ----------
        y : pl.DataFrame
            New target observations.

        """
        if not hasattr(self, "adapters_"):
            return

        fh = self.fit_forecasting_horizon_
        n_rows = len(y)
        symmetric = self._adapter_symmetric_
        coverage_rates = self.fit_coverage_rates_

        y_pred = self.point_forecaster_.predict(forecasting_horizon=fh)

        per_step_errors: dict[str, list[dict[str, dict[float, object]]]] = {}
        for step in range(1, 1 + fh):
            key = f"step_{step}"
            scorer = self.conformity_scorers_[key]
            y_pred_step = y_pred[step - 1 :: fh].drop("vintage_time", strict=False)
            scores_new = scorer.score(y, y_pred_step)
            calib = self.conformity_scores_.filter(pl.col("step") == step).drop("step")
            per_step_errors[key] = self._adapter_step_errors(
                y=y,
                scores_new=scores_new,
                calib=calib,
                y_pred_step=y_pred_step,
                step_key=key,
                coverage_rates=coverage_rates,
                symmetric=symmetric,
                n_rows=n_rows,
            )

        if self.adapter_pooling_ == "shared":
            pooled = self._pool_adapter_errors(per_step_errors, coverage_rates, symmetric, n_rows)
            # One object per column under sharing, so drive each column once from
            # the pooled trajectory. Walking the step keys would advance the same
            # object once per horizon step, moving the level fh times too far per
            # observed row.
            seen: set[int] = set()
            for step_adapters in self.adapters_.values():
                for column, adapter in step_adapters.items():
                    if id(adapter) in seen:
                        continue
                    seen.add(id(adapter))
                    adapter.observe([row[column] for row in pooled])
        else:
            for key, errors in per_step_errors.items():
                for column, adapter in self.adapters_[key].items():
                    adapter.observe([row[column] for row in errors])

    def _distinct_adapters(self) -> list:
        """Return each adapter object once, however many keys point at it.

        Under ``alpha_pooling="shared"`` one object serves every horizon step of
        a column, so the nested mapping holds ``fh`` references to it. Anything
        that advances or rolls back state must walk objects rather than
        references, or a single row moves the level ``fh`` times.

        Returns
        -------
        list
            The distinct adapter objects, in first-seen order.

        """
        seen: set[int] = set()
        distinct = []
        for step_adapters in self.adapters_.values():
            for adapter in step_adapters.values():
                if id(adapter) not in seen:
                    seen.add(id(adapter))
                    distinct.append(adapter)
        return distinct

    def _rewind_adapter(self, y: pl.DataFrame) -> None:
        """Roll each per-step adapter back by ``len(y)`` observations.

        Parameters
        ----------
        y : pl.DataFrame
            Target observations to rewind (used only for the row count).

        """
        if not hasattr(self, "adapters_"):
            return

        n_rewind = len(y)
        # Every distinct clone advanced by the same row count, so every distinct
        # clone rolls back by it too. Iterating the mapping rather than the
        # objects would roll a shared adapter back once per step key, and the
        # overshoot would not cancel the matching one in observe because rewind
        # floors at the fit-time seed while observe has no ceiling.
        for adapter in self._distinct_adapters():
            adapter.rewind(n_rewind)

    def observe(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> "SplitConformalForecaster":
        """Observe new data and update conformity scores and the point forecaster.

        Updates conformity scores and similarity weights with the new
        observations, then advances the wrapped point forecaster's
        observation buffers without refitting.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Passed directly to the wrapped
            ``point_forecaster_``; any feature transformation is the
            responsibility of that inner estimator. If ``None``, only
            target-derived features are used.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns. Passed to the wrapped ``point_forecaster_``; any
            feature transformation is the responsibility of that inner
            estimator, reachable as ``point_forecaster__forecast_transformer``.

        Returns
        -------
        self
            The forecaster with updated observation buffers.

        """
        check_is_fitted(
            self,
            ["point_forecaster_", "local_y_schema_", "local_X_actual_schema_", "shared_X_actual_schema_", "groups_"],
        )

        y, X_actual, groups = validate_forecaster_data(self, y, X_actual, reset=False, groups=groups)

        # Update similarity / conformity scores *before* the point forecaster
        # absorbs the new data so we can still call predict() to obtain the
        # prediction-vs-actual residual. This update is panel-agnostic, so it
        # runs unconditionally ahead of the panel/standard branch; routing it
        # through the standard branch alone would silently skip it on panel
        # data.
        #
        # The adapter runs *before* _observe_conformity so its realized-coverage
        # check sees the calibration scores and similarity weights as they stood
        # pre-observe, before this batch is appended.
        self._observe_adapter(y)
        self._observe_conformity(y)

        if self.groups_ is None:
            super()._observe_standard(y, X_actual=X_actual)
        else:
            BasePanelForecaster._observe_panel(self, y, X_actual=X_actual, groups=groups)

        self.point_forecaster_.observe(y=y, X_actual=X_actual, groups=groups, X_future=X_future, X_forecast=X_forecast)
        self.observed_time_ = self.point_forecaster_.observed_time_
        return self

    def rewind(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> "SplitConformalForecaster":
        """Rewind conformity scores, similarity state, and the point forecaster.

        Removes the most recently observed conformity scores and similarity
        state, then rewinds the wrapped point forecaster's observation
        buffers, mirroring the order used by ``observe()``.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations to restore the observation
            state to. Must align with ``y``.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.

        Returns
        -------
        self
            The forecaster with rewound observation buffers.

        """
        check_is_fitted(
            self,
            ["point_forecaster_", "local_y_schema_", "local_X_actual_schema_", "shared_X_actual_schema_", "groups_"],
        )

        y, X_actual, groups = validate_forecaster_data(self, y, X_actual, reset=False, groups=groups)

        # Rewind conformity / similarity state *before* the point forecaster
        # rolls back, mirroring the order used by observe() so both methods
        # maintain the same state invariant. As in observe(), this rollback is
        # panel-agnostic and runs unconditionally ahead of the branch.
        self._rewind_adapter(y)
        self._rewind_conformity(y, X_actual=X_actual)

        if self.groups_ is None:
            super()._rewind_standard(y, X_actual=X_actual)
        else:
            BasePanelForecaster._rewind_panel(self, y, X_actual=X_actual, groups=groups)

        self.point_forecaster_.rewind(y=y, X_actual=X_actual, groups=groups, X_future=X_future, X_forecast=X_forecast)
        self.observed_time_ = self.point_forecaster_.observed_time_
        return self

    def predict(
        self,
        forecasting_horizon: StrictInt | None = None,
        groups: list[str] | None = None,
        predict_transformed: bool = False,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate point forecasts.

        Delegates to the wrapped point forecaster.

        Parameters
        ----------
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future.  If ``None``,
            uses the horizon specified at fit time.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.
        predict_transformed : bool, default=False
            If ``True``, return predictions in the transformed space without
            applying inverse target transformation.
        X_future : pl.DataFrame or None, default=None
            Known future features override. Re-derives step columns
            without mutating forecaster state.
        X_forecast : pl.DataFrame or None, default=None
            External forecast override with ``"vintage_time"`` and
            ``"time"`` columns. Re-derives step columns without mutating
            forecaster state.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Point predictions with ``"vintage_time"``, ``"time"``, and one
            column per target variable.

        """
        check_is_fitted(
            self,
            ["local_y_schema_", "local_X_actual_schema_", "shared_X_actual_schema_", "groups_"],
        )

        _, _, groups = validate_forecaster_data(
            self,
            y=None,
            X_actual=None,
            reset=False,
            groups=groups,
        )

        return self.point_forecaster_.predict(
            forecasting_horizon=forecasting_horizon,
            groups=groups,
            predict_transformed=predict_transformed,
            X_future=X_future,
            X_forecast=X_forecast,
        )

    def observe_predict(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        groups: list[str] | None = None,
        stride: StrictInt | None = None,
        predict_transformed: bool = False,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Alternate recursive observe and predict.

        Produces a rolling sequence of predictions, observing ``stride``
        rows between each. Returns the concatenation of the initial
        prediction and one prediction per stride step.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Sliced and observed incrementally at each step
            of the rolling loop.
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future.  If ``None``,
            uses the horizon specified at fit time.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.
        stride : int or None, default=None
            Step size for rolling update-predict.  If ``None``, defaults to
            the forecasting horizon used at fit time
            (``fit_forecasting_horizon_``).
        predict_transformed : bool, default=False
            If ``True``, return predictions in the transformed space without
            applying inverse target transformation.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Point predictions with ``"vintage_time"``, ``"time"``, and one
            column per target variable.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the forecaster has not been fitted yet.
        ValueError
            If ``y`` / ``X`` have invalid structure or ``groups``
            contains names not seen during fit.

        """
        check_is_fitted(
            self,
            ["local_y_schema_", "local_X_actual_schema_", "shared_X_actual_schema_", "groups_"],
        )

        y, X_actual, groups = validate_forecaster_data(
            self,
            y=y,
            X_actual=X_actual,
            reset=False,
            groups=groups,
            X_future=X_future,
            X_forecast=X_forecast,
        )

        forecasting_horizon, _ = self._validate_predict_params(forecasting_horizon)
        if stride is None:
            stride = self.fit_forecasting_horizon_

        return self._observe_predict_loop(
            predict_fn=self.predict,
            y=y,
            X_actual=X_actual,
            X_future=X_future,
            X_forecast=X_forecast,
            groups=groups,
            stride=stride,
            observe_fn=self.observe,
            forecasting_horizon=forecasting_horizon,
            predict_transformed=predict_transformed,
            **params,
        )

    def observe_predict_interval(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt | None = None,
        coverage_rates: list[float] | None = None,
        strategy: Literal["mean", "median", "point"] | None = "point",
        groups: list[str] | None = None,
        stride: StrictInt | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Alternate recursive observe and predict_interval.

        Equivalent to calling ``observe(y, X_actual)`` then
        ``predict_interval()``.  Returns interval predictions.

        Overrides the parent implementation to pass ``observe_fn`` so that
        the wrapped ``point_forecaster_`` observation state is correctly
        advanced at each stride step.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more numeric value columns.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Sliced and observed incrementally at each step
            of the rolling loop.
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future.  If ``None``,
            uses the horizon specified at fit time.
        coverage_rates : list of float or None, default=None
            Coverage levels for prediction intervals (e.g., ``[0.9, 0.95]``
            for 90 % and 95 % intervals).  If ``None``, defaults to the rates
            used at fit time.
        strategy : {"mean", "median", "point"} or None, default=None
            Strategy for deriving point predictions from prediction intervals
            during recursive multi-step forecasting:

            - ``"mean"``: use the mean of the interval bounds
            - ``"median"``: use the median of the interval bounds
            - ``"point"``: use the point forecast directly (if available)

            If ``None``, defaults to ``"mean"``.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.
        stride : int or None, default=None
            Step size for rolling update-predict.  If ``None``, defaults to
            the forecasting horizon used at fit time
            (``fit_forecasting_horizon_``).
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Interval predictions with ``"vintage_time"``, ``"time"``, and
            lower/upper bound columns for each target at each coverage rate.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the forecaster has not been fitted yet.
        ValueError
            If ``y`` / ``X_actual`` have invalid structure, ``coverage_rates`` not in
            [0, 1], or ``groups`` contains names not seen during fit.

        """
        self._validate_strategy(strategy)

        check_is_fitted(
            self,
            ["local_y_schema_", "local_X_actual_schema_", "shared_X_actual_schema_", "groups_"],
        )

        y, X_actual, groups = validate_forecaster_data(
            self,
            y=y,
            X_actual=X_actual,
            reset=False,
            groups=groups,
            X_future=X_future,
            X_forecast=X_forecast,
        )

        forecasting_horizon, _ = self._validate_predict_params(forecasting_horizon, coverage_rates)

        if stride is None:
            stride = self.fit_forecasting_horizon_

        return self._observe_predict_loop(
            predict_fn=self.predict_interval,
            y=y,
            X_actual=X_actual,
            X_future=X_future,
            X_forecast=X_forecast,
            groups=groups,
            stride=stride,
            observe_fn=self.observe,
            forecasting_horizon=forecasting_horizon,
            coverage_rates=coverage_rates,
            strategy=strategy,
            **params,
        )

    def _weighted_inverse_score(
        self,
        y_pred_step: pl.DataFrame,
        conformity_scores_step: pl.DataFrame,
        coverage_rate: float,
        weights: np.ndarray,
        conformity_scorer_step: BaseConformityScorer,
        step: int = 1,
    ) -> pl.DataFrame:
        """Compute prediction intervals using similarity-weighted quantiles.

        Parameters
        ----------
        y_pred_step : pl.DataFrame
            Point predictions for one step (single row) with ``"time"``
            and value columns.
        conformity_scores_step : pl.DataFrame
            Conformity scores for this step with ``"time"`` and value
            columns.
        coverage_rate : float
            Target coverage probability.
        weights : np.ndarray
            Similarity weights of shape ``(n_calibration,)``.
        conformity_scorer_step : BaseConformityScorer
            Fitted conformity scorer (used for type detection and
            formatting).

        Returns
        -------
        pl.DataFrame
            Interval columns (no ``"time"`` column).

        """
        value_cols = [c for c in conformity_scores_step.columns if c != "time"]
        scores_no_time = conformity_scores_step.drop("time", strict=False)
        y_pred_values = y_pred_step.drop("time")

        # One weight row serves every value column, so the collapse check runs
        # once here rather than inside the per-column loop below. Checking it
        # per column would emit the same warning n times for one event.
        warn_if_weights_collapsed(weights, step, coverage_rate)

        # Read scorer characteristics from tags instead of checking types
        tags = conformity_scorer_step.__sklearn_tags__()
        assert tags.scorer_tags is not None
        symmetric = tags.scorer_tags.symmetric
        multiplicative = tags.scorer_tags.multiplicative
        epsilon = conformity_scorer_step.get_params().get("epsilon", 0.0)

        lower_data: dict[str, list[float]] = {}
        upper_data: dict[str, list[float]] = {}

        # Under pooling one quantile is drawn from every column's scores, with each
        # column at a calibration time carrying that time's affinity. The tiling has
        # to run on the raw affinities, which pooled_weights handles.
        pooled = self.calibration_strategy == "global"
        empty = np.empty(0, dtype=np.float64)
        pooled_scores = scores_no_time.to_numpy().astype(np.float64).reshape(-1) if pooled else empty
        pooled_w = pooled_weights(weights, len(value_cols)) if pooled else empty
        column_scales = getattr(conformity_scorer_step, "column_scales_", None)

        for col in value_cols:
            scores_col = pooled_scores if pooled else scores_no_time[col].to_numpy().astype(np.float64)
            col_weights = pooled_w if pooled else weights
            pred_val = float(y_pred_values[col][0])
            scale = (pred_val + epsilon) if multiplicative else 1.0
            if pooled and column_scales is not None:
                # A dispersion-normalized scorer rebuilds the bound with the
                # column's own fitted scale, which keeps pooled widths per column.
                scale = column_scales[col]

            if symmetric:
                q = weighted_quantile(scores_col, 1.0 - coverage_rate, col_weights)
                lower_data[col] = [pred_val - q * scale]
                upper_data[col] = [pred_val + q * scale]
            else:
                alpha = 1.0 - coverage_rate
                lower_q = weighted_quantile(scores_col, 1.0 - alpha / 2.0, col_weights)
                upper_q = weighted_quantile(scores_col, alpha / 2.0, col_weights)
                lower_data[col] = [pred_val + lower_q * scale]
                upper_data[col] = [pred_val + upper_q * scale]

        lower_bound = pl.DataFrame(lower_data)
        upper_bound = pl.DataFrame(upper_data)
        return conformity_scorer_step._format_y_pred_interval(lower_bound, upper_bound, coverage_rate)

    def _adapter_inverse_score(
        self,
        y_pred_step: pl.DataFrame,
        conformity_scores_step: pl.DataFrame,
        coverage_rate: float,
        levels: dict[str, object],
        weights: np.ndarray,
        conformity_scorer_step: BaseConformityScorer,
        step: int = 1,
    ) -> pl.DataFrame:
        """Build an interval at the adapter's effective level.

        Constructs the interval from the effective miscoverage level the
        adapter currently tracks, then labels the output columns with the
        *nominal* ``coverage_rate`` so callers still read a "90% interval"
        even though its width was adapted. Uses similarity weights when
        supplied and uniform weights otherwise, mirroring
        ``_weighted_inverse_score``.

        Parameters
        ----------
        y_pred_step : pl.DataFrame
            Point predictions for one step (single row) with ``"time"``.
        conformity_scores_step : pl.DataFrame
            Conformity scores for this step with ``"time"``.
        coverage_rate : float
            Nominal coverage rate used only to label the output columns.
        levels : dict of str to (float or tuple of float)
            Effective level per value column: ``alpha`` for symmetric
            scorers, ``(beta_lower, beta_upper)`` for asymmetric ones.
        weights : np.ndarray
            Per-calibration weights (similarity weights, or uniform).
        conformity_scorer_step : BaseConformityScorer
            Fitted conformity scorer (for tags and formatting).
        step : int, default=1
            Horizon step, used only to name the step in a collapsed-weight
            warning.

        Returns
        -------
        pl.DataFrame
            Interval columns (no ``"time"`` column), labeled with the
            nominal ``coverage_rate``.

        """
        value_cols = [c for c in conformity_scores_step.columns if c != "time"]
        scores_no_time = conformity_scores_step.drop("time", strict=False)
        y_pred_values = y_pred_step.drop("time")

        # Once per weight row, for the same reason as in _weighted_inverse_score.
        warn_if_weights_collapsed(weights, step, coverage_rate)

        tags = conformity_scorer_step.__sklearn_tags__()
        assert tags.scorer_tags is not None
        symmetric = tags.scorer_tags.symmetric
        multiplicative = tags.scorer_tags.multiplicative
        epsilon = conformity_scorer_step.get_params().get("epsilon", 0.0)

        lower_data: dict[str, list[float]] = {}
        upper_data: dict[str, list[float]] = {}

        for col in value_cols:
            scores_col = scores_no_time[col].to_numpy().astype(np.float64)
            pred_val = float(y_pred_values[col][0])
            scale = (pred_val + epsilon) if multiplicative else 1.0

            thresholds = self._adapter_thresholds(scores_col, weights, levels[col], symmetric)
            if symmetric:
                q = float(thresholds)  # ty: ignore[invalid-argument-type]
                lower_data[col] = [pred_val - q * scale]
                upper_data[col] = [pred_val + q * scale]
            else:
                lower_q, upper_q = thresholds  # ty: ignore[not-iterable]
                lower_data[col] = [pred_val + lower_q * scale]
                upper_data[col] = [pred_val + upper_q * scale]

        lower_bound = pl.DataFrame(lower_data)
        upper_bound = pl.DataFrame(upper_data)
        return conformity_scorer_step._format_y_pred_interval(lower_bound, upper_bound, coverage_rate)

    @staticmethod
    def _validate_strategy(strategy: str | None) -> None:
        """Reject a recursion strategy this forecaster cannot honour.

        ``strategy`` selects how a recursive step derives its next observation
        from the previous step's bounds. This forecaster has no such step, so
        only ``"point"`` (and ``None``, meaning the default) describes it.
        Silently accepting ``"mean"`` or ``"median"`` would let a caller ask for
        bound-midpoint recursion, receive point-based behaviour, and never learn
        the difference.

        Parameters
        ----------
        strategy : str or None
            The requested strategy.

        Raises
        ------
        ValueError
            If ``strategy`` is neither ``None`` nor ``"point"``.

        """
        if strategy not in (None, "point"):
            raise ValueError(
                f"SplitConformalForecaster always recurses on the point forecast, so "
                f"strategy={strategy!r} cannot be honoured. The wrapped point forecaster "
                f"produces the whole horizon in one call and the conformal bands are "
                f"derived from it, so bound midpoints are never fed back. Pass "
                f"strategy='point' (the default) or omit it."
            )

    def predict_interval(  # ty: ignore[invalid-method-override]
        self,
        forecasting_horizon: StrictInt | None = None,
        coverage_rates: list[float] | None = None,
        strategy: Literal["mean", "median", "point"] | None = "point",
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate interval forecasts.

        Uses calibrated conformity scores to construct prediction intervals
        around point forecasts.

        Parameters
        ----------
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast into the future.  If ``None``,
            uses the horizon specified at fit time.
        coverage_rates : list of float or None, default=None
            Coverage levels for prediction intervals (e.g., ``[0.9, 0.95]``
            for 90 % and 95 % intervals).  If ``None``, defaults to the rates
            used at fit time.
        strategy : {"point"} or None, default="point"
            Retained for interface parity with
            [`BaseIntervalForecaster`][yohou.interval.base.BaseIntervalForecaster],
            where it selects how a recursive step derives its next observation.
            This forecaster has no such step: the wrapped ``point_forecaster_``
            produces the whole horizon in one call and the bands are draped over
            the result, so any multi-step recursion happens inside that
            forecaster and always on point values, never on bound midpoints.
            ``"point"`` therefore describes what this class does, and is the
            default. ``"mean"`` and ``"median"`` cannot be honoured and raise.
        groups : list of str or None, default=None
            Panel group prefixes to operate on.  If ``None``, all groups
            are used.  Ignored when the forecaster was not fitted on panel
            data.
        X_future : pl.DataFrame or None, default=None
            Known future features override. Re-derives step columns
            without mutating forecaster state.
        X_forecast : pl.DataFrame or None, default=None
            External forecast override with ``"vintage_time"`` and
            ``"time"`` columns. Re-derives step columns without mutating
            forecaster state.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Interval predictions with ``"vintage_time"``, ``"time"``, a bare
            point-forecast column per target, and lower/upper bound columns for
            each target at each coverage rate. The bare column is the wrapped
            point forecaster's prediction, the value the bands are centred on.
            Reading it is exact for any conformity scorer, whereas re-deriving
            the point forecast from the bounds only recovers it when the scorer
            is symmetric.

        """
        self._validate_strategy(strategy)

        check_is_fitted(
            self,
            ["local_y_schema_", "local_X_actual_schema_", "shared_X_actual_schema_", "groups_"],
        )

        _, _, groups = validate_forecaster_data(
            self,
            y=None,
            X_actual=None,
            reset=False,
            groups=groups,
        )

        forecasting_horizon, coverage_rates = self._validate_predict_params(forecasting_horizon, coverage_rates)

        y_pred_full = self.point_forecaster_.predict(X_future=X_future, X_forecast=X_forecast)
        has_vintage_time = "vintage_time" in y_pred_full.columns
        if has_vintage_time:
            vintage_time_col = y_pred_full["vintage_time"]
            y_pred = y_pred_full.drop("vintage_time")
        else:
            y_pred = y_pred_full

        # Extract time for later reconstruction
        y_pred_time = y_pred.select("time")
        y_pred_values = y_pred.drop("time")

        # An adapter only adapts the coverage rates it seeded at fit time.
        # Rates it never tracked fall back to the static level, with a warning
        # so the fallback is never silent.
        tracked_rates: set[float] = set()
        if hasattr(self, "adapters_"):
            # Every clone is seeded with the same rates at fit, so any one of
            # them answers which rates are tracked.
            any_adapter = next(iter(self.adapters_["step_1"].values()))
            tracked_rates = set(any_adapter.predict().keys())
            for coverage_rate in coverage_rates:
                if coverage_rate not in tracked_rates:
                    warnings.warn(
                        f"Coverage rate {coverage_rate} was not tracked by the adaptive conformal "
                        f"adapter (tracked rates: {sorted(tracked_rates)}); falling back to the static "
                        f"calibrated level for it.",
                        UserWarning,
                        stacklevel=2,
                    )

        y_pred_intervals_list: list[pl.DataFrame] = []
        for step in range(1, 1 + forecasting_horizon):
            # Get step predictions
            y_pred_step_values = y_pred_values.slice(step - 1, 1)
            y_pred_step_time = y_pred_time.slice(step - 1, 1)

            # Combine time and values for inverse_score (conformity scorers need time)
            y_pred_step = y_pred_step_time.hstack(y_pred_step_values)

            conformity_scorer_step = self.conformity_scorers_[f"step_{step}"]
            conformity_scores_step = self.conformity_scores_.filter(pl.col("step") == step).drop("step")

            # One scorer per step, so its symmetry is fixed for every rate below.
            step_tags = conformity_scorer_step.__sklearn_tags__()
            assert step_tags.scorer_tags is not None
            step_symmetric = step_tags.scorer_tags.symmetric

            rate_parts: list[pl.DataFrame] = []
            for coverage_rate in coverage_rates:
                # Checked once per (step, rate), before any path computes a
                # quantile, so it covers the static, weighted and adapted paths
                # alike rather than being repeated in each.
                # Under pooling the quantile draws from every column's scores, so
                # the resolution ceiling is the pooled count. Counting per column
                # would report a rate unreachable exactly when pooling made it
                # reachable, which is the point of the mode.
                n_value_columns = len([c for c in conformity_scores_step.columns if c != "time"])
                available = conformity_scores_step.height * (
                    n_value_columns if self.calibration_strategy == "global" else 1
                )
                warn_if_calibration_too_small(available, coverage_rate, step_symmetric, step)

                adapter_active = hasattr(self, "adapters_") and coverage_rate in tracked_rates
                # Empty unless the adapter is active, in which case it carries one
                # effective level per value column.
                effective_levels: dict[str, object] = (
                    {
                        column: adapter.predict()[coverage_rate]
                        for column, adapter in self.adapters_[f"step_{step}"].items()
                    }
                    if adapter_active
                    else {}
                )

                if self.similarity is not None and hasattr(self, "similarities_"):
                    similarity_step = self.similarities_[f"step_{step}"]
                    weights_array = similarity_step.predict(y_pred=y_pred_step)
                    step_weights = weights_array[0].astype(np.float64)

                    if adapter_active:
                        y_pred_interval_rate_step = self._adapter_inverse_score(
                            y_pred_step=y_pred_step,
                            conformity_scores_step=conformity_scores_step,
                            coverage_rate=coverage_rate,
                            levels=effective_levels,
                            weights=step_weights,
                            conformity_scorer_step=conformity_scorer_step,
                            step=step,
                        )
                    else:
                        y_pred_interval_rate_step = self._weighted_inverse_score(
                            y_pred_step=y_pred_step,
                            conformity_scores_step=conformity_scores_step,
                            coverage_rate=coverage_rate,
                            weights=step_weights,
                            conformity_scorer_step=conformity_scorer_step,
                            step=step,
                        )
                elif adapter_active:
                    n_calib = conformity_scores_step.height
                    # Reserved-mass uniform weights; see _adapter_step_errors.
                    uniform_weights = np.full(n_calib, 1.0 / (n_calib + 1), dtype=np.float64)
                    y_pred_interval_rate_step = self._adapter_inverse_score(
                        y_pred_step=y_pred_step,
                        conformity_scores_step=conformity_scores_step,
                        coverage_rate=coverage_rate,
                        levels=effective_levels,
                        weights=uniform_weights,
                        conformity_scorer_step=conformity_scorer_step,
                        step=step,
                    )
                else:
                    y_pred_interval_rate_step = conformity_scorer_step.inverse_score(
                        y_pred=y_pred_step,
                        conformity_scores=conformity_scores_step,
                        coverage_rate=coverage_rate,
                        pooled=self.calibration_strategy == "global",
                    ).drop("time")

                rate_parts.append(y_pred_interval_rate_step)

            # Add time column once at the front, then the bare point columns.
            # ``y_pred_step_values`` is the wrapped point forecaster's own output
            # for this step, already sliced above for conformity scoring, so
            # emitting it costs no second ``predict`` call. It lets a caller read
            # the point forecast the bands were built around, rather than
            # re-deriving it from the bounds, which only recovers the point
            # forecast when the conformity scorer is symmetric.
            y_pred_intervals_step = pl.concat([y_pred_step_time, y_pred_step_values, *rate_parts], how="horizontal")

            y_pred_intervals_list.append(y_pred_intervals_step)

        y_pred_intervals = pl.concat(y_pred_intervals_list)

        if has_vintage_time:
            return y_pred_intervals.insert_column(0, vintage_time_col.head(len(y_pred_intervals)))
        return y_pred_intervals
