"""Implementation of conformal forecasters."""

import math
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
from yohou.utils._compat import Interval, _fit_context

from .base import BaseConformalAdapter, BaseIntervalForecaster, BaseSimilarity
from .utils import weighted_quantile

__all__ = ["SplitConformalForecaster"]

# Fewest conformity scores the deepest horizon step may collect under a strided
# calibration replay. Below this the tail quantiles of the score distribution
# are estimated from a handful of samples, so intervals get wide and unstable
# in a way that surfaces weeks later as bad coverage rather than at fit. The
# value is a judgment call: 30 keeps the 90th percentile estimate inside the
# sample rather than at its extremes, while staying reachable with a few weeks
# of daily origins.
MIN_STRIDED_SCORES_PER_STEP = 30


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
        Number of observations to use for calibration. Always counted in
        rows, whatever ``calibration_stride`` is set to.
    calibration_stride : int >= 1 or None, default=None
        Rows observed between successive calibration replay origins. ``None``
        keeps the stride-1 replay: one conformity score per calibration row
        per horizon step. When set, the replay still observes every
        calibration row but scores only origins ``calibration_stride`` rows
        apart, ending on the last calibration row, so the scores match a
        production cadence of one forecast per stride. ``calibration_size``
        must be a multiple of the stride. Step ``h`` then collects
        ``calibration_size / calibration_stride - ceil(h / calibration_stride) + 1``
        scores; fit refuses a configuration whose deepest step falls below
        ``MIN_STRIDED_SCORES_PER_STEP``.
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
        How to handle panel data. See `BaseForecaster` for details.

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
        "calibration_stride": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "conformity_scorer": [BaseConformityScorer],
        "similarity": [BaseSimilarity, None],
        "adapter": [BaseConformalAdapter, None],
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
        calibration_stride: StrictInt | None = None,
    ):
        BaseIntervalForecaster.__init__(self, panel_strategy=panel_strategy)

        self.point_forecaster = point_forecaster
        self.conformity_scorer = conformity_scorer
        self.similarity = similarity
        self.adapter = adapter
        self.calibration_size = calibration_size
        self.calibration_stride = calibration_stride

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

        self._validate_calibration_stride(forecasting_horizon)

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

        # At the default calibration_stride=None the replay runs at stride 1: each row
        # of y_calib produces one prediction window of length forecasting_horizon,
        # yielding calibration_size - step + 1 conformity scores per horizon step
        # instead of the ~2-3 that stride=forecasting_horizon would leave. More
        # calibration scores per step gives quantiles that are stable and
        # well-separated.
        #
        # The stride is not a cost knob: each origin yields exactly one score per
        # horizon step, so raising it buys speed only by discarding scores one for
        # one. A configured calibration_stride is a *cadence* knob instead: it
        # restricts scores to origins matching a production decision cadence (one
        # forecast per day), and _validate_calibration_stride has already refused a
        # configuration whose score budget could not hold.
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
        replay_stride = self.calibration_stride or 1

        # Three paths, most to least aggressive. The bulk one additionally skips the
        # rolling observe, which is sound only where every transformer declares
        # `batch_invariant`; that path moves rolling-statistic columns by about one ULP,
        # so it is taken only on a declared stack. Anything undeclared keeps a
        # bit-identical path, which is what makes a missing declaration cost speed alone.
        # Recorded as a fitted attribute, not just chosen. Which path ran decides what
        # state the forecaster is left in, so a state defect is only attributable to a
        # path if the fitted object can say which one it took. Establishing that for the
        # 2026-08-08 regression cost a bisect across two submodule bumps.
        # The bulk path replays every origin by construction, so a configured
        # calibration_stride routes to the batched or rolling path, both of which
        # take the stride and still observe every calibration row.
        if direct and bulk is not None and replay_stride == 1 and point._chains_are_batch_invariant():
            replay_path = "bulk"
        elif direct and batched is not None:
            replay_path = "batched"
        else:
            replay_path = "rolling"
        self.replay_path_ = replay_path

        # The asserts restate what the path selection above guarantees; the string
        # dispatch hides that narrowing from the type checker.
        if replay_path == "bulk":
            assert bulk is not None
            y_pred_calib = bulk(
                y=y_calib,
                X_actual=X_actual_calib,
                groups=point.groups_ or [],
                X_future=X_future,
                X_forecast=X_forecast,
                predict_transformed=False,
            )
        elif replay_path == "batched":
            assert batched is not None
            y_pred_calib = batched(
                y=y_calib,
                X_actual=X_actual_calib,
                groups=point.groups_ or [],
                stride=replay_stride,
                X_future=X_future,
                X_forecast=X_forecast,
                predict_transformed=False,
            )
        else:
            y_pred_calib = self.point_forecaster_.observe_predict(
                y=y_calib,
                X_actual=X_actual_calib,
                forecasting_horizon=None,
                stride=replay_stride,
                predict_transformed=False,
                X_future=X_future,
                X_forecast=X_forecast,
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
            self.adapters_ = {}
            for step in range(1, 1 + forecasting_horizon):
                key = f"step_{step}"
                self.adapters_[key] = clone(self.adapter).fit(
                    self.fit_coverage_rates_, symmetric=self._adapter_symmetric_
                )

        return self

    def _validate_calibration_stride(self, forecasting_horizon: int) -> None:
        """Reject a strided calibration whose score budget cannot hold.

        Two failure modes are caught at fit time rather than surfacing as bad
        intervals later. A ``calibration_size`` that is not a multiple of the
        stride would leave the last replay origin short of the block end, so
        the origins would drift off the production cadence the stride exists
        to match. And a block too small for the stride leaves the deepest
        horizon step with too few conformity scores: step ``h`` collects
        ``C/k - ceil(h/k) + 1`` scores (``C`` rows, stride ``k``), so the
        deepest step is always the binding one.

        Raises
        ------
        ValueError
            If ``calibration_size`` is not a multiple of ``calibration_stride``,
            or the deepest step's score count falls below
            ``MIN_STRIDED_SCORES_PER_STEP``.
        """
        if self.calibration_stride is None:
            return
        k = self.calibration_stride
        c = self.calibration_size

        if c % k != 0:
            raise ValueError(
                f"calibration_size={c} is not a multiple of calibration_stride={k}. "
                f"The strided replay tail-anchors its origins on the last calibration "
                f"row, so the block must hold a whole number of strides."
            )

        depth = math.ceil(forecasting_horizon / k)
        worst = c // k - depth + 1
        if worst < MIN_STRIDED_SCORES_PER_STEP:
            needed = (MIN_STRIDED_SCORES_PER_STEP - 1 + depth) * k
            binding_lo = (depth - 1) * k + 1
            raise ValueError(
                f"calibration_stride={k} with calibration_size={c} leaves "
                f"{worst} conformity scores at the binding deep steps "
                f"({binding_lo} to {forecasting_horizon}), below the floor of "
                f"{MIN_STRIDED_SCORES_PER_STEP}. The smallest calibration_size "
                f"whose worst step clears the floor is {needed}."
            )

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
    ) -> list[dict[float, object]]:
        """Build one per-row miscoverage dict per observed row for a step.

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
            Length ``n_rows``; each dict maps a coverage rate to its
            miscoverage signal.

        """
        value_cols = [c for c in calib.columns if c != "time"]
        calib_cols = {c: calib[c].to_numpy().astype(np.float64) for c in value_cols}
        n_calib = calib.height
        levels = self.adapters_[step_key].predict()

        def sentinel(coverage_rate: float) -> object:
            """Return the zero-update error for a row this step could not score."""
            alpha_target = 1.0 - coverage_rate
            return alpha_target if symmetric else (alpha_target / 2.0, alpha_target / 2.0)

        # Default every row to the zero-update sentinel, then fill scored rows.
        errors: list[dict[float, object]] = [{cr: sentinel(cr) for cr in coverage_rates} for _ in range(n_rows)]

        if scores_new.height == 0:
            return errors

        # Weights per scored row (uniform when no similarity is configured).
        scored_pred = y_pred_step.join(scores_new.select("time"), on="time", how="semi")
        if self.similarity is not None and hasattr(self, "similarities_"):
            weight_matrix = self.similarities_[step_key].predict(y_pred=scored_pred).astype(np.float64)
        else:
            weight_matrix = np.ones((scores_new.height, n_calib), dtype=np.float64)

        row_index = {t: i for i, t in enumerate(y["time"].to_list())}
        scored_values = {c: scores_new[c].to_numpy().astype(np.float64) for c in value_cols}

        for j, t in enumerate(scores_new["time"].to_list()):
            i = row_index[t]
            weights = weight_matrix[j]
            for coverage_rate in coverage_rates:
                level = levels[coverage_rate]
                if symmetric:
                    misses = []
                    for c in value_cols:
                        q = float(self._adapter_thresholds(calib_cols[c], weights, level, True))  # ty: ignore[invalid-argument-type]
                        misses.append(1.0 if scored_values[c][j] > q else 0.0)
                    errors[i][coverage_rate] = float(np.mean(misses))
                else:
                    lower_misses, upper_misses = [], []
                    for c in value_cols:
                        lower_q, upper_q = self._adapter_thresholds(calib_cols[c], weights, level, False)  # ty: ignore[not-iterable]
                        s = scored_values[c][j]
                        lower_misses.append(1.0 if s < lower_q else 0.0)
                        upper_misses.append(1.0 if s > upper_q else 0.0)
                    errors[i][coverage_rate] = (float(np.mean(lower_misses)), float(np.mean(upper_misses)))

        return errors

    @staticmethod
    def _pool_adapter_errors(
        per_step_errors: dict[str, list[dict[float, object]]],
        coverage_rates: list[float],
        symmetric: bool,
        n_rows: int,
    ) -> list[dict[float, object]]:
        """Average per-step miscoverage into one shared per-row trajectory.

        Used for ``alpha_pooling="shared"``: a single pooled update is fed to
        every step's adapter, so the per-step dict alone (which cannot see
        across steps) still yields one shared level.

        Parameters
        ----------
        per_step_errors : dict
            Maps each ``"step_k"`` key to its length-``n_rows`` error list.
        coverage_rates : list of float
            Tracked coverage rates.
        symmetric : bool
            Whether the conformity scorer is symmetric.
        n_rows : int
            Number of observed rows.

        Returns
        -------
        list of dict
            Length ``n_rows`` pooled error list.

        """
        keys = list(per_step_errors)
        pooled: list[dict[float, object]] = []
        for i in range(n_rows):
            row: dict[float, object] = {}
            for coverage_rate in coverage_rates:
                if symmetric:
                    row[coverage_rate] = float(np.mean([per_step_errors[k][i][coverage_rate] for k in keys]))
                else:
                    lowers = [per_step_errors[k][i][coverage_rate][0] for k in keys]  # ty: ignore[not-subscriptable]
                    uppers = [per_step_errors[k][i][coverage_rate][1] for k in keys]  # ty: ignore[not-subscriptable]
                    row[coverage_rate] = (float(np.mean(lowers)), float(np.mean(uppers)))
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

        per_step_errors: dict[str, list[dict[float, object]]] = {}
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

        if getattr(self.adapter, "alpha_pooling", "per_step") == "shared":
            pooled = self._pool_adapter_errors(per_step_errors, coverage_rates, symmetric, n_rows)
            for key in per_step_errors:
                self.adapters_[key].observe(pooled)
        else:
            for key, errors in per_step_errors.items():
                self.adapters_[key].observe(errors)

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
        for adapter in self.adapters_.values():
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

        # Read scorer characteristics from tags instead of checking types
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

            if symmetric:
                q = weighted_quantile(scores_col, 1.0 - coverage_rate, weights)
                lower_data[col] = [pred_val - q * scale]
                upper_data[col] = [pred_val + q * scale]
            else:
                alpha = 1.0 - coverage_rate
                lower_q = weighted_quantile(scores_col, 1.0 - alpha / 2.0, weights)
                upper_q = weighted_quantile(scores_col, alpha / 2.0, weights)
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
        level: object,
        weights: np.ndarray,
        conformity_scorer_step: BaseConformityScorer,
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
        level : float or tuple of float
            Effective level from the adapter: ``alpha`` for symmetric
            scorers, ``(beta_lower, beta_upper)`` for asymmetric ones.
        weights : np.ndarray
            Per-calibration weights (similarity weights, or uniform).
        conformity_scorer_step : BaseConformityScorer
            Fitted conformity scorer (for tags and formatting).

        Returns
        -------
        pl.DataFrame
            Interval columns (no ``"time"`` column), labeled with the
            nominal ``coverage_rate``.

        """
        value_cols = [c for c in conformity_scores_step.columns if c != "time"]
        scores_no_time = conformity_scores_step.drop("time", strict=False)
        y_pred_values = y_pred_step.drop("time")

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

            thresholds = self._adapter_thresholds(scores_col, weights, level, symmetric)
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
            tracked_rates = set(self.adapters_["step_1"].predict().keys())
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

            rate_parts: list[pl.DataFrame] = []
            for coverage_rate in coverage_rates:
                adapter_active = hasattr(self, "adapters_") and coverage_rate in tracked_rates
                effective_level = self.adapters_[f"step_{step}"].predict()[coverage_rate] if adapter_active else None

                if self.similarity is not None and hasattr(self, "similarities_"):
                    similarity_step = self.similarities_[f"step_{step}"]
                    weights_array = similarity_step.predict(y_pred=y_pred_step)
                    step_weights = weights_array[0].astype(np.float64)

                    if adapter_active:
                        y_pred_interval_rate_step = self._adapter_inverse_score(
                            y_pred_step=y_pred_step,
                            conformity_scores_step=conformity_scores_step,
                            coverage_rate=coverage_rate,
                            level=effective_level,
                            weights=step_weights,
                            conformity_scorer_step=conformity_scorer_step,
                        )
                    else:
                        y_pred_interval_rate_step = self._weighted_inverse_score(
                            y_pred_step=y_pred_step,
                            conformity_scores_step=conformity_scores_step,
                            coverage_rate=coverage_rate,
                            weights=step_weights,
                            conformity_scorer_step=conformity_scorer_step,
                        )
                elif adapter_active:
                    uniform_weights = np.ones(conformity_scores_step.height, dtype=np.float64)
                    y_pred_interval_rate_step = self._adapter_inverse_score(
                        y_pred_step=y_pred_step,
                        conformity_scores_step=conformity_scores_step,
                        coverage_rate=coverage_rate,
                        level=effective_level,
                        weights=uniform_weights,
                        conformity_scorer_step=conformity_scorer_step,
                    )
                else:
                    y_pred_interval_rate_step = conformity_scorer_step.inverse_score(
                        y_pred=y_pred_step,
                        conformity_scores=conformity_scores_step,
                        coverage_rate=coverage_rate,
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
