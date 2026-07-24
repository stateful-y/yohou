"""AdditiveForecaster meta-forecaster for parallel observed-term additive composition."""

import polars as pl
from pydantic import StrictInt
from sklearn.base import BaseEstimator, clone
from sklearn.utils import Bunch
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    process_routing,
)
from sklearn.utils.validation import check_is_fitted

from yohou.base import BaseActualTransformer
from yohou.point import BasePointForecaster
from yohou.utils import POINT, Tags, inspect_panel, validate_forecaster_data
from yohou.utils._compat import _BaseComposition, _fit_context, _raise_for_params

__all__ = ["AdditiveForecaster"]


class AdditiveForecaster(BasePointForecaster, _BaseComposition):
    """Point meta-forecaster that sums independently fitted per-term forecasts.

    AdditiveForecaster forecasts an aggregate target ``y`` as the sum of
    independently fitted "term" forecasters, in level space:
    ``y = sum_i y_i (+ eps)``. It is the observed-term, parallel-fit dual of
    [`DecompositionPipeline`][yohou.compose.decomposition_pipeline.DecompositionPipeline],
    which emits the same output algebra but by sequential residual decomposition.

    Each term carries an ``extractor``: a fit-time recipe (a stateless
    `BaseActualTransformer`) that manufactures the term's target series from the
    training data (the target ``y`` joined with ``X_actual`` on ``"time"``). The
    extractor runs at ``fit``, ``observe``, and ``rewind``, but never at
    ``predict``, where each term forecaster predicts from its own features.

    Granularity is inferred from the extractor output: a single unprefixed
    series is a global term (broadcast across every panel group), and a
    ``__``-prefixed panel output is a per-group term (aligned to its group).

    An optional ``residual_forecaster`` closes the gap the declared terms leave.
    It is fitted on ``eps = y - sum_i y_i`` computed from the observed extracted
    term targets, and its forecast is added to the summed term forecasts at
    predict.

    Parameters
    ----------
    terms : list of (str, BaseActualTransformer, BasePointForecaster) tuples
        List of ``(name, extractor, forecaster)`` triples with unique names.

        name : str
            Unique name for the term. Used for metadata routing and
            ``get_params``/``set_params``.
        extractor : BaseActualTransformer
            Stateless transformer that manufactures the term's target series
            from the ``y``-joined-``X_actual`` frame. A stateful extractor is
            rejected at fit.
        forecaster : BasePointForecaster
            Point forecaster fitted on the extracted term target.

    residual_forecaster : BasePointForecaster or None, default=None
        Optional point forecaster fitted on the residual
        ``eps = y - sum_i y_i``. When ``None``, the prediction is the plain
        bottom-up sum of the term forecasts.

    Attributes
    ----------
    forecasters_ : list of (str, BasePointForecaster, str) tuples
        Fitted term forecasters as ``(name, fitted_forecaster, granularity)``,
        where ``granularity`` is ``"global"`` or ``"panel"``.

    extractors_ : dict of str to BaseActualTransformer
        Fitted extractors keyed by term name, reused at ``observe``/``rewind``.

    residual_forecaster_ : BasePointForecaster or None
        The fitted residual forecaster, or ``None`` when ``residual_forecaster``
        is ``None``.

    See Also
    --------
    - [`DecompositionPipeline`][yohou.compose.decomposition_pipeline.DecompositionPipeline] : Sequential residual-based dual of this class.
    - [`ColumnForecaster`][yohou.compose.column_forecaster.ColumnForecaster] : Separate forecasters for target/feature columns.
    - [`SeasonalNaive`][yohou.point.naive.SeasonalNaive] : Simple point forecaster usable as a term.
    - [`PolynomialTrendForecaster`][yohou.stationarity.trend.PolynomialTrendForecaster] : Polynomial trend forecaster usable as a term.

    Notes
    -----
    **Additive identity in level space**::

        y = y_1 + y_2 + ... + y_k + eps

    The additive identity holds in level space, so this forecaster exposes no
    pipeline-level ``target_transformer``; any per-term scaling lives inside each
    term forecaster's own ``target_transformer``.

    All term forecasters and the residual forecaster must be point forecasters
    (`BasePointForecaster`); a non-point forecaster raises ``ValueError`` at fit.

    Terms are fitted in parallel and independently: ``observe`` and ``rewind``
    re-extract each term target and forward it to the corresponding term
    forecaster without the sequential residual threading
    `DecompositionPipeline` needs.

    Raises
    ------
    ValueError
        If any term forecaster or the residual forecaster is not a
        `BasePointForecaster`, if ``terms`` is empty, if term names are not
        unique, if an extractor carries observation state
        (``observation_horizon > 0``), or if ``forecasting_horizon`` < 1.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.compose import AdditiveForecaster
    >>> from yohou.preprocessing import FunctionTransformer
    >>> from yohou.point import SeasonalNaive
    >>>
    >>> time = pl.datetime_range(
    ...     start=datetime(2020, 1, 1), end=datetime(2020, 2, 29), interval="1d", eager=True
    ... )
    >>> y = pl.DataFrame({"time": time, "spp": [30.0 + (i % 7) for i in range(len(time))]})
    >>>
    >>> # An extractor manufactures each term's target at fit time.
    >>> anchor = FunctionTransformer(func=lambda df: df.select((pl.col("spp") * 0.5).alias("anchor")))
    >>> forecaster = AdditiveForecaster(
    ...     terms=[("anchor", anchor, SeasonalNaive(seasonality=7))],
    ...     residual_forecaster=SeasonalNaive(seasonality=7),
    ... )
    >>> forecaster.fit(y, forecasting_horizon=7)  # doctest: +ELLIPSIS
    AdditiveForecaster(...)
    >>> y_pred = forecaster.predict(forecasting_horizon=7)

    """

    _parameter_constraints: dict = {
        "terms": [list],
        "residual_forecaster": [BasePointForecaster, None],
    }

    def __init__(
        self,
        terms: list[tuple[str, BaseActualTransformer, BasePointForecaster]],
        *,
        residual_forecaster: BasePointForecaster | None = None,
    ):
        BasePointForecaster.__init__(
            self,
            target_transformer=None,
            actual_transformer=None,
            forecast_transformer=None,
            target_as_feature=None,
            panel_strategy="global",
        )
        self.terms = terms
        self.residual_forecaster = residual_forecaster

    @property
    def _terms(self) -> list[tuple[str, BasePointForecaster]]:
        """Adapter exposing ``(name, forecaster)`` 2-tuples for ``_BaseComposition``.

        ``_BaseComposition._get_params``/``_set_params`` expect a list of
        ``(name, estimator)`` 2-tuples, so the extractor is stripped from each
        ``(name, extractor, forecaster)`` triple.

        Returns
        -------
        list of (str, BasePointForecaster)
            Term tuples with the extractor stripped.

        """
        try:
            return [(name, forecaster) for name, _, forecaster in self.terms]  # ty: ignore[invalid-assignment]
        except (TypeError, ValueError):
            return self.terms  # ty: ignore[invalid-return-type]

    @_terms.setter
    def _terms(self, value: list[tuple[str, BasePointForecaster]]) -> None:
        """Merge new ``(name, forecaster)`` 2-tuples back with the original extractors.

        Parameters
        ----------
        value : list of (str, BasePointForecaster)
            New term 2-tuples produced by ``_set_params``.

        """
        try:
            self.terms = [
                (name, extractor, forecaster)
                for ((name, forecaster), (_, extractor, _)) in zip(value, self.terms, strict=True)  # ty: ignore[invalid-assignment]
            ]
        except (TypeError, ValueError):
            self.terms = value

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, return the parameters for this estimator and contained
            subobjects that are estimators.

        Returns
        -------
        dict
            Parameter names mapped to their values.

        """
        return self._get_params("_terms", deep=deep)

    def set_params(self, **params) -> "AdditiveForecaster":
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``. Nested term
        parameters use double-underscore notation, e.g. ``terms__lambda__degree``.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self
            Estimator instance.

        """
        self._set_params("_terms", **params)
        return self

    def _collect_children(self) -> list[BasePointForecaster]:
        """Collect the term and residual forecasters for tag aggregation.

        Uses the fitted forecasters when available, otherwise the unfitted
        constructor arguments, so tags are accessible before and after fit.

        Returns
        -------
        list of BasePointForecaster
            The term forecasters plus the residual forecaster when set.

        """
        if hasattr(self, "forecasters_"):
            children = [forecaster for _, forecaster, _ in self.forecasters_]
            if self.residual_forecaster_ is not None:
                children.append(self.residual_forecaster_)
        else:
            children = [forecaster for _, _, forecaster in self.terms]
            if self.residual_forecaster is not None:
                children.append(self.residual_forecaster)
        return children

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes: point type,
            ``requires_exogenous=False``, ``tracks_observations=False``, and
            ``stateful``/``supports_panel_data`` aggregated from the children.

        """
        tags = super().__sklearn_tags__()
        assert tags.forecaster_tags is not None

        tags.forecaster_tags.forecaster_type = POINT
        # The meta-forecaster forwards X_actual to children but does not require it.
        tags.forecaster_tags.requires_exogenous = False
        # Observation tracking is delegated to the child forecasters.
        tags.forecaster_tags.tracks_observations = False

        children = self._collect_children()
        if children:
            tags.forecaster_tags.stateful = any(
                getattr(child.__sklearn_tags__().forecaster_tags, "stateful", False) for child in children
            )
            tags.forecaster_tags.supports_panel_data = all(
                getattr(child.__sklearn_tags__().forecaster_tags, "supports_panel_data", True) for child in children
            )

        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> "AdditiveForecaster":
        """Fit each term forecaster on its extracted target, then the residual.

        Parameters
        ----------
        y : pl.DataFrame
            Aggregate target time series with a ``"time"`` column (datetime)
            and one or more numeric value columns.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned with
            ``y``. Joined with ``y`` to form the extractor input frame and
            forwarded to each term forecaster.
        forecasting_horizon : int, default=1
            Number of time steps to forecast into the future.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column. Forwarded to each
            term forecaster.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"`` columns.
            Forwarded to each term forecaster.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted AdditiveForecaster instance.

        Raises
        ------
        ValueError
            If any term forecaster or the residual forecaster is not a
            `BasePointForecaster`, if term names are not unique, if an
            extractor is stateful, or if ``forecasting_horizon`` < 1.

        """
        forecasting_horizon = self._validate_fit_params(forecasting_horizon)
        _raise_for_params(params, self, "fit")

        if not self.terms:
            raise ValueError("AdditiveForecaster requires at least one term, but `terms` is empty.")

        # Validate term names are unique (and routing-safe).
        self._validate_names([name for name, _, _ in self.terms])  # ty: ignore[invalid-assignment]

        # Validate all term forecasters and the residual are point forecasters.
        for name, _extractor, forecaster in self.terms:  # ty: ignore[invalid-assignment]
            if not isinstance(forecaster, BasePointForecaster):
                raise ValueError(
                    f"All term forecasters must be BasePointForecaster instances. "
                    f"Term '{name}' has a {type(forecaster).__name__}."
                )
        if self.residual_forecaster is not None and not isinstance(self.residual_forecaster, BasePointForecaster):
            raise ValueError(
                f"residual_forecaster must be a BasePointForecaster instance, got "
                f"{type(self.residual_forecaster).__name__}."
            )

        # Self-scaffolding: sets groups_, local_y_schema_, interval_, observed_time_,
        # _step_column_names_, _X_forecast_t_, etc. that predict/validate rely on.
        # target_transformer/actual_transformer are None, so the returned frames
        # equal the inputs and are not needed here.
        self._pre_fit(
            y=y,
            X_actual=X_actual,
            forecasting_horizon=forecasting_horizon,
            X_future=X_future,
            X_forecast=X_forecast,
        )

        routed_params = process_routing(self, "fit", **params)

        # The extractor sees "time" + y columns + X columns. A left join keeps
        # every y row even when X_actual omits some timestamps.
        frame = y if X_actual is None else y.join(X_actual, on="time", how="left")

        self.forecasters_ = []
        self.extractors_ = {}
        observed_contributions: list[tuple[str, pl.DataFrame]] = []
        for name, extractor, forecaster in self.terms:  # ty: ignore[invalid-assignment]
            extractor_ = clone(extractor)
            # Manufacture the term target. Reject only extractors that carry
            # observation state (they consume past rows, so a fit-time per-batch
            # label build is not well defined). observation_horizon is readable
            # only after fit, so the guard follows fit_transform; the term
            # forecaster is not fitted until the extractor passes.
            y_i = extractor_.fit_transform(frame)
            if extractor_.observation_horizon > 0:
                raise ValueError(
                    f"AdditiveForecaster supports extractors without observation state (v1). "
                    f"Term '{name}' has an extractor that consumes {extractor_.observation_horizon} "
                    f"past row(s) (observation_horizon > 0); a term extractor manufactures fit-time "
                    f"labels and must not require past rows. Extractor type: {type(extractor).__name__}."
                )

            _, panel_groups = inspect_panel(y_i)
            granularity = "panel" if panel_groups else "global"

            forecaster_ = clone(forecaster)
            step_params = routed_params.get(name, Bunch(fit={}))
            forecaster_.fit(
                y=y_i,
                X_actual=X_actual,
                forecasting_horizon=forecasting_horizon,
                X_future=X_future,
                X_forecast=X_forecast,
                **step_params.fit,
            )
            self.forecasters_.append((name, forecaster_, granularity))
            self.extractors_[name] = extractor_
            observed_contributions.append((granularity, y_i))

        self.residual_forecaster_ = None
        if self.residual_forecaster is not None:
            eps = self._compute_residual(y, observed_contributions)
            residual_params = routed_params.get("residual_forecaster", Bunch(fit={}))
            self.residual_forecaster_ = clone(self.residual_forecaster)
            self.residual_forecaster_.fit(
                y=eps,
                X_actual=X_actual,
                forecasting_horizon=forecasting_horizon,
                X_future=X_future,
                X_forecast=X_forecast,
                **residual_params.fit,
            )

        self._y_observed = self._bounded_observed(y)

        return self

    def _bounded_observed(self, y: pl.DataFrame) -> pl.DataFrame:
        """Store the observation buffer trimmed to ``observation_horizon`` rows.

        This forecaster has no pipeline-level ``target_transformer``, so
        ``observation_horizon`` is ``0`` and the buffer is the full frame. The
        method mirrors `DecompositionPipeline` so long-running streams stay
        bounded if a future revision introduces a stateful pipeline transformer.

        Parameters
        ----------
        y : pl.DataFrame
            The aggregate target frame for this call.

        Returns
        -------
        pl.DataFrame
            The bounded observation buffer.

        """
        horizon = self.observation_horizon
        return y.tail(horizon) if horizon else y

    def _broadcast_sum(
        self,
        contributions: list[tuple[str, pl.DataFrame]],
        index_cols: tuple[str, ...],
        groups: list[str] | None,
        residual: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Sum term contributions into the aggregate target, broadcasting globals.

        Every frame is aligned on the ``index_cols`` pair (never ``"time"``
        alone, which would fan rows across vintages sharing a timestamp). A
        global term's single value column is added to every aggregate target
        column; a panel term's group-g column is added to aggregate group g.
        A group present in the aggregate but absent from a panel term's frame
        raises rather than being silently zero-filled. The residual, when
        provided, carries the aggregate's granularity and is added by exact
        name match.

        Parameters
        ----------
        contributions : list of (str, pl.DataFrame)
            ``(granularity, frame)`` pairs, where ``granularity`` is
            ``"global"`` or ``"panel"`` and ``frame`` carries ``index_cols``
            plus value columns.
        index_cols : tuple of str
            The alignment key, ``("time",)`` for observed targets or
            ``("vintage_time", "time")`` for forecasts.
        groups : list of str or None
            Aggregate groups to produce (panel), or ``None`` for non-panel data.
        residual : pl.DataFrame or None, default=None
            Aggregate-granularity residual to add, or ``None``.

        Returns
        -------
        pl.DataFrame
            The aggregate frame with ``index_cols`` plus the summed target
            columns (``{group}__{suffix}`` in panel mode, ``{suffix}`` otherwise).

        Raises
        ------
        ValueError
            If a panel term forecast omits a group present in the aggregate.

        """
        index_list = list(index_cols)
        suffixes = list(self.local_y_schema_.keys())

        wide = contributions[0][1].select(index_list)
        global_cols: list[str] = []
        panel_group_cols: dict[str, list[str]] = {g: [] for g in (groups or [])}

        for idx, (granularity, frame) in enumerate(contributions):
            value_cols = [c for c in frame.columns if c not in index_list]
            rename = {c: f"__term{idx}__{c}" for c in value_cols}
            wide = wide.join(frame.rename(rename), on=index_list, how="left")
            if granularity == "panel":
                for group_name in groups or []:
                    matched = [
                        f"__term{idx}__{c}" for c in value_cols if c == group_name or c.startswith(f"{group_name}__")
                    ]
                    if not matched:
                        raise ValueError(
                            f"A panel term forecast is missing group '{group_name}', which is present "
                            f"in the aggregate target. AdditiveForecaster does not zero-fill a dropped group."
                        )
                    panel_group_cols[group_name].extend(matched)
            else:
                global_cols.extend(f"__term{idx}__{c}" for c in value_cols)

        residual_prefix = None
        if residual is not None:
            residual_value_cols = [c for c in residual.columns if c not in index_list]
            residual_rename = {c: f"__residual__{c}" for c in residual_value_cols}
            wide = wide.join(residual.rename(residual_rename), on=index_list, how="left")
            residual_prefix = "__residual__"

        def _sum_expr(cols: list[str]) -> pl.Expr:
            """Sum the named columns, defaulting to a float zero when empty.

            Parameters
            ----------
            cols : list of str
                Column names to add together.

            Returns
            -------
            pl.Expr
                The polars expression summing ``cols``.

            """
            expr = pl.lit(0.0)
            for col in cols:
                expr = expr + pl.col(col)
            return expr

        global_sum = _sum_expr(global_cols)

        agg_exprs: list[pl.Expr] = []
        if groups is None:
            for suffix in suffixes:
                expr = global_sum
                residual_col = f"{residual_prefix}{suffix}" if residual_prefix is not None else None
                if residual_col is not None and residual_col in wide.columns:
                    expr = expr + pl.col(residual_col)
                agg_exprs.append(expr.alias(suffix))
        else:
            for group_name in groups:
                group_sum = global_sum + _sum_expr(panel_group_cols[group_name])
                for suffix in suffixes:
                    agg_col = f"{group_name}__{suffix}"
                    expr = group_sum
                    residual_col = f"{residual_prefix}{agg_col}" if residual_prefix is not None else None
                    if residual_col is not None and residual_col in wide.columns:
                        expr = expr + pl.col(residual_col)
                    agg_exprs.append(expr.alias(agg_col))

        return wide.select([pl.col(col) for col in index_list] + agg_exprs)

    def _compute_residual(
        self,
        y: pl.DataFrame,
        observed_contributions: list[tuple[str, pl.DataFrame]],
    ) -> pl.DataFrame:
        """Compute the observed residual ``eps = y - sum_i y_i``.

        The term sum is built from the observed extracted targets, broadcasting
        global terms across the aggregate's groups, and subtracted from the
        aggregate target column by column on ``"time"``.

        Parameters
        ----------
        y : pl.DataFrame
            The aggregate target with a ``"time"`` column.
        observed_contributions : list of (str, pl.DataFrame)
            ``(granularity, observed_target)`` pairs from the fitted extractors.

        Returns
        -------
        pl.DataFrame
            The residual frame with the same target schema as ``y``.

        """
        sum_frame = self._broadcast_sum(observed_contributions, ("time",), self.groups_)
        agg_cols = [c for c in y.columns if c != "time"]
        joined = y.join(sum_frame, on="time", how="inner", suffix="__additive_sum")
        return joined.select(
            [pl.col("time")] + [(pl.col(col) - pl.col(f"{col}__additive_sum")).alias(col) for col in agg_cols]
        )

    def predict(  # ty: ignore[invalid-method-override]
        self,
        forecasting_horizon: StrictInt | None = None,
        groups: list[str] | None = None,
        predict_transformed: bool = False,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> pl.DataFrame:
        """Generate forecasts as the broadcast sum of the term forecasts.

        Parameters
        ----------
        forecasting_horizon : int >= 1 or None, default=None
            Horizon to forecast. If ``None``, uses ``fit_forecasting_horizon_``.
        groups : list of str or None, default=None
            Panel group prefixes to operate on. If ``None``, all groups are
            predicted. Ignored when the forecaster was not fitted on panel data.
        predict_transformed : bool, default=False
            Present for API parity. This forecaster has no pipeline-level
            target transform, so predictions are always level-space.
        X_future : pl.DataFrame or None, default=None
            Known future features override forwarded to each term forecaster.
        X_forecast : pl.DataFrame or None, default=None
            External forecast override forwarded to each term forecaster.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Predictions with ``"vintage_time"``, ``"time"``, and the aggregate
            target columns.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the forecaster has not been fitted yet.
        ValueError
            If ``groups`` contains names not seen during fit, or if a panel
            term forecast omits a group present in the aggregate.

        """
        check_is_fitted(self, ["forecasters_", "groups_"])
        _, _, groups = validate_forecaster_data(
            self,
            y=None,
            X_actual=None,
            reset=False,
            groups=groups,
        )

        forecasting_horizon = self._validate_predict_params(forecasting_horizon)
        _raise_for_params(params, self, "predict")
        routed_params = process_routing(self, "predict", **params)

        contributions: list[tuple[str, pl.DataFrame]] = []
        for name, forecaster, granularity in self.forecasters_:
            step_params = routed_params.get(name, Bunch(predict={}))
            term_groups = groups if granularity == "panel" else None
            y_hat_i = forecaster.predict(
                forecasting_horizon=forecasting_horizon,
                groups=term_groups,
                predict_transformed=False,
                X_future=X_future,
                X_forecast=X_forecast,
                **step_params.predict,
            )
            contributions.append((granularity, y_hat_i))

        residual_pred = None
        if self.residual_forecaster_ is not None:
            residual_params = routed_params.get("residual_forecaster", Bunch(predict={}))
            residual_pred = self.residual_forecaster_.predict(
                forecasting_horizon=forecasting_horizon,
                groups=groups,
                predict_transformed=False,
                X_future=X_future,
                X_forecast=X_forecast,
                **residual_params.predict,
            )

        return self._broadcast_sum(contributions, ("vintage_time", "time"), groups, residual=residual_pred)

    def observe(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> "AdditiveForecaster":
        """Observe new data, re-extracting each term target and forwarding it.

        Each term's stored extractor re-manufactures the term target from the
        new ``y``-joined-``X_actual`` frame, which is forwarded to the term
        forecaster's ``observe``. When a residual forecaster is set, ``eps`` is
        recomputed on the batch and observed. No sequential residual threading
        across terms is performed.

        Parameters
        ----------
        y : pl.DataFrame
            New aggregate target observations with a ``"time"`` column.
        X_actual : pl.DataFrame or None, default=None
            New actual feature observations with a ``"time"`` column aligned
            with ``y``.
        groups : list of str or None, default=None
            Panel group prefixes to operate on. If ``None``, all groups are
            observed.
        X_future : pl.DataFrame or None, default=None
            Known future features forwarded to each term forecaster.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts forwarded to each term forecaster.

        Returns
        -------
        self
            The forecaster with updated observation state.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the forecaster has not been fitted yet.

        """
        check_is_fitted(self, ["forecasters_", "groups_"])
        y, X_actual, groups = validate_forecaster_data(
            self,
            y=y,
            X_actual=X_actual,
            reset=False,
            groups=groups,
        )

        frame = y if X_actual is None else y.join(X_actual, on="time", how="left")

        observed_contributions: list[tuple[str, pl.DataFrame]] = []
        for name, forecaster, granularity in self.forecasters_:
            y_i = self.extractors_[name].transform(frame)
            term_groups = groups if granularity == "panel" else None
            forecaster.observe(y_i, X_actual=X_actual, groups=term_groups, X_future=X_future, X_forecast=X_forecast)
            observed_contributions.append((granularity, y_i))

        if self.residual_forecaster_ is not None:
            eps = self._compute_residual(y, observed_contributions)
            self.residual_forecaster_.observe(
                eps, X_actual=X_actual, groups=groups, X_future=X_future, X_forecast=X_forecast
            )

        self._y_observed = self._bounded_observed(y)

        return self

    def rewind(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        groups: list[str] | None = None,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
    ) -> "AdditiveForecaster":
        """Rewind each term forecaster to the end of the provided window.

        Like ``observe``, but each term forecaster and the residual are rewound
        (their observation buffers replaced) rather than extended.

        Parameters
        ----------
        y : pl.DataFrame
            Aggregate target observations with a ``"time"`` column.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations to restore the observation state to.
        groups : list of str or None, default=None
            Panel group prefixes to operate on. If ``None``, all groups are
            rewound.
        X_future : pl.DataFrame or None, default=None
            Known future features forwarded to each term forecaster.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts forwarded to each term forecaster.

        Returns
        -------
        self
            The forecaster with rewound observation state.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the forecaster has not been fitted yet.

        """
        check_is_fitted(self, ["forecasters_", "groups_"])
        y, X_actual, groups = validate_forecaster_data(
            self,
            y=y,
            X_actual=X_actual,
            reset=False,
            groups=groups,
        )

        frame = y if X_actual is None else y.join(X_actual, on="time", how="left")

        observed_contributions: list[tuple[str, pl.DataFrame]] = []
        for name, forecaster, granularity in self.forecasters_:
            y_i = self.extractors_[name].transform(frame)
            term_groups = groups if granularity == "panel" else None
            forecaster.rewind(y_i, X_actual=X_actual, groups=term_groups, X_future=X_future, X_forecast=X_forecast)
            observed_contributions.append((granularity, y_i))

        if self.residual_forecaster_ is not None:
            eps = self._compute_residual(y, observed_contributions)
            self.residual_forecaster_.rewind(
                eps, X_actual=X_actual, groups=groups, X_future=X_future, X_forecast=X_forecast
            )

        self._y_observed = self._bounded_observed(y)

        return self

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
        """Alternate rolling observe and predict across ``y``.

        Overrides the base ``observe_predict`` so the rolling loop calls this
        forecaster's custom ``observe`` at each stride step (re-extracting term
        targets) rather than the flat base observe.

        Parameters
        ----------
        y : pl.DataFrame
            Aggregate target time series with a ``"time"`` column.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations sliced and observed incrementally.
        forecasting_horizon : int or None, default=None
            Number of time steps to forecast. If ``None``, uses the fit horizon.
        groups : list of str or None, default=None
            Panel group prefixes to operate on. If ``None``, all groups are used.
        stride : int or None, default=None
            Step size for rolling update then predict. If ``None``, defaults to
            ``fit_forecasting_horizon_``.
        predict_transformed : bool, default=False
            Present for API parity; predictions are always level-space.
        X_future : pl.DataFrame or None, default=None
            Known future features forwarded to each term forecaster.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts forwarded to each term forecaster.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Point predictions with ``"vintage_time"``, ``"time"``, and the
            aggregate target columns.

        """
        check_is_fitted(self, ["forecasters_", "groups_"])
        y, X_actual, groups = validate_forecaster_data(
            self,
            y=y,
            X_actual=X_actual,
            reset=False,
            groups=groups,
            X_future=X_future,
            X_forecast=X_forecast,
        )

        fh = self._validate_predict_params(forecasting_horizon)
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
            forecasting_horizon=fh,
            predict_transformed=predict_transformed,
            **params,
        )

    def get_metadata_routing(self) -> MetadataRouter:
        """Get metadata routing for this estimator.

        Returns
        -------
        MetadataRouter
            Router forwarding metadata to each term forecaster and the residual
            forecaster for ``fit``, ``predict``, and ``observe_predict``.

        """
        router = MetadataRouter(owner=self)

        method_mapping = (
            MethodMapping()
            .add(caller="fit", callee="fit")
            .add(caller="predict", callee="predict")
            .add(caller="observe_predict", callee="observe_predict")
        )

        for name, _extractor, forecaster in self.terms:  # ty: ignore[invalid-assignment]
            router.add(**{name: forecaster}, method_mapping=method_mapping)

        if isinstance(self.residual_forecaster, BaseEstimator):
            router.add(residual_forecaster=self.residual_forecaster, method_mapping=method_mapping)

        return router
