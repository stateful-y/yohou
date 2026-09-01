"""Implementation of reduction-based class-probability forecasters."""

from __future__ import annotations

from typing import Any, Literal, cast

import polars as pl
from pydantic import StrictInt
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from yohou.base import BaseActualTransformer, BaseForecastTransformer, BaseReductionForecaster, BaseStepTransformer
from yohou.utils._compat import HasMethods, StrOptions, _fit_context
from yohou.weighting import BaseWeighter

from .base import BaseClassProbaForecaster

__all__ = ["ClassProbaReductionForecaster"]


class ClassProbaReductionForecaster(BaseReductionForecaster, BaseClassProbaForecaster):
    """Class-probability forecaster using sklearn classifiers on tabularized time series.

    Converts categorical time series forecasting to a tabular classification task.
    The target is encoded to integer codes before tabularization; predictions use
    ``predict_proba`` to return per-class probability distributions.

    Parameters
    ----------
    estimator : BaseEstimator, default=LogisticRegression()
        Classifier used to fit the tabularized data. Must implement
        ``fit``, ``predict``, and ``predict_proba``.
    reduction_strategy : {"direct", "multi-output"}, default="multi-output"
        Strategy for multi-step forecasting.

        - ``"multi-output"``: a single classifier predicts all H horizon
          steps simultaneously. Simple and fast.
        - ``"direct"``: H independent classifiers, one per horizon step.
          Each model specialises in its own step, avoiding error
          accumulation but ignoring inter-step dependencies.

        See [`BaseReductionForecaster`][yohou.base.reduction.BaseReductionForecaster]
        for full per-option semantics.
    target_transformer : BaseActualTransformer or None, default=None
        Transformer for target preprocessing.
    actual_transformer : BaseActualTransformer or None, default=None
        Transformer for feature engineering (typically LagTransformer).
    forecast_transformer : BaseForecastTransformer or None, default=None
        Transformer applied to ``X_forecast`` before step columns are derived,
        so the step columns reaching the estimator are built from transformed
        values. Must be forecast-kind (vintage-indexed); an actual-kind
        transformer is rejected. ``None`` leaves ``X_forecast`` untouched.
    step_transformer : BaseStepTransformer or None, default=None
        Transformer applied to the derived ``{base}_step_1..H`` frame after
        step columns are built from ``X_future``/``X_forecast`` and before they
        join the design matrix. Reduces or rescales along the horizon axis.
        ``None`` leaves the step columns as derived.
    target_as_feature : {"transformed", "raw"} or None, default="transformed"
        Whether to include the target variable as a feature for reduction.
        If ``"transformed"``, the transformed target is used. If ``"raw"``,
        the raw target is used. If ``None``, the target is not included as
        a feature.
    step_feature_alignment : {"all", "matched", "cumulative"}, default="all"
        Controls which step-indexed feature columns each direct estimator
        sees. Only the ``"direct"`` strategy applies this parameter; a
        non-default value on any other strategy warns at fit and changes
        nothing. ``"multi-output"`` cannot filter, since one estimator reads a
        different step column per output and needs them all; ``"dir-rec"``
        could, but is excluded by a deliberate scope decision.

        - ``"all"``: every estimator receives all step columns.
        - ``"matched"``: estimator for step h receives only ``*_step_h``.
        - ``"cumulative"``: estimator for step h receives ``*_step_1..h``.

        Filtering applies identically whatever the step columns were derived
        from and under either panel strategy. A non-default alignment that
        cannot recognize any step column in the feature matrix raises
        ``RuntimeError`` rather than falling back to ``"all"``.

    training_stride : int, default=1
        Keep one tabularized training instance every ``training_stride`` rows,
        tail-anchored so the most recent instance is always kept. The default 1
        keeps every instance. See
        [`BaseReductionForecaster`][yohou.base.reduction.BaseReductionForecaster]
        for the full semantics.
    validation_size : int or None, default=None
        Number of trailing time steps (per group on panel data) to hold out
        from classifier training and deliver to the wrapped estimator's
        ``fit`` as ``eval_set``, enabling estimator-side early stopping
        (LightGBM, XGBoost, CatBoost). Class discovery and label encoding
        use the remaining head only, so a class occurring only inside the
        tail raises ``ValueError``. Early stopping itself is configured on
        the estimator. Fitting also raises ``ValueError`` when the
        estimator's ``fit`` accepts neither ``eval_set`` nor ``**kwargs``,
        the estimator is a ``sklearn.multioutput`` wrapper, the head left
        after the split cannot build one training row, ``validation_size``
        is smaller than ``forecasting_horizon`` while
        ``validation_overlap=False``, or a raw ``eval_set`` is also passed
        through fit ``**params``. See
        [`BaseReductionForecaster`][yohou.base.reduction.BaseReductionForecaster]
        for the full semantics.
    validation_overlap : bool, default=False
        Only used when ``validation_size`` is set. By default only rows
        whose entire target window lies inside the held-out tail are
        evaluated (``validation_size - forecasting_horizon + 1`` rows).
        When ``True``, the ``forecasting_horizon - 1`` boundary rows whose
        target windows straddle the split are also evaluated, yielding
        ``validation_size`` rows; those rows score some time points the
        model also trained on.
    nan_handling : {"drop", "pass"}, default="pass"
        How to handle NaN values in tabularized data.
        ``"pass"`` leaves NaN in place (suitable for estimators that
        handle NaN natively, such as tree-based models). ``"drop"``
        removes any training instance where X or y contains NaN before
        fitting the estimator, and emits a warning with the count of
        dropped rows. At predict time, returns NaN predictions for any
        time step whose features contain NaN.
    panel_strategy : {"global", "multivariate"}, default="global"
        How to handle panel data. See `BaseForecaster` for details.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel for the ``"direct"`` strategy.
        ``None`` means 1. ``-1`` means using all processors.
    time_weighter : BaseWeighter or None, default=None
        Per-timestep training-sample weighter (e.g.
        [`ExponentialDecayWeighter`][yohou.weighting.weighters.ExponentialDecayWeighter]).
        Its parameters are tunable via search. If None, samples are unweighted.
    vintage_weighter : BaseWeighter or None, default=None
        Per-vintage training-sample weighter, combined multiplicatively with
        ``time_weighter``. If None, no vintage weighting is applied.
    sample_weight_alignment : {"first_step", "mean_step", "weighted_mean_step", \
"max_weight_step", "min_weight_step"}, default="first_step"
        Strategy for converting ``time_weighter`` weights to sklearn
        ``sample_weight`` across forecast horizons.

    Attributes
    ----------
    classes_ : dict[str, list[str]]
        Mapping from target column name to its class labels, discovered at
        fit time from the unique values in each target column.
    n_classes_ : dict[str, int]
        Mapping from target column name to the number of classes.
    label_to_code_ : dict[str, dict[str, int]]
        Mapping from target column name to a dict mapping class labels to
        integer codes.
    estimator_ : BaseEstimator or list[BaseEstimator]
        Fitted sklearn classifier(s).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime
    >>> from yohou.class_proba import ClassProbaReductionForecaster
    >>>
    >>> df = pl.DataFrame({
    ...     "time": pl.datetime_range(
    ...         start=datetime(2021, 1, 1),
    ...         end=datetime(2021, 1, 10),
    ...         interval="1d",
    ...         eager=True,
    ...     ),
    ...     "weather": ["sun", "sun", "rain", "rain", "cloud", "sun", "rain", "cloud", "sun", "rain"],
    ... })
    >>>
    >>> train = df[:8]
    >>> forecaster = ClassProbaReductionForecaster()
    >>> _ = forecaster.fit(y=train, forecasting_horizon=1)
    >>>
    >>> y_proba = forecaster.predict_class_proba(forecasting_horizon=1)
    >>> len(y_proba)
    1

    Notes
    -----
    The target columns are label-encoded to integer codes before
    tabularization. The encoding is stored in ``classes_`` and
    ``label_to_code_`` so that ``predict_class_proba`` can map the
    classifier's probability output back to the original class labels.

    See Also
    --------
    - [`BaseClassProbaForecaster`][yohou.class_proba.base.BaseClassProbaForecaster] : Base class for class-probability forecasters.
    - [`PointReductionForecaster`][yohou.point.reduction.PointReductionForecaster] : ML-based point forecaster.
    - [`BaseReductionForecaster`][yohou.base.reduction.BaseReductionForecaster] : Base class for reduction forecasters.

    """

    _parameter_constraints: dict = {
        **BaseReductionForecaster._parameter_constraints,
        "estimator": [HasMethods(["fit", "predict", "predict_proba"])],
        "reduction_strategy": [StrOptions({"direct", "multi-output"})],
    }

    _supports_panel = True

    def __init__(
        self,
        estimator: BaseEstimator = LogisticRegression(),
        *,
        reduction_strategy: Literal["direct", "multi-output"] = "multi-output",
        target_transformer: BaseActualTransformer | None = None,
        actual_transformer: BaseActualTransformer | None = None,
        forecast_transformer: BaseForecastTransformer | None = None,
        step_transformer: BaseStepTransformer | None = None,
        target_as_feature: Literal["transformed", "raw"] | None = "transformed",
        step_feature_alignment: Literal["all", "matched", "cumulative"] = "all",
        training_stride: int = 1,
        validation_size: int | None = None,
        validation_overlap: bool = False,
        nan_handling: Literal["drop", "pass"] = "pass",
        n_jobs: int | None = None,
        panel_strategy: Literal["global", "multivariate"] = "global",
        time_weighter: BaseWeighter | None = None,
        vintage_weighter: BaseWeighter | None = None,
        sample_weight_alignment: str = "first_step",
    ) -> None:
        BaseReductionForecaster.__init__(
            self,
            estimator=estimator,
            reduction_strategy=reduction_strategy,
            target_as_feature=target_as_feature,
            target_transformer=target_transformer,
            actual_transformer=actual_transformer,
            forecast_transformer=forecast_transformer,
            step_transformer=step_transformer,
            step_feature_alignment=step_feature_alignment,
            training_stride=training_stride,
            validation_size=validation_size,
            validation_overlap=validation_overlap,
            nan_handling=nan_handling,
            n_jobs=n_jobs,
            panel_strategy=panel_strategy,
            time_weighter=time_weighter,
            vintage_weighter=vintage_weighter,
            sample_weight_alignment=sample_weight_alignment,
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        y: pl.DataFrame,
        X_actual: pl.DataFrame | None = None,
        forecasting_horizon: StrictInt = 1,
        X_future: pl.DataFrame | None = None,
        X_forecast: pl.DataFrame | None = None,
        **params,
    ) -> ClassProbaReductionForecaster:
        """Fit the forecaster to historical data.

        Encodes categorical targets to integer codes, tabularizes the time
        series, and fits the wrapped sklearn classifier.

        Parameters
        ----------
        y : pl.DataFrame
            Target time series with a ``"time"`` column (datetime) and one
            or more categorical (String, Categorical, or Enum) value columns.
        X_actual : pl.DataFrame or None, default=None
            Actual feature observations with a ``"time"`` column aligned
            with ``y``. Processed by the actual transformer to produce
            lags, rolling statistics, and other derived features. If
            ``None``, only target-derived features are used.
        forecasting_horizon : int, default=1
            Number of time steps to forecast into the future.
        X_future : pl.DataFrame or None, default=None
            Known future features with a ``"time"`` column. Deterministic
            values available for past and future dates. Bypasses the
            actual transformer.
        X_forecast : pl.DataFrame or None, default=None
            External forecasts with ``"vintage_time"`` and ``"time"``
            columns. Bypasses the actual transformer.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        self
            The fitted forecaster instance.

        """
        forecasting_horizon = self._validate_fit_params(forecasting_horizon)
        self._warn_inapplicable_step_alignment()

        y_fit, X_fit = y, X_actual
        y_tail: pl.DataFrame | None = None
        X_tail: pl.DataFrame | None = None
        if self.validation_size is not None:
            y_fit, X_fit, y_tail, X_tail = self._prepare_validation_fit(y, X_actual, forecasting_horizon, params)

        # Discover classes from the training head before _pre_fit (which may
        # transform y). The validation tail never contributes classes: the
        # encoder is fitted on the head only, and a tail-only class raises.
        # Use unprefixed (base) column names so panel groups share class labels.
        self.classes_: dict[str, list[str]] = {}
        self.n_classes_: dict[str, int] = {}
        self.label_to_code_: dict[str, dict[str, int]] = {}
        for col in y_fit.columns:
            if col == "time":
                continue
            base_col = col.split("__", 1)[1] if "__" in col else col
            unique_vals = sorted(y_fit[col].drop_nulls().unique().cast(pl.String).to_list())
            if base_col in self.classes_:
                merged = sorted(set(self.classes_[base_col]) | set(unique_vals))
                self.classes_[base_col] = merged
            else:
                self.classes_[base_col] = unique_vals
        for base_col, labels in self.classes_.items():
            self.n_classes_[base_col] = len(labels)
            self.label_to_code_[base_col] = {label: i for i, label in enumerate(labels)}

        # Encode target columns to integer codes for tabularization
        y_encoded = self._encode_target(y_fit)
        y_tail_encoded: pl.DataFrame | None = None
        if y_tail is not None:
            self._check_tail_classes(y_tail)
            y_tail_encoded = self._encode_target(y_tail)

        y_t, X_t = self._pre_fit(
            y=y_encoded,
            X_actual=X_fit,
            forecasting_horizon=forecasting_horizon,
            X_future=X_future,
            X_forecast=X_forecast,
        )

        eval_data = None
        if y_tail_encoded is not None:
            eval_data = self._build_validation_eval_data(
                y_t, X_t, y_tail_encoded, X_tail, forecasting_horizon, X_future, X_forecast
            )

        self.estimator_ = self._estimator_fit_one(
            y_t,
            X_t,
            forecasting_horizon,
            estimator_fit_params=params,
            eval_data=eval_data,
        )

        return self

    def _check_tail_classes(self, y_tail: pl.DataFrame) -> None:
        """Reject validation-tail classes the head-fitted encoder never saw.

        Parameters
        ----------
        y_tail : pl.DataFrame
            The held-out raw (unencoded) target rows.

        Raises
        ------
        ValueError
            If any target class occurs only inside the validation tail.

        """
        for col in y_tail.columns:
            if col == "time":
                continue
            base_col = col.split("__", 1)[1] if "__" in col else col
            known = set(self.classes_.get(base_col, []))
            tail_vals = set(y_tail[col].drop_nulls().unique().cast(pl.String).to_list())
            unseen = sorted(tail_vals - known)
            if unseen:
                raise ValueError(
                    f"Target column {col!r} contains class(es) {unseen} that occur "
                    f"only inside the validation_size holdout tail. The label "
                    f"encoder is fitted on the training head only, so every class "
                    f"must appear before the holdout. Reduce validation_size or "
                    f"provide more data."
                )

    def _encode_target(self, y: pl.DataFrame) -> pl.DataFrame:
        """Encode categorical target columns to float codes.

        Parameters
        ----------
        y : pl.DataFrame
            Target data with categorical columns.

        Returns
        -------
        pl.DataFrame
            Target data with categorical columns replaced by float codes.

        """
        return self._apply_label_encoding(y)

    def _predict_class_proba_one(
        self,
        groups: list[str],
        **params,
    ) -> pl.DataFrame:
        """Produce probability forecasts for one fit-horizon block.

        Parameters
        ----------
        groups : list of str
            Panel group names to predict for.
        **params : dict
            Metadata to route to nested estimators.

        Returns
        -------
        pl.DataFrame
            Probability predictions with ``"vintage_time"``, ``"time"``,
            and columns ``{target}_proba_{class_label}`` for each class.

        """
        y_proba = self._estimator_predict_proba_one(
            self.estimator_,
            groups=groups,
        )
        y_proba = self._add_time_columns(y_proba)
        return y_proba

    def _estimator_predict_proba_one(
        self,
        estimator: BaseEstimator | list[BaseEstimator],
        groups: list[str],
    ) -> pl.DataFrame:
        """Dispatch estimator probability prediction to the strategy-specific method.

        Parameters
        ----------
        estimator : BaseEstimator or list[BaseEstimator]
            Fitted estimator(s).
        groups : list of str
            Panel group names to predict for.

        Returns
        -------
        pl.DataFrame
            Probability predictions.

        """
        if self.reduction_strategy == "direct":
            if not isinstance(estimator, list):
                raise TypeError(
                    f"Expected a list of estimators for the 'direct' strategy, got {type(estimator).__name__}."
                )
            return self._estimator_predict_proba_direct(cast(list[BaseEstimator], estimator), groups)
        if not isinstance(estimator, BaseEstimator):
            raise TypeError(
                f"Expected a single estimator for the 'multi-output' strategy, got {type(estimator).__name__}."
            )
        return self._estimator_predict_proba_multi_output(estimator, groups)

    def _estimator_predict_proba_multi_output(
        self,
        estimator: BaseEstimator,
        groups: list[str],
    ) -> pl.DataFrame:
        """Generate probability predictions using a fitted multi-output estimator.

        Parameters
        ----------
        estimator : BaseEstimator
            Fitted sklearn classifier.
        groups : list of str
            Panel group names to predict for.

        Returns
        -------
        pl.DataFrame
            Probability predictions.

        """
        if self.groups_ is None:
            X_tab = self._get_predict_features()
            return self._predict_proba_and_reshape(estimator, X_tab)

        y_pred_dict: dict[str, pl.DataFrame] = {}
        for panel_group_name in groups:
            X_tab = self._get_predict_features(panel_group_name)
            y_pred_dict[panel_group_name] = self._predict_proba_and_reshape(estimator, X_tab, panel_group_name)
        return pl.concat(list(y_pred_dict.values()), how="horizontal")

    def _estimator_predict_proba_direct(
        self,
        estimators: list[BaseEstimator],
        groups: list[str],
    ) -> pl.DataFrame:
        """Generate probability predictions using H independent direct estimators.

        Parameters
        ----------
        estimators : list[BaseEstimator]
            H fitted estimators, one per horizon step.
        groups : list of str
            Panel group names to predict for.

        Returns
        -------
        pl.DataFrame
            Probability predictions.

        Notes
        -----
        The output schema differs between panel and non-panel modes. In
        non-panel mode the per-step single-row frames are stacked
        vertically, yielding H rows (one per horizon step) of global
        probability columns. In panel mode each group's H per-step rows are
        first stacked vertically, then the per-group blocks are concatenated
        horizontally, so every group's probability columns sit side by side
        in the same H rows.

        """
        if self.groups_ is None:
            X_tab = self._get_predict_features()
            frames = []
            for estimator in estimators:
                frames.append(self._predict_proba_and_reshape_single_step(estimator, X_tab))
            return pl.concat(frames)

        y_pred_dict: dict[str, list[pl.DataFrame]] = {g: [] for g in groups}
        for panel_group_name in groups:
            X_tab = self._get_predict_features(panel_group_name)
            for estimator in estimators:
                y_pred_dict[panel_group_name].append(
                    self._predict_proba_and_reshape_single_step(estimator, X_tab, panel_group_name)
                )
        return pl.concat(
            [pl.concat(v) for v in y_pred_dict.values()],
            how="horizontal",
        )

    def _predict_proba_and_reshape(
        self,
        estimator: BaseEstimator,
        X_tab: pl.DataFrame,
        panel_group_name: str | None = None,
    ) -> pl.DataFrame:
        """Call predict_proba and reshape to probability DataFrame.

        For multi-output, the estimator predicts all H steps at once.
        Each step has n_targets columns; each target column's integer
        prediction maps to n_classes probability columns.

        Parameters
        ----------
        estimator : BaseEstimator
            Fitted classifier.
        X_tab : pl.DataFrame
            Feature DataFrame, typically of shape ``(1, n_features)`` at
            predict time (the shape is a caller convention, not enforced here).
        panel_group_name : str or None
            Panel group prefix for column naming.

        Returns
        -------
        pl.DataFrame
            Probability DataFrame with ``{target}_proba_{class}`` columns.

        Notes
        -----
        ``classes_`` is the *globally merged* class set across every panel
        group (see ``fit``); each group's underlying sklearn classifier may
        have been trained on a subset of those classes, so its
        ``predict_proba`` can return fewer columns than ``len(classes_)``.
        When a global class index has no corresponding column from the
        estimator (``c_idx >= len(step_proba)``), its probability is filled
        with ``0.0``: the class was unseen by this group, so it carries zero
        mass. This is expected reconciliation, not data loss, and keeps the
        per-group output aligned to the shared global class layout.

        The non-list (single-output) branch only emits a row when
        ``proba`` has shape ``(1, n_classes)``, i.e. ``fh=1`` with a
        single-target estimator; ``fh>1`` multi-step stepping is handled by
        the recursive loop in ``predict_class_proba`` (via a
        ``MultiOutputClassifier``, which takes the list path). If neither
        condition holds, this method returns an empty DataFrame.

        """
        assert self.local_y_t_schema_ is not None
        y_cols = list(self.local_y_t_schema_.keys())
        fh = self.fit_forecasting_horizon_
        n_targets = len(y_cols)

        # For multi-output with H*n_targets outputs, sklearn wraps in
        # MultiOutputClassifier or similar. We handle both cases.
        proba = estimator.predict_proba(X_tab)  # ty: ignore[unresolved-attribute]

        # Build result row by row (one row per forecast step)
        result_data: dict[str, list[Any]] = {}
        for target_col in y_cols:
            for label in self.classes_[target_col]:
                col_name = f"{target_col}_proba_{label}"
                if panel_group_name is not None:
                    col_name = f"{panel_group_name}__{col_name}"
                result_data[col_name] = []

        if isinstance(proba, list):
            # MultiOutputClassifier returns list of arrays, one per output
            # Each array has shape (1, n_classes_for_that_output)
            # Outputs are ordered: target1_step1, target2_step1, ..., target1_step2, ...
            for step in range(fh):
                for t_idx, target_col in enumerate(y_cols):
                    output_idx = step * n_targets + t_idx
                    step_proba = proba[output_idx][0]  # shape (n_classes,)
                    classes_for_target = self.classes_[target_col]
                    for c_idx, label in enumerate(classes_for_target):
                        col_name = f"{target_col}_proba_{label}"
                        if panel_group_name is not None:
                            col_name = f"{panel_group_name}__{col_name}"
                        if c_idx < len(step_proba):
                            result_data[col_name].append(float(step_proba[c_idx]))
                        else:
                            # Global class unseen by this group's estimator; zero mass.
                            result_data[col_name].append(0.0)
        else:
            # Single-output classifier or single-target: proba shape (1, n_classes)
            # For multi-step, we need H rows from a single prediction. If
            # multi-output is used, the model predicts H*n_targets columns
            # and predict_proba returns a single array.
            # Fall back: treat as single-step single-target
            assert n_targets == 1
            target_col = y_cols[0]
            classes_for_target = self.classes_[target_col]

            if proba.ndim == 2 and proba.shape[0] == 1:
                # Single row prediction, map to fh=1
                step_proba = proba[0]
                for c_idx, label in enumerate(classes_for_target):
                    col_name = f"{target_col}_proba_{label}"
                    if panel_group_name is not None:
                        col_name = f"{panel_group_name}__{col_name}"
                    result_data[col_name].append(float(step_proba[c_idx]) if c_idx < len(step_proba) else 0.0)
                # This branch builds a single row and is only reached with
                # fh=1 in single-output mode; multi-step stepping is handled
                # by the recursive loop in predict_class_proba.

        return pl.DataFrame(result_data)

    def _predict_proba_and_reshape_single_step(
        self,
        estimator: BaseEstimator,
        X_tab: pl.DataFrame,
        panel_group_name: str | None = None,
    ) -> pl.DataFrame:
        """Call predict_proba for a single-step direct estimator.

        Parameters
        ----------
        estimator : BaseEstimator
            Fitted single-step classifier.
        X_tab : pl.DataFrame
            Feature DataFrame, typically of shape ``(1, n_features)`` at
            predict time (the shape is a caller convention, not enforced here).
        panel_group_name : str or None
            Panel group prefix for column naming.

        Returns
        -------
        pl.DataFrame
            Single-row probability DataFrame.

        Notes
        -----
        As in ``_predict_proba_and_reshape``, a global class index without a
        matching column from the estimator (``c_idx >= len(step_proba)``) is
        filled with ``0.0``. This reconciles a group whose estimator saw only
        a subset of the globally merged ``classes_``; the unseen class
        correctly carries zero mass rather than being dropped.

        """
        assert self.local_y_t_schema_ is not None
        y_cols = list(self.local_y_t_schema_.keys())

        proba = estimator.predict_proba(X_tab)  # ty: ignore[unresolved-attribute]

        result_data: dict[str, list[float]] = {}

        if isinstance(proba, list):
            # Multiple targets
            for t_idx, target_col in enumerate(y_cols):
                step_proba = proba[t_idx][0]
                for c_idx, label in enumerate(self.classes_[target_col]):
                    col_name = f"{target_col}_proba_{label}"
                    if panel_group_name is not None:
                        col_name = f"{panel_group_name}__{col_name}"
                    result_data[col_name] = [float(step_proba[c_idx]) if c_idx < len(step_proba) else 0.0]
        else:
            # Single target
            assert len(y_cols) == 1
            target_col = y_cols[0]
            step_proba = proba[0]
            for c_idx, label in enumerate(self.classes_[target_col]):
                col_name = f"{target_col}_proba_{label}"
                if panel_group_name is not None:
                    col_name = f"{panel_group_name}__{col_name}"
                result_data[col_name] = [float(step_proba[c_idx]) if c_idx < len(step_proba) else 0.0]

        return pl.DataFrame(result_data)
