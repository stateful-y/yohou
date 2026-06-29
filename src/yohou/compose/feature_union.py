"""FeaturePipeline and FeatureUnion utilities for chaining transformers."""

import numbers
from collections import Counter
from collections.abc import Iterator
from typing import Any

import polars as pl
import polars.selectors as cs
from sklearn.pipeline import (
    FeatureUnion as sklearn_FeatureUnion,
)
from sklearn.utils import (
    Bunch,
)
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    process_routing,
)
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    check_is_fitted,
)

from yohou.base import BaseTransformer
from yohou.utils import Tags
from yohou.utils._compat import (
    _BaseComposition,
    _fit_one,
    _fit_transform_one,
    _raise_for_params,
    _transform_one,
)
from yohou.utils.panel import panel_aware_prefix

from .utils import _hstack, _observe_transform_one, _rewind_transform_one

__all__ = ["FeatureUnion"]


class FeatureUnion(BaseTransformer, _BaseComposition):
    """Concatenates results of multiple transformer objects.

    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.

    Parameters of the transformers may be set using its name and the parameter
    name separated by a '__'. A transformer may be replaced entirely by
    setting the parameter with its name to another transformer, removed by
    setting to 'drop' or disabled by setting to 'passthrough' (features are
    passed without transformation).

    Parameters
    ----------
    transformer_list : list of (str, transformer) tuples
        List of transformer objects to be applied to the data. The first
        half of each tuple is the name of the transformer. The transformer can
        be 'drop' for it to be ignored or can be 'passthrough' for features to
        be passed unchanged.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a ``joblib.parallel_backend`` context.
        ``-1`` means using all processors.

    transformer_weights : dict, default=None
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.
        Raises ValueError if key not present in ``transformer_list``.

    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    verbose_feature_names_out : bool, default=True
        If True, `get_feature_names_out` will prefix all feature names
        with the name of the transformer that generated that feature
        using a single underscore separator (e.g., ``lags_sales``).
        For panel data columns, the prefix is inserted after the group
        separator to preserve panel structure
        (e.g., ``store_1__lags_sales``).
        If False, `get_feature_names_out` will not prefix any feature
        names and will error if feature names are not unique.

    Attributes
    ----------
    named_transformers : `Bunch`
        Dictionary-like object, with the following attributes.
        Read-only attribute to access any transformer parameter by user
        given name. Keys are transformer names and values are
        transformer parameters.

    n_features_in_ : int
        Number of features seen during ``fit``. Only defined if the
        underlying first transformer in `transformer_list` exposes such an
        attribute when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when
        `X` has feature names that are all strings.

    See Also
    --------
    `sklearn.pipeline.FeatureUnion` : Underlying scikit-learn feature union class.
    - [`FeaturePipeline`][yohou.compose.feature_pipeline.FeaturePipeline] : Sequential transformer chaining.
    - [`BaseTransformer`][yohou.base.transformer.BaseTransformer] : Base class for transformers.
    - [`LagTransformer`][yohou.preprocessing.window.LagTransformer] : Common transformer for lag features.

    Notes
    -----
    Transformers run in parallel when `n_jobs` is set to a value other than 1.
    This can significantly improve performance for computationally expensive transformers.

    Results are concatenated horizontally with automatic time alignment. The
    internal `_hstack()` function aligns transformer outputs by taking the
    intersection of their `"time"` columns, so only timestamps present in all
    transformer outputs are retained.

    The `observation_horizon` property returns the MAXIMUM across all transformers
    (not the sum). This is because all transformers operate on the same input data,
    and the union needs enough history to satisfy the most demanding transformer.

    Useful for multi-scale feature engineering, such as combining short-term and
    long-term lag features, or mixing different preprocessing approaches in parallel.

    All transformers must accept the same input time series with a `time` column.

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import datetime, timedelta
    >>> from yohou.compose import FeatureUnion
    >>> from yohou.preprocessing import LagTransformer
    >>>
    >>> # Create sample weekly time series data (52 weeks)
    >>> time = pl.datetime_range(
    ...     start=datetime(2023, 1, 1),
    ...     end=datetime(2023, 1, 1) + timedelta(weeks=51),
    ...     interval="1w",
    ...     eager=True,
    ... )
    >>> data = pl.DataFrame({"time": time, "demand": range(1, 53)})
    >>>
    >>> # Example 1: Combine short-term and long-term lags for multi-scale features
    >>> union = FeatureUnion([
    ...     ("short_lags", LagTransformer(lag=[1, 2, 3])),
    ...     ("long_lags", LagTransformer(lag=[7, 14, 21])),
    ... ])
    >>>
    >>> # Example 2: Access transformers by name
    >>> union.named_transformers["short_lags"]  # doctest: +ELLIPSIS
    LagTransformer(...)
    >>>
    >>> # Example 3: Access transformers by position
    >>> union[0]  # doctest: +ELLIPSIS
    LagTransformer(...)

    """

    _required_parameters = ["transformer_list"]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict[str, Any]
            Parameter names mapped to their values.

        """
        return _BaseComposition._get_params(self, attr="transformer_list", deep=deep)

    def set_params(self, **params: Any) -> "FeatureUnion":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : FeatureUnion
            FeatureUnion instance.

        """
        _BaseComposition._set_params(self, attr="transformer_list", **params)
        return self

    def _iter(self) -> Iterator[tuple[str, Any, float]]:
        """Generate (name, trans, weight) tuples excluding None and 'drop' transformers.

        Yields
        ------
        name : str
            Transformer name.
        trans : Any
            Transformer instance.
        weight : float
            Transformer weight.

        """
        return sklearn_FeatureUnion._iter(self)  # ty: ignore[invalid-argument-type]

    def __getitem__(self, ind: int | str | slice) -> Any:
        """Return a sub-union or a single transformer.

        Parameters
        ----------
        ind : int, str, or slice
            Index, name, or slice of the transformer to retrieve.

        Returns
        -------
        transformer : Any
            The transformer or sub-union.

        """
        if isinstance(ind, slice):
            if ind.step is not None and ind.step != 1:
                raise ValueError("FeatureUnion slicing only supports a step of 1")
            return self.__class__(
                transformer_list=self.transformer_list[ind],
                n_jobs=self.n_jobs,
                transformer_weights=self.transformer_weights,
                verbose=self.verbose,
            )
        elif isinstance(ind, int):
            _, est = self.transformer_list[ind]
            return est
        else:
            # String case - get by name
            return self.named_transformers[ind]

    @property
    def named_transformers(self) -> Bunch:
        """Access the transformers by name.

        Returns
        -------
        named_transformers : Bunch
            Dictionary-like object with transformer names as keys.

        """
        return Bunch(**dict(self.transformer_list))

    def _log_message(self, name: str, idx: int, total: int) -> str:
        """Get log message for a transformer.

        Parameters
        ----------
        name : str
            Transformer name.
        idx : int
            Current index.
        total : int
            Total number of transformers.

        Returns
        -------
        message : str
            Log message.

        """
        return f"(step {idx} of {total}) Processing {name}"

    def _parallel_func(self, X: pl.DataFrame, y: pl.DataFrame | None, func: Any, routed_params: Any) -> Any:
        """Run func in parallel on X and y.

        Parameters
        ----------
        X : pl.DataFrame
            Input data.
        y : pl.DataFrame | None
            Target data.
        func : Any
            Function to apply.
        routed_params : Any
            Routed parameters.

        Returns
        -------
        results : Any
            Results from parallel execution.

        """
        return sklearn_FeatureUnion._parallel_func(self, X, y, func, routed_params)  # ty: ignore[invalid-argument-type]

    def _update_transformer_list(self, transformers: Any) -> None:
        """Update transformer_list with fitted transformers.

        Parameters
        ----------
        transformers : Any
            Fitted transformers.

        """
        transformers_iter = iter(transformers)
        self.transformer_list[:] = [
            (name, next(transformers_iter) if old is not None else None) for name, old in self.transformer_list
        ]

    def get_feature_names_out(self, input_features: list[str] | None = None) -> Any:
        """Get output feature names.

        Parameters
        ----------
        input_features : list[str] | None, default=None
            Input feature names.

        Returns
        -------
        feature_names_out : Any
            Output feature names.

        """
        return super().get_feature_names_out(input_features)

    @property
    def n_features_in_(self) -> int:
        """Number of features seen during fit.

        Returns
        -------
        n_features_in_ : int
            Number of input features.

        """
        # Delegate to first transformer
        for _, trans in self.transformer_list:
            if hasattr(trans, "n_features_in_"):
                return trans.n_features_in_
        raise AttributeError("n_features_in_ not available")

    @property
    def feature_names_in_(self) -> Any:
        """Names of features seen during fit.

        Returns
        -------
        feature_names_in_ : Any
            Names of input features.

        """
        for _, trans in self.transformer_list:
            if hasattr(trans, "feature_names_in_"):
                return trans.feature_names_in_
        raise AttributeError("feature_names_in_ not available")

    def _add_prefix_for_feature_names_out(self, feature_names_out: list[list[str]]) -> list[str]:
        """Add prefixes to feature names.

        Uses single underscore ``_`` as separator (not ``__``) to avoid
        conflicts with the panel data ``<GROUP>__<SERIES>`` convention.
        For panel columns, the prefix is inserted after the group separator
        (e.g., ``store_1__lags_sales``).

        Parameters
        ----------
        feature_names_out : list[list[str]]
            Feature names from each transformer.

        Returns
        -------
        prefixed_names : list[str]
            Feature names with prefixes.

        """
        return [panel_aware_prefix(col, name) for name, cols in feature_names_out for col in cols]

    def __sklearn_tags__(self) -> Tags:
        """Get estimator tags.

        Returns
        -------
        Tags
            Estimator tags with yohou-specific attributes.

        """
        tags = super().__sklearn_tags__()

        # Aggregate tags from transformers (static capability check)
        if hasattr(self, "transformer_list") and self.transformer_list is not None:
            transformers = [t for _, t in self.transformer_list if t not in ("drop", "passthrough") and t is not None]
            if transformers:
                assert tags.transformer_tags is not None
                assert tags.input_tags is not None
                # Stateful if any transformer is stateful
                tags.transformer_tags.stateful = any(
                    t.__sklearn_tags__().transformer_tags.stateful for t in transformers
                )

                # Not invertible unless there is only one transformer and it is invertible
                tags.transformer_tags.invertible = (
                    len(transformers) == 1 and transformers[0].__sklearn_tags__().transformer_tags.invertible
                )

                # Aggregate min_value: take the maximum (most restrictive)
                # All transformers receive the same input, so we need to satisfy all constraints
                min_values = [t.__sklearn_tags__().input_tags.min_value for t in transformers]
                non_none_min_values = [v for v in min_values if v is not None]
                tags.input_tags.min_value = max(non_none_min_values) if non_none_min_values else None

        return tags

    def __sklearn_is_fitted__(self) -> bool:
        """Check if fitted.

        Returns
        -------
        is_fitted : bool
            True if the union is fitted.

        """
        return sklearn_FeatureUnion.__sklearn_is_fitted__(self)  # ty: ignore[invalid-argument-type]

    def _sk_visual_block_(self) -> Any:
        """Get visual block representation.

        Returns
        -------
        visual_block : Any
            Visual block representation.

        """
        return sklearn_FeatureUnion._sk_visual_block_(self)  # ty: ignore[invalid-argument-type]

    def _get_observation_horizons(self) -> list[int]:
        """Get observation horizons from all transformers.

        Returns
        -------
        observation_horizons : list[int]
            List of observation horizons from each transformer.

        """
        observation_horizons = []
        for _, t, _ in self._iter():
            # _iter() already replaces 'passthrough' with a FunctionTransformer
            # and skips 'drop'/None, so t is always a transformer here.
            observation_horizon = t.observation_horizon if hasattr(t, "observation_horizon") else 0
            observation_horizons.append(observation_horizon)

        return observation_horizons

    def _build_column_names(self, Xs: list[pl.DataFrame]) -> list[list[str]]:
        """Compute per-transformer output column names.

        Extracts each transformer's non-time output columns and, when
        ``verbose_feature_names_out`` is True, prefixes them with the
        transformer name. When False, raises if any output names collide.

        Parameters
        ----------
        Xs : list of pl.DataFrame
            Transformed outputs, one per transformer (each with a ``"time"``
            column).

        Returns
        -------
        list of list of str
            Final column names for each transformer, in order.

        """
        transformer_names = [name for name, _, _ in self._iter()]
        raw_column_names = [[col for col in X_t.columns if col != "time"] for X_t in Xs]

        if self.verbose_feature_names_out:
            return [
                [panel_aware_prefix(col, name) for col in cols]
                for name, cols in zip(transformer_names, raw_column_names, strict=False)
            ]

        flat_names = [col for cols in raw_column_names for col in cols]
        counts = Counter(flat_names)
        duplicates = [name for name, count in counts.items() if count > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate feature names found: {duplicates}. "
                "Either use transformers that produce unique names or set "
                "verbose_feature_names_out=True to add transformer name prefixes."
            )
        return raw_column_names

    @property
    def observation_horizon(self) -> int:
        """Maximum observation horizon across all transformers.

        Returns
        -------
        int
            Maximum observation horizon needed.

        Raises
        ------
        NotFittedError
            If the feature union has not been fitted yet.

        """
        check_is_fitted(self)

        observation_horizons = self._get_observation_horizons()
        observation_horizon = max(observation_horizons, default=0)

        return observation_horizon

    _parameter_constraints: dict = {
        "transformer_list": [list],
        "n_jobs": [numbers.Integral, None],
        "transformer_weights": [dict, None],
        "verbose": ["boolean"],
        "verbose_feature_names_out": ["boolean"],
    }

    def __init__(
        self,
        transformer_list: list[tuple[str, Any]],
        *,
        n_jobs: int | None = None,
        transformer_weights: dict[str, float] | None = None,
        verbose: bool = False,
        verbose_feature_names_out: bool = True,
    ) -> None:
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self.verbose_feature_names_out = verbose_feature_names_out

    def _validate_transformers(self) -> None:
        """Validate all transformers are BaseTransformer instances.

        Raises
        ------
        TypeError
            If any transformer is invalid.

        """
        names, transformers = zip(*self.transformer_list, strict=False)

        # validate names
        self._validate_names(names)

        # validate estimators
        for t in transformers:
            if t in ("drop", "passthrough"):
                continue
            if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(t, "transform"):
                raise TypeError(f"All estimators should implement fit and transform. '{t}' (type {type(t)}) doesn't")

    def _validate_transformer_weights(self) -> None:
        """Validate transformer weights dictionary.

        Raises
        ------
        ValueError
            If weight keys don't match transformer names.

        """
        if not self.transformer_weights:
            return

        transformer_names = {name for name, _ in self.transformer_list}
        for name in self.transformer_weights:
            if name not in transformer_names:
                raise ValueError(
                    f'Attempting to weight transformer "{name}", but it is not present in transformer_list.'
                )

    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **fit_params: Any) -> "FeatureUnion":
        """Fit all transformers using X.

        Parameters
        ----------
        X : pl.DataFrame
            Input data with a mandatory ``"time"`` column, used to fit
            transformers.

        y : pl.DataFrame or None, default=None
            Target time series passed to the sub-transformers.

        **fit_params : dict, default=None
            Parameters routed to the `fit` methods of the sub-transformers via
            the metadata routing API. See the sklearn Metadata Routing User
            Guide for more details.

        Returns
        -------
        self : FeatureUnion
            FeatureUnion class instance.
        """
        _raise_for_params(fit_params, self, "fit")
        routed_params = process_routing(self, "fit", **fit_params)
        transformers = self._parallel_func(X, y, _fit_one, routed_params)

        if not transformers:
            # All transformers are None
            return self

        self._update_transformer_list(transformers)
        return self

    def fit_transform(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params: object) -> pl.DataFrame:
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : pl.DataFrame
            Input data with a mandatory ``"time"`` column to be transformed.

        y : pl.DataFrame or None, default=None
            Target time series passed to the sub-transformers.

        **params : dict, default=None
            Parameters routed to the `fit` methods of the sub-transformers via
            the metadata routing API. See the sklearn Metadata Routing User
            Guide for more details.

        Returns
        -------
        X_t : pl.DataFrame
            Polars DataFrame with a ``"time"`` column and the horizontally
            concatenated feature columns from all transformers, aligned to the
            intersection of their time grids.
        """
        routed_params = process_routing(self, "fit_transform", **params)
        results = self._parallel_func(X, y, _fit_transform_one, routed_params)
        if not results:
            # All transformers are None
            time = X.select(cs.by_name("time"))
            return time

        Xs, transformers = zip(*results, strict=False)
        self._update_transformer_list(transformers)

        column_names = self._build_column_names(list(Xs))

        result = _hstack(list(Xs), column_names=column_names)
        return result

    def transform(self, X: pl.DataFrame, **params: Any) -> pl.DataFrame:
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : pl.DataFrame
            Input data with a mandatory ``"time"`` column to be transformed.

        **params : dict, default=None
            Parameters routed to the `transform` method of the sub-transformers via the
            metadata routing API. See [Metadata Routing User Guide](https://scikit-learn.org/stable/metadata_routing.html) for more details.

        Returns
        -------
        X_t : pl.DataFrame
            Polars DataFrame with a ``"time"`` column and the horizontally
            concatenated feature columns from all transformers, aligned to the
            intersection of their time grids.
        """
        _raise_for_params(params, self, "transform")
        routed_params = process_routing(self, "transform", **params)

        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight, routed_params[name]) for name, trans, weight in self._iter()
        )
        if not Xs:
            # All transformers are None
            time = X.select(cs.by_name("time"))
            return time

        column_names = self._build_column_names(list(Xs))

        result = _hstack(Xs, column_names=column_names)
        return result

    def observe_transform(self, X: pl.DataFrame, **params: Any) -> pl.DataFrame:
        """Observe and transform X in parallel for each transformer, concatenate results.

        This method atomically observes each transformer with new data and
        transforms it, concurrently when ``n_jobs != 1`` and sequentially
        otherwise. The transformation uses the pre-observe state, then updates
        the memory. This is more efficient and correct than calling observe()
        then transform() separately.

        Parameters
        ----------
        X : pl.DataFrame
            New data to observe with and transform.

        **params : dict, default=None
            Parameters routed to the `observe_transform` methods of the sub-transformers
            via the metadata routing API. See [Metadata Routing User Guide](https://scikit-learn.org/stable/metadata_routing.html) for more details.

        Returns
        -------
        X_t : pl.DataFrame
            Horizontally stacked results of transformers, aligned by observation horizons.

        """
        _raise_for_params(params, self, "observe_transform")
        routed_params = process_routing(self, "observe_transform", **params)

        # Parallel execution of observe_transform on all transformers
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_observe_transform_one)(trans, X, None, weight, routed_params[name])
            for name, trans, weight in self._iter()
        )

        if not Xs:
            # All transformers are None
            time = X.select(cs.by_name("time"))
            return time

        column_names = self._build_column_names(list(Xs))

        result = _hstack(Xs, column_names=column_names)

        return result

    def rewind_transform(self, X: pl.DataFrame, **params: Any) -> pl.DataFrame:
        """Rewind and transform X in parallel for each transformer, concatenate results.

        This method applies rewind_transform semantics to each transformer in parallel:
        transforms from scratch without using pre-existing memory, discards warmup rows,
        and rewinds the internal state with the input data.

        Parameters
        ----------
        X : pl.DataFrame
            Data to transform and use for rewinding state.

        **params : dict, default=None
            Parameters routed to the `rewind_transform` methods of the sub-transformers
            via the metadata routing API. See [Metadata Routing User Guide](https://scikit-learn.org/stable/metadata_routing.html) for more details.

        Returns
        -------
        X_t : pl.DataFrame
            Horizontally stacked results of transformers, aligned by observation horizons,
            with warmup rows discarded.

        """
        _raise_for_params(params, self, "rewind_transform")
        routed_params = process_routing(self, "rewind_transform", **params)

        # Parallel execution of rewind_transform on all transformers
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_rewind_transform_one)(trans, X, None, weight, routed_params[name])
            for name, trans, weight in self._iter()
        )

        if not Xs:
            # All transformers are None
            time = X.select(cs.by_name("time"))
            return time

        column_names = self._build_column_names(list(Xs))

        result = _hstack(Xs, column_names=column_names)

        return result

    def observe(self, X: pl.DataFrame) -> "FeatureUnion":
        """Observe new data, fanning out to each child transformer.

        FeatureUnion holds no buffer of its own; the union's stateful memory is
        the union of its children's buffers. This override fans ``observe`` out
        to each child so their memory advances. The inherited
        ``BaseTransformer.observe`` cannot be used because it guards on
        ``X_schema_``, which FeatureUnion never sets.

        Parameters
        ----------
        X : pl.DataFrame
            New data with a mandatory ``"time"`` column to observe.

        Returns
        -------
        self : FeatureUnion
            The union with each child transformer's memory advanced.

        """
        check_is_fitted(self)
        for _, transformer, _ in self._iter():
            if hasattr(transformer, "observe"):
                transformer.observe(X)
        return self

    def rewind(self, X: pl.DataFrame) -> "FeatureUnion":
        """Rewind internal memory, fanning out to each child transformer.

        Mirrors :meth:`observe` by delegating to each child's ``rewind`` so the
        union's per-child buffers are rolled back to the provided window. The
        inherited ``BaseTransformer.rewind`` cannot be used because it guards on
        ``X_schema_``, which FeatureUnion never sets.

        Parameters
        ----------
        X : pl.DataFrame
            Data with a mandatory ``"time"`` column to rewind state to.

        Returns
        -------
        self : FeatureUnion
            The union with each child transformer's memory rewound.

        """
        check_is_fitted(self)
        for _, transformer, _ in self._iter():
            if hasattr(transformer, "rewind"):
                transformer.rewind(X)
        return self

    def get_metadata_routing(self) -> MetadataRouter:
        """Get metadata routing of this object.

        Please check [Metadata Routing User Guide](https://scikit-learn.org/stable/metadata_routing.html) on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A `MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self)

        for name, transformer in self.transformer_list:
            router.add(
                **{name: transformer},
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="fit_transform", callee="fit_transform")
                .add(caller="fit_transform", callee="fit")
                .add(caller="fit_transform", callee="transform")
                .add(caller="transform", callee="transform")
                .add(caller="observe_transform", callee="observe_transform")
                .add(caller="rewind_transform", callee="rewind_transform"),
            )

        return router
