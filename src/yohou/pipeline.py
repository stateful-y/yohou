"""Pipeline utilities for chaining transformers and forecasters."""

from collections import Counter
from copy import deepcopy
from itertools import chain
from numbers import Integral

import polars as pl
import polars.selectors as cs
from joblib import Memory
from sklearn.base import _fit_context, clone
from sklearn.compose._column_transformer import (
    ColumnTransformer as sklearn_ColumnTransformer,
)
from sklearn.compose._column_transformer import (
    _check_X,
)
from sklearn.pipeline import (
    FeatureUnion as sklearn_FeatureUnion,
)
from sklearn.pipeline import (
    Pipeline as sklearn_Pipeline,
)
from sklearn.pipeline import (
    _fit_one,
    _fit_transform_one,
    _transform_one,
)
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import (
    Bunch,
    _safe_indexing,
)
from sklearn.utils._dataframe import is_pandas_df
from sklearn.utils._param_validation import HasMethods, Hidden, StrOptions
from sklearn.utils._set_output import (
    _get_output_config,
)
from sklearn.utils._user_interface import _print_elapsed_time
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    process_routing,
)
from sklearn.utils.metaestimators import _BaseComposition, available_if
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_feature_names,
    _check_n_features,
    _get_feature_names,
    _num_samples,
    check_is_fitted,
)

from yohou.base import BaseTransformer

__all__ = ["Pipeline", "FeatureUnion", "ColumnTransformer"]


class Pipeline(BaseTransformer, _BaseComposition):  # type: ignore[misc]
    """
    A sequence of data transformers with an optional final predictor.

    `Pipeline` allows you to sequentially apply a list of time series
    transformers to preprocess the data.

    Steps of the pipeline must be 'transforms', that is, they must implement
    `fit`, `transform` and `update` methods.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters. For this, it
    enables setting parameters of the various steps using their names and the
    parameter name separated by a `'__'`, as in the example below. A step's
    estimator may be replaced entirely by setting the parameter with its name
    to another estimator, or a transformer removed by setting it to
    `'passthrough'` or `None`.

    Parameters
    ----------
    steps : list of tuples
        List of (name of step, estimator) tuples that are to be chained in
        sequential order. To be compatible with the scikit-learn API, all steps
        must define `fit`. All non-last steps must also define `transform`. See
        :ref:`Combining Estimators <combining_estimators>` for more details.

    transform_input : list of str, default=None
        The names of the :term:`metadata` parameters that should be transformed by the
        pipeline before passing it to the step consuming it.

        This enables transforming some input arguments to ``fit`` (other than ``X``)
        to be transformed by the steps of the pipeline up to the step which requires
        them. Requirement is defined via :ref:`metadata routing <metadata_routing>`.
        For instance, this can be used to pass a validation set through the pipeline.

        You can only set this if metadata routing is enabled, which you
        can enable using ``sklearn.set_config(enable_metadata_routing=True)``.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. The last step
        will never be cached, even if it is a transformer. By default, no
        caching is performed. If a string is given, it is the path to the
        caching directory. Enabling caching triggers a clone of the transformers
        before fitting. Therefore, the transformer instance given to the
        pipeline cannot be inspected directly. Use the attribute ``named_steps``
        or ``steps`` to inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Attributes
    ----------
    named_steps : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying first estimator in `steps` exposes such an attribute
        when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

    """

    # BaseEstimator interface
    _required_parameters = ["steps"]

    _parameter_constraints: dict[str, object] = {
        "steps": [list, Hidden(tuple)],
        "transform_input": [list, None],
        "memory": [None, str, HasMethods(["cache"])],
        "verbose": ["boolean"],
    }

    def __init__(
        self,
        steps: list[tuple[str, object]],
        *,
        transform_input: list[str] | None = None,
        memory: None | Memory | str = None,
        verbose: bool = False,
    ) -> None:
        """Initialize pipeline.

        Parameters
        ----------
        steps : list of tuple
            List of (name, transformer) tuples.

        transform_input : list of str or None, default=None
            Which inputs to transform.

        memory : Memory, str, or None, default=None
            Caching mechanism.

        verbose : bool, default=False
            Enable verbose output.

        """
        self.steps = steps
        self.transform_input = transform_input
        self.memory = memory
        self.verbose = verbose

    get_params = sklearn_Pipeline.get_params
    set_params = sklearn_Pipeline.set_params
    _iter = sklearn_Pipeline._iter
    __len__ = sklearn_Pipeline.__len__
    __getitem__ = sklearn_Pipeline.__getitem__
    _fit = sklearn_Pipeline._fit
    named_steps = sklearn_Pipeline.named_steps
    _final_estimator = sklearn_Pipeline._final_estimator
    _log_message = sklearn_Pipeline._log_message
    _check_method_params = sklearn_Pipeline._check_method_params
    get_feature_names_out = sklearn_Pipeline.get_feature_names_out
    n_features_in_ = sklearn_Pipeline.n_features_in_
    feature_names_in_ = sklearn_Pipeline.feature_names_in_
    __sklearn_is_fitted__ = sklearn_Pipeline.__sklearn_is_fitted__
    _sk_visual_block_ = sklearn_Pipeline._sk_visual_block_
    _get_metadata_for_step = sklearn_Pipeline._get_metadata_for_step

    @property
    def observation_horizon(self) -> int:
        """Get cumulative observation horizon across all steps.

        Returns
        -------
        int
            Total observation horizon needed.

        """
        observation_horizon = 0
        for _, t in self.steps:
            if t != "passthrough":
                observation_horizon += t.observation_horizon  # type: ignore[attr-defined]

        return observation_horizon

    def _validate_steps(self) -> None:
        """Validate that all steps are BaseTransformer instances.

        Raises
        ------
        TypeError
            If any step is not a BaseTransformer or 'passthrough'.

        """
        names, transformers = zip(*self.steps)

        # validate names
        self._validate_names(names)

        for t in transformers:
            if t is None or t == "passthrough":
                continue
            if not isinstance(t, BaseTransformer):
                raise TypeError(
                    "All steps should be instances of `BaseTransformer` "
                    "or be the string 'passthrough' "
                    "'%s' (type %s) doesn't" % (t, type(t))
                )

    @_fit_context(  # type: ignore[untyped-decorator]
        # estimators in Pipeline.steps are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params: object) -> "Pipeline":
        """Fit the model.

        Fit all the transformers one after the other and sequentially transform the
        data. Finally, fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters passed to the ``fit`` method of each step, where
                each parameter name is prefixed such that parameter ``p`` for step
                ``s`` has key ``s__p``.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

        Returns
        -------
        self : object
            Pipeline with fitted steps.
        """
        routed_params = self._check_method_params(method="fit", props=params)
        X_t = self._fit(X, y, routed_params)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                last_step_params = routed_params[self.steps[-1][0]]
                self._final_estimator.fit(X_t, y, **last_step_params["fit"])

        return self

    @_fit_context(  # type: ignore[untyped-decorator]
        # estimators in Pipeline.steps are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit_transform(
        self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params: object
    ) -> object:
        """Fit the model and transform with the final estimator.

        Fit all the transformers one after the other and sequentially transform
        the data. Only valid if the final estimator either implements
        `fit_transform` or `fit` and `transform`.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **params : dict of str -> object
            - If `enable_metadata_routing=False` (default):

                Parameters passed to the ``fit`` method of each step, where
                each parameter name is prefixed such that parameter ``p`` for step
                ``s`` has key ``s__p``.

            - If `enable_metadata_routing=True`:

                Parameters requested and accepted by steps. Each step must have
                requested certain metadata for these parameters to be forwarded to
                them.

        Returns
        -------
        X_t : ndarray of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        routed_params = self._check_method_params(method="fit_transform", props=params)
        X_t = self._fit(X, y, routed_params)

        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return X_t

            last_step_params = routed_params[self.steps[-1][0]]
            return last_step.fit_transform(X_t, y, **last_step_params["fit_transform"])

    def transform(self, X: pl.DataFrame, **params: object) -> object:
        """Transform the data, and apply `transform` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `transform` method. Only valid if the final estimator
        implements `transform`.

        This also works where final estimator is `None` in which case all prior
        transformations are applied.

        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.

        **params : dict of str -> object
            Parameters requested and accepted by steps. Each step must have
            requested certain metadata for these parameters to be forwarded to
            them.

        Returns
        -------
        X_t : ndarray of shape (n_samples, n_transformed_features)
            Transformed data.
        """
        _raise_for_params(params, self, "transform")

        # not branching here since params is only available if
        # enable_metadata_routing=True
        routed_params = process_routing(self, "transform", **params)
        X_t = X
        for _, name, transform in self._iter():
            X_t = transform.transform(X_t, **routed_params[name].transform)
        return X_t

    def _can_inverse_transform(self) -> bool:
        """Check if all steps support inverse_transform.

        Returns
        -------
        bool
            True if all steps have inverse_transform method.

        """
        return all(hasattr(t, "inverse_transform") for _, _, t in self._iter())

    @available_if(_can_inverse_transform)  # type: ignore[untyped-decorator]
    def inverse_transform(self, X_t: pl.DataFrame, X_p: pl.DataFrame, **params: object) -> object:
        """Apply `inverse_transform` for each step in a reverse order.

        All estimators in the pipeline must support `inverse_transform`.

        Parameters
        ----------
        X_t : array-like of shape (n_samples, n_transformed_features)
            Data samples, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features. Must fulfill
            input requirements of last step of pipeline's
            ``inverse_transform`` method.

        **params : dict of str -> object
            Parameters requested and accepted by steps. Each step must have
            requested certain metadata for these parameters to be forwarded to
            them.

        Returns
        -------
        X_t : ndarray of shape (n_samples, n_features)
            Inverse transformed data, that is, data in the original feature
            space.
        """
        _raise_for_params(params, self, "inverse_transform")

        # we don't have to branch here, since params is only non-empty if
        # enable_metadata_routing=True.
        routed_params = process_routing(self, "inverse_transform", **params)
        reverse_iter = reversed(list(self._iter()))

        if self.observation_horizon:
            # TODO: Routes params
            # routed_params_previous = process_routing(self, "fit_transform", **params)

            # Build X_p_iter_list by transforming X_p through each step
            # The key insight: for the first transformer's inverse, we need
            # X_p[sum_of_other_observation_horizons : observation_horizon], not X_p[:first_observation_horizon]
            steps_list = list(self._iter())

            X_p_iter = X_p
            X_p_iter_list = []
            for idx, (_, _, transform) in enumerate(steps_list[:-1]):
                # Transform X_p_iter through this step using the fitted transform
                X_p_iter = (
                    deepcopy(transform)
                    .reset(X_p_iter[: transform.observation_horizon])
                    .transform(
                        X_p_iter[transform.observation_horizon :],  # **routed_params_previous
                    )
                )
                X_p_iter_list.append(X_p_iter)

            # For the first transformer's inverse, we need the slice of X_p
            # that comes after all the memory used by other transformers
            first_transform = steps_list[0][2]
            offset = sum(t.observation_horizon for _, _, t in steps_list[1:])
            X_p_iter_list.append(X_p[offset : offset + first_transform.observation_horizon])

            X_p_iter_list.reverse()

            X = X_t
            for (_, _, transform), X_p_iter in zip(reverse_iter, X_p_iter_list):
                X = transform.inverse_transform(X_t=X, X_p=X_p_iter)

        else:
            for _, name, transform in reverse_iter:
                X = transform.inverse_transform(X_t, **routed_params[name].inverse_transform)

        return X

    def get_metadata_routing(self) -> object:
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self.__class__.__name__)

        # first we add all steps except the last one
        for _, name, trans in self._iter(with_final=False, filter_passthrough=True):
            method_mapping = MethodMapping()
            # fit, fit_predict, and fit_transform call fit_transform if it
            # exists, or else fit and transform
            if hasattr(trans, "fit_transform"):
                (
                    method_mapping.add(caller="fit", callee="fit_transform")
                    .add(caller="fit_transform", callee="fit_transform")
                    .add(caller="fit_predict", callee="fit_transform")
                )
            else:
                (
                    method_mapping.add(caller="fit", callee="fit")
                    .add(caller="fit", callee="transform")
                    .add(caller="fit_transform", callee="fit")
                    .add(caller="fit_transform", callee="transform")
                    .add(caller="fit_predict", callee="fit")
                    .add(caller="fit_predict", callee="transform")
                )

            (
                method_mapping.add(caller="predict", callee="transform")
                .add(caller="predict", callee="transform")
                .add(caller="predict_proba", callee="transform")
                .add(caller="decision_function", callee="transform")
                .add(caller="predict_log_proba", callee="transform")
                .add(caller="transform", callee="transform")
                .add(caller="inverse_transform", callee="inverse_transform")
                .add(caller="score", callee="transform")
            )

            router.add(method_mapping=method_mapping, **{name: trans})

        final_name, final_est = self.steps[-1]
        if final_est is None or final_est == "passthrough":
            return router

        # then we add the last step
        method_mapping = MethodMapping()
        if hasattr(final_est, "fit_transform"):
            method_mapping.add(caller="fit_transform", callee="fit_transform")
        else:
            method_mapping.add(caller="fit", callee="fit").add(caller="fit", callee="transform")
        (
            method_mapping.add(caller="fit", callee="fit")
            .add(caller="predict", callee="predict")
            .add(caller="fit_predict", callee="fit_predict")
            .add(caller="predict_proba", callee="predict_proba")
            .add(caller="decision_function", callee="decision_function")
            .add(caller="predict_log_proba", callee="predict_log_proba")
            .add(caller="transform", callee="transform")
            .add(caller="inverse_transform", callee="inverse_transform")
            .add(caller="score", callee="score")
        )

        router.add(method_mapping=method_mapping, **{final_name: final_est})
        return router


def _hstack(
    Xs: list[object], column_names: list[list[str]], observation_horizons: list[int]
) -> pl.DataFrame:
    """Stack transformed features horizontally, aligning observation horizons.

    Parameters
    ----------
    Xs : list
        List of transformed DataFrames.

    column_names : list of list of str
        Column names for each DataFrame.

    observation_horizons : list of int
        Observation horizon for each transformer.

    Returns
    -------
    pl.DataFrame
        Horizontally concatenated features.

    """
    ref_observation_horizon = max(observation_horizons)
    time = Xs[0].select(cs.by_name("time"))[ref_observation_horizon - observation_horizons[0] :]  # type: ignore[attr-defined]
    Xs_concat = pl.concat(
        [
            X.select(~cs.by_name("time"))[ref_observation_horizon - observation_horizon :]  # type: ignore[attr-defined]
            for X, observation_horizon in zip(Xs, observation_horizons)
        ],
        how="horizontal",
    )
    Xs_concat.columns = column_names
    result = pl.concat([time, Xs_concat], how="horizontal")

    return result


class FeatureUnion(BaseTransformer, _BaseComposition):  # type: ignore[misc]
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
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    transformer_weights : dict, default=None
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.
        Raises ValueError if key not present in ``transformer_list``.

    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    verbose_feature_names_out : bool, default=True
        If True, :meth:`get_feature_names_out` will prefix all feature names
        with the name of the transformer that generated that feature.
        If False, :meth:`get_feature_names_out` will not prefix any feature
        names and will error if feature names are not unique.

    Attributes
    ----------
    named_transformers : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        Read-only attribute to access any transformer parameter by user
        given name. Keys are transformer names and values are
        transformer parameters.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying first transformer in `transformer_list` exposes such an
        attribute when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when
        `X` has feature names that are all strings.

    """

    _required_parameters = ["transformer_list"]

    get_params = sklearn_FeatureUnion.get_params
    set_params = sklearn_FeatureUnion.set_params
    _iter = sklearn_FeatureUnion._iter
    __getitem__ = sklearn_FeatureUnion.__getitem__
    named_transformers = sklearn_FeatureUnion.named_transformers
    _log_message = sklearn_FeatureUnion._log_message
    _parallel_func = sklearn_FeatureUnion._parallel_func
    _update_transformer_list = sklearn_FeatureUnion._update_transformer_list
    get_feature_names_out = sklearn_FeatureUnion.get_feature_names_out
    n_features_in_ = sklearn_FeatureUnion.n_features_in_
    feature_names_in_ = sklearn_FeatureUnion.feature_names_in_
    _add_prefix_for_feature_names_out = sklearn_FeatureUnion._add_prefix_for_feature_names_out
    __sklearn_is_fitted__ = sklearn_FeatureUnion.__sklearn_is_fitted__
    _sk_visual_block_ = sklearn_FeatureUnion._sk_visual_block_

    def _get_observation_horizons(self) -> list[int]:
        """Get observation horizons from all transformers."""
        observation_horizons = []
        for _, t, _ in self._iter():
            observation_horizon = 0
            if t != "passthrough":
                observation_horizon = t.observation_horizon

            observation_horizons.append(observation_horizon)

        return observation_horizons

    @property
    def observation_horizon(self) -> int:
        """Maximum observation horizon across all transformers."""
        observation_horizons = self._get_observation_horizons()
        observation_horizon = max(observation_horizons)

        return observation_horizon

    def __init__(
        self,
        transformer_list: list[tuple[str, object]],
        *,
        n_jobs: int | None = None,
        transformer_weights: dict[str, float] | None = None,
        verbose: bool = False,
        verbose_feature_names_out: bool = True,
    ) -> None:
        """Initialize feature union.

        Parameters
        ----------
        transformer_list : list of tuple
            List of (name, transformer) tuples.

        n_jobs : int or None, default=None
            Number of parallel jobs.

        transformer_weights : dict or None, default=None
            Weights for transformer outputs.

        verbose : bool, default=False
            Enable verbose output.

        verbose_feature_names_out : bool, default=True
            Prefix feature names with transformer names.

        """
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
        names, transformers = zip(*self.transformer_list)

        # validate names
        self._validate_names(names)

        # validate estimators
        for t in transformers:
            if t in ("drop", "passthrough"):
                continue
            if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(
                t, "transform"
            ):
                raise TypeError(
                    "All estimators should implement fit and "
                    "transform. '%s' (type %s) doesn't" % (t, type(t))
                )

    def _validate_transformer_weights(self) -> None:
        """Validate transformer weights dictionary.

        Raises
        ------
        ValueError
            If weight keys don't match transformer names.

        """
        if not self.transformer_weights:
            return

        transformer_names = set(name for name, _ in self.transformer_list)
        for name in self.transformer_weights:
            if name not in transformer_names:
                raise ValueError(
                    f'Attempting to weight transformer "{name}", '
                    "but it is not present in transformer_list."
                )

    def fit(self, X: object, y: object = None, **fit_params: object) -> "FeatureUnion":
        """Fit all transformers using X.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data, used to fit transformers.

        y : array-like of shape (n_samples, n_outputs), default=None
            Targets for supervised learning.

        **fit_params : dict, default=None
            - If `enable_metadata_routing=False` (default):
              Parameters directly passed to the `fit` methods of the
              sub-transformers.

            - If `enable_metadata_routing=True`:
              Parameters safely routed to the `fit` methods of the
              sub-transformers. See :ref:`Metadata Routing User Guide
              <metadata_routing>` for more details.

        Returns
        -------
        self : object
            FeatureUnion class instance.
        """
        if _routing_enabled():
            routed_params = process_routing(self, "fit", **fit_params)
        else:
            # TODO(SLEP6): remove when metadata routing cannot be disabled.
            routed_params = Bunch()
            for name, _ in self.transformer_list:
                routed_params[name] = Bunch(fit={})
                routed_params[name].fit = fit_params

        transformers = self._parallel_func(X, y, _fit_one, routed_params)

        if not transformers:
            # All transformers are None
            return self

        self._update_transformer_list(transformers)
        return self

    def fit_transform(
        self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params: object
    ) -> object:
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : array-like of shape (n_samples, n_outputs), default=None
            Targets for supervised learning.

        **params : dict, default=None
            - If `enable_metadata_routing=False` (default):
              Parameters directly passed to the `fit` methods of the
              sub-transformers.

            - If `enable_metadata_routing=True`:
              Parameters safely routed to the `fit` methods of the
              sub-transformers. See :ref:`Metadata Routing User Guide
              <metadata_routing>` for more details.

        Returns
        -------
        X_t : array-like or sparse matrix of \
                shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
        """
        if _routing_enabled():
            routed_params = process_routing(self, "fit_transform", **params)
        else:
            # TODO(SLEP6): remove when metadata routing cannot be disabled.
            routed_params = Bunch()
            for name, obj in self.transformer_list:
                if hasattr(obj, "fit_transform"):
                    routed_params[name] = Bunch(fit_transform={})
                    routed_params[name].fit_transform = params
                else:
                    routed_params[name] = Bunch(fit={})
                    routed_params[name] = Bunch(transform={})
                    routed_params[name].fit = params

        results = self._parallel_func(X, y, _fit_transform_one, routed_params)
        if not results:
            # All transformers are None
            time = X.select(cs.by_name("time"))
            return time

        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)

        return _hstack(
            list(Xs),
            column_names=self.get_feature_names_out(),
            observation_horizons=self._get_observation_horizons(),
        )

    def transform(self, X: pl.DataFrame, **params: object) -> object:
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        **params : dict, default=None

            Parameters routed to the `transform` method of the sub-transformers via the
            metadata routing API. See :ref:`Metadata Routing User Guide
            <metadata_routing>` for more details.

        Returns
        -------
        X_t : array-like or sparse matrix of shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
        """
        _raise_for_params(params, self, "transform")

        if _routing_enabled():
            routed_params = process_routing(self, "transform", **params)
        else:
            # TODO(SLEP6): remove when metadata routing cannot be disabled.
            routed_params = Bunch()
            for name, _ in self.transformer_list:
                routed_params[name] = Bunch(transform={})

        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight, routed_params[name])
            for name, trans, weight in self._iter()
        )
        if not Xs:
            # All transformers are None
            time = X.select(cs.by_name("time"))
            return time

        return _hstack(
            Xs,
            column_names=self.get_feature_names_out(),
            observation_horizons=self._get_observation_horizons(),
        )

    def get_metadata_routing(self) -> object:
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self.__class__.__name__)

        for name, transformer in self.transformer_list:
            router.add(
                **{name: transformer},
                method_mapping=MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="fit_transform", callee="fit_transform")
                .add(caller="fit_transform", callee="fit")
                .add(caller="fit_transform", callee="transform")
                .add(caller="transform", callee="transform"),
            )

        return router


_ERR_MSG_1DCOLUMN = (
    "1D data passed to a transformer that expects 2D data. "
    "Try to specify the column selection as a list of one "
    "item instead of a scalar."
)


class ColumnTransformer(BaseTransformer, _BaseComposition):  # type: ignore[misc]
    """Applies transformers to columns of a polars DataFrame.

    This estimator allows different columns or column subsets of the input
    to be transformed separately and the features generated by each transformer
    will be concatenated to form a single feature space.
    This is useful for heterogeneous or columnar data, to combine several
    feature extraction mechanisms or transformations into a single transformer.

    Parameters
    ----------
    transformers : list of tuples
        List of (name, transformer, columns) tuples specifying the
        transformer objects to be applied to subsets of the data.

        name : str
            Like in Pipeline and FeatureUnion, this allows the transformer and
            its parameters to be set using ``set_params`` and searched in grid
            search.
        transformer : {'drop', 'passthrough'} or estimator
            Estimator must support :term:`fit` and :term:`transform`.
            Special-cased strings 'drop' and 'passthrough' are accepted as
            well, to indicate to drop the columns or to pass them through
            untransformed, respectively.
        columns :  str, array-like of str, int, array-like of int, \
                array-like of bool, slice or callable
            Indexes the data on its second axis. Integers are interpreted as
            positional columns, while strings can reference DataFrame columns
            by name.  A scalar string or int should be used where
            ``transformer`` expects X to be a 1d array-like (vector),
            otherwise a 2d array will be passed to the transformer.
            A callable is passed the input data `X` and can return any of the
            above. To select multiple columns by name or dtype, you can use
            :obj:`make_column_selector`.

    remainder : {'drop', 'passthrough'} or estimator, default='drop'
        By default, only the specified columns in `transformers` are
        transformed and combined in the output, and the non-specified
        columns are dropped. (default of ``'drop'``).
        By specifying ``remainder='passthrough'``, all remaining columns that
        were not specified in `transformers`, but present in the data passed
        to `fit` will be automatically passed through. This subset of columns
        is concatenated with the output of the transformers. For dataframes,
        extra columns not seen during `fit` will be excluded from the output
        of `transform`.
        By setting ``remainder`` to be an estimator, the remaining
        non-specified columns will use the ``remainder`` estimator. The
        estimator must support :term:`fit` and :term:`transform`.
        Note that using this feature requires that the DataFrame columns
        input at :term:`fit` and :term:`transform` have identical order.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    transformer_weights : dict, default=None
        Multiplicative weights for features per transformer. The output of the
        transformer is multiplied by these weights. Keys are transformer names,
        values the weights.

    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    verbose_feature_names_out : bool, default=True
        If True, :meth:`ColumnTransformer.get_feature_names_out` will prefix
        all feature names with the name of the transformer that generated that
        feature.
        If False, :meth:`ColumnTransformer.get_feature_names_out` will not
        prefix any feature names and will error if feature names are not
        unique.

    Attributes
    ----------
    transformers_ : list
        The collection of fitted transformers as tuples of (name,
        fitted_transformer, column). `fitted_transformer` can be an estimator,
        or `'drop'`; `'passthrough'` is replaced with an equivalent
        :class:`~sklearn.preprocessing.FunctionTransformer`. In case there were
        no columns selected, this will be the unfitted transformer. If there
        are remaining columns, the final element is a tuple of the form:
        ('remainder', transformer, remaining_columns) corresponding to the
        ``remainder`` parameter. If there are remaining columns, then
        ``len(transformers_)==len(transformers)+1``, otherwise
        ``len(transformers_)==len(transformers)``.

    named_transformers_ : :class:`~sklearn.utils.Bunch`
        Read-only attribute to access any transformer by given name.
        Keys are transformer names and values are the fitted transformer
        objects.

    output_indices_ : dict
        A dictionary from each transformer name to a slice, where the slice
        corresponds to indices in the transformed output. This is useful to
        inspect which transformer is responsible for which transformed
        feature(s).

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying transformers expose such an attribute when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    Notes
    -----
    The order of the columns in the transformed feature matrix follows the
    order of how the columns are specified in the `transformers` list.
    Columns of the original feature matrix that are not specified are
    dropped from the resulting transformed feature matrix, unless specified
    in the `passthrough` keyword. Those columns specified with `passthrough`
    are added at the right to the output of the transformers.
    """

    _parameter_constraints: dict[str, object] = {
        "transformers": [list, Hidden(tuple)],
        "remainder": [
            StrOptions({"drop", "passthrough"}),
            HasMethods(["fit", "transform"]),
            HasMethods(["fit_transform", "transform"]),
        ],
        "n_jobs": [Integral, None],
        "transformer_weights": [dict, None],
        "verbose": ["verbose"],
        "verbose_feature_names_out": ["boolean"],
    }

    get_params = sklearn_ColumnTransformer.get_params
    set_params = sklearn_ColumnTransformer.set_params
    _transformers = sklearn_ColumnTransformer._transformers
    _iter = sklearn_ColumnTransformer._iter
    __getitem__ = sklearn_ColumnTransformer.__getitem__
    _log_message = sklearn_ColumnTransformer._log_message
    _update_fitted_transformers = sklearn_ColumnTransformer._update_fitted_transformers
    _get_feature_name_out_for_transformer = (
        sklearn_ColumnTransformer._get_feature_name_out_for_transformer
    )
    get_feature_names_out = sklearn_ColumnTransformer.get_feature_names_out
    _get_remainder_cols = sklearn_ColumnTransformer._get_remainder_cols
    _get_remainder_cols_dtype = sklearn_ColumnTransformer._get_remainder_cols_dtype
    _add_prefix_for_feature_names_out = sklearn_ColumnTransformer._add_prefix_for_feature_names_out
    _sk_visual_block_ = sklearn_ColumnTransformer._sk_visual_block_
    _get_empty_routing = sklearn_ColumnTransformer._get_empty_routing
    _validate_remainder = sklearn_ColumnTransformer._validate_remainder
    _validate_column_callables = sklearn_ColumnTransformer._validate_column_callables
    _record_output_indices = sklearn_ColumnTransformer._record_output_indices

    def __init__(
        self,
        transformers: list[tuple[str, object, object]],
        *,
        remainder: str | object = "drop",
        n_jobs: int | None = None,
        transformer_weights: dict[str, float] | None = None,
        verbose: bool = False,
        verbose_feature_names_out: bool = True,
    ) -> None:
        """Initialize ColumnTransformer."""
        self.transformers = transformers
        self.remainder = remainder
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self.verbose_feature_names_out = verbose_feature_names_out

    def _get_observation_horizons(self) -> list[int]:
        """Get observation horizons from all fitted transformers."""
        observation_horizons = []
        for _, t, _, _ in self._iter(
            fitted=True,
            column_as_labels=True,
            skip_drop=False,
            skip_empty_columns=False,
        ):
            observation_horizon = 0
            if t not in ("drop", "passthrough"):
                observation_horizon = t.observation_horizon

            observation_horizons.append(observation_horizon)

        return observation_horizons

    @property
    def observation_horizon(self) -> int:
        """Maximum observation horizon across all transformers."""
        observation_horizons = self._get_observation_horizons()
        observation_horizon = max(observation_horizons)

        return observation_horizon

    def _validate_transformers(self) -> None:
        """Validate names of transformers and the transformers themselves.

        This checks whether given transformers have the required methods, i.e.
        `fit` or `fit_transform` and `transform` implemented.
        """
        if not self.transformers:
            return

        names, transformers, _ = zip(*self.transformers)

        # validate names
        self._validate_names(names)

        # validate estimators
        for t in transformers:
            if t == "passthrough":
                continue
            if not isinstance(t, BaseTransformer):
                # Used to validate the transformers in the `transformers` list
                raise TypeError(
                    "All estimators should be instances of `BaseTransformer` "
                    "or be the string 'passthrough' "
                    "'%s' (type %s) doesn't" % (t, type(t))
                )

    def _call_func_on_transformers(
        self,
        X: pl.DataFrame,
        y: pl.DataFrame | None,
        func: str,
        column_as_labels: bool,
        routed_params: dict[str, dict[str, dict[str, object]]],
    ) -> list[object]:
        """
        Private function to fit and/or transform on demand.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            The data to be used in fit and/or transform.

        y : array-like of shape (n_samples,)
            Targets.

        func : callable
            Function to call, which can be _fit_transform_one or
            _transform_one.

        column_as_labels : bool
            Used to iterate through transformers. If True, columns are returned
            as strings. If False, columns are returned as they were given by
            the user. Can be True only if the ``ColumnTransformer`` is already
            fitted.

        routed_params : dict
            The routed parameters as the output from ``process_routing``.

        Returns
        -------
        Return value (transformers and/or transformed X data) depends
        on the passed function.
        """
        time = X.select(cs.by_name("time"))
        X = X.select(~cs.by_name("time"))

        if func is _fit_transform_one:
            fitted = False
        else:  # func is _transform_one
            fitted = True

        def safe_indexing(X: pl.DataFrame, columns: object, axis: int) -> object:
            """Safe indexing helper for polars DataFrames."""
            Xi = _safe_indexing(X, columns, axis=axis)

            if isinstance(Xi, pl.Series):
                Xi = Xi.to_frame()

            return Xi

        transformers = list(
            self._iter(
                fitted=fitted,
                column_as_labels=column_as_labels,
                skip_drop=True,
                skip_empty_columns=True,
            )
        )
        try:
            jobs = []
            for idx, (name, trans, column, weight) in enumerate(transformers, start=1):
                if func is _fit_transform_one:
                    if trans == "passthrough":
                        output_config = _get_output_config("transform", self)
                        trans = FunctionTransformer(
                            check_inverse=False,
                            feature_names_out="one-to-one",
                        ).set_output(transform=output_config["dense"])

                    extra_args = dict(
                        message_clsname="ColumnTransformer",
                        message=self._log_message(name, idx, len(transformers)),
                    )
                else:  # func is _transform_one
                    extra_args = {}
                jobs.append(
                    delayed(func)(
                        transformer=clone(trans) if not fitted else trans,
                        X=pl.concat(
                            [time, safe_indexing(X, column, axis=1)],
                            how="horizontal",
                        ),
                        y=y,
                        weight=weight,
                        **extra_args,
                        params=routed_params[name],
                    )
                )

            return Parallel(n_jobs=self.n_jobs)(jobs)  # type: ignore[no-any-return]

        except ValueError as e:
            if "Expected 2D array, got 1D array instead" in str(e):
                raise ValueError(_ERR_MSG_1DCOLUMN) from e
            else:
                raise

    def fit(
        self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params: object
    ) -> "ColumnTransformer":
        """Fit all transformers using X.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            Input data, of which specified subsets are used to fit the
            transformers.

        y : array-like of shape (n_samples,...), default=None
            Targets for supervised learning.

        **params : dict, default=None
            Parameters to be passed to the underlying transformers' ``fit`` and
            ``transform`` methods.

            You can only pass this if metadata routing is enabled, which you
            can enable using ``sklearn.set_config(enable_metadata_routing=True)``.

        Returns
        -------
        self : ColumnTransformer
            This estimator.
        """
        _raise_for_params(params, self, "fit")
        # we use fit_transform to make sure to set sparse_output_ (for which we
        # need the transformed data) to have consistent output type in predict
        self.fit_transform(X, y=y, **params)
        return self

    @_fit_context(  # type: ignore[untyped-decorator]
        # estimators in ColumnTransformer.transformers are not validated yet
        prefer_skip_nested_validation=False
    )
    def fit_transform(
        self, X: pl.DataFrame, y: pl.DataFrame | None = None, **params: object
    ) -> object:
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            Input data, of which specified subsets are used to fit the
            transformers.

        y : array-like of shape (n_samples,), default=None
            Targets for supervised learning.

        **params : dict, default=None
            Parameters to be passed to the underlying transformers' ``fit`` and
            ``transform`` methods.

            You can only pass this if metadata routing is enabled, which you
            can enable using ``sklearn.set_config(enable_metadata_routing=True)``.

        Returns
        -------
        X_t : {array-like, sparse matrix} of \
                shape (n_samples, sum_n_components)
            Horizontally stacked results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers. If
            any result is a sparse matrix, everything will be converted to
            sparse matrices.
        """
        _raise_for_params(params, self, "fit_transform")
        _check_feature_names(self, X, reset=True)

        X = _check_X(X)
        # set n_features_in_ attribute
        _check_n_features(self, X, reset=True)
        self._validate_transformers()
        n_samples = _num_samples(X)

        self._validate_column_callables(X)
        self._validate_remainder(X)

        if _routing_enabled():
            routed_params = process_routing(self, "fit_transform", **params)
        else:
            routed_params = self._get_empty_routing()

        result = self._call_func_on_transformers(
            X,
            y,
            _fit_transform_one,
            column_as_labels=False,
            routed_params=routed_params,
        )

        if not result:
            self._update_fitted_transformers([])
            # All transformers are None
            time = X.select(cs.by_name("time"))
            return time

        Xs, transformers = zip(*result)

        self.sparse_output_ = False

        self._update_fitted_transformers(transformers)
        self._record_output_indices(Xs)

        return self._hstack(list(Xs), n_samples=n_samples)

    def transform(self, X: pl.DataFrame, **params: object) -> object:
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            The data to be transformed by subset.

        **params : dict, default=None
            Parameters to be passed to the underlying transformers' ``transform``
            method.

            You can only pass this if metadata routing is enabled, which you
            can enable using ``sklearn.set_config(enable_metadata_routing=True)``.

        Returns
        -------
        X_t : {array-like, sparse matrix} of \
                shape (n_samples, sum_n_components)
            Horizontally stacked results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers. If
            any result is a sparse matrix, everything will be converted to
            sparse matrices.
        """
        _raise_for_params(params, self, "transform")
        check_is_fitted(self)
        X = _check_X(X)

        # If ColumnTransformer is fit using a dataframe, and now a dataframe is
        # passed to be transformed, we select columns by name instead. This
        # enables the user to pass X at transform time with extra columns which
        # were not present in fit time, and the order of the columns doesn't
        # matter.
        fit_dataframe_and_transform_dataframe = hasattr(self, "feature_names_in_") and (
            is_pandas_df(X) or hasattr(X, "__dataframe__")
        )

        n_samples = _num_samples(X)
        column_names = _get_feature_names(X)

        if fit_dataframe_and_transform_dataframe:
            named_transformers = self.named_transformers_
            # check that all names seen in fit are in transform, unless
            # they were dropped
            non_dropped_indices = [
                ind
                for name, ind in self._transformer_to_input_indices.items()
                if name in named_transformers and named_transformers[name] != "drop"
            ]

            all_indices = set(chain(*non_dropped_indices))
            all_names = set(self.feature_names_in_[ind] for ind in all_indices)

            diff = all_names - set(column_names)
            if diff:
                raise ValueError(f"columns are missing: {diff}")
        else:
            # ndarray was used for fitting or transforming, thus we only
            # check that n_features_in_ is consistent
            self._check_n_features(X, reset=False)

        if _routing_enabled():
            routed_params = process_routing(self, "transform", **params)
        else:
            routed_params = self._get_empty_routing()

        Xs = self._call_func_on_transformers(
            X,
            None,
            _transform_one,
            column_as_labels=fit_dataframe_and_transform_dataframe,
            routed_params=routed_params,
        )

        if not Xs:
            # All transformers are None
            time = X.select(cs.by_name("time"))
            return time

        return self._hstack(list(Xs), n_samples=n_samples)

    def _hstack(self, Xs: list[pl.DataFrame], *, n_samples: int) -> pl.DataFrame:
        """Stacks Xs horizontally.

        This allows subclasses to control the stacking behavior, while reusing
        everything else from ColumnTransformer.

        Parameters
        ----------
        Xs : list of {array-like, sparse matrix, dataframe}
            The container to concatenate.

        n_samples : int
            The number of samples in the input data to checking the transformation
            consistency.
        """
        # rename before stacking as it avoids to error on temporary duplicated
        # columns
        transformer_names = [
            t[0]
            for t in self._iter(
                fitted=True,
                column_as_labels=False,
                skip_drop=True,
                skip_empty_columns=True,
            )
        ]
        feature_names_outs = [
            [col for col in X.columns if col != "time"] for X in Xs if X.shape[1] != 1
        ]
        if self.verbose_feature_names_out:
            # `_add_prefix_for_feature_names_out` takes care about raising
            # an error if there are duplicated columns.
            feature_names_outs = self._add_prefix_for_feature_names_out(
                list(zip(transformer_names, feature_names_outs))
            )
        else:
            # check for duplicated columns and raise if any
            feature_names_outs = list(chain.from_iterable(feature_names_outs))
            feature_names_count = Counter(feature_names_outs)
            if any(count > 1 for count in feature_names_count.values()):
                duplicated_feature_names = sorted(
                    name for name, count in feature_names_count.items() if count > 1
                )
                err_msg = (
                    "Duplicated feature names found before concatenating the"
                    " outputs of the transformers:"
                    f" {duplicated_feature_names}.\n"
                )
                for transformer_name, X in zip(transformer_names, Xs):
                    if X.shape[1] == 1:
                        continue
                    dup_cols_in_transformer = sorted(
                        set(X.columns).intersection(duplicated_feature_names)
                    )
                    if len(dup_cols_in_transformer):
                        err_msg += (
                            f"Transformer {transformer_name} has conflicting "
                            f"columns names: {dup_cols_in_transformer}.\n"
                        )
                raise ValueError(
                    err_msg + "Either make sure that the transformers named above "
                    "do not generate columns with conflicting names or set "
                    "verbose_feature_names_out=True to automatically "
                    "prefix to the output feature names with the name "
                    "of the transformer to prevent any conflicting "
                    "names."
                )

        output = _hstack(
            Xs,
            column_names=feature_names_outs,
            observation_horizons=self._get_observation_horizons(),
        )
        output_samples = output.shape[0]
        if output_samples != n_samples - self.observation_horizon:
            raise ValueError(
                "Concatenating DataFrames from the transformer's output lead to"
                " an inconsistent number of samples."
            )

        return output

    def get_metadata_routing(self) -> object:
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self.__class__.__name__)
        # Here we don't care about which columns are used for which
        # transformers, and whether or not a transformer is used at all, which
        # might happen if no columns are selected for that transformer. We
        # request all metadata requested by all transformers.
        transformers = chain(self.transformers, [("remainder", self.remainder, None)])
        for name, step, _ in transformers:
            method_mapping = MethodMapping()
            if hasattr(step, "fit_transform"):
                (
                    method_mapping.add(caller="fit", callee="fit_transform").add(
                        caller="fit_transform", callee="fit_transform"
                    )
                )
            else:
                (
                    method_mapping.add(caller="fit", callee="fit")
                    .add(caller="fit", callee="transform")
                    .add(caller="fit_transform", callee="fit")
                    .add(caller="fit_transform", callee="transform")
                )
            method_mapping.add(caller="transform", callee="transform")
            router.add(method_mapping=method_mapping, **{name: step})

        return router
