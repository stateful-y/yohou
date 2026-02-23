import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Yohou Quickstart: Time Series Forecasting with a Scikit-Learn API

    Welcome to **yohou**, a scikit-learn-compatible time series forecasting framework built on **polars**.

    ## What You'll Learn

    This tour covers some of the major capabilities of the framework in a single notebook:

    1. **Data & visualisation**: load data, plot time series
    2. **Baseline forecasting**: SeasonalNaive
    3. **Preprocessing pipelines**: FeaturePipeline with LogTransformer, SeasonalDifferencing, LagTransformer
    4. **Decomposition**: DecompositionPipeline with trend + seasonality components
    5. **Incremental learning**: observe, predict, observe_predict streaming workflow
    6. **Cross-validation & search**: ExpandingWindowSplitter, GridSearchCV, RandomizedSearchCV
    7. **Scoring**: point scorers, aggregation modes
    8. **Interval forecasting**: SplitConformalForecaster for prediction intervals
    9. **Time-weighted training**: exponential_decay_weight, compose_weights
    10. **Panel data**: forecasting multiple series with the `__` separator convention

    ## Prerequisites

    Basic Python and familiarity with sklearn's `fit` / `predict` API.
    """)
    return


@app.cell(hide_code=True)
async def _():
    import sys as _sys

    if "pyodide" in _sys.modules:
        import micropip

        await micropip.install(["plotly", "scikit-learn", "scipy", "yohou"])
    _ = None
    return


@app.cell(hide_code=True)
def _():
    import copy

    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots
    from scipy.stats import randint, uniform
    from sklearn.base import clone
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    from yohou.compose import DecompositionPipeline, FeaturePipeline
    from yohou.datasets import load_air_passengers, load_store_sales
    from yohou.interval import SplitConformalForecaster
    from yohou.metrics import MeanAbsoluteError, MeanSquaredError
    from yohou.model_selection import (
        ExpandingWindowSplitter,
        GridSearchCV,
        RandomizedSearchCV,
        SlidingWindowSplitter,
    )
    from yohou.plotting import (
        plot_calibration,
        plot_cv_results_scatter,
        plot_forecast,
        plot_splits,
        plot_time_series,
        plot_time_weight,
    )
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer
    from yohou.stationarity import (
        FourierSeasonalityForecaster,
        LogTransformer,
        PolynomialTrendForecaster,
        SeasonalDifferencing,
    )
    from yohou.utils.panel import inspect_locality
    from yohou.utils.weighting import (
        compose_weights,
        exponential_decay_weight,
        linear_decay_weight,
        seasonal_emphasis_weight,
    )

    return (
        DecompositionPipeline,
        ExpandingWindowSplitter,
        FeaturePipeline,
        FourierSeasonalityForecaster,
        GridSearchCV,
        LagTransformer,
        LogTransformer,
        MeanAbsoluteError,
        MeanSquaredError,
        PointReductionForecaster,
        PolynomialTrendForecaster,
        RandomizedSearchCV,
        Ridge,
        SeasonalDifferencing,
        SeasonalNaive,
        SlidingWindowSplitter,
        SplitConformalForecaster,
        clone,
        compose_weights,
        copy,
        exponential_decay_weight,
        inspect_locality,
        linear_decay_weight,
        load_air_passengers,
        load_store_sales,
        pl,
        plot_calibration,
        plot_cv_results_scatter,
        plot_forecast,
        plot_splits,
        plot_time_series,
        plot_time_weight,
        randint,
        seasonal_emphasis_weight,
        train_test_split,
        uniform,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Data & Visualisation

    We use the classic **Air Passengers** dataset, monthly international airline
    passenger totals from 1949 to 1960.

    Yohou requires a polars DataFrame with a **`"time"` column** (datetime type).
    """)
    return


@app.cell
def _(load_air_passengers):
    y = load_air_passengers().rename({"Passengers": "passengers"})
    print(f"Shape: {y.shape}  |  Range: {y['time'].min()} → {y['time'].max()}")
    y.head()
    return (y,)


@app.cell
def _(plot_time_series, y):
    plot_time_series(
        y,
        columns="passengers",
        title="Air Passengers (1949 – 1960)",
        y_label="Passengers (thousands)",
        height=380,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Observations**: strong upward trend, yearly seasonality (summer peaks), and
    increasing variance, hallmarks of a non-stationary, multiplicative series.
    """)
    return


@app.cell
def _(train_test_split, y):
    y_train, y_test = train_test_split(y, test_size=24, shuffle=False)
    forecasting_horizon = len(y_test)

    print(f"Train: {len(y_train)} rows  |  Test: {len(y_test)} rows  |  Horizon: {forecasting_horizon}")
    return forecasting_horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Baseline: Seasonal Naive

    The simplest seasonal model: repeat the values from one year ago
    (`seasonality=12` for monthly data).

    Every more complex model should **beat this baseline**.
    """)
    return


@app.cell
def _(MeanAbsoluteError, SeasonalNaive, forecasting_horizon, y_test, y_train):
    baseline = SeasonalNaive(seasonality=12)
    baseline.fit(y_train, X=None, forecasting_horizon=forecasting_horizon)
    y_pred_baseline = baseline.predict(forecasting_horizon=forecasting_horizon)

    scorer = MeanAbsoluteError()
    scorer.fit(y_train)
    mae_baseline = scorer.score(y_test, y_pred_baseline)
    print(f"Baseline MAE: {mae_baseline:.2f}")
    return mae_baseline, scorer, y_pred_baseline


@app.cell
def _(plot_forecast, y_pred_baseline, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_baseline,
        y_train=y_train,
        title="Baseline: Seasonal Naive",
        y_label="Passengers (thousands)",
        height=380,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Preprocessing Pipelines

    Reduction forecasters convert time-series forecasting to supervised learning.
    A **FeaturePipeline** chains transforms:

    1. `LogTransformer`: stabilises multiplicative variance
    2. `SeasonalDifferencing(12)`: removes yearly seasonality and trend
    3. `LagTransformer`: creates autoregressive features from past values

    When used as a `target_transformer`, all transformations are automatically inverted at prediction time. In the case of a `FeaturePipeline`, this means all transforms it includes have to be invertible.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    max_lag_slider = mo.ui.slider(start=1, stop=12, value=3, label="Max lag", show_value=True)
    max_lag_slider
    return (max_lag_slider,)


@app.cell
def _(
    FeaturePipeline,
    LagTransformer,
    LogTransformer,
    PointReductionForecaster,
    Ridge,
    SeasonalDifferencing,
    forecasting_horizon,
    mae_baseline,
    max_lag_slider,
    scorer,
    y_test,
    y_train,
):
    pipeline_target = FeaturePipeline([
        ("log", LogTransformer(offset=1.0)),
        ("diff", SeasonalDifferencing(seasonality=12)),
    ])
    pipeline_feature = FeaturePipeline([
        ("lag", LagTransformer(lag=list(range(1, max_lag_slider.value + 1)))),
    ])

    reduction = PointReductionForecaster(
        estimator=Ridge(alpha=10),
        target_transformer=pipeline_target,
        feature_transformer=pipeline_feature,
    )
    reduction.fit(y_train, X=None, forecasting_horizon=forecasting_horizon)
    y_pred_reduction = reduction.predict(forecasting_horizon=forecasting_horizon)

    mae_reduction = scorer.score(y_test, y_pred_reduction)
    improvement = (mae_baseline - mae_reduction) / mae_baseline * 100
    print(f"Reduction MAE: {mae_reduction:.2f}  ({improvement:+.1f}% vs baseline)")
    return reduction, y_pred_reduction


@app.cell
def _(plot_forecast, y_pred_baseline, y_pred_reduction, y_test, y_train):
    plot_forecast(
        y_test,
        {"Baseline": y_pred_baseline, "Reduction": y_pred_reduction},
        y_train=y_train,
        title="Reduction Forecaster vs Baseline",
        y_label="Passengers (thousands)",
        height=380,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Decomposition Pipeline

    A **DecompositionPipeline** is a meta-forecaster made of several forecasters that sequentially fit on the residual of the previous forecaster. I can be used to explicitly models structural components:

    - **Trend** via `PolynomialTrendForecaster`
    - **Seasonality** via `FourierSeasonalityForecaster`
    - **Residual** via any forecaster (here `PointReductionForecaster`)
    """)
    return


@app.cell
def _(
    DecompositionPipeline,
    FourierSeasonalityForecaster,
    LagTransformer,
    LogTransformer,
    PointReductionForecaster,
    PolynomialTrendForecaster,
    Ridge,
    forecasting_horizon,
    scorer,
    y_test,
    y_train,
):
    decomp = DecompositionPipeline(
        forecasters=[
            ("trend", PolynomialTrendForecaster(degree=1)),
            ("season", FourierSeasonalityForecaster(seasonality=12, harmonics=[1, 2, 3, 4])),
            (
                "residual",
                PointReductionForecaster(
                    estimator=Ridge(alpha=1.0),
                    feature_transformer=LagTransformer(lag=[1, 2, 3, 12]),
                ),
            ),
        ],
        target_transformer=LogTransformer(offset=1.0),
    )
    decomp.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred_decomp = decomp.predict(forecasting_horizon=forecasting_horizon)
    mae_decomp = scorer.score(y_test, y_pred_decomp)
    print(f"Decomposition MAE: {mae_decomp:.2f}")
    return (y_pred_decomp,)


@app.cell
def _(
    plot_forecast,
    y_pred_baseline,
    y_pred_decomp,
    y_pred_reduction,
    y_test,
    y_train,
):
    plot_forecast(
        y_test,
        {
            "Baseline": y_pred_baseline,
            "Reduction": y_pred_reduction,
            "Decomposition": y_pred_decomp,
        },
        y_train=y_train,
        title="Model Comparison (so far)",
        y_label="Passengers (thousands)",
        height=400,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Incremental Learning: `observe` / `predict`

    In production you rarely retrain from scratch. Yohou's **streaming workflow**:

    1. `fit(y_train)`: initial training
    2. `observe(y_new)`: feed new observations (updates memory, NOT model weights)
    3. `predict(horizon)`: forecast from latest observation
    4. `observe_predict(y_new)`: atomic shortcut for observe + predict

    Here we compare a **static** prediction (no new observations) against one that
    first calls `observe_predict` to absorb half the test period before predicting
    the remaining horizon:
    """)
    return


@app.cell
def _(copy, forecasting_horizon, plot_forecast, reduction, y_test, y_train):
    _pivot = forecasting_horizon // 2

    # Static: predict straight from the training end, no new data
    _static = copy.deepcopy(reduction)
    _y_pred_static = _static.predict(forecasting_horizon=forecasting_horizon)

    # Incremental: observe the first half of the test window, then predict the rest
    _incr = copy.deepcopy(reduction)
    _y_pred_incr = _incr.observe_predict(
        y=y_test[:_pivot],
        forecasting_horizon=_pivot,
    )

    plot_forecast(
        y_test,
        {"Static (predict only)": _y_pred_static, "Incremental (observe_predict)": _y_pred_incr},
        y_train=y_train,
        title="Static vs Incremental: single observe_predict call",
        y_label="Passengers (thousands)",
        height=400,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Cross-Validation & Hyperparameter Search

    Standard k-fold CV destroys temporal order. Yohou provides **temporal splitters**:

    - `ExpandingWindowSplitter`: training window grows each fold
    - `SlidingWindowSplitter`: training window slides at fixed size

    Pair with `GridSearchCV` or `RandomizedSearchCV` to tune hyperparameters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `plot_splits` visualizes the train/test structure across folds.
    An **expanding window** grows the training set with each fold:
    """)
    return


@app.cell
def _(ExpandingWindowSplitter, plot_splits, y_train):
    plot_splits(
        y_train,
        ExpandingWindowSplitter(n_splits=5, test_size=12),
        title="Expanding Window (5 folds)",
        height=310,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A **sliding window** keeps training size fixed — oldest data is dropped each fold:
    """)
    return


@app.cell
def _(SlidingWindowSplitter, plot_splits, y_train):
    plot_splits(
        y_train,
        SlidingWindowSplitter(n_splits=5, test_size=12, stride=12),
        title="Sliding Window (5 folds)",
        height=310,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### RandomizedSearchCV

    Randomly sample from continuous / discrete parameter distributions.
    Efficient for larger search spaces.
    """)
    return


@app.cell
def _(
    ExpandingWindowSplitter,
    MeanAbsoluteError,
    RandomizedSearchCV,
    clone,
    forecasting_horizon,
    randint,
    reduction,
    uniform,
    y_train,
):
    cv = ExpandingWindowSplitter(n_splits=2, test_size=12)

    random_search = RandomizedSearchCV(
        forecaster=clone(reduction),
        param_distributions={
            "estimator__alpha": uniform(0.01, 10.0),
            "feature_transformer__lag__lag": randint(1, 7),
        },
        scoring=MeanAbsoluteError(),
        cv=cv,
        n_iter=9,
        refit=True,
        return_train_score=False,
        n_jobs=1,
    )
    random_search.fit(y_train, X=None, forecasting_horizon=forecasting_horizon)

    print(f"Best params: {random_search.best_params_}")
    print(f"Best CV MAE:  {random_search.best_score_:.2f}")
    return (random_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GridSearchCV

    Exhaustively test every combination, best for small discrete grids.
    """)
    return


@app.cell
def _(
    ExpandingWindowSplitter,
    GridSearchCV,
    MeanAbsoluteError,
    clone,
    forecasting_horizon,
    reduction,
    y_train,
):
    grid_search = GridSearchCV(
        forecaster=clone(reduction),
        param_grid={
            "estimator__alpha": [0.001, 0.01, 0.1],
            "feature_transformer__lag__lag": [1, 2, 3],
        },
        scoring=MeanAbsoluteError(),
        cv=ExpandingWindowSplitter(n_splits=2, test_size=12),
        refit=True,
        return_train_score=False,
        n_jobs=1,
    )
    grid_search.fit(y_train, X=None, forecasting_horizon=forecasting_horizon)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV MAE:  {grid_search.best_score_:.2f}")
    return (grid_search,)


@app.cell
def _(forecasting_horizon, grid_search, random_search, scorer, y_test):
    y_pred_random = random_search.predict(forecasting_horizon=forecasting_horizon)
    y_pred_grid = grid_search.predict(forecasting_horizon=forecasting_horizon)

    mae_random = scorer.score(y_test, y_pred_random)
    mae_grid = scorer.score(y_test, y_pred_grid)
    print(f"RandomizedSearchCV test MAE: {mae_random:.2f}")
    print(f"GridSearchCV       test MAE: {mae_grid:.2f}")
    return y_pred_grid, y_pred_random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compare the tuned models against the baseline on the test set:
    """)
    return


@app.cell
def _(
    plot_forecast,
    y_pred_baseline,
    y_pred_grid,
    y_pred_random,
    y_test,
    y_train,
):
    plot_forecast(
        y_test,
        {
            "Baseline": y_pred_baseline,
            "RandomizedSearchCV": y_pred_random,
            "GridSearchCV": y_pred_grid,
        },
        y_train=y_train,
        title="Model Comparison (tuning)",
        y_label="Passengers (thousands)",
        height=400,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `plot_cv_results_scatter` shows how each hyperparameter value affects the
    cross-validated score, making it easy to identify the optimal range:
    """)
    return


@app.cell
def _(grid_search, plot_cv_results_scatter):
    plot_cv_results_scatter(
        grid_search.cv_results_,
        param_name="estimator__alpha",
        higher_is_better=False,
        title="GridSearchCV: alpha vs MAE",
        y_label="Mean MAE",
        height=380,
    )
    return


@app.cell
def _(plot_cv_results_scatter, random_search):
    plot_cv_results_scatter(
        random_search.cv_results_,
        param_name="estimator__alpha",
        higher_is_better=False,
        title="RandomizedSearchCV: alpha vs MAE",
        y_label="Mean MAE",
        height=380,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Scoring & Aggregation

    Yohou scorers extend sklearn with time-series-specific features:

    - **`score(y_truth, y_pred)`** returns a float (default `aggregation_method="all"`)
    - Change `aggregation_method` for richer output:
      - `"timewise" -> per-compotent scores (DataFrame)
      - `"componentwise"` → per-timestep scores (DataFrame)
      - `"all"` → single scalar across all columns and timesteps
    """)
    return


@app.cell
def _(MeanAbsoluteError, MeanSquaredError, y_pred_reduction, y_test, y_train):
    _scorers = {
        "MAE": MeanAbsoluteError(),
        "MSE": MeanSquaredError(),
        "MAE-componentwise": MeanAbsoluteError(aggregation_method="componentwise"),
    }
    for _name, _s in _scorers.items():
        _s.fit(y_train)
        _result = _s.score(y_test, y_pred_reduction)
        print(f"{_name}: {_result}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With `"componentwise"` you get a DataFrame with one error per timestep,
    useful for diagnosing where in the forecast horizon accuracy degrades.

    See `examples/metrics/` for an exhaustive scorer survey.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Interval Forecasting: Prediction Intervals

    `SplitConformalForecaster` wraps any point forecaster and produces
    **calibrated prediction intervals** using split conformal prediction.

    Pass `coverage_rates` to `predict_interval` to get intervals at the
    desired confidence levels.
    """)
    return


@app.cell
def _(
    FeaturePipeline,
    LagTransformer,
    LogTransformer,
    PointReductionForecaster,
    Ridge,
    SeasonalDifferencing,
    SplitConformalForecaster,
    forecasting_horizon,
    y_train,
):
    conformal = SplitConformalForecaster(
        point_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=10),
            target_transformer=FeaturePipeline([
                ("log", LogTransformer(offset=1.0)),
                ("diff", SeasonalDifferencing(seasonality=12)),
            ]),
            feature_transformer=FeaturePipeline([
                ("lag", LagTransformer(lag=[1, 2, 3])),
            ]),
        ),
        calibration_size=30,
    )
    conformal.fit(y_train, forecasting_horizon=forecasting_horizon)

    coverage_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    _y_pred_point = conformal.predict(forecasting_horizon=forecasting_horizon).drop("observed_time")
    _y_pred_pi = conformal.predict_interval(
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage_rates,
    )
    y_pred_interval = _y_pred_point.join(_y_pred_pi, on="time")
    print(f"Interval columns: {y_pred_interval.columns}")
    y_pred_interval.head()
    return coverage_rates, y_pred_interval


@app.cell
def _(plot_forecast, y_pred_interval, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_interval,
        y_train=y_train,
        coverage_rates=[0.5, 0.7, 0.9],
        title="50 / 70 / 90% Prediction Intervals (Split Conformal)",
        y_label="Passengers (thousands)",
        height=400,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Calibration

    A well-calibrated forecaster should hit its nominal coverage: a 90% interval
    should contain the actual value ~90% of the time. The calibration plot compares
    **empirical coverage** (observed fraction of actuals inside the interval) against
    **nominal coverage** (the requested rate) for every level. Points on the diagonal
    indicate perfect calibration; points below mean the intervals are too narrow
    (under-coverage), points above mean they are too wide (over-coverage).
    """)
    return


@app.cell
def _(coverage_rates, plot_calibration, y_pred_interval, y_test):
    plot_calibration(
        y_pred_interval,
        y_test,
        coverage_rates=coverage_rates,
        columns="passengers",
        title="Prediction Interval Calibration",
        height=400,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Time-Weighted Training

    Give more importance to **recent** observations during model fitting via
    `time_weight`, a callable `(pl.Series) → pl.Series`.

    Built-in weight functions:

    | Function | Effect |
    |----------|--------|
    | `exponential_decay_weight(half_life=…)` | Recent data gets exponentially more weight |
    | `linear_decay_weight(max_steps=…)` | Linear ramp from 0 → 1 |
    | `seasonal_emphasis_weight(seasonality=…, emphasis=…)` | Boost specific seasonal positions |
    | `compose_weights(fn1, fn2, …)` | Multiply multiple weight functions element-wise |
    """)
    return


@app.cell
def _(
    exponential_decay_weight,
    linear_decay_weight,
    pl,
    plot_time_weight,
    seasonal_emphasis_weight,
    y_train,
):
    _fns = {
        "exponential": exponential_decay_weight(half_life=15),
        "linear": linear_decay_weight(max_steps=None),
        "seasonal": seasonal_emphasis_weight(seasonality=12, emphasis=3.0),
    }
    _weights_df = y_train.select("time").with_columns([
        pl.Series(f"time_weight__{name}", fn(y_train["time"]).to_list())
        for name, fn in _fns.items()
    ])
    plot_time_weight(
        _weights_df,
        weight_column="time_weight",
        facet_n_cols=1,
        title="Time Weight Functions",
        height=500,
    )
    return


@app.cell
def _(
    FeaturePipeline,
    LagTransformer,
    LogTransformer,
    PointReductionForecaster,
    Ridge,
    SeasonalDifferencing,
    compose_weights,
    exponential_decay_weight,
    forecasting_horizon,
    scorer,
    seasonal_emphasis_weight,
    y_test,
    y_train,
):
    _tw = compose_weights(
        exponential_decay_weight(half_life=20),
        seasonal_emphasis_weight(seasonality=12, emphasis=3.0),
    )

    weighted_forecaster = PointReductionForecaster(
        estimator=Ridge(alpha=10),
        target_transformer=FeaturePipeline([
            ("log", LogTransformer(offset=1.0)),
            ("diff", SeasonalDifferencing(seasonality=12)),
        ]),
        feature_transformer=FeaturePipeline([
            ("lag", LagTransformer(lag=[1, 2, 3])),
        ]),
    ).set_fit_request(time_weight=True)

    weighted_forecaster.fit(y_train, forecasting_horizon=forecasting_horizon, time_weight=_tw)
    y_pred_weighted = weighted_forecaster.predict(forecasting_horizon=forecasting_horizon)
    mae_weighted = scorer.score(y_test, y_pred_weighted)
    print(f"Time-weighted MAE: {mae_weighted:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Panel Data

    Yohou uses **column prefixes with `__`** to represent panel groups:

    ```
    store_1_item_1__sales   store_2_item_1__sales   store_3_item_1__sales
    ```

    Any forecaster automatically handles all groups when it sees this pattern.
    """)
    return


@app.cell
def _(inspect_locality, load_store_sales):
    _df = load_store_sales()
    _cols = ["store_1_item_1__sales", "store_2_item_1__sales", "store_3_item_1__sales"]
    y_panel = _df.select(["time"] + _cols)

    _global, _groups = inspect_locality(y_panel)
    print(f"Panel groups: {list(_groups.keys())}")
    y_panel.head()
    return (y_panel,)


@app.cell
def _(MeanAbsoluteError, SeasonalNaive, y_panel):
    _split = len(y_panel) - 90
    y_panel_train, y_panel_test = y_panel[:_split], y_panel[_split:]

    panel_baseline = SeasonalNaive(seasonality=365)
    panel_baseline.fit(y_panel_train, forecasting_horizon=91)
    y_pred_panel = panel_baseline.predict(forecasting_horizon=91)

    _scorer = MeanAbsoluteError()
    _scorer.fit(y_panel_test)
    print(f"Panel baseline MAE: {_scorer.score(y_panel_test, y_pred_panel):.2f}")
    return y_panel_test, y_panel_train, y_pred_panel


@app.cell
def _(plot_forecast, y_panel_test, y_panel_train, y_pred_panel):
    plot_forecast(
        y_panel_test,
        y_pred_panel,
        y_train=y_panel_train,
        facet_n_cols=1,
        title="Panel Forecast: SeasonalNaive",
        y_label="Sales",
        height=700,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Scikit-learn API**: `fit(y, X, forecasting_horizon)` → `predict(forecasting_horizon)` → `score(y_truth, y_pred)`
    - **Preprocessing**: `FeaturePipeline` chains invertible transforms (log, differencing, lags)
    - **Decomposition**: `DecompositionPipeline` models trend + seasonality + residual
    - **Cross-validation**: `ExpandingWindowSplitter` / `SlidingWindowSplitter` respect temporal order
    - **Hyperparameter search**: `GridSearchCV` (exhaustive) and `RandomizedSearchCV` (efficient)
    - **Time weighting**: `exponential_decay_weight`, `compose_weights` for recency emphasis
    - **Streaming**: `observe_predict` updates memory without retraining
    - **Intervals**: `SplitConformalForecaster` + `predict_interval(coverage_rates=[0.9])`
    - **Panel data**: `__` separator convention: one forecaster handles all groups


    ## Next Steps

    | Topic | Notebook |
    |-------|----------|
    | Point forecasters | `point/naive_forecasters.py`, `point/reduction_forecaster.py` |
    | Feature engineering | `point/feature_forecasting.py`, `preprocessing/window_transformers.py` |
    | Interval forecasting | `interval/conformal_forecasting.py`, `interval/interval_reduction.py` |
    | Decomposition deep dive | `stationarity/decomposition.py` |
    | Metrics guide | `metrics/point_metrics.py`, `metrics/interval_metrics.py` |
    | Splitters & search | `model_selection/cv_splitters.py`, `model_selection/hyperparameter_search.py` |
    | Dataset explorers | `datasets/air_passengers.py`, `datasets/store_sales.py`, … |
    | Plotting gallery | `plotting/exploration.py`, `plotting/forecasting_visualization.py`, … |
    """)
    return


if __name__ == "__main__":
    app.run()
