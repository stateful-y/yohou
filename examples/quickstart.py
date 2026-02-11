import marimo

__generated_with = "0.19.11"
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

    This tour covers every major capability in a single notebook:

    1. **Data & visualisation**: load data, plot time series
    2. **Baseline forecasting**: SeasonalNaive
    3. **Preprocessing pipelines**: FeaturePipeline with LogTransformer, SeasonalDifferencing, LagTransformer
    4. **Decomposition**: DecompositionPipeline with trend + seasonality components
    5. **Cross-validation & search**: ExpandingWindowSplitter, GridSearchCV, RandomizedSearchCV
    6. **Scoring**: point scorers, aggregation modes
    7. **Panel data**: forecasting multiple series with the `__` separator convention
    8. **Time-weighted training**: exponential_decay_weight, compose_weights
    9. **Incremental learning**: observe, predict, observe_predict streaming workflow
    10. **Interval forecasting**: SplitConformalForecaster for prediction intervals

    ## Prerequisites

    Basic Python and familiarity with sklearn's `fit` / `predict` API.
    """)


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
        plot_forecast,
        plot_time_series,
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
        go,
        inspect_locality,
        linear_decay_weight,
        load_air_passengers,
        load_store_sales,
        make_subplots,
        pl,
        plot_forecast,
        plot_time_series,
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


@app.cell
def _(load_air_passengers):
    y = load_air_passengers().rename({"Passengers": "passengers"})
    print(f"Shape: {y.shape}  |  Range: {y['time'].min()} → {y['time'].max()}")
    y.head()
    return (y,)


@app.cell(hide_code=True)
def _(plot_time_series, y):
    plot_time_series(
        y,
        columns="passengers",
        title="Air Passengers (1949 – 1960)",
        y_label="Passengers (thousands)",
        height=380,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Observations**: strong upward trend, yearly seasonality (summer peaks), and
    increasing variance, hallmarks of a non-stationary, multiplicative series.
    """)


@app.cell
def _(train_test_split, y):
    y_train, y_test = train_test_split(y, test_size=0.2, shuffle=False)
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


@app.cell(hide_code=True)
def _(plot_forecast, y_pred_baseline, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_baseline,
        y_train=y_train,
        title="Baseline: Seasonal Naive",
        y_label="Passengers (thousands)",
        height=380,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Preprocessing Pipelines

    Reduction forecasters convert time-series forecasting to supervised learning.
    A **FeaturePipeline** chains invertible transforms:

    1. `LogTransformer`: stabilises multiplicative variance
    2. `SeasonalDifferencing(12)`: removes yearly seasonality and trend
    3. `LagTransformer`: creates autoregressive features from past values

    All transformations are automatically inverted at prediction time.
    """)


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
    return mae_reduction, reduction, y_pred_reduction


@app.cell(hide_code=True)
def _(plot_forecast, y_pred_baseline, y_pred_reduction, y_test, y_train):
    plot_forecast(
        y_test,
        {"Baseline": y_pred_baseline, "Reduction": y_pred_reduction},
        y_train=y_train,
        title="Reduction Forecaster vs Baseline",
        y_label="Passengers (thousands)",
        height=380,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Decomposition Pipeline

    A **DecompositionPipeline** explicitly models structural components:

    - **Trend** via `PolynomialTrendForecaster`
    - **Seasonality** via `FourierSeasonalityForecaster`
    - **Residual** via any forecaster (here `PointReductionForecaster`)

    Each component is fitted on the residuals left by the previous one.
    """)


@app.cell
def _(
    DecompositionPipeline,
    FeaturePipeline,
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
            ("trend", PolynomialTrendForecaster(degree=2)),
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
    return decomp, mae_decomp, y_pred_decomp


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Cross-Validation & Hyperparameter Search

    Standard k-fold CV destroys temporal order. Yohou provides **temporal splitters**:

    - `ExpandingWindowSplitter`: training window grows each fold
    - `SlidingWindowSplitter`: training window slides at fixed size

    Pair with `GridSearchCV` or `RandomizedSearchCV` to tune hyperparameters.
    """)


@app.cell(hide_code=True)
def _(ExpandingWindowSplitter, SlidingWindowSplitter, go, make_subplots, y_train):
    _fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Expanding Window (5 folds)", "Sliding Window (5 folds)"),
        vertical_spacing=0.15,
    )

    for _row, _splitter in enumerate(
        [
            ExpandingWindowSplitter(n_splits=5, test_size=12),
            SlidingWindowSplitter(n_splits=5, test_size=12, stride=12),
        ],
        start=1,
    ):
        for _i, (_tr, _te) in enumerate(_splitter.split(y_train)):
            _t_tr = y_train[_tr]["time"]
            _t_te = y_train[_te]["time"]
            _fig.add_trace(
                go.Scatter(
                    x=[_t_tr[0], _t_tr[-1]], y=[_i, _i],
                    mode="lines", line=dict(color="steelblue", width=8),
                    showlegend=(_i == 0 and _row == 1), name="Train",
                ),
                row=_row, col=1,
            )
            _fig.add_trace(
                go.Scatter(
                    x=[_t_te[0], _t_te[-1]], y=[_i, _i],
                    mode="lines", line=dict(color="orange", width=8),
                    showlegend=(_i == 0 and _row == 1), name="Test",
                ),
                row=_row, col=1,
            )

    _fig.update_layout(height=420, template="plotly_white")
    _fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### RandomizedSearchCV

    Randomly sample from continuous / discrete parameter distributions.
    Efficient for larger search spaces.
    """)


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
    cv = ExpandingWindowSplitter(n_splits=2, test_size=36)

    random_search = RandomizedSearchCV(
        forecaster=clone(reduction),
        param_distributions={
            "estimator__alpha": uniform(0.01, 10.0),
            "feature_transformer__lag__lag": randint(1, 7),
        },
        scoring=MeanAbsoluteError(),
        cv=cv,
        n_iter=10,
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
            "estimator__alpha": [0.1, 1.0, 10.0],
            "feature_transformer__lag__lag": [2, 4, 6],
        },
        scoring=MeanAbsoluteError(),
        cv=ExpandingWindowSplitter(n_splits=2, test_size=36),
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
    return mae_grid, mae_random, y_pred_grid, y_pred_random


@app.cell(hide_code=True)
def _(go, grid_search, mae_grid, mae_random, pl, random_search):
    _fig = go.Figure()

    _rand_res = pl.from_dict(random_search.cv_results_)
    _fig.add_trace(go.Scatter(
        x=list(range(len(_rand_res))),
        y=_rand_res["mean_test_score"].to_list(),
        mode="markers", name="RandomizedSearchCV",
        marker=dict(size=10, color="orange"),
    ))

    _grid_res = pl.from_dict(grid_search.cv_results_)
    _fig.add_trace(go.Scatter(
        x=list(range(len(_grid_res))),
        y=_grid_res["mean_test_score"].to_list(),
        mode="markers", name="GridSearchCV",
        marker=dict(size=10, color="purple", symbol="diamond"),
    ))

    _fig.add_hline(y=mae_random, line_dash="dash", line_color="orange",
                   annotation_text=f"Random best: {mae_random:.2f}")
    _fig.add_hline(y=mae_grid, line_dash="dash", line_color="purple",
                   annotation_text=f"Grid best: {mae_grid:.2f}")
    _fig.update_layout(
        title="Search Strategy Comparison",
        xaxis_title="Trial", yaxis_title="CV MAE",
        height=400, template="plotly_white",
    )
    _fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Scoring & Aggregation

    Yohou scorers extend sklearn with time-series-specific features:

    - **`score(y_truth, y_pred)`** returns a float (default `aggregation_method="timewise"`)
    - Change `aggregation_method` for richer output:
      - `"componentwise"` → per-timestep scores (DataFrame)
      - `"all"` → single scalar across all columns and timesteps
    """)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With `"componentwise"` you get a DataFrame with one error per timestep,
    useful for diagnosing where in the forecast horizon accuracy degrades.

    See `examples/metrics/` for an exhaustive scorer survey including interval
    scorers and conformity scorers.
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Panel Data

    Yohou uses **column prefixes with `__`** to represent panel groups:

    ```
    store_1_item_1__sales   store_2_item_1__sales   store_3_item_1__sales
    ```

    Any forecaster automatically handles all groups when it sees this pattern.
    """)


@app.cell
def _(inspect_locality, load_store_sales, pl):
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

    panel_baseline = SeasonalNaive(seasonality=7)
    panel_baseline.fit(y_panel_train, forecasting_horizon=90)
    y_pred_panel = panel_baseline.predict(forecasting_horizon=90)

    _scorer = MeanAbsoluteError()
    _scorer.fit(y_panel_test)
    print(f"Panel baseline MAE: {_scorer.score(y_panel_test, y_pred_panel):.2f}")
    return y_panel_test, y_panel_train, y_pred_panel


@app.cell(hide_code=True)
def _(go, make_subplots, y_panel_test, y_panel_train, y_pred_panel):
    _cols = [c for c in y_panel_train.columns if c != "time"]
    _fig = make_subplots(rows=len(_cols), cols=1, subplot_titles=_cols, vertical_spacing=0.08)

    for _i, _col in enumerate(_cols, 1):
        _fig.add_trace(
            go.Scatter(x=y_panel_train["time"], y=y_panel_train[_col], name="Train",
                       line=dict(color="steelblue"), showlegend=(_i == 1)),
            row=_i, col=1,
        )
        _fig.add_trace(
            go.Scatter(x=y_panel_test["time"], y=y_panel_test[_col], name="Actual",
                       line=dict(color="gray", dash="dot"), showlegend=(_i == 1)),
            row=_i, col=1,
        )
        _fig.add_trace(
            go.Scatter(x=y_pred_panel["time"], y=y_pred_panel[_col], name="Predicted",
                       line=dict(color="orange"), showlegend=(_i == 1)),
            row=_i, col=1,
        )

    _fig.update_layout(
        height=220 * len(_cols), template="plotly_white",
        title="Panel Forecast: SeasonalNaive (weekly)",
    )
    _fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Time-Weighted Training

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


@app.cell(hide_code=True)
def _(
    exponential_decay_weight,
    go,
    linear_decay_weight,
    make_subplots,
    seasonal_emphasis_weight,
    y_train,
):
    _fns = {
        "Exponential (half_life=15)": exponential_decay_weight(half_life=15),
        "Linear": linear_decay_weight(max_steps=None),
        "Seasonal (period=12, emphasis=3)": seasonal_emphasis_weight(seasonality=12, emphasis=3.0),
    }
    _fig = make_subplots(rows=len(_fns), cols=1, subplot_titles=list(_fns.keys()), vertical_spacing=0.1)

    for _i, (_name, _fn) in enumerate(_fns.items(), 1):
        _w = _fn(y_train["time"])
        _fig.add_trace(
            go.Scatter(y=_w.to_list(), mode="lines", name=_name, showlegend=False),
            row=_i, col=1,
        )

    _fig.update_layout(height=500, template="plotly_white", title="Time Weight Functions")
    _fig


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
    return (y_pred_weighted,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Incremental Learning: `observe` / `predict`

    In production you rarely retrain from scratch. Yohou's **streaming workflow**:

    1. `fit(y_train)`: initial training
    2. `observe(y_new)`: feed new observations (updates memory, NOT model weights)
    3. `predict(horizon)`: forecast from latest observation
    4. `observe_predict(y_new)`: atomic shortcut for observe + predict

    Compare a **static** model (never updates) to an **incremental** one:
    """)


@app.cell
def _(copy, go, reduction, scorer, y_test):
    _static = copy.deepcopy(reduction)
    _incremental = copy.deepcopy(reduction)

    _mae_static, _mae_incr = [], []

    for _i in range(1, len(y_test)):
        _y_cur = y_test[_i : _i + 1]

        # Static: predict growing horizon from training end
        _pred_s = _static.predict(forecasting_horizon=_i + 1).tail(1)
        _mae_static.append(scorer.score(_y_cur, _pred_s))

        # Incremental: observe previous step, predict 1 step ahead
        _pred_i = _incremental.observe_predict(
            y=y_test[_i - 1 : _i], X=None, forecasting_horizon=1,
        )
        _mae_incr.append(scorer.score(_y_cur, _pred_i))

    _steps = list(range(1, len(_mae_static) + 1))
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_steps, y=_mae_static, mode="lines",
        name=f"Static (avg {sum(_mae_static) / len(_mae_static):.1f})",
        line=dict(color="red"),
    ))
    _fig.add_trace(go.Scatter(
        x=_steps, y=_mae_incr, mode="lines",
        name=f"Incremental (avg {sum(_mae_incr) / len(_mae_incr):.1f})",
        line=dict(color="green"),
    ))
    _fig.update_layout(
        title="Rolling MAE: Static vs Incremental",
        xaxis_title="Step", yaxis_title="MAE",
        height=380, template="plotly_white",
    )
    _fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Interval Forecasting: Prediction Intervals

    `SplitConformalForecaster` wraps any point forecaster and produces
    **calibrated prediction intervals** using split conformal prediction.

    Pass `coverage_rates` to `predict_interval` to get intervals at the
    desired confidence levels.
    """)


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
    y_test,
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

    y_pred_interval = conformal.predict_interval(
        forecasting_horizon=forecasting_horizon,
        coverage_rates=[0.9],
    )
    print(f"Interval columns: {y_pred_interval.columns}")
    y_pred_interval.head()
    return (y_pred_interval,)


@app.cell(hide_code=True)
def _(plot_forecast, y_pred_interval, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_interval,
        y_train=y_train,
        coverage_rates=[0.9],
        title="90% Prediction Interval (Split Conformal)",
        y_label="Passengers (thousands)",
        height=400,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **Scikit-learn API**: `fit(y, X, forecasting_horizon)` → `predict(forecasting_horizon)` → `score(y_truth, y_pred)`
    - **Preprocessing**: `FeaturePipeline` chains invertible transforms (log, differencing, lags)
    - **Decomposition**: `DecompositionPipeline` models trend + seasonality + residual
    - **Cross-validation**: `ExpandingWindowSplitter` / `SlidingWindowSplitter` respect temporal order
    - **Hyperparameter search**: `GridSearchCV` (exhaustive) and `RandomizedSearchCV` (efficient)
    - **Panel data**: `__` separator convention: one forecaster handles all groups
    - **Time weighting**: `exponential_decay_weight`, `compose_weights` for recency emphasis
    - **Streaming**: `observe_predict` updates memory without retraining
    - **Intervals**: `SplitConformalForecaster` + `predict_interval(coverage_rates=[0.9])`

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


if __name__ == "__main__":
    app.run()
