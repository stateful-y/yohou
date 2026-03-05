# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Reduction Strategies",
    "description": "Compare multi-output, direct, and dir-rec reduction strategies for multi-step forecasting, including target_as_feature control and fitted estimator inspection.",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reduction Strategies

    [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) converts time series forecasting into tabular
    regression. The `reduction_strategy` parameter controls **how** multi-step
    forecasts are produced from the tabularized data.

    ## What You'll Learn

    - The three reduction strategies: **multi-output**, **direct**, and **dir-rec**
    - How each strategy affects the fitted `estimator_` attribute
    - Comparing strategies visually and with [`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.point.MeanAbsoluteError/)
    - How `target_as_feature` controls feature construction
    - When to choose each strategy

    ## Prerequisites

    Basic familiarity with [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) -
    see [`reduction_forecaster.py`](/examples/point/reduction_forecaster/) for an introduction.
    """)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    from yohou.datasets import fetch_sunspot
    from yohou.metrics import MeanAbsoluteError
    from yohou.plotting import plot_forecast, plot_model_comparison_bar, plot_time_series
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        LagTransformer,
        MeanAbsoluteError,
        PointReductionForecaster,
        Ridge,
        fetch_sunspot,
        pl,
        plot_forecast,
        plot_model_comparison_bar,
        plot_time_series,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load and Prepare Data

    We use the Sunspot dataset resampled to monthly frequency -
    over 200 years of solar activity with a strong ~11-year cycle.
    """)


@app.cell
def _(fetch_sunspot, pl, train_test_split):
    y = fetch_sunspot().frame.group_by_dynamic("time", every="1mo").agg(pl.col("sunspot_number").mean())

    y_train, y_test = train_test_split(y, test_size=0.05, shuffle=False)
    forecasting_horizon = 24

    print(f"Training: {len(y_train)} obs, Test: {len(y_test)} obs")
    print(f"Forecasting horizon: {forecasting_horizon} months")
    return forecasting_horizon, y, y_test, y_train


@app.cell
def _(plot_time_series, y):
    plot_time_series(y, title="Monthly Sunspot Numbers (1818-2020)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Multi-Output Strategy (Default)

    A **single model** predicts all H horizon steps simultaneously.
    The sklearn estimator is wrapped in `MultiOutputRegressor` (if needed)
    to output H values at once.

    - Simple and fast: one model to train
    - Assumes the same model structure suits every step
    - `estimator_` is a **single** `BaseEstimator`
    """)


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, forecasting_horizon, y_test, y_train):
    fc_multi = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        reduction_strategy="multi-output",
        feature_transformer=LagTransformer(lag=list(range(1, 25))),
    )
    fc_multi.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred_multi = fc_multi.predict(forecasting_horizon=len(y_test))

    print(f"estimator_ type: {type(fc_multi.estimator_).__name__}")
    y_pred_multi.head()
    return fc_multi, y_pred_multi


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Direct Strategy

    **H independent models**, one per horizon step. Each model specialises
    in predicting a single step ahead, avoiding the error accumulation
    that recursive prediction can suffer from.

    - No error propagation between steps
    - Ignores inter-step dependencies
    - `estimator_` is a **list** of H `BaseEstimator` objects
    """)


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, forecasting_horizon, y_test, y_train):
    fc_direct = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        reduction_strategy="direct",
        feature_transformer=LagTransformer(lag=list(range(1, 25))),
    )
    fc_direct.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred_direct = fc_direct.predict(forecasting_horizon=len(y_test))

    print(f"estimator_ type: {type(fc_direct.estimator_).__name__}")
    print(f"Number of fitted models: {len(fc_direct.estimator_)}")
    y_pred_direct.head()
    return fc_direct, y_pred_direct


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Dir-Rec Strategy (Direct-Recursive Hybrid)

    **H models fitted sequentially**. Model h predicts step h using the original
    features **augmented** with in-sample predictions from models 1 to h-1.
    This combines per-step specialisation with inter-step information flow.

    - Each model sees progressively more features
    - Captures dependencies between horizon steps
    - `estimator_` is a **list** of H `BaseEstimator` objects (like direct)
    """)


@app.cell
def _(LagTransformer, PointReductionForecaster, Ridge, forecasting_horizon, y_test, y_train):
    fc_dirrec = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        reduction_strategy="dir-rec",
        feature_transformer=LagTransformer(lag=list(range(1, 25))),
    )
    fc_dirrec.fit(y_train, forecasting_horizon=forecasting_horizon)
    y_pred_dirrec = fc_dirrec.predict(forecasting_horizon=len(y_test))

    print(f"estimator_ type: {type(fc_dirrec.estimator_).__name__}")
    print(f"Number of fitted models: {len(fc_dirrec.estimator_)}")
    print(f"Original features: {fc_dirrec._dir_rec_n_original_features_}")
    y_pred_dirrec.head()
    return fc_dirrec, y_pred_dirrec


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Visual Comparison

    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) accepts a `dict[str, pl.DataFrame]` to overlay
    multiple models on the same chart.
    """)


@app.cell
def _(plot_forecast, y_pred_direct, y_pred_dirrec, y_pred_multi, y_test, y_train):
    plot_forecast(
        y_test,
        {
            "Multi-Output": y_pred_multi,
            "Direct": y_pred_direct,
            "Dir-Rec": y_pred_dirrec,
        },
        y_train=y_train,
        n_history=120,
        title="Reduction Strategy Comparison",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. MAE Comparison

    We score each strategy with [`MeanAbsoluteError`](/pages/api/generated/yohou.metrics.point.MeanAbsoluteError/)
    and compare them in a bar chart.
    """)


@app.cell
def _(
    MeanAbsoluteError,
    plot_model_comparison_bar,
    y_pred_direct,
    y_pred_dirrec,
    y_pred_multi,
    y_test,
    y_train,
):
    mae = MeanAbsoluteError()
    mae.fit(y_train)

    scores = {}
    for _name, _pred in [("Multi-Output", y_pred_multi), ("Direct", y_pred_direct), ("Dir-Rec", y_pred_dirrec)]:
        _y_test_trimmed = y_test.head(len(_pred))
        scores[_name] = {"MAE": mae.score(_y_test_trimmed, _pred)}

    plot_model_comparison_bar(scores, title="MAE by Reduction Strategy")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Feature Construction with `target_as_feature`

    The `target_as_feature` parameter controls what enters the feature matrix
    before `feature_transformer` generates lags:

    | Value | Features built from | Use case |
    |---|---|---|
    | `"transformed"` (default) | Target after `target_transformer` | Lag features see the transformed scale |
    | `"raw"` | Original untransformed target | Lag features on raw scale even when target is transformed |
    | `None` | Exogenous X only (no target) | Purely exogenous-driven forecasting (requires X) |

    We compare `"transformed"` vs `"raw"` with the direct strategy.
    Setting `target_as_feature=None` is useful when exogenous features (X) are
    available and you want to exclude the target from the feature matrix entirely.
    """)


@app.cell
def _(
    LagTransformer,
    MeanAbsoluteError,
    PointReductionForecaster,
    Ridge,
    forecasting_horizon,
    mo,
    y_test,
    y_train,
):
    taf_scores = {}
    for _taf in ["transformed", "raw"]:
        _fc = PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            reduction_strategy="direct",
            target_as_feature=_taf,
            feature_transformer=LagTransformer(lag=list(range(1, 25))),
        )
        _fc.fit(y_train, forecasting_horizon=forecasting_horizon)
        _pred = _fc.predict(forecasting_horizon=len(y_test))
        _mae = MeanAbsoluteError()
        _mae.fit(y_train)
        taf_scores[_taf] = _mae.score(y_test.head(len(_pred)), _pred)

    mo.ui.table(
        [{"target_as_feature": k, "MAE": f"{v:.2f}"} for k, v in taf_scores.items()],
        label="MAE by target_as_feature",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. When to Use Each Strategy

    | Strategy | Models fitted | Error accumulation | Inter-step dependencies | Best for |
    |---|---|---|---|---|
    | **Multi-output** | 1 | None (single model) | Implicit (joint prediction) | Fast baseline, short horizons |
    | **Direct** | H | None (independent) | None | Long horizons, heterogeneous steps |
    | **Dir-Rec** | H (sequential) | Partial (augmented features) | Explicit (feature augmentation) | When steps depend on prior predictions |

    All strategies can be **applied recursively** for horizons beyond the fit horizon
    by passing a larger `forecasting_horizon` at predict time.
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - `reduction_strategy` controls how multi-step forecasts are produced from tabularized data
    - **Multi-output**: one model, all steps at once - simple and fast
    - **Direct**: H independent models - avoids error accumulation
    - **Dir-Rec**: H sequential models with feature augmentation - captures inter-step dependencies
    - `target_as_feature` controls whether lags come from the transformed target, raw target, or neither
    - For direct and dir-rec, `estimator_` becomes a `list[BaseEstimator]` of length H
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next Steps

    - **Reduction basics**: See [`reduction_forecaster.py`](/examples/point/reduction_forecaster/) for transformers and GridSearchCV
    - **Interval strategies**: See [`interval_reduction.py`](/examples/interval/interval_reduction/) for quantile regression with reduction strategies
    - **Panel data**: See [`panel_reduction.py`](/examples/point/panel_reduction/) for panel strategies (orthogonal to reduction strategies)
    - **Time weighting**: See [`time_weighted_reduction.py`](/examples/point/time_weighted_reduction/) for sample weight alignment
    """)


if __name__ == "__main__":
    app.run()
