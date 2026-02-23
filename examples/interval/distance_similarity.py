"""Distance-Based Similarity Weighting for Conformal Prediction.

Demonstrates DistanceSimilarity and its role in adaptive conformal intervals.
"""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
async def _():
    import sys as _sys

    if "pyodide" in _sys.modules:
        import micropip

        await micropip.install(["plotly", "scikit-learn", "yohou"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Distance-Based Similarity Weighting

    Standard conformal prediction gives every calibration residual equal weight.
    `DistanceSimilarity` instead up-weights residuals from calibration points
    **similar** to the current test point, yielding **adaptive** prediction
    intervals that narrow where the model is confident and widen where it
    is uncertain.

    ## What You'll Learn

    - What similarity-based conformal prediction is and why it matters
    - Configuring `DistanceSimilarity` with different distance metrics
    - Plugging it into `SplitConformalForecaster` via the `similarity` parameter
    - Comparing interval widths and calibration with vs without similarity weighting
    - Tuning `metric_params` for metrics like Minkowski

    ## Prerequisites

    Familiarity with `SplitConformalForecaster` and basic conformal prediction
    (see `examples/interval/conformal_forecasting.py`).
    """)
    return


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.linear_model import Ridge

    from yohou.datasets import fetch_tourism_monthly
    from yohou.interval import DistanceSimilarity, SplitConformalForecaster
    from yohou.metrics import (
        AbsoluteResidual,
        EmpiricalCoverage,
        IntervalScore,
        MeanIntervalWidth,
    )
    from yohou.plotting import plot_calibration, plot_forecast, plot_score_per_horizon
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        AbsoluteResidual,
        DistanceSimilarity,
        EmpiricalCoverage,
        IntervalScore,
        LagTransformer,
        MeanIntervalWidth,
        PointReductionForecaster,
        Ridge,
        SplitConformalForecaster,
        fetch_tourism_monthly,
        pl,
        plot_calibration,
        plot_forecast,
        plot_score_per_horizon,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We use Monthly Tourism, its clear trend and seasonality make
    similarity effects visible because the error structure changes over time.
    """)
    return


@app.cell
def _(fetch_tourism_monthly):
    y = (
        fetch_tourism_monthly()
        .frame.select("time", "T1__tourists")
        .rename({"T1__tourists": "passengers"})
    )

    split_idx = int(len(y) * 0.8)
    y_train = y.head(split_idx)
    y_test = y.tail(len(y) - split_idx)
    forecasting_horizon = len(y_test)
    return forecasting_horizon, split_idx, y, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Baseline: Standard Conformal (No Similarity)

    First we build a standard `SplitConformalForecaster` **without** similarity
    weighting. All calibration residuals get equal weight.
    """)
    return


@app.cell
def _(
    AbsoluteResidual,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SplitConformalForecaster,
    forecasting_horizon,
    y_train,
):
    coverage = [0.80, 0.90, 0.95]

    base_forecaster = PointReductionForecaster(
        estimator=Ridge(alpha=1.0),
        feature_transformer=LagTransformer(lag=[1, 2, 3, 12]),
    )

    conformal_standard = SplitConformalForecaster(
        point_forecaster=base_forecaster,
        conformity_scorer=AbsoluteResidual(),
        calibration_size=30,
        similarity=None,
    )
    conformal_standard.fit(
        y_train,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage,
    )

    y_pred_standard = conformal_standard.predict_interval(
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage,
    )
    return conformal_standard, coverage, y_pred_standard


@app.cell
def _(plot_forecast, y_pred_standard, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_standard,
        y_train=y_train,
        coverage_rates=[0.90],
        n_history=36,
        title="Standard Conformal (Uniform Weights)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Euclidean Similarity

    Now we add `DistanceSimilarity(metric="euclidean")`. Calibration points
    whose lag features are close to the test point get higher weight, producing
    intervals that adapt to local error structure.
    """)
    return


@app.cell
def _(
    AbsoluteResidual,
    DistanceSimilarity,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SplitConformalForecaster,
    coverage,
    forecasting_horizon,
    y_train,
):
    conformal_euclidean = SplitConformalForecaster(
        point_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 2, 3, 12]),
        ),
        conformity_scorer=AbsoluteResidual(),
        calibration_size=30,
        similarity=DistanceSimilarity(metric="euclidean"),
    )
    conformal_euclidean.fit(
        y_train,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage,
    )

    y_pred_euclidean = conformal_euclidean.predict_interval(
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage,
    )
    return conformal_euclidean, y_pred_euclidean


@app.cell
def _(plot_forecast, y_pred_euclidean, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_euclidean,
        y_train=y_train,
        coverage_rates=[0.90],
        n_history=36,
        title="Euclidean Similarity Weighting",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Comparing Distance Metrics

    `DistanceSimilarity` accepts any metric from `scipy.spatial.distance.cdist`.
    Let us compare **euclidean**, **cosine**, and **cityblock** (Manhattan distance)
    side by side.
    """)
    return


@app.cell
def _(
    AbsoluteResidual,
    DistanceSimilarity,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SplitConformalForecaster,
    coverage,
    forecasting_horizon,
    y_train,
):
    _metrics_to_compare = ["cosine", "cityblock"]
    similarity_predictions = {}

    for _metric in _metrics_to_compare:
        _forecaster = SplitConformalForecaster(
            point_forecaster=PointReductionForecaster(
                estimator=Ridge(alpha=1.0),
                feature_transformer=LagTransformer(lag=[1, 2, 3, 12]),
            ),
            conformity_scorer=AbsoluteResidual(),
            calibration_size=30,
            similarity=DistanceSimilarity(metric=_metric),
        )
        _forecaster.fit(
            y_train,
            forecasting_horizon=forecasting_horizon,
            coverage_rates=coverage,
        )
        similarity_predictions[_metric] = _forecaster.predict_interval(
            forecasting_horizon=forecasting_horizon,
            coverage_rates=coverage,
        )
    return (similarity_predictions,)


@app.cell
def _(
    mo,
    MeanIntervalWidth,
    similarity_predictions,
    y_pred_euclidean,
    y_pred_standard,
    y_test,
    y_train,
):
    _width_scorer = MeanIntervalWidth(coverage_rates=[0.90])
    _width_scorer.fit(y_train)
    _all_preds = {
        "standard": y_pred_standard,
        "euclidean": y_pred_euclidean,
        **similarity_predictions,
    }
    _rows = []
    for _name, _pred in _all_preds.items():
        _width = float(_width_scorer.score(y_test, _pred))
        _rows.append(f"| {_name} | {_width:.2f} |")

    mo.md(
        "### Mean Interval Width (90% coverage)\n\n"
        "| Method | Width |\n|--------|-------|\n"
        + "\n".join(_rows)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Calibration Assessment

    Good intervals should achieve empirical coverage close to the nominal rate.
    `plot_calibration` shows whether each method is over- or under-covering.
    """)
    return


@app.cell
def _(plot_calibration, coverage, y_pred_standard, y_test):
    plot_calibration(
        y_pred_standard,
        y_test,
        coverage_rates=coverage,
        title="Calibration: Standard Conformal",
    )
    return


@app.cell
def _(plot_calibration, coverage, y_pred_euclidean, y_test):
    plot_calibration(
        y_pred_euclidean,
        y_test,
        coverage_rates=coverage,
        title="Calibration: Euclidean Similarity",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Horizon Degradation

    Prediction intervals typically widen at longer horizons. `plot_score_per_horizon`
    shows how interval scores change across forecast steps.
    """)
    return


@app.cell
def _(IntervalScore, plot_score_per_horizon, y_pred_euclidean, y_pred_standard, y_test):
    plot_score_per_horizon(
        IntervalScore(coverage_rates=[0.90]),
        y_test,
        {"standard": y_pred_standard, "euclidean": y_pred_euclidean},
        kind="line",
        title="Interval Score per Horizon Step (90% Coverage)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Custom metric_params

    For metrics like Minkowski, pass `metric_params` to control the distance:
    `metric_params={"p": 1.5}` gives a p-norm between Manhattan (p=1) and
    Euclidean (p=2).
    """)
    return


@app.cell
def _(
    AbsoluteResidual,
    DistanceSimilarity,
    LagTransformer,
    PointReductionForecaster,
    Ridge,
    SplitConformalForecaster,
    coverage,
    forecasting_horizon,
    y_train,
):
    conformal_minkowski = SplitConformalForecaster(
        point_forecaster=PointReductionForecaster(
            estimator=Ridge(alpha=1.0),
            feature_transformer=LagTransformer(lag=[1, 2, 3, 12]),
        ),
        conformity_scorer=AbsoluteResidual(),
        calibration_size=30,
        similarity=DistanceSimilarity(
            metric="minkowski", metric_params={"p": 1.5}
        ),
    )
    conformal_minkowski.fit(
        y_train,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage,
    )

    y_pred_minkowski = conformal_minkowski.predict_interval(
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage,
    )
    return conformal_minkowski, y_pred_minkowski


@app.cell
def _(plot_forecast, y_pred_minkowski, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_minkowski,
        y_train=y_train,
        coverage_rates=[0.90],
        n_history=36,
        title="Minkowski (p=1.5) Similarity",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    - **DistanceSimilarity** is the concrete `BaseSimilarity` subclass that produces locally-weighted conformal intervals
    - **Adaptive intervals** narrow in well-predicted regions and widen where the model struggles, unlike uniform conformal
    - **Distance metric choice** matters: `euclidean` is the default; `cosine` ignores magnitude; `cityblock` is more robust to outliers
    - **metric_params** allow fine-tuning (e.g., Minkowski p-norm) without subclassing
    - **Calibration** should be checked after any similarity choice: local weighting can improve or hurt coverage depending on the data

    ## Next Steps

    - **Conformity scorers**: See `examples/interval/conformal_conformity_scorers.py` for comparing Residual, GammaResidual, etc.
    - **Panel intervals**: See `examples/interval/panel_intervals.py` for prediction intervals on panel data
    - **Interval metrics**: See `examples/metrics/interval_metrics.py` for EmpiricalCoverage, IntervalScore, and more
    """)
    return


if __name__ == "__main__":
    app.run()
