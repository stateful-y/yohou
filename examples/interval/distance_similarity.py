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
    "title": "Distance-Based Similarity",
    "description": "Adaptive prediction intervals via similarity-weighted conformal prediction using DistanceSimilarity with configurable distance metrics and bandwidths.",
    "category": "how-to",
    "companion": "/pages/explanation/interval-forecasting/#similarity-based-adaptive-intervals",
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Distance-Based Similarity Weighting

    Standard conformal prediction gives every calibration residual equal weight.
    [`DistanceSimilarity`](/pages/api/generated/yohou.interval.similarity.DistanceSimilarity/) instead up-weights residuals from calibration points
    **similar** to the current test point, yielding **adaptive** prediction
    intervals that narrow where the model is confident and widen where it
    is uncertain.

    ## What You'll Learn

    - What similarity-based conformal prediction is and why it matters
    - Configuring [`DistanceSimilarity`](/pages/api/generated/yohou.interval.similarity.DistanceSimilarity/) with different distance metrics
    - Plugging it into [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/) via the `similarity` parameter
    - Comparing interval widths and calibration with vs without similarity weighting
    - Tuning `metric_params` for metrics like Minkowski

    ## Prerequisites

    Familiarity with [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/) and basic conformal prediction.
    """)


@app.cell(hide_code=True)
def _():
    from copy import deepcopy

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    from yohou.datasets import fetch_tourism_monthly
    from yohou.interval import DistanceSimilarity, SplitConformalForecaster
    from yohou.metrics import (
        AbsoluteResidual,
        IntervalScore,
        MeanIntervalWidth,
    )
    from yohou.plotting import (
        plot_calibration,
        plot_forecast,
        plot_score_heatmap,
        plot_score_per_step,
        plot_score_per_vintage,
    )
    from yohou.point import PointReductionForecaster
    from yohou.preprocessing import LagTransformer

    return (
        AbsoluteResidual,
        DistanceSimilarity,
        IntervalScore,
        LagTransformer,
        MeanIntervalWidth,
        PointReductionForecaster,
        Ridge,
        SplitConformalForecaster,
        deepcopy,
        fetch_tourism_monthly,
        plot_calibration,
        plot_forecast,
        plot_score_heatmap,
        plot_score_per_step,
        plot_score_per_vintage,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Prepare Data

    We use Monthly Tourism, its clear trend and seasonality make
    similarity effects visible because the error structure changes over time.
    """)


@app.cell
def _(fetch_tourism_monthly, train_test_split):
    y = fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})

    y_train, y_test = train_test_split(y, test_size=24, shuffle=False)
    forecasting_horizon = len(y_test)
    return forecasting_horizon, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Baseline: Standard Conformal (No Similarity)

    First we build a standard [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/) **without** similarity
    weighting. All calibration residuals get equal weight.
    """)


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
    coverage = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

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
    _y_point = conformal_standard.predict(forecasting_horizon=forecasting_horizon)
    y_pred_standard = y_pred_standard.hstack(_y_point.drop("time", "vintage_time"))
    return coverage, y_pred_standard


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) displays the standard conformal intervals with uniform
    weighting across all calibration residuals.
    """)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Euclidean Similarity

    Now we add `DistanceSimilarity(metric="euclidean")`. Calibration points
    whose lag features are close to the test point get higher weight, producing
    intervals that adapt to local error structure.
    """)


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
    _y_point = conformal_euclidean.predict(forecasting_horizon=forecasting_horizon)
    y_pred_euclidean = y_pred_euclidean.hstack(_y_point.drop("time", "vintage_time"))
    return (y_pred_euclidean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) shows how Euclidean similarity weighting narrows or
    widens intervals depending on the local calibration residuals.
    """)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Comparing Distance Metrics

    [`DistanceSimilarity`](/pages/api/generated/yohou.interval.similarity.DistanceSimilarity/) accepts any metric from `scipy.spatial.distance.cdist`.
    Let us compare **euclidean**, **cosine**, and **cityblock** (Manhattan distance)
    side by side.
    """)


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
    MeanIntervalWidth,
    mo,
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

    mo.md("### Mean Interval Width (90% coverage)\n\n| Method | Width |\n|--------|-------|\n" + "\n".join(_rows))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Calibration Assessment

    Good intervals should achieve empirical coverage close to the nominal rate.
    [`plot_calibration`](/pages/api/generated/yohou.plotting.evaluation.plot_calibration/) shows whether each method is over- or under-covering.
    """)


@app.cell
def _(coverage, plot_calibration, y_pred_standard, y_test):
    plot_calibration(
        y_pred_standard,
        y_test,
        coverage_rates=coverage,
        title="Calibration: Standard Conformal",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_calibration`](/pages/api/generated/yohou.plotting.evaluation.plot_calibration/) for the Euclidean similarity approach shows whether
    the locally-weighted intervals achieve their nominal coverage.
    """)


@app.cell
def _(coverage, plot_calibration, y_pred_euclidean, y_test):
    plot_calibration(
        y_pred_euclidean,
        y_test,
        coverage_rates=coverage,
        title="Calibration: Euclidean Similarity",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Horizon Degradation

    Prediction intervals typically widen at longer horizons. [`plot_score_per_step`](/pages/api/generated/yohou.plotting.evaluation.plot_score_per_step/)
    shows how interval scores change across forecast steps.
    """)


@app.cell
def _(
    IntervalScore,
    plot_score_per_step,
    y_pred_euclidean,
    y_pred_standard,
    y_test,
):
    plot_score_per_step(
        IntervalScore(coverage_rates=[0.90]),
        y_test,
        {"standard": y_pred_standard, "euclidean": y_pred_euclidean},
        kind="line",
        title="Interval Score per Horizon Step (90% Coverage)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Custom metric_params

    For metrics like Minkowski, pass `metric_params` to control the distance:
    `metric_params={"p": 1.5}` gives a p-norm between Manhattan (p=1) and
    Euclidean (p=2).
    """)


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
        calibration_size=36,
        similarity=DistanceSimilarity(metric="minkowski", metric_params={"p": 2.5}),
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
    _y_point = conformal_minkowski.predict(forecasting_horizon=forecasting_horizon)
    y_pred_minkowski = y_pred_minkowski.hstack(_y_point.drop("time", "vintage_time"))
    return (y_pred_minkowski,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) shows how the Minkowski distance (p=1.5) affects
    the resulting interval shape.
    """)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-vintage Scoring

    The `observe_predict_interval` method with `stride=1` produces one
    interval forecast per observation point, creating multiple *vintages*.
    Each vintage represents a different forecast origin, so you can analyse
    how interval quality evolves as the model absorbs more data.
    """)


@app.cell
def _(conformal_standard, coverage, deepcopy, forecasting_horizon, y_test):
    _vintage_model = deepcopy(conformal_standard)
    y_pred_vintages = _vintage_model.observe_predict_interval(
        y=y_test,
        stride=1,
        forecasting_horizon=forecasting_horizon,
        coverage_rates=coverage,
    )
    print(f"Vintages: {y_pred_vintages['vintage_time'].n_unique()}")
    y_pred_vintages.head(10)
    return (y_pred_vintages,)


@app.cell
def _(IntervalScore, y_train):
    vintage_scorer = IntervalScore()
    vintage_scorer.fit(y_train)
    return (vintage_scorer,)


@app.cell
def _(vintage_scorer, plot_score_per_vintage, y_pred_vintages, y_test):
    plot_score_per_vintage(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="Interval Score per Vintage",
        y_label="Interval Score",
        height=380,
    )


@app.cell
def _(vintage_scorer, plot_score_heatmap, y_pred_vintages, y_test):
    plot_score_heatmap(
        vintage_scorer,
        y_test,
        y_pred_vintages,
        title="Score Heatmap (Step x Vintage)",
    )


if __name__ == "__main__":
    app.run()
