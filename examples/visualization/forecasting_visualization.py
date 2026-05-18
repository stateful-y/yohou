# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn",
#     "yohou[plotting]",
# ]
# ///

import marimo

__generated_with = "0.20.2"
__gallery__ = {
    "title": "Forecast Visualization",
    "description": "Visualise point forecasts from single and multiple models, decomposition pipeline components, and time weight decay functions with interactive Plotly.",
    "category": "tutorial",
    "companion": "/pages/tutorials/forecast-visualization/",
    "section": "visualization",
    "api_references": ["ClassProbaReductionForecaster", "DecompositionPipeline", "PatternSeasonalityForecaster", "PointReductionForecaster", "PolynomialTrendForecaster", "SeasonalNaive", "SplitConformalForecaster", "plot_calibration", "plot_decomposition", "plot_time_weight"],
}
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import polars as pl
    from sklearn.tree import DecisionTreeClassifier

    from yohou.class_proba import ClassProbaReductionForecaster
    from yohou.compose import DecompositionPipeline
    from yohou.datasets import (
        fetch_air_quality_classification,
        fetch_sunspot,
        fetch_tourism_monthly,
    )
    from yohou.interval import SplitConformalForecaster
    from yohou.model_selection import train_test_split
    from yohou.plotting import (
        plot_calibration,
        plot_decomposition,
        plot_forecast,
        plot_time_weight,
    )
    from yohou.point import PointReductionForecaster, SeasonalNaive
    from yohou.preprocessing import LagTransformer
    from yohou.stationarity import (
        PatternSeasonalityForecaster,
        PolynomialTrendForecaster,
    )
    from yohou.utils.weighting import (
        exponential_decay_weight,
        linear_decay_weight,
    )

    return (
        ClassProbaReductionForecaster,
        DecisionTreeClassifier,
        DecompositionPipeline,
        LagTransformer,
        PatternSeasonalityForecaster,
        PointReductionForecaster,
        PolynomialTrendForecaster,
        SeasonalNaive,
        SplitConformalForecaster,
        exponential_decay_weight,
        fetch_air_quality_classification,
        fetch_sunspot,
        fetch_tourism_monthly,
        linear_decay_weight,
        pl,
        plot_calibration,
        plot_decomposition,
        plot_forecast,
        plot_time_weight,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Forecast Visualization and Comparison

    In this notebook we will plot forecasts from single and multiple models,
    visualize categorical and class-probability predictions, check probability
    calibration, inspect decomposition components, and render time weight
    functions.

    ## Prerequisites

    Basic familiarity with yohou's fit/predict API (see `examples/quickstart.py`).
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prepare Data and Models

    We load the Monthly Tourism dataset and the Sunspot dataset, then fit three
    forecasters: [`SeasonalNaive`](/pages/api/generated/yohou.point.naive.SeasonalNaive/) (repeats the last seasonal cycle),
    [`PointReductionForecaster`](/pages/api/generated/yohou.point.reduction.PointReductionForecaster/) (sklearn regressor with lag features), and
    [`SplitConformalForecaster`](/pages/api/generated/yohou.interval.split_conformal.SplitConformalForecaster/) (wraps a point forecaster to produce prediction
    intervals via conformal calibration). Their predictions are reused across
    all visualization sections below.
    """)


@app.cell
def _(fetch_tourism_monthly, train_test_split):

    tourism = (
        fetch_tourism_monthly().frame.select("time", "T1__tourists").drop_nulls().rename({"T1__tourists": "tourists"})
    )
    fh = 12
    y_train, y_test = train_test_split(tourism, test_size=fh)
    return fh, y_test, y_train


@app.cell
def _(PointReductionForecaster, SeasonalNaive, fh, y_train):
    naive = SeasonalNaive(seasonality=12)
    naive.fit(y_train, forecasting_horizon=fh)
    y_pred_naive = naive.predict(forecasting_horizon=fh)

    reduction = PointReductionForecaster()
    reduction.fit(y_train, forecasting_horizon=fh)
    y_pred_reduction = reduction.predict(forecasting_horizon=fh)
    return y_pred_naive, y_pred_reduction


@app.cell
def _(SeasonalNaive, SplitConformalForecaster, fetch_sunspot, fh):
    sunspots = fetch_sunspot().frame
    ss_train = sunspots.head(len(sunspots) - fh)
    ss_test = sunspots.tail(fh)

    conformal = SplitConformalForecaster(
        point_forecaster=SeasonalNaive(seasonality=132),
    )
    conformal.fit(ss_train, forecasting_horizon=fh)
    y_pred_conformal = conformal.predict_interval(
        forecasting_horizon=fh,
        coverage_rates=[0.7],
    )
    return ss_test, ss_train, y_pred_conformal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Single Forecast Plot

    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) displays actuals vs predictions. Pass **y_train** to include
    history, and use **n_history** to trim the visible window.
    """)


@app.cell
def _(plot_forecast, y_pred_naive, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_naive,
        y_train=y_train,
        title="Seasonal Naive - Full History",
    )


@app.cell
def _(plot_forecast, y_pred_naive, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_naive,
        y_train=y_train,
        n_history=50,
        title="Seasonal Naive - Last 50 Steps of History",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Multi-Model Forecast

    Pass a **dict** of predictions to overlay multiple models. For interval
    forecasters, include **coverage_rates** to display prediction bands.
    Toggle **show_transition** to control the training/test boundary marker.
    """)


@app.cell
def _(plot_forecast, y_pred_naive, y_pred_reduction, y_test):
    plot_forecast(
        y_test,
        {"Naive": y_pred_naive, "Reduction": y_pred_reduction},
        title="Multi-Model Comparison (Overlay)",
    )


@app.cell
def _(plot_forecast, ss_test, ss_train, y_pred_conformal):
    plot_forecast(
        ss_test,
        y_pred_conformal,
        y_train=ss_train,
        coverage_rates=[0.7],
        n_history=120,
        title="Conformal Forecaster - 70% PI on Sunspot",
    )


@app.cell
def _(plot_forecast, y_pred_naive, y_test, y_train):
    plot_forecast(
        y_test,
        y_pred_naive,
        y_train=y_train,
        show_transition=False,
        title="Seasonal Naive - No Transition Marker",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Categorical (Hard-Label) Forecasts

    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) **auto-detects** columns with `String` or `Categorical`
    dtype and renders a **step chart**. Each class is mapped to a horizontal
    band so temporal transitions between categories are easy to follow.

    We use [`fetch_air_quality_classification`](/pages/api/generated/yohou.datasets._fetchers.fetch_air_quality_classification/)
    (4-class air quality target derived from PM2.5 data) and fit a
    [`ClassProbaReductionForecaster`](/pages/api/generated/yohou.class_proba.reduction.ClassProbaReductionForecaster/)
    with a [`DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).
    """)


@app.cell
def _(
    ClassProbaReductionForecaster,
    DecisionTreeClassifier,
    LagTransformer,
    fetch_air_quality_classification,
    train_test_split,
):
    cls_data = fetch_air_quality_classification()
    cls_y, cls_X = cls_data.y, cls_data.X_actual
    cls_y_train, cls_y_test, cls_X_train, cls_X_test = train_test_split(
        cls_y,
        cls_X,
        test_size=200,
    )
    cls_fh = 24

    cls_forecaster = ClassProbaReductionForecaster(
        estimator=DecisionTreeClassifier(random_state=42),
        feature_transformer=LagTransformer(lag=[1, 2, 3, 6, 12, 24]),
    )
    cls_forecaster.fit(cls_y_train, cls_X_train, forecasting_horizon=cls_fh)

    cls_y_pred_labels = cls_forecaster.predict(
        forecasting_horizon=cls_fh,
    )
    cls_y_proba = cls_forecaster.predict_class_proba(
        forecasting_horizon=cls_fh,
    )
    return cls_X_test, cls_fh, cls_forecaster, cls_y_pred_labels, cls_y_proba, cls_y_test, cls_y_train


@app.cell
def _(cls_y_pred_labels, cls_y_test, cls_y_train, plot_forecast):
    plot_forecast(
        cls_y_test,
        cls_y_pred_labels,
        y_train=cls_y_train,
        n_history=50,
        title="Categorical Forecast with Training History",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Multi-Model Categorical Comparison

    Passing a **dict** of categorical predictions overlays step traces for
    each model. Here we compare two rolling forecasts produced by
    `observe_predict()` so the model updates its memory at every step.
    """)


@app.cell
def _(cls_X_test, cls_fh, cls_forecaster, cls_y_test, cls_y_train, plot_forecast):
    cls_forecaster_obs = cls_forecaster
    cls_obs_labels = cls_forecaster_obs.observe_predict(
        cls_y_test[:cls_fh],
        X_actual=cls_X_test[:cls_fh],
        forecasting_horizon=cls_fh,
    )
    plot_forecast(
        cls_y_test,
        {
            "Static": cls_forecaster.predict(forecasting_horizon=cls_fh),
            "After Observe": cls_obs_labels,
        },
        y_train=cls_y_train,
        n_history=50,
        title="Multi-Model Categorical Comparison",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Class-Probability (Soft) Forecasts

    When predictions contain `_proba_` columns (from `predict_class_proba()`),
    [`plot_forecast`](/pages/api/generated/yohou.plotting.forecasting.plot_forecast/) renders a **stacked area chart** where each class is a
    coloured band whose height equals the predicted probability. Diamond
    markers show the true class at each time step.
    """)


@app.cell
def _(cls_y_proba, cls_y_test, plot_forecast):
    plot_forecast(
        cls_y_test,
        cls_y_proba,
        title="Class-Probability Forecast (Stacked Area)",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reliability Diagram

    [`plot_calibration`](/pages/api/generated/yohou.plotting.evaluation.plot_calibration/)
    compares predicted probabilities against empirical frequencies. A
    perfectly calibrated model follows the diagonal. Use **n_bins** to
    control the number of probability bins.
    """)


@app.cell
def _(cls_y_proba, cls_y_test, plot_calibration):
    cls_y_truth = cls_y_test.head(len(cls_y_proba))
    plot_calibration(
        cls_y_proba,
        cls_y_truth,
        target="air_quality",
        n_bins=8,
        title="Reliability Diagram - Air Quality",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Decomposition Visualization

    [`plot_decomposition`](/pages/api/generated/yohou.plotting.forecasting.plot_decomposition/) displays a fitted [`DecompositionPipeline`](/pages/api/generated/yohou.compose.decomposition_pipeline.DecompositionPipeline/)'s components
    as stacked subplots. Toggle **show_original** and pass a subset of components.
    """)


@app.cell
def _(
    DecompositionPipeline,
    PatternSeasonalityForecaster,
    PolynomialTrendForecaster,
    fh,
    y_train,
):
    decomp = DecompositionPipeline(
        forecasters=[
            ("trend", PolynomialTrendForecaster(degree=2)),
            ("seasonality", PatternSeasonalityForecaster(seasonality=12)),
        ],
        store_residuals=True,
    )
    decomp.fit(y_train, forecasting_horizon=fh)

    decomp_components = {}
    for _name, _fc, *_ in decomp.forecasters_:
        decomp_components[_name] = _fc.predict(forecasting_horizon=fh)
    return (decomp_components,)


@app.cell
def _(decomp_components, fh, plot_decomposition, y_test):
    plot_decomposition(
        y_test.tail(fh),
        decomp_components,
        show_original=True,
        title="Decomposition - Trend + Seasonality with Original",
    )


@app.cell
def _(decomp_components, fh, plot_decomposition, y_train):
    plot_decomposition(
        y_train.tail(fh),
        decomp_components,
        show_original=False,
        title="Decomposition - Components Only",
    )


@app.cell
def _(decomp_components, fh, plot_decomposition, y_test):
    plot_decomposition(
        y_test.tail(fh),
        {"trend": decomp_components["trend"]},
        title="Decomposition - Trend Only",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Time Weight Visualization

    [`plot_time_weight`](/pages/api/generated/yohou.plotting.forecasting.plot_time_weight/) renders weighting functions over time. Toggle the
    **fill** kwarg for an area chart and adjust **fill_opacity**.
    """)


@app.cell(hide_code=True)
def _(exponential_decay_weight, linear_decay_weight, pl, y_train):
    _times = y_train["time"]
    _exp_fn = exponential_decay_weight(half_life=24)
    _lin_fn = linear_decay_weight()

    exp_weight_df = pl.DataFrame({"time": _times, "time_weight": _exp_fn(_times)})
    lin_weight_df = pl.DataFrame({"time": _times, "time_weight": _lin_fn(_times)})
    return exp_weight_df, lin_weight_df


@app.cell
def _(exp_weight_df, plot_time_weight):
    plot_time_weight(
        exp_weight_df,
        fill=True,
        title="Exponential Decay Weight (half_life=24)",
    )


@app.cell
def _(lin_weight_df, plot_time_weight):
    plot_time_weight(
        lin_weight_df,
        fill=False,
        title="Linear Decay Weight - Line Only",
    )


@app.cell
def _(lin_weight_df, plot_time_weight):
    plot_time_weight(
        lin_weight_df,
        fill=True,
        fill_opacity=0.5,
        title="Linear Decay Weight - Semi-Transparent Fill",
    )



@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Next Steps

        - [Forecast Visualization](/pages/tutorials/forecast-visualization/) for the full guide
        - [Visualize Forecasts](/pages/how-to/visualize-forecasts/) for related techniques
        """
    )
    return

if __name__ == "__main__":
    app.run()
