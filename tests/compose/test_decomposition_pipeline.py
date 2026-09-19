"""DecompositionPipeline forwards the actual transformer's step columns to its components."""

from sklearn.linear_model import LinearRegression

from point.test_step_output_alignment import _StepProbe
from yohou.compose import DecompositionPipeline, FeatureUnion
from yohou.point import PointReductionForecaster
from yohou.preprocessing import LagTransformer


def test_actual_step_columns_reach_components(y_X_factory):
    """Only derived step columns are stripped before forwarding; step-output columns survive."""
    y, X, X_future, _ = y_X_factory(
        length=80, n_targets=1, n_features=1, n_future_features=1, forecasting_horizon=3, return_exogenous=True
    )
    pipeline = DecompositionPipeline(
        [
            ("a", PointReductionForecaster(estimator=LinearRegression(), nan_handling="drop")),
            ("b", PointReductionForecaster(estimator=LinearRegression(), nan_handling="drop")),
        ],
        actual_transformer=FeatureUnion([("lag", LagTransformer(lag=1)), ("seasonal", _StepProbe())]),
    )
    pipeline.fit(y, X, forecasting_horizon=3, X_future=X_future)

    probe = {f"seasonal_X_0_probe_step_{h}" for h in range(1, 4)}
    assert pipeline._step_column_names_ == {f"F_0_step_{h}" for h in range(1, 4)}
    assert pipeline._actual_step_column_names_ == probe
    for _, forecaster in pipeline.forecasters_:
        assert probe <= set(forecaster._X_t_observed.columns)
