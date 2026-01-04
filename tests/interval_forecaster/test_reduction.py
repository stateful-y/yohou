from datetime import datetime

import polars as pl
import polars.testing
import pytest
from sklearn.model_selection import train_test_split

from yohou.interval_forecaster import IntervalReductionForecaster

length = 22

time = pl.DataFrame(
    {
        "time": pl.datetime_range(
            start=datetime(2021, 12, 16),
            end=datetime(2021, 12, 16, 0, 0, length - 1),
            interval="1s",
            eager=True,
        ),
    }
)
y = pl.DataFrame(
    {
        "a": range(length),
        "b": range(10, length + 10),
    },
    schema={
        "a": pl.Float64,
        "b": pl.Float64,
    },
)
y = pl.concat([time, y], how="horizontal")

X = pl.DataFrame(
    {
        "c": range(length),
        "d": range(10, length + 10),
        "e": range(20, length + 20),
    },
    schema={
        "c": pl.Float64,
        "d": pl.Float64,
        "e": pl.Float64,
    },
)
X = pl.concat([time, X], how="horizontal")

y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.2, shuffle=False)


@pytest.mark.parametrize(
    "fit_forecasting_horizon, predict_forecasting_horizon, expected_a",
    [
        (1, 5, [17, 17.4, 17.56, 17.624, 17.6496]),
        (3, 5, [17, 18, 19, 18.2, 19.2]),
        (3, 2, [17, 18]),
    ],
)
def test_predict(fit_forecasting_horizon, predict_forecasting_horizon, expected_a):
    coverage_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    forecaster = IntervalReductionForecaster(coverage_rates=coverage_rates)

    forecaster.fit(y=y_train, X=X_train, forecasting_horizon=fit_forecasting_horizon)

    y_pred = forecaster.predict(
        forecasting_horizon=predict_forecasting_horizon,
        X=X_test,
        predict_transformed=False,
    )

    # Extract non-time column names from y_train
    y_columns = [col for col in y_train.columns if col != "time"]
    for col in y_columns:
        for coverage_rate in coverage_rates:
            assert all(
                y_pred[f"{col}_upper_{coverage_rate}"] + 1e-14
                >= y_pred[f"{col}_lower_{coverage_rate}"]
            )


@pytest.mark.parametrize(
    "fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a",
    [
        (1, 5, 1, [17, 17.4, 17.56, 17.624, 17.6496]),
        (3, 5, 2, [17, 18, 19, 18.2, 19.2]),
        (3, 2, 1, [17, 18]),
    ],
)
def test_update_predict(fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a):
    coverage_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    forecaster = IntervalReductionForecaster(coverage_rates=coverage_rates)

    forecaster.fit(y=y_train, X=X_train, forecasting_horizon=fit_forecasting_horizon)

    # Truncate y_test to ensure X_test covers the future horizon
    y_test_truncated = y_test[:-predict_forecasting_horizon]

    y_pred = forecaster.update_predict(
        y=y_test_truncated,
        X=X_test,
        forecasting_horizon=predict_forecasting_horizon,
        stride=stride,
    )

    # Extract non-time column names from y_train
    y_columns = [col for col in y_train.columns if col != "time"]
    for col in y_columns:
        for coverage_rate in coverage_rates:
            assert all(
                y_pred[f"{col}_upper_{coverage_rate}"] + 1e-14
                >= y_pred[f"{col}_lower_{coverage_rate}"]
            )


y_panel = pl.DataFrame(
    {
        "x__a": range(length),
        "x__b": range(10, length + 10),
        "y__a": range(10, length + 10),
        "y__b": range(20, length + 20),
    }
)
y_panel = pl.concat([time, y_panel], how="horizontal")

X_panel = pl.DataFrame(
    {
        "x__c": range(length),
        "y__c": range(10, length + 10),
        "d": range(10, length + 10),
        "e": range(20, length + 20),
    }
)
X_panel = pl.concat([time, X_panel], how="horizontal")

y_train_panel, y_test_panel, X_train_panel, X_test_panel = train_test_split(
    y_panel, X_panel, test_size=0.2, shuffle=False
)


@pytest.mark.parametrize(
    "fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a",
    [
        (1, 5, 1, [17, 17.4, 17.56, 17.624, 17.6496]),
        (3, 5, 2, [17, 18, 19, 18.2, 19.2]),
        (3, 2, 1, [17, 18]),
    ],
)
def test_update_predict_global(
    fit_forecasting_horizon, predict_forecasting_horizon, stride, expected_a
):
    coverage_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    forecaster = IntervalReductionForecaster(coverage_rates=coverage_rates)

    forecaster.fit(
        y=y_train_panel,
        X=X_train_panel,
        forecasting_horizon=fit_forecasting_horizon,
    )

    # Truncate y_test to ensure X_test covers the future horizon
    y_test_truncated = y_test_panel[:-predict_forecasting_horizon]

    y_pred = forecaster.update_predict(
        y=y_test_truncated,
        X=X_test_panel,
        forecasting_horizon=predict_forecasting_horizon,
        stride=stride,
    )

    # Check that upper bounds >= lower bounds for columns with __ separator
    # Columns are like group1__a_lower_0.1, group1__a_upper_0.1, etc.
    for col in y_pred.columns:
        if col in ["time", "observed_time"]:
            continue
        if "_upper_" in col:
            # Extract coverage rate and find corresponding lower bound
            parts = col.split("_upper_")
            if len(parts) == 2:
                lower_col = f"{parts[0]}_lower_{parts[1]}"
                if lower_col in y_pred.columns:
                    assert all(
                        y_pred[col] + 1e-13 >= y_pred[lower_col]
                    ), f"Upper bound {col} should be >= lower bound {lower_col}"


def test_y_pred_local_columns_interval_global_data():
    """Test y_pred_local_columns_ for interval forecaster with global data."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 31),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": range(31)})

    forecaster = IntervalReductionForecaster(coverage_rates=[0.1, 0.5, 0.9])
    forecaster.fit(y[:20], forecasting_horizon=3)

    # Should match local_y_t_schema_ keys (separate estimators for lower/upper)
    assert hasattr(forecaster, "y_pred_local_columns_")
    assert forecaster.y_pred_local_columns_ == ["value"]


def test_y_pred_local_columns_interval_multiple_coverage():
    """Test y_pred_local_columns_ with multiple coverage rates."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 31),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({"time": time, "value": range(31)})

    # Multiple coverage rates - should still use same y_pred_local_columns_
    forecaster = IntervalReductionForecaster(coverage_rates=[0.1, 0.3, 0.5, 0.7, 0.9])
    forecaster.fit(y[:20], forecasting_horizon=3)

    assert forecaster.y_pred_local_columns_ == ["value"]


def test_y_pred_local_columns_interval_multiple_targets():
    """Test y_pred_local_columns_ with multiple target columns."""
    time = pl.datetime_range(
        start=datetime(2020, 1, 1),
        end=datetime(2020, 1, 31),
        interval="1d",
        eager=True,
    )
    y = pl.DataFrame({
        "time": time,
        "sales": range(31),
        "revenue": [x * 10 for x in range(31)],
    })

    forecaster = IntervalReductionForecaster(coverage_rates=[0.5])
    forecaster.fit(y[:20], forecasting_horizon=3)

    # Should match all target columns
    assert set(forecaster.y_pred_local_columns_) == {"sales", "revenue"}


def test_y_pred_local_columns_interval_panel_data(panel_time_series_factory):
    """Test y_pred_local_columns_ for interval forecaster with panel data."""
    y = panel_time_series_factory(length=50, n_series=2, seed=42)

    forecaster = IntervalReductionForecaster(coverage_rates=[0.1, 0.9])
    forecaster.fit(y[:30], forecasting_horizon=3)

    # Should match local_y_t_schema_ keys (separate estimators for lower/upper)
    assert hasattr(forecaster, "y_pred_local_columns_")
    assert set(forecaster.y_pred_local_columns_) == set(forecaster.local_y_t_schema_.keys())
