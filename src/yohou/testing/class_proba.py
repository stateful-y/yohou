"""Check functions for yohou class-probability forecasters.

This module provides validation functions specific to class-probability
forecasters (BaseClassProbaForecaster implementations).
"""

__all__ = [
    "check_class_proba_classes_attribute",
    "check_class_proba_predict_returns_labels",
    "check_class_proba_prediction_bounds",
    "check_class_proba_prediction_structure",
    "check_class_proba_prediction_sums",
    "check_class_proba_prediction_types",
]

import polars as pl

from yohou.utils import inspect_panel


def check_class_proba_prediction_structure(
    forecaster, y_test: pl.DataFrame, X_test: pl.DataFrame | None = None
) -> None:
    """Check class-probability predictions have correct column structure.

    Validates that ``predict_class_proba`` output contains ``"vintage_time"``
    and ``"time"`` columns, plus ``{target}_proba_{class}`` columns for every
    target and class.

    Parameters
    ----------
    forecaster : BaseClassProbaForecaster
        Fitted class-probability forecaster instance.
    y_test : pl.DataFrame
        Test target data.
    X_test : pl.DataFrame or None, default=None
        Test features.

    Raises
    ------
    AssertionError
        If prediction structure is incorrect.

    """
    forecasting_horizon = min(3, len(y_test))
    y_pred = forecaster.predict_class_proba(forecasting_horizon=forecasting_horizon, X=X_test)

    assert "vintage_time" in y_pred.columns, "Class-proba predictions must have 'vintage_time'"
    assert "time" in y_pred.columns, "Class-proba predictions must have 'time'"

    # Should NOT have interval columns
    interval_cols = [col for col in y_pred.columns if "_lower_" in col or "_upper_" in col]
    assert len(interval_cols) == 0, f"Class-proba predictions should not have interval columns, found: {interval_cols}"

    # Should have _proba_ columns for each target and class
    _, y_panel_groups = inspect_panel(y_test)
    if len(y_panel_groups) > 0:
        # Panel data: classes_ keys are unprefixed target names
        for target_col, class_labels in forecaster.classes_.items():
            for group_prefix in y_panel_groups:
                for label in class_labels:
                    col_name = f"{group_prefix}__{target_col}_proba_{label}"
                    assert col_name in y_pred.columns, f"Missing probability column: {col_name}"
    else:
        for target_col, class_labels in forecaster.classes_.items():
            for label in class_labels:
                col_name = f"{target_col}_proba_{label}"
                assert col_name in y_pred.columns, f"Missing probability column: {col_name}"


def check_class_proba_prediction_bounds(forecaster, y_test: pl.DataFrame, X_test: pl.DataFrame | None = None) -> None:
    """Check all probability values are in [0, 1].

    Parameters
    ----------
    forecaster : BaseClassProbaForecaster
        Fitted class-probability forecaster instance.
    y_test : pl.DataFrame
        Test target data.
    X_test : pl.DataFrame or None, default=None
        Test features.

    Raises
    ------
    AssertionError
        If any probability value is outside [0, 1].

    """
    forecasting_horizon = min(3, len(y_test))
    y_pred = forecaster.predict_class_proba(forecasting_horizon=forecasting_horizon, X=X_test)

    proba_cols = [col for col in y_pred.columns if "_proba_" in col]
    assert len(proba_cols) > 0, "No probability columns found"

    for col in proba_cols:
        values = y_pred[col]
        min_val = values.min()
        max_val = values.max()
        assert min_val >= 0.0, f"Probability column {col} has negative values (min={min_val})"
        assert max_val <= 1.0, f"Probability column {col} has values > 1 (max={max_val})"


def check_class_proba_prediction_sums(forecaster, y_test: pl.DataFrame, X_test: pl.DataFrame | None = None) -> None:
    """Check probabilities sum to approximately 1.0 per row per target.

    Parameters
    ----------
    forecaster : BaseClassProbaForecaster
        Fitted class-probability forecaster instance.
    y_test : pl.DataFrame
        Test target data.
    X_test : pl.DataFrame or None, default=None
        Test features.

    Raises
    ------
    AssertionError
        If probabilities do not sum to approximately 1.0.

    """
    forecasting_horizon = min(3, len(y_test))
    y_pred = forecaster.predict_class_proba(forecasting_horizon=forecasting_horizon, X=X_test)

    _, y_panel_groups = inspect_panel(y_test)

    if len(y_panel_groups) > 0:
        for group_prefix in y_panel_groups:
            for target_col, class_labels in forecaster.classes_.items():
                proba_cols = [f"{group_prefix}__{target_col}_proba_{label}" for label in class_labels]
                row_sums = y_pred.select(proba_cols).sum_horizontal()
                for i, s in enumerate(row_sums):
                    assert abs(s - 1.0) < 1e-6, (
                        f"Probabilities for {group_prefix}__{target_col} at row {i} sum to {s}, expected ~1.0"
                    )
    else:
        for target_col, class_labels in forecaster.classes_.items():
            proba_cols = [f"{target_col}_proba_{label}" for label in class_labels]
            row_sums = y_pred.select(proba_cols).sum_horizontal()
            for i, s in enumerate(row_sums):
                assert abs(s - 1.0) < 1e-6, f"Probabilities for {target_col} at row {i} sum to {s}, expected ~1.0"


def check_class_proba_prediction_types(forecaster) -> None:
    """Check class-proba forecaster has 'class_proba' in forecaster_type tag.

    Parameters
    ----------
    forecaster : BaseClassProbaForecaster
        Class-probability forecaster instance.

    Raises
    ------
    AssertionError
        If forecaster_type is not 'class_proba'.

    """
    tags = forecaster.__sklearn_tags__()
    forecaster_type = tags.forecaster_tags.forecaster_type if tags.forecaster_tags else None

    assert forecaster_type is not None and "class_proba" in forecaster_type, (
        f"Class-proba forecaster should have 'class_proba' in forecaster_type tag, got {forecaster_type}"
    )


def check_class_proba_classes_attribute(forecaster) -> None:
    """Check classes_ and n_classes_ attributes are populated correctly after fit.

    Validates that ``classes_`` is a dict mapping target column names to
    sorted lists of class labels, that ``n_classes_`` maps each target
    to an integer class count, and that ``label_to_code_`` maps each
    label to a numeric code.

    Parameters
    ----------
    forecaster : BaseClassProbaForecaster
        Fitted class-probability forecaster instance.

    Raises
    ------
    AssertionError
        If classes_, n_classes_, or label_to_code_ are invalid.

    """
    assert hasattr(forecaster, "classes_"), "Fitted forecaster must have classes_ attribute"
    assert isinstance(forecaster.classes_, dict), f"classes_ should be dict, got {type(forecaster.classes_)}"
    assert len(forecaster.classes_) > 0, "classes_ should not be empty"

    for target_col, labels in forecaster.classes_.items():
        assert isinstance(labels, list), f"classes_[{target_col!r}] should be list, got {type(labels)}"
        assert len(labels) >= 2, f"classes_[{target_col!r}] should have at least 2 classes, got {len(labels)}"
        assert labels == sorted(labels), f"classes_[{target_col!r}] should be sorted, got {labels}"

    assert hasattr(forecaster, "n_classes_"), "Fitted forecaster must have n_classes_ attribute"
    assert isinstance(forecaster.n_classes_, dict), f"n_classes_ should be dict, got {type(forecaster.n_classes_)}"
    for target_col, n in forecaster.n_classes_.items():
        assert target_col in forecaster.classes_, f"n_classes_ key {target_col!r} not in classes_"
        assert n == len(forecaster.classes_[target_col]), (
            f"n_classes_[{target_col!r}]={n} doesn't match len(classes_[{target_col!r}])={len(forecaster.classes_[target_col])}"
        )

    assert hasattr(forecaster, "label_to_code_"), "Fitted forecaster must have label_to_code_ attribute"
    assert isinstance(forecaster.label_to_code_, dict), (
        f"label_to_code_ should be dict, got {type(forecaster.label_to_code_)}"
    )

    for target_col, mapping in forecaster.label_to_code_.items():
        assert target_col in forecaster.classes_, f"label_to_code_ key {target_col!r} not in classes_"
        assert set(mapping.keys()) == set(forecaster.classes_[target_col]), (
            f"label_to_code_[{target_col!r}] keys {set(mapping.keys())} "
            f"don't match classes_ {set(forecaster.classes_[target_col])}"
        )


def check_class_proba_predict_returns_labels(
    forecaster, y_test: pl.DataFrame, X_test: pl.DataFrame | None = None
) -> None:
    """Check predict() returns argmax class labels, not probabilities.

    Validates that ``predict()`` returns a DataFrame with string class labels
    (the most-likely class per target), not probability columns.

    Parameters
    ----------
    forecaster : BaseClassProbaForecaster
        Fitted class-probability forecaster instance.
    y_test : pl.DataFrame
        Test target data.
    X_test : pl.DataFrame or None, default=None
        Test features.

    Raises
    ------
    AssertionError
        If predict() output is incorrect.

    """
    forecasting_horizon = min(3, len(y_test))
    y_pred = forecaster.predict(forecasting_horizon=forecasting_horizon, X=X_test)

    assert "vintage_time" in y_pred.columns, "predict() must have 'vintage_time'"
    assert "time" in y_pred.columns, "predict() must have 'time'"

    # Should NOT have _proba_ columns
    proba_cols = [col for col in y_pred.columns if "_proba_" in col]
    assert len(proba_cols) == 0, f"predict() should not have _proba_ columns, found: {proba_cols}"

    # Should have target columns with string/categorical labels
    target_cols = [col for col in y_pred.columns if col not in ("vintage_time", "time")]
    assert len(target_cols) > 0, "predict() must have at least one target column"

    for col in target_cols:
        # Strip panel prefix if present
        base_col = col.split("__")[-1] if "__" in col else col
        if base_col in forecaster.classes_:
            values = y_pred[col].cast(pl.String).unique().to_list()
            valid_classes = set(forecaster.classes_[base_col])
            for val in values:
                assert val in valid_classes, (
                    f"predict() column {col!r} has value {val!r} not in classes {valid_classes}"
                )
