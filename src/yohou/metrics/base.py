"""Base classes for forecasting metrics and scoring functions."""

import abc

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator


class BaseScorer(BaseEstimator, metaclass=abc.ABCMeta):
    """Base class for all forecasting metrics.

    Defines the interface for scoring forecast quality. All scorers must implement
    the :meth:`score` method and can optionally override :meth:`fit` for metrics
    that require training data statistics.

    Attributes
    ----------
    prediction_types : set of str
        Types of predictions this scorer evaluates ({"point"} or {"interval"}).

    """

    @property
    @abc.abstractmethod
    def prediction_types(self) -> set[str]:
        """Get the prediction types this scorer handles.

        Returns
        -------
        set of str
            Either {"point"} or {"interval"}.

        """
        raise NotImplementedError()

    def _validate_inputs(
        self, y_truth: pl.DataFrame, y_pred: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Align ground truth and predictions by matching time indices.

        Ensures that predictions and actuals are properly aligned and removes
        time columns after alignment for metric computation.

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth values with "time" column.

        y_pred : pl.DataFrame
            Predicted values with "observed_time" and "time" columns.

        Returns
        -------
        y_truth : pl.DataFrame
            Aligned ground truth with time column removed.

        y_pred : pl.DataFrame
            Aligned predictions with time columns removed.

        """
        y_truth = y_truth.join(y_pred[["time"]], on="time")
        y_pred = y_pred.filter(pl.col("time").is_in(y_truth["time"].implode()))

        y_truth = y_truth.drop("time")
        y_pred = y_pred.drop("observed_time", "time")

        return y_truth, y_pred

    def fit(self, y_train: pl.DataFrame | None) -> "BaseScorer":
        """Fit the scorer on training data if needed.

        Most metrics are stateless and don't require fitting, but some (e.g.,
        scale-dependent metrics) may need training data statistics.

        Parameters
        ----------
        y_train : pl.DataFrame or None
            Training set target values.

        Returns
        -------
        self

        """
        return self

    @abc.abstractmethod
    def score(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Compute the metric score.

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth values with "time" column.

        y_pred : pl.DataFrame
            Predicted values with "observed_time" and "time" columns.

        Returns
        -------
        pl.DataFrame
            Metric value dataframe. Lower is better for error metrics.

        """
        raise NotImplementedError()

    def __call__(self, y_truth: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
        """Compute score using callable interface.

        Enables using scorers as functions: scorer(y_truth, y_pred).

        Parameters
        ----------
        y_truth : pl.DataFrame
            Ground truth values.

        y_pred : pl.DataFrame
            Predicted values.

        Returns
        -------
        pl.DataFrame
            Metric score dataframe.

        """
        return self.score(y_truth, y_pred)


class BasePointScorer(BaseScorer, metaclass=abc.ABCMeta):
    """Base class for point forecast metrics.

    Point forecasters produce single-value predictions. Metrics derived from this
    class evaluate prediction accuracy (e.g., MAE, RMSE, MAPE).

    See Also
    --------
    :mod:`yohou.metrics.point` : Concrete implementations (MAE, MSE, RMSE, MAPE)
    :class:`yohou.point_forecaster.base.BasePointForecaster` : Produces point forecasts

    """

    @property
    def prediction_types(self) -> set[str]:
        """Get the prediction types this scorer handles.

        Returns
        -------
        set of str
            {"point"}

        """
        return {"point"}


class BaseIntervalScorer(BaseScorer, metaclass=abc.ABCMeta):
    """Base class for interval forecast metrics.

    Interval forecasters produce prediction intervals. Metrics derived from this
    class evaluate coverage and width trade-offs.

    See Also
    --------
    :mod:`yohou.metrics.interval` : Concrete implementations
    :class:`yohou.interval_forecaster.base.BaseIntervalForecaster` : Produces intervals

    """

    @property
    def prediction_types(self) -> set[str]:
        """Get the prediction types this scorer handles.

        Returns
        -------
        set of str
            {"interval"}

        """
        return {"interval"}


class BaseConformityScorer(BaseScorer, metaclass=abc.ABCMeta):
    """Base class for conformal prediction conformity scorers.

    Conformity scorers quantify how "unusual" a prediction is compared to the
    calibration set. Used in conformal prediction to construct valid prediction
    intervals with coverage guarantees.

    See Also
    --------
    :mod:`yohou.metrics.conformity` : Concrete conformity scorers
    :class:`yohou.interval_forecaster.split_conformal.SplitConformalForecaster` : Uses conformity
    scores

    """

    @staticmethod
    def _compute_assymetric_quantiles(
        conformity_scores: pl.DataFrame, coverage_rate: float
    ) -> tuple[float, float]:
        """Compute lower and upper quantiles for asymmetric intervals.

        Parameters
        ----------
        conformity_scores : pl.DataFrame
            Conformity scores from calibration.

        coverage_rate : float
            Target coverage rate.

        Returns
        -------
        lower_quantile : float
            Lower quantile value.

        upper_quantile : float
            Upper quantile value.

        """
        lower_quantile: float = np.quantile(conformity_scores, coverage_rate / 2.0, method="lower")  # type: ignore[call-overload]

        upper_quantile: float = np.quantile(
            conformity_scores, 1 - coverage_rate / 2.0, method="upper"
        )  # type: ignore[call-overload]

        return lower_quantile, upper_quantile

    @staticmethod
    def _compute_symetric_quantiles(conformity_scores: pl.DataFrame, coverage_rate: float) -> float:
        """Compute quantile for symmetric intervals.

        Parameters
        ----------
        conformity_scores : pl.DataFrame
            Conformity scores from calibration.

        coverage_rate : float
            Target coverage rate.

        Returns
        -------
        float
            Quantile value for symmetric intervals.

        """
        quantile: float = np.quantile(conformity_scores, 1 - coverage_rate, method="lower")  # type: ignore[call-overload]

        return quantile

    @staticmethod
    def _format_y_pred_interval(
        lower_bound: pl.DataFrame, upper_bound: pl.DataFrame, coverage_rate: float
    ) -> pl.DataFrame:
        """Format lower and upper bounds into interval DataFrame.

        Parameters
        ----------
        lower_bound : pl.DataFrame
            Lower bound predictions.

        upper_bound : pl.DataFrame
            Upper bound predictions.

        coverage_rate : float
            Coverage rate for labeling columns.

        Returns
        -------
        pl.DataFrame
            Formatted prediction intervals.

        """
        lower_bound.columns = [f"{col}_lower_{coverage_rate}" for col in lower_bound.columns]
        upper_bound.columns = [f"{col}_upper_{coverage_rate}" for col in upper_bound.columns]

        y_pred_interval = pl.concat([lower_bound, upper_bound], how="horizontal")

        return y_pred_interval

    @abc.abstractmethod
    def inverse_score(
        self, y_pred: pl.DataFrame, conformity_scores: pl.DataFrame, coverage_rate: float
    ) -> pl.DataFrame:
        """Transform conformity scores into prediction intervals.

        Parameters
        ----------
        y_pred : pl.DataFrame
            Point predictions.

        conformity_scores : pl.DataFrame
            Conformity scores from calibration.

        coverage_rate : float
            Target coverage probability.

        Returns
        -------
        pl.DataFrame
            Prediction intervals.

        """
        raise NotImplementedError()
