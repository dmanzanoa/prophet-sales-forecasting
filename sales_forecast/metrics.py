"""Evaluation metrics for regression and forecasting.

This module defines simple functions to compute error metrics commonly used in
forecasting: mean absolute percentage error (MAPE), symmetric MAPE (SMAPE),
mean absolute error (MAE), root mean squared error (RMSE) and the coefficient
of determination (R²).  These functions operate on numpy arrays or list-like
objects and return float results.  A convenience function ``compute_metrics``
aggregates them into a dictionary.
"""

from typing import Sequence, Dict, Optional
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute percentage error (MAPE).

    MAPE expresses the prediction error as a percentage of the true values.
    Values close to zero indicate better performance.  Note that MAPE can
    produce extreme values when the true series contains zeros; this function
    clips the divisor at a very small positive value to avoid division by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100.0)


def smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Symmetric mean absolute percentage error (SMAPE).

    SMAPE is similar to MAPE but symmetrically penalises over- and
    under-estimation by dividing the absolute error by the average of the
    absolute true and predicted values.  SMAPE values are in the range
    ``[0, 200]``; lower values indicate better forecasts.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(
        np.mean(
            2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
        )
        * 100.0
    )


def compute_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, Optional[float]]:
    """Compute a suite of regression metrics.

    The returned dictionary contains MAE, RMSE, R², MAPE and SMAPE.  When the
    true series has fewer than two non-NaN values the R² score is set to
    ``None`` because it is undefined in that case.

    Parameters
    ----------
    y_true : sequence of float
        The ground truth values.
    y_pred : sequence of float
        The predicted values.

    Returns
    -------
    dict
        A dictionary with keys ``MAE``, ``RMSE``, ``R2``, ``MAPE``, ``SMAPE``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = None
    # r2_score requires at least two samples to be defined
    if len(y_true) > 1 and np.any(~np.isnan(y_true)):
        try:
            r2 = float(r2_score(y_true, y_pred))
        except ValueError:
            r2 = None
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
    }
