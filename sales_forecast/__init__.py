"""Sales forecasting package using Prophet.

This package contains utility functions for loading and preprocessing data,
metric computations, and a simple wrapper around Facebook’s Prophet library to
train and forecast models on grouped time series.  See the README in the
repository root for usage instructions.
"""

from .data import load_data, winsorize_per_region  # noqa: F401
from .metrics import mape, smape, compute_metrics  # noqa: F401
from .model import ProphetForecaster  # noqa: F401

__all__ = [
    "load_data",
    "winsorize_per_region",
    "mape",
    "smape",
    "compute_metrics",
    "ProphetForecaster",
]
