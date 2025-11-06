"""ProphetForecaster class for training and forecasting time series by region.

This module defines a lightweight wrapper around Facebook’s Prophet library.  It
handles the repetitive tasks of creating a model with common hyper‑parameters,
adding conditional seasonalities and holiday effects, and fitting separate
models for each region in a grouped dataset.  During training, it can also
compute standard error metrics by reserving the last `horizon` observations as
a hold‑out set when sufficient history exists.  Predictions are generated
simply by calling the ``predict_region`` method with a trained instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Iterable

import pandas as pd
import numpy as np
from prophet import Prophet
import joblib

from .metrics import compute_metrics


def _es_cierre(ts: pd.Timestamp) -> bool:
    """Return True if a timestamp falls within the final week of its month.

    In many business settings, sales behaviour changes near the end of a month
    (e.g. due to closing of ledgers or promotional campaigns).  This helper
    identifies dates whose day component is >=24, indicating they occur in the
    closing week of the month.  The returned value can be used to define
    conditional seasonalities in Prophet.
    """
    # Accept either pandas Timestamp or objects convertible via pd.to_datetime
    ts = pd.to_datetime(ts)
    return ts.day >= 24


@dataclass
class ProphetForecaster:
    """Train and use Prophet models for grouped time series.

    Parameters
    ----------
    prediction_length : int
        How many periods ahead each model should forecast when training.
    country_holidays : Optional[str], default None
        Country code to pass to Prophet's built‑in holiday generator.  If None,
        no holidays are added.  See ``Prophet.add_country_holidays`` for
        supported country names.
    use_holidays : bool, default True
        If False, holiday effects will not be included even if a country
        name is provided.
    """

    prediction_length: int = 4
    country_holidays: Optional[str] = None
    use_holidays: bool = True
    models: Dict[str, Prophet] = field(default_factory=dict, init=False)
    history: Optional[pd.DataFrame] = field(default=None, init=False)

    # ---------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------
    def _create_model(self) -> Prophet:
        """Instantiate a Prophet model with default hyper‑parameters.

        The configuration mirrors typical use cases for weekly data.  You can
        customise this method in subclasses or by editing the source code to
        experiment with different seasonality settings or changepoint priors.
        """
        m = Prophet(
            yearly_seasonality=52,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_range=1.0,
            changepoint_prior_scale=0.5,
        )
        # Add holidays if requested and supported by Prophet
        if self.use_holidays and self.country_holidays:
            try:
                m.add_country_holidays(country_name=self.country_holidays)
            except Exception:
                # Ignore failures silently; Prophet will warn internally if the
                # country code is not recognised.
                pass
        # Monthly seasonality captures intra‑month patterns
        m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        # Conditional seasonalities based on end‑of‑month flag
        m.add_seasonality(name="cierre", period=1.0, fourier_order=3, condition_name="es_cierre")
        m.add_seasonality(name="no_cierre", period=1.0, fourier_order=3, condition_name="no_es_cierre")
        return m

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required columns and derive end‑of‑month indicators.

        Prophet expects at least two columns: ``ds`` (dates) and ``y`` (values).
        This helper also adds boolean columns ``es_cierre`` and
        ``no_es_cierre`` based on whether the date falls in the last week of
        its month.  These columns are used as conditional regressors.
        """
        out = df.copy()
        if "ds" not in out.columns or "y" not in out.columns:
            raise ValueError("DataFrame must contain 'ds' and 'y' columns")
        out["ds"] = pd.to_datetime(out["ds"])
        out.sort_values("ds", inplace=True)
        out["es_cierre"] = out["ds"].apply(_es_cierre).astype(bool)
        out["no_es_cierre"] = ~out["es_cierre"]
        return out

    # ---------------------------------------------------------------------
    # Public methods
    # ---------------------------------------------------------------------
    def train_region(
        self,
        df_region: pd.DataFrame,
        evaluate: bool = True,
        horizon: Optional[int] = None,
    ) -> Tuple[Prophet, Dict[str, Optional[float]], pd.DataFrame]:
        """Fit a Prophet model on one region and optionally compute metrics.

        The model is trained on the provided DataFrame.  If ``evaluate`` is
        True and the DataFrame contains at least double the number of records
        specified by ``horizon`` (or ``self.prediction_length`` if not given),
        the last ``horizon`` observations are held out for evaluation.  The
        returned DataFrame contains the forecast for the next ``horizon``
        periods beyond the last date in the input data.

        Parameters
        ----------
        df_region : pd.DataFrame
            Data for a single region with at least ``ds`` and ``y`` columns.
        evaluate : bool, default True
            Whether to compute error metrics using a hold‑out set.
        horizon : int, optional
            Forecast horizon.  Defaults to ``self.prediction_length``.

        Returns
        -------
        model : Prophet
            The fitted Prophet model.
        metrics : dict
            A dictionary of evaluation metrics; values are ``None`` if
            evaluation was not performed.
        forecast : pd.DataFrame
            A DataFrame with columns ``ds``, ``yhat``, ``yhat_lower`` and
            ``yhat_upper`` for the forecast horizon.  The region column is
            added later by the caller.
        """
        if horizon is None:
            horizon = self.prediction_length
        # Prepare data and derive regressors
        prepared = self._prepare_dataframe(df_region)
        n = len(prepared)
        # Split into training and testing sets if evaluation is requested and possible
        if evaluate and n >= 2 * horizon and horizon > 0:
            train_df = prepared.iloc[:-horizon].copy()
            test_df = prepared.iloc[-horizon:].copy()
        else:
            train_df = prepared.copy()
            test_df = pd.DataFrame(columns=prepared.columns)
        # Initialise and fit model
        m = self._create_model()
        m.fit(train_df)
        # Create future dataframe for prediction
        # Prophet will extend from the last observed date by the specified number of periods
        future = m.make_future_dataframe(periods=horizon, freq="W-MON")
        # Add regressors for future dates
        future["es_cierre"] = future["ds"].apply(_es_cierre)
        future["no_es_cierre"] = ~future["es_cierre"]
        # Fill any missing columns with zeros (e.g. additional regressors)
        future = future.fillna(0)
        fcst = m.predict(future)
        # Extract only the forecasts for the horizon
        fcst_horizon = fcst.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        # Compute metrics if there is a test set
        if not test_df.empty:
            pred = fcst.iloc[-horizon:]["yhat"].values
            true = test_df["y"].values
            metrics = compute_metrics(true, pred)
        else:
            metrics = {"MAE": None, "RMSE": None, "R2": None, "MAPE": None, "SMAPE": None}
        return m, metrics, fcst_horizon

    def fit(
        self,
        df: pd.DataFrame,
        evaluate: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit Prophet models for all regions in the provided DataFrame.

        The DataFrame must contain a ``region`` column indicating which rows
        belong together.  After fitting, the trained models and original
        history are stored on the instance.  Metrics and forecasts for all
        regions are returned to the caller as DataFrames.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with columns ``ds``, ``y`` and ``region``.
        evaluate : bool, default True
            Whether to compute evaluation metrics by holding out the last
            ``prediction_length`` periods for each region when possible.

        Returns
        -------
        metrics_df : pd.DataFrame
            Per‑region evaluation metrics.
        forecasts_df : pd.DataFrame
            Forecasts for the next ``prediction_length`` periods for each region.
        """
        metrics_rows = []
        fcst_rows: Iterable[pd.DataFrame] = []  # type: ignore[assignment]
        history_list = []
        for region, group in df.groupby("region"):
            model, metrics, fcst = self.train_region(group, evaluate=evaluate)
            self.models[region] = model
            metrics_rows.append({"region": region, **metrics})
            fcst["region"] = region
            fcst_rows.append(fcst)
            history_list.append(group)
        # Concatenate results
        metrics_df = pd.DataFrame(metrics_rows)
        forecasts_df = pd.concat(list(fcst_rows), ignore_index=True)
        self.history = pd.concat(history_list, ignore_index=True)
        return metrics_df, forecasts_df

    def save(
        self,
        model_path: str,
        history_path: str,
        metrics_df: pd.DataFrame,
        forecasts_df: pd.DataFrame,
    ) -> None:
        """Persist trained models and associated artefacts to disk.

        Parameters
        ----------
        model_path : str
            File path to save the dictionary of trained models using joblib.
        history_path : str
            File path to save the full training history as a Parquet file.
        metrics_df : pd.DataFrame
            DataFrame of evaluation metrics to write alongside the models.
        forecasts_df : pd.DataFrame
            DataFrame of last‑horizon forecasts to write alongside the models.
        """
        # Save models as a joblib file
        joblib.dump(self.models, model_path)
        # Save the history so that future predictions can be made without retraining
        if self.history is not None:
            self.history.to_parquet(history_path, index=False)
        # Also persist metrics and forecasts to CSV for convenience
        metrics_file = model_path.replace(".pkl", "_metrics.csv")
        forecasts_file = model_path.replace(".pkl", "_forecast.csv")
        metrics_df.to_csv(metrics_file, index=False)
        forecasts_df.to_csv(forecasts_file, index=False)

    def load(self, model_path: str, history_path: str) -> None:
        """Load previously saved models and history from disk.

        This method populates the instance’s ``models`` and ``history``
        attributes.  It does not return anything.
        """
        self.models = joblib.load(model_path)
        self.history = pd.read_parquet(history_path)

    def predict_region(
        self,
        region: str,
        horizon: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate future forecasts for a single region.

        Parameters
        ----------
        region : str
            The region key whose model should be used.
        horizon : int, optional
            Number of periods to forecast.  Defaults to ``self.prediction_length``.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns ``ds``, ``yhat``, ``yhat_lower``,
            ``yhat_upper`` and ``region``.  Dates are strings in ISO format for
            ease of serialisation.
        """
        if horizon is None:
            horizon = self.prediction_length
        if region not in self.models:
            raise KeyError(f"Region '{region}' has no trained model")
        if self.history is None:
            raise RuntimeError("History must be loaded before generating forecasts")
        # Retrieve model and history for region
        model = self.models[region]
        hist_region = self.history[self.history["region"] == region].copy()
        if hist_region.empty:
            raise ValueError(f"No history available for region '{region}'")
        last_ds = pd.to_datetime(hist_region["ds"].max())
        # Build future dates from the first Monday after the last observation
        future_ds = pd.date_range(
            last_ds + pd.offsets.Week(1, weekday=0), periods=horizon, freq="W-MON"
        )
        future = pd.DataFrame({"ds": future_ds})
        future["es_cierre"] = future["ds"].apply(_es_cierre)
        future["no_es_cierre"] = ~future["es_cierre"]
        future = future.fillna(0)
        forecast = model.predict(future)
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        result["region"] = region
        result["ds"] = result["ds"].astype(str)  # convert to string for easier JSON serialisation
        return result
