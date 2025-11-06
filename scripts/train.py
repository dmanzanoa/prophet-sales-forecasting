#!/usr/bin/env python3
"""Train Prophet models for grouped time series.

This command line script wraps the ``ProphetForecaster`` class in a
convenient interface.  It reads a CSV file with columns ``ds``, ``y`` and
``region``, optionally applies per‑region winsorization, trains one model
per region and persists the results to disk.

Usage example::

    python scripts/train.py --input-csv data/train.csv \
        --output-dir artifacts --prediction-length 6 --country-holidays Australia \
        --winsorize --evaluate

The above command will train models forecasting six weeks ahead, include
Australian public holidays, winsorize the target variable to mitigate
outliers and compute evaluation metrics if the history permits.  The
resulting files `models.pkl`, `history.parquet`, `models_metrics.csv` and
`models_forecast.csv` will be written to the specified `artifacts` directory.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Ensure the parent directory is on sys.path so the sales_forecast package
# can be imported when this script is executed directly.  Without this,
# running the script with ``python scripts/train.py`` would result in a
# ``ModuleNotFoundError``.  When packaged and installed via pip, this is
# unnecessary because the package is on the Python path by default.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sales_forecast.data import load_data, winsorize_per_region
from sales_forecast.model import ProphetForecaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Prophet models per region.")
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to the input CSV containing at least ds, y and region columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="model_output",
        help="Directory to save the trained models and artefacts.",
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=4,
        help="Number of future periods (weeks) to forecast.",
    )
    parser.add_argument(
        "--country-holidays",
        default=None,
        help=(
            "Country name for Prophet's built-in holidays (e.g. 'Australia', 'UnitedStates')."
        ),
    )
    parser.add_argument(
        "--no-holidays",
        action="store_true",
        help="Disable adding country holidays even if --country-holidays is given.",
    )
    parser.add_argument(
        "--winsorize",
        action="store_true",
        help="Apply per-region winsorization to the target variable to reduce the impact of outliers.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Compute evaluation metrics by holding out the last horizon periods for each region when possible.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Read and clean data
    df = load_data(args.input_csv)
    # Drop rows with missing target and ensure y is numeric
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(float)
    # Optionally winsorize
    if args.winsorize:
        df = winsorize_per_region(df, column="y")
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # Initialise forecaster
    forecaster = ProphetForecaster(
        prediction_length=args.prediction_length,
        country_holidays=args.country_holidays,
        use_holidays=not args.no_holidays,
    )
    # Fit models
    metrics_df, forecasts_df = forecaster.fit(df, evaluate=args.evaluate)
    # Save artefacts
    model_file = output_path / "models.pkl"
    history_file = output_path / "history.parquet"
    forecaster.save(str(model_file), str(history_file), metrics_df, forecasts_df)
    print(f"✅ Models saved to {model_file}")
    print(f"📝 Training history saved to {history_file}")
    print(f"📊 Metrics written to {model_file.with_suffix('_metrics.csv')}")
    print(f"📈 Forecasts written to {model_file.with_suffix('_forecast.csv')}")


if __name__ == "__main__":
    main()
