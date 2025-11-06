#!/usr/bin/env python3
"""Generate future forecasts for a single region using a trained model.

This script loads previously trained Prophet models and their training history
from disk, then produces future forecasts for a specified region.  The output
can be printed to stdout or written to a CSV file.

Example usage::

    python scripts/predict.py --model-file model_output/models.pkl \
        --history-file model_output/history.parquet --region "Region 1" \
        --horizon 8 --output-file forecast_region1.csv

If the `--output-file` argument is omitted the forecasts will be printed to
stdout as a formatted table.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to sys.path so that the sales_forecast package can be
# imported when executing this script directly.  When the package is
# installed or the scripts are run via ``python -m``, this is not needed.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sales_forecast.model import ProphetForecaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast future periods for a given region.")
    parser.add_argument(
        "--model-file",
        required=True,
        help="Path to the joblib file containing the trained models (models.pkl).",
    )
    parser.add_argument(
        "--history-file",
        required=True,
        help="Path to the Parquet file containing the training history (history.parquet).",
    )
    parser.add_argument(
        "--region",
        required=True,
        help="Region name to forecast.  Must match one of the regions used during training.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Number of future periods to forecast.  Defaults to the value used during training.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional path to write the forecast to a CSV file.  If omitted, the forecast is printed to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forecaster = ProphetForecaster()
    # Load models and history
    forecaster.load(args.model_file, args.history_file)
    # Generate prediction
    forecast = forecaster.predict_region(args.region, horizon=args.horizon)
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        forecast.to_csv(output_path, index=False)
        print(f"✅ Forecast written to {output_path}")
    else:
        # Print as a nice table
        print(forecast.to_string(index=False))


if __name__ == "__main__":
    main()
