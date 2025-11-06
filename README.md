# Sales Forecasting with Prophet

This project demonstrates how to build a simple weekly sales forecasting system using [Facebook’s Prophet](https://facebook.github.io/prophet/) library.  The code is designed to be run locally from the command line and can be easily extended or integrated into cloud environments such as AWS SageMaker.

## What it does

- **Train per‑region models:** The training script fits a separate Prophet model for each region found in your dataset.  Each model captures trends, seasonal patterns and holidays specific to that region.
- **Generate forecasts:** After training, the models can produce future forecasts for a specified horizon (in weeks).
- **Evaluate performance:** If the dataset is large enough, the last `horizon` weeks of each region are held out to compute common evaluation metrics (MAE, RMSE, R², MAPE and SMAPE).
- **Save artefacts:** Trained models, raw training history and summary metrics are saved to disk so they can be re‑used later for inference without retraining.

## Repository structure

```
sales_forecast/
│
├── sales_forecast/        # Python package with core code
│   ├── __init__.py
│   ├── data.py            # Data loading and preprocessing utilities
│   ├── metrics.py         # Metric functions used during evaluation
│   └── model.py           # ProphetForecaster class for training and prediction
│
├── scripts/
│   ├── train.py           # Command line interface to train models
│   └── predict.py         # Command line interface to run inference
│
├── requirements.txt       # Python package requirements
└── README.md              # This file
```

## Getting started

1. **Install dependencies.**  Prophet depends on [CmdStan](https://mc-stan.org/users/interfaces/cmdstan) to run its underlying probabilistic model.  On most systems Prophet will automatically download and build CmdStan the first time it is used.  You can install all Python dependencies with:

   ```bash
   python -m pip install -r requirements.txt
   ```

   Depending on your platform you may also need a C++ compiler available to compile Stan.  On Windows you can install [RTools](https://cran.r-project.org/bin/windows/Rtools/).  See the [Prophet installation guide](https://facebook.github.io/prophet/docs/installation.html) for detailed instructions.

2. **Prepare your data.**  The training script expects a CSV file with at least three columns:

   - `ds`: the date of each observation (any pandas‑parsable date format)
   - `y`: the numeric value to forecast (e.g. weekly sales)
   - `region`: a categorical field indicating which region or segment each row belongs to

   Additional regressors can be added as extra columns, although the default model in this example only uses automatically generated end‑of‑month indicators.

3. **Train the models.**  Run the `train.py` script, specifying your input CSV and the forecast horizon (in weeks).  For example:

   ```bash
   python scripts/train.py --input-csv data/train.csv --prediction-length 4 --country-holidays Australia --winsorize
   ```

   The script creates a `models.pkl` file containing all trained models, a `history.parquet` file with the original training data, and CSV files with metrics and the last `horizon` weeks of predictions.

4. **Run inference.**  Once models have been trained, you can generate future forecasts from the command line:

   ```bash
   python scripts/predict.py --model-file models.pkl --history-file history.parquet --region "RegionName" --horizon 6
   ```

   The script prints a table of future dates and the corresponding forecast intervals for the requested region.  You can redirect the output to a CSV by adding `--output-file forecast.csv`.

## Code design

The training and inference logic is encapsulated in the `ProphetForecaster` class defined in `sales_forecast/model.py`.  The class handles:

- Initialising Prophet models with optional country holidays and custom seasonalities.
- Adding boolean regressors indicating whether a date falls within the final week of the month (`es_cierre`) and its complement (`no_es_cierre`).  These variables allow the model to learn different behaviours around month‑end closures without hard‑coding the dates.
- Fitting per‑region models and computing evaluation metrics when enough historical data is available.
- Generating future frames for prediction and merging forecasts back with the associated region.

Separating this logic into modules makes the code easier to test and maintain.  Should you wish to experiment with different modelling techniques (e.g. RandomForest or gradient boosted trees), you can add additional classes under `sales_forecast/` and expose them via new scripts.

## Sensitive information

The original code this repository was based on included real S3 bucket names and proprietary prefixes.  For security reasons all such values have been removed or replaced with generic placeholders.  If you wish to persist your artefacts to cloud storage, modify the `train.py` script to upload the output files to your own bucket using `boto3`.  Never commit credentials, account numbers or other secrets to version control.

## License

This project is provided for demonstration purposes and does not include any warranty.  Feel free to use it as a starting point for your own forecasting pipelines.