# Dataset and model methodology

## Data source

The fallback dataset is synthetic and generated deterministically (seed 42). It is not a record of real home sales or listings. For stronger academic evidence, supply a licensed real dataset and cite its provenance, access date, geography, and limitations.

## CSV schema

Required columns: Area, Bedrooms, Age, Region, Price. Area is in square feet; Age is years; Price is a numeric target. Region must be one of the six app region labels.

## Preprocessing and validation

The pipeline converts numeric fields, removes incomplete rows, unsupported regions, invalid positive/range values, and exact duplicates. It uses an 80/20 train/test split with random_state=42. Three regressors are compared: Linear Regression, Random Forest, and HistGradientBoosting. Selection is based on lowest test RMSE. MAE, RMSE, and R-squared are reported.

## Generated artifacts

Running python model.py creates model.pkl, reports/model_metrics.json, and diagnostic plots under reports/plots. Do not quote synthetic-data scores as real-world accuracy. A single random split is a teaching baseline; stronger research should add cross-validation and temporal/geographic validation.
