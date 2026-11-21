# Dataset directory

The default training pipeline generates a deterministic synthetic dataset of 2,000 rows. It is an educational example, not real property-market evidence.

To use a documented real dataset, add housing.csv with columns Area, Bedrooms, Age, Region, Price. Region must match one of the six options in the current UI. Alternatively run python model.py --data path/to/dataset.csv.

Record the dataset URL, licence, collection date, geographic coverage, and preprocessing decisions in your report. The repository does not currently include a verified licensed real property dataset.