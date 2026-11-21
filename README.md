# Home Prediction App

A responsive machine-learning web application that estimates property prices from property characteristics. Built with Python, Flask, Pandas, NumPy, Scikit-learn, and Joblib.

## Overview

The application provides an end-to-end prediction workflow:

1. Generate or load a housing dataset.
2. Clean and validate the data.
3. Compare regression models using a held-out test set.
4. Select a model based on test RMSE.
5. Save the trained model and evaluation artifacts.
6. Serve predictions through a Flask web interface and JSON API.

**Data limitation:** The default dataset is synthetic. Predictions are illustrative estimates and must not be treated as professional property valuations or evidence of actual market accuracy.

## Live Demo

[Open the deployed app](https://home-prediction-app.onrender.com/)

## Preview

![Home Prediction App preview](assets/screenshot.svg)

## Features

- Responsive property prediction interface
- Input validation on client and server
- Prediction loading and result states
- Model comparison and evaluation metrics
- Actual-versus-predicted and residual diagnostic plots
- Model transparency endpoint
- Health-check endpoint
- CSV dataset support
- Automated tests with GitHub Actions

## Technology Stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python, Flask
- **Machine learning:** Scikit-learn (Linear Regression, Random Forest, HistGradientBoosting)
- **Data processing:** Pandas, NumPy
- **Model persistence:** Joblib
- **Deployment:** Render

## Prediction Inputs

- Area: 700–6,000 sq ft
- Bedrooms: 1–7
- Property age: 0–69 years
- Region: North America, Europe, Asia, Latin America, Africa, or Oceania

## API

### `POST /predict`

Example request:

```json
{
  "area": 1800,
  "bedrooms": 3,
  "age": 8,
  "region": "North America"
}
```

Example response:

```json
{
  "success": true,
  "prediction": "$271,234",
  "region": "North America"
}
```

### `GET /health`

Returns application and model availability.

### `GET /metrics`

Returns the selected model, evaluation metrics, dataset details, and training information.

## Model Training and Evaluation

The training pipeline compares Linear Regression, Random Forest, and HistGradientBoosting using an 80/20 train-test split with a fixed random seed. The model with the lowest test RMSE is selected. The pipeline reports MAE, RMSE, and R².

Run training:

```bash
python model.py
```

Generated artifacts include:

- `model.pkl` — trained model bundle
- `reports/model_metrics.json` — evaluation summary
- `reports/plots/actual_vs_predicted.png`
- `reports/plots/residual_plot.png`
- `reports/plots/feature_importance.png` (when supported)

For details about data processing and limitations, see [Dataset and Model Methodology](docs/DATA_AND_MODEL.md).

## Use a CSV Dataset

Run the pipeline with your own dataset:

```bash
python model.py --data data/your_housing_dataset.csv
```

Required columns: `Area`, `Bedrooms`, `Age`, `Region`, `Price`.

The dataset must have documented provenance and appropriate usage rights. The repository does not currently contain a verified licensed real-property dataset.

## Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/Shivamxxpathak/Home-Prediction-app.git
   cd Home-Prediction-app
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   Windows:

   ```bash
   venv\\Scripts\\activate
   ```

   macOS/Linux:

   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Train the model and start the server:

   ```bash
   python model.py
   python app.py
   ```

5. Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Tests

Run the test suite:

```bash
pytest -q
```

GitHub Actions runs the tests on repository updates.

## Deployment

The repository includes a Render configuration. The app uses the `PORT` environment variable and binds to `0.0.0.0`.

## Repository Owner

**Shivam Pathak**

- [GitHub](https://github.com/Shivamxxpathak)
- [Portfolio](https://shivamxxpathak.github.io/)
