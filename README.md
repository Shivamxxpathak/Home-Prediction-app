# Home Prediction App

A polished full-stack machine learning project that predicts house prices from area, bedroom count, and property age using a linear regression model.

## Sample Screenshot

![Home Prediction App preview](assets/screenshot.svg)

This preview shows the polished prediction form and results card used by the app.

## Overview

This repository demonstrates a clean end-to-end workflow for building a small ML web app:
- train a regression model with synthetic data,
- serialize it with joblib,
- serve it through Flask,
- and present it with a modern glassmorphism-style interface.

## Tech Stack
- Frontend: HTML, CSS, JavaScript
- Backend: Python, Flask
- Machine Learning: scikit-learn, pandas, numpy, joblib

## Features
- Synthetic dataset generation for 500 training samples
- Model training pipeline in a dedicated script
- REST-style prediction endpoint for the UI
- Organized static assets and templates for a cleaner repository layout

## Quick Start

### 1. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python model.py
```

### 4. Run the app
```bash
python app.py
```

Then open http://127.0.0.1:5000/ in your browser.

## Repository Structure
```text
Home-Prediction-app/
├── app.py                  # Flask application entry point
├── model.py                # Model training script
├── requirements.txt        # Python dependencies
├── templates/              # HTML pages for the web UI
│   └── index.html
├── static/                 # CSS and frontend assets
│   └── style.css
├── assets/                 # Project screenshots and media
│   └── screenshot.svg
└── .gitignore              # Files to ignore in Git
```

## Notes
If you run into install issues with scikit-learn on Windows, you may need the Microsoft Visual C++ Build Tools installed on your machine.

## Deploying to Render
This repository includes deployment files for Render:
- Procfile
- render.yaml
- runtime.txt

When the app starts on a fresh host, it will automatically train the model if the model file is missing.
