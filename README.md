# House Price Predictor Web App

A beautiful, full-stack machine learning web application that predicts house prices based on Area, Bedrooms, and Age.

![House Price Predictor](assets/screenshot.png)

## Overview

This project demonstrates how to build a complete Machine Learning application from scratch and integrate it into a modern web interface. It uses **Multiple Linear Regression** on the backend and a premium **Glassmorphism** design on the frontend.

### Tech Stack
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism UI), JavaScript (Fetch API)
- **Backend**: Python, Flask, RESTful API
- **Machine Learning**: Scikit-Learn (Linear Regression), Pandas, Numpy, Joblib

## Features

- **Synthetic Data Generation**: Generates 500 records of synthetic house data for training.
- **Model Training Pipeline**: A standalone script (`model.py`) to train the linear regression model and serialize it to disk.
- **REST API**: A Flask application that exposes a `/predict` endpoint to accept input features and return a price estimate.
- **Premium UI**: A sleek, responsive, glassmorphic frontend with smooth micro-animations and gradients.

## Quick Start

### 1. Install Dependencies
Make sure you have Python installed. Then, create a virtual environment and install the required packages:

```bash
python -m venv venv

# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

> **Note**: If you encounter errors installing `scikit-learn` on Windows, you may need to install the **Microsoft Visual C++ 14.0 Build Tools**.

### 2. Train the Model
Run the model script to generate data, train the Linear Regression model, and create the `model.pkl` file:

```bash
python model.py
```

### 3. Run the Web Server
Start the Flask backend application:

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000/`.

## Project Structure

```
├── app.py                  # Flask backend server
├── model.py                # Model training script
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Frontend UI
├── static/
│   └── style.css           # Glassmorphism styling
└── assets/                 # Images for documentation
```
