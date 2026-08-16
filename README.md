# 🏠 Home Prediction App

A polished end-to-end machine learning web application that estimates house prices from property characteristics using **Python, Flask, Pandas, NumPy, Scikit-learn, and Joblib**.

## ✨ What’s New

- Modern responsive glassmorphism UI
- Clear three-step prediction workflow
- Client-side and server-side input validation
- Loading state with prediction feedback
- Reset button for quick new predictions
- Accessible live result area and descriptive labels
- `/health` endpoint for deployment monitoring
- Safer API error handling without exposing internal exceptions
- Responsive layout for desktop, tablet, and mobile

## 📸 Preview

![Home Prediction App preview](assets/screenshot.svg)

## 🎯 Overview

The application demonstrates a complete ML workflow:

1. Generate a synthetic housing dataset with 2,000 samples.
2. Add region-aware market adjustments.
3. Train a Linear Regression model.
4. Serialize the trained model with Joblib.
5. Serve predictions through Flask.
6. Collect user inputs through a responsive web interface.
7. Return a formatted estimated property value through a JSON API.

> **Important:** This is an educational machine learning project based on synthetic data. Predictions are estimates and should not be treated as professional property valuations.

## 🚀 Live Demo

Hosted on Render: https://home-prediction-app.onrender.com/

## 🧰 Tech Stack

- **Frontend:** HTML5, CSS3, JavaScript
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn, Linear Regression
- **Data:** Pandas, NumPy
- **Model Persistence:** Joblib
- **Deployment:** Render

## 🔥 Features

### Prediction
- Area: 700–6,000 sq ft
- Bedrooms: 1–7
- Property age: 0–69 years
- Market region: North America, Europe, Asia, Latin America, Africa, Oceania

### User Experience
- Responsive modern interface
- Real-time validation messages
- Loading animation while predicting
- Form reset control
- Success and error result states
- Keyboard-friendly form controls

### API

**POST `/predict`**

Accepts JSON containing:

```json
{
  "area": 1800,
  "bedrooms": 3,
  "age": 8,
  "region": "North America"
}
```

Returns a formatted prediction such as:

```json
{
  "success": true,
  "prediction": "$271,234",
  "region": "North America"
}
```

**GET `/health`**

Returns model availability and application status for deployment monitoring.

## 🧪 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Shivamxxpathak/Home-Prediction-app.git
cd Home-Prediction-app
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python model.py
```

### 5. Start Flask

```bash
python app.py
```

Open `http://127.0.0.1:5000/`.

## 📁 Repository Structure

```text
Home-Prediction-app/
├── app.py                  # Flask application and prediction API
├── model.py                # Dataset generation and model training
├── model.pkl               # Serialized trained model, when generated
├── requirements.txt        # Python dependencies
├── Procfile                # Render process configuration
├── render.yaml             # Render deployment configuration
├── templates/
│   └── index.html          # Prediction interface
├── static/
│   └── style.css           # Responsive application styling
├── assets/
│   └── screenshot.svg      # Project preview
└── README.md
```

## 📌 Model Notes

The training pipeline creates synthetic data with relationships between area, bedrooms, age, and regional market multipliers. A Linear Regression model is trained on those generated relationships.

Because the training data is synthetic, model performance should not be interpreted as evidence of real-world housing-market accuracy.

## ☁️ Deploying to Render

The repository includes the Render deployment configuration. The application reads the `PORT` environment variable and binds to `0.0.0.0`, making it compatible with Render's web service runtime.

## 👨‍💻 Author

**Shivam Pathak**

- GitHub: https://github.com/Shivamxxpathak
- Portfolio: https://shivamxxpathak.github.io/
