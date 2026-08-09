import os
from pathlib import Path

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model.pkl'

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}.")
except FileNotFoundError:
    print(f"Warning: {MODEL_PATH.name} not found. Please run model.py first to train the model.")
    model = None

@app.route('/')
def home():
    """Renders the home page with the prediction form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests from the frontend."""
    if model is None:
        return jsonify({'error': 'Model not trained. Please train the model first.'}), 500
        
    try:
        # Get data from JSON payload
        data = request.json
        area = float(data.get('area', 0))
        bedrooms = int(data.get('bedrooms', 0))
        age = float(data.get('age', 0))
        
        # Prepare feature vector for prediction
        # The model was trained on ['Area', 'Bedrooms', 'Age']
        features = np.array([[area, bedrooms, age]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Return formatted result
        return jsonify({
            'success': True,
            'prediction': f"${prediction:,.2f}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
