import os
from pathlib import Path

from flask import Flask, render_template, request, jsonify
import joblib

from model import create_and_train_model, prepare_features, GLOBAL_REGIONS

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model.pkl'

app = Flask(__name__, template_folder='templates', static_folder='static')


def load_model():
    try:
        model_data = joblib.load(MODEL_PATH)

        if isinstance(model_data, dict) and 'model' in model_data and 'feature_columns' in model_data:
            print(f"Model loaded successfully from {MODEL_PATH}.")
            return model_data

        print("Saved model uses an older format. Retraining with the current global market model.")
        create_and_train_model()
        return joblib.load(MODEL_PATH)

    except FileNotFoundError:
        print(f"Warning: {MODEL_PATH.name} not found. Training a new model...")
        create_and_train_model()
        try:
            return joblib.load(MODEL_PATH)
        except FileNotFoundError:
            print("Model training did not create a model file.")
            return None


model_data = load_model()
model = model_data['model'] if model_data else None

@app.route('/')
def home():
    """Renders the home page with the prediction form."""
    return render_template('index.html', regions=GLOBAL_REGIONS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests from the frontend."""
    if model is None:
        return jsonify({'error': 'Model not trained. Please train the model first.'}), 500

    try:
        data = request.json
        area = float(data.get('area', 0))
        bedrooms = int(data.get('bedrooms', 0))
        age = float(data.get('age', 0))
        region = data.get('region', 'North America')

        features = prepare_features(area, bedrooms, age, region)
        prediction = model.predict(features)[0]

        return jsonify({
            'success': True,
            'prediction': f"${prediction:,.2f}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
