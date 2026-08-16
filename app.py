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

        print("Saved model uses an older format. Retraining with the current model.")
        create_and_train_model()
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"{MODEL_PATH.name} not found. Training a new model...")
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
    return render_template('index.html', regions=GLOBAL_REGIONS)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok' if model is not None else 'model_unavailable',
        'model_loaded': model is not None,
        'regions': GLOBAL_REGIONS,
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': 'Model is unavailable. Please try again later.'}), 503

    try:
        data = request.get_json(silent=True) or {}
        area = float(data.get('area'))
        bedrooms = int(data.get('bedrooms'))
        age = float(data.get('age'))
        region = data.get('region', 'North America')

        if not 700 <= area <= 6000:
            raise ValueError('Area must be between 700 and 6,000 sq ft.')
        if not 1 <= bedrooms <= 7:
            raise ValueError('Bedrooms must be between 1 and 7.')
        if not 0 <= age <= 69:
            raise ValueError('Age must be between 0 and 69 years.')
        if region not in GLOBAL_REGIONS:
            raise ValueError('Please select a valid market region.')

        features = prepare_features(area, bedrooms, age, region)
        prediction = float(model.predict(features)[0])
        prediction = max(0.0, prediction)

        return jsonify({
            'success': True,
            'prediction': f"${prediction:,.0f}",
            'region': region,
        })

    except (TypeError, ValueError) as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception:
        app.logger.exception('Prediction request failed')
        return jsonify({'success': False, 'error': 'Something went wrong while generating the prediction.'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
