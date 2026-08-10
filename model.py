import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
import joblib

MODEL_PATH = Path(__file__).resolve().parent / 'model.pkl'

GLOBAL_REGIONS = [
    'North America',
    'Europe',
    'Asia',
    'Latin America',
    'Africa',
    'Oceania'
]

REGION_MULTIPLIERS = {
    'North America': 1.00,
    'Europe': 0.95,
    'Asia': 0.80,
    'Latin America': 0.60,
    'Africa': 0.45,
    'Oceania': 1.05,
}


def _feature_columns():
    columns = ['Area', 'Bedrooms', 'Age']
    columns.extend([f'Region_{region}' for region in GLOBAL_REGIONS])
    return columns


def prepare_features(area, bedrooms, age, region):
    if region not in GLOBAL_REGIONS:
        raise ValueError(f"Unknown region: {region}. Valid regions: {', '.join(GLOBAL_REGIONS)}")

    df = pd.DataFrame([
        {
            'Area': area,
            'Bedrooms': bedrooms,
            'Age': age,
            'Region': region,
        }
    ])
    features = pd.get_dummies(df, columns=['Region'])
    return features.reindex(columns=_feature_columns(), fill_value=0).values


def create_and_train_model():
    print("Generating global synthetic dataset...")
    np.random.seed(42)

    n_samples = 2000

    area = np.random.randint(700, 6000, n_samples)
    bedrooms = np.random.randint(1, 8, n_samples)
    age = np.random.randint(0, 70, n_samples)
    regions = np.random.choice(GLOBAL_REGIONS, n_samples)

    base_price = 30000
    price = base_price + (area * 130) + (bedrooms * 12000) - (age * 800)
    price = price * np.array([REGION_MULTIPLIERS[r] for r in regions])
    price += np.random.normal(0, 20000, n_samples)

    df = pd.DataFrame({
        'Area': area,
        'Bedrooms': bedrooms,
        'Age': age,
        'Region': regions,
        'Price': price,
    })

    print(f"Dataset generated with {n_samples} global samples.")

    X = pd.get_dummies(df[['Area', 'Bedrooms', 'Age', 'Region']], columns=['Region'])
    X = X.reindex(columns=_feature_columns(), fill_value=0)
    y = df['Price']

    print("Training Linear Regression model for global markets...")
    model = LinearRegression()
    model.fit(X, y)

    r2_score = model.score(X, y)
    print(f"Model trained! R^2 Score: {r2_score:.4f}")

    model_data = {
        'model': model,
        'feature_columns': _feature_columns(),
        'regions': GLOBAL_REGIONS,
    }

    joblib.dump(model_data, MODEL_PATH)
    print(f"Model saved successfully to {MODEL_PATH}")
    return model_data


if __name__ == '__main__':
    create_and_train_model()
