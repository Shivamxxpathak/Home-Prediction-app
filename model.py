import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
import joblib

MODEL_PATH = Path(__file__).resolve().parent / 'model.pkl'

def create_and_train_model():
    print("Generating synthetic dataset...")
    # Generate random synthetic data for house prices
    np.random.seed(42)
    
    # 500 samples
    n_samples = 500
    
    # Features:
    # Area in sq ft: 800 to 4000
    area = np.random.randint(800, 4000, n_samples)
    
    # Bedrooms: 1 to 6
    bedrooms = np.random.randint(1, 7, n_samples)
    
    # Age in years: 0 to 50
    age = np.random.randint(0, 51, n_samples)
    
    # Base price
    base_price = 50000
    
    # Linear combination with some noise
    # Price = Base + Area*150 + Bedrooms*15000 - Age*1000 + Noise
    noise = np.random.normal(0, 15000, n_samples)
    
    price = base_price + (area * 150) + (bedrooms * 15000) - (age * 1000) + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'Area': area,
        'Bedrooms': bedrooms,
        'Age': age,
        'Price': price
    })
    
    print("Dataset generated with 500 samples.")
    
    # Features and Target
    X = df[['Area', 'Bedrooms', 'Age']]
    y = df['Price']
    
    print("Training Linear Regression model...")
    # Initialize and train model
    model = LinearRegression()
    model.fit(X, y)
    
    print(f"Model trained! R^2 Score: {model.score(X, y):.4f}")
    
    # Save the model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved successfully to {MODEL_PATH}")

if __name__ == '__main__':
    create_and_train_model()
