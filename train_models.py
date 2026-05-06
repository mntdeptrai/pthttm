import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

def main():
    print("Loading data from Google Sheets...")
    sheet_id = '1ZfnNy48RwtK0D_i_B3CmoXUKcpIBt8rBfdBNqzypEko'
    gid = '695684010'
    data_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}'
    
    try:
        df = pd.read_csv(data_url)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Preprocessing data...")
    # Fill missing values with mean
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    X = df.drop('target', axis=1)
    y = df['target']

    # Scale the features
    print("Scaling features...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train models
    print("Training models...")
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_scaled, y)
    print("Logistic Regression trained.")

    # KNN (using n_neighbors=1 as in original notebook)
    knn_model = KNeighborsClassifier(n_neighbors=1)
    knn_model.fit(X_scaled, y)
    print("KNN trained.")

    # Create model directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save scaler and models
    print("Saving models to disk...")
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(lr_model, 'models/logistic_model.joblib')
    joblib.dump(knn_model, 'models/knn_model.joblib')
    print("Done! Models are saved in the 'models/' directory.")

if __name__ == '__main__':
    main()
