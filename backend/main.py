from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import json

app = FastAPI(title="Cardiovascular Disease Prediction API")

# Setup CORS to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and scaler at startup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

try:
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    lr_model = joblib.load(os.path.join(MODELS_DIR, 'logistic_model.joblib'))
    svm_model = joblib.load(os.path.join(MODELS_DIR, 'svm_model.joblib'))
    knn_model = joblib.load(os.path.join(MODELS_DIR, 'knn_model.joblib'))
    rf_model = joblib.load(os.path.join(MODELS_DIR, 'rf_model.joblib'))
    
    # Load training results if available
    results_path = os.path.join(MODELS_DIR, 'training_results.json')
    training_results = None
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            training_results = json.load(f)
    
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    scaler, lr_model, svm_model, knn_model, rf_model = None, None, None, None, None
    training_results = None

class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Hypertension Prediction API"}

@app.get("/model-info")
def model_info():
    """Return training results and model performance metrics."""
    if training_results is None:
        raise HTTPException(status_code=500, detail="Training results not available.")
    return training_results

@app.post("/predict")
def predict_hypertension(data: PatientData):
    if not scaler or not lr_model or not svm_model or not knn_model or not rf_model:
        raise HTTPException(status_code=500, detail="Models are not loaded on the server.")
    
    # Convert input data to DataFrame for scaler
    df_input = pd.DataFrame([data.model_dump()])
    
    try:
        # Scale the features
        X_scaled = scaler.transform(df_input)
        
        # Predict with Logistic Regression
        lr_pred = lr_model.predict(X_scaled)[0]
        lr_prob = lr_model.predict_proba(X_scaled)[0][1] if hasattr(lr_model, "predict_proba") else None
        
        # Predict with SVM
        svm_pred = svm_model.predict(X_scaled)[0]
        svm_prob = svm_model.predict_proba(X_scaled)[0][1] if hasattr(svm_model, "predict_proba") else None
        
        # Predict with KNN
        knn_pred = knn_model.predict(X_scaled)[0]
        knn_prob = knn_model.predict_proba(X_scaled)[0][1] if hasattr(knn_model, "predict_proba") else None
        
        # Predict with Random Forest
        rf_pred = rf_model.predict(X_scaled)[0]
        rf_prob = rf_model.predict_proba(X_scaled)[0][1] if hasattr(rf_model, "predict_proba") else None
        
        return {
            "input_data": data.model_dump(),
            "logistic_regression": {
                "prediction": int(lr_pred),
                "probability": float(lr_prob) if lr_prob is not None else None,
                "status": "Bình thường (Healthy)" if lr_pred == 1 else "Nguy cơ cao (High Risk)"
            },
            "svm": {
                "prediction": int(svm_pred),
                "probability": float(svm_prob) if svm_prob is not None else None,
                "status": "Bình thường (Healthy)" if svm_pred == 1 else "Nguy cơ cao (High Risk)"
            },
            "knn": {
                "prediction": int(knn_pred),
                "probability": float(knn_prob) if knn_prob is not None else None,
                "status": "Bình thường (Healthy)" if knn_pred == 1 else "Nguy cơ cao (High Risk)"
            },
            "random_forest": {
                "prediction": int(rf_pred),
                "probability": float(rf_prob) if rf_prob is not None else None,
                "status": "Bình thường (Healthy)" if rf_pred == 1 else "Nguy cơ cao (High Risk)"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
