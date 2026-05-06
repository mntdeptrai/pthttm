from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Hypertension Prediction API")

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
    knn_model = joblib.load(os.path.join(MODELS_DIR, 'knn_model.joblib'))
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    scaler, lr_model, knn_model = None, None, None

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

@app.post("/predict")
def predict_hypertension(data: PatientData):
    if not scaler or not lr_model or not knn_model:
        raise HTTPException(status_code=500, detail="Models are not loaded on the server.")
    
    # Convert input data to DataFrame for scaler
    df_input = pd.DataFrame([data.dict()])
    
    try:
        # Scale the features
        X_scaled = scaler.transform(df_input)
        
        # Predict with Logistic Regression
        lr_pred = lr_model.predict(X_scaled)[0]
        lr_prob = lr_model.predict_proba(X_scaled)[0][1] if hasattr(lr_model, "predict_proba") else None
        
        # Predict with KNN
        knn_pred = knn_model.predict(X_scaled)[0]
        knn_prob = knn_model.predict_proba(X_scaled)[0][1] if hasattr(knn_model, "predict_proba") else None
        
        return {
            "logistic_regression": {
                "prediction": int(lr_pred),
                "probability": float(lr_prob) if lr_prob is not None else None,
                "status": "High Risk" if lr_pred == 1 else "Low Risk"
            },
            "knn": {
                "prediction": int(knn_pred),
                "probability": float(knn_prob) if knn_prob is not None else None,
                "status": "High Risk" if knn_pred == 1 else "Low Risk"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
