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

def calculate_guideline_risk(data: PatientData):
    """Calculate risk stratification based on Guideline 2.5 (Table 2.5)."""
    bp = data.trestbps
    risk_factors = [
        data.age > 55,
        data.chol > 200,
        data.fbs == 1,
        data.ca > 0,
        data.thal <= 2 # 1 or 2 is risk in the updated scale
    ]
    rf_count = sum(risk_factors)
    
    # Blood pressure levels
    bp_level = 0
    if bp >= 180: bp_level = 4
    elif bp >= 160: bp_level = 3
    elif bp >= 140: bp_level = 2
    elif bp >= 130: bp_level = 1
    
    # Simplified Guideline Matrix (0: Low, 1: Medium, 2: High, 3: Very High)
    matrix = [
        [0, 0, 0, 1, 2], # 0 risk factors
        [0, 0, 1, 1, 3], # 1-2 risk factors
        [1, 2, 2, 2, 3], # >= 3 risk factors or diabetes/organ damage
        [3, 3, 3, 3, 3]  # Established disease (handled by ca/thal logic)
    ]
    
    row = 0
    if data.ca > 0 or data.thal == 1: row = 3
    elif rf_count >= 3 or data.fbs == 1: row = 2
    elif rf_count >= 1: row = 1
    
    risk_level = matrix[row][bp_level]
    labels = ["Nguy cơ thấp", "Nguy cơ trung bình", "Nguy cơ cao", "Nguy cơ rất cao"]
    
    return {
        "level": risk_level,
        "label": labels[risk_level],
        "factors_count": rf_count,
        "bp_level": bp_level
    }

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
    
    # [NEW] ADD GUIDELINE RISK AS A FEATURE
    guideline_info = calculate_guideline_risk(data)
    df_input['guideline_risk'] = guideline_info['level']
    
    try:
        # Scale the features (now 14 features)
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
        
        # Calculate Guideline 2.5 Risk
        guideline_info = calculate_guideline_risk(data)
        
        def map_risk_tier(prob_1, guideline_level):
            # Class 0 is Risk, Class 1 is Healthy. risk probability = 1 - prob_1
            prob_risk = 1 - prob_1
            
            # 1. VERY HIGH RISK: Prob > 85% or Guideline Very High
            if prob_risk > 0.85 or guideline_level >= 3: return "Nguy cơ rất cao"
            
            # 2. HIGH RISK: Prob > 65% or Guideline High
            if prob_risk > 0.65 or guideline_level >= 2: return "Nguy cơ cao"
            
            # 3. MEDIUM RISK: Prob > 40% or Guideline Medium
            # If guideline says Low (0), we need a stronger AI signal (>45%) to call it Medium
            if guideline_level >= 1 or prob_risk > 0.45: return "Nguy cơ trung bình"
            
            # 4. LOW RISK / NORMAL
            return "Bình thường (Thấp)"

        return {
            "input_data": data.model_dump(),
            "guideline_2_5": guideline_info,
            "logistic_regression": {
                "prediction": int(lr_pred),
                "probability": float(lr_prob) if lr_prob is not None else None,
                "status": map_risk_tier(lr_prob, guideline_info['level'])
            },
            "svm": {
                "prediction": int(svm_pred),
                "probability": float(svm_prob) if svm_prob is not None else None,
                "status": map_risk_tier(svm_prob, guideline_info['level'])
            },
            "knn": {
                "prediction": int(knn_pred),
                "probability": float(knn_prob) if knn_prob is not None else None,
                "status": map_risk_tier(knn_prob, guideline_info['level'])
            },
            "random_forest": {
                "prediction": int(rf_pred),
                "probability": float(rf_prob) if rf_prob is not None else None,
                "status": map_risk_tier(rf_prob, guideline_info['level'])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
