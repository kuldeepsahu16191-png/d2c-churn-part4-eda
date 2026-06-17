import os

import pip
# import jobfile  # or pip install numpy 

# import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
 
# define
class CustomerFeatures(BaseModel):
    customer_id: str
    recency: int
    frequency: int
    monetary_value: float
    support_complaints: int
    return_rate: float
    app_web_activity_score: int

class SinglePredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    predicted_class: int
    risk_level: str
    risk_explanation: str

class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures]

class BatchPredictionResponse(BaseModel):
    predictions: list[SinglePredictionResponse]

app = FastAPI(
    title="D2C Customer Churn Intelligence API",
    description="Production scoring service to identify high-risk customers for targeted retention campaigns.",
    version="1.0.0"
)

# Global model placeholder
# Fix: Use os.getcwd() instead of __file__ for Colab compatibility
MODEL_PATH = os.path.join(os.getcwd(), "models", "model.pkl")
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            # Replace with joblib.load or your specific model loading logic
            import pickle
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load model artifact from {MODEL_PATH}: {str(e)}")
    else:
        # Graceful fallback or warning for initialization without breaking CI/CD pipelines
        print(f"Warning: Model artifact not found at {MODEL_PATH}. Running in mock mode.")

def generate_risk_explanation(features: CustomerFeatures, prob: float) -> tuple[str, str]:
    """Generates an interpretable risk explanation based on rules or SHAP baselines."""
    reasons = []
    if features.support_complaints >= 3:
        reasons.append("high volume of support tickets/complaints")
    if features.recency > 30:
        reasons.append("prolonged period of purchasing inactivity (high recency)")
    if features.return_rate > 0.30:
        reasons.append("elevated product return rate")
    if features.app_web_activity_score < 15:
        reasons.append("dropping app/web digital engagement scores")
        
    risk_level = "High" if prob >= 0.70 else ("Medium" if prob >= 0.40 else "Low")
    
    if not reasons:
        explanation = "Customer maintains stable behavioral patterns; baseline fluctuation."
    else:
        explanation = f"Risk driven by: {', '.join(reasons)}."
        
    return risk_level, explanation

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Checks the operational health of the application and model status."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=SinglePredictionResponse, status_code=status.HTTP_200_OK)
def predict_single(payload: CustomerFeatures):
    """Evaluates churn risk for a single customer payload."""
    # Handling mock inference if model file is missing during initialization
    if model is None:
        # Deterministic mock calculation for demonstration based on features
        mock_prob = min(0.95, (payload.recency * 0.01) + (payload.support_complaints * 0.15))
        prob = round(mock_prob, 4)
        pred_class = 1 if prob >= 0.50 else 0
    else:
        try:
            # Extracting expected feature vector matrix order matching Part 3 training
            feature_vector = [[
                payload.recency,
                payload.frequency,
                payload.monetary_value,
                payload.support_complaints,
                payload.return_rate,
                payload.app_web_activity_score
            ]]
            prob = float(model.predict_proba(feature_vector)[0][1])
            #  business-justified threshold calculated in Part 3 (e.g., 0.45)
            # Hardcoded to 0.50 here as a general standard fallback
            pred_class = 1 if prob >= 0.50 else 0
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference pipeline execution error: {str(e)}")

    risk_level, risk_explanation = generate_risk_explanation(payload, prob)

    return SinglePredictionResponse(
        customer_id=payload.customer_id,
        churn_probability=prob,
        predicted_class=pred_class,
        risk_level=risk_level,
        risk_explanation=risk_explanation
    )

@app.post("/batch_predict", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
def predict_batch(payload: BatchPredictionRequest):
    """Evaluates churn risk vectors across multiple batch customer records.""" 
    if not payload.customers:
        raise HTTPException(status_code=400, detail="The input customer batch array cannot be empty.")
        
    results = []
    for customer in payload.customers:
        # Call single prediction routing logic
        res = predict_single(customer)
        results.append(res)
        
    return BatchPredictionResponse(predictions=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)