from pydantic import BaseModel, Field
from typing import List, Optional

class CustomerFeatures(BaseModel):
    customer_id: str = Field(..., example="CUST_10294")
    recency: int = Field(..., ge=0, description="Days since last order")
    frequency: int = Field(..., ge=0, description="Total number of orders placed")
    monetary_value: float = Field(..., ge=0.0, description="Total revenue from customer")
    support_complaints: int = Field(..., ge=0, description="Count of support tickets raised")
    return_rate: float = Field(..., ge=0.0, le=1.0, description="Ratio of returned orders")
    app_web_activity_score: int = Field(..., ge=0, description="Activity index on digital platforms")

    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_10294",
                "recency": 45,
                "frequency": 2,
                "monetary_value": 120.50,
                "support_complaints": 3,
                "return_rate": 0.40,
                "app_web_activity_score": 12
            }
        }

class SinglePredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    predicted_class: int
    risk_level: str
    risk_explanation: str

class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]

class BatchPredictionResponse(BaseModel):
    predictions: List[SinglePredictionResponse]
