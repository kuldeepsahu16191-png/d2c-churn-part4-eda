import pytest
from fastapi.testclient import TestClient

# Initialize the test client to simulate API requests
client = TestClient(app)

def test_health_endpoint():
    """Test 1: Validates that the service is running and healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_single_valid():
    """Test 2: Validates successful scoring for a standard valid payload[cite: 1]."""
    # Sample data matching your Pydantic schema[cite: 1]
    payload = {
        "customer_id": "CUST_999",
        "recency": 10,
        "frequency": 5,
        "monetary_value": 200.0,
        "support_complaints": 1,
        "return_rate": 0.1,
        "app_web_activity_score": 30
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "CUST_999"
    assert "churn_probability" in data
    assert "risk_level" in data

def test_predict_invalid_input():
    """Test 3: Validates that Pydantic properly blocks and rejects bad input[cite: 1]."""
    # Sending a negative frequency should trigger a 422 error[cite: 1]
    invalid_payload = {
        "customer_id": "CUST_ERR",
        "recency": 5,
        "frequency": -5,
        "monetary_value": 50.00,
        "support_complaints": 0,
        "return_rate": 0.0,
        "app_web_activity_score": 20
    }
    response = client.post("/predict", json=invalid_payload)
    # 422 is the standard FastAPI response for validation failures[cite: 1]
    assert response.status_code == 422
