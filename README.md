# d2c-churn-part4-eda


## Project Overview
This repository contains the production-ready FastAPI scoring service and reproducible machine learning workflow for the D2C Customer Churn Intelligence & Retention API. It exposes endpoints to serve the trained churn prediction model (built in Part 3) for both real-time individual predictions and high-throughput batch scoring.

The application implements strict Pydantic input validation schemas, automated endpoint tests, and incorporates an operational monitoring and responsible-use plan for the retention team.

---

## Repository Structure
```text
├── .vscode/
│   └── launch.json      # VS Code local debugging and service launch configuration
├── app/
│   ├── main.py          # FastAPI application core logic and routes 
│   └── schemas.py       # Pydantic data models for request/response validation 
├── models/
│   └── model.pkl        # Serialized production machine learning model artifact 
├── tests
│   └── test_api.py      # Automated API integration and endpoint test suite 
├── README.md            # Execution instructions and API sample documentation 
├── monitoring_plan.md   # Post-deployment data drift & retraining strategy 
└── requirements.txt     # Locked production dependencies
