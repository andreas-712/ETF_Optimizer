# Internal Backend Developer Guide & Data Contracts

The backend directory contains the backend API routing, database schema definitions, and machine-learning and mathematical engines for our custom stock portfolio optimizer.

## 1. Directory Tree & Ownership Boundary

To distribute workloads, the codebase separates web app infrastructure from our mathematical components:

**Andreas's Workspace (`/resources`, `/schemas`, `/models`, `/ml_engine`):** Owns API endpoints, HTTP request validation, SQLAlchemy database design, cloud deployment orchestration, Gemini model inference orchestration, and end-to-end async predictive time-series model inference.
**Theo's Workspace (`/math_engine`):** Owns deterministic mathematical code.

**Both workspaces remain decoupled from API endpoint logic for the Flask app**

```text
backend/
├── app.py                         # Flask application entry point
├── db.py                          # Database integration module
├── config.py                      # Backend application configuration
├── requirements.txt               # Engine and application dependencies
│
├── resources/                     # Flask REST endpoint modules
├── schemas/                       # Request and response validation schemas
├── models/                        # Database model modules
│
├── ml_engine/                     # Andreas: predictive ML engine
│   ├── market_data_collection.py  # Stock queries and live/historical data collection (numerical and unstructured text)
│   ├── gemini.py                  # Gemini client, configs, and structured scoring prompts
│   ├── train.py                   # Feature frame construction, model training, storing
│   ├── predictor.py               # Standardized feature selection for model inference
│   ├── batch_collection.py        # End-to-end historical data: numerical, unstructured, and batch inference input + extraction
│   ├── live_inference.py          # End-to-end live collection of candidate tickers and running numerical ML inference
│   ├── model_orchestrator.py      # Timeline-specific training, loading, and inference lifecycle
│   └── sandbox_ml_models.py       # Model training, backtesting and live-testing
│
└── math_engine/                   # Theo: deterministic mathematical engine
    └── Kalman_Filter.py           # Per-ticker Kalman price and velocity smoothing
```
