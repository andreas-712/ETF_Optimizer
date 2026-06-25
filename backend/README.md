# Internal Backend Developer Guide & Data Contracts

The backend directory contains the backend API routing, database schema definitions, and machine-learning and mathematical engines for our custom stock portfolio optimizer.

## 1. Directory Tree & Ownership Boundary

To distribute workloads, the codebase separates web app infrastructure from our mathematical components:

**Andreas's Workspace (`/resources`, `/schemas`, `/models`, `/ml_engine`):** Owns API endpoints, HTTP request validation, SQLAlchemy database design, cloud deployment orchestration, Gemini JSON feature mining, and predictive time-series models.
**Theo's Workspace (`/math_engine`):** Owns deterministic mathematical code. Scripts written here must remain 100% pure and decoupled from API routes, network requests, or database connection frameworks.

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
│   ├── configs.py                 # Shared live/backtest execution state
│   ├── market_data_collection.py  # Price, FMP metric, and live/historical data collection
│   ├── gemini.py                  # Gemini client, prompts, and structured scoring prompts
│   ├── gemini_data_collection.py  # Historical Gemini inference collection and CSV generation
│   ├── train.py                   # Feature frame construction, model training, storing
│   ├── predictor.py               # Standardized feature selection for model inference
│   ├── model_orchestrator.py      # Timeline-specific training, loading, and inference lifecycle
│   └── sandbox_ml_models.py       # Model backtesting experiments and prediction reports
│
└── math_engine/                   # Theo: deterministic mathematical engine
    └── Kalman_Filter.py           # Per-ticker Kalman price and velocity smoothing
```
