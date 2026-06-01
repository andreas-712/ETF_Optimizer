# Internal Backend Developer Guide & Data Contracts

The backend directory contains the backend API routing, database schema definitions, and machine learning / mathematical engines for our custom stock portfolio optimizer.

## 1. Directory Tree & Ownership Boundary

To distribute workloads, the codebase separates web app infrastructure from our mathemtical components:

**Andreas's Workspace (`/resources`, `/schemas`, `/models`, `/ml_engine`):** Owns API endpoints, HTTP request validation, SQLAlchemy database design, cloud deployment orchestration, Gemini JSON feature mining, and predictive time-series models.
* **Theo's Workspace (`/math_engine`):** Owns deterministic mathematical code. **Scripts written here must remain 100% pure and decoupled from API routes, network requests, or database connection frameworks.**

```text
backend/
├── app.py                 # Flask app & extension configuration
├── db.py                  # Global SQLAlchemy instance
├── config.py              # Environment configs: .flaskenv and cloud secrets
├── requirements.txt       # Engine and app library dependencies
│
├── resources/             # Flask REST endpoints
│   └── optimize.py        # Core route that triggers the execution pipeline
├── schemas/               # Marshmallow JSON payload bouncers and field validation
├── models/                # Database tables structured as declarative Python classes
│
├── ml_engine/             # Andreas: Predictive AI Engine
│   ├── gemini_pipeline.py # News text ingestion & structured feature mapping
│   └── predictor.py       # ML forward estimation models (XGBoost/Scikit-learn)
│
└── math_engine/           # Theo: Deterministic Math Engine
    ├── kalman_filter.py   # Signal pre-processor and time-series trend smoother
    └── optimizer.py       # Portfolio allocation cost function