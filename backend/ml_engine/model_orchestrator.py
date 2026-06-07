# Orchestrates the full ML lifecycle for a given timeline instance model

from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Training helpers
from ml_engine.train import (
    build_training_frame,
    train_return_predictor,
    train_volatility_predictor,
    save_model,
)

# Inference data helpers
from ml_engine.predictor import build_inference_features, FEATURE_COLUMNS

# Instantiate a model once per timeline and let the instance live statically for re-use
class TimelineModel:
    def __init__(self, timeline_days: int):
        self.timeline_days = timeline_days
        self.return_model = None
        self.volatility_model = None

    # Trains and saves models under /ml_engine/saved_models
    def train(self, market_df: pd.DataFrame, gemini_data: pd.DataFrame, kalman_r=1e-2, kalman_q=1e-5):
        training_df = build_training_frame(
            df = market_df,
            horizon_days = self.timeline_days,
            gemini_data = gemini_data,
            feature_columns = FEATURE_COLUMNS,
            kalman_r = kalman_r,
            kalman_q = kalman_q,
        )

        self.return_model = train_return_predictor(
            training_df,
            FEATURE_COLUMNS,
            "future_return_outcome",
            self.timeline_days,
        )

        self.volatility_model = train_volatility_predictor(
            training_df,
            FEATURE_COLUMNS,
            "future_volatility_outcome",
            self.timeline_days,
        )

        save_model(self.return_model, f"gbr_return_model_{self.timeline_days}d.pkl")
        save_model(self.volatility_model, f"rfr_volatility_model_{self.timeline_days}d.pkl")


    # Loads trained model from directory (only do this once per prediction timeline and save instance statically)
    def load(self, model_dir: str):
        self.return_model = joblib.load(
            Path(model_dir) / f"gbr_return_model_{self.timeline_days}d.pkl"
        )
        self.volatility_model = joblib.load(
            Path(model_dir) / f"rfr_volatility_model_{self.timeline_days}d.pkl"
        )

    def return_inference(self, df: pd.DataFrame, kalman_toggle = True) -> np.ndarray:
        X = build_inference_features(self.timeline_days, df, kalman_toggle)
        return self.return_model.predict(X)
    
    def volatility_inference(self, df: pd.DataFrame, kalman_toggle = True) -> np.ndarray:
        X = build_inference_features(self.timeline_days, df, kalman_toggle)
        return self.volatility_model.predict(X)
