"""
This file contains functions and classes for model orchestration:
    - Managing one return and volatility model per prediction horizon
    - Loading saved models and running inference through timeline instances
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Training helpers
from ml_engine.train import (
    train_return_predictor,
    train_volatility_predictor,
    save_model,
)
from ml_engine.predictor import select_inference_features, FEATURE_COLUMNS

SAVED_MODEL_DIR = Path(__file__).resolve().parent / "saved_models"


# Instantiate a model once per timeline and let the instance live statically for re-use
class TimelineModel:
    def __init__(self, timeline_days: int):
        self.timeline_days = timeline_days
        self.return_model = None
        self.volatility_model = None

    # Train and save models from rows for one prediction horizon.
    def train(self, training_df: pd.DataFrame) -> None:
        required_columns = {
            "date",
            "prediction_horizon_days",
            *FEATURE_COLUMNS,
            "future_return_outcome",
            "future_volatility_outcome",
        }
        missing_columns = required_columns - set(training_df.columns)
        if missing_columns:
            raise ValueError(
                f"Flattened training frame is missing columns: {sorted(missing_columns)}"
            )

        training_horizons = set(training_df["prediction_horizon_days"].unique())
        if training_horizons != {self.timeline_days}:
            raise ValueError(
                f"Training frame must contain only the model's prediction horizon ({self.timeline_days}); received {sorted(training_horizons)}"
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
    def _load(self):
        self.return_model = joblib.load(
            Path(SAVED_MODEL_DIR) / f"gbr_return_model_{self.timeline_days}d.pkl"
        )
        self.volatility_model = joblib.load(
            Path(SAVED_MODEL_DIR) / f"rfr_volatility_model_{self.timeline_days}d.pkl"
        )

    def return_inference(self, processed_df: pd.DataFrame) -> np.ndarray:
        if self.return_model == None:
            print(f"Return model for {self.timeline_days} day horizons not loaded yet\n")
            return np.array([])
        X = select_inference_features(processed_df)
        return self.return_model.predict(X)
    
    def volatility_inference(self, processed_df: pd.DataFrame) -> np.ndarray:
        if self.volatility_model == None:
            print(f"Volatility model for {self.timeline_days} day horizons not loaded yet\n")
            return np.array([])
        X = select_inference_features(processed_df)
        return self.volatility_model.predict(X)


MODELS = {
    3: TimelineModel(3),
    20: TimelineModel(20),
    90: TimelineModel(90)
}

def load_models() -> None:
    """Loads all available timeline models"""
    for model in MODELS.values():
        model._load()
