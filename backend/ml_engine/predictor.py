"""
Exposes helpers for ML model prediction data
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Keep feature column names standardized across modules (source of truth)
FEATURE_COLUMNS = [
    "price_trend_deviation",
    "rolling_volatility",
    "gemini_sentiment_score",
]


def build_inference_features(timeline: int, df: pd.DataFrame, kalman_filter = True) -> pd.DataFrame:
    """
    Processes feature columns for prediction-ready data
    Params: timeline in days, dataframe with features, Kalman Filter toggle
    Returns: Processed feature columns
    """
    live_df = df.copy()
    
    # 1. Kalman filter toggle
    if kalman_filter:
        live_df['price_trend_deviation'] = live_df['adjusted_close'] - live_df['kalman_smoothed_price']
    else:
        # Simple windowing based directly on the timeline parameter
        live_df['price_trend_deviation'] = live_df['adjusted_close'] - live_df['adjusted_close'].rolling(window = timeline, min_periods = 1).mean()
        
    # 2. Extract feature columns the model needs
    return live_df[FEATURE_COLUMNS]
