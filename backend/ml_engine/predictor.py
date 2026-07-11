"""Selects fully processed model features for prediction."""

import pandas as pd


# Keep feature column names standardized across modules (source of truth)
FEATURE_COLUMNS = [
    "price_trend_deviation",
    "rolling_volatility",
    "gemini_sentiment_score",
]


def select_inference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return processed feature columns for the ML models"""
    return df[FEATURE_COLUMNS]
