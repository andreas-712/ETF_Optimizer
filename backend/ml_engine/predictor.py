"""
This file contains functions for prediction feature selection:
    - Defining the shared model feature columns
    - Selecting fully processed inference features for return and volatility models
"""

import pandas as pd


# Keep feature column names standardized across modules (source of truth)
SECTOR_FEATURE_COLUMNS = [
    "energy",
    "financial",
    "technology",
]
SECTORS = {"energy", "financial", "technology"}

FEATURE_COLUMNS = [
    "price_trend_deviation",
    "rolling_volatility",
    "gemini_sentiment_score",
    *SECTOR_FEATURE_COLUMNS,
]


def add_sector_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model-ready sector feature columns."""
    result = df.copy()
    for column in SECTOR_FEATURE_COLUMNS:
        if column not in result.columns:
            result[column] = 0

    if "sector" in result.columns:
        normalized_sectors = result["sector"].astype(str).str.lower()
        for sector in SECTORS:
            # Finds and sets all matching values to 1 for each sector
            result.loc[normalized_sectors == sector, sector] = 1

    return result


def select_inference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return processed feature columns for the ML models"""
    result = add_sector_features(df)
    return result[FEATURE_COLUMNS]
