"""
This file contains functions for prediction feature selection:
    - Defining the shared model feature columns
    - Selecting fully processed inference features for return and volatility models
"""

import pandas as pd


# Keep feature column names standardized across modules (source of truth)
INDUSTRY_FEATURE_COLUMNS = [
    "energy",
    "financial",
    "technology",
]
INDUSTRIES = {"energy", "financial", "technology"}

FEATURE_COLUMNS = [
    "price_trend_deviation",
    "rolling_volatility",
    "gemini_sentiment_score",
    *INDUSTRY_FEATURE_COLUMNS,
]


def add_industry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model-ready industry feature columns."""
    result = df.copy()
    for column in INDUSTRY_FEATURE_COLUMNS:
        if column not in result.columns:
            result[column] = 0

    if "industry" in result.columns:
        normalized_industries = result["industry"].astype(str).str.lower()
        for industry in INDUSTRIES:
            # Finds and sets all matching values to 1 for each industry
            result.loc[normalized_industries == industry, industry] = 1

    return result


def select_inference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return processed feature columns for the ML models"""
    result = add_industry_features(df)
    return result[FEATURE_COLUMNS]
