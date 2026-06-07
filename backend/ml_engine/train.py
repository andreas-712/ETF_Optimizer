# Exposes functions for training ML model on existing data

import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import math

from math_engine.Kalman_Filter import Kalman_Filter

SAVED_MODEL_DIR = Path(__file__).resolve().parent / "saved_models"


def save_model(model, filename: str) -> Path:
    SAVED_MODEL_DIR.mkdir(parents = True, exist_ok = True)
    output_path = SAVED_MODEL_DIR / filename
    joblib.dump(model, output_path)
    return output_path


"""
Builds the Gemini inference feature frame for overall market sentiment impact
Sentiment score = relevance * polarity * urgency
"""
def build_gemini_feature_frame(gemini_df: pd.DataFrame) -> pd.DataFrame:
    # 1. Checks for data integrity
    required_columns = {"ticker", "relevance", "polarity", "urgency"}
    missing_columns = required_columns - set(gemini_df.columns)
    if missing_columns:
        raise ValueError(f"Gemini data is missing columns: {sorted(missing_columns)}")

    result = gemini_df.copy()

    duplicate_tickers = result.loc[result["ticker"].duplicated(), "ticker"].unique()
    if len(duplicate_tickers) > 0:
        raise ValueError(f"Gemini data has duplicate tickers: {sorted(duplicate_tickers)}")

    allowed_scores = {
        "relevance": {0.5, 1.0, 1.5},
        "polarity": {-1, 1},
        "urgency": {1, 2, 3},
    }
    for column, allowed_values in allowed_scores.items():
        invalid_values = result.loc[~result[column].isin(allowed_values), column].unique()
        if len(invalid_values) > 0:
            raise ValueError(
                f"Gemini {column} has invalid values: {sorted(invalid_values)}"
            )

    # 2. Compute sentiment scores for tickers
    result["gemini_sentiment_score"] = (
        result["relevance"] * result["polarity"] * result["urgency"]
    )

    return result[["ticker", "gemini_sentiment_score"]]


def add_gemini_inputs(df: pd.DataFrame, gemini_df: pd.DataFrame) -> pd.DataFrame:
    gemini_features = build_gemini_feature_frame(gemini_df)
    return df.merge(gemini_features, on="ticker", how="left")


def build_training_frame(
    df: pd.DataFrame,
    horizon_days: int,
    gemini_data: pd.DataFrame,
    feature_columns: list[str],
    rolling_volatility_window: int = 10,
    kalman_q: float = 1e-5,
    kalman_r: float = 1e-2,
) -> pd.DataFrame:
    # Populate df with relevant data, process with Kalman filter, add Gemini inference
    result = df.sort_values(["ticker", "date"]).copy()
    result = Kalman_Filter(Q=kalman_q, R=kalman_r).smooth_dataframe(result)
    result = add_gemini_inputs(result, gemini_data)

    # Sort by ticker
    grouped = result.groupby("ticker", group_keys = False)
    result["daily_return"] = grouped["adjusted_close"].pct_change()

    result["price_trend_deviation"] = (
        result["adjusted_close"] - result["kalman_smoothed_price"]
    )
    result["rolling_volatility"] = grouped["daily_return"].transform(
        lambda values: values.rolling(rolling_volatility_window).std()
    )
    result["future_return_outcome"] = grouped["adjusted_close"].transform(
        lambda prices: prices.pct_change(horizon_days).shift(-horizon_days)
    )
    result["future_volatility_outcome"] = grouped["daily_return"].transform(
        lambda values: values.rolling(horizon_days).std().shift(-horizon_days)
    )

    return result.dropna(
        subset=feature_columns + ["future_return_outcome", "future_volatility_outcome"]
    )

"""
Trains a sequential Gradient Boosting model
Captures asset momentum/inflection signals based on 
Kalman and Gemini features.
"""
def train_return_predictor(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    timeline_days: int,
):

    # Extract input and output columns
    X = df[feature_columns]
    y = df[target_column]
    
    # Set model hyperparameters
    return_model = GradientBoostingRegressor(
        n_estimators = 50,     # Sequential tree learning steps
        learning_rate = 0.005,   # Step size down loss gradient
        max_depth = 3,          # Capture interactive feature variables
        subsample = 0.85,       # Minimize variance (hide some data from predictors)
        random_state = 10
    )

    # Build exponential decay (half life) weight array
    # a is lower for return model compared to volatility model
    a = math.log(2) / (timeline_days * 3)
    dates = pd.to_datetime(df["date"])
    age_days = (dates.max() - dates).dt.days
    weights = [math.exp(-a * age) for age in age_days]
    
    return_model.fit(X, y, sample_weight = weights)

    return return_model

"""
Trains a Random Forest Regression model
Predicts market volatility / risk for specified assets
based on Kalman and Gemini features.
"""
def train_volatility_predictor(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    timeline_days: int,
):

    X = df[feature_columns]
    y = df[target_column]

    volatility_model = RandomForestRegressor(
        n_estimators = 200,      # Parallel trees: can push higher
        max_depth = 6,           # Less prone to overfitting: push depth higher
        min_samples_split = 5,   # Lower = more specific rules (potential overfitting)
        n_jobs = -1,              # Spread calculations across available cores
        random_state = 10
    )

    # Build exponential decay (half life) weight array
    a = math.log(2) / timeline_days
    dates = pd.to_datetime(df["date"])
    age_days = (dates.max() - dates).dt.days
    weights = [math.exp(-a * age) for age in age_days]

    volatility_model.fit(X, y, sample_weight = weights)

    return volatility_model
