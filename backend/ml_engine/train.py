"""
Exposes functions for training ML model on existing data
File writes: saved_models/{filename}.pkl
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import math

from math_engine.Kalman_Filter import Kalman_Filter

SAVED_MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
RETURN_HALF_LIFE_MULTIPLIER = 2
MIN_RETURN_HALF_LIFE_DAYS = 30
MAX_RETURN_HALF_LIFE_DAYS = 360


def return_half_life_days(timeline_days: int) -> int:
    raw_half_life = timeline_days * RETURN_HALF_LIFE_MULTIPLIER
    return max(
        MIN_RETURN_HALF_LIFE_DAYS,
        min(raw_half_life, MAX_RETURN_HALF_LIFE_DAYS),
    )


def return_model_config(timeline_days: int) -> dict:
    if timeline_days <= 20:
        return {
            "n_estimators": 120,
            "learning_rate": 0.10,
        }

    if timeline_days >= 360:
        return {
            "n_estimators": 80,
            "learning_rate": 0.08,
        }

    return {
        "n_estimators": 100,
        "learning_rate": 0.09,
    }


def save_model(model, filename: str) -> Path:
    SAVED_MODEL_DIR.mkdir(parents = True, exist_ok = True)
    output_path = SAVED_MODEL_DIR / filename
    joblib.dump(model, output_path)
    return output_path


def build_gemini_feature_frame(gemini_data: list[dict]) -> pd.DataFrame:
    """
    Builds the Gemini inference feature frame for overall market sentiment impact
    Sentiment score = relevance * polarity * urgency
    """
    # 1. Checks for data integrity
    result = pd.DataFrame(gemini_data)
    required_columns = {"ticker", "relevance", "polarity", "urgency"}
    missing_columns = required_columns - set(result.columns)
    if missing_columns:
        raise ValueError(f"Gemini data is missing columns: {sorted(missing_columns)}")

    duplicate_tickers = result.loc[result["ticker"].duplicated(), "ticker"].unique()
    if len(duplicate_tickers) > 0:
        raise ValueError(f"Gemini data has duplicate tickers: {sorted(duplicate_tickers)}")

    allowed_scores = {
        "relevance": {i for i in range(0, 11)},
        "polarity": {-1, 1},
        "urgency": {i for i in range(0, 11)},
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


def add_gemini_inputs(df: pd.DataFrame, gemini_data) -> pd.DataFrame:
    gemini_features = build_gemini_feature_frame(gemini_data)
    return df.merge(gemini_features, on="ticker", how="left")


def build_model_feature_frame(
    df: pd.DataFrame,
    gemini_data,
    rolling_volatility_window: int = 10,
    kalman_q: float = 1e-5,
    kalman_r: float = 1e-2,
) -> pd.DataFrame:
    result = df.sort_values(["ticker", "date"]).copy()
    result = Kalman_Filter(Q=kalman_q, R=kalman_r).smooth_dataframe(result)
    result = add_gemini_inputs(result, gemini_data)

    grouped = result.groupby("ticker", group_keys = False)
    result["daily_return"] = grouped["adjusted_close"].pct_change()
    result["price_trend_deviation"] = (
        result["adjusted_close"] - result["kalman_smoothed_price"]
    )
    result["rolling_volatility"] = grouped["daily_return"].transform(
        lambda values: values.rolling(rolling_volatility_window).std()
    )

    return result


def build_training_frame(
    df: pd.DataFrame,
    horizon_days: int,
    gemini_data: list[dict],
    feature_columns: list[str],
    rolling_volatility_window: int = 10,
    kalman_q: float = 1e-5,
    kalman_r: float = 1e-2,
) -> pd.DataFrame:
    result = build_model_feature_frame(
        df,
        gemini_data,
        rolling_volatility_window=rolling_volatility_window,
        kalman_q=kalman_q,
        kalman_r=kalman_r,
    )

    grouped = result.groupby("ticker", group_keys = False)
    result["future_return_outcome"] = grouped["adjusted_close"].transform(
        lambda prices: prices.pct_change(horizon_days).shift(-horizon_days)
    )
    result["future_volatility_outcome"] = grouped["daily_return"].transform(
        lambda values: values.rolling(horizon_days).std().shift(-horizon_days)
    )

    return result.dropna(
        subset=feature_columns + ["future_return_outcome", "future_volatility_outcome"]
    )


def train_return_predictor(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    timeline_days: int,
):
    """
    Trains a sequential Gradient Boosting model.
    Captures asset momentum/inflection signals based on 
    Kalman and Gemini features.
    """

    # Extract input and output columns
    X = df[feature_columns]
    y = df[target_column]
    config = return_model_config(timeline_days)
    
    # Set model hyperparameters
    return_model = GradientBoostingRegressor(
        n_estimators = config["n_estimators"],     # Sequential tree learning steps
        learning_rate = config["learning_rate"],   # Step size down loss gradient
        max_depth = 3,          # Capture interactive feature variables
        subsample = 0.85,       # Minimize variance (hide some data from predictors)
        random_state = 10
    )

    # Build exponential decay (half life) weight array
    half_life_days = return_half_life_days(timeline_days)
    a = math.log(2) / half_life_days
    dates = pd.to_datetime(df["date"])
    age_days = (dates.max() - dates).dt.days
    weights = [math.exp(-a * age) for age in age_days]
    
    return_model.fit(X, y, sample_weight = weights)

    return return_model


def train_volatility_predictor(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    timeline_days: int,
):
    """
    Trains a Random Forest Regression model.
    Predicts market volatility / risk for specified assets
    based on Kalman and Gemini features.
    """

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
