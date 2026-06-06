import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from math_engine.Kalman_Filter import Kalman_Filter

SAVED_MODEL_DIR = Path(__file__).resolve().parent / "saved_models"


def save_model(model, filename: str) -> Path:
    SAVED_MODEL_DIR.mkdir(parents = True, exist_ok = True)
    output_path = SAVED_MODEL_DIR / filename
    joblib.dump(model, output_path)
    return output_path


def add_gemini_inputs(df: pd.DataFrame, ratings: dict) -> pd.DataFrame:
    result = df.copy()
    result["gemini_sentiment_score"] = result["ticker"].map(
        lambda ticker: ratings[ticker]["sentiment"]
    )
    result["gemini_risk_flag"] = result["ticker"].map(
        lambda ticker: ratings[ticker]["risk"]
    )
    return result


def build_training_frame(
    df: pd.DataFrame,
    horizon_days: int,
    gemini_ratings: dict,
    feature_columns: list[str],
    rolling_volatility_window: int = 10,
    kalman_q: float = 1e-5,
    kalman_r: float = 1e-2,
) -> pd.DataFrame:
    # Populate df with relevant data, process with Kalman filter, add Gemini inference
    result = df.sort_values(["ticker", "date"]).copy()
    result = Kalman_Filter(Q=kalman_q, R=kalman_r).smooth_dataframe(result)
    result = add_gemini_inputs(result, gemini_ratings)

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

'''
Trains a sequential Gradient Boosting model
Captures asset momentum/inflection signals based on 
Kalman and Gemini features.
'''
def train_return_predictor(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
):

    X = df[feature_columns]
    y = df[target_column]
    
    return_model = GradientBoostingRegressor(
        n_estimators = 100,     # Sequential tree learning steps
        learning_rate = 0.05,   # Step size down loss gradient
        max_depth = 4,          # Capture interactive feature variables
        subsample = 0.85,       # Minimize variance (hide some data from predictors)
        random_state = 10       # Reduce unecessary variables for tests 
    )

    return_model.fit(X, y)

    return return_model

'''
Trains a Random Forest Regression model
Predicts market volatility / risk for specified assets
based on Kalman and Gemini features.
'''
def train_risk_predictor(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
):

    X = df[feature_columns]
    y = df[target_column]

    risk_model = RandomForestRegressor(
        n_estimators = 200,      # Parallel trees: faster execution
        max_depth = 6,           # Less prone to overfitting: more depth
        min_samples_split = 5,   # Smooth isolated trend deviations
        n_jobs = 1,              # Spread calculations across all cores
        random_state = 10
    )

    risk_model.fit(X, y)

    return risk_model
