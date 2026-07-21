"""
This file contains functions for the model training flow:
    - Building return, volatility, and Gemini feature frames
    - Training and saving return and volatility predictors

File writes: saved_models/{filename}.pkl
"""

from typing import Any

import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import math

from ml_engine.gemini import GEMINI_RESPONSE_FIELDS
from ml_engine.predictor import INDUSTRY_FEATURE_COLUMNS, add_industry_features
from math_engine.Kalman_Filter import Kalman_Filter

SAVED_MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
RETURN_HALF_LIFE_MULTIPLIER = 2
MIN_RETURN_HALF_LIFE_DAYS = 5
MAX_RETURN_HALF_LIFE_DAYS = 180
ROLLING_VOLATILITY_WINDOW = 10
ROLLING_PRICE_WINDOW = 10


def return_half_life_days(timeline_days: int) -> int:
    """Return the time-series half-life weights used by the return model"""
    raw_half_life = timeline_days * RETURN_HALF_LIFE_MULTIPLIER
    return max(
        MIN_RETURN_HALF_LIFE_DAYS,
        min(raw_half_life, MAX_RETURN_HALF_LIFE_DAYS)
    )


def _return_model_config(timeline_days: int) -> dict[str, int | float]:
    """Returns Gradient Boosting hyperparameters based on prediction horizon."""
    if timeline_days == 5:
        return {
            "n_estimators": 130,
            "learning_rate": 0.13
        }

    if timeline_days == 20:
        return {
            "n_estimators": 110,
            "learning_rate": 0.11
        }


    if timeline_days == 90:
        return {
            "n_estimators": 80,
            "learning_rate": 0.08
        }
    
    # Default
    return {
        "n_estimators": 100,
        "learning_rate": 0.1
    }


def save_model(model: Any, filename: str) -> Path:
    """Serialize a fitted model under : data : SAVED_MODEL_DIR."""
    SAVED_MODEL_DIR.mkdir(parents = True, exist_ok = True)
    output_path = SAVED_MODEL_DIR / filename
    joblib.dump(model, output_path)
    return output_path


def _build_gemini_feature_frame(
    gemini_data: list[dict[str, Any]],
    horizon_days: int
) -> pd.DataFrame:
    """
    Build one validated feature row per ticker and date.
    The score is relevance * polarity * urgency and ranges from -100 to 100.
    The input must contain one row per ticker, date, and prediction horizon.
    Industry is optional metadata; when present, it is converted to one-hot columns.
    """
    # 1. Checks for data integrity
    result = pd.DataFrame(gemini_data)
    missing_columns = GEMINI_RESPONSE_FIELDS - set(result.columns)
    if missing_columns:
        raise ValueError(f"Gemini data is missing columns: {missing_columns}")

    result = result[
        result["prediction_horizon_days"] == horizon_days
    ].copy()
    result["date"] = pd.to_datetime(result["date"]).dt.date

    duplicate_rows = result.duplicated(subset = ["ticker", "date"])
    if duplicate_rows.any():
        raise ValueError("Gemini data has duplicate ticker/date rows")

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

    # 3. Map each ticker to its respective industry
    result = add_industry_features(result)

    feature_columns = [
        "ticker",
        "date",
        "gemini_sentiment_score",
        *INDUSTRY_FEATURE_COLUMNS,
    ]
    return result[feature_columns]


def _add_gemini_outputs(
    df: pd.DataFrame,
    gemini_data: list[dict[str, Any]],
    horizon_days: int
) -> pd.DataFrame:
    """Join daily Gemini sentiment scores onto market data"""
    gemini_features = _build_gemini_feature_frame(gemini_data, horizon_days)
    return df.merge(gemini_features, on=["ticker", "date"], how = "left")


def build_model_feature_frame(
    df: pd.DataFrame,
    gemini_outputs: list[dict[str, Any]],
    horizon_days: int,
    prediction_window_start: str = None, # Should be dt.now for live
    kalman_q: float = 1e-5,
    kalman_r: float = 1e-2,
) -> pd.DataFrame:
    """
    Create chronologically ordered model features for each ticker.
    Adds the rolling mean price deviation, daily percentage return, 
    price_trend_deviation, rolling_volatility and gemini_sentiment_score.
    df: unprocessed numerical df. gemini_outputs: unprocessed sentiment scores.
    """
    result = df.sort_values(["ticker", "date"]).copy()
    result = Kalman_Filter(Q=kalman_q, R=kalman_r).smooth_dataframe(result)

    grouped = result.groupby("ticker", group_keys = False)
    result["daily_return"] = grouped["adjusted_close"].pct_change()
    rolling_price = grouped["adjusted_close"].transform(
        lambda values: values.rolling(ROLLING_PRICE_WINDOW).mean()
    )
    result["price_trend_deviation"] = (
        (result["adjusted_close"] / rolling_price) - 1
    )
    result["rolling_volatility"] = grouped["daily_return"].transform(
        lambda values: values.rolling(ROLLING_VOLATILITY_WINDOW).std()
    )

    daily_ticker_frames = []
    columns_to_forward_fill = [
        "adjusted_close",
        "volume",
        "kalman_smoothed_price",
        "kalman_velocity",
        "price_trend_deviation",
        "rolling_volatility"
    ]

    for ticker, ticker_df in result.groupby("ticker"):
        ticker_df = ticker_df.copy()
        ticker_df["date"] = pd.to_datetime(ticker_df["date"])
        ticker_df = ticker_df.set_index("date")
        calendar_end = ticker_df.index.max()
        if prediction_window_start is not None:
            calendar_end = pd.to_datetime(prediction_window_start)
        calendar_dates = pd.date_range(
            ticker_df.index.min(),
            calendar_end,
            freq = "D"
        )
        ticker_df = ticker_df.reindex(calendar_dates)
        ticker_df["ticker"] = ticker
        ticker_df[columns_to_forward_fill] = ticker_df[
            columns_to_forward_fill
        ].ffill()
        ticker_df["date"] = ticker_df.index.date
        daily_ticker_frames.append(ticker_df.reset_index(drop = True))

    result = pd.concat(daily_ticker_frames, ignore_index = True)
    return _add_gemini_outputs(result, gemini_outputs, horizon_days)


def calendar_future_returns(ticker_df: pd.DataFrame, horizon_days: int) -> pd.Series:
    """
    Creates training labels for each return prediction horizon.
    Handles one ticker, across all given backtesting dates, for one horizon.
    """
    ticker_df = ticker_df.sort_values("date")
    dates = pd.to_datetime(ticker_df["date"]).reset_index(drop = True)
    prices = ticker_df["adjusted_close"].reset_index(drop = True)
    future_returns = []

    for index, current_date in enumerate(dates):
        target_date = current_date + pd.Timedelta(days = horizon_days)
        target_index = dates.searchsorted(target_date)

        if target_index >= len(ticker_df):
            future_returns.append(float("nan"))
            continue

        current_price = prices.iloc[index]
        future_price = prices.iloc[target_index]
        future_returns.append((future_price / current_price) - 1)

    return pd.Series(future_returns, index = ticker_df.index)


def calendar_future_volatility(ticker_df: pd.DataFrame,horizon_days: int) -> pd.Series:
    """
    Creates training labels for each volatility prediction horizon.
    Handles one ticker, across all given backtesting dates, for one horizon.
    """
    ticker_df = ticker_df.sort_values("date")
    dates = pd.to_datetime(ticker_df["date"]).reset_index(drop = True)
    daily_returns = ticker_df["daily_return"].reset_index(drop = True)
    future_volatility = []

    for current_date in dates:
        target_date = current_date + pd.Timedelta(days = horizon_days)
        future_returns = daily_returns[
            (dates > current_date) & (dates <= target_date)
        ].dropna()
        future_volatility.append(future_returns.std())

    return pd.Series(future_volatility, index = ticker_df.index)


def build_training_frame(
    df: pd.DataFrame, # Processed numerical df
    horizon_days: int,
    gemini_data: list[dict[str, Any]],
    feature_columns: list[str],
    kalman_q: float = 1e-5,
    kalman_r: float = 1e-2
) -> pd.DataFrame:
    """
    Adds future outcomes and removes rows that cannot be used for training.
    future_return_outcome is the percentage price change through the calendar target date.
    future_volatility_outcome uses real trading returns observed inside that calendar window.
    """
    result = build_model_feature_frame(
        df,
        gemini_data,
        horizon_days,
        kalman_q=kalman_q,
        kalman_r=kalman_r
    )

    grouped = result.groupby("ticker", group_keys = False)
    future_return_groups = (
        calendar_future_returns(ticker_df, horizon_days)
        for _, ticker_df in grouped
    )
    result["future_return_outcome"] = pd.concat(future_return_groups).reindex(
        result.index
    )
    future_volatility_groups = (
        calendar_future_volatility(ticker_df, horizon_days)
        for _, ticker_df in grouped
    )
    result["future_volatility_outcome"] = pd.concat(
        future_volatility_groups
    ).reindex(
        result.index
    )

    return result.dropna(
        subset=feature_columns + ["future_return_outcome", "future_volatility_outcome"]
    )


def train_return_predictor(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    timeline_days: int,
) -> GradientBoostingRegressor:
    """Fits a time-series-weighted Gradient Boosting return predictor"""

    # Extract input and output columns
    X = df[feature_columns]
    y = df[target_column]
    config = _return_model_config(timeline_days)
    
    # Set model hyperparameters
    return_model = GradientBoostingRegressor(
        n_estimators = config["n_estimators"], # Sequential tree learning steps
        learning_rate = config["learning_rate"], # Step size down loss gradient
        max_depth = 3, # Capture interactive feature variables
        subsample = 0.85, # Minimize variance (hide some data from predictors)
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
) -> RandomForestRegressor:
    """Fits a time-series-weighted Random Forest volatility predictor"""

    X = df[feature_columns]
    y = df[target_column]

    volatility_model = RandomForestRegressor(
        n_estimators = 200, # Parallel trees: can push higher
        max_depth = 6, # Less prone to overfitting: push depth higher
        min_samples_split = 5, # Lower = more specific rules (potential overfitting)
        n_jobs = -1, # Spread calculations across available cores
        random_state = 10
    )

    # Build exponential decay (half life) weight array
    a = math.log(2) / timeline_days
    dates = pd.to_datetime(df["date"])
    age_days = (dates.max() - dates).dt.days
    weights = [math.exp(-a * age) for age in age_days]

    volatility_model.fit(X, y, sample_weight = weights)

    return volatility_model
