'''
This predictor module is responsible for determining:
- Expected returns over time (Gradient Boosting)
- Future volatility trends (Random Forest)
These take in:
    1. price_trend_deviation (float)
    2. rolling_volatility (float)
    3. gemini_sentiment_score (float)
    4. gemini_risk_flag (0 - low, 1 - med, 2 - high)
'''

import pandas as pd
import numpy as np
import joblib

# Load trained model binaries from disk
return_model = joblib.load('ml_engine/saved_models/gbr_return_model.pkl')
volatility_model = joblib.load('ml_engine/saved_models/rfr_volatility_model.pkl')

# Makes an inference on future returns over given timeline
def return_inference(timeline: int, df: pd.DataFrame, kalman_filter = True) -> np.ndarray:
    live_df = df.copy()
    
    # 1. Kalman filter toggle
    if kalman_filter:
        live_df['price_trend_deviation'] = live_df['adjusted_close'] - live_df['kalman_smoothed_price']
    else:
        # Simple windowing based directly on the timeline parameter
        live_df['price_trend_deviation'] = live_df['adjusted_close'] - live_df['adjusted_close'].rolling(window = timeline, min_periods = 1).mean()
        
    # 2. Extract feature columns the model needs
    X = live_df[['price_trend_deviation', 'rolling_volatility', 'gemini_sentiment_score', 'gemini_risk_flag']]
    
    # 3. Run prediction
    return return_model.predict(X)

# Makes an inference on future volatility over given timeline
def volatility_inference(timeline: int, df: pd.DataFrame, kalman_filter = True) -> np.ndarray:
    live_df = df.copy()
    
    if kalman_filter:
        live_df['price_trend_deviation'] = live_df['adjusted_close'] - live_df['kalman_smoothed_price']
    else:
        live_df['price_trend_deviation'] = live_df['adjusted_close'] - live_df['adjusted_close'].rolling(window=timeline, min_periods=1).mean()
        
    X = live_df[['price_trend_deviation', 'rolling_volatility', 'gemini_sentiment_score', 'gemini_risk_flag']]
    
    return volatility_model.predict(X)