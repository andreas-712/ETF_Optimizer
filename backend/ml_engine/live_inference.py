import datetime as dt
from zoneinfo import ZoneInfo # Standardized to ny time

from ml_engine.model_orchestrator import MODELS
from ml_engine.market_data_collection import fetch_numerical_ticker_data, fetch_ticker_gemini_inputs
from ml_engine.gemini import fetch_gemini_ticker_inference, GEMINI_RESPONSE_FIELDS
from ml_engine.train import build_model_feature_frame, ROLLING_PRICE_WINDOW, ROLLING_VOLATILITY_WINDOW


async def predict_ticker(ticker: str, horizon_days: int) -> dict[str, float | int]:
    """
    Master async function for returning volatility and percent change data over the given horizon.
    (To be used in math engine as 2 variables in ETF composition).
    Returns volatility pct, return pct, horizon days.
    """

    if horizon_days not in MODELS:
        print(f"Error: prediction horizon {horizon_days} unavailable")
        return  {}

    # Record current date
    now = dt.datetime.now(ZoneInfo("America/New_York"))
    now_date = now.date()
    # Fetch 3x calendar days to capture enough trading days
    start_rolling_window_date = now_date - dt.timedelta(days = max(ROLLING_PRICE_WINDOW, ROLLING_VOLATILITY_WINDOW) * 3)

    # Fetch numerical and Gemini input data
    numerical_data_df = fetch_numerical_ticker_data([ticker], start_rolling_window_date.strftime("%Y-%m-%d"), now_date.strftime("%Y-%m-%d"))
    gemini_input_data_dict = fetch_ticker_gemini_inputs(ticker)

    # Wait for Gemini inference
    gemini_inference_dict = await fetch_gemini_ticker_inference(ticker, now_date.strftime("%Y-%m-%d"), horizon_days, gemini_input_data_dict)
    if len(gemini_inference_dict) != len(GEMINI_RESPONSE_FIELDS):
        print("Invalid field count")
        return {}
    for field in gemini_inference_dict.keys():
        if field not in GEMINI_RESPONSE_FIELDS:
            print("Invalid field name")
            return {}

    # Combine the numerical and processed gemini frames into final input frame
    ml_processed_input_frame = build_model_feature_frame(
        numerical_data_df, 
        [gemini_inference_dict], 
        horizon_days,
        now_date.strftime("%Y-%m-%d"),
    )
    # Return only latest (live) row
    live_input_frame = ml_processed_input_frame.tail(1)

    # Load and predict
    prediction_model = MODELS[horizon_days]
    return_pct = float(prediction_model.return_inference(live_input_frame)[0])
    volatility_pct = float(prediction_model.volatility_inference(live_input_frame)[0])

    return {"volatility": volatility_pct, "return": return_pct, "horizon_days": horizon_days}