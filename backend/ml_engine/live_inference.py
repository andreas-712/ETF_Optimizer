"""
This file contains functions for the live inference flow:
    - Selecting candidate tickers
    - Parallelized inference for return and volatility, abstracting the end-to-end flow
"""
import datetime as dt
from zoneinfo import ZoneInfo # Standardized to ny time
import yfinance as yf
import asyncio

from ml_engine.model_orchestrator import MODELS
from ml_engine.market_data_collection import fetch_numerical_ticker_data, fetch_ticker_gemini_inputs
from ml_engine.gemini import fetch_gemini_ticker_inference, GEMINI_RESPONSE_FIELDS
from ml_engine.train import build_model_feature_frame, ROLLING_PRICE_WINDOW, ROLLING_VOLATILITY_WINDOW

# Can choose multiple sectors, company sizes, and blacklisted tickers
# USER_INPUTS = ["sectors", "company_sizes", "risk_tolerance", "blacklisted", "max_pool", "min_pool"]
# RISK_TOLERANCES = {"high", "medium", "low"}
# SECTORS = {"technology", "financial", "energy"}
# COMPANY_SIZES = {"big-cap", "mid-cap", "small-cap"}

SMALL_CAP_LOW = 300_000_000 # $300 million USD
SMALL_CAP_HIGH = 2_000_000_000 # $2 billion USD
BIG_CAP_LOW = 10_000_000_000 # $10 billion USD
MIN_POOL_LOW_BOUND = 5
MAX_POOL_UPPER_BOUND = 50


async def predict_tickers(tickers: list[str], horizon_days: int) -> dict[str, dict[str, float | int]]:
    """
    Master async function for returning volatility and percent change data over the given horizon.
    Returns volatility pct, return pct, horizon days for each ticker.
    Contains values {ticker: {volatility: float, return: float, horizon_days: int}}
    """
    if horizon_days not in MODELS:
        print(f"Prediction horizon {horizon_days} unavailable")
        return  {}

    # Record current date
    now = dt.datetime.now(ZoneInfo("America/New_York"))
    now_date = now.date()
    # Fetch 3x calendar days to capture enough trading days
    start_rolling_window_date = now_date - dt.timedelta(days = max(ROLLING_PRICE_WINDOW, ROLLING_VOLATILITY_WINDOW) * 3)

    # Fetch numerical data for the whole ticker batch
    numerical_data_df = fetch_numerical_ticker_data(
        tickers,
        start_rolling_window_date.strftime("%Y-%m-%d"),
        now_date.strftime("%Y-%m-%d")
    )

    # Fetch Gemini input data concurrently because yfinance calls are synchronous
    gemini_input_tasks = [asyncio.to_thread(fetch_ticker_gemini_inputs, ticker) for ticker in tickers]
    gemini_input_data = await asyncio.gather(*gemini_input_tasks)

    # Wait for all Gemini inferences concurrently
    gemini_inference_tasks = [
        fetch_gemini_ticker_inference(
            ticker,
            now_date.strftime("%Y-%m-%d"),
            horizon_days,
            ticker_gemini_input_data
        )
        for ticker, ticker_gemini_input_data in zip(tickers, gemini_input_data)
    ]
    gemini_inference_dicts = await asyncio.gather(*gemini_inference_tasks)

    valid_gemini_inference_dicts = []
    for gemini_inference_dict in gemini_inference_dicts:
        if len(gemini_inference_dict) != len(GEMINI_RESPONSE_FIELDS):
            print(f"Invalid field count for {gemini_inference_dict.get("ticker")}")
            continue
        invalid_fields = set(gemini_inference_dict.keys()) - GEMINI_RESPONSE_FIELDS
        if invalid_fields:
            print(f"Invalid field names for {gemini_inference_dict.get("ticker")}: {invalid_fields}")
            continue
        valid_gemini_inference_dicts.append(gemini_inference_dict)

    if not valid_gemini_inference_dicts:
        print("No valid Gemini input data received")
        return {}

    valid_tickers = [gemini_inference_dict["ticker"] for gemini_inference_dict in valid_gemini_inference_dicts]
    numerical_data_df = numerical_data_df[numerical_data_df["ticker"].isin(valid_tickers)]

    # Combine the numerical and processed gemini frames into final input frame
    ml_processed_input_frame = build_model_feature_frame(
        numerical_data_df, 
        valid_gemini_inference_dicts, 
        horizon_days,
        now_date.strftime("%Y-%m-%d"),
    )
    # Return only latest (live) row for each ticker
    live_input_frame = ml_processed_input_frame.groupby("ticker", group_keys = False).tail(1)

    # Load and predict
    prediction_model = MODELS[horizon_days]
    return_pcts = prediction_model.return_inference(live_input_frame)
    volatility_pcts = prediction_model.volatility_inference(live_input_frame)

    predictions = {}
    for index, ticker in enumerate(live_input_frame["ticker"]):
        predictions[ticker] = {
            "volatility": float(volatility_pcts[index]),
            "return": float(return_pcts[index]),
            "horizon_days": horizon_days
        }

    return predictions


def _build_ticker_query(user_inputs: dict) -> yf.EquityQuery:
    """Builds the yfinance query for gathering preliminary candidates"""
    # US-only
    filters = [yf.EquityQuery("eq", ["region", "us"])]

    sectors = user_inputs["sectors"]
    if not sectors:
        print("No sectors selected")

    # Add sectors to filter
    filters.append(yf.EquityQuery("is-in", ["sector"] + [sector for sector in sectors]))

    sizes = user_inputs["sizes"]
    if not sizes:
        print("No company sizes selected")

    size_filters = []
    if "big-cap" in sizes:
        size_filters.append(yf.EquityQuery("gte", ["intradaymarketcap", BIG_CAP_LOW]))
    if "mid-cap" in sizes:
        size_filters.append(yf.EquityQuery("btwn", ["intradaymarketcap", SMALL_CAP_HIGH, BIG_CAP_LOW]))
    if "small-cap" in sizes:
        size_filters.append(yf.EquityQuery("btwn", ["intradaymarketcap", SMALL_CAP_LOW, SMALL_CAP_HIGH]))

    filters.append(yf.EquityQuery("or", size_filters))

    return yf.EquityQuery("and", filters)


def _sync_fetch(ticker: str):
    """Synchronous worker to avoid freezing state loop"""
    try:
        return yf.Ticker(ticker).recommendations
    except Exception:
        print(f"Analyst ratings retrieval for ticker {ticker} failed")
        return None
        
async def _fetch_analyst_ratings(symbol: str) -> float:
    """Asynchronously retrieves aggregated analyst rating scores for preliminary ticker filtering"""
    df = await asyncio.to_thread(_sync_fetch, symbol)
    if df is None or df.empty or "period" not in df.columns:
        return -999.0 # Sentinel value

    live_month = df[df["period"] == "0m"]
    if live_month.empty:
        return -999.0
    
    row = live_month.iloc[0]
    bullish_signals = (row.get('strongBuy') * 2) + (row.get('buy') * 1)
    bearish_signals = (row.get('strongSell') * 2) + (row.get('sell') * 1)

    return float(bullish_signals - bearish_signals)

async def get_ticker_pool(user_inputs: dict) -> list:
    """Returns a filtered pool of candidate tickers of up to max_pool tickers"""
    # Build and get query
    query = _build_ticker_query(user_inputs)

    # 2. Get initial candidates (2x max)
    clamped_max_pool = min(max(user_inputs["max_pool"], MIN_POOL_LOW_BOUND * 2), MAX_POOL_UPPER_BOUND * 2)
    screener_data = yf.screen(query, size = clamped_max_pool * 2)
    quotes = screener_data.get('quotes', [])
    full_candidates = [quote["symbol"] for quote in quotes if "symbol" in quote]

    # Begin filtering
    # 1. Blacklisted
    filtered_candidates = [t for t in full_candidates if t not in user_inputs["blacklisted"]]
    
    # 2. Analyst ratings
    tasks = [_fetch_analyst_ratings(ticker) for ticker in filtered_candidates]
    scores = await asyncio.gather(*tasks)

    valid_pool = [(t, s) for t, s in zip(filtered_candidates, scores)]
    valid_pool.sort(key = lambda x: x[1], reverse = True)

    iterator = max(clamped_max_pool, len(valid_pool))
    return [valid_pool[0] for _ in range(iterator)]
