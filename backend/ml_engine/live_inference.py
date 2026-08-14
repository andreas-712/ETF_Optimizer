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
from ml_engine.exceptions import userInputError
from ml_engine.model_orchestrator import load_models

# Can choose multiple sectors, company sizes, and blacklisted tickers
# USER_INPUTS = ["sectors", "company_sizes", "risk_tolerance", "blacklisted", "max_pool", "min_pool"]
# RISK_TOLERANCES = {"high", "medium", "low"}
# SECTORS = {"technology": max_pct, "financial": max_pct, "energy": 1}, where 1 is the uncapped sentinel

YFINANCE_SECTORS = {
    "energy": "Energy",
    "financial": "Financial Services",
    "technology": "Technology",
}
COMPANY_SIZES = {"big-cap", "mid-cap", "small-cap"}
SMALL_CAP_LOW = 300_000_000 # $300 million USD
SMALL_CAP_HIGH = 2_000_000_000 # $2 billion USD
BIG_CAP_LOW = 10_000_000_000 # $10 billion USD
MIN_POOL_LOW_BOUND = 10
MAX_POOL_UPPER_BOUND = 100


async def predict_tickers(
    horizon_days: int,
    ticker_industries: dict[str, str],
) -> dict[str, dict[str, float | int]]:
    """
    Master async function for returning volatility and percent change data over the given horizon.
    Returns volatility pct, return pct, horizon days for each ticker.
    Contains values {ticker: {volatility: float, return: float, horizon_days: int}}
    """
    if horizon_days not in MODELS:
        print(f"Prediction horizon {horizon_days} unavailable")
        return  {}

    tickers = list(ticker_industries)

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
            ticker_gemini_input_data,
            mode="live",
        )
        for ticker, ticker_gemini_input_data in zip(tickers, gemini_input_data)
    ]
    gemini_inference_dicts = await asyncio.gather(*gemini_inference_tasks)

    valid_gemini_inference_dicts = []
    for gemini_inference_dict in gemini_inference_dicts:
        if len(gemini_inference_dict) != len(GEMINI_RESPONSE_FIELDS):
            t = gemini_inference_dict.get("ticker")
            print(f"Invalid field count for {t}")
            continue
        invalid_fields = set(gemini_inference_dict.keys()) - GEMINI_RESPONSE_FIELDS
        if invalid_fields:
            t = gemini_inference_dict.get("ticker")
            print(f"Invalid field names for {t}: {invalid_fields}")
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
    live_input_frame = ml_processed_input_frame.groupby("ticker", group_keys = False).tail(1).copy()
    live_input_frame["industry"] = live_input_frame["ticker"].map(ticker_industries)

    # Load and predict
    prediction_model = MODELS[horizon_days]
    return_pcts = prediction_model.return_inference(live_input_frame)
    volatility_pcts = prediction_model.volatility_inference(live_input_frame)

    predictions = {}
    for index, ticker in enumerate(live_input_frame["ticker"]):
        predictions[ticker] = {
            "volatility": float(volatility_pcts[index]),
            "return": float(return_pcts[index]),
            "horizon_days": horizon_days,
        }

    return predictions


def run_live_inference(live_inputs: dict) -> dict:
    """
    Select live candidates, predict their returns and volatility, and print them.
    Master function which returns all predictions with the given user input parameters.
    """
    if live_inputs["horizon_days"] not in MODELS:
        raise userInputError("Invalid input horizon")
    if not live_inputs["sectors"] or not set(live_inputs["sectors"].keys()).issubset(YFINANCE_SECTORS.keys()):
        raise userInputError("Missing or invalid sector(s) given")
    if not live_inputs["sizes"] or not set(live_inputs["sizes"]).issubset(COMPANY_SIZES):
        raise userInputError("Missing or invalid size(s) given")
    if live_inputs["min_pool"] < MIN_POOL_LOW_BOUND or live_inputs["max_pool"] > MAX_POOL_UPPER_BOUND or live_inputs["min_pool"] > live_inputs["max_pool"]:
        raise userInputError("Invalid pool size bound(s)")
    # Min/max now validated
    s = 0
    ones = 0
    for _, val in live_inputs["sectors"].items():
        if 1 > val > (1. - 1. / live_inputs["max_pool"]) or val < (1. / live_inputs["max_pool"]): # Dynamic bounds based on granularity
            raise userInputError("Invalid bounds: over")
        if val == 1:
            ones += 1
        s += val
    if ones != 1:
        raise userInputError("Need exactly one value of 1 assigned to a sector to fill")
    if s < 1 or s >= (2 - 1 / (live_inputs["max_pool"])):
        raise userInputError("Incorrect ETF sector(s) composition given")
    
    load_models()
    candidates = asyncio.run(get_ticker_pool(live_inputs))
    ticker_industries = {
        candidate["ticker"]: candidate["industry"]
        for candidate in candidates
    }
    predictions = asyncio.run(
        predict_tickers(
            live_inputs["horizon_days"],
            ticker_industries,
        )
    )
    candidate_metadata = {
        candidate["ticker"]: {
            "industry": candidate["industry"],
            "market_cap_category": candidate["market_cap_category"],
        }
        for candidate in candidates
    }
    for ticker, prediction in predictions.items():
        prediction.update(candidate_metadata[ticker])

    return predictions


def _build_ticker_query(user_inputs: dict, sector: str) -> yf.EquityQuery:
    """Builds the yfinance query for gathering preliminary candidates"""
    try:
        yfinance_sector = YFINANCE_SECTORS[sector]
    except KeyError:
        allowed_sectors = ", ".join(sorted(YFINANCE_SECTORS))
        # Prevent exception chaining
        raise ValueError(f"Unsupported sector {sector!r}; expected one of: {allowed_sectors}") from None

    # US-only
    filters = [yf.EquityQuery("eq", ["region", "us"])]

    # Add sectors to filter
    filters.append(yf.EquityQuery("eq", ["sector", yfinance_sector]))

    sizes = user_inputs["sizes"]
    if not sizes:
        raise ValueError("Select at least one company size")

    size_filters = []
    if "big-cap" in sizes:
        size_filters.append(yf.EquityQuery("gte", ["intradaymarketcap", BIG_CAP_LOW]))
    if "mid-cap" in sizes:
        size_filters.append(yf.EquityQuery("btwn", ["intradaymarketcap", SMALL_CAP_HIGH, BIG_CAP_LOW]))
    if "small-cap" in sizes:
        size_filters.append(yf.EquityQuery("btwn", ["intradaymarketcap", SMALL_CAP_LOW, SMALL_CAP_HIGH]))

    # Allow query to return any valid company size
    if len(size_filters) == 1:
        filters.append(size_filters[0])
    else:
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
    """Asynchronously retrieves aggregated analyst rating scores for preliminary ticker filtering."""
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


def _get_market_cap(quote: dict) -> float | None:
    """Return Yahoo's market-cap value."""
    for field in ("intradayMarketCap", "intradaymarketcap", "marketCap"):
        value = quote.get(field)
        if value is not None:
            return float(value)
    return None


def _market_cap_category(market_cap: float | None) -> str | None:
    """Map a market-cap value to the size buckets used by the screener."""
    if market_cap is None:
        return None
    if market_cap < SMALL_CAP_HIGH:
        return "small-cap"
    if market_cap < BIG_CAP_LOW:
        return "mid-cap"
    return "big-cap"

async def get_ticker_pool(user_inputs: dict) -> list:
    """Returns up to the max pool sector-limited candidates."""
    clamped_min_pool = min(
        max(user_inputs["min_pool"], MIN_POOL_LOW_BOUND),
        MAX_POOL_UPPER_BOUND,
    )
    clamped_max_pool = min(
        max(user_inputs["max_pool"], clamped_min_pool),
        MAX_POOL_UPPER_BOUND,
    )
    selected_pool = []
    sector_pools = []

    # Query sector-capped industries first to fill the remainder with remaining sector
    for sector, cap in sorted(user_inputs["sectors"].items(), key=lambda item: item[1]):
        query = _build_ticker_query(user_inputs, sector)
        screener_data = yf.screen(query, size = clamped_max_pool * 2)
        quotes = screener_data.get("quotes", [])
        candidates = [
            {
                "ticker": quote["symbol"],
                "market_cap": _get_market_cap(quote),
            }
            for quote in quotes
            if "symbol" in quote and quote["symbol"] not in user_inputs["blacklisted"]
        ]

        scores = await asyncio.gather(
            *[_fetch_analyst_ratings(candidate["ticker"]) for candidate in candidates]
        )
        sector_pool = [
            {
                "ticker": candidate["ticker"],
                "industry": sector,
                "market_cap_category": _market_cap_category(candidate["market_cap"]),
                "score": score,
            }
            for candidate, score in zip(candidates, scores)
        ]
        sector_pool.sort(key = lambda row: row["score"], reverse = True)
        sector_pools.append(sector_pool)

        pool_space = clamped_max_pool - len(selected_pool)
        sector_limit = clamped_max_pool if cap == 1 else int(clamped_max_pool * cap)
        sector_limit = min(pool_space, sector_limit)
        selected_pool.extend(sector_pool[:sector_limit])

    # Truncated sector limits can leave fewer than min_pool candidates
    # In this case, use the best remaining candidates without exceeding max_pool
    if len(selected_pool) < clamped_min_pool:
        selected_tickers = {row["ticker"] for row in selected_pool}
        remaining_candidates = [
            row
            for sector_pool in sector_pools
            for row in sector_pool
            if row["ticker"] not in selected_tickers
        ]
        remaining_candidates.sort(key = lambda row: row["score"], reverse = True)
        required_candidates = clamped_min_pool - len(selected_pool)
        selected_pool.extend(remaining_candidates[:required_candidates])

    return [
        {
            "ticker": row["ticker"],
            "industry": row["industry"],
            "market_cap_category": row["market_cap_category"],
        }
        for row in selected_pool[:clamped_max_pool]
    ]
