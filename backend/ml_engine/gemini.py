import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load env variables
BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".flaskenv")

LIVE_MODE = "live"
DATA_MODE = os.getenv("DATA_MODE", "backtest").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "No Gemini API key found"
    )
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_USE_SEARCH = os.getenv("GEMINI_USE_SEARCH", "false").lower() == "true"

# Initialize the SDK client
client = genai.Client(api_key=GEMINI_API_KEY)
grounding_tool = types.Tool(google_search=types.GoogleSearch())

BACKTESTING_PROMPT = """You are a quantitative financial data extractor for historical backtesting. Analyze only the provided article summaries, balance sheets, and historical grades for the given stock ticker and requested date, then return one aggregate score for the requested prediction timeline.

Use only the provided article summaries and company metrics as your reference for market signals and information. Do not use your general knowledge about the company, ticker, market, earnings, products, lawsuits, analyst updates, or events unless that information is explicitly stated in the provided texts and company metrics. If the provided data does not contain sufficient signals, return conservative low relevance and urgency scores.

Return exactly one strict JSON object with exactly these keys:
{
  "ticker": string,
  "date": "YYYY-MM-DD",
  "prediction_horizon_days": number,
  "relevance": number,
  "polarity": number,
  "urgency": number
}

Copy ticker, date, and prediction_horizon_days exactly from the user request.

Scoring rules:
1. relevance: 0 to 10 whole number increments. High means the available information is directly impactful to the ticker's performance.
2. polarity: -1 or 1. Negative means bearish, positive means bullish.
3. urgency: 0 to 10 whole number increments. Higher means the catalyst is more likely to cause price changes for the given ticker over the requested timeline.

Calibration rules:
1. Scores of 9 or 10 is rare. Use them only for company moving, absolutely compelling information with clear relevance to the requested timeline. Scores of 0 or 1 indicate minimal visible headwinds or tailwinds for price movement.
2. A major scheduled event can be highly relevant without being bullish. Score polarity from expected market reaction, not from the company's general importance or brand strength. 
3. If the evidence is mixed, expectation-heavy, already priced in, or mostly speculative, choose conservative urgency scores (near 0).
4. For longer timelines, reduce urgency unless the catalyst is likely to keep affecting the stock throughout most of the requested window.

Aggregate all relevant events into one final score. Do not return event lists, nested objects, arrays, explanations, markdown, or any extra keys."""

LIVE_PREDICTION_PROMPT = """You are a quantitative financial data extractor for future ticker price prediction. Analyze only the provided article summaries, balance sheets, and historical grades for the given stock ticker, then return one aggregate score for the requested prediction timeline.

Use only the provided article summaries and company metrics as your reference for market signals and information. Do not use your general knowledge about the company, ticker, market, earnings, products, lawsuits, analyst updates, or events unless that information is explicitly given in the provided texts and company metrics. If the provided data do not contain sufficient signals, return conservative low relevance and urgency scores.

Return exactly one strict JSON object with exactly these keys:
{
  "relevance": number,
  "polarity": number,
  "urgency": number
}

Scoring rules:
1. relevance: 0 to 10 whole number increments. High means the available information is directly impactful to the ticker's performance.
2. polarity: -1 or 1. Negative means bearish, positive means bullish.
3. urgency: 0 to 10 whole number increments. Higher means the catalyst is more likely to cause price changes for the given ticker over the requested timeline.

Calibration rules:
1. Scores of 9 or 10 is rare. Use them only for company moving, absolutely compelling information with clear relevance to the requested timeline. Scores of 0 or 1 indicate minimal visible headwinds or tailwinds for price movement.
2. A major scheduled event can be highly relevant without being bullish. Score polarity from expected market reaction, not from the company's general importance or brand strength. 
3. If the evidence is mixed, expectation-heavy, already priced in, or mostly speculative, choose conservative urgency scores (near 0).
4. For longer timelines, reduce urgency unless the catalyst is likely to keep affecting the stock throughout most of the requested window.

Aggregate all relevant events into one final score. Do not return event lists, nested objects, arrays, explanations, markdown, ticker, date, or any extra keys."""

if DATA_MODE == LIVE_MODE:
    SYSTEM_PROMPT = LIVE_PREDICTION_PROMPT
else:
    SYSTEM_PROMPT = BACKTESTING_PROMPT

async def fetch_gemini_ticker_inference(
    ticker: str,
    date: str,
    timeline_days: int,
    ticker_data: dict
):
    # Stateless, asynchronous call for a single ticker
    print(f"Dispatching request for {ticker}...")
    
    try:
        config_kwargs = {
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0.0,
            "response_mime_type": "application/json",
        }
        if GEMINI_USE_SEARCH:
            config_kwargs["tools"] = [grounding_tool]

        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=(
                f"Extract data for {ticker} using only the provided information. "
                f"Balance sheet, analyst recommendations, and news headlines: {ticker_data}. "
                f"Score the expected price movement over approximately {timeline_days} days after {date}."
            ),
            config=types.GenerateContentConfig(**config_kwargs)
        )
        
        # 2. Return the parsed dictionary instead of raw string
        return {"ticker": ticker, **json.loads(response.text)}
        
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}
    
