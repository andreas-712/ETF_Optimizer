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

BACKTESTING_PROMPT = """You are a quantitative financial data extractor for historical backtesting. Analyze only the provided article texts for the given stock ticker and requested date, then return one aggregate score for the requested prediction timeline.

Use only the provided article texts as your reference for market signals and information. Do not use your general knowledge about the company, ticker, market, earnings, products, lawsuits, analyst updates, or events unless that information is explicitly stated in the provided article texts. If the provided article texts do not contain enough evidence, return conservative low relevance and urgency scores instead of filling gaps from memory.

Return exactly one strict JSON object with exactly these keys:
{
  "relevance": number,
  "polarity": number,
  "urgency": number
}

Scoring rules:
1. relevance: 0 to 10 whole number increments. Higher means the available information is more directly impactful to the ticker's performance.
2. polarity: -1 or 1. Negative means bearish, positive means bullish.
3. urgency: 0 to 10 whole number increments. Higher means the catalyst is more likely to matter to the ticker's price inside the requested timeline.

Calibration rules:
1. Scores of 9 or 10 is rare. Use them only for unusually strong, company-moving information with clear relevance to the requested timeline. Scores of 0 or 1 indicate no visible headwinds or tailwinds for price movement.
2. A major scheduled event can be highly relevant without being bullish. Score polarity from expected market reaction, not from the company's general importance or brand strength. 
Do not mention or rely on actual stock price movement, returns, or market reaction after the requested date. Score only what a market participant could infer before the future window begins.
3. If the evidence is mixed, expectation-heavy, already priced in, or mostly speculative, choose conservative relevance and urgency scores.
4. For longer timelines, reduce urgency unless the catalyst is likely to keep affecting the stock throughout most of the requested window.

Aggregate all relevant events into one final score. Do not return event lists, nested objects, arrays, explanations, markdown, ticker, date, or any extra keys."""

LIVE_PREDICTION_PROMPT = """You are a quantitative financial data extractor for live prediction. Analyze the provided article texts for the given stock ticker as of the requested date, then return one aggregate score for the requested prediction timeline.

Use the provided article texts as your primary reference for market signals and information. You may additionally search the web data to find any other sources of sentiment or rapidly-changing information.

Return exactly one strict JSON object with exactly these keys:
{
  "relevance": number,
  "polarity": number,
  "urgency": number
}

Scoring rules:
1. relevance: 0 to 10 whole number increments. Higher means the available information is more directly impactful to the ticker's performance.
2. polarity: -1 or 1. Negative means bearish, positive means bullish.
3. urgency: 0 to 10 whole number increments. Higher means the catalyst is more likely to matter to the ticker's price inside the requested timeline.

Calibration rules:
1. Scores of 9 or 10 is rare. Use them only for unusually strong, company-moving information with clear relevance to the requested timeline. Scores of 0 or 1 indicate no visible headwinds or tailwinds for price movement.
2. A major scheduled event can be highly relevant without being bullish. Score polarity from expected market reaction, not from the company's general importance or brand strength.
3. If the evidence is mixed, expectation-heavy, already priced in, or mostly speculative, choose conservative relevance and urgency scores.
4. For longer timelines, reduce urgency unless the catalyst is likely to keep affecting the stock throughout most of the requested window.

Aggregate all relevant events into one final score. Do not return event lists, nested objects, arrays, explanations, markdown, ticker, date, or any extra keys."""

if DATA_MODE == LIVE_MODE:
    SYSTEM_PROMPT = LIVE_PREDICTION_PROMPT
else:
    SYSTEM_PROMPT = BACKTESTING_PROMPT

async def fetch_gemini_ticker_data(
    ticker: str,
    date: str,
    timeline_days: int,
    executive_summaries: list,
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
            contents=f"Extract data for {ticker} using only information available on or before {date}. Base your prediction on the following provided company metrics and executive summaries: {executive_summaries}. Score the expected direction and catalyst strength over approximately {timeline_days} days after {date}.",
            config=types.GenerateContentConfig(**config_kwargs)
        )
        
        # 2. Return the parsed dictionary instead of raw string
        return {"ticker": ticker, **json.loads(response.text)}
        
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}
    
