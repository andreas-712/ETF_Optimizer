'''
Collects historical Gemini inference ratings for ML training.
'''
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys
import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_engine.gemini import fetch_gemini_ticker_data
from ml_engine.market_data_collection import fetch_ticker_summaries


TRAINING_TIMEFRAME_DAYS = 1
FINAL_TRAINING_DATE = "2025-06-08"
OUTPUT_PATH = Path(__file__).resolve().parent / "gemini_training_data.csv"
RATE_LIMIT_WAIT_SECONDS = 20 # over 100 seconds on 5th retry
RATE_LIMIT_MAX_RETRIES = 5

TICKERS = ["AAPL", "NVDA"]
PREDICTION_HORIZON_DAYS = [3, 20]

OUTPUT_COLUMNS = [
    "date",
    "ticker",
    "prediction_horizon_days",
    "relevance",
    "polarity",
    "urgency",
    "gemini_sentiment_score",
]

REQUIRED_GEMINI_FIELDS = {"ticker", "relevance", "polarity", "urgency"}


def normalize_gemini_record(record: dict) -> dict | None:
    if not REQUIRED_GEMINI_FIELDS.issubset(record):
        return None

    return {
        "ticker": record["ticker"],
        "relevance": float(record["relevance"]),
        "polarity": float(record["polarity"]),
        "urgency": float(record["urgency"]),
    }


async def fetch_gemini_records(tickers: list[str], date: str, horizon_days: int) -> list[dict]:
    """
    Fetches Gemini inferences for the given ticker, date, and prediction horizon
    """
    records = []
    for ticker in tickers:
        retries = 0

        while True:
            summaries_and_data = fetch_ticker_summaries(ticker, horizon_days, date)
            if len(summaries_and_data) == 0:
                print(f"No data or executive summaries found for {ticker}, {horizon_days}d, {date}")
                records.append({
                    "ticker": ticker,
                    "error": "No data or executive summaries found",
                })
                break

            record = await fetch_gemini_ticker_data(ticker, date, horizon_days, summaries_and_data)
            error = str(record.get("error", "")).lower()
            rate_limited = "429" in error

            if not rate_limited or retries >= RATE_LIMIT_MAX_RETRIES:
                records.append(record)
                break

            retries += 1
            print(f"Rate limit hit. Waiting {RATE_LIMIT_WAIT_SECONDS}s before retry {retries} for {ticker}.")
            await asyncio.sleep(RATE_LIMIT_WAIT_SECONDS)

    return records


async def main_async():
    final_date = datetime.strptime(FINAL_TRAINING_DATE, "%Y-%m-%d").date()
    start_date = final_date - timedelta(days=TRAINING_TIMEFRAME_DAYS - 1)

    for day_offset in range(TRAINING_TIMEFRAME_DAYS):
        date = (start_date + timedelta(days=day_offset)).isoformat()

        for horizon_days in PREDICTION_HORIZON_DAYS:
            print(f"Collecting {date}, {horizon_days}d")
            gemini_data = await fetch_gemini_records(TICKERS, date, horizon_days)

            rows = []
            for record in gemini_data:
                if not isinstance(record, dict):
                    print(f"Skipping non-JSON response: {record}")
                    continue

                if "error" in record:
                    print(f"Skipping {record['ticker']}: {record['error']}")
                    continue

                normalized_record = normalize_gemini_record(record)
                if normalized_record is None:
                    print(f"Skipping invalid JSON response: {record}")
                    continue

                row = {
                    "date": date,
                    "ticker": normalized_record["ticker"],
                    "prediction_horizon_days": horizon_days,
                    "relevance": normalized_record["relevance"],
                    "polarity": normalized_record["polarity"],
                    "urgency": normalized_record["urgency"],
                }
                row["gemini_sentiment_score"] = (
                    row["relevance"] * row["polarity"] * row["urgency"]
                )
                rows.append(row)

            rows_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
            rows_df.to_csv(
                OUTPUT_PATH,
                mode="a",
                header=not OUTPUT_PATH.exists(),
                index=False,
            )
            print(f"Appended {len(rows)} rows to {OUTPUT_PATH}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
