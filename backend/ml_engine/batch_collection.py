"""Collect Finnhub article summaries in 30-day batches"""

import datetime as dt
import json
import os
from pathlib import Path
import requests
from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".flaskenv")

START_DATE = dt.date(2025, 9, 26)
END_DATE = dt.date(2026, 3, 26)
CHUNK_DAYS = 30
TICKERS = ["NVDA", "AAPL", "AMZN", "GOOG", "MSFT"]

FINNHUB_URL = os.getenv("FINNHUB_URL")
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
FINNHUB_ENDPOINT = "/company-news?"
OUTPUT_PATH = Path(__file__).resolve().parent / "finnhub_executive_summaries.json"


def main():
    summaries_by_ticker = {ticker: {} for ticker in TICKERS}

    for ticker in TICKERS:
        chunk_start = START_DATE

        while chunk_start <= END_DATE:
            chunk_end = min(
                chunk_start + dt.timedelta(days=CHUNK_DAYS - 1),
                END_DATE,
            )
            params = {
                "symbol": ticker,
                "from": chunk_start.isoformat(),
                "to": chunk_end.isoformat(),
                "token": FINNHUB_KEY,
            }

            print(f"Fetching {ticker}: {chunk_start} to {chunk_end}")
            articles = requests.get(
                FINNHUB_URL + FINNHUB_ENDPOINT,
                params = params,
            ).json()

            for article in articles:
                article_date = dt.datetime.fromtimestamp( # Convert from unix timestamp to datetime
                    article["datetime"],
                    tz = dt.timezone.utc,
                ).date().isoformat()
                summaries_by_ticker[ticker].setdefault(article_date, []).append(
                    article["summary"]
                )

            chunk_start = chunk_end + dt.timedelta(days = 1)

        summaries_by_ticker[ticker] = dict(
            sorted(summaries_by_ticker[ticker].items())
        )

    OUTPUT_PATH.write_text(
        json.dumps(summaries_by_ticker, indent = 2),
        encoding = "utf-8",
    )
    print(f"Saved article summaries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
