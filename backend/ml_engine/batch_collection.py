"""Collect Finnhub article summaries in 30-day batches"""

import datetime as dt
import json
import os
from pathlib import Path
import requests
from dotenv import load_dotenv
from ml_engine.market_data_collection import FMP_ENDPOINTS


BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".flaskenv")

START_DATE = dt.date(2025, 9, 26)
END_DATE = dt.date(2026, 3, 26)
CHUNK_DAYS = 30
TICKERS = ["NVDA", "AAPL", "AMZN", "GOOG", "MSFT"]
NUM_QUARTERS = 6

COLLECTION_STATES = {
    "summaries": "Y",
    "balance_sheets": "Y",
    "historical_grades": "Y",
}

FINNHUB_URL = os.getenv("FINNHUB_URL")
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
FINNHUB_ENDPOINT = "/company-news?"
FMP_URL = os.getenv("FMP_URL")
FMP_KEY = os.getenv("FMP_KEY")
OUTPUT_PATH = Path(__file__).resolve().parent / "finnhub_and_fmp_data.json"


def collect_summaries() -> dict:
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

    return summaries_by_ticker


def collect_balance_sheets() -> dict:
    """
    Collects balance sheets ordered from newest to oldest by filing date
    """
    balance_sheets = {ticker: {} for ticker in TICKERS}

    for ticker in TICKERS:
        params = {"symbol": ticker.upper(), "limit": NUM_QUARTERS, "period": "quarter"}
        params["apikey"] = FMP_KEY
        req = FMP_URL + FMP_ENDPOINTS[0]
        response = requests.get(req, params = params).json()

        for balance_sheet in response:
            balance_sheets[ticker][balance_sheet["filingDate"]] = {
                "cashAndShortTermInvestments": balance_sheet["cashAndShortTermInvestments"],
                "totalCurrentAssets": balance_sheet["totalCurrentAssets"],
                "totalLiabilitiesAndTotalEquity": balance_sheet["totalLiabilitiesAndTotalEquity"],
                "totalDebt": balance_sheet["totalDebt"],
            }

        balance_sheets[ticker] = dict(
            sorted(balance_sheets[ticker].items(), reverse = True)
        )

    return balance_sheets


def collect_historical_grades() -> dict:
    """
    Collects historical grades ordered from newest to oldest by date
    """
    historical_grades = {ticker: {} for ticker in TICKERS}

    for ticker in TICKERS:
        params = {
            "symbol": ticker.upper(), "limit": NUM_QUARTERS * 20, "apikey": FMP_KEY}
        req = FMP_URL + FMP_ENDPOINTS[1]
        response = requests.get(req, params = params).json()

        for analyst_rating in response:
            historical_grades[ticker][analyst_rating["date"]] = {
                "analystRatingsStrongBuy": analyst_rating["analystRatingsStrongBuy"],
                "analystRatingsBuy": analyst_rating["analystRatingsBuy"],
                "analystRatingsHold": analyst_rating["analystRatingsHold"],
                "analystRatingsSell": analyst_rating["analystRatingsSell"],
                "analystRatingsStrongSell": analyst_rating["analystRatingsStrongSell"]
            }

        historical_grades[ticker] = dict(
            sorted(historical_grades[ticker].items(), reverse = True)
        )

    return historical_grades


def main():
    collected_data = {ticker: {} for ticker in TICKERS}

    if COLLECTION_STATES["summaries"] == "Y":
        summaries_by_ticker = collect_summaries()
        for ticker in TICKERS:
            collected_data[ticker]["ummaries"] = summaries_by_ticker[ticker]

    if COLLECTION_STATES["balance_sheets"] == "Y":
        balance_sheets = collect_balance_sheets()
        for ticker in TICKERS:
            collected_data[ticker]["balance_sheets"] = balance_sheets[ticker]

    if COLLECTION_STATES["historical_grades"] == "Y":
        historical_grades = collect_historical_grades()
        for ticker in TICKERS:
            collected_data[ticker]["historical_grades"] = historical_grades[ticker]

    OUTPUT_PATH.write_text(
        json.dumps(collected_data, indent = 2),
        encoding = "utf-8",
    )
    print(f"Saved batch data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
