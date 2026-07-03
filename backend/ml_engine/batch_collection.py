"""
Collects executive summaries, balance sheets and historical grades for backtesting
File writes: batch_data / [finnhub_summaries.json, fmp_balance_sheets.json, fmp_historical_grades.json]
"""

import datetime as dt
import json
import os
import time
from pathlib import Path
import requests
from dotenv import load_dotenv
from ml_engine.market_data_collection import FMP_ENDPOINTS
import re


BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".flaskenv")

SUMMARY_START_DATE = dt.date(2025, 8, 26)
START_DATE = dt.date(2025, 9, 26) # Inclusive
END_DATE = dt.date(2026, 3, 26) # Exclusive
CHUNK_DAYS = 30
TICKERS = ["NVDA", "AAPL", "AMZN", "META", "MSFT"]
# 2 quarters of data + starting 1 quarter before today + 1 quarter buffer
NUM_QUARTERS = 5

COLLECTION_STATES = {
    "summaries": "N",
    "balance_sheets": "N",
    "historical_grades": "N",
    "filter_summaries": "N",
    "produce_training_batch": "Y"
}

FINNHUB_URL = os.getenv("FINNHUB_URL")
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
FINNHUB_ENDPOINT = "/company-news?"
FMP_URL = os.getenv("FMP_URL")
FMP_KEY = os.getenv("FMP_KEY")
OUTPUT_DIR = Path(__file__).resolve().parent / "batch_data"
OUTPUT_DIR.mkdir(exist_ok=True)
SUMMARIES_OUTPUT_PATH = OUTPUT_DIR / "finnhub_summaries.json"
BALANCE_SHEETS_OUTPUT_PATH = OUTPUT_DIR / "fmp_balance_sheets.json"
HISTORICAL_GRADES_OUTPUT_PATH = OUTPUT_DIR / "fmp_historical_grades.json"
FILTERED_SUMMARIES_OUTPUT_PATH = OUTPUT_DIR / "filtered_finnhub_summaries.json"
TRAINING_BATCH_OUTPUT_PATH = OUTPUT_DIR / "training_batch.json"
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")



def collect_summaries() -> dict:
    """Collect article summaries ordered from newest to oldest by date."""
    summaries_by_ticker = {ticker: {} for ticker in TICKERS}

    for ticker in TICKERS:
        chunk_start = SUMMARY_START_DATE

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

            time.sleep(1.5)

            for article in sorted(
                articles,
                key = lambda article: article["datetime"],
                reverse = True,
            ):
                article_date = dt.datetime.fromtimestamp( # Convert from unix timestamp to datetime
                    article["datetime"],
                    tz = dt.timezone.utc,
                ).date().isoformat()
                summaries_by_ticker[ticker].setdefault(article_date, []).append(
                    article["summary"]
                )

            chunk_start = chunk_end + dt.timedelta(days = 1)

        summaries_by_ticker[ticker] = dict(
            sorted(summaries_by_ticker[ticker].items(), reverse = True)
        )

    return summaries_by_ticker


def collect_balance_sheets() -> dict:
    """
    Collects balance sheets ordered from newest to oldest by filing date
    """
    balance_sheets = {ticker: {} for ticker in TICKERS}

    for ticker in TICKERS:
        params = {"symbol": ticker.upper(), "limit": NUM_QUARTERS, "period": "quarter"} # Capped at 5 quarters
        params["apikey"] = FMP_KEY
        req = FMP_URL + FMP_ENDPOINTS[0]
        response = requests.get(req, params = params).json()

        time.sleep(6)

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
        params = {"symbol": ticker.upper(), "limit": 10, "apikey": FMP_KEY}
        req = FMP_URL + FMP_ENDPOINTS[1]
        response = requests.get(req, params = params).json()
        
        time.sleep(6)

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


def filter_summaries() -> None:
    company_terms = {"NVDA": ["nvidia", "nvda", "jensen huang", "gpu"],
                     "AAPL": ["apple", "aapl", "tim cook", "john ternus", "iphone"],
                     "MSFT": ["microsoft", "msft", "satya nadella", "azure"],
                     "META": ["meta", "llama", "mark zuckerberg"],
                     "AMZN": ["amazon", "amzn", "andy jassy", "jeff bezos", "aws"]}

    summaries_by_ticker = json.loads(
        SUMMARIES_OUTPUT_PATH.read_text(encoding = "utf-8")
    )

    # Go through each ticker
    for ticker, ticker_data in summaries_by_ticker.items():
        terms = company_terms[ticker]
        filtered_dates = {}
        previous_summaries = []

        # Build every date from oldest to newest so missing dates can be forwarded
        for day_offset in range(181+31): # 31 days buffer for first month forwarding
            date = (SUMMARY_START_DATE + dt.timedelta(days = day_offset)).isoformat()
            summaries = ticker_data.get(date, [])
            matching_summaries = []

            # Check each summary against keywords and remove URLs
            for summary in summaries:
                if len(matching_summaries) >= 3: # 3 summaries per date
                    break

                cleaned_summary = URL_PATTERN.sub("", summary).strip()
                if not 30 < len(cleaned_summary) < 600:
                    continue
                summary_lower = cleaned_summary.lower()

                for term in terms:
                    if term in summary_lower:
                        if cleaned_summary not in matching_summaries:
                            matching_summaries.append(cleaned_summary)
                        break

            current_summaries = matching_summaries.copy()
            forwarded_count = 0

            # Forward previous valid data if not enough made it through filters
            for previous_summary in previous_summaries:
                if len(matching_summaries) >= 3:
                    break
                if previous_summary not in matching_summaries:
                    matching_summaries.append(previous_summary)
                    forwarded_count += 1

            if forwarded_count:
                print(f"Forwarded {forwarded_count} articles for date {date}, ticker {ticker}")

            filtered_dates[date] = matching_summaries
            previous_summaries = matching_summaries

        summaries_by_ticker[ticker] = {
            "summaries": dict(sorted(filtered_dates.items(), reverse = True))
        }

    FILTERED_SUMMARIES_OUTPUT_PATH.write_text(
        json.dumps(summaries_by_ticker),
        encoding = "utf-8",
    )
    print(f"Filtered summaries written to {FILTERED_SUMMARIES_OUTPUT_PATH}")

def produce_training_batch() -> None:
    """
    Produce self-contained daily training data for each ticker
    Balance sheets and historical grades are forwarded until newer data exists
    """
    filtered_summaries = json.loads(
        FILTERED_SUMMARIES_OUTPUT_PATH.read_text(encoding = "utf-8")
    )
    balance_sheets = json.loads(
        BALANCE_SHEETS_OUTPUT_PATH.read_text(encoding = "utf-8")
    )
    historical_grades = json.loads(
        HISTORICAL_GRADES_OUTPUT_PATH.read_text(encoding = "utf-8")
    )

    training_batch = {}

    for ticker in TICKERS:
        training_batch[ticker] = {}
        ticker_summaries = filtered_summaries[ticker]["summaries"]
        ticker_balance_sheets = balance_sheets[ticker]
        ticker_historical_grades = historical_grades[ticker]

        # Get starting balance sheet
        current_balance_sheet = ticker_balance_sheets[max(
            date for date in ticker_balance_sheets
            if date <= START_DATE.isoformat()
        )]
        # Get starting historical grade
        current_historical_grades = ticker_historical_grades[max(
            date for date in ticker_historical_grades
            if date <= START_DATE.isoformat()
        )]

        for day_offset in range(181):
            date = (START_DATE + dt.timedelta(days = day_offset)).isoformat()

            # Update latest balance sheet as it becomes available
            if date in ticker_balance_sheets:
                current_balance_sheet = ticker_balance_sheets[date]
            # Update latest historical grade as it becomes available
            if date in ticker_historical_grades:
                current_historical_grades = ticker_historical_grades[date]

            training_batch[ticker][date] = {
                "filtered_summaries": ticker_summaries[date],
                "balance_sheet": current_balance_sheet,
                "historical_grades": current_historical_grades,
            }

    TRAINING_BATCH_OUTPUT_PATH.write_text(
        json.dumps(training_batch),
        encoding = "utf-8",
    )
    print(f"Training batch written to {TRAINING_BATCH_OUTPUT_PATH}")


def main():
    if COLLECTION_STATES["summaries"] == "Y":
        summaries_by_ticker = collect_summaries()
        SUMMARIES_OUTPUT_PATH.write_text(
            json.dumps(summaries_by_ticker),
            encoding = "utf-8",
        )
        print(f"Saved summaries to {SUMMARIES_OUTPUT_PATH}")

    if COLLECTION_STATES["balance_sheets"] == "Y":
        balance_sheets = collect_balance_sheets()
        BALANCE_SHEETS_OUTPUT_PATH.write_text(
            json.dumps(balance_sheets),
            encoding = "utf-8",
        )
        print(f"Saved balance sheets to {BALANCE_SHEETS_OUTPUT_PATH}")

    if COLLECTION_STATES["historical_grades"] == "Y":
        historical_grades = collect_historical_grades()
        HISTORICAL_GRADES_OUTPUT_PATH.write_text(
            json.dumps(historical_grades),
            encoding = "utf-8",
        )
        print(f"Saved historical grades to {HISTORICAL_GRADES_OUTPUT_PATH}")

    if COLLECTION_STATES["filter_summaries"] == "Y":
        filter_summaries()

    if COLLECTION_STATES["produce_training_batch"] == "Y":
        produce_training_batch()


if __name__ == "__main__":
    main()
