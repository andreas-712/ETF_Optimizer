'''
Collects market data for tickers
Currently uses Financial Modeling Prep (FMP) API for market news
'''

import os
from pathlib import Path
import yfinance as yf
import pandas as pd
import datetime as dt
from zoneinfo import ZoneInfo
import requests
from dotenv import load_dotenv
from ml_engine.configs import DATA_MODE, LIVE_MODE
import math
from collections import defaultdict

BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".flaskenv")

FMP_KEY = os.getenv("FMP_KEY")
FMP_URL = os.getenv("FINANCIAL_URL")
FMP_ENDPOINTS = {}


def _as_series(column_data):
    """
    Standardizes yf data to a series column
    """
    if isinstance(column_data, pd.DataFrame):
        return column_data.iloc[:, 0]
    return column_data


def fetch_ticker_data(tickers: list, lookback_years: int) -> pd.DataFrame:
    """
    Fetches daily market data times. Standardized to EST
    """
    print(f"Fetching {lookback_years} years of data for: {tickers}")

    # Use NY time
    market_timezone = ZoneInfo("America/New_York")
    ny_today = dt.datetime.now(market_timezone)

    # Calculate lookback window
    end_date = ny_today.strftime('%Y-%m-%d')
    start_date = (ny_today - dt.timedelta(days = lookback_years * 365)).strftime('%Y-%m-%d')

    compiled_records = []

    # Download market data for given ticker
    for symbol in tickers:
        try:
            raw_yf_df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
            )

            if raw_yf_df.empty:
                print(f"No data returned for {symbol}")
                continue
            
            raw_yf_df = raw_yf_df.reset_index()

            # Add in adjusted close column
            adjusted_close_column = "Adj Close" if "Adj Close" in raw_yf_df.columns else "Close"

            # Reformat
            formatted_df = pd.DataFrame({
                'date': pd.to_datetime(_as_series(raw_yf_df['Date'])).dt.date,
                'ticker': symbol,
                'adjusted_close': _as_series(raw_yf_df[adjusted_close_column]).astype(float),
                'volume': _as_series(raw_yf_df['Volume']).astype(int)
            })

            # Append ticker with data to df
            compiled_records.append(formatted_df)
            print(f"Successfully fetched {len(formatted_df)} rows for {symbol}")

        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")

    if compiled_records:
        raw_market_matrix = pd.concat(compiled_records, ignore_index = True)
        return raw_market_matrix # Filled dataframe

    return pd.DataFrame() # Empty data frame on failure


def fetch_ticker_summaries(ticker: str, horizon_days: int, cutoff_date: str) -> list:
    """
    Returns executive summaries for articles about the given ticker.
    """
    cutoff = pd.to_datetime(cutoff_date).date()
    start_date = cutoff - dt.timedelta(days=horizon_days)
    parsed_data_text = ""
    responses = []

    if DATA_MODE == LIVE_MODE:
        limit = 1
    else:
        limit = _get_fmp_limit(cutoff_date)

    for i in range(len(FMP_ENDPOINTS)): # 3 APIs for FMP
        if i == 0:
            # Balance sheet
            params = {"symbol": ticker, "limit": limit, "period": "quarter"} # limit = number of quarters
        if i == 1:
            # Financial growth
            params = {"symbol": ticker, "limit": limit, "period": "quarter"} # limit = number of quarters
        if i == 2:
            # Historical grades
            params = {"symbol": ticker, "limit": limit * 15 if not DATA_MODE == LIVE_MODE else 1} # Estimate reports at 15 per quarter

        endpoint = FMP_ENDPOINTS[i]
        responses.append(requests.get(endpoint, params = params).json())

    # Fetch stock news data
    if DATA_MODE == LIVE_MODE:
        responses.append(_fetch_live_fundamentals(ticker))
    else:
        responses.append(_fetch_historical_fundamentals(ticker, cutoff_date))

    parsed_data_text = _parse_responses(responses, cutoff_date)

    return parsed_data_text


def _fetch_live_fundamentals(ticker: str) -> dict:
    """
    Fetches fundamentals for live inference using using yfinance API
    """

def _fetch_historical_fundamentals(ticker: str, cutoff_date: str) -> dict:
    """
    Fetches historical fundamentals for backtesting using Finnhub API
    """

def _get_fmp_limit(cutoff_date_str: str) -> int:
    """
    Calculates the limit parameter for querying quarterly statements for backtesting
    """
    now = dt.datetime.now(dt.timezone.utc).date()
    cutoff_date = dt.datetime.strptime(cutoff_date_str, "%Y-%m-%d").date()

    delta_days = (now - cutoff_date).days
    if delta_days <= 0:
        print(f"Error, returned {delta_days} days")
        return 1 # Default to latest report
    
    # Get the latest n reports
    days_per_quarter = 91.25
    limit = math.ceil(delta_days / days_per_quarter) + 1

    return limit


def _binary_search_index(dates: list, target: str) -> int:
    """
    Returns index of most recent index up to target date
    """
    l, r = 0, len(dates)-1
    dt_target = dt.datetime.strptime(target, "%Y-%m-%d")

    while l < r:
        mid = (l+r) // 2
        dt_mid = dt.datetime.strptime(dates[mid], "%Y-%m-%d")

        if (dt_target - dt_mid).days > 0:
            r = mid
        else:
            l = mid + 1

    return l


def _parse_responses(responses: list, cutoff_date: str) -> dict:
    """
    Returns a parsed, prompt-ready dict for inference
    """
    parsed_responses_dict = defaultdict(str)

    # For backtesting, -1 indexes oldest (valid) entry
    # For live testing, -1 indexes the only entry

    # 1. Add balance sheet data
    balance_sheet = responses[0][-1]
    parsed_responses_dict["Cash and short term investments"] = balance_sheet["cashAndShortTermInvestments"]
    parsed_responses_dict["Total current assets"] = balance_sheet["totalCurrentAssets"]
    parsed_responses_dict["Total liabilities and total equity"] = balance_sheet["totalLiabilitiesAndTotalEquity"]
    parsed_responses_dict["Total debt"] = balance_sheet["totalDebt"]

    # 2. Add financial growth data
    financial_statement = responses[1][-1]
    parsed_responses_dict["YoY revenue growth"] = f"{financial_statement["revenueGrowth"] * 100}%"
    parsed_responses_dict["YoY EPS growth"] = f"{financial_statement["epsgrowth"] * 100}%"
    parsed_responses_dict["YoY debt growth"] = f"{financial_statement["debtGrowth"] * 100}%"
    parsed_responses_dict["Net income growth"] = f"{financial_statement["netIncomeGrowth"] * 100}%"

    # 3. Latest historical ratings
    analyst_rating_dates = [rating["date"] for rating in responses[2]]
    analyst_ratings = responses[2][
        _binary_search_index(analyst_rating_dates, cutoff_date)
    ]
    parsed_responses_dict["Analyst strong buys"] = analyst_ratings["analystRatingsStrongBuy"]
    parsed_responses_dict["Analyst buys"] = analyst_ratings["analystRatingsBuy"]
    parsed_responses_dict["Analyst holds"] = analyst_ratings["analystRatingsHold"]
    parsed_responses_dict["Analyst sells"] = analyst_ratings["analystRatingsSell"]
    parsed_responses_dict["Analyst strong sells"] = analyst_ratings["analystRatingsStrongSell"]

    # 4. Latest stock news
    parsed_responses_dict.update(responses[3]) # Up to 3 headlines in an array

    return parsed_responses_dict
