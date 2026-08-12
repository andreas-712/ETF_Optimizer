"""
This file contains functions for collecting market data:
    - Fetching daily numerical ticker data from yfinance
    - Fetching and parsing live ticker inputs for Gemini inference
"""

import yfinance as yf
import pandas as pd
import datetime as dt


def _as_series(column_data):
    """
    Standardizes yf data to a series column
    """
    if isinstance(column_data, pd.DataFrame):
        return column_data.iloc[:, 0]
    return column_data


def fetch_numerical_ticker_data(
    tickers: list,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Fetches daily market data standardized to EST.
    Batches requests in parallel.
    """
    print(f"Fetching numerical data from {start_date} to {end_date} for: {tickers}")

    raw_yf_df = yf.download(
        tickers,
        start = start_date,
        end = end_date,
        progress = False,
        auto_adjust = False,
        group_by = "ticker",
        threads = True,
    )

    if raw_yf_df.empty:
        print("No numerical data returned")
        return pd.DataFrame()

    compiled_records = []
    for symbol in tickers:
        try:
            symbol_df = raw_yf_df[symbol].reset_index()

            if symbol_df.empty:
                print(f"No data returned for {symbol}")
                continue
            
            # Add in adjusted close column
            adjusted_close_column = "Adj Close" if "Adj Close" in symbol_df.columns else "Close"

            # Reformat
            formatted_df = pd.DataFrame({
                "date": pd.to_datetime(_as_series(symbol_df["Date"])).dt.date,
                "ticker": symbol,
                "adjusted_close": _as_series(symbol_df[adjusted_close_column]).astype(float),
                "volume": _as_series(symbol_df["Volume"]).astype(int)
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


def fetch_ticker_gemini_inputs(ticker: str) -> dict:
    """
    Returns the latest balance sheet, analyst recommendations, and news for a ticker
    """
    ticker_data = yf.Ticker(ticker)
    responses = []

    responses.append(
        ticker_data.get_balance_sheet(
            as_dict = False,
            pretty = False,
            freq = "quarterly"
        )
    )
    responses.append(ticker_data.get_recommendations(as_dict = False))
    responses.append(ticker_data.get_news(count = 8, tab = "news"))

    return _parse_responses(responses)


def _parse_responses(responses: list) -> dict:
    """Returns a parsed dict for direct extraction for inference."""
    def value_or_nan(field: str, getter):
        """Fills NaN for any failed metric retrieval"""
        try:
            return getter()
        except KeyError:
            print(f"Missing live Gemini input field: {field}")
            return "NaN"

    parsed_responses_dict = {}

    # 1. Latest Balance Sheet Data (iloc[0] grabs most recent / leftmost col)
    balance_sheet = responses[0]

    parsed_responses_dict["Cash and short term investments"] = value_or_nan(
        "Cash and short term investments",
        lambda: balance_sheet.loc["CashCashEquivalentsAndShortTermInvestments"].iloc[0],
    )
    parsed_responses_dict["Total current assets"] = value_or_nan(
        "Total current assets",
        lambda: balance_sheet.loc["CurrentAssets"].iloc[0],
    )
    parsed_responses_dict["Total liabilities and total equity"] = value_or_nan(
        "Total liabilities and total equity",
        lambda: balance_sheet.loc["TotalLiabilitiesNetMinorityInterest"].iloc[0],
    )
    parsed_responses_dict["Total debt"] = value_or_nan(
        "Total debt",
        lambda: balance_sheet.loc["TotalDebt"].iloc[0],
    )

    # 2. Latest analyst ratings
    analyst_ratings = responses[1]
    try:
        current_ratings = analyst_ratings[analyst_ratings["period"] == "0m"].iloc[0]
    except KeyError:
        print("Missing live Gemini input field: Analyst ratings")
        current_ratings = {}

    parsed_responses_dict["Analyst strong buys"] = value_or_nan(
        "Analyst strong buys",
        lambda: current_ratings["strongBuy"],
    )
    parsed_responses_dict["Analyst buys"] = value_or_nan(
        "Analyst buys",
        lambda: current_ratings["buy"],
    )
    parsed_responses_dict["Analyst holds"] = value_or_nan(
        "Analyst holds",
        lambda: current_ratings["hold"],
    )
    parsed_responses_dict["Analyst sells"] = value_or_nan(
        "Analyst sells",
        lambda: current_ratings["sell"],
    )
    parsed_responses_dict["Analyst strong sells"] = value_or_nan(
        "Analyst strong sells",
        lambda: current_ratings["strongSell"],
    )

    # 3. Latest stock news
    parsed_responses_dict["News summaries"] = value_or_nan(
        "News summaries",
        lambda: [
            "Publisher:"
            + article["content"]["provider"]["displayName"]
            + "."
            + article["content"]["title"]
            for article in responses[2][:8]
        ],
    )

    return parsed_responses_dict

