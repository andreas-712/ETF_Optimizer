'''
Collects market data for tickers
Currently uses Financial Modeling Prep (FMP) API for market news
'''

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
    Fetches daily market data times. Standardized to EST
    """
    print(f"Fetching numerical data from {start_date} to {end_date} for: {tickers}")

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
    """
    Returns a parsed dict for direct extraction for inference
    """
    parsed_responses_dict = {}

    # 1. Latest Balance Sheet Data (iloc[0] grabs most recent / leftmost col)
    balance_sheet = responses[0]

    parsed_responses_dict["Cash and short term investments"] = balance_sheet.loc[
        "Cash Cash Equivalents And Short Term Investments"
    ].iloc[0]
    parsed_responses_dict["Total current assets"] = balance_sheet.loc[
        "Current Assets"
    ].iloc[0]
    parsed_responses_dict["Total liabilities and total equity"] = balance_sheet.loc[
        "Total Liabilities Net Minority Interest"
    ].iloc[0]
    parsed_responses_dict["Total debt"] = balance_sheet.loc["Total Debt"].iloc[0]

    # 2. Latest analyst ratings
    analyst_ratings = responses[1]

    # Filter for the current live period (0m) and convert to a dict
    current_ratings = analyst_ratings[analyst_ratings["period"] == "0m"].iloc[0]

    parsed_responses_dict["Analyst strong buys"] = current_ratings["strongBuy"]
    parsed_responses_dict["Analyst buys"] = current_ratings["buy"]
    parsed_responses_dict["Analyst holds"] = current_ratings["hold"]
    parsed_responses_dict["Analyst sells"] = current_ratings["sell"]
    parsed_responses_dict["Analyst strong sells"] = current_ratings["strongSell"]

    # 3. Latest stock news
    parsed_responses_dict["News summaries"] = [
        "Publisher:" + article["publisher"] + "." + article["title"] for article in responses[2][:8]
    ]

    return parsed_responses_dict


def _binary_search_index(dates: list, target: str) -> int | None:
    """
    Returns index of most recent index up to target date. Assumes newest to oldest order
    """
    l, r = 0, len(dates)-1
    dt_target = dt.datetime.strptime(target, "%Y-%m-%d")
    result = None

    while l <= r:
        mid = (l+r) // 2
        dt_mid = dt.datetime.strptime(dates[mid], "%Y-%m-%d")

        if dt_mid <= dt_target:
            result = mid
            r = mid - 1
        else:
            l = mid + 1

    return result
