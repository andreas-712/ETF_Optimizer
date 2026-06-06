import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def _as_series(column_data):
    if isinstance(column_data, pd.DataFrame):
        return column_data.iloc[:, 0]
    return column_data

def fetch_ticker_data(tickers: list, lookback_years: int) -> pd.DataFrame:
    # Fetches daily market data times. are standardized to EST
    print(f"Fetching {lookback_years} years of data for: {tickers}")

    # Use NY time
    market_timezone = ZoneInfo("America/New_York")
    ny_today = datetime.now(market_timezone)

    # Calculate lookback window
    end_date = ny_today.strftime('%Y-%m-%d')
    start_date = (ny_today - timedelta(days = lookback_years * 365)).strftime('%Y-%m-%d')

    compiled_records = []

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

            # Reformat
            adjusted_close_column = "Adj Close" if "Adj Close" in raw_yf_df.columns else "Close"

            formatted_df = pd.DataFrame({
                'date': pd.to_datetime(_as_series(raw_yf_df['Date'])).dt.date,
                'ticker': symbol,
                'adjusted_close': _as_series(raw_yf_df[adjusted_close_column]).astype(float),
                'volume': _as_series(raw_yf_df['Volume']).astype(int)
            })

            compiled_records.append(formatted_df)
            print(f"Successfully fetched {len(formatted_df)} rows for {symbol}")

        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")

    if compiled_records:
        raw_market_matrix = pd.concat(compiled_records, ignore_index = True)
        return raw_market_matrix # Filled dataframe

    return pd.DataFrame() # Empty data frame on failure
