"""
Contains tests for the ML pipeline functionality.
From backend/, run:
    PYTHONPATH=. ETF_Opt/bin/python -m pytest -q ml_engine/tests.py -m "not integration"
        - For non-integration tests (no API calls)
    PYTHONPATH=. ETF_Opt/bin/python -m pytest -q ml_engine/tests.py -m integration
        - For only integration tests (only API calls)
    PYTHONPATH=. ETF_Opt/bin/python -m pytest -q ml_engine/tests.py
        - For comprehensive testing
"""

import pytest as t
import json
import datetime as dt

from ml_engine.sandbox_ml_models import run_backtesting_inference, run_direct_ticker_inference
from ml_engine.exceptions import userInputError
from ml_engine.live_inference import run_live_inference, MIN_POOL_LOW_BOUND, MAX_POOL_UPPER_BOUND
from ml_engine.batch_collection import (
    TRAINING_FILE_OUTPUT_PATH,
    NUMERICAL_DATA_OUTPUT_PATHS,
    START_DATE,
    END_DATE,
)
from ml_engine.model_orchestrator import MODELS

TRAINING_FILE = TRAINING_FILE_OUTPUT_PATH # Current training file to test


"""
----------------------------------Live inference tests for price predictions and volatility----------------------------------
***Some tests make API calls***
User input parameters: 
{
    "horizon_days": int,
    "sectors": dict,
    "sizes": list,
    "blacklisted": list,
    "min_pool": int,
    "max_pool": int,
}
"""
def test_invalid_horizon():
    with t.raises(userInputError): run_live_inference({
    "horizon_days": 25,
    "sectors": {
        "technology": 1,
        "financial": 0.33,
    },
    "sizes": ["big-cap"],
    "blacklisted": [],
    "min_pool": 5,
    "max_pool": 30,
})
        
def test_invalid_sectors():
    with t.raises(userInputError): run_live_inference({
    "horizon_days": 20,
    "sectors": {
        "nothing": 1,
    },
    "sizes": ["big-cap"],
    "blacklisted": [],
    "min_pool": 5,
    "max_pool": 30,
})
        
def test_empty_sectors():
    with t.raises(userInputError): run_live_inference({
    "horizon_days": 20,
    "sectors": {},
    "sizes": ["big-cap"],
    "blacklisted": [],
    "min_pool": 5,
    "max_pool": 30,
})
        
def test_invalid_sector_partition_all():
    with t.raises(userInputError): run_live_inference({
    "horizon_days": 20,
    "sectors": {
        "technology": 1,
        "financial": 1.5,
        "energy": -1.2
    },
    "sizes": ["small-cap", "big-cap"],
    "blacklisted": [],
    "min_pool": MIN_POOL_LOW_BOUND,
    "max_pool": MAX_POOL_UPPER_BOUND,
})
        
def test_invalid_sector_partition_ones_under():
    with t.raises(userInputError): run_live_inference({
    "horizon_days": 20,
    "sectors": {
        "technology": 0.6,
        "financial": 0.4
    },
    "sizes": ["small-cap", "big-cap"],
    "blacklisted": [],
    "min_pool": MIN_POOL_LOW_BOUND,
    "max_pool": MAX_POOL_UPPER_BOUND,
})
        
def test_invalid_sector_partition_ones_over():
    with t.raises(userInputError): run_live_inference({
    "horizon_days": 20,
    "sectors": {
        "technology": 1,
        "financial": 1,
    },
    "sizes": ["small-cap", "big-cap"],
    "blacklisted": [],
    "min_pool": MIN_POOL_LOW_BOUND,
    "max_pool": MAX_POOL_UPPER_BOUND,
})
        
def test_invalid_sector_partition_negative():
    with t.raises(userInputError): run_live_inference({
    "horizon_days": 20,
    "sectors": {
        "technology": 1,
        "financial": -0.2
    },
    "sizes": ["small-cap", "big-cap"],
    "blacklisted": [],
    "min_pool": MIN_POOL_LOW_BOUND,
    "max_pool": MAX_POOL_UPPER_BOUND,
})
        
def test_invalid_sector_partition_over_two():
    with t.raises(userInputError): run_live_inference({
    "horizon_days": 20,
    "sectors": {
        "technology": 1,
        "financial": 0.6,
        "energy": 0.7
    },
    "sizes": ["small-cap", "big-cap"],
    "blacklisted": [],
    "min_pool": MIN_POOL_LOW_BOUND,
    "max_pool": MAX_POOL_UPPER_BOUND,
})
        
def test_invalid_sector_partition_granularity_lower_bound():
    with t.raises(userInputError): run_live_inference({
    "horizon_days": 20,
    "sectors": {
        "technology": 1,
        "financial": (0.99 / MAX_POOL_UPPER_BOUND),
    },
    "sizes": ["small-cap", "big-cap"],
    "blacklisted": [],
    "min_pool": MIN_POOL_LOW_BOUND,
    "max_pool": MAX_POOL_UPPER_BOUND,
})
        
def test_invalid_sector_partition_granularity_upper_bound():
    with t.raises(userInputError): run_live_inference({
    "horizon_days": 20,
    "sectors": {
        "technology": 1,
        "financial": 1 - (0.99 / MAX_POOL_UPPER_BOUND),
    },
    "sizes": ["small-cap", "big-cap"],
    "blacklisted": [],
    "min_pool": MIN_POOL_LOW_BOUND,
    "max_pool": MAX_POOL_UPPER_BOUND,
})
        
def test_invalid_sector_partition_granularity_total_upper_bound():
    with t.raises(userInputError): run_live_inference({
    "horizon_days": 20,
    "sectors": {
        "technology": 1,
        "financial": ((1 - 0.99 / MAX_POOL_UPPER_BOUND) / 2),
        "energy": ((1 - 0.99 / MAX_POOL_UPPER_BOUND) / 2)
    },
    "sizes": ["small-cap", "big-cap"],
    "blacklisted": [],
    "min_pool": MIN_POOL_LOW_BOUND,
    "max_pool": MAX_POOL_UPPER_BOUND,
})
        
def test_invalid_sizes():
    with t.raises(userInputError): run_live_inference({
    "horizon_days": 20,
    "sectors": {
        "technology": 1,
        "financial": 0.33,
    },
    "sizes": ["nothing"],
    "blacklisted": [],
    "min_pool": 5,
    "max_pool": 30,
})
        
def test_empty_sizes():
    with t.raises(userInputError): run_live_inference({
    "horizon_days": 20,
    "sectors": {
        "technology": 1,
        "financial": 0.33,
    },
    "sizes": [],
    "blacklisted": [],
    "min_pool": 5,
    "max_pool": 30,
})
        
def test_swapped_bounds():
    with t.raises(userInputError): run_live_inference({
    "horizon_days": 20,
    "sectors": {
        "technology": 1,
        "financial": 0.33,
    },
    "sizes": ["small-cap", "big-cap"],
    "blacklisted": [],
    "min_pool": 10,
    "max_pool": 5,
})
        
def test_invalid_lower_bound():
    with t.raises(userInputError): run_live_inference({
    "horizon_days": 20,
    "sectors": {
        "technology": 1,
        "financial": 0.33,
    },
    "sizes": ["small-cap", "big-cap"],
    "blacklisted": [],
    "min_pool": MIN_POOL_LOW_BOUND - 1,
    "max_pool": MAX_POOL_UPPER_BOUND,
})
        
def test_invalid_upper_bound():
    with t.raises(userInputError): run_live_inference({
    "horizon_days": 20,
    "sectors": {
        "technology": 1,
        "financial": 0.33,
    },
    "sizes": ["small-cap", "big-cap"],
    "blacklisted": [],
    "min_pool": MIN_POOL_LOW_BOUND,
    "max_pool": MAX_POOL_UPPER_BOUND + 1,
})
        
# Live API call to yfinance and GCP
@t.mark.integration
def test_validate_output_live():
    size = len(run_live_inference({
    "horizon_days": 20,
    "sectors": {
        "technology": 1,
        "financial": 0.33,
    },
    "sizes": ["big-cap"],
    "blacklisted": [],
    "min_pool": MIN_POOL_LOW_BOUND,
    "max_pool": MIN_POOL_LOW_BOUND + 1,
}))
    assert size <= MAX_POOL_UPPER_BOUND # Min is not guaranteed


"""     
----------------------------------ML model invariant tests (backtesting)----------------------------------
"""

def test_gemini_aggregate_scores():
    with open(TRAINING_FILE, "r", encoding = "utf-8") as f:
        data = json.load(f)
        for row in data:
            assert -100 < row["gemini_sentiment_score"] < 100

def test_prediction_horizons():
    with open(TRAINING_FILE, "r", encoding = "utf-8") as f:
        data = json.load(f)
        for row in data:
            assert row["prediction_horizon_days"] in MODELS

def test_batch_dates():
    with open(TRAINING_FILE, "r", encoding = "utf-8") as file:
        data = json.load(file)
        for row in data:
            assert START_DATE <= dt.date.fromisoformat(row["date"]) < END_DATE

def test_future_return_outcomes_match_numerical_source_data():
    """Training return outcomes must match historical source file price data"""
    prices_by_ticker = {}
    for file_path in NUMERICAL_DATA_OUTPUT_PATHS.values():
        with open(file_path, "r", encoding = "utf-8") as source_file:
            for source_row in json.load(source_file):
                prices_by_ticker.setdefault(source_row["ticker"], {})[
                    dt.date.fromisoformat(source_row["date"])
                ] = float(source_row["adjusted_close"])

    with open(TRAINING_FILE, "r", encoding = "utf-8") as training_file:
        training_rows = {
            (row["ticker"], dt.date.fromisoformat(row["date"]), row["prediction_horizon_days"]): row
            for row in json.load(training_file)
        }

    final_date = END_DATE + dt.timedelta(days = max(MODELS))
    for ticker, trading_prices in prices_by_ticker.items():
        calendar_prices = {}
        latest_price = None # Keep track of latest trading day price to lookup forward-filled date in training file
        current_date = min(trading_prices)

        while current_date <= final_date:
            latest_price = trading_prices.get(current_date, latest_price)
            calendar_prices[current_date] = latest_price
            current_date += dt.timedelta(days=1)

        current_date = START_DATE
        while current_date < END_DATE:
            for horizon_days in MODELS:
                row = training_rows.get((ticker, current_date, horizon_days))
                if row: # Some non-trading days dropped over shorter horizons, and Gemini response integrity
                    expected_return = (
                        calendar_prices[current_date + dt.timedelta(days=horizon_days)]
                        / calendar_prices[current_date]) - 1
                    assert row["future_return_outcome"] == t.approx(expected_return, rel = 1e-12, abs = 1e-12)
            current_date += dt.timedelta(days = 1)
