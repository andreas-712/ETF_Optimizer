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
from ml_engine.sandbox_ml_models import run_backtesting_inference, run_direct_ticker_inference
from ml_engine.exceptions import userInputError
from ml_engine.live_inference import run_live_inference, MIN_POOL_LOW_BOUND, MAX_POOL_UPPER_BOUND


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

