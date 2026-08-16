"""
math_engine/ETF_bucket.py

Orchestration layer -- NOT part of the pure math. This is where I/O happens
(calling ml_engine.run_live_inference, calling yfinance for price history);
math_engine.Optimizer and math_engine.Covariance stay pure and are only
called into from here.

ml_inputs must match the contract ml_engine.live_inference.run_live_inference
expects (see main.py.shape_user_inputs for how the CLI builds it):

    {
        "horizon_days": int,
        "sectors": {"technology": 0.4, "energy": 1.0, ...},  # cap fractions, 1.0 = uncapped "fill" sector
        "sizes": ["big-cap", "mid-cap", "small-cap"],
        "blacklisted": ["XOM", ...],
        "min_pool": int,
        "max_pool": int,
    }

run_live_inference already handles candidate screening (sector/size/blacklist)
and return + volatility inference, and is synchronous (it manages its own
asyncio internally), so none of this needs to be async either.
"""

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np

from ml_engine.live_inference import run_live_inference
from ml_engine.market_data_collection import fetch_numerical_ticker_data
from ml_engine.exceptions import userInputError

from math_engine.Optimizer import optimize_weights
from math_engine.Covariance import build_covariance_matrix

RISK_MULTIPLIERS = {
    "conservative": 0.7,
    "moderate": 1.0,
    "aggressive": 1.4,
}

COVARIANCE_HISTORY_DAYS = 182  # ~6 trading months for the covariance estimate


def get_optimized_etf_bucket(
    ml_inputs: dict,
    risk_tolerance: str,
    max_single_weight: float | None = None,
) -> dict[str, dict]:
    """
    Screens candidates and predicts return/volatility per ticker (ml_engine),
    then solves for minimum-variance weights hitting a risk-scaled target
    return (math_engine). Returns one record per selected ticker:
    {ticker: {weight, return, volatility, horizon_days, sector, market_cap_category}}
    """
    if risk_tolerance not in RISK_MULTIPLIERS:
        raise userInputError(
            f"Unknown risk tolerance {risk_tolerance!r}; expected one of: {', '.join(RISK_MULTIPLIERS)}"
        )

    predictions = run_live_inference(ml_inputs)
    if not predictions:
        return {}

    tickers = list(predictions)

    now_date = dt.datetime.now(ZoneInfo("America/New_York")).date()
    start_date = now_date - dt.timedelta(days=COVARIANCE_HISTORY_DAYS)
    numerical_data = fetch_numerical_ticker_data(
        tickers,
        start_date.strftime("%Y-%m-%d"),
        now_date.strftime("%Y-%m-%d"),
    )
    if numerical_data.empty:
        raise userInputError("No historical price data available to compute a covariance matrix")

    wide_prices = numerical_data.pivot(index="date", columns="ticker", values="adjusted_close").sort_index()

    # Drop any tickers price history failed to return -- keeps predictions/mu/Sigma aligned
    final_tickers = [t for t in tickers if t in wide_prices.columns]
    if not final_tickers:
        raise userInputError("No tickers had enough historical price data to compute covariance")
    predictions = {t: predictions[t] for t in final_tickers}

    mu = np.array([predictions[t]["return"] for t in final_tickers])
    Sigma = build_covariance_matrix(wide_prices, final_tickers, annualize=True)

    target_return = float(np.mean(mu) * RISK_MULTIPLIERS[risk_tolerance])

    # get_ticker_pool already shapes the candidate pool by these sector caps;
    # re-applying them here also caps the *weighted* sector exposure, not just
    # the candidate count. cap >= 1 is the uncapped "fill" sentinel.
    sector_caps = ml_inputs.get("sectors", {})
    group_caps = []
    for sector, cap in sector_caps.items():
        if cap >= 1:
            continue
        idx = [i for i, t in enumerate(final_tickers) if predictions[t]["sector"] == sector]
        if idx:
            group_caps.append((idx, cap))

    try:
        weights = optimize_weights(
            mu,
            Sigma,
            target_return,
            long_only=True,
            max_weight=max_single_weight,
            group_caps=group_caps or None,
        )
    except ValueError:
        # target_return unreachable under the given caps -- relax to the best
        # return achievable using the unscaled mean of mu instead, then retry once.
        weights = optimize_weights(
            mu,
            Sigma,
            float(np.mean(mu)),
            long_only=True,
            max_weight=max_single_weight,
            group_caps=group_caps or None,
        )

    return {
        ticker: {**predictions[ticker], "weight": weight}
        for ticker, weight in zip(final_tickers, weights.tolist())
    }
