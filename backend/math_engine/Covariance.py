import numpy as np
import pandas as pd


def build_covariance_matrix(
    wide_prices: pd.DataFrame,
    tickers: list[str],
    annualize: bool = True,
) -> np.ndarray:
    """
    Inputs:
        wide_prices: DataFrame indexed by date, one column per ticker
            (adjusted close price).
        tickers: order to return Sigma in. This MUST match the order used
            to build `mu` elsewhere -- the optimizer assumes index i in mu
            and Sigma refer to the same asset.
        annualize: if True, scales daily variance up to annual (x252
            trading days). Keep this consistent with whatever units
            `target_return` and mu are expressed in -- if mu is an annual
            predicted return, Sigma needs to be annualized too, or the two
            won't be on the same scale.

    Returns:
        Sigma: covariance matrix, shape (len(tickers), len(tickers))
    """
    prices = wide_prices[tickers]
    daily_returns = prices.pct_change().dropna()
    Sigma = daily_returns.cov().to_numpy()

    if annualize:
        Sigma = Sigma * 252

    return Sigma