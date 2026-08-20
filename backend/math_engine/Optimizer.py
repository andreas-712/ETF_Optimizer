import numpy as np
from scipy.optimize import minimize


def _min_variance_weights_closed_form(mu: np.ndarray, Sigma: np.ndarray, target_return: float) -> np.ndarray:
    """
    Inputs :
    mu: 1D array of expected return assets, shape (n,)
    Sigma: covariance matrix of asset returns, shape (n, n)
    target_return: what the client wants out of it
    Returns:
    1D array of portfolio weights, of shape (n,)
    """

    Sigma_inv = np.linalg.inv(Sigma)
    ones = np.ones(len(mu))
    A = ones @ Sigma_inv @ ones
    B = ones @ Sigma_inv @ mu
    C = mu @ Sigma_inv @ mu
    det = A * C - B**2
    with np.errstate(divide="ignore", invalid="ignore"):
        lam1 = (C-B * target_return) / det
        lam2 = (A * target_return - B) / det
    return lam1 * (Sigma_inv @ ones) + lam2 * (Sigma_inv @ mu)

def _min_variance_weights_constrained(mu: np.ndarray, Sigma: np.ndarray, target_return: float, long_only: bool = True, max_weight: float | None=None, group_caps: list[tuple[list[int], float]] | None=None) -> np.ndarray:
    """
    some inputs are the same as the closed form
    long_only: If true, every weight is bounded below by 0(no shorting)
    max_weight: optional upper bound on all weights
    group_caps: list of (indices, max group weights) pairs. The indices refer to the positions in mu sigma. Like for example a limit of 0.3 in tech 
    """
    n = len(mu)

    # A single asset leaves no free variable: sum(w) = 1 already forces w = [1.0],
    # which makes the target-return equality redundant. SLSQP can't handle that --
    # it always rejects 2 equality constraints against 1 variable, even when they're
    # consistent -- so resolve it directly instead of routing through the solver.
    if n == 1:
        lower = 0.0 if long_only else -1.0
        if lower > 1.0 + 1e-8 or (max_weight is not None and max_weight < 1.0 - 1e-8):
            raise ValueError("Constrained optimization infeasible: bounds exclude the only feasible weight (1.0) for a single asset")
        if not np.isclose(mu[0], target_return, atol=1e-6):
            raise ValueError(
                f"Constrained optimization infeasible: single remaining asset return ({mu[0]}) cannot hit target_return ({target_return})"
            )
        if group_caps and any(0 in idx and cap < 1.0 - 1e-8 for idx, cap in group_caps):
            raise ValueError("Constrained optimization infeasible: a group cap excludes the only feasible weight (1.0) for a single asset")
        return np.array([1.0])

    def portfolio_variance(w: np.ndarray) -> float:
        return float(w @ Sigma @ w)
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: w @ mu - target_return},
    ]
    if group_caps:
        for indices, cap in group_caps:
            constraints.append({"type":"ineq", "fun": lambda w, idx=indices, cap=cap: cap-np.sum(w[idx])})
    lower = 0.0 if long_only else -1.0
    bounds = [(lower, max_weight) for _ in range(n)]
    w0 = np.ones(n)/n
    result = minimize(portfolio_variance, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError(f"Constrained optimization failed: {result.message}")
    return result.x
def optimize_weights(
    mu: np.ndarray,
    Sigma: np.ndarray,
    target_return: float,
    long_only: bool = True,
    max_weight: float | None = None,
    group_caps: list[tuple[list[int], float]] | None = None,
) -> np.ndarray:
    """
 
    Solves closed-form first since it's fast and exact, then checks whether
    the result actually violates the constraints you asked for. Only reaches
    for the slower iterative solver if it has to.
 
    Inputs / returns: same as min_variance_weights_constrained.
    """
    w = _min_variance_weights_closed_form(mu, Sigma, target_return)

    
    violates_budget = not np.all(np.isfinite(w)) or not np.isclose(np.sum(w), 1.0, atol=1e-6)
    violates_target = not np.isclose(w @ mu, target_return, atol=1e-6) if np.all(np.isfinite(w)) else True
    violates_long_only = long_only and np.any(w < -1e-8)
    violates_cap = max_weight is not None and np.any(w > max_weight + 1e-8)
    violates_group = group_caps is not None and any(
        np.sum(w[idx]) > cap + 1e-8 for idx, cap in group_caps
    )

    if violates_budget or violates_target or violates_long_only or violates_cap or violates_group:
        w = _min_variance_weights_constrained(
            mu, Sigma, target_return, long_only=long_only, max_weight=max_weight, group_caps=group_caps
        )

    return w

def risk_contributions(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Decomposes total portfolio volatility into a per-asset marginal contribution,
    via the standard Euler decomposition: each asset's contribution is
    (Sigma @ w)_i / portfolio_vol, chosen so that sum(w_i * contribution_i) ==
    portfolio_vol == sqrt(w @ Sigma @ w) exactly (not an approximation).

    This is what callers should report as "this asset's volatility" when the
    number needs to roll up linearly to the true portfolio volatility -- a
    weighted sum of assets' standalone volatilities does NOT equal portfolio
    volatility except in the degenerate case of perfect correlation.

    Inputs:
    w: portfolio weights, shape (n,)
    Sigma: covariance matrix, shape (n, n)
    Returns:
    1D array of per-asset marginal risk contributions, shape (n,)
    """
    portfolio_vol = float(np.sqrt(w @ Sigma @ w))
    if portfolio_vol <= 0:
        return np.zeros_like(w)
    return (Sigma @ w) / portfolio_vol
