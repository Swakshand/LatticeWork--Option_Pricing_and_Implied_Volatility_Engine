"""
test_pricing.py
================

Pytest unit tests for the pure numerical core in :mod:`pricing`. Deliberately
has **no network dependency** (unlike the live market-data routes in
``app.py``/``market_data.py``), so it runs deterministically anywhere,
including CI.

Run with:

    pytest -v

Covers:
  * Put-call parity (a model-free, no-arbitrage identity).
  * Binomial -> Black-Scholes convergence.
  * The two American-exercise theorems (call == European; put > European).
  * Internal consistency between the fast (`crr_price`) and full-lattice
    (`crr_tree`) binomial implementations.
  * The implied-volatility solver: round-trip recovery across a range of
    moneyness/vol/maturity combinations, and correct rejection of prices that
    violate no-arbitrage bounds.
"""

from __future__ import annotations

from math import exp

import pytest

import pricing

S, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.20, 1.0





@pytest.mark.parametrize("option_type", ["call", "put"])
def test_binomial_converges_to_black_scholes(option_type):
    bs = pricing.black_scholes(S, K, R, SIGMA, T, option_type)
    price = pricing.crr_price(S, K, R, SIGMA, T, 500, option_type, "european")
    assert abs(price - bs.price) < 0.05


def test_crr_tree_matches_crr_price():
    """The slow, full-lattice tree and the fast vectorised pricer must agree."""
    tree = pricing.crr_tree(S, K, R, SIGMA, T, 50, "call", "european")
    fast = pricing.crr_price(S, K, R, SIGMA, T, 50, "call", "european")
    assert abs(tree.price - fast) < 1e-9


def test_put_call_parity():
    call = pricing.black_scholes(S, K, R, SIGMA, T, "call")
    put = pricing.black_scholes(S, K, R, SIGMA, T, "put")
    assert abs((call.price - put.price) - (S - K * exp(-R * T))) < 1e-8





def test_american_call_equals_european_call_without_dividends():
    american = pricing.crr_price(S, K, R, SIGMA, T, 500, "call", "american")
    european = pricing.crr_price(S, K, R, SIGMA, T, 500, "call", "european")
    assert abs(american - european) < 1e-3


def test_american_put_exceeds_european_put():
    american = pricing.crr_price(S, K, R, SIGMA, T, 500, "put", "american")
    european = pricing.crr_price(S, K, R, SIGMA, T, 500, "put", "european")
    assert american > european + 1e-4


@pytest.mark.parametrize("N", [1, 2, 5, 25])
def test_no_early_exercise_nodes_flagged_for_calls(N):
    """A call on a non-dividend stock should never show early exercise."""
    tree = pricing.crr_tree(S, K, R, SIGMA, T, N, "call", "american")
    flagged = [
        tree.exercise_tree[i][j]
        for i in range(N)
        for j in range(i + 1)
    ]
    assert not any(flagged)





@pytest.mark.parametrize(
    "s, k, r, sigma, t, option_type",
    [
        (100, 100, 0.05, 0.20, 1.0, "call"),
        (100, 100, 0.05, 0.20, 1.0, "put"),
        (100, 130, 0.03, 0.35, 0.5, "call"),
        (100, 70, 0.03, 0.15, 2.0, "put"),
        (100, 100, 0.01, 0.05, 0.05, "call"),
        (50, 55, 0.04, 0.60, 1.5, "put"),
        (200, 150, 0.02, 0.80, 0.1, "call"),
    ],
)
def test_implied_volatility_round_trip(s, k, r, sigma, t, option_type):
    """Price with a known sigma, then solve for it back out from the price."""
    price = pricing.black_scholes(s, k, r, sigma, t, option_type).price
    result = pricing.implied_volatility(price, s, k, r, t, option_type)
    assert result.sigma is not None
    assert result.converged
    assert abs(result.sigma - sigma) < 1e-4


def test_implied_volatility_rejects_price_below_intrinsic():
    """A price that violates the model-free no-arbitrage lower bound has no solution."""
    result = pricing.implied_volatility(
        target_price=0.01, S=100, K=50, r=0.05, T=1.0, option_type="call"
    )
    assert result.sigma is None
    assert result.method == "failed"


def test_implied_volatility_rejects_price_above_spot():
    """A call can never be worth more than the stock itself."""
    result = pricing.implied_volatility(
        target_price=150, S=100, K=100, r=0.05, T=1.0, option_type="call"
    )
    assert result.sigma is None
    assert result.method == "failed"


def test_implied_volatility_falls_back_to_bisection_when_vega_is_tiny():
    """Deep out-of-the-money options have ~0 Vega; Newton should hand off cleanly."""
    price = pricing.black_scholes(100, 300, 0.03, 0.9, 0.05, "call").price
    result = pricing.implied_volatility(price, 100, 300, 0.03, 0.05, "call")
    assert result.sigma is not None
    assert abs(result.sigma - 0.9) < 1e-3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
