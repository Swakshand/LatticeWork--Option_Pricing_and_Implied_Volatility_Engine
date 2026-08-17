"""
smoke_test.py
=============

Headless sanity checks for Latticework. Run with:

    python smoke_test.py

Checks performed:
  1. The binomial *European* price converges to Black-Scholes as N grows
     (|diff| < 0.05 at N = 500 for a standard at-the-money case).
  2. American call on a non-dividend stock equals the European call (a known
     no-early-exercise result), while an American put is worth strictly more
     than its European counterpart.
  3. Put-call parity holds for the Black-Scholes prices.
  4. The implied-volatility solver recovers a known sigma from the price it
     itself produced (a "round trip" check).
  5. The Flask app boots and GET / returns HTTP 200. No server is left running.

For a more thorough, network-free unit test suite (edge cases, parametrised
implied-vol round trips across many strikes/maturities, etc.), see
``test_pricing.py`` (run with ``pytest``). This script stays intentionally
small and dependency-light so it can be run directly with plain Python.
"""

from __future__ import annotations

import pricing


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ->  {detail}" if detail else ""))
    return condition


def main() -> int:
    ok = True
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0


    bs = pricing.black_scholes(S, K, r, sigma, T, "call")
    print(f"\nBlack-Scholes European call: {bs.price:.6f}\n")
    print("  N      binomial       |diff|")
    prev_diff = None
    for N in (1, 5, 25, 100, 500):
        price = pricing.crr_price(S, K, r, sigma, T, N, "call", "european")
        diff = abs(price - bs.price)
        print(f"  {N:<5}  {price:>12.6f}  {diff:>10.6f}")
        if N == 500:
            prev_diff = diff
    ok &= check(
        "European binomial call -> Black-Scholes at N=500",
        prev_diff is not None and prev_diff < 0.05,
        f"|diff| = {prev_diff:.6f} < 0.05",
    )


    bs_put = pricing.black_scholes(S, K, r, sigma, T, "put")
    put_500 = pricing.crr_price(S, K, r, sigma, T, 500, "put", "european")
    ok &= check(
        "European binomial put -> Black-Scholes at N=500",
        abs(put_500 - bs_put.price) < 0.05,
        f"|diff| = {abs(put_500 - bs_put.price):.6f}",
    )


    am_call = pricing.crr_price(S, K, r, sigma, T, 500, "call", "american")
    eu_call = pricing.crr_price(S, K, r, sigma, T, 500, "call", "european")
    ok &= check(
        "American call == European call (no dividends)",
        abs(am_call - eu_call) < 1e-3,
        f"american={am_call:.6f}, european={eu_call:.6f}",
    )

    am_put = pricing.crr_price(S, K, r, sigma, T, 500, "put", "american")
    ok &= check(
        "American put > European put",
        am_put > put_500 + 1e-4,
        f"american={am_put:.6f} > european={put_500:.6f}",
    )


    from math import exp

    parity_lhs = bs.price - bs_put.price
    parity_rhs = S - K * exp(-r * T)
    ok &= check(
        "Put-call parity: C - P == S - K e^{-rT}",
        abs(parity_lhs - parity_rhs) < 1e-8,
        f"{parity_lhs:.6f} == {parity_rhs:.6f}",
    )


    known_sigma = 0.35
    mkt_price = pricing.black_scholes(S, K, r, known_sigma, T, "call").price
    iv = pricing.implied_volatility(mkt_price, S, K, r, T, "call")
    ok &= check(
        "Implied vol solver recovers a known sigma from its own price",
        iv.sigma is not None and abs(iv.sigma - known_sigma) < 1e-4,
        f"recovered={iv.sigma}, expected={known_sigma}, method={iv.method}",
    )


    tree = pricing.crr_tree(S, K, r, sigma, T, 50, "call", "european")
    fast = pricing.crr_price(S, K, r, sigma, T, 50, "call", "european")
    ok &= check(
        "crr_tree root value matches crr_price",
        abs(tree.price - fast) < 1e-9,
        f"tree={tree.price:.6f}, fast={fast:.6f}",
    )


    import app as flask_app

    client = flask_app.app.test_client()
    resp = client.get("/")
    ok &= check("GET / returns 200", resp.status_code == 200, f"status={resp.status_code}")

    api = client.post(
        "/api/price",
        json=dict(S=S, K=K, r=r, sigma=sigma, T=T, N=5, option_type="call", exercise="european"),
    )
    ok &= check("POST /api/price returns 200", api.status_code == 200, f"status={api.status_code}")
    if api.status_code == 200:
        body = api.get_json()
        ok &= check(
            "API payload has figures + prices",
            all(k in body for k in ("binom_price", "bs_price", "lattice_fig", "conv_fig")),
        )

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
