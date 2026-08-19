from __future__ import annotations
from dataclasses import dataclass, field
from math import exp, log, pi, sqrt
from typing import Literal
import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]

@dataclass
class BinomialResult:
    #Full output of a binomial valuation, including the lattices.
    price: float
    dt: float
    u: float
    d: float
    p: float
    disc: float
    n_steps: int

    asset_tree: list[list[float]] = field(default_factory=list)
    value_tree: list[list[float]] = field(default_factory=list)
    exercise_tree: list[list[bool]] = field(default_factory=list)


@dataclass
class BlackScholesResult:
    #Closed-form European price plus the standard greeks.
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    d1: float
    d2: float


@dataclass
class ImpliedVolResult:
    #Output of the implied-volatility solver.
    sigma: float | None
    iterations: int
    converged: bool
    method: Literal["newton", "bisection", "failed"]
    error: str | None = None


def black_scholes(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: OptionType = "call",
) -> BlackScholesResult:
    #Price a European option with the Black-Scholes-Merton formula.
    if T <= 0 or sigma <= 0:

        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return BlackScholesResult(intrinsic, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * sqrt(T))
            - r * K * exp(-r * T) * norm.cdf(d2)
        )
        rho = K * T * exp(-r * T) * norm.cdf(d2)
    else:
        price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * sqrt(T))
            + r * K * exp(-r * T) * norm.cdf(-d2)
        )
        rho = -K * T * exp(-r * T) * norm.cdf(-d2)

    gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
    vega = S * norm.pdf(d1) * sqrt(T)

    return BlackScholesResult(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        d1=d1,
        d2=d2,
    )


def implied_volatility(
    target_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: OptionType = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> ImpliedVolResult:
    #Back out the volatility that makes `black_scholes` reproduce a quoted market price.
    if T <= 0:
        return ImpliedVolResult(None, 0, False, "failed", error="T must be positive.")
    if S <= 0 or K <= 0:
        return ImpliedVolResult(None, 0, False, "failed", error="S and K must be positive.")

    discounted_K = K * exp(-r * T)
    if option_type == "call":
        lower_bound = max(S - discounted_K, 0.0)
        upper_bound = S
    else:
        lower_bound = max(discounted_K - S, 0.0)
        upper_bound = discounted_K

    if target_price < lower_bound - 1e-8 or target_price > upper_bound + 1e-8:
        return ImpliedVolResult(
            None, 0, False, "failed",
            error=(
                f"Price {target_price:.4f} is outside the no-arbitrage bounds "
                f"[{lower_bound:.4f}, {upper_bound:.4f}] for these S, K, r, T -- "
                "no volatility could ever produce it (likely a stale/bad quote)."
            ),
        )

    sigma = sqrt(2 * pi / T) * (target_price / S)
    sigma = min(max(sigma, 0.02), 3.0)


    for i in range(max_iter):
        bs = black_scholes(S, K, r, sigma, T, option_type)
        if bs.vega < 1e-8:
            break
        step = (bs.price - target_price) / bs.vega
        if abs(step) < tol:
            return ImpliedVolResult(sigma - step, i + 1, True, "newton")
        sigma_next = sigma - step
        if not (1e-4 < sigma_next < 5.0):
            break
        sigma = sigma_next


    lo, hi = 1e-4, 5.0
    f_lo = black_scholes(S, K, r, lo, T, option_type).price - target_price
    f_hi = black_scholes(S, K, r, hi, T, option_type).price - target_price
    if f_lo * f_hi > 0:
        return ImpliedVolResult(
            None, max_iter, False, "failed",
            error="Could not bracket a solution between sigma=0.01% and 500%.",
        )

    for i in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = black_scholes(S, K, r, mid, T, option_type).price - target_price
        if (hi - lo) < tol:
            return ImpliedVolResult(mid, i + 1, True, "bisection")
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid

    return ImpliedVolResult(0.5 * (lo + hi), max_iter, False, "bisection", error="Max iterations reached.")


def _payoff(spots: np.ndarray, K: float, option_type: OptionType) -> np.ndarray:
    #Vanilla payoff evaluated element-wise on an array of spot prices.
    if option_type == "call":
        return np.maximum(spots - K, 0.0)
    return np.maximum(K - spots, 0.0)


def crr_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int,
    option_type: OptionType = "call",
    exercise: ExerciseStyle = "european",
) -> float:
    #Fast CRR price (no lattice retained) using vectorised backward induction.
    if N < 1:
        raise ValueError("N must be a positive integer.")

    dt = T / N
    u = exp(sigma * sqrt(dt))
    d = 1.0 / u
    disc = exp(-r * dt)
    p = (exp(r * dt) - d) / (u - d)

    j = np.arange(N + 1)
    spots = S * (u**j) * (d ** (N - j))
    values = _payoff(spots, K, option_type)

    for i in range(N - 1, -1, -1):

        values = disc * (p * values[1 : i + 2] + (1 - p) * values[0 : i + 1])
        if exercise == "american":
            spots = S * (u ** np.arange(i + 1)) * (d ** (np.arange(i + 1)[::-1]))
            values = np.maximum(values, _payoff(spots, K, option_type))

    return float(values[0])


def crr_tree(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int,
    option_type: OptionType = "call",
    exercise: ExerciseStyle = "european",
) -> BinomialResult:
    #CRR price **with** the full asset and option-value lattices retained.
    if N < 1:
        raise ValueError("N must be a positive integer.")

    dt = T / N
    u = exp(sigma * sqrt(dt))
    d = 1.0 / u
    disc = exp(-r * dt)
    p = (exp(r * dt) - d) / (u - d)

    asset_tree: list[list[float]] = []
    for i in range(N + 1):
        level = [S * (u**j) * (d ** (i - j)) for j in range(i + 1)]
        asset_tree.append(level)


    value_tree: list[list[float]] = [[0.0] * (i + 1) for i in range(N + 1)]
    exercise_tree: list[list[bool]] = [[False] * (i + 1) for i in range(N + 1)]

    for j in range(N + 1):
        payoff = _intrinsic(asset_tree[N][j], K, option_type)
        value_tree[N][j] = payoff

        exercise_tree[N][j] = payoff > 0.0

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            continuation = disc * (
                p * value_tree[i + 1][j + 1] + (1 - p) * value_tree[i + 1][j]
            )
            if exercise == "american":
                intrinsic = _intrinsic(asset_tree[i][j], K, option_type)
                if intrinsic > continuation:
                    value_tree[i][j] = intrinsic
                    exercise_tree[i][j] = True
                else:
                    value_tree[i][j] = continuation
            else:
                value_tree[i][j] = continuation

    return BinomialResult(
        price=value_tree[0][0],
        dt=dt,
        u=u,
        d=d,
        p=p,
        disc=disc,
        n_steps=N,
        asset_tree=asset_tree,
        value_tree=value_tree,
        exercise_tree=exercise_tree,
    )


def _intrinsic(spot: float, K: float, option_type: OptionType) -> float:
    #Scalar vanilla payoff (helper for the node-by-node backward pass).
    if option_type == "call":
        return max(spot - K, 0.0)
    return max(K - spot, 0.0)


def convergence_series(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: OptionType = "call",
    exercise: ExerciseStyle = "european",
    n_max: int = 200,
) -> tuple[list[int], list[float]]:
    #Binomial price as a function of the number of steps `N`. Returns `(steps, prices)` where `steps = [1, 2, ..., n_max]`. Plotting `prices` against `steps` 
    #produces the characteristic *sawtooth* that oscillates around and converges to the Black-Scholes value.

    steps = list(range(1, n_max + 1))
    prices = [
        crr_price(S, K, r, sigma, T, n, option_type, exercise) for n in steps
    ]
    return steps, prices


if __name__ == "__main__":

    params = dict(S=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)
    bs = black_scholes(option_type="call", **params)
    binom = crr_price(N=500, option_type="call", exercise="european", **params)
    print(f"Black-Scholes call : {bs.price:.6f}")
    print(f"Binomial (N=500)   : {binom:.6f}")
    print(f"|difference|       : {abs(bs.price - binom):.6f}")
    
