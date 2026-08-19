# Latticework: An Option Pricing Platform

An interactive Flask dashboard that teaches **risk-neutral pricing**
from the ground up. It builds the Cox–Ross–Rubinstein (CRR) binomial tree,
prices European and American vanilla options by backward induction, and shows —
visually — how the binomial price **converges to Black–Scholes** as the number
of steps grows. This is the conceptual on-ramp to no-arbitrage / risk-neutral
valuation.

It also connects that theory to **real markets**: a Newton–Raphson implied-
volatility solver inverts live option-chain quotes (via `yfinance`) to plot
the **volatility smile/skew** — direct, visible evidence that the constant-
volatility assumption behind Black–Scholes and the CRR tree doesn't hold in
practice.

---

## Real market data

Panel 4 of the dashboard ("Real market data") connects the theory above to
an actual, live market:

- **Historical (realised) volatility** — `market_data.get_historical_stats`
  pulls a year of daily closes for a ticker and computes the annualised
  standard deviation of daily log returns: the textbook estimator, with no
  GARCH or exponential weighting, so every step is easy to reproduce by
  hand. Clicking **"Fetch live data"** pre-fills the pricer's spot and
  volatility inputs with these real numbers instead of guesses.
- **A live option chain** — `market_data.get_option_chain` fetches every
  listed strike for a chosen expiry and reduces it to a "market price" per
  strike: the bid/ask **midpoint** when both are quoted and positive (an
  actually-executable price), falling back to the **last traded price**
  otherwise (common for illiquid, far out-of-the-money strikes). Quotes with
  no usable price at all are dropped rather than guessed at.
- **Implied volatility per strike**, computed by *our own* solver above (not
  a data vendor's built-in column), plotted as the smile.

Real market data is messy by nature — stale quotes, zero bid/ask, prices
that momentarily violate the model's no-arbitrage bounds because the
underlying has moved since the quote was last updated. Rather than silently
producing a garbage number, every stage reports *why* a row was dropped
(`n_used` / `n_dropped` / `n_iv_failed` in the API response and the panel's
summary line), and `market_data.MarketDataError` surfaces a clear message
for network failures, invalid tickers, or expiries with no listed options.

---

## Model assumptions & limitations

Every number this dashboard produces is only as good as the assumptions
behind it. Being explicit about where each model is known to break matters
as much as the model itself:

- **Constant volatility.** Both the CRR tree and Black–Scholes assume one
  \(\sigma\) for the option's entire life. The volatility smile above is
  live, visible proof this is false for real markets — implied volatility
  varies by strike (and, in reality, by expiry too — the full "volatility
  surface").
- **No dividends.** The underlying is assumed to pay nothing before expiry.
  With dividends, the "American call = European call" result checked in
  `smoke_test.py`/`test_pricing.py` can actually break — early exercise can
  become optimal just before an ex-dividend date.
- **Frictionless markets.** No transaction costs, no bid-ask spread,
  unlimited borrowing/lending at one risk-free rate, and infinite liquidity
  at every price. The real option chain in panel 4 shows actual bid/ask
  spreads, volume, and open interest — exactly the frictions this model
  ignores.
- **Continuous, costless hedging (Black–Scholes specifically).** The
  closed-form price assumes the replicating portfolio can be rebalanced
  continuously and for free. The CRR tree only rebalances at `N` discrete
  steps, which is part of why it differs from Black–Scholes at finite `N`
  (the sawtooth in section 5).
- **Historical vs. implied volatility are different things.** "Fetch live
  data" fills \(\sigma\) with *historical* (backward-looking, realised)
  volatility, while the smile shows *implied* (forward-looking,
  market-priced) volatility at each strike. They are rarely equal —
  comparing the two is itself a simple, real volatility-trading signal.

---

## What's in the box

| File | Purpose |
| --- | --- |
| `pricing.py` | Pure pricing engine: CRR tree (fast + full-lattice variants), Black–Scholes with greeks, implied-volatility solver (Newton–Raphson + bisection fallback), convergence series. Heavily documented, zero network dependency. |
| `market_data.py` | Thin `yfinance` wrapper: historical volatility, live spot price, and option-chain fetching/cleaning. The only module allowed to touch the network; raises a single clear `MarketDataError` on any failure. |
| `app.py` | Flask server (port **5002**). Builds Plotly figures server-side and serves them as JSON to the front-end, including the new live market-data routes. |
| `templates/index.html` | Single-page dashboard with MathJax-rendered LaTeX. |
| `static/style.css` | "Research notebook" dark theme. |
| `static/app.js` | Front-end controller: gathers inputs, calls the API, renders results. |
| `smoke_test.py` | Headless correctness + boot checks (run directly with plain Python, no test framework). |
| `test_pricing.py` | `pytest` unit-test suite for the pure pricing/greeks/implied-vol logic — parametrised edge cases, no network dependency, runs in CI. |
| `.github/workflows/tests.yml` | GitHub Actions CI: runs `test_pricing.py` and `smoke_test.py` on every push/PR. |
| `requirements.txt` | `flask`, `numpy`, `scipy`, `plotly`, `yfinance`, `pytest`. |

### The four visualisations

1. **The recombining lattice** — drawn as a node graph (step on the x-axis, net
   up-moves on the y-axis). Nodes are coloured by option value; hover to see the
   asset price and option value at each node. American early-exercise nodes get
   a red ring.
2. **Convergence chart** — the binomial price as a function of \(N\), oscillating
   in a decaying sawtooth around the flat Black–Scholes benchmark line.
3. **Risk-neutral intuition panel** — displays \(\Delta t, u, d, p\), the step
   and total discount factors, and shows explicitly that the *discounted
   expected payoff under \(\mathbb{Q}\)* reproduces the price.
4. **Volatility smile** — implied volatility (solved by our own Newton–Raphson
   root-finder) plotted against strike for a real, live option chain, with a
   per-strike table showing exactly where each market price came from.


## Verifying correctness

Two complementary layers of checks:

**`smoke_test.py`** (plain Python, no framework — run it directly) asserts,
among other things, that for the standard case
\(S=K=100,\ r=5\%,\ \sigma=20\%,\ T=1\):

- the European binomial call price agrees with Black–Scholes to
  \(|\text{diff}| < 0.05\) at \(N = 500\);
- the American call equals the European call (no dividends);
- the American put exceeds the European put;
- Black–Scholes put–call parity holds;
- the implied-volatility solver recovers a known \(\sigma\) from the price it
  itself produced (a "round trip" check);
- the Flask app boots and both `GET /` and `POST /api/price` return HTTP 200.

**`test_pricing.py`** (`pytest`, run in CI on every push via
`.github/workflows/tests.yml`) covers the same properties more thoroughly and
with no network dependency: put-call parity, binomial→Black–Scholes
convergence, both American-exercise theorems, fast/full-lattice consistency,
and — most extensively — the implied-volatility solver, parametrised across
seven very different moneyness/volatility/maturity combinations (including a
deep out-of-the-money, short-dated case specifically chosen to stress-test
numerical underflow), plus explicit tests that impossible (arbitrage-violating)
prices are correctly rejected rather than silently mishandled.

---
