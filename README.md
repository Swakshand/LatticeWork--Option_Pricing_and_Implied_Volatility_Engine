# Latticework

A beautiful, interactive Flask dashboard that teaches **risk-neutral pricing**
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

![lattice + convergence dashboard](https://img.shields.io/badge/Flask-dashboard-6ea8fe)
![tests](https://github.com/OWNER/REPO/actions/workflows/tests.yml/badge.svg)

---

## Why this project exists

Almost every derivatives-pricing idea a quant researcher uses — martingale
measures, no-arbitrage, replication, PDE/expectation duality — can be seen in
miniature in a two-state (binomial) world. The binomial tree is the smallest
model where "the price is a *discounted expected payoff under a special
probability measure*" becomes concrete and computable. Once you internalise it,
Black–Scholes is just the continuous-time limit.

---

## The mathematics

### 1. No-arbitrage and the risk-neutral measure

Consider one time step in which the stock either goes up to \(S u\) or down to
\(S d\). Any option payoff over that step can be **replicated** by a portfolio
of \(\Delta\) shares of stock plus a cash position in the risk-free bond.
Because two portfolios with identical future payoffs must cost the same today
(otherwise there is an arbitrage), the option's price equals the cost of the
replicating portfolio.

Working through the algebra, that replication cost can be rewritten as a
**discounted expectation** — but under a synthetic probability, not the real
one:

$$
V_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\!\big[\text{payoff}(S_T)\big].
$$

Under the *risk-neutral measure* \(\mathbb{Q}\), every asset earns the risk-free
rate on average. Crucially, the **real-world probability of an up move never
appears** in the option price. Only volatility (which sets \(u\) and \(d\)) and
the risk-free rate matter.

### 2. The CRR parameterisation

Split maturity \(T\) into \(N\) steps of length \(\Delta t = T/N\). Cox, Ross
and Rubinstein chose the up/down factors to match the stock's volatility:

$$
u = e^{\sigma\sqrt{\Delta t}}, \qquad
d = \frac{1}{u} = e^{-\sigma\sqrt{\Delta t}}.
$$

The condition \(u\,d = 1\) makes the tree **recombine**: an up-then-down move
lands back where you started. So after \(i\) steps there are only \(i+1\)
distinct prices,

$$
S_i^{\,j} = S_0\, u^{\,j} d^{\,i-j}, \qquad j = 0,1,\dots,i,
$$

instead of \(2^i\). This is what keeps the model tractable.

The **risk-neutral up-probability** is pinned down by requiring the (discounted)
stock price to be a martingale — i.e. the stock earns exactly \(r\) on average:

$$
p\,u + (1-p)\,d = e^{r\Delta t}
\quad\Longrightarrow\quad
p = \frac{e^{r\Delta t} - d}{u - d}.
$$

For the model to be arbitrage-free we need \(d < e^{r\Delta t} < u\), which
guarantees \(0 < p < 1\).

### 3. Backward induction

Set the option's value at the terminal nodes to its payoff, then roll backwards.
At each node the value is the discounted risk-neutral expectation of its two
children:

$$
V_i^{\,j} = e^{-r\Delta t}\Big(p\,V_{i+1}^{\,j+1} + (1-p)\,V_{i+1}^{\,j}\Big).
$$

Repeating this from the leaves to the root gives today's price \(V_0^0\). Each
local step *is* the one-period risk-neutral pricing formula; chaining them is
just the law of iterated expectations.

### 4. American early exercise

An American option can be exercised at any node. So at every node we compare the
**continuation value** (holding) against the **intrinsic value** (exercising
now) and keep the larger:

$$
V_i^{\,j} = \max\!\Big(
\text{payoff}(S_i^{\,j}),\;
e^{-r\Delta t}\big(p\,V_{i+1}^{\,j+1} + (1-p)\,V_{i+1}^{\,j}\big)
\Big).
$$

Two classic results fall out of this and are checked in the smoke test:

- An **American call on a non-dividend stock equals the European call** — it is
  never optimal to exercise early (you would throw away time value and the
  interest on the strike).
- An **American put is worth strictly more** than the European put, because
  early exercise can be optimal when the option is deep in the money.

The dashboard rings the nodes where early exercise is optimal in red.

### 5. Convergence to Black–Scholes, and the sawtooth

As \(N \to \infty\), the CRR European price converges to the Black–Scholes
closed form. The error shrinks like \(\mathcal{O}(1/N)\), but it does **not**
decrease monotonically — it **oscillates**. This is the "money shot" of the
dashboard.

**Why the sawtooth?** For a fixed strike \(K\), the pricing error is dominated by
how the \(N+1\) discrete terminal nodes straddle the strike. As \(N\) increases
by one, the grid of terminal prices shifts, and whether a node lands just above
or just below \(K\) flips between "even \(N\)" and "odd \(N\)". That flips how
much probability mass sits just in- vs. just out-of-the-money near the money,
producing an even/odd oscillation whose amplitude decays as the grid gets finer.
The Black–Scholes value is the smooth limit that the sawtooth wraps around.

### 6. Implied volatility, and why the "smile" matters

Black–Scholes price is a **strictly increasing** function of \(\sigma\) (its
Vega is always \(\ge 0\)), so for any price inside the model's no-arbitrage
bounds there is exactly one \(\sigma\) that reproduces it. **Implied
volatility** is that \(\sigma\), solved *backwards* from a real, quoted
market price \(V_{\text{mkt}}\) instead of assumed up front:

$$
\sigma_{\text{implied}} : \quad \text{BS}(S, K, r, \sigma_{\text{implied}}, T) = V_{\text{mkt}}.
$$

This is implemented in `pricing.implied_volatility` as:

1. **Reject impossible prices.** Any price outside the model-free
   no-arbitrage bounds — \([\max(S - Ke^{-rT}, 0),\ S]\) for a call — cannot
   come from *any* \(\sigma\); it's rejected outright rather than fed to a
   root-finder that would return a meaningless answer.
2. **Newton–Raphson**, using the model's own analytic Vega as the local
   slope: \(\sigma \leftarrow \sigma - \dfrac{\text{BS}(\sigma) -
   V_{\text{mkt}}}{\text{Vega}(\sigma)}\). This converges in a handful of
   iterations when Vega is well-behaved.
3. **Bisection fallback.** Deep in/out-of-the-money options have Vega close
   to zero, which makes Newton's step numerically unstable (or pushes the
   estimate outside a sane volatility range). When that happens the solver
   falls back to bisection on a bracketed interval — slower, but guaranteed
   to converge.

Convergence in both methods is judged by how much the **volatility estimate
itself** stops changing between iterations, not by the raw dollar price
residual — this matters because deep out-of-the-money, short-dated options
can have real prices far smaller than any sensible fixed price tolerance
(fractions of a cent), which would otherwise make a price-based check
declare false convergence on the very first guess.

**Why plot it against strike?** If Black–Scholes' constant-\(\sigma\)
assumption were literally true, every strike's implied volatility would come
back identical — a flat line. Real option chains never look like that: the
curve traced out (the **volatility smile/skew**) is direct, visible evidence
that the market prices tail risk (deep in/out-of-the-money outcomes)
differently than a single flat \(\sigma\) would predict. The dashboard's
panel 4 fetches a real, live option chain (via `yfinance`) and plots exactly
this curve, computed by the solver above, alongside the underlying market
data (bid/ask/last price, volume, open interest) so every number is
traceable back to its source.

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

> **Note:** this feature requires internet access at request time (the Flask
> server calls out to Yahoo Finance via `yfinance` when you click "Fetch live
> data"). Everything else in the dashboard works fully offline.

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

---

## How to run

From a PowerShell prompt in this folder:

```powershell
# 1. (optional) create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. install dependencies
pip install -r requirements.txt

# 3. run the headless smoke test (no server left running)
python smoke_test.py

# 4. (optional) run the pytest unit-test suite
pytest -v

# 5. launch the dashboard
python app.py
```

Then open <http://127.0.0.1:5002> in your browser.

The dashboard prices an at-the-money 1-year call on load. Adjust the inputs in
the left panel and click **Price option** to re-price and redraw everything.

> **Tip:** set the number of steps to a small value (e.g. `N = 5`) to see an
> individual lattice clearly, and push the "Convergence up to N" slider to ~200
> to watch the sawtooth settle onto the Black–Scholes line.

> **Tip:** in panel 4, type a real ticker (e.g. `AAPL`, `MSFT`, `SPY`) and
> click **"Fetch live data"** — it pulls the real spot price and historical
> volatility into the inputs above, and plots the live implied-volatility
> smile below. Requires internet access; everything else works offline.

---

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

*Educational use only — not investment advice.*
