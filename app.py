"""
app.py
======

Flask dashboard for *Latticework*.

The server does all of the numerical heavy lifting (in :mod:`pricing`) and
constructs Plotly figures as JSON, which the front-end renders with Plotly.js.
Keeping figure construction on the server means the browser only ever handles
presentation, and the interesting mathematics lives in one documented place.

Run with:

    python app.py

then open http://127.0.0.1:5002 in a browser.
"""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, jsonify, render_template, request

import market_data
import pricing

app = Flask(__name__)






PALETTE = {
    "bg": "#0f1420",
    "panel": "#151b2b",
    "grid": "#26304a",
    "text": "#e6ecff",
    "muted": "#8a97b8",
    "accent": "#6ea8fe",
    "accent2": "#f5a97f",
    "up": "#63d2a4",
    "down": "#f28fad",
    "exercise": "#f7768e",
}


def _base_layout(title: str) -> dict:
    """Common Plotly layout so every chart shares the dashboard aesthetic."""
    return dict(
        title=dict(text=title, font=dict(size=16, color=PALETTE["text"])),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], family="Inter, system-ui, sans-serif"),
        margin=dict(l=60, r=30, t=50, b=50),
        hoverlabel=dict(
            bgcolor=PALETTE["panel"],
            bordercolor=PALETTE["grid"],
            font=dict(color=PALETTE["text"]),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=PALETTE["muted"]),
        ),
    )





def build_lattice_figure(res: pricing.BinomialResult, option_type: str) -> go.Figure:
    """Draw the recombining CRR lattice as a node-graph.

    Node ``(i, j)`` (``i`` = step, ``j`` = up-moves) is placed at::

        x = i            (time / step index)
        y = 2*j - i      (net up-moves; keeps the tree symmetric & recombining)

    Edges connect each node to its up and down children. Nodes are coloured by
    option value; nodes where early exercise is optimal get a distinct ring.
    """
    N = res.n_steps


    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for i in range(N):
        for j in range(i + 1):
            x0, y0 = i, 2 * j - i

            for jj in (j + 1, j):
                x1, y1 = i + 1, 2 * jj - (i + 1)
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color=PALETTE["grid"], width=1),
        hoverinfo="skip",
        showlegend=False,
    )


    node_x, node_y = [], []
    colors, texts, ring_x, ring_y = [], [], [], []
    for i in range(N + 1):
        for j in range(i + 1):
            x, y = i, 2 * j - i
            node_x.append(x)
            node_y.append(y)
            asset = res.asset_tree[i][j]
            value = res.value_tree[i][j]
            colors.append(value)
            texts.append(
                f"<b>Step {i}</b> &nbsp; up-moves {j}<br>"
                f"Asset price&nbsp;&nbsp;S = {asset:,.4f}<br>"
                f"Option value&nbsp; V = {value:,.4f}"
                + (
                    "<br><b>early exercise optimal</b>"
                    if res.exercise_tree[i][j] and i < N
                    else ""
                )
            )
            if res.exercise_tree[i][j] and i < N:
                ring_x.append(x)
                ring_y.append(y)


    marker_size = max(6, min(22, 420 / (N + 2)))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            size=marker_size,
            color=colors,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title=dict(text="Option<br>value", font=dict(color=PALETTE["muted"])),
                tickfont=dict(color=PALETTE["muted"]),
                outlinewidth=0,
                thickness=12,
            ),
            line=dict(color=PALETTE["bg"], width=1),
        ),
        text=texts,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )

    traces = [edge_trace, node_trace]


    if ring_x:
        traces.append(
            go.Scatter(
                x=ring_x,
                y=ring_y,
                mode="markers",
                marker=dict(
                    size=marker_size + 6,
                    color="rgba(0,0,0,0)",
                    line=dict(color=PALETTE["exercise"], width=2),
                ),
                name="early exercise",
                hoverinfo="skip",
            )
        )

    fig = go.Figure(data=traces)
    layout = _base_layout(f"Recombining CRR lattice&nbsp; ({option_type}, N = {N})")
    layout.update(
        xaxis=dict(
            title="step i",
            showgrid=False,
            zeroline=False,
            color=PALETTE["muted"],
        ),
        yaxis=dict(
            title="net up-moves (2j - i)",
            showgrid=False,
            zeroline=False,
            color=PALETTE["muted"],
        ),
        showlegend=bool(ring_x),
    )
    fig.update_layout(layout)
    return fig


def build_convergence_figure(
    steps: list[int],
    prices: list[float],
    bs_price: float,
    option_type: str,
) -> go.Figure:
    """Binomial price vs. N, converging to the flat Black-Scholes line."""
    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=[steps[0], steps[-1]],
            y=[bs_price, bs_price],
            mode="lines",
            line=dict(color=PALETTE["accent2"], width=2, dash="dash"),
            name=f"Black-Scholes = {bs_price:.4f}",
        )
    )


    fig.add_trace(
        go.Scatter(
            x=steps,
            y=prices,
            mode="lines+markers",
            line=dict(color=PALETTE["accent"], width=1.5),
            marker=dict(size=3, color=PALETTE["accent"]),
            name="Binomial price",
            hovertemplate="N = %{x}<br>price = %{y:.4f}<extra></extra>",
        )
    )

    layout = _base_layout(
        f"Binomial → Black-Scholes convergence&nbsp; ({option_type})"
    )
    layout.update(
        xaxis=dict(
            title="number of steps N",
            gridcolor=PALETTE["grid"],
            zeroline=False,
            color=PALETTE["muted"],
        ),
        yaxis=dict(
            title="option price",
            gridcolor=PALETTE["grid"],
            zeroline=False,
            color=PALETTE["muted"],
        ),
    )
    fig.update_layout(layout)
    return fig


def build_smile_figure(
    smile_rows: list[dict],
    spot: float,
    ticker: str,
    expiry: str,
    option_type: str,
) -> go.Figure:
    """Implied volatility (computed by *our own* Newton-Raphson solver, not
    taken from a data vendor's own column) plotted against strike.

    A flat Black-Scholes world would show a single horizontal line -- the one
    sigma the whole dashboard otherwise assumes is constant. Real quotes
    almost never look like that; the shape traced out here (smile/skew) is
    live evidence of the "constant volatility" assumption breaking down.
    """
    ok_rows = [row for row in smile_rows if row["iv"] is not None]
    strikes = [row["strike"] for row in ok_rows]
    ivs = [row["iv"] * 100 for row in ok_rows]
    methods = [row["method"] for row in ok_rows]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=strikes,
            y=ivs,
            mode="lines+markers",
            line=dict(color=PALETTE["accent"], width=1.5),
            marker=dict(size=7, color=PALETTE["accent"]),
            name=f"Implied vol ({option_type})",
            customdata=methods,
            hovertemplate=(
                "Strike = %{x}<br>Implied vol = %{y:.2f}%<br>solver: %{customdata}<extra></extra>"
            ),
        )
    )
    if strikes:
        fig.add_vline(
            x=spot,
            line=dict(color=PALETTE["accent2"], width=1.5, dash="dot"),
            annotation_text=f"spot = {spot:,.2f}",
            annotation_font_color=PALETTE["muted"],
        )

    layout = _base_layout(f"Implied volatility smile &middot; {ticker} {expiry} ({option_type}s)")
    layout.update(
        xaxis=dict(title="strike K", gridcolor=PALETTE["grid"], zeroline=False, color=PALETTE["muted"]),
        yaxis=dict(title="implied volatility (%)", gridcolor=PALETTE["grid"], zeroline=False, color=PALETTE["muted"]),
    )
    fig.update_layout(layout)
    return fig





@app.route("/")
def index():
    """Serve the single-page dashboard."""
    return render_template("index.html")


@app.route("/api/price", methods=["POST"])
def api_price():
    """Price the option and return figures + numbers as JSON.

    Expects a JSON body with the pricing inputs. Returns:
      * scalar prices (binomial, Black-Scholes) and greeks,
      * risk-neutral quantities (u, d, p, discount factor, ...),
      * two Plotly figures (lattice + convergence) as JSON.
    """
    data = request.get_json(force=True)

    try:
        S = float(data["S"])
        K = float(data["K"])
        r = float(data["r"])
        sigma = float(data["sigma"])
        T = float(data["T"])
        N = int(data["N"])
        option_type = data.get("option_type", "call")
        exercise = data.get("exercise", "european")
        n_max = int(data.get("n_max", 200))
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify(error=f"Invalid input: {exc}"), 400


    if min(S, K, sigma, T) <= 0:
        return jsonify(error="S, K, sigma and T must all be positive."), 400
    if not (1 <= N <= 400):
        return jsonify(error="N (lattice steps) must be between 1 and 400."), 400
    if not (1 <= n_max <= 1000):
        return jsonify(error="Convergence N must be between 1 and 1000."), 400




    draw_N = min(N, 60)
    res = pricing.crr_tree(S, K, r, sigma, T, draw_N, option_type, exercise)


    binom_price = pricing.crr_price(S, K, r, sigma, T, N, option_type, exercise)

    bs = pricing.black_scholes(S, K, r, sigma, T, option_type)

    steps, conv_prices = pricing.convergence_series(
        S, K, r, sigma, T, option_type, exercise, n_max=n_max
    )

    lattice_fig = build_lattice_figure(res, option_type)
    conv_fig = build_convergence_figure(steps, conv_prices, bs.price, option_type)



    terminal_spots = np.array(res.asset_tree[draw_N])
    terminal_payoffs = (
        np.maximum(terminal_spots - K, 0.0)
        if option_type == "call"
        else np.maximum(K - terminal_spots, 0.0)
    )
    j = np.arange(draw_N + 1)

    from math import comb

    rn_probs = np.array(
        [comb(draw_N, int(jj)) * res.p**jj * (1 - res.p) ** (draw_N - jj) for jj in j]
    )
    expected_payoff = float(np.sum(rn_probs * terminal_payoffs))
    total_discount = float(res.disc**draw_N)

    payload = {
        "binom_price": binom_price,
        "binom_price_drawn": res.price,
        "bs_price": bs.price,
        "abs_diff": abs(binom_price - bs.price),
        "greeks": {
            "delta": bs.delta,
            "gamma": bs.gamma,
            "vega": bs.vega / 100.0,
            "theta": bs.theta / 365.0,
            "rho": bs.rho / 100.0,
            "d1": bs.d1,
            "d2": bs.d2,
        },
        "risk_neutral": {
            "dt": res.dt,
            "u": res.u,
            "d": res.d,
            "p": res.p,
            "disc_step": res.disc,
            "disc_total": total_discount,
            "draw_N": draw_N,
            "expected_payoff": expected_payoff,
            "discounted_expected_payoff": expected_payoff * total_discount,
        },
        "lattice_fig": json.loads(
            json.dumps(lattice_fig, cls=plotly.utils.PlotlyJSONEncoder)
        ),
        "conv_fig": json.loads(
            json.dumps(conv_fig, cls=plotly.utils.PlotlyJSONEncoder)
        ),
    }
    return jsonify(payload)





@app.route("/api/market/stats", methods=["POST"])
def api_market_stats():
    """Spot price + annualised historical volatility for a ticker.

    Used to pre-fill the pricer's S and sigma inputs with real numbers
    instead of guesses.
    """
    data = request.get_json(force=True)
    ticker = data.get("ticker", "")
    lookback_days = int(data.get("lookback_days", 252))

    try:
        stats = market_data.get_historical_stats(ticker, lookback_days)
    except market_data.MarketDataError as exc:
        return jsonify(error=str(exc)), 400

    return jsonify(
        ticker=stats.ticker,
        spot=stats.spot,
        hist_vol=stats.hist_vol,
        n_obs=stats.n_obs,
        as_of=stats.as_of,
    )


@app.route("/api/market/expirations", methods=["POST"])
def api_market_expirations():
    """List every option-expiry date Yahoo Finance has for a ticker."""
    data = request.get_json(force=True)
    ticker = data.get("ticker", "")

    try:
        expirations = market_data.list_expirations(ticker)
    except market_data.MarketDataError as exc:
        return jsonify(error=str(exc)), 400

    return jsonify(ticker=ticker.strip().upper(), expirations=expirations)


@app.route("/api/market/smile", methods=["POST"])
def api_market_smile():
    """Fetch a real option chain and back out implied volatility at every
    usable strike, using our own :func:`pricing.implied_volatility` solver.

    Returns the smile chart plus a per-strike table so every number can be
    inspected (market price, where it came from, and the recovered vol).
    """
    data = request.get_json(force=True)
    ticker = data.get("ticker", "")
    option_type = data.get("option_type", "call")
    expiry = data.get("expiry") or None
    try:
        r = float(data.get("r", 0.05))
    except (TypeError, ValueError):
        return jsonify(error="Invalid risk-free rate."), 400

    try:
        chain = market_data.get_option_chain(ticker, option_type, expiry)
    except market_data.MarketDataError as exc:
        return jsonify(error=str(exc)), 400

    smile_rows = []
    n_iv_failed = 0
    for row in chain.rows:
        iv_res = pricing.implied_volatility(
            row.market_price, chain.spot, row.strike, r, chain.T, option_type
        )
        if iv_res.sigma is None:
            n_iv_failed += 1
        smile_rows.append(
            {
                "strike": row.strike,
                "market_price": row.market_price,
                "price_source": row.price_source,
                "volume": row.volume,
                "iv": iv_res.sigma,
                "method": iv_res.method,
                "error": iv_res.error,
            }
        )

    fig = build_smile_figure(smile_rows, chain.spot, chain.ticker, chain.expiry, option_type)

    return jsonify(
        ticker=chain.ticker,
        spot=chain.spot,
        expiry=chain.expiry,
        T=chain.T,
        option_type=option_type,
        r=r,
        n_used=len(chain.rows),
        n_dropped=chain.n_dropped,
        n_iv_failed=n_iv_failed,
        rows=smile_rows,
        fig=json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5002)
