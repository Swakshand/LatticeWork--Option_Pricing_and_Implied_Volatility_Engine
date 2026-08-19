from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
import numpy as np
import yfinance as yf

OptionType = str

class MarketDataError(RuntimeError):
    #Raised whenever live market data cannot be fetched or is unusable.


@dataclass
class HistoricalStats:
    #Spot price + realised volatility estimated from recent daily closes.
    ticker: str
    spot: float
    hist_vol: float
    n_obs: int
    as_of: str


def get_historical_stats(ticker: str, lookback_days: int = 252) -> HistoricalStats:
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise MarketDataError("Please enter a ticker symbol.")

    try:
        hist = yf.Ticker(ticker).history(
            period=f"{max(lookback_days, 30)}d", interval="1d"
        )
    except Exception as exc:
        raise MarketDataError(f"Could not download price history for '{ticker}': {exc}") from exc

    if hist is None or hist.empty or "Close" not in hist:
        raise MarketDataError(
            f"No price history found for '{ticker}'. Check that the ticker symbol is correct."
        )

    closes = hist["Close"].dropna().to_numpy(dtype=float)
    if len(closes) < 10:
        raise MarketDataError(
            f"Only {len(closes)} usable price points found for '{ticker}' -- "
            "too few to estimate volatility reliably."
        )

    log_returns = np.diff(np.log(closes))
    daily_vol = float(np.std(log_returns, ddof=1))
    annualised_vol = daily_vol * sqrt(252)

    return HistoricalStats(
        ticker=ticker,
        spot=float(closes[-1]),
        hist_vol=annualised_vol,
        n_obs=len(log_returns),
        as_of=hist.index[-1].strftime("%Y-%m-%d"),
    )


@dataclass
class ChainRow:
    #One usable strike from a real option chain.
    strike: float
    market_price: float
    price_source: str
    volume: float
    open_interest: float
    in_the_money: bool


@dataclass
class OptionChainData:
    ticker: str
    spot: float
    expiry: str
    T: float
    option_type: str
    rows: list[ChainRow]
    n_dropped: int


def list_expirations(ticker: str) -> list[str]:
    """Return every expiry date Yahoo Finance lists options for."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise MarketDataError("Please enter a ticker symbol.")

    try:
        expirations = list(yf.Ticker(ticker).options)
    except Exception as exc:
        raise MarketDataError(f"Could not fetch option expirations for '{ticker}': {exc}") from exc

    if not expirations:
        raise MarketDataError(f"'{ticker}' has no listed options on Yahoo Finance.")
    return expirations


def get_option_chain(
    ticker: str,
    option_type: str = "call",
    expiry: str | None = None,
    min_volume: float = 0,
) -> OptionChainData:

    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise MarketDataError("Please enter a ticker symbol.")
    if option_type not in ("call", "put"):
        raise MarketDataError("option_type must be 'call' or 'put'.")

    tk = yf.Ticker(ticker)

    try:
        expirations = list(tk.options)
    except Exception as exc:
        raise MarketDataError(f"Could not fetch option expirations for '{ticker}': {exc}") from exc
    if not expirations:
        raise MarketDataError(f"'{ticker}' has no listed options on Yahoo Finance.")

    if expiry is None:
        expiry = expirations[0]
    elif expiry not in expirations:
        preview = ", ".join(expirations[:6]) + ("..." if len(expirations) > 6 else "")
        raise MarketDataError(f"'{expiry}' is not a listed expiry for '{ticker}'. Available: {preview}")

    try:
        chain = tk.option_chain(expiry)
        spot = float(tk.fast_info["last_price"])
    except Exception as exc:
        raise MarketDataError(f"Could not fetch the option chain for '{ticker}': {exc}") from exc

    table = (chain.calls if option_type == "call" else chain.puts).fillna(0)
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days_to_expiry = max((expiry_dt - datetime.now(timezone.utc)).days, 1)
    T = days_to_expiry / 365.0

    rows: list[ChainRow] = []
    n_dropped = 0
    for _, r in table.iterrows():
        bid, ask, last = float(r["bid"]), float(r["ask"]), float(r["lastPrice"])
        volume = float(r["volume"])
        open_interest = float(r["openInterest"])

        if bid > 0 and ask > 0 and ask >= bid:
            price, source = (bid + ask) / 2.0, "mid"
        elif last > 0:
            price, source = last, "last"
        else:
            n_dropped += 1
            continue

        if volume < min_volume:
            n_dropped += 1
            continue

        rows.append(
            ChainRow(
                strike=float(r["strike"]),
                market_price=price,
                price_source=source,
                volume=volume,
                open_interest=open_interest,
                in_the_money=bool(r["inTheMoney"]),
            )
        )

    if not rows:
        raise MarketDataError(f"No usable {option_type} quotes found for '{ticker}' expiring {expiry}.")

    rows.sort(key=lambda row: row.strike)
    return OptionChainData(
        ticker=ticker, spot=spot, expiry=expiry, T=T, option_type=option_type,
        rows=rows, n_dropped=n_dropped,
    )
    
