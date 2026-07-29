"""
fetch_data.py
Pulls daily adjusted close prices for a cross-asset basket (equities,
bonds, commodities, crypto, currency) and computes daily returns for
use in rolling correlation analysis.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TICKERS = {
    "SPY": "US Equities",
    "QQQ": "Nasdaq 100",
    "TLT": "Long Treasuries",
    "LQD": "Corporate Bonds",
    "GLD": "Gold",
    "USO": "Oil",
    "BTC-USD": "Bitcoin",
    "UUP": "US Dollar Index",
}

START_DATE = "2018-01-01"
END_DATE = "2026-07-28"


def fetch_prices(tickers=None, start=START_DATE, end=END_DATE):
    """Download adjusted close prices for the cross-asset basket."""
    tickers = tickers or list(TICKERS.keys())
    print(f"Fetching price data for {len(tickers)} assets...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]

    # BTC-USD trades 7 days/week while equities/bonds trade only on
    # weekdays. Align everything to actual trading days by dropping
    # any row where a traditional market asset has no price (weekends,
    # market holidays), then forward-fill BTC's price on any gaps.
    equity_bond_cols = [t for t in tickers if t != "BTC-USD"]
    data = data.dropna(subset=equity_bond_cols, how="any")

    return data


def compute_returns(price_df):
    """Convert price levels to simple daily returns."""
    return price_df.pct_change().dropna(how="all")


def save_data(prices, returns):
    DATA_DIR.mkdir(exist_ok=True)
    prices.to_csv(DATA_DIR / "prices.csv")
    returns.to_csv(DATA_DIR / "returns.csv")
    print(f"Saved prices and returns to {DATA_DIR}/")


def main():
    prices = fetch_prices()
    returns = compute_returns(prices)
    save_data(prices, returns)
    print(returns.tail())


if __name__ == "__main__":
    main()