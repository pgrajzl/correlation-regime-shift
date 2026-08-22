"""
fetch_data.py
Pulls daily adjusted close prices for:
  1. A cross-asset macro basket (equities, bonds, commodities, crypto,
     currency, volatility, rates) for rolling correlation / regime analysis.
  2. The full current S&P 500 universe, for cross-sectional signal research.

Both datasets are cleaned, aligned to trading days, and saved to disk as
prices + simple daily returns.
"""

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cross-asset macro basket
# ---------------------------------------------------------------------------
# Expanded beyond the original set to cover rates, credit, vol, and breadth
# -- the things that actually drive regime shifts, not just asset returns.
MACRO_TICKERS = {
    # Equities
    "SPY": "US Equities (S&P 500)",
    "RSP": "US Equities (S&P 500 Equal Weight)",
    "QQQ": "Nasdaq 100",
    "IWM": "US Small Caps (Russell 2000)",
    "EFA": "Developed ex-US Equities",
    "EEM": "Emerging Market Equities",
    # Rates / bonds
    "TLT": "Long Treasuries (20Y+)",
    "IEF": "Intermediate Treasuries (7-10Y)",
    "SHY": "Short Treasuries (1-3Y)",
    "LQD": "Investment Grade Corporate Bonds",
    "HYG": "High Yield Corporate Bonds",
    "^TNX": "10Y Treasury Yield",
    "^IRX": "13-Week T-Bill Yield",
    # Commodities
    "GLD": "Gold",
    "SLV": "Silver",
    "USO": "Oil (WTI)",
    # Currency
    "UUP": "US Dollar Index",
    # Crypto
    "BTC-USD": "Bitcoin",
    # Volatility / risk sentiment
    "^VIX": "CBOE Volatility Index",
}

START_DATE = "2018-01-01"
END_DATE = "2026-07-28"

# Tickers that trade 24/7 or on a different calendar than US equities/bonds.
# These get forward-filled onto the traditional trading-day index rather
# than being used to define which rows survive the dropna alignment step.
CONTINUOUS_TICKERS = {"BTC-USD", "ETH-USD"}

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


# ---------------------------------------------------------------------------
# S&P 500 universe
# ---------------------------------------------------------------------------
def get_sp500_tickers() -> list[str]:
    """Scrape the current S&P 500 constituent list from Wikipedia.

    This reflects whatever the index composition is *today* -- it will
    silently drift over time as constituents change, which is the correct
    behavior for "current universe" cross-sectional work but means this
    is NOT survivorship-bias-free for historical backtests. Note that
    limitation explicitly wherever this universe feeds a backtest.
    """
    logger.info("Fetching current S&P 500 constituent list from Wikipedia...")
    tables = pd.read_html(SP500_WIKI_URL)
    sp500_table = tables[0]
    tickers = sp500_table["Symbol"].str.replace(".", "-", regex=False).tolist()
    logger.info(f"Found {len(tickers)} S&P 500 constituents.")
    return tickers


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _download_in_chunks(
    tickers: list[str],
    start: str,
    end: str,
    chunk_size: int = 50,
    max_retries: int = 3,
    pause: float = 2.0,
) -> pd.DataFrame:
    """Download Close prices in chunks, retrying failed chunks.

    yfinance batch downloads over large ticker lists (e.g. 500+ names)
    are prone to partial failures -- rate limits, delisted/renamed
    tickers, transient network errors. Chunking + retry makes this
    resilient instead of failing the whole run on one bad ticker.
    """
    frames = []
    failed_tickers: list[str] = []

    chunks = [tickers[i : i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    for i, chunk in enumerate(chunks, start=1):
        logger.info(f"Downloading chunk {i}/{len(chunks)} ({len(chunk)} tickers)...")
        attempt = 0
        while attempt < max_retries:
            try:
                data = yf.download(
                    chunk,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )["Close"]
                # yf collapses to a Series (not DataFrame) when chunk size is 1
                if isinstance(data, pd.Series):
                    data = data.to_frame(name=chunk[0])
                frames.append(data)
                break
            except Exception as e:
                attempt += 1
                logger.warning(
                    f"Chunk {i} failed (attempt {attempt}/{max_retries}): {e}"
                )
                time.sleep(pause)
        else:
            logger.error(f"Chunk {i} failed after {max_retries} attempts, skipping.")
            failed_tickers.extend(chunk)

    if not frames:
        raise RuntimeError("All download chunks failed -- no data retrieved.")

    combined = pd.concat(frames, axis=1)

    # Drop any ticker that came back entirely empty (delisted, bad symbol,
    # renamed, etc.) rather than letting it silently poison downstream NaNs.
    empty_cols = combined.columns[combined.isna().all()].tolist()
    if empty_cols:
        logger.warning(f"Dropping {len(empty_cols)} tickers with no data: {empty_cols}")
        combined = combined.drop(columns=empty_cols)
        failed_tickers.extend(empty_cols)

    if failed_tickers:
        logger.warning(
            f"{len(set(failed_tickers))} tickers could not be fetched or had no data."
        )

    return combined


def fetch_macro_basket(
    tickers: dict[str, str] | None = None,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """Download the cross-asset macro basket, aligned to traditional trading days."""
    tickers = tickers or MACRO_TICKERS
    ticker_list = list(tickers.keys())
    logger.info(f"Fetching macro basket: {len(ticker_list)} assets...")

    data = _download_in_chunks(ticker_list, start, end, chunk_size=20)

    # Align to traditional (equity/bond) trading days: drop rows where any
    # non-continuous asset is missing (weekends, market holidays), then
    # forward-fill the 24/7 assets (crypto) onto that index.
    continuous_cols = [c for c in data.columns if c in CONTINUOUS_TICKERS]
    traditional_cols = [c for c in data.columns if c not in CONTINUOUS_TICKERS]

    data = data.dropna(subset=traditional_cols, how="any")
    if continuous_cols:
        data[continuous_cols] = data[continuous_cols].ffill()

    return data


def fetch_sp500_universe(
    start: str = START_DATE,
    end: str = END_DATE,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Download Close prices for the full current S&P 500 universe."""
    tickers = tickers or get_sp500_tickers()
    logger.info(f"Fetching S&P 500 universe: {len(tickers)} constituents...")

    data = _download_in_chunks(tickers, start, end, chunk_size=50)

    # Cross-sectional universes commonly have partial-history names (recent
    # IPOs, recent index additions). Don't drop rows for this -- that would
    # wipe out the whole history. Leave NaNs in place; downstream signal
    # code should handle per-column availability explicitly (e.g. require
    # a minimum lookback window per ticker) rather than assuming a dense panel.
    coverage = data.notna().mean().sort_values()
    sparse = coverage[coverage < 0.5]
    if len(sparse) > 0:
        logger.info(
            f"{len(sparse)} tickers have <50% coverage over the sample window "
            f"(likely recent IPOs / index additions): {sparse.index.tolist()[:10]}"
            f"{'...' if len(sparse) > 10 else ''}"
        )

    return data


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Convert price levels to simple daily returns."""
    return price_df.pct_change().dropna(how="all")


def save_dataset(prices: pd.DataFrame, returns: pd.DataFrame, prefix: str) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    prices.to_csv(DATA_DIR / f"{prefix}_prices.csv")
    returns.to_csv(DATA_DIR / f"{prefix}_returns.csv")
    logger.info(f"Saved {prefix} prices/returns to {DATA_DIR}/")


def main():
    # Macro basket
    macro_prices = fetch_macro_basket()
    macro_returns = compute_returns(macro_prices)
    save_dataset(macro_prices, macro_returns, prefix="macro")
    print("\nMacro basket (tail):")
    print(macro_returns.tail())

    # S&P 500 universe
    sp500_prices = fetch_sp500_universe()
    sp500_returns = compute_returns(sp500_prices)
    save_dataset(sp500_prices, sp500_returns, prefix="sp500")
    print(f"\nS&P 500 universe: {sp500_prices.shape[1]} tickers, "
          f"{sp500_prices.shape[0]} trading days.")


if __name__ == "__main__":
    main()