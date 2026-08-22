"""
rolling_correlation.py
Pure calculation module for rolling and static correlation analysis
across any asset pair or universe. No plotting, no hardcoded periods --
visualize.py (or any caller) decides what timeframe/assets to slice.
"""

import pandas as pd


def compute_rolling_correlation(
    returns: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    window: int = 60,
) -> pd.Series:
    """
    Rolling pairwise correlation between two assets over time.
    Returns a Series indexed by date.
    """
    return returns[asset_a].rolling(window).corr(returns[asset_b])


def compute_correlation_matrix(
    returns: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Static correlation matrix over a given date range
    (or the full dataset if no range is given).
    """
    data = returns
    if start and end:
        data = returns.loc[start:end]
    return data.corr()


def compute_rolling_correlation_matrix_series(
    returns: pd.DataFrame,
    window: int = 60,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """
    Full rolling correlation matrix for every date in the series,
    returned as {date: correlation_matrix}. Useful for animation/slider
    views in visualize.py.
    Note: this is O(n) correlation-matrix computations, each O(assets^2) --
    expensive for long histories or wide universes. Callers should trim
    the date range or asset set before calling this on the full S&P universe.
    """
    dates = returns.index[window:]
    matrices = {}
    for date in dates:
        window_data = returns.loc[:date].tail(window)
        matrices[date] = window_data.corr()
    return matrices