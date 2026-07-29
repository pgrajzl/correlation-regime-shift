"""
rolling_correlation.py
Computes rolling correlation matrices over time, and extracts
correlation snapshots for defined market stress periods.
"""

import pandas as pd

# Defined stress/calm periods for comparison snapshots
STRESS_PERIODS = {
    "Pre-COVID Calm (2019)": ("2019-01-01", "2019-12-31"),
    "COVID Crash (Feb-Apr 2020)": ("2020-02-15", "2020-04-15"),
    "2022 Rate Shock": ("2022-01-01", "2022-12-31"),
    "Recent (Last 6M)": None,  # filled dynamically based on latest data
}


def compute_rolling_correlation(returns, pair_a, pair_b, window=60):
    """
    Compute rolling correlation between two specific assets over time.
    Returns a Series indexed by date.
    """
    return returns[pair_a].rolling(window).corr(returns[pair_b])


def compute_correlation_matrix(returns, start=None, end=None):
    """
    Compute a static correlation matrix over a given date range
    (or the full dataset if no range is given).
    """
    data = returns
    if start and end:
        data = returns.loc[start:end]
    return data.corr()


def get_period_snapshots(returns, periods=STRESS_PERIODS):
    """
    Returns a dict of {period_name: correlation_matrix} for each
    defined stress period.
    """
    snapshots = {}
    for name, date_range in periods.items():
        if date_range is None:
            # "Recent" period: last 6 months of available data
            end_date = returns.index.max()
            start_date = end_date - pd.DateOffset(months=6)
            snapshots[name] = compute_correlation_matrix(returns, start_date, end_date)
        else:
            start, end = date_range
            subset = returns.loc[start:end]
            if subset.empty:
                continue
            snapshots[name] = compute_correlation_matrix(returns, start, end)
    return snapshots


def compute_rolling_correlation_matrix_series(returns, window=60):
    """
    Computes the full rolling correlation matrix for every date,
    returned as a dict of {date: correlation_matrix} for the last
    N dates (useful for animation/slider views).
    Note: this can be expensive for long histories/many assets,
    so it's typically called on a trimmed date range.
    """
    dates = returns.index[window:]
    matrices = {}
    for date in dates:
        window_data = returns.loc[:date].tail(window)
        matrices[date] = window_data.corr()
    return matrices