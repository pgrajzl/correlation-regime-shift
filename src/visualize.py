"""
visualize.py
Interactive visualizations for correlation analysis:
  1. Rolling correlation dashboard — pick any two assets side by side
     and a timeframe, see how their correlation moves over time.
  2. Correlation matrix dashboard — pick any subset of assets and a
     timeframe, see the static correlation heatmap for that selection.
Both pull from rolling_correlation.py's calculation functions; this
module is presentation only.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display

from .rolling_correlation import compute_correlation_matrix

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

plt.rcParams["font.family"] = "Times New Roman"

RANGE_OPTIONS = {
    "6M": 126,
    "1Y": 252,
    "3Y": 756,
    "5Y": 1260,
    "All": None,
}


def _filter_by_range(data, range_label):
    """Trim a Series or DataFrame to the trailing N trading days for a
    given range label. 'All' returns the data unchanged."""
    n_days = RANGE_OPTIONS[range_label]
    if range_label == "All" or n_days is None:
        return data
    return data.tail(n_days)


def _display_name(ticker, label_map):
    return f"{ticker} ({label_map[ticker]})" if ticker in label_map else ticker


def plot_rolling_correlation_pair(returns, ticker_a, ticker_b, window=60, pair_label=None):
    """
    Static rolling-correlation plot for a specific asset pair, saved to disk.
    Useful for a one-off export outside the interactive dashboard.
    """
    corr_series = returns[ticker_a].rolling(window).corr(returns[ticker_b]).dropna()
    label = pair_label or f"{ticker_a} vs {ticker_b}"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(corr_series.index, corr_series.values, color="black", linewidth=1.3)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title(f"Rolling {window}-Day Correlation: {label}")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1, 1)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"rolling_corr_{label.replace(' ', '_').replace('/', '-')}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved rolling correlation plot to {out_path}")
    plt.show()


def build_pair_correlation_dashboard(returns, ticker_labels=None, window=60, ma_window=126):
    """
    Interactive dashboard: pick Asset A and Asset B side by side (B excludes
    whatever's selected in A), a timeframe, and plot their rolling
    correlation over that window. A checkbox toggles a long-run moving
    average of the correlation series on/off, so you can see how current
    correlation stacks up against its own trend.
    """
    all_tickers = list(returns.columns)
    label_map = ticker_labels or {t: t for t in all_tickers}
    display_to_ticker = {_display_name(t, label_map): t for t in all_tickers}

    asset_a_dropdown = widgets.Dropdown(
        options=[_display_name(t, label_map) for t in all_tickers],
        value=_display_name(all_tickers[0], label_map),
        description="Asset A:",
    )
    asset_b_dropdown = widgets.Dropdown(
        options=[_display_name(t, label_map) for t in all_tickers if t != all_tickers[0]],
        value=_display_name(all_tickers[1], label_map),
        description="Asset B:",
    )
    range_dropdown = widgets.Dropdown(
        options=list(RANGE_OPTIONS.keys()), value="1Y", description="Range:"
    )
    show_ma_checkbox = widgets.Checkbox(
        value=False, description=f"Show {ma_window}-day MA of correlation"
    )

    controls = widgets.HBox([asset_a_dropdown, asset_b_dropdown, range_dropdown, show_ma_checkbox])
    output = widgets.Output()

    def update_asset_b_options(change=None):
        selected_a = display_to_ticker[asset_a_dropdown.value]
        available = [_display_name(t, label_map) for t in all_tickers if t != selected_a]
        current_b = asset_b_dropdown.value
        asset_b_dropdown.options = available
        if current_b not in available:
            asset_b_dropdown.value = available[0]

    def redraw(change=None):
        output.clear_output(wait=True)
        ticker_a = display_to_ticker[asset_a_dropdown.value]
        ticker_b = display_to_ticker[asset_b_dropdown.value]

        # Compute the MA on the FULL correlation history before trimming to
        # the display range, so the average isn't distorted by truncation
        # at the start of the visible window.
        full_corr = returns[ticker_a].rolling(window).corr(returns[ticker_b]).dropna()
        corr_ma = full_corr.rolling(ma_window).mean()

        corr_series = _filter_by_range(full_corr, range_dropdown.value)
        corr_ma_display = _filter_by_range(corr_ma, range_dropdown.value)

        with output:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(corr_series.index, corr_series.values, color="black", linewidth=2.0,
                    label=f"{window}-day rolling correlation")
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)

            if show_ma_checkbox.value:
                ax.plot(corr_ma_display.index, corr_ma_display.values,
                        color="#c0392b", linewidth=1.6, linestyle="--",
                        label=f"{ma_window}-day MA of correlation")
                ax.legend(loc="upper left", fontsize=9)

            ax.set_title(
                f"Rolling {window}-Day Correlation: "
                f"{asset_a_dropdown.value} vs {asset_b_dropdown.value}"
            )
            ax.set_ylabel("Correlation")
            ax.set_ylim(-1, 1)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    asset_a_dropdown.observe(update_asset_b_options, names="value")
    asset_a_dropdown.observe(redraw, names="value")
    asset_b_dropdown.observe(redraw, names="value")
    range_dropdown.observe(redraw, names="value")
    show_ma_checkbox.observe(redraw, names="value")

    display(controls, output)
    redraw()