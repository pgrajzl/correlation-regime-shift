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


def build_pair_correlation_dashboard(returns, ticker_labels=None, window=60):
    """
    Interactive dashboard: pick Asset A and Asset B side by side (B excludes
    whatever's selected in A) and a timeframe, and plot their rolling
    correlation over that window.
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

    controls = widgets.HBox([asset_a_dropdown, asset_b_dropdown, range_dropdown])
    output = widgets.Output()

    def update_asset_b_options(change=None):
        """Prevent Asset B from offering the same ticker as Asset A."""
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

        corr_series = returns[ticker_a].rolling(window).corr(returns[ticker_b]).dropna()
        corr_series = _filter_by_range(corr_series, range_dropdown.value)

        with output:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(corr_series.index, corr_series.values, color="black", linewidth=2.0)
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
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

    display(controls, output)
    redraw()


def build_correlation_matrix_dashboard(returns, ticker_labels=None):
    """
    Interactive dashboard: pick any subset of assets (multi-select) and a
    timeframe, and see the correlation heatmap for that selection over
    that window. Replaces the old fixed year/month picker with free
    asset + range selection.
    """
    all_tickers = list(returns.columns)
    label_map = ticker_labels or {t: t for t in all_tickers}
    display_to_ticker = {_display_name(t, label_map): t for t in all_tickers}
    all_display_names = [_display_name(t, label_map) for t in all_tickers]

    asset_select = widgets.SelectMultiple(
        options=all_display_names,
        value=tuple(all_display_names),  # default: everything selected
        description="Assets:",
        rows=min(len(all_display_names), 10),
    )
    range_dropdown = widgets.Dropdown(
        options=list(RANGE_OPTIONS.keys()), value="1Y", description="Range:"
    )

    controls = widgets.VBox([asset_select, range_dropdown])
    output = widgets.Output()

    def redraw(change=None):
        output.clear_output(wait=True)

        selected_tickers = [display_to_ticker[name] for name in asset_select.value]
        if len(selected_tickers) < 2:
            with output:
                print("Select at least two assets to compute a correlation matrix.")
            return

        subset = returns[selected_tickers]
        subset = _filter_by_range(subset, range_dropdown.value)

        corr_matrix = compute_correlation_matrix(subset)
        labels = [label_map.get(t, t) for t in corr_matrix.columns]
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        with output:
            fig_size = max(6, len(selected_tickers) * 0.8)
            fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))
            sns.heatmap(
                corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5,
            )
            ax.set_title(f"Correlation Matrix — {range_dropdown.value}", fontsize=13)
            plt.tight_layout()
            plt.show()

    asset_select.observe(redraw, names="value")
    range_dropdown.observe(redraw, names="value")

    display(controls, output)
    redraw()