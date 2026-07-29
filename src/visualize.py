"""
visualize.py
Visualizes correlation regime shifts: (1) side-by-side heatmaps
comparing correlation structure across defined stress periods, and
(2) a rolling correlation line plot for a specific asset pair over time.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display
import pandas as pd

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
plt.rcParams["font.family"] = "serif"


def plot_period_comparison(snapshots, ticker_labels=None):
    """
    Plots a grid of correlation heatmaps, one per stress period,
    for easy side-by-side comparison of regime shifts.
    """
    n = len(snapshots)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, (period_name, corr_matrix) in zip(axes, snapshots.items()):
        labels = corr_matrix.columns
        if ticker_labels:
            labels = [ticker_labels.get(t, t) for t in corr_matrix.columns]

        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax, cbar=True,
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5
        )
        ax.set_title(period_name, fontsize=12)

    # Hide any unused subplots
    for ax in axes[len(snapshots):]:
        ax.axis("off")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "regime_comparison_heatmaps.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved regime comparison heatmaps to {out_path}")
    plt.show()


def plot_rolling_correlation_pair(rolling_corr_series, pair_label, highlight_periods=None):
    """
    Plots the rolling correlation between a specific asset pair over
    time, with optional shaded regions for known stress periods.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rolling_corr_series.index, rolling_corr_series.values,
            color="black", linewidth=1.3)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)

    if highlight_periods:
        colors = ["#f4cccc", "#cfe2f3", "#d9ead3"]
        for i, (name, (start, end)) in enumerate(highlight_periods.items()):
            if start is None:
                continue
            ax.axvspan(start, end, color=colors[i % len(colors)], alpha=0.4, label=name)

    ax.set_title(f"Rolling 60-Day Correlation: {pair_label}")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-1, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"rolling_corr_{pair_label.replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved rolling correlation plot to {out_path}")
    plt.show()


RANGE_OPTIONS = {
    "6M": 126,
    "1Y": 252,
    "3Y": 756,
    "5Y": 1260,
    "All": None,
}


def filter_by_range(series, range_label):
    if range_label == "All" or RANGE_OPTIONS[range_label] is None:
        return series
    n_days = RANGE_OPTIONS[range_label]
    return series.tail(n_days)


def build_pair_correlation_dashboard(returns, ticker_labels=None, window=60):
    """
    Interactive dashboard: pick any two assets via dropdowns (Asset B
    excludes whatever is selected in Asset A) and a time range, and
    plot their rolling correlation over that window.
    """
    all_tickers = list(returns.columns)
    label_map = ticker_labels or {t: t for t in all_tickers}

    def display_name(ticker):
        return f"{ticker} ({label_map[ticker]})" if ticker in label_map else ticker

    display_to_ticker = {display_name(t): t for t in all_tickers}

    asset_a_dropdown = widgets.Dropdown(
        options=[display_name(t) for t in all_tickers],
        value=display_name(all_tickers[0]),
        description="Asset A:",
    )
    asset_b_dropdown = widgets.Dropdown(
        options=[display_name(t) for t in all_tickers if t != all_tickers[0]],
        value=display_name(all_tickers[1]),
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
        available = [display_name(t) for t in all_tickers if t != selected_a]

        current_b = asset_b_dropdown.value
        asset_b_dropdown.options = available
        # If Asset B's current value is now invalid (matched A), reset it
        if current_b not in available:
            asset_b_dropdown.value = available[0]

    def redraw(change=None):
        output.clear_output(wait=True)
        ticker_a = display_to_ticker[asset_a_dropdown.value]
        ticker_b = display_to_ticker[asset_b_dropdown.value]

        corr_series = returns[ticker_a].rolling(window).corr(returns[ticker_b])
        corr_series = filter_by_range(corr_series.dropna(), range_dropdown.value)

        with output:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(corr_series.index, corr_series.values, color="black", linewidth=2.0)
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
            ax.set_title(f"Rolling {window}-Day Correlation: {asset_a_dropdown.value} vs {asset_b_dropdown.value}")
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

def build_interactive_heatmap_dashboard(returns, ticker_labels=None):
    """
    Interactive dashboard: pick a year and month via dropdowns, and
    the correlation heatmap updates to show that month's correlation
    structure across the asset basket.
    """
    label_map = ticker_labels or {t: t for t in returns.columns}
    labels = [label_map.get(t, t) for t in returns.columns]

    available_years = sorted(returns.index.year.unique())
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }

    year_dropdown = widgets.Dropdown(
        options=available_years,
        value=available_years[-1],
        description="Year:",
    )
    month_dropdown = widgets.Dropdown(
        options=[(name, num) for num, name in month_names.items()],
        value=1,
        description="Month:",
    )

    controls = widgets.HBox([year_dropdown, month_dropdown])
    output = widgets.Output()

    def redraw(change=None):
        output.clear_output(wait=True)
        year = year_dropdown.value
        month = month_dropdown.value

        month_data = returns[(returns.index.year == year) & (returns.index.month == month)]

        with output:
            if month_data.empty or len(month_data) < 2:
                print(f"No data available for {month_names[month]} {year}.")
                return

            corr_matrix = month_data.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))


            fig, ax = plt.subplots(figsize=(8, 7))
            sns.heatmap(
                corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5
            )
            ax.set_title(f"Correlation Matrix — {month_names[month]} {year}", fontsize=13)
            plt.tight_layout()
            plt.show()

    year_dropdown.observe(redraw, names="value")
    month_dropdown.observe(redraw, names="value")

    display(controls, output)
    redraw()