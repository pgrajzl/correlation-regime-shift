"""
visualize.py
Visualizes correlation regime shifts: (1) side-by-side heatmaps
comparing correlation structure across defined stress periods, and
(2) a rolling correlation line plot for a specific asset pair over time.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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