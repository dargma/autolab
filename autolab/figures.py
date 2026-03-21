"""CVPR-quality figure generation utilities."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Colorblind-safe palette
PALETTE = sns.color_palette("colorblind")
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.figsize": (8, 5),
})


def plot_sweep_comparison(results, target_accuracy=None, output_path="fig-sweep.svg",
                          title="Architecture Sweep Results"):
    """Bar chart comparing accuracy across candidates. Baseline shown as dashed line."""
    names = [r["name"] for r in results]
    accs = [r["accuracy"] * 100 for r in results]

    fig, ax = plt.subplots()
    bars = ax.bar(range(len(names)), accs, color=PALETTE[:len(names)], edgecolor="black", linewidth=0.5)

    if target_accuracy is not None:
        ax.axhline(y=target_accuracy * 100, color="red", linestyle="--", linewidth=1.5,
                   label=f"Target ({target_accuracy*100:.0f}%)")
        ax.legend()

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_pareto(results, output_path="fig-pareto.svg",
                title="Pareto Frontier: Accuracy vs Latency"):
    """Scatter plot with Pareto-optimal models highlighted."""
    accs = [r["accuracy"] * 100 for r in results]
    lats = [r["avg_latency_ms"] for r in results]
    names = [r["name"] for r in results]
    params = [r["params"] for r in results]

    # Find Pareto-optimal points (maximize accuracy, minimize latency)
    pareto = []
    for i, (a, l) in enumerate(zip(accs, lats)):
        dominated = False
        for j, (a2, l2) in enumerate(zip(accs, lats)):
            if i != j and a2 >= a and l2 <= l and (a2 > a or l2 < l):
                dominated = True
                break
        if not dominated:
            pareto.append(i)

    fig, ax = plt.subplots()

    # Non-Pareto points
    for i in range(len(results)):
        marker = "s" if i not in pareto else None
        if i not in pareto:
            ax.scatter(lats[i], accs[i], c=[PALETTE[3]], s=60, alpha=0.6, zorder=2)
            ax.annotate(names[i], (lats[i], accs[i]), fontsize=7,
                        textcoords="offset points", xytext=(5, 5))

    # Pareto points
    pareto_lats = [lats[i] for i in pareto]
    pareto_accs = [accs[i] for i in pareto]
    ax.scatter(pareto_lats, pareto_accs, c=[PALETTE[0]], s=100, marker="*",
               edgecolors="black", linewidth=0.5, zorder=3, label="Pareto-optimal")
    for i in pareto:
        ax.annotate(names[i], (lats[i], accs[i]), fontsize=8, fontweight="bold",
                    textcoords="offset points", xytext=(5, -10))

    # Connect Pareto frontier
    pareto_sorted = sorted(pareto, key=lambda i: lats[i])
    ax.plot([lats[i] for i in pareto_sorted], [accs[i] for i in pareto_sorted],
            "k--", alpha=0.4, linewidth=1)

    ax.set_xlabel("Average Latency (ms)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_ralph_convergence(iterations, metric_name, target, output_path="fig-convergence.svg",
                           title="Ralph-Loop Convergence"):
    """Line chart showing metric convergence over iterations."""
    iters = list(range(len(iterations)))
    values = [it.get("best_value", 0) for it in iterations]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, [v * 100 for v in values], "o-", color=PALETTE[0], linewidth=2,
            markersize=8, label=f"Best {metric_name}")
    ax.axhline(y=target * 100, color="red", linestyle="--", linewidth=1.5,
               label=f"Target ({target*100:.0f}%)")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(f"{metric_name} (%)")
    ax.set_title(title)
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_cross_dataset(results_by_dataset, output_path="fig-cross-dataset.svg",
                       title="Cross-Dataset Comparison"):
    """Grouped bar chart comparing models across datasets."""
    datasets = list(results_by_dataset.keys())
    # Get all model names (union across datasets)
    all_names = []
    for ds_results in results_by_dataset.values():
        for r in ds_results:
            if r["name"] not in all_names:
                all_names.append(r["name"])

    fig, ax = plt.subplots(figsize=(10, 5))
    n_ds = len(datasets)
    width = 0.8 / n_ds
    x = list(range(len(all_names)))

    for di, ds in enumerate(datasets):
        acc_map = {r["name"]: r["accuracy"] * 100 for r in results_by_dataset[ds]}
        vals = [acc_map.get(n, 0) for n in all_names]
        offsets = [xi + (di - n_ds / 2 + 0.5) * width for xi in x]
        ax.bar(offsets, vals, width=width, label=ds, color=PALETTE[di],
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=45, ha="right")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def load_results(summary_json_path):
    """Load results from a summary.json file, returning the ranking list."""
    with open(summary_json_path) as f:
        data = json.load(f)
    return data.get("ranking", [])
