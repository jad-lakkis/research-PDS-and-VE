"""
plot_grouped_comparison.py - grouped bar chart, metric x method, faceted
by metric, in the style of ASL360's Fig. 5 (metric x method, faceted by
video). This project trains/evaluates on one video (Runner), so the
faceting dimension that actually varies here is the BUDGET ARM, used as
the x-axis category within each metric's subplot instead - method is the
grouped/colored dimension, same as ASL360's own bars.

Usage: python plot_grouped_comparison.py <out_dir>
Edit ARMS below to add/remove runs (same convention as
plot_tradeoff_frontier.py) - only baseline A arms are demonstrated here;
re-run once Run D has a clean (ent_coef=0) converged result.
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONVERGED_WINDOW = 200

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
METHOD_COLOR = {"PPO": "#2a78d6", "PPO+PDS": "#eb6834"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
})


def load_raw_progress(pattern):
    files = sorted(glob.glob(pattern), key=lambda p: int(re.search(r"block_v(\d+)", p).group(1)) if "block_v" in p else 0)
    frames = []
    for f in files:
        p = pd.read_csv(f, dtype=str)
        if "eval/feasible" not in p.columns:
            continue
        p = p[p["eval/feasible"] != "eval/feasible"].apply(pd.to_numeric, errors="coerce")
        p = p[p["eval/feasible"].notna()]
        frames.append(p)
    return pd.concat(frames, ignore_index=True) if frames else None


ARMS = [
    ("PPO", "0.3/6.5/8",  "runpod_results/dbar_0.3_pbar_6.5_bbar_8.0_seed0/block_v*/progress.csv"),
    ("PPO", "0.12/4/7.4", "runpod_results_summary/02_most_important_runs/dbar_0.12_pbar_4_bbar_7.4_seed0/block_v*/progress.csv"),
    ("PPO", "0.12/4/8",   "runpod_results_v2/v3/dbar_0.12_pbar_4_bbar_8.0_seed0/block_v*/progress.csv"),
    ("PPO", "0.1/6.5/8",  "runpod_results/dbar_0.1_pbar_6.5_bbar_8.0_seed0/block_v*/progress.csv"),
    ("PPO+PDS", "0.3/6.5/8",  "run_D_clean_results/run_D_clean_dbar_0.3_pbar_6.5_bbar_8.0_seed0/progress.csv"),
    ("PPO+PDS", "0.12/4/7.4", "run_D_clean_results/run_D_clean_dbar_0.12_pbar_4_bbar_7.4_seed0/progress.csv"),
    ("PPO+PDS", "0.12/4/8",   "run_D_clean_results/run_D_clean_dbar_0.12_pbar_4_bbar_8.0_seed0/progress.csv"),
    ("PPO+PDS", "0.1/6.5/8",  "run_D_clean_results/run_D_clean_dbar_0.1_pbar_6.5_bbar_8.0_seed0/progress.csv"),
]

# (progress.csv column, display label, higher-is-better)
METRICS = [("eval/J_Q", "Viewport quality " + r"$J_Q$", True),
           ("eval/violation", "Constraint violation", False),
           ("eval/mean_power_mw", "Transmit power (mW)", False),
           ("eval/mean_stall_sec", "Stall time (s)", False)]


def generate(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for method, label, pattern in ARMS:
        df = load_raw_progress(pattern)
        if df is None:
            print(f"SKIP {label}: no data found for {pattern}")
            continue
        tail = df.tail(CONVERGED_WINDOW)
        row = {"method": method, "arm": label}
        for col, _, _ in METRICS:
            row[col] = tail[col].mean()
        rows.append(row)
    data = pd.DataFrame(rows)
    print(data.round(4).to_string(index=False))

    arms = data["arm"].unique().tolist()
    methods = data["method"].unique().tolist()
    x = np.arange(len(arms))
    width = 0.8 / max(len(methods), 1)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4.4))
    if len(METRICS) == 1:
        axes = [axes]
    for ax, (col, title, _) in zip(axes, METRICS):
        for i, method in enumerate(methods):
            sub = data[data["method"] == method].set_index("arm").reindex(arms)
            ax.bar(x + (i - (len(methods) - 1) / 2) * width, sub[col], width=width * 0.9,
                   color=METHOD_COLOR.get(method, INK2), label=method, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(arms, rotation=20, ha="right", fontsize=8.5)
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.grid(axis="y", alpha=0.6)
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(f"Converged performance by budget arm (mean of last {CONVERGED_WINDOW} rollouts)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(out_dir, "grouped_comparison.png")
    fig.savefig(out, dpi=140)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "graphs_grouped"
    generate(out_dir)
