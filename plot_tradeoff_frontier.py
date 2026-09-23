"""
plot_tradeoff_frontier.py - pairwise constraint-cost tradeoff curves, in
the style of the MMSP'25 baseline paper's Fig. 6 (outage vs. power, stall
vs. outage, stall vs. power - each swept across a reward weight) and
ASL360's Figs. 6-7 (metric vs. quality, one point per method).

This project doesn't sweep a reward weight directly - its analogous
"knob" is the budget triple (d_bar, p_bar, b_bar), which is already swept
across several arms. So instead of one continuous curve from one
training run, each point here is one ARM's own converged state (mean
over its last CONVERGED_WINDOW rollouts), and the "frontier" is traced
out across arms - the empirical tradeoff the constrained-RL mechanism
actually reaches, not a synthetic sweep.

Usage: python plot_tradeoff_frontier.py <out_dir>
Edit ARMS below to add/remove runs - each entry is (label, loader, kwargs).
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CONVERGED_WINDOW = 200

# dataviz reference palette (references/palette.md) - color follows the
# entity (method), never rank; categorical slots 1/2/3 (blue/orange/aqua)
# are the ones validated all-pairs for a scatter form.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
METHOD_COLOR = {"PPO": "#2a78d6", "PPO+PDS": "#eb6834", "PPO+PDS (clip100)": "#1baf7a"}

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


def converged_point(df, d_bar, p_bar, b_bar):
    tail = df.tail(CONVERGED_WINDOW)
    return {
        "J_D": tail["eval/J_D"].mean(), "J_P": tail["eval/J_P"].mean(), "J_B": tail["eval/J_B"].mean(),
        "d_bar": d_bar, "p_bar": p_bar, "b_bar": b_bar,
        "feasible_pct": tail["eval/feasible"].mean() * 100,
    }


# Each entry: label shown on the plot, (method for color), pattern, budgets.
# Baseline A (4 budgets) vs clean Run D (ent_coef=0, PDS-GAE) - clean D is
# still training as of this pull, so its point is a snapshot, not a
# final converged value; re-run as it progresses. clip100 variant kept
# separate (own label) rather than averaged in with clean D, since
# whether clipping helps is still an open question - see decision_log.md #26.
ARMS = [
    ("PPO", "0.3/6.5/8",  "runpod_results/dbar_0.3_pbar_6.5_bbar_8.0_seed0/block_v*/progress.csv", (0.3, 6.5, 8.0)),
    ("PPO", "0.12/4/7.4", "runpod_results_summary/02_most_important_runs/dbar_0.12_pbar_4_bbar_7.4_seed0/block_v*/progress.csv", (0.12, 4.0, 7.4)),
    ("PPO", "0.12/4/8",   "runpod_results_v2/v3/dbar_0.12_pbar_4_bbar_8.0_seed0/block_v*/progress.csv", (0.12, 4.0, 8.0)),
    ("PPO", "0.1/6.5/8",  "runpod_results/dbar_0.1_pbar_6.5_bbar_8.0_seed0/block_v*/progress.csv", (0.1, 6.5, 8.0)),
    ("PPO+PDS", "0.3/6.5/8",  "run_D_clean_results/run_D_clean_dbar_0.3_pbar_6.5_bbar_8.0_seed0/progress.csv", (0.3, 6.5, 8.0)),
    ("PPO+PDS", "0.12/4/7.4", "run_D_clean_results/run_D_clean_dbar_0.12_pbar_4_bbar_7.4_seed0/progress.csv", (0.12, 4.0, 7.4)),
    ("PPO+PDS", "0.12/4/8",   "run_D_clean_results/run_D_clean_dbar_0.12_pbar_4_bbar_8.0_seed0/progress.csv", (0.12, 4.0, 8.0)),
    ("PPO+PDS", "0.1/6.5/8",  "run_D_clean_results/run_D_clean_dbar_0.1_pbar_6.5_bbar_8.0_seed0/progress.csv", (0.1, 6.5, 8.0)),
    ("PPO+PDS (clip100)", "0.12/4/7.4", "run_D_clean_results/run_D_clip100_dbar_0.12_pbar_4_bbar_7.4_seed0/progress.csv", (0.12, 4.0, 7.4)),
]

PAIRS = [("J_D", "J_P", r"$J_D$ (stall)", r"$J_P$ (power)"), ("J_D", "J_B", r"$J_D$ (stall)", r"$J_B$ (tiles)"),
         ("J_P", "J_B", r"$J_P$ (power)", r"$J_B$ (tiles)")]


def generate(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    points = []
    for method, label, pattern, (d, p, b) in ARMS:
        df = load_raw_progress(pattern)
        if df is None:
            print(f"SKIP {label}: no data found for {pattern}")
            continue
        pt = converged_point(df, d, p, b)
        pt.update({"method": method, "label": label})
        points.append(pt)
    if not points:
        raise SystemExit("No arms loaded - nothing to plot")
    pts = pd.DataFrame(points)
    print(pts[["label", "method", "J_D", "J_P", "J_B", "feasible_pct"]].round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle(f"Empirical constraint-cost tradeoff frontier (mean of last {CONVERGED_WINDOW} rollouts per arm)",
                 fontsize=11, color=INK)
    methods_seen = pts["method"].unique().tolist()
    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, PAIRS):
        for method in methods_seen:
            sub = pts[pts["method"] == method]
            ax.scatter(sub[xcol], sub[ycol], s=70, color=METHOD_COLOR.get(method, INK2),
                       edgecolors=SURFACE, linewidths=1.5, zorder=3, label=method)
            for _, row in sub.iterrows():
                ax.annotate(row["label"], (row[xcol], row[ycol]), fontsize=7.5, color=MUTED,
                            xytext=(6, 4), textcoords="offset points")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(out_dir, "tradeoff_frontier.png")
    fig.savefig(out, dpi=140)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "graphs_tradeoff"
    generate(out_dir)
