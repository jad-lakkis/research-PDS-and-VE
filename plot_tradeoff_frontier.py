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


def converged_point(df, d_bar, p_bar, b_bar):
    tail = df.tail(CONVERGED_WINDOW)
    return {
        "J_D": tail["eval/J_D"].mean(), "J_P": tail["eval/J_P"].mean(), "J_B": tail["eval/J_B"].mean(),
        "d_bar": d_bar, "p_bar": p_bar, "b_bar": b_bar,
        "feasible_pct": tail["eval/feasible"].mean() * 100,
    }


def multiseed_point(paths: list, d_bar, p_bar, b_bar):
    """Same as converged_point, but averaged across several seeds' own
    tail means, with the across-seed std attached for error bars - real
    seed-to-seed spread, not just one run's noise."""
    per_seed = []
    for path in paths:
        df = load_raw_progress(path)
        if df is None:
            continue
        per_seed.append(converged_point(df, d_bar, p_bar, b_bar))
    if not per_seed:
        return None
    seed_df = pd.DataFrame(per_seed)
    out = {col: seed_df[col].mean() for col in ["J_D", "J_P", "J_B", "feasible_pct"]}
    out.update({f"{col}_std": seed_df[col].std() for col in ["J_D", "J_P", "J_B"]})
    out.update({"d_bar": d_bar, "p_bar": p_bar, "b_bar": b_bar, "n_seeds": len(per_seed)})
    return out


# Each entry: label shown on the plot, (method for color), pattern, budgets.
# Baseline A (4 budgets) vs clean Run D (ent_coef=0, PDS-GAE). Clean D is
# still training as of this pull, so its point is a snapshot, not a
# final converged value; re-run as it progresses. clip100 dropped per
# explicit decision - not part of the official comparison.
ARMS = [
    ("PPO", "0.3/6.5/8",  "runpod_results/dbar_0.3_pbar_6.5_bbar_8.0_seed0/block_v*/progress.csv", (0.3, 6.5, 8.0)),
    ("PPO", "0.12/4/7.4", "runpod_results_summary/02_most_important_runs/dbar_0.12_pbar_4_bbar_7.4_seed0/block_v*/progress.csv", (0.12, 4.0, 7.4)),
    ("PPO", "0.12/4/8",   "runpod_results_v2/v3/dbar_0.12_pbar_4_bbar_8.0_seed0/block_v*/progress.csv", (0.12, 4.0, 8.0)),
    ("PPO", "0.1/6.5/8",  "runpod_results/dbar_0.1_pbar_6.5_bbar_8.0_seed0/block_v*/progress.csv", (0.1, 6.5, 8.0)),
    ("PPO+PDS", "0.3/6.5/8",  "run_D_clean_results/run_D_clean_dbar_0.3_pbar_6.5_bbar_8.0_seed0/progress.csv", (0.3, 6.5, 8.0)),
    ("PPO+PDS", "0.12/4/7.4", "run_D_clean_results/run_D_clean_dbar_0.12_pbar_4_bbar_7.4_seed0/progress.csv", (0.12, 4.0, 7.4)),
    ("PPO+PDS", "0.12/4/8",   "run_D_clean_results/run_D_clean_dbar_0.12_pbar_4_bbar_8.0_seed0/progress.csv", (0.12, 4.0, 8.0)),
    ("PPO+PDS", "0.1/6.5/8",  "run_D_clean_results/run_D_clean_dbar_0.1_pbar_6.5_bbar_8.0_seed0/progress.csv", (0.1, 6.5, 8.0)),
]

# Official multi-seed EXP1 cells - (method, label, [progress.csv per seed], budget).
# Plotted with real error bars (std across seeds), distinct from the
# single-seed exploratory ARMS above. A_full (all 3 constraints active)
# was retired from the official design - budget A is now drop-P only, at
# 4 seeds (seeds 3/4 added in its place, total still 14 runs). Both
# methods within a cell always use the SAME seed list - keep it that way,
# see _draw_frontier's docstring for why (the seed-count-mismatch bug).
OFFICIAL_ROOT = "FINAL_PART1_EXP/exp1_official"
OFFICIAL_ARMS = [
    ("PPO", "0.3/6.5/8 dropP (official)",
     [f"{OFFICIAL_ROOT}/ppo_budgetA_dropP_seed{s}/progress.csv" for s in (1, 2, 3, 4)], (0.3, 6.5, 8.0)),
    ("PPO+PDS", "0.3/6.5/8 dropP (official)",
     [f"{OFFICIAL_ROOT}/pds_budgetA_dropP_seed{s}/progress.csv" for s in (1, 2, 3, 4)], (0.3, 6.5, 8.0)),
    ("PPO", "0.1/6.5/8 dropP (official)",
     [f"{OFFICIAL_ROOT}/ppo_budgetB_dropP_seed{s}/progress.csv" for s in (1, 2, 3)], (0.1, 6.5, 8.0)),
    ("PPO+PDS", "0.1/6.5/8 dropP (official)",
     [f"{OFFICIAL_ROOT}/pds_budgetB_dropP_seed{s}/progress.csv" for s in (1, 2, 3)], (0.1, 6.5, 8.0)),
]

PAIRS = [("J_D", "J_P", r"$J_D$ (stall)", r"$J_P$ (power)"), ("J_D", "J_B", r"$J_D$ (stall)", r"$J_B$ (tiles)"),
         ("J_P", "J_B", r"$J_P$ (power)", r"$J_B$ (tiles)")]


def _draw_frontier(pts, title, use_errorbars, out_path):
    """One self-contained figure. Every method plotted here uses the SAME
    marker style and the SAME seed provenance as every other method on the
    same axes - never hollow-vs-filled or errorbar-vs-no-errorbar mixed on
    one chart. That mixing (single-seed exploratory points next to
    multi-seed official points with error bars, for what looks like the
    same comparison) is exactly the "one method has more seeds than the
    other on the same graph" bug the user flagged - fixed by never letting
    both provenances share an Axes at all, not by styling them differently."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle(title, fontsize=10.5, color=INK)
    methods_seen = pts["method"].unique().tolist()
    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, PAIRS):
        for method in methods_seen:
            sub = pts[pts["method"] == method]
            if not len(sub):
                continue
            if use_errorbars:
                xerr = sub.get(f"{xcol}_std")
                yerr = sub.get(f"{ycol}_std")
                ax.errorbar(sub[xcol], sub[ycol], xerr=xerr, yerr=yerr, fmt="o",
                            ms=9, color=METHOD_COLOR.get(method, INK2), ecolor=METHOD_COLOR.get(method, INK2),
                            elinewidth=1.2, capsize=3, zorder=4, label=method)
            else:
                ax.scatter(sub[xcol], sub[ycol], s=55, facecolors="none",
                           edgecolors=METHOD_COLOR.get(method, INK2), linewidths=1.5, zorder=3,
                           label=method)
            for _, row in sub.iterrows():
                ax.annotate(row["label"], (row[xcol], row[ycol]), fontsize=7, color=MUTED,
                            xytext=(6, 4), textcoords="offset points")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")


def generate(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    exploratory_points = []
    for method, label, pattern, (d, p, b) in ARMS:
        df = load_raw_progress(pattern)
        if df is None:
            print(f"SKIP {label}: no data found for {pattern}")
            continue
        pt = converged_point(df, d, p, b)
        pt.update({"method": method, "label": label})
        exploratory_points.append(pt)

    official_points = []
    for method, label, paths, (d, p, b) in OFFICIAL_ARMS:
        pt = multiseed_point(paths, d, p, b)
        if pt is None:
            print(f"SKIP {label} ({method}): no data found")
            continue
        pt.update({"method": method, "label": label})
        official_points.append(pt)

    if not exploratory_points and not official_points:
        raise SystemExit("No arms loaded - nothing to plot")

    # Two fully separate figures, never combined on one Axes. Official
    # cells are matched seed-for-seed by construction (see CELLS/OFFICIAL_ARMS
    # above and plot_official_exp1.py) - both methods in a cell always use
    # the same seed list, so this figure is always an apples-to-apples
    # comparison. The old combined chart put exploratory single-seed points
    # (no error bars) and official multi-seed points (with error bars) on
    # the same axes for what read as "the same" budget - that's the bug.
    if official_points:
        off = pd.DataFrame(official_points)
        print("Official (multi-seed, matched seed counts per cell):")
        print(off[["label", "method", "n_seeds", "J_D", "J_P", "J_B", "feasible_pct"]].round(3).to_string(index=False))
        _draw_frontier(
            off,
            f"Empirical constraint-cost tradeoff frontier - official multi-seed runs only "
            f"(mean of last {CONVERGED_WINDOW} rollouts per arm; error bars = std across seeds, "
            f"matched seed count per method within each cell)",
            use_errorbars=True,
            out_path=os.path.join(out_dir, "tradeoff_frontier_official.png"),
        )
    else:
        print("SKIP official frontier figure: no official arms loaded")

    if exploratory_points:
        exp = pd.DataFrame(exploratory_points)
        print("\nExploratory (single seed each, preliminary - not paper data):")
        print(exp[["label", "method", "J_D", "J_P", "J_B", "feasible_pct"]].round(3).to_string(index=False))
        _draw_frontier(
            exp,
            f"Empirical constraint-cost tradeoff frontier - exploratory single-seed runs "
            f"(mean of last {CONVERGED_WINDOW} rollouts per arm; PRELIMINARY, one seed per point - "
            f"superseded by the official multi-seed figure wherever a matching cell exists there)",
            use_errorbars=False,
            out_path=os.path.join(out_dir, "tradeoff_frontier_exploratory.png"),
        )
    else:
        print("SKIP exploratory frontier figure: no exploratory arms loaded")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "graphs_tradeoff"
    generate(out_dir)
