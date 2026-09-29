"""
plot_ve_v1_vs_pds.py - VE v1 (PPO+PDS+VE, pre-duplicate-rejection-sampler
commit) vs PDS on Budget A, on whatever data has been extracted so far.

Primary, matched comparison: VE v1 (6 seeds: 1,2,3,5,6,7) vs Exp1 PDS
(same 6 seeds, FINAL_PART1_EXP/exp1_official/ - the long, mature, already-
finished run). New PDS (two-pass, no VE) only has 2 seeds (1,2) extracted
so far - shown as a separate, clearly-labeled small-n reference, never
blended into the matched 6-seed band (standing rule this project has
repeatedly enforced: never combine mismatched seed counts on one line).

Usage: .venv/Scripts/python.exe plot_ve_v1_vs_pds.py <out_dir>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
METHOD_COLOR = {
    "Exp1 PDS": "#eb6834",       # matches this project's established PPO+PDS orange
    "VE v1": "#1f8a70",          # distinct teal - VE-specific color, new to this script
    "new PDS (n=2)": "#7a6fd1",  # muted purple, signals "exploratory/small-n"
}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
})

VE_ROOT = "EXP2_VE/ve_v1_vs_pds/raw_snapshot/exp2_ve"
EXP1_ROOT = "FINAL_PART1_EXP/exp1_official"

VE_SEEDS = [1, 2, 3, 5, 6, 7]
NEW_PDS_SEEDS = [1, 2]


def load_progress(path):
    df = pd.read_csv(path, dtype=str)
    df = df[df["eval/feasible"] != "eval/feasible"].apply(pd.to_numeric, errors="coerce")
    df = df[df["eval/feasible"].notna()].reset_index(drop=True)
    df["rollout"] = np.arange(1, len(df) + 1)
    return df


def _mean_std_band(ax, x, series_2d, color, label, clip=None):
    mean = series_2d.mean(axis=0)
    std = series_2d.std(axis=0)
    lo, hi = mean - std, mean + std
    if clip is not None:
        lo, hi = np.clip(lo, *clip), np.clip(hi, *clip)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(x, mean, color=color, lw=2.0, label=f"{label} (mean +/- std, n={series_2d.shape[0]})", zorder=3)


def _thin_lines(ax, x, series_2d, color, label):
    """For the n=2 new-PDS reference - a band/std at n=2 is not meaningful
    (matches this project's own established reasoning: a formal CI at n=2
    is misleadingly wide, and std of 2 points is a single number, not a
    real spread estimate) - shown as thin per-seed dashed lines instead,
    clearly distinguishing "too few seeds for a band" from the matched
    6-seed comparison's real mean+/-std bands."""
    for i, row in enumerate(series_2d):
        ax.plot(x, row, color=color, lw=1.1, linestyle="--", alpha=0.75, zorder=2,
                 label=(f"{label} (n={series_2d.shape[0]}, individual seeds)" if i == 0 else None))


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    ve = [load_progress(f"{VE_ROOT}/pdsve_budgetA_dropP_seed{s}/progress.csv") for s in VE_SEEDS]
    exp1 = [load_progress(f"{EXP1_ROOT}/pds_budgetA_dropP_seed{s}/progress.csv") for s in VE_SEEDS]
    newpds = [load_progress(f"{VE_ROOT}/pdsnew_budgetA_dropP_seed{s}/progress.csv") for s in NEW_PDS_SEEDS]

    n = min(min(len(d) for d in ve), min(len(d) for d in exp1), min(len(d) for d in newpds))
    ve = [d.iloc[:n] for d in ve]
    exp1 = [d.iloc[:n] for d in exp1]
    newpds = [d.iloc[:n] for d in newpds]
    print(f"Matched to n={n} rollouts (shortest run across all arms shown).")
    print(f"VE v1 seeds: {VE_SEEDS} (n={len(ve)})  Exp1 PDS seeds: {VE_SEEDS} (n={len(exp1)})  "
          f"new PDS seeds: {NEW_PDS_SEEDS} (n={len(newpds)})")

    x = ve[0]["rollout"].values

    # --- Figure 1: violation over rollouts ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ve_v = np.stack([d["eval/violation"].values for d in ve], axis=0)
    exp1_v = np.stack([d["eval/violation"].values for d in exp1], axis=0)
    newpds_v = np.stack([d["eval/violation"].values for d in newpds], axis=0)
    _mean_std_band(ax, x, exp1_v, METHOD_COLOR["Exp1 PDS"], "Exp1 PDS")
    _mean_std_band(ax, x, ve_v, METHOD_COLOR["VE v1"], "VE v1")
    _thin_lines(ax, x, newpds_v, METHOD_COLOR["new PDS (n=2)"], "new PDS")
    ax.set_xlabel("Rollout")
    ax.set_ylabel("Held-out violation $v$ (lower is better)")
    ax.set_title(f"Budget A: violation over training (matched to {n} rollouts)")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_violation.png"), dpi=150)
    plt.close(fig)

    # --- Figure 2: rolling-50 feasibility ---
    fig, ax = plt.subplots(figsize=(9, 5))
    def roll_feas(d):
        return d["eval/feasible"].rolling(50).mean().values * 100.0
    ve_f = np.stack([roll_feas(d) for d in ve], axis=0)
    exp1_f = np.stack([roll_feas(d) for d in exp1], axis=0)
    newpds_f = np.stack([roll_feas(d) for d in newpds], axis=0)
    _mean_std_band(ax, x, exp1_f, METHOD_COLOR["Exp1 PDS"], "Exp1 PDS", clip=(0.0, 100.0))
    _mean_std_band(ax, x, ve_f, METHOD_COLOR["VE v1"], "VE v1", clip=(0.0, 100.0))
    _thin_lines(ax, x, newpds_f, METHOD_COLOR["new PDS (n=2)"], "new PDS")
    ax.set_xlabel("Rollout")
    ax.set_ylabel("Held-out feasible, rolling-50 mean (%)")
    ax.set_ylim(-3, 103)
    ax.set_title(f"Budget A: rolling feasibility (matched to {n} rollouts)")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_rolling_feasible.png"), dpi=150)
    plt.close(fig)

    # --- Figure 3: per-seed final-100-rollout snapshot (bar) ---
    fig, ax = plt.subplots(figsize=(9, 5))
    labels, vals, colors = [], [], []
    for d, s in zip(ve, VE_SEEDS):
        labels.append(f"VE A{s}"); vals.append(d["eval/violation"].iloc[-100:].mean()); colors.append(METHOD_COLOR["VE v1"])
    for d, s in zip(exp1, VE_SEEDS):
        labels.append(f"Exp1 A{s}"); vals.append(d["eval/violation"].iloc[-100:].mean()); colors.append(METHOD_COLOR["Exp1 PDS"])
    order = np.argsort(vals)
    labels = [labels[i] for i in order]; vals = [vals[i] for i in order]; colors = [colors[i] for i in order]
    ax.barh(labels, vals, color=colors)
    ax.set_xlabel("Mean violation, last 100 rollouts (lower is better)")
    ax.set_title(f"Budget A: current standing per seed, at rollout {n}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_current_standing.png"), dpi=150)
    plt.close(fig)

    print(f"Saved 3 figures to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "EXP2_VE/ve_v1_vs_pds/graphs")
