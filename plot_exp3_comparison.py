"""
plot_exp3_comparison.py - graphs for EXP3_PDS_VE_COMPARISON: old PDS+VE,
VE v2, new PDS (no VE), and the B_VE/T hyperparameter variants, each
against Exp1 PDS / new PDS as relevant, plus PPO everywhere, and one
combined "all methods" figure. Same conventions as plot_ve_v1_vs_pds.py
(SURFACE/INK palette, mean+/-std bands for n=6-seed groups, thin dashed
lines for n=1 seed - never a std band at n=1).

Usage: .venv/Scripts/python.exe plot_exp3_comparison.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {
    "PPO": "#3b6fa0",
    "Exp1 PDS": "#eb6834",
    "New PDS": "#2f9e44",
    "Old PDS+VE": "#a0479e",
    "VE v2": "#c99a1e",
    "Base (B_VE=8,T=10)": "#a0479e",
    "B_VE=25,T=10": "#c9762b",
    "B_VE=10,T=5": "#4a6fb5",
}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
})

SEEDS = [1, 2, 3, 5, 6, 7]
# Fresh PPO rerun (replaces the original Exp1 PPO run, which was manually
# stopped at ~4000-4800 rollouts with no resumable checkpoint). The rerun
# has since passed that point on every seed and revealed late-training
# violation onset on 2/6 seeds (A6, A7) past ~5400-5700 rollouts - see
# EXP3_PDS_VE_COMPARISON/05_ppo_baseline/README.txt.
PPO_ROOT = "EXP3_PDS_VE_COMPARISON/05_ppo_baseline/raw"
NEWPDS_ROOT = "EXP3_PDS_VE_COMPARISON/03_new_pds/metrics/raw"
OLDVE_ROOT = "EXP3_PDS_VE_COMPARISON/01_old_pds_plus_ve/metrics/raw"
VE2_ROOT = "EXP3_PDS_VE_COMPARISON/02_ve_v2/metrics/raw"
VAR_ROOT = "EXP3_PDS_VE_COMPARISON/04_ve_hyperparameter_variants/metrics/raw"
CKPT_CSV = "EXP3_PDS_VE_COMPARISON/03_new_pds/metrics/per_rollout_checkpoints.csv"
OUT_ROOT = "EXP3_PDS_VE_COMPARISON/graphs"


def load(path):
    df = pd.read_csv(path, dtype=str)
    df = df[df["eval/feasible"] != "eval/feasible"].apply(pd.to_numeric, errors="coerce")
    df = df[df["eval/feasible"].notna()].reset_index(drop=True)
    df["rollout"] = np.arange(1, len(df) + 1)
    return df


def mean_std_band(ax, x, series_2d, color, label, clip=None):
    mean = series_2d.mean(axis=0)
    std = series_2d.std(axis=0)
    lo, hi = mean - std, mean + std
    if clip is not None:
        lo, hi = np.clip(lo, *clip), np.clip(hi, *clip)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0, zorder=2)
    ax.plot(x, mean, color=color, lw=2.0, label=f"{label} (n={series_2d.shape[0]})", zorder=3)


def thin_line(ax, x, series_1d, color, label):
    ax.plot(x, series_1d, color=color, lw=1.3, linestyle="--", alpha=0.85, zorder=3, label=f"{label} (n=1)")


def bin_feasible(d, n, bin_size=20):
    """Mean eval/feasible (%) in each non-overlapping bin_size-rollout bin
    - per explicit request, replacing raw per-rollout (too jagged/messy at
    6 seeds x binary values) without going back to a sliding/rolling mean
    (rejected earlier for panel 15's threshold crossings, same "no
    smoothing over a window" preference). Non-overlapping bins, not a
    rolling window - each point is an independent, unduplicated 20-rollout
    slice. Trailing partial bin (n % bin_size rows) is dropped."""
    v = d["eval/feasible"].values[:n] * 100.0
    n_bins = len(v) // bin_size
    trimmed = v[:n_bins * bin_size]
    binned = trimmed.reshape(n_bins, bin_size).mean(axis=1)
    x = (np.arange(n_bins) + 0.5) * bin_size
    return x, binned


def violation_fig(groups, n, title, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(1, n + 1)
    for label, dfs in groups.items():
        color = COLOR[label]
        if len(dfs) == 1:
            thin_line(ax, x, dfs[0]["eval/violation"].values[:n], color, label)
        else:
            stacked = np.stack([d["eval/violation"].values[:n] for d in dfs], axis=0)
            mean_std_band(ax, x, stacked, color, label)
    ax.set_xlabel("Rollout")
    ax.set_ylabel("Held-out violation $v$ (lower is better)")
    ax.set_title(f"{title} (matched to {n} rollouts)")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def feasibility_fig(groups, n, title, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, dfs in groups.items():
        color = COLOR[label]
        if len(dfs) == 1:
            xb, yb = bin_feasible(dfs[0], n)
            ax.plot(xb, yb, color=color, lw=1.3, linestyle="--", alpha=0.85, zorder=3, label=f"{label} (n=1)")
        else:
            xb = None
            ys = []
            for d in dfs:
                xb, yb = bin_feasible(d, n)
                ys.append(yb)
            mean_std_band(ax, xb, np.stack(ys, axis=0), color, label, clip=(0.0, 100.0))
    ax.set_xlabel("Rollout")
    ax.set_ylabel("Held-out feasible, per 20 rollouts (%)")
    ax.set_ylim(-3, 103)
    ax.set_title(f"{title} (matched to {n} rollouts)")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_section(out_dir, groups, title_prefix):
    os.makedirs(out_dir, exist_ok=True)
    multi_seed_dfs = [d for dfs in groups.values() if len(dfs) > 1 for d in dfs]
    single_seed_dfs = [dfs[0] for dfs in groups.values() if len(dfs) == 1]
    n = min(len(d) for d in multi_seed_dfs + single_seed_dfs)
    violation_fig(groups, n, f"{title_prefix}: violation over training", os.path.join(out_dir, "01_violation.png"))
    feasibility_fig(groups, n, f"{title_prefix}: rolling feasibility", os.path.join(out_dir, "02_feasibility.png"))
    print(f"{title_prefix}: matched n={n}, saved to {out_dir}")
    return n


def main():
    ppo = {s: load(f"{PPO_ROOT}/ppo_rerun_A{s}_progress.csv") for s in SEEDS}
    newpds = {s: load(f"{NEWPDS_ROOT}/newpds_own_A{s}_progress.csv") for s in SEEDS}
    oldve = {s: load(f"{OLDVE_ROOT}/oldve_own_A{s}_progress.csv") for s in SEEDS}
    ve2 = {s: load(f"{VE2_ROOT}/ve2_own_A{s}_progress.csv") for s in SEEDS}

    def L(d):
        return [d[s] for s in SEEDS]

    # 01, 02, 03: no longer generated here - plot_exp3_full_panels.py now
    # writes its full 15-panel suite (panel 01 = violation+feasibility
    # combined, panel 07 = feasibility standalone) directly into each of
    # these same folders, superseding the simpler 2-file version this
    # script used to write there. Run that script for those.

    # --- 03's late-training checkpoint chart: New PDS vs Exp1 PDS, a
    # different, already-established comparison about the late-training
    # crossover, not a "vs PPO" chart, so it's kept as its own file -
    # numbered 16_ (not 03_) so it can't collide with the full-panel
    # suite's own 01-15 files living in this same folder. ---
    ck = pd.read_csv(CKPT_CSV)
    pts = sorted(ck["rollout_point"].unique())
    new_mean = [ck.loc[ck.rollout_point == p, "new_pds_stays_feasible_pct"].mean() for p in pts]
    new_n = [ck.loc[ck.rollout_point == p, "new_pds_stays_feasible_pct"].notna().sum() for p in pts]
    exp1_mean = [ck.loc[ck.rollout_point == p, "exp1_pds_stays_feasible_pct"].mean() for p in pts]
    exp1_n = [ck.loc[ck.rollout_point == p, "exp1_pds_stays_feasible_pct"].notna().sum() for p in pts]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(pts, new_mean, color=COLOR["New PDS"], lw=2.0, marker="o", ms=6,
             label="New PDS (mean across seeds that reached this point)")
    ax.plot(pts, exp1_mean, color=COLOR["Exp1 PDS"], lw=2.0, marker="o", ms=6,
             label="Exp1 PDS (mean across seeds that reached this point)")
    for p, nn, ne in zip(pts, new_n, exp1_n):
        ax.annotate(f"n={nn}/{ne}", (p, 3), fontsize=7, color=MUTED, ha="center")
    ax.set_xlabel("Rollout checkpoint")
    ax.set_ylabel("Stays-feasible %, 100 rollouts before checkpoint")
    ax.set_ylim(-3, 103)
    ax.set_title("New PDS vs Exp1 PDS: late-training feasibility (varying n per point - see labels)")
    ax.legend(loc="center left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    os.makedirs(f"{OUT_ROOT}/03_new_pds", exist_ok=True)
    fig.savefig(f"{OUT_ROOT}/03_new_pds/16_late_training_checkpoints.png", dpi=150)
    plt.close(fig)

    # --- 04: B_VE/T hyperparameter variants, seed A2 only, vs PPO (n=1,
    # thin lines) - the 3 variants stay together (they're what this
    # section is about, not "other methods"), PPO replaces Exp1 PDS as
    # the one reference line ---
    out4 = f"{OUT_ROOT}/04_hyperparameter_variants"
    os.makedirs(out4, exist_ok=True)
    base = load(f"{VAR_ROOT}/base_bve8_t10_A2_progress.csv")
    v25 = load(f"{VAR_ROOT}/variant_bve25_t10_A2_progress.csv")
    v10t5 = load(f"{VAR_ROOT}/variant_bve10_t5_A2_progress.csv")
    ppo_a2 = ppo[2]
    n4 = min(len(base), len(v25), len(v10t5), len(ppo_a2))
    groups4 = {
        "Base (B_VE=8,T=10)": [base], "B_VE=25,T=10": [v25], "B_VE=10,T=5": [v10t5], "PPO": [ppo_a2],
    }
    violation_fig(groups4, n4, "B_VE/T variants (seed A2 only)", f"{out4}/01_violation.png")
    feasibility_fig(groups4, n4, "B_VE/T variants (seed A2 only)", f"{out4}/02_feasibility.png")
    print(f"04 variants: matched n={n4}, saved to {out4}")

    # --- 05: everything together vs PPO, 6-seed matched ---
    all_multi = L(ppo) + L(newpds) + L(oldve) + L(ve2)
    n5 = min(len(d) for d in all_multi)
    groups5 = {
        "PPO": L(ppo), "New PDS": L(newpds),
        "Old PDS+VE": L(oldve), "VE v2": L(ve2),
    }
    out5 = f"{OUT_ROOT}/05_all_methods_vs_ppo"
    os.makedirs(out5, exist_ok=True)
    violation_fig(groups5, n5, "All methods vs PPO", f"{out5}/01_violation.png")
    feasibility_fig(groups5, n5, "All methods vs PPO", f"{out5}/02_feasibility.png")
    print(f"05 all-methods: matched n={n5}, saved to {out5}")


if __name__ == "__main__":
    main()
