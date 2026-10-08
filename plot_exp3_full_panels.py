"""
plot_exp3_full_panels.py - the SAME full panel suite plot_official_exp1.py
generates for PPO vs PPO+PDS (2 methods), now used as a reusable
generate_cell() over multiple cells for Budget A / drop-P: one cell with
all 5 methods together, plus one cell per non-PPO method shown ALONE
against PPO (Old PDS+VE vs PPO, VE v2 vs PPO, New PDS vs PPO) - per
explicit request, so each method gets its own full 15-panel comparison
against the PPO baseline, not mixed in with the others.

15 panels per cell, same numbering/content as plot_official_exp1.py, each
drawn as one mean+/-std band per method in that cell (2 bands for a
single-method-vs-PPO cell, 5 for the combined one; panel 13's PDS-
residual-variance panel further restricts to non-PPO methods only, since
PPO has no pds_diag/* columns). Each cell truncated to n = min rollouts
across its OWN methods x 6 seeds (1,2,3,5,6,7 - seed 4 dropped, since the
VE-family arms never ran it and every method in a comparison must share
the identical seed list) - so a method's own cell isn't truncated by
unrelated VE arms it isn't being compared against here.

Usage: .venv/Scripts/python.exe plot_exp3_full_panels.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
METHOD_COLOR = {
    "PPO": "#3b6fa0", "New PDS": "#2f9e44",
    "Old PDS+VE": "#a0479e", "VE v2": "#c99a1e",
}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
})

SEEDS = [1, 2, 3, 5, 6, 7]
D_BAR, P_BAR, B_BAR = 0.3, 6.5, 8.0
DROPPED = frozenset({"P"})
PATHS = {
    # Fresh PPO rerun (launched 2026-09-30, still running) - replaces the
    # original Exp1 PPO run, which was manually stopped at ~4000-4800
    # rollouts with no resumable checkpoint. The rerun has since passed
    # that point on every seed and revealed late-training violation onset
    # on 2/6 seeds (A6, A7) past ~5400-5700 rollouts - PPO is NOT a
    # uniformly-stable baseline at long horizons, see
    # EXP3_PDS_VE_COMPARISON/05_ppo_baseline/README.txt.
    "PPO": "EXP3_PDS_VE_COMPARISON/05_ppo_baseline/raw/ppo_rerun_A{s}_progress.csv",
    "New PDS": "EXP3_PDS_VE_COMPARISON/03_new_pds/metrics/raw/newpds_own_A{s}_progress.csv",
    "Old PDS+VE": "EXP3_PDS_VE_COMPARISON/01_old_pds_plus_ve/metrics/raw/oldve_own_A{s}_progress.csv",
    "VE v2": "EXP3_PDS_VE_COMPARISON/02_ve_v2/metrics/raw/ve2_own_A{s}_progress.csv",
}

CELLS = [
    ("all_4_methods", list(METHOD_COLOR.keys()), "EXP3_PDS_VE_COMPARISON/graphs/full_A_dropP_4methods"),
    ("Old PDS+VE vs PPO", ["Old PDS+VE", "PPO"], "EXP3_PDS_VE_COMPARISON/graphs/01_old_pds_plus_ve"),
    ("VE v2 vs PPO", ["VE v2", "PPO"], "EXP3_PDS_VE_COMPARISON/graphs/02_ve_v2"),
    ("New PDS vs PPO", ["New PDS", "PPO"], "EXP3_PDS_VE_COMPARISON/graphs/03_new_pds"),
]


def load_progress(path):
    df = pd.read_csv(path, dtype=str)
    df = df[df["eval/feasible"] != "eval/feasible"].apply(pd.to_numeric, errors="coerce")
    df = df[df["eval/feasible"].notna()].reset_index(drop=True)
    df["rollout"] = np.arange(1, len(df) + 1)
    return df


def first_rollout_below(df, col, thresh, sustained=20):
    roll = df[col].rolling(sustained).mean()
    hit = roll[roll < thresh]
    return int(hit.index[0]) + 1 if len(hit) else None


def bin_feasible(d, bin_size=20):
    """Mean eval/feasible (%) per non-overlapping bin_size-rollout bin -
    replaces raw per-rollout (too jagged at 6 seeds x binary values)
    without a sliding/rolling window (rejected for panel 15's threshold
    crossings, same "no window smoothing" preference)."""
    v = d["eval/feasible"].values * 100.0
    n_bins = len(v) // bin_size
    trimmed = v[:n_bins * bin_size]
    binned = trimmed.reshape(n_bins, bin_size).mean(axis=1)
    x = (np.arange(n_bins) + 0.5) * bin_size
    return x, binned


def _mean_std_band(ax, x, series_2d, color, label, clip=None):
    mean = series_2d.mean(axis=0)
    std = series_2d.std(axis=0)
    lo, hi = mean - std, mean + std
    if clip is not None:
        lo, hi = np.clip(lo, *clip), np.clip(hi, *clip)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0, zorder=2)
    ax.plot(x, mean, color=color, lw=2.0, label=f"{label} (n={series_2d.shape[0]})", zorder=3)


def _mean_std_band_nan(ax, x, series_2d, color, label):
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(series_2d, axis=0)
        std = np.nanstd(series_2d, axis=0)
    n_defined = np.sum(~np.isnan(series_2d), axis=0)
    mean = np.where(n_defined > 0, mean, np.nan)
    std = np.where(n_defined > 0, std, np.nan)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0, zorder=2)
    ax.plot(x, mean, color=color, lw=2.0, label=f"{label} (up to n={series_2d.shape[0]})", zorder=3)


def plot_multi(ax, data, col, ylabel, methods, budget=None, clip=None):
    for method in methods:
        runs = data[method]
        color = METHOD_COLOR[method]
        x = runs[0]["rollout"]
        vals = np.stack([d[col].values for d in runs], axis=0)
        _mean_std_band(ax, x, vals, color, method, clip=clip)
    if budget is not None:
        ax.axhline(budget, color="#e34948", ls="--", lw=1.0, label=f"budget = {budget}")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("rollout")
    ax.legend(frameon=False, fontsize=7.5)


def generate_cell(title_prefix, methods, out_dir, match_across_methods=True):
    """match_across_methods=True (default): every method truncated to the
    SAME n (the shortest method's own min-seed length) - needed whenever
    methods are being read off a shared x-axis for a literal "at the same
    rollout count" comparison. False: each method keeps its OWN min-seed
    length (still matched across its own 6 seeds, just not across
    methods) - every panel here already loops and plots one method at a
    time with that method's own x (see plot_multi/the per-method loops
    below), so this naturally lets methods end at different points on a
    shared axes without any other change - some lines will just be
    longer than others, which is the point."""
    pds_family = [m for m in methods if m != "PPO"]
    full_data = {method: [load_progress(PATHS[method].format(s=s)) for s in SEEDS] for method in methods}
    if match_across_methods:
        n = min(len(d) for runs in full_data.values() for d in runs)
        data = {method: [d.iloc[:n] for d in full_data[method]] for method in methods}
        print(f"{title_prefix}: {len(methods)} methods, {len(SEEDS)} seeds/method, truncated to n={n}")
    else:
        data = {}
        for method in methods:
            n_m = min(len(d) for d in full_data[method])
            data[method] = [d.iloc[:n_m] for d in full_data[method]]
        lens = {m: len(data[m][0]) for m in methods}
        print(f"{title_prefix}: {len(methods)} methods, {len(SEEDS)} seeds/method, own lengths: {lens}")
    os.makedirs(out_dir, exist_ok=True)

    # 01: violation + rolling feasibility
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    plot_multi(axes[0], data, "eval/violation", "violation", methods)
    axes[0].set_title(f"{title_prefix}: constraint violation (bold=mean, shaded=+/-1 std)")
    for method in methods:
        runs = data[method]
        color = METHOD_COLOR[method]
        # Binned to 20-rollout, non-overlapping windows - raw per-rollout
        # was too jagged/messy at 6 seeds x binary values; a sliding
        # window was rejected earlier (panel 15's threshold crossings),
        # so this bins instead of smoothing.
        xb = None
        ys = []
        for d in runs:
            xb, yb = bin_feasible(d)
            ys.append(yb)
        _mean_std_band(axes[1], xb, np.stack(ys, axis=0), color, method, clip=(0.0, 100.0))
    axes[1].set_ylabel("feasible % (per 20 rollouts)")
    axes[1].set_xlabel("rollout")
    axes[1].legend(frameon=False, fontsize=7.5)
    axes[1].set_title("feasible % (decision-log #10's primary metric)")
    fig.tight_layout(); fig.savefig(f"{out_dir}/01_violation.png"); plt.close(fig)

    # 02: constraint returns vs budgets (D, B - P dropped)
    specs = [("eval/J_D", D_BAR, "J_D (stall)"), ("eval/J_B", B_BAR, "J_B (tiles)")]
    fig, axes = plt.subplots(len(specs), 1, figsize=(11, 3.3 * len(specs)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (col, bud, lbl) in zip(axes, specs):
        plot_multi(ax, data, col, lbl, methods, budget=bud)
    axes[0].set_title(f"{title_prefix}: constraint returns vs budgets")
    fig.tight_layout(); fig.savefig(f"{out_dir}/02_constraint_returns.png"); plt.close(fig)

    # 03: multipliers (mu_D, mu_B - P dropped)
    mu_specs = [("lagrangian/mu_D", "mu_D (stall)"), ("lagrangian/mu_B", "mu_B (tiles)")]
    fig, axes = plt.subplots(len(mu_specs), 1, figsize=(11, 3.3 * len(mu_specs)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (col, lbl) in zip(axes, mu_specs):
        plot_multi(ax, data, col, lbl, methods)
    axes[0].set_title(f"{title_prefix}: Lagrange multipliers")
    fig.tight_layout(); fig.savefig(f"{out_dir}/03_multipliers.png"); plt.close(fig)

    # 04: physical metrics (4-panel)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plot_multi(axes[0, 0], data, "eval/mean_coverage", "coverage (fraction)", methods)
    plot_multi(axes[0, 1], data, "physical/mean_n_enhanced_tiles", "enhanced tiles (of 64)", methods)
    plot_multi(axes[1, 0], data, "eval/mean_stall_sec", "stall time (sec)", methods)
    plot_multi(axes[1, 1], data, "physical/mean_power_mW", "transmit power (mW)", methods)
    fig.suptitle(f"{title_prefix}: physical metrics")
    fig.tight_layout(); fig.savefig(f"{out_dir}/04_physical_metrics.png"); plt.close(fig)

    # 05: reward
    fig, ax = plt.subplots(figsize=(11, 4.5))
    plot_multi(ax, data, "rollout/ep_rew_mean", "mean episode reward", methods)
    ax.set_title(f"{title_prefix}: reward")
    fig.tight_layout(); fig.savefig(f"{out_dir}/05_reward.png"); plt.close(fig)

    # 06: PSNR
    fig, ax = plt.subplots(figsize=(11, 4.5))
    plot_multi(ax, data, "eval/mean_psnr_db", "PSNR (dB)", methods)
    ax.set_title(f"{title_prefix}: viewport PSNR")
    fig.tight_layout(); fig.savefig(f"{out_dir}/06_psnr.png"); plt.close(fig)

    # 07: feasibility, per 20 rollouts (standalone)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for method in methods:
        runs = data[method]
        color = METHOD_COLOR[method]
        xb = None
        ys = []
        for d in runs:
            xb, yb = bin_feasible(d)
            ys.append(yb)
        _mean_std_band(ax, xb, np.stack(ys, axis=0), color, method, clip=(0.0, 100.0))
    ax.set_ylabel("feasible % (per 20 rollouts)")
    ax.set_xlabel("rollout")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title(f"{title_prefix}: feasibility, per 20 rollouts")
    fig.tight_layout(); fig.savefig(f"{out_dir}/07_rolling_feasible.png"); plt.close(fig)

    # 08: policy entropy
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for method in methods:
        runs = data[method]
        color = METHOD_COLOR[method]
        x = runs[0]["rollout"]
        vals = np.stack([-d["train/entropy_loss"].values for d in runs], axis=0)
        _mean_std_band(ax, x, vals, color, method)
    ax.set_ylabel("policy entropy (= -train/entropy_loss)")
    ax.set_xlabel("rollout")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title(f"{title_prefix}: policy entropy")
    fig.tight_layout(); fig.savefig(f"{out_dir}/08_policy_entropy.png"); plt.close(fig)

    # 09: training diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_multi(axes[0], data, "train/explained_variance", "explained variance", methods)
    axes[0].axhline(1.0, color="black", ls="--", lw=0.8)
    plot_multi(axes[1], data, "train/approx_kl", "approx. KL per update", methods)
    fig.suptitle(f"{title_prefix}: training diagnostics")
    fig.tight_layout(); fig.savefig(f"{out_dir}/09_training_diagnostics.png"); plt.close(fig)

    # 10: J_Q vs theoretical max
    episode_len = 36
    j_q_max = sum(config.PPO_GAMMA ** t for t in range(episode_len))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for method in methods:
        runs = data[method]
        color = METHOD_COLOR[method]
        x = runs[0]["rollout"]
        vals = np.stack([d["eval/J_Q"].values for d in runs], axis=0)
        _mean_std_band(ax, x, vals, color, method)
    ax.axhline(j_q_max, color="black", ls="--", lw=0.8, label=f"theoretical max = {j_q_max:.2f}")
    ax.set_ylabel(r"$J_Q$")
    ax.set_xlabel("rollout")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title(f"{title_prefix}: viewport quality return vs. its theoretical max")
    fig.tight_layout(); fig.savefig(f"{out_dir}/10_j_q_vs_max.png"); plt.close(fig)

    # 11: stall time
    fig, ax = plt.subplots(figsize=(11, 4.5))
    plot_multi(ax, data, "eval/mean_stall_sec", "stall time (sec)", methods)
    ax.set_title(f"{title_prefix}: stall time")
    fig.tight_layout(); fig.savefig(f"{out_dir}/11_stall_time.png"); plt.close(fig)

    # 12: extra (discretionary) tiles
    mandatory_avg = 12.52
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for method in methods:
        runs = data[method]
        color = METHOD_COLOR[method]
        x = runs[0]["rollout"]
        vals = np.stack([d["physical/mean_n_enhanced_tiles"].values - mandatory_avg for d in runs], axis=0)
        _mean_std_band(ax, x, vals, color, method)
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set_ylabel("extra tiles (approx.)")
    ax.set_xlabel("rollout")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title(f"{title_prefix}: discretionary extra tiles (enhanced - {mandatory_avg} mandatory avg)")
    fig.tight_layout(); fig.savefig(f"{out_dir}/12_extra_tiles.png"); plt.close(fig)

    # 13: PDS residual std - non-PPO methods in THIS cell only (PPO has no
    # pds_diag/* columns). Only std_delta_pds per method shown (not each
    # method's own ordinary-TD counterfactual too) - see plot_official_exp1.py
    # for why that comparison is a within-run estimator check, not a
    # between-method one, and is dropped here in favor of the latter.
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for method in pds_family:
        runs = data[method]
        color = METHOD_COLOR[method]
        x = runs[0]["rollout"]
        vals = np.stack([d["pds_diag/std_delta_pds"].rolling(20, min_periods=5).mean().values for d in runs], axis=0)
        _mean_std_band(ax, x, vals, color, method)
    ax.set_ylabel("PDS residual std (rolling-20)")
    ax.set_xlabel("rollout")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title(f"{title_prefix}: PDS residual variance, non-PPO methods only (n={len(pds_family)} methods)")
    fig.tight_layout(); fig.savefig(f"{out_dir}/13_pds_residual_variance.png"); plt.close(fig)

    # 14: quality conditioned on feasibility
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for method in methods:
        runs = data[method]
        color = METHOD_COLOR[method]
        x = runs[0]["rollout"]
        psnr_feas = np.stack([d["eval/mean_psnr_db"].where(d["eval/feasible"] == 1)
                               .rolling(100, min_periods=5).mean().values for d in runs], axis=0)
        jq_feas = np.stack([d["eval/J_Q"].where(d["eval/feasible"] == 1)
                             .rolling(100, min_periods=5).mean().values for d in runs], axis=0)
        _mean_std_band_nan(axes[0], x, psnr_feas, color, method)
        _mean_std_band_nan(axes[1], x, jq_feas, color, method)
    axes[0].set_ylabel("PSNR (dB), feasible rollouts only")
    axes[0].set_xlabel("rollout")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].axhline(j_q_max, color="black", ls="--", lw=0.8, label=f"theoretical max = {j_q_max:.2f}")
    axes[1].set_ylabel(r"$J_Q$, feasible rollouts only")
    axes[1].set_xlabel("rollout")
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{title_prefix}: quality conditioned on feasibility (rolling-100 over feasible rows only)")
    fig.tight_layout(); fig.savefig(f"{out_dir}/14_quality_conditioned_on_feasible.png"); plt.close(fig)

    # 15: speed to constraint satisfaction. Uses full_data (each method's
    # OWN un-truncated run length), NOT the shared n used by panels 01-14 -
    # a per-seed first-passage-time statistic doesn't need a common x-axis,
    # and truncating PPO/Exp1 PDS down to a short VE arm's window silently
    # drops their own slower seeds, biasing the mean toward their luckiest
    # ones (verified case: PPO's true viol=0 mean is ~1596, matching PDS
    # being faster than PPO; a truncated version gave a fake ~913 that
    # inverted the comparison). sustained=1: first single rollout whose raw
    # eval/violation crosses thresh, no rolling window - per explicit
    # request, in place of plot_official_exp1.py's usual sustained-20.
    thresholds = [("viol<5", 5.0), ("viol<1", 1.0), ("viol<0.2", 0.2), ("viol=0", 1e-6)]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(thresholds))
    width = 0.8 / len(methods)
    for j, method in enumerate(methods):
        runs = full_data[method]
        color = METHOD_COLOR[method]
        means, seed_vals = [], []
        for _, thresh in thresholds:
            vals = [first_rollout_below(d, "eval/violation", thresh, sustained=1) for d in runs]
            seed_vals.append(vals)
            defined = [v for v in vals if v is not None]
            means.append(np.mean(defined) if defined else np.nan)
        offset = (j - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, means, width, color=color, alpha=0.85, label=f"{method} (n={len(runs)})")
        for xi, vals in zip(x + offset, seed_vals):
            defined = [v for v in vals if v is not None]
            if defined:
                ax.scatter([xi] * len(defined), defined, color=INK, s=10, zorder=5)
            n_missing = len(vals) - len(defined)
            if n_missing:
                ax.annotate(f"{n_missing}/{len(vals)} missed", (xi, 0), xytext=(0, 6),
                            textcoords="offset points", ha="center", fontsize=6.5, color=INK,
                            rotation=90, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
    ax.set_xticks(x); ax.set_xticklabels([t[0] for t in thresholds])
    ax.set_ylabel("rollouts needed (first single rollout to cross)")
    ax.set_title(f"{title_prefix}: speed to constraint satisfaction - first single rollout, not sustained "
                 "(each method's own full run length; lower = faster)")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(f"{out_dir}/15_speed_to_threshold.png"); plt.close(fig)

    print(f"  wrote 15 panels to {out_dir}")


def main():
    for title_prefix, methods, out_dir in CELLS:
        generate_cell(title_prefix, methods, out_dir)


if __name__ == "__main__":
    main()
