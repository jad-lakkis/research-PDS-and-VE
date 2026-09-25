"""
plot_official_exp1.py - multi-seed PPO vs PPO+PDS comparison for the
official EXP1 experiment set (FINAL_PART1_EXP/exp1_official/), replacing
the incomplete 3-panel pairs/ folder (violation, multipliers, rolling-
feasible only) with the full panel set PLUS real seed-to-seed spread,
which single-seed comparisons earlier in this project couldn't show.

Each cell (budget x constraint-set) has 2-4 seeds per method. Every panel
shows a bold mean line per method plus a shaded +/-1 std band across
seeds (real seed-to-seed spread, not hidden by the average) - not
individual per-seed lines, which mostly add clutter at this few seeds
per cell. Truncated to the shortest run's rollout count within a cell,
same convention as every other comparison script in this project.

Usage: python plot_official_exp1.py <out_dir>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
METHOD_COLOR = {"PPO": "#2a78d6", "PPO+PDS": "#eb6834"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
})

# The all-3-constraint cell (A_full) was retired from the official design -
# budget A runs drop-P only. Full seed set as of 2026-09-25: all 7 A seeds
# and all 5 B seeds are mature (3559-4984 rollouts each, checked before
# this update - no more "too early, exclude" arms). A_full's already-
# completed data (seeds 1-2) is left on disk, just no longer part of the
# official multi-seed comparison - see EXP1_D_VS_A/summary's exploratory
# frontier if that historical data is ever needed again.
#
# IMPORTANT (repeatedly flagged by the user): ppo/pds seed LISTS within a
# cell must always be IDENTICAL - never let one method have more seeds
# than the other in the same comparison. That was a real, fixed bug once
# already (plot_tradeoff_frontier.py's old combined chart); keep it fixed.
ROOT = "FINAL_PART1_EXP/exp1_official"
CELLS = {
    "A_dropP": {"budget": (0.3, 6.5, 8.0), "seeds": [1, 2, 3, 4, 5, 6, 7],
                "ppo": "ppo_budgetA_dropP_seed{}", "pds": "pds_budgetA_dropP_seed{}",
                "dropped": frozenset({"P"})},
    "B_dropP": {"budget": (0.1, 6.5, 8.0), "seeds": [1, 2, 3, 4, 5],
                "ppo": "ppo_budgetB_dropP_seed{}", "pds": "pds_budgetB_dropP_seed{}",
                "dropped": frozenset({"P"})},
}


def load_progress(path):
    df = pd.read_csv(path, dtype=str)
    df = df[df["eval/feasible"] != "eval/feasible"].apply(pd.to_numeric, errors="coerce")
    df = df[df["eval/feasible"].notna()].reset_index(drop=True)
    df["rollout"] = np.arange(1, len(df) + 1)
    return df


def load_cell(spec):
    ppo = [load_progress(f"{ROOT}/{spec['ppo'].format(s)}/progress.csv") for s in spec["seeds"]]
    pds = [load_progress(f"{ROOT}/{spec['pds'].format(s)}/progress.csv") for s in spec["seeds"]]
    n = min(min(len(d) for d in ppo), min(len(d) for d in pds))
    ppo = [d.iloc[:n] for d in ppo]
    pds = [d.iloc[:n] for d in pds]
    return ppo, pds, n


def first_rollout_below(df, col, thresh, sustained=20):
    """First rollout (1-based) where a rolling-`sustained` mean of `col`
    drops under `thresh` and stays there - requiring a sustained window
    (not just one lucky dip) before crediting "reached," same convention
    used throughout this project's diagnostics. None if it never happens
    within the truncated window."""
    roll = df[col].rolling(sustained).mean()
    hit = roll[roll < thresh]
    return int(hit.index[0]) + 1 if len(hit) else None


def _mean_std_band_nan(ax, x, series_2d, color, label):
    """Same as _mean_std_band but NaN-aware (np.nanmean/nanstd) - for
    panels conditioned on feasibility, where a seed with zero feasible
    rows in a given window is genuinely undefined there, not zero. Plain
    .mean()/.std() would propagate a single seed's NaN into the whole
    band; nan-aware stats instead average over whichever seeds have data
    at that rollout. A point is only plotted where at least one seed has
    a real value."""
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(series_2d, axis=0)
        std = np.nanstd(series_2d, axis=0)
    n_defined = np.sum(~np.isnan(series_2d), axis=0)
    mean = np.where(n_defined > 0, mean, np.nan)
    std = np.where(n_defined > 0, std, np.nan)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(x, mean, color=color, lw=2.0, label=f"{label} (mean +/- std, up to n={series_2d.shape[0]})", zorder=3)


def _mean_std_band(ax, x, series_2d, color, label, clip=None):
    """series_2d: (n_seeds, n_rollouts). Draws a bold mean line + a shaded
    +/-1 std band across seeds, in place of one thin line per seed - with
    only 2-4 seeds/cell, individual raw lines add clutter without adding
    real signal, and a formal 95% CI would be misleadingly wide at n=2 (a
    t-based CI needs t_{0.975,df=1}~=12.7, ballooning the band far past
    what 2 points actually support) - std is the honest, standard choice
    for this few seeds (matches common RL-benchmark plotting convention,
    e.g. SB3/Spinning Up style)."""
    mean = series_2d.mean(axis=0)
    std = series_2d.std(axis=0)
    lo, hi = mean - std, mean + std
    if clip is not None:
        lo, hi = np.clip(lo, *clip), np.clip(hi, *clip)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
    ax.plot(x, mean, color=color, lw=2.0, label=f"{label} (mean +/- std, n={series_2d.shape[0]})", zorder=3)


def plot_panel(ax, ppo, pds, col, ylabel, budget=None):
    for runs, method in [(ppo, "PPO"), (pds, "PPO+PDS")]:
        color = METHOD_COLOR[method]
        x = runs[0]["rollout"]
        vals = np.stack([d[col].values for d in runs], axis=0)
        _mean_std_band(ax, x, vals, color, method)
    if budget is not None:
        ax.axhline(budget, color="#e34948", ls="--", lw=1.0, label=f"budget = {budget}")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("rollout")


def generate_cell(cell_name, spec, out_dir):
    d_bar, p_bar, b_bar = spec["budget"]
    ppo, pds, n = load_cell(spec)
    cell_dir = os.path.join(out_dir, cell_name)
    os.makedirs(cell_dir, exist_ok=True)

    dropped_note = f" (dropped: {','.join(sorted(spec['dropped']))})" if spec["dropped"] else ""
    print(f"{cell_name}: budget {d_bar}/{p_bar}/{b_bar}{dropped_note}, "
          f"{len(spec['seeds'])} seeds/method, truncated to n={n}")

    # Feasibility paired directly with violation - same "going up while
    # violation goes down" view added to plot_pds_graphs.py's per-arm
    # panel the same day, kept consistent between both pipelines. 07 below
    # still has feasibility as its own standalone panel too - not
    # redundant, just a different framing (paired-with-violation here,
    # deep-dive on its own there).
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    plot_panel(axes[0], ppo, pds, "eval/violation", "violation")
    axes[0].set_title(f"{cell_name}: constraint violation (bold=mean, shaded=+/-1 std)")
    for runs, method in [(ppo, "PPO"), (pds, "PPO+PDS")]:
        color = METHOD_COLOR[method]
        x = runs[0]["rollout"]
        roll = np.stack([d["eval/feasible"].rolling(50, min_periods=10).mean().values * 100 for d in runs], axis=0)
        _mean_std_band(axes[1], x, roll, color, method, clip=(0.0, 100.0))
    axes[1].set_ylabel("feasible % (rolling-50)")
    axes[1].set_xlabel("rollout")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title(f"{cell_name}: feasible % - decision-log #10's primary metric")
    fig.tight_layout(); fig.savefig(os.path.join(cell_dir, "01_violation.png")); plt.close(fig)

    # A dropped constraint gets NO subplot here, not just a hidden budget
    # line - showing J_P's raw trend next to a "budget" axis label reads
    # as an active constraint even without the red line. Real bug, caught
    # in FINAL_PART1_EXP's own auto-generated per-run graphs (same issue,
    # fixed in plot_pds_graphs.py's plot_constraint_returns_vs_budgets()).
    all_specs = [("eval/J_D", d_bar, "J_D (stall)", "D"), ("eval/J_P", p_bar, "J_P (power)", "P"),
                 ("eval/J_B", b_bar, "J_B (tiles)", "B")]
    active_specs = [(col, bud, lbl) for col, bud, lbl, code in all_specs if code not in spec["dropped"]]
    fig, axes = plt.subplots(len(active_specs), 1, figsize=(11, 3.3 * len(active_specs)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (col, bud, lbl) in zip(axes, active_specs):
        plot_panel(ax, ppo, pds, col, lbl, bud)
    axes[0].set_title(f"{cell_name}: constraint returns vs budgets")
    fig.tight_layout(); fig.savefig(os.path.join(cell_dir, "02_constraint_returns.png")); plt.close(fig)

    # Same "no trace at all" rule as the constraint-returns panel above -
    # a dropped multiplier is excluded, not shown pinned at 0.
    mu_specs = [(c, l, code) for c, l, code in
                [("lagrangian/mu_D", "mu_D (stall)", "D"), ("lagrangian/mu_P", "mu_P (power)", "P"),
                 ("lagrangian/mu_B", "mu_B (tiles)", "B")] if code not in spec["dropped"]]
    fig, axes = plt.subplots(len(mu_specs), 1, figsize=(11, 3.3 * len(mu_specs)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (col, lbl, _) in zip(axes, mu_specs):
        plot_panel(ax, ppo, pds, col, lbl)
    axes[0].set_title(f"{cell_name}: Lagrange multipliers")
    fig.tight_layout(); fig.savefig(os.path.join(cell_dir, "03_multipliers.png")); plt.close(fig)

    # Full 4-panel version (stall + power added) - matches the fix applied
    # to plot_pds_graphs.py's per-arm version the same day; these columns
    # were always in progress.csv, just not wired into this panel before.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    plot_panel(axes[0, 0], ppo, pds, "eval/mean_coverage", "coverage (fraction)")
    plot_panel(axes[0, 1], ppo, pds, "physical/mean_n_enhanced_tiles", "enhanced tiles (of 64)")
    plot_panel(axes[1, 0], ppo, pds, "eval/mean_stall_sec", "stall time (sec)")
    plot_panel(axes[1, 1], ppo, pds, "physical/mean_power_mW", "transmit power (mW)")
    fig.suptitle(f"{cell_name}: physical metrics")
    fig.tight_layout(); fig.savefig(os.path.join(cell_dir, "04_physical_metrics.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    plot_panel(ax, ppo, pds, "rollout/ep_rew_mean", "mean episode reward")
    ax.set_title(f"{cell_name}: reward")
    fig.tight_layout(); fig.savefig(os.path.join(cell_dir, "05_reward.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    plot_panel(ax, ppo, pds, "eval/mean_psnr_db", "PSNR (dB)")
    ax.set_title(f"{cell_name}: viewport PSNR (now available for both methods)")
    fig.tight_layout(); fig.savefig(os.path.join(cell_dir, "06_psnr.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for runs, method in [(ppo, "PPO"), (pds, "PPO+PDS")]:
        color = METHOD_COLOR[method]
        x = runs[0]["rollout"]
        roll = np.stack([d["eval/feasible"].rolling(50, min_periods=10).mean().values * 100 for d in runs], axis=0)
        _mean_std_band(ax, x, roll, color, method, clip=(0.0, 100.0))
    ax.set_ylabel("feasible % (rolling-50)")
    ax.set_xlabel("rollout")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{cell_name}: rolling feasibility")
    fig.tight_layout(); fig.savefig(os.path.join(cell_dir, "07_rolling_feasible.png")); plt.close(fig)

    # Panels 08-11: same 4 new metrics added to plot_pds_graphs.py's
    # per-arm output the same day (train/* columns exist for both PPO and
    # PPO+PDS, no skip logic needed here).
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for runs, method in [(ppo, "PPO"), (pds, "PPO+PDS")]:
        color = METHOD_COLOR[method]
        x = runs[0]["rollout"]
        vals = np.stack([-d["train/entropy_loss"].values for d in runs], axis=0)
        _mean_std_band(ax, x, vals, color, method)
    ax.set_ylabel("policy entropy (= -train/entropy_loss)")
    ax.set_xlabel("rollout")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{cell_name}: policy entropy")
    fig.tight_layout(); fig.savefig(os.path.join(cell_dir, "08_policy_entropy.png")); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_panel(axes[0], ppo, pds, "train/explained_variance", "explained variance")
    axes[0].axhline(1.0, color="black", ls="--", lw=0.8)
    plot_panel(axes[1], ppo, pds, "train/approx_kl", "approx. KL per update")
    fig.suptitle(f"{cell_name}: training diagnostics")
    fig.tight_layout(); fig.savefig(os.path.join(cell_dir, "09_training_diagnostics.png")); plt.close(fig)

    # Same theoretical-max formula as plot_pds_graphs.py's plot_j_q_vs_max
    # (cost_Q^t in [0,1] every step, 36-step episode).
    episode_len = 36
    j_q_max = sum(config.PPO_GAMMA ** t for t in range(episode_len))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for runs, method in [(ppo, "PPO"), (pds, "PPO+PDS")]:
        color = METHOD_COLOR[method]
        x = runs[0]["rollout"]
        vals = np.stack([d["eval/J_Q"].values for d in runs], axis=0)
        _mean_std_band(ax, x, vals, color, method)
    ax.axhline(j_q_max, color="black", ls="--", lw=0.8, label=f"theoretical max = {j_q_max:.2f}")
    ax.set_ylabel(r"$J_Q$")
    ax.set_xlabel("rollout")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{cell_name}: viewport quality return vs. its theoretical max")
    fig.tight_layout(); fig.savefig(os.path.join(cell_dir, "10_j_q_vs_max.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    plot_panel(ax, ppo, pds, "eval/mean_stall_sec", "stall time (sec)")
    ax.set_title(f"{cell_name}: stall time")
    fig.tight_layout(); fig.savefig(os.path.join(cell_dir, "11_stall_time.png")); plt.close(fig)

    # Same mandatory-tile-count baseline as plot_pds_graphs.py's
    # plot_extra_tiles_approx, so the two pipelines' "extra tiles" reads
    # match - total enhanced minus the ~12.52-tile mandatory-viewport
    # average, isolating the agent's own discretionary picks.
    mandatory_avg = 12.52
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for runs, method in [(ppo, "PPO"), (pds, "PPO+PDS")]:
        color = METHOD_COLOR[method]
        x = runs[0]["rollout"]
        vals = np.stack([d["physical/mean_n_enhanced_tiles"].values - mandatory_avg for d in runs], axis=0)
        _mean_std_band(ax, x, vals, color, method)
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set_ylabel("extra tiles (approx.)")
    ax.set_xlabel("rollout")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{cell_name}: discretionary extra tiles (enhanced - {mandatory_avg} mandatory avg)")
    fig.tight_layout(); fig.savefig(os.path.join(cell_dir, "12_extra_tiles.png")); plt.close(fig)

    # PDS residual variance vs. the ordinary-TD counterfactual, PDS arms
    # only - PPO has no pds_diag/* columns at all (it never computes a PDS
    # residual), so there is no PPO line here. Both lines come from the
    # SAME PDS rollouts: std_delta_pds is the real PDS residual actually
    # used; std_delta_ordinary is a zero-extra-cost counterfactual ordinary
    # TD residual computed on the identical rollout (pds_buffer.py's
    # _compute_variance_diagnostics) - this is a within-PDS-run comparison
    # of two estimators, not a PPO-vs-PDS method comparison, so it's
    # plotted separately from the METHOD_COLOR convention used everywhere
    # else in this file to avoid implying otherwise. Lightly smoothed
    # (rolling-20) since the raw per-rollout values are noisy estimates.
    fig, ax = plt.subplots(figsize=(11, 4.5))
    smooth_pds = np.stack([d["pds_diag/std_delta_pds"].rolling(20, min_periods=5).mean().values for d in pds], axis=0)
    smooth_ord = np.stack([d["pds_diag/std_delta_ordinary"].rolling(20, min_periods=5).mean().values for d in pds], axis=0)
    x = pds[0]["rollout"]
    _mean_std_band(ax, x, smooth_pds, "#eb6834", "PDS residual (std_delta_pds)")
    _mean_std_band(ax, x, smooth_ord, "#6b6b6b", "ordinary-TD counterfactual, same rollout (std_delta_ordinary)")
    ax.set_ylabel("residual std (rolling-20)")
    ax.set_xlabel("rollout")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{cell_name}: PDS residual vs. ordinary-TD counterfactual variance (PDS arms only, n={len(pds)})")
    fig.tight_layout(); fig.savefig(os.path.join(cell_dir, "13_pds_residual_variance.png")); plt.close(fig)
    n_panels = 14

    # PSNR / J_Q conditioned on eval/feasible==1 - the unconditional 06/10
    # panels mix in infeasible rollouts and muddy the quality-among-feasible
    # edge found earlier this session. NaN-masking a rollout's value when
    # that seed wasn't feasible there, before the rolling mean, means the
    # rolling window naturally averages only feasible rows while keeping
    # true rollout position on the x-axis (see _mean_std_band_nan).
    episode_len = 36
    j_q_max = sum(config.PPO_GAMMA ** t for t in range(episode_len))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for runs, method in [(ppo, "PPO"), (pds, "PPO+PDS")]:
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
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(j_q_max, color="black", ls="--", lw=0.8, label=f"theoretical max = {j_q_max:.2f}")
    axes[1].set_ylabel(r"$J_Q$, feasible rollouts only")
    axes[1].set_xlabel("rollout")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{cell_name}: quality conditioned on feasibility (rolling-100 over feasible rows only; "
                 f"blank where no seed had a feasible row in that window)")
    fig.tight_layout()
    fig.savefig(os.path.join(cell_dir, f"{n_panels}_quality_conditioned_on_feasible.png"))
    plt.close(fig)
    n_panels += 1

    # Speed to constraint satisfaction at 4 thresholds, this cell only -
    # per-seed dots shown alongside the mean bar so the spread isn't
    # hidden (e.g. B_dropP's PPO near-zero bar is pulled way up by one
    # late-converging seed; a bare mean bar would hide that entirely).
    # "viol=0" uses a tiny epsilon (1e-6), not literally testing float
    # equality - a rolling MEAN of many rollouts essentially never lands
    # on exactly 0.0 even once every individual rollout in the window
    # does, since floating-point rounding differs rollout to rollout.
    thresholds = [("viol<5", 5.0), ("viol<1", 1.0), ("viol<0.2", 0.2), ("viol=0", 1e-6)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(thresholds))
    width = 0.35
    for j, (runs, method) in enumerate([(ppo, "PPO"), (pds, "PPO+PDS")]):
        color = METHOD_COLOR[method]
        means, seed_vals = [], []
        for _, thresh in thresholds:
            vals = [first_rollout_below(d, "eval/violation", thresh) for d in runs]
            seed_vals.append(vals)
            defined = [v for v in vals if v is not None]
            means.append(np.mean(defined) if defined else np.nan)
        offset = (j - 0.5) * width
        ax.bar(x + offset, means, width, color=color, alpha=0.85, label=f"{method} (mean, n={len(runs)})")
        for xi, vals in zip(x + offset, seed_vals):
            defined = [v for v in vals if v is not None]
            if defined:
                ax.scatter([xi] * len(defined), defined, color=INK, s=18, zorder=5,
                           label="individual seed" if (j == 0 and xi == x[0] + offset) else None)
            n_missing = len(vals) - len(defined)
            if n_missing:
                ax.annotate(f"{n_missing} seed(s)\nnever reached", (xi, 0), xytext=(0, 8),
                            textcoords="offset points", ha="center", fontsize=6.5, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels([t[0] for t in thresholds])
    ax.set_ylabel("rollouts needed (sustained-20 window)")
    ax.set_title(f"{cell_name}: speed to constraint satisfaction (lower = faster; dots = individual seeds)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(cell_dir, f"{n_panels}_speed_to_threshold.png"))
    plt.close(fig)
    n_panels += 1

    print(f"  wrote {n_panels} panels to {cell_dir}")


def generate(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for cell_name, spec in CELLS.items():
        generate_cell(cell_name, spec, out_dir)


if __name__ == "__main__":
    generate(sys.argv[1] if len(sys.argv) > 1 else "graphs_exp1")
