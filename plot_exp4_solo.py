"""
plot_exp4_solo.py - the full graph suite (15-panel full-panels + 14-panel
meeting-followup) for EACH method ALONE, using that method's own full
available rollout count - no PPO, no cross-method matching/truncation at
all. Built because every comparison so far matches n to the shortest
method in the comparison, which for the "X vs PPO" cells meant PPO's own
~4000-rollout length was silently capping methods (e.g. New PDS) that
have run much further (8000+ rollouts) since the last data pull.

Reuses plot_exp3_full_panels.generate_cell() directly for the 15-panel
suite (it's already N-generic and works with a single-method list - no
PPO required, n naturally becomes that one method's own min-seed length).
The meeting-followup suite needed a real rewrite (plot_meeting_followup_exp3's
generate_cell() is hardcoded 2-method PPO-vs-X throughout, including
panel 05's std-RATIO-to-PPO) - this file's generate_cell_solo() drops
every PPO-pairing and panel 05 reports raw across-seed std instead of a
ratio (no PPO to divide by).

Usage: .venv/Scripts/python.exe plot_exp4_solo.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from plot_exp3_full_panels import generate_cell as generate_full_panels_cell
from plot_meeting_followup_exp3 import (
    D_BAR, B_BAR, SEEDS, SURFACE, INK, MUTED,
    load_progress, smooth, mean_std_band, pooled_tail,
    sliding_prob_exceed, per_seed_convergence_fraction,
)

METHODS = {
    # Fresh PPO rerun - replaces the original Exp1 PPO run (manually
    # stopped at ~4000-4800 rollouts, no resumable checkpoint). Revealed
    # late-training violation onset on 2/6 seeds past ~5400-5700 rollouts -
    # see EXP3_PDS_VE_COMPARISON/05_ppo_baseline/README.txt.
    "PPO": ("#3b6fa0", "EXP3_PDS_VE_COMPARISON/05_ppo_baseline/raw/ppo_rerun_A{s}_progress.csv"),
    "New PDS": ("#2f9e44", "EXP3_PDS_VE_COMPARISON/03_new_pds/metrics/raw/newpds_own_A{s}_progress.csv"),
    "Old PDS+VE": ("#a0479e", "EXP3_PDS_VE_COMPARISON/01_old_pds_plus_ve/metrics/raw/oldve_own_A{s}_progress.csv"),
    "VE v2": ("#c99a1e", "EXP3_PDS_VE_COMPARISON/02_ve_v2/metrics/raw/ve2_own_A{s}_progress.csv"),
}
OUT_ROOT = "EXP4_SOLO"
FOLDER_NAME = {
    "PPO": "ppo", "New PDS": "new_pds",
    "Old PDS+VE": "old_pds_ve", "VE v2": "ve_v2",
}


def generate_cell_solo(title_prefix, method_name, color, path_tmpl, out_dir):
    """Meeting-followup's 14 panels, ONE method only, own full rollout
    length. Every panel that compared 2 methods (PPO | X) now shows just
    the one; panel 05 (deviation by metric) reports raw across-seed std
    per metric instead of a ratio, since there's no PPO to divide by."""
    runs = [load_progress(path_tmpl.format(s=s)) for s in SEEDS]
    n = min(len(d) for d in runs)
    os.makedirs(out_dir, exist_ok=True)
    print(f"{title_prefix} (solo): n={n} rollouts, {len(SEEDS)} seeds")
    x = runs[0]["rollout"].values[:n]
    runs_t = [d.iloc[:n] for d in runs]
    tail500, tail1000 = min(500, n), min(1000, n)

    # ---- 01: budget utilisation ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, col, bar, ylabel in [(axes[0, 0], "eval/J_D", D_BAR, "J_D (stall)"), (axes[0, 1], "eval/J_B", B_BAR, "J_B (tiles)")]:
        vals = np.stack([smooth(d[col]).values for d in runs_t], axis=0)
        mean_std_band(ax, x, vals, color, method_name)
        ax.axhline(bar, color="#e34948", ls="--", lw=1.0, label=f"budget = {bar}")
        ax.set_ylabel(ylabel); ax.set_xlabel("rollout"); ax.legend(frameon=False, fontsize=8)
    for ax, col, bar, title in [(axes[1, 0], "eval/J_D", D_BAR, "Stall budget used"), (axes[1, 1], "eval/J_B", B_BAR, "Tile budget used")]:
        pts = [d[col].values[-tail500:].mean() / bar * 100 for d in runs_t]
        ax.axhline(100, color="#e34948", ls="--", lw=1.0, label="100% = exactly at budget")
        jitter = (np.random.RandomState(0).rand(len(pts)) - 0.5) * 0.15
        ax.scatter(np.zeros(len(pts)) + jitter, pts, color=color, s=40, zorder=3)
        ax.hlines(np.mean(pts), -0.2, 0.2, color=INK, lw=2.5, zorder=4)
        ax.annotate(f"mean {np.mean(pts):.1f}%", (0.22, np.mean(pts)), fontsize=8, va="center")
        ax.set_xticks([0]); ax.set_xticklabels([method_name])
        ax.set_ylabel(f"% of allowed budget used ({tail500} rollouts, per seed)")
        ax.set_title(title); ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"{title_prefix}: where does it operate? (solo, n={n})")
    fig.tight_layout(); fig.savefig(f"{out_dir}/01_budget_utilisation.png"); plt.close(fig)

    # ---- 02: convergence in probability ----
    eps_list = [("eps=0.5", 0.5), ("eps=0.2", 0.2), ("eps=0.1", 0.1), ("eps=0.05", 0.05), ("eps=0.01", 0.01)]
    fig, ax = plt.subplots(figsize=(9, 5))
    xs, curves = sliding_prob_exceed(runs_t, "eval/violation", eps_list)
    alphas = np.linspace(0.35, 1.0, len(eps_list))
    for (lbl, _), a in zip(eps_list, alphas):
        ax.plot(xs, curves[lbl], color=color, alpha=a, lw=2.0 if a == 1.0 else 1.2, label=lbl)
    ax.axhline(0.05, color="#e34948", ls="--", lw=1.0, label="delta=0.05")
    ax.set_ylim(-0.02, 1.02); ax.set_xlabel("rollout n"); ax.set_ylabel("P(violation > eps)")
    ax.set_title(f"{title_prefix}: convergence in probability (sliding window of 200 rollouts x 6 seeds)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out_dir}/02_convergence_in_probability.png"); plt.close(fig)

    # ---- 03: violation CDF and tile-distance ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    pooled_v = pooled_tail(runs_t, "eval/violation", tail1000)
    xs_ = np.sort(pooled_v); ys_ = np.arange(1, len(xs_) + 1) / len(xs_)
    axes[0].step(xs_, ys_, color=color, lw=2.0, where="post",
                  label=f"median={np.median(pooled_v):.3f}, P90={np.percentile(pooled_v,90):.3f}, mean={pooled_v.mean():.3f}")
    axes[0].set_xscale("symlog", linthresh=1e-3); axes[0].axhline(0.5, color=INK, ls=":", lw=0.8)
    axes[0].set_xlabel(f"violation (last {tail1000} rollouts x 6 seeds)"); axes[0].set_ylabel("cumulative probability")
    axes[0].set_title("CDF of violation"); axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    dist = B_BAR - pooled_tail(runs_t, "eval/J_B", tail1000)
    axes[1].hist(dist, bins=60, density=True, color=color, alpha=0.75)
    axes[1].axvline(0, color="#e34948", ls="--", lw=1.0, label="0 = exactly at tile budget")
    axes[1].axvline(dist.mean(), color=INK, lw=1.5, label=f"mean={dist.mean():.3f}")
    axes[1].set_xlabel("unused tile budget (budget - J_B); <0 means over"); axes[1].set_ylabel("density")
    axes[1].set_title("Distance from the tile constraint"); axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{title_prefix}: violation CDF and tile-distance")
    fig.tight_layout(); fig.savefig(f"{out_dir}/03_violation_cdf_and_distance.png"); plt.close(fig)

    # ---- 04: multiplier stability ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, col, ylabel in [(axes[0], "lagrangian/mu_D", "mu_D (stall)"), (axes[1], "lagrangian/mu_B", "mu_B (tiles)")]:
        vals = np.stack([smooth(d[col]).values for d in runs_t], axis=0)
        std = np.array([d[col].values[-tail500:].mean() for d in runs_t]).std()
        mean_std_band(ax, x, vals, color, f"{method_name} (across-seed std={std:.2f})")
        ax.set_ylabel(ylabel); ax.set_xlabel("rollout"); ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"{ylabel}: stability")
    fig.suptitle(f"{title_prefix}: Lagrange multiplier stability")
    fig.tight_layout(); fig.savefig(f"{out_dir}/04_multiplier_stability.png"); plt.close(fig)

    # ---- 05: across-seed std per metric (no PPO to ratio against) ----
    metric_specs = [
        ("stall", "eval/mean_stall_sec"), ("J_D", "eval/J_D"), ("J_B", "eval/J_B"),
        ("tiles", "physical/mean_n_enhanced_tiles"), ("J_Q", "eval/J_Q"),
        ("coverage", "eval/mean_coverage"), ("PSNR", "eval/mean_psnr_db"),
        ("mu_B", "lagrangian/mu_B"), ("mu_D", "lagrangian/mu_D"),
    ]
    rows = [(name, np.array([d[col].values[-tail1000:].mean() for d in runs_t]).std()) for name, col in metric_specs]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=color)
    for i, (name, v) in enumerate(rows):
        ax.annotate(f"{v:.3f}", (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_xlabel(f"across-seed std, last {tail1000} rollouts (no PPO baseline - raw value, not a ratio)")
    ax.set_title(f"{title_prefix}: per-metric stability (solo - lower = steadier across seeds)")
    fig.tight_layout(); fig.savefig(f"{out_dir}/05_deviation_by_metric.png"); plt.close(fig)

    # ---- 06: entropy ----
    fig, ax = plt.subplots(figsize=(10, 4.5))
    vals = np.stack([smooth(-d["train/entropy_loss"]).values for d in runs_t], axis=0)
    final = np.mean([(-d["train/entropy_loss"].values[-tail500:]).mean() for d in runs_t])
    mean_std_band(ax, x, vals, color, f"{method_name} (final={final:.2f})")
    ax.set_ylabel("policy entropy (= -train/entropy_loss)"); ax.set_xlabel("rollout")
    ax.legend(frameon=False, fontsize=8); ax.set_title(f"{title_prefix}: policy entropy")
    fig.tight_layout(); fig.savefig(f"{out_dir}/06_entropy.png"); plt.close(fig)

    # ---- 07/08: convergence isolated ----
    for panel_num, col, bar, eps_list_iso, cname in [
        ("07", "eval/J_D", D_BAR, [("eps=0 (literally over budget)", 0.0), ("eps=0.05", 0.05), ("eps=0.1", 0.1), ("eps=0.2", 0.2)], "STALL"),
        ("08", "eval/J_B", B_BAR, [("eps=0 (literally over budget)", 0.0), ("eps=0.2", 0.2), ("eps=0.5", 0.5), ("eps=1.0", 1.0)], "TILES"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        thresh_list = [(lbl, bar + e) for lbl, e in eps_list_iso]
        xs, curves = sliding_prob_exceed(runs_t, col, thresh_list)
        alphas = np.linspace(0.35, 1.0, len(thresh_list))
        for (lbl, _), a in zip(thresh_list, alphas[::-1]):
            is_main = lbl.startswith("eps=0 ")
            ax.plot(xs, curves[lbl], color=color, alpha=1.0 if is_main else 0.4,
                     lw=2.2 if is_main else 1.2, label=lbl)
        ax.axhline(0.05, color="#e34948", ls="--", lw=1.0, label="delta=0.05")
        ax.set_ylim(-0.02, 1.02); ax.set_xlabel("rollout n")
        ax.set_ylabel(f"P({'J_D' if cname=='STALL' else 'J_B'} > {bar}+eps)")
        ax.set_title(f"{title_prefix}: {cname.lower()} constraint isolated (budget={bar})")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(f"{out_dir}/{panel_num}_convergence_{cname.lower()}_isolated.png"); plt.close(fig)

    # ---- 09/10: CDF + distance isolated ----
    for panel_num, col, bar, cname in [("09", "eval/J_D", D_BAR, "stall"), ("10", "eval/J_B", B_BAR, "tiles")]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        over = np.maximum(0, pooled_tail(runs_t, col, tail1000) - bar)
        xs_ = np.sort(over); ys_ = np.arange(1, len(xs_) + 1) / len(xs_)
        axes[0].step(xs_, ys_, color=color, lw=2.0, where="post")
        axes[0].set_xscale("symlog", linthresh=1e-3)
        axes[0].set_xlabel(f"amount over {cname} budget"); axes[0].set_ylabel("cumulative probability")
        axes[0].set_title("CDF of over-budget amount")
        dist = bar - pooled_tail(runs_t, col, tail1000)
        axes[1].hist(dist, bins=60, density=True, color=color, alpha=0.75)
        axes[1].axvline(0, color="#e34948", ls="--", lw=1.0, label="0 = exactly at budget")
        axes[1].axvline(dist.mean(), color=INK, lw=1.5, label=f"mean={dist.mean():.3f}")
        axes[1].set_xlabel(f"unused {cname} budget"); axes[1].set_ylabel("density")
        axes[1].set_title(f"Distance from {cname} constraint"); axes[1].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{title_prefix}: {cname.upper()} constraint isolated")
        fig.tight_layout(); fig.savefig(f"{out_dir}/{panel_num}_cdf_distance_{cname}.png"); plt.close(fig)

    # ---- 11: power descriptive ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    vals = np.stack([smooth(d["eval/mean_power_mw"]).values for d in runs_t], axis=0)
    final = np.mean([d["eval/mean_power_mw"].values[-tail500:].mean() for d in runs_t])
    mean_std_band(axes[0], x, vals, color, f"{method_name} (final={final:.2f} mW)")
    axes[0].set_ylabel("power (mW)"); axes[0].set_xlabel("rollout"); axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Power usage over training (NOT constrained - P dropped)")
    pooled = pooled_tail(runs_t, "eval/mean_power_mw", tail1000)
    axes[1].hist(pooled, bins=50, density=True, color=color, alpha=0.75, label=f"mean={pooled.mean():.2f} mW")
    axes[1].set_xlabel("power (mW)"); axes[1].set_ylabel("density"); axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title(f"Power distribution, last {tail1000} rollouts")
    fig.suptitle(f"{title_prefix}: power - descriptive, since P is dropped in this experiment")
    fig.tight_layout(); fig.savefig(f"{out_dir}/11_power_descriptive.png"); plt.close(fig)

    # ---- 12: violation contribution ----
    fig, ax = plt.subplots(figsize=(10, 5))
    viol_d = np.stack([np.maximum(0, d["eval/J_D"].values - D_BAR) for d in runs_t], axis=0).mean(axis=0)
    viol_b = np.stack([np.maximum(0, d["eval/J_B"].values - B_BAR) for d in runs_t], axis=0).mean(axis=0)
    ax.stackplot(x, viol_d, viol_b, colors=["#3b6fa0", "#e8a33d"], labels=["viol_D (stall)", "viol_B (tiles)"], alpha=0.85)
    ax.plot(x, viol_d + viol_b, color=INK, lw=0.8, label="total")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_ylabel("violation contribution (log scale)"); ax.set_xlabel("rollout")
    ax.set_title(f"{title_prefix}: composition, log-y so the tail is visible"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out_dir}/12_violation_contribution.png"); plt.close(fig)

    # ---- 13: per-seed convergence fraction ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, bar, cname in [(axes[0], "eval/J_D", D_BAR, "stall"), (axes[1], "eval/J_B", B_BAR, "tiles")]:
        fractions = per_seed_convergence_fraction(runs_t, col, bar, [5.0, 1.0, 0.2])
        for eps, alpha, lw in [(5.0, 0.3, 1.0), (1.0, 0.5, 1.0), (0.2, 1.0, 2.0)]:
            ax.plot(x, fractions[eps], color=color, alpha=alpha, lw=lw, label=f"eps={eps}" if eps == 0.2 else None)
        ax.set_ylabel(f"fraction of {len(SEEDS)} seeds converged"); ax.set_xlabel("rollout")
        ax.set_title(f"{cname}: per-seed convergence fraction (eps=0.2 bold, 5.0/1.0 faint)")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{title_prefix}: fraction of SEEDS individually converged, vs rollout")
    fig.tight_layout(); fig.savefig(f"{out_dir}/13_per_seed_convergence_fraction.png"); plt.close(fig)

    # ---- 14: policy convergence (KL) ----
    fig, ax = plt.subplots(figsize=(11, 5))
    vals = np.stack([d["train/approx_kl"].values for d in runs_t], axis=0)
    tail_mean = np.mean([d["train/approx_kl"].values[-tail1000:].mean() for d in runs_t])
    mean_std_band(ax, x, vals, color, f"{method_name} (tail-{tail1000} mean={tail_mean:.5f})")
    ax.set_ylabel("approx_kl (policy change per update)"); ax.set_xlabel("rollout")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{title_prefix}: policy convergence - how much the policy still changes each update")
    fig.tight_layout(); fig.savefig(f"{out_dir}/14_policy_convergence_kl.png"); plt.close(fig)

    print(f"  wrote 14 solo meeting-followup panels to {out_dir}")


if __name__ == "__main__":
    for method_name, (color, path_tmpl) in METHODS.items():
        folder = FOLDER_NAME[method_name]
        out_dir = f"{OUT_ROOT}/{folder}"
        generate_full_panels_cell(f"{method_name} (solo)", [method_name], out_dir)
        generate_cell_solo(f"{method_name}", method_name, color, path_tmpl, f"{out_dir}/meeting_followup")
