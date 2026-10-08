"""
plot_meeting_followup_exp3.py - reproduces the FINAL_PART1_EXP_M/meeting_followup/
16-panel analysis suite (PPO vs PPO+PDS) for the new EXP3 method-vs-PPO cells.

No source script survived for the original 16 PNGs - each panel's exact
statistic/computation was reverse-engineered directly from the reference
images (titles, axis labels, legends). Where a reference panel's exact
smoothing kernel/window couldn't be read off the image precisely, a
reasonable, explicitly-documented choice was made instead (noted per
panel below) - prioritizing correct underlying statistics over pixel-
identical cosmetic replication.

Panels 15/16 (instantaneous per-timestep stall/tiles) are NOT built here:
they require eval_steps.csv, which only exists for the original Exp1
PPO/PDS runs - it was never extracted for the new EXP3 runs. Blocked
until that data is synced.

Usage: .venv/Scripts/python.exe plot_meeting_followup_exp3.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PPO_COLOR = "#3b6fa0"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
})

SEEDS = [1, 2, 3, 5, 6, 7]
D_BAR, P_BAR, B_BAR = 0.3, 6.5, 8.0
PPO_PATH = "EXP3_PDS_VE_COMPARISON/05_ppo_baseline/raw/ppo_rerun_A{s}_progress.csv"


def load_progress(path):
    df = pd.read_csv(path, dtype=str)
    df = df[df["eval/feasible"] != "eval/feasible"].apply(pd.to_numeric, errors="coerce")
    df = df[df["eval/feasible"].notna()].reset_index(drop=True)
    df["rollout"] = np.arange(1, len(df) + 1)
    return df


def smooth(series, window=101):
    """Moderate rolling-mean smoothing for trend-line panels (matches the
    visibly smooth curves in the reference images at n~4000 rollouts) -
    the reference script's exact window is unrecoverable, so this is an
    explicit, documented choice, not a guess at exact replication."""
    return series.rolling(window, center=True, min_periods=1).mean()


def mean_std_band(ax, x, series_2d, color, label, clip=None):
    mean = series_2d.mean(axis=0)
    std = series_2d.std(axis=0)
    lo, hi = mean - std, mean + std
    if clip is not None:
        lo, hi = np.clip(lo, *clip), np.clip(hi, *clip)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0, zorder=2)
    ax.plot(x, mean, color=color, lw=2.0, label=label, zorder=3)


def pooled_tail(runs, col, window):
    """Pool the last `window` rollouts x all seeds into one flat array -
    "last N rollouts x 6 seeds" pooling, used throughout the reference
    suite (panels 01, 03, 05, 09, 10, 11)."""
    return np.concatenate([d[col].values[-window:] for d in runs])


def ecdf_symlog(ax, pooled_a, pooled_b, label_a, label_b, color_a, color_b, xlabel):
    for pooled, color, label in [(pooled_a, color_a, label_a), (pooled_b, color_b, label_b)]:
        x = np.sort(pooled)
        y = np.arange(1, len(x) + 1) / len(x)
        median, p90, mean = np.median(pooled), np.percentile(pooled, 90), np.mean(pooled)
        ax.step(x, y, color=color, lw=2.0, where="post",
                 label=f"{label} (median={median:.3f}, P90={p90:.3f}, mean={mean:.3f})")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.axhline(0.5, color=INK, ls=":", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cumulative probability")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")


def sliding_prob_exceed(runs, col, thresh_fn_list, window=200):
    """P(col > threshold) via a sliding window of `window` rollouts x
    len(runs) seeds, at every rollout n >= window - matches "sliding
    window of 200 rollouts x 6 seeds" in the reference panels exactly."""
    n = len(runs[0])
    stacked = np.stack([d[col].values for d in runs], axis=0)  # (n_seeds, n_rollouts)
    xs = np.arange(window, n + 1)
    out = {}
    for label, thresh in thresh_fn_list:
        probs = []
        for end in xs:
            pool = stacked[:, end - window:end].flatten()
            probs.append(np.mean(pool > thresh))
        out[label] = np.array(probs)
    return xs, out


def per_seed_convergence_fraction(runs, col, bar, eps_list):
    n = len(runs[0])
    out = {}
    for eps in eps_list:
        frac = np.zeros(n)
        for i in range(n):
            frac[i] = np.mean([d[col].values[i] <= bar + eps for d in runs])
        out[eps] = frac
    return out


def generate_cell(title_prefix, method_name, method_color, method_path_tmpl, out_dir):
    ppo = [load_progress(PPO_PATH.format(s=s)) for s in SEEDS]
    method = [load_progress(method_path_tmpl.format(s=s)) for s in SEEDS]
    n = min(min(len(d) for d in ppo), min(len(d) for d in method))
    ppo = [d.iloc[:n] for d in ppo]
    method = [d.iloc[:n] for d in method]
    os.makedirs(out_dir, exist_ok=True)
    print(f"{title_prefix}: n={n} rollouts, {len(SEEDS)} seeds/method")
    x = ppo[0]["rollout"].values
    tail500 = min(500, n)
    tail1000 = min(1000, n)

    # ---- 01: budget utilisation ----
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, col, bar, ylabel in [(axes[0, 0], "eval/J_D", D_BAR, "J_D (stall)"),
                                   (axes[0, 1], "eval/J_B", B_BAR, "J_B (tiles)")]:
        for runs, color, label in [(ppo, PPO_COLOR, "PPO"), (method, method_color, method_name)]:
            vals = np.stack([smooth(d[col]).values for d in runs], axis=0)
            mean_std_band(ax, x, vals, color, f"{label} (mean +/- std, n=6)")
        ax.axhline(bar, color="#e34948", ls="--", lw=1.0, label=f"budget = {bar}")
        ax.set_ylabel(ylabel); ax.set_xlabel("rollout"); ax.legend(frameon=False, fontsize=8)
    for ax, col, bar, title in [(axes[1, 0], "eval/J_D", D_BAR, "Stall budget used"),
                                  (axes[1, 1], "eval/J_B", B_BAR, "Tile budget used")]:
        cats, vals_pts, colors = [], [], []
        for runs, color, label in [(ppo, PPO_COLOR, "PPO"), (method, method_color, method_name)]:
            per_seed = [d[col].values[-tail500:].mean() / bar * 100 for d in runs]
            cats.append(label); vals_pts.append(per_seed); colors.append(color)
        ax.axhline(100, color="#e34948", ls="--", lw=1.0, label="100% = exactly at budget")
        for i, (cat, pts, color) in enumerate(zip(cats, vals_pts, colors)):
            jitter = (np.random.RandomState(0).rand(len(pts)) - 0.5) * 0.15
            ax.scatter(np.full(len(pts), i) + jitter, pts, color=color, s=40, zorder=3)
            ax.hlines(np.mean(pts), i - 0.2, i + 0.2, color=INK, lw=2.5, zorder=4)
            ax.annotate(f"mean {np.mean(pts):.1f}%", (i + 0.22, np.mean(pts)), fontsize=8, va="center")
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats)
        ax.set_ylabel(f"% of allowed budget used ({tail500} rollouts, per seed)")
        ax.set_title(title); ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"{title_prefix}: PPO vs {method_name} - where does each method operate?")
    fig.tight_layout(); fig.savefig(f"{out_dir}/01_budget_utilisation.png"); plt.close(fig)

    # ---- 02: convergence in probability (total violation) ----
    eps_list = [("eps=0.5", 0.5), ("eps=0.2", 0.2), ("eps=0.1", 0.1), ("eps=0.05", 0.05), ("eps=0.01", 0.01)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, runs, color, label in [(axes[0], ppo, PPO_COLOR, "PPO"), (axes[1], method, method_color, method_name)]:
        xs, curves = sliding_prob_exceed(runs, "eval/violation", eps_list)
        alphas = np.linspace(0.35, 1.0, len(eps_list))
        for (lbl, _), a in zip(eps_list, alphas):
            ax.plot(xs, curves[lbl], color=color, alpha=a, lw=2.0 if a == 1.0 else 1.2, label=lbl)
        ax.axhline(0.05, color="#e34948", ls="--", lw=1.0, label="delta=0.05")
        ax.set_ylim(-0.02, 1.02); ax.set_xlabel("rollout n"); ax.set_ylabel("P(violation > eps)")
        ax.set_title(f"{label}: convergence in probability"); ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"{title_prefix}: P(violation > eps) - sliding window of 200 rollouts x {len(SEEDS)} seeds")
    fig.tight_layout(); fig.savefig(f"{out_dir}/02_convergence_in_probability.png"); plt.close(fig)

    # ---- 03: violation CDF and tile-distance ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    v_ppo = pooled_tail(ppo, "eval/violation", tail1000)
    v_meth = pooled_tail(method, "eval/violation", tail1000)
    ecdf_symlog(axes[0], v_ppo, v_meth, "PPO", method_name, PPO_COLOR, method_color,
                f"violation  (last {tail1000} rollouts x {len(SEEDS)} seeds)")
    axes[0].set_title("CDF of violation")
    dist_ppo = B_BAR - pooled_tail(ppo, "eval/J_B", tail1000)
    dist_meth = B_BAR - pooled_tail(method, "eval/J_B", tail1000)
    axes[1].hist(dist_ppo, bins=60, density=True, color=PPO_COLOR, alpha=0.55, label=f"PPO (mean={dist_ppo.mean():.3f})")
    axes[1].hist(dist_meth, bins=60, density=True, color=method_color, alpha=0.55, label=f"{method_name} (mean={dist_meth.mean():.3f})")
    axes[1].axvline(dist_ppo.mean(), color=PPO_COLOR, lw=2.0)
    axes[1].axvline(dist_meth.mean(), color=method_color, lw=2.0)
    axes[1].axvline(0, color="#e34948", ls="--", lw=1.0, label="0 = exactly at tile budget")
    axes[1].set_xlabel("unused tile budget (budget - J_B); <0 means over"); axes[1].set_ylabel("density")
    axes[1].set_title("Distance from the tile constraint"); axes[1].legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"{title_prefix}: violation CDF and tile-distance, with means marked")
    fig.tight_layout(); fig.savefig(f"{out_dir}/03_violation_cdf_and_distance.png"); plt.close(fig)

    # ---- 04: multiplier stability ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, col, ylabel in [(axes[0], "lagrangian/mu_D", "mu_D (stall)"), (axes[1], "lagrangian/mu_B", "mu_B (tiles)")]:
        for runs, color, label in [(ppo, PPO_COLOR, "PPO"), (method, method_color, method_name)]:
            vals = np.stack([smooth(d[col]).values for d in runs], axis=0)
            seed_tail_means = np.array([d[col].values[-tail500:].mean() for d in runs])
            across_seed_std = seed_tail_means.std()
            mean_std_band(ax, x, vals, color, f"{label} (across-seed std={across_seed_std:.2f})")
        ax.set_ylabel(ylabel); ax.set_xlabel("rollout"); ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"{ylabel}: stability")
    fig.suptitle(f"{title_prefix}: Lagrange multiplier stability")
    fig.tight_layout(); fig.savefig(f"{out_dir}/04_multiplier_stability.png"); plt.close(fig)

    # ---- 05: deviation by metric ----
    metric_specs = [
        ("stall", "eval/mean_stall_sec"), ("J_D", "eval/J_D"), ("J_B", "eval/J_B"),
        ("tiles", "physical/mean_n_enhanced_tiles"), ("J_Q", "eval/J_Q"),
        ("coverage", "eval/mean_coverage"), ("PSNR", "eval/mean_psnr_db"),
        ("mu_B", "lagrangian/mu_B"), ("mu_D", "lagrangian/mu_D"),
    ]
    rows = []
    for name, col in metric_specs:
        std_ppo = np.array([d[col].values[-tail1000:].mean() for d in ppo]).std()
        std_meth = np.array([d[col].values[-tail1000:].mean() for d in method]).std()
        ratio = std_meth / std_ppo if std_ppo > 0 else np.nan
        rows.append((name, ratio))
    rows.sort(key=lambda r: -r[1] if not np.isnan(r[1]) else -np.inf)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [method_color if r > 1 else PPO_COLOR for _, r in rows]
    ax.barh([r[0] for r in rows][::-1], [r[1] for r in rows][::-1], color=colors[::-1])
    for i, (name, r) in enumerate(rows[::-1]):
        ax.annotate(f"{r:.2f}x", (r, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    ax.axvline(1.0, color="#e34948", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(f"across-seed std ratio ({method_name} / PPO)  -  left of line = {method_name} steadier")
    ax.set_title(f"{title_prefix}: which metrics is {method_name} steadier on? (last {tail1000} rollouts)")
    fig.tight_layout(); fig.savefig(f"{out_dir}/05_deviation_by_metric.png"); plt.close(fig)

    # ---- 06: entropy ----
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for runs, color, label in [(ppo, PPO_COLOR, "PPO"), (method, method_color, method_name)]:
        vals = np.stack([smooth(-d["train/entropy_loss"]).values for d in runs], axis=0)
        final = np.mean([(-d["train/entropy_loss"].values[-tail500:]).mean() for d in runs])
        mean_std_band(ax, x, vals, color, f"{label} (final={final:.2f})")
    ax.set_ylabel("policy entropy (= -train/entropy_loss)"); ax.set_xlabel("rollout")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{title_prefix}: policy entropy")
    fig.tight_layout(); fig.savefig(f"{out_dir}/06_entropy.png"); plt.close(fig)

    # ---- 07/08: convergence isolated (stall / tiles) ----
    for panel_num, col, bar, eps_list_iso, cname in [
        ("07", "eval/J_D", D_BAR, [("eps=0 (literally over budget)", 0.0), ("eps=0.05", 0.05), ("eps=0.1", 0.1), ("eps=0.2", 0.2)], "STALL"),
        ("08", "eval/J_B", B_BAR, [("eps=0 (literally over budget)", 0.0), ("eps=0.2", 0.2), ("eps=0.5", 0.5), ("eps=1.0", 1.0)], "TILES"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax, runs, color, label in [(axes[0], ppo, PPO_COLOR, "PPO"), (axes[1], method, method_color, method_name)]:
            thresh_list = [(lbl, bar + e) for lbl, e in eps_list_iso]
            xs, curves = sliding_prob_exceed(runs, col, thresh_list)
            alphas = np.linspace(0.35, 1.0, len(thresh_list))
            for (lbl, _), a in zip(thresh_list, alphas[::-1]):
                is_main = lbl.startswith("eps=0 ")
                ax.plot(xs, curves[lbl], color=color, alpha=1.0 if is_main else 0.4,
                         lw=2.2 if is_main else 1.2, label=lbl)
            ax.axhline(0.05, color="#e34948", ls="--", lw=1.0, label="delta=0.05")
            ax.set_ylim(-0.02, 1.02); ax.set_xlabel("rollout n")
            ax.set_ylabel(f"P({'J_D' if cname=='STALL' else 'J_B'} > {bar}+eps)")
            ax.set_title(f"{label}: {cname.lower()} constraint only"); ax.legend(frameon=False, fontsize=7)
        fig.suptitle(f"{title_prefix}: convergence in probability - {cname} constraint isolated (budget={bar})")
        fig.tight_layout(); fig.savefig(f"{out_dir}/{panel_num}_convergence_{cname.lower()}_isolated.png"); plt.close(fig)

    # ---- 09/10: CDF + distance isolated (stall / tiles) ----
    for panel_num, col, bar, cname in [("09", "eval/J_D", D_BAR, "stall"), ("10", "eval/J_B", B_BAR, "tiles")]:
        fig, axes = plt.subplots(1, 3, figsize=(19, 5))
        over_ppo = np.maximum(0, pooled_tail(ppo, col, tail1000) - bar)
        over_meth = np.maximum(0, pooled_tail(method, col, tail1000) - bar)
        ecdf_symlog(axes[0], over_ppo, over_meth, "PPO", method_name, PPO_COLOR, method_color,
                    f"amount over {cname} budget")
        axes[0].set_title("CDF of over-budget amount")
        for ax, runs, color, label in [(axes[1], ppo, PPO_COLOR, "PPO"), (axes[2], method, method_color, method_name)]:
            dist = bar - pooled_tail(runs, col, tail1000)
            ax.hist(dist, bins=60, density=True, color=color, alpha=0.75)
            ax.axvline(0, color="#e34948", ls="--", lw=1.0, label="0 = exactly at budget")
            ax.axvline(dist.mean(), color=INK, lw=1.5, label=f"mean={dist.mean():.3f}")
            ax.set_xlabel(f"unused {cname} budget (budget - {cname})"); ax.set_ylabel("density")
            ax.set_title(f"{label} distance from {cname} constraint  (own scale)")
            ax.legend(frameon=False, fontsize=7.5)
        fig.suptitle(f"{title_prefix}: {cname.upper()} constraint isolated - CDF + separately-scaled distance panels")
        fig.tight_layout(); fig.savefig(f"{out_dir}/{panel_num}_cdf_distance_{cname}.png"); plt.close(fig)

    # ---- 11: power descriptive ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for runs, color, label in [(ppo, PPO_COLOR, "PPO"), (method, method_color, method_name)]:
        vals = np.stack([smooth(d["eval/mean_power_mw"]).values for d in runs], axis=0)
        final = np.mean([d["eval/mean_power_mw"].values[-tail500:].mean() for d in runs])
        mean_std_band(axes[0], x, vals, color, f"{label} (final={final:.2f} mW)")
    axes[0].set_ylabel("power (mW)"); axes[0].set_xlabel("rollout"); axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Power usage over training (NOT constrained - P dropped)")
    for runs, color, label in [(ppo, PPO_COLOR, "PPO"), (method, method_color, method_name)]:
        pooled = pooled_tail(runs, "eval/mean_power_mw", tail1000)
        axes[1].hist(pooled, bins=50, density=True, color=color, alpha=0.6, label=f"{label} (mean={pooled.mean():.2f} mW)")
    axes[1].set_xlabel("power (mW)"); axes[1].set_ylabel("density"); axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title(f"Power distribution, last {tail1000} rollouts (descriptive only, no budget)")
    fig.suptitle(f"{title_prefix}: power - descriptive, since P is dropped (unconstrained) in this experiment")
    fig.tight_layout(); fig.savefig(f"{out_dir}/11_power_descriptive.png"); plt.close(fig)

    # ---- 12: violation contribution (stacked area, log-y) ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, runs, label in [(axes[0], ppo, "PPO"), (axes[1], method, method_name)]:
        viol_d = np.stack([np.maximum(0, d["eval/J_D"].values - D_BAR) for d in runs], axis=0).mean(axis=0)
        viol_b = np.stack([np.maximum(0, d["eval/J_B"].values - B_BAR) for d in runs], axis=0).mean(axis=0)
        ax.stackplot(x, viol_d, viol_b, colors=["#3b6fa0", "#e8a33d"], labels=["viol_D (stall)", "viol_B (tiles)"], alpha=0.85)
        ax.plot(x, viol_d + viol_b, color=INK, lw=0.8, label="total")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_ylabel("violation contribution (log scale)"); ax.set_xlabel("rollout")
        ax.set_title(f"{label}: composition, log-y so the tail is visible"); ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{title_prefix}: violation composition - log scale reveals late-training behavior")
    fig.tight_layout(); fig.savefig(f"{out_dir}/12_violation_contribution.png"); plt.close(fig)

    # ---- 13: per-seed convergence fraction ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, col, bar, cname in [(axes[0], "eval/J_D", D_BAR, "stall"), (axes[1], "eval/J_B", B_BAR, "tiles")]:
        for runs, color, label in [(ppo, PPO_COLOR, "PPO"), (method, method_color, method_name)]:
            fractions = per_seed_convergence_fraction(runs, col, bar, [5.0, 1.0, 0.2])
            for eps, alpha, lw in [(5.0, 0.35, 1.2), (1.0, 0.55, 1.2), (0.2, 1.0, 2.2)]:
                ax.plot(x, fractions[eps], color=color, alpha=alpha, lw=lw,
                         label=f"{label} eps={eps}" if eps == 0.2 else None)
        ax.set_ylabel(f"fraction of {len(SEEDS)} seeds converged"); ax.set_xlabel("rollout")
        ax.set_title(f"{cname}: per-seed convergence fraction (eps=0.2 bold, 5.0/1.0 faint)")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{title_prefix}: fraction of SEEDS individually converged, vs rollout")
    fig.tight_layout(); fig.savefig(f"{out_dir}/13_per_seed_convergence_fraction.png"); plt.close(fig)

    # ---- 14: policy convergence (KL), raw/unsmoothed ----
    fig, ax = plt.subplots(figsize=(12, 5))
    for runs, color, label in [(ppo, PPO_COLOR, "PPO"), (method, method_color, method_name)]:
        vals = np.stack([d["train/approx_kl"].values for d in runs], axis=0)
        tail_mean = np.mean([d["train/approx_kl"].values[-tail1000:].mean() for d in runs])
        mean_std_band(ax, x, vals, color, f"{label} (tail-{tail1000} mean={tail_mean:.5f})")
    ax.set_ylabel("approx_kl (policy change per update)"); ax.set_xlabel("rollout")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{title_prefix}: policy convergence - how much the policy still changes each update")
    fig.tight_layout(); fig.savefig(f"{out_dir}/14_policy_convergence_kl.png"); plt.close(fig)

    print(f"  wrote 14 panels to {out_dir} (15/16 skipped - eval_steps.csv not available for {method_name})")


def generate_cell_multi(title_prefix, methods, colors, path_tmpls, out_dir, match_across_methods=True):
    """Same 14 panels as generate_cell(), generalized to N methods (all 4:
    PPO + New PDS + Old PDS+VE + VE v2 - Exp1 PDS dropped from the main
    roster per explicit request). Several panels needed real layout
    changes, not just a longer loop:
    - 02/07/08 (sliding-window P(exceed)): one subplot per method instead
      of a fixed 2.
    - 05 (deviation by metric): grouped bars (one per non-PPO method) per
      metric row, instead of a single PDS/PPO ratio bar.
    - 03/09/10's histogram panels: switched from filled overlaid
      histograms to unfilled step outlines (03/11) or kept as separate
      own-scale panels per method (09/10, now 1+N panels instead of 3) -
      N=5 filled, overlapping histograms were unreadable.
    """
    data = {m: [load_progress(path_tmpls[m].format(s=s)) for s in SEEDS] for m in methods}
    if match_across_methods:
        n = min(len(d) for runs in data.values() for d in runs)
        data = {m: [d.iloc[:n] for d in data[m]] for m in methods}
        print(f"{title_prefix}: n={n} rollouts, {len(SEEDS)} seeds/method, {len(methods)} methods")
    else:
        # Each method keeps its OWN min-seed length, not matched across
        # methods - every x below is now derived per-method inside its
        # own loop (not from a single shared array), so lines of
        # different lengths on the same axes are expected and fine.
        for m in methods:
            n_m = min(len(d) for d in data[m])
            data[m] = [d.iloc[:n_m] for d in data[m]]
        n = min(len(data[m][0]) for m in methods)  # only used below for tail500/tail1000 sizing
        lens = {m: len(data[m][0]) for m in methods}
        print(f"{title_prefix}: {len(SEEDS)} seeds/method, {len(methods)} methods, own lengths: {lens}")
    os.makedirs(out_dir, exist_ok=True)
    tail500, tail1000 = min(500, n), min(1000, n)
    non_ppo = [m for m in methods if m != "PPO"]
    ML = [(data[m], colors[m], m) for m in methods]

    # ---- 01 ----
    fig, axes = plt.subplots(2, 2, figsize=(17, 9))
    for ax, col, bar, ylabel in [(axes[0, 0], "eval/J_D", D_BAR, "J_D (stall)"), (axes[0, 1], "eval/J_B", B_BAR, "J_B (tiles)")]:
        for runs, color, label in ML:
            x = runs[0]["rollout"].values
            vals = np.stack([smooth(d[col]).values for d in runs], axis=0)
            mean_std_band(ax, x, vals, color, label)
        ax.axhline(bar, color="#e34948", ls="--", lw=1.0, label=f"budget = {bar}")
        ax.set_ylabel(ylabel); ax.set_xlabel("rollout"); ax.legend(frameon=False, fontsize=7)
    for ax, col, bar, title in [(axes[1, 0], "eval/J_D", D_BAR, "Stall budget used"), (axes[1, 1], "eval/J_B", B_BAR, "Tile budget used")]:
        ax.axhline(100, color="#e34948", ls="--", lw=1.0, label="100% = exactly at budget")
        for i, (runs, color, label) in enumerate(ML):
            pts = [d[col].values[-tail500:].mean() / bar * 100 for d in runs]
            jitter = (np.random.RandomState(0).rand(len(pts)) - 0.5) * 0.15
            ax.scatter(np.full(len(pts), i) + jitter, pts, color=color, s=36, zorder=3)
            ax.hlines(np.mean(pts), i - 0.2, i + 0.2, color=INK, lw=2.2, zorder=4)
        ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=7.5, rotation=15)
        ax.set_ylabel(f"% of allowed budget used ({tail500} rollouts, per seed)")
        ax.set_title(title); ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"{title_prefix}: where does each method operate?")
    fig.tight_layout(); fig.savefig(f"{out_dir}/01_budget_utilisation.png"); plt.close(fig)

    # ---- 02 ----
    eps_list = [("eps=0.5", 0.5), ("eps=0.2", 0.2), ("eps=0.1", 0.1), ("eps=0.05", 0.05), ("eps=0.01", 0.01)]
    fig, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 5))
    for ax, (runs, color, label) in zip(axes, ML):
        xs, curves = sliding_prob_exceed(runs, "eval/violation", eps_list)
        alphas = np.linspace(0.35, 1.0, len(eps_list))
        for (lbl, _), a in zip(eps_list, alphas):
            ax.plot(xs, curves[lbl], color=color, alpha=a, lw=2.0 if a == 1.0 else 1.2, label=lbl)
        ax.axhline(0.05, color="#e34948", ls="--", lw=1.0, label="delta=0.05")
        ax.set_ylim(-0.02, 1.02); ax.set_xlabel("rollout n")
        if ax is axes[0]: ax.set_ylabel("P(violation > eps)")
        ax.set_title(label); ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle(f"{title_prefix}: P(violation > eps) - sliding window of 200 rollouts x {len(SEEDS)} seeds")
    fig.tight_layout(); fig.savefig(f"{out_dir}/02_convergence_in_probability.png"); plt.close(fig)

    # ---- 03 ----
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
    for runs, color, label in ML:
        pooled = pooled_tail(runs, "eval/violation", tail1000)
        xs_ = np.sort(pooled); ys_ = np.arange(1, len(xs_) + 1) / len(xs_)
        axes[0].step(xs_, ys_, color=color, lw=2.0, where="post",
                      label=f"{label} (median={np.median(pooled):.3f}, mean={pooled.mean():.3f})")
    axes[0].set_xscale("symlog", linthresh=1e-3); axes[0].axhline(0.5, color=INK, ls=":", lw=0.8)
    axes[0].set_xlabel(f"violation (last {tail1000} rollouts x {len(SEEDS)} seeds)"); axes[0].set_ylabel("cumulative probability")
    axes[0].set_title("CDF of violation"); axes[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    for runs, color, label in ML:
        dist = B_BAR - pooled_tail(runs, "eval/J_B", tail1000)
        axes[1].hist(dist, bins=60, density=True, histtype="step", linewidth=2.0, color=color,
                      label=f"{label} (mean={dist.mean():.3f})")
    axes[1].axvline(0, color="#e34948", ls="--", lw=1.0, label="0 = exactly at tile budget")
    axes[1].set_xlabel("unused tile budget (budget - J_B); <0 means over"); axes[1].set_ylabel("density")
    axes[1].set_title("Distance from the tile constraint (step outlines)"); axes[1].legend(frameon=False, fontsize=6.5)
    fig.suptitle(f"{title_prefix}: violation CDF and tile-distance")
    fig.tight_layout(); fig.savefig(f"{out_dir}/03_violation_cdf_and_distance.png"); plt.close(fig)

    # ---- 04 ----
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
    for ax, col, ylabel in [(axes[0], "lagrangian/mu_D", "mu_D (stall)"), (axes[1], "lagrangian/mu_B", "mu_B (tiles)")]:
        for runs, color, label in ML:
            x = runs[0]["rollout"].values
            vals = np.stack([smooth(d[col]).values for d in runs], axis=0)
            std = np.array([d[col].values[-tail500:].mean() for d in runs]).std()
            mean_std_band(ax, x, vals, color, f"{label} (across-seed std={std:.2f})")
        ax.set_ylabel(ylabel); ax.set_xlabel("rollout"); ax.legend(frameon=False, fontsize=7)
        ax.set_title(f"{ylabel}: stability")
    fig.suptitle(f"{title_prefix}: Lagrange multiplier stability")
    fig.tight_layout(); fig.savefig(f"{out_dir}/04_multiplier_stability.png"); plt.close(fig)

    # ---- 05: grouped bars, one group per metric, one bar per non-PPO method ----
    metric_specs = [
        ("stall", "eval/mean_stall_sec"), ("J_D", "eval/J_D"), ("J_B", "eval/J_B"),
        ("tiles", "physical/mean_n_enhanced_tiles"), ("J_Q", "eval/J_Q"),
        ("coverage", "eval/mean_coverage"), ("PSNR", "eval/mean_psnr_db"),
        ("mu_B", "lagrangian/mu_B"), ("mu_D", "lagrangian/mu_D"),
    ]
    std_ppo_by_metric = {name: np.array([d[col].values[-tail1000:].mean() for d in data["PPO"]]).std()
                          for name, col in metric_specs}
    ratios = {m: [] for m in non_ppo}
    for name, col in metric_specs:
        for m in non_ppo:
            std_m = np.array([d[col].values[-tail1000:].mean() for d in data[m]]).std()
            ratios[m].append(std_m / std_ppo_by_metric[name] if std_ppo_by_metric[name] > 0 else np.nan)
    order = np.argsort([np.nanmean([ratios[m][i] for m in non_ppo]) for i in range(len(metric_specs))])[::-1]
    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(metric_specs))
    height = 0.8 / len(non_ppo)
    for j, m in enumerate(non_ppo):
        vals = [ratios[m][i] for i in order]
        offset = (j - (len(non_ppo) - 1) / 2) * height
        ax.barh(y + offset, vals, height, color=colors[m], label=m)
    ax.axvline(1.0, color="#e34948", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels([metric_specs[i][0] for i in order])
    ax.set_xlabel(f"across-seed std ratio (method / PPO)  -  left of line = steadier than PPO")
    ax.set_title(f"{title_prefix}: which metrics is each method steadier on? (last {tail1000} rollouts)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out_dir}/05_deviation_by_metric.png"); plt.close(fig)

    # ---- 06 ----
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for runs, color, label in ML:
        x = runs[0]["rollout"].values
        vals = np.stack([smooth(-d["train/entropy_loss"]).values for d in runs], axis=0)
        final = np.mean([(-d["train/entropy_loss"].values[-tail500:]).mean() for d in runs])
        mean_std_band(ax, x, vals, color, f"{label} (final={final:.2f})")
    ax.set_ylabel("policy entropy (= -train/entropy_loss)"); ax.set_xlabel("rollout")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title(f"{title_prefix}: policy entropy")
    fig.tight_layout(); fig.savefig(f"{out_dir}/06_entropy.png"); plt.close(fig)

    # ---- 07/08 ----
    for panel_num, col, bar, eps_list_iso, cname in [
        ("07", "eval/J_D", D_BAR, [("eps=0 (literally over budget)", 0.0), ("eps=0.05", 0.05), ("eps=0.1", 0.1), ("eps=0.2", 0.2)], "STALL"),
        ("08", "eval/J_B", B_BAR, [("eps=0 (literally over budget)", 0.0), ("eps=0.2", 0.2), ("eps=0.5", 0.5), ("eps=1.0", 1.0)], "TILES"),
    ]:
        fig, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 5))
        for ax, (runs, color, label) in zip(axes, ML):
            thresh_list = [(lbl, bar + e) for lbl, e in eps_list_iso]
            xs, curves = sliding_prob_exceed(runs, col, thresh_list)
            alphas = np.linspace(0.35, 1.0, len(thresh_list))
            for (lbl, _), a in zip(thresh_list, alphas[::-1]):
                is_main = lbl.startswith("eps=0 ")
                ax.plot(xs, curves[lbl], color=color, alpha=1.0 if is_main else 0.4,
                         lw=2.2 if is_main else 1.2, label=lbl)
            ax.axhline(0.05, color="#e34948", ls="--", lw=1.0, label="delta=0.05")
            ax.set_ylim(-0.02, 1.02); ax.set_xlabel("rollout n")
            if ax is axes[0]: ax.set_ylabel(f"P({'J_D' if cname=='STALL' else 'J_B'} > {bar}+eps)")
            ax.set_title(label); ax.legend(frameon=False, fontsize=6.5)
        fig.suptitle(f"{title_prefix}: convergence in probability - {cname} constraint isolated (budget={bar})")
        fig.tight_layout(); fig.savefig(f"{out_dir}/{panel_num}_convergence_{cname.lower()}_isolated.png"); plt.close(fig)

    # ---- 09/10: CDF + 1 own-scale histogram per method ----
    for panel_num, col, bar, cname in [("09", "eval/J_D", D_BAR, "stall"), ("10", "eval/J_B", B_BAR, "tiles")]:
        n_panels = 1 + len(ML)
        ncols = 2 if n_panels <= 4 else 3
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.5 * nrows), squeeze=False)
        flat = axes.ravel()
        for extra in flat[n_panels:]:
            extra.set_visible(False)
        ax_cdf = flat[0]
        for runs, color, label in ML:
            over = np.maximum(0, pooled_tail(runs, col, tail1000) - bar)
            xs_ = np.sort(over); ys_ = np.arange(1, len(xs_) + 1) / len(xs_)
            ax_cdf.step(xs_, ys_, color=color, lw=2.0, where="post", label=label)
        ax_cdf.set_xscale("symlog", linthresh=1e-3)
        ax_cdf.set_xlabel(f"amount over {cname} budget"); ax_cdf.set_ylabel("cumulative probability")
        ax_cdf.set_title("CDF of over-budget amount"); ax_cdf.legend(frameon=False, fontsize=7)
        for ax, (runs, color, label) in zip(flat[1:n_panels], ML):
            dist = bar - pooled_tail(runs, col, tail1000)
            ax.hist(dist, bins=50, density=True, color=color, alpha=0.75)
            ax.axvline(0, color="#e34948", ls="--", lw=1.0)
            ax.axvline(dist.mean(), color=INK, lw=1.5)
            ax.set_xlabel(f"unused {cname} budget"); ax.set_ylabel("density")
            ax.set_title(f"{label} (own scale, mean={dist.mean():.3f})", fontsize=9)
        fig.suptitle(f"{title_prefix}: {cname.upper()} constraint isolated - CDF + per-method distance panels")
        fig.tight_layout(); fig.savefig(f"{out_dir}/{panel_num}_cdf_distance_{cname}.png"); plt.close(fig)

    # ---- 11 ----
    fig, axes = plt.subplots(1, 2, figsize=(17, 5))
    for runs, color, label in ML:
        x = runs[0]["rollout"].values
        vals = np.stack([smooth(d["eval/mean_power_mw"]).values for d in runs], axis=0)
        final = np.mean([d["eval/mean_power_mw"].values[-tail500:].mean() for d in runs])
        mean_std_band(axes[0], x, vals, color, f"{label} (final={final:.2f} mW)")
    axes[0].set_ylabel("power (mW)"); axes[0].set_xlabel("rollout"); axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("Power usage over training (NOT constrained - P dropped)")
    for runs, color, label in ML:
        pooled = pooled_tail(runs, "eval/mean_power_mw", tail1000)
        axes[1].hist(pooled, bins=50, density=True, histtype="step", linewidth=2.0, color=color,
                      label=f"{label} (mean={pooled.mean():.2f} mW)")
    axes[1].set_xlabel("power (mW)"); axes[1].set_ylabel("density"); axes[1].legend(frameon=False, fontsize=7)
    axes[1].set_title(f"Power distribution, last {tail1000} rollouts (step outlines)")
    fig.suptitle(f"{title_prefix}: power - descriptive, since P is dropped in this experiment")
    fig.tight_layout(); fig.savefig(f"{out_dir}/11_power_descriptive.png"); plt.close(fig)

    # ---- 12 ----
    fig, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 5))
    for ax, (runs, color, label) in zip(axes, ML):
        x = runs[0]["rollout"].values
        viol_d = np.stack([np.maximum(0, d["eval/J_D"].values - D_BAR) for d in runs], axis=0).mean(axis=0)
        viol_b = np.stack([np.maximum(0, d["eval/J_B"].values - B_BAR) for d in runs], axis=0).mean(axis=0)
        ax.stackplot(x, viol_d, viol_b, colors=["#3b6fa0", "#e8a33d"], labels=["viol_D", "viol_B"], alpha=0.85)
        ax.plot(x, viol_d + viol_b, color=INK, lw=0.8, label="total")
        ax.set_yscale("symlog", linthresh=1e-3); ax.set_xlabel("rollout")
        if ax is axes[0]: ax.set_ylabel("violation contribution (log scale)")
        ax.set_title(label, fontsize=9); ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle(f"{title_prefix}: violation composition - log scale reveals late-training behavior")
    fig.tight_layout(); fig.savefig(f"{out_dir}/12_violation_contribution.png"); plt.close(fig)

    # ---- 13 ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, col, bar, cname in [(axes[0], "eval/J_D", D_BAR, "stall"), (axes[1], "eval/J_B", B_BAR, "tiles")]:
        for runs, color, label in ML:
            x = runs[0]["rollout"].values
            fractions = per_seed_convergence_fraction(runs, col, bar, [5.0, 1.0, 0.2])
            for eps, alpha, lw in [(5.0, 0.3, 1.0), (1.0, 0.5, 1.0), (0.2, 1.0, 2.0)]:
                ax.plot(x, fractions[eps], color=color, alpha=alpha, lw=lw,
                         label=f"{label}" if eps == 0.2 else None)
        ax.set_ylabel(f"fraction of {len(SEEDS)} seeds converged"); ax.set_xlabel("rollout")
        ax.set_title(f"{cname}: per-seed convergence fraction (eps=0.2 bold, 5.0/1.0 faint)")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"{title_prefix}: fraction of SEEDS individually converged, vs rollout")
    fig.tight_layout(); fig.savefig(f"{out_dir}/13_per_seed_convergence_fraction.png"); plt.close(fig)

    # ---- 14 ----
    fig, ax = plt.subplots(figsize=(13, 5))
    for runs, color, label in ML:
        vals = np.stack([d["train/approx_kl"].values for d in runs], axis=0)
        tail_mean = np.mean([d["train/approx_kl"].values[-tail1000:].mean() for d in runs])
        mean_std_band(ax, x, vals, color, f"{label} (tail-{tail1000} mean={tail_mean:.5f})")
    ax.set_ylabel("approx_kl (policy change per update)"); ax.set_xlabel("rollout")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title(f"{title_prefix}: policy convergence - how much the policy still changes each update")
    fig.tight_layout(); fig.savefig(f"{out_dir}/14_policy_convergence_kl.png"); plt.close(fig)

    print(f"  wrote 14 panels to {out_dir} (15/16 skipped - eval_steps.csv not available)")


if __name__ == "__main__":
    generate_cell(
        "Old PDS+VE vs PPO", "Old PDS+VE", "#a0479e",
        "EXP3_PDS_VE_COMPARISON/01_old_pds_plus_ve/metrics/raw/oldve_own_A{s}_progress.csv",
        "EXP3_PDS_VE_COMPARISON/graphs/01_old_pds_plus_ve/meeting_followup",
    )
    generate_cell(
        "VE v2 vs PPO", "VE v2", "#c99a1e",
        "EXP3_PDS_VE_COMPARISON/02_ve_v2/metrics/raw/ve2_own_A{s}_progress.csv",
        "EXP3_PDS_VE_COMPARISON/graphs/02_ve_v2/meeting_followup",
    )
    generate_cell(
        "New PDS vs PPO", "New PDS", "#2f9e44",
        "EXP3_PDS_VE_COMPARISON/03_new_pds/metrics/raw/newpds_own_A{s}_progress.csv",
        "EXP3_PDS_VE_COMPARISON/graphs/03_new_pds/meeting_followup",
    )
    generate_cell_multi(
        "All 4 methods",
        ["PPO", "New PDS", "Old PDS+VE", "VE v2"],
        {"PPO": "#3b6fa0", "New PDS": "#2f9e44",
         "Old PDS+VE": "#a0479e", "VE v2": "#c99a1e"},
        {
            "PPO": "EXP3_PDS_VE_COMPARISON/05_ppo_baseline/raw/ppo_rerun_A{s}_progress.csv",
            "New PDS": "EXP3_PDS_VE_COMPARISON/03_new_pds/metrics/raw/newpds_own_A{s}_progress.csv",
            "Old PDS+VE": "EXP3_PDS_VE_COMPARISON/01_old_pds_plus_ve/metrics/raw/oldve_own_A{s}_progress.csv",
            "VE v2": "EXP3_PDS_VE_COMPARISON/02_ve_v2/metrics/raw/ve2_own_A{s}_progress.csv",
        },
        "EXP3_PDS_VE_COMPARISON/graphs/full_A_dropP_4methods/meeting_followup",
    )
