"""
plot_experiment_x_extra.py - new Experiment X panels (17-21) for findings
not covered by the standard suite: cumulative violation, multiplier peak/
final levels, early-checkpoint violation, late multiplier volatility,
violation size when infeasible. Experiment X only: PPO, New PDS, VE v2,
seeds 1,2,3,5,7, matched to the shortest method's length.

Usage: .venv/Scripts/python.exe plot_experiment_x_extra.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_exp3_full_panels as p3

SEEDS_X = [1, 2, 3, 5, 7]
METHODS = ["PPO", "New PDS", "VE v2"]
COLORS = {"PPO": "#3b6fa0", "New PDS": "#2f9e44", "VE v2": "#c99a1e"}
OUT_DIR = "EXPERIMENT_X_drop_seed6"
TAG = "Experiment X (seed 6 dropped, 5 seeds)"
TAIL = 1000


def load():
    raw = {m: [p3.load_progress(p3.PATHS[m].format(s=s)) for s in SEEDS_X] for m in METHODS}
    n = min(len(d) for r in raw.values() for d in r)
    return {m: [d.iloc[:n] for d in raw[m]] for m in METHODS}, n


def bars_with_dots(ax, groups, per_method_vals, ylabel, log=False, fmt="{:.2f}"):
    """per_method_vals[m] = list (one entry per group) of per-seed lists."""
    x = np.arange(len(groups))
    width = 0.8 / len(METHODS)
    for j, m in enumerate(METHODS):
        offset = (j - (len(METHODS) - 1) / 2) * width
        means = [np.mean(v) for v in per_method_vals[m]]
        ax.bar(x + offset, means, width, color=COLORS[m], alpha=0.85, label=m)
        for xi, v, mu in zip(x + offset, per_method_vals[m], means):
            ax.scatter([xi] * len(v), v, color=p3.INK, s=10, zorder=5)
            ax.annotate(fmt.format(mu), (xi, mu), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5, color=p3.INK)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)


def main():
    data, n = load()
    os.makedirs(OUT_DIR, exist_ok=True)

    # 17: cumulative violation over training
    fig, ax = plt.subplots(figsize=(11, 5))
    for m in METHODS:
        x = data[m][0]["rollout"].values
        cum = np.stack([d["eval/violation"].cumsum().values for d in data[m]], axis=0)
        p3._mean_std_band(ax, x, cum, COLORS[m], f"{m} (final={cum[:, -1].mean():.0f})")
    ax.set_xlabel("rollout"); ax.set_ylabel("cumulative violation (sum over evaluations)")
    ax.set_title(f"{TAG}: total constraint violation accumulated during training (lower = better)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/17_cumulative_violation.png"); plt.close(fig)

    # 18: multiplier peak and final levels
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, name in [(axes[0], "lagrangian/mu_D", "mu_D (stall)"), (axes[1], "lagrangian/mu_B", "mu_B (tiles)")]:
        vals = {m: [[d[col].max() for d in data[m]], [d[col].values[-TAIL:].mean() for d in data[m]]] for m in METHODS}
        bars_with_dots(ax, ["peak over training", f"final (last {TAIL})"], vals, name)
        ax.set_title(f"{name}: how much penalty pressure each method needed")
    fig.suptitle(f"{TAG}: Lagrange multiplier levels (lower = constraint satisfied with less penalty)")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/18_multiplier_peak_and_final.png"); plt.close(fig)

    # 19: violation at fixed early checkpoints (log scale)
    checkpoints = [250, 500, 1000, 1500]
    vals = {m: [[d["eval/violation"].values[c - 50:c].mean() for d in data[m]] for c in checkpoints] for m in METHODS}
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars_with_dots(ax, [f"rollout {c}" for c in checkpoints], vals,
                   "violation (mean of the 50 rollouts ending at checkpoint)", log=True)
    ax.set_title(f"{TAG}: violation at fixed points early in training (log scale, lower = better)")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/19_violation_at_early_checkpoints.png"); plt.close(fig)

    # 20: late-training multiplier volatility (within-run std)
    vals = {m: [[d[c].values[-TAIL:].std() for d in data[m]] for c in ["lagrangian/mu_D", "lagrangian/mu_B"]]
            for m in METHODS}
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_with_dots(ax, ["mu_D (stall)", "mu_B (tiles)"], vals,
                   f"within-run std of multiplier, last {TAIL} rollouts", fmt="{:.3f}")
    ax.set_title(f"{TAG}: late-training multiplier volatility (lower = steadier controller)")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/20_multiplier_late_volatility.png"); plt.close(fig)

    # 21: when a method violates late in training, by how much
    def cond_size(d):
        v = d["eval/violation"].values[-TAIL:]
        v = v[v > 1e-9]
        return v.mean() if len(v) else 0.0

    def large_share(d):
        return np.mean(d["eval/violation"].values[-TAIL:] > 0.5) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bars_with_dots(axes[0], ["violation size when infeasible"],
                   {m: [[cond_size(d) for d in data[m]]] for m in METHODS}, "mean violation | infeasible", fmt="{:.3f}")
    axes[0].set_title(f"Size of a violation when one happens (last {TAIL})")
    bars_with_dots(axes[1], ["% rollouts with violation > 0.5"],
                   {m: [[large_share(d) for d in data[m]]] for m in METHODS}, "% of rollouts", fmt="{:.1f}")
    axes[1].set_title(f"Large violations (> 0.5), last {TAIL}")
    fig.suptitle(f"{TAG}: late-training violations are small boundary touches, not large failures")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/21_violation_size_when_infeasible.png"); plt.close(fig)

    # 22: all-seed alignment on each constraint, two definitions. Tolerance
    # is RELATIVE to each budget (eps_rel * budget), not a fixed absolute
    # 0.2 - a fixed 0.2 is 67% of the stall budget but only 2.5% of the
    # tile budget, so it wasn't comparable across the two constraints.
    eps_rel = 0.20
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, cname, col, bar in [(axes[0], "Stall", "eval/J_D", p3.D_BAR), (axes[1], "Tiles", "eval/J_B", p3.B_BAR)]:
        thresh = bar * (1 + eps_rel)
        firsts, simult = {}, {}
        for m in METHODS:
            ok = np.stack([d[col].values <= thresh for d in data[m]])
            firsts[m] = [int(np.argmax(r)) + 1 for r in ok]
            simult[m] = int(np.argmax(ok.all(axis=0))) + 1
        groups = ["slowest seed's first pass\n(every seed has reached it once)",
                  "all 5 seeds passing\nat the same rollout"]
        x = np.arange(len(groups))
        width = 0.8 / len(METHODS)
        ppo_vals = [max(firsts["PPO"]), simult["PPO"]]
        for j, m in enumerate(METHODS):
            offset = (j - (len(METHODS) - 1) / 2) * width
            vals = [max(firsts[m]), simult[m]]
            ax.bar(x + offset, vals, width, color=COLORS[m], alpha=0.85, label=m)
            ax.scatter([x[0] + offset] * len(firsts[m]), firsts[m], color=p3.INK, s=12, zorder=5,
                       label="each seed's first pass" if j == 0 else None)
            for xi, v, pv in zip(x + offset, vals, ppo_vals):
                txt = f"{v}" if m == "PPO" else f"{v}\n{pv / v:.2f}x"
                ax.annotate(txt, (xi, v), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=7.5, color=p3.INK)
        ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8.5)
        ax.set_ylabel("rollout (lower = faster)")
        ax.set_title(f"{cname}: {'J_D' if cname == 'Stall' else 'J_B'} <= budget {bar} x {1 + eps_rel:.2f} = {thresh:.2f}")
        ax.set_ylim(0, max(max(ppo_vals), max(simult.values())) * 1.22)
        ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=4)
    fig.suptitle(f"{TAG}: when does every seed get within {eps_rel:.0%} of each budget? "
                 "(held-out test; x = speedup vs PPO)")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/22_all_seeds_reach_constraint.png"); plt.close(fig)

    print(f"wrote panels 17-22 to {OUT_DIR} (matched n={n})")


if __name__ == "__main__":
    main()
