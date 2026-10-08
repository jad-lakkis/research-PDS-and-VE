"""
plot_experiment_x_tradeoffs.py - tradeoff CURVES (panels 23-29) for
Experiment X: PPO, New PDS, VE v2, seeds 1,2,3,5,7, matched to the shortest
method's length. Every panel is a curve traced by a parameter - either
training progress (rollout) or the violation tolerance eps - so each
figure shows how one quantity is exchanged for another.

Usage: .venv/Scripts/python.exe plot_experiment_x_tradeoffs.py
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
K = 50  # rolling window for training-path curves
EPS_GRID = [5, 3, 2, 1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.0]
EPS_TICKS = [5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0]
MARKS = [50, 250, 500, 1000, 2000]
LABELED = [250, 1000]  # checkpoints that get a text label (plus the end)


def load():
    raw = {m: [p3.load_progress(p3.PATHS[m].format(s=s)) for s in SEEDS_X] for m in METHODS}
    n = min(len(d) for r in raw.values() for d in r)
    return {m: [d.iloc[:n] for d in raw[m]] for m in METHODS}, n


def smooth(a):
    return np.convolve(a, np.ones(K) / K, mode="valid")


def mean_path(data, m, col):
    return smooth(np.stack([d[col].values for d in data[m]]).mean(axis=0))


def mark_path(ax, xs, ys, m, n):
    for r in MARKS + [n]:
        i = min(max(r - K, 0), len(xs) - 1)
        ax.scatter(xs[i], ys[i], color=COLORS[m], s=28, edgecolor=p3.INK, linewidth=0.6, zorder=4)
        if r in LABELED or r == n:
            ax.annotate("end" if r == n else f"r{r}", (xs[i], ys[i]), xytext=(5, 5),
                        textcoords="offset points", fontsize=7.5, color=p3.INK)


def eps_axis(ax):
    ax.set_xscale("symlog", linthresh=0.01); ax.invert_xaxis()
    ax.set_xticks(EPS_TICKS); ax.set_xticklabels([str(e) for e in EPS_TICKS])
    ax.minorticks_off()
    ax.set_xlabel("allowed violation eps (stricter to the right; 0 = fully feasible)")


def first_reach(v, eps):
    hit = np.where(v <= max(eps, 1e-9))[0]
    return int(hit[0]) + 1 if len(hit) else np.nan


def main():
    data, n = load()
    os.makedirs(OUT_DIR, exist_ok=True)
    for old in ["23_tradeoff_operating_point.png", "24_tradeoff_speed_vs_reliability.png",
                "26_tradeoff_resources_vs_coverage.png", "27_tradeoff_samples_vs_wallclock.png",
                "28_tradeoff_resources_vs_coverage_path.png"]:
        if os.path.exists(f"{OUT_DIR}/{old}"):
            os.remove(f"{OUT_DIR}/{old}")

    # 23: stall cost vs tile cost, traced through training
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for m in METHODS:
        xd, yb = mean_path(data, m, "eval/J_D"), mean_path(data, m, "eval/J_B")
        ax.plot(xd, yb, color=COLORS[m], lw=2.0, label=m)
        mark_path(ax, xd, yb, m, n)
    ax.axvline(p3.D_BAR, color="#e34948", ls="--", lw=1.0, label=f"stall budget {p3.D_BAR}")
    ax.axhline(p3.B_BAR, color="#e34948", ls=":", lw=1.0, label=f"tile budget {p3.B_BAR}")
    ax.set_xscale("symlog", linthresh=0.01); ax.set_xlim(left=0)
    ax.set_xlabel("stall cost J_D (symlog)"); ax.set_ylabel("tile cost J_B")
    ax.set_title(f"{TAG}\nstall vs tile cost through training (rolling-{K}, seed mean) - "
                 "each path starts top-right and ends inside the budget box")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/23_tradeoff_stall_vs_tiles_path.png"); plt.close(fig)

    # 25: quality vs violation, traced through training
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for m in METHODS:
        xv, yq = mean_path(data, m, "eval/violation"), mean_path(data, m, "eval/J_Q")
        ax.plot(xv, yq, color=COLORS[m], lw=2.0, label=m)
        mark_path(ax, xv, yq, m, n)
    ax.set_xscale("symlog", linthresh=0.01); ax.set_xlim(left=0); ax.invert_xaxis()
    ax.set_xlabel("held-out violation (symlog; feasible to the right)"); ax.set_ylabel("held-out quality J_Q")
    ax.set_title(f"{TAG}\nquality given up to become feasible (rolling-{K}, seed mean) - paths run left to right")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/25_tradeoff_quality_vs_violation_path.png"); plt.close(fig)

    # precompute eps sweeps
    reach, reli = {}, {}
    for m in METHODS:
        reach[m] = np.array([[first_reach(d["eval/violation"].values, e) for d in data[m]] for e in EPS_GRID])
        reli[m] = np.array([[np.mean(d["eval/violation"].values[-TAIL:] <= max(e, 1e-9)) * 100 for d in data[m]]
                            for e in EPS_GRID])
    eps_x = np.array(EPS_GRID)

    # 24: strictness vs time to reach it
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in METHODS:
        mu, sd = np.nanmean(reach[m], axis=1), np.nanstd(reach[m], axis=1)
        ax.fill_between(eps_x, mu - sd, mu + sd, color=COLORS[m], alpha=0.12, linewidth=0)
        ax.plot(eps_x, mu, color=COLORS[m], lw=2.0, marker="o", ms=4, label=f"{m} (mean +/- std over seeds)")
    eps_axis(ax)
    ax.set_ylabel("rollouts until first reaching violation <= eps")
    ax.set_title(f"{TAG}\nstrictness vs learning time - how long each level of compliance takes")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/24_tradeoff_strictness_vs_time.png"); plt.close(fig)

    # 26: tolerance vs late reliability
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in METHODS:
        mu, sd = reli[m].mean(axis=1), reli[m].std(axis=1)
        ax.fill_between(eps_x, np.clip(mu - sd, 0, 100), np.clip(mu + sd, 0, 100), color=COLORS[m], alpha=0.12, linewidth=0)
        ax.plot(eps_x, mu, color=COLORS[m], lw=2.0, marker="o", ms=4, label=m)
    eps_axis(ax)
    ax.set_ylim(-3, 103)
    ax.set_ylabel(f"% of last {TAIL} rollouts with violation <= eps")
    ax.set_title(f"{TAG}\ntolerance vs late reliability - how much slack each method needs to be reliable")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/26_tradeoff_tolerance_vs_reliability.png"); plt.close(fig)

    # 27: speed-reliability frontier, traced by eps
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for m in METHODS:
        xs, ys = np.nanmean(reach[m], axis=1), reli[m].mean(axis=1)
        ax.plot(xs, ys, color=COLORS[m], lw=2.0, marker="o", ms=4, label=m)
        for e, x_, y_ in zip(EPS_GRID, xs, ys):
            if e in (1, 0.2, 0.05, 0.01, 0.0):
                ax.annotate(f"eps={e}", (x_, y_), xytext=(4, -10), textcoords="offset points", fontsize=7, color=COLORS[m])
    ax.set_xlabel("rollouts to first reach violation <= eps (lower = faster)")
    ax.set_ylabel(f"% of last {TAIL} rollouts with violation <= eps (higher = more reliable)")
    ax.set_title(f"{TAG}\nspeed-reliability frontier traced by eps (curves further top-left are better)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/27_tradeoff_speed_reliability_frontier.png"); plt.close(fig)

    # 28: what each resource buys - tiles buy coverage, power buys rate (fewer stalls)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    for m in METHODS:
        for ax, xc, yc in [(axes[0], "eval/mean_n_tiles", "eval/mean_coverage"),
                           (axes[1], "eval/mean_power_mw", "eval/mean_stall_sec")]:
            xs, ys = mean_path(data, m, xc), mean_path(data, m, yc)
            ax.plot(xs, ys, color=COLORS[m], lw=2.0, label=m)
            mark_path(ax, xs, ys, m, n)
    axes[0].set_xlabel("tiles sent per segment"); axes[0].set_ylabel("viewport coverage")
    axes[0].set_title("tiles spent vs coverage bought")
    axes[1].set_xlabel("transmit power (mW, unconstrained)"); axes[1].set_ylabel("stall seconds per segment (symlog)")
    axes[1].set_yscale("symlog", linthresh=0.01); axes[1].set_ylim(bottom=0)
    axes[1].set_title("power spent vs stall")
    for a in axes:
        a.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{TAG}: what each resource buys, traced through training (held-out eval, rolling-{K})")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/28_tradeoff_resources_path.png"); plt.close(fig)

    # 29: violation vs rollouts and vs wall-clock (compute cost)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    for m in METHODS:
        v = mean_path(data, m, "eval/violation")
        r = np.arange(K, K + len(v))
        sec = np.mean([np.median(np.diff(d["time/time_elapsed"].values)) for d in data[m]])
        axes[0].plot(r, v, color=COLORS[m], lw=2.0, label=m)
        axes[1].plot(r * sec / 3600, v, color=COLORS[m], lw=2.0, label=f"{m} (~{sec:.1f} s/rollout)")
    for a, lab in [(axes[0], "rollouts (environment interaction)"), (axes[1], "wall-clock hours")]:
        a.set_yscale("symlog", linthresh=0.01); a.set_xlabel(lab)
        a.set_ylabel("held-out violation (rolling-50, symlog)"); a.legend(frameon=False, fontsize=8)
    axes[0].set_title("violation per environment interaction")
    axes[1].set_title("violation per wall-clock hour (different pods - indicative)")
    fig.suptitle(f"{TAG}: sample efficiency vs compute cost")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/29_tradeoff_samples_vs_wallclock.png"); plt.close(fig)

    # numbers for the write-up
    for m in METHODS:
        print(f"{m}: reach(mean) " + ", ".join(f"eps={e}:{np.nanmean(reach[m][i]):.0f}" for i, e in enumerate(EPS_GRID)))
        print(f"{m}: late reliability " + ", ".join(f"eps={e}:{reli[m][i].mean():.1f}%" for i, e in enumerate(EPS_GRID)))
    print(f"wrote tradeoff curves 23-29 to {OUT_DIR} (matched n={n})")


if __name__ == "__main__":
    main()
