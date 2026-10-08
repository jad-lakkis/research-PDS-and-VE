"""
plot_experiment_x.py - "Experiment X": PPO vs New PDS vs VE v2, with seed
6 dropped (5 seeds: 1,2,3,5,7) - an explicit, labeled sensitivity check
requested to test specific theories, NOT a replacement for the real
6-seed comparisons in EXP3/EXP4/EXP5. Old PDS+VE is deliberately excluded
per request (PPO, New PDS, VE v2 only).

Overrides the SEEDS constant on the two shared modules before calling
their already-built, N-generic cell generators (plot_exp3_full_panels's
generate_cell, plot_meeting_followup_exp3's generate_cell_multi) - both
reference a module-level SEEDS name at call time, so this is a clean,
non-invasive way to get a different seed set without touching the
shared files that every other experiment still depends on with the
real 6-seed list.

Usage: .venv/Scripts/python.exe plot_experiment_x.py
"""
import matplotlib.pyplot as plt
import numpy as np

import plot_exp3_full_panels as p3
import plot_meeting_followup_exp3 as pmf

SEEDS_X = [1, 2, 3, 5, 7]
p3.SEEDS = SEEDS_X
pmf.SEEDS = SEEDS_X

METHODS = ["PPO", "New PDS", "VE v2"]
COLORS = {"PPO": "#3b6fa0", "New PDS": "#2f9e44", "VE v2": "#c99a1e"}
PATHS = {m: p3.PATHS[m] for m in METHODS}
OUT_DIR = "EXPERIMENT_X_drop_seed6"
TITLE = "Experiment X: PPO vs New PDS vs VE v2 (seed 6 dropped, 5 seeds)"


def panel_16_ratio_vs_ppo(out_dir):
    """New panel 16: same first-single-rollout crossing rule as panel 15
    (kept as-is, raw rollout counts), just re-expressed as a speedup
    ratio with PPO normalized to 1.0x. Does NOT touch panel 15 itself -
    both panels now coexist, showing the same underlying numbers two
    ways."""
    full_data = {m: [p3.load_progress(PATHS[m].format(s=s)) for s in SEEDS_X] for m in METHODS}
    thresholds = [("viol<5", 5.0), ("viol<1", 1.0), ("viol<0.2", 0.2), ("viol=0", 1e-6)]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(thresholds))
    width = 0.8 / len(METHODS)

    raw_means = {m: [] for m in METHODS}
    for method in METHODS:
        runs = full_data[method]
        for _, thresh in thresholds:
            vals = [p3.first_rollout_below(d, "eval/violation", thresh, sustained=1) for d in runs]
            defined = [v for v in vals if v is not None]
            raw_means[method].append(np.mean(defined) if defined else np.nan)

    for j, method in enumerate(METHODS):
        color = COLORS[method]
        ratios = [raw_means["PPO"][i] / raw_means[method][i] if raw_means[method][i] else np.nan
                  for i in range(len(thresholds))]
        offset = (j - (len(METHODS) - 1) / 2) * width
        bars = ax.bar(x + offset, ratios, width, color=color, alpha=0.85, label=f"{method} (n={len(full_data[method])})")
        for xi, r in zip(x + offset, ratios):
            ax.annotate(f"{r:.2f}x", (xi, r), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=p3.INK)
    ax.axhline(1.0, color="#e34948", ls="--", lw=1.0, label="PPO baseline = 1.0x")
    ax.set_xticks(x); ax.set_xticklabels([t[0] for t in thresholds])
    ax.set_ylabel("speedup vs PPO (PPO = 1.0x; higher = faster than PPO)")
    ax.set_title(f"{TITLE}: speed to threshold, normalized to PPO=1x - first single rollout "
                 "(each method's own full run length)")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(f"{out_dir}/16_speed_to_threshold_ratio_vs_ppo.png"); plt.close(fig)
    print(f"  wrote 16_speed_to_threshold_ratio_vs_ppo.png to {out_dir}")


if __name__ == "__main__":
    p3.generate_cell(TITLE, METHODS, OUT_DIR)
    pmf.generate_cell_multi(TITLE, METHODS, COLORS, PATHS, f"{OUT_DIR}/meeting_followup")
    panel_16_ratio_vs_ppo(OUT_DIR)
