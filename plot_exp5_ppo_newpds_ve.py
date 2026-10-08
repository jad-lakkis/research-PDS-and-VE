"""
plot_exp5_ppo_newpds_ve.py - "PPO vs New PDS vs VE" comparison: PPO, New
PDS, Old PDS+VE, VE v2 together (Exp1 PDS deliberately excluded - this
cell is specifically about the two VE variants against the structural
PDS fix and the baseline). Matched to the same rollout count (shortest
method's own min-seed length), same convention as every other multi-
method cell in this project.

Usage: .venv/Scripts/python.exe plot_exp5_ppo_newpds_ve.py
"""
import matplotlib.pyplot as plt
import numpy as np

import plot_exp3_full_panels as p3
from plot_exp3_full_panels import generate_cell as generate_full_panels_cell
from plot_meeting_followup_exp3 import generate_cell_multi

METHODS = ["PPO", "New PDS", "Old PDS+VE", "VE v2"]
COLORS = {"PPO": "#3b6fa0", "New PDS": "#2f9e44", "Old PDS+VE": "#a0479e", "VE v2": "#c99a1e"}
PATHS = {
    "PPO": "EXP3_PDS_VE_COMPARISON/05_ppo_baseline/raw/ppo_rerun_A{s}_progress.csv",
    "New PDS": "EXP3_PDS_VE_COMPARISON/03_new_pds/metrics/raw/newpds_own_A{s}_progress.csv",
    "Old PDS+VE": "EXP3_PDS_VE_COMPARISON/01_old_pds_plus_ve/metrics/raw/oldve_own_A{s}_progress.csv",
    "VE v2": "EXP3_PDS_VE_COMPARISON/02_ve_v2/metrics/raw/ve2_own_A{s}_progress.csv",
}
OUT_DIR = "EXP5_PPO_NEWPDS_VE"
TITLE = "PPO vs New PDS vs VE"


def panel_16_ratio_vs_ppo(out_dir):
    """Same panel-16 concept as Experiment X: panel 15's first-single-
    rollout crossing rule, re-expressed as a speedup ratio with PPO
    normalized to 1.0x, for all 3 non-PPO methods here (New PDS,
    Old PDS+VE, VE v2). Each method's own full (own-min-seed) length,
    same as panel 15 itself - does not touch panel 15."""
    non_ppo = [m for m in METHODS if m != "PPO"]
    full_data = {m: [p3.load_progress(PATHS[m].format(s=s)) for s in p3.SEEDS] for m in METHODS}
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
        ax.bar(x + offset, ratios, width, color=color, alpha=0.85, label=f"{method} (n={len(full_data[method])})")
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
    generate_full_panels_cell(TITLE, METHODS, OUT_DIR)
    generate_cell_multi(TITLE, METHODS, COLORS, PATHS, f"{OUT_DIR}/meeting_followup")
    panel_16_ratio_vs_ppo(OUT_DIR)
