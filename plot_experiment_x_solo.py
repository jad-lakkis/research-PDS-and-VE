"""
plot_experiment_x_solo.py - Experiment X's solo runs: PPO, New PDS, VE v2
each ALONE (no cross-method comparison), using Experiment X's own 5-seed
set (seed 6 dropped: 1,2,3,5,7) - same "solo" concept as EXP4_SOLO, just
restricted to Experiment X's method roster and seed set.

Overrides SEEDS on plot_exp3_full_panels (used by the imported
generate_cell for the 15-panel suite) AND on plot_exp4_solo itself (its
own generate_cell_solo references the module-level SEEDS it imported
from plot_meeting_followup_exp3 at import time - a local copy, not a
live link, so it must be overridden directly on plot_exp4_solo, not on
plot_meeting_followup_exp3).

Usage: .venv/Scripts/python.exe plot_experiment_x_solo.py
"""
import plot_exp3_full_panels as p3
import plot_exp4_solo as solo

SEEDS_X = [1, 2, 3, 5, 7]
p3.SEEDS = SEEDS_X
solo.SEEDS = SEEDS_X

METHODS_X = ["PPO", "New PDS", "VE v2"]
OUT_ROOT = "EXPERIMENT_X_drop_seed6/solo"

if __name__ == "__main__":
    for method_name in METHODS_X:
        color, path_tmpl = solo.METHODS[method_name]
        folder = solo.FOLDER_NAME[method_name]
        out_dir = f"{OUT_ROOT}/{folder}"
        solo.generate_full_panels_cell(f"{method_name} (Experiment X solo, seed 6 dropped)", [method_name], out_dir)
        solo.generate_cell_solo(method_name, method_name, color, path_tmpl, f"{out_dir}/meeting_followup")
