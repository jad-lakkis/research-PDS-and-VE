"""
plot_budget_methods_lumos.py - every method at one budget pair (test runs): PSNR,
avoidable stall (vs the allowance) and tiles per segment (vs the tile budget).
Hatched bar = outside the budgets. Uniform uses all 64 tiles by design (judged on stall only).

Usage: .venv/Scripts/python.exe plot_budget_methods_lumos.py [allowance] [b_bar] [csv]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fastsim as fs

ALLOW = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
B_BAR = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
CSV = sys.argv[3] if len(sys.argv) > 3 else "VIDEO_SURVEY/lumos5g/budget_sweep.csv"
TILES = 64 / np.sum(fs.DISC)   # J_B -> average tiles per segment
ORDER = [("oracle (actual viewport + future link)", "oracle (actual viewport + future link)", "#222222"),
         ("RL estimate", "RL estimate", "#1f77b4"),
         ("RL estimate, direction-blind (today's observation)", "RL estimate, direction-blind", "#9ecae1"),
         ("uniform, same bits as RL estimate", "uniform, same bits as RL estimate", "#2ca02c"),
         ("uniform, same bits as RL estimate (all 64 tiles)", "uniform, same bits as RL estimate", "#2ca02c"),
         ("literature, tuned to our budgets", "literature, tuned to our budgets", "#d62728"),
         ("threshold 10 s base layer, enlarged viewport (ASL360, adapted)", "ASL360 threshold (adapted)", "#ff7f0e"),
         ("average-based, fitted to our error data (MMSP rule)", "average-based, fitted margin (MMSP)", "#ff7f0e"),
         ("average-based, bounds 30/15 (MMSP)", "average-based, 30/15 (MMSP)", "#ff7f0e"),
         ("QER 200x100 (MMSP)", "QER 200x100 (MMSP)", "#ff7f0e"),
         ("QER 160x80 (MMSP)", "QER 160x80 (MMSP)", "#ff7f0e"),
         ("QER 120x60 (MMSP)", "QER 120x60 (MMSP)", "#ff7f0e"),
         ("predicted viewport only", "predicted viewport only", "#7f7f7f")]

d = pd.read_csv(CSV)
s = d[(d.split == "test") & (d.allowance == ALLOW) & (d.b_bar == B_BAR)].set_index("method")
rows = [(lab, c, s.loc[m]) for m, lab, c in ORDER if m in s.index]
y = np.arange(len(rows))[::-1]
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
for ax, (col, f, title, limit) in zip(axes, [("psnr", 1.0, "viewport PSNR (dB), higher = better", None),
                                            ("avoidable_stall", 1.0, "avoidable stall, lower = better", ALLOW),
                                            ("J_B", TILES, "tiles per segment, lower = better", B_BAR * TILES)]):
    for yi, (lab, c, r) in zip(y, rows):
        ok = r.within_budgets in (True, "True")
        ax.barh(yi, r[col] * f, color=c if ok else "white", edgecolor=c, hatch=None if ok else "///", lw=1.5)
        ax.text(r[col] * f, yi, f" {r[col] * f:.2f}", va="center", fontsize=8)
    if limit is not None:
        ax.axvline(limit, color="k", ls="--", lw=1.2)
        ax.text(limit, y[0] + 0.6, " budget", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
axes[0].set_xlim(51.0, 55.8)
axes[0].set_yticks(y)
axes[0].set_yticklabels([lab for lab, _, _ in rows], fontsize=9)
fig.suptitle(f"stall allowance {ALLOW}, tile budget {B_BAR:g} (test runs)", fontsize=11)
fig.tight_layout()
out = f"VIDEO_SURVEY/lumos5g/methods_allow{ALLOW}_B{B_BAR:g}.png"
fig.savefig(out, dpi=150)
print(out)
