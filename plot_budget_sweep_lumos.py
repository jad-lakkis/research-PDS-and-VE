"""
plot_budget_sweep_lumos.py - PSNR vs tile budget at three stall allowances (test runs),
from VIDEO_SURVEY/lumos5g/budget_sweep.csv. Hollow marker = outside the budgets.

Usage: .venv/Scripts/python.exe plot_budget_sweep_lumos.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

d = pd.read_csv("VIDEO_SURVEY/lumos5g/budget_sweep.csv")
d = d[d.split == "test"]
METHODS = [("oracle (actual viewport + future link)", "oracle", "#222222"),
           ("RL estimate", "RL estimate", "#1f77b4"),
           ("literature, tuned to our budgets", "literature, tuned", "#d62728"),
           ("best published rule within budgets", "best published rule", "#ff7f0e"),
           ("uniform, same bits as RL estimate", "uniform, same bits", "#2ca02c"),
           ("predicted viewport only", "predicted viewport only", "#7f7f7f")]
ALLOW = (0.1, 0.2, 0.3)
fig, axes = plt.subplots(1, len(ALLOW), figsize=(15, 4.6), sharey=True)
for ax, a in zip(axes, ALLOW):
    s = d[d.allowance == a]
    for m, label, c in METHODS:
        x = s[s.method == m].sort_values("b_bar")
        ax.plot(x.b_bar, x.psnr, "-", color=c, lw=2, label=label)
        ok = x.within_budgets.astype(bool)
        ax.plot(x.b_bar[ok], x.psnr[ok], "o", color=c, ms=6)
        ax.plot(x.b_bar[~ok], x.psnr[~ok], "o", mfc="white", mec=c, ms=6)
    ax.set_title(f"stall allowance {a} (D̄ = floor + {a})")
    ax.set_xlabel("tile budget B̄")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("viewport PSNR (dB), higher = better")
axes[-1].legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig("VIDEO_SURVEY/lumos5g/budget_sweep.png", dpi=150)
print("saved")
