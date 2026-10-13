"""
report_T.py - every method on channel T, against both budgets, overall and second by second (replay only).

Channel T = today's synthetic model with bandwidth 15 MHz, Rician K 2 dB, slow shadowing 12 dB (memory 0.95 per s),
blockages 29 dB deep, 9% of the time, 6 s long on average. Held-out viewers x 300 windows (900 episodes).
Budgets: avoidable stall <= 0.1 (stall beyond what sending only the forced tiles would cause, same episodes),
tile usage J_B <= 11 (~23.2 tiles every second). No buffer cap.
Methods: H1 viewport only, H2 QER 120x60 / 160x80, H3 average-based, H4 ASL360 threshold (10 s of playable video),
link oracle (knows the future link, not the actual viewport), oracle (also the actual viewport).

Outputs (CHANNEL_VARIANTS/): report_T.csv, fig11_channel_T_scorecard.png, fig12_channel_T_over_time.png
Usage (from the project folder): .venv/Scripts/python.exe CHANNEL_VARIANTS/report_T.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explain_A as ex  # noqa: E402
import replay_channels as rc  # noqa: E402

OUT = "CHANNEL_VARIANTS"
T_KW = dict(W_MHz=15.0, K_dB=2.0, sh_sigma=12.0, sh_rho=0.95, blk_att=29.0, blk_frac=0.09, blk_mean=6.0)
G, H, DISC = rc.G, rc.H, rc.DISC
ALLOW, B_BAR = 0.1, 11.0
B_TILES = B_BAR * 64 / DISC.sum()
COL = {"H1 viewport only": "#7f7f7f", "H2 QER 120x60": "#e377c2", "H2 QER 160x80": "#8c564b", "H3 average-based": "#d62728",
       "H4 ASL360 threshold": "#9467bd", "link oracle": "#1f77b4", "oracle": "#000000"}


def main():
    eps, R_list, _ = rc.channel_eps(T_KW, 11, list(G["held"]), rc.N_WIN)
    P, R = [p for p, _ in eps], np.stack(R_list)
    forced = np.stack([G["A_m"][p] for p in P])
    M = {m: np.stack([rc.MASK[m][p] for p in P]) for m in rc.MASK}
    Xo = rc.oracle(eps, R_list, B_BAR, G["EO"], None, return_masks=True)
    Xl = rc.oracle(eps, R_list, B_BAR, G["EM"], None, rc.TOP_LINK, return_masks=True)
    res = {m: ex.follow(P, R, lambda t, Z, m=m: M[m][:, t]) for m in rc.MASK}
    res["H4 ASL360 threshold"] = ex.follow(P, R, lambda t, Z: np.where((Z >= 10.0 * forced[:, t])[:, None],
                                                                       M["H3 average-based"][:, t], M["H1 viewport only"][:, t]))
    res["link oracle"] = ex.follow(P, R, lambda t, Z: Xl[:, t])
    res["oracle"] = ex.follow(P, R, lambda t, Z: Xo[:, t])
    d1 = res["H1 viewport only"]["stall"]
    rows = []
    for m, r in res.items():
        avoid = ((r["stall"] - d1) * DISC).sum(axis=1).mean()
        jb = (r["tiles"] / 64 * DISC).sum(axis=1).mean()
        rows.append(dict(method=m, psnr=r["psnr"].mean(), avoidable_stall=avoid, tile_usage_JB=jb, tiles_per_s=r["tiles"].mean(),
                         stall_seconds_pct=100 * ((r["stall"] - d1) > 1e-9).mean(),
                         episodes_over_allowance_pct=100 * (((r["stall"] - d1) * DISC).sum(axis=1) > ALLOW).mean(),
                         within_stall=avoid <= ALLOW + 1e-9, within_tiles=jb <= B_BAR + 1e-9))
    tab = pd.DataFrame(rows)
    tab["within_both"] = tab.within_stall & tab.within_tiles
    tab.to_csv(f"{OUT}/report_T.csv", index=False)
    pd.set_option("display.width", 250)
    print(tab.round(3).to_string(index=False))

    # scorecard: quality, stall vs allowance, tiles vs budget
    order = tab.sort_values("psnr", ascending=False).method.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    y = np.arange(len(order))[::-1]
    for yi, m in zip(y, order):
        r = tab.set_index("method").loc[m]
        sty = dict(color=COL[m] if r.within_both else "white", edgecolor=COL[m], hatch=None if r.within_both else "///", lw=1.5)
        axes[0].barh(yi, r.psnr - 51, left=51, **sty)
        axes[0].text(r.psnr + 0.04, yi, f"{r.psnr:.2f}", va="center", fontsize=9)
        axes[1].barh(yi, max(r.avoidable_stall, 0), **sty)
        axes[1].text(max(r.avoidable_stall, 0) + 0.02, yi, f"{r.avoidable_stall:.2f}", va="center", fontsize=9)
        axes[2].barh(yi, r.tile_usage_JB, **sty)
        axes[2].text(r.tile_usage_JB + 0.1, yi, f"{r.tile_usage_JB:.2f}", va="center", fontsize=9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order, fontsize=9)
    axes[0].set_xlim(51, 56.3)
    axes[1].axvline(ALLOW, color="black", ls="--", lw=1.2)
    axes[2].axvline(B_BAR, color="black", ls="--", lw=1.2)
    axes[1].set_xlim(0, max(1.7, tab.avoidable_stall.max() * 1.15))
    axes[2].set_xlim(0, 13)
    axes[0].set_xlabel("viewport PSNR (dB), higher = better")
    axes[1].set_xlabel("avoidable stall per episode (allowance 0.1, dashed)")
    axes[2].set_xlabel("tile usage J_B (budget 11, dashed)")
    for a in axes:
        a.grid(axis="x", alpha=0.3)
    axes[0].legend(handles=[Patch(facecolor="#bbbbbb", label="within both budgets"),
                            Patch(facecolor="white", edgecolor="#555555", hatch="///", label="breaks a budget")],
                   loc="lower right", fontsize=8.5)
    fig.suptitle("Channel T, 900 held-out test episodes: quality, stall against its allowance, tiles against their budget", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig11_channel_T_scorecard.png", dpi=150)
    plt.close(fig)

    # over the 36 s of an episode (average over the 900 episodes)
    t = np.arange(H)
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    below = (R < forced).mean(axis=0) * 100
    axes[0].bar(t, below, color="#d62728", alpha=0.5)
    axes[0].set_ylabel("% of episodes")
    axes[0].set_title("Share of episodes where the link is below the forced-tiles need in that second", fontsize=10, loc="left")
    for m in order:
        r = res[m]
        kw = dict(color=COL[m], lw=2.2 if "oracle" in m else 1.5, ls="--" if m == "link oracle" else "-", label=m)
        axes[1].plot(t, r["psnr"].mean(axis=0), **kw)
        axes[2].plot(t, np.cumsum(DISC * (r["stall"] - d1).mean(axis=0)), **kw)
        axes[3].plot(t, r["tiles"].mean(axis=0), **kw)
    axes[1].set_ylabel("dB")
    axes[1].set_title("Viewport quality in that second (average)", fontsize=10, loc="left")
    axes[2].axhline(ALLOW, color="black", ls="--", lw=1.2)
    axes[2].text(35, ALLOW + 0.03, "stall allowance 0.1 (whole episode)", ha="right", fontsize=8.5)
    axes[2].set_ylabel("seconds")
    axes[2].set_ylim(bottom=0)
    axes[2].set_title("Avoidable stall accumulated so far (the constraint counts the total at second 35)", fontsize=10, loc="left")
    axes[3].axhline(B_TILES, color="black", ls="--", lw=1.2)
    axes[3].text(0.3, B_TILES + 0.3, f"tile budget 11 = {B_TILES:.1f} tiles every second", fontsize=8.5)
    axes[3].set_ylabel("tiles")
    axes[3].set_title("Tiles sent in that second (average)", fontsize=10, loc="left")
    axes[3].set_xlabel("second of the 36-s episode")
    for a in axes:
        a.grid(alpha=0.3)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("Channel T: each method over the episode (average of 900 test episodes)", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(f"{OUT}/fig12_channel_T_over_time.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
