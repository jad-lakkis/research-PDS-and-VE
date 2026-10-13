"""
compare_volatile.py - channel T and three more volatile versions, side by side (replay only, no training).

All are the MMSP'25 channel (d 50 m, 50 mW) at 15 MHz with blockages 9% of the time, 6 s long; they differ in
shadowing strength / speed, fading depth and blockage depth. For each: 200 s of the link, then every method's quality
and avoidable stall on 900 held-out test episodes (allowance 0.1, tile budget 11, faithful ASL360).
Output: CHANNEL_VARIANTS/fig13_volatile_channels.png
Usage (from the project folder): .venv/Scripts/python.exe CHANNEL_VARIANTS/compare_volatile.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import channel_variants as cv  # noqa: E402
import replay_channels as rc  # noqa: E402

BASE = dict(W_MHz=15.0, blk_frac=0.09, blk_mean=6.0)
CHANNELS = [("T: shadowing 12 dB, slow swings, K 2 dB, blockages 29 dB", dict(sh_sigma=12.0, sh_rho=0.95, K_dB=2.0, blk_att=29.0)),
            ("T with fast swings (memory 0.8)", dict(sh_sigma=12.0, sh_rho=0.8, K_dB=2.0, blk_att=29.0)),
            ("shadowing 16 dB, fast swings", dict(sh_sigma=16.0, sh_rho=0.8, K_dB=2.0, blk_att=29.0)),
            ("most volatile: shadowing 16 dB, Rayleigh fading, blockages 35 dB", dict(sh_sigma=16.0, sh_rho=0.95, K_dB=None, blk_att=35.0))]
METHODS = ["H1 viewport only", "H2 QER 120x60", "H2 QER 160x80", "H3 average-based", "H4 ASL360 threshold", "link oracle", "oracle"]
COL = {"H1 viewport only": "#7f7f7f", "H2 QER 120x60": "#e377c2", "H2 QER 160x80": "#8c564b", "H3 average-based": "#d62728",
       "H4 ASL360 threshold": "#9467bd", "link oracle": "#1f77b4", "oracle": "#000000"}


def main(channels=CHANNELS, out="CHANNEL_VARIANTS/fig13_volatile_channels.png", title="Channel T vs more volatile versions"):
    fig, axes = plt.subplots(len(channels), 3, figsize=(17, 3.4 * len(channels)), gridspec_kw=dict(width_ratios=[1.6, 1, 1]), squeeze=False)
    for row, (name, kw) in zip(axes, channels):
        kw = {**cv.BASE, **BASE, **kw}
        x = cv.rates(**kw, n=600, seed=5)[100:300]
        t = np.arange(len(x))
        row[0].plot(t, x, color="#1f77b4", lw=1.1)
        row[0].fill_between(t, 0, x, where=x < cv.NEED_FORCED, color="#d62728", alpha=0.25, step="mid")
        row[0].axhline(cv.NEED_FORCED, color="#d62728", ls="--", lw=1)
        row[0].axhline(2 * cv.NEED_FORCED, color="#2ca02c", ls=":", lw=1)
        row[0].set_ylim(0, 450)
        row[0].set_ylabel("Mbit/s")
        row[0].set_title(name, fontsize=9.5, loc="left")
        row[0].grid(alpha=0.3)
        eps, R, r_all = rc.channel_eps(kw, 11, list(rc.G["held"]), rc.N_WIN)
        hs = rc.heuristics([p for p, _ in eps], np.stack(R), None)
        hs["link oracle"] = rc.oracle(eps, R, 11.0, rc.G["EM"], None, rc.TOP_LINK)
        hs["oracle"] = rc.oracle(eps, R, 11.0, rc.G["EO"], None)
        floor = hs["H1 viewport only"]["J_D"]
        y = np.arange(len(METHODS))[::-1]
        best_ok = max(hs[m]["psnr"] for m in METHODS[:5] if hs[m]["J_D"] - floor <= 0.1 + 1e-9)
        for yi, m in zip(y, METHODS):
            avoid = hs[m]["J_D"] - floor
            ok = avoid <= 0.1 + 1e-9 and hs[m]["J_B"] <= 11.0 + 1e-9
            sty = dict(color=COL[m] if ok else "white", edgecolor=COL[m], hatch=None if ok else "///", lw=1.4)
            row[1].barh(yi, hs[m]["psnr"] - 51, left=51, **sty)
            row[1].text(hs[m]["psnr"] + 0.05, yi, f"{hs[m]['psnr']:.2f}", va="center", fontsize=8)
            row[2].barh(yi, max(avoid, 0), **sty)
            row[2].text(max(avoid, 0) + 0.03, yi, f"{avoid:.2f}", va="center", fontsize=8)
        s = cv.stats(r_all / 1e6)
        row[1].set_title(f"quality (dB) - room for RL {hs['link oracle']['psnr'] - best_ok:.2f} dB", fontsize=9.5)
        row[2].set_title(f"avoidable stall (allowance 0.1); link below need {s['pct_below_forced_need']:.0f}% of seconds", fontsize=9.5)
        row[2].axvline(0.1, color="black", ls="--", lw=1.1)
        row[1].set_xlim(51, 56.3)
        row[2].set_xlim(0, 2.5)
        for a in row[1:]:
            a.set_yticks(y)
            a.set_yticklabels(METHODS if a is row[1] else [], fontsize=8)
            a.grid(axis="x", alpha=0.3)
        print(f"{name}: done", flush=True)
    axes[-1][0].set_xlabel("time (s)  (red dashed = forced-tiles need, green dotted = twice that, red = link below the need)")
    fig.suptitle(f"{title} (hatched = breaks a budget)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=140)


if __name__ == "__main__":
    main()
