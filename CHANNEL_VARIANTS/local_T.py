"""
local_T.py - last, bounded search around channel T (replay only, no training).

Grid fixed in advance (memory kept at 0.95 so the link stays predictable; K 2 dB, blockage depth 29 dB as in T):
  bandwidth 12, 13, 14, 15, 16 MHz | shadowing 10, 12 dB | blockages 9% or 15% of the time | 6 or 10 s long
Each setting: held-out viewers x 300 windows, allowance 0.1, tile budget 11, no buffer cap, faithful ASL360.
The best setting is re-checked on two fresh sets of random draws.
Output: CHANNEL_VARIANTS/local_T.csv
Usage (from the project folder): .venv/Scripts/python.exe CHANNEL_VARIANTS/local_T.py
"""
import itertools
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_channels as rc  # noqa: E402
import search_channels as sc  # noqa: E402

BASE_T = dict(K_dB=2.0, sh_rho=0.95, blk_att=29.0)


def run(kw, seed):
    eps, R, r = rc.channel_eps(kw, seed, sc.HELD, rc.N_WIN)
    return sc.evaluate(eps, R, r)


def main():
    t0, rows = time.time(), []
    for W, sh, frac, mean in itertools.product((12.0, 13.0, 14.0, 15.0, 16.0), (10.0, 12.0), (0.09, 0.15), (6.0, 10.0)):
        kw = dict(BASE_T, W_MHz=W, sh_sigma=sh, blk_frac=frac, blk_mean=mean)
        rows.append(dict(kw, seed=11, **run(kw, 11)))
        print(f"W {W:.0f} sh {sh:.0f} blocked {frac:.2f} x {mean:.0f} s: room {rows[-1]['room']:.2f} ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    best = df.sort_values("room", ascending=False).iloc[0]
    kw = {k: best[k] for k in ("K_dB", "sh_rho", "blk_att", "W_MHz", "sh_sigma", "blk_frac", "blk_mean")}
    for seed in (21, 31):
        rows.append(dict(kw, seed=seed, **run(kw, seed)))
    df = pd.DataFrame(rows)
    df.to_csv("CHANNEL_VARIANTS/local_T.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["W_MHz", "sh_sigma", "blk_frac", "blk_mean", "seed", "below_need_pct", "over_2x_pct", "lag1", "floor", "best",
            "psnr H4", "avoid H4", "link_oracle", "room"]
    print(df.sort_values("room", ascending=False)[cols].head(12).round(2).to_string(index=False))
    print(df[df.seed != 11][cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
