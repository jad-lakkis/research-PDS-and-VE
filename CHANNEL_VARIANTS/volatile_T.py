"""
volatile_T.py - more volatility around channel T (replay only, no training).

Channel T = MMSP'25 channel (d 50 m, 50 mW, same path loss and rate formula) at 15 MHz, with slow shadowing and
blockages (29 dB, 9% of the time, 6 s long). Grid, fixed in advance:
  shadowing 12, 14, 16 dB | shadowing memory 0.95 (slow swings) or 0.8 (fast swings) | fading K 2 dB or Rayleigh |
  blockage depth 29 or 35 dB
Each setting: held-out viewers x 300 windows, allowance 0.1, tile budget 11, no buffer cap, faithful ASL360.
Output: CHANNEL_VARIANTS/volatile_T.csv
Usage (from the project folder): .venv/Scripts/python.exe CHANNEL_VARIANTS/volatile_T.py
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


def main():
    t0, rows = time.time(), []
    for sh, rho, K, att in itertools.product((12.0, 14.0, 16.0), (0.95, 0.8), (2.0, None), (29.0, 35.0)):
        kw = dict(W_MHz=15.0, K_dB=K, sh_sigma=sh, sh_rho=rho, blk_att=att, blk_frac=0.09, blk_mean=6.0)
        eps, R, r = rc.channel_eps(kw, 11, sc.HELD, rc.N_WIN)
        rows.append(dict(kw, **sc.evaluate(eps, R, r)))
        print(f"shadowing {sh:.0f} memory {rho} K {K} blockage {att:.0f}: room {rows[-1]['room']:.2f} ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("CHANNEL_VARIANTS/volatile_T.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["sh_sigma", "sh_rho", "K_dB", "blk_att", "below_need_pct", "over_2x_pct", "lag1", "floor", "best", "psnr H4",
            "avoid H4", "avoid H2 QER 120" if "avoid H2 QER 120" in df else "avoid QER 120", "avoid H3", "link_oracle", "oracle", "room"]
    print(df.sort_values("room", ascending=False)[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
