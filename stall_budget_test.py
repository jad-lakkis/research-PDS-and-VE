"""
stall_budget_test.py - same as combined_test.py (motion-aware RL estimate with
buffer reserve vs tuned literature rules vs rate-matched uniform) on the
medium-bitrate videos, with link outages, for tighter stall budgets.

Usage: .venv/Scripts/python.exe stall_budget_test.py
"""
from multiprocessing import Pool

import pandas as pd

import combined_test as ct
import design_options_scan as ds

OUT = "VIDEO_SURVEY/design_options/stall_budget.csv"
VIDEOS = ["Runner", "GateNight", "Bridge"]


def run(args):
    rows = []
    for d_bar in (0.1, 0.15, 0.3):
        ds.D_BAR = d_bar
        for r in ct.run(args):
            r["D_bar"] = d_bar
            rows.append(r)
    return rows


def main():
    prepared = [ct.prepare(n) for n in VIDEOS]
    with Pool(len(VIDEOS)) as pool:
        res = pool.map(run, prepared)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    df = df[df.B_bar == 8.0]
    p = df.pivot_table(index=["D_bar", "outage_dB", "video"], columns="method", values="psnr")
    f = df.pivot_table(index=["D_bar", "outage_dB", "video"], columns="method", values="held_feasible", aggfunc="first")
    p["RL - literature"] = p["RL estimate"] - p["literature"]
    p["RL - trivial"] = p["RL estimate"] - p["trivial"]
    print(p.round(2).to_string())
    print(f.to_string())


if __name__ == "__main__":
    main()
