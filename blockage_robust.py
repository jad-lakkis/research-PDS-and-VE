"""
blockage_robust.py - robustness re-run of blockage_test.py on the medium-bitrate
videos with 10 channel/blockage realisations per viewer (training and held-out)
and three blockage depths. Same methods and selection as blockage_test.py.

Usage: .venv/Scripts/python.exe blockage_robust.py
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

import blockage_test as bt
import config
import design_options_scan as ds
import fastsim as fs

OUT = "VIDEO_SURVEY/design_options/blockage_robust.csv"
VIDEOS = ["Runner", "GateNight", "Bridge"]
TRAIN_SEEDS = tuple(range(2000, 2010))
HELD_SEEDS = tuple(range(1000, 1010))


def prepare(name):
    ds.TRAIN_SEEDS, ds.HELD_SEEDS = TRAIN_SEEDS, HELD_SEEDS
    return ds.prepare(name)


def run(args):
    ds.TRAIN_SEEDS, ds.HELD_SEEDS = TRAIN_SEEDS, HELD_SEEDS
    bt.ATTS = (10.0, 12.5, 15.0)
    return bt.run(args)


def main():
    prepared = [prepare(n) for n in VIDEOS]
    with Pool(len(VIDEOS)) as pool:
        res = pool.map(run, prepared)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    cols = ["trivial", "literature", "reactive (RL estimate)", "planning (RL estimate)", "foresight (not causal)"]
    for metric in ["psnr", "J_D"]:
        p = df.pivot_table(index=["blockage_dB", "video"], columns="method", values=metric)
        print(f"===== {metric} =====")
        print(p[[c for c in cols if c in p.columns]].round(3).to_string())
    f = df.pivot_table(index=["blockage_dB", "video"], columns="method", values="held_feasible", aggfunc="first")
    print("===== within both budgets on held-out viewers =====")
    print(f[[c for c in cols if c in f.columns]].to_string())


if __name__ == "__main__":
    main()
