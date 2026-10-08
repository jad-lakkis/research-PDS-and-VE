"""
motion_room_test.py - does knowing which way the head just moved create room
for RL? Compares, on held-out viewers, the RL estimate that ranks extra tiles
by expected coverage under the training error distribution (as in
design_options_scan.py) with a motion-aware version whose error distribution
is conditioned on the previous one-second head move (direction and size,
known at decision time). Literature and ceiling values are read from
design_options.csv for the same designs.

Usage: .venv/Scripts/python.exe motion_room_test.py
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config
import design_options_scan as ds
import fastsim as fs
import room_to_learn as rl

OUT = "VIDEO_SURVEY/design_options/motion_room.csv"
DESIGNS = [("D0", 0.5, 8.0), ("D1", 0.5, 7.5), ("D2", 0.25, 8.0)]
BINS = [-np.inf, -30.0, -10.0, 10.0, 30.0, np.inf]


def prev_move(tr):
    """previous one-second azimuth move, known at decision time (= pred_t - pred_{t-1})."""
    return np.concatenate([[0.0], rl.wsigned(np.diff(tr["pt"]))])


def run(args):
    v, train, held = args
    T = v.traj
    tr_eps = [(p, s) for p in train for s in ds.TRAIN_SEEDS]
    he_eps = [(p, s) for p in held for s in ds.HELD_SEEDS]
    e_t = np.concatenate([rl.wsigned(T[p]["at"] - T[p]["pt"])[2:] for p in train])
    e_p = np.concatenate([(T[p]["ap"] - T[p]["pp"])[2:] for p in train])
    mv = np.concatenate([prev_move(T[p])[2:] for p in train])
    mbin = np.digitize(mv, BINS)
    scores = {}
    for p in T:
        pm = np.digitize(prev_move(T[p]), BINS)
        E0 = np.stack([rl.weights(T[p]["pt"][t] + e_t, T[p]["pp"][t] + e_p).mean(axis=0) for t in range(fs.H)])
        EM = np.stack([rl.weights(T[p]["pt"][t] + e_t[mbin == pm[t]], T[p]["pp"][t] + e_p[mbin == pm[t]]).mean(axis=0)
                       if (mbin == pm[t]).sum() >= 20 else E0[t] for t in range(fs.H)])
        scores[p] = (np.where(T[p]["mand"], 0.0, E0), np.where(T[p]["mand"], 0.0, EM))
    rows = []
    for key, top_frac, b_bar in DESIGNS:
        P = fs.levels(top_frac); top = len(P) - 1

        def adaptive(which, tau, alpha):
            def pol(s):
                p, t = s["pos"], s["t"]
                sc = scores[p][which][t]
                cand = np.where(sc >= tau)[0]
                cand = cand[np.argsort(-sc[cand])]
                a_m = v.A_base[t] + v.delta[t][T[p]["mand"][t]].sum()
                cap = alpha * (s["Z"] + fs.T0 * fs.rate(P[top], s["h"]))
                n_add = int(np.searchsorted(a_m + np.cumsum(v.delta[t][cand]), cap, side="right"))
                ex = np.zeros(fs.NT, bool); ex[cand[:n_add]] = True
                return ex, top
            return pol

        ok = lambda m, mg=1.0: m["J_D"] <= ds.D_BAR * mg and m["J_B"] <= b_bar * mg
        for which, label in [(0, "rl_estimate"), (1, "rl_estimate_motion")]:
            vals = np.concatenate([scores[p][which][scores[p][which] > 0] for p in train])
            cands = []
            for tau in np.quantile(vals, [0.5, 0.75, 0.9, 0.95, 0.98]):
                for a in (0.5, 0.7, 0.85, 1.0):
                    pol = adaptive(which, tau, a)
                    m = fs.simulate(v, tr_eps, pol, P)
                    if ok(m, ds.MARGIN):
                        cands.append((m["J_cov"], pol))
            if cands:
                pol = max(cands, key=lambda c: c[0])[1]
                m = fs.simulate(v, he_eps, pol, P)
                rows.append(dict(video=v.name, design=key, method=label, held_feasible=ok(m), **m))
    return rows


def main():
    prepared = [ds.prepare(n) for n in ds.VIDEOS]
    with Pool(7) as pool:
        res = pool.map(run, prepared)
    new = pd.DataFrame([r for rows in res for r in rows])
    new.to_csv(OUT, index=False)
    old = pd.read_csv("VIDEO_SURVEY/design_options/design_options.csv")
    old = old[old.design.isin([d[0] for d in DESIGNS]) & old.method.isin(["trivial", "literature", "ceiling"])]
    allr = pd.concat([old[new.columns.intersection(old.columns)], new], ignore_index=True)
    for metric in ["psnr", "cov"]:
        p = allr.pivot_table(index=["design", "video"], columns="method", values=metric)
        p["room (RL est - lit)"] = p["rl_estimate"] - p["literature"]
        p["room with motion"] = p["rl_estimate_motion"] - p["literature"]
        print(f"===== {metric} =====")
        print(p.round(3).to_string())


if __name__ == "__main__":
    main()
