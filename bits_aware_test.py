"""
bits_aware_test.py - when the link (bits) is the bottleneck, does choosing
tiles by value PER BIT create room for RL over the literature rules?

Per video, the transmit power is fixed at the lowest level (from 50, 25, 12.5,
6.25, 3.1 mW) at which the predicted viewport alone still uses at most half
of the stall budget on the training viewers - so the link, not the tile
count, is what limits extra tiles. Stall budget 0.3; tile budget 8 or 12.
Objectives: coverage or viewport PSNR.
Methods (tuned on training viewers, tested on held-out viewers):
  trivial       predicted viewport only
  literature    best of QER / average-based / ASL360 threshold (fine margin grid)
  RL est (tile) ranks extra tiles by expected value
  RL est (bit)  ranks extra tiles by expected value per enhancement bit
Expected value = expected coverage (coverage objective) or expected coverage
x the tile's MSE gain (PSNR objective). Both RL estimates add tiles while the
current channel and buffer can deliver them.

Usage: .venv/Scripts/python.exe bits_aware_test.py
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config
import design_options_scan as ds
import fastsim as fs
import room_to_learn as rl

OUT = "VIDEO_SURVEY/design_options/bits_aware.csv"
FRACS = [0.5, 0.25, 0.125, 0.0625, 0.03125]
LAMS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0)


def run(args):
    v, train, held = args
    T = v.traj
    tr_eps = [(p, s) for p in train for s in ds.TRAIN_SEEDS]
    he_eps = [(p, s) for p in held for s in ds.HELD_SEEDS]
    eth = np.concatenate([rl.wsigned(T[p]["at"] - T[p]["pt"])[1:] for p in train])
    eph = np.concatenate([(T[p]["ap"] - T[p]["pp"])[1:] for p in train])
    s_fit = (float(np.mean(np.abs(eth))), float(np.mean(np.abs(eph))))
    Ecov = {p: np.where(T[p]["mand"], 0.0, np.stack([rl.weights(T[p]["pt"][t] + eth, T[p]["pp"][t] + eph).mean(axis=0)
                                                     for t in range(fs.H)])) for p in T}
    zero = {p: np.zeros((fs.H, fs.NT), bool) for p in T}
    # power level at which the link becomes the bottleneck
    frac = FRACS[0]
    for f in FRACS:
        P = fs.levels(f)
        m = fs.simulate(v, tr_eps, lambda s, P=P: (np.zeros(fs.NT, bool), len(P) - 1), P)
        if m["J_D"] <= 0.5 * ds.D_BAR:
            frac = f
        else:
            break
    P = fs.levels(frac); top = len(P) - 1

    lit = []
    for w, h in [(120, 60), (160, 80), (200, 100)]:
        mm = {}
        for p in T:
            a = np.zeros((fs.H, fs.NT), bool)
            for t in range(fs.H):
                d = [np.hypot(rl.wsigned(T[p]["pt"][t] - c[0]), T[p]["pp"][t] - c[1]) for c in ds.QER_CENTRES]
                c = ds.QER_CENTRES[int(np.argmin(d))]
                a[t] = fs.region_masks(c[0], c[1], w / 2, h / 2)[0]
            mm[p] = a
        lit.append((f"QER {w}x{h}", mm, False))
    for lam in LAMS:
        mm = {p: fs.region_masks(T[p]["pt"], T[p]["pp"], config.V_THETA_DEG + lam * s_fit[0], config.V_PHI_DEG + lam * s_fit[1]) for p in T}
        lit += [(f"average-based {lam}x", mm, False), (f"threshold, margin {lam}x", mm, True)]

    def lit_pol(masks, th):
        return lambda s: ((masks[s["pos"]][s["t"]] if (not th or s["Z"] / v.mean_base_bits >= 10.0) else np.zeros(fs.NT, bool)), top)

    def rl_pol(score, tau, alpha):
        def pol(s):
            p, t = s["pos"], s["t"]
            sc = score[p][t]
            cand = np.where(sc >= tau)[0]
            cand = cand[np.argsort(-sc[cand])]
            a_m = v.A_base[t] + v.delta[t][T[p]["mand"][t]].sum()
            cap = alpha * (s["Z"] + fs.T0 * fs.rate(P[top], s["h"]))
            n_add = int(np.searchsorted(a_m + np.cumsum(v.delta[t][cand]), cap, side="right"))
            ex = np.zeros(fs.NT, bool); ex[cand[:n_add]] = True
            return ex, top
        return pol

    rows = []
    for obj in ["cov", "psnr"]:
        val = Ecov if obj == "cov" else {p: Ecov[p] * v.gain for p in T}
        per_bit = {p: val[p] / np.maximum(v.delta, 1.0) for p in T}
        objk = "J_cov" if obj == "cov" else "J_psnr"
        for b_bar in (8.0, 12.0):
            ok = lambda m, mg=1.0: m["J_D"] <= ds.D_BAR * mg and m["J_B"] <= b_bar * mg

            def best(cands):
                ev = [(n, pol, fs.simulate(v, tr_eps, pol, P)) for n, pol in cands]
                f = [e for e in ev if ok(e[2], ds.MARGIN)]
                if not f:
                    return None
                n, pol, _ = max(f, key=lambda e: e[2][objk])
                return n, fs.simulate(v, he_eps, pol, P)
            tag = dict(video=v.name, top_power_mW=P[top] * 1e3, objective=obj, B_bar=b_bar)
            m = fs.simulate(v, he_eps, lit_pol(zero, False), P)
            rows.append(dict(**tag, method="trivial", choice="predicted viewport only", held_feasible=ok(m), **m))
            for method, cands in [("literature", [(n, lit_pol(mm, th)) for n, mm, th in lit]),
                                  ("RL est (tile)", [(f"q{q} a{a}", rl_pol(val, t_, a)) for q, t_ in
                                                     zip([50, 75, 90, 95, 98], np.quantile(np.concatenate([val[p][val[p] > 0] for p in train]), [0.5, 0.75, 0.9, 0.95, 0.98]))
                                                     for a in (0.5, 0.7, 0.85, 1.0)]),
                                  ("RL est (bit)", [(f"q{q} a{a}", rl_pol(per_bit, t_, a)) for q, t_ in
                                                    zip([50, 75, 90, 95, 98], np.quantile(np.concatenate([per_bit[p][per_bit[p] > 0] for p in train]), [0.5, 0.75, 0.9, 0.95, 0.98]))
                                                    for a in (0.5, 0.7, 0.85, 1.0)])]:
                b = best(cands)
                if b:
                    rows.append(dict(**tag, method=method, choice=b[0], held_feasible=ok(b[1]), **b[1]))
    return rows


def main():
    prepared = [ds.prepare(n) for n in ds.VIDEOS]
    with Pool(7) as pool:
        res = pool.map(run, prepared)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(OUT, index=False)
    p = df.pivot_table(index=["objective", "B_bar", "video", "top_power_mW"], columns="method", values="psnr")
    p["best RL est"] = p[["RL est (tile)", "RL est (bit)"]].max(axis=1)
    p["room (best RL est - lit), dB"] = p["best RL est"] - p["literature"]
    p["per-bit gain, dB"] = p["RL est (bit)"] - p["RL est (tile)"]
    pd.set_option("display.width", 250)
    print(p.round(3).to_string())
    print(p.groupby(level=["objective", "B_bar"])[["room (best RL est - lit), dB", "per-bit gain, dB"]].agg(["mean", "min", "max"]).round(3))


if __name__ == "__main__":
    main()
