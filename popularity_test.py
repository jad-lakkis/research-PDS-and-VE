"""
popularity_test.py - can a learned policy beat the literature rules by
exploiting where OTHER viewers looked at the same moment of the video?

Popularity map pop[t, i] = average coverage weight of tile i in the actual
viewports of the TRAINING viewers at segment t. When a training viewer is
being simulated (for tuning), its own trace is left out of the map
(leave-one-out), so tuning never sees the viewer it is scored on. Held-out
viewers use the map built from all training viewers.

RL estimates rank extra tiles by a score and add them while the link can
deliver them (adaptive rule; NOT a paper baseline):
  own          expected coverage from the viewer's own predicted direction
  own+motion   same, error distribution conditioned on the previous head move
  +popularity  (1 - beta) * own(+motion) + beta * popularity, beta tuned
Literature: best of QER / average-based / ASL360 threshold, fine margin grid.
Everything tuned on training viewers (5% margin), tested on held-out viewers.
Power fixed at 50 mW, stall budget 0.3, tile budget 8 or 10.

Usage: .venv/Scripts/python.exe popularity_test.py
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config
import design_options_scan as ds
import fastsim as fs
import room_to_learn as rl

OUT = "VIDEO_SURVEY/design_options/popularity.csv"
LAMS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0)
BINS = [-np.inf, -30.0, -10.0, 10.0, 30.0, np.inf]


def run(args):
    v, train, held = args
    T = v.traj
    tr_eps = [(p, s) for p in train for s in ds.TRAIN_SEEDS]
    he_eps = [(p, s) for p in held for s in ds.HELD_SEEDS]
    P = fs.levels(0.5); top = len(P) - 1
    pm = {p: np.concatenate([[0.0], rl.wsigned(np.diff(T[p]["pt"]))]) for p in T}
    e_t = np.concatenate([rl.wsigned(T[p]["at"] - T[p]["pt"])[2:] for p in train])
    e_p = np.concatenate([(T[p]["ap"] - T[p]["pp"])[2:] for p in train])
    mv = np.digitize(np.concatenate([pm[p][2:] for p in train]), BINS)
    eth = np.concatenate([rl.wsigned(T[p]["at"] - T[p]["pt"])[1:] for p in train])
    eph = np.concatenate([(T[p]["ap"] - T[p]["pp"])[1:] for p in train])
    s_fit = (float(np.mean(np.abs(eth))), float(np.mean(np.abs(eph))))
    own, mot = {}, {}
    for p in T:
        b = np.digitize(pm[p], BINS)
        E0 = np.stack([rl.weights(T[p]["pt"][t] + eth, T[p]["pp"][t] + eph).mean(axis=0) for t in range(fs.H)])
        EM = np.stack([rl.weights(T[p]["pt"][t] + e_t[mv == b[t]], T[p]["pp"][t] + e_p[mv == b[t]]).mean(axis=0)
                       if (mv == b[t]).sum() >= 20 else E0[t] for t in range(fs.H)])
        own[p], mot[p] = E0, EM
    W_sum = sum(T[p]["w"] for p in train)
    pop = {p: ((W_sum - T[p]["w"]) / (len(train) - 1) if p in train else W_sum / len(train)) for p in T}

    def score(kind, beta):
        base = own if kind.startswith("own") and "motion" not in kind else mot
        return {p: np.where(T[p]["mand"], 0.0, (1 - beta) * base[p] + beta * pop[p]) for p in T}

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

    def rl_pol(sc_map, tau, alpha):
        def pol(s):
            p, t = s["pos"], s["t"]
            sc = sc_map[p][t]
            cand = np.where(sc >= tau)[0]
            cand = cand[np.argsort(-sc[cand])]
            a_m = v.A_base[t] + v.delta[t][T[p]["mand"][t]].sum()
            cap = alpha * (s["Z"] + fs.T0 * fs.rate(P[top], s["h"]))
            n_add = int(np.searchsorted(a_m + np.cumsum(v.delta[t][cand]), cap, side="right"))
            ex = np.zeros(fs.NT, bool); ex[cand[:n_add]] = True
            return ex, top
        return pol

    rows = []
    for b_bar in (8.0, 10.0):
        ok = lambda m, mg=1.0: m["J_D"] <= ds.D_BAR * mg and m["J_B"] <= b_bar * mg

        def best(cands):
            ev = [(n, pol, fs.simulate(v, tr_eps, pol, P)) for n, pol in cands]
            f = [e for e in ev if ok(e[2], ds.MARGIN)]
            if not f:
                return None
            n, pol, _ = max(f, key=lambda e: e[2]["J_cov"])
            return n, fs.simulate(v, he_eps, pol, P)

        def rl_cands(kind, betas):
            out = []
            for beta in betas:
                sc = score(kind, beta)
                vals = np.concatenate([sc[p][sc[p] > 0] for p in train])
                for q, tau in zip([50, 75, 90, 95, 98], np.quantile(vals, [0.5, 0.75, 0.9, 0.95, 0.98])):
                    for a in (0.5, 0.7, 0.85, 1.0):
                        out.append((f"beta {beta} q{q} a{a}", rl_pol(sc, tau, a)))
            return out
        tag = dict(video=v.name, B_bar=b_bar)
        m = fs.simulate(v, he_eps, lit_pol({p: np.zeros((fs.H, fs.NT), bool) for p in T}, False), P)
        rows.append(dict(**tag, method="trivial", choice="predicted viewport only", held_feasible=ok(m), **m))
        for method, cands in [("literature", [(n, lit_pol(mm, th)) for n, mm, th in lit]),
                              ("RL est: own", rl_cands("own", [0.0])),
                              ("RL est: own+motion", rl_cands("own+motion", [0.0])),
                              ("RL est: own+motion+popularity", rl_cands("own+motion", [0.25, 0.5, 0.75, 1.0]))]:
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
    pd.set_option("display.width", 250)
    for metric in ["psnr", "cov"]:
        p = df.pivot_table(index=["B_bar", "video"], columns="method", values=metric)
        best_rl = p[[c for c in p.columns if c.startswith("RL est")]].max(axis=1)
        p["room: popularity RL - lit"] = p["RL est: own+motion+popularity"] - p["literature"]
        p["room: best RL - lit"] = best_rl - p["literature"]
        print(f"===== {metric} =====")
        print(p.round(3).to_string())
        print(p.groupby(level="B_bar")[["room: popularity RL - lit", "room: best RL - lit"]].agg(["mean", "min", "max"]).round(3))
    print(df[df.method.str.contains("popularity")][["video", "B_bar", "choice", "held_feasible"]].to_string(index=False))


if __name__ == "__main__":
    main()
