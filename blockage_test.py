"""
blockage_test.py - do realistic link blockages (capacity collapsing for a few
seconds, as in mmWave / UAV traces) create a fair margin for RL?

Channel: the environment's Rician gain times a two-state Markov blockage
process (clear / blocked). Blocked periods attenuate the gain by ATT dB and
last ~3 s on average; ~20% of the time is blocked. The agent sees the current
channel gain, so it can tell when it is blocked (the state persists).
Design: coverage objective, stall budget 0.3, tile budget 8, power fixed at 50 mW.
Methods (tuned on training viewers, tested on held-out viewers):
  trivial      predicted viewport only
  literature   best of QER / average-based / ASL360 threshold (fine margin grid)
  reactive     RL estimate: extra tiles while the CURRENT channel + buffer can deliver
  planning     RL estimate: as reactive, but only adds tiles while keeping a buffer
               reserve of r segments (r tuned) - what a policy that learns to
               prepare for blockages could do (causal)
  foresight    knows the next L seconds of channel (not causal; upper reference)

Usage: .venv/Scripts/python.exe blockage_test.py
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config
import design_options_scan as ds
import fastsim as fs
import room_to_learn as rl

OUT = "VIDEO_SURVEY/design_options/blockage.csv"
LAMS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0)
P_BG = 1.0 / 3.0                     # mean blocked duration 3 s
P_GB = P_BG * 0.2 / 0.8              # ~20% of time blocked
ATTS = (15.0, 20.0)
B_BAR = 8.0


def run(args):
    v, train, held = args
    T = v.traj
    tr_eps = [(p, s) for p in train for s in ds.TRAIN_SEEDS]
    he_eps = [(p, s) for p in held for s in ds.HELD_SEEDS]
    P = fs.levels(0.5); top = len(P) - 1
    base_h = {k: h.copy() for k, h in v.h.items()}
    draws = {k: np.random.default_rng(abs(hash(("blk",) + k)) % 2**32).random(fs.H) for k in base_h}
    eth = np.concatenate([rl.wsigned(T[p]["at"] - T[p]["pt"])[1:] for p in train])
    eph = np.concatenate([(T[p]["ap"] - T[p]["pp"])[1:] for p in train])
    s_fit = (float(np.mean(np.abs(eth))), float(np.mean(np.abs(eph))))
    E = {p: np.where(T[p]["mand"], 0.0, np.stack([rl.weights(T[p]["pt"][t] + eth, T[p]["pp"][t] + eph).mean(axis=0)
                                                  for t in range(fs.H)])) for p in T}
    A_m = {p: v.A_base + np.array([v.delta[t][T[p]["mand"][t]].sum() for t in range(fs.H)]) for p in T}
    taus = np.quantile(np.concatenate([E[p][E[p] > 0] for p in train]), [0.5, 0.75, 0.9, 0.95, 0.98])
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

    def tiles_pol(tau, alpha, reserve, L):
        def pol(s):
            p, t, seed = s["pos"], s["t"], s["seed"]
            hs = v.h[(p, seed)]
            sc = E[p][t]
            cand = np.where(sc >= tau)[0]
            cand = cand[np.argsort(-sc[cand])]
            r_now = fs.T0 * fs.rate(P[top], hs[t])
            need = reserve * A_m[p][t]
            if L > 1 and t + 1 < fs.H:
                k = np.arange(t + 1, min(t + L, fs.H))
                need = max(need, np.max(np.cumsum(A_m[p][k] - fs.T0 * fs.rate(P[top], hs[k]))))
            room = alpha * (s["Z"] + r_now) - need
            n_add = int(np.searchsorted(A_m[p][t] + np.cumsum(v.delta[t][cand]), room, side="right"))
            ex = np.zeros(fs.NT, bool); ex[cand[:n_add]] = True
            return ex, top
        return pol

    rows = []
    for att in ATTS:
        for k, h in base_h.items():
            u = draws[k]
            blocked = np.zeros(fs.H, bool)
            blocked[0] = u[0] < 0.2
            for t in range(1, fs.H):
                blocked[t] = (u[t] >= P_BG) if blocked[t - 1] else (u[t] < P_GB)
            v.h[k] = h * np.where(blocked, 10 ** (-att / 10), 1.0)
        ok = lambda m, mg=1.0: m["J_D"] <= ds.D_BAR * mg and m["J_B"] <= B_BAR * mg

        def best(cands):
            ev = [(n, pol, fs.simulate(v, tr_eps, pol, P)) for n, pol in cands]
            f = [e for e in ev if ok(e[2], ds.MARGIN)]
            if not f:
                return None
            n, pol, _ = max(f, key=lambda e: e[2]["J_cov"])
            return n, fs.simulate(v, he_eps, pol, P)
        tag = dict(video=v.name, blockage_dB=att)
        m = fs.simulate(v, he_eps, lit_pol({p: np.zeros((fs.H, fs.NT), bool) for p in T}, False), P)
        rows.append(dict(**tag, method="trivial", choice="predicted viewport only", held_feasible=ok(m), **m))
        for method, cands in [("literature", [(n, lit_pol(mm, th)) for n, mm, th in lit]),
                              ("reactive (RL estimate)", [(f"tau {tau:.4f} a {a}", tiles_pol(tau, a, 0.0, 1)) for tau in taus for a in (0.5, 0.7, 0.85, 1.0)]),
                              ("planning (RL estimate)", [(f"tau {tau:.4f} a {a} reserve {r}", tiles_pol(tau, a, r, 1))
                                                          for tau in taus for a in (0.7, 1.0) for r in (0.25, 0.5, 1.0, 1.5)]),
                              ("foresight (not causal)", [(f"tau {tau:.4f} L {L}", tiles_pol(tau, 1.0, 0.0, L)) for tau in taus for L in (3, 5)])]:
            b = best(cands)
            if b:
                rows.append(dict(**tag, method=method, choice=b[0], held_feasible=ok(b[1]), **b[1]))
    for k, h in base_h.items():
        v.h[k] = h
    return rows


def main():
    prepared = [ds.prepare(n) for n in ds.VIDEOS]
    with Pool(7) as pool:
        res = pool.map(run, prepared)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    for metric in ["psnr", "cov", "J_D"]:
        p = df.pivot_table(index=["blockage_dB", "video"], columns="method", values=metric)
        if metric != "J_D":
            best_rl = p[["reactive (RL estimate)", "planning (RL estimate)"]].max(axis=1)
            p["room: best causal RL est - lit"] = best_rl - p["literature"]
            p["foresight - best causal"] = p["foresight (not causal)"] - best_rl
        print(f"===== {metric} =====")
        print(p.round(3).to_string())
    print(df[["video", "blockage_dB", "method", "choice", "held_feasible"]].to_string(index=False))


if __name__ == "__main__":
    main()
