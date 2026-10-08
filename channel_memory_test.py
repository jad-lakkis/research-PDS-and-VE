"""
channel_memory_test.py - does a channel with memory (slow shadowing: good and
bad periods lasting several seconds) create room for RL?

Channel: the environment's Rician gain h_t times slow log-normal shadowing
10^(s_t/10), s_t = rho s_{t-1} + sqrt(1 - rho^2) sigma eps_t. rho = 0 is
memoryless, rho = 0.95 has memory; both have the same marginal distribution
and use the same random draws, so only the memory differs.
Design: coverage objective, stall budget 0.3, tile budget 8, power fixed at 50 mW.
Methods, tuned on training viewers and tested on held-out viewers:
  trivial      predicted viewport only
  literature   best of QER / average-based / ASL360 threshold (fine margin grid)
  adaptive     RL estimate: extra tiles ranked by expected coverage, added while
               the CURRENT channel and buffer can deliver them
  foresight    same, but knows the next L seconds of channel (not causal: it is
               the most a policy could gain by predicting the channel)
With memory, a learned policy can partly predict the channel from its current
value, so foresight - adaptive becomes room that RL can learn to capture.

Usage: .venv/Scripts/python.exe channel_memory_test.py
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config
import design_options_scan as ds
import fastsim as fs
import room_to_learn as rl

OUT = "VIDEO_SURVEY/design_options/channel_memory.csv"
VIDEOS = ["Runner", "GateNight", "Bridge", "SiyuanGate"]
CONFIGS = [(sig, rho) for sig in (4.0, 6.0) for rho in (0.0, 0.95)]
B_BAR = 8.0
LAMS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0)


def run(args):
    v, train, held = args
    T = v.traj
    tr_eps = [(p, s) for p in train for s in ds.TRAIN_SEEDS]
    he_eps = [(p, s) for p in held for s in ds.HELD_SEEDS]
    base_h = {k: h.copy() for k, h in v.h.items()}
    eps_draw = {k: np.random.default_rng(abs(hash(k)) % 2**32).standard_normal(fs.H) for k in base_h}
    eth = np.concatenate([rl.wsigned(T[p]["at"] - T[p]["pt"])[1:] for p in train])
    eph = np.concatenate([(T[p]["ap"] - T[p]["pp"])[1:] for p in train])
    s_fit = (float(np.mean(np.abs(eth))), float(np.mean(np.abs(eph))))
    E = {p: np.where(T[p]["mand"], 0.0, np.stack([rl.weights(T[p]["pt"][t] + eth, T[p]["pp"][t] + eph).mean(axis=0)
                                                  for t in range(fs.H)])) for p in T}
    A_m = {p: v.A_base + np.array([v.delta[t][T[p]["mand"][t]].sum() for t in range(fs.H)]) for p in T}
    P = fs.levels(0.5); top = len(P) - 1
    taus = np.quantile(np.concatenate([E[p][E[p] > 0] for p in train]), [0.5, 0.75, 0.9, 0.95, 0.98])

    lit = []
    for w, h in [(120, 60), (160, 80), (200, 100)]:
        m = {}
        for p in T:
            mm = np.zeros((fs.H, fs.NT), bool)
            for t in range(fs.H):
                d = [np.hypot(rl.wsigned(T[p]["pt"][t] - c[0]), T[p]["pp"][t] - c[1]) for c in ds.QER_CENTRES]
                c = ds.QER_CENTRES[int(np.argmin(d))]
                mm[t] = fs.region_masks(c[0], c[1], w / 2, h / 2)[0]
            m[p] = mm
        lit.append((f"QER {w}x{h}", m, False))
    for lam in LAMS:
        m = {p: fs.region_masks(T[p]["pt"], T[p]["pp"], config.V_THETA_DEG + lam * s_fit[0], config.V_PHI_DEG + lam * s_fit[1]) for p in T}
        lit += [(f"average-based {lam}x", m, False), (f"threshold, margin {lam}x", m, True)]

    def lit_pol(masks, th):
        return lambda s: ((masks[s["pos"]][s["t"]] if (not th or s["Z"] / v.mean_base_bits >= 10.0) else np.zeros(fs.NT, bool)), top)

    def tiles_pol(tau, alpha, L):
        def pol(s):
            p, t, seed = s["pos"], s["t"], s["seed"]
            hs = v.h[(p, seed)]
            sc = E[p][t]
            cand = np.where(sc >= tau)[0]
            cand = cand[np.argsort(-sc[cand])]
            r_now = fs.T0 * fs.rate(P[top], hs[t])
            need = 0.0
            if L > 1 and t + 1 < fs.H:
                k = np.arange(t + 1, min(t + L, fs.H))
                need = max(0.0, np.max(np.cumsum(A_m[p][k] - fs.T0 * fs.rate(P[top], hs[k]))))
            room = alpha * (s["Z"] + r_now) - need
            n_add = int(np.searchsorted(A_m[p][t] + np.cumsum(v.delta[t][cand]), room, side="right"))
            ex = np.zeros(fs.NT, bool); ex[cand[:n_add]] = True
            return ex, top
        return pol

    rows = []
    for sig, rho in CONFIGS:
        for k, h in base_h.items():
            e = eps_draw[k]
            s = np.zeros(fs.H); s[0] = sig * e[0]
            for t in range(1, fs.H):
                s[t] = rho * s[t - 1] + np.sqrt(1 - rho ** 2) * sig * e[t]
            v.h[k] = h * 10 ** (s / 10)
        ok = lambda m, mg=1.0: m["J_D"] <= ds.D_BAR * mg and m["J_B"] <= B_BAR * mg

        def best(cands):
            ev = [(n, pol, fs.simulate(v, tr_eps, pol, P)) for n, pol in cands]
            f = [e for e in ev if ok(e[2], ds.MARGIN)]
            if not f:
                return None
            n, pol, _ = max(f, key=lambda e: e[2]["J_cov"])
            return n, fs.simulate(v, he_eps, pol, P)
        tag = dict(video=v.name, shadowing_sigma_dB=sig, memory_rho=rho)
        m = fs.simulate(v, he_eps, lit_pol({p: np.zeros((fs.H, fs.NT), bool) for p in T}, False), P)
        rows.append(dict(**tag, method="trivial", choice="predicted viewport only", held_feasible=ok(m), **m))
        for method, cands in [("literature", [(n, lit_pol(mm, th)) for n, mm, th in lit]),
                              ("adaptive (RL estimate)", [(f"tau {tau:.4f} alpha {a}", tiles_pol(tau, a, 1)) for tau in taus for a in (0.5, 0.7, 0.85, 1.0)]),
                              ("foresight (knows future channel)", [(f"tau {tau:.4f} L {L}", tiles_pol(tau, 1.0, L)) for tau in taus for L in (3, 5)])]:
            b = best(cands)
            if b:
                rows.append(dict(**tag, method=method, choice=b[0], held_feasible=ok(b[1]), **b[1]))
    return rows


def main():
    prepared = [ds.prepare(n) for n in VIDEOS]
    with Pool(len(VIDEOS)) as pool:
        res = pool.map(run, prepared)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(OUT, index=False)
    for metric in ["psnr", "cov"]:
        p = df.pivot_table(index=["shadowing_sigma_dB", "memory_rho", "video"], columns="method", values=metric)
        p["RL est - literature"] = p["adaptive (RL estimate)"] - p["literature"]
        p["foresight - RL est"] = p["foresight (knows future channel)"] - p["adaptive (RL estimate)"]
        print(f"===== {metric} =====")
        print(p.round(3).to_string())
    print(df[["video", "shadowing_sigma_dB", "memory_rho", "method", "held_feasible", "J_D"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
