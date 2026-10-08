"""
combined_test.py - all allowed changes together (problem formulation fixed):
  agent information: tiles relative to the predicted direction + recent
                     head-motion direction (+ buffer and channel, as now)
  channel:           no outages, or link blockages (10 / 12.5 dB, ~20% of the
                     time, ~3 s each)
  power fixed at 50 mW; stall budget 0.3; tile budget 8 or 10.
Methods (tuned on training viewers, tested on held-out viewers, 10 channel
realisations per viewer):
  trivial            predicted viewport only
  literature         best of QER / average-based / ASL360 threshold (fine margin grid)
  RL estimate        motion-aware expected-coverage ranking + buffer reserve (causal rule,
                     NOT a paper baseline - estimates what a learned policy can reach)
  uniform (rate-matched)  spends the same total bits per segment as the RL estimate,
                     spread evenly over all 64 tiles at one quality level (the highest
                     level whose rate does not exceed the RL estimate's); compared on
                     PSNR and stall only (coverage is 1 and it uses all 64 tiles by design)

Usage: .venv/Scripts/python.exe combined_test.py
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config
import design_options_scan as ds
import fastsim as fs
import room_to_learn as rl
from streaming_rl import data_loader

OUT = "VIDEO_SURVEY/design_options/combined.csv"
TRAIN_SEEDS = tuple(range(2000, 2010))
HELD_SEEDS = tuple(range(1000, 1010))
LAMS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0)
BINS = [-np.inf, -30.0, -10.0, 10.0, 30.0, np.inf]
P_BG = 1.0 / 3.0
P_GB = P_BG * 0.2 / 0.8


def prepare(name):
    ds.TRAIN_SEEDS, ds.HELD_SEEDS = TRAIN_SEEDS, HELD_SEEDS
    v, train, held = ds.prepare(name)
    rd = data_loader.load_rd_data()
    br, ym = rd["video_bitrate_data"][v.vi], rd["video_ymse_data"][v.vi]
    f = np.arange(fs.H) * config.GOP_SIZE_FRAMES
    v.lv_bits = {k: br[config.qp_array_index_for_enhancement_level(k)][:, f].T for k in range(0, 7)}
    v.lv_mse = {k: ym[config.qp_array_index_for_enhancement_level(k)][:, f].T for k in range(0, 7)}
    return v, train, held


def run(args):
    v, train, held = args
    T = v.traj
    tr_eps = [(p, s) for p in train for s in TRAIN_SEEDS]
    he_eps = [(p, s) for p in held for s in HELD_SEEDS]
    P = fs.levels(0.5); top = len(P) - 1
    base_h = {k: h.copy() for k, h in v.h.items()}
    draws = {k: np.random.default_rng(abs(hash(("blk",) + k)) % 2**32).random(fs.H) for k in base_h}
    pm = {p: np.concatenate([[0.0], rl.wsigned(np.diff(T[p]["pt"]))]) for p in T}
    e_t = np.concatenate([rl.wsigned(T[p]["at"] - T[p]["pt"])[2:] for p in train])
    e_p = np.concatenate([(T[p]["ap"] - T[p]["pp"])[2:] for p in train])
    mv = np.digitize(np.concatenate([pm[p][2:] for p in train]), BINS)
    eth = np.concatenate([rl.wsigned(T[p]["at"] - T[p]["pt"])[1:] for p in train])
    eph = np.concatenate([(T[p]["ap"] - T[p]["pp"])[1:] for p in train])
    s_fit = (float(np.mean(np.abs(eth))), float(np.mean(np.abs(eph))))
    EM = {}
    for p in T:
        b = np.digitize(pm[p], BINS)
        rows_ = []
        for t in range(fs.H):
            sel = mv == b[t]
            if sel.sum() < 20:
                sel = np.ones(len(e_t), bool)
            rows_.append(rl.weights(T[p]["pt"][t] + e_t[sel], T[p]["pp"][t] + e_p[sel]).mean(axis=0))
        EM[p] = np.where(T[p]["mand"], 0.0, np.stack(rows_))
    A_m = {p: v.A_base + np.array([v.delta[t][T[p]["mand"][t]].sum() for t in range(fs.H)]) for p in T}
    taus = np.quantile(np.concatenate([EM[p][EM[p] > 0] for p in train]), [0.5, 0.75, 0.9, 0.95, 0.98])

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

    def rl_pol(tau, alpha, reserve):
        def pol(s):
            p, t = s["pos"], s["t"]
            sc = EM[p][t]
            cand = np.where(sc >= tau)[0]
            cand = cand[np.argsort(-sc[cand])]
            room = alpha * (s["Z"] + fs.T0 * fs.rate(P[top], s["h"])) - reserve * A_m[p][t]
            n_add = int(np.searchsorted(A_m[p][t] + np.cumsum(v.delta[t][cand]), room, side="right"))
            ex = np.zeros(fs.NT, bool); ex[cand[:n_add]] = True
            return ex, top
        return pol

    def uniform(eps, k):
        out = []
        for pos, seed in eps:
            tr, hs = T[pos], v.h[(pos, seed)]
            Z, acc = 0.0, dict(psnr=0.0, J_D=0.0, bits=0.0)
            for t in range(fs.H):
                A = v.lv_bits[k][t].sum()
                R = fs.rate(P[top], hs[t])
                D = 0.0 if Z + fs.T0 * R >= A else fs.T0 * (1.0 - (Z + fs.T0 * R) / A)
                Z = max(Z + fs.T0 * R - A, 0.0)
                w = tr["w"][t]
                acc["psnr"] += 10 * np.log10(255.0 ** 2 / (np.sum(w * v.lv_mse[k][t]) / np.sum(w))) / fs.H
                acc["J_D"] += fs.DISC[t] * D / fs.T0
                acc["bits"] += A / fs.H
            out.append(acc)
        return {kk: float(np.mean([o[kk] for o in out])) for kk in out[0]}

    rows = []
    for att in (0.0, 10.0, 12.5):
        for k, h in base_h.items():
            u = draws[k]
            blocked = np.zeros(fs.H, bool)
            blocked[0] = att > 0 and u[0] < 0.2
            for t in range(1, fs.H):
                blocked[t] = att > 0 and ((u[t] >= P_BG) if blocked[t - 1] else (u[t] < P_GB))
            v.h[k] = h * np.where(blocked, 10 ** (-att / 10), 1.0)
        for b_bar in (8.0, 10.0):
            ok = lambda m, mg=1.0: m["J_D"] <= ds.D_BAR * mg and m["J_B"] <= b_bar * mg

            def best(cands):
                ev = [(n, pol, fs.simulate(v, tr_eps, pol, P)) for n, pol in cands]
                f = [e for e in ev if ok(e[2], ds.MARGIN)]
                if not f:
                    return None
                n, pol, mtr = max(f, key=lambda e: e[2]["J_cov"])
                return n, fs.simulate(v, he_eps, pol, P), mtr
            tag = dict(video=v.name, outage_dB=att, B_bar=b_bar)
            m = fs.simulate(v, he_eps, lit_pol({p: np.zeros((fs.H, fs.NT), bool) for p in T}, False), P)
            rows.append(dict(**tag, method="trivial", choice="predicted viewport only", held_feasible=ok(m), **m))
            bl = best([(n, lit_pol(mm, th)) for n, mm, th in lit])
            if bl:
                rows.append(dict(**tag, method="literature", choice=bl[0], held_feasible=ok(bl[1]), **bl[1]))
            br_ = best([(f"tau {tau:.4f} a {a} reserve {r}", rl_pol(tau, a, r))
                        for tau in taus for a in (0.7, 1.0) for r in (0.0, 0.25, 0.5, 1.0)])
            if br_:
                rows.append(dict(**tag, method="RL estimate", choice=br_[0], held_feasible=ok(br_[1]), **br_[1]))
                target = br_[2]["bits"]
                ks = [k for k in range(1, 7) if uniform(tr_eps[:6], k)["bits"] <= target]
                if ks:
                    k = max(ks)
                    mu = uniform(he_eps, k)
                    rows.append(dict(**tag, method="uniform (rate-matched)", choice=f"all 64 tiles at level {k}",
                                     held_feasible=mu["J_D"] <= ds.D_BAR, psnr=mu["psnr"], J_D=mu["J_D"], bits=mu["bits"]))
    for k, h in base_h.items():
        v.h[k] = h
    return rows


def main():
    prepared = [prepare(n) for n in ds.VIDEOS]
    with Pool(7) as pool:
        res = pool.map(run, prepared)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    p = df.pivot_table(index=["outage_dB", "B_bar", "video"], columns="method", values="psnr")
    f = df.pivot_table(index=["outage_dB", "B_bar", "video"], columns="method", values="held_feasible", aggfunc="first")
    p["RL - literature"] = p["RL estimate"] - p["literature"]
    p["RL - uniform"] = p["RL estimate"] - p["uniform (rate-matched)"]
    print(p.round(2).to_string())
    print(f.to_string())
    print(p.groupby(level=["outage_dB", "B_bar"])[["RL - literature", "RL - uniform"]].agg(["mean", "min", "max"]).round(2))


if __name__ == "__main__":
    main()
