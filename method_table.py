"""
method_table.py - one comparison table per video, all methods evaluated the
same way: held-out viewers (10 channel realisations each), stall budget 0.3,
tile budget 8, power 50 mW, no outages, simulated with fastsim (matches the
training environment).

Methods:
  predicted viewport only
  QER 120x60 / 160x80 / 200x100 (MMSP, sizes as published)
  average-based, MMSP bounds (enlargement 30, 15 deg as in MMSP)
  ASL360 threshold (literal: extra enhancement only once the buffer holds 10 s;
      our predicted-viewport tiles are always enhanced, so = predicted viewport only)
  literature, tuned: best of QER / average-based / threshold with the margin size
      tuned on training viewers to meet the budgets
  uniform, rate-matched: same total bits per segment as the RL estimate, spread
      over all 64 tiles at one quality level
  RL estimate (fixed observation): tiles relative to the predicted direction + recent
      head-motion direction; ranks extra tiles by expected coverage, adds them while the
      link can deliver (a rule standing in for what the agent could learn)
  RL estimate, excellent prediction: same, but knows the direction and rough size of the
      next head move (not available in practice; optimistic upper estimate for RL)
  oracle: knows the actual viewport; best feasible use of the tile budget

Usage: .venv/Scripts/python.exe method_table.py
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

import combined_test as ct
import config
import design_options_scan as ds
import fastsim as fs
import room_to_learn as rl

OUT = "VIDEO_SURVEY/method_table.csv"
B_BAR = 8.0
BINS = [-np.inf, -30.0, -10.0, 10.0, 30.0, np.inf]
LAMS = ct.LAMS


def run(args):
    v, train, held = args
    T = v.traj
    tr_eps = [(p, s) for p in train for s in ct.TRAIN_SEEDS]
    he_eps = [(p, s) for p in held for s in ct.HELD_SEEDS]
    P = fs.levels(0.5); top = len(P) - 1
    ok = lambda m, mg=1.0: m["J_D"] <= ds.D_BAR * mg and m["J_B"] <= B_BAR * mg
    pm = {p: np.concatenate([[0.0], rl.wsigned(np.diff(T[p]["pt"]))]) for p in T}
    err = {p: rl.wsigned(T[p]["at"] - T[p]["pt"]) for p in T}
    e_t = np.concatenate([err[p][2:] for p in train])
    e_p = np.concatenate([(T[p]["ap"] - T[p]["pp"])[2:] for p in train])
    prev_bin = np.digitize(np.concatenate([pm[p][2:] for p in train]), BINS)
    next_bin = np.digitize(e_t, BINS)
    eth = np.concatenate([err[p][1:] for p in train])
    eph = np.concatenate([(T[p]["ap"] - T[p]["pp"])[1:] for p in train])
    s_fit = (float(np.mean(np.abs(eth))), float(np.mean(np.abs(eph))))

    def scores(cond):
        out = {}
        for p in T:
            key = np.digitize(pm[p], BINS) if cond == "motion" else np.digitize(err[p], BINS)
            ref = prev_bin if cond == "motion" else next_bin
            rows = []
            for t in range(fs.H):
                sel = ref == key[t]
                if sel.sum() < 20:
                    sel = np.ones(len(e_t), bool)
                rows.append(rl.weights(T[p]["pt"][t] + e_t[sel], T[p]["pp"][t] + e_p[sel]).mean(axis=0))
            out[p] = np.where(T[p]["mand"], 0.0, np.stack(rows))
        return out
    S_mot, S_opt = scores("motion"), scores("next")
    A_m = {p: v.A_base + np.array([v.delta[t][T[p]["mand"][t]].sum() for t in range(fs.H)]) for p in T}

    def lit_pol(masks, th):
        return lambda s: ((masks[s["pos"]][s["t"]] if (not th or s["Z"] / v.mean_base_bits >= 10.0) else np.zeros(fs.NT, bool)), top)

    def qer_masks(w, h):
        out = {}
        for p in T:
            a = np.zeros((fs.H, fs.NT), bool)
            for t in range(fs.H):
                d = [np.hypot(rl.wsigned(T[p]["pt"][t] - c[0]), T[p]["pp"][t] - c[1]) for c in ds.QER_CENTRES]
                c = ds.QER_CENTRES[int(np.argmin(d))]
                a[t] = fs.region_masks(c[0], c[1], w / 2, h / 2)[0]
            out[p] = a
        return out

    def avg_masks(st, sp):
        return {p: fs.region_masks(T[p]["pt"], T[p]["pp"], config.V_THETA_DEG + st, config.V_PHI_DEG + sp) for p in T}

    def rl_pol(S, tau, alpha):
        def pol(s):
            p, t = s["pos"], s["t"]
            sc = S[p][t]
            cand = np.where(sc >= tau)[0]
            cand = cand[np.argsort(-sc[cand])]
            room = alpha * (s["Z"] + fs.T0 * fs.rate(P[top], s["h"]))
            n_add = int(np.searchsorted(A_m[p][t] + np.cumsum(v.delta[t][cand]), room, side="right"))
            ex = np.zeros(fs.NT, bool); ex[cand[:n_add]] = True
            return ex, top
        return pol

    def best(cands):
        ev = [(n, pol, fs.simulate(v, tr_eps, pol, P)) for n, pol in cands]
        f = [e for e in ev if ok(e[2], ds.MARGIN)]
        if not f:
            return None
        n, pol, mtr = max(f, key=lambda e: e[2]["J_cov"])
        return n, fs.simulate(v, he_eps, pol, P), mtr

    rows = []

    def add(name, kind, m, choice=""):
        rows.append(dict(video=v.name, method=name, kind=kind, choice=choice, within_budgets=ok(m) if "J_B" in m else m["J_D"] <= ds.D_BAR, **m))

    zero = {p: np.zeros((fs.H, fs.NT), bool) for p in T}
    add("predicted viewport only", "heuristic", fs.simulate(v, he_eps, lit_pol(zero, False), P))
    for w, h in [(120, 60), (160, 80), (200, 100)]:
        add(f"QER {w}x{h} (MMSP)", "heuristic", fs.simulate(v, he_eps, lit_pol(qer_masks(w, h), False), P))
    add("average-based, MMSP bounds (MMSP)", "heuristic", fs.simulate(v, he_eps, lit_pol(avg_masks(30.0, 15.0), False), P))
    add("threshold 10 s (ASL360, literal)", "heuristic", fs.simulate(v, he_eps, lit_pol(zero, False), P))
    lit = [(f"QER {w}x{h}", lit_pol(qer_masks(w, h), False)) for w, h in [(120, 60), (160, 80), (200, 100)]]
    for lam in LAMS:
        mm = avg_masks(lam * s_fit[0], lam * s_fit[1])
        lit += [(f"average-based {lam}x", lit_pol(mm, False)), (f"threshold, margin {lam}x", lit_pol(mm, True))]
    b = best(lit)
    if b:
        add("literature, tuned on training viewers", "heuristic", b[1], b[0])
    taus_m = np.quantile(np.concatenate([S_mot[p][S_mot[p] > 0] for p in train]), [0.5, 0.75, 0.9, 0.95, 0.98])
    br = best([(f"tau {t_:.4f} a {a}", rl_pol(S_mot, t_, a)) for t_ in taus_m for a in (0.5, 0.7, 0.85, 1.0)])
    if br:
        add("RL estimate, fixed observation", "RL estimate", br[1], br[0])
        target = br[2]["bits"]
        unif = None
        for k in range(6, 0, -1):
            mu = uniform_eval(v, tr_eps[:6], k, P)
            if mu["bits"] <= target:
                unif = k
                break
        if unif:
            mu = uniform_eval(v, he_eps, unif, P)
            rows.append(dict(video=v.name, method="uniform, rate-matched (all 64 tiles)", kind="heuristic",
                             choice=f"level {unif}", within_budgets=None, psnr=mu["psnr"], J_D=mu["J_D"], bits=mu["bits"],
                             cov=1.0, J_B=float(np.sum(fs.DISC)), extra=fs.NT))
    taus_o = np.quantile(np.concatenate([S_opt[p][S_opt[p] > 0] for p in train]), [0.5, 0.75, 0.9, 0.95, 0.98])
    bo = best([(f"tau {t_:.4f} a {a}", rl_pol(S_opt, t_, a)) for t_ in taus_o for a in (0.5, 0.7, 0.85, 1.0)])
    if bo:
        add("RL estimate, excellent prediction", "RL estimate (optimistic)", bo[1], bo[0])
    # oracle: perfect viewport knowledge, pooled tile budget, shrunk to stay feasible
    items, base = [], 0.0
    for p in held:
        base += np.sum(fs.DISC * T[p]["mand"].sum(axis=1)) / fs.NT
        items += [(T[p]["w"][t, i], p, t, i) for t in range(fs.H) for i in np.where(~T[p]["mand"][t] & (T[p]["w"][t] > 0))[0]]
    items.sort(key=lambda x: -x[0])

    def oracle(frac):
        budget = frac * (len(held) * B_BAR - base)
        masks = {p: np.zeros((fs.H, fs.NT), bool) for p in held}
        for _, p, t, i in items:
            c_ = fs.DISC[t] / fs.NT
            if budget < c_:
                break
            masks[p][t, i] = True; budget -= c_
        return fs.simulate(v, he_eps, lit_pol(masks, False), P)
    lo, hi, bm = 0.0, 1.0, None
    for _ in range(12):
        mid = (lo + hi) / 2
        m = oracle(mid)
        if ok(m):
            lo, bm = mid, m
        else:
            hi = mid
    add("oracle (perfect viewport knowledge)", "upper bound", bm, f"{lo:.0%} of extra tile budget")
    return rows


def uniform_eval(v, eps, k, P):
    top = len(P) - 1
    out = []
    for pos, seed in eps:
        tr, hs = v.traj[pos], v.h[(pos, seed)]
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


def main():
    prepared = [ct.prepare(n) for n in ds.VIDEOS]
    with Pool(7) as pool:
        res = pool.map(run, prepared)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 45); pd.set_option("display.max_rows", 200)
    print(df[["video", "method", "choice", "within_budgets", "psnr", "cov", "J_D", "J_B"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
