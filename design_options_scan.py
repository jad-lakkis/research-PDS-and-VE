"""
design_options_scan.py - which problem design leaves room for RL?

For each video and design: trivial policy (predicted viewport only), the best
literature rule (QER, average-based, ASL360 threshold; tuned on training
viewers to meet the budgets), an RL estimate (adaptive rule: ranks extra
tiles by expected value from the predicted direction and adds them while
the link can deliver; NOT a paper baseline - only an estimate of what a
learned policy could reach), and the perfect-prediction ceiling.
All evaluated on held-out viewers with fastsim (validated against the env).

Designs (stall budget 0.3 throughout):
  D0 current        objective coverage, power free at 50 mW, tile budget 8
  D1 tight tiles    as D0, tile budget 7.5
  D2 weak link      as D0, power fixed at 25 mW
  D3 power budget   as D0, power is a decision (0-50 mW) with a binding power budget
                    (1.25 x the power the trivial policy needs)
  D4 PSNR           objective = viewport PSNR, otherwise D0
  D5 PSNR+25mW      objective PSNR, power fixed at 25 mW
  D6 PSNR+budget    objective PSNR, power budget as D3
For PSNR designs the RL estimate ranks tiles by expected coverage x the tile's
MSE gain (needs per-tile content information in the agent's observation);
"RL estimate, no content" ranks by expected coverage only.

Usage: .venv/Scripts/python.exe design_options_scan.py
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config
import fastsim as fs
import room_to_learn as rl
from streaming_rl import channel_model, data_loader

OUT = "VIDEO_SURVEY/design_options"
VIDEOS = ["Runner", "GateNight", "Bridge", "SiyuanGate", "Academic", "SouthGate", "StudyRoom"]
D_BAR, MARGIN = 0.3, 0.95
TRAIN_SEEDS, HELD_SEEDS = (2000, 2001, 2002), (1000, 1001, 1002)
DESIGNS = [  # key, label, objective, top power fraction, power mode, tile budget
    ("D0", "current (coverage, 50 mW, tiles 8)", "cov", 0.5, "free", 8.0),
    ("D1", "tile budget 7.5", "cov", 0.5, "free", 7.5),
    ("D2", "fixed 25 mW", "cov", 0.25, "free", 8.0),
    ("D3", "power budget", "cov", 0.5, "budget", 8.0),
    ("D4", "PSNR objective", "psnr", 0.5, "free", 8.0),
    ("D5", "PSNR + fixed 25 mW", "psnr", 0.25, "free", 8.0),
    ("D6", "PSNR + power budget", "psnr", 0.5, "budget", 8.0),
]
QER_CENTRES = [(-180.0 + 36.0 + 72.0 * k, -90.0 + 18.0 + 36.0 * j) for j in range(5) for k in range(5)]
_rng = np.random.default_rng(0)
H_SAMPLES = np.array([channel_model.sample_channel_gain(_rng) for _ in range(20000)])


def prepare(name):
    cat = data_loader.load_video_catalog()
    rd, hn = data_loader.load_rd_data(), data_loader.load_hn_data()
    v = fs.Video(name, cat, rd, hn)
    held = list(config.EVAL_TRACE_INDICES) if name == "Runner" else [i for i in range(len(v.traces)) if i % 4 == 2]
    train = [i for i in range(len(v.traces)) if i not in held]
    for p in train + held:
        v.trajectory(p)
        for s in (TRAIN_SEEDS if p in train else HELD_SEEDS):
            v.channel(p, s)
    v.bundle = None
    v.traces = None
    return v, train, held


def scan(args):
    v, train, held = args
    tr_eps = [(p, s) for p in train for s in TRAIN_SEEDS]
    he_eps = [(p, s) for p in held for s in HELD_SEEDS]
    T = v.traj
    eth = np.concatenate([rl.wsigned(T[p]["at"] - T[p]["pt"])[1:] for p in train])
    eph = np.concatenate([(T[p]["ap"] - T[p]["pp"])[1:] for p in train])
    s_fit = (float(np.mean(np.abs(eth))), float(np.mean(np.abs(eph))))
    Ecov = {p: np.where(T[p]["mand"], 0.0, np.stack([rl.weights(T[p]["pt"][t] + eth, T[p]["pp"][t] + eph).mean(axis=0)
                                                     for t in range(fs.H)])) for p in T}
    Epsnr = {p: Ecov[p] * v.gain for p in T}
    Acov = {p: np.where(T[p]["mand"], 0.0, T[p]["w"]) for p in T}
    Apsnr = {p: Acov[p] * v.gain for p in T}

    # literature tile rules: extra-tile masks per viewer (threshold rule decides at run time)
    def region(p, hw, hh):
        return fs.region_masks(T[p]["pt"], T[p]["pp"], hw, hh)

    def qer(p, w, h):
        out = np.zeros((fs.H, fs.NT), bool)
        for t in range(fs.H):
            d = [np.hypot(rl.wsigned(T[p]["pt"][t] - c[0]), T[p]["pp"][t] - c[1]) for c in QER_CENTRES]
            c = QER_CENTRES[int(np.argmin(d))]
            out[t] = fs.region_masks(c[0], c[1], w / 2, h / 2)[0]
        return out

    lit_tiles = [("predicted viewport only", {p: np.zeros((fs.H, fs.NT), bool) for p in T}, False)]
    lit_tiles += [(f"QER {w}x{h}", {p: qer(p, w, h) for p in T}, False) for w, h in [(120, 60), (160, 80), (200, 100)]]
    for lam in (0.25, 0.5, 0.75, 1.0):
        lit_tiles.append((f"average-based {lam}x", {p: region(p, config.V_THETA_DEG + lam * s_fit[0], config.V_PHI_DEG + lam * s_fit[1]) for p in T}, False))
    lit_tiles.append(("average-based (MMSP 30,15)", {p: region(p, config.V_THETA_DEG + 30, config.V_PHI_DEG + 15) for p in T}, False))
    for lam in (0.5, 1.0):
        lit_tiles.append((f"threshold (10 s), margin {lam}x", {p: region(p, config.V_THETA_DEG + lam * s_fit[0], config.V_PHI_DEG + lam * s_fit[1]) for p in T}, True))

    rows = []
    for key, label, obj, top_frac, pmode, b_bar in DESIGNS:
        P = fs.levels(top_frac)
        top = len(P) - 1
        objk = "J_cov" if obj == "cov" else "J_psnr"

        def power(rule, A, Z, h):
            if rule == "top":
                return top
            if rule.startswith("fixed"):
                return int(rule.split(":")[1])
            k = fs.pmin_level(A, Z, h, P)
            if rule.startswith("buf"):
                b = float(rule.split(":")[1])
                while k < top and Z + fs.T0 * fs.rate(P[k], h) - A < b * A:
                    k += 1
            elif rule.startswith("opp") and h >= np.quantile(H_SAMPLES, float(rule.split(":")[1])):
                k = min(k + 1, top)
            return k

        def lit_policy(masks, thresh, rule):
            def pol(s):
                p, t = s["pos"], s["t"]
                ex = masks[p][t] if (not thresh or s["Z"] / v.mean_base_bits >= 10.0) else np.zeros(fs.NT, bool)
                x = ex | T[p]["mand"][t]
                return ex, power(rule, v.A_base[t] + v.delta[t][x].sum(), s["Z"], s["h"])
            return pol

        def adaptive(score, tau, alpha, cap_k, rule):
            def pol(s):
                p, t = s["pos"], s["t"]
                sc = score[p][t]
                cand = np.where(sc >= tau)[0]
                cand = cand[np.argsort(-sc[cand])]
                a_m = v.A_base[t] + v.delta[t][T[p]["mand"][t]].sum()
                cap = alpha * (s["Z"] + fs.T0 * fs.rate(P[cap_k], s["h"]))
                n_add = int(np.searchsorted(a_m + np.cumsum(v.delta[t][cand]), cap, side="right"))
                ex = np.zeros(fs.NT, bool); ex[cand[:n_add]] = True
                return ex, power(rule, a_m + v.delta[t][cand[:n_add]].sum(), s["Z"], s["h"])
            return pol

        triv_rule = "top" if pmode == "free" else "pmin"
        triv_pol = lit_policy({p: np.zeros((fs.H, fs.NT), bool) for p in T}, False, triv_rule)
        p_bar = None
        if pmode == "budget":
            p_bar = 1.25 * fs.simulate(v, tr_eps, triv_pol, P)["J_P"]

        def ok(m, mg=1.0):
            return m["J_D"] <= D_BAR * mg and m["J_B"] <= b_bar * mg and (p_bar is None or m["J_P"] <= p_bar * mg)

        def best_of(cands):
            evald = [(name, pol, fs.simulate(v, tr_eps, pol, P)) for name, pol in cands]
            feas = [e for e in evald if ok(e[2], MARGIN)]
            if not feas:
                return None
            name, pol, _ = max(feas, key=lambda e: e[2][objk])
            return name, fs.simulate(v, he_eps, pol, P)

        def record(method, choice, m):
            rows.append(dict(video=v.name, design=key, design_label=label, objective=obj, P_bar=p_bar, method=method,
                             choice=choice, held_feasible=ok(m), **m))

        record("trivial", "predicted viewport only", fs.simulate(v, he_eps, triv_pol, P))
        rules = ["top"] if pmode == "free" else ["pmin", "fixed:1", "fixed:2", "top"]
        best = best_of([(f"{n} [{r}]", lit_policy(m, th, r)) for n, m, th in lit_tiles for r in rules])
        if best:
            record("literature", *best)

        def rl_cands(score):
            vals = np.concatenate([score[p][score[p] > 0] for p in train])
            taus = np.quantile(vals, [0.5, 0.75, 0.9, 0.95, 0.98])
            caps = [top] if pmode == "free" else [2, 3, 4]
            prules = ["top"] if pmode == "free" else ["pmin", "buf:0.3", "opp:0.8"]
            return [(f"tau q{q} alpha {a} cap {c} {r}", adaptive(score, tau, a, c, r))
                    for q, tau in zip([50, 75, 90, 95, 98], taus) for a in (0.5, 0.7, 0.85, 1.0) for c in caps for r in prules]
        best = best_of(rl_cands(Ecov if obj == "cov" else Epsnr))
        if best:
            record("rl_estimate", *best)
        if obj == "psnr":
            best = best_of(rl_cands(Ecov))
            if best:
                record("rl_estimate_no_content", *best)

        # ceiling: perfect prediction, pooled tile budget on the highest actual values, shrunk to stay feasible
        A = Acov if obj == "cov" else Apsnr
        items, base = [], 0.0
        for p in held:
            base += np.sum(fs.DISC * T[p]["mand"].sum(axis=1)) / fs.NT
            items += [(A[p][t, i], p, t, i) for t in range(fs.H) for i in np.where(A[p][t] > 0)[0]]
        items.sort(key=lambda x: -x[0])

        def oracle(frac):
            budget = frac * (len(held) * b_bar - base)
            masks = {p: np.zeros((fs.H, fs.NT), bool) for p in held}
            for _, p, t, i in items:
                c_ = fs.DISC[t] / fs.NT
                if budget < c_:
                    break
                masks[p][t, i] = True; budget -= c_
            return fs.simulate(v, he_eps, lit_policy(masks, False, triv_rule), P)
        lo, hi, bm = 0.0, 1.0, None
        for _ in range(12):
            mid = (lo + hi) / 2
            m = oracle(mid)
            if ok(m):
                lo, bm = mid, m
            else:
                hi = mid
        record("ceiling", f"perfect prediction ({lo:.0%} of extra tile budget)", bm if bm is not None else oracle(0.0))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    prepared = [prepare(n) for n in VIDEOS]
    print("prepared", flush=True)
    with Pool(min(len(VIDEOS), 7)) as pool:
        res = pool.map(scan, prepared)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(f"{OUT}/design_options.csv", index=False)
    print("saved", f"{OUT}/design_options.csv")


if __name__ == "__main__":
    main()
