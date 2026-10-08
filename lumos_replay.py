"""
lumos_replay.py - replay the baselines on Lumos5G link traces before any training.

Setup (formulation unchanged): Runner, stall budget 0.3, tile budget 8, power fixed
(R_t comes from the trace, R_t = scale * 1e6 * Throughput_t, known at decision time),
forced predicted-viewport tiles, coverage objective.
Episodes: a viewer paired with a 36-s window of one Lumos5G run.
  tuning (tuned variants, RL estimate): training viewers x training runs
  reported:                             held-out viewers (2, 6, 9) x test runs
Configurations (enhancement level, trace scale, windows): kappa = mean link rate / mean
bits of (base layer + predicted-viewport tiles) per segment.
  windows "all":         every 36-s window of the split's runs
  windows "sustainable": only windows whose link carries the forced stream (base layer +
                         predicted-viewport tiles, average training-viewer bits per second)
                         with no stall from an empty buffer - a property of the trace alone

Methods:
  predicted viewport only                    (mandatory transmission alone)
  QER 120x60 / 160x80 / 200x100 (MMSP)       published sizes
  average-based, MMSP bounds 30/15 (MMSP)    published enlargement
  average-based, fitted to our error data    MMSP rule, margin = mean prediction error
  ASL360 threshold (adapted)                 extra tiles (fitted-error enlarged viewport) only once
                                             the buffer holds 10 s of base layer (45.3 Mbit)
  literature, tuned to our budgets           best feasible of QER / average-based / threshold
                                             over a fine margin grid (reported separately)
  uniform, same bits as RL estimate          every segment spends exactly the RL estimate's bits
                                             for that segment, spread evenly over all 64 tiles
  RL estimate                                motion-aware expected-coverage ranking + link/buffer
                                             reserve (a rule standing in for a learned policy;
                                             not a paper baseline)
  RL estimate, perfect viewport              same rule ranking by the actual viewport (ceiling)

Usage: .venv/Scripts/python.exe lumos_replay.py
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import combined_test as ct
import config
import design_options_scan as ds
import fastsim as fs
import room_to_learn as rl
from streaming_rl import data_loader, lumos5g

OUT_DIR = "VIDEO_SURVEY/lumos5g"
# stall budgets for "all" windows follow the original calibration rule: a fixed position
# between predicted-viewport-only (J_D 1.94) and all-64-tiles (J_D 8.22) on training runs;
# 5% -> 2.25, 10% -> 2.57 (Experiment X's 0.3 sat at 1.6% of the synthetic-channel range)
CONFIGS = [dict(level=5, scale=1.0, windows="all", d_bar=d, b_bar=b) for d in (2.25, 2.57) for b in (8.0, 12.0, 16.0)] + [
    dict(level=5, scale=1.0, windows="sustainable", d_bar=0.3, b_bar=b) for b in (8.0, 12.0, 16.0)]
N_HIST = 5
N_TRAIN_WIN, VIEWERS_PER_WIN = 160, 3
LAMS = ct.LAMS + (1.5, 2.0, 3.0)
BINS = ct.BINS


def prepare(level=fs.LEVEL):
    ds.TRAIN_SEEDS, ds.HELD_SEEDS = (), ()   # no synthetic channels needed
    v, train, held = ds.prepare("Runner")
    rd = data_loader.load_rd_data()
    br, ym = rd["video_bitrate_data"][v.vi], rd["video_ymse_data"][v.vi]
    f = np.arange(fs.H) * config.GOP_SIZE_FRAMES
    v.lv_bits = {k: br[config.qp_array_index_for_enhancement_level(k)][:, f].T for k in range(7)}
    v.lv_mse = {k: ym[config.qp_array_index_for_enhancement_level(k)][:, f].T for k in range(7)}
    if level != fs.LEVEL:
        v.delta = v.lv_bits[level] - v.lv_bits[0]
        v.mse_e = v.lv_mse[level]
        v.gain = v.mse_b - v.mse_e
    return v, train, held


def sustainable(runs, ws, A_avg, scale):
    """windows where the forced stream (average bits A_avg[t]) never stalls from an empty buffer"""
    keep = []
    for w in ws:
        R = lumos5g.rate_window(runs, w[0], w[1], fs.H, scale)
        Z, ok_ = 0.0, True
        for t in range(fs.H):
            if Z + fs.T0 * R[t] < A_avg[t]:
                ok_ = False
                break
            Z += fs.T0 * R[t] - A_avg[t]
        if ok_:
            keep.append(w)
    return keep


def run(cfg):
    scale, B_BAR, D_BAR = cfg["scale"], cfg["b_bar"], cfg["d_bar"]
    v, train, held = prepare(cfg["level"])
    T = v.traj
    A_m = {p: v.A_base + np.array([v.delta[t][T[p]["mand"][t]].sum() for t in range(fs.H)]) for p in T}
    runs, info = lumos5g.load()
    sp = lumos5g.split_runs(info)
    tr_all, te_all = lumos5g.windows(runs, sp["train"], fs.H), lumos5g.windows(runs, sp["test"], fs.H)
    if cfg["windows"] == "sustainable":
        A_avg = np.mean([A_m[p] for p in train], axis=0)
        tr_all, te_w = sustainable(runs, tr_all, A_avg, scale), sustainable(runs, te_all, A_avg, scale)
    else:
        te_w = te_all
    kept = (len(tr_all) / len(lumos5g.windows(runs, sp["train"], fs.H)), len(te_w) / len(te_all))
    tr_w = [tr_all[i] for i in np.random.default_rng(0).choice(len(tr_all), min(N_TRAIN_WIN, len(tr_all)), replace=False)]
    rates = {w: lumos5g.rate_window(runs, w[0], w[1], fs.H, scale, N_HIST) for w in tr_w + te_w}
    tr_eps = [(train[(i + j * 3) % len(train)], w) for i, w in enumerate(tr_w) for j in range(VIEWERS_PER_WIN)]
    he_eps = [(p, w) for w in te_w for p in held]
    kappa = np.mean([rates[w][N_HIST:].mean() for w in te_w]) / np.mean([A_m[p].mean() for p in held])
    ok = lambda m, mg=1.0: m["J_D"] <= D_BAR * mg and m["J_B"] <= B_BAR * mg

    # prediction-error statistics from training viewers (as in combined_test / method_table)
    pm = {p: np.concatenate([[0.0], rl.wsigned(np.diff(T[p]["pt"]))]) for p in T}
    e_t = np.concatenate([rl.wsigned(T[p]["at"] - T[p]["pt"])[2:] for p in train])
    e_p = np.concatenate([(T[p]["ap"] - T[p]["pp"])[2:] for p in train])
    mv = np.digitize(np.concatenate([pm[p][2:] for p in train]), BINS)
    eth = np.concatenate([rl.wsigned(T[p]["at"] - T[p]["pt"])[1:] for p in train])
    eph = np.concatenate([(T[p]["ap"] - T[p]["pp"])[1:] for p in train])
    s_fit = (float(np.mean(np.abs(eth))), float(np.mean(np.abs(eph))))
    EM, EO = {}, {}
    for p in T:
        b = np.digitize(pm[p], BINS)
        rows_ = []
        for t in range(fs.H):
            sel = mv == b[t]
            if sel.sum() < 20:
                sel = np.ones(len(e_t), bool)
            rows_.append(rl.weights(T[p]["pt"][t] + e_t[sel], T[p]["pp"][t] + e_p[sel]).mean(axis=0))
        EM[p] = np.where(T[p]["mand"], 0.0, np.stack(rows_))
        EO[p] = np.where(T[p]["mand"], 0.0, T[p]["w"])

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

    def avg_masks(st, sp_):
        return {p: fs.region_masks(T[p]["pt"], T[p]["pp"], config.V_THETA_DEG + st, config.V_PHI_DEG + sp_) for p in T}

    zero = np.zeros(fs.NT, bool)

    def lit_pol(masks, th):
        return lambda s: masks[s["pos"]][s["t"]] if (not th or s["Z"] / v.mean_base_bits >= 10.0) else zero

    def rl_pol(S, tau, alpha, reserve, record=None):
        def pol(s):
            p, t = s["pos"], s["t"]
            sc = S[p][t]
            cand = np.where(sc >= tau)[0]
            cand = cand[np.argsort(-sc[cand])]
            room = alpha * (s["Z"] + fs.T0 * s["R"]) - reserve * A_m[p][t]
            n_add = int(np.searchsorted(A_m[p][t] + np.cumsum(v.delta[t][cand]), room, side="right"))
            ex = np.zeros(fs.NT, bool); ex[cand[:n_add]] = True
            if record is not None:
                record[(p, s["key"], t)] = A_m[p][t] + v.delta[t][cand[:n_add]].sum()
            return ex
        return pol

    def best(cands):
        ev = [(n, pol, fs.simulate_trace(v, tr_eps, pol, rates, N_HIST)) for n, pol in cands]
        f = [e for e in ev if ok(e[2], ds.MARGIN)]
        if not f:
            return None
        return max(f, key=lambda e: e[2]["J_cov"])

    rows = []

    def add(name, kind, m, choice=""):
        rows.append(dict(level=cfg["level"], scale=scale, b_bar=B_BAR, d_bar=D_BAR, windows=cfg["windows"], kappa=kappa, method=name, kind=kind, choice=choice, within_budgets=ok(m), **m))

    sim = lambda pol: fs.simulate_trace(v, he_eps, pol, rates, N_HIST)
    zm = {p: np.zeros((fs.H, fs.NT), bool) for p in T}
    per_ep = fs.simulate_trace(v, he_eps, lit_pol(zm, False), rates, N_HIST, per_episode=True)
    m0 = {k: float(np.mean([o[k] for o in per_ep])) for k in per_ep[0]}
    add("predicted viewport only", "published", m0)
    frac_bad = float(np.mean([o["J_D"] > D_BAR for o in per_ep]))
    q = {f"{w}x{h}": qer_masks(w, h) for w, h in [(120, 60), (160, 80), (200, 100)]}
    for k, mm in q.items():
        add(f"QER {k} (MMSP)", "published", sim(lit_pol(mm, False)))
    add("average-based, bounds 30/15 (MMSP)", "published", sim(lit_pol(avg_masks(30.0, 15.0), False)))
    fit = avg_masks(*s_fit)
    add("average-based, fitted to our error data (MMSP rule)", "adapted", sim(lit_pol(fit, False)))
    add("threshold 10 s base layer, enlarged viewport (ASL360, adapted)", "adapted", sim(lit_pol(fit, True)))
    lit = [(f"QER {k}", lit_pol(mm, False)) for k, mm in q.items()]
    for lam in LAMS:
        mm = avg_masks(lam * s_fit[0], lam * s_fit[1])
        lit += [(f"average-based {lam}x", lit_pol(mm, False)), (f"threshold, margin {lam}x", lit_pol(mm, True))]
    b = best(lit)
    if b:
        add("literature, tuned to our budgets", "tuned", sim(b[1]), b[0])
    taus = np.quantile(np.concatenate([EM[p][EM[p] > 0] for p in train]), [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98])
    grid = [(tau, a, r) for tau in taus for a in (0.5, 0.7, 1.0) for r in (0.0, 0.25, 0.5, 1.0, 2.0)]
    br = best([(f"tau {tau:.4f} a {a} reserve {r}", rl_pol(EM, tau, a, r)) for tau, a, r in grid])
    if br:
        rec = {}
        tau, a, r = grid[[f"tau {tau:.4f} a {a} reserve {r}" for tau, a, r in grid].index(br[0])]
        add("RL estimate", "RL estimate", sim(rl_pol(EM, tau, a, r, rec)), br[0])
        mu = uniform_trace(v, he_eps, rates, rec)
        add("uniform, same bits as RL estimate (all 64 tiles)", "uniform", mu, "per-segment match")
        rows[-1]["within_budgets"] = mu["J_D"] <= D_BAR   # stall only: uses all 64 tiles by design
    taus_o = np.quantile(np.concatenate([EO[p][EO[p] > 0] for p in train]), [0.0, 0.25, 0.5, 0.75])
    bo = best([(f"tau {tau:.4f} a {a} reserve {r}", rl_pol(EO, tau, a, r))
               for tau in taus_o for a in (0.7, 1.0) for r in (0.0, 0.25, 0.5, 1.0)])
    if bo:
        add("RL estimate, perfect viewport (ceiling)", "upper bound", sim(bo[1]), bo[0])
    return rows, dict(level=cfg["level"], scale=scale, b_bar=B_BAR, d_bar=D_BAR, windows=cfg["windows"], kappa=kappa,
                      train_windows_kept=kept[0], test_windows_kept=kept[1], mean_rate_Mbps=np.mean([rates[w][N_HIST:].mean() for w in te_w]) / 1e6,
                      pred_only_episodes_over_stall_budget=frac_bad, train_eps=len(tr_eps), test_eps=len(he_eps))


def uniform_trace(v, eps, rates, rec):
    """All 64 tiles at one level, some raised one level (fixed random order) so the
    segment's bits equal the RL estimate's bits for the same (viewer, window, t)."""
    order = np.random.default_rng(0).permutation(fs.NT)
    tot = {k: v.lv_bits[k].sum(axis=1) for k in range(7)}
    out = []
    for pos, key in eps:
        tr, R = v.trajectory(pos), rates[key]
        Z, acc, prev = 0.0, dict(psnr=0.0, J_D=0.0, bits=0.0, stall_pct=0.0, qvar_dB=0.0, cov=1.0,
                                 J_B=float(np.sum(fs.DISC))), None
        for t in range(fs.H):
            target = rec[(pos, key, t)]
            k = max([k for k in range(7) if tot[k][t] <= target] or [0])
            lv = np.full(fs.NT, k)
            bits = tot[k][t]
            if k < 6:
                for i in order:
                    add = v.lv_bits[k + 1][t][i] - v.lv_bits[k][t][i]
                    if bits + add <= target:
                        lv[i] = k + 1; bits += add
            Rt = R[N_HIST + t]
            D = 0.0 if Z + fs.T0 * Rt >= bits else fs.T0 * (1.0 - (Z + fs.T0 * Rt) / bits)
            Z = max(Z + fs.T0 * Rt - bits, 0.0)
            w = tr["w"][t]
            mse = np.choose(lv, [v.lv_mse[j][t] for j in range(7)])
            psnr = 10 * np.log10(255.0 ** 2 / (np.sum(w * mse) / np.sum(w)))
            acc["psnr"] += psnr / fs.H; acc["J_D"] += fs.DISC[t] * D / fs.T0; acc["bits"] += bits / fs.H
            acc["stall_pct"] += 100.0 * (D > 0) / fs.H
            if prev is not None:
                acc["qvar_dB"] += abs(psnr - prev) / (fs.H - 1)
            prev = psnr
        out.append(acc)
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def data_summary():
    runs, info = lumos5g.load()
    sp = lumos5g.split_runs(info)
    rows = []
    for name, ids in sp.items():
        y = np.concatenate([runs[r] for r in ids])
        rows.append(dict(split=name, runs=len(ids), windows_36s=len(lumos5g.windows(runs, ids, fs.H)),
                         walking=int((info.loc[ids, "mode"] == "walking").sum()), driving=int((info.loc[ids, "mode"] == "driving").sum()),
                         mean_Mbps=y.mean(), median_Mbps=np.median(y), p10_Mbps=np.quantile(y, 0.1),
                         zero_pct=100 * np.mean(y == 0), below_126_pct=100 * np.mean(y < 126)))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summ = data_summary()
    summ.to_csv(f"{OUT_DIR}/splits.csv", index=False)
    with Pool(min(len(CONFIGS), 8)) as pool:
        res = pool.map(run, CONFIGS)
    df = pd.DataFrame([r for rows, _ in res for r in rows])
    meta = pd.DataFrame([m for _, m in res])
    df.to_csv(f"{OUT_DIR}/replay.csv", index=False)
    meta.to_csv(f"{OUT_DIR}/replay_scales.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 200)
    print(summ.round(1).to_string(index=False))
    print(meta.round(3).to_string(index=False))
    cols = ["windows", "d_bar", "b_bar", "method", "within_budgets", "psnr", "cov", "J_D", "J_B", "stall_pct", "qvar_dB", "buffer_seg", "bits", "choice"]
    d = df.copy(); d["bits"] = d["bits"] / 1e6
    print(d[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
