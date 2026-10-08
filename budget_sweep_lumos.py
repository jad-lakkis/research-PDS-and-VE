"""
budget_sweep_lumos.py - which stall / tile budgets to use with Lumos5G link traces?

Setup as lumos_replay.py: Runner, level 5, raw Lumos5G rates (R_t known at decision
time), power fixed, forced predicted-viewport tiles, coverage objective, all 36-s windows.

Stall budget = outage floor + allowance. On real traces even the most conservative
policy (predicted viewport only) stalls during outages; that stall cannot be avoided
by any policy. D_bar = (viewport-only J_D on training runs) + allowance, and a method's
AVOIDABLE stall = its J_D minus the viewport-only J_D on the same episodes. A method is
within the stall budget when its avoidable stall <= allowance (judged on the same
episodes, so the random outage stall cancels instead of swamping the comparison).

Episodes: training viewers x training runs (36-s windows) for tuning; held-out viewers
(2, 6, 9) x validation runs and x test runs (windows every 12 s) for evaluation. Budgets
are chosen on validation; test confirms.

Methods: see lumos_replay.py, plus
  oracle   knows the actual viewport AND the whole future link of every test window;
           adds actual-viewport tiles (largest coverage first, pooled over episodes like
           the expected-cost constraints) while the tile budget and the avoidable-stall
           allowance last. A greedy upper bound, not exactly optimal.

Each policy is simulated once; only the selection changes with the budgets.

Usage: .venv/Scripts/python.exe budget_sweep_lumos.py
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config
import design_options_scan as ds
import fastsim as fs
import lumos_replay as lr
import room_to_learn as rl
from streaming_rl import lumos5g

OUT_DIR = "VIDEO_SURVEY/lumos5g"
ALLOW = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
B_BARS = (7.5, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0)
EVAL_STRIDE, N_HIST, MARGIN = 12, 5, 0.95
G = {}


def setup():
    v, train, held = lr.prepare()
    T = v.traj
    runs, info = lumos5g.load()
    sp = lumos5g.split_runs(info)
    win = {"train": lumos5g.windows(runs, sp["train"], fs.H),
           "val": lumos5g.windows(runs, sp["val"], fs.H, EVAL_STRIDE),
           "test": lumos5g.windows(runs, sp["test"], fs.H, EVAL_STRIDE)}
    eps = {"train": [(train[(i + j * 3) % len(train)], w) for i, w in enumerate(win["train"]) for j in range(3)],
           "val": [(p, w) for w in win["val"] for p in held], "test": [(p, w) for w in win["test"] for p in held]}
    rates = {w: lumos5g.rate_window(runs, w[0], w[1], fs.H, 1.0, N_HIST) for ws in win.values() for w in ws}
    A_m = {p: v.A_base + np.array([v.delta[t][T[p]["mand"][t]].sum() for t in range(fs.H)]) for p in T}
    pm = {p: np.concatenate([[0.0], rl.wsigned(np.diff(T[p]["pt"]))]) for p in T}
    e_t = np.concatenate([rl.wsigned(T[p]["at"] - T[p]["pt"])[2:] for p in train])
    e_p = np.concatenate([(T[p]["ap"] - T[p]["pp"])[2:] for p in train])
    mv = np.digitize(np.concatenate([pm[p][2:] for p in train]), lr.BINS)
    eth = np.concatenate([rl.wsigned(T[p]["at"] - T[p]["pt"])[1:] for p in train])
    eph = np.concatenate([(T[p]["ap"] - T[p]["pp"])[1:] for p in train])
    s_fit = (float(np.mean(np.abs(eth))), float(np.mean(np.abs(eph))))
    EM, EO = {}, {}
    for p in T:
        b = np.digitize(pm[p], lr.BINS)
        rows_ = []
        for t in range(fs.H):
            sel = mv == b[t]
            if sel.sum() < 20:
                sel = np.ones(len(e_t), bool)
            rows_.append(rl.weights(T[p]["pt"][t] + e_t[sel], T[p]["pp"][t] + e_p[sel]).mean(axis=0))
        EM[p] = np.where(T[p]["mand"], 0.0, np.stack(rows_))
        EO[p] = np.where(T[p]["mand"], 0.0, T[p]["w"])
    masks = {}
    for w, h in [(120, 60), (160, 80), (200, 100)]:
        mm = {}
        for p in T:
            a = np.zeros((fs.H, fs.NT), bool)
            for t in range(fs.H):
                d = [np.hypot(rl.wsigned(T[p]["pt"][t] - c[0]), T[p]["pp"][t] - c[1]) for c in ds.QER_CENTRES]
                c = ds.QER_CENTRES[int(np.argmin(d))]
                a[t] = fs.region_masks(c[0], c[1], w / 2, h / 2)[0]
            mm[p] = a
        masks[f"QER {w}x{h}"] = mm
    avg = lambda st, sp_: {p: fs.region_masks(T[p]["pt"], T[p]["pp"], config.V_THETA_DEG + st, config.V_PHI_DEG + sp_) for p in T}
    masks["none"] = {p: np.zeros((fs.H, fs.NT), bool) for p in T}
    masks["avg 30/15"] = avg(30.0, 15.0)
    for lam in lr.LAMS:
        masks[f"avg {lam}x"] = avg(lam * s_fit[0], lam * s_fit[1])
    taus = np.quantile(np.concatenate([EM[p][EM[p] > 0] for p in train]), [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98])
    taus_o = np.quantile(np.concatenate([EO[p][EO[p] > 0] for p in train]), [0.0, 0.25, 0.5, 0.75])
    G.update(v=v, train=train, held=held, eps=eps, rates=rates, A_m=A_m, EM=EM, EO=EO, masks=masks, taus=taus, taus_o=taus_o,
             tau_sets={"EM": taus, "EO": taus_o})


def candidates():
    """(family, name, spec) - specs are plain tuples so they can be sent to workers."""
    c = [("published", "predicted viewport only", ("mask", "none", False)),
         ("published", "QER 120x60 (MMSP)", ("mask", "QER 120x60", False)),
         ("published", "QER 160x80 (MMSP)", ("mask", "QER 160x80", False)),
         ("published", "QER 200x100 (MMSP)", ("mask", "QER 200x100", False)),
         ("published", "average-based, bounds 30/15 (MMSP)", ("mask", "avg 30/15", False)),
         ("published", "average-based, fitted to our error data (MMSP rule)", ("mask", "avg 1.0x", False)),
         ("published", "threshold 10 s base layer, enlarged viewport (ASL360, adapted)", ("mask", "avg 1.0x", True))]
    c += [("lit", n, ("mask", n, False)) for n in ("QER 120x60", "QER 160x80", "QER 200x100")]
    for lam in lr.LAMS:
        c += [("lit", f"average-based {lam}x", ("mask", f"avg {lam}x", False)),
              ("lit", f"threshold, margin {lam}x", ("mask", f"avg {lam}x", True))]
    for k in range(7):
        for a in (0.5, 0.7, 1.0):
            for r in (0.0, 0.25, 0.5, 1.0, 2.0):
                for link in ("now", "min5"):
                    c.append(("rl", f"tau q{k} a {a} reserve {r} link {link}", ("rl", "EM", k, a, r, link)))
    for k in range(4):
        for a in (0.5, 0.7, 1.0):
            for r in (0.0, 0.25, 0.5, 1.0, 2.0):
                c.append(("ceil", f"tau q{k} a {a} reserve {r}", ("rl", "EO", k, a, r, "now")))
    return c


def policy(spec, record=None):
    v, A_m, zero = G["v"], G["A_m"], np.zeros(fs.NT, bool)
    if spec[0] == "mask":
        mm, th = G["masks"][spec[1]], spec[2]
        return lambda s: mm[s["pos"]][s["t"]] if (not th or s["Z"] / v.mean_base_bits >= 10.0) else zero
    _, which, k, alpha, reserve, link = spec
    S = G[which]
    tau = G["tau_sets"][which][k]

    def pol(s):
        p, t = s["pos"], s["t"]
        sc = S[p][t]
        cand = np.where(sc >= tau)[0]
        cand = cand[np.argsort(-sc[cand])]
        R = s["R"] if link == "now" else min(s["R"], s["Rh"].min())
        room = alpha * (s["Z"] + fs.T0 * R) - reserve * A_m[p][t]
        n_add = int(np.searchsorted(A_m[p][t] + np.cumsum(v.delta[t][cand]), room, side="right"))
        ex = np.zeros(fs.NT, bool); ex[cand[:n_add]] = True
        if record is not None:
            record[(p, s["key"], t)] = A_m[p][t] + v.delta[t][cand[:n_add]].sum()
        return ex
    return pol


def evaluate(spec):
    return {sp: fs.simulate_trace(G["v"], G["eps"][sp], policy(spec), G["rates"], N_HIST) for sp in ("train", "val", "test")}


def uniform_for(spec):
    out = {}
    for sp in ("val", "test"):
        rec = {}
        fs.simulate_trace(G["v"], G["eps"][sp], policy(spec, rec), G["rates"], N_HIST)
        out[sp] = lr.uniform_trace(G["v"], G["eps"][sp], G["rates"], rec)
    return out


def ep_stall(A, R):
    Z, J = 0.0, 0.0
    for t in range(fs.H):
        s = Z + fs.T0 * R[t]
        if s < A[t]:
            J += fs.DISC[t] * (1.0 - s / A[t]); Z = 0.0
        else:
            Z = s - A[t]
    return J


def oracle(args, return_masks=False):
    """pooled greedy with perfect viewport and link foresight; returns the metrics of its masks"""
    split, allow, b_bar = args
    v, T, eps, rates = G["v"], G["v"].traj, G["eps"][split], G["rates"]
    A = [G["A_m"][p].astype(float).copy() for p, _ in eps]
    R = [rates[w][N_HIST:] for _, w in eps]
    S = [ep_stall(A[e], R[e]) for e in range(len(eps))]
    b_left = len(eps) * b_bar - sum(np.sum(fs.DISC * T[p]["mand"].sum(axis=1)) / fs.NT for p, _ in eps)
    d_left = len(eps) * allow
    items = [(T[p]["w"][t, i], e, t, i) for e, (p, _) in enumerate(eps) for t in range(fs.H)
             for i in np.where(~T[p]["mand"][t] & (T[p]["w"][t] > 0))[0]]
    items.sort(key=lambda x: -x[0])
    chosen = {e: np.zeros((fs.H, fs.NT), bool) for e in range(len(eps))}
    for pass_ in (0, 1):
        for _, e, t, i in items:
            cost = fs.DISC[t] / fs.NT
            if chosen[e][t, i] or cost > b_left:
                continue
            p = eps[e][0]
            A[e][t] += v.delta[t][i]
            s_new = ep_stall(A[e], R[e])
            ds_ = s_new - S[e]
            if ds_ <= 1e-12 or (pass_ == 1 and ds_ <= d_left):
                chosen[e][t, i] = True; b_left -= cost; d_left -= max(ds_, 0.0); S[e] = s_new
            else:
                A[e][t] -= v.delta[t][i]
    lookup = {(eps[e][0], eps[e][1]): chosen[e] for e in chosen}
    if return_masks:
        return lookup
    m = fs.simulate_trace(v, eps, lambda s: lookup[(s["pos"], s["key"])][s["t"]], rates, N_HIST)
    return dict(split=split, allowance=allow, b_bar=b_bar, **m)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cands = candidates()
    with Pool(8, initializer=setup) as pool:
        res = pool.map(evaluate, [c[2] for c in cands], chunksize=4)
        evals = pd.DataFrame([dict(family=f, name=n, split=sp, **m) for (f, n, _), r in zip(cands, res) for sp, m in r.items()])
        evals.to_csv(f"{OUT_DIR}/budget_sweep_candidates.csv", index=False)
        floor = evals[evals.name == "predicted viewport only"].set_index("split")
        evals["avoidable_stall"] = evals.J_D - evals.split.map(floor.J_D)
        D_floor_train = float(floor.loc["train", "J_D"])

        # per budget pair: tuned literature / RL estimate / ceiling chosen on training runs
        tr = evals[evals.split == "train"].set_index("name")
        picks = {}
        for allow in ALLOW:
            for b in B_BARS:
                for fam in ("lit", "rl", "ceil"):
                    f = tr[(tr.family == fam) & (tr.avoidable_stall <= MARGIN * allow) & (tr.J_B <= MARGIN * b)]
                    picks[(allow, b, fam)] = f.J_cov.idxmax() if len(f) else None
        spec = {n: s for _, n, s in cands}
        rl_picks = sorted({n for (a, b, f), n in picks.items() if f == "rl" and n})
        uni = dict(zip(rl_picks, pool.map(uniform_for, [spec[n] for n in rl_picks])))
        orc = pool.map(oracle, [(sp, a, b) for sp in ("val", "test") for a in ALLOW for b in B_BARS])
    orc = pd.DataFrame(orc)

    rows = []
    for sp in ("val", "test"):
        ev = evals[evals.split == sp].set_index("name")
        fl = ev.loc["predicted viewport only"]
        for allow in ALLOW:
            for b in B_BARS:
                ok = lambda m: (m["J_D"] - fl["J_D"]) <= allow + 1e-9 and m["J_B"] <= b + 1e-9
                tag = dict(split=sp, allowance=allow, d_bar=round(D_floor_train + allow, 3), b_bar=b)

                def add(method, m, choice="", within=None):
                    rows.append(dict(**tag, method=method, choice=choice, psnr=m["psnr"], cov=m.get("cov", 1.0),
                                     avoidable_stall=m["J_D"] - fl["J_D"], J_D=m["J_D"], J_B=m["J_B"],
                                     stall_pct=m["stall_pct"], qvar_dB=m["qvar_dB"], bits_Mbit=m["bits"] / 1e6,
                                     within_budgets=ok(m) if within is None else within))
                pub = ev[ev.family == "published"]
                for n, m in pub.iterrows():
                    add(n, m)
                feas = [n for n, m in pub.iterrows() if ok(m)]
                bp = max(feas, key=lambda n: pub.loc[n, "psnr"])
                add("best published rule within budgets", pub.loc[bp], bp)
                for fam, label in (("lit", "literature, tuned to our budgets"), ("rl", "RL estimate"),
                                   ("ceil", "RL estimate, perfect viewport")):
                    n = picks[(allow, b, fam)]
                    if n:
                        add(label, ev.loc[n], n)
                        if fam == "rl":
                            u = uni[n][sp]
                            add("uniform, same bits as RL estimate", u, "all 64 tiles", within=(u["J_D"] - fl["J_D"]) <= allow + 1e-9)
                o = orc[(orc.split == sp) & (orc.allowance == allow) & (orc.b_bar == b)].iloc[0]
                add("oracle (actual viewport + future link)", o)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/budget_sweep.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(f"stall floor (predicted viewport only, training runs): J_D = {D_floor_train:.3f}")
    for sp in ("val", "test"):
        p = df[df.split == sp].pivot_table(index=["allowance", "b_bar"], columns="method", values="psnr")
        f = df[df.split == sp].pivot_table(index=["allowance", "b_bar"], columns="method", values="within_budgets", aggfunc="first")
        out = pd.DataFrame({"viewport only": p["predicted viewport only"], "best published": p["best published rule within budgets"],
                            "lit tuned": p.get("literature, tuned to our budgets"), "uniform": p.get("uniform, same bits as RL estimate"),
                            "RL est": p.get("RL estimate"), "RL est ok": f.get("RL estimate"), "oracle": p["oracle (actual viewport + future link)"]})
        out["RL - lit tuned"] = out["RL est"] - out["lit tuned"]
        out["RL - best published"] = out["RL est"] - out["best published"]
        out["RL - uniform"] = out["RL est"] - out["uniform"]
        out["oracle - RL"] = out["oracle"] - out["RL est"]
        out["oracle - lit tuned"] = out["oracle"] - out["lit tuned"]
        print(f"===== {sp} runs (PSNR dB) =====")
        print(out.round(2).to_string())


if __name__ == "__main__":
    main()
