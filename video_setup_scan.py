"""
video_setup_scan.py - for every 36-s video, with and without the power
budget: does the trivial heuristic (predicted viewport only) already meet
all budgets, how well do tuned rules do, and how far is the
perfect-prediction ceiling? Rules only, no training.

Same environment physics as training (the env is pointed at each video by
setting config.TRAINING_VIDEO_ARRAY_INDEX in-process; nothing on disk
changes). Rules are tuned on each video's training viewers and evaluated on
held-out viewers (every 4th viewer; Runner keeps its usual split).

Settings:  S1 = stall <= 0.3, tiles <= 8, power free (Experiment X)
           S2 = stall <= 0.3, power <= 6.5, tiles <= 8 (original formulation)
Tile rules:  pred      predicted viewport only
             nodir(k)  + k fixed tiles most useful on training (no direction)
             dir(tau)  + tiles whose expected coverage >= tau (uses the predicted direction)
Power rules: p50 (S1);  pmin = lowest level avoiding a stall (P_min),
             opp(q) = 50 mW when the current channel is above its q-quantile, else pmin (S2)
Ceiling:     oracle tiles placed with the actual viewport known, budget-fitted on held-out.

Usage: .venv/Scripts/python.exe video_setup_scan.py
"""
import os

import numpy as np
import pandas as pd

import config
import room_to_learn as rl
import run_nonrl_baselines as rb
from streaming_rl import channel_model, data_loader, viewport
from streaming_rl.environment import TileStreamingEnv

OUT = "VIDEO_SURVEY/setup_scan.csv"
H, NT = 36, config.N_TILES
D_BAR, P_BAR, B_BAR = 0.3, 6.5, 8.0
MARGIN = 0.95
TRAIN_SEEDS, HELD_SEEDS = (2000, 2001, 2002), (1000, 1001, 1002)
TAUS = np.geomspace(0.005, 0.15, 8)
KS = range(1, 7)
_rng = np.random.default_rng(0)
H_SAMPLES = np.array([channel_model.sample_channel_gain(_rng) for _ in range(50000)])


def feasible(m, setting, margin=1.0):
    ok = m["J_D"] <= D_BAR * margin and m["J_B"] <= B_BAR * margin
    return ok and (setting == "S1" or m["J_P"] <= P_BAR * margin)


def run(bundle, eps, masks, prule, hq=None):
    rs = []
    for pos, seed in eps:
        env = TileStreamingEnv(trace_indices=[pos], bundle=bundle)
        env.reset(seed=seed)
        cov, psnr = [], []
        for t in range(H):
            prop = masks[pos][t]
            mand = viewport.mandatory_tile_mask(env._predicted_theta, env._predicted_phi)
            if prule == "p50":
                k = 4
            else:
                k = rb.power_idx("pwmin", env, prop | mand)
                if prule == "opp" and env._h >= hq:
                    k = 4
            _, _, te, tr, info = env.step(np.concatenate([prop.astype(np.int64), [k]]))
            cov.append(info["coverage"]); psnr.append(info["viewport_psnr_db"])
            if te or tr:
                break
        rs.append(dict(J_Q=info["J_Q"], J_D=info["J_D"], J_P=info["J_P"], J_B=info["J_B"], cov=np.mean(cov), psnr=np.mean(psnr)))
    return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}


def trajectories(bundle, positions):
    return {p: rl.trajectory(bundle, p, 0) for p in positions}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cat = data_loader.load_video_catalog()
    hn = data_loader.load_hn_data()
    rd = data_loader.load_rd_data()
    rows = []
    for _, v in cat[cat.frames == 1080].iterrows():
        config.TRAINING_VIDEO_ARRAY_INDEX = int(v.array_index)
        traces = data_loader.get_valid_traces(hn, v.array_index)
        bundle = {"rd_data": rd, "valid_traces": traces}
        n_v = len(traces)
        held = list(config.EVAL_TRACE_INDICES) if v["name"] == "Runner" else [i for i in range(n_v) if i % 4 == 2]
        train = [i for i in range(n_v) if i not in held]
        tr_eps = [(p, s) for p in train for s in TRAIN_SEEDS]
        he_eps = [(p, s) for p in held for s in HELD_SEEDS]
        tj = trajectories(bundle, train + held)

        # per-trace tile options (decisions depend only on the viewing trajectory)
        mand = {p: rl.weights(tj[p]["pt"], tj[p]["pp"]) > 0 for p in tj}
        eth = np.concatenate([rl.wsigned(tj[p]["at"] - tj[p]["pt"])[1:] for p in train])
        eph = np.concatenate([(tj[p]["ap"] - tj[p]["pp"])[1:] for p in train])
        E = {p: np.where(mand[p], -1.0, np.stack([rl.weights(tj[p]["pt"][t] + eth, tj[p]["pp"][t] + eph).mean(axis=0)
                                                  for t in range(H)])) for p in tj}
        gain = sum(np.where(mand[p], 0.0, rl.weights(tj[p]["at"], tj[p]["ap"])).sum(axis=0) for p in train)
        order = np.argsort(-gain)
        options = [("pred", None, {p: np.zeros((H, NT), bool) for p in tj})]
        for k in KS:
            m = np.zeros((H, NT), bool); m[:, order[:k]] = True
            options.append(("nodir", k, {p: m for p in tj}))
        for tau in TAUS:
            options.append(("dir", round(float(tau), 4), {p: E[p] >= tau for p in tj}))

        for setting, prules in [("S1", [("p50", None)]), ("S2", [("pmin", None), ("opp", 0.5), ("opp", 0.8)])]:
            cand = []
            for fam, par, masks in options:
                for pr, q in prules:
                    hq = np.quantile(H_SAMPLES, q) if q is not None else None
                    m_tr = run(bundle, tr_eps, masks, pr, hq)
                    cand.append(dict(fam=fam, par=par, prule=f"{pr}{'' if q is None else q}", masks=masks, pr=pr, hq=hq,
                                     tr=m_tr, ok=feasible(m_tr, setting, MARGIN)))
            for fam in ["pred", "nodir", "dir"]:
                c = [x for x in cand if x["fam"] == fam and x["ok"]]
                if not c:
                    c = [min([x for x in cand if x["fam"] == fam],
                             key=lambda x: max(0, x["tr"]["J_D"] - D_BAR) + max(0, x["tr"]["J_B"] - B_BAR)
                             + (0 if setting == "S1" else max(0, x["tr"]["J_P"] - P_BAR)))]
                best = max(c, key=lambda x: x["tr"]["J_Q"])
                m_he = run(bundle, he_eps, best["masks"], best["pr"], best["hq"])
                rows.append(dict(video=v["name"], setting=setting, method=fam, param=best["par"], power=best["prule"],
                                 train_feasible=best["ok"], held_feasible=feasible(m_he, setting), **{f"held_{k}": val for k, val in m_he.items()}))
            # perfect-prediction ceiling: oracle tile knapsack, shrunk until feasible on held-out
            items, base = [], 0.0
            for p in held:
                wa = rl.weights(tj[p]["at"], tj[p]["ap"])
                base += np.sum(rl.DISC * mand[p].sum(axis=1)) / NT
                items += [(wa[t, i], p, t, i) for t in range(H) for i in np.where(~mand[p][t] & (wa[t] > 0))[0]]
            items.sort(key=lambda x: -x[0])
            pr_or = "p50" if setting == "S1" else "pmin"

            def oracle(frac):
                budget = frac * (len(held) * B_BAR - base)
                masks = {p: np.zeros((H, NT), bool) for p in held}
                for w_, p, t, i in items:
                    c_ = rl.DISC[t] / NT
                    if budget < c_:
                        break
                    masks[p][t, i] = True; budget -= c_
                return run(bundle, he_eps, masks, pr_or)
            lo, hi, best_m = 0.0, 1.0, oracle(0.0)
            for _ in range(10):
                mid = (lo + hi) / 2
                m_ = oracle(mid)
                if feasible(m_, setting):
                    lo, best_m = mid, m_
                else:
                    hi = mid
            rows.append(dict(video=v["name"], setting=setting, method="oracle", param=round(lo, 3), power=pr_or,
                             train_feasible=None, held_feasible=feasible(best_m, setting), **{f"held_{k}": val for k, val in best_m.items()}))
            print(f"{v['name']} {setting} done", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    config.TRAINING_VIDEO_ARRAY_INDEX = 4
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_rows", 200, "display.float_format", "{:.3f}".format):
        print(df)


if __name__ == "__main__":
    main()
