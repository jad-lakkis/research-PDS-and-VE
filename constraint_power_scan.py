"""
constraint_power_scan.py - hypothesis test: keep the two constraints (stall,
tiles; power free) but change the power action's ceiling and the budget
values, so that the stall budget can bind. Do the heuristics still do well?

Configs: top power level in {50, 25, 12.5} mW (5 levels from 0, as now)
         x tile budget B_bar in {8, 10}; stall budget D_bar = 0.3.
Videos:  Runner (usual held-out split), GateNight (every 4th viewer held out).
Methods (all causal except the ceiling; tuned on training viewers, tested on held-out):
  pred     predicted viewport only, top power
  fixed    best of: + k most useful fixed tiles (no direction) / + tiles with
           expected coverage >= tau (uses predicted direction); top power
  smart    state-aware: rank extra tiles by expected coverage, add them (above a
           value threshold) only while the segment fits what the link can deliver
           now: A_t <= alpha * (Z_t + T0 * R_top(h_t)) - uses current buffer and channel
  ceiling  the smart rule with the ACTUAL viewport known (perfect prediction),
           threshold fitted on held-out
Power is free here, so every method transmits at the top level.

Usage: .venv/Scripts/python.exe constraint_power_scan.py [videos] [top-power fractions] [out.csv]
       e.g.  constraint_power_scan.py SiyuanGate,Academic 0.25,0.125,0.0625 VIDEO_SURVEY/constraint_power_scan_2.csv
"""
import os
import sys

import numpy as np
import pandas as pd

import config
import room_to_learn as rl
import run_nonrl_baselines as rb
from streaming_rl import channel_model, data_loader, layer_model, viewport
from streaming_rl.environment import TileStreamingEnv

OUT = "VIDEO_SURVEY/constraint_power_scan.csv"
H, NT = 36, config.N_TILES
D_BAR = 0.3
MARGIN = 0.95
TRAIN_SEEDS, HELD_SEEDS = (2000, 2001, 2002), (1000, 1001, 1002)
W, T0 = config.W_HZ, config.T0_SEC
WN0 = W * channel_model.dbm_to_watts(config.N0_DBM_PER_HZ)
PMAX = channel_model.dbm_to_watts(config.P_MAX_DBM)
LEVEL = config.INITIAL_ENHANCEMENT_LEVELS[1]


def run(bundle, eps, policy):
    rs = []
    for pos, seed in eps:
        env = TileStreamingEnv(trace_indices=[pos], bundle=bundle)
        env.reset(seed=seed)
        cov, psnr = [], []
        for t in range(H):
            prop = policy(env, pos, t)
            _, _, te, tr, info = env.step(np.concatenate([prop.astype(np.int64), [config.N_POWER_LEVELS - 1]]))
            cov.append(info["coverage"]); psnr.append(info["viewport_psnr_db"])
            if te or tr:
                break
        rs.append(dict(J_Q=info["J_Q"], J_D=info["J_D"], J_B=info["J_B"], cov=np.mean(cov), psnr=np.mean(psnr),
                       tiles=np.mean([0])))
    return {k: float(np.mean([r[k] for r in rs])) for k in ["J_Q", "J_D", "J_B", "cov", "psnr"]}


def main():
    videos = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Runner", "GateNight"]
    fracs = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.5, 0.25, 0.125]
    out = sys.argv[3] if len(sys.argv) > 3 else OUT
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cat = data_loader.load_video_catalog()
    hn = data_loader.load_hn_data()
    rd = data_loader.load_rd_data()
    rows = []
    for vname in videos:
        v = cat[cat.name == vname].iloc[0]
        config.TRAINING_VIDEO_ARRAY_INDEX = int(v.array_index)
        traces = data_loader.get_valid_traces(hn, v.array_index)
        bundle = {"rd_data": rd, "valid_traces": traces}
        held = list(config.EVAL_TRACE_INDICES) if vname == "Runner" else [i for i in range(len(traces)) if i % 4 == 2]
        train = [i for i in range(len(traces)) if i not in held]
        tr_eps = [(p, s) for p in train for s in TRAIN_SEEDS]
        he_eps = [(p, s) for p in held for s in HELD_SEEDS]
        tj = {p: rl.trajectory(bundle, p, 0) for p in train + held}
        mand = {p: rl.weights(tj[p]["pt"], tj[p]["pp"]) > 0 for p in tj}
        actual = {p: rl.weights(tj[p]["at"], tj[p]["ap"]) for p in tj}
        eth = np.concatenate([rl.wsigned(tj[p]["at"] - tj[p]["pt"])[1:] for p in train])
        eph = np.concatenate([(tj[p]["ap"] - tj[p]["pp"])[1:] for p in train])
        E = {p: np.where(mand[p], -1.0, np.stack([rl.weights(tj[p]["pt"][t] + eth, tj[p]["pp"][t] + eph).mean(axis=0)
                                                  for t in range(H)])) for p in tj}
        gain = sum(np.where(mand[p], 0.0, actual[p]).sum(axis=0) for p in train)
        order = np.argsort(-gain)
        A_base, DELTA = {}, {}
        for t in range(H):
            r = layer_model.compute_A_t(rd, int(v.array_index), t, np.zeros(NT, bool), enhanced_level=LEVEL)
            A_base[t], DELTA[t] = r["A_base"], np.asarray(r["delta"], float)

        def fixed_policy(masks):
            return lambda env, p, t: masks[p][t]

        def smart_policy(score, tau, alpha, p_top):
            def pol(env, p, t):
                m = mand[p][t]
                s = score[p][t]
                cand = np.where(~m & (s >= tau))[0]
                cand = cand[np.argsort(-s[cand])]
                a_m = A_base[t] + DELTA[t][m].sum()
                cap = alpha * (env._Z + T0 * W * np.log2(1.0 + p_top * env._h / WN0))
                n_add = int(np.searchsorted(a_m + np.cumsum(DELTA[t][cand]), cap, side="right"))
                prop = np.zeros(NT, bool); prop[cand[:n_add]] = True
                return prop
            return pol

        s_fit = (float(np.mean(np.abs(eth))), float(np.mean(np.abs(eph))))  # this video's mean training error

        def literature_policy(kind, param):
            dec = rb.make_policy(kind, param, "p50", 1.0)
            return lambda env, p, t: dec(env)[0]

        for cap_frac in fracs:
            config.POWER_LEVEL_MAX_FRACTION = cap_frac
            p_top = cap_frac * PMAX
            for b_bar in [8.0, 10.0]:
                ok = lambda m, mg=1.0: m["J_D"] <= D_BAR * mg and m["J_B"] <= b_bar * mg
                tag = dict(video=vname, top_power_mW=p_top * 1e3, B_bar=b_bar)
                pred = {p: np.zeros((H, NT), bool) for p in tj}
                m_pred = run(bundle, he_eps, fixed_policy(pred))
                rows.append(dict(**tag, method="pred", held_feasible=ok(m_pred), **m_pred))
                for name, kind, par in [("QER 120x60", "qer", (120.0, 60.0)),
                                        ("average-based 0.25x", "average", (0.25 * s_fit[0], 0.25 * s_fit[1])),
                                        ("average-based 0.5x", "average", (0.5 * s_fit[0], 0.5 * s_fit[1]))]:
                    m = run(bundle, he_eps, literature_policy(kind, par))
                    rows.append(dict(**tag, method=name, held_feasible=ok(m), **m))
                # fixed rules (no state): tuned on training
                best = None
                opts = [(f"nodir k={k}", {p: np.tile(np.isin(np.arange(NT), order[:k]), (H, 1)) for p in tj}) for k in range(1, 9)]
                opts += [(f"dir tau={tau:.3f}", {p: E[p] >= tau for p in tj}) for tau in np.geomspace(0.005, 0.15, 8)]
                for name, masks in opts:
                    m_tr = run(bundle, tr_eps, fixed_policy(masks))
                    if ok(m_tr, MARGIN) and (best is None or m_tr["J_Q"] > best[1]["J_Q"]):
                        best = (name, m_tr, masks)
                if best:
                    m = run(bundle, he_eps, fixed_policy(best[2]))
                    rows.append(dict(**tag, method=f"fixed ({best[0]})", held_feasible=ok(m), **m))
                # smart state-aware rule: tuned on training
                best = None
                for tau in [0.005, 0.01, 0.02, 0.04]:
                    for alpha in [0.5, 0.7, 0.85, 1.0]:
                        m_tr = run(bundle, tr_eps, smart_policy(E, tau, alpha, p_top))
                        if ok(m_tr, MARGIN) and (best is None or m_tr["J_Q"] > best[1]["J_Q"]):
                            best = ((tau, alpha), m_tr)
                if best:
                    m = run(bundle, he_eps, smart_policy(E, *best[0], p_top))
                    rows.append(dict(**tag, method=f"smart (tau={best[0][0]}, alpha={best[0][1]})", held_feasible=ok(m), **m))
                # ceiling: smart structure with the actual viewport, fitted on held-out
                best = None
                for tau in [0.0005, 0.005, 0.02, 0.05, 0.1]:
                    for alpha in [0.5, 0.7, 0.85, 1.0]:
                        m = run(bundle, he_eps, smart_policy(actual, tau, alpha, p_top))
                        if ok(m) and (best is None or m["J_Q"] > best[1]["J_Q"]):
                            best = ((tau, alpha), m)
                if best:
                    rows.append(dict(**tag, method=f"ceiling (perfect prediction; tau={best[0][0]}, alpha={best[0][1]})",
                                     held_feasible=True, **best[1]))
                print(f"{vname} top={p_top * 1e3:.1f} mW B_bar={b_bar} done", flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)
    config.POWER_LEVEL_MAX_FRACTION = 0.5
    config.TRAINING_VIDEO_ARRAY_INDEX = 4
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_rows", 200,
                           "display.max_colwidth", 60, "display.float_format", "{:.3f}".format):
        print(df)


if __name__ == "__main__":
    main()
