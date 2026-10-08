"""
room_by_video.py - per video: how much room is there for RL above the
literature rules, and where do the literature rules' extra tiles go?

Literature rules (the paper's baselines), each tuned to the budgets on the
video's training viewers and tested on held-out viewers:
  predicted viewport only; MMSP QER (120x60, 160x80, 200x100);
  MMSP average-based (fixed margin, several sizes); ASL360 threshold
  (extra tiles only once the buffer holds 10 s of base layer).
Reference (not a baseline): random extra tiles.
RL estimate (NOT a paper baseline): adaptive rule that ranks extra tiles by
  expected coverage from the predicted direction and adds them while the
  link can deliver them - used only to estimate what a learned policy can reach.
Ceiling: perfect viewport prediction, tile budget spent optimally.

Power fixed at the top level (50 mW); stall budget 0.3; tile budget swept.

Usage: .venv/Scripts/python.exe room_by_video.py
"""
import os

import numpy as np
import pandas as pd

import config
import room_to_learn as rl
import run_nonrl_baselines as rb
from streaming_rl import channel_model, data_loader, layer_model, viewport
from streaming_rl.environment import TileStreamingEnv

OUT_DIR = "VIDEO_SURVEY/room_by_video"
VIDEOS = ["Runner", "GateNight", "SiyuanGate", "Academic"]
BUDGETS = [7.0, 7.5, 8.0, 9.0, 10.0, 11.0]
H, NT = 36, config.N_TILES
D_BAR, MARGIN = 0.3, 0.95
TRAIN_SEEDS, HELD_SEEDS = (2000, 2001, 2002), (1000, 1001, 1002)
W, T0 = config.W_HZ, config.T0_SEC
WN0 = W * channel_model.dbm_to_watts(config.N0_DBM_PER_HZ)
P_TOP = config.POWER_LEVEL_MAX_FRACTION * channel_model.dbm_to_watts(config.P_MAX_DBM)
LEVEL = config.INITIAL_ENHANCEMENT_LEVELS[1]


def run(bundle, eps, policy, keep_steps=False):
    """policy(env, pos, t) -> proposed extra-tile mask. Top power always."""
    rs, steps = [], []
    for pos, seed in eps:
        env = TileStreamingEnv(trace_indices=[pos], bundle=bundle)
        env.reset(seed=seed)
        cov, extra, useful = [], [], []
        for t in range(H):
            prop = policy(env, pos, t)
            _, _, te, tr, info = env.step(np.concatenate([prop.astype(np.int64), [config.N_POWER_LEVELS - 1]]))
            mand = info["mandatory_tile_mask"].astype(bool)
            ext = prop & ~mand
            hit = rl.weights(info["actual_theta"], info["actual_phi"])[0] > 0
            cov.append(info["coverage"]); extra.append(ext.sum()); useful.append((ext & hit).sum())
            if keep_steps:
                steps.append(dict(pos=pos, seed=seed, t=t, extra=ext, mand=mand, hit=hit,
                                  pred=(info["predicted_theta"], info["predicted_phi"]),
                                  actual=(info["actual_theta"], info["actual_phi"])))
            if te or tr:
                break
        rs.append(dict(J_Q=info["J_Q"], J_D=info["J_D"], J_B=info["J_B"], cov=np.mean(cov),
                       extra=np.mean(extra), useful=np.mean(useful)))
    m = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
    return (m, steps) if keep_steps else m


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cat = data_loader.load_video_catalog()
    hn = data_loader.load_hn_data()
    rd = data_loader.load_rd_data()
    rows, example_steps = [], {}
    for vname in VIDEOS:
        v = cat[cat.name == vname].iloc[0]
        vi = int(v.array_index)
        config.TRAINING_VIDEO_ARRAY_INDEX = vi
        traces = data_loader.get_valid_traces(hn, vi)
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
        s_fit = (float(np.mean(np.abs(eth))), float(np.mean(np.abs(eph))))
        E = {p: np.where(mand[p], -1.0, np.stack([rl.weights(tj[p]["pt"][t] + eth, tj[p]["pp"][t] + eph).mean(axis=0)
                                                  for t in range(H)])) for p in tj}
        A_base, DELTA = {}, {}
        for t in range(H):
            r = layer_model.compute_A_t(rd, vi, t, np.zeros(NT, bool), enhanced_level=LEVEL)
            A_base[t], DELTA[t] = r["A_base"], np.asarray(r["delta"], float)
        mean_base = np.mean(list(A_base.values()))

        def lit(kind, param):
            dec = rb.make_policy(kind, param, "p50", mean_base)
            return lambda env, p, t: dec(env)[0]

        def rnd(k, seed=0):
            g = np.random.default_rng(seed)

            def pol(env, p, t):
                m = viewport.mandatory_tile_mask(env._predicted_theta, env._predicted_phi)
                prop = np.zeros(NT, bool); prop[g.choice(np.where(~m)[0], size=k, replace=False)] = True
                return prop
            return pol

        def adaptive(score, tau, alpha):
            def pol(env, p, t):
                m = mand[p][t]; s = score[p][t]
                cand = np.where(~m & (s >= tau))[0]
                cand = cand[np.argsort(-s[cand])]
                cap = alpha * (env._Z + T0 * W * np.log2(1.0 + P_TOP * env._h / WN0))
                n_add = int(np.searchsorted(A_base[t] + DELTA[t][m].sum() + np.cumsum(DELTA[t][cand]), cap, side="right"))
                prop = np.zeros(NT, bool); prop[cand[:n_add]] = True
                return prop
            return pol

        options = [("literature", "predicted viewport only", lambda env, p, t: np.zeros(NT, bool))]
        options += [("literature", f"QER {w}x{h}", lit("qer", (float(w), float(h)))) for w, h in [(120, 60), (160, 80), (200, 100)]]
        options += [("literature", f"average-based {lam}x", lit("average", (lam * s_fit[0], lam * s_fit[1]))) for lam in (0.25, 0.5, 0.75, 1.0)]
        options += [("literature", "average-based (MMSP bounds 30,15)", lit("average", (30.0, 15.0)))]
        options += [("literature", f"threshold, EL = margin {lam}x", lit("threshold", (lam * s_fit[0], lam * s_fit[1]))) for lam in (0.5, 1.0)]
        options += [("random", f"random +{k}", rnd(k)) for k in range(1, 9)]
        options += [("rl_estimate", f"adaptive tau={tau} alpha={a}", adaptive(E, tau, a))
                    for tau in (0.003, 0.005, 0.01, 0.02, 0.04) for a in (0.5, 0.7, 0.85, 1.0)]
        res = []
        for fam, name, pol in options:
            res.append(dict(fam=fam, name=name, pol=pol, tr=run(bundle, tr_eps, pol), he=run(bundle, he_eps, pol)))
        print(f"{vname}: evaluated {len(res)} options", flush=True)

        # ceiling: perfect prediction, pooled tile budget spent on the highest-coverage tiles
        items, base = [], 0.0
        for p in held:
            base += np.sum(rl.DISC * mand[p].sum(axis=1)) / NT
            items += [(actual[p][t, i], p, t, i) for t in range(H) for i in np.where(~mand[p][t] & (actual[p][t] > 0))[0]]
        items.sort(key=lambda x: -x[0])

        def oracle_masks(budget):
            masks = {p: np.zeros((H, NT), bool) for p in held}
            for w_, p, t, i in items:
                c_ = rl.DISC[t] / NT
                if budget < c_:
                    break
                masks[p][t, i] = True; budget -= c_
            return masks

        for b_bar in BUDGETS:
            ok = lambda m, mg=1.0: m["J_D"] <= D_BAR * mg and m["J_B"] <= b_bar * mg
            for fam in ["literature", "random", "rl_estimate"]:
                cands = [r for r in res if r["fam"] == fam and ok(r["tr"], MARGIN)]
                if not cands:
                    continue
                best = max(cands, key=lambda r: r["tr"]["J_Q"])
                rows.append(dict(video=vname, B_bar=b_bar, family=fam, choice=best["name"], held_feasible=ok(best["he"]), **best["he"]))
            lo, hi, best_m = 0.0, 1.0, None
            full = len(held) * b_bar - base
            for _ in range(10):
                mid = (lo + hi) / 2
                ms = oracle_masks(mid * full)
                m = run(bundle, he_eps, lambda env, p, t: ms[p][t])
                if ok(m):
                    lo, best_m = mid, m
                else:
                    hi = mid
            if best_m is not None:
                rows.append(dict(video=vname, B_bar=b_bar, family="ceiling", choice=f"perfect prediction ({lo:.0%} of extra budget)",
                                 held_feasible=True, **best_m))
            print(f"{vname} B_bar={b_bar} done", flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT_DIR}/room_by_video.csv", index=False)

        # per-step records of the methods at B_bar = 8 for the waste picture
        b8 = [r for r in rows if r["video"] == vname and r["B_bar"] == 8.0 and r["family"] in ("literature", "rl_estimate")]
        for r in b8:
            pol = next(o["pol"] for o in res if o["name"] == r["choice"])
            _, steps = run(bundle, he_eps[:len(held)], pol, keep_steps=True)
            example_steps[(vname, r["family"], r["choice"])] = steps
    config.TRAINING_VIDEO_ARRAY_INDEX = 4
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/room_by_video.csv", index=False)
    pd.to_pickle(example_steps, f"{OUT_DIR}/example_steps_B8.pkl")
    print("saved", f"{OUT_DIR}/room_by_video.csv")


if __name__ == "__main__":
    main()
