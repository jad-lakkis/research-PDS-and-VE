"""
room_to_learn.py - "is there room to learn?" test for Experiment X's setup,
rules and oracles only (no training). Held-out traces 2,6,9 with channel
seeds 1000-1002 - the exact evaluation the learned policies get.

Levels compared (tile side, power free = Experiment X setting):
  1. existing rules (predicted viewport only, QER 120x60, average-based 0.25x)
  2. Bayes threshold rule (CAUSAL): add tile i if its EXPECTED viewport
     coverage, under the prediction-error samples of the 9 TRAINING traces,
     is >= tau; tau set on training traces to meet the tile budget.
     "+motion" conditions the error samples on the size of the previous
     error (information the agent already observes).
  3. viewport oracle (NON-CAUSAL): knows the actual viewport, spends the
     pooled tile budget on the highest-coverage tiles (greedy knapsack;
     fractional version = a true upper bound, integer version simulated).
Power side (power-budget setting P_bar=6.5): for each tile plan, greedy
per-slot minimum power vs a schedule planned with the FUTURE channel
known (continuous power up to 50 mW, earliest-deadline greedy).

Usage: .venv/Scripts/python.exe room_to_learn.py
"""
import os

import numpy as np
import pandas as pd

import config
import run_nonrl_baselines as rb
from streaming_rl import channel_model, data_loader, layer_model, viewport
from streaming_rl.environment import TileStreamingEnv

OUT_DIR = "EXPERIMENT_X_drop_seed6/room_to_learn"
D_BAR, P_BAR, B_BAR = 0.3, 6.5, 8.0
GAM = config.DISCOUNT_FACTOR_LAMBDA
H, NT = 36, config.N_TILES
W, T0 = config.W_HZ, config.T0_SEC
WN0 = W * channel_model.dbm_to_watts(config.N0_DBM_PER_HZ)
P_MAX = channel_model.dbm_to_watts(config.P_MAX_DBM)
P_TOP = rb.P_LEVELS[-1]
DISC = GAM ** np.arange(H)
TB = np.array([viewport.tile_bounds(i) for i in range(NT)])  # (64,4): th0, th1, ph0, ph1
VP_AREA = (2 * config.V_THETA_DEG) * (2 * config.V_PHI_DEG)


def wsigned(d):
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


def weights(theta, phi):
    """(N,64) coverage weight of each tile for viewports centred at (theta,phi):
    area of tile inside the viewport / viewport area (sums to 1)."""
    theta, phi = np.atleast_1d(theta)[:, None], np.atleast_1d(phi)[:, None]
    ov_t = np.zeros((theta.shape[0], NT))
    for s in (-360.0, 0.0, 360.0):
        ov_t += np.clip(np.minimum(theta + config.V_THETA_DEG, TB[:, 1] + s)
                        - np.maximum(theta - config.V_THETA_DEG, TB[:, 0] + s), 0, None)
    ov_p = np.clip(np.minimum(phi + config.V_PHI_DEG, TB[:, 3]) - np.maximum(phi - config.V_PHI_DEG, TB[:, 2]), 0, None)
    return ov_t * ov_p / VP_AREA


def trajectory(bundle, pos, seed):
    env = TileStreamingEnv(trace_indices=[pos], bundle=bundle)
    env.reset(seed=seed)
    rec = []
    for _ in range(H):
        _, _, te, tr, info = env.step(np.concatenate([np.zeros(NT, np.int64), [4]]))
        rec.append((info["predicted_theta"], info["predicted_phi"], info["actual_theta"], info["actual_phi"], info["h_t"]))
        if te or tr:
            break
    r = np.array(rec)
    return dict(pt=r[:, 0], pp=r[:, 1], at=r[:, 2], ap=r[:, 3], h=r[:, 4])


def motion(tr):
    """size of the previous slot's error (what the agent observes at decision time)."""
    e = np.hypot(wsigned(tr["at"] - tr["pt"]), tr["ap"] - tr["pp"])
    return np.concatenate([[0.0], e[:-1]])


def simulate(bundle, pos, seed, masks, power="p50"):
    """Run the real env with the given per-slot extra-tile masks."""
    env = TileStreamingEnv(trace_indices=[pos], bundle=bundle)
    env.reset(seed=seed)
    cov, psnr, A, h = [], [], [], []
    for t in range(H):
        executed = masks[t] | viewport.mandatory_tile_mask(env._predicted_theta, env._predicted_phi)
        k = rb.power_idx("pwmin", env, executed) if power == "pwmin" else 4
        _, _, te, tr, info = env.step(np.concatenate([masks[t].astype(np.int64), [k]]))
        cov.append(info["coverage"]); psnr.append(info["viewport_psnr_db"]); A.append(info["A_t"]); h.append(info["h_t"])
        if te or tr:
            break
    return dict(J_Q=info["J_Q"], J_D=info["J_D"], J_P=info["J_P"], J_B=info["J_B"], cov=np.mean(cov), psnr=np.mean(psnr),
                A=np.array(A), h=np.array(h))


def planned_power(A, h):
    """Zero-stall schedule with the future channel known: deliver each slot's
    data by its deadline from the cheapest earlier-or-same slot (continuous
    power <= 50 mW). Returns (J_P, undeliverable bits)."""
    c = DISC[:len(A)] * WN0 / (h * P_MAX)
    rmax = W * np.log2(1.0 + P_TOP * h / WN0)
    R = np.zeros(len(A))
    short = 0.0
    for t in range(len(A)):
        need = A[t]
        chunk = A[t] / 200.0
        while need > 1e-6:
            d = min(chunk, need)
            k = np.arange(t + 1)
            ok = R[k] + d <= rmax[k] + 1e-9
            if not ok.any():
                short += need
                break
            marg = np.where(ok, c[k] * (2.0 ** ((R[k] + d) / W) - 2.0 ** (R[k] / W)), np.inf)
            j = k[int(np.argmin(marg))]
            R[j] += d
            need -= d
    return float(np.sum(c * (2.0 ** (R / W) - 1.0))), short


def myopic_power(A, h):
    """Per-slot exact minimum power with no buffer planning (continuous)."""
    c = DISC[:len(A)] * WN0 / (h * P_MAX)
    rmax = W * np.log2(1.0 + P_TOP * h / WN0)
    R = np.minimum(A, rmax)
    return float(np.sum(c * (2.0 ** (R / W) - 1.0))), float(np.sum(A - R))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    bundle = data_loader.load_training_video_bundle()
    mean_base = np.mean([layer_model.compute_A_t(bundle["rd_data"], config.TRAINING_VIDEO_ARRAY_INDEX, t, np.zeros(NT, bool),
                                                 enhanced_level=config.INITIAL_ENHANCEMENT_LEVELS[1])["A_base"] for t in range(H)])
    held = [(p, s, trajectory(bundle, p, s)) for p, s in zip(config.EVAL_TRACE_INDICES, config.EVAL_SEEDS)]
    train = [(p, 2000, trajectory(bundle, p, 2000)) for p in rb.TRAIN_POS]
    train_eps = [(p, s) for p in rb.TRAIN_POS for s in (2000, 2001, 2002)]  # channel seeds for stall checks

    # sanity: vectorized weights reproduce viewport.coverage exactly
    tr0 = held[0][2]
    m0 = viewport.mandatory_tile_mask(tr0["pt"][5], tr0["pp"][5])
    assert abs(weights(tr0["at"][5], tr0["ap"][5])[0] @ m0 - viewport.coverage(m0, tr0["at"][5], tr0["ap"][5])) < 1e-12

    # training error samples, tagged by previous-error size
    eth, eph, mot = [], [], []
    for _, _, tr in train:
        eth.append(wsigned(tr["at"] - tr["pt"])[1:]); eph.append((tr["ap"] - tr["pp"])[1:]); mot.append(motion(tr)[1:])
    eth, eph, mot = map(np.concatenate, (eth, eph, mot))
    edges = np.quantile(mot, [0.25, 0.5, 0.75])
    print(f"training error samples: {len(eth)}; previous-error quartile edges (deg): {np.round(edges, 1).tolist()}")

    def expected(tr, conditional):
        mand = weights(tr["pt"], tr["pp"]) > 0
        mb = np.digitize(motion(tr), edges)
        E = np.zeros((H, NT))
        for t in range(H):
            sel = (np.digitize(mot, edges) == mb[t]) if conditional else np.ones(len(eth), bool)
            E[t] = weights(tr["pt"][t] + eth[sel], tr["pp"][t] + eph[sel]).mean(axis=0)
        return np.where(mand, -1.0, E), mand

    def bayes_masks(trs, tau, conditional, cache):
        out = []
        for key, tr in trs:
            if (key, conditional) not in cache:
                cache[(key, conditional)] = expected(tr, conditional)
            E, _ = cache[(key, conditional)]
            out.append(E >= tau)
        return out

    def jb(trs, masks):
        return np.mean([np.sum(DISC * ((weights(tr["pt"], tr["pp"]) > 0) | m).sum(axis=1)) / NT for (_, tr), m in zip(trs, masks)])

    cache = {}
    tr_items = [(("train", p), tr) for p, _, tr in train]
    he_items = [(("held", p), tr) for p, _, tr in held]
    rows = []

    def record(name, masks, info_kind):
        sims = [simulate(bundle, p, s, m) for (p, s, _), m in zip(held, masks)]
        g = lambda k: float(np.mean([x[k] for x in sims]))
        v = max(0, g("J_D") - D_BAR) + max(0, g("J_B") - B_BAR)
        # power-budget setting: same tiles, greedy pwmin vs planned-with-future-channel
        simp = [simulate(bundle, p, s, m, power="pwmin") for (p, s, _), m in zip(held, masks)]
        jp_greedy = float(np.mean([x["J_P"] for x in simp])); jd_greedy = float(np.mean([x["J_D"] for x in simp]))
        pl = [planned_power(x["A"], x["h"]) for x in sims]
        my = [myopic_power(x["A"], x["h"]) for x in sims]
        rows.append(dict(method=name, information=info_kind, J_Q=g("J_Q"), coverage=g("cov"), psnr_db=g("psnr"), J_D=g("J_D"),
                         J_B=g("J_B"), violation_powerfree=v, feasible_powerfree=v <= 1e-9,
                         powerbudget_greedy_J_P=jp_greedy, powerbudget_greedy_J_D=jd_greedy,
                         powerbudget_myopic_cont_J_P=float(np.mean([m_[0] for m_ in my])),
                         myopic_undeliverable_Mbit=float(np.mean([m_[1] for m_ in my])) / 1e6,
                         powerbudget_planned_future_channel_J_P=float(np.mean([p_[0] for p_ in pl])),
                         planned_undeliverable_Mbit=float(np.mean([p_[1] for p_ in pl])) / 1e6))

    # 1. existing rules (rebuilt per held-out trajectory)
    for name, kind, param in [("Predicted viewport only", "mandatory", None), ("QER 120x60", "qer", (120.0, 60.0)),
                              ("Average-based 0.25x fitted", "average", (7.68, 0.83))]:
        dec = rb.make_policy(kind, param, "p50", mean_base)
        masks = []
        for p, s, _ in held:
            env = TileStreamingEnv(trace_indices=[p], bundle=bundle); env.reset(seed=s); ms = []
            for t in range(H):
                prop, _ = dec(env); ms.append(prop)
                env.step(np.concatenate([prop.astype(np.int64), [4]]))
            masks.append(np.array(ms))
        record(name, masks, "causal rule")

    # 2. Bayes threshold rules. tau set on TRAINING traces only: smallest tau
    # (most tiles) with training J_B <= 8 AND training J_D <= 0.3 at 50 mW.
    def train_ok(tau, conditional):
        bm = dict(zip([p for p, _, _ in train], bayes_masks(tr_items, tau, conditional, cache)))
        sims = [simulate(bundle, p, s, bm[p]) for p, s in train_eps]
        return np.mean([x["J_B"] for x in sims]) <= B_BAR and np.mean([x["J_D"] for x in sims]) <= D_BAR

    for conditional, label in [(False, "Bayes threshold rule"), (True, "Bayes threshold rule + motion")]:
        lo, hi = 0.0, 0.2
        for _ in range(18):
            mid = (lo + hi) / 2
            if train_ok(mid, conditional):
                hi = mid
            else:
                lo = mid
        tau = hi
        print(f"{label}: tau={tau:.4f} (training-feasible on both budgets)")
        record(f"{label} (tau set on training)", bayes_masks(he_items, tau, conditional, cache), "causal rule")

    # 3. viewport oracle: greedy knapsack on ACTUAL coverage weights, pooled tile
    # budget; the budget is then shrunk until the held-out stall budget also
    # holds (the oracle may use held-out knowledge - it is an upper reference).
    items, base_cost, base_val = [], 0.0, 0.0
    for e, (p, s, tr) in enumerate(held):
        wa = weights(tr["at"], tr["ap"]); mand = weights(tr["pt"], tr["pp"]) > 0
        base_cost += np.sum(DISC * mand.sum(axis=1)) / NT
        base_val += np.sum(DISC * (wa * mand).sum(axis=1))
        for t in range(H):
            for i in np.where(~mand[t] & (wa[t] > 0))[0]:
                items.append((wa[t, i], e, t, i))
    items.sort(key=lambda x: -x[0])

    def oracle_masks(budget):
        masks = [np.zeros((H, NT), bool) for _ in held]
        frac_val = base_val
        for w_, e, t, i in items:
            c = DISC[t] / NT
            if budget >= c:
                masks[e][t, i] = True; budget -= c; frac_val += DISC[t] * w_
            else:
                frac_val += DISC[t] * w_ * max(budget, 0) / c
                break
        return masks, frac_val / len(held)

    full_budget = len(held) * B_BAR - base_cost
    _, ub = oracle_masks(full_budget)
    print(f"viewport oracle: tile-budget-only fractional upper bound J_Q = {ub:.3f}")
    lo, hi = 0.0, 1.0
    for _ in range(14):
        mid = (lo + hi) / 2
        ms, _ = oracle_masks(mid * full_budget)
        if np.mean([simulate(bundle, p, s, m)["J_D"] for (p, s, _), m in zip(held, ms)]) <= D_BAR:
            lo = mid
        else:
            hi = mid
    ms, _ = oracle_masks(lo * full_budget)
    print(f"viewport oracle: stall-feasible at {lo * 100:.1f}% of the extra-tile budget")
    record("Viewport oracle (knows actual viewport)", ms, "NON-causal")
    rows[-1]["J_Q_fractional_upper_bound"] = ub

    df = pd.DataFrame(rows)
    comp = pd.read_csv("EXPERIMENT_X_drop_seed6/nonrl_baselines/comparison_heldout_with_learned.csv", index_col=0)
    for name in [n for n in comp.index if n.startswith("LEARNED")]:
        r = comp.loc[name]
        df.loc[len(df)] = dict(method=name, information="learned (no direction)", J_Q=r["J_Q"], coverage=r["mean_coverage"],
                               psnr_db=r["mean_psnr_db"], J_D=r["J_D"], J_B=r["J_B"], violation_powerfree=r["violation"])
    df.to_csv(f"{OUT_DIR}/summary_heldout.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(df.drop(columns=["J_Q_fractional_upper_bound"], errors="ignore"))


if __name__ == "__main__":
    main()
