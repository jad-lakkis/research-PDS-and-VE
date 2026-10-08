"""
robustness_random.py - (1) random tile allocation baseline, (2) re-test of
rules AND the saved PPO policies on new channel draws and on all 12 viewers,
to check the results are not an artifact of the 3 fixed held-out evaluations.

Scoring is identical to the learned policies' held-out evaluation: each
episode's discounted J_Q/J_D/J_B, averaged over the episodes of one
evaluation, then checked against the budgets (D_bar 0.3, B_bar 8).

PPO models: FINAL_PART1_EXP/exp1_official/ppo_budgetA_dropP_seed{s}/best
(Exp1 PPO is bit-identical to Experiment X's PPO rerun; "best" = the
checkpoint the training run itself selected on the standard held-out
evaluation, so on the standard evaluation it is optimistic for PPO, on new
channel draws it is a fair re-test).

Usage: .venv/Scripts/python.exe robustness_random.py
"""
import os
import pickle

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

import config
import room_to_learn as rl
import run_nonrl_baselines as rb
from streaming_rl import data_loader, layer_model, viewport
from streaming_rl.environment import TileStreamingEnv

OUT = "EXPERIMENT_X_drop_seed6/nonrl_baselines/robustness"
H, NT = rl.H, rl.NT
SEEDS_X = [1, 2, 3, 5, 7]
NEW_DRAWS = 10


def episode(bundle, pos, seed, policy):
    env = TileStreamingEnv(trace_indices=[pos], bundle=bundle)
    obs, _ = env.reset(seed=seed)
    cov, psnr, done = [], [], False
    while not done:
        obs, _, te, tr, info = env.step(policy(env, obs))
        done = te or tr
        cov.append(info["coverage"]); psnr.append(info["viewport_psnr_db"])
    return dict(J_Q=info["J_Q"], J_D=info["J_D"], J_B=info["J_B"], cov=np.mean(cov), psnr=np.mean(psnr))


def evaluation(bundle, eps, policy):
    rs = [episode(bundle, p, s, policy) for p, s in eps]
    m = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
    m["feasible"] = m["J_D"] <= rl.D_BAR and m["J_B"] <= rl.B_BAR
    return m


def act(prop, k=4):
    return np.concatenate([prop.astype(np.int64), [k]])


def main():
    os.makedirs(OUT, exist_ok=True)
    bundle = data_loader.load_training_video_bundle()
    mean_base = np.mean([layer_model.compute_A_t(bundle["rd_data"], config.TRAINING_VIDEO_ARRAY_INDEX, t, np.zeros(NT, bool),
                                                 enhanced_level=config.INITIAL_ENHANCEMENT_LEVELS[1])["A_base"] for t in range(H)])
    std_eps = list(zip(config.EVAL_TRACE_INDICES, config.EVAL_SEEDS))
    new_evals = [[(p, 5000 + 3 * j + i) for i, p in enumerate(config.EVAL_TRACE_INDICES)] for j in range(NEW_DRAWS)]
    all_eps = [(p, 6000 + i) for p in range(12) for i in range(3)]

    # ---------- policies ----------
    def pred_only(env, obs):
        return act(np.zeros(NT, bool))

    def random_k(k, rng):
        def pol(env, obs):
            mand = viewport.mandatory_tile_mask(env._predicted_theta, env._predicted_phi)
            prop = np.zeros(NT, bool)
            free = np.where(~mand)[0]
            prop[rng.choice(free, size=min(k, len(free)), replace=False)] = True
            return act(prop)
        return pol

    qer_dec = rb.make_policy("qer", (120.0, 60.0), "p50", mean_base)

    def qer(env, obs):
        return act(qer_dec(env)[0])

    train = [rl.trajectory(bundle, p, 2000) for p in rb.TRAIN_POS]
    eth = np.concatenate([rl.wsigned(t["at"] - t["pt"])[1:] for t in train])
    eph = np.concatenate([(t["ap"] - t["pp"])[1:] for t in train])

    def expcov(tau):
        def pol(env, obs):
            th, ph = env._predicted_theta, env._predicted_phi
            mand = viewport.mandatory_tile_mask(th, ph)
            E = rl.weights(th + eth, ph + eph).mean(axis=0)
            return act((E >= tau) & ~mand)
        return pol

    # threshold for the expected-coverage rule: training traces only, 5% margin fixed in advance
    train_eps = [(p, s) for p in rb.TRAIN_POS for s in (2000, 2001, 2002)]
    lo, hi = 0.0, 0.2
    for _ in range(16):
        mid = (lo + hi) / 2
        m = evaluation(bundle, train_eps, expcov(mid))
        if m["J_B"] <= 0.95 * rl.B_BAR and m["J_D"] <= 0.95 * rl.D_BAR:
            hi = mid
        else:
            lo = mid
    tau = hi
    print(f"expected-coverage rule threshold (training, 5% margin): {tau:.4f}")

    ppo = []
    for s in SEEDS_X:
        d = f"FINAL_PART1_EXP/exp1_official/ppo_budgetA_dropP_seed{s}/best"
        model = PPO.load(f"{d}/best_model.zip", device="cpu")
        with open(f"{d}/best_vecnormalize.pkl", "rb") as f:
            vn = pickle.load(f)
        vn.training = False
        ppo.append((s, model, vn))

    def ppo_pol(model, vn):
        def pol(env, obs):
            return model.predict(vn.normalize_obs(obs), deterministic=True)[0]
        return pol

    # ---------- (1) random allocation on the standard held-out evaluation ----------
    rows = []
    for k in range(0, 7):
        reps = [evaluation(bundle, std_eps, random_k(k, np.random.default_rng(100 + r))) for r in range(30)]
        rows.append(dict(method=f"Random: predicted viewport + {k} random tiles", evaluation="standard held-out (30 random repeats)",
                         coverage=np.mean([x["cov"] for x in reps]), coverage_std=np.std([x["cov"] for x in reps]),
                         psnr=np.mean([x["psnr"] for x in reps]), J_D=np.mean([x["J_D"] for x in reps]),
                         J_B=np.mean([x["J_B"] for x in reps]), feasible_pct=np.mean([x["feasible"] for x in reps]) * 100))

    # ---------- (2) robustness: standard, 10 new channel draws, all 12 viewers ----------
    methods = [("Predicted viewport only", lambda r: pred_only), ("Random: + 2 random tiles", lambda r: random_k(2, np.random.default_rng(r))),
               ("QER 120x60", lambda r: qer), (f"Expected-coverage rule (training, 5% margin)", lambda r: expcov(tau))]
    methods += [(f"PPO best model seed {s}", (lambda m_, v_: (lambda r: ppo_pol(m_, v_)))(m, v)) for s, m, v in ppo]
    for name, make in methods:
        for label, evals in [("standard held-out", [std_eps]), (f"held-out viewers, {NEW_DRAWS} new channel draws", new_evals),
                             ("all 12 viewers x 3 new channel draws", [all_eps])]:
            res = [evaluation(bundle, ev, make(1000 + j)) for j, ev in enumerate(evals)]
            rows.append(dict(method=name, evaluation=label, coverage=np.mean([x["cov"] for x in res]),
                             coverage_std=np.std([x["cov"] for x in res]), psnr=np.mean([x["psnr"] for x in res]),
                             J_D=np.mean([x["J_D"] for x in res]), J_B=np.mean([x["J_B"] for x in res]),
                             feasible_pct=np.mean([x["feasible"] for x in res]) * 100))
        print(f"done {name}")

    df = pd.DataFrame(rows)
    # PPO: also the mean over its 5 seeds
    for label in df["evaluation"].unique():
        d = df[(df.evaluation == label) & df.method.str.startswith("PPO best")]
        if len(d):
            rows.append(dict(method="PPO best models (mean of 5 seeds)", evaluation=label, coverage=d.coverage.mean(),
                             coverage_std=d.coverage.std(), psnr=d.psnr.mean(), J_D=d.J_D.mean(), J_B=d.J_B.mean(),
                             feasible_pct=d.feasible_pct.mean()))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/random_and_robustness.csv", index=False)
    with pd.option_context("display.width", 230, "display.max_columns", 12, "display.max_colwidth", 48,
                           "display.float_format", "{:.3f}".format):
        print(df[~df.method.str.startswith("PPO best model seed")])
    print(f"saved {OUT}/random_and_robustness.csv")


if __name__ == "__main__":
    main()
