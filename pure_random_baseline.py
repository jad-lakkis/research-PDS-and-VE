"""
pure_random_baseline.py - "all random tiles, no viewport" reference: k tiles
drawn uniformly from all 64 each segment, the forced predicted-viewport
tiles switched OFF for this baseline only (the environment's mandatory rule
is patched to an empty mask inside this script; nothing on disk changes).
Same held-out traces, channel seeds and scoring as the learned policies.
This measures what viewport prediction is worth - it is NOT a like-for-like
competitor to the learned policies, which always get the predicted tiles.

Usage: .venv/Scripts/python.exe pure_random_baseline.py
"""
import numpy as np
import pandas as pd

import config
from streaming_rl import data_loader, viewport
from streaming_rl.environment import TileStreamingEnv

OUT = "EXPERIMENT_X_drop_seed6/nonrl_baselines/robustness/pure_random_no_viewport.csv"
D_BAR, B_BAR = 0.3, 8.0
NT = config.N_TILES
REPEATS = 30


def evaluate(bundle, k, rng):
    res = []
    for pos, seed in zip(config.EVAL_TRACE_INDICES, config.EVAL_SEEDS):
        env = TileStreamingEnv(trace_indices=[pos], bundle=bundle)
        env.reset(seed=seed)
        cov, psnr, done = [], [], False
        while not done:
            mask = np.zeros(NT, np.int64)
            mask[rng.choice(NT, size=k, replace=False)] = 1
            _, _, te, tr, info = env.step(np.concatenate([mask, [4]]))  # 50 mW
            done = te or tr
            assert info["n_enhanced_tiles"] == k  # forced tiles really are off
            cov.append(info["coverage"]); psnr.append(info["viewport_psnr_db"])
        res.append(dict(J_Q=info["J_Q"], J_D=info["J_D"], J_B=info["J_B"], cov=np.mean(cov), psnr=np.mean(psnr)))
    m = {key: float(np.mean([r[key] for r in res])) for key in res[0]}
    m["feasible"] = m["J_D"] <= D_BAR and m["J_B"] <= B_BAR
    return m


def main():
    bundle = data_loader.load_training_video_bundle()
    original = viewport.mandatory_tile_mask
    viewport.mandatory_tile_mask = lambda th, ph: np.zeros(NT, dtype=bool)
    try:
        rows = []
        for k in [0, 4, 8, 12, 14, 16, 17, 20, 24, 32]:
            reps = [evaluate(bundle, k, np.random.default_rng(300 + r)) for r in range(REPEATS)]
            rows.append(dict(method=f"Pure random: {k} tiles anywhere (no viewport)", tiles_per_segment=k,
                             coverage=np.mean([x["cov"] for x in reps]), coverage_std=np.std([x["cov"] for x in reps]),
                             psnr=np.mean([x["psnr"] for x in reps]), J_D=np.mean([x["J_D"] for x in reps]),
                             J_B=np.mean([x["J_B"] for x in reps]), feasible_pct=np.mean([x["feasible"] for x in reps]) * 100))
    finally:
        viewport.mandatory_tile_mask = original
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 12, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
