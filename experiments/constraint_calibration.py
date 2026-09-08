"""
constraint_calibration.py - re-derive D_BAR/P_BAR/B_BAR under the
literal eqs. 6-8 discounted-return definition (J_D/J_P/J_B), replacing
the Stage-6 values that were calibrated against a windowed running-MEAN
of per-step costs - a different quantity (see other/decision_log.md
Stage 7 #1-6).

Measures the FULL, exact J_D/J_P/J_B (and J_Q, the objective analog) -
not the step-0 shortcut - under the same three reference tile-policies
used by experiments/el_selection_test.py, crossed with min/mid/max
power, run ONLY on the 9 TRAINING traces (config.EVAL_TRACE_INDICES are
never touched here - held-out traces must stay unseen by any
training-configuration decision, not just training itself), with fixed,
documented seeds so this is exactly reproducible.

Proposes feasible RANGES/sweep candidates, not single convenient
numbers, per the professor's original "sweep several values and assess
impact" guidance - a human confirms before config.py is touched.

Run: .venv\\Scripts\\python.exe experiments\\constraint_calibration.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from streaming_rl import data_loader
from streaming_rl.environment import TileStreamingEnv

POWER_LEVEL_LABELS = {0: "min (0%)", 2: "mid (50%)", 4: "max (100%)"}

# Fixed, reused seed per TRAINING trace - not fresh-random - so this
# script's output is exactly reproducible run to run. Arbitrary values,
# deliberately disjoint from config.EVAL_SEEDS (1000-1002) to avoid any
# confusion between calibration and evaluation runs.
CALIBRATION_SEED_BASE = 3000


def make_action(tile_mask: np.ndarray, power_level_idx: int) -> np.ndarray:
    return np.concatenate([tile_mask.astype(np.int64), [power_level_idx]])


def policy_action(policy: str, power_level_idx: int, rng: np.random.Generator) -> np.ndarray:
    if policy == "mandatory_only":
        tile_mask = np.zeros(config.N_TILES, dtype=bool)
    elif policy == "mandatory_random50":
        tile_mask = rng.random(config.N_TILES) < 0.5
    elif policy == "mandatory_all":
        tile_mask = np.ones(config.N_TILES, dtype=bool)
    else:
        raise ValueError(policy)
    return make_action(tile_mask, power_level_idx)


def run_calibration_episode(env: TileStreamingEnv, seed: int, policy: str, power_level_idx: int) -> dict:
    action_rng = np.random.default_rng(hash((seed, policy, power_level_idx)) % (2**32))
    obs, _info = env.reset(seed=seed)
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        action = policy_action(policy, power_level_idx, action_rng)
        obs, _reward, terminated, truncated, info = env.step(action)
    return {"J_Q": info["J_Q"], "J_D": info["J_D"], "J_P": info["J_P"], "J_B": info["J_B"]}


def main():
    # Loaded once, shared across every per-trace env below (rd_data is
    # read-only after load, ~388MB - avoid reloading it 9 times).
    bundle = data_loader.load_training_video_bundle()
    n_traces = len(bundle["valid_traces"])
    train_indices = [i for i in range(n_traces) if i not in config.EVAL_TRACE_INDICES]
    print(f"Training traces (positions, excluding EVAL_TRACE_INDICES={config.EVAL_TRACE_INDICES}): "
          f"{train_indices}")
    print(f"lambda = {config.DISCOUNT_FACTOR_LAMBDA}  (config.DISCOUNT_FACTOR_LAMBDA)")
    print()

    # One standalone env per training trace, exactly as HeldOutTraceEvalCallback
    # does for the held-out ones - forces deterministic trace selection.
    train_envs = {idx: TileStreamingEnv(trace_indices=[idx], bundle=bundle) for idx in train_indices}

    rows = []
    for policy in ("mandatory_only", "mandatory_random50", "mandatory_all"):
        for power_idx in (0, 2, 4):
            episode_results = [
                run_calibration_episode(env, seed=CALIBRATION_SEED_BASE + trace_idx, policy=policy,
                                         power_level_idx=power_idx)
                for trace_idx, env in train_envs.items()
            ]
            for key in ("J_Q", "J_D", "J_P", "J_B"):
                values = [r[key] for r in episode_results]
                rows.append({
                    "policy": policy, "power": POWER_LEVEL_LABELS[power_idx], "power_idx": power_idx,
                    "metric": key, "mean": float(np.mean(values)), "min": float(np.min(values)),
                    "max": float(np.max(values)),
                })

    print(f"{'policy':>19} {'power':>9} {'metric':>5} {'mean':>8} {'min':>8} {'max':>8}")
    for r in rows:
        print(f"{r['policy']:>19} {r['power']:>9} {r['metric']:>5} "
              f"{r['mean']:>8.4f} {r['min']:>8.4f} {r['max']:>8.4f}")

    def get(policy, power_idx, metric):
        return next(r for r in rows if r["policy"] == policy and r["power_idx"] == power_idx
                    and r["metric"] == metric)

    print()
    print("--- Proposed ranges (max power: mandatory_only = efficient, mandatory_all = wasteful) ---")
    for metric, budget_name in (("J_D", "D_BAR"), ("J_P", "P_BAR"), ("J_B", "B_BAR")):
        efficient = get("mandatory_only", 4, metric)
        wasteful = get("mandatory_all", 4, metric)
        lo, hi = efficient["mean"], wasteful["mean"]
        if lo > hi:
            lo, hi = hi, lo
        print(f"{budget_name}: efficient policy mean {metric}={efficient['mean']:.4f} "
              f"(range [{efficient['min']:.4f}, {efficient['max']:.4f}]), "
              f"wasteful policy mean {metric}={wasteful['mean']:.4f} "
              f"(range [{wasteful['min']:.4f}, {wasteful['max']:.4f}]) "
              f"-> candidate sweep range roughly [{lo:.4f}, {hi:.4f}]")

    print()
    print("These are candidate RANGES for D_BAR_CANDIDATES/P_BAR_CANDIDATES/B_BAR_CANDIDATES "
          "(config.py) - report to the user for confirmation before picking concrete values.")


if __name__ == "__main__":
    main()
