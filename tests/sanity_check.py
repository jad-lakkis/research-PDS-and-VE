"""
sanity_check.py - run TileStreamingEnv with a random policy and report
whether the numbers behave sensibly, before trusting the environment
for actual training. Same discipline used for explore_data.ipynb:
verify real, executed output, don't just trust that the code "should"
work.

Also stands as a standing regression check for eq. 2's mandatory
predicted-viewport enhancement (streaming_rl/environment.py,
viewport.mandatory_tile_mask): every step's n_mandatory_tiles must fall
in [B_BAR_MIN_TILES, B_BAR_MAX_TILES], the geometric bound derived in
config.py. A violation here would mean the mandatory-mask geometry is
wrong, independent of whatever policy is running.

Run: .venv\\Scripts\\python.exe tests\\sanity_check.py
"""

import sys
from pathlib import Path

import numpy as np

# This file lives in tests/, one level below the project root where
# config.py and streaming_rl/ live - put the root on sys.path so
# `import config` resolves when running `python tests/sanity_check.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from streaming_rl.environment import TileStreamingEnv


def run_episode(env: TileStreamingEnv, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    coverages, stalls, n_enhanced, n_mandatory, n_extra, buffer_trace = [], [], [], [], [], [obs[0]]
    steps = 0

    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        coverages.append(step_info["coverage"])
        stalls.append(step_info["D_t"])
        n_enhanced.append(step_info["n_enhanced_tiles"])
        n_mandatory.append(step_info["n_mandatory_tiles"])
        n_extra.append(step_info["n_agent_extra_tiles"])
        buffer_trace.append(obs[0])
        steps += 1

        assert config.B_BAR_MIN_TILES <= step_info["n_mandatory_tiles"] <= config.B_BAR_MAX_TILES, (
            f"n_mandatory_tiles={step_info['n_mandatory_tiles']} outside the geometric "
            f"bound [{config.B_BAR_MIN_TILES}, {config.B_BAR_MAX_TILES}] - the mandatory-"
            f"viewport-tile mask geometry (viewport.mandatory_tile_mask) looks wrong"
        )

    return {
        "steps": steps,
        "total_reward": total_reward,
        "mean_reward": total_reward / steps,
        "mean_coverage": float(np.mean(coverages)),
        "n_stalls": int(np.sum(np.array(stalls) > 0)),
        "max_stall_sec": float(np.max(stalls)),
        "mean_n_enhanced": float(np.mean(n_enhanced)),
        "mean_n_mandatory": float(np.mean(n_mandatory)),
        "mean_n_agent_extra": float(np.mean(n_extra)),
        "max_buffer": float(np.max(buffer_trace)),
        "min_buffer": float(np.min(buffer_trace)),
        "any_negative_buffer": bool(np.any(np.array(buffer_trace) < 0)),
        "any_nan_or_inf": bool(not np.all(np.isfinite(buffer_trace + coverages + stalls))),
    }


def main():
    env = TileStreamingEnv()
    print(f"Environment: {len(env._valid_traces)} valid traces for "
          f"{config.TRAINING_VIDEO_NAME} (array_index={config.TRAINING_VIDEO_ARRAY_INDEX})")
    print(f"observation_space: {env.observation_space}")
    print(f"action_space: {env.action_space}")
    print()

    n_episodes = 5
    results = [run_episode(env, seed=i) for i in range(n_episodes)]

    print(f"{'ep':>3} {'steps':>6} {'total_r':>10} {'mean_r':>8} {'mean_cov':>9} "
          f"{'n_stalls':>9} {'max_stall':>10} {'mean_enh':>9} {'mean_mand':>10} "
          f"{'mean_extra':>11} {'max_buf':>14} {'neg_buf?':>9} {'nan/inf?':>9}")
    for i, r in enumerate(results):
        print(f"{i:>3} {r['steps']:>6} {r['total_reward']:>10.2f} {r['mean_reward']:>8.3f} "
              f"{r['mean_coverage']:>9.3f} {r['n_stalls']:>9} {r['max_stall_sec']:>10.4f} "
              f"{r['mean_n_enhanced']:>9.1f} {r['mean_n_mandatory']:>10.1f} "
              f"{r['mean_n_agent_extra']:>11.1f} {r['max_buffer']:>14,.0f} "
              f"{str(r['any_negative_buffer']):>9} {str(r['any_nan_or_inf']):>9}")

    print()
    all_steps = [r["steps"] for r in results]
    any_bad = any(r["any_negative_buffer"] or r["any_nan_or_inf"] for r in results)
    print(f"Episode lengths: {all_steps}  (Runner has 1080 frames / "
          f"{config.GOP_SIZE_FRAMES} = 36 GOPs -> expect every episode to run exactly 36 steps)")
    print(f"All episode lengths == 36: {all(s == 36 for s in all_steps)}")
    print(f"Any negative buffer or NaN/inf seen: {any_bad}  (must be False to trust this environment)")
    print(f"All n_mandatory_tiles in [{config.B_BAR_MIN_TILES}, {config.B_BAR_MAX_TILES}]: "
          f"True (would have raised an assertion above otherwise)")
    print(f"Mean coverage under a fully RANDOM policy: "
          f"{np.mean([r['mean_coverage'] for r in results]):.3f}  "
          f"(includes the mandatory predicted-viewport tiles plus ~32/64 random extra "
          f"tiles each step, so this is expected to be noticeably higher than pre-eq.2 "
          f"runs - not a regression, see config.py's B_BAR history)")


if __name__ == "__main__":
    main()
