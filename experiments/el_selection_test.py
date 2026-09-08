"""
el_selection_test.py - decide the enhancement level (EL) empirically.

Compares EL = array index 0 ("best quality", k=6) vs. EL = array index 1
("one step below best", k=5) - the base layer stays array index 6 (the
worst quality) either way, per config.qp_array_index_for_enhancement_level.
Run *after* eq. 2's mandatory predicted-viewport enhancement
(streaming_rl/environment.py) lands, since that changes the underlying
stall/coverage dynamics enough that pre-eq.2 numbers wouldn't be
representative of the environment this EL will actually run in.

For each candidate EL, three reference tile-policies are crossed with
three power levels (min/mid/max of the 5 discrete levels):
  - mandatory_only:    the agent adds nothing beyond the forced
                        predicted-viewport tiles - a well-defined lower
                        bound now that eq. 2 makes "zero tiles enhanced"
                        impossible.
  - mandatory_random50: the agent additionally enhances each remaining
                        tile independently with probability 0.5.
  - mandatory_all:      the agent enhances every tile (maximally
                        wasteful reference).

Decision rule (stated before running, not after): prefer the EL where,
at max power, the mandatory_only (efficient) policy stalls rarely while
mandatory_all (wasteful) stalls meaningfully more often - i.e. the
constraint actually binds enough to distinguish good from bad policies,
without binding so hard that both fail regardless of policy. The full
table is reported either way; this script only recommends, the
resulting INITIAL_ENHANCEMENT_LEVELS change in config.py needs
confirmation before it's applied.

This test's stall-rate numbers also ground config.py's provisional
D_BAR (see that file's comment): set a touch above the efficient
policy's stall rate and a touch below the wasteful policy's.

Run: .venv\\Scripts\\python.exe experiments\\el_selection_test.py
"""

import sys
from pathlib import Path

import numpy as np

# This file lives in experiments/, one level below the project root
# where config.py and streaming_rl/ live.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from streaming_rl.environment import TileStreamingEnv

CANDIDATE_ELS = {
    "array_idx_0 (best quality, k=6)": 6,
    "array_idx_1 (one below best, k=5)": 5,
}

POWER_LEVEL_LABELS = {0: "min (0%)", 2: "mid (50%)", 4: "max (100%)"}
N_EPISODES = 12          # matches the 12 valid Runner traces
RUNAWAY_BUFFER_BITS = 1e9  # flag if the buffer grows past this - "too cheap, nothing to learn"


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


def run_combo(enhanced_level: int, policy: str, power_level_idx: int) -> dict:
    env = TileStreamingEnv(enhanced_level=enhanced_level)
    action_rng = np.random.default_rng(hash((enhanced_level, policy, power_level_idx)) % (2**32))

    stall_flags, stall_durations, coverages, episode_rewards, max_buffers = [], [], [], [], []
    for seed in range(N_EPISODES):
        obs, info = env.reset(seed=seed)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy_action(policy, power_level_idx, action_rng)
            obs, reward, terminated, truncated, step_info = env.step(action)
            total_reward += reward
            stall_flags.append(step_info["D_t"] > 0.0)
            stall_durations.append(step_info["D_t"])
            coverages.append(step_info["coverage"])
        episode_rewards.append(total_reward)
        max_buffers.append(obs[0])

    return {
        "enhanced_level": enhanced_level,
        "policy": policy,
        "power_level_idx": power_level_idx,
        "stall_frequency": float(np.mean(stall_flags)),
        "mean_stall_duration": float(np.mean(stall_durations)),
        "mean_coverage": float(np.mean(coverages)),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "max_buffer": float(np.max(max_buffers)),
        "runaway_buffer": bool(np.max(max_buffers) > RUNAWAY_BUFFER_BITS),
    }


def main():
    results = []
    for el_label, el_k in CANDIDATE_ELS.items():
        for policy in ("mandatory_only", "mandatory_random50", "mandatory_all"):
            for power_idx in (0, 2, 4):
                results.append(run_combo(el_k, policy, power_idx))

    print(f"{'EL (k)':>6} {'policy':>19} {'power':>9} {'stall_freq':>10} "
          f"{'mean_stall':>10} {'mean_cov':>9} {'mean_ep_r':>10} {'max_buf':>14} {'runaway?':>9}")
    for r in results:
        print(f"{r['enhanced_level']:>6} {r['policy']:>19} "
              f"{POWER_LEVEL_LABELS[r['power_level_idx']]:>9} {r['stall_frequency']:>10.3f} "
              f"{r['mean_stall_duration']:>10.4f} {r['mean_coverage']:>9.3f} "
              f"{r['mean_episode_reward']:>10.2f} {r['max_buffer']:>14,.0f} "
              f"{str(r['runaway_buffer']):>9}")

    print()
    print("--- Decision (max power, mandatory_only vs. mandatory_all) ---")
    by_key = {(r["enhanced_level"], r["policy"], r["power_level_idx"]): r for r in results}
    scored = []
    for el_label, el_k in CANDIDATE_ELS.items():
        efficient = by_key[(el_k, "mandatory_only", 4)]
        wasteful = by_key[(el_k, "mandatory_all", 4)]
        gap = wasteful["stall_frequency"] - efficient["stall_frequency"]
        scored.append((el_label, el_k, efficient, wasteful, gap))
        print(f"k={el_k:>2} ({el_label}): mandatory_only stall_freq={efficient['stall_frequency']:.3f}, "
              f"mandatory_all stall_freq={wasteful['stall_frequency']:.3f}, gap={gap:.3f}")

    print()
    viable = [s for s in scored if s[2]["stall_frequency"] < 0.5]
    if viable:
        best = max(viable, key=lambda s: s[4])
        print(f"RECOMMENDATION: k={best[1]} ({best[0]}) - the efficient (mandatory_only) policy "
              f"stalls rarely (stall_freq={best[2]['stall_frequency']:.3f}) while the wasteful "
              f"(mandatory_all) policy stalls meaningfully more often "
              f"(stall_freq={best[3]['stall_frequency']:.3f}), the largest such gap among "
              f"candidates whose efficient policy isn't already failing.")
    else:
        print("RECOMMENDATION: none of the candidates have a stall_freq<0.5 mandatory_only "
              "policy at max power - even the efficient reference policy is struggling. "
              "Needs a closer look before picking either candidate.")
    print()
    print("This is a recommendation, not an automatic config.py change - confirm before applying.")


if __name__ == "__main__":
    main()
