"""
plot_episode_trace.py - qualitative per-step trace over a few consecutive
episodes on a held-out eval trace, in the style of ASL360's Fig. 4
(buffer level + corresponding action, aligned, last 3 episodes). This is
the one figure type from the reference papers that shows mechanism (what
the converged policy actually does, step by step) rather than another
aggregate number - also the direct answer to decision_log.md #30's "show
stall behavior" ask.

Supports overlaying multiple runs (e.g. PPO vs PPO+PDS) on the same 3
panels, same method-color convention as plot_tradeoff_frontier.py /
plot_grouped_comparison.py, so all three paper-style figures read as one
consistent set.

Reuses the project's own eval pattern (streaming_rl/eval.py): a
standalone TileStreamingEnv stepped directly, observations normalized
via VecNormalize.normalize_obs() (read-only, never mutates training
stats) rather than driving eval through the training VecEnv stack.
Episode length is fixed at exactly 36 steps (never truncated - decision
log #1), so n_episodes*36 steps line up exactly across runs with no
resampling needed.

Usage: python plot_episode_trace.py <out_dir> [n_episodes] [trace_position_index]
  Edit RUNS below (label, model_dir) - each model_dir needs either
  model.zip+vecnormalize.pkl, or best_model.zip+best_vecnormalize.pkl
  (both checked automatically).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import config
from streaming_rl.environment import TileStreamingEnv

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
METHOD_COLOR = {"PPO": "#2a78d6", "PPO+PDS": "#eb6834"}
STALL_COLOR = "#e34948"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
})

# One trace per budget arm - (budget label, out subdir, PPO model_dir,
# PPO+PDS model_dir). Both methods use the SAME held-out trace/budget
# per entry, so behavior differences reflect the algorithm, not a
# different problem. clip100 dropped, not part of the official
# comparison. PPO checkpoints below are each arm's LATEST block that
# still has a saved model.zip (some final blocks save progress.csv but
# not a fresh checkpoint) - see conversation history for how each was
# picked.
BUDGET_ARMS = [
    ("0.3_6.5_8.0",
     "runpod_results/dbar_0.3_pbar_6.5_bbar_8.0_seed0/block_v10",
     "run_D_clean_results/run_D_clean_dbar_0.3_pbar_6.5_bbar_8.0_seed0/best"),
    ("0.12_4_7.4",
     "runpod_results/dbar_0.12_pbar_4_bbar_7.4_seed0/block_v7",
     "run_D_clean_results/run_D_clean_dbar_0.12_pbar_4_bbar_7.4_seed0/best"),
    ("0.12_4_8.0",
     "runpod_results_v2/v3/dbar_0.12_pbar_4_bbar_8.0_seed0/block_v9",
     "run_D_clean_results/run_D_clean_dbar_0.12_pbar_4_bbar_8.0_seed0/best"),
    ("0.1_6.5_8.0",
     "runpod_results/dbar_0.1_pbar_6.5_bbar_8.0_seed0/block_v6",
     "run_D_clean_results/run_D_clean_dbar_0.1_pbar_6.5_bbar_8.0_seed0/best"),
]


def _find_checkpoint(model_dir: str):
    for model_name, vecnorm_name in [("model", "vecnormalize.pkl"), ("best_model", "best_vecnormalize.pkl")]:
        model_path = os.path.join(model_dir, model_name)
        vecnorm_path = os.path.join(model_dir, vecnorm_name)
        if os.path.exists(model_path + ".zip"):
            return model_path, (vecnorm_path if os.path.exists(vecnorm_path) else None)
    raise FileNotFoundError(f"No model.zip or best_model.zip found under {model_dir}")


def collect_trace(model_dir: str, n_episodes: int, trace_idx_pos: int):
    trace_position = config.EVAL_TRACE_INDICES[trace_idx_pos]
    base_seed = config.EVAL_SEEDS[trace_idx_pos]
    model_path, vecnorm_path = _find_checkpoint(model_dir)

    env = TileStreamingEnv(trace_indices=[trace_position])
    model = PPO.load(model_path, device="cpu")
    if vecnorm_path is not None:
        # VecNormalize.load() needs a VecEnv to wrap - this throwaway
        # DummyVecEnv is only there to satisfy that API; its own
        # reset()/step() are never called, only the loaded obs_rms via
        # .normalize_obs() below (read-only, same as streaming_rl/eval.py).
        dummy_venv = DummyVecEnv([lambda: TileStreamingEnv(trace_indices=[trace_position])])
        vecnorm = VecNormalize.load(vecnorm_path, dummy_venv)
    else:
        vecnorm = None

    rows = []
    for ep in range(n_episodes):
        obs, _info = env.reset(seed=base_seed + ep)
        step = 0
        terminated = truncated = False
        while not (terminated or truncated):
            norm_obs = vecnorm.normalize_obs(obs) if vecnorm is not None else obs
            action, _state = model.predict(norm_obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            rows.append({
                "episode": ep, "step": step, "Z": info["Z"], "D_t": info["D_t"],
                "n_enhanced_tiles": info["n_enhanced_tiles"], "power_mw": info["power_watts"] * 1000.0,
                "coverage": info["coverage"],
            })
            step += 1
    return rows, trace_position


def generate(runs: list, out_dir: str, n_episodes: int = 3, trace_idx_pos: int = 0):
    """runs: list of (label, model_dir), e.g. [("PPO", dir1), ("PPO+PDS", dir2)]."""
    os.makedirs(out_dir, exist_ok=True)

    traces = {}
    trace_position = None
    for label, model_dir in runs:
        rows, trace_position = collect_trace(model_dir, n_episodes, trace_idx_pos)
        traces[label] = rows
        stall_n = sum(1 for r in rows if r["D_t"] > 0)
        print(f"{label}: {len(rows)} steps, {stall_n} stall steps, "
              f"mean coverage={np.mean([r['coverage'] for r in rows]):.3f}, "
              f"mean tiles={np.mean([r['n_enhanced_tiles'] for r in rows]):.2f}, "
              f"mean power_mw={np.mean([r['power_mw'] for r in rows]):.2f}")

    any_rows = next(iter(traces.values()))
    x = list(range(len(any_rows)))
    ep_boundaries = [i for i in range(1, len(any_rows)) if any_rows[i]["episode"] != any_rows[i - 1]["episode"]]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    fig.suptitle(f"Policy behavior, {n_episodes} consecutive episodes on the same held-out trace "
                 f"(position {trace_position})", fontsize=11, color=INK)

    for label, rows in traces.items():
        color = METHOD_COLOR.get(label, INK2)
        Z = [r["Z"] for r in rows]
        tiles = [r["n_enhanced_tiles"] for r in rows]
        power = [r["power_mw"] for r in rows]
        stall_x = [i for i, r in enumerate(rows) if r["D_t"] > 0]

        axes[0].plot(x, Z, color=color, lw=1.6, label=label, alpha=0.9)
        axes[1].plot(x, tiles, color=color, lw=1.4, label=label, alpha=0.9)
        if stall_x:
            axes[1].scatter(stall_x, [tiles[i] for i in stall_x], color=STALL_COLOR, s=32, zorder=5,
                            marker="x" if label != runs[0][0] else "o")
        axes[2].plot(x, power, color=color, lw=1.4, label=label, alpha=0.9)

    # Z is a DATA buffer (bits), not a time buffer - Z_next = Z + T0*R_t -
    # A_t with R_t a bitrate and A_t a bit-volume (environment.py step()),
    # despite reading as a playout-seconds buffer in the MMSP'25 paper's
    # own notation.
    axes[0].set_ylabel("Buffer level $Z$ (bits)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    axes[1].set_ylabel("Enhanced tiles (of 64)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right",
                   title="line = tiles; o/x = stall (D_t>0)", title_fontsize=7.5)
    axes[2].set_ylabel("Transmit power (mW)")
    axes[2].set_xlabel("Step (concatenated across episodes)")

    for ax in axes:
        for b in ep_boundaries:
            ax.axvline(b, color=AXIS, lw=1.0, ls="--", alpha=0.7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(out_dir, "episode_trace.png")
    fig.savefig(out, dpi=140)
    print(f"Wrote {out}")


if __name__ == "__main__":
    # <base_out_dir>/<budget>/episode_trace.png per arm in BUDGET_ARMS,
    # so each run's trace sits alongside that run's overlay panels.
    base_out_dir = sys.argv[1] if len(sys.argv) > 1 else "graphs_trace"
    n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    trace_idx_pos = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    for budget_label, ppo_dir, pds_dir in BUDGET_ARMS:
        print(f"\n=== {budget_label} ===")
        generate([("PPO", ppo_dir), ("PPO+PDS", pds_dir)],
                 os.path.join(base_out_dir, budget_label), n_episodes, trace_idx_pos)
