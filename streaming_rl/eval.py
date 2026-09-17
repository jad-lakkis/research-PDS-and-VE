"""
eval.py - deterministic, constraint-aware evaluation on the held-out
traces (config.EVAL_TRACE_INDICES / config.EVAL_SEEDS).

Bypasses SB3's VecEnv layer entirely for evaluation: VecEnv.reset() in
the installed stable_baselines3 has no per-call seed/options support
(older-style API), and driving eval through the *training* VecNormalize/
DummyVecEnv stack out-of-band would desync OnPolicyAlgorithm's cached
_last_obs between rollouts. Instead each held-out trace gets its own
standalone TileStreamingEnv instance, stepped directly; the training
run's learned observation normalization is applied manually via
VecNormalize.normalize_obs() (confirmed stateless/read-only - never
calls .update(), so this can't leak eval statistics into training
stats) before every model.predict() call.

Model selection (eqs. 5-8, directly - see other/decision_log.md #10):
a checkpoint is feasible iff J_D<=D_BAR and J_P<=P_BAR and J_B<=B_BAR
(small float tolerance - fixed seeds make repeat evals of the same
checkpoint exactly reproducible, so this is not a noise tolerance).
Among feasible checkpoints seen so far this run, keep the one with the
highest J_Q. If none are feasible yet, keep the one with the smallest
total normalized violation, J_Q as tiebreak. Mean Lagrangian reward is
never used for selection - mu_D/mu_P/mu_B change during training, so
it isn't comparable across checkpoints.
"""

import os
from dataclasses import dataclass

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

import config
from streaming_rl.environment import TileStreamingEnv


@dataclass
class EpisodeResult:
    trace_position: int
    seed: int
    steps: int
    mean_coverage: float
    mean_stall_sec: float
    mean_power_mw: float
    mean_n_tiles: float
    mean_psnr_db: float
    J_Q: float
    J_D: float
    J_P: float
    J_B: float


def run_eval_episode(env: TileStreamingEnv, model, vecnormalize, seed: int,
                      trace_position: int) -> EpisodeResult:
    """Play one full, deterministic episode on `env` (already restricted
    to a single held-out trace) and return its summary. `vecnormalize`
    may be None (no normalization applied - e.g. before Part H's
    VecNormalize wiring lands, or when training with observations left
    raw); when present, only its read-only normalize_obs() is used."""
    obs, _info = env.reset(seed=seed)
    coverages, stalls_sec, powers_mw, n_tiles, psnr_db = [], [], [], [], []
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        norm_obs = vecnormalize.normalize_obs(obs) if vecnormalize is not None else obs
        action, _state = model.predict(norm_obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        coverages.append(info["coverage"])
        stalls_sec.append(info["D_t"])
        powers_mw.append(info["power_watts"] * 1000.0)
        n_tiles.append(info["n_enhanced_tiles"])
        psnr_db.append(info["viewport_psnr_db"])

    for key in ("J_Q", "J_D", "J_P", "J_B"):
        assert key in info, (
            f"terminal info dict is missing '{key}' - environment.py's discounted-return "
            f"accumulator hasn't landed, or uses a different key name"
        )

    return EpisodeResult(
        trace_position=trace_position, seed=seed, steps=len(coverages),
        mean_coverage=float(np.mean(coverages)), mean_stall_sec=float(np.mean(stalls_sec)),
        mean_power_mw=float(np.mean(powers_mw)), mean_n_tiles=float(np.mean(n_tiles)),
        mean_psnr_db=float(np.mean(psnr_db)),
        J_Q=float(info["J_Q"]), J_D=float(info["J_D"]), J_P=float(info["J_P"]), J_B=float(info["J_B"]),
    )


def _checkpoint_key(feasible: bool, violation: float, J_Q: float) -> tuple:
    """Smaller = better. (0, ...) always sorts below (1, ...), so
    feasible checkpoints always beat infeasible ones with no extra flag
    needed; ties within a tier break on -J_Q (higher J_Q = smaller key)."""
    return (0, -J_Q, 0.0) if feasible else (1, violation, -J_Q)


class HeldOutTraceEvalCallback(BaseCallback):
    """Top-level callback (NOT nested in EveryNTimesteps - EventCallback
    never forwards on_rollout_end to a wrapped child, confirmed against
    the installed stable_baselines3 source). Fires once per PPO rollout,
    same as LagrangianMultiplierCallback."""

    def __init__(self, eval_trace_indices=config.EVAL_TRACE_INDICES,
                 eval_seeds=config.EVAL_SEEDS, d_bar: float = config.D_BAR,
                 p_bar: float = config.P_BAR, b_bar: float = config.B_BAR,
                 best_model_save_path: str = "runs/best", eval_every_n_rollouts: int = 1,
                 feasibility_tol: float = 1e-9, verbose: int = 0):
        super().__init__(verbose)
        assert len(eval_trace_indices) == len(eval_seeds), (
            "eval_trace_indices and eval_seeds must be the same length (one seed per held-out trace)"
        )
        self.eval_trace_indices = list(eval_trace_indices)
        self.eval_seeds = list(eval_seeds)
        self.d_bar, self.p_bar, self.b_bar = d_bar, p_bar, b_bar
        self.best_model_save_path = best_model_save_path
        self.eval_every_n_rollouts = eval_every_n_rollouts
        self.feasibility_tol = feasibility_tol
        self._rollout_count = 0
        self._eval_envs = None
        self.best_key = (1, float("inf"), float("inf"))
        self.best_checkpoint_step = None

    def _init_callback(self) -> None:
        os.makedirs(self.best_model_save_path, exist_ok=True)
        # Built ONCE - data_loader.load_training_video_bundle() is
        # uncached and reloads rd.mat (78MB)/hn.mat (4MB) from disk on
        # every TileStreamingEnv construction otherwise.
        self._eval_envs = [TileStreamingEnv(trace_indices=[idx]) for idx in self.eval_trace_indices]

    def _on_step(self) -> bool:
        return True   # all logic runs in _on_rollout_end; abstract method must exist

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count % self.eval_every_n_rollouts != 0:
            return

        vecnorm: VecNormalize = self.model.get_vec_normalize_env()
        results = [
            run_eval_episode(env, self.model, vecnorm, seed=self.eval_seeds[i],
                              trace_position=self.eval_trace_indices[i])
            for i, env in enumerate(self._eval_envs)
        ]

        J_Q = float(np.mean([r.J_Q for r in results]))
        J_D = float(np.mean([r.J_D for r in results]))
        J_P = float(np.mean([r.J_P for r in results]))
        J_B = float(np.mean([r.J_B for r in results]))
        violation = (max(0.0, J_D - self.d_bar) + max(0.0, J_P - self.p_bar)
                     + max(0.0, J_B - self.b_bar))
        feasible = violation <= self.feasibility_tol

        for name, val in [
            ("J_Q", J_Q), ("J_D", J_D), ("J_P", J_P), ("J_B", J_B),
            ("violation", violation), ("feasible", float(feasible)),
            ("mean_coverage", float(np.mean([r.mean_coverage for r in results]))),
            ("mean_stall_sec", float(np.mean([r.mean_stall_sec for r in results]))),
            ("mean_power_mw", float(np.mean([r.mean_power_mw for r in results]))),
            ("mean_n_tiles", float(np.mean([r.mean_n_tiles for r in results]))),
            ("mean_psnr_db", float(np.mean([r.mean_psnr_db for r in results]))),
        ]:
            self.logger.record(f"eval/{name}", val)

        new_key = _checkpoint_key(feasible, violation, J_Q)
        if new_key < self.best_key:
            self.best_key = new_key
            self.best_checkpoint_step = self.num_timesteps
            self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            if vecnorm is not None:
                vecnorm.save(os.path.join(self.best_model_save_path, "best_vecnormalize.pkl"))
            if self.verbose >= 1:
                print(f"New best checkpoint @ {self.num_timesteps}: "
                      f"feasible={feasible} J_Q={J_Q:.4f} violation={violation:.4f}")
