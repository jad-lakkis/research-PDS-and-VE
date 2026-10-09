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

Lumos5G link (link="lumos5g"): held-out viewers x config.LUMOS5G_EVAL_WINDOWS
fixed validation-run windows. Outage stall that no policy can avoid
differs between windows and between splits, so the stall budget is judged
on the SAME episodes as "predicted viewport only" (computed once, at
init): an episode's stall budget is its own viewport-only J_D plus the
stall allowance, and the eval's D_bar is the mean of those.
"""

import csv
import os
from dataclasses import dataclass
from itertools import product

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

import config
from streaming_rl import lumos5g
from streaming_rl.environment import TileStreamingEnv


def encode_mask_hex(mask: np.ndarray) -> str:
    """Pack a 64-element boolean tile mask into 16 hex chars.

    Paired with decode_mask_hex() below - they are kept together on
    purpose so nothing downstream has to guess the bit order. Flat tile
    index i maps to grid (i // config.TILE_GRID_COLS, i % TILE_GRID_COLS)
    (raster row-major, config.TILE_INDEX_ORDER).
    """
    return np.packbits(np.asarray(mask, dtype=bool), bitorder="little").tobytes().hex()


def decode_mask_hex(hex_str: str, n_tiles: int = config.N_TILES) -> np.ndarray:
    """Inverse of encode_mask_hex() - returns a boolean array of n_tiles."""
    packed = np.frombuffer(bytes.fromhex(hex_str), dtype=np.uint8)
    return np.unpackbits(packed, bitorder="little")[:n_tiles].astype(bool)


# Column order for <log_dir>/eval_steps.csv, written by EvalStepTraceWriter.
# The combined enhanced mask is deliberately NOT stored - it is exactly
# (agent_mask | mandatory_mask), recoverable via decode_mask_hex().
EVAL_STEP_COLUMNS = [
    "rollout", "total_timesteps", "trace_position", "seed", "step",
    "agent_mask_hex", "mandatory_mask_hex", "n_agent_extra_tiles", "n_enhanced_tiles",
    "pred_theta_deg", "pred_phi_deg", "actual_theta_deg", "actual_phi_deg",
    "coverage", "coverage_el_only", "psnr_db", "psnr_db_el_only",
    "power_watts", "D_t", "Z",
]


class EvalStepTraceWriter:
    """Per-eval-step spatial/geometry trace -> <log_dir>/eval_steps.csv.

    Deliberately a side CSV rather than extra progress.csv columns (same
    reasoning as train_ppo.py's TileEnhancementTrackerCallback: SB3's
    CSVOutputFormat rewrites the whole file whenever a new key appears).

    Opened in APPEND mode with the header written only when the file is
    missing/empty, so a --resume-from continuation extends the trace
    instead of truncating it - note TileEnhancementTrackerCallback uses
    "w" and does silently lose history across resumed blocks.

    Every field is read with info.get(..., nan): an env constructed
    without log_counterfactual simply leaves those two columns empty
    rather than raising partway through a long run.

    Rows are STAGED in memory during an eval and only written on
    commit(). Whether an eval is worth keeping depends on the checkpoint
    comparison, which isn't known until after all its episodes have run -
    so the caller stages first and decides after. One eval is 3x36=108
    rows, so buffering is negligible.
    """

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self._path = os.path.join(log_dir, "eval_steps.csv")
        write_header = (not os.path.exists(self._path)) or os.path.getsize(self._path) == 0
        self._fh = open(self._path, "a", newline="")
        self._writer = csv.writer(self._fh)
        if write_header:
            self._writer.writerow(EVAL_STEP_COLUMNS)
            self._fh.flush()
        self._staged = []

    def stage_step(self, rollout: int, total_timesteps: int, trace_position: int,
                   seed: int, step: int, info: dict) -> None:
        nan = float("nan")
        agent_mask = info.get("agent_tile_mask")
        mandatory_mask = info.get("mandatory_tile_mask")
        self._staged.append([
            rollout, total_timesteps, trace_position, seed, step,
            encode_mask_hex(agent_mask) if agent_mask is not None else "",
            encode_mask_hex(mandatory_mask) if mandatory_mask is not None else "",
            info.get("n_agent_extra_tiles", nan), info.get("n_enhanced_tiles", nan),
            info.get("predicted_theta", nan), info.get("predicted_phi", nan),
            info.get("actual_theta", nan), info.get("actual_phi", nan),
            info.get("coverage", nan), info.get("coverage_el_only", nan),
            info.get("viewport_psnr_db", nan), info.get("viewport_psnr_db_el_only", nan),
            info.get("power_watts", nan), info.get("D_t", nan), info.get("Z", nan),
        ])

    def commit(self) -> int:
        """Write everything staged since the last commit/discard. One
        flush per eval, not per row (these runs write to network FS)."""
        n = len(self._staged)
        if n:
            self._writer.writerows(self._staged)
            self._fh.flush()
            self._staged.clear()
        return n

    def discard(self) -> None:
        self._staged.clear()


# Column order for <log_dir>/eval_episodes.csv, written by
# EvalEpisodeWriter. One row per held-out eval episode (3/eval) - the
# per-episode J_D/J_P/J_B _on_rollout_end below already computes via
# EpisodeResult, but previously only logged as an already-averaged
# eval/J_D etc. This is what per-episode (rather than per-rollout)
# framing of the constraint/violation needs.
EVAL_EPISODE_COLUMNS = [
    "rollout", "total_timesteps", "trace_position", "seed",
    "J_D", "J_P", "J_B", "viol_D", "viol_P", "viol_B", "violation", "feasible",
]


class EvalEpisodeWriter:
    """Per-eval-episode J_D/J_P/J_B and derived violation ->
    <log_dir>/eval_episodes.csv.

    Unlike EvalStepTraceWriter, every eval's rows are kept unconditionally -
    3 rows/eval is negligible, and unlike the spatial trace there is no
    "is this eval worth it" question to defer. Same append-mode,
    header-written-only-if-missing/empty convention as EvalStepTraceWriter,
    for the same --resume-from reason.
    """

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self._path = os.path.join(log_dir, "eval_episodes.csv")
        write_header = (not os.path.exists(self._path)) or os.path.getsize(self._path) == 0
        self._fh = open(self._path, "a", newline="")
        self._writer = csv.writer(self._fh)
        if write_header:
            self._writer.writerow(EVAL_EPISODE_COLUMNS)
            self._fh.flush()

    def write_rows(self, rows: list) -> None:
        if rows:
            self._writer.writerows(rows)
            self._fh.flush()


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
                      trace_position: int, step_sink=None, options: dict = None) -> EpisodeResult:
    """Play one full, deterministic episode on `env` (already restricted
    to a single held-out trace) and return its summary. `vecnormalize`
    may be None (no normalization applied - e.g. before Part H's
    VecNormalize wiring lands, or when training with observations left
    raw); when present, only its read-only normalize_obs() is used.

    `step_sink`, when given, is called as step_sink(step_index, info)
    once per step - used by EvalStepTraceWriter to record the per-step
    spatial/geometry trace. Default None leaves behavior unchanged.

    `options` is passed to env.reset() - the Lumos5G link's fixed
    {"window": (run, start)}; None for the rician link."""
    obs, _info = env.reset(seed=seed, options=options)
    coverages, stalls_sec, powers_mw, n_tiles, psnr_db = [], [], [], [], []
    terminated = truncated = False
    info = {}
    step_index = 0
    while not (terminated or truncated):
        norm_obs = vecnormalize.normalize_obs(obs) if vecnormalize is not None else obs
        action, _state = model.predict(norm_obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        if step_sink is not None:
            step_sink(step_index, info)
        step_index += 1
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
                 feasibility_tol: float = 1e-9, log_dir: str = None,
                 spatial_log_every_n_rollouts: int = 150,
                 dropped_constraints: frozenset = frozenset(), verbose: int = 0,
                 link: str = "rician", stall_allowance: float = None, lumos_scale: float = 1.0,
                 lumos_data: dict = None, lumos_eval_windows: int = config.LUMOS5G_EVAL_WINDOWS,
                 tile_frame: str = "absolute", obs_forced_bits: bool = False):
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
        # A dropped constraint is excluded from BOTH the feasibility check
        # and the checkpoint-selection violation sum below - a run that
        # never had power in its reward shouldn't be judged infeasible
        # for using a lot of it. J_P/mean_power_mw etc. are still logged
        # normally (see run_eval_episode/EpisodeResult) so the dropped
        # constraint's actual behavior remains visible for comparison,
        # it just doesn't gate feasibility or best-checkpoint selection.
        self.dropped_constraints = frozenset(dropped_constraints)
        # Where eval_steps.csv goes. Falls back to the parent of
        # best_model_save_path (which is <log_dir>/best) so existing
        # callers that don't pass log_dir still land in the run's own dir.
        self.log_dir = log_dir if log_dir is not None else os.path.dirname(best_model_save_path)
        self.spatial_log_every_n_rollouts = spatial_log_every_n_rollouts
        self._rollout_count = 0
        self._eval_envs = None
        self._step_trace_writer = None
        self._episode_writer = None
        self.best_key = (1, float("inf"), float("inf"))
        self.best_checkpoint_step = None
        self.link = link
        if link == "lumos5g":
            assert stall_allowance is not None, "the Lumos5G link judges stall as viewport-only floor + stall_allowance"
        self.stall_allowance = stall_allowance
        self.lumos_scale = lumos_scale
        self.lumos_data = lumos_data
        self.lumos_eval_windows = lumos_eval_windows
        self.tile_frame = tile_frame   # must match the training env's (it defines the action and observation)
        self.obs_forced_bits = obs_forced_bits   # same: changes the observation
        # (env, seed, trace_position, reset options) per eval episode, and
        # each episode's own stall budget - built in _init_callback.
        self._episodes = None
        self._episode_d_bars = None
        self._stall_floor = None

    def _init_callback(self) -> None:
        os.makedirs(self.best_model_save_path, exist_ok=True)
        # Built ONCE - data_loader.load_training_video_bundle() is
        # uncached and reloads rd.mat (78MB)/hn.mat (4MB) from disk on
        # every TileStreamingEnv construction otherwise.
        #
        # log_counterfactual=True is set HERE and only here: these eval
        # envs run 108 steps per eval, so the extra per-step PSNR call is
        # cheap, while the training envs never pay it.
        if self.link == "lumos5g":
            self._init_lumos_episodes()
        else:
            self._eval_envs = [
                TileStreamingEnv(trace_indices=[idx], log_counterfactual=True, tile_frame=self.tile_frame)
                for idx in self.eval_trace_indices
            ]
            self._episodes = [(env, self.eval_seeds[i], self.eval_trace_indices[i], None)
                              for i, env in enumerate(self._eval_envs)]
            self._episode_d_bars = [self.d_bar] * len(self._episodes)
        self._step_trace_writer = EvalStepTraceWriter(self.log_dir)
        self._episode_writer = EvalEpisodeWriter(self.log_dir)

    def _init_lumos_episodes(self) -> None:
        """Held-out viewers x fixed validation windows; the "seed" column of the
        eval CSVs is the episode's index into <log_dir>/eval_windows.csv."""
        data = self.lumos_data if self.lumos_data is not None else lumos5g.load_bundle()
        envs = {idx: TileStreamingEnv(trace_indices=[idx], log_counterfactual=True, link="lumos5g",
                                      lumos_split="val", lumos_scale=self.lumos_scale, lumos_data=data,
                                      tile_frame=self.tile_frame, obs_forced_bits=self.obs_forced_bits)
                for idx in self.eval_trace_indices}
        self._eval_envs = list(envs.values())
        ep_len = self._eval_envs[0]._ep_len
        wins = lumos5g.eval_windows(data["runs"], data["splits"]["val"], ep_len, self.lumos_eval_windows)
        self._episodes = [(envs[idx], i, idx, {"window": w})
                          for i, (w, idx) in enumerate(product(wins, self.eval_trace_indices))]
        # viewport-only stall of every eval episode (no extra tiles) - the
        # unavoidable part, deterministic given the window and viewer
        no_extra = np.zeros(config.N_TILES, dtype=np.int64)
        floors = []
        for env, seed, _idx, opts in self._episodes:
            env.reset(seed=seed, options=opts)
            done, info = False, {}
            while not done:
                _obs, _r, term, trunc, info = env.step(no_extra)
                done = term or trunc
            floors.append(float(info["J_D"]))
        self._stall_floor = float(np.mean(floors))
        self._episode_d_bars = [f + self.stall_allowance for f in floors]
        self.d_bar = self._stall_floor + self.stall_allowance
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, "eval_windows.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed", "trace_position", "run", "start", "viewport_only_J_D", "d_bar"])
            for (env, seed, idx, opts), fl, db in zip(self._episodes, floors, self._episode_d_bars):
                w.writerow([seed, idx, opts["window"][0], opts["window"][1], fl, db])
        if self.verbose >= 1:
            print(f"Lumos5G eval: {len(self._episodes)} episodes, viewport-only J_D {self._stall_floor:.4f}, "
                  f"stall allowance {self.stall_allowance} -> eval D_bar {self.d_bar:.4f}")

    def _on_step(self) -> bool:
        return True   # all logic runs in _on_rollout_end; abstract method must exist

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count % self.eval_every_n_rollouts != 0:
            return

        vecnorm: VecNormalize = self.model.get_vec_normalize_env()

        # Stage the per-step spatial trace for every eval; whether it is
        # kept is decided after the checkpoint comparison below.
        def _make_sink(trace_position: int, seed: int):
            def _sink(step_index: int, info: dict) -> None:
                self._step_trace_writer.stage_step(
                    rollout=self._rollout_count, total_timesteps=self.num_timesteps,
                    trace_position=trace_position, seed=seed, step=step_index, info=info,
                )
            return _sink

        results = [
            run_eval_episode(env, self.model, vecnorm, seed=seed, trace_position=pos,
                              step_sink=_make_sink(pos, seed), options=opts)
            for env, seed, pos, opts in self._episodes
        ]

        J_Q = float(np.mean([r.J_Q for r in results]))
        J_D = float(np.mean([r.J_D for r in results]))
        J_P = float(np.mean([r.J_P for r in results]))
        J_B = float(np.mean([r.J_B for r in results]))
        viol_D = 0.0 if "D" in self.dropped_constraints else max(0.0, J_D - self.d_bar)
        viol_P = 0.0 if "P" in self.dropped_constraints else max(0.0, J_P - self.p_bar)
        viol_B = 0.0 if "B" in self.dropped_constraints else max(0.0, J_B - self.b_bar)
        violation = viol_D + viol_P + viol_B
        feasible = violation <= self.feasibility_tol

        episode_rows = []
        for r, e_d_bar in zip(results, self._episode_d_bars):
            e_viol_D = 0.0 if "D" in self.dropped_constraints else max(0.0, r.J_D - e_d_bar)
            e_viol_P = 0.0 if "P" in self.dropped_constraints else max(0.0, r.J_P - self.p_bar)
            e_viol_B = 0.0 if "B" in self.dropped_constraints else max(0.0, r.J_B - self.b_bar)
            e_violation = e_viol_D + e_viol_P + e_viol_B
            e_feasible = float(e_violation <= self.feasibility_tol)
            episode_rows.append([
                self._rollout_count, self.num_timesteps, r.trace_position, r.seed,
                r.J_D, r.J_P, r.J_B, e_viol_D, e_viol_P, e_viol_B, e_violation, e_feasible,
            ])
        self._episode_writer.write_rows(episode_rows)

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
        if self._stall_floor is not None:
            self.logger.record("eval/avoidable_stall", J_D - self._stall_floor)

        new_key = _checkpoint_key(feasible, violation, J_Q)
        is_new_best = new_key < self.best_key
        if is_new_best:
            self.best_key = new_key
            self.best_checkpoint_step = self.num_timesteps
            self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            if vecnorm is not None:
                vecnorm.save(os.path.join(self.best_model_save_path, "best_vecnormalize.pkl"))
            if self.verbose >= 1:
                print(f"New best checkpoint @ {self.num_timesteps}: "
                      f"feasible={feasible} J_Q={J_Q:.4f} violation={violation:.4f}")

        # Keep this eval's per-step trace if it lands on the sampling
        # cadence, OR if it produced a new best checkpoint - the best
        # checkpoint is the one the figures are generated from, so its
        # trace should be captured regardless of where the cadence fell.
        on_cadence = (self.spatial_log_every_n_rollouts > 0
                      and self._rollout_count % self.spatial_log_every_n_rollouts == 0)
        if on_cadence or is_new_best:
            self._step_trace_writer.commit()
        else:
            self._step_trace_writer.discard()
