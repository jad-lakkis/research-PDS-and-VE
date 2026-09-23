"""
train_ppo.py - train PPO on TileStreamingEnv, by default through the
Lagrangian constrained-RL reward (adaptive mu_D/mu_P/mu_B in place of
the fixed BETA_S/BETA_P/BETA_B weights - see streaming_rl/lagrangian.py).
--no-lagrangian trains against the env's own fixed-beta reward (eq. 9)
instead.

Verified against the installed stable_baselines3==2.9.0: VecNormalize is
itself a VecEnv, so PPO's usual auto-wrap (Monitor + DummyVecEnv) never
triggers once it's in the stack - Monitor/DummyVecEnv/VecNormalize are
all built by hand below. env_method()/get_attr()/set_attr() pass
straight through VecNormalize to the wrapped DummyVecEnv unchanged, so
LagrangianMultiplierCallback needs no adjustment for VecNormalize being
present. tensorboard is NOT installed in this project's .venv, so
logging goes through stable_baselines3.common.logger.configure(log_dir,
["stdout", "csv"]) + model.set_logger(...) instead of tensorboard_log=.

Constraint enforcement (eqs. 6-8) uses the literal per-episode
discounted returns J_D/J_P/J_B (streaming_rl/environment.py, exposed in
info only on each episode's terminal step), not a windowed running
mean - see other/decision_log.md Stage 7 #1-4 for why, including a
flagged, deliberately-accepted consequence (J_D/J_P/J_B are numerically
dominated by each episode's first ~2 steps at DISCOUNT_FACTOR_LAMBDA=
0.01). Multiplier updates and physical-metrics logging both fire once
per PPO rollout (_on_rollout_end) rather than on an independent step
cadence - confirmed EventCallback/EveryNTimesteps never forwards
on_rollout_end to a wrapped child, so both callbacks are top-level
entries in callbacks=[...], not nested.

Run: .venv\\Scripts\\python.exe train_ppo.py [--no-lagrangian] [--total-timesteps N] [--d-bar F] [--p-bar F] [--b-bar F]
"""

import argparse
import csv
import json
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as sb3_configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import config
from streaming_rl import data_loader
from streaming_rl.environment import TileStreamingEnv
from streaming_rl.lagrangian import LagrangianRewardWrapper, dual_ascent_step
from streaming_rl.eval import HeldOutTraceEvalCallback


class LagrangianMultiplierCallback(BaseCallback):
    """Top-level callback (see module docstring for why, re: on_rollout_end
    forwarding). _on_step() collects info["J_D"]/["J_P"]/["J_B"] out of
    self.locals["infos"] wherever present (i.e. on whichever env just hit
    its terminal step) throughout the rollout. _on_rollout_end() fires
    once collect_rollouts() has finished the whole rollout's env.step()
    calls and strictly before the policy update - averages J_D/J_P/J_B
    across however many episodes fully completed this rollout and does
    one dual_ascent_step per multiplier.
    """

    def __init__(self, d_bar: float, p_bar: float, b_bar: float, eta: float, verbose: int = 0):
        super().__init__(verbose)
        self.d_bar, self.p_bar, self.b_bar, self.eta = d_bar, p_bar, b_bar, eta
        self._J_D_buffer: list = []
        self._J_P_buffer: list = []
        self._J_B_buffer: list = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "J_D" in info:   # terminal step only
                self._J_D_buffer.append(info["J_D"])
                self._J_P_buffer.append(info["J_P"])
                self._J_B_buffer.append(info["J_B"])
        return True

    def _on_rollout_end(self) -> None:
        n = len(self._J_D_buffer)
        self.logger.record("lagrangian/n_episodes", n)
        if n == 0:
            # No episode completed within this rollout (n_steps too
            # small relative to episode length) - skip the update rather
            # than divide by zero; multipliers hold their current values.
            return

        avg_J_D = sum(self._J_D_buffer) / n
        avg_J_P = sum(self._J_P_buffer) / n
        avg_J_B = sum(self._J_B_buffer) / n

        current = self.training_env.env_method("get_multipliers")[0]
        new_mu_D = dual_ascent_step(current["mu_D"], avg_J_D, self.d_bar, self.eta)
        new_mu_P = dual_ascent_step(current["mu_P"], avg_J_P, self.p_bar, self.eta)
        new_mu_B = dual_ascent_step(current["mu_B"], avg_J_B, self.b_bar, self.eta)
        self.training_env.env_method("set_multipliers", mu_D=new_mu_D, mu_P=new_mu_P, mu_B=new_mu_B)

        for name, val in [("mu_D", new_mu_D), ("mu_P", new_mu_P), ("mu_B", new_mu_B),
                           ("avg_J_D", avg_J_D), ("avg_J_P", avg_J_P), ("avg_J_B", avg_J_B)]:
            self.logger.record(f"lagrangian/{name}", val)

        self._J_D_buffer.clear()
        self._J_P_buffer.clear()
        self._J_B_buffer.clear()


class PhysicalMetricsCallback(BaseCallback):
    """Raw physical-unit metrics (mean coverage, stall seconds, power in
    mW, enhanced-tile count) logged once per PPO rollout, read directly
    off TileStreamingEnv's own per-step info fields via
    self.locals["infos"] - present regardless of which reward wrapper is
    active, so this works identically in --no-lagrangian mode too. Same
    top-level, dual-hook (_on_step accumulates, _on_rollout_end flushes)
    pattern as LagrangianMultiplierCallback, for the same reason.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._coverage_buf: list = []
        self._stall_seconds_buf: list = []
        self._power_watts_buf: list = []
        self._n_enhanced_tiles_buf: list = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            self._coverage_buf.append(info["coverage"])
            self._stall_seconds_buf.append(info["D_t"])
            self._power_watts_buf.append(info["power_watts"])
            self._n_enhanced_tiles_buf.append(info["n_enhanced_tiles"])
        return True

    def _on_rollout_end(self) -> None:
        n = len(self._coverage_buf)
        if n == 0:
            return
        self.logger.record("physical/mean_coverage", sum(self._coverage_buf) / n)
        self.logger.record("physical/mean_stall_seconds", sum(self._stall_seconds_buf) / n)
        self.logger.record("physical/mean_power_mW", 1000.0 * sum(self._power_watts_buf) / n)
        self.logger.record("physical/mean_n_enhanced_tiles", sum(self._n_enhanced_tiles_buf) / n)
        self._coverage_buf.clear()
        self._stall_seconds_buf.clear()
        self._power_watts_buf.clear()
        self._n_enhanced_tiles_buf.clear()


class TileEnhancementTrackerCallback(BaseCallback):
    """Stage 8b: coarse spatial enhancement frequency, so roughly WHERE on
    the panorama gets enhanced (not just how many tiles) can be plotted
    later - e.g. a heatmap. Collapses the 8x8=64 tile grid down to a 4x4=16
    region grid (each region = one 2x2 block of tiles, frequency-averaged)
    - confirmed with the user: keep per-rollout logging, but not one column
    per individual tile (64 was "extracting everything"). Kept OUT of the
    main SB3 CSV logger deliberately: even 16 extra columns would bloat
    progress.csv and repeatedly trigger CSVOutputFormat's full-file
    rewrite-on-new-key behavior (see other/decision_log.md's SB3-source
    verification notes) right at the start of training. Instead writes its
    own small CSV, one row per rollout, alongside progress.csv. Also logs
    two cheap summary scalars (std/max frequency across the full 64 tiles,
    unaffected by this region reduction) to the main logger, so there's a
    quick sanity signal without needing the full file. Same top-level,
    dual-hook pattern as the other Stage-8-era callbacks.
    """

    N_REGION_ROWS = 4
    N_REGION_COLS = 4

    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self._csv_path = os.path.join(log_dir, "tile_enhancement_freq.csv")
        self._tile_count = np.zeros(config.N_TILES, dtype=np.int64)
        self._n_steps_this_rollout = 0
        self._rollout_idx = 0

    def _init_callback(self) -> None:
        os.makedirs(os.path.dirname(self._csv_path) or ".", exist_ok=True)
        region_cols = [f"region_r{r}_c{c}" for r in range(self.N_REGION_ROWS) for c in range(self.N_REGION_COLS)]
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rollout", "total_timesteps"] + region_cols)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            self._tile_count += info["tile_mask"]
            self._n_steps_this_rollout += 1
        return True

    def _on_rollout_end(self) -> None:
        if self._n_steps_this_rollout == 0:
            return
        self._rollout_idx += 1
        freq = self._tile_count / self._n_steps_this_rollout
        # raster row-major (config.TILE_INDEX_ORDER): flat index i -> (i//8, i%8).
        # Block-average each 2x2 tile block into one region: (8,8) -> (4,2,4,2)
        # -> mean over the two size-2 axes -> (4,4).
        row_block = config.TILE_GRID_ROWS // self.N_REGION_ROWS
        col_block = config.TILE_GRID_COLS // self.N_REGION_COLS
        region_freq = freq.reshape(config.TILE_GRID_ROWS, config.TILE_GRID_COLS) \
                          .reshape(self.N_REGION_ROWS, row_block, self.N_REGION_COLS, col_block) \
                          .mean(axis=(1, 3))
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self._rollout_idx, self.num_timesteps] + region_freq.flatten().tolist())
        self.logger.record("tiles/enhancement_freq_std", float(freq.std()))
        self.logger.record("tiles/enhancement_freq_max", float(freq.max()))
        self._tile_count[:] = 0
        self._n_steps_this_rollout = 0


def build_env(use_lagrangian: bool, trace_indices: list, bundle: dict):
    """One (unvectorized) env, Monitor on the outside: Monitor(Lagrangian
    RewardWrapper(TileStreamingEnv(...))) or Monitor(TileStreamingEnv(...))
    for --no-lagrangian. Manual Monitor wrap is required (see module
    docstring: VecNormalize is itself a VecEnv, so PPO's auto Monitor+
    DummyVecEnv wrap never triggers once VecNormalize is in the stack).
    """
    env = TileStreamingEnv(trace_indices=trace_indices, bundle=bundle)
    if use_lagrangian:
        env = LagrangianRewardWrapper(env)
    return Monitor(env)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-lagrangian", action="store_true",
                    help="train against the env's fixed-beta reward instead of the Lagrangian wrapper")
    p.add_argument("--total-timesteps", type=int, default=200_000)
    # Stage 8: 1728 = 48*36, a clean multiple of both the 36-step episode
    # length (Part C - zero episode/rollout boundary contamination) and
    # the PPO batch_size=64 default (no truncated-minibatch warning).
    # Gives 48 episodes/rollout, ~270 policy gradient updates per one
    # multiplier update - close to the originally-validated 320:1 ratio,
    # confirmed with the user after reverting env parallelization (below).
    p.add_argument("--n-steps", type=int, default=1728, help="PPO rollout length (per env)")
    # Defaults to 1 (no parallelization) - confirmed with the user: the
    # ratio-widening side effect of n_envs>1 (multiplier reacts to real
    # constraint data proportionally less often) wasn't worth the
    # bigger-Monte-Carlo-sample benefit. Left as a supported option, not
    # removed, in case that trade-off gets revisited later.
    p.add_argument("--n-envs", type=int, default=1,
                    help="number of parallel training env instances, sharing one loaded bundle")
    p.add_argument("--d-bar", type=float, default=None, help="overrides config.D_BAR for this run")
    p.add_argument("--p-bar", type=float, default=None, help="overrides config.P_BAR for this run")
    p.add_argument("--b-bar", type=float, default=None, help="overrides config.B_BAR for this run")
    p.add_argument("--eval-freq-rollouts", type=int, default=1,
                    help="run held-out-trace evaluation every N rollouts")
    p.add_argument("--log-dir", type=str, default="runs/ppo_run")
    p.add_argument("--seed", type=int, default=0)
    # Stage 8: continue an existing run instead of starting fresh - points
    # at a PREVIOUS run's --log-dir (must contain model.zip, vecnormalize.pkl,
    # multipliers.json - all written by this script's own final-save step).
    # VERIFIED BY ACTUALLY RUNNING (an initial doc comment here had this
    # backwards): with reset_num_timesteps=False, SB3's _setup_learn() does
    # `total_timesteps += self.num_timesteps` internally - so --total-timesteps
    # is an INCREMENT (how many MORE steps to run), not an absolute target.
    # A prior run at 518,400 steps + --total-timesteps 522432 therefore runs
    # 522,432 MORE steps, ending at 1,040,832 - not at 522,432.
    p.add_argument("--resume-from", type=str, default=None,
                    help="log-dir of a previous run to continue from (model+vecnormalize+multipliers)")
    # RunPod GPU support: SB3 auto-detects CUDA when available and installed
    # torch has CUDA support - "auto" (default) already does the right thing
    # on both a CPU-only laptop and a CUDA-enabled RunPod instance with no
    # other code change needed. Explicit flag only for clarity/override.
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                    help="torch device for PPO - 'auto' picks cuda if available, else cpu")
    # PDS-vs-baseline ablation (Run B): isolates the one-step advantage from
    # the PDS critic by using this same ordinary-critic PPO path with
    # gae_lambda=0 instead of PPOWithPDS. Default 0.95 is SB3's own default -
    # unset, this flag changes nothing for existing runs.
    p.add_argument("--gae-lambda", type=float, default=0.95,
                    help="GAE lambda for advantage estimation (0 = one-step, matching PDS's estimator)")
    return p.parse_args()


def main():
    args = parse_args()
    use_lagrangian = not args.no_lagrangian

    d_bar = args.d_bar if args.d_bar is not None else config.D_BAR
    p_bar = args.p_bar if args.p_bar is not None else config.P_BAR
    b_bar = args.b_bar if args.b_bar is not None else config.B_BAR
    if use_lagrangian and (d_bar is None or p_bar is None or b_bar is None):
        raise ValueError(
            "Lagrangian training requires concrete D_BAR/P_BAR/B_BAR "
            "(currently unset placeholders) - pick values, pass --d-bar/--p-bar/--b-bar, "
            "or run with --no-lagrangian for the fixed-beta baseline."
        )

    # Loaded once, shared between the training env and every held-out
    # eval env - rd_data is read-only after load (measured ~388MB;
    # data_loader.load_training_video_bundle() is uncached).
    bundle = data_loader.load_training_video_bundle()
    n_traces = len(bundle["valid_traces"])
    assert max(config.EVAL_TRACE_INDICES) < n_traces, (
        f"config.EVAL_TRACE_INDICES={config.EVAL_TRACE_INDICES} references an index >= "
        f"the {n_traces} valid traces actually loaded"
    )
    eval_indices = list(config.EVAL_TRACE_INDICES)
    train_indices = [i for i in range(n_traces) if i not in eval_indices]

    # Stage 8 (environment parallelization): N env instances in one process,
    # all sharing the same loaded `bundle` (rd_data is read-only after load,
    # so this is safe) - avoids the ~388MB-per-process duplication a
    # SubprocVecEnv would require, since bundle can't be shared across a
    # process boundary. Each lambda is independent (no closure/late-binding
    # issue - none of them close over a per-iteration loop variable).
    train_venv = DummyVecEnv([
        lambda: build_env(use_lagrangian, train_indices, bundle) for _ in range(args.n_envs)
    ])

    if args.resume_from is not None:
        # VecNormalize.load wants the RAW (not-yet-normalized) VecEnv - same
        # pattern streaming_rl/eval.py already uses for eval envs.
        train_venv = VecNormalize.load(os.path.join(args.resume_from, "vecnormalize.pkl"), train_venv)
        train_venv.training = True  # frozen (eval-mode) copies may exist elsewhere; this one keeps learning
        model = PPO.load(os.path.join(args.resume_from, "model"), env=train_venv, device=args.device)
        if use_lagrangian:
            with open(os.path.join(args.resume_from, "multipliers.json")) as f:
                saved_mu = json.load(f)
            train_venv.env_method("set_multipliers", **saved_mu)
            print(f"Resumed from {args.resume_from}: multipliers={saved_mu}, "
                  f"num_timesteps={model.num_timesteps}")
    else:
        train_venv = VecNormalize(train_venv, training=True, norm_obs=True, norm_reward=False,
                                   gamma=config.PPO_GAMMA)
        model = PPO("MlpPolicy", train_venv, n_steps=args.n_steps, gamma=config.PPO_GAMMA,
                    gae_lambda=args.gae_lambda, seed=args.seed, verbose=1, device=args.device)
    model.set_logger(sb3_configure(args.log_dir, ["stdout", "csv"]))

    callbacks = [PhysicalMetricsCallback(), TileEnhancementTrackerCallback(log_dir=args.log_dir)]
    if use_lagrangian:
        callbacks.append(LagrangianMultiplierCallback(
            d_bar=d_bar, p_bar=p_bar, b_bar=b_bar, eta=config.MU_LEARNING_RATE,
        ))
    callbacks.append(HeldOutTraceEvalCallback(
        d_bar=d_bar, p_bar=p_bar, b_bar=b_bar,
        best_model_save_path=os.path.join(args.log_dir, "best"),
        eval_every_n_rollouts=args.eval_freq_rollouts,
        verbose=1,
    ))

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks,
                reset_num_timesteps=(args.resume_from is None))

    os.makedirs(args.log_dir, exist_ok=True)
    model.save(os.path.join(args.log_dir, "model"))
    train_venv.save(os.path.join(args.log_dir, "vecnormalize.pkl"))
    print(f"Saved model to {os.path.join(args.log_dir, 'model')}")
    print(f"Saved VecNormalize stats to {os.path.join(args.log_dir, 'vecnormalize.pkl')}")
    if use_lagrangian:
        # Stage 8: multipliers live only in the LagrangianRewardWrapper
        # instance's memory otherwise - lost on process exit. Persisted here
        # so --resume-from can restore them instead of silently resetting to
        # MU_D_INIT/MU_P_INIT/MU_B_INIT.
        final_mu = train_venv.env_method("get_multipliers")[0]
        with open(os.path.join(args.log_dir, "multipliers.json"), "w") as f:
            json.dump(final_mu, f)
        print(f"Saved multipliers to {os.path.join(args.log_dir, 'multipliers.json')}: {final_mu}")


if __name__ == "__main__":
    main()
