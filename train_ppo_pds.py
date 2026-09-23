"""
train_ppo_pds.py - train PPO+PDS (post-decision state) on TileStreamingEnv:
same environment, reward, budgets, discount, seeds, and rollout size as
train_ppo.py's Lagrangian path, differing ONLY in the advantage/critic
mechanism (streaming_rl.pds_ppo.PPOWithPDS's one-step PDS advantage plus a
second PDS critic, in place of GAE) - the fair-comparison protocol
confirmed in the PDS design plan, Section 6.

Always trains through the Lagrangian path (LagrangianRewardWrapper, live
mu_D/mu_P/mu_B) - there is no --no-lagrangian mode here, unlike
train_ppo.py: the PDS reward split (streaming_rl.pds.split_reward) is only
defined in terms of the live multipliers (PDS design plan, Section 4), so
the fixed-beta path isn't a meaningful PDS configuration.

Reuses build_env/LagrangianMultiplierCallback/PhysicalMetricsCallback/
TileEnhancementTrackerCallback from train_ppo.py directly (imported, not
duplicated) - train_ppo.py itself is NOT modified by this file; the two
entrypoints are independently runnable (PDS design plan, Sections 9/13).

Run: .venv\\Scripts\\python.exe train_ppo_pds.py --d-bar F --p-bar F --b-bar F [--total-timesteps N] [--seed N]
"""

import argparse
import json
import os

from stable_baselines3.common.logger import configure as sb3_configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import config
from streaming_rl import data_loader
from streaming_rl.eval import HeldOutTraceEvalCallback
from streaming_rl.lagrangian import parse_dropped_constraints
from streaming_rl.pds_ppo import PPOWithPDS
from train_ppo import (
    LagrangianMultiplierCallback,
    PhysicalMetricsCallback,
    TileEnhancementTrackerCallback,
    build_env,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--n-steps", type=int, default=1728,
                    help="PPO rollout length (per env) - same default/meaning as train_ppo.py")
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--d-bar", type=float, default=None, help="overrides config.D_BAR for this run")
    p.add_argument("--p-bar", type=float, default=None, help="overrides config.P_BAR for this run")
    p.add_argument("--b-bar", type=float, default=None, help="overrides config.B_BAR for this run")
    p.add_argument("--ent-coef", type=float, default=None,
                    help="overrides config.PDS_ENTROPY_COEF for this run - PPOWithPDS's actor-loss "
                         "entropy coefficient (stock PPO's own ent_coef*entropy_loss term, previously "
                         "always 0.0 here); only meaningful on a fresh start, not --resume-from, since "
                         "a resumed run keeps its own already-baked-in value")
    # PDS-vs-baseline ablation (Run D): chains the PDS residual through the
    # same backward GAE recursion as baseline PPO instead of stopping at one
    # step (see PDSRolloutBuffer.compute_pds_returns_and_advantage). 0 =
    # the original one-step PDS advantage (Run C, matches every prior PDS
    # run in runpod_pds_results/); 0.95 (default here, matching train_ppo.py
    # and SB3's own default) = Run D. Only meaningful on a fresh start, not
    # --resume-from - baked into the saved model's own hyperparameters.
    p.add_argument("--gae-lambda", type=float, default=0.95,
                    help="GAE lambda for the PDS-residual advantage recursion (0 = original one-step "
                         "PDS advantage / Run C, 0.95 = PDS-GAE / Run D matching baseline PPO)")
    # Optional stabilizer, default OFF (None = SB3's own default, no value
    # clipping) - a standalone lever from --gae-lambda/--ent-coef, only
    # meant to be turned on as its own separate, labeled comparison (a "D
    # + clipping" arm) once a clean ent-coef=0 Run D has already been
    # checked, not bundled into the same run as another change. SB3's own
    # docstring: "this clipping depends on the reward scaling" - it's a
    # raw-magnitude clip on (new_value - old_value), not a fraction like
    # --clip-range, so pick it from this project's own healthy value/return
    # scale (converged episode reward is O(10) in this project), not
    # copied from clip_range's 0.2.
    p.add_argument("--clip-range-vf", type=float, default=None,
                    help="value-function clipping (raw units, depends on reward scale - see SB3 docs); "
                         "None (default) = off, matching every run so far")
    # Genuine removal of a constraint from the Lagrangian, not a loosened
    # budget - see streaming_rl.lagrangian.parse_dropped_constraints and
    # train_ppo.py's own --drop-constraints for exactly what this does.
    p.add_argument("--drop-constraints", type=str, default="",
                    help="comma-separated subset of D,P,B to remove entirely from the "
                         "reward and feasibility check, e.g. 'P' or 'P,B' (default: none dropped)")
    p.add_argument("--eval-freq-rollouts", type=int, default=1,
                    help="run held-out-trace evaluation every N rollouts")
    p.add_argument("--log-dir", type=str, default="runs/ppo_pds_run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume-from", type=str, default=None,
                    help="log-dir of a previous PPO+PDS run to continue from (model+vecnormalize+multipliers)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()
    dropped_constraints = parse_dropped_constraints(args.drop_constraints)
    if dropped_constraints:
        print(f"Dropping constraints entirely (not just loosening): {sorted(dropped_constraints)}")

    d_bar = args.d_bar if args.d_bar is not None else config.D_BAR
    p_bar = args.p_bar if args.p_bar is not None else config.P_BAR
    b_bar = args.b_bar if args.b_bar is not None else config.B_BAR
    ent_coef = args.ent_coef if args.ent_coef is not None else config.PDS_ENTROPY_COEF

    bundle = data_loader.load_training_video_bundle()
    n_traces = len(bundle["valid_traces"])
    assert max(config.EVAL_TRACE_INDICES) < n_traces, (
        f"config.EVAL_TRACE_INDICES={config.EVAL_TRACE_INDICES} references an index >= "
        f"the {n_traces} valid traces actually loaded"
    )
    eval_indices = list(config.EVAL_TRACE_INDICES)
    train_indices = [i for i in range(n_traces) if i not in eval_indices]

    train_venv = DummyVecEnv([
        lambda: build_env(True, train_indices, bundle, dropped_constraints) for _ in range(args.n_envs)
    ])

    if args.resume_from is not None:
        train_venv = VecNormalize.load(os.path.join(args.resume_from, "vecnormalize.pkl"), train_venv)
        train_venv.training = True
        model = PPOWithPDS.load(os.path.join(args.resume_from, "model"), env=train_venv, device=args.device)
        with open(os.path.join(args.resume_from, "multipliers.json")) as f:
            saved_mu = json.load(f)
        train_venv.env_method("set_multipliers", **saved_mu)
        print(f"Resumed from {args.resume_from}: multipliers={saved_mu}, num_timesteps={model.num_timesteps}")
    else:
        train_venv = VecNormalize(train_venv, training=True, norm_obs=True, norm_reward=False,
                                   gamma=config.PPO_GAMMA)
        model = PPOWithPDS(env=train_venv, n_steps=args.n_steps, gamma=config.PPO_GAMMA,
                            gae_lambda=args.gae_lambda, ent_coef=ent_coef,
                            clip_range_vf=args.clip_range_vf, seed=args.seed,
                            verbose=1, device=args.device)
        print(f"Fresh start: d_bar={d_bar} p_bar={p_bar} b_bar={b_bar} ent_coef={ent_coef} "
              f"gae_lambda={args.gae_lambda} clip_range_vf={args.clip_range_vf} "
              f"pds_critic_architecture={config.PDS_CRITIC_ARCHITECTURE}")
    model.set_logger(sb3_configure(args.log_dir, ["stdout", "csv"]))

    callbacks = [
        PhysicalMetricsCallback(),
        TileEnhancementTrackerCallback(log_dir=args.log_dir),
        LagrangianMultiplierCallback(d_bar=d_bar, p_bar=p_bar, b_bar=b_bar, eta=config.MU_LEARNING_RATE,
                                      dropped_constraints=dropped_constraints),
        HeldOutTraceEvalCallback(
            d_bar=d_bar, p_bar=p_bar, b_bar=b_bar,
            best_model_save_path=os.path.join(args.log_dir, "best"),
            eval_every_n_rollouts=args.eval_freq_rollouts,
            log_dir=args.log_dir,
            dropped_constraints=dropped_constraints,
            verbose=1,
        ),
    ]

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks,
                reset_num_timesteps=(args.resume_from is None))

    os.makedirs(args.log_dir, exist_ok=True)
    model.save(os.path.join(args.log_dir, "model"))
    train_venv.save(os.path.join(args.log_dir, "vecnormalize.pkl"))
    print(f"Saved model to {os.path.join(args.log_dir, 'model')}")
    print(f"Saved VecNormalize stats to {os.path.join(args.log_dir, 'vecnormalize.pkl')}")
    final_mu = train_venv.env_method("get_multipliers")[0]
    with open(os.path.join(args.log_dir, "multipliers.json"), "w") as f:
        json.dump(final_mu, f)
    print(f"Saved multipliers to {os.path.join(args.log_dir, 'multipliers.json')}: {final_mu}")


if __name__ == "__main__":
    main()
