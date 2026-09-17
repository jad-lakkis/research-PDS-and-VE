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

    d_bar = args.d_bar if args.d_bar is not None else config.D_BAR
    p_bar = args.p_bar if args.p_bar is not None else config.P_BAR
    b_bar = args.b_bar if args.b_bar is not None else config.B_BAR

    bundle = data_loader.load_training_video_bundle()
    n_traces = len(bundle["valid_traces"])
    assert max(config.EVAL_TRACE_INDICES) < n_traces, (
        f"config.EVAL_TRACE_INDICES={config.EVAL_TRACE_INDICES} references an index >= "
        f"the {n_traces} valid traces actually loaded"
    )
    eval_indices = list(config.EVAL_TRACE_INDICES)
    train_indices = [i for i in range(n_traces) if i not in eval_indices]

    train_venv = DummyVecEnv([
        lambda: build_env(True, train_indices, bundle) for _ in range(args.n_envs)
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
                            seed=args.seed, verbose=1, device=args.device)
    model.set_logger(sb3_configure(args.log_dir, ["stdout", "csv"]))

    callbacks = [
        PhysicalMetricsCallback(),
        TileEnhancementTrackerCallback(log_dir=args.log_dir),
        LagrangianMultiplierCallback(d_bar=d_bar, p_bar=p_bar, b_bar=b_bar, eta=config.MU_LEARNING_RATE),
        HeldOutTraceEvalCallback(
            d_bar=d_bar, p_bar=p_bar, b_bar=b_bar,
            best_model_save_path=os.path.join(args.log_dir, "best"),
            eval_every_n_rollouts=args.eval_freq_rollouts,
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
