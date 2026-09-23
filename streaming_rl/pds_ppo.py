"""
pds_ppo.py - PPOWithPDS: PPO with the ordinary GAE residual replaced by
the PDS residual (delta_t^PDS = r_known^t + Vtilde_psi(omega~^t) -
V_phi(omega^t)), chained through the same backward GAE(lambda) recursion
as baseline PPO ("PDS-GAE" - see PDSRolloutBuffer.compute_pds_returns_and_advantage),
plus a second, separate PDS critic Vtilde_psi trained alongside the
actor/ordinary critic (PDS design plan, Sections 4/6/8). gae_lambda is
inherited unchanged from stock PPO's own constructor kwarg/self.gae_lambda -
0 reproduces the original one-step PDS advantage, 0.95 matches baseline
PPO's horizon.

Overrides collect_rollouts() (Phase A: also builds omega~^t and the
known/random reward split every step, using the LIVE mu_D/mu_P/mu_B) and
train() (Phase C: uses A^t_PDS-GAE instead of ordinary GAE, adds the PDS
critic's own MSE loss term). Phase D (multiplier updates) is untouched -
train_ppo.py's LagrangianMultiplierCallback works identically here,
hooked to _on_rollout_end the same way, since PPOWithPDS still calls
callback.on_rollout_end() at exactly the same point stock
OnPolicyAlgorithm.collect_rollouts() does.

Deliberately does NOT replicate stock OnPolicyAlgorithm.collect_rollouts()'s
own "handle timeout by bootstrapping with value function" reward-patching
block (adding gamma*V(terminal_obs) directly into rewards[idx] for a
TimeLimit.truncated step): that patch exists to fix the ordinary
single-critic GAE bootstrap specifically, and doing it here too would
double-apply the correction (once via that patch mutating rewards[idx],
again via compute_pds_returns_and_advantage's own truncated-branch) and
would also break the r_known+r_random==r_total invariant asserted below,
since rewards[idx] would no longer be the raw Lagrangian reward. PPOWithPDS
handles truncation entirely through PDSRolloutBuffer's own
terminated/truncated/true_next_obs bookkeeping instead.
"""

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance, obs_as_tensor

import config
from streaming_rl import pds
from streaming_rl.pds_buffer import PDSRolloutBuffer
from streaming_rl.pds_policy import PDSActorCriticPolicy


class PPOWithPDS(PPO):
    def __init__(self, policy=PDSActorCriticPolicy, env=None, pds_net_arch: list = None, **kwargs):
        assert policy is PDSActorCriticPolicy, (
            "PPOWithPDS always uses PDSActorCriticPolicy - it's not a swappable "
            "policy= argument like stock PPO's (needed so predict_pds_values() exists)"
        )
        policy_kwargs = dict(kwargs.pop("policy_kwargs", None) or {})
        policy_kwargs.setdefault("pds_obs_dim", pds.PDS_OBS_DIM)
        policy_kwargs.setdefault("pds_net_arch", pds_net_arch or config.PDS_CRITIC_ARCHITECTURE)
        kwargs["policy_kwargs"] = policy_kwargs
        kwargs["rollout_buffer_class"] = PDSRolloutBuffer
        super().__init__(PDSActorCriticPolicy, env, **kwargs)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        # Multipliers are constant for the whole rollout - LagrangianMultiplierCallback
        # only updates them in _on_rollout_end(), which this method itself
        # triggers (via callback.on_rollout_end() below) only AFTER the loop
        # below finishes. One fetch here is exactly equivalent to fetching
        # every step (as the design plan's pseudocode literally writes it),
        # without ~n_rollout_steps redundant env_method() round trips through
        # the VecNormalize/Monitor/DummyVecEnv stack.
        multipliers = env.env_method("get_multipliers")[0]
        mu_d, mu_p, mu_b = multipliers["mu_D"], multipliers["mu_P"], multipliers["mu_B"]

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # --- Phase A: PDS-specific per-env bookkeeping ---
            pds_obs = np.zeros((env.num_envs, pds.PDS_OBS_DIM), dtype=np.float32)
            true_next_obs = np.zeros_like(self._last_obs)
            known_rewards = np.zeros(env.num_envs, dtype=np.float32)
            random_rewards = np.zeros(env.num_envs, dtype=np.float32)
            terminateds = np.zeros(env.num_envs, dtype=np.float32)
            truncateds = np.zeros(env.num_envs, dtype=np.float32)

            for idx, done in enumerate(dones):
                info = infos[idx]
                # DummyVecEnv contract (verified against the installed SB3
                # source): dones[idx] = terminated or truncated, and
                # info["TimeLimit.truncated"] = truncated and not terminated
                # - so truncated/terminated are recoverable exactly from
                # these two values, including the (here never occurring)
                # edge case of both firing on the same step.
                truncated = bool(info.get("TimeLimit.truncated", False))
                terminated = bool(done) and not truncated
                terminateds[idx] = float(terminated)
                truncateds[idx] = float(truncated)

                if done:
                    true_next_obs[idx] = info["terminal_observation"]
                else:
                    true_next_obs[idx] = new_obs[idx]

                pds_obs[idx] = pds.build_pds_observation(self._last_obs[idx], true_next_obs[idx], info["tile_mask"])

                r_known, r_random = pds.split_reward(info, mu_d, mu_p, mu_b)
                known_rewards[idx] = r_known
                random_rewards[idx] = r_random
                # rewards[idx] is the raw Lagrangian reward as env.step()
                # returns it - VecNormalize.norm_reward=False project-wide
                # (train_ppo.py/train_ppo_pds.py), so it's never scaled.
                # np.isclose, not a bare absolute epsilon: float32 forward
                # passes and reordered floating-point sums can differ by
                # more than a strict 1e-9 even when the split is exactly
                # correct.
                assert np.isclose(r_known + r_random, rewards[idx]), (
                    f"PDS reward split mismatch at step {n_steps}, env {idx}: "
                    f"r_known={r_known} + r_random={r_random} = {r_known + r_random} != r_total={rewards[idx]}"
                )

            rollout_buffer.add(
                self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs,
                pds_obs, true_next_obs, known_rewards, random_rewards, terminateds, truncateds,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            last_values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_pds_returns_and_advantage(
            self.policy, last_values=last_values, gae_lambda=self.gae_lambda,
        )

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, ordinary_value_losses, pds_value_losses = [], [], []
        clip_fractions = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                pds_values = self.policy.predict_pds_values(rollout_data.pds_observations).flatten()

                # Normalize advantage - same per-minibatch convention stock
                # PPO.train() uses (not a single global-rollout pass).
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                ordinary_value_loss = F.mse_loss(rollout_data.ordinary_returns, values_pred)
                pds_value_loss = F.mse_loss(rollout_data.pds_returns, pds_values)
                ordinary_value_losses.append(ordinary_value_loss.item())
                pds_value_losses.append(pds_value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * (ordinary_value_loss + pds_value_loss)

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.ordinary_returns.flatten()
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(ordinary_value_losses))
        self.logger.record("train/pds_value_loss", np.mean(pds_value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
