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
    def __init__(self, policy=PDSActorCriticPolicy, env=None, pds_net_arch: list = None,
                 ve_enabled: bool = False, ve_batch_size: int = 0, ve_period: int = 10, **kwargs):
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

        # Virtual experience (v3 formulation, Algorithm 2) - disabled by
        # default, so existing PDS-only runs/behavior are completely
        # unaffected unless explicitly turned on.
        self.ve_enabled = bool(ve_enabled)
        self.ve_batch_size = int(ve_batch_size)
        self.ve_period = int(ve_period)
        if self.ve_enabled:
            assert self.ve_batch_size > 0, "ve_enabled=True requires ve_batch_size > 0"
            assert self.ve_period > 0, "ve_period must be positive"

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

        # Virtual experience needs rd_data/enhanced_level for compute_A_t -
        # per-env-instance attributes but constant (never reassigned after
        # __init__) and content-identical across all n_envs (same source
        # video), so one fetch ever, cached, is exactly as correct as
        # fetching every step (verified: env.get_attr("_rd_data")[0] is
        # literally the same object TileStreamingEnv uses internally,
        # reached through the full VecNormalize/DummyVecEnv/Monitor/
        # LagrangianRewardWrapper stack via gym.Wrapper's __getattr__
        # delegation).
        if self.ve_enabled and not hasattr(self, "_ve_rd_data"):
            self._ve_rd_data = env.get_attr("_rd_data")[0]
            self._ve_enhanced_level = env.get_attr("_enhanced_level")[0]

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            # Rollout-local, 0-indexed step - matches v3 Algorithm 2's own
            # "t" exactly (the doc's Algorithm 2 is nested for rollout /
            # for t=0..n_steps-1, same structure as Algorithm 1 - NOT a
            # flat counter across rollout boundaries). n_steps hasn't been
            # incremented yet this iteration, so it already equals t.
            t_local = n_steps
            ve_triggered = self.ve_enabled and (t_local % self.ve_period == 0)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)

                # Virtual-experience hypothetical-action sampling (v3:
                # "sample additional valid hypothetical actions from the
                # policy distribution already computed for the real
                # action"). self.policy.action_dist is the SAME object
                # forward() just configured via proba_distribution() (SB3
                # ActorCriticPolicy._get_action_dist_from_latent returns
                # self.action_dist.proba_distribution(...), which mutates
                # and returns self.action_dist - verified against the
                # installed SB3 2.9.0 source and against
                # PDSActorCriticPolicy, which overrides neither forward()
                # nor _get_action_dist_from_latent()) - so drawing more
                # samples from it costs zero extra actor forward passes.
                # MUST happen before any other policy call, since the
                # distribution is mutable and gets reconfigured by the
                # next one. RNG snapshot/restore keeps these extra draws
                # from perturbing the real action sequence: verified
                # (ad-hoc test, not yet a committed test file) that a
                # frozen policy given the same seed produces bit-identical
                # actions/values/log_probs on the NEXT call whether or not
                # VE sampling happened in between.
                virtual_actions_np = None
                if ve_triggered:
                    rng_state = th.get_rng_state()
                    virtual_actions = [
                        self.policy.action_dist.get_actions(deterministic=False)
                        for _ in range(self.ve_batch_size)
                    ]
                    th.set_rng_state(rng_state)
                    # (ve_batch_size, n_envs, action_dim) -> per-branch numpy arrays
                    virtual_actions_np = th.stack(virtual_actions, dim=0).cpu().numpy()
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

            if ve_triggered:
                pds_obs_v, returns_v = self._build_virtual_pairs(
                    env, infos, terminateds, mu_d, mu_p, mu_b, virtual_actions_np,
                )
                rollout_buffer.add_virtual_pairs(pds_obs_v, returns_v)

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

    def _build_virtual_pairs(self, env, infos, terminateds, mu_d, mu_p, mu_b, virtual_actions_np):
        """Recompute physics and assemble (pds_obs, target) for every
        virtual branch sampled this VE-triggering step, across all
        n_envs x ve_batch_size branches (v3 Algorithm 2 lines 13-18).

        virtual_actions_np: (ve_batch_size, n_envs, action_dim) raw
        actions drawn from self.policy.action_dist right after the real
        forward pass this step (see collect_rollouts). infos/terminateds
        are THIS step's real per-env info/terminal flags - reused, never
        recomputed per branch (that's the whole basis for VE being
        valid). self._last_obs is still the PRE-step observation here
        (collect_rollouts reassigns it to new_obs only after this call).

        Returns (pds_obs_batch, returns_batch), flattened across
        (env, branch) in that nesting order - shape
        (n_envs*ve_batch_size, PDS_OBS_DIM) / (n_envs*ve_batch_size,).
        """
        n_envs = env.num_envs
        n_branches = self.ve_batch_size
        n_total = n_envs * n_branches

        raw_next_states = np.zeros((n_total, 4), dtype=np.float32)
        tile_masks = np.zeros((n_total, config.N_TILES), dtype=np.float32)
        r_random_batch = np.zeros(n_total, dtype=np.float32)

        k = 0
        for env_idx in range(n_envs):
            info = infos[env_idx]
            # Delta_theta^t/Delta_phi^t: action-independent (function of
            # the REAL observed viewport and the prediction only), so
            # shared across every branch of this env - same reasoning as
            # h_next below. Recomputed from already-exposed info fields
            # rather than adding yet another info field for it.
            delta_theta_t = abs(info["actual_theta"] - info["predicted_theta"])
            delta_phi_t = abs(info["actual_phi"] - info["predicted_phi"])
            for b in range(n_branches):
                agent_tile_mask_b, power_watts_b = pds.unpack_action(virtual_actions_np[b, env_idx])
                phys = pds.compute_virtual_branch_physics(
                    agent_tile_mask_b=agent_tile_mask_b, power_watts_b=power_watts_b,
                    mandatory_tile_mask=info["mandatory_tile_mask"],
                    rd_data=self._ve_rd_data, video_array_index=config.TRAINING_VIDEO_ARRAY_INDEX,
                    gop_index=info["gop_index"], enhanced_level=self._ve_enhanced_level,
                    Z_t=info["Z_t"], h_t=info["h_t"],
                    actual_theta_t=info["actual_theta"], actual_phi_t=info["actual_phi"],
                    mu_d=mu_d, mu_p=mu_p, mu_b=mu_b,
                )
                raw_next_states[k] = pds.build_raw_next_pds_state(
                    phys["Z_next_b"], delta_theta_t, delta_phi_t, info["h_next"],
                )
                tile_masks[k] = phys["tile_mask_b"].astype(np.float32)
                r_random_batch[k] = phys["r_random_b"]
                k += 1

        # One batched normalize_obs() call over all n_envs*ve_batch_size
        # branches (not one per branch) - VecNormalize.normalize_obs() is
        # stateless/read-only (never calls .update()), same guarantee
        # eval.py already relies on, so this can't leak into VecNormalize's
        # running training statistics.
        next_states_norm = env.normalize_obs(raw_next_states)

        pds_obs_batch = np.zeros((n_total, pds.PDS_OBS_DIM), dtype=np.float32)
        k = 0
        for env_idx in range(n_envs):
            obs_t = self._last_obs[env_idx]
            for b in range(n_branches):
                pds_obs_batch[k] = pds.build_pds_observation(obs_t, next_states_norm[k], tile_masks[k])
                k += 1

        # One batched predict_values() call for V_old(omega^{t+1,(b)})
        # across every branch - unlike real steps (which reuse the next
        # stored step's already-computed value), virtual branches have no
        # such shortcut available (omega^{t+1,(b)} isn't any real stored
        # step's own observation), so every branch needs its own forward
        # pass; batching them into one call is still far cheaper than
        # n_total separate calls.
        with th.no_grad():
            v_old_batch = self.policy.predict_values(
                obs_as_tensor(next_states_norm, self.device)
            ).cpu().numpy().flatten()

        # Terminal masking, shared per real env across all its branches
        # (v3: "the fixed H-step terminal flag is shared") - termination
        # here is purely time-based (environment.py: next_t >= len(trace)),
        # never action-dependent, so every hypothetical branch at env_idx
        # inherits that env's REAL terminated flag exactly; "truncated"
        # has no separate case for virtual branches (see
        # compute_virtual_branch_physics docstring) since there is no
        # VecEnv auto-reset/true_next_obs distinction to guard against
        # for a state this function constructed itself.
        terminated_per_branch = np.repeat(terminateds.astype(bool), n_branches)
        v_old_batch = np.where(terminated_per_branch, 0.0, v_old_batch)

        returns_batch = r_random_batch + self.gamma * v_old_batch
        return pds_obs_batch.astype(np.float32), returns_batch.astype(np.float32)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, ordinary_value_losses = [], []
        clip_fractions = []

        # Snapshot the real PDS pairs in flat (N, dim)/(N,) form NOW, before
        # Pass 1's rollout_buffer.get() call below flattens
        # pds_observations/pds_returns in place as an internal side effect
        # (RolloutBuffer.get()'s own swap_and_flatten-on-first-call
        # behavior) - capturing an independent copy here means Pass 2 below
        # never depends on that side effect or on Pass 1 having run first.
        real_pds_obs = self.rollout_buffer.to_torch(
            self.rollout_buffer.swap_and_flatten(self.rollout_buffer.pds_observations)
        )
        # .flatten(): pds_returns is 2D (buffer_size, n_envs) going in -
        # swap_and_flatten's own "pad to 3D if under 3 dims" rule (see its
        # source) turns that into (N, 1), not (N,), which would otherwise
        # silently broadcast against pds_values' (N,) shape in the
        # F.mse_loss calls below (confirmed: this exact bug was live and
        # producing a PyTorch broadcasting UserWarning + wrong loss values
        # until caught here).
        real_pds_returns = self.rollout_buffer.to_torch(
            self.rollout_buffer.swap_and_flatten(self.rollout_buffer.pds_returns)
        ).flatten()

        # --- Pass 1: actor + ordinary critic V, real rollout data only ---
        # (v3 Algorithm 2 line 21: "update pi_theta, V on real rollout
        # data"). The PDS critic Vtilde is trained entirely separately
        # below (Pass 2) - deliberately NOT folded into this same loss/
        # backward pass. zero_grad() immediately before every backward()
        # (unchanged from before VE) means clip_grad_norm_ below only
        # ever sees gradients from THIS pass's own loss, regardless of
        # whether it's scoped to self.policy.parameters() or narrower -
        # this is what keeps Pass 2's (possibly larger, real+virtual-
        # pooled) PDS critic gradient from silently shrinking the actor's
        # clipped step via a shared global norm, a real risk that was
        # flagged and is resolved by this per-pass zero_grad discipline,
        # not by the two losses having separate parameters alone.
        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

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
                ordinary_value_losses.append(ordinary_value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * ordinary_value_loss

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

        # --- Pass 2: PDS critic Vtilde, pooled real+virtual pairs ---
        # (v3 Algorithm 2 line 22: "Fit Vtilde on all real and virtual PDS
        # pairs from this rollout"). Real pairs come from the SAME
        # rollout_buffer (flattened across buffer_size*n_envs, exactly
        # what rollout_data.pds_observations/pds_returns drew from in the
        # pre-VE code); virtual pairs (possibly none, if VE is disabled
        # or no VE event fired) come from get_virtual_pairs(). Pooled and
        # shuffled BEFORE minibatching, so F.mse_loss's per-minibatch mean
        # weights every pair - real or virtual - equally; no separate
        # virtual-loss coefficient. Runs for the same self.n_epochs, but
        # as its own independent shuffle/minibatch loop - a virtual pair
        # is never required to land in the same minibatch as the real
        # step it came from.
        virtual_pds_obs, virtual_pds_returns = self.rollout_buffer.get_virtual_pairs()

        if virtual_pds_obs is not None:
            pooled_obs = th.cat([real_pds_obs, virtual_pds_obs], dim=0)
            pooled_returns = th.cat([real_pds_returns, virtual_pds_returns], dim=0)
        else:
            pooled_obs, pooled_returns = real_pds_obs, real_pds_returns
        n_virtual = 0 if virtual_pds_obs is None else virtual_pds_obs.shape[0]
        n_pooled = pooled_obs.shape[0]

        pds_value_losses = []
        for epoch in range(self.n_epochs):
            perm = th.randperm(n_pooled, device=pooled_obs.device)
            for start in range(0, n_pooled, self.batch_size):
                idx = perm[start:start + self.batch_size]
                pds_values = self.policy.predict_pds_values(pooled_obs[idx]).flatten()
                pds_value_loss = F.mse_loss(pooled_returns[idx], pds_values)
                pds_value_losses.append(pds_value_loss.item())

                self.policy.optimizer.zero_grad()
                pds_value_loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

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
        if self.ve_enabled:
            self.logger.record("ve/n_virtual_pairs", n_virtual)
            self.logger.record("ve/virtual_fraction", n_virtual / n_pooled if n_pooled else 0.0)
        self.logger.record("train/clip_range", clip_range)

        # PDS variance-reduction evidence (PDSRolloutBuffer._compute_variance_diagnostics):
        # is delta_pds actually lower-variance than the ordinary TD residual
        # on the same rollout? Recorded on EVERY train() call from rollout 1
        # so SB3's CSVOutputFormat never has to rewrite progress.csv for a
        # newly-appearing key mid-run. The plain-PPO baseline never reaches
        # this code path (it uses stock PPO.train()), so no shared-key
        # coupling between the two methods.
        for name, val in self.rollout_buffer.last_diagnostics.items():
            self.logger.record(f"pds_diag/{name}", val)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
