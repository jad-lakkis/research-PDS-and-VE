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
from streaming_rl import pds, tile_frame
from streaming_rl.pds_buffer import PDSRolloutBuffer
from streaming_rl.pds_policy import PDSActorCriticPolicy


def _sample_multicategorical(distribution, generator: th.Generator) -> th.Tensor:
    """One fresh sample from an already-configured MultiCategoricalDistribution
    (SB3 stable_baselines3.common.distributions), drawn from `generator`
    instead of torch's default/global RNG stream.

    Replicates MultiCategoricalDistribution.sample()'s own behavior
    (th.stack([dist.sample() for dist in self.distribution], dim=1)) but
    substitutes torch.multinomial's generator= argument, which
    Categorical.sample()'s own public API does not expose. Verified to
    produce identical shape/dtype to the original .sample() (each
    sub-Categorical's .probs has shape (n_envs, num_categories);
    multinomial(..., num_samples=1) gives (n_envs, 1), squeezed to
    (n_envs,) then stacked across the 65 tile/power dimensions to
    (n_envs, 65) - matching MultiDiscrete([2]*N_TILES + [N_POWER_LEVELS])
    exactly, same as the real action's own shape.
    """
    samples = [
        th.multinomial(cat.probs, num_samples=1, replacement=True, generator=generator).squeeze(-1)
        for cat in distribution.distribution
    ]
    return th.stack(samples, dim=1)


class PPOWithPDS(PPO):
    def __init__(self, policy=PDSActorCriticPolicy, env=None, pds_net_arch: list = None,
                 ve_enabled: bool = False, ve_batch_size: int = 0, ve_period: int = 10,
                 pds_two_pass: bool = False, ve_shared_pass: bool = False,
                 pds_critic_lambda: float = None, **kwargs):
        assert policy is PDSActorCriticPolicy, (
            "PPOWithPDS always uses PDSActorCriticPolicy - it's not a swappable "
            "policy= argument like stock PPO's (needed so predict_pds_values() exists)"
        )
        policy_kwargs = dict(kwargs.pop("policy_kwargs", None) or {})
        # PDS observation size follows the env's observation (68 for the
        # rician link, 73 for Lumos5G - see pds.pds_obs_dim). env is None only
        # inside PPOWithPDS.load(), which restores policy_kwargs from the save.
        obs_dim = env.observation_space.shape[0] if env is not None else 4
        policy_kwargs.setdefault("pds_obs_dim", pds.pds_obs_dim(obs_dim))
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

        # Whether train() uses the two-pass structure (separate gradient-
        # clip norm and independent minibatch shuffle for the PDS critic,
        # vs one combined loss/backward/clip/step shared with the actor).
        # ve_enabled normally FORCES this True regardless of pds_two_pass's
        # own value - combining VE's pooled real+virtual gradient into a
        # single clip norm with the actor is the exact contamination risk
        # the split exists to prevent. ve_shared_pass is an explicit,
        # deliberate override of that safety default: True folds VE's
        # pooled data into the SAME single combined pass as the actor and
        # V instead (the "old PDS + VE" experiment - does VE help the
        # already-working single-pass procedure, at the cost of
        # reopening the shared-clip-norm risk on purpose, to measure
        # whether it actually matters in practice). Meaningless without
        # ve_enabled=True; ignored if VE is off (pds_two_pass alone still
        # selects the structure in that case). With VE off, pds_two_pass
        # is a genuinely independent, experimentable choice on its own:
        # does the training-PROCEDURE change alone (not virtual data)
        # affect results? All three structures apply the same vf_coef
        # weighting to the PDS-critic loss (see _train_pds_critic/
        # _train_policy_and_value) - never a deliberate difference.
        # Lambda for the ORDINARY critic's target (see
        # PDSRolloutBuffer.compute_pds_returns_and_advantage). None = same as
        # gae_lambda (original behaviour, Experiment X); 0 = one-step target
        # r_known + Vtilde(omega~), removing the two critics' mutual error
        # amplification while the actor keeps the gae_lambda trace. Restored
        # by load() like the other attributes; checkpoints saved before this
        # existed keep the None set here.
        self.pds_critic_lambda = None if pds_critic_lambda is None else float(pds_critic_lambda)

        self.ve_shared_pass = self.ve_enabled and bool(ve_shared_pass)
        if self.ve_shared_pass:
            self.pds_two_pass_training = False
        else:
            self.pds_two_pass_training = self.ve_enabled or bool(pds_two_pass)

        # Dedicated RNG stream for VE's hypothetical-action sampling -
        # NEVER touches torch's default/global generator that the real
        # action sequence depends on, so there is nothing to snapshot/
        # restore and no possibility of a later real draw reusing bits a
        # virtual draw already consumed (a real, if narrow, correlation
        # risk with the snapshot/restore approach this replaces).
        # Reproducible per-run (offset from self.seed, distinct from it)
        # when a seed is given; OS-entropy-seeded otherwise, matching
        # self.seed's own None-means-unseeded convention.
        ve_seed = None if self.seed is None else (self.seed + 1_000_003)
        self._ve_generator = th.Generator(device=self.device)
        if ve_seed is not None:
            self._ve_generator.manual_seed(ve_seed)

    def _setup_model(self) -> None:
        super()._setup_model()
        # PDSRolloutBuffer sizes its PDS-observation array from a class-level
        # default (68, the rician link); match the policy's actual input size.
        # Runs on fresh construction and on load() alike.
        if self.rollout_buffer.pds_obs_dim != self.policy.pds_obs_dim:
            self.rollout_buffer.pds_obs_dim = self.policy.pds_obs_dim
            self.rollout_buffer.reset()

    def _excluded_save_params(self) -> list:
        # _ve_rd_data caches the ENTIRE rd.mat (all 15 videos' RD tables,
        # not just the training one - data_loader.load_rd_data()) directly
        # on self (see collect_rollouts) so it isn't re-fetched via
        # env.get_attr() every rollout. Without this exclusion it gets
        # pickled into data.pkl on every model.save() call - the actual
        # cause of VE checkpoints being ~640MB instead of ~0.4MB. Safe to
        # exclude: collect_rollouts re-fetches both attributes fresh from
        # the env on first use after load/resume (hasattr(self,
        # "_ve_rd_data") is False on a freshly constructed/loaded model
        # regardless), identical content either way - not needed for
        # correctness, unlike _ve_generator (kept in the save - its RNG
        # state carrying across --resume-from is a verified, relied-on
        # property, not an oversight).
        return super()._excluded_save_params() + ["_ve_rd_data", "_ve_enhanced_level"]

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
            actions = actions.cpu().numpy()

            # self.policy.action_dist stays correctly configured for THIS
            # step's real forward pass until the NEXT self.policy(...)
            # call (start of the next loop iteration) - nothing between
            # here and _build_virtual_pairs() below touches it, so VE's
            # hypothetical-action sampling can safely happen later, inside
            # _build_virtual_pairs(), once the mandatory-tile mask (needed
            # to check for duplicates correctly - see that method) is
            # actually known. This ordering used to matter more: an
            # earlier version drew samples HERE, before env.step(), purely
            # because the RNG mechanism at the time (snapshot/restore on
            # torch's shared default generator) needed precise timing.
            # self._ve_generator (see __init__) is fully independent of
            # that shared generator, so there is no longer any timing
            # constraint on when VE sampling happens at all.

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
            pds_obs = np.zeros((env.num_envs, self.policy.pds_obs_dim), dtype=np.float32)
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

                # executed mask in the agent's own coordinates (relative tile frame) or physical
                pds_obs[idx] = pds.build_pds_observation(self._last_obs[idx], true_next_obs[idx],
                                                         info.get("tile_mask_rel", info["tile_mask"]))

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
                pds_obs_v, returns_v, ve_stats = self._build_virtual_pairs(
                    env, infos, terminateds, mu_d, mu_p, mu_b,
                )
                rollout_buffer.add_virtual_pairs(pds_obs_v, returns_v)
                rollout_buffer.add_ve_sample_stats(*ve_stats)

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
            critic_lambda=getattr(self, "pds_critic_lambda", None),
        )

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def _sample_distinct_virtual_actions(self, env_idx, mandatory_tile_mask, real_tile_mask_packed,
                                          n_branches, max_retries=20, tile_shift=None):
        """Draw n_branches hypothetical actions from self.policy.action_dist
        (via self._ve_generator - see collect_rollouts for why this can
        safely happen at any point, not just immediately after the real
        forward pass), rejecting and redrawing any candidate whose
        post-mandatory-merge tile_mask duplicates an already-accepted
        branch or the real executed action, up to max_retries per slot.
        Falls back to accepting a duplicate if max_retries is exhausted
        (bounds worst-case compute; can happen once the policy's
        effective support is smaller than n_branches).

        Measured need for this (best-checkpoint estimate, 2026-09-30):
        policy-distribution draws collapse from 100% distinct at
        initialization to ~71-77% distinct by rollout ~1000 as the
        policy sharpens, with ~20% exactly matching the real action by
        then (contributing zero new information to Vtilde's training set
        beyond what the real transition already provides).

        Draws for ALL envs every attempt (self.policy.action_dist is
        batched over n_envs) but only this env's own slice is used/
        checked - some redraws are wasted work for envs that already
        have enough distinct branches when n_envs>1. Harmless
        (self._ve_generator draws are cheap - no forward pass) and
        every real launch uses n_envs=1 anyway, where it's exact.

        Returns a list of n_branches raw action arrays (still needs
        pds.unpack_action() + mandatory-tile OR-in, same as any other
        raw action - this only decides WHICH actions to keep).

        tile_shift: the real step's relative-tile-frame shift (None = the
        agent's bits are physical tiles). Duplicates are judged on PHYSICAL
        masks, like the real executed one.
        """
        accepted_actions = []
        accepted_masks_packed = {real_tile_mask_packed}
        for _ in range(n_branches):
            candidate_action = None
            for _attempt in range(max_retries):
                sample = _sample_multicategorical(self.policy.action_dist, self._ve_generator)
                action_np = sample[env_idx].cpu().numpy()
                agent_mask, _ = pds.unpack_action(action_np)
                if tile_shift is not None:
                    agent_mask = tile_frame.rel_to_abs(agent_mask, tile_shift)
                merged = (agent_mask | mandatory_tile_mask).astype(bool)
                packed = np.packbits(merged).tobytes()
                if packed not in accepted_masks_packed:
                    candidate_action = action_np
                    accepted_masks_packed.add(packed)
                    break
            if candidate_action is None:
                candidate_action = action_np  # retries exhausted - accept the last draw anyway
            accepted_actions.append(candidate_action)
        return accepted_actions

    def _build_virtual_pairs(self, env, infos, terminateds, mu_d, mu_p, mu_b):
        """Recompute physics and assemble (pds_obs, target) for every
        virtual branch this VE-triggering step, across all
        n_envs x ve_batch_size branches (v3 Algorithm 2 lines 13-18).

        infos/terminateds are THIS step's real per-env info/terminal
        flags - reused, never recomputed per branch (that's the whole
        basis for VE being valid). self._last_obs is still the PRE-step
        observation here (collect_rollouts reassigns it to new_obs only
        after this call).

        Returns (pds_obs_batch, returns_batch, ve_stats), flattened across
        (env, branch) in that nesting order - shape
        (n_envs*ve_batch_size, PDS_OBS_DIM) / (n_envs*ve_batch_size,).
        ve_stats = (n_branches_total, n_unique_total, n_match_real_total),
        summed across every env this call - "how much of the B_VE draws
        is actually new information, once mandatory tiles collapse some
        of them onto the same effective tile_mask" (including collapsing
        onto the real executed action itself). Should now read close to
        100% unique except when the retry cap in
        _sample_distinct_virtual_actions is exhausted - kept as a live
        diagnostic specifically to catch that case, not just historical
        record of the old (pre-rejection-sampling) behavior.
        """
        n_envs = env.num_envs
        n_branches = self.ve_batch_size
        n_total = n_envs * n_branches

        raw_next_states = np.zeros((n_total, self.observation_space.shape[0]), dtype=np.float32)
        tile_masks = np.zeros((n_total, config.N_TILES), dtype=np.float32)      # physical
        pds_masks = np.zeros((n_total, config.N_TILES), dtype=np.float32)       # the PDS critic's coordinates
        r_random_batch = np.zeros(n_total, dtype=np.float32)

        n_unique_total = 0
        n_match_real_total = 0

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
            k_env_start = k

            mandatory_mask_bool = info["mandatory_tile_mask"].astype(bool)
            real_mask_packed = np.packbits(info["tile_mask"].astype(bool)).tobytes()
            tile_shift = info.get("tile_shift")   # relative tile frame: same shift for every branch
            branch_actions = self._sample_distinct_virtual_actions(
                env_idx, mandatory_mask_bool, real_mask_packed, n_branches, tile_shift=tile_shift,
            )

            for action_np in branch_actions:
                agent_tile_mask_b, power_watts_b = pds.unpack_action(action_np)
                if tile_shift is not None:
                    agent_tile_mask_b = tile_frame.rel_to_abs(agent_tile_mask_b, tile_shift)
                phys = pds.compute_virtual_branch_physics(
                    agent_tile_mask_b=agent_tile_mask_b, power_watts_b=power_watts_b,
                    mandatory_tile_mask=info["mandatory_tile_mask"],
                    rd_data=self._ve_rd_data, video_array_index=config.TRAINING_VIDEO_ARRAY_INDEX,
                    gop_index=info["gop_index"], enhanced_level=self._ve_enhanced_level,
                    Z_t=info["Z_t"], h_t=info["h_t"],
                    actual_theta_t=info["actual_theta"], actual_phi_t=info["actual_phi"],
                    mu_d=mu_d, mu_p=mu_p, mu_b=mu_b, R_t=info.get("R_t_exogenous"),
                )
                if info.get("next_obs_raw") is not None:
                    # trace-driven link: next state = real next observation with this branch's buffer
                    raw_next_states[k] = pds.raw_next_state_from_obs(
                        info["next_obs_raw"], phys["Z_next_b"], info["z_obs_scale"],
                    )
                else:
                    raw_next_states[k] = pds.build_raw_next_pds_state(
                        phys["Z_next_b"], delta_theta_t, delta_phi_t, info["h_next"],
                    )
                tile_masks[k] = phys["tile_mask_b"].astype(np.float32)
                pds_masks[k] = (tile_frame.abs_to_rel(phys["tile_mask_b"], tile_shift) if tile_shift is not None
                                else phys["tile_mask_b"]).astype(np.float32)
                r_random_batch[k] = phys["r_random_b"]
                k += 1

            # Duplicate bookkeeping for THIS env's n_branches slice, post-
            # mandatory-OR (matches what actually enters training, not the
            # raw pre-mandatory agent picks) - packed to hashable rows via
            # np.packbits so set() can dedupe them cheaply.
            env_masks = tile_masks[k_env_start:k].astype(bool)
            packed = [np.packbits(row).tobytes() for row in env_masks]
            n_unique_total += len(set(packed))
            n_match_real_total += sum(1 for p in packed if p == real_mask_packed)

        # One batched normalize_obs() call over all n_envs*ve_batch_size
        # branches (not one per branch) - VecNormalize.normalize_obs() is
        # stateless/read-only (never calls .update()), same guarantee
        # eval.py already relies on, so this can't leak into VecNormalize's
        # running training statistics.
        next_states_norm = env.normalize_obs(raw_next_states)

        pds_obs_batch = np.zeros((n_total, self.policy.pds_obs_dim), dtype=np.float32)
        k = 0
        for env_idx in range(n_envs):
            obs_t = self._last_obs[env_idx]
            for b in range(n_branches):
                pds_obs_batch[k] = pds.build_pds_observation(obs_t, next_states_norm[k], pds_masks[k])
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
        ve_stats = (n_total, n_unique_total, n_match_real_total)
        return pds_obs_batch.astype(np.float32), returns_batch.astype(np.float32), ve_stats

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

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

        # self.pds_two_pass_training selects the structure (see __init__
        # for exactly what differs and why VE forces this True). False
        # reproduces the original single-combined-pass formula exactly -
        # verified bit-for-bit against an independently-reconstructed
        # copy of that formula, from the same starting weights AND
        # optimizer momentum state (that path's PDS-critic loss IS
        # vf_coef-weighted, matching Exp1's real formula). True is also
        # available WITHOUT VE (pds_two_pass=True, ve_enabled=False) as a
        # deliberate, standalone experiment: does the separate-clip-norm/
        # independent-shuffle structure alone change results, apart from
        # virtual data? That path's PDS-critic loss is deliberately
        # UNWEIGHTED (see _train_pds_critic's docstring) - matches every
        # two-pass run already collected, past and future, on one
        # consistent procedure, rather than the single-pass default's
        # weighting.
        if self.pds_two_pass_training:
            pass1 = self._train_policy_and_value(include_pds_loss=False)
            pass2 = self._train_pds_critic(real_pds_obs, real_pds_returns)
        elif self.ve_shared_pass:
            # "Old PDS + VE": VE's pooled real+virtual pairs fold into the
            # SAME single combined pass as the actor/V, on purpose (see
            # __init__ for why this is an explicit override, not the
            # default). include_pds_loss's per-minibatch loop draws a
            # freshly-shuffled batch_size-sized slice of this pooled set
            # each iteration, riding along with the actor's own real-data
            # minibatch loop - so Vtilde is capped at the ACTOR's own
            # iteration count (n_epochs * ceil(real_size/batch_size)),
            # touching fewer total pooled samples per epoch than the
            # two-pass structure's own dedicated pass over the full
            # pooled set would give it. That's an inherent, honest
            # consequence of sharing one pass, not an arbitrary choice.
            virtual_pds_obs, virtual_pds_returns = self.rollout_buffer.get_virtual_pairs()
            if virtual_pds_obs is not None:
                pooled_pds_obs = th.cat([real_pds_obs, virtual_pds_obs], dim=0)
                pooled_pds_returns = th.cat([real_pds_returns, virtual_pds_returns], dim=0)
            else:
                pooled_pds_obs, pooled_pds_returns = real_pds_obs, real_pds_returns
            pass1 = self._train_policy_and_value(
                include_pds_loss=True, pooled_pds_obs=pooled_pds_obs, pooled_pds_returns=pooled_pds_returns,
            )
            n_virtual = 0 if virtual_pds_obs is None else virtual_pds_obs.shape[0]
            pass2 = {
                "pds_value_losses": pass1["pds_value_losses"],
                "n_virtual": n_virtual, "n_pooled": pooled_pds_obs.shape[0],
            }
        else:
            pass1 = self._train_policy_and_value(include_pds_loss=True)
            pass2 = {"pds_value_losses": pass1["pds_value_losses"], "n_virtual": 0, "n_pooled": 0}

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.ordinary_returns.flatten()
        )

        self.logger.record("train/entropy_loss", np.mean(pass1["entropy_losses"]))
        self.logger.record("train/policy_gradient_loss", np.mean(pass1["pg_losses"]))
        self.logger.record("train/value_loss", np.mean(pass1["ordinary_value_losses"]))
        self.logger.record("train/pds_value_loss", np.mean(pass2["pds_value_losses"]))
        self.logger.record("train/approx_kl", np.mean(pass1["approx_kl_divs"]))
        self.logger.record("train/clip_fraction", np.mean(pass1["clip_fractions"]))
        self.logger.record("train/loss", pass1["last_loss"])
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if self.ve_enabled:
            self.logger.record("ve/n_virtual_pairs", pass2["n_virtual"])
            self.logger.record(
                "ve/virtual_fraction", pass2["n_virtual"] / pass2["n_pooled"] if pass2["n_pooled"] else 0.0
            )
            # How much of ve_batch_size draws is actually new information,
            # once mandatory tiles are OR'd in - pooled (not averaged-of-
            # averages) across every env/VE-event this rollout. Directly
            # informs whether a larger ve_batch_size is worth its extra
            # compute: unique_fraction near 1.0 means draws rarely
            # collapse onto each other; match_real_fraction is how often a
            # draw collapses onto the real executed action specifically
            # (those add zero new (state, target) pairs beyond what the
            # real step already contributes).
            ve_dup = self.rollout_buffer.get_ve_duplicate_summary()
            self.logger.record("ve/unique_fraction", ve_dup["unique_fraction"])
            self.logger.record("ve/match_real_fraction", ve_dup["match_real_fraction"])
        self.logger.record("train/clip_range", pass1["clip_range"])
        if pass1["clip_range_vf"] is not None:
            self.logger.record("train/clip_range_vf", pass1["clip_range_vf"])

        # PDS variance-reduction evidence (PDSRolloutBuffer._compute_variance_diagnostics):
        # is delta_pds actually lower-variance than the ordinary TD residual
        # on the same rollout? Recorded on EVERY train() call from rollout 1
        # so SB3's CSVOutputFormat never has to rewrite progress.csv for a
        # newly-appearing key mid-run. The plain-PPO baseline never reaches
        # this code path (it uses stock PPO.train()), so no shared-key
        # coupling between the two methods.
        for name, val in self.rollout_buffer.last_diagnostics.items():
            self.logger.record(f"pds_diag/{name}", val)

    def _train_policy_and_value(self, include_pds_loss: bool,
                                 pooled_pds_obs: th.Tensor = None, pooled_pds_returns: th.Tensor = None) -> dict:
        """Actor + ordinary critic V, real rollout data only (v3 Algorithm
        2 line 21: "update pi_theta, V on real rollout data").

        include_pds_loss selects between the two training procedures:
        - False (VE enabled, two-pass): the PDS critic Vtilde is trained
          entirely separately (_train_pds_critic), deliberately NOT
          folded into this loss/backward pass. zero_grad() immediately
          before every backward() means clip_grad_norm_ below only ever
          sees gradients from THIS pass's own loss - this is what keeps
          Pass 2's (real+virtual-pooled) PDS critic gradient from
          silently shrinking the actor's clipped step via a shared
          global norm. Separately verified (test, not just this
          reasoning): actor/ordinary-critic parameters are bit-for-bit
          unchanged by a _train_pds_critic()-only call, and vice versa.
        - True, pooled_pds_obs=None (VE disabled): reproduces the
          ORIGINAL, pre-VE single-combined-pass formula exactly - loss =
          policy_loss + ent_coef*entropy_loss + vf_coef*(ordinary_value_loss
          + pds_value_loss), one zero_grad/backward/clip/step per
          minibatch, PDS critic interleaved with the actor using the
          SAME minibatch shuffle (rollout_data.pds_observations/
          pds_returns, drawn from the real rollout data only).
        - True, pooled_pds_obs given (VE enabled, ve_shared_pass=True,
          the deliberate "old PDS + VE" override - see __init__): same
          single combined pass - the actor's OWN iteration count and
          minibatch size are untouched, exactly matching what plain
          old-PDS always gave it, no inflation. Vtilde's own per-
          iteration slice of the pooled real+virtual set is sized
          independently and LARGER (pds_batch_size, computed below from
          n_pooled and the actor's own iteration count) so that, over
          the same number of iterations, Vtilde covers the full pooled
          set once per epoch (reshuffled every epoch) - full VE exposure
          without changing anything about the actor's own training
          regimen.
        """
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf is not None else None

        entropy_losses = []
        pg_losses, ordinary_value_losses = [], []
        pds_value_losses = [] if include_pds_loss else None
        clip_fractions = []
        approx_kl_divs = []
        last_loss = None

        use_pooled_pds = pooled_pds_obs is not None
        n_pooled = pooled_pds_obs.shape[0] if use_pooled_pds else 0
        if use_pooled_pds:
            # Vtilde's own per-iteration batch is sized independently of
            # the actor's (self.batch_size) - the actor's iteration count/
            # minibatch size stays exactly what plain old-PDS always used
            # (no inflation of its own training), but Vtilde's slice is
            # made BIGGER so that, over the SAME number of iterations, it
            # covers the full pooled set once per epoch instead of only
            # ~real_size/n_pooled of it. Ceil-div throughout so the last
            # iteration's slightly-larger draw still fits.
            real_size = self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs
            n_iterations_per_epoch = -(-real_size // self.batch_size)
            pds_batch_size = -(-n_pooled // n_iterations_per_epoch)

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            pooled_perm = th.randperm(n_pooled, device=pooled_pds_obs.device) if use_pooled_pds else None
            pooled_cursor = 0
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

                if include_pds_loss:
                    if use_pooled_pds:
                        # pds_batch_size (computed above the epoch loop),
                        # NOT the actor's own minibatch size - this is
                        # what gives Vtilde full pooled-set coverage per
                        # epoch without changing the actor's own
                        # iteration count or minibatch size at all.
                        idx = pooled_perm[pooled_cursor:pooled_cursor + pds_batch_size]
                        if idx.shape[0] < pds_batch_size:
                            # Epoch's shuffle exhausted (last iteration,
                            # since n_iterations*pds_batch_size >= n_pooled
                            # by construction) - pad with a fresh random
                            # draw rather than error.
                            extra = th.randint(
                                0, n_pooled, (pds_batch_size - idx.shape[0],), device=pooled_pds_obs.device,
                            )
                            idx = th.cat([idx, extra])
                        pooled_cursor += pds_batch_size
                        pds_values = self.policy.predict_pds_values(pooled_pds_obs[idx]).flatten()
                        pds_value_loss = F.mse_loss(pooled_pds_returns[idx], pds_values)
                    else:
                        pds_values = self.policy.predict_pds_values(rollout_data.pds_observations).flatten()
                        pds_value_loss = F.mse_loss(rollout_data.pds_returns, pds_values)
                    pds_value_losses.append(pds_value_loss.item())
                    loss = loss + self.vf_coef * pds_value_loss

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
                last_loss = loss.item()

            self._n_updates += 1
            if not continue_training:
                break

        return {
            "entropy_losses": entropy_losses, "pg_losses": pg_losses,
            "ordinary_value_losses": ordinary_value_losses, "clip_fractions": clip_fractions,
            "approx_kl_divs": approx_kl_divs, "last_loss": last_loss,
            "clip_range": clip_range, "clip_range_vf": clip_range_vf,
            "pds_value_losses": pds_value_losses,
        }

    def _train_pds_critic(self, real_pds_obs: th.Tensor, real_pds_returns: th.Tensor) -> dict:
        """Pass 2: PDS critic Vtilde, pooled real+virtual pairs (v3
        Algorithm 2 line 22: "Fit Vtilde on all real and virtual PDS pairs
        from this rollout"). real_pds_obs/real_pds_returns are the SAME
        rollout's real pairs, already flattened by train() before Pass 1
        touched the buffer's internal state; virtual pairs (possibly
        none, if VE is disabled or no VE event fired) come from
        get_virtual_pairs(). Pooled and shuffled BEFORE minibatching, so
        F.mse_loss's per-minibatch mean weights every pair - real or
        virtual - equally; no separate virtual-loss coefficient. Runs for
        the same self.n_epochs, but as its own independent shuffle/
        minibatch loop - a virtual pair is never required to land in the
        same minibatch as the real step it came from. Never touches the
        actor or ordinary critic's loss - see _train_policy_and_value's
        docstring for why a shared optimizer/zero_grad discipline still
        keeps this pass from perturbing them. UNWEIGHTED (no vf_coef) by
        explicit choice, matching the already-running two-pass arms
        exactly (both the 3 no-VE "new PDS" controls and the 12 VE arms
        use weight 1 here) - a deliberate decision to keep every
        two-pass run, past and future, on one consistent procedure,
        rather than "fixing" this to match the single-pass default's
        0.5 weighting (which stays as-is, since it's reproducing Exp1's
        actual original formula, not this one).
        """
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

        return {"pds_value_losses": pds_value_losses, "n_virtual": n_virtual, "n_pooled": n_pooled}
