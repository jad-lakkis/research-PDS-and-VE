"""
pds_buffer.py - PDSRolloutBuffer: SB3's RolloutBuffer, extended with the
extra per-step fields PPO+PDS needs (the 68-dim PDS observation
omega~^t, the known/random reward split, and per-step
terminated/truncated/true_next_obs bookkeeping - PDS design plan,
Section 8/Phase B).

compute_pds_returns_and_advantage() replaces GAE with the one-step PDS
advantage/targets (Section 4/6 of the plan) - a genuinely different
formula, not a variant of GAE, so this is a separate method, not an
override of the base class's compute_returns_and_advantage() (which is
simply never called anywhere in the PPO+PDS path).
"""

from collections.abc import Generator
from typing import NamedTuple

import numpy as np
import torch as th

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize

from streaming_rl.pds import PDS_OBS_DIM


class PDSRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    pds_observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    ordinary_returns: th.Tensor   # y^t   - target for the ordinary critic V_phi
    pds_returns: th.Tensor        # y~^t  - target for the PDS critic Vtilde_psi


class PDSRolloutBuffer(RolloutBuffer):
    pds_obs_dim: int = PDS_OBS_DIM

    def reset(self) -> None:
        super().reset()
        self.pds_observations = np.zeros((self.buffer_size, self.n_envs, self.pds_obs_dim), dtype=np.float32)
        self.true_next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.known_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.random_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.terminateds = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.truncateds = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.ordinary_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pds_returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        pds_obs: np.ndarray,
        true_next_obs: np.ndarray,
        known_reward: np.ndarray,
        random_reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> None:
        """Same call as RolloutBuffer.add(), plus every PDS-specific field
        for this same step, at the same self.pos - stored BEFORE delegating
        to the base add() (which is what advances self.pos)."""
        pos = self.pos
        self.pds_observations[pos] = np.array(pds_obs)
        self.true_next_observations[pos] = np.array(true_next_obs)
        self.known_rewards[pos] = np.array(known_reward)
        self.random_rewards[pos] = np.array(random_reward)
        self.terminateds[pos] = np.array(terminated)
        self.truncateds[pos] = np.array(truncated)
        super().add(obs, action, reward, episode_start, value, log_prob)

    def compute_pds_returns_and_advantage(self, policy, last_values: th.Tensor) -> None:
        """Phase B of the PPO+PDS pseudocode. Must be called once, after the
        rollout is fully collected (self.full is True) and BEFORE any policy
        update this rollout - every target below is computed with the
        CURRENT, pre-update critics (under no_grad here); PPOWithPDS.train()
        later recomputes fresh, grad-carrying forward passes for the actual
        loss terms.

        V_next^t = 0                              if terminated at t (true episode end - no bootstrap)
                 = V_phi(true_next_obs^t)          if truncated at t (bootstrap from the TRUE cutoff state,
                                                     not the next episode's freshly auto-reset observation)
                 = V_phi(omega^{t+1})               otherwise (the next stored step's own omega, or the
                                                     end-of-rollout bootstrap value on the buffer's last step)
        """
        assert self.full, "compute_pds_returns_and_advantage() must be called on a full buffer"
        last_values_np = last_values.clone().cpu().numpy().flatten()

        with th.no_grad():
            pds_obs_flat = self.to_torch(self.pds_observations.reshape(-1, self.pds_obs_dim))
            true_next_obs_flat = self.to_torch(self.true_next_observations.reshape(-1, *self.obs_shape))

            v_pds = policy.predict_pds_values(pds_obs_flat).cpu().numpy().reshape(self.buffer_size, self.n_envs)
            v_true_next = policy.predict_values(true_next_obs_flat).cpu().numpy().reshape(self.buffer_size, self.n_envs)

        # V_phi(omega^{t+1}) for every non-terminal, non-truncated step: the
        # NEXT stored step's own v_t (self.values, already computed in Phase
        # A by the SAME frozen policy - policy is frozen for the whole
        # rollout, so no need to recompute). Shifted by one; the final row
        # is overwritten with the end-of-rollout bootstrap value passed in
        # (last_values) - same idiom as RolloutBuffer.compute_returns_and_advantage's
        # own last-step case.
        v_next_otherwise = np.empty_like(self.values)
        v_next_otherwise[:-1] = self.values[1:]
        v_next_otherwise[-1] = last_values_np

        terminated_mask = self.terminateds.astype(bool)
        truncated_mask = self.truncateds.astype(bool)
        otherwise_mask = ~terminated_mask & ~truncated_mask

        v_next = np.zeros_like(self.values)   # terminated_mask rows stay 0 - no bootstrap past a true episode end
        v_next[truncated_mask] = v_true_next[truncated_mask]
        v_next[otherwise_mask] = v_next_otherwise[otherwise_mask]

        self.pds_returns = self.random_rewards + self.gamma * v_next   # y~^t
        self.ordinary_returns = self.known_rewards + v_pds             # y^t
        self.advantages = self.ordinary_returns - self.values          # A^t_PDS = y^t - v_t

    def get(self, batch_size: int | None = None) -> Generator[PDSRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            _tensor_names = [
                "observations", "pds_observations", "actions", "values",
                "log_probs", "advantages", "ordinary_returns", "pds_returns",
            ]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: VecNormalize | None = None) -> PDSRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.pds_observations[batch_inds],
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.ordinary_returns[batch_inds].flatten(),
            self.pds_returns[batch_inds].flatten(),
        )
        return PDSRolloutBufferSamples(*tuple(map(self.to_torch, data)))
