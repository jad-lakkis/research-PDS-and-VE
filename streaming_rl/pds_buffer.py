"""
pds_buffer.py - PDSRolloutBuffer: SB3's RolloutBuffer, extended with the
extra per-step fields PPO+PDS needs (the 68-dim PDS observation
omega~^t, the known/random reward split, and per-step
terminated/truncated/true_next_obs bookkeeping - PDS design plan,
Section 8/Phase B).

compute_pds_returns_and_advantage() computes the PDS one-step residual
delta_t^PDS = r_known^t + Vtilde_psi(omega~^t) - V_phi(omega^t), then
chains it through the same backward GAE(lambda) recursion Schulman et
al.'s GAE uses on the ordinary TD residual - "PDS-GAE". gae_lambda=0
collapses this to exactly the original one-step PDS advantage (A_t =
delta_t^PDS, no temporal chaining); gae_lambda=0.95 matches baseline
PPO's own horizon (the Run-D ablation: same GAE parameters as baseline,
differing only in whether the one-step residual is estimated
conventionally or through the PDS decomposition). Not an override of
the base class's compute_returns_and_advantage() (never called anywhere
in the PPO+PDS path) since the residual itself is PDS-specific.
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
    # Populated by compute_pds_returns_and_advantage(); empty until then
    # so PPOWithPDS.train() can log defensively without an AttributeError
    # if the call order ever changes.
    last_diagnostics: dict = {}

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

    def compute_pds_returns_and_advantage(self, policy, last_values: th.Tensor, gae_lambda: float) -> None:
        """Phase B of the PPO+PDS pseudocode. Must be called once, after the
        rollout is fully collected (self.full is True) and BEFORE any policy
        update this rollout - every target below is computed with the
        CURRENT, pre-update critics (under no_grad here); PPOWithPDS.train()
        later recomputes fresh, grad-carrying forward passes for the actual
        loss terms.

        gae_lambda has no default - callers must choose explicitly (0 for
        the original one-step PDS advantage, 0.95 to match baseline PPO's
        own GAE horizon).

        V_next^t = 0                              if terminated at t (true episode end - no bootstrap)
                 = V_phi(true_next_obs^t)          if truncated at t (bootstrap from the TRUE cutoff state,
                                                     not the next episode's freshly auto-reset observation)
                 = V_phi(omega^{t+1})               otherwise (the next stored step's own omega, or the
                                                     end-of-rollout bootstrap value on the buffer's last step)

        This V_next/pds_returns machinery is for the PDS CRITIC's own
        training target only (ytilde_t = r_random^t + gamma*V_next, left
        exactly as before by the Run-D change) - it plays no part in the
        actor's advantage, which is self-contained per step through
        Vtilde_psi(omega~^t) and never needs to look at the next stored row.
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

        self.pds_returns = self.random_rewards + self.gamma * v_next   # y~^t - UNCHANGED by Run D

        # delta_t^PDS = r_known^t + Vtilde_psi(omega~^t) - V_phi(omega^t):
        # the PDS one-step residual, fully self-contained per t (unlike the
        # ordinary TD residual, it never needs the next stored step's own
        # value - Vtilde_psi(omega~^t) already stands in for it). This is
        # Run C's entire advantage, and Run D's per-step building block.
        delta_pds = self.known_rewards + v_pds - self.values

        # Backward PDS-GAE recursion (Schulman et al.'s GAE applied to
        # delta_pds instead of the ordinary TD residual):
        #   A_t = delta_t^PDS + gamma*lambda*next_non_terminal*A_{t+1}
        # next_non_terminal is 0 at any terminated OR truncated step -
        # chaining across an episode boundary would mix an unrelated future
        # episode's residuals into this one. At gae_lambda=0 this collapses
        # to exactly A_t = delta_t^PDS (bit-identical to the old one-step
        # formula, Run C); at gae_lambda=0.95 it matches baseline PPO's own
        # GAE horizon (Run D).
        next_non_terminal = otherwise_mask.astype(np.float32)
        advantages = np.zeros_like(self.values)
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            last_gae_lam = delta_pds[step] + self.gamma * gae_lambda * next_non_terminal[step] * last_gae_lam
            advantages[step] = last_gae_lam
        self.advantages = advantages

        # Ordinary critic target follows SB3's own returns=advantages+values
        # convention, so the ordinary critic trains toward the same horizon
        # as the actor. At gae_lambda=0 this is bit-identical to the old
        # y^t = r_known^t + Vtilde_psi(omega~^t), since advantages+values
        # reduces to delta_pds+values = known_rewards+v_pds exactly.
        self.ordinary_returns = self.advantages + self.values          # R^t_PDS-GAE

        self._compute_variance_diagnostics(delta_pds, v_next, next_non_terminal, gae_lambda)

    def _compute_variance_diagnostics(self, delta_pds, v_next, next_non_terminal,
                                      gae_lambda: float) -> None:
        """Evidence for PDS's core claim: that delta_pds is a LOWER-VARIANCE
        estimate than the ordinary TD residual. Computes what ordinary GAE
        would have produced on this very same rollout and compares.

        Uses only arrays already in scope - no extra network forward
        passes, so this costs two vector ops and one backward recursion
        over an already-collected rollout.

        Results land in self.last_diagnostics for PPOWithPDS.train() to
        log. Means are included alongside the stds deliberately: a
        residual that is lower-variance but biased is NOT the claim, so
        the means have to be inspectable too.
        """
        # Ordinary TD residual: the total reward (not the known/random
        # split) plus the same bootstrap v_next the PDS critic target
        # uses, minus the same ordinary-critic baseline.
        delta_ordinary = self.rewards + self.gamma * v_next - self.values

        adv_ordinary = np.zeros_like(self.values)
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            last_gae_lam = (delta_ordinary[step]
                            + self.gamma * gae_lambda * next_non_terminal[step] * last_gae_lam)
            adv_ordinary[step] = last_gae_lam

        a_pds = self.advantages.flatten()
        a_ord = adv_ordinary.flatten()

        def _corr(x, y):
            # np.corrcoef warns and returns nan on a zero-variance input;
            # return nan explicitly instead of emitting a runtime warning
            # every rollout.
            if x.std() == 0.0 or y.std() == 0.0:
                return float("nan")
            return float(np.corrcoef(x, y)[0, 1])

        std_pds = float(delta_pds.std())
        std_ord = float(delta_ordinary.std())

        self.last_diagnostics = {
            "std_delta_pds": std_pds,
            "std_delta_ordinary": std_ord,
            "std_adv_pds": float(a_pds.std()),
            "std_adv_ordinary": float(a_ord.std()),
            # <1 means the PDS residual really is the lower-variance one.
            "std_ratio_pds_over_ord": (std_pds / std_ord) if std_ord != 0.0 else float("nan"),
            "corr_adv_pds_ord": _corr(a_pds, a_ord),
            "sign_agreement": float(np.mean(np.sign(a_pds) == np.sign(a_ord))),
            "mean_delta_pds": float(delta_pds.mean()),
            "mean_delta_ordinary": float(delta_ordinary.mean()),
            # Proves r_known + r_random == r_total exactly. If this drifts
            # from 0 the known/random split is broken and the whole
            # PDS-vs-ordinary comparison above is apples-to-oranges.
            "reward_decomp_max_err": float(
                np.abs(self.known_rewards + self.random_rewards - self.rewards).max()
            ),
        }

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
