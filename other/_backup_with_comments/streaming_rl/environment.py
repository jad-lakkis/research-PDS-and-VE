"""
environment.py - the gymnasium Env: TileStreamingEnv.

Ties together data_loader, channel_model, layer_model, and viewport
into the actual RL loop, following the timing breakdown already worked
out in other/environment_design.md Sec. 4 and the post-decision-state
(PDS) formalism from the problem formulation (eqs. 10-20):

  reset():
    - pick a random valid user trace for config.TRAINING_VIDEO_ARRAY_INDEX
    - Z^0 = 0, bootstrap prediction = actual position at gop 0 (so the
      bootstrap Delta^{-1} is exactly 0, not just initialized to 0),
      sample h^0
    - return omega^0 = (Z^0, Delta^{-1}_theta, Delta^{-1}_phi, h^0)

  step(action) at internal step index t (self._gop_index):
    - unpack action into a per-tile enhancement mask x^t and a discrete
      power level -> P^t (watts)
    - compute A^t (layer_model, using the CURRENT gop's data), R^t
      (channel_model, using h^t - already known from the previous step/
      reset), D^t, Z^{t+1}  <- this whole block is the PDS omega~^t,
      known immediately after the action, before any randomness resolves
    - REVEAL the actual viewport for time t (same t as the action - not
      t+1) by reading trace[frame_index_for_gop(t)]; compute C^t_cov via
      viewport.coverage(x^t, actual_theta_t, actual_phi_t), and
      Delta^t = |actual_t - predicted_t| (predicted_t was computed at
      the END of the previous step, using the "last viewport as
      prediction" rule: theta_hat^t = theta_tilde^{t-1})
    - r^t = r^t_known + r^t_random (eqs. 18-20)
    - sample h^{t+1}, compute theta_hat^{t+1} = theta_tilde^t (this
      step's just-revealed actual) for use at the START of the next step
    - return omega^{t+1}, r^t, terminated, truncated, info

Action space layout (see module docstring in __init__ for why this is
a flat MultiDiscrete rather than a Dict/Tuple): a length-(N_TILES+1)
array. The first N_TILES entries are each in {0,1} (x^t_i - enhance
tile i or not). The last entry is in {0, ..., N_POWER_LEVELS-1} and
selects a linearly-spaced power level between 0 and P_max (watts).

Scope note: this first implementation does NOT yet force the predicted
viewport's tiles to be enhanced (eq. 2's "predicted viewport tiles are
assigned x_i=1") - the agent currently has a fully free choice over all
64 tiles. That constraint is not implemented yet; flagged here rather
than silently deviating from the formulation without saying so.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import config
from streaming_rl import data_loader, channel_model, layer_model, viewport


class TileStreamingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # Loaded ONCE, at construction (environment_design.md Sec. 4,
        # "loaded once" bucket) - never modified afterward.
        bundle = data_loader.load_training_video_bundle()
        self._rd_data = bundle["rd_data"]
        self._valid_traces = bundle["valid_traces"]   # list of (user_slot, Nx4 array)
        assert len(self._valid_traces) > 0, "no valid HMD traces found for the training video"

        # observation = (Z^t, Delta^{t-1}_theta, Delta^{t-1}_phi, h^t)
        # Z and h are left unbounded above (>=0) - the formulation places
        # no ceiling on either (eq. 15 only floors Z at 0; h has no
        # stated upper bound). Delta is an absolute difference of two
        # angles in [-180,180], so its max possible value is 360.
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, 360.0, 360.0, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

        # Flat MultiDiscrete instead of a Dict/Tuple action space: SB3's
        # PPO (the algorithm this project is heading toward) does not
        # support Dict/Tuple *action* spaces natively, only Box/
        # Discrete/MultiDiscrete/MultiBinary. A single MultiDiscrete
        # keeps this compatible without redesigning later.
        self.action_space = spaces.MultiDiscrete([2] * config.N_TILES + [config.N_POWER_LEVELS])

        self._P_max_watts = channel_model.dbm_to_watts(config.P_MAX_DBM)

        # Per-episode state, set in reset()
        self._trace = None
        self._gop_index = None
        self._Z = None
        self._predicted_theta = None
        self._predicted_phi = None
        self._h = None

    # -- internal helpers -------------------------------------------------

    def _frame_idx(self, gop_index: int) -> int:
        return layer_model.frame_index_for_gop(gop_index)

    def _actual_theta_phi(self, gop_index: int) -> tuple:
        yaw, pitch, roll, ts = self._trace[self._frame_idx(gop_index)]
        return viewport.yaw_pitch_to_theta_phi(yaw, pitch)

    def _unpack_action(self, action) -> tuple:
        action = np.asarray(action)
        tile_mask = action[: config.N_TILES].astype(bool)
        power_level_idx = int(action[config.N_TILES])
        power_watts = (power_level_idx / (config.N_POWER_LEVELS - 1)) * self._P_max_watts
        return tile_mask, power_watts

    def _observation(self, Z, delta_theta, delta_phi, h) -> np.ndarray:
        return np.array([Z, delta_theta, delta_phi, h], dtype=np.float32)

    # -- gymnasium API ------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # sets self.np_random per gymnasium convention

        trace_idx = self.np_random.integers(0, len(self._valid_traces))
        _, self._trace = self._valid_traces[trace_idx]

        self._gop_index = 0
        self._Z = 0.0

        actual_theta_0, actual_phi_0 = self._actual_theta_phi(0)
        # Bootstrap: assume a perfect first prediction, so Delta^{-1}=0
        # by construction (design choice - environment_design.md Sec. 2).
        self._predicted_theta = actual_theta_0
        self._predicted_phi = actual_phi_0
        delta_theta_prev = abs(actual_theta_0 - self._predicted_theta)  # == 0.0
        delta_phi_prev = abs(actual_phi_0 - self._predicted_phi)        # == 0.0

        self._h = channel_model.sample_channel_gain(self.np_random_as_generator())

        obs = self._observation(self._Z, delta_theta_prev, delta_phi_prev, self._h)
        info = {}
        return obs, info

    def step(self, action):
        t = self._gop_index
        tile_mask, power_watts = self._unpack_action(action)

        # --- known immediately after the action (the PDS omega~^t) ---
        a_result = layer_model.compute_A_t(self._rd_data, config.TRAINING_VIDEO_ARRAY_INDEX,
                                            t, tile_mask, enhanced_level=1)
        A_t = a_result["A_t"]
        R_t = channel_model.transmission_rate(power_watts, self._h)

        T0 = config.T0_SEC
        if self._Z + T0 * R_t >= A_t:
            D_t = 0.0
        else:
            D_t = T0 * (1.0 - (self._Z + T0 * R_t) / A_t)
        Z_next = max(self._Z + T0 * R_t - A_t, 0.0)

        r_known = (
            -config.BETA_S * (D_t / T0)
            - config.BETA_P * (power_watts / self._P_max_watts)
            - config.BETA_B * (tile_mask.sum() / config.N_TILES)
        )

        # --- reveal the random outcome for time t ---
        actual_theta_t, actual_phi_t = self._actual_theta_phi(t)
        coverage_t = viewport.coverage(tile_mask, actual_theta_t, actual_phi_t)
        delta_theta_t = abs(actual_theta_t - self._predicted_theta)
        delta_phi_t = abs(actual_phi_t - self._predicted_phi)
        r_random = config.BETA_Q * coverage_t

        r_t = r_known + r_random

        # --- prepare state for the NEXT step ---
        h_next = channel_model.sample_channel_gain(self.np_random_as_generator())
        predicted_theta_next = actual_theta_t   # "last viewport as prediction"
        predicted_phi_next = actual_phi_t

        next_t = t + 1
        terminated = self._frame_idx(next_t) >= len(self._trace)

        self._Z = Z_next
        self._h = h_next
        self._predicted_theta = predicted_theta_next
        self._predicted_phi = predicted_phi_next
        self._gop_index = next_t

        obs = self._observation(Z_next, delta_theta_t, delta_phi_t, h_next)
        info = {
            "A_t": A_t,
            "A_base": a_result["A_base"],
            "A_enh": a_result["A_enh"],
            "R_t": R_t,
            "D_t": D_t,
            "coverage": coverage_t,
            "n_enhanced_tiles": int(tile_mask.sum()),
            "power_watts": power_watts,
            "r_known": r_known,
            "r_random": r_random,
        }
        return obs, r_t, terminated, False, info

    def np_random_as_generator(self) -> np.random.Generator:
        """gymnasium's self.np_random (set by super().reset(seed=...)) is
        already a numpy Generator in modern gymnasium versions - this
        thin wrapper exists so channel_model's `rng: np.random.Generator`
        parameter has one clear call site, in case that ever changes.
        """
        return self.np_random
