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
    - force every tile touching the PREDICTED viewport into x^t (eq. 2:
      "The predicted viewport tiles are sent as enhanced tiles. In
      addition, the agent selects any other tile...") via
      viewport.mandatory_tile_mask - the agent's own raw pick is OR'd
      with this mandatory mask before anything else is computed, so the
      agent only has free choice over tiles beyond the predicted viewport
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
    - return omega^{t+1}, r^t, terminated, truncated, info - info also
      carries the three normalized constraint costs (cost_D, cost_P,
      cost_B) consumed by streaming_rl/lagrangian.py, computed once here
      as the single source of truth rather than re-derived downstream

Action space layout (see module docstring in __init__ for why this is
a flat MultiDiscrete rather than a Dict/Tuple): a length-(N_TILES+1)
array. The first N_TILES entries are each in {0,1} (x^t_i - enhance
tile i or not, before the mandatory-viewport OR above is applied). The
last entry is in {0, ..., N_POWER_LEVELS-1} and selects a linearly-
spaced power level between 0 and P_max (watts).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import config
from streaming_rl import data_loader, channel_model, layer_model, viewport


class TileStreamingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, enhanced_level: int = None, trace_indices: list = None, bundle: dict = None):
        super().__init__()

        # Which k level (config.qp_array_index_for_enhancement_level)
        # enhanced tiles are sent at. Defaults to
        # config.INITIAL_ENHANCEMENT_LEVELS[1] (the confirmed non-base
        # level - currently k=5, see that constant's comment for the
        # el_selection_test.py evidence behind it), overridable so
        # experiments/el_selection_test.py can instantiate the SAME
        # environment logic under different candidate EL values without
        # duplicating step()'s behavior in a separate script.
        self._enhanced_level = (
            config.INITIAL_ENHANCEMENT_LEVELS[1] if enhanced_level is None else enhanced_level
        )

        # Loaded ONCE, at construction (environment_design.md Sec. 4,
        # "loaded once" bucket) - never modified afterward. `bundle=` lets
        # a caller (e.g. train_ppo.py, building one train-env and one
        # eval-env instance) share the already-loaded rd_data (measured
        # ~388MB) instead of paying for scipy.io.loadmat twice for no
        # benefit - rd_data is only ever read, never mutated.
        bundle = bundle if bundle is not None else data_loader.load_training_video_bundle()
        self._rd_data = bundle["rd_data"]
        all_traces = bundle["valid_traces"]   # list of (user_slot, Nx4 array)

        # trace_indices: POSITIONS into bundle["valid_traces"] (0..11 for
        # the 12 valid Runner traces) - NOT hn.mat user_slot values (those
        # are [0,2,4,5,6,7,8,9,10,11,13,25], confirmed different). None
        # (default) reproduces the original behavior: every valid trace
        # is in play. Used to restrict a given env instance to a train-
        # only or eval-only subset (config.EVAL_TRACE_INDICES) - reset()
        # needs no further changes, it already only touches
        # self._valid_traces, whatever subset that is.
        self._valid_traces = all_traces if trace_indices is None else [all_traces[i] for i in trace_indices]
        assert len(self._valid_traces) > 0, f"no valid traces for trace_indices={trace_indices}"

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

        # Per-episode discounted-return accumulators (eqs. 5-8, literal
        # form): J_X = sum_{t=0}^{T-1} lambda^t * X^t. Every episode
        # reaches terminated=True at exactly 36 steps (never truncated),
        # so this is an exact Monte Carlo sum computed as the episode
        # plays out - no cost/value-function critic needed. Exposed in
        # info only on the terminal step (see step()).
        self._J_Q = None
        self._J_D = None
        self._J_P = None
        self._J_B = None

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
        power_fraction = (power_level_idx / (config.N_POWER_LEVELS - 1)) * config.POWER_LEVEL_MAX_FRACTION
        power_watts = power_fraction * self._P_max_watts
        return tile_mask, power_watts

    def _observation(self, Z, delta_theta, delta_phi, h) -> np.ndarray:
        # h is pre-scaled here ONLY - every other consumer of h (channel_model,
        # info["..."], etc.) still uses the raw physical value. See
        # config.H_OBSERVATION_PRESCALE's comment / decision_log.md #15.
        return np.array([Z, delta_theta, delta_phi, h * config.H_OBSERVATION_PRESCALE], dtype=np.float32)

    # -- gymnasium API ------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # sets self.np_random per gymnasium convention

        trace_idx = self.np_random.integers(0, len(self._valid_traces))
        _, self._trace = self._valid_traces[trace_idx]

        self._gop_index = 0
        self._Z = 0.0
        self._J_Q = 0.0
        self._J_D = 0.0
        self._J_P = 0.0
        self._J_B = 0.0

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
        agent_tile_mask, power_watts = self._unpack_action(action)

        # eq. 2: predicted-viewport tiles are always enhanced ("assigned
        # x_i=1"), on top of whatever the agent's own action picked -
        # the agent only has free choice over the remaining tiles. Uses
        # the PREDICTED (theta, phi), not the actual one - self._predicted_theta/
        # phi still hold the prediction that was live when the agent
        # chose this action (not overwritten until the end of this method).
        mandatory_tile_mask = viewport.mandatory_tile_mask(self._predicted_theta, self._predicted_phi)
        tile_mask = agent_tile_mask | mandatory_tile_mask

        # --- known immediately after the action (the PDS omega~^t) ---
        a_result = layer_model.compute_A_t(self._rd_data, config.TRAINING_VIDEO_ARRAY_INDEX,
                                            t, tile_mask, enhanced_level=self._enhanced_level)
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
        viewport_psnr_db = layer_model.compute_viewport_psnr_db(
            self._rd_data, config.TRAINING_VIDEO_ARRAY_INDEX, t, tile_mask,
            self._enhanced_level, actual_theta_t, actual_phi_t,
        )
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

        # Normalized per-step constraint costs (eqs. 6-8) - single source
        # of truth for both info["cost_*"] (per-step, consumed by
        # streaming_rl.lagrangian.LagrangianRewardWrapper's reward
        # reconstruction) and the discounted per-episode sums below.
        cost_D_t = D_t / T0
        cost_P_t = power_watts / self._P_max_watts
        cost_B_t = float(tile_mask.sum()) / config.N_TILES

        # Literal per-episode discounted accumulation (eqs. 5-8):
        # J_X = sum_{t=0}^{T-1} lambda^t * X^t. Re-zeroed each reset().
        discount_pow_t = config.DISCOUNT_FACTOR_LAMBDA ** t
        self._J_Q += discount_pow_t * coverage_t
        self._J_D += discount_pow_t * cost_D_t
        self._J_P += discount_pow_t * cost_P_t
        self._J_B += discount_pow_t * cost_B_t

        obs = self._observation(Z_next, delta_theta_t, delta_phi_t, h_next)
        info = {
            "A_t": A_t,
            "A_base": a_result["A_base"],
            "A_enh": a_result["A_enh"],
            "R_t": R_t,
            "D_t": D_t,
            "coverage": coverage_t,
            "viewport_psnr_db": viewport_psnr_db,
            "tile_mask": tile_mask.astype(np.uint8),  # which of the 64 tiles, for Stage 8 per-tile tracking
            "n_enhanced_tiles": int(tile_mask.sum()),
            "n_mandatory_tiles": int(mandatory_tile_mask.sum()),
            "n_agent_extra_tiles": int((agent_tile_mask & ~mandatory_tile_mask).sum()),
            "power_watts": power_watts,
            "r_known": r_known,
            "r_random": r_random,
            "cost_D": cost_D_t,
            "cost_P": cost_P_t,
            "cost_B": cost_B_t,
        }
        if terminated:
            # Complete (not partial) discounted returns, only on the
            # terminal step - same idiom as SB3's own Monitor wrapper
            # only adding info["episode"] at episode end. Consumed by
            # train_ppo.py's LagrangianMultiplierCallback (J_D/J_P/J_B)
            # and streaming_rl/eval.py's model-selection logic (all four).
            info["J_Q"] = self._J_Q
            info["J_D"] = self._J_D
            info["J_P"] = self._J_P
            info["J_B"] = self._J_B
        return obs, r_t, terminated, False, info

    def np_random_as_generator(self) -> np.random.Generator:
        """gymnasium's self.np_random (set by super().reset(seed=...)) is
        already a numpy Generator in modern gymnasium versions - this
        thin wrapper exists so channel_model's `rng: np.random.Generator`
        parameter has one clear call site, in case that ever changes.
        """
        return self.np_random
