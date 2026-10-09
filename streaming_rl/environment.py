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

Link model (constructor argument "link"):
  "rician" (default) - R^t = W log2(1 + P^t h^t / (W N0)), h^t sampled
    i.i.d. each step (channel_model). Everything above, unchanged.
  "lumos5g" - R^t is MEASURED throughput (streaming_rl/lumos5g.py; each
    episode is a contiguous 36-s window of one Lumos5G run, drawn uniformly
    over every (run, start second) of the chosen split, or fixed through
    reset(options={"window": (run, start)})). Power cannot change a
    measured rate, so it is fixed (config.FIXED_POWER_WATTS) and the action
    is the N_TILES tile bits alone. The buffer/stall/coverage equations are
    unchanged - they only use R^t. Observation:
    (Z^t, Delta_theta^{t-1}, Delta_phi^{t-1}, R^t, R^{t-1}, ..., R^{t-5})
    with Z and R in Mbit(/s); R^t is known before the decision, exactly
    as h^t is in the rician model.

Tile frame (constructor argument "tile_frame"):
  "absolute" (default) - action bit i is physical tile i. Everything above.
  "relative" - the agent's tile bits are in viewport-relative coordinates
    (streaming_rl/tile_frame.py): rows physical, columns counted from the
    PREDICTED viewport's column. step() maps them to physical tiles (a
    cyclic column shift fixed by the prediction) before anything else, so
    transmitted tiles, buffer, stall, reward and constraints are computed on
    physical tiles exactly as before. The observation carries the viewing
    context the relative bits need:
    (Z^t, signed last head move dtheta, dphi, predicted elevation phi_hat,
     predicted azimuth's offset inside its tile column, sin/cos of the
     predicted azimuth theta_hat, video time t/T, link part)
    with link part = R^t, ..., R^{t-5} (Lumos5G, 14 numbers in total) or
    h^t (rician, 9 in total).

Lumos5G-only options (both off by default):
  obs_forced_bits - append the bits of this second's forced stream
    A_forced^t = A_base^t + sum of the predicted-viewport tiles' enhancement
    (Mbit) and the ratio (Z^t + T0 R^t) / A_forced^t: whether the link and
    buffer can carry the forced tiles right now, the quantity a
    link-adaptive rule decides on.
  episode_stall_floor - report, on the terminal step, the episode's own
    predicted-viewport-only J_D (info["J_D_floor"]): the stall no tile
    choice could avoid in this (viewer, window), used by the training
    multiplier to compare only the avoidable part with the allowance.
  obs_buffer - "linear" (default): buffer in Mbit, forced-bits ratio as is;
    "log": log(1 + Z / config.LUMOS5G_OBS_BUFFER_REF_BITS) and log(1 + ratio).
    The buffer has no cap, so 5G stretches fill it to tens of Gbit and, on
    the linear scale, VecNormalize squashes the near-empty region where the
    stall decision is made. Only the observation changes.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import config
from streaming_rl import data_loader, channel_model, layer_model, lumos5g, tile_frame, viewport
from streaming_rl import pds as pds_module


class TileStreamingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, enhanced_level: int = None, trace_indices: list = None, bundle: dict = None,
                 log_counterfactual: bool = False, link: str = "rician", lumos_split: str = "train",
                 lumos_scale: float = 1.0, lumos_data: dict = None, tile_frame: str = "absolute",
                 obs_forced_bits: bool = False, episode_stall_floor: bool = False,
                 obs_buffer: str = "linear"):
        super().__init__()

        # Eval-only diagnostic: when True, step() additionally reports what
        # the viewport PSNR/coverage WOULD have been if only the mandatory
        # predicted-viewport tiles had been enhanced (i.e. without any of
        # the agent's own extra tile picks). Isolates what the learned
        # policy's extra tiles actually buy, versus the reference paper's
        # viewport-enlargement mechanism alone.
        #
        # PSNR/coverage ONLY - it does NOT re-simulate A_t, the buffer, or
        # stalls under that counterfactual mask, so it is not a full
        # "EL-alone policy" rollout, just the per-step quality delta.
        #
        # Default False so the training hot path (1728 steps/rollout) pays
        # only one boolean test per step; enabled on the eval envs built in
        # streaming_rl/eval.py::HeldOutTraceEvalCallback._init_callback,
        # which run 3x36=108 steps per eval.
        self._log_counterfactual = log_counterfactual

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

        assert link in ("rician", "lumos5g"), f"unknown link model {link!r}"
        assert tile_frame in ("absolute", "relative"), f"unknown tile frame {tile_frame!r}"
        self._link = link
        self._tile_frame = tile_frame
        if link == "lumos5g" or tile_frame == "relative":
            # Every trace of a video has the same length (Runner: 1080 frames
            # = 36 GOPs). Lumos5G windows are sized to it (plus one extra
            # second for the terminal observation's R^{t+1}); the relative
            # frame's observation reports video time as t / this.
            self._ep_len = max(-(-len(tr) // config.GOP_SIZE_FRAMES) for _, tr in self._valid_traces)
        if link == "lumos5g":
            self._lumos = lumos_data if lumos_data is not None else lumos5g.load_bundle()
            self._lumos_scale = float(lumos_scale)
            self._n_hist = config.LUMOS5G_N_HISTORY
            self._start_runs, self._start_secs = lumos5g.all_starts(
                self._lumos["runs"], self._lumos["splits"][lumos_split], self._ep_len)
            self._window = None
            self._R = None
            # observation = (Z^t, Delta^{t-1}_theta, Delta^{t-1}_phi, R^t, R^{t-1}, ..., R^{t-n_hist})
            n_obs = 3 + 1 + self._n_hist
            self.observation_space = spaces.Box(
                low=np.zeros(n_obs, dtype=np.float32),
                high=np.array([np.inf, 360.0, 360.0] + [np.inf] * (1 + self._n_hist), dtype=np.float32),
                dtype=np.float32,
            )
            self.action_space = spaces.MultiDiscrete([2] * config.N_TILES)
        else:
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

        if tile_frame == "relative":
            # (Z, dtheta, dphi, phi_hat, column offset, sin theta_hat, cos theta_hat, t/T) + link part;
            # the action space is unchanged in size - only what each tile bit refers to changes
            n_link = 1 + self._n_hist if link == "lumos5g" else 1
            half = config.TILE_WIDTH_DEG / 2
            self.observation_space = spaces.Box(
                low=np.array([0.0, -180.0, -180.0, -90.0, -half, -1.0, -1.0, 0.0] + [0.0] * n_link, dtype=np.float32),
                high=np.array([np.inf, 180.0, 180.0, 90.0, half, 1.0, 1.0, 1.0] + [np.inf] * n_link, dtype=np.float32),
                dtype=np.float32,
            )

        assert link == "lumos5g" or not (obs_forced_bits or episode_stall_floor), (
            "obs_forced_bits / episode_stall_floor need the Lumos5G link (R^t known before the decision)")
        self._obs_forced_bits = bool(obs_forced_bits)
        self._episode_stall_floor = bool(episode_stall_floor)
        assert obs_buffer in lumos5g.OBS_BUFFER_MODES, f"unknown obs_buffer {obs_buffer!r}"
        assert link == "lumos5g" or obs_buffer == "linear", "obs_buffer='log' needs the Lumos5G link"
        self._obs_buffer = obs_buffer
        self._A_forced = None
        self._J_D_floor = None
        if self._obs_forced_bits:
            n = self.observation_space.shape[0]
            self.observation_space = spaces.Box(
                low=np.concatenate([self.observation_space.low, [0.0, 0.0]]).astype(np.float32),
                high=np.concatenate([self.observation_space.high, [np.inf, np.inf]]).astype(np.float32),
                dtype=np.float32,
            )
            self._ratio_index = n + 1

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
        # Delegates to pds.unpack_action() - single source of truth, also
        # used by virtual-experience hypothetical-action decoding, so the
        # two can never silently drift apart.
        return pds_module.unpack_action(action)

    def _observation(self, Z, delta_theta, delta_phi, h) -> np.ndarray:
        # h is pre-scaled here ONLY - every other consumer of h (channel_model,
        # info["..."], etc.) still uses the raw physical value. See
        # config.H_OBSERVATION_PRESCALE's comment / decision_log.md #15.
        return np.array([Z, delta_theta, delta_phi, h * config.H_OBSERVATION_PRESCALE], dtype=np.float32)

    def _lumos_observation(self, Z, delta_theta, delta_phi, t) -> np.ndarray:
        # self._R holds n_hist samples before the window, then the window:
        # R^t is self._R[n_hist + t]; R^t, R^{t-1}, ..., R^{t-n_hist} is
        # that slice reversed. Z/R scaled to Mbit(/s) here only - every
        # physics quantity (info, buffer, stall) stays in bits.
        s = config.LUMOS5G_OBS_SCALE
        link = self._R[t:t + self._n_hist + 1][::-1] * s
        return np.concatenate([[lumos5g.buffer_obs(Z, self._obs_buffer), delta_theta, delta_phi], link]).astype(np.float32)

    def _relative_observation(self, Z, dtheta, dphi, pred_theta, pred_phi, t, h) -> np.ndarray:
        # Viewing context for viewport-relative tile bits (see module docstring).
        # dtheta/dphi are SIGNED (dtheta wrapped to [-180, 180)): the last head
        # move = last actual - its prediction (the viewport before it).
        if self._link == "lumos5g":
            zs = config.LUMOS5G_OBS_SCALE
            link = self._R[t:t + self._n_hist + 1][::-1] * zs
            z_obs = lumos5g.buffer_obs(Z, self._obs_buffer)
        else:
            zs = 1.0
            link = [h * config.H_OBSERVATION_PRESCALE]
            z_obs = Z * zs
        th = np.deg2rad(pred_theta)
        view = [z_obs, dtheta, dphi, pred_phi, tile_frame.column_offset_deg(pred_theta),
                np.sin(th), np.cos(th), t / self._ep_len]
        return np.concatenate([view, link]).astype(np.float32)

    def _forced_stream(self):
        """A_forced^t for every t of the episode: base layer + predicted-viewport
        tiles (prediction = last actual viewport, as in step()) - fixed by the
        viewer trace, independent of the agent."""
        A = np.zeros(self._ep_len)
        for t in range(self._ep_len):
            th, ph = self._actual_theta_phi(max(t - 1, 0))
            mask = viewport.mandatory_tile_mask(th, ph)
            A[t] = layer_model.compute_A_t(self._rd_data, config.TRAINING_VIDEO_ARRAY_INDEX, t, mask,
                                           enhanced_level=self._enhanced_level)["A_t"]
        return A

    def _viewport_only_stall(self):
        """J_D of sending only the forced stream over this episode's window."""
        Z, J = 0.0, 0.0
        T0 = config.T0_SEC
        for t in range(self._ep_len):
            R, A = self._R[self._n_hist + t], self._A_forced[t]
            if Z + T0 * R < A:
                J += config.DISCOUNT_FACTOR_LAMBDA ** t * (1.0 - (Z + T0 * R) / A)
                Z = 0.0
            else:
                Z = Z + T0 * R - A
        return J

    def _forced_part(self, Z, t):
        k = min(t, self._ep_len - 1)
        A = self._A_forced[k]
        return np.array([A * config.LUMOS5G_OBS_SCALE,
                         lumos5g.ratio_obs((Z + config.T0_SEC * self._R[self._n_hist + t]) / A, self._obs_buffer)],
                        dtype=np.float32)

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

        if self._link == "lumos5g":
            window = (options or {}).get("window")
            if window is None:
                k = self.np_random.integers(0, len(self._start_runs))
                window = (int(self._start_runs[k]), int(self._start_secs[k]))
            self._window = (int(window[0]), int(window[1]))
            self._R = lumos5g.rate_window(self._lumos["runs"], self._window[0], self._window[1],
                                          self._ep_len + 1, self._lumos_scale, self._n_hist)
            self._h = None
            obs = self._lumos_observation(self._Z, delta_theta_prev, delta_phi_prev, 0)
        else:
            self._h = channel_model.sample_channel_gain(self.np_random_as_generator())
            obs = self._observation(self._Z, delta_theta_prev, delta_phi_prev, self._h)
        if self._tile_frame == "relative":
            obs = self._relative_observation(self._Z, 0.0, 0.0, actual_theta_0, actual_phi_0, 0, self._h)
        if self._obs_forced_bits or self._episode_stall_floor:
            self._A_forced = self._forced_stream()
            self._J_D_floor = self._viewport_only_stall() if self._episode_stall_floor else None
        if self._obs_forced_bits:
            obs = np.concatenate([obs, self._forced_part(self._Z, 0)]).astype(np.float32)
        info = {}
        return obs, info

    def step(self, action):
        t = self._gop_index
        # Snapshot the PRE-step Z^t/h^t before the end-of-step block below
        # overwrites self._Z/self._h with Z_next/h_next - same reasoning/bug
        # shape as pred_theta_used/pred_phi_used below (self._Z/self._h
        # would otherwise read as the NEXT step's values by the time info
        # is built). Needed for virtual-experience hypothetical-branch
        # physics recomputation, which requires the real, already-known
        # Z^t/h^t - not available anywhere else in raw physical units
        # (info["Z"] is the POST-step Z~^{t+1}, and self._last_obs in
        # collect_rollouts is VecNormalize-normalized, not raw).
        Z_t_used = self._Z
        h_t_used = self._h
        agent_tile_mask, power_watts = self._unpack_action(action)

        # eq. 2: predicted-viewport tiles are always enhanced ("assigned
        # x_i=1"), on top of whatever the agent's own action picked -
        # the agent only has free choice over the remaining tiles. Uses
        # the PREDICTED (theta, phi), not the actual one - self._predicted_theta/
        # phi still hold the prediction that was live when the agent
        # chose this action (not overwritten until the end of this method).
        # Snapshot the prediction THIS step actually used, before the
        # end-of-step block below overwrites self._predicted_theta/_phi
        # with this step's actual angles ("last viewport as prediction").
        # Without this snapshot, anything reading self._predicted_theta
        # down in the info dict would get the NEXT step's prediction, and
        # the logged prediction error would be identically zero forever.
        pred_theta_used = self._predicted_theta
        pred_phi_used = self._predicted_phi

        # Relative tile frame: the agent's bits -> physical tiles, using the
        # prediction known now. From here on everything is physical.
        tile_shift = None
        if self._tile_frame == "relative":
            tile_shift = tile_frame.column_shift(pred_theta_used)
            agent_tile_mask = tile_frame.rel_to_abs(agent_tile_mask, tile_shift)

        mandatory_tile_mask = viewport.mandatory_tile_mask(pred_theta_used, pred_phi_used)
        tile_mask = agent_tile_mask | mandatory_tile_mask

        # --- known immediately after the action (the PDS omega~^t) ---
        a_result = layer_model.compute_A_t(self._rd_data, config.TRAINING_VIDEO_ARRAY_INDEX,
                                            t, tile_mask, enhanced_level=self._enhanced_level)
        A_t = a_result["A_t"]
        if self._link == "lumos5g":
            R_t = float(self._R[self._n_hist + t])   # measured, known before the decision
        else:
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
        h_next = None if self._link == "lumos5g" else channel_model.sample_channel_gain(self.np_random_as_generator())
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

        if self._tile_frame == "relative":
            obs = self._relative_observation(Z_next, tile_frame.wrap_deg(actual_theta_t - pred_theta_used),
                                             actual_phi_t - pred_phi_used, predicted_theta_next,
                                             predicted_phi_next, next_t, h_next)
        elif self._link == "lumos5g":
            obs = self._lumos_observation(Z_next, delta_theta_t, delta_phi_t, next_t)
        else:
            obs = self._observation(Z_next, delta_theta_t, delta_phi_t, h_next)
        if self._obs_forced_bits:
            obs = np.concatenate([obs, self._forced_part(Z_next, next_t)]).astype(np.float32)
        info = {
            "A_t": A_t,
            "A_base": a_result["A_base"],
            "A_enh": a_result["A_enh"],
            "R_t": R_t,
            "D_t": D_t,
            # Buffer level after this step. NOTE: bits, not seconds -
            # Z_next = Z + T0*R_t - A_t with R_t a bitrate and A_t a
            # bit-volume, even though MMSP'25's own notation reads as a
            # playout-seconds buffer. Purely additive, for diagnostics.
            "Z": Z_next,
            # Raw, pre-step Z^t/h^t (physical units, NOT VecNormalize-
            # normalized) - for virtual-experience hypothetical-branch
            # physics recomputation only; every other consumer of buffer/
            # channel state uses "Z" (post-step) or the normalized obs.
            "Z_t": Z_t_used,
            "h_t": h_t_used,
            # Env-internal GOP index THIS step used for its video-data
            # lookup (compute_A_t/coverage/_actual_theta_phi all key off
            # this) - cycles 0..H-1 per episode, distinct from the
            # rollout-local step counter collect_rollouts tracks for the
            # VE-trigger condition. Needed so a virtual branch's
            # compute_A_t call indexes the exact same GOP the real step did.
            "gop_index": t,
            # Raw next-step channel gain h^{t+1} - already sampled above
            # (independent of the action, reused identically across every
            # virtual-experience branch), but otherwise only consumed
            # internally to build the next observation/self._h.
            "h_next": h_next,
            "coverage": coverage_t,
            "viewport_psnr_db": viewport_psnr_db,
            "tile_mask": tile_mask.astype(np.uint8),  # which of the 64 tiles, for Stage 8 per-tile tracking
            "n_enhanced_tiles": int(tile_mask.sum()),
            "n_mandatory_tiles": int(mandatory_tile_mask.sum()),
            "n_agent_extra_tiles": int((agent_tile_mask & ~mandatory_tile_mask).sum()),
            # The two masks SEPARATELY (tile_mask above is their OR, so the
            # agent's own picks aren't recoverable from it alone), plus the
            # geometry needed to place them: which direction we PREDICTED
            # the viewport would be (what mandatory_tile_mask was built
            # from) versus where it ACTUALLY turned out to be. Both masks
            # are freshly-allocated arrays already, so these are reference
            # stores - no copy, no new computation on the hot path.
            "agent_tile_mask": agent_tile_mask,
            "mandatory_tile_mask": mandatory_tile_mask,
            "predicted_theta": float(pred_theta_used),
            "predicted_phi": float(pred_phi_used),
            "actual_theta": float(actual_theta_t),
            "actual_phi": float(actual_phi_t),
            "power_watts": power_watts,
            "r_known": r_known,
            "r_random": r_random,
            "cost_D": cost_D_t,
            "cost_P": cost_P_t,
            "cost_B": cost_B_t,
        }
        if self._link == "lumos5g" or self._tile_frame == "relative":
            # For virtual experience: every part of the next observation except
            # Z (head move, next prediction, video time, link) is the same for
            # every hypothetical branch - only its buffer differs.
            info["next_obs_raw"] = obs.copy()
            info["z_obs_scale"] = config.LUMOS5G_OBS_SCALE if self._link == "lumos5g" else 1.0
        if self._link == "lumos5g":
            # measured rate: the same for every hypothetical action
            info["R_t_exogenous"] = R_t
            info["lumos_window"] = self._window
            info["obs_buffer"] = self._obs_buffer   # how a virtual next state must encode its buffer/ratio
        if self._obs_forced_bits:
            # the next observation's ratio depends on the buffer, so a virtual
            # branch must recompute it (pds.raw_next_state_from_obs)
            info["obs_ratio_index"] = self._ratio_index
            info["next_R"] = float(self._R[self._n_hist + next_t])
            info["next_A_forced"] = float(self._A_forced[min(next_t, self._ep_len - 1)])
        if self._tile_frame == "relative":
            # agent_tile_mask/tile_mask above are PHYSICAL; the PDS critic and
            # virtual experience also need the shift and the executed mask in
            # the agent's relative coordinates
            info["tile_shift"] = tile_shift
            info["tile_mask_rel"] = tile_frame.abs_to_rel(tile_mask, tile_shift).astype(np.uint8)

        # Eval-only counterfactual (see __init__'s log_counterfactual):
        # quality if ONLY the mandatory predicted-viewport tiles had been
        # enhanced. Same call as the real viewport_psnr_db above, differing
        # only in the mask, and evaluated against the same ACTUAL angles -
        # so (viewport_psnr_db - viewport_psnr_db_el_only) is exactly the
        # quality the agent's extra tile picks bought on this step.
        if self._log_counterfactual:
            info["viewport_psnr_db_el_only"] = layer_model.compute_viewport_psnr_db(
                self._rd_data, config.TRAINING_VIDEO_ARRAY_INDEX, t, mandatory_tile_mask,
                self._enhanced_level, actual_theta_t, actual_phi_t,
            )
            info["coverage_el_only"] = viewport.coverage(
                mandatory_tile_mask, actual_theta_t, actual_phi_t
            )

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
            if self._episode_stall_floor:
                info["J_D_floor"] = self._J_D_floor
        return obs, r_t, terminated, False, info

    def np_random_as_generator(self) -> np.random.Generator:
        """gymnasium's self.np_random (set by super().reset(seed=...)) is
        already a numpy Generator in modern gymnasium versions - this
        thin wrapper exists so channel_model's `rng: np.random.Generator`
        parameter has one clear call site, in case that ever changes.
        """
        return self.np_random
