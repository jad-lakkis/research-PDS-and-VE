"""
pds.py - pure functions for PPO+PDS (post-decision state): assembling the
68-dim PDS observation omega~^t (eq. 16) and splitting the Lagrangian
reward into its known/random parts (eqs. 18-20), using the LIVE
mu_D/mu_P/mu_B multipliers - NOT environment.py's own info["r_known"]/
["r_random"] fields, which use the fixed BETA_S/BETA_P/BETA_B weights
instead (see the PDS design plan, Section 9 - those fields exist for the
--no-lagrangian baseline path and would silently use the wrong weights
here). No SB3 dependency - both functions operate on plain numpy arrays /
the env's own info dict, so they're testable in isolation.
"""

import numpy as np

import config
from streaming_rl import channel_model, layer_model, viewport

def pds_obs_dim(obs_dim: int) -> int:
    """Z-tilde (1) + every other component of omega^t, reused as-is
    (obs_dim - 1) + tile_mask E^t(x^t) (config.N_TILES)  (eq. 16)."""
    return obs_dim + config.N_TILES


# Rician link: Z-tilde (1) + Delta_theta^{t-1}, Delta_phi^{t-1}, h^t (3) +
# tile_mask (config.N_TILES) = 68. The Lumos5G link observes R^t and its
# history instead of h^t, so its PDS observation is pds_obs_dim(9) = 73.
# Lumos5G with the relative tile frame: the 14-entry observation (viewing
# context + link) with the post-decision buffer, plus the executed mask in
# relative coordinates (environment info["tile_mask_rel"]) = 78 - the
# viewing context stays in, since the same relative mask has different
# encoded sizes and quality effects at different places and times.
PDS_OBS_DIM = pds_obs_dim(4)


def build_pds_observation(obs: np.ndarray, true_next_obs: np.ndarray, tile_mask: np.ndarray) -> np.ndarray:
    """omega~^t = concat(Z~^{t+1}, Delta_theta^{t-1}, Delta_phi^{t-1}, h^t, E^t(x^t))  (eq. 16).
    With the Lumos5G link, h^t is replaced by R^t, R^{t-1}, ..., R^{t-5} -
    same rule: everything after Z in omega^t is reused as-is.

    obs: this step's own 4-dim observation omega^t = (Z^t, Delta_theta^{t-1},
        Delta_phi^{t-1}, h^t) - Delta_theta/Delta_phi/h are reused from here
        AS-IS (PDS design plan, Section 4): same physical quantities at the
        same timestep, unaffected by this step's own action, already
        VecNormalize-consistent with omega^t.
    true_next_obs: the TRUE final observation of this step - info["terminal_observation"]
        if this step ended an episode (terminated OR truncated), else the
        plain next observation unchanged. Only its Z component (index 0) is
        used, as Z~^{t+1} - using the auto-reset observation here instead
        would silently splice in the wrong episode's state.
    tile_mask: the executed tile mask E^t(x^t) (info["tile_mask"]) - the
        mandatory-viewport-forced mask actually sent, not the agent's raw
        pre-mandatory pick.
    """
    obs = np.asarray(obs, dtype=np.float32)
    true_next_obs = np.asarray(true_next_obs, dtype=np.float32)
    tile_mask = np.asarray(tile_mask, dtype=np.float32)
    assert obs.ndim == 1, f"expected a 1-D omega^t, got shape {obs.shape}"
    assert true_next_obs.shape == obs.shape, f"true_next_obs shape {true_next_obs.shape} != omega^t shape {obs.shape}"
    assert tile_mask.shape == (config.N_TILES,), f"expected tile_mask shape ({config.N_TILES},), got {tile_mask.shape}"

    z_tilde = true_next_obs[0:1]
    pre_decision_rest = obs[1:]
    pds_obs = np.concatenate([z_tilde, pre_decision_rest, tile_mask]).astype(np.float32)
    expected = pds_obs_dim(obs.shape[0])
    assert pds_obs.shape == (expected,), f"expected PDS obs shape ({expected},), got {pds_obs.shape}"
    return pds_obs


def unpack_action(action) -> tuple:
    """(tile_mask, power_watts) from a raw MultiDiscrete([2]*N_TILES +
    [N_POWER_LEVELS]) action array. Single source of truth, shared by
    TileStreamingEnv._unpack_action (the real executed action) and
    virtual-experience hypothetical-action sampling - avoids the two
    ever drifting apart. A tiles-only action (the Lumos5G link's
    MultiDiscrete([2]*N_TILES)) means fixed power, config.FIXED_POWER_WATTS.
    """
    action = np.asarray(action)
    tile_mask = action[:config.N_TILES].astype(bool)
    if action.shape[0] == config.N_TILES:
        return tile_mask, config.FIXED_POWER_WATTS
    power_level_idx = int(action[config.N_TILES])
    power_fraction = (power_level_idx / (config.N_POWER_LEVELS - 1)) * config.POWER_LEVEL_MAX_FRACTION
    power_watts = power_fraction * channel_model.dbm_to_watts(config.P_MAX_DBM)
    return tile_mask, power_watts


def build_raw_next_pds_state(Z_next_b: float, delta_theta_t: float, delta_phi_t: float, h_next: float) -> np.ndarray:
    """Raw (un-normalized) omega^{t+1,(b)} = (Z_next_b, delta_theta_t,
    delta_phi_t, h_next*PRESCALE) - matches environment.py::_observation()'s
    own raw representation exactly, including the h prescale
    (config.H_OBSERVATION_PRESCALE) applied BEFORE VecNormalize ever sees
    it. Caller still has to run this through VecNormalize.normalize_obs()
    before it's network-ready - this module stays SB3/torch-free.
    """
    return np.array(
        [Z_next_b, delta_theta_t, delta_phi_t, h_next * config.H_OBSERVATION_PRESCALE], dtype=np.float32
    )


def raw_next_state_from_obs(next_obs_raw: np.ndarray, Z_next_b: float, z_obs_scale: float) -> np.ndarray:
    """Raw omega^{t+1,(b)} for a trace-driven link (Lumos5G): the real raw
    next observation with only its buffer replaced by the branch's own
    Z_next_b - every other component (Delta^t, R^{t+1} and its history) is
    the same for every hypothetical action. Same scaling as
    environment.py::_lumos_observation()."""
    out = np.array(next_obs_raw, dtype=np.float32, copy=True)
    out[0] = np.float32(Z_next_b * z_obs_scale)
    return out


def split_reward(info: dict, mu_d: float, mu_p: float, mu_b: float, beta_q: float = config.BETA_Q) -> tuple:
    """(r_known^t, r_random^t)  (eqs. 18-20), from the env's own per-step
    cost/coverage fields plus the LIVE Lagrange multipliers.
    """
    r_known = -mu_d * info["cost_D"] - mu_p * info["cost_P"] - mu_b * info["cost_B"]
    r_random = beta_q * info["coverage"]
    return float(r_known), float(r_random)


def compute_virtual_branch_physics(
    agent_tile_mask_b: np.ndarray, power_watts_b: float, mandatory_tile_mask: np.ndarray,
    rd_data: dict, video_array_index: int, gop_index: int, enhanced_level: int,
    Z_t: float, h_t: float, actual_theta_t: float, actual_phi_t: float,
    mu_d: float, mu_p: float, mu_b: float, beta_q: float = config.BETA_Q, R_t: float = None,
) -> dict:
    """Recompute the deterministic known-quantities (eqs. 11-14) and
    known/random reward split (eqs. 18-20) for ONE hypothetical action
    alpha^{t,(b)} = (agent_tile_mask_b, power_watts_b) at the real
    pre-decision state - the physics half of a virtual-experience branch
    (v3 formulation, "Virtual experience" section: "recompute the known
    data, rate, stall, buffer, coverage, and hypothetical next state").

    Z_t/h_t/actual_theta_t/actual_phi_t are the REAL, already-known
    quantities at this step (Z_t/h_t pre-decision, actual_theta_t/phi_t
    the REAL observed viewport, reused unchanged - never recomputed per
    branch, since neither depends on the action). Deliberately does NOT
    touch h^{t+1}, Delta_theta^t/phi^t, VecNormalize, or the PDS critic's
    forward pass - this module stays SB3/torch-free (see module
    docstring); the caller assembles omega~^{t,(b)}/omega^{t+1,(b)} and
    calls predict_values() itself, batched across the whole VE batch.

    R_t: a trace-driven link's measured rate (info["R_t_exogenous"]) - the
    same for every branch, since power cannot change it. None (default) =
    rician link, rate recomputed from power_watts_b and h_t.

    Returns {"tile_mask_b", "Z_next_b", "r_known_b", "r_random_b",
    "coverage_b"} - all raw Python/numpy, no torch.
    """
    tile_mask_b = np.asarray(agent_tile_mask_b, dtype=bool) | np.asarray(mandatory_tile_mask, dtype=bool)

    a_result_b = layer_model.compute_A_t(
        rd_data, video_array_index, gop_index, tile_mask_b, enhanced_level=enhanced_level,
    )
    A_t_b = a_result_b["A_t"]
    R_t_b = R_t if R_t is not None else channel_model.transmission_rate(power_watts_b, h_t)

    T0 = config.T0_SEC
    if Z_t + T0 * R_t_b >= A_t_b:
        D_t_b = 0.0
    else:
        D_t_b = T0 * (1.0 - (Z_t + T0 * R_t_b) / A_t_b)
    Z_next_b = max(Z_t + T0 * R_t_b - A_t_b, 0.0)

    P_max_watts = channel_model.dbm_to_watts(config.P_MAX_DBM)
    cost_D_b = D_t_b / T0
    cost_P_b = power_watts_b / P_max_watts
    cost_B_b = float(tile_mask_b.sum()) / config.N_TILES
    r_known_b = -mu_d * cost_D_b - mu_p * cost_P_b - mu_b * cost_B_b

    coverage_b = viewport.coverage(tile_mask_b, actual_theta_t, actual_phi_t)
    r_random_b = beta_q * coverage_b

    return {
        "tile_mask_b": tile_mask_b,
        "Z_next_b": float(Z_next_b),
        "r_known_b": float(r_known_b),
        "r_random_b": float(r_random_b),
        "coverage_b": float(coverage_b),
    }
