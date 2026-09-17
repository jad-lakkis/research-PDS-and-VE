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

# Z-tilde (1) + Delta_theta^{t-1}, Delta_phi^{t-1}, h^t (3, reused from
# omega^t as-is) + tile_mask E^t(x^t) (config.N_TILES) = 68  (eq. 16)
PDS_OBS_DIM = 1 + 3 + config.N_TILES


def build_pds_observation(obs: np.ndarray, true_next_obs: np.ndarray, tile_mask: np.ndarray) -> np.ndarray:
    """omega~^t = concat(Z~^{t+1}, Delta_theta^{t-1}, Delta_phi^{t-1}, h^t, E^t(x^t))  (eq. 16).

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
    assert obs.shape == (4,), f"expected omega^t shape (4,), got {obs.shape}"
    assert true_next_obs.shape == (4,), f"expected true_next_obs shape (4,), got {true_next_obs.shape}"
    assert tile_mask.shape == (config.N_TILES,), f"expected tile_mask shape ({config.N_TILES},), got {tile_mask.shape}"

    z_tilde = true_next_obs[0:1]
    delta_theta_delta_phi_h = obs[1:4]
    pds_obs = np.concatenate([z_tilde, delta_theta_delta_phi_h, tile_mask]).astype(np.float32)
    assert pds_obs.shape == (PDS_OBS_DIM,), f"expected PDS obs shape ({PDS_OBS_DIM},), got {pds_obs.shape}"
    return pds_obs


def split_reward(info: dict, mu_d: float, mu_p: float, mu_b: float, beta_q: float = config.BETA_Q) -> tuple:
    """(r_known^t, r_random^t)  (eqs. 18-20), from the env's own per-step
    cost/coverage fields plus the LIVE Lagrange multipliers.
    """
    r_known = -mu_d * info["cost_D"] - mu_p * info["cost_P"] - mu_b * info["cost_B"]
    r_random = beta_q * info["coverage"]
    return float(r_known), float(r_random)
