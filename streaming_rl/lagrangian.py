"""
lagrangian.py - Lagrangian relaxation of the constrained MDP (eqs. 5-9).

The environment (streaming_rl.environment.TileStreamingEnv) reports the
three normalized per-step constraint costs via info["cost_D"],
info["cost_P"], info["cost_B"], and (on each episode's terminal step
only) the literal eq. 5-8 discounted returns info["J_Q"]/["J_D"]/
["J_P"]/["J_B"] (see environment.py::step()). This module wraps an env
exposing the per-step fields and replaces its fixed BETA_S/BETA_P/
BETA_B penalty weights with adaptive Lagrange multipliers mu_D, mu_P,
mu_B. The multipliers themselves are updated once per PPO rollout (see
train_ppo.py's LagrangianMultiplierCallback, hooked to _on_rollout_end)
using the average of J_D/J_P/J_B across that rollout's completed
episodes: mu <- max(0, mu + eta*(avg_J - budget)) via dual_ascent_step
below. mu rises when a budget is exceeded, falls when it's satisfied -
directly implementing eqs. 6-8 as adaptive weights rather than the
fixed-weight stand-in (eq. 9). BETA_Q is NOT adaptive: viewport quality
is the objective (eq. 5), not a constraint.

Verified against the installed gymnasium==1.3.0/stable_baselines3==2.9.0
(see other/decision_log.md): gym.Wrapper.step() passes the inner env's
info dict through unmodified, and env_method()/get_attr()/set_attr()
tunnel through the VecNormalize/DummyVecEnv/Monitor stack (train_ppo.py)
via gymnasium's own get_wrapper_attr - no manual handling needed here.
"""

import gymnasium as gym

import config


class LagrangianRewardWrapper(gym.Wrapper):
    """Reconstructs the reward from info["coverage"] and the three
    cost_* fields on every step, discarding the wrapped env's own
    fixed-beta reward entirely (rather than arithmetically subtracting
    its penalty terms back out) - avoids double-penalizing, and needs
    zero knowledge of T0_SEC/P_MAX_DBM/N_TILES (already normalized once,
    in environment.py).

    Does NOT override reset(): mu_D/mu_P/mu_B must persist across
    episode boundaries - LagrangianMultiplierCallback (train_ppo.py)
    updates them once per PPO rollout, which spans many episodes.
    """

    def __init__(
        self,
        env: gym.Env,
        mu_D_init: float = config.MU_D_INIT,
        mu_P_init: float = config.MU_P_INIT,
        mu_B_init: float = config.MU_B_INIT,
        beta_q: float = config.BETA_Q,
    ):
        super().__init__(env)
        self.mu_D = float(mu_D_init)
        self.mu_P = float(mu_P_init)
        self.mu_B = float(mu_B_init)
        self.beta_q = float(beta_q)

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        reward = (
            self.beta_q * info["coverage"]
            - self.mu_D * info["cost_D"]
            - self.mu_P * info["cost_P"]
            - self.mu_B * info["cost_B"]
        )

        info["mu_D"], info["mu_P"], info["mu_B"] = self.mu_D, self.mu_P, self.mu_B
        info["fixed_beta_reward"] = base_reward   # discarded reward, kept for comparison/debugging
        info["lagrangian_reward"] = reward

        return obs, reward, terminated, truncated, info

    def set_multipliers(self, mu_D: float = None, mu_P: float = None, mu_B: float = None) -> None:
        if mu_D is not None:
            self.mu_D = float(mu_D)
        if mu_P is not None:
            self.mu_P = float(mu_P)
        if mu_B is not None:
            self.mu_B = float(mu_B)

    def get_multipliers(self) -> dict:
        return {"mu_D": self.mu_D, "mu_P": self.mu_P, "mu_B": self.mu_B}


def dual_ascent_step(mu: float, avg_cost: float, budget: float, eta: float) -> float:
    """One projected-gradient-ascent step on a single Lagrange multiplier:
    mu <- max(0, mu + eta*(avg_cost - budget)). Pure function, no gym/SB3
    dependency - rises when avg_cost exceeds budget, falls (floored at 0)
    when it's satisfied.
    """
    return max(0.0, mu + eta * (avg_cost - budget))


def parse_dropped_constraints(spec: str) -> frozenset:
    """Parse --drop-constraints ("", "P", "P,B", ...) into a frozenset of
    single-letter codes from {"D","P","B"}. A dropped constraint is a
    GENUINE removal, not a loosened budget: its multiplier is pinned at 0
    for the whole run (LagrangianRewardWrapper's mu_*_init below, plus
    LagrangianMultiplierCallback/HeldOutTraceEvalCallback in train_ppo.py/
    streaming_rl/eval.py skipping it), so it contributes exactly 0 to the
    reward and is excluded from the feasibility/violation check - the
    agent gets zero signal about it in either direction, not just a loose
    one. Kept here (not inline in each entrypoint) so train_ppo.py and
    train_ppo_pds.py can't drift into interpreting the flag differently.
    """
    if not spec:
        return frozenset()
    codes = frozenset(c.strip().upper() for c in spec.split(",") if c.strip())
    assert codes <= {"D", "P", "B"}, f"--drop-constraints must be from {{D,P,B}}, got {spec!r}"
    return codes


def mu_init_kwargs(dropped_constraints: frozenset) -> dict:
    """mu_*_init overrides for LagrangianRewardWrapper - 0.0 (and pinned
    there, never updated) for any dropped constraint, config default
    otherwise."""
    return {
        "mu_D_init": 0.0 if "D" in dropped_constraints else config.MU_D_INIT,
        "mu_P_init": 0.0 if "P" in dropped_constraints else config.MU_P_INIT,
        "mu_B_init": 0.0 if "B" in dropped_constraints else config.MU_B_INIT,
    }
