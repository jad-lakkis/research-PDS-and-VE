"""
test_lagrangian.py - unit + small integration tests for
streaming_rl.lagrangian (dual_ascent_step and LagrangianRewardWrapper).
Plain assertions, no pytest dependency, matching the project's existing
test convention.

Run: .venv\\Scripts\\python.exe tests\\test_lagrangian.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from streaming_rl.environment import TileStreamingEnv
from streaming_rl.lagrangian import LagrangianRewardWrapper, dual_ascent_step


def test_dual_ascent_step_rises_on_violation():
    mu_next = dual_ascent_step(mu=1.0, avg_cost=0.8, budget=0.1, eta=0.5)
    assert mu_next > 1.0, f"expected mu to rise when avg_cost > budget, got {mu_next}"
    print(f"[PASS] dual_ascent_step rises on violation: 1.0 -> {mu_next}")


def test_dual_ascent_step_falls_on_satisfaction():
    mu_next = dual_ascent_step(mu=1.0, avg_cost=0.0, budget=0.1, eta=0.5)
    assert mu_next < 1.0, f"expected mu to fall when avg_cost < budget, got {mu_next}"
    print(f"[PASS] dual_ascent_step falls on satisfaction: 1.0 -> {mu_next}")


def test_dual_ascent_step_floors_at_zero():
    mu_next = dual_ascent_step(mu=0.05, avg_cost=0.0, budget=1.0, eta=10.0)
    assert mu_next == 0.0, f"expected mu to floor at exactly 0.0, got {mu_next}"
    print(f"[PASS] dual_ascent_step floors at 0.0 (got {mu_next}) instead of going negative")


def test_wrapper_reward_reconstruction_and_multiplier_roundtrip():
    env = LagrangianRewardWrapper(TileStreamingEnv(), mu_D_init=2.0, mu_P_init=3.0, mu_B_init=4.0)

    mus = env.get_multipliers()
    assert mus == {"mu_D": 2.0, "mu_P": 3.0, "mu_B": 4.0}, f"unexpected initial multipliers: {mus}"

    env.set_multipliers(mu_D=0.5)
    mus = env.get_multipliers()
    assert mus["mu_D"] == 0.5 and mus["mu_P"] == 3.0 and mus["mu_B"] == 4.0, (
        f"set_multipliers should only change the passed-in fields, got {mus}"
    )

    obs, info = env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    expected = (
        env.beta_q * info["coverage"]
        - mus["mu_D"] * info["cost_D"]
        - mus["mu_P"] * info["cost_P"]
        - mus["mu_B"] * info["cost_B"]
    )
    assert abs(reward - expected) < 1e-9, f"reward={reward} != reconstructed {expected}"
    assert info["mu_D"] == 0.5 and info["mu_P"] == 3.0 and info["mu_B"] == 4.0
    print(f"[PASS] wrapper reward matches beta_q*coverage - mu.cost reconstruction "
          f"(reward={reward:.4f})")


def test_episode_terminal_info_has_discounted_returns():
    """environment.py's J_Q/J_D/J_P/J_B accumulator (eqs. 5-8, literal
    per-episode discounted sum) - present ONLY on the terminal step, and
    matches an independently-recomputed sum from the per-step cost_*/
    coverage fields collected along the way."""
    env = LagrangianRewardWrapper(TileStreamingEnv())
    obs, _info = env.reset(seed=2)

    coverages, cost_Ds, cost_Ps, cost_Bs = [], [], [], []
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert not any(k in info for k in ("J_Q", "J_D", "J_P", "J_B")) or terminated, (
            "J_Q/J_D/J_P/J_B must only appear on the terminal step"
        )
        coverages.append(info["coverage"])
        cost_Ds.append(info["cost_D"])
        cost_Ps.append(info["cost_P"])
        cost_Bs.append(info["cost_B"])

    for key in ("J_Q", "J_D", "J_P", "J_B"):
        assert key in info, f"terminal info dict is missing '{key}'"

    lam = config.DISCOUNT_FACTOR_LAMBDA
    expected_J_Q = sum(lam**t * c for t, c in enumerate(coverages))
    expected_J_D = sum(lam**t * c for t, c in enumerate(cost_Ds))
    expected_J_P = sum(lam**t * c for t, c in enumerate(cost_Ps))
    expected_J_B = sum(lam**t * c for t, c in enumerate(cost_Bs))

    assert abs(info["J_Q"] - expected_J_Q) < 1e-9, f"J_Q={info['J_Q']} != recomputed {expected_J_Q}"
    assert abs(info["J_D"] - expected_J_D) < 1e-9, f"J_D={info['J_D']} != recomputed {expected_J_D}"
    assert abs(info["J_P"] - expected_J_P) < 1e-9, f"J_P={info['J_P']} != recomputed {expected_J_P}"
    assert abs(info["J_B"] - expected_J_B) < 1e-9, f"J_B={info['J_B']} != recomputed {expected_J_B}"
    print(f"[PASS] terminal J_Q/J_D/J_P/J_B match an independently-recomputed discounted sum "
          f"over {len(coverages)} steps (J_Q={info['J_Q']:.4f}, J_D={info['J_D']:.4f}, "
          f"J_P={info['J_P']:.4f}, J_B={info['J_B']:.4f})")


if __name__ == "__main__":
    test_dual_ascent_step_rises_on_violation()
    test_dual_ascent_step_falls_on_satisfaction()
    test_dual_ascent_step_floors_at_zero()
    test_wrapper_reward_reconstruction_and_multiplier_roundtrip()
    test_episode_terminal_info_has_discounted_returns()
    print()
    print("All lagrangian tests passed.")
