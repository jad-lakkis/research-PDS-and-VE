"""
test_channel_model.py - empirical verification of channel_model.sample_psi
against the paper's stated properties: E[|psi|^2]=1, a 12dB (nu=10^1.2)
LOS-to-scatter power ratio, and nonnegative channel gains. Plain
assertions, no pytest dependency (not installed in this project yet).

Run: .venv\\Scripts\\python.exe tests\\test_channel_model.py
"""

import sys
from pathlib import Path

import numpy as np

# Allow running this file directly (python tests/test_channel_model.py)
# by putting the project root - where config.py and streaming_rl/ live -
# on sys.path, since this file is one level below the root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from streaming_rl import channel_model

N_SAMPLES = 500_000
TOLERANCE = 0.02          # relative tolerance for E[|psi|^2] (target 1.0)
RATIO_TOLERANCE = 0.03    # relative tolerance for the LOS/scatter ratio (target nu)


def test_mean_power_is_one():
    rng = np.random.default_rng(1)
    psi = channel_model.sample_psi(rng, size=N_SAMPLES)
    mean_power = np.mean(np.abs(psi) ** 2)
    target = config.RICIAN_E_PSI_SQUARED
    assert abs(mean_power - target) < TOLERANCE, (
        f"mean(|psi|^2)={mean_power:.5f}, expected ~{target} (tolerance {TOLERANCE})"
    )
    print(f"[PASS] mean(|psi|^2) = {mean_power:.5f}  (target {target}, N={N_SAMPLES:,})")


def test_los_to_scatter_ratio_matches_nu():
    rng = np.random.default_rng(2)
    psi = channel_model.sample_psi(rng, size=N_SAMPLES)
    x, y = psi.real, psi.imag

    # Empirical LOS power: squared sample means of the real/imag parts.
    los_power = np.mean(x) ** 2 + np.mean(y) ** 2
    # Empirical scatter power: sample variances of the real/imag parts.
    scatter_power = np.var(x) + np.var(y)
    empirical_ratio = los_power / scatter_power

    nu = channel_model.rician_k_linear()
    rel_error = abs(empirical_ratio - nu) / nu
    assert rel_error < RATIO_TOLERANCE, (
        f"empirical LOS/scatter ratio={empirical_ratio:.3f}, expected ~nu={nu:.3f} "
        f"(relative error {rel_error:.3%}, tolerance {RATIO_TOLERANCE:.0%})"
    )
    print(f"[PASS] empirical LOS/scatter power ratio = {empirical_ratio:.3f}  "
          f"(target nu = 10^1.2 = {nu:.3f}, N={N_SAMPLES:,})")


def test_channel_gains_nonnegative():
    rng = np.random.default_rng(3)
    gains = [channel_model.sample_channel_gain(rng) for _ in range(10_000)]
    assert all(g >= 0.0 for g in gains), "found a negative channel gain"
    assert all(np.isfinite(g) for g in gains), "found a non-finite channel gain"
    print(f"[PASS] all {len(gains):,} sampled channel gains are finite and >= 0 "
          f"(min={min(gains):.3e}, max={max(gains):.3e})")


def test_seeded_reproducibility():
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    psi_a = channel_model.sample_psi(rng_a, size=1000)
    psi_b = channel_model.sample_psi(rng_b, size=1000)
    assert np.array_equal(psi_a, psi_b), "same seed produced different psi samples"

    rng_c = np.random.default_rng(123)
    rng_d = np.random.default_rng(456)
    gain_c = channel_model.sample_channel_gain(rng_c)
    gain_d = channel_model.sample_channel_gain(rng_d)
    assert gain_c != gain_d, "different seeds produced identical channel gains (suspicious)"
    print("[PASS] same seed -> identical samples; different seeds -> different samples")


if __name__ == "__main__":
    test_mean_power_is_one()
    test_los_to_scatter_ratio_matches_nu()
    test_channel_gains_nonnegative()
    test_seeded_reproducibility()
    print()
    print("All channel_model tests passed.")
