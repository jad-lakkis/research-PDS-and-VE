"""
channel_model.py - UAV-to-base-station channel gain h^t.

Implements the Rician-fading and path-loss channel model from
MMSP'25 Sec. II.D:

    h = CHANNEL_CONST * |psi|^2
        / (d * atan(ANTENNA_BEAMWIDTH_ARG))^2

For the initial implementation, the UAV-to-BS distance is fixed at
d = config.UAV_DISTANCE_M = 50 m.

The paper specifies a 12 dB Rician factor and E[|psi|^2] = 1:

    nu = 10 ** (RICIAN_K_FACTOR_DB / 10) = 10 ** 1.2

and states psi's distribution directly as:

    psi ~ CN( sqrt(nu / (2 + 2*nu)),  1 / (2 + 2*nu) )

Implemented through its real and imaginary components:

    sigma_squared  = 1 / (2 * (1 + nu))
    component_mean = sqrt(nu / (2 * (1 + nu)))

    X ~ Normal(component_mean, sigma_squared)
    Y ~ Normal(component_mean, sigma_squared)
    psi = X + 1j * Y

Therefore:

    E[|psi|^2]
      = E[X^2] + E[Y^2]
      = 2 * component_mean**2 + 2 * sigma_squared
      = 1.

(component_mean/sigma_squared are scaled by config.RICIAN_E_PSI_SQUARED
in the code below rather than hardcoded to 1, so the formula still
reduces to exactly the above - and still integrates with the rest of
config.py - if that constant is ever anything other than 1.)

The resulting channel gain is:

    h = CHANNEL_CONST * abs(psi)**2
        / (UAV_DISTANCE_M * atan(ANTENNA_BEAMWIDTH_ARG))**2

See tests/test_channel_model.py for empirical verification (large
sample) that this construction produces E[|psi|^2]~=1, an implied
LOS-to-scatter power ratio ~=nu, and only nonnegative channel gains.
"""

import math

import numpy as np

import config


def rician_k_linear() -> float:
    """K-factor (nu) in linear scale (config stores it in dB)."""
    return 10.0 ** (config.RICIAN_K_FACTOR_DB / 10.0)


def sample_psi(rng: np.random.Generator, size=None):
    """Draw one (or `size` many) Rician-fading complex sample(s) psi,
    following the paper's stated distribution directly (see module
    docstring): psi = X + jY, X and Y i.i.d. Normal(component_mean,
    sigma_squared).
    """
    nu = rician_k_linear()
    omega = config.RICIAN_E_PSI_SQUARED  # target E[|psi|^2], nominally 1.0

    component_mean = np.sqrt(omega * nu / (2.0 * (1.0 + nu)))
    component_std = np.sqrt(omega / (2.0 * (1.0 + nu)))

    x = rng.normal(component_mean, component_std, size=size)
    y = rng.normal(component_mean, component_std, size=size)
    return x + 1j * y


def sample_channel_gain(rng: np.random.Generator, distance_m: float = None) -> float:
    """Draw one channel gain h (linear power gain, dimensionless), using
    the static-distance case by default (config.UAV_DISTANCE_M).
    """
    d = config.UAV_DISTANCE_M if distance_m is None else distance_m
    psi = sample_psi(rng)
    beamwidth_rad = np.arctan(config.ANTENNA_BEAMWIDTH_ARG)
    h = config.CHANNEL_CONST * np.abs(psi) ** 2 / (d * beamwidth_rad) ** 2
    return float(h)


def transmission_rate(power_watts: float, h: float) -> float:
    """Shannon rate R^t = W * log2(1 + P*h / (W*N0))  (eq. 13), in bits/sec.

    power_watts: transmit power in WATTS (convert from dBm first - see
    dbm_to_watts below).
    """
    n0_watts_per_hz = dbm_to_watts(config.N0_DBM_PER_HZ)
    snr = (power_watts * h) / (config.W_HZ * n0_watts_per_hz)
    return config.W_HZ * math.log2(1.0 + snr)


def dbm_to_watts(dbm: float) -> float:
    """dBm -> watts. (0 dBm = 1 mW = 1e-3 W.)"""
    return (10.0 ** (dbm / 10.0)) * 1e-3
