"""
synthetic_link.py - the synthetic link ("option 2") as playable rate traces.

Model: the MMSP'25 UAV channel of streaming_rl/channel_model.py (d = 50 m, 50 mW, same path loss and antenna
gain, R = W log2(1 + P h / (W N0))), one value per second, with
  bandwidth W (MHz);
  Rician fading with factor K (dB; None = Rayleigh), a fresh draw every second (as now);
  slow log-normal shadowing s_t (dB), correlated in time: s_t = rho s_{t-1} + sqrt(1 - rho^2) sigma eps_t;
  blockages: a two-state Markov chain (blocked / clear) with a given share of time blocked and mean blocked
  duration; while blocked the gain drops by blk_att dB.
Shadowing and blockages depend on the previous seconds, so the link is generated ahead of time as long "runs"
(train / validation / test, each run its own seed) and played back through the same code path as the Lumos5G
traces: TileStreamingEnv(link="lumos5g", lumos_data=load_bundle()). Rates are in Mbit/s, like the Lumos5G data.

rates() is identical to CHANNEL_VARIANTS/channel_variants.rates (checked by other/verify_synthetic_link.py).
"""
import numpy as np
import pandas as pd

import config
from streaming_rl import channel_model


def rates(K_dB=12.0, sh_sigma=0.0, sh_rho=0.95, blk_att=0.0, blk_frac=0.2, blk_mean=3.0, W_MHz=10.0, n=1000, seed=0):
    """link rate in Mbit/s, one value per second"""
    rng = np.random.default_rng(seed)
    if K_dB is None:
        psi2 = rng.exponential(1.0, n)
    else:   # same construction as channel_model.sample_psi
        nu = 10 ** (K_dB / 10)
        m, s = np.sqrt(nu / (2 * (1 + nu))), np.sqrt(1 / (2 * (1 + nu)))
        psi2 = rng.normal(m, s, n) ** 2 + rng.normal(m, s, n) ** 2
    h_base = config.CHANNEL_CONST / (config.UAV_DISTANCE_M * np.arctan(config.ANTENNA_BEAMWIDTH_ARG)) ** 2
    gain_db = 10 * np.log10(h_base * psi2)
    if sh_sigma > 0:
        e = rng.standard_normal(n)
        sh = np.empty(n)
        sh[0] = sh_sigma * e[0]
        a, b = sh_rho, np.sqrt(1 - sh_rho ** 2) * sh_sigma
        for t in range(1, n):
            sh[t] = a * sh[t - 1] + b * e[t]
        gain_db = gain_db + sh
    if blk_att > 0:
        p_exit = 1.0 / blk_mean
        p_enter = p_exit * blk_frac / (1 - blk_frac)
        u = rng.random(n)
        blk = np.zeros(n, bool)
        blk[0] = u[0] < blk_frac
        for t in range(1, n):
            blk[t] = (u[t] >= p_exit) if blk[t - 1] else (u[t] < p_enter)
        gain_db = gain_db - blk_att * blk
    W = W_MHz * 1e6
    snr = config.FIXED_POWER_WATTS * 10 ** (gain_db / 10) / (W * channel_model.dbm_to_watts(config.N0_DBM_PER_HZ))
    return W * np.log2(1 + snr) / 1e6


def load_bundle(params=None, seed=None, n_runs=None, run_seconds=None):
    """{"runs": {run: Mbit/s array}, "info": per-run frame, "splits": {"train", "val", "test": [runs]}} in the
    format of lumos5g.load_bundle(); run i is generated with seed (seed + i)"""
    params = dict(config.SYNTH_LINK if params is None else params)
    seed = config.SYNTH_SEED if seed is None else seed
    n_runs = config.SYNTH_RUNS if n_runs is None else n_runs
    run_seconds = config.SYNTH_RUN_SECONDS if run_seconds is None else run_seconds
    runs, splits, k = {}, {}, 0
    for split, count in zip(("train", "val", "test"), n_runs):
        splits[split] = []
        for _ in range(count):
            runs[k] = rates(**params, n=run_seconds, seed=seed + k)
            splits[split].append(k)
            k += 1
    info = pd.DataFrame({"mode": "synthetic", "mean_mbps": {r: x.mean() for r, x in runs.items()}})
    return {"runs": runs, "info": info, "splits": splits}
