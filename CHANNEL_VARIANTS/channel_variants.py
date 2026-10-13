"""
channel_variants.py - what the link rate looks like under different synthetic channel settings (no video, no training).

Today's synthetic link (MMSP'25 model, streaming_rl/channel_model.py), one draw per second:
    h = CHANNEL_CONST * |psi|^2 / (d * atan(BEAMWIDTH_ARG))^2,  psi Rician with factor K (12 dB), i.i.d. every second
    R = W log2(1 + P h / (W N0)),  P = 50 mW, W = 10 MHz, d = 50 m
At this operating point (SNR ~49 dB) R moves only W*log2(10)/10 = 3.3 Mbit/s per dB of gain, so it stays in
153-170 Mbit/s. Knobs varied here, one at a time from today's model, then in combinations:
    K           Rician factor (12 dB = strong line of sight; 0 dB / Rayleigh = deep fades)
    shadowing   slow log-normal shadowing s_t (dB), AR(1) in time: s_t = rho s_{t-1} + sqrt(1-rho^2) sigma eps_t
    blockage    two-state Markov blockage: attenuation (dB), share of time blocked, mean blocked duration (s)
    W           bandwidth (MHz), at the same power and distance (wider band: higher rate, lower SNR, larger swing)
Reference needs (Runner, enhancement level 5): base layer + forced predicted-viewport tiles ~129 Mbit/s;
all 64 tiles enhanced ~430 Mbit/s.

Outputs (CHANNEL_VARIANTS/): channel_variants.csv (statistics per setting), fig1_distributions.png,
fig2_time_series.png, fig3_candidates_vs_old_and_lumos5g.png.
Usage (from the project folder): .venv/Scripts/python.exe CHANNEL_VARIANTS/channel_variants.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import config  # noqa: E402
from streaming_rl import channel_model, lumos5g  # noqa: E402

OUT = "CHANNEL_VARIANTS"
N_SEC = 200_000
NEED_FORCED, NEED_ALL = 129.0, 430.0          # Mbit/s, Runner at enhancement level 5
H_BASE = config.CHANNEL_CONST / (config.UAV_DISTANCE_M * np.arctan(config.ANTENNA_BEAMWIDTH_ARG)) ** 2
N0 = channel_model.dbm_to_watts(config.N0_DBM_PER_HZ)
BASE = dict(K_dB=12.0, sh_sigma=0.0, sh_rho=0.95, blk_att=0.0, blk_frac=0.2, blk_mean=3.0, W_MHz=10.0)


def rates(K_dB=12.0, sh_sigma=0.0, sh_rho=0.95, blk_att=0.0, blk_frac=0.2, blk_mean=3.0, W_MHz=10.0, n=N_SEC, seed=0):
    """R_t in Mbit/s, one value per second."""
    rng = np.random.default_rng(seed)
    if K_dB is None:                                   # Rayleigh
        psi2 = rng.exponential(1.0, n)
    else:                                              # same construction as channel_model.sample_psi
        nu = 10 ** (K_dB / 10)
        m, s = np.sqrt(nu / (2 * (1 + nu))), np.sqrt(1 / (2 * (1 + nu)))
        psi2 = rng.normal(m, s, n) ** 2 + rng.normal(m, s, n) ** 2
    gain_db = 10 * np.log10(H_BASE * psi2)
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
    snr = config.FIXED_POWER_WATTS * 10 ** (gain_db / 10) / (W * N0)
    return W * np.log2(1 + snr) / 1e6


def stats(r):
    below = r < NEED_FORCED
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, below.astype(int), 0])))[::2] if below.any() else np.array([0])
    return dict(mean=r.mean(), p5=np.percentile(r, 5), median=np.median(r), p95=np.percentile(r, 95),
                pct_below_forced_need=100 * below.mean(), pct_over_2x_need=100 * (r >= 2 * NEED_FORCED).mean(),
                pct_over_all64=100 * (r >= NEED_ALL).mean(), mean_outage_length_s=runs.mean(),
                lag1_autocorr=np.corrcoef(r[:-1], r[1:])[0, 1])


SWEEPS = {
    "Rician factor K (fading depth)": [("K 12 dB (today)", dict(K_dB=12.0)), ("K 6 dB", dict(K_dB=6.0)), ("K 3 dB", dict(K_dB=3.0)),
                                       ("K 0 dB", dict(K_dB=0.0)), ("Rayleigh", dict(K_dB=None))],
    "slow shadowing, memory rho 0.95": [("none (today)", {}), ("sigma 4 dB", dict(sh_sigma=4.0)), ("sigma 8 dB", dict(sh_sigma=8.0)),
                                        ("sigma 12 dB", dict(sh_sigma=12.0))],
    "blockage, 20% of time, 3 s on average": [("none (today)", {}), ("10 dB", dict(blk_att=10.0)), ("20 dB", dict(blk_att=20.0)),
                                              ("30 dB", dict(blk_att=30.0))],
    "bandwidth W (same power, distance)": [("10 MHz (today)", {}), ("20 MHz", dict(W_MHz=20.0)), ("40 MHz", dict(W_MHz=40.0)),
                                           ("100 MHz", dict(W_MHz=100.0))],
}
# combinations picked from a grid (W 15-30 MHz, shadowing 6-12 dB, blockage 0-30 dB, K 12/3 dB) for: below the forced
# need 8-22% of seconds, at least 2x the need >= 40%, room for all 64 tiles 3-30%, memory (lag-1 correlation >= 0.7)
CANDIDATES = [
    ("A: 20 MHz, shadowing 12 dB, blockage 30 dB (15%)", dict(W_MHz=20.0, sh_sigma=12.0, blk_att=30.0, blk_frac=0.15)),
    ("B: 25 MHz, shadowing 10 dB, blockage 30 dB (15%)", dict(W_MHz=25.0, sh_sigma=10.0, blk_att=30.0, blk_frac=0.15)),
    ("C: 25 MHz, shadowing 12 dB, blockage 30 dB (15%), K 3 dB", dict(W_MHz=25.0, sh_sigma=12.0, blk_att=30.0, blk_frac=0.15, K_dB=3.0)),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    # today's model, built exactly as the environment does, as a check on the generator
    rng = np.random.default_rng(1)
    env_r = np.array([channel_model.transmission_rate(config.FIXED_POWER_WATTS, channel_model.sample_channel_gain(rng))
                      for _ in range(20000)]) / 1e6
    gen_r = rates(n=20000, seed=1)
    print("check vs the environment's channel (5/50/95%):", np.percentile(env_r, [5, 50, 95]).round(1), np.percentile(gen_r, [5, 50, 95]).round(1))

    data = lumos5g.load_bundle()
    lum = np.concatenate([data["runs"][r] for r in data["splits"]["train"]])
    series, rows = {}, []
    for sweep, items in SWEEPS.items():
        for name, kw in items:
            r = rates(**{**BASE, **kw})
            series[(sweep, name)] = r
            rows.append(dict(sweep=sweep, setting=name, **stats(r)))
    for name, kw in CANDIDATES:
        r = rates(**{**BASE, **kw})
        series[("candidates", name)] = r
        rows.append(dict(sweep="candidates", setting=name, **stats(r)))
    rows.append(dict(sweep="reference", setting="Lumos5G (training runs)", **stats(lum)))
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{OUT}/channel_variants.csv", index=False)
    pd.set_option("display.width", 250)
    print(tab.round(2).to_string(index=False))

    # fig 1: distribution of the rate, one panel per knob
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, (sweep, items) in zip(axes.flat, SWEEPS.items()):
        cols = plt.cm.Blues(np.linspace(0.35, 1.0, len(items)))
        hi = 0
        for (name, _), c in zip(items, cols):
            r = np.sort(series[(sweep, name)])
            ax.plot(r, np.arange(1, len(r) + 1) / len(r), color=c, lw=2, label=name)
            hi = max(hi, np.percentile(r, 99.5))
        ax.axvline(NEED_FORCED, color="#d62728", ls="--", lw=1.2)
        ax.text(NEED_FORCED, 0.02, " forced tiles need", color="#d62728", fontsize=8, rotation=90, va="bottom")
        if hi > NEED_ALL:
            ax.axvline(NEED_ALL, color="#7f7f7f", ls=":", lw=1.2)
            ax.text(NEED_ALL, 0.02, " all 64 tiles need", color="#555555", fontsize=8, rotation=90, va="bottom")
        ax.set_xlim(0, hi * 1.05)
        ax.set_title(sweep, fontsize=10)
        ax.set_xlabel("link rate (Mbit/s)")
        ax.set_ylabel("share of seconds at or below")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Synthetic channel: distribution of the link rate, one setting changed at a time (all else as today)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_distributions.png", dpi=150)
    plt.close(fig)

    # fig 2: 120 s of each knob's strongest setting vs today's model and the needs
    picks = [("today's model", series[("Rician factor K (fading depth)", "K 12 dB (today)")]),
             ("Rayleigh fading", series[("Rician factor K (fading depth)", "Rayleigh")]),
             ("shadowing 12 dB, memory", series[("slow shadowing, memory rho 0.95", "sigma 12 dB")]),
             ("blockage 30 dB", series[("blockage, 20% of time, 3 s on average", "30 dB")]),
             ("bandwidth 100 MHz", series[("bandwidth W (same power, distance)", "100 MHz")])]
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 2.1 * len(picks)), sharex=True)
    for ax, (name, r) in zip(axes, picks):
        ax.plot(np.arange(120), r[1000:1120], color="#1f77b4", lw=1.3)
        ax.axhline(NEED_FORCED, color="#d62728", ls="--", lw=1.1)
        if r[1000:1120].max() > NEED_ALL:
            ax.axhline(NEED_ALL, color="#7f7f7f", ls=":", lw=1.1)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Mbit/s")
        ax.set_title(name, fontsize=9.5, loc="left")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("120 seconds of each: red dashed = forced tiles need (~129 Mbit/s), grey dotted = all 64 tiles (~430)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_time_series.png", dpi=150)
    plt.close(fig)

    # fig 3: candidates vs today's model and Lumos5G
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1, 1.4]))
    ref = [("today's synthetic model", series[("Rician factor K (fading depth)", "K 12 dB (today)")], "#7f7f7f"),
           ("Lumos5G (measured)", lum, "#222222")]
    cand_cols = ["#1f77b4", "#2ca02c", "#9467bd"]
    allc = ref + [(n, series[("candidates", n)], c) for (n, _), c in zip(CANDIDATES, cand_cols)]
    for name, r, c in allc:
        rs = np.sort(r)
        axes[0].plot(rs, np.arange(1, len(rs) + 1) / len(rs), color=c, lw=2, label=name)
    axes[0].axvline(NEED_FORCED, color="#d62728", ls="--", lw=1.2)
    axes[0].axvline(NEED_ALL, color="#7f7f7f", ls=":", lw=1.2)
    axes[0].set_xlim(0, 1500)
    axes[0].set_xlabel("link rate (Mbit/s)")
    axes[0].set_ylabel("share of seconds at or below")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    for name, r, c in allc[2:]:
        axes[1].plot(np.arange(300), r[5000:5300], color=c, lw=1.2, label=name)
    axes[1].axhline(NEED_FORCED, color="#d62728", ls="--", lw=1.2, label="forced tiles need")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("Mbit/s")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Candidate synthetic channels vs today's model and Lumos5G", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_candidates_vs_old_and_lumos5g.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
