"""
verify_synthetic_link.py - checks for the synthetic link "option 2" (--link synthetic) before any training.

1. generator: streaming_rl.synthetic_link.rates is identical to CHANNEL_VARIANTS/channel_variants.rates (the
   replay study the channel was chosen from)
2. physics: the environment on the synthetic traces reproduces fastsim.simulate_trace episode by episode
   (J_D, J_B, PSNR) for viewport-only and random extra tiles, and its observation carries R^t..R^{t-5}
3. stall floor: viewport-only J_D, exact over the training distribution (training viewers x every start second of
   every training run) - the value for config.SYNTH_STALL_FLOOR_TRAIN
4. baseline on the TEST runs (held-out viewers x windows every 12 s): every heuristic and both oracles, against
   the budgets (allowance 0.1, tile budget 11) - what the trained agents will be compared with.
   Written to CHANNEL_VARIANTS/option2_test_baseline.csv

Usage: .venv/Scripts/python.exe other/verify_synthetic_link.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "CHANNEL_VARIANTS"))
os.chdir(ROOT)
import config  # noqa: E402
import fastsim as fs  # noqa: E402
import channel_variants as cv  # noqa: E402
import replay_channels as rc  # noqa: E402
from streaming_rl import data_loader, lumos5g, synthetic_link  # noqa: E402
from streaming_rl.environment import TileStreamingEnv  # noqa: E402

# 1 ---------------------------------------------------------------------------------------------------------------
gen = max(float(np.max(np.abs(synthetic_link.rates(**config.SYNTH_LINK, n=3000, seed=s)
                                - cv.rates(**{**cv.BASE, **config.SYNTH_LINK}, n=3000, seed=s)))) for s in (0, 1, 7))
print(f"1. generator vs the replay study's: max diff {gen:.1e} Mbit/s")

data = synthetic_link.load_bundle()
runs, sp = data["runs"], data["splits"]
v, G = rc.v, rc.G
held, train = list(G["held"]), list(G["train"])

# 2 ---------------------------------------------------------------------------------------------------------------
bundle = data_loader.load_training_video_bundle()
rng = np.random.default_rng(0)
test_w = lumos5g.windows(runs, sp["test"], fs.H)
episodes = [(p, test_w[i]) for p in held for i in rng.choice(len(test_w), 6, replace=False)]
rates = {w: lumos5g.rate_window(runs, w[0], w[1], fs.H, 1.0, 5) for _, w in episodes}
extras = {(p, w): rng.random((fs.H, config.N_TILES)) < 0.15 for p, w in episodes}
worst = dict(J=0.0, psnr=0.0, obs=0.0)
for kind in ("viewport only", "random extra tiles"):
    pol = (lambda s: np.zeros(fs.NT, bool)) if kind == "viewport only" else (lambda s: extras[(s["pos"], s["key"])][s["t"]])
    ref = fs.simulate_trace(v, episodes, pol, rates, 5, per_episode=True)
    for (p, w), r in zip(episodes, ref):
        env = TileStreamingEnv(trace_indices=[p], bundle=bundle, link="lumos5g", lumos_split="test", lumos_data=data)
        obs, _ = env.reset(seed=0, options={"window": w})
        R = runs[w[0]] * 1e6
        psnr = []
        for t in range(fs.H):
            exp_R = [R[max(w[1] + t - k, 0)] * 1e-6 for k in range(6)]
            worst["obs"] = max(worst["obs"], float(np.max(np.abs(obs[3:] - np.float32(exp_R)))))
            a = np.zeros(config.N_TILES, np.int64) if kind == "viewport only" else extras[(p, w)][t].astype(np.int64)
            obs, _r, term, trunc, info = env.step(a)
            psnr.append(info["viewport_psnr_db"])
        assert term
        worst["J"] = max(worst["J"], abs(info["J_D"] - r["J_D"]), abs(info["J_B"] - r["J_B"]))
        worst["psnr"] = max(worst["psnr"], abs(np.mean(psnr) - r["psnr"]))
print(f"2. environment vs replay simulator ({len(episodes)} test episodes x 2 policies): max |dJ| {worst['J']:.1e}, "
      f"max |dPSNR| {worst['psnr']:.1e}; observation R history max diff {worst['obs']:.1e} Mbit/s")

# 3 ---------------------------------------------------------------------------------------------------------------
starts = [(r, s) for r in sp["train"] for s in range(len(runs[r]) - fs.H + 1)]
Rw = np.stack([runs[r][s:s + fs.H] for r, s in starts]) * 1e6
floors = []
for p in train:
    A = np.broadcast_to(G["A_m"][p], Rw.shape)
    D = rc.stall(A, Rw, None)
    floors.append((D * fs.DISC).sum(axis=1).mean())
floor = float(np.mean(floors))
print(f"3. viewport-only stall over the training distribution ({len(train)} viewers x {len(starts)} windows): {floor:.4f} "
      f"(config.SYNTH_STALL_FLOOR_TRAIN = {config.SYNTH_STALL_FLOOR_TRAIN})")

# 4 ---------------------------------------------------------------------------------------------------------------
wins = lumos5g.windows(runs, sp["test"], fs.H, 12)
eps = [(p, w) for w in wins for p in held]
R_list = [lumos5g.rate_window(runs, w[0], w[1], fs.H, 1.0, 5)[5:] for _, w in eps]
hs = rc.heuristics([p for p, _ in eps], np.stack(R_list), None)
hs["link oracle"] = rc.oracle(eps, R_list, config.LUMOS5G_B_BAR, G["EM"], None, rc.TOP_LINK)
hs["oracle"] = rc.oracle(eps, R_list, config.LUMOS5G_B_BAR, G["EO"], None)
f_test = hs["H1 viewport only"]["J_D"]
rows = []
for m, r in hs.items():
    avoid = r["J_D"] - f_test
    rows.append(dict(method=m, psnr=r["psnr"], J_D=r["J_D"], avoidable_stall=avoid,
                     stall_over_budget=max(avoid - config.LUMOS5G_STALL_ALLOWANCE, 0.0), J_B=r["J_B"],
                     tiles_over_budget=max(r["J_B"] - config.LUMOS5G_B_BAR, 0.0), tiles_per_s=r["tiles"],
                     within_both=avoid <= config.LUMOS5G_STALL_ALLOWANCE + 1e-9 and r["J_B"] <= config.LUMOS5G_B_BAR + 1e-9))
tab = pd.DataFrame(rows)
tab.to_csv("CHANNEL_VARIANTS/option2_test_baseline.csv", index=False)
pd.set_option("display.width", 250)
print(f"4. baseline on the test runs ({len(eps)} episodes; unavoidable stall {f_test:.3f}, stall budget {f_test + 0.1:.3f}, tile budget 11):")
print(tab.round(3).to_string(index=False))
