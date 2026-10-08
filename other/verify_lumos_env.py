"""
verify_lumos_env.py - checks for TileStreamingEnv(link="lumos5g").

1. physics: the environment reproduces fastsim.simulate_trace (the simulator behind
   every Lumos5G replay table) episode by episode - J_D, J_B, J_Q, coverage, PSNR,
   buffer - for "predicted viewport only" and for random extra tiles
2. observation: R^t and its 5-sample history are the right trace seconds, in Mbit/s;
   Z in Mbit; terminal observation uses R^{t+1}
3. virtual experience: feeding the REAL action through pds.compute_virtual_branch_physics
   + pds.raw_next_state_from_obs reproduces the real step (buffer, rewards, coverage,
   next observation) exactly
4. stall floor: viewport-only J_D over random training episodes matches
   config.LUMOS5G_STALL_FLOOR_TRAIN

Usage: .venv/Scripts/python.exe other/verify_lumos_env.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
import fastsim as fs  # noqa: E402
import lumos_replay as lr  # noqa: E402
from streaming_rl import data_loader, lumos5g, pds  # noqa: E402
from streaming_rl.environment import TileStreamingEnv  # noqa: E402

bundle = data_loader.load_training_video_bundle()
data = lumos5g.load_bundle()
v, train, held = lr.prepare()
rng = np.random.default_rng(0)
test_w = lumos5g.windows(data["runs"], data["splits"]["test"], fs.H)
episodes = [(p, test_w[i]) for p in held for i in rng.choice(len(test_w), 6, replace=False)]
rates = {w: lumos5g.rate_window(data["runs"], w[0], w[1], fs.H, 1.0, 5) for _, w in episodes}
extras = {(p, w): rng.random((fs.H, config.N_TILES)) < 0.15 for p, w in episodes}

# 1 + 2 + 3 -------------------------------------------------------------------
worst = dict(J=0.0, psnr=0.0, obs=0.0, ve=0.0)
for kind in ("viewport only", "random extra tiles"):
    pol = (lambda s: np.zeros(fs.NT, bool)) if kind == "viewport only" else (lambda s: extras[(s["pos"], s["key"])][s["t"]])
    ref = fs.simulate_trace(v, episodes, pol, rates, 5, per_episode=True)
    for (p, w), r in zip(episodes, ref):
        env = TileStreamingEnv(trace_indices=[p], bundle=bundle, link="lumos5g", lumos_split="test", lumos_data=data)
        obs, _ = env.reset(seed=0, options={"window": w})
        R = data["runs"][w[0]] * 1e6
        psnr, cov, Zs = [], [], []
        for t in range(fs.H):
            # observation: Z, |dtheta|, |dphi|, R^t, R^{t-1..t-5} (clipped at the run start), Mbit
            exp_R = [R[max(w[1] + t - k, 0)] * 1e-6 for k in range(6)]
            worst["obs"] = max(worst["obs"], float(np.max(np.abs(obs[3:] - np.float32(exp_R)))))
            Zs.append(obs[0] * 1e6)
            a = np.zeros(config.N_TILES, np.int64) if kind == "viewport only" else extras[(p, w)][t].astype(np.int64)
            Z_t = env._Z
            obs, _r, term, trunc, info = env.step(a)
            psnr.append(info["viewport_psnr_db"]); cov.append(info["coverage"])
            # VE: the real action as a "hypothetical" branch
            m, pw = pds.unpack_action(a)
            ph = pds.compute_virtual_branch_physics(
                agent_tile_mask_b=m, power_watts_b=pw, mandatory_tile_mask=info["mandatory_tile_mask"],
                rd_data=env._rd_data, video_array_index=config.TRAINING_VIDEO_ARRAY_INDEX, gop_index=info["gop_index"],
                enhanced_level=env._enhanced_level, Z_t=info["Z_t"], h_t=info["h_t"],
                actual_theta_t=info["actual_theta"], actual_phi_t=info["actual_phi"],
                mu_d=1.3, mu_p=0.0, mu_b=0.7, R_t=info.get("R_t_exogenous"))
            nxt = pds.raw_next_state_from_obs(info["next_obs_raw"], ph["Z_next_b"], info["z_obs_scale"])
            r_known = -1.3 * info["cost_D"] - 0.7 * info["cost_B"]
            worst["ve"] = max(worst["ve"], abs(ph["Z_next_b"] - info["Z"]), float(np.max(np.abs(nxt - obs))),
                              abs(ph["r_known_b"] - r_known), abs(ph["coverage_b"] - info["coverage"]))
            assert Z_t == info["Z_t"]
            if term or trunc:
                break
        assert t == fs.H - 1 and term
        # terminal observation's rate part is R^{36}, R^{35}, ... (clipped at the run end)
        exp_last = [R[min(w[1] + fs.H - k, len(R) - 1)] * 1e-6 for k in range(6)]
        worst["obs"] = max(worst["obs"], float(np.max(np.abs(obs[3:] - np.float32(exp_last)))))
        worst["J"] = max(worst["J"], abs(info["J_D"] - r["J_D"]), abs(info["J_B"] - r["J_B"]), abs(info["J_Q"] - r["J_cov"]))
        worst["psnr"] = max(worst["psnr"], abs(np.mean(psnr) - r["psnr"]), abs(np.mean(cov) - r["cov"]))
    print(f"{kind}: checked {len(episodes)} test episodes")
print(f"1. env vs fastsim: max |dJ| {worst['J']:.2e}, max |dPSNR or coverage| {worst['psnr']:.2e}")
print(f"2. observation: max |obs - expected| (Mbit/s, float32) {worst['obs']:.2e}")
print(f"3. VE real-action branch vs real step: max abs difference {worst['ve']:.2e}")

# 4 ---------------------------------------------------------------------------
# The floor's per-episode distribution is heavy-tailed (median 0, std 4.2, the worst
# 5% of episodes carry 43% of the stall), so a few thousand stepped episodes can land
# well below it. Instead: exact viewport-only J_D of every training (viewer, run, start)
# from the stall recursion (checked against the env above), looked up at 20,000
# environment-sampled episodes - tests the env's sampling, not Monte Carlo luck.
T = v.traj
A_m = {p: v.A_base + np.array([v.delta[t][T[p]["mand"][t]].sum() for t in range(fs.H)]) for p in train}
table = {}
for r in data["splits"]["train"]:
    y = data["runs"][r] * 1e6
    Rw = np.stack([y[s:s + fs.H] for s in range(len(y) - fs.H + 1)])
    for p in train:
        Z, j = np.zeros(len(Rw)), np.zeros(len(Rw))
        for t in range(fs.H):
            s_ = Z + Rw[:, t]
            st = s_ < A_m[p][t]
            j += fs.DISC[t] * np.where(st, 1 - s_ / A_m[p][t], 0.0)
            Z = np.where(st, 0.0, s_ - A_m[p][t])
        table[(p, r)] = j
exact = np.concatenate(list(table.values()))
env = TileStreamingEnv(trace_indices=train, bundle=bundle, link="lumos5g", lumos_split="train", lumos_data=data)
pos = {id(tr): train[i] for i, (_, tr) in enumerate(env._valid_traces)}
look, viewers = [], []
for k in range(20_000):
    env.reset(seed=10_000 + k)
    viewers.append(pos[id(env._trace)])
    look.append(table[(viewers[-1], env._window[0])][env._window[1]])
look = np.array(look)
print(f"4. viewport-only J_D: exact over all {len(exact)} training episodes {exact.mean():.4f} "
      f"(config {config.LUMOS5G_STALL_FLOOR_TRAIN}); at 20,000 env-sampled episodes {look.mean():.3f} "
      f"+- {exact.std() / np.sqrt(len(look)):.3f}; viewer counts {np.unique(viewers, return_counts=True)[1].tolist()}")
print(f"   action space {env.action_space}, observation shape {env.observation_space.shape}")
