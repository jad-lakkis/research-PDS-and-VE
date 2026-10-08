"""
verify_tile_frame.py - checks for TileStreamingEnv(tile_frame="relative").

1. mapping: rel_to_abs / abs_to_rel are inverse; the predicted viewport's column is
   relative column 4; the column offset lies in [-22.5, 22.5)
2. physics: a relative-frame action gives exactly the same step as the equivalent
   physical action in the absolute frame (tile mask, bits, rate, stall, buffer,
   coverage, PSNR, costs, J's) - Lumos5G and rician links, random actions
3. observation: signed head move (wrapped), predicted elevation, column offset,
   sin/cos predicted azimuth, video time, link part
4. replay equivalence: the replay simulator's per-episode results for random
   physical masks are reproduced when the same masks are sent as relative bits
5. virtual experience + PDS input: the real relative action through the VE path
   reproduces the real step and next observation; the PDS observation is
   [post-decision Z, the other 13 observation entries, relative executed mask] = 78

Usage: .venv/Scripts/python.exe other/verify_tile_frame.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
import fastsim as fs  # noqa: E402
import lumos_replay as lr  # noqa: E402
from streaming_rl import data_loader, lumos5g, pds, tile_frame  # noqa: E402
from streaming_rl.environment import TileStreamingEnv  # noqa: E402

rng = np.random.default_rng(0)
bundle = data_loader.load_training_video_bundle()
data = lumos5g.load_bundle()

# 1 -------------------------------------------------------------------------
for _ in range(2000):
    m = rng.random(64) < 0.4
    s = int(rng.integers(-8, 8))
    assert np.array_equal(tile_frame.abs_to_rel(tile_frame.rel_to_abs(m, s), s), m)
for th in np.linspace(-180, 180, 7201):
    s = tile_frame.column_shift(th)
    rel = np.zeros(64, bool); rel[tile_frame.CENTER_COL] = True          # row 0, relative column 4
    assert int(np.argmax(tile_frame.rel_to_abs(rel, s))) == tile_frame.predicted_column(th)
    off = tile_frame.column_offset_deg(th)
    assert -22.5 <= off < 22.5
print("1. mapping: inverse, centre column and offset range OK")

# 2 + 3 + 5 -----------------------------------------------------------------
worst = dict(step=0.0, obs=0.0, ve=0.0)
FIELDS = ("A_t", "R_t", "D_t", "Z", "coverage", "viewport_psnr_db", "cost_D", "cost_P", "cost_B")
for link in ("lumos5g", "rician"):
    kw = dict(link=link, lumos_split="test", lumos_data=data) if link == "lumos5g" else {}
    test_w = lumos5g.windows(data["runs"], data["splits"]["test"], fs.H)
    for p in (2, 6, 9):
        for rep in range(4):
            e_rel = TileStreamingEnv(trace_indices=[p], bundle=bundle, tile_frame="relative", **kw)
            e_abs = TileStreamingEnv(trace_indices=[p], bundle=bundle, **kw)
            opts = {"window": test_w[int(rng.integers(len(test_w)))]} if link == "lumos5g" else None
            o_rel, _ = e_rel.reset(seed=50 + rep, options=opts)
            e_abs.reset(seed=50 + rep, options=opts)
            assert o_rel.shape == ((14,) if link == "lumos5g" else (9,))
            for t in range(fs.H):
                bits = (rng.random(64) < rng.uniform(0, 0.5)).astype(np.int64)
                power = [] if link == "lumos5g" else [int(rng.integers(config.N_POWER_LEVELS))]
                pred_th, pred_ph = e_rel._predicted_theta, e_rel._predicted_phi
                o_next, r1, term, _, i_rel = e_rel.step(np.r_[bits, power])
                phys = tile_frame.rel_to_abs(bits, i_rel["tile_shift"]).astype(np.int64)
                _, r2, _, _, i_abs = e_abs.step(np.r_[phys, power])
                assert i_rel["tile_shift"] == tile_frame.column_shift(pred_th)
                assert np.array_equal(i_rel["tile_mask"], i_abs["tile_mask"])
                assert np.array_equal(i_rel["tile_mask_rel"], tile_frame.abs_to_rel(i_abs["tile_mask"], i_rel["tile_shift"]))
                worst["step"] = max([worst["step"], abs(r1 - r2)] + [abs(i_rel[f] - i_abs[f]) for f in FIELDS])
                # observation after this step: context for the NEXT decision
                th, ph = i_rel["actual_theta"], i_rel["actual_phi"]
                exp = [tile_frame.wrap_deg(th - pred_th), ph - pred_ph, ph, tile_frame.column_offset_deg(th),
                       np.sin(np.deg2rad(th)), np.cos(np.deg2rad(th)), (t + 1) / fs.H]
                worst["obs"] = max(worst["obs"], float(np.max(np.abs(o_next[1:8] - np.float32(exp)))))
                zs = config.LUMOS5G_OBS_SCALE if link == "lumos5g" else 1.0
                worst["obs"] = max(worst["obs"], abs(float(o_next[0]) - np.float32(i_rel["Z"] * zs)) / max(1.0, i_rel["Z"] * zs))
                # VE: the real relative action as a hypothetical branch
                m_b, pw_b = pds.unpack_action(np.r_[bits, power])
                m_b = tile_frame.rel_to_abs(m_b, i_rel["tile_shift"])
                ph_b = pds.compute_virtual_branch_physics(
                    agent_tile_mask_b=m_b, power_watts_b=pw_b, mandatory_tile_mask=i_rel["mandatory_tile_mask"],
                    rd_data=e_rel._rd_data, video_array_index=config.TRAINING_VIDEO_ARRAY_INDEX,
                    gop_index=i_rel["gop_index"], enhanced_level=e_rel._enhanced_level, Z_t=i_rel["Z_t"],
                    h_t=i_rel["h_t"], actual_theta_t=th, actual_phi_t=ph, mu_d=1.3, mu_p=0.4, mu_b=0.7,
                    R_t=i_rel.get("R_t_exogenous"))
                nxt = pds.raw_next_state_from_obs(i_rel["next_obs_raw"], ph_b["Z_next_b"], i_rel["z_obs_scale"])
                worst["ve"] = max(worst["ve"], abs(ph_b["Z_next_b"] - i_rel["Z"]), float(np.max(np.abs(nxt - o_next))),
                                  abs(ph_b["coverage_b"] - i_rel["coverage"]))
                assert np.array_equal(tile_frame.abs_to_rel(ph_b["tile_mask_b"], i_rel["tile_shift"]), i_rel["tile_mask_rel"])
                if term:
                    assert t == fs.H - 1 and abs(i_rel["J_D"] - i_abs["J_D"]) < 1e-12 and abs(i_rel["J_Q"] - i_abs["J_Q"]) < 1e-12
                    break
    print(f"   {link}: 12 episodes stepped in lockstep")
print(f"2. relative vs equivalent physical action: max abs difference {worst['step']:.2e}")
print(f"3. observation entries vs expected: max abs difference {worst['obs']:.2e}")
print(f"5. VE branch with the real relative action vs real step: max abs difference {worst['ve']:.2e}")

# PDS observation: 14 + 64 = 78
o = rng.random(14).astype(np.float32); o2 = rng.random(14).astype(np.float32); m = (rng.random(64) < 0.3).astype(np.uint8)
po = pds.build_pds_observation(o, o2, m)
assert po.shape == (78,) and po[0] == o2[0] and np.array_equal(po[1:14], o[1:]) and np.array_equal(po[14:], m)
print("   PDS observation: 78 = [post-decision Z, 13 pre-decision entries, 64 relative mask bits]")

# 4 -------------------------------------------------------------------------
v, train, held = lr.prepare()
test_w = lumos5g.windows(data["runs"], data["splits"]["test"], fs.H)
eps = [(p, test_w[i]) for p in held for i in rng.choice(len(test_w), 5, replace=False)]
rates = {w: lumos5g.rate_window(data["runs"], w[0], w[1], fs.H, 1.0, 5) for _, w in eps}
masks = {e: rng.random((fs.H, 64)) < 0.15 for e in eps}
ref = fs.simulate_trace(v, eps, lambda s: masks[(s["pos"], s["key"])][s["t"]], rates, 5, per_episode=True)
dmax = 0.0
for (p, w), r in zip(eps, ref):
    env = TileStreamingEnv(trace_indices=[p], bundle=bundle, link="lumos5g", lumos_split="test", lumos_data=data,
                           tile_frame="relative")
    env.reset(seed=0, options={"window": w})
    for t in range(fs.H):
        rel = tile_frame.abs_to_rel(masks[(p, w)][t], tile_frame.column_shift(env._predicted_theta))
        _o, _r, term, _, info = env.step(rel.astype(np.int64))
    dmax = max(dmax, abs(info["J_D"] - r["J_D"]), abs(info["J_B"] - r["J_B"]), abs(info["J_Q"] - r["J_cov"]))
print(f"4. replay masks sent as relative bits vs replay simulator: max |dJ| {dmax:.2e}")
