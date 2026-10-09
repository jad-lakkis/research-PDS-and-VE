"""
verify_forced_bits.py - checks for TileStreamingEnv(obs_forced_bits=True, episode_stall_floor=True).

1. without the flags the observation is unchanged; with them it is the same plus 2 entries
2. obs[-2] = forced-stream bits of the coming second (= A_t of a step with no extra tiles), in Mbit;
   obs[-1] = (Z + T0 R_t) / forced bits
3. info["J_D_floor"] = the episode's J_D when only the forced tiles are sent
4. virtual experience: the real action through raw_next_state_from_obs (with the ratio recomputed)
   reproduces the real next observation

Usage: .venv/Scripts/python.exe other/verify_forced_bits.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from streaming_rl import data_loader, lumos5g, pds  # noqa: E402
from streaming_rl.environment import TileStreamingEnv  # noqa: E402

bundle, data = data_loader.load_training_video_bundle(), lumos5g.load_bundle()
rng = np.random.default_rng(0)
wins = lumos5g.windows(data["runs"], data["splits"]["test"], 36)
worst = dict(base=0.0, bits=0.0, ratio=0.0, floor=0.0, ve=0.0)
for frame in ("relative", "absolute"):
    for p in (2, 6, 9):
        for _ in range(3):
            w = wins[int(rng.integers(len(wins)))]
            kw = dict(trace_indices=[p], bundle=bundle, link="lumos5g", lumos_split="test", lumos_data=data, tile_frame=frame)
            e0 = TileStreamingEnv(**kw)
            e1 = TileStreamingEnv(obs_forced_bits=True, episode_stall_floor=True, **kw)
            ez = TileStreamingEnv(**kw)                      # forced tiles only, for the floor
            o0, _ = e0.reset(seed=1, options={"window": w})
            o1, _ = e1.reset(seed=1, options={"window": w})
            ez.reset(seed=1, options={"window": w})
            assert o1.shape[0] == o0.shape[0] + 2 == e1.observation_space.shape[0]
            for t in range(36):
                worst["base"] = max(worst["base"], float(np.max(np.abs(o1[:-2] - o0))))
                a = (rng.random(64) < 0.2).astype(np.int64)
                Z_t = e1._Z
                _, _, _, _, iz = ez.step(np.zeros(64, np.int64))
                worst["bits"] = max(worst["bits"], abs(o1[-2] - np.float32(iz["A_t"] * 1e-6)))
                worst["ratio"] = max(worst["ratio"], abs(o1[-1] - np.float32((Z_t + iz["R_t"]) / iz["A_t"])))
                o0, *_ = e0.step(a)
                o1, _, term, _, i1 = e1.step(a)
                m, pw = pds.unpack_action(a)
                if frame == "relative":
                    from streaming_rl import tile_frame
                    m = tile_frame.rel_to_abs(m, i1["tile_shift"])
                ph = pds.compute_virtual_branch_physics(
                    agent_tile_mask_b=m, power_watts_b=pw, mandatory_tile_mask=i1["mandatory_tile_mask"],
                    rd_data=e1._rd_data, video_array_index=config.TRAINING_VIDEO_ARRAY_INDEX, gop_index=i1["gop_index"],
                    enhanced_level=e1._enhanced_level, Z_t=i1["Z_t"], h_t=i1["h_t"], actual_theta_t=i1["actual_theta"],
                    actual_phi_t=i1["actual_phi"], mu_d=1.0, mu_p=0.0, mu_b=1.0, R_t=i1["R_t_exogenous"])
                nxt = pds.raw_next_state_from_obs(i1["next_obs_raw"], ph["Z_next_b"], i1["z_obs_scale"],
                                                  i1["obs_ratio_index"], i1["next_R"], i1["next_A_forced"])
                worst["ve"] = max(worst["ve"], float(np.max(np.abs(nxt - o1))))
            assert term
            worst["floor"] = max(worst["floor"], abs(i1["J_D_floor"] - iz["J_D"]))
print("1. observation without the 2 new entries vs flags off: max diff %.1e" % worst["base"])
print("2. forced bits vs A_t of a forced-only step: max diff %.1e Mbit; ratio: max diff %.1e" % (worst["bits"], worst["ratio"]))
print("3. episode floor vs forced-only J_D: max diff %.1e" % worst["floor"])
print("4. VE next state vs real next observation: max diff %.1e" % worst["ve"])
