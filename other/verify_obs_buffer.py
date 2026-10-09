"""
verify_obs_buffer.py - checks for TileStreamingEnv(obs_buffer="log") (--obs-buffer log, Lumos5G only).

1. with obs_buffer="log" every observation entry except the buffer (and the forced-bits ratio) equals the
   default env's, step for step under the same actions
2. buffer entry = log(1 + Z / LUMOS5G_OBS_BUFFER_REF_BITS) of the env's own buffer; ratio entry = log(1 + ratio)
   of the default env's ratio
3. physics unchanged: rewards, stall, tiles and the buffer itself are identical to the default env
4. virtual experience: the real action through raw_next_state_from_obs reproduces the real next observation
5. what the change buys: the gap between an empty buffer and one forced second, in standard deviations of
   the buffer entry over these episodes, linear vs log

Usage: .venv/Scripts/python.exe other/verify_obs_buffer.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from streaming_rl import data_loader, lumos5g, pds, tile_frame  # noqa: E402
from streaming_rl.environment import TileStreamingEnv  # noqa: E402

bundle, data = data_loader.load_training_video_bundle(), lumos5g.load_bundle()
rng = np.random.default_rng(0)
wins = lumos5g.windows(data["runs"], data["splits"]["test"], 36)
worst = dict(rest=0.0, buf=0.0, ratio=0.0, phys=0.0, ve=0.0)
Z_lin, Z_log = [], []
for frame in ("relative", "absolute"):
    for forced in (True, False):
        for p in (2, 6, 9):
            for _ in range(3):
                w = wins[int(rng.integers(len(wins)))]
                kw = dict(trace_indices=[p], bundle=bundle, link="lumos5g", lumos_split="test", lumos_data=data,
                          tile_frame=frame, obs_forced_bits=forced)
                e0, e1 = TileStreamingEnv(**kw), TileStreamingEnv(obs_buffer="log", **kw)
                o0, _ = e0.reset(seed=1, options={"window": w})
                o1, _ = e1.reset(seed=1, options={"window": w})
                ri = e1._ratio_index if forced else None
                keep = [i for i in range(len(o0)) if i not in (0, ri)]
                for t in range(36):
                    worst["rest"] = max(worst["rest"], float(np.max(np.abs(o1[keep] - o0[keep]))))
                    worst["buf"] = max(worst["buf"], abs(float(o1[0]) - float(np.float32(np.log1p(e1._Z / config.LUMOS5G_OBS_BUFFER_REF_BITS)))))
                    if forced:
                        worst["ratio"] = max(worst["ratio"], abs(float(o1[ri]) - float(np.float32(np.log1p(o0[ri])))))
                    Z_lin.append(o0[0])
                    Z_log.append(o1[0])
                    a = (rng.random(64) < 0.2).astype(np.int64)
                    o0, r0, _, _, i0 = e0.step(a)
                    o1, r1, term, _, i1 = e1.step(a)
                    worst["phys"] = max(worst["phys"], abs(r0 - r1), abs(i0["D_t"] - i1["D_t"]), abs(e0._Z - e1._Z),
                                        float(np.any(i0["tile_mask"] != i1["tile_mask"])))
                    m, pw = pds.unpack_action(a)
                    if frame == "relative":
                        m = tile_frame.rel_to_abs(m, i1["tile_shift"])
                    ph = pds.compute_virtual_branch_physics(
                        agent_tile_mask_b=m, power_watts_b=pw, mandatory_tile_mask=i1["mandatory_tile_mask"],
                        rd_data=e1._rd_data, video_array_index=config.TRAINING_VIDEO_ARRAY_INDEX, gop_index=i1["gop_index"],
                        enhanced_level=e1._enhanced_level, Z_t=i1["Z_t"], h_t=i1["h_t"], actual_theta_t=i1["actual_theta"],
                        actual_phi_t=i1["actual_phi"], mu_d=1.0, mu_p=0.0, mu_b=1.0, R_t=i1["R_t_exogenous"])
                    nxt = pds.raw_next_state_from_obs(i1["next_obs_raw"], ph["Z_next_b"], i1["z_obs_scale"],
                                                      i1.get("obs_ratio_index"), i1.get("next_R"), i1.get("next_A_forced"),
                                                      i1["obs_buffer"])
                    worst["ve"] = max(worst["ve"], float(np.max(np.abs(nxt - o1))))
                assert term
print("1. every other observation entry vs the default env: max diff %.1e" % worst["rest"])
print("2. buffer entry vs log(1 + Z/ref): max diff %.1e; ratio entry vs log(1 + ratio): max diff %.1e" % (worst["buf"], worst["ratio"]))
print("3. reward / stall / buffer / tiles vs the default env: max diff %.1e" % worst["phys"])
print("4. VE next state vs real next observation: max diff %.1e" % worst["ve"])
one = 1.29e8   # one second of Runner's forced stream
print("5. empty buffer vs one forced second, in std of the buffer entry: linear %.3f, log %.3f"
      % (one * config.LUMOS5G_OBS_SCALE / np.std(Z_lin), np.log1p(one / config.LUMOS5G_OBS_BUFFER_REF_BITS) / np.std(Z_log)))
