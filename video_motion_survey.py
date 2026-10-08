"""
video_motion_survey.py - head motion and tile-selection headroom for every
video in the dataset, under our exact setup (1-s segments = 30 frames,
previous viewport as prediction, 8x8 tiles, 120x60 viewport, mandatory
predicted-viewport tiles, enhancement level 5, 50 mW link).

Per video, over all valid head traces:
  - per-segment head move (azimuth wrapped, elevation)
  - coverage of the predicted-viewport tiles alone
  - + 2 random extra tiles (exact expectation) and + 2 best extra tiles
    chosen with the actual viewport known (oracle) -> how much 2 extra
    tiles can possibly matter
  - mandatory-only stall/tile cost at 50 mW (3 channel draws per trace)

Usage: .venv/Scripts/python.exe video_motion_survey.py
"""
import os

import numpy as np
import pandas as pd

import config
import room_to_learn as rl
from streaming_rl import channel_model, data_loader, layer_model

OUT = "VIDEO_SURVEY"
GOP = config.GOP_SIZE_FRAMES
GAM = config.DISCOUNT_FACTOR_LAMBDA
W, T0 = config.W_HZ, config.T0_SEC
WN0 = W * channel_model.dbm_to_watts(config.N0_DBM_PER_HZ)
P_TOP = 0.5 * channel_model.dbm_to_watts(config.P_MAX_DBM)
LEVEL = config.INITIAL_ENHANCEMENT_LEVELS[1]


def main():
    os.makedirs(OUT, exist_ok=True)
    cat = data_loader.load_video_catalog()
    hn = data_loader.load_hn_data()
    rd = data_loader.load_rd_data()
    rows = []
    for _, v in cat.iterrows():
        traces = data_loader.get_valid_traces(hn, v.array_index)
        n_seg = v.frames // GOP
        dth, dph, cov_m, cov_r2, cov_o2, big = [], [], [], [], [], []
        jd, jb, stall_slots = [], [], []
        for _, arr in traces:
            th = arr[np.arange(n_seg) * GOP, 0]
            ph = arr[np.arange(n_seg) * GOP, 1]
            pt, pp = np.concatenate([[th[0]], th[:-1]]), np.concatenate([[ph[0]], ph[:-1]])
            wa = rl.weights(th, ph)
            mand = rl.weights(pt, pp) > 0
            e = np.abs(rl.wsigned(th - pt))
            for g in range(1, n_seg):
                free = ~mand[g]
                cm = wa[g][mand[g]].sum()
                cov_m.append(cm)
                cov_r2.append(cm + 2 * wa[g][free].mean())
                cov_o2.append(cm + np.sort(wa[g][free])[-2:].sum())
                dth.append(e[g]); dph.append(abs(ph[g] - pp[g])); big.append(e[g] > 60)
            A = np.array([layer_model.compute_A_t(rd, v.array_index, g, mand[g], enhanced_level=LEVEL)["A_t"]
                          for g in range(n_seg)])
            for s in range(3):
                rng = np.random.default_rng(7000 + s)
                Z, J = 0.0, 0.0
                for g in range(n_seg):
                    h = channel_model.sample_channel_gain(rng)
                    R = W * np.log2(1.0 + P_TOP * h / WN0)
                    D = 0.0 if Z + T0 * R >= A[g] else T0 * (1.0 - (Z + T0 * R) / A[g])
                    Z = max(Z + T0 * R - A[g], 0.0)
                    J += GAM ** g * D / T0
                    stall_slots.append(D > 1e-9)
                jd.append(J)
            jb.append(np.sum(GAM ** np.arange(n_seg) * mand.sum(axis=1)) / config.N_TILES)
        rows.append(dict(video=v["name"], array_index=v.array_index, seconds=n_seg, viewers=len(traces),
                         move_az_mean=np.mean(dth), move_az_median=np.median(dth), move_el_mean=np.mean(dph),
                         pct_jumps_over_60=np.mean(big) * 100, cov_predicted=np.mean(cov_m),
                         cov_plus2_random=np.mean(cov_r2), cov_plus2_oracle=np.mean(cov_o2),
                         room_oracle_minus_random=np.mean(cov_o2) - np.mean(cov_r2),
                         mandatory_Mbit=np.mean(A) / 1e6, mandonly_J_D_50mW=np.mean(jd),
                         mandonly_stall_slot_pct=np.mean(stall_slots) * 100, mandonly_J_B=np.mean(jb)))
    df = pd.DataFrame(rows).sort_values("move_az_mean", ascending=False)
    df.to_csv(f"{OUT}/video_motion_survey.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))
    print(f"saved {OUT}/video_motion_survey.csv")


if __name__ == "__main__":
    main()
