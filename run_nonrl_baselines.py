"""
run_nonrl_baselines.py - adapted non-RL reference policies (MMSP'25
average-based and QER-based, ASL360 threshold-based) evaluated in our
environment on the SAME held-out traces / channel seeds as the learned
Experiment X policies (PPO, New PDS, VE v2; seeds 1,2,3,5,7), plus on the
9 training traces (for design-choice checks - nothing here is selected
from held-out outcomes). See OUT_DIR/README.md for what is taken from the
papers and what is our adaptation.

Usage: .venv/Scripts/python.exe run_nonrl_baselines.py
"""
import os
import time

import numpy as np
import pandas as pd

import config
import plot_exp3_full_panels as p3
from streaming_rl import channel_model, data_loader, layer_model, viewport
from streaming_rl.environment import TileStreamingEnv
from streaming_rl.eval import encode_mask_hex

OUT_DIR = "EXPERIMENT_X_drop_seed6/nonrl_baselines"
D_BAR, B_BAR = 0.3, 8.0
SEEDS_X = [1, 2, 3, 5, 7]
TAIL = 1000
TRAIN_POS = [i for i in range(12) if i not in config.EVAL_TRACE_INDICES]
TRAIN_SEEDS = (2000, 2001, 2002)
P_LEVELS = [k / (config.N_POWER_LEVELS - 1) * config.POWER_LEVEL_MAX_FRACTION
            * channel_model.dbm_to_watts(config.P_MAX_DBM) for k in range(config.N_POWER_LEVELS)]
WN0 = config.W_HZ * channel_model.dbm_to_watts(config.N0_DBM_PER_HZ)

# MMSP'25 Sec. IV values
SIGMA_MAX, OMEGA_MAX = 60.0, 30.0          # max enlargement (azimuth, altitude), deg
QER_SIZES = [(120.0, 60.0), (160.0, 80.0), (200.0, 100.0)]
QER_THETA = [-180.0 + 36.0 + 72.0 * k for k in range(5)]   # 5x5 cell centres
QER_PHI = [-90.0 + 18.0 + 36.0 * j for j in range(5)]
QER_CENTRES = [(th, ph) for ph in QER_PHI for th in QER_THETA]  # row-major, tie-break = lowest index
# ASL360 Sec. IV
BUFFER_THRESHOLD_SEC = 10.0


def wrap(d):
    d = abs(d) % 360.0
    return min(d, 360.0 - d)


def region_mask(theta_c, phi_c, half_w, half_h):
    """Tiles with nonzero overlap with a (theta,phi) rectangle - the same
    rule viewport.mandatory_tile_mask uses for the predicted viewport."""
    m = np.zeros(config.N_TILES, dtype=bool)
    for i in range(config.N_TILES):
        t0, t1, f0, f1 = viewport.tile_bounds(i)
        ov_t = viewport._interval_overlap_wrapped(theta_c - half_w, theta_c + half_w, t0, t1)
        ov_f = viewport._interval_overlap(phi_c - half_h, phi_c + half_h, f0, f1)
        m[i] = ov_t * ov_f > 0.0
    return m


def nearest_qer(theta, phi):
    d = [np.hypot(wrap(theta - th), phi - ph) for th, ph in QER_CENTRES]
    return QER_CENTRES[int(np.argmin(d))]


def a_of(env, mask):
    return layer_model.compute_A_t(env._rd_data, config.TRAINING_VIDEO_ARRAY_INDEX, env._gop_index,
                                   mask, enhanced_level=env._enhanced_level)["A_t"]


def power_idx(rule, env, executed_mask):
    """'pwmin': lowest level whose rate delivers the executed mask with no
    stall, from information known BEFORE acting (segment size A_t, buffer
    Z_t, current channel h_t) - eq. P_min; falls back to the top level.
    'p50': fixed top level (50 mW)."""
    if rule == "p50":
        return config.N_POWER_LEVELS - 1
    A, Z, h = a_of(env, executed_mask), env._Z, env._h
    pmin = (WN0 / h) * (2.0 ** (max(A - Z, 0.0) / (config.T0_SEC * config.W_HZ)) - 1.0)
    for k, p in enumerate(P_LEVELS):
        if p >= pmin:
            return k
    return config.N_POWER_LEVELS - 1


def make_policy(kind, param, prule, mean_base_bits):
    def decide(env):
        th, ph = env._predicted_theta, env._predicted_phi
        mand = viewport.mandatory_tile_mask(th, ph)
        if kind == "mandatory":
            prop = np.zeros(config.N_TILES, dtype=bool)
        elif kind == "average":
            s_t, s_f = param
            prop = region_mask(th, ph, config.V_THETA_DEG + s_t, config.V_PHI_DEG + s_f)
        elif kind == "qer":
            w, hh = param
            qt, qf = nearest_qer(th, ph)
            prop = region_mask(qt, qf, w / 2.0, hh / 2.0)
        elif kind == "threshold":
            buffer_sec = env._Z / mean_base_bits
            if buffer_sec >= BUFFER_THRESHOLD_SEC:
                prop = (mand.copy() if param is None
                        else region_mask(th, ph, config.V_THETA_DEG + param[0], config.V_PHI_DEG + param[1]))
            else:
                prop = np.zeros(config.N_TILES, dtype=bool)
        executed = prop | mand
        return prop, power_idx(prule, env, executed)
    return decide


def run_episode(decide, bundle, pos, seed, name, rows):
    env = TileStreamingEnv(trace_indices=[pos], bundle=bundle)
    env.reset(seed=seed)
    cov, psnr, stall, pw, nprop, nexec, outage, dt_us = [], [], [], [], [], [], [], []
    done, info, step = False, {}, 0
    while not done:
        z_pre = env._Z
        t0 = time.perf_counter()
        prop, k = decide(env)
        dt_us.append((time.perf_counter() - t0) * 1e6)
        action = np.concatenate([prop.astype(np.int64), [k]])
        _, _, term, trunc, info = env.step(action)
        done = term or trunc
        executed = info["tile_mask"].astype(bool)
        cov.append(info["coverage"]); psnr.append(info["viewport_psnr_db"]); stall.append(info["D_t"])
        pw.append(info["power_watts"] * 1e3); nprop.append(int(prop.sum())); nexec.append(int(executed.sum()))
        outage.append(info["coverage"] < 1.0 - 1e-9)
        rows.append(dict(variant=name, trace_position=pos, seed=seed, step=step, Z_pre_bits=z_pre,
                         proposed_mask_hex=encode_mask_hex(prop), executed_mask_hex=encode_mask_hex(executed),
                         n_proposed=int(prop.sum()), n_executed=int(executed.sum()),
                         n_mandatory=int(info["n_mandatory_tiles"]), power_mW=info["power_watts"] * 1e3,
                         D_t=info["D_t"], Z_post_bits=info["Z"], coverage=info["coverage"],
                         psnr_db=info["viewport_psnr_db"]))
        step += 1
    return dict(variant=name, trace_position=pos, seed=seed, J_Q=info["J_Q"], J_D=info["J_D"], J_B=info["J_B"],
                mean_coverage=np.mean(cov), mean_psnr_db=np.mean(psnr), mean_stall_sec=np.mean(stall),
                stall_slot_pct=np.mean(np.array(stall) > 1e-9) * 100, mean_power_mW=np.mean(pw),
                mean_proposed_tiles=np.mean(nprop), mean_executed_tiles=np.mean(nexec),
                full_coverage_outage_pct=np.mean(outage) * 100, decision_time_us=np.mean(dt_us))


def summarize(ep):
    g = ep.groupby("variant", sort=False).mean(numeric_only=True).drop(columns=["trace_position", "seed"])
    g["viol_D"] = (g["J_D"] - D_BAR).clip(lower=0)
    g["viol_B"] = (g["J_B"] - B_BAR).clip(lower=0)
    g["violation"] = g["viol_D"] + g["viol_B"]
    g["feasible"] = g["violation"] <= 1e-9
    return g


def prediction_error_stats(bundle):
    """Wrapped |error| of the 'previous viewport' predictor on the 9
    training traces (action-independent, so any action works here)."""
    dth, dph = [], []
    for pos in TRAIN_POS:
        env = TileStreamingEnv(trace_indices=[pos], bundle=bundle)
        env.reset(seed=0)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(np.zeros(config.N_TILES + 1, dtype=np.int64))
            done = term or trunc
            if info["predicted_theta"] is not None and len(dth) >= 0:
                dth.append(wrap(info["actual_theta"] - info["predicted_theta"]))
                dph.append(abs(info["actual_phi"] - info["predicted_phi"]))
    return np.array(dth), np.array(dph)


def learned_rows():
    raw = {m: [p3.load_progress(p3.PATHS[m].format(s=s)) for s in SEEDS_X] for m in ["PPO", "New PDS", "VE v2"]}
    n = min(len(d) for r in raw.values() for d in r)
    out = []
    for m, runs in raw.items():
        T = [d.iloc[:n].iloc[-TAIL:] for d in runs]
        g = lambda c: np.mean([t[c].mean() for t in T])
        sd = lambda c: np.std([t[c].mean() for t in T])
        out.append(dict(variant=f"LEARNED {m} (Exp X, last {TAIL} of {n}, 5 seeds)", J_Q=g("eval/J_Q"), J_D=g("eval/J_D"),
                        J_B=g("eval/J_B"), mean_coverage=g("eval/mean_coverage"), mean_psnr_db=g("eval/mean_psnr_db"),
                        mean_stall_sec=g("eval/mean_stall_sec"), mean_power_mW=g("eval/mean_power_mw"),
                        mean_executed_tiles=g("eval/mean_n_tiles"), violation=g("eval/violation"),
                        feasible_checkpoint_pct=np.mean([(t["eval/violation"] <= 1e-9).mean() * 100 for t in T]),
                        J_Q_seed_std=sd("eval/J_Q"), coverage_seed_std=sd("eval/mean_coverage")))
    return pd.DataFrame(out).set_index("variant")


def rl_inference_us():
    """Indicative per-decision actor cost: a same-codebase Lagrangian-PPO
    checkpoint (Exp X model files are not stored locally)."""
    from stable_baselines3 import PPO
    path = "expirments_ALL/runpod_results/dbar_0.3_pbar_6.5_bbar_8.0_seed0/block_v1/model.zip"
    model = PPO.load(path, device="cpu")
    obs = np.zeros(4, dtype=np.float32)
    for _ in range(50):
        model.predict(obs, deterministic=True)
    t0 = time.perf_counter()
    for _ in range(2000):
        model.predict(obs, deterministic=True)
    return (time.perf_counter() - t0) / 2000 * 1e6


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    bundle = data_loader.load_training_video_bundle()
    mean_base_bits = np.mean([layer_model.compute_A_t(bundle["rd_data"], config.TRAINING_VIDEO_ARRAY_INDEX, t,
                                                      np.zeros(config.N_TILES, bool),
                                                      enhanced_level=config.INITIAL_ENHANCEMENT_LEVELS[1])["A_base"]
                              for t in range(36)])
    dth, dph = prediction_error_stats(bundle)
    s_fit = (float(dth.mean()), float(dph.mean()))
    s_paper = (SIGMA_MAX / 2.0, OMEGA_MAX / 2.0)
    print(f"training-trace wrapped |error|: theta mean {dth.mean():.2f} p95 {np.quantile(dth, .95):.2f} max {dth.max():.2f};"
          f" phi mean {dph.mean():.2f} p95 {np.quantile(dph, .95):.2f} max {dph.max():.2f}")
    print(f"mean base layer per 1-s segment: {mean_base_bits / 1e6:.3f} Mbit -> 10 s threshold = {10 * mean_base_bits / 1e6:.1f} Mbit")

    variants = [("Predicted viewport only [pwmin]", "mandatory", None, "pwmin"),
                ("Predicted viewport only [50mW]", "mandatory", None, "p50")]
    for prule, tag in [("pwmin", "pwmin"), ("p50", "50mW")]:
        variants += [(f"MMSP average-based, fitted s=({s_fit[0]:.1f},{s_fit[1]:.1f}) [{tag}]", "average", s_fit, prule),
                     (f"MMSP average-based, paper s=({s_paper[0]:.0f},{s_paper[1]:.0f}) [{tag}]", "average", s_paper, prule)]
        variants += [(f"MMSP QER {int(w)}x{int(h)} [{tag}]", "qer", (w, h), prule) for w, h in QER_SIZES]
    variants += [("ASL360 threshold, EL=predicted viewport [50mW]", "threshold", None, "p50"),
                 (f"ASL360 threshold, EL=enlarged s=({s_paper[0]:.0f},{s_paper[1]:.0f}) [50mW]", "threshold", s_paper, "p50"),
                 (f"ASL360 threshold, EL=enlarged s=({s_fit[0]:.1f},{s_fit[1]:.1f}) [50mW]", "threshold", s_fit, "p50")]
    # extra average-based candidates for budget calibration: scaled fitted enlargement
    for lam in (0.25, 0.5, 0.75):
        s = (lam * s_fit[0], lam * s_fit[1])
        for prule, tag in [("pwmin", "pwmin"), ("p50", "50mW")]:
            variants.append((f"MMSP average-based, {lam:.2f}x fitted s=({s[0]:.1f},{s[1]:.1f}) [{tag}]", "average", s, prule))
    families = {"MMSP average-based": lambda n: n.startswith("MMSP average-based"),
                "MMSP QER": lambda n: n.startswith("MMSP QER"),
                "ASL360 threshold": lambda n: n.startswith("ASL360 threshold")}
    summaries = {}

    for split, plist, slist in [("heldout", list(config.EVAL_TRACE_INDICES), None), ("train", TRAIN_POS, TRAIN_SEEDS)]:
        eps, rows = [], []
        for name, kind, param, prule in variants:
            decide = make_policy(kind, param, prule, mean_base_bits)
            for i, pos in enumerate(plist):
                seeds = [config.EVAL_SEEDS[i]] if slist is None else list(slist)
                for sd in seeds:
                    eps.append(run_episode(decide, bundle, pos, sd, name, rows))
        ep = pd.DataFrame(eps)
        ep.to_csv(f"{OUT_DIR}/episodes_{split}.csv", index=False)
        if split == "heldout":
            pd.DataFrame(rows).to_csv(f"{OUT_DIR}/steps_heldout.csv", index=False)
        summ = summarize(ep)
        summ.to_csv(f"{OUT_DIR}/summary_{split}.csv")
        print(f"\n=== {split} ({len(plist)} traces) ===")
        cols = ["J_Q", "mean_coverage", "mean_psnr_db", "J_D", "J_B", "violation", "feasible", "stall_slot_pct",
                "mean_power_mW", "mean_proposed_tiles", "mean_executed_tiles", "full_coverage_outage_pct", "decision_time_us"]
        with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
            print(summ[cols])
        summaries[split] = summ
    held, train = summaries["heldout"], summaries["train"]

    # Budget calibration, decided on TRAINING traces only: within each family,
    # the variant with the highest training J_Q among training-feasible ones
    # (if none is training-feasible, the lowest training violation).
    picks = []
    for fam, is_member in families.items():
        tr = train[[is_member(n) for n in train.index]]
        feas = tr[tr["feasible"]]
        pick = feas["J_Q"].idxmax() if len(feas) else tr["violation"].idxmin()
        picks.append(dict(family=fam, selected_on_train=pick, train_J_Q=tr.loc[pick, "J_Q"],
                          train_violation=tr.loc[pick, "violation"], n_candidates=len(tr), n_train_feasible=len(feas)))
    picks = pd.DataFrame(picks)
    picks.to_csv(f"{OUT_DIR}/budget_calibrated_selection.csv", index=False)
    print("\n=== budget-calibrated pick per family (chosen on training traces) ===")
    print(picks.to_string(index=False))

    learned = learned_rows()
    rl_us = rl_inference_us()
    learned["decision_time_us"] = rl_us
    comp = pd.concat([held, learned], axis=0, sort=False)
    comp.to_csv(f"{OUT_DIR}/comparison_heldout_with_learned.csv")
    print("\n=== learned (Exp X) on the same held-out traces ===")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(learned[["J_Q", "mean_coverage", "mean_psnr_db", "J_D", "J_B", "violation", "feasible_checkpoint_pct",
                       "mean_power_mW", "mean_executed_tiles", "decision_time_us"]])
    pd.Series(dict(train_theta_err_mean=dth.mean(), train_phi_err_mean=dph.mean(),
                   train_theta_err_p95=np.quantile(dth, .95), train_phi_err_p95=np.quantile(dph, .95),
                   mean_base_bits_per_segment=mean_base_bits, rl_actor_inference_us=rl_us)).to_csv(f"{OUT_DIR}/design_constants.csv")


if __name__ == "__main__":
    main()
