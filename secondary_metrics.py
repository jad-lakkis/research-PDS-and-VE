"""
secondary_metrics.py - PPO (Experiment X / Exp1 PPO, bit-identical) vs every
heuristic baseline on the metrics other papers report: PSNR, coverage, stall
time, stall frequency, buffer level, quality variation between consecutive
segments, tiles, power. Runner, the standard held-out evaluation (traces 2, 6, 9
with channel seeds 1000-1002, exactly what PPO's own evaluation uses), stall
budget 0.3, tile budget 8.

PPO: per-step evaluation logs (eval_steps.csv) of the last ~1000 rollouts of
each of seeds 1, 2, 3, 5, 7. Heuristics: simulated with fastsim (matches the
environment), power fixed at 50 mW ("the reference methods use the same
transmit power", MMSP).

Usage: .venv/Scripts/python.exe secondary_metrics.py
"""
import numpy as np
import pandas as pd

import config
import combined_test as ct
import design_options_scan as ds
import fastsim as fs
import room_to_learn as rl
from streaming_rl.eval import decode_mask_hex

OUT = "VIDEO_SURVEY/secondary_metrics.csv"
EPS = list(zip(config.EVAL_TRACE_INDICES, config.EVAL_SEEDS))


def episode_metrics(psnr, cov, D, Z, P_mw, ntiles, bits):
    psnr, D = np.asarray(psnr), np.asarray(D)
    return dict(psnr=psnr.mean(), cov=np.mean(cov), quality_variation_dB=np.mean(np.abs(np.diff(psnr))),
                stall_s_per_segment=D.mean(), stall_segments_pct=np.mean(D > 1e-9) * 100,
                J_D=float(np.sum(fs.DISC[:len(D)] * D / fs.T0)), J_B=float(np.sum(fs.DISC[:len(D)] * np.asarray(ntiles) / fs.NT)),
                buffer_s=np.mean(Z), power_mW=np.mean(P_mw), tiles=np.mean(ntiles), bits_Mbit=np.mean(bits) / 1e6)


def main():
    v, train, held = ct.prepare("Runner")
    T = v.traj
    P = fs.levels(0.5); top = len(P) - 1
    seg_bits = v.mean_base_bits  # buffer in seconds of base layer, like ASL360's buffer threshold

    def run(extra_fn, power_fn=None, level_fn=None):
        rows = []
        for pos, seed in EPS:
            tr, hs = T[pos], v.channel(pos, seed)
            Z, rec = 0.0, dict(psnr=[], cov=[], D=[], Z=[], P=[], n=[], bits=[])
            for t in range(fs.H):
                s = dict(pos=pos, seed=seed, t=t, Z=Z, h=hs[t], tr=tr, video=v)
                w = tr["w"][t]
                if level_fn is None:
                    ex = extra_fn(s)
                    x = ex | tr["mand"][t]
                    A = v.A_base[t] + v.delta[t][x].sum()
                    mse = np.where(x, v.mse_e[t], v.mse_b[t])
                    n, cv = x.sum(), float(w[x].sum())
                else:
                    lv = level_fn(s)
                    A = sum(v.lv_bits[k][t][lv == k].sum() for k in np.unique(lv))
                    mse = np.choose(lv, [v.lv_mse[k][t] for k in range(7)])
                    n, cv = fs.NT, 1.0
                k = top if power_fn is None else power_fn(s, A)
                R = fs.rate(P[k], hs[t])
                D = 0.0 if Z + fs.T0 * R >= A else fs.T0 * (1.0 - (Z + fs.T0 * R) / A)
                rec["Z"].append(Z / seg_bits)
                Z = max(Z + fs.T0 * R - A, 0.0)
                rec["psnr"].append(10 * np.log10(255.0 ** 2 / (np.sum(w * mse) / np.sum(w))))
                rec["cov"].append(cv); rec["D"].append(D); rec["P"].append(P[k] * 1e3); rec["n"].append(n); rec["bits"].append(A)
            rows.append(episode_metrics(rec["psnr"], rec["cov"], rec["D"], rec["Z"], rec["P"], rec["n"], rec["bits"]))
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    eth = np.concatenate([rl.wsigned(T[p]["at"] - T[p]["pt"])[1:] for p in train])
    eph = np.concatenate([(T[p]["ap"] - T[p]["pp"])[1:] for p in train])
    s_fit = (float(np.mean(np.abs(eth))), float(np.mean(np.abs(eph))))

    def region(hw, hh):
        return lambda s: fs.region_masks(s["tr"]["pt"][s["t"]], s["tr"]["pp"][s["t"]], hw, hh)[0]

    def qer(w, h):
        def f(s):
            pt, pp = s["tr"]["pt"][s["t"]], s["tr"]["pp"][s["t"]]
            d = [np.hypot(rl.wsigned(pt - c[0]), pp - c[1]) for c in ds.QER_CENTRES]
            c = ds.QER_CENTRES[int(np.argmin(d))]
            return fs.region_masks(c[0], c[1], w / 2, h / 2)[0]
        return f

    def threshold(hw, hh):
        reg = region(hw, hh)
        return lambda s: reg(s) if s["Z"] / v.mean_base_bits >= 10.0 else np.zeros(fs.NT, bool)

    none = lambda s: np.zeros(fs.NT, bool)
    rows = {}
    rows["predicted viewport only"] = run(none)
    for w, h in [(120, 60), (160, 80), (200, 100)]:
        rows[f"QER {w}x{h} (MMSP)"] = run(qer(w, h))
    rows["average-based, MMSP bounds 30/15 (MMSP)"] = run(region(config.V_THETA_DEG + 30, config.V_PHI_DEG + 15))
    rows["average-based, fitted to our error data (MMSP)"] = run(region(config.V_THETA_DEG + s_fit[0], config.V_PHI_DEG + s_fit[1]))
    rows["threshold 10 s, literal (ASL360)"] = run(threshold(config.V_THETA_DEG, config.V_PHI_DEG))
    rows["threshold 10 s, EL = enlarged viewport (ASL360)"] = run(threshold(config.V_THETA_DEG + s_fit[0], config.V_PHI_DEG + s_fit[1]))
    rows["average-based 0.35x, tuned to our budgets"] = run(region(config.V_THETA_DEG + 0.35 * s_fit[0], config.V_PHI_DEG + 0.35 * s_fit[1]))

    # PPO from its own evaluation logs
    ppo = []
    for sd in [1, 2, 3, 5, 7]:
        d = pd.read_csv(f"FINAL_PART1_EXP/exp1_official/ppo_budgetA_dropP_seed{sd}/eval_steps.csv")
        d = d[d.rollout > d.rollout.max() - 1000]
        for _, ep in d.groupby(["rollout", "trace_position"]):
            ep = ep.sort_values("step")
            bits = []
            for r in ep.itertuples(index=False):
                x = decode_mask_hex(r.agent_mask_hex) | decode_mask_hex(r.mandatory_mask_hex)
                bits.append(v.A_base[int(r.step)] + v.delta[int(r.step)][x].sum())
            Zpre = np.r_[0.0, ep["Z"].values[:-1]] / seg_bits
            ppo.append(episode_metrics(ep["psnr_db"].values, ep["coverage"].values, ep["D_t"].values, Zpre,
                                       ep["power_watts"].values * 1e3, ep["n_enhanced_tiles"].values, bits))
    rows["PPO (Experiment X, late, 5 seeds)"] = {k: float(np.mean([r[k] for r in ppo])) for k in ppo[0]}
    target = rows["PPO (Experiment X, late, 5 seeds)"]["bits_Mbit"] * 1e6
    order = np.random.default_rng(0).permutation(fs.NT)

    def uniform_levels(s):
        t = s["t"]
        lv = np.full(fs.NT, 3); bits = v.lv_bits[3][t].sum()
        for i in order:
            add = v.lv_bits[4][t][i] - v.lv_bits[3][t][i]
            if bits + add <= target:
                lv[i] = 4; bits += add
        return lv
    rows["uniform, same bit rate as PPO (all 64 tiles)"] = run(None, level_fn=uniform_levels)

    df = pd.DataFrame(rows).T
    df["within budgets"] = (df["J_D"] <= 0.3) & (df["J_B"] <= 8.0)
    df.to_csv(OUT)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 50)
    cols = ["psnr", "cov", "quality_variation_dB", "stall_s_per_segment", "stall_segments_pct", "J_D", "J_B", "within budgets",
            "buffer_s", "power_mW", "tiles", "bits_Mbit"]
    print(df[cols].round(3).to_string())


if __name__ == "__main__":
    main()
