"""
literature_fine_check.py - fairness check for design_options_scan.py: give
the literature rules a finer grid of margin sizes (average-based and ASL360
threshold) so a tight tile budget is not lost only because the margin grid
was coarse. Same selection (training viewers, 5% safety margin) and held-out
evaluation as design_options_scan.py.

Usage: .venv/Scripts/python.exe literature_fine_check.py
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config
import design_options_scan as ds
import fastsim as fs
import room_to_learn as rl

DESIGNS = [("D0", 0.5, 8.0), ("D1", 0.5, 7.5), ("D2", 0.25, 8.0)]
LAMS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0)


def run(args):
    v, train, held = args
    T = v.traj
    tr_eps = [(p, s) for p in train for s in ds.TRAIN_SEEDS]
    he_eps = [(p, s) for p in held for s in ds.HELD_SEEDS]
    eth = np.concatenate([rl.wsigned(T[p]["at"] - T[p]["pt"])[1:] for p in train])
    eph = np.concatenate([(T[p]["ap"] - T[p]["pp"])[1:] for p in train])
    s_fit = (float(np.mean(np.abs(eth))), float(np.mean(np.abs(eph))))
    opts = []
    for lam in LAMS:
        m = {p: fs.region_masks(T[p]["pt"], T[p]["pp"], config.V_THETA_DEG + lam * s_fit[0], config.V_PHI_DEG + lam * s_fit[1]) for p in T}
        opts.append((f"average-based {lam}x", m, False))
        opts.append((f"threshold (10 s), margin {lam}x", m, True))
    rows = []
    for key, top_frac, b_bar in DESIGNS:
        P = fs.levels(top_frac); top = len(P) - 1
        ok = lambda m, mg=1.0: m["J_D"] <= ds.D_BAR * mg and m["J_B"] <= b_bar * mg

        def pol_for(masks, thresh):
            def pol(s):
                p, t = s["pos"], s["t"]
                ex = masks[p][t] if (not thresh or s["Z"] / v.mean_base_bits >= 10.0) else np.zeros(fs.NT, bool)
                return ex, top
            return pol
        best = None
        for name, masks, th in opts:
            m = fs.simulate(v, tr_eps, pol_for(masks, th), P)
            if ok(m, ds.MARGIN) and (best is None or m["J_cov"] > best[1]["J_cov"]):
                best = (name, m, pol_for(masks, th))
        if best:
            m = fs.simulate(v, he_eps, best[2], P)
            rows.append(dict(video=v.name, design=key, method="literature_fine", choice=best[0], held_feasible=ok(m), **m))
    return rows


def main():
    prepared = [ds.prepare(n) for n in ds.VIDEOS]
    with Pool(7) as pool:
        res = pool.map(run, prepared)
    fine = pd.DataFrame([r for rows in res for r in rows])
    fine.to_csv("VIDEO_SURVEY/design_options/literature_fine.csv", index=False)
    old = pd.read_csv("VIDEO_SURVEY/design_options/design_options.csv")
    mot = pd.read_csv("VIDEO_SURVEY/design_options/motion_room.csv")
    old = old[old.design.isin(["D0", "D1", "D2"]) & (old.method == "literature")][["video", "design", "psnr", "cov", "choice"]]
    f = fine[["video", "design", "psnr", "cov", "choice"]].rename(columns={"psnr": "psnr_fine", "cov": "cov_fine", "choice": "choice_fine"})
    m = mot[mot.method == "rl_estimate_motion"][["video", "design", "psnr", "cov"]].rename(columns={"psnr": "psnr_rl_motion", "cov": "cov_rl_motion"})
    t = old.merge(f, on=["video", "design"], how="outer").merge(m, on=["video", "design"], how="outer")
    t["best_lit_psnr"] = t[["psnr", "psnr_fine"]].max(axis=1)
    t["best_lit_cov"] = t[["cov", "cov_fine"]].max(axis=1)
    t["room_psnr"] = t["psnr_rl_motion"] - t["best_lit_psnr"]
    t["room_cov"] = t["cov_rl_motion"] - t["best_lit_cov"]
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 34)
    print(t[["design", "video", "choice", "psnr", "choice_fine", "psnr_fine", "psnr_rl_motion", "room_psnr", "room_cov"]]
          .sort_values(["design", "video"]).round(3).to_string(index=False))
    print(t.groupby("design")[["room_psnr", "room_cov"]].agg(["mean", "min", "max"]).round(3))


if __name__ == "__main__":
    main()
