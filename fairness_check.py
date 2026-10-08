"""
fairness_check.py - is the rule-vs-RL comparison fair?
 (1) SAME-INFORMATION rule: no viewing direction, like our agent. Always adds
     the k tiles that were most useful as extras on the 9 training traces;
     k chosen on training traces (largest k within both budgets).
 (2) Expected-coverage rule (uses the predicted direction) with its threshold
     chosen ONLY on training traces, with a safety margin fixed in advance
     (0%, 5%, 10% below each budget) - no held-out selection.
Held-out evaluation identical to the learned policies'. Power: 50 mW.

Usage: .venv/Scripts/python.exe fairness_check.py
"""
import numpy as np
import pandas as pd

import config
import room_to_learn as rl
import run_nonrl_baselines as rb
from streaming_rl import data_loader

OUT = "EXPERIMENT_X_drop_seed6/nonrl_baselines/fairness_check.csv"
H, NT = rl.H, rl.NT


def main():
    bundle = data_loader.load_training_video_bundle()
    held = [(p, s, rl.trajectory(bundle, p, s)) for p, s in zip(config.EVAL_TRACE_INDICES, config.EVAL_SEEDS)]
    train_tr = {p: rl.trajectory(bundle, p, 2000) for p in rb.TRAIN_POS}
    train_eps = [(p, s) for p in rb.TRAIN_POS for s in (2000, 2001, 2002)]

    def run(eps, masks_for):
        sims = [rl.simulate(bundle, p, s, masks_for(p)) for p, s in eps]
        return {k: float(np.mean([x[k] for x in sims])) for k in ["J_Q", "J_D", "J_B", "cov", "psnr"]}

    held_eps = [(p, s) for p, s, _ in held]
    held_tr = {p: tr for p, _, tr in held}
    rows = []

    # (1) same-information rule: fixed most-useful tiles from training
    gain = np.zeros(NT); n = 0
    for tr in train_tr.values():
        wa = rl.weights(tr["at"], tr["ap"]); mand = rl.weights(tr["pt"], tr["pp"]) > 0
        gain += np.where(mand, 0.0, wa).sum(axis=0); n += H
    order = np.argsort(-gain)
    print("most useful fixed extra tiles on training (row,col):", [(int(i) // 8, int(i) % 8) for i in order[:6]])
    pick_k = 0
    for k in range(0, 9):
        m = np.zeros((H, NT), bool); m[:, order[:k]] = True
        tr_r = run(train_eps, lambda p: m)
        he_r = run(held_eps, lambda p: m)
        ok = tr_r["J_B"] <= rl.B_BAR and tr_r["J_D"] <= rl.D_BAR
        if ok:
            pick_k = k
        rows.append(dict(method=f"Same-info rule: always add top {k} tiles", train_feasible=ok, train_J_B=tr_r["J_B"],
                         train_J_D=tr_r["J_D"], **{f"held_{a}": b for a, b in he_r.items()}))
    for r in rows:
        r["training_pick"] = r["method"].endswith(f"top {pick_k} tiles")

    # (2) expected-coverage rule, threshold from training only, margin fixed in advance
    eth = np.concatenate([rl.wsigned(t["at"] - t["pt"])[1:] for t in train_tr.values()])
    eph = np.concatenate([(t["ap"] - t["pp"])[1:] for t in train_tr.values()])

    def expected(tr):
        mand = rl.weights(tr["pt"], tr["pp"]) > 0
        E = np.stack([rl.weights(tr["pt"][t] + eth, tr["pp"][t] + eph).mean(axis=0) for t in range(H)])
        return np.where(mand, -1.0, E)

    E_train = {p: expected(tr) for p, tr in train_tr.items()}
    E_held = {p: expected(tr) for p, tr in held_tr.items()}
    for margin in (0.0, 0.05, 0.10):
        jb_t, jd_t = rl.B_BAR * (1 - margin), rl.D_BAR * (1 - margin)
        lo, hi = 0.0, 0.2
        for _ in range(18):
            mid = (lo + hi) / 2
            r = run(train_eps, lambda p: E_train[p] >= mid)
            if r["J_B"] <= jb_t and r["J_D"] <= jd_t:
                hi = mid
            else:
                lo = mid
        tau = hi
        he_r = run(held_eps, lambda p: E_held[p] >= tau)
        tr_r = run(train_eps, lambda p: E_train[p] >= tau)
        rows.append(dict(method=f"Expected-coverage rule, training margin {int(margin * 100)}%", train_feasible=True,
                         train_J_B=tr_r["J_B"], train_J_D=tr_r["J_D"], training_pick=True,
                         **{f"held_{a}": b for a, b in he_r.items()}))

    df = pd.DataFrame(rows)
    df["held_feasible"] = (df["held_J_B"] <= rl.B_BAR) & (df["held_J_D"] <= rl.D_BAR)
    df.to_csv(OUT, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(df[["method", "training_pick", "train_J_B", "train_J_D", "held_cov", "held_psnr", "held_J_B", "held_J_D", "held_feasible"]])
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
